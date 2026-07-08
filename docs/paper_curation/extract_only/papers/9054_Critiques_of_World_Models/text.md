# arXiv:2507.05169v5[cs.LG]17Jun2026

## Critique of World Model

Eric Xing⋄,†∗, Mingkai Deng⋄,†∗, Jinyu Hou⋄,† ⋄Institute of Foundation Models, Mohamed bin Zayed University of Artificial Intelligence † School of Computer Science, Carnegie Mellon University {eric.xing, mingkai.deng, jinyu.hou}@mbzuai.ac.ae June 1, 2026

Abstract

World Model, the algorithmic simulator of the real-world environment which biological agents experience and act upon, has been an emerging topic in recent years due to the rising need to develop virtual agents with artificial (general) intelligence. There has been much discussion on what a world model really is, how to build it, how to use it, and how to evaluate it. In this essay, starting from the imagination in the famed Sci-Fi classic Dune, and drawing inspiration from the concept of “hypothetical thinking” in psychology literature, we argue the primary goal of a world model to be simulating all actionable possibilities of the real world for purposeful reasoning and acting. We examine the key design dimensions of world modeling: data, representation, architecture, learning objective, and usage, surveying existing approaches and analyzing their tradeoffs. Building on this examination, we propose a new Generative Latent Prediction (GLP) architecture for a general-purpose world model, based on stateful, hierarchical, multi-level, and mixed continuous/discrete representations, and a generative and self-supervised learning framework, with an outlook of a Physical, Agentic, and Nested (PAN) AGI system enabled by such a model.

### 1 Introduction

A Large Language Model (LLM) simulates the next word in human languages, which has led to systems like ChatGPT that allow people to perform a wide range of tasks facilitated through language, such as common conversation, standardized tests, professional writing, and advanced math reasoning, at a level rivaling human intelligence.

What would you do if you could perfectly simulate the next world – every possible future in the environment that we reside? Dune, a classic of science fiction that inspired the likes of George Lucas’ Star Wars and Miyazaki’s Valley of the Wind, boldly imagines such a possibility. The series is centered around the Kwisatz Haderach, a prophesized human being who inherits their ancestors’ memories and simulates the outcomes of all possible plans in order to chart the best path to achieve their goals [35]. Such superhuman ability allows them to command armies to win galactic wars, or to oversee global-scale projects that turn a desert planet into a green paradise. Is it possible to build towards computer systems with similar functionalities using a similar approach?

∗ Co-first author

Unlike a chat-bot, human consists of a hierarchy of abilities that go from immediate, concrete ones (e.g., body control/movement/action, reading/listening, and speaking/drawing) to far-reaching and abstract ones (e.g., planning, collaboration, and strategizing). Furthermore, the same human, while not necessarily perfectly, can perform a broad range of tasks (e.g., do household chores, complete risky expeditions, conduct research investigations, navigate social situations, and manage complex enterprises) all with the same cognitive architecture of the human brain. Can a single artificial intelligence (AI) system perform all these tasks? Each of these problems can be seen as a goal-oriented agent acting in a multimodal environment, requiring purposeful reasoning of massive temporal-spacial, social-physical, and emotional-cognitive complexity and depth, to the point that traditional approaches built on logical inference (e.g., induction, deduction, abduction) are often easily overwhelmed. It emerges that the key to generalized decision-making toward such complexity lies in reasoning by “seeing the future” as seen in Dune – an ability formally known as Hypothetical Thinking [5] in the psychology literature, or “thought experiments” in common practice – the ability to simulate the next worlds using a mental model of the world. We call such a mental model a World Model.

Specifically, a World Model (WM) is a generative model that simulates the possibilities in diverse scenarios (e.g., physical world, mental world, social world, and evolutionary world). Operationally, a WM takes previous world state s and action a, and predicts or simulates the next world state s′ through a transformation function, such as a conditional probability distribution:

s′ ∼ p(s′|s,a) (1)

With a WM, machines can perform thought experiments by simulating actions and plans in complex scenarios, including counterfactual ones, and extracting the best ones among them. This is consistent with the hypothesis that humans reason not just linearly with formal rules toward goals (e.g., imagine a self-serving individual immediately offering help to another person upon seeing them cry in the hopes of stopping them from crying) via a deterministic optimization algorithm, but also based on simulations using an internal mental model (e.g., imagine the same individual decides what to do by mentally simulating multiple possible outcomes including self-exhaustion, the other person stops crying, they continue to cry but are grateful, etc., with the best expected reward in mind) [28, 56].

This view of the world model as fundamentally a simulator, rather than a generator of visual content, is gaining broad recognition across the field [32, 33]. While some framings emphasize physical simulation (geometry, physics, and dynamics), we argue that the concept extends further: a world model should serve as the substrate for simulative reasoning, the capacity to mentally enact hypothetical scenarios (e.g., physical, social, strategic, or abstract) in service of purposeful decisionmaking.

Such a WM also enables transfer of knowledge to solving novel tasks, thanks to the fact that real world dynamics, even in different scenarios, share many mechanistic commonalities. For instance, a scuba diver experiences how their body moves in response to low gravity, which is likely helpful to walking on the moon. A mountaineer would be good at predicting how terrain conceals individual movements, which is useful when it’s their turn to lead a mountain ambush during a war. On the other hand, a skilled gamer has deep knowledge of how digital characters respond to control signals, which will come in handy when they become drone operators. Therefore, like humans often employ their mental model to help extrapolate from past experiences to act novelly in new environments; machines can leverage a WM similarly to better achieve zero-shot capabilities in unfamiliar environments.

How should we create such a general WM? The key desiderata for building and training a WM include the following 5 aspects: identifying and preparing training data with the desired world information; adopting a general representation space for the latent world state with possibly richer

[Figure 1]

𝜋 Agent

[Figure 2]

[Figure 3]

World State 𝑠𝑡 𝑎𝑡 𝑠𝑡+1 𝑔 Goal 𝑇

| | |
|---|---|
|𝑉| |

|𝜇|
|---|

[Figure 4]

Critic

[Figure 5]

Universe

Figure 1: A possible definition of an optimal agent

meaning than the observation data in plain sight; designing an architecture that allows effective reasoning over the representations; choosing an objective that properly guides the model training; determining how to use the world model in a decision-making system. Recent years have seen a surge in efforts toward a world model. In this paper, we examine the empirical design landscape along these five dimensions, survey existing technical approaches including recent systematic proposals, and present our perspective on what a general-purpose world model requires.

We conclude our examination with a brief preview of an alternative architecture – PAN – a Physical, Agentic, and Nested world model, which we argue offers the potential for a truly general-purpose and actionable world model, based on the following design principles: 1) data from all modalities of experience; 2) mixed continuous and discrete representation; 3) hierarchical generative latent prediction (GLP) modeling paradigm with an extended LLM backbone (for discrete concept-based reasoning), as well as a generative embedding predictive module (for continuous gradient-based reasoning), as the reasoning engines; 4) generative loss grounded in observation data; and 5) usage for simulating experience for training agents with reinforcement learning (RL). Full details of the PAN world models and results will be provided in a separate dedicated manuscript [66].

### 2 World Model and Agent Decision-Making

World model arises in the context of agent decision-making. An agent is an autonomous system that acts in an environment – the universe, with both physical and social worlds – to achieve a goal (e.g., climbing a mountain, winning a military campaign). We consider an environment with discrete timesteps indexed by t (continuous timesteps can be approximated by infinitesimally small discrete time-steps). Formally, the agent takes the current world state st and outputs the next action at based on a distribution pπ(at|st), known as the “policy” in reinforcement learning literature. The optimal agent is thus one that best achieves its goals across all environments. The concept of a World Model arises as a surrogate of the environment in general agentic reasoning.

##### 2.1 Agent-Environment Model and Optimal Agent

Consider a sequential interaction between an agent and an environment (Figure 1). At time step t, the agent outputs action at, and the universe µ takes the current state st and the action at, and outputs the next state st+1 based on distribution pµ(st+1|st,at). The distribution of the interaction trajectory until timestep T, or (at,st+1,...,aT−1,sT), given the current state st is thus denoted by:

T−1

pπµ(at,st+1,...,sT | st) =

pπ(ak | sk) agent

pµ(sk+1 | sk,ak) universe

(2)

k=t

In each state st, the agent also receives a reward r(g,st) based on its goal g. We evaluate the agent by its discounted cumulative reward, denoted as ∞k=t γkr(g,sk) (with the discount parameter γt decaying to zero with time, i.e., limt→∞ γt = 0). Note that this reward function can be dense (e.g., gaming scores), but perhaps frequently sparse (e.g., curing a disease). The agent’s long-term success can thus be measured by its expected future discounted reward, also known as value function [53]:

∞

Vπ,µg (st) := Eπ,µ

γkr(g,sk) st

k=t

T k=t γkr(g,sk)

pπµ(at,st+1,...,sT | st) trajectory

= lim

T→∞

(at,st+1,...,sT )

goal

(3)

Based on Equations 2 and 3, we can define the optimal agent in this universe µ as one that maximizes the value function, written formally as below:

πµ∗ := arg max

Vπ,µg (4)

π

Some simple derivation will show that the optimal agent in state st will select actions based on the following decision rule πµ∗(st) when planning for actions at:T−1:

T−1

T−1

πµ∗(st) = arg max

γkr(g,sk) + γTVπ,µg (sT)

pµ(si+1|si,ai) universe response

(5)

at:T−1 possible actions

st+1:T

i=t

k=t

goal progress

##### 2.2 World Model and Simulative Reasoning

Note that optimal decision-making defined in Equation 5 requires the agent to have access to the ground-truth world state s from the universe µ to experience and optimize. However, these are often not the case aside from simple scenarios like the Go and Chess games [47, 48] – imagine building an agent to land on Mars, or even a real-world robot relying on noisy sensors in daily environments. World Model thus arises as a crucial component for predicting the universe’s response to a general agent. Specifically, as illustrated in Figure 2, a WM f operates on an internal (continuous or discrete) representation of the world state, denoted as a belief state sˆt, which is derived from sensory inputs ot via an Encoder h (unlike the optimal agent described in §2.1 which has direct access to the true world state st). Given a proposed action a′t (as opposed to the true action at used by the optimal agent), the WM predicts the next belief state sˆt+1 according to the distribution pf(ˆst+1|sˆt,a′t). Note that for such simulative reasoning to be possible at all, the belief state must be stateful: sˆt must constitute an identifiable, persistent estimate of the world state that the agent can hold in long/short-term memory, revisit, and update across simulation steps, rather than a transient encoding of the current sensory stream. We return to this requirement, and to what representations satisfy it, in §4.2. This predicted belief state then allows the agent to propose the next action, continuing the cycle of prediction and action up to a desired time horizon T′.

The agent can simulate multiple such sequences of proposed actions and belief states, and select the actual action at (upon observing ot) based on some external function such as a Critic that evaluates the outcome against a given Goal. Thus, a WM essentially functions as a generative model of possible future world states, which enables simulative reasoning, or ”thought experiments”. Formally, for the optimal agent πf∗ equipped with WM f in belief state sˆt, we define the simulation-based decision rule in Equation 6 as follows:

πf∗(ˆst) = arg max

a′t:T′−1 possible actions

s ˆt+1:T′

T′−1

γkr(g,sˆk) + γT′Vπ,fg (ˆsT′)

k=t

goal progress

T′−1

pf(ˆsi+1|sˆi,a′i) simulation with world model

i=t

(6)

[Figure 6]

[Figure 7]

[Figure 8]

𝜋 Agent

Encoder ℎ

𝑓

World Model

continuous or discrete internal representations

[Figure 9]

[Figure 10]

𝑎𝑡′ 𝑠Ƹ𝑡+1′ Goal

Belief 𝑔

𝑠Ƹ𝑡

simulative reasoning to

find the best action

𝑇′

agent estimation

| | |
|---|---|
|𝑉| |

of world state

[Figure 11]

[Figure 12]

𝑜𝑡

𝑎𝑡

Observation

Critic

actual action

𝑇

sensory data contains incomplete information

|𝜇|
|---|

[Figure 13]

Universe

inaccessible

- Figure 2: An agent in real world where groundtruth world state and universe are unavailable to experience or experiment, so world model is crucial for simulation.

A general-purpose WM enables simulation of diverse possibilities across a wide range of domains, enabling agents to reason about outcomes without direct interaction with the environment. This includes, but is not limited to the following examples:

- • Physical dynamics: Mechanics of the real world, such as how water pours, how an object moves when thrown, or how a machine operates under varying conditions.
- • Embodied experiences: Internal bodily states (e.g., balance, posture), sensations (e.g., heat, pain, dizziness), and complex motor activities like getting dressed or tying shoes.
- • Emotional states: Affective responses such as happiness, sadness, or fear, which can facilitate planning in emotionally charged contexts (e.g., therapy or social interactions).
- • Social situations: The actions and internal states of other individuals, including their embodied or emotional experiences, needs, intentions, and expectations.
- • Mental world: Abstract “thought processes” such as logistics, tactics, and strategies, potentially in multi-agent or adversarial settings.
- • Counterfactual world: Alternative realities or “what if” scenarios to guide better decisionmaking under uncertainty or incomplete information.
- • Evolutionary world: Generational dynamics such as genetic inheritance, adaptation, and survival of organisms.

As stated earlier, a major function of the WM is to enable simulative reasoning – where an agent performs a series of thought experiments by simulating the outcomes of plans with the WM, based on which it can choose the best plan. This reasoning approach contrasts with alternative approaches also explored in AI systems, such as logical reasoning inspired by a type of human mental activity that aims to arrive at a conclusion through rigorous inferences or arguments starting from a set of premises and then using stepwise relational formal reasoning (e.g., Lambda calculus) to reach a conclusion supported by these premises; or model predictive control based reasoning used in the process industries in chemical plants and oil refineries to infer optimal action sequence through mathematical programming (e.g., convex optimization) to satisfy a set of constraints. A hallmark of simulative reasoning enabled by a WM is its flexibility, generalizability, and scalability with respect to changing computing resource, memory, environment, and problem complexity, thanks to the WM’s intrinsic ability to simulate all possibilities across domains. In this regard, a WM bears important empirical (e.g., end-to-end experience) similarity to a modern LLM such as ChatGPT

that can operate in a subject-agonistic way in the lingual intelligence space. Indeed, as we discuss later, a general-purpose WM can make use of LLMs as its key building blocks.

Combined with an encoder that estimates beliefs of the world states from arbitrary sensory observations, WM supports machines to perform thought experiments computationally with controlled depth (i.e., number of steps) and width (number of trajectories). AlphaGo [47], for example, can be seen as a special case of simulative reasoning with Monte-Carlo Tree Search (MCTS) using a known (trivial) WM. In the physical world, simulative reasoning enables autonomous vehicles to drive safely by forecasting future street scenarios [e.g., 57, 52] or military commanders to develop battle-winning tactics by anticipating the outcomes of troop movements [51]. WM also enables simulations on different time scales, allowing one to answer questions about billions of years on Earth’s evolutionary path, or just a few moments on a hypothetical Martian civilization.

### 3 The World Model Landscape

Recent work on world modeling have led to a variety of systems, many of which optimized for a specific domain or type of simulation. Interestingly, a commonality nevertheless can be found in these diverse systems in that they all put a significant emphasis on video/image generation, and on visual quality of the generated contents:

Gaming World Models Systems such as Genie 2 [39] (Google DeepMind), Muse [29] (Microsoft), and Oasis [16] (Decart and Etched) simulate video game environments using generative models. These models can render plausible trajectories from visual and action input, producing up to 1-2 minutes of continuous gameplay content. More recently, Genie 3 [6] and Ant Group’s LingBotWorld [54] simulate highly realistic 3D environments with navigation-based actions and promptable world events. Despite their advances, these systems remain domain-specific – for instance, Genie 2 and Muse rely on restrictive console-style inputs, while Oasis is limited to Minecraft-like settings – and action control remains limited, with the inclusion of interacting agents still largely absent. Furthermore, their temporal coherence remains shallow, as current generation horizons (1-2 minutes) fall short of representing full gameplay sessions, which often span several hours. As such, gaming world models lack the flexibility, generality, and long-term reasoning capabilities required for more open-ended, agent-driven tasks.

- 3D Scene World Models World Labs [65] and related efforts focus on stylized 3D scene generation and egocentric navigation. More recently, World Labs released additional details about its system Marble [64], which generates explorable 3D environments from image or text prompts, and Meta published WorldGen [59] with similar functionalities. While visually appealing and demonstrating strong spatial realism, there is currently no clear indication that these models support interactions beyond simple navigation. Richer simulation capabilities appear to rely on exported Gaussian-splat representations, which operate outside the world model itself. This amounts to a program-as-simulator approach: the learned model generates geometric structure, but the dynamics (e.g., how objects move, collide, and deform) are governed by prescribed physics engines, making it closer to a digital twin than a learned world model. Indeed, from available demonstrations, these models simulate static environments without dynamic agents, physics, or rich interactivity. This results in incomplete simulations that are insufficient for tasks involving physical causality, multi-agent behavior, or goal-driven planning. Although these systems push the boundary of spatial realism, they do not support full-fledged world modeling for decision-making or agent learning.

Physical World Models Generative models such as Wayve GAIA-2 [46] and NVIDIA Cosmos [1] are trained specifically on physical control tasks, including autonomous driving, robotic manipulation, and embodied navigation. These systems demonstrate impressive fidelity in modeling low-level

physics and sensory-motor control under diverse conditions (e.g., varying weather, lighting, and geography). More recently, NVIDIA has introduced Cosmos-Predict2.5 [3] and Cosmos 3 [37], while 1X developed 1XWM [25], further improving fidelity and control in autonomous driving and robotic manipulation. Runway ML has also introduced GWM-1 [45], described as a “general world model family,” though based on currently available information it functions more as a collection of specialized models targeting robotics, 3D environments, and avatar-based scenarios. However, these systems remain tightly coupled to their respective domains, often relying on task-specific sensors, data, or control architectures. They excel in their respective constrained settings, but have yet to address the broader challenge of simulating complex, multi-agent, or socially grounded worlds.

Video Generation Models Another popular class of models focus on general-purpose video generation, with recent examples including OpenAI’s Sora [10], Google DeepMind’s Veo [20], and ByteDance’s Seedance 2.0 [11]. These models aim to generate high-quality video sequences from textual prompts and/or prior frames, with the latest systems also supporting multimodal outputs and interactive extension or editing capabilities. While visually stunning, these models only generate fixed trajectories and do not support interactions based on alternative actions. Specifically, they lack explicit notions of state, action, and possibly even object-level or conceptual representation within the video frames. They also provide no simulation control that would allow for reasoning about counterfactual outcomes or evaluate different decisions. Consequently, these systems fall outside the definition of world models for reasoning and planning, and are better understood as contentgeneration tools (focused on pixel-level synthesis) rather than parts of decision-making systems.

Joint Embedding Predictive Models Last but not least, the joint embedding predictive architecture (JEPA) family, including the V-JEPA series [7, 4], DINO-WM [72], and PLDM [50] from Meta FAIR, has attracted significant attention for its conceptually elegant approach to world modeling. These models forego pixel-level generation and instead predict future latent embeddings, often using an encoder-encoder architecture and supervising the outputs in the latent space with energybased losses. While this design promises to improve tractability, the evidence for practical usability remains scarce as these models have mainly been demonstrated in toy environments with simple heuristics and action spaces. The very recent V-JEPA 2 [4] marks a step forward by applying joint embedding prediction to robotic arm manipulation tasks, but it remains unclear whether such models can generalize across more diverse tasks (e.g., making breakfast) or scale to higher-complexity environments with long-term dependencies (e.g., mountaineering).

In summary, although the systems surveyed above have demonstrated significant progress in modeling aspects of the world, most fall short of enabling purposeful reasoning and planning in real world applications due to limitations in scope, abstraction, controllability, interactability, and generalizability. In particular, except for JEPA, contemporary WM systems almost universally emphasize video generation as a core function, yet this emphasis remains largely unexamined and lacks a compelling justification. This focus may reflect the underlying conceptual ambiguities, or even misunderstandings, regarding a fundamental question – what is a world model? We argue that a world model is not fundamentally about generating videos, but about serving as a simulator for reasoning and thought-experiment, which was echoed in recent discussions [32, 33]. Our discussion below is organized around this definition to examine the plausibility and feasibility of current technical approaches to world modeling.

### 4 Critique of World Modeling

Building a general-purpose world model capable of supporting simulative reasoning across diverse domains requires addressing several interrelated design dimensions: what data to train on, how to represent world states, what architecture to use for prediction, what learning objective to

Next Sensory Input

Sensory Input

(video, audio, smell, …)

𝑜

𝑜′

| |
|---|
|Encoder ℎ|

| |
|---|
|Encoder ℎ|

𝑠Ƹ′∗ − 𝑠′Ƹ

𝑠Ƹ′∗

𝑠Ƹ World Model𝑓 𝑠′Ƹ

Fixed-Size, Continuous (Abstract) Sensory

Reconstruction Loss in Latent Space

Deterministic NextEmbedding Prediction

𝑎

Embeddings

- Figure 3: A systematic technical proposal that results from common wisdom on world modeling, most prominently instantiated by the joint embedding predictive architecture (JEPA).

optimize, and how the trained model is used for decision-making. Across the diverse efforts surveyed above, several common wisdoms regarding these dimensions have emerged, which can be summarized

- as follows:

- 1. Sensory inputs should be emphasized over text, because of the larger data volume from the physical world (e.g., a 4-year-old processes approximately 1.114 bytes of visual data, whereas all textual data used to train modern LLMs amount to approximately 0.914 bytes).
- 2. World states should be represented as continuous vectors to enable gradient-based optimization, rather than discrete tokens.
- 3. An encoder-encoder architecture that predicts the next latent state is preferred over autoregressive generation of obervations, to avoid compounding prediction errors.
- 4. Supervision by latent reconstruction objectives is preferable to reconstructing raw observations, due to not modeling irrelevant or unpredictable details.
- 5. The world model should be used for action selection via model-predictive control (MPC) rather than reinforcement learning (RL), for sample efficiency and safety.

These common wisdoms have led to a systematic set of technical proposals, most prominently instantiated in the joint embedding predictive architecture (JEPA) [31]. As illustrated in Figure 3, this line of work is centered around an idea that can be summarized as “next representation prediction” rather than next data prediction:

- • Sensory-First Pretraining: The framework prioritizes continuous sensory data such as video, audio, and other modalities over text.
- • Fixed-Size, Continuous State Embeddings: Given sensory input o, an encoder h estimates the world state sˆ = h(o) as an abstract continuous embedding with fixed dimensions (e.g., sˆ ∈ Rd).
- • Encoder-Encoder Architecture: Based on action input a, the world model f predicts the next state embedding sˆ′ = f(ˆs,a) in a deterministic manner. In particular, the architecture does not use a decoder g to reconstruct the next observation o′, but instead applies the encoder again to bootstrap sˆ′∗ = h(o′) as a ground-truth next state for supervision.
- • Reconstruction Loss in Latent Space: Instead of supervising the world model by the difference between the reconstructed next sensory input oˆ′ and the actual data o′, this frame-

- work bases learning on the deviation between the predicted next state sˆ′ and the bootstrapped ground truth sˆ′∗ (e.g., L2 loss ∥sˆ′ − sˆ′∗∥).
- • Action Selection via MPC: Given current observation ot, the framework favors proposing an initial action sequence (at,at+1,...,aT−1), using the world model f to simulate the next states (st+1,st+2,...,sT), and optimizing the actions based on goal progress Vg(sT).

These design choices represent an intellectually coherent perspective on world modeling and make important contributions – particularly the emphasis on learning abstract, predictable structure rather than reconstructing every low-level detail, and the recognition that a world model should focus on what is learnable and relevant [32]. However, when the goal is a general-purpose world model capable of supporting simulative reasoning across diverse domains, each of these choices involves tradeoffs that merit careful examination. In the following, we analyze each design dimension, survey the landscape of existing approaches, and present our perspective on what a general-purpose world model requires.

##### 4.1 Data: Information Density, Not Just Volume

What data should a world model be trained on? One perspective emphasizes sensory inputs due to their larger raw volume compared to text: as estimated, LLMs are trained on 0.9 × 1014 bytes of textual data, whereas a 4-year-old child has already seen 1.1 × 1014 bytes of vision data [31].

Although sensory data streams such as video appear massive in raw volume, much of that data is low in semantic content and highly redundant [19, 61]. In contrast, natural language is an evolved compression of human experiences, optimized over generations of abstract communication and conceptual reasoning [34, 41].

Text captures not only physical realities but also mental, social, and counterfactual phenomena, which are otherwise difficult or impossible to observe directly [15]. For example, concepts such as justice, motivation, or regret are richly encoded in language but have no direct sensory equivalent. Moreover, language provides an interface to collective human memory (e.g., documented observations, scientific discoveries, engineering failures), which are difficult, if not impossible, to derive from raw perceptual input alone [58]. Indeed, models trained on text can write software [60] or solve Olympiad-level math problems [13], whereas those trained on raw visual and motion data alone have been more suited for physical navigation [62] or manipulation tasks [2].

Thus, the path towards general-purpose world modeling must leverage all modalities of experience, whether it be text, images, videos, touch, audio, or more. Crucially, these modalities are not interchangeable, but reflect different layers of experience (e.g., video captures spatiotemporal dynamics in the embodied and physical world, while language encodes abstract concepts and social norms). A general-purpose world model should not favor any single modality at the expense of others, but rather learn from this stratified structure of experience to generalize across diverse tasks. Ignoring any level, be it low-level perception or high-level abstraction, risks omitting crucial information needed for intelligent behavior.

##### 4.2 Representation: Continuous? Discrete? Or Both?

How should world states be represented? Current approaches (e.g., V-JEPA 2 [4] and Cosmos 3 [37]) typically show a preference for continuous embeddings over discrete tokens, arguing that continuous representations enable smoother gradient-based optimization and richer encoding of perceptual detail.

Do humans perform gradient optimization (e.g., SGD) over continuous neural signals or pattern search (e.g., kNN) over discrete concepts during reasoning? We don’t know for sure. What we do

Sensors

(video, audio, smell, …)

𝑜

𝑜

Scaling Up

| |𝒱|
|---|---|
| | |

|Encoder ℎ|
|---|

|Encoder ℎ|
|---|

|The|brown-fox|jumps-over|the-fence|
|---|---|---|---|

Embeddings

Concept Tokens

Increase vocabulary size

| | |
|---|---|
| | |

|Cat Labrador<br><br>Dog|
|---|

𝑠Ƹ

𝑠Ƹ

Scaling Out

Fixed-Size, Continuous

| |𝒱|
|---|---|
| | |

|𝒱|
|---|

|The|brown|fox|jumps|over|the|fence|…|
|---|---|---|---|---|---|---|---|

Embeddings

Increase code length

Vocabulary

- Figure 4: Vocabulary-based tokens is an effective way to categorize perceptual inputs into discrete concepts for reasoning (left). We may scale up or scale out discrete code to deal with increasing data complexity (right). Thm.1 shows either is effective, but scaling out is more efficient.

know is that reasoning can be cognitive or physiological, or both, and it might be unlikely to have one algorithm fit for all.

While continuous representations allow for smoother gradient flow, an important complementary consideration is the inherent noise and high variability associated with continuous sensory inputs, which can make them brittle for certain forms of reasoning. In effect, fixed-size continuous embeddings derived directly from raw sensory streams are stateless: they fluctuate with noise, viewpoint, and other nuisance factors rather than with the underlying world, and therefore have difficulty with functioning as identifiable states that can anchor reasoning across time. Human cognition has evolved to counter this variability by categorizing raw perception into discrete concepts [8], which are what we typically encode in language, symbols, and structured thoughts (Figure 4, left).

By contrast, we call a representation stateful if it constitutes an identifiable, persistent, and semantic estimate of the world state – one that can be stored in, recalled from, and reasoned over as long- and short-term memory, that remains stable under nuisance variation in the sensory stream, and that is sufficient for predicting the dynamics that matter (i.e., a genuine state in the state-space sense of the belief state sˆ introduced in §2.2). For world modeling, we argue, statefulness is not optional but a must: a representation that cannot be re-identified and carried forward in memory cannot support simulative reasoning over extended horizons.

Vocabulary-based tokens are thus not a liability, but an asset: they offer a stable, composable medium for representing concepts at various levels of abstraction. They form the foundation for designing and building language-based AI systems of today, such as the LLMs, which ground reasoning on sequence of discrete words which are human tokens that correspond to diverse perceptions from the universe (e.g., physical, mental, or social worlds), and allow a form of long-term memory to be employed by implementing (ideally, dynamically controllable) context length. Although not entirely accurate, it is reasonable to consider the space of language as a man-made (through evolution and learning) latent space of the perceived and describable universe that humans live – a substantial subspace of the whole universe. Benefitting from massive text-based pretraining, an LLM can learn to simulate contents in this latent space formed by natural language. Indeed, recent work that represents world states in natural language has seen success for reasoning and planning in a wide range of practical tasks [23, 17]. Complementing natural language tokens, modern techniques like VQ-VAE [55] allow us to further convert sensory data (e.g., images or audio) into discrete tokens while preserving structure and semantics.

While such discrete representations are expected to offer stability and symbolic structure like natural language tokens, a natural concern is whether they can faithfully capture the richness of high-

dimensional, continuous sensory data due to the risk of information loss through distillation. This concern grows with the complexity of the world: Will discrete tokens be sufficient for distinguishing between subtly different world states? Indeed, the world often contains deeper layers of meaning than what is directly observable through sensory input (e.g., a puppet’s movements may reflect the hidden intentions of the puppet master). Capturing such latent structure requires representations that can scale in expressive capacity.

In an attempt to understand more deeply the potential and limitation of using discrete tokens, we present a theoretical result showing that discrete representations can, in principle, preserve arbitrarily fine distinctions between real-valued inputs, provided we scale them appropriately. Specifically, we consider two intuitive strategies for increasing representational capacity:

- • Learning a larger modality tokenizer (scale up): Keep the number of tokens fixed, increase the vocabulary size to allows each token to encode a finer-grained chunk of information.
- • Finding a longer language expression (scale out): Keep the vocabulary fixed, increase the sequence length and combine more tokens to express more complex inputs.

As we show in Theorem 1 below, it is more efficient to scale out by increasing the length of the encoding.

- Theorem 1 (Completeness of Language Representation). Assume real inputs x = [x1,...,xT], where xt ∈ RD and ∥xt∥ < K. For any ϵ > 0, there exists a language Lϵ = (V,N,fϵ) with vocabulary V, maximal sentence length N < ∞, and a mapping function fϵ : RTD → VN such that for all x,x′ ∈ RTD, ∥x − x′∥ > ϵ ⇒ fϵ(x) ̸= fϵ(x′).

Explanation. If you have a sequence of continuous sensor readings or data points, no matter how small a difference you want to be able to distinguish between two sequences, you can always create a language (a system of words or symbols) that can represent these sequences uniquely.

Proof Sketch. We will prove the contrapositive that fϵ(x) = fϵ(x′) ⇒ ∥x − x′∥ ≤ ϵ. Specifically, we propose two ways to scale the discrete code:

- • Case 1 (Learning a Larger Modality Tokenizer). Keep the code length constant at T, increase

the vocabulary size to Mϵ = ⌈

√

TDK˜ϵ˜−1⌉D (i.e., scaling up).

- • Case 2 (Finding a Longer Language Expression). Keep the vocabulary size constant at M,

√

TDK˜ϵ˜−1⌉ (i.e., scaling out).

increase the maximum sentence length to Nϵ = TD⌈logM

| |
|---|

The detailed proof can be found in Appendix A. As the proof shows, complete representations are achievable with vocabulary-based discrete tokens. However, the way to scale the representation matters (Figure 4, right). In Case 1, the vocabulary size must grow in O((TD)D), i.e., exponentially with the input size we’d like to capture, which is likely not sustainable. In Case 2, on the other hand, the sequence length need only increase in the order of O(TD log TD), which is much more manageable. So in theory, scaling out the token sequences, which can be implemented with an enhanced LLM (with visual tokenizers and vocabulary mergers or switchers), offers a more flexible and efficient pathway to capturing complex structure in data generally. In other words, opting for discrete, vocabulary-grounded codes for the sake of statefulness need not come at the cost of expressive fidelity: Theorem 1 guarantees that suitably scaled discrete codes can preserve arbitrarily fine distinctions between world states, answering the information-loss concern raised above. In practice, the model may also represent complex inputs efficiently by dynamically resizing the dictionary, and describe novel inputs by growing the vocabulary with more new observations.

In summary, given that discrete and continuous latent representations offer complementary level of abstraction, representation power, and operationalizability, we advocate for the approach of mixed representations, in which discrete tokens supply the stateful backbone of the world state – a stable, identifiable, and memory-anchored medium for more robust, interpretable, and symbolic forms of reasoning, while continuous embeddings still play a role in capturing fine-grained sensory nuance. While this form of mixed representation is still in its early stages, recent work has demonstrated its promise for generalization in world modeling [67] and other forms of reasoning [e.g., 36].

##### 4.3 Architecture: Pure Abstraction, or Autoregressive Generation

What architecture should a world model adopt? Approaches range from direct next-observation generation to joint embedding prediction in latent space, each involving distinct tradeoffs in grounding, efficiency, and scalability. In this subsection, we analyze two competing approaches, examine their tradeoffs for general-purpose world modeling, and present our blueprint of a generative latent prediction (GLP) architecture as the synthesis.

###### 4.3.1 Two Competing Approaches

The first approach, exemplified by encoder-decoder-based systems such as Genie 2 [39] and NVIDIA Cosmos series [1, 3, 37], prioritizes validation by training video-generation architectures that directly predict the next observation o′ from the current observation o and action a, resulting in a closed loop that checks predictions against observable data.

An influential alternative, represented by the JEPA framework [31], prioritizes abstraction through a non-autoregressive, non-generative, encoder-encoder design that predicts the next latent state directly, sidestepping the need to reconstruct raw observations in an open loop setup. This represents a deliberate and intellectually coherent design choice, motivated by the valid concern that autoregressive observation generation may compound errors over long sequences and waste capacity on irrelevant details. As we analyze below, however, the recursive application of latent prediction is itself functionally autoregressive and generative, and the practical limitations attributed to autoregressive models may be addressed through architectural choices rather than architectural avoidance.

Below, we provide a systematic analysis of JEPA and an alternative architecture we propose that not only inherits important advantages (e.g., abstraction) of JEPA, but also draws inspiration from encoder-decoder approaches (e.g., validation) such as Genie. Formally, JEPA defines two core functions (Figure 5, left):

sˆ = h(o), sˆ′ = f(ˆs,a),

where h is an encoder from observations to latent states, and f is the world model which predicts the next latent state given the current state and an action. Recursive application of these two operators defines a latent transition model that is effectively autoregressive and generative, even though it symbolically lacks an explicit probabilistic decoder to generate what can be compared against real next observation data. (It does not mean such comparisons are avoided, as the second encoder

- at the output end, in fact, indirectly still makes that comparison, but with poorer mathematical controllability, as we discuss in the next section.) More precisely, JEPA can be viewed as specifying a degenerate conditional distribution, denoted informally as below:

pf(ˆs′|s,aˆ ) = δ(ˆs′ − f(ˆs,a)),

where δ(·) is the Dirac delta function centered at the deterministic prediction. Thus, JEPA is not generative in the probabilistic sense (i.e., it does not model uncertainty or samples from a distribution over outcomes), but it is generative in the functional sense of recursively simulating the evolution of the latent states over time, and is therefore subject to the same issues of autoregressive models. This is not to say, however, that autoregressive models are inherently flawed due to error

Encoder-Encoder Architecture 𝑜 ℎ 𝑠Ƹ𝑖 𝑓 𝑠Ƹ𝑖′

𝑜

𝑜′

Enhanced 𝑜′ො

Generative Reconstruction

LLM +

Deterministic NextEmbedding Prediction

|Encoder ℎ|
|---|

|Encoder ℎ|
|---|

Diffusion

|Decoder g|
|---|
| |

𝑠Ƹ′∗

𝑠Ƹ World Model𝑓 𝑠′Ƹ

𝑁𝑠Ƹ 𝑁𝑠Ƹ′

Abstract EmbeddingsFixed-Size, Cont’s, 𝑎

Hierarchical latent spaces for reasoning at higher semantic level

𝑎

- Figure 5: Comparison of the joint embedding predictive architecture (JEPA, left) and the proposed Generative Latent Prediction (GLP) architecture (right). JEPA operates in open loop, predicting the next latent state without reconstructing observations. GLP closes the loop through a decoder that validates predictions against observable data, while incorporating both an enhanced LLM backbone for discrete reasoning and a diffusion-based predictor for continuous dynamics.

accumulation. Many real-world systems (e.g., three-body problem, fluid dynamics, or financial markets) are fundamentally chaotic, with small deviations growing exponentially over time [40]. In such settings, exact prediction is impossible regardless of model class. However, well-structured autoregressive models (e.g., Kalman filters [9] for continuous cases and HMMs [9] for discrete cases) can still learn useful, abstract properties of the system (e.g., whether water will spill, the direction of price movement) that are often surprisingly stable and predictable – an insight grounded in ergodic theory and statistical mechanics [38].

###### 4.3.2 Tradeoffs for General-Purpose Modeling

A common concern with encoder-decoder architectures (which defines an additional function oˆ′ = g(ˆs′) with g as a decoder from latent states back to observations) is that they may compel the model to reconstruct aspects of the environment that are either inherently unpredictable or irrelevant to task performance. Examples often cited include fine-grained visual details, inconsequential events, or out-of-scene content, which may mislead the model into learning unstable or spurious correlations. Proponents of encoder-only architectures thus suggest that by avoiding this reconstruction step, the resulting WM can focus more selectively on predictable and task-relevant elements.

However, the abstraction-first approach introduces a complementary risk: a predictor can only be as good as the representation it operates on. In JEPA, the encoder h is intentionally lossy and noninvertible, discarding information deemed irrelevant. But if the encoder discards information that turns out to matter for the dynamics being predicted, the world model is learning transitions in an ill-formed state space, and no amount of predictor sophistication can recover what was thrown away. Put differently, a lossy, non-invertible encoder cannot yield a stateful representation in the sense of §4.2: the resulting “state” is not a sufficient statistic for the dynamics being modeled, so prediction degrades regardless of predictor capacity. A closed-loop generative architecture provides a built-in mechanism to detect this: if the decoder cannot reconstruct the next observation from the predicted state, the encoder was too lossy. The decoder thus serves not merely as a generation module but as a diagnostic tool that continuously pressures the representation to retain all dynamically relevant information. Without it, supervision occurs solely in the latent space, trading challenges of pixel-level variability for the risk of indefinability: the predicted latents are not directly grounded in observable data, which makes it difficult to diagnose whether the model is learning meaningful dynamics or collapsing to trivial solutions, an issue we discuss formally in §4.4.

More broadly, the architectural divide between open-loop and closed-loop approaches reflects a deeper split in the representation learning community between semantic representations (e.g., CLIP [42] and DINO series [12, 49]), which are optimized to capture invariances across observations and discard the rest, and generative representations (e.g., VAE [30], MAE [24], and diffusion models [26, 43]), which preserve enough variation to reconstruct the full observation space. JEPA’s encoder is semantic by design due to its energy-based loss functions with heuristics-based regularization, which is powerful when the relevant semantics are known in advance, but limiting when the world model must generalize to tasks whose relevant features were not anticipated.

On the other hand, generative representations subsume semantic ones: a sufficiently powerful generative model can develop rich semantic structure as a byproduct of learning to reconstruct observations, as demonstrated by LLMs, which acquire deep semantic understanding through next-token prediction. For a general-purpose world model, therefore, this insight supports adopting a generative architecture: the completeness of generative representations ensures that all dynamically relevant information is retained, while semantic structure can be organized by a dedicated reasoning layer (e.g., an LLM) that has already internalized language-based semantic structure through large-scale pretraining. Information discarded by a semantic encoder cannot be recovered, but information preserved by a generative encoder can always be further abstracted. This complementarity matters especially because what a semantic encoder treats as “important” is determined by its training distribution: phenomena that are rare in the training data (e.g., car crashes in autonomous driving datasets) will be abstracted away, even when they are precisely the scenarios where the world model’s predictions carry the highest stakes.

Finally, a practical argument for abstraction-first architectures has been efficiency due to obviating the need of a separate decoder g, but recent advances are narrowing this gap. Systems such as FastVideo [71, 70], which combines trainable sparse attention with an LLM-based reasoning backbone, can generate 30 seconds of 1080P video in approximately 3 seconds, while architectures such

- as Causal Swin-DPM [66] use chunk-wise causal attention to dampen error propagation and signal variability over long horizons. These developments suggest that the marginal efficiency gain of dropping the decoder does not justify the fundamental information loss.

###### 4.3.3 Generative Latent Prediction as a Synthesis

Therefore, rather than abandoning generative modeling to avoid signal variability, an alternative and well-established strategy is to adopt hierarchical abstraction through what we call a Generative Latent Prediction (GLP) architecture (Figure 5, right). Concretely, GLP consists of an Encoder h and Decoder g that together form a generative bottleneck grounding predictions in observable data, plus a Latent Reasoning Backbone f composed of an enhanced LLM for discrete concept-based reasoning and a diffusion-based next-embedding predictor for continuous perceptual dynamics. Instead of modeling the full world at a single level of detail, GLP decomposes the problem across multiple layers of latent prediction, each specialized for different representational granularities, whether it be continuous perceptual features or discrete conceptual tokens. This allows each layer to operate at an appropriate level of abstraction while remaining generative and predictive. For instance:

- • At the lowest level, next-embedding predictors (e.g., latent diffusion models) can handle stochasticity and fine-grained variation in raw, continuous perceptual data (e.g., pixels, audio, proprioception). These models incorporate generative mechanisms (e.g., encoder-decoder architecture) that directly ground predictions in observable data, which leads to stronger supervision as we show in §4.4.
- • At the intermediate level, a next-token predictor (e.g., autoregressive Transformer decoder) can reason over discrete modality tokens derived via VQ-VAE-style encoders, capturing the

- symbolic and compositional structure.
- • At the highest level, a large language model (LLM) operating in a “thought space” composed of language tokens can support long-horizon planning, mental simulation, and counterfactual reasoning. Together with the intermediate level, these two levels of discrete reasoning can be jointly implemented through an enhanced LLM architecture performing next-token prediction.

The GLP paradigm not only supports structured, abstract reasoning through next-latent prediction, but also preserves the capacity for detailed reconstruction of the input world, enabling generative supervision and external use. This not only mitigates the compounding of prediction errors by isolating low-level variability within the bottom encoder-decoder layer, but also enables more expressive reasoning and generalization at higher layers of abstraction. Importantly, it allows the model to flexibly mix continuous embeddings for perceptual nuance with discrete tokens for abstract structure, which aligns with our discussion of representation in §4.2.

Crucially, GLP does not reject semantic abstraction, but harnesses the fact that generative pretraining naturally gives rise to semantic structure. The LLM backbone, itself a product of large-scale generative pretraining, provides structured abstractions ready-made, while the generative decoder ensures completeness by retaining information the semantic layer may not have known it needed. The result is a system where semantic abstraction is made sufficient by a dedicated reasoning backbone and kept rich by the generative decoder. In this sense, GLP is an inclusive paradigm: it naturally accommodates the growing family of efficient generative techniques, including FastVideo [71, 70] and Causal Swin-DPM [66], which make closed-loop validation practical at scale, while leveraging the powerful abstract reasoning capabilities of LLMs. As we further elaborate in §4.4, this encoderworld-model-decoder design leads to stronger supervision and more stable training dynamics than open-loop approaches, a claim we support formally in Propositions 1–2 and Theorem 2.

##### 4.4 Objective: Learning in Data Space, or Latent Space?

What learning objective should guide world model training? Recent models (e.g., 1XWM [25] and LingBot-World [54]) typically supervise the prediction based on reconstruction error against the observation data. An alternative is latent reconstruction (e.g., JEPA series [7, 4]), which supervises transitions in the learned embedding space rather than reconstructing raw observations, motivated by tractability and the desire to avoid modeling irrelevant details.

A key argument for adopting latent supervision is that modeling a probabilistic distribution over raw observations (e.g., pixels in a video) for reconstruction may be unnecessary or counterproductive, and that learning in latent space can be more effective. This perspective has given rise to energy-based latent reconstruction objectives, which bypass the decoder and directly supervise transitions between encoded states. Formally, given encoder h and world model f, the latent reconstruction loss is defined as:

Llatent(h,f) = E(o,a,o′)∼D [∥f(h(o),a) − h(o′)∥], (7)

where the model predicts the next latent state sˆ′ and compares it to the encoded form of the next observation, without reconstruction o′ itself.

Despite its apparent simplicity, this objective is prone to collapse, as we show in Proposition 1: the model can trivially minimize the loss by mapping all observations to a constant vector and learning an invariant transition. Such collapse is the limiting case of statelessness (§4.2): the learned representation retains no identifiable information about the world state at all. To counteract this tendency, latent reconstruction objectives often require complex regularizers (e.g., maximizing the information I(ˆs) of latent states). These regularizers, however, are often hard to tune and difficult to understand, which can make training brittle and limits scalability. By contrast, the generative

|𝑜′ො − 𝑜′| |
|---|---|

𝑜

𝑜′

𝑜 ℎ 𝑠Ƹ𝑖 𝑓 𝑠Ƹ𝑖′

𝑜′ො 𝑜′

Reconstruction Loss in Latent Space

|Encoder ℎ|
|---|

|Encoder ℎ|
|---|

|Decoder g|
|---|
| |

Generative loss

|𝑠′ − 𝑠′∗|
|---|

Ƹ 𝑠′∗

𝑠Ƹ World Model𝑓 𝑠′Ƹ

𝑁𝑠Ƹ 𝑁𝑠Ƹ′

Ƹ regularizations

𝑎

|𝐼(𝑠)|
|---|

|𝐼(𝑠′∗)|
|---|

𝑎

- Figure 6: Comparison of latent-space reconstruction objectives (left) and generative data reconstruction objectives (right).

reconstruction loss grounds the learning objective in observable data by introducing a decoder g, and supervising the predicted next observation directly as below:

Lgen(h,f,g) = E(o,a,o′)∼D [∥g ◦ f(h(o),a) − o′∥]. (8)

Indeed, the generative loss Lgen anchors the learned representation to the structure of the sensory world, and thus avoids the collapse suffered by the latent loss Llatent, as we show in Proposition 2.

- Proposition 1 (Collapse of Latent Reconstruction Loss). Given O, S, A ⊆ Rd and functions h : O → S, f : S × A → S, and latent reconstruction loss:

Llatent(h,f) = E(o,a,o′)∼D [∥f(h(o),a) − h(o′)∥],

There exists (h∗,f∗) and c ∈ S, such that h∗(o) = c for all o ∈ O and f∗(c,a) = c for all a ∈ A, such that:

Llatent(h∗,f∗) = min

h,f

Llatent(h,f)

Explanation. If you have an encoder and a world model, and you train it using the latent reconstruction loss, there is a cheat configuration for the model to minimize the loss while learning nothing about the true dynamics.

Proof Sketch. If we construct such a degenerate solution (h∗,f∗), this solution satisfies Llatent(h∗,f∗) = 0, which is a global minimum as Llatent(h,f) ≥ 0 for all (h,f).

| |
|---|

- Proposition 2 (Non-Collapse of Generative Loss). Given functions h : O → S, f : S × A → S, g : S → O, and generative loss:

Lgen(h,f,g) = E(o,a,o′)∼D [∥g ◦ f(h(o),a) − o′∥],

Assuming ∃(o1,a1,o2), (o3,a3,o4) ∈ D such that o2 ̸= o4, then given (h′,f′), fixed g′ and c ∈ S such that h′(o) = c ∀o ∈ O and f′(c,a) = c ∀a ∈ A, there exists (h,˜ f˜) such that:

Lgen(h,˜ f,g˜ ′) < Lgen(h′,f′,g′)

Explanation. If you add a decoder to the model and train it with the generative loss, assuming the data contains different next-observation targets, there will always be another set of encoder and world model that gets lower loss than the previous cheat configuration.

Proof Sketch. Given degenerate solution (h′,f′) and fixed g′, construct (h,˜ f˜) to be equal to (h′,f′)

- at every point except (o1,a1,o2) and (o3,a3,o4) where o2 ̸= o4 and the constant-valued (h′,f′) will get non-zero loss. Instead, set (h,˜ f˜) to perfectly fit these two targets, so they will get zero loss. Thus we have:

Lgen(h,˜ f,g˜ ′) < Lgen(h′,f′,g′)

| |
|---|

Details of the proofs of both propositions are available in Appendices B and C, respectively.

Beyond the issue of collapsing, a more fundamental structural limitation of the latent reconstruction objective is that it essentially acts as a loose surrogate for observation-level consistency, as we show in Theorem 2. This means that minimizing Llatent does not, in general, guarantee consistency with what the agent would observe in the world, which can lead to misaligned or brittle representations. In general-purpose settings, we argue that anchoring to the next observation o′ via generative loss provides a more stable and mechanistically interpretable training signal.

- Theorem 2 (Latent reconstruction is an upper-bounded surrogate for generative reconstruction). Given sufficiently powerful encoder h : O → S and decoder g : S → O, such that for all latent states sˆ ∈ S, the roundtrip reconstruction error satisfies ∥h ◦ g(ˆs) − sˆ∥ ≤ ϵ for some small ϵ > 0. For world model f : S ×A → S and transition data (o,a,o′) ∼ D, define the latent-space loss Llatent and the generative loss Lgen as below:

Llatent = ∥f(h(o),a) − h(o′)∥, Lgen = ∥g ◦ f(h(o),a) − o′∥. Assume that encoder h, decoder g, and world model f induce the following conditional distributions:

sˆ | o ∼ N(h(o),I), oˆ | sˆ ∼ N(g(ˆs),I), and sˆ′ | s,aˆ ∼ N(f(ˆs,a),I). Then, the following inequality holds:

Llatent ≤ Lgen + ϵ, with Llatent = Lgen when h ◦ g(ˆs) = sˆ for all sˆ ∈ S and supp(D) ⊆ Im(g).

Explanation. If your encoder and decoder approximately undo each other, then the JEPA latent reconstruction loss is upper-bounded by the generative loss plus a small reconstruction error. These losses are only equal when the encoder-decoder pair perfectly invert each other, which is unrealistic in practice. As such, minimizing the latent loss does not guarantee consistency with observed data, which is required for minimizing the generative loss (Figure 7).

Proof Sketch. Observe that the two losses Llatent and Lgen are scaled KL divergences of the Gaussian prediction distributions in the latent and observation spaces, respectively. Apply the encoder h to both the observation prediction and true observation distributions in Lgen, and the data processing inequality states that the augmented KL divergence is upper-bounded by the original Lgen. After that, apply the triangle inequality to show that Llatent is upper-bounded by the sum of this augmented KL (upper-bounded by Lgen) and the roundtrip reconstruction error (upper-bounded by ϵ), thus completing the proof. (We include the detailed proof in Appendix D.)

| |
|---|

In practice, ϵ is small (as is typical for strong modern autoencoders), so Llatent ≤ Lgen usually holds, meaning that the former can miss semantically important mistakes that the latter will penalize. Additionally, the use of less-understood regularizers in conjunction with Llatent as the objective makes its outcome even more difficult to assess without necessary boundary conditions imposed by observation data.

|ℒgen + 𝜖|
|---|
| |
|ℒgen|

Encoder-decoder reconstruction error (small)

|ℒlatent ≤ ℒgen + 𝜖| |
|---|---|
| |ℒlatent|
| | |

Task-relevant information missing from encoder

- Figure 7: As Thm. 2 shows, the energy-based latent reconstruction loss (Llatent) is upper bounded by the generative data reconstruction loss (Lgen) plus a small encoder-decoder reconstruction error (ϵ). ϵ is small in practice, meaning Llatent ≤ Lgen usually holds. Minimizing Llatent, therefore, does not guarantee consistency with observed data, which is required for minimizing Lgen.

More broadly, all representation necessarily involves compression; the question is not whether to compress, but how to avoid premature loss of task-relevant information. The emphasis on learning abstract, predictable structure (e.g., JEPA) is well-founded, as a model should indeed focus on what is learnable and avoid squandering capacity on irrelevant noise [32]. Our perspective extends this insight by arguing that generative objectives provide a natural safeguard: by requiring the model to reconstruct observations, the learning process itself determines what information is worth retaining, rather than relying on a priori design decisions about which details matter.

In conclusion, our argument is not that world models must operate in pixel space, but that they should learn from it. Framing the distinction as next-representation prediction versus next-observation

prediction creates a false dichotomy that can lead to theoretical ambiguities and practical instability. The purpose of predicting the next observation is to ensure that the predicted latent representations are meaningfully grounded in the real world, whether conceptually or physically. Conversely, reliable prediction in latent space depends on continual validation through observable data. Mathematically, any latent representation of real-world signal intrinsically suffers from issues of identifiability and stability. As such, alignment and calibration with real data are essential to ensure the representations remain meaningful and robust. Generative reconstruction objectives tether the learned representations to the observable world, providing richer and more stable learning signal that supports meaningful distinctions, general usability, and human interpretability. These properties are critical for downstream usage, whether it be planning trajectories or training agents through reinforcement learning, which we discuss more in §4.5 below.

##### 4.5 Usage: MPC or RL?

How should a trained world model be used for decision-making? There has also been debate over whether model-predictive control (MPC) is favored over reinforcement learning (RL) in using the WM for reasoning, due to sample efficiency and safety advantages [18]. Here we describe a typical MPC setup (Figure 8, left) which is often adopted by recent work [50, 4]: at timestep t during inference, the agent infers its current latent state sˆt = h(ot), proposes an initial sequence of actions (at,...,aT−1) until some decision horizon T, and uses the world model to predict the corresponding next-state sequence (ˆst+1,...,sˆT). These simulated states can then be evaluated using a cost function C(g,sˆ) for goal g (e.g., L2 distance between sˆ and encoded goal sˆg = h(g)), based on which the agent may propose the next action with lower cost. Decision-making thus amounts to finding the action sequence that minimizes the cost function, formalized as below:

(a∗t,...,a∗T−1) = arg min

at,...,aT−1

T−1

C(g,f(ˆsk,ak)).

k=t

𝑜1

Separate agent model selects action

World model simulates experience

𝑜1

Optimize actions based on objective

Simulate next latent states using WM

ℎ 𝑠Ƹ1𝑖 𝑓 𝑠Ƹ2𝑖 𝑎1

|Encoder ℎ| |
|---|---|
| | |

𝑓 …

𝑠Ƹ𝑇𝑖

|…| |
|---|---|
| | |

𝑠Ƹ2 𝑓

𝑠Ƹ𝑇

𝑠Ƹ1 World Model𝑓

𝑁1 𝑁2

𝑁𝑇

Simple action initialization

|Goal𝑔|
|---|

𝑎1

|Goal (𝑔)|
|---|

𝑎1

𝑎2

Agent Model (𝜋)

𝜋

Optimize agent model based on goal with RL

- Figure 8: Comparison of model-predictive control (left) and reinforcement learning from worldmodel-simulated experience (right) as approaches to using the world model for decision-making.

In practice, the (continuous) action optimization is often performed using traditional numerical algorithms (e.g., MPPI [63] and CEM [44]) involving decision horizons of 1-20 steps and 100s upon 1000s of action samples, and the agent executes the first action in the final action sequence a∗t before replanning in the next step t+1. The appeal of MPC lies in learning from offline trajectories (o1,a1,...,oT) ∼ D without potentially unsafe exploration in the real world, as well as potential for higher-quality decision-making from world-model-based simulation.

However, MPC can suffer from practical limitations. Simulation of latent trajectories using the world model, for instance, must be performed repeatedly at every timestep during inference, leading to high computational overhead and making it difficult to respond effectively in fast-changing environments. Beyond computational efficiency, MPC typically plans only a few steps ahead (e.g., up to 10-20 steps) in terms of searching horizon. This limits the extent of its foresight, as long planning horizons (e.g., 100s of steps) can be difficult due to the exploding number of trajectories and world-model errors. As the horizon increases, MPC also becomes more difficult to implement and optimize, since the proposal distribution must sample entire action sequences at once over the full planning horizon. This is why MPC often relies on relatively simple proposal distributions, such as uniform random sampling or multivariate Gaussians. Indeed, MPC so far has shown promise primarily in simplified settings (e.g., Go) where environment dynamics are simple and slower decision-making is rewarded, but struggles to extend to real-world tasks (e.g., customer service), which typically involve complex dynamics and require a mixture of short- and long-term decision-making.

On the other hand, RL is a general, flexible, and scalable approach to training agents without restrictions on the decision-making method or search horizon. In particular, one can replace the true universe with a world model f for exploration and learning (as discussed in §2.2). Below we describe an RL setup (Figure 8, right) where the agent interacts with the world model instead of the environment [22]: in each time step t with world state representation sˆt (which may be encoded from some observation data ot or completely imagined from scratch), the agent π takes action at ∼ pπ(at | sˆt) and the world model f simulates the next state sˆt+1 ∼ pf(ˆst+1 | sˆt,at). This may repeat until some rollout horizon T or in a never-ending manner. Computing the reward at each step r(g,sˆt) based on goal g, the optimal agent πf∗ may thus learn by maximizing the expected discounted cumulative reward (with well-formed discount schedule {γk}∞k=t to ensure numerical

stability) as defined below:

πf∗(ˆst) = arg max

Eπ,f

π

= arg max

π

∞

γkr(g,sˆk) s ˆt (9)

k=t

T

T

pf(ˆsi+1 | sˆi,ai). simulation with world model

pπ(ai | sˆi) select action

lim

γkr(g,sˆk)

T→∞

i=t

k=t

(at,sˆt+1,...,sˆT )

Operationally, as we show above, both MPC and RL can use world models, the former only for decision-making while the latter also for learning. We recognize the latter as part of a broader paradigm: learning from experience [27]. In this framework, the agent model continually interacts with and learns from an infinite space of imagined worlds, simulated by a world model. The countless hypothetical trajectories can then be used to train the agent via RL, imitation learning, or other learning signals that make full use of all experiences. These updates can occur entirely offline, using batches of rollouts from the world model rather than interacting with the real environment.

Compared to MPC which is computationally expensive at decision-making time, RL with world model (as in Equation 9) shifts part of the computational cost to the training phase. Instead of planning from scratch at each step, the world model is used offline to train a policy network that can later be reused for fast action selection at every state. Crucially, Both RL and MPC along with the world model may be included as components inside the agent model that must carry both deliberate planning and reactive actioning, while another fast policy can still learn to react swiftly when needed. Whereas recent work like o1, o3, and R1 [21] can be seen as special cases in math and coding where model-free policy-based methods enables fast-reacting behaviors, our view is to generalize this pattern: Agents should both reason with and learn from the worlds that they simulate, allowing flexible decision-making, continual improvement, and emergence of intelligence with experience.

In summary, as demonstrated above, unlike MPC, RL can learn a policy function that reflects long-term cumulative rewards, enabling more strategic reasoning over extended time horizons. This makes it applicable in practical settings like goal-conditioned robotic manipulation, multi-turn dialog systems, or autonomous driving.

### 5 The PAN World Model

Drawing from the examination of world model design dimensions above, we arrive at the following conclusion regarding design principles for a general-purpose WM capable of supporting simulative reasoning: 1) use data from all modalities of experience; 2) employ a stateful, mixed continuous and discrete representation; 3) adopt a hierarchical Generative Latent Prediction (GLP) modeling paradigm with an extended-LLM backbone (for discrete concept-based reasoning), as well as a generative embedding predictive module (for continuous gradient-based reasoning), as the reasoning engines; 4) train over a generative loss grounded in observation data; and 5) apply world model to simulate experiences for training agents using reinforcement learning. We conclude our examination with a brief preview of a new architecture, PAN – a Physical, Agentic, and Nested world model, based on the aforementioned design principles. Details and preliminary results of PAN will be presented in [66].

##### 5.1 A Motivating Usecase

A truly versatile and generalizable WM must be grounded in tasks that reflect the full complexity of real-world reasoning demands. These may include variations in data modality (e.g., verbal, visual,

Current Observation Predicted Observations

Reconstruct

Sensory Observations

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

perceptual inputs for generative supervision

…

(Video, Sound, Temperature, Pain, …)

and external usage

Sensory Encoder (Tokenizer + Embedder)

Multimodal Decoder

Short-Term Consistency

…

| | | | |
|---|---|---|---|
| | | | |

Implicitly simulated world by hierarchical latent

Estimated World State

…

representations for

(Objects, Tools,

agentic reasoning

Agents, Emotions, …)

World Model Backbone (Enhanced LLM + Diffusion Embedding Predictor)

Long-Term

Consistency

…

| | | | | |
|---|---|---|---|---|
| | | | | |

Simulated Action

Ask Isabella to lend a hand…

Apply pressure on the icepick…

Explore a route to go down…

Share lessons with others…

…

(Motion, Interaction, Logistics, Strategy, …)

- Figure 9: Illustration of the generative latent prediction (GLP) architecture of PAN-World, our proposed framework for general-purpose world modeling. At each time point, PAN-World estimates the world state sˆ based on the current sensory observation o, predicts the future world states sˆ′ based on proposed actions a for agentic reasoning, and reconstructs the future observations oˆ′ for generative supervision and external usage.

sensory), spatio-temporal scope (from one second in a room to days in a whole country), action granularity (e.g., fine motor control, bodily movement, expressive gestures), and decision scale (from immediate actions, tactics, to long-term strategies). While many existing WMs are demonstrated on simplified, toy tasks (e.g., manipulating kitchen tools) and simple scenarios (seconds to minutes of video in 3D worlds), such settings fall short of capturing the richness of real-world agentic experience. A WM designed around these tasks, therefore, is unlikely to scale to complexities required for real-world applications. For instance, a WM that only enables tool manipulation in a kitchen is inadequate for planning and executing an end-to-end dinner service in a restaurant.

In contrast, PAN is motivated by a more complex and realistic use case: a mountaineering expedition. In this setting, the WM must internalize multimodal sensory inputs and simulate future world states in service of a demanding, structured task. This task naturally decomposes into multiple interconnected sub-tasks at different levels: high-level decisions like gear selection, route and segment planning, navigation, weather assessment, pacing, etc., low-level actions like climbing, roping, and precise motor control in response to terrain and surface conditions, and social coordination with teammates through verbal and non-verbal communication, among others.

The mountaineer’s sensory experience includes not only sights and sounds – snowfields, cliffs, a partner calling out from ahead – but also tactile and motion signals such as wind, cold, and muscular strain. The actionable world states that drive purposeful reasoning, such as terrain affordances, team dynamics, or latent risks, exist at multiple levels of abstraction beneath them. PAN thus begins by ingesting this continuous stream of multimodal signals: inputs from vision, sound, temperature, motion, potentially even pain, which may each be relevant for different tasks but together constitute a holistic reality.

##### 5.2 The GLP Architecture for PAN World Model

Following a mixed representation and multi-scale reasoning principle, PAN processes multimodal sensory inputs o using its Sensory Encoder (h), which maps inputs via both discrete and continuous pathways to capture complementary aspects of the world. On one hand (Figure 9), a tokenizer hierarchically maps raw signals into discrete tokens grounded in PAN’s vocabulary, which spans

multiple levels of abstraction. These tokens may consist of abstract tokens learned via a VQ-VAEstyle approach [55] as well as concrete words drawn from natural language. The representation may include a flexible number of such tokens to compactly reflect deep layers of world information: Where am I? Who is with me? What tools do I have? What is my emotional state? This tokenbased component of the world state is stateful by construction: grounded in a persistent vocabulary, each token is identifiable and re-addressable, allowing the estimated state to be held in memory, queried, and updated across the long horizons of the expedition rather than recomputed transiently from each new sensory frame. As discussed in §4.2, this form of representation can be sufficient to capture relevant information, even for continuous data like video. On the other hand (Figure 5, right), PAN may also encode low-level details into continuous latent embeddings to capture the full nuanced perceptual experience where necessary. Together, these tokens and embeddings form a layered estimate of the world state sˆ = {sˆi}Ni=1 over which PAN performs simulation and purposeful reasoning.

Given a proposed action a (e.g., “buckle the carabiner to my harness”), PAN predicts the next world state sˆ′ (e.g., a conceptual state like “I am safely anchored”, or a physical state like “that rope is tightening”) using a World Model Backbone (f) built on an enhanced LLM and a diffusionbased next-latent-embedding predictor (Figure 9). This design is a concrete instantiation of the GLP architecture introduced earlier in §4.3 (Figure 5, right). The LLM-based backbone reasons over both natural language tokens and a learned conceptual vocabulary – some explicit (e.g., a particular shape of icepicks), others implicit or emergent (e.g., feelings that arise when sharing hardearned knowledge). This supports broad generalization across domains [17]. During both training and inference, the model can also dynamically extend its vocabulary by introducing new tokens or merging existing ones to maximize the prediction quality.

On the other hand, the diffusion-based embedding predictor is responsible for fast, low-level, and often subconscious reasoning that are critical for embodied responses yet difficult to express in language. This module simulates detailed perceptual experiences, such as whether a foothold is secured, or how the body might shift its weight during a climb [67]. A Learned Switch allows PAN to predict the next world state hierarchically ({sˆ′i}N

′

i=1) by adaptively combining the LLM-based backbone, multiple vocabularies, and the diffusion-based embedding predictor, depending on task demands. These mechanisms enable PAN-WM to scale across spatio-temporal scopes and action granularities as is required for general usability – from concrete physical scenarios like mountain climbing and social interaction, to abstract, far-reaching strategic consequences like nationwide policy changes.

To supervise its predictions, and to allow the trained WM to interface with external agents (or human) who may use its outputs, PAN reconstructs the next observation oˆ′ using a Multimodal Decoder (g) and compares it to the actual observation o′. Crucially, the decoder’s outputs are not limited to videos, but includes a full sensory experience, which may include sound, temperature, motion, pain, other embodied signals, and/or even text. As discussed in §4.3 and §4.4, this generative supervision grounds the predicted world state sˆ′ in sensory reality, ensuring that the representation retains all possible information while allowing residual variability to be absorbed by the decoder g. This approach contrasts sharply with models trained on next-representation prediction (e.g., V-JEPA 2 [4]), which supervise the world model purely in latent space. The latter objectives are, at best, loose surrogates of generative objectives and prone to representation collapse or unidentifiability, as they lack grounding in real sensory input. Formally, PAN models the conditional distribution of the next

observation o′ given the current observation o and proposed action a as below: pPAN(o′ | o,a)

pf(ˆs′ | s,aˆ ) world model

pg(o′ | sˆ′) decoder

ph(ˆs | o) encoder

=

s, ˆ sˆ′

N′

N

pg(o′ | sˆ′)

pf(ˆs′j | sˆ′<j,s,aˆ )

ph(ˆsi | sˆ<i,o)

=

s, ˆ sˆ′

i=1

j=1

hierarchical world state inference

switch-based next-state prediction

Overall, with its hierarchical, multi-level, and mixed representation architecture, and an encoderdecoder pipeline that threads perception o, action o, belief sˆi, simulated belief sˆ′i, and simulated worlds o′, PAN represents a general-purpose generative model for simulating actionable real-world possibilities for an agent to perform purposeful reasoning, as we will briefly allude to in §5.4. PAN does not sidestep the variability in raw perceptual input, but instead modularizes and organizes it. This enables richer internal simulation of every layer of experience for more powerful agent reasoning and planning.

##### 5.3 Training the PAN World Model

It should be obvious from the mountaineering example that simply watching videos is not enough to learn all the reasoning capability needed to accomplish the final goal, which can take days of time and hundreds upon thousands of actions and steps from the onset, and is built on rich background knowledge about geography, climate, equipments, sports, and even history. The training of PANWM shall use a divide-and-conquer approach that begins with pretraining each of its modules independently through self-supervision (e.g., LLM for text data, and diffusion model for video data). These modality- and level-specific modules are then aligned or integrated during the post-training phase using multimodal data, cascaded embedding, and gradient propagation. Modules operating over continuous embeddings can be trained using standard gradient-based optimization techniques. In contrast, components using discrete tokens may benefit from gradient-free methods similar to reinforcement learning [21]. As proven in §4.4, generative, data-reconstruction-based objectives are grounded in observed data and provide a stable and reliable learning signal for the entire system.

A key strength of the PAN architecture is its data efficiency, because of its use of a multi-scale and stratified view of the world. In the mountaineering task, when reasoning about navigation and path-finding, the world states do not need to include snow or rock surface details at pixel level, whereas when deciding where to lay hands or feet during climbing, the world states can ignore geographical contexts. Therefore it is not necessary for WMs simulating highly complex possibilities to be contingent on data that capture all such complexity all at once (e.g., videos that visually cover mountaineering at all levels), but to take advantage of data of different kinds offering information at different levels (e.g., travel book for trail guide and map reading, indoor video for rock climbing and gear usage). After all, it is unrealistic to expect a large corpus of videos that comprehensively cover all aspects of alpine climbing. Many general capabilities (e.g., social reasoning, travel planning, cold weather survival) can be learned from abundant language data. Only directly embodied skills (e.g., foot placement, rock climbing technique) require physical data like videos or proprioception, which can be obtained in controlled or simulated environments.

Indeed, PAN’s pretrain-then-align/integrate strategy enables sensory information (e.g., from a video diffusion model) to be grounded within higher-level, richer contexts through LLMs, thereby facilitating cross-modal generalization. At the same time, abstract knowledge embedded in LLMs can be anchored to concrete, embodied experiences, increasing the precision and realism of the system’s

Expected Reward

|Goal|
|---|

###### +

| |
|---|

WM

Predicted

WM

Future States

Action

Action

|Goal|
|---|

+

WM

Possibles

World Model (WM)

WM

World States

Action

|Goal|
|---|

o

WM

Action

Action

Action

WM

Select precomputed plans based on belief and expected reward

|Goal|
|---|

−

WM

Planner Action

Action

[Figure 18]

Estimated

World State

#### Simulative Reasoning Agent

Perception

Output

[Figure 19]

Action

- Figure 10: Illustration of our proposed simulative reasoning agent powered by the PAN World Model. Unlike traditional RL agents that rely on reactive policies, or model-precitive control (MPC) agents that expensively simulates futures at decision time, this agent leverages a cache of precomputed simulations generated by the PAN WM. During decision-making, the agent selects actions based on its current beliefs and expected outcomes, enabling a more efficient, flexible, and intentional form of planning, which, as we argue, is closer to the flexibility of human reasoning.

reasoning [68]. The result is a WM that, like humans, derives commonsense understanding from a diverse set of experiences. Consequently, it does not require exhaustive training data for each specific task (e.g., mountaineering or autonomous driving), but can instead draw on conceptual knowledge acquired from many domains. We believe this kind of general-purpose WM is well-suited to simulating experience for agent decision-making and/or training, as elaborated below.

##### 5.4 Towards Agentic Reasoning with PAN

Recall in §2 we outlined an agent architecture for simulative reasoning using the world model. PAN fits naturally into this paradigm, functioning not merely as a video generator, but as a rich internal sandbox for simulation, experimentation, and foresight.

As illustrated in Figure 10, a PAN-Agent, prompted by a goal and in reception of continuous stream of perception from the real world, is expected to come up with actions, plans (sequence of actions), or strategies (plans in light of counterfactual situations), which would involve using the PAN-WM to precompute and cache a diverse set of possible world states, plausible actions within those states, and their simulated outcomes [14]. At decision time, rather than only relying on performing expensive real-time simulations, the agent may consult this cache and select actions based on current beliefs and expected rewards. This decoupling of simulation from action selection allows the agent to reason more deliberately, adaptively, and selectively, avoiding the rigidity of purely reactive policy in endto-end RL and the computational burden of constant forward rollout in MPC. The result is an agent that more closely mirrors human-like cognition – planning ahead, navigating uncertainty, and acting

with both flexibility and foresight. We believe this represents a promising step towards agents with richer forms of agency – one capable not only of simulative reasoning, but also of choosing among imagined futures with intent. Such agents may ultimately approach the adaptability, resilience, and autonomy characteristic of human intelligence.

### 6 Conclusion

We have examined the foundations, debates, and practical challenges in the pursuit of generalpurpose world modeling. Our intent in writing this critique is to inspire further discussions and deeper reflections on the following fundamental questions: what is indeed a world model?, what is a world model for, and how to build a world model of practical and general utility?

We argue that a WM is not about video or virtual reality generation, but is about simulating all possibilities in real world; and such outcome is not for visual pleasure, but for simulative reasoning; and current paradigms and efforts towards this end remain primitive.

It is our wish that, by offering a constructive and analytical examination of the key dimensions of world model design, surveying the diverse perspectives in the field, and presenting our alternative proposal – the Generative Latent Prediction (GLP) architecture, we can spark further advancements in both theory and implementation of stronger world models.

The PAN world model we previewed is a concrete instantiation of GLP, designed for simulating all possible worlds and enabling agent reasoning and planning. By combining multimodal data, stateful, hierarchical representations, multi-level generative modeling, and observation-grounded objectives, PAN supports long-horizon simulation and flexible decision-making across tasks in the physical and cyber domains.

It has been recognized that world models should serve as core components of agentic planning and decision-making, as echoed in recent work [33, 37]. We, however, argue that how to act on worldmodel simulations is the function of a separate agent model, as previewed in §5.4. Moreover, the agent’s relationship to the world model goes well beyond planning and action selection, and we believe that these capabilities should arise as properties internal to the agent itself, rather than externally prescribed procedures. We address these questions in a forthcoming companion manuscript [69].

Looking ahead, the PAN framework opens several promising directions: scaling from single-agent to multi-agent simulations (e.g., collective behaviors of a business, a society, consequences to public health), extending across time scales (e.g., from milliseconds to millennia), improving simulation fidelity across modalities, and enabling agent learning directly through imagined experience. As world models increasingly serve as the substrate for reasoning, imagination, and action, we believe that frameworks like PAN, with its experience grounding, multi-layer abstraction, and empirical scalability, offers a compelling foundation for the development of robust, general-purpose AI.

### References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [2] Figure AI. Helix: A vision-language-action model for generalist humanoid control, February

2025. Accessed: 2025-05-01.

- [3] Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025.
- [4] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning, 2025.
- [5] Linden J. Ball. Hypothetical Thinking, page 514–528. Cambridge Handbooks in Psychology. Cambridge University Press, 2020.
- [6] Philip J. Ball, Jakob Bauer, Frank Belletti, Bethanie Brownfield, Ariel Ephrat, Shlomi Fruchter, Agrim Gupta, Kristian Holsheimer, Aleksander Holynski, Jiri Hron, Christos Kaplanis, Marjorie Limont, Matt McGill, Yanko Oliveira, Jack Parker-Holder, Frank Perbet, Guy Scully, Jeremy Shar, Stephen Spencer, Omer Tov, Ruben Villegas, Emma Wang, Jessica Yung, Cip Baetu, Jordi Berbel, David Bridson, Jake Bruce, Gavin Buttimore, Sarah Chakera, Bilva Chandra, Paul Collins, Alex Cullum, Bogdan Damoc, Vibha Dasagi, Maxime Gazeau, Charles Gbadamosi, Woohyun Han, Ed Hirst, Ashyana Kachra, Lucie Kerley, Kristian Kjems, Eva Knoepfel, Vika Koriakin, Jessica Lo, Cong Lu, Zeb Mehring, Alex Moufarek, Henna Nandwani, Valeria Oliveira, Fabio Pardo, Jane Park, Andrew Pierson, Ben Poole, Helen Ran, Tim Salimans, Manuel Sanchez, Igor Saprykin, Amy Shen, Sailesh Sidhwani, Duncan Smith, Joe Stanton, Hamish Tomlinson, Dimple Vijaykumar, Luyu Wang, Piers Wingfield, Nat Wong, Keyang Xu, Christopher Yew, Nick Young, Vadim Zubov, Douglas Eck, Dumitru Erhan, Koray Kavukcuoglu, Demis Hassabis, Zoubin Gharamani, Raia Hadsell, A¨ron van den Oord, Inbar Mosseri, Adrian Bolton, Satinder Singh, and Tim Rockta¨schel. Genie 3: A new frontier for world models. 2025.
- [7] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024.
- [8] Lisa Feldman Barrett. How emotions are made: The secret life of the brain. Pan Macmillan, 2017.
- [9] L. E. Baum and T. Petrie. Statistical inference for probabilistic functions of finite state markov chains. The Annals of Mathematical Statistics, 37(6):1554–1563, 1966.
- [10] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. 2024. URL https://openai. com/research/video-generation-models-as-world-simulators, 3:1, 2024.
- [11] ByteDance Seed. Seedance 2.0, 2026. Accessed: 2026-02-22.

- [12] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.
- [13] Yuri Chervonyi, Trieu H Trinh, Miroslav Olsˇ´k, Xiaomeng Yang, Hoang Nguyen, Marcelo Menegali, Junehyuk Jung, Vikas Verma, Quoc V Le, and Thang Luong. Gold-medalist performance in solving olympiad geometry with alphageometry2. arXiv preprint arXiv:2502.03544, 2025.
- [14] Brandon Chiou, Mason Choey, Mingkai Deng, Jinyu Hou, Jackie Wang, Ariel Wu, Frank Xu, Zhiting Hu, Hongxia Jin, Li Erran Li, Graham Neubig, Yilin Shen, and Eric P. Xing. Reasoneragent: A fully open source, ready-to-run agent that does research in a web browser and answers your queries, February 2025.
- [15] Terrence W Deacon. The symbolic species: The co-evolution of language and the brain. WW Norton & Company, 1998.
- [16] Decart, Julian Quevedo, Quinn McIntyre, Spruce Campbell, Xinlei Chen, and Robert Wachen. Oasis: A universe in a transformer. 2024.
- [17] Mingkai Deng, Jinyu Hou, Zhiting Hu, Graham Neubig, Hongxia Jin, Yilin Shen, and Eric P. Xing. Simura: Towards general goal-oriented agent via simulative reasoning architecture with llm-based world model, 2 2025.
- [18] Chelsea Finn and Sergey Levine. Deep visual foresight for planning robot motion, 2017.
- [19] Rafael C Gonzales and Paul Wintz. Digital image processing. Addison-Wesley Longman Publishing Co., Inc., 1987.
- [20] Google DeepMind. Veo. https://deepmind.google/models/veo/, 2025. Latest release version: Veo 3 (May 20, 2025), accessed: 2025-06-10.
- [21] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [22] David Ha and Ju¨rgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.
- [23] Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. Reasoning with language model is planning with world model. arXiv preprint arXiv:2305.14992, 2023.
- [24] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´r, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000–16009, 2022.
- [25] D HO, J MONAS, JT REN, and C YU. 1x world model: evaluating bits, not atoms, 2025.
- [26] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [27] Zhiting Hu and Eric P Xing. Toward a standard model of machine learning. arXiv preprint arXiv:2108.07783, 2021.
- [28] Philip N Johnson-Laird. Mental models and human reasoning. Proceedings of the National Academy of Sciences, 107(43):18243–18250, 2010.

- [29] Anssi Kanervisto, Dave Bignell, Linda Yilin Wen, Martin Grayson, Raluca Georgescu, Sergio Valcarcel Macua, Shan Zheng Tan, Tabish Rashid, Tim Pearce, Yuhan Cao, et al. World and human action models towards gameplay ideation. Nature, 638(8051):656–663, 2025.
- [30] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.
- [31] Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1):1–62, 2022.
- [32] Yann LeCun and Eric Xing. How should ai learn to understand the world? yann lecun & eric xing on jepa and glp, 2026. YouTube video; debate at Spring School AI for Impact 2026, Ben Guerir, Morocco, March 25, 2026.
- [33] Fei-Fei Li. A functional taxonomy of world models. X post, June 2026. Accessed: 2026-06-05.
- [34] David JC MacKay. Information theory, inference and learning algorithms. Cambridge university press, 2003.
- [35] Willis E. McNelly, editor. The Dune Encyclopedia. Berkley Publishing Group, New York, 1984. Published by arrangement with the author. Designed by Jeremiah B. Lighter.
- [36] Chancharik Mitra, Brandon Huang, Trevor Darrell, and Roei Herzig. Compositional chain-ofthought prompting for large multimodal models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14420–14431, 2024.
- [37] NVIDIA. Cosmos 3: Omnimodal world models for physical ai. arXiv preprint arXiv:2606.02800, 2026.
- [38] Donald S Ornstein and Benjamin Weiss. Statistical properties of chaotic systems. Bulletin of the American Mathematical Society, 24(1):11–116, 1991.
- [39] Jack Parker-Holder, Philip Ball, Jake Bruce, Vibhavari Dasagi, Kristian Holsheimer, Christos Kaplanis, Alexandre Moufarek, Guy Scully, Jeremy Shar, Jimmy Shi, Stephen Spencer, Jessica Yung, Michael Dennis, Sultan Kenjeyev, Shangbang Long, Vlad Mnih, Harris Chan, Maxime Gazeau, Bonnie Li, Fabio Pardo, Luyu Wang, Lei Zhang, Frederic Besse, Tim Harley, Anna Mitenkova, Jane Wang, Jeff Clune, Demis Hassabis, Raia Hadsell, Adrian Bolton, Satinder Singh, and Tim Rockt¨schel. Genie 2: A large-scale foundation world model. 2024.
- [40] Lawrence Perko. Differential equations and dynamical systems, volume 7. Springer Science & Business Media, 2013.
- [41] Steven Pinker. The language instinct (1994/2007). 2007.
- [42] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [43] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨rn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [44] Reuven Y. Rubinstein. Optimization of computer simulation models with rare events. European Journal of Operational Research, 99(1):89–112, 1997.
- [45] Runway. Introducing gwm-1, December 2025. Accessed: 2026-02-22.

- [46] Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, and Gianluca Corrado. Gaia-2: A controllable multi-view generative world model for autonomous driving. arXiv preprint arXiv:2503.20523, 2025.
- [47] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature, 529(7587):484–489, 2016.
- [48] David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al. Mastering chess and shogi by self-play with a general reinforcement learning algorithm. arXiv preprint arXiv:1712.01815, 2017.
- [49] Oriane Sim´eoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.
- [50] Vlad Sobal, Wancong Zhang, Kynghyun Cho, Randall Balestriero, Tim GJ Rudner, and Yann LeCun. Learning from reward-free offline data: A case for planning with latent dynamics models. arXiv preprint arXiv:2502.14819, 2025.
- [51] Ralph E Strauch. Battle simulation for command and control training. 1982.
- [52] Qiao Sun, Xin Huang, Junru Gu, Brian C Williams, and Hang Zhao. M2i: From factored marginal trajectory prediction to interactive prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6543–6552, 2022.
- [53] Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.
- [54] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, et al. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.
- [55] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.
- [56] Nicole Van Hoeck, Patrick D Watson, and Aron K Barbey. Cognitive neuroscience of human counterfactual reasoning. Frontiers in human neuroscience, 9:420, 2015.
- [57] Balakrishnan Varadarajan, Ahmed Hefny, Avikalp Srivastava, Khaled S Refaat, Nigamaa Nayakanti, Andre Cornman, Kan Chen, Bertrand Douillard, Chi Pang Lam, Dragomir Anguelov, et al. Multipath++: Efficient information fusion and trajectory aggregation for behavior prediction. In 2022 International Conference on Robotics and Automation (ICRA), pages 7814–7821. IEEE, 2022.
- [58] Lev S Vygotsky. Thought and language, volume 29. MIT press, 2012.
- [59] Dilin Wang, Hyunyoung Jung, Tom Monnier, Kihyuk Sohn, Chuhang Zou, Xiaoyu Xiang, YuYing Yeh, Di Liu, Zixuan Huang, Thu Nguyen-Phuoc, et al. Worldgen: From text to traversable and interactive 3d worlds. arXiv preprint arXiv:2511.16825, 2025.
- [60] Xingyao Wang, Boxuan Li, Yufan Song, Frank F Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. Openhands: An open platform for ai software developers as generalist agents. In The Thirteenth International Conference on Learning Representations, 2024.

- [61] Yao Wang, J¨rn Ostermann, and Ya-Qin Zhang. Video processing and communications, volume 1. Prentice hall Upper Saddle River, 2002.
- [62] Waymo. Waymo Driver: Self-Driving Car Technology for a Reliable Ride, 2025. Accessed: 2025-05-01.
- [63] Grady Williams, Andrew Aldrich, and Evangelos Theodorou. Model predictive path integral control using covariance variable importance sampling. arXiv preprint arXiv:1509.01149, 2015.
- [64] World Labs. Marble: A multimodal world model, November 2025. Accessed: 2026-02-22.
- [65] World Labs Technical Staff. Generating worlds. https://www.worldlabs.ai/blog, 2025. Accessed: 2025-06-10.
- [66] Jiannan Xiang, Yi Gu, Zihan Liu, Zeyu Feng, Qiyue Gao, Yiyan Hu, Benhao Huang, Guangyi Liu, Yichi Yang, Kun Zhou, et al. Pan: A world model for general, interactable, and longhorizon world simulation. arXiv preprint arXiv:2511.09057, 2025.
- [67] Jiannan Xiang, Guangyi Liu, Yi Gu, Qiyue Gao, Yuting Ning, Yuheng Zha, Zeyu Feng, Tianhua Tao, Shibo Hao, Yemin Shi, et al. Pandora: Towards general world model with natural language actions and video states. arXiv preprint arXiv:2406.09455, 2024.
- [68] Jiannan Xiang, Tianhua Tao, Yi Gu, Tianmin Shu, Zirui Wang, Zichao Yang, and Zhiting Hu. Language models meet world models: Embodied experiences enhance language models. Advances in neural information processing systems, 36:75392–75412, 2023.
- [69] Eric Xing, Mingkai Deng, and Jinyu Hou. Critiques of agents. Manuscript in preparation, 2026.
- [70] Peiyuan Zhang, Yongqi Chen, Haofeng Huang, Will Lin, Zhengzhong Liu, Ion Stoica, Eric Xing, and Hao Zhang. Faster video diffusion with trainable sparse attention. Advances in Neural Information Processing Systems, 38:152509–152534, 2026.
- [71] Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhengzhong Liu, and Hao Zhang. Fast video generation with sliding tile attention. arXiv preprint arXiv:2502.04507, 2025.
- [72] Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. Dino-wm: World models on pretrained visual features enable zero-shot planning. arXiv preprint arXiv:2411.04983, 2024.

### A Proof for Theorem 1

Proof. We will prove the contrapositive that fϵ(x) = fϵ(x′) ⇒ ∥x − x′∥ ≤ ϵ. Based on the conditions, we have it that ∥x−x′∥ = T,Dt=1,d=1(xtd −x′td)2 ≤ ϵ = ϵ˜2, which is satisfied when |xtd − x′td| ≤ √TDϵ˜ . Because ∥xt∥ ≤ K := K˜

4 , we have that |xtd| ≤ K2˜ ∀d ∈ [1,D]. Therefore, to satisfy the condition

2

that ∥x − x′∥ ≤ ϵ, we need to divide the interval −K2˜ , K2˜ into subintervals with length at most √ ϵ˜

√

TDK˜ϵ˜−1⌉ of them. Further, because there are TD such intervals to divide, the support of our code should be at least as large as the number of required subintervals as below:

TD, of which there are ⌈

√

TDK˜ϵ˜−1⌉TD

|V|M ≥ ⌈

In the two cases below, we will attempt to construct this support by scaling the vocabulary size |V| and the code length M individually. It is technically also possible to scale both at the same time, but we focus on the existence for this proof, and leave more efficient solutions to future work.

- Case 1 (Modality Tokenizer): Take N = T, and construct a vocabulary V with size Mϵ =

⌈

√

TDK˜ϵ˜−1⌉D. We then define the encoding function for x as fϵ(x) = (c1,...,cT) where ct ∈ V. We define the index of ct as a D-digit number of base-Mϵ

(d1(ct),d2(ct),...,dD(ct)), where dn(ct) = ⌊x

tn+K/˜ 2 ϵ/˜

√

TD ⌋ is the n-th digit. Note that this is well defined because dn(ct) ∈ 0,⌊

√

TDK˜ϵ˜−1⌋ , and thus enables the representation of ⌈

√

TDK˜ϵ˜−1⌉D unique values for ct using D

digits. This way, given fϵ(x) = fϵ(x′), since ct = c′t ∀t ∈ [1,T], we have dn(ct) = dn(c′t) ∀n ∈ [1,D], which implies that |xtn − x′tn| ≤ ϵ/˜

√

TD. Thus, we have

∥x − x′∥ =

T,D

t=1,d=1

(xtd − x′td)2 ≤

T,D

t=1,d=1

ϵ˜2 TD

= ϵ (10)

- Case 2 (Finding a Language Expression): Assume that the vocabulary has a fixed size |V| = N (which applies to a wide range of natural and machine languages). Take sequence length Nϵ = TD⌈logM

√

TDK˜ϵ˜−1⌉ and define

), ct ∈ V,

fϵ(x) = (c1,...,cN

ϵ

We map each xtd to a number utd ∈ [0,1] represented by L digits of base-N, where the n-th digit dn(utd) = cs+n. In this manner, xtd will correspond to subsequence (cs+1,cs+2,...,cs+L), where s = (td − 1)⌈logM

√

√

TDK˜ϵ˜−1⌉ and L = ⌈logN

TDK˜ϵ˜−1⌉. Thus, we have that

L

dn(utd) Mn −

- 1

- 2

K. ˜ (11)

xtd =

n=1

Now, given x,x′ where fϵ(x) = fϵ(x′), we have fϵ(x)td = fϵ(x′)td and dn(utd) = dn(utd), so utd and u′td must lie in the same interval formed by numbers for which all digits dn are identical as below:

[u : dn(u) = vn, n = 1,...,L] =

L

vn Mn

,

n=1

L

vn Mn

n=1

1 ML

+

(12)

Thus we have

L

L

L

L

dn(u′td) Mn

dn(u′td) Mn −

- 1

- 2

- 1

- 2

dn(utd) Mn −

dn(utd) Mn −

K ˜ −

K ˜ =

|xtd − x′td| =

n=1

n=1

n=1

n=1

√

√

ϵ ˜ √

TDK˜ϵ˜−1⌉K˜ ≤ M−logM

TDK˜ϵ˜−1K˜ ≤

≤ M−LK˜ = M−⌈logM

TD And by extension ∥x − x′∥ ≤ ϵ

K ˜

| |
|---|

### B Proof for Proposition 1

Proof. We will prove the proposition by constructing such a (h∗,f∗) pair such that Llatent(h∗,f∗) = minh,f Llatent(h,f). ((h∗,f∗) is the global minimum of Llatent.) This could be simply done by finding arbitrary c ∈ Rn and letting h∗(o) = c for all o ∈ O and f∗(c,a) = c for all a ∈ A. Then, we have that Llatent(h∗,f∗) = 0. Since Llatent is a distance measurement, we have that Llatent ≥ 0, which means that (h∗,f∗) is the global minimum.

| |
|---|

### C Proof for Proposition 2

Proof. We prove the proposition by showing that, with the generative loss Lgen, there is no solution that can be both degenerate and optimal at the same time. More specifically, for any degenerate solution (h′,f′), and fixed g, there must be a way to construct another solution (h,˜ f˜) which yields a lower loss value. More formally, without loss of generality, assume that we have:

- • Some arbitrary constant c ∈ Rn;
- • h′ : O → S such that h′(o) = c for all o ∈ O;
- • f′ : S × A → S such that f′(c,a) = c for all a ∈ A;
- • A fixed decoder g : S → O;
- • A dataset D ⊂ O × A × O containing at least two distinct triplets: (o1,a1,o2), (o3,a3,o4) ∈ D with o2 ̸= o4.

Assuming all the function families are sufficiently expressive. Then, under the degenerate solution (h′,f′), for any (o,a,o′) ∈ D, we have: g ◦f′(h′(o),a) = g(f′(c,a)) = g(c). Therefore, the generative loss becomes:

Lgen(h′,f′,g) = E(o,a,o′)∼D [∥g(c) − o′∥]. Note that this loss is non-zero if there exists at least one (o,a,o′) ∈ D such that g(c) ̸= o′, which must be the case here because o2 ̸= o4 while g(c) is fixed across all inputs. We now construct an alternative pair (h,˜ f˜) such that:

- • For all (o,a,o′) ∈ D\{(o1,a1,o2),(o3,a3,o4)}, define h˜(o) = c and f˜(c,a) = c, i.e., h,˜ f˜behave identically to the degenerate solution.
- • For the two special cases, define s1,s2 ∈ S such that g′(s1) = o2 and g′(s2) = o4. (This is possible since g′ is fixed and assumed expressive enough to approximate o2 and o4.) Then, set h˜(o1) = sˆ1, f˜(ˆs1,a1) = s1, and h˜(o3) = sˆ2, f˜(ˆs2,a3) = s2, where sˆ1,sˆ2 ∈ S are any distinct intermediate representations to encode o1 and o3.

Now, consider the loss under this construction:

- • At (o1,a1,o2): h˜(o1) = sˆ1, f˜(ˆs1,a1) = s1, g′(s1) = o2, so loss is ∥o2 − o2∥ = 0.
- • At (o3,a3,o4): similarly, loss is zero for h˜ and f˜.
- • While for h′ and f′, there must be one of (o1,a1,o2) and (o3,a3,o4) at which the loss is strictly positive. In other words, ∥g(c) − o2∥ + ∥g(c) − o4∥ > 0
- • At all other datapoints, the loss is unchanged (equal to the degenerate solution’s loss) because we did not modify the outputs for those inputs.

Hence, the total loss under (h,˜ f˜) is strictly less than under (h′,f′): Lgen(h,˜ f,g˜ ) < Lgen(h′,f′,g)

This indicates that the degenerate solution must not be optimal, thus proving that it cannot be a global minimizer when g is trained or expressive and when the dataset includes different outputs for different inputs.

| |
|---|

### D Proof of Theorem 2

Proof. Because the conditional distributions in the statement are all isotropic Gaussians, we can use them to construct the following transition distributions (assuming qˆ is the empirical data distribution):

Latent Prediction ph◦f : sˆ′ | o,a ∼ N(f(h(o),a),I) Latent Target qˆh : sˆ′ | o′ ∼ N(h(o′),I)

Observation Prediction ph◦f◦g : oˆ′ | o,a ∼ N(g ◦ f(h(o),a),I)

It emerges that the two loss functions described above can be expressed as scaled KL divergences of those distributions, as shown below:

Llatent = ∥f(h(o),a) − h(o′)∥ = 2DKL(ˆqh(ˆs′ | o′)∥ph◦f(ˆs′ | o,a)), Lgen = ∥g ◦ f(h(o),a) − o′∥ = 2DKL(ˆq(o′)∥ph◦f◦g(o′ | o,a)).

Consider applying the encoder transition to qˆ and ph◦f◦g. In the former case, we will recover the latent target qˆh, and in the latter case we will have performed a roundtrip from latent to observation, and back to latent, resulting in the following conditional distribution:

Roundtrip Latent Prediction ph◦f◦g◦h : sˆ′ | o,a ∼ N(h ◦ g ◦ f(h(o),a),I). Applying the data processing inequality gives the following relation:

DKL(ˆqh(ˆs′ | o′)∥ph◦f◦g◦h(ˆs′ | o,a)) ≤ DKL(ˆq(o′)∥ph◦f◦g(o′ | o,a)),

And because the left-hand side is 12∥h ◦ g ◦ f(h(o),a) − h(o′)∥2 and the right-hand side is 12L2gen, we plug them into the inequality above and get:

∥h ◦ g ◦ f(h(o),a) − h(o′)∥ ≤ Lgen. (13)

Next, because the encoder h and decoder g satisfy the roundtrip consistency of ∥h ◦ g(ˆs) − sˆ∥ ≤ ϵ, plugging in sˆ = f(h(o),a) gives:

∥h ◦ g ◦ f(h(o),a) − f(h(o),a)∥ ≤ ϵ. (14) Finally, we apply the triangle inequality to upper-bound the latent loss as below:

Llatent = ∥f(h(o),a) − h(o′)∥ ≤ ∥f(h(o),a) − h ◦ g ◦ f(h(o),a)∥

+ ∥h ◦ g ◦ f(h(o),a) − h(o′)∥

≤Lgen by Inequality 13 ≤ Lgen + ϵ

≤ϵ by Inequality 14

When h ◦ g(ˆs) = sˆ for all sˆ ∈ S, we have that g : S → Im(g) ⊆ O is an injective function which maps each point in S to a unique point in its image Im(g). On the other hand, h is the left inverse of g on Im(g), which maps each oˆ ∈ Im(g) to sˆ ∈ S s.t. g(ˆs) = oˆ. Thus h is bijective in Im(g) by mapping each element o ∈ Im(g) to exactly one sˆ ∈ S.

This means h defines a parameter transformation Th where Th(o) = h(o) for all o ∈ Im(g). Because qˆ(o′) and ph◦f◦g(o′ | o,a) are supported entirely inside Im(g) and KL divergence is invariant under

parameter transformation, we have:

- 1

- 2L2gen = DKL(ˆq(o′)∥ph◦f◦g(o′ | o,a))

= DKL(ˆqh(ˆs′ | o′)∥ph◦f◦g◦h(ˆs′ | o,a))

- 1

- 2∥h ◦ g ◦ f(h(o),a) − h(o′)∥2

= =

1 2∥f(h(o),a) − h(o′)∥2 (h ◦ g(ˆs) = sˆ)

- 1

- 2L2latent.

=

Thus we have Llatent = Lgen.

| |
|---|

