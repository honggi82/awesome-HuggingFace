# arXiv:2604.11805v1[cs.LG]13Apr2026

### Solving Physics Olympiad via Reinforcement Learning on Physics Simulators

###### Mihir Prabhudesai∗,† Aryan Satpathy∗,† Yangmin Li† Zheyang Qin† Nikash Bhardwaj Amir Zadeh Chuan Li Katerina Fragkiadaki Deepak Pathak

Carnegie Mellon University Lambda

###### Zero-Shot generalization to International Physics Olympiad

[Figure 1]

[Figure 2]

What is the momentum of the sphere after 5.3 seconds?

At what height does the projectile hit the Jenga tower?

Ans: 4.3 Kg m / s

Ans: 3.45 m

+4.4

Base Model Finetuned (+ Synthetic Data)

40

| |
|---|

###### Improvements

30

+5.4

Accuracy(%)

[Figure 3]

[Figure 4]

At what time does the bowling ball strike the pins?

20

How many times has the ball bounced at the end of 4 seconds?

+7.5

Ans: 3 s

Ans: 2

| | | |
|---|---|---|
| | | |

10

Qwen2.5 3B

Qwen2.5 32B

Qwen3 30B

Figure 1 | We present SIM2REASON: a method for turning physics simulators into scalable generators of question–answer pairs to improve LLM reasoning, removing the need of human annotation in the data-generation pipeline. The core idea is to structure the randomization with a domain-specific language (DSL) and use it to procedurally generate reasoning problems, as illustrated in the examples above. LLMs finetuned on this synthetic data get zero-shot improvement on real world benchmarks such as International Physics Olympiad.

##### Abstract

We have witnessed remarkable advances in LLM reasoning capabilities with the advent of DeepSeek-R1. However, much of this progress has been fueled by the abundance of internet question–answer (QA) pairs—a major bottleneck going forward, since such data is limited in scale and concentrated mainly in domains like mathematics. In contrast, other sciences such as physics lack large-scale QA datasets to effectively train reasoning-capable models. In this work, we show that physics simulators can serve as a powerful alternative source of supervision for training LLMs for physical reasoning. We generate random scenes in physics engines, create synthetic question–answer pairs from simulated interactions, and train LLMs using reinforcement learning on this synthetic data. Our models exhibit zero-shot sim-to-real transfer to real-world physics benchmarks: for example, training solely on synthetic simulated data improves performance on IPhO (International Physics Olympiad) problems by 5-10 percentage points across model sizes. These results demonstrate that physics simulators can act as scalable data generators, enabling LLMs to acquire deep physical reasoning skills beyond the limitations of internet-scale QA data. Code available at: https://sim2reason.github.io/.

*Project co-leads & Equal contribution. † Core contributors. Correspondence to {mprabhud, asatpath}@andrew.cmu.edu

###### 1. Introduction

Reinforcement learning with verifiable rewards (RLVR) has enabled large language models (LLMs) to cross the threshold from pattern matching to multi-step reasoning. However, this progress is fundamentally constrained by the availability of high-quality question–answer (QA) pairs: textbook- and internet-derived QA corpora are finite, unevenly distributed across domains, and difficult to scale beyond a few million examples. As a result, RLVR systems such as DeepSeek-R1 DeepSeek-AI et al. (2025) are ultimately bottlenecked not by model capacity, but by the scarcity of supervision data Wu et al. (2025).

This limitation is most visible in the physical sciences. While mathematics benefits from abundant question–answer pairs, physics, chemistry, and other empirical sciences lack comparable large-scale datasets. For example, less than 1% of the 800K QA pairs used in DeepSeek-R1 involve STEM topics, leading to poor generalization on standard physics benchmarks. The root issue is that internet QA data is sparse, unevenly distributed, and not systematically varied, leaving large gaps in the supervision signal required for scientific reasoning.

Physics engines, on the other hand, encode physical laws in executable form. Instead of describing phenomena in text, they compute future states by numerically integrating systems of ordinary differential equations under constraints. This gives them the ability to generate unlimited trajectories with high-fidelity supervision signals—such as instantaneous forces, momentum, and energy transfers—that are rarely captured in static internet corpora. However, this information is not directly usable by LLMs to improve their physics problem solving skills: simulator outputs are approximate, continuous, forward-time numerical traces, whereas physics problem solving requires accurate, inverse, symbolic, and counterfactual reasoning. The challenge, then, is how to represent simulator-derived physical information in a way that helps improve an LLM’s physics problem solving ability.

One potential solution is utilizing physics simulators as external tools Sarch et al. (2025); Schick et al. (2023). However, this approach is non-trivial as it shifts the primary challenge from physical reasoning to code generation; the LLM must master complex simulator-specific APIs to model a problem. Our early experiments with this paradigm were unsuccessful, as models frequently struggled to produce executable and physically accurate simulation code. Furthermore, many physical phenomena are not natively supported by simulators, and implementing them requires human-in-the-loop engineering, which renders this approach unscalable. In contrast, we find that our method allows us to generalize beyond the scope of our simulator (Section 3.6).

To address these limitations, we propose Sim2Reason: a framework that transforms the physics simulator into a scalable QA generator (Figure 1). Instead of relying on the LLM’s initial coding capabilities, we procedurally construct diverse physical systems in the physics simulator and simulate their dynamics to automatically generate verified question-answer pairs. Our pipeline produces three reasoning modes: numeric (state queries), reverse (parameter inference), and symbolic (closed-form expressions). These systems span a broad spectrum of classical mechanics, covering the majority of core phenomena encountered in undergraduate and Olympiad-level physics. The procedural nature of our Domain Specific Language (DSL) enables the dynamic composition of heterogeneous physical scenes—such as combining pulley systems with rotational dynamics—generating millions of unique, physically grounded training samples (Figure 2).

(f) RL training using generated QA data

- (a) Simulate a procedurally generated random scene in the Physics Simulator

- (b) Record sensor data from the simulator

- (c) Procedurally generate scene description

|A smooth table is inclined at an angle 0 degrees with the horizontal. There are multiple systems on this table. In a system called ComplexCollisionPlane.mass-0, a block of mass 2.0 kg s placed at 2.5 m on the table. It is initially moving with a velocity of -0.3 m/s.A wall is fixed at 0.0 m on the table. ComplexCollisionPlane.spring_mass-3, 2 blocks are placed at 3.0 and 4.0 m on the table. They have masses 1.0 and 1.0 Kg respectively. These blocks ............|
|---|

- (d) Procedurally generate a random question

- (e) Retrieve the answer from the sensor data.

[Figure 5]

[Figure 6]

[Figure 7]

###### LLM

###### <think>

<think>

time

To ﬁnd the magnitude of the velocity of the second block in the spring_mass-3 system, we need to track the sequence of elastic collisions and then solve the internal dynamics of the spring-mass system.

To ﬁnd the magnitude of the velocity of the second block in the spring_mass-3 system at $t = 5.44$ seconds, we need to track the sequence of elastic collisions and the subsequent oscillatory motion of the spring-mass system.

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

- 1. Initial State and First Collision

We have a block ($m_A = 2.0\text{ kg}$) at $x = 1.0\text{ m}$ moving at $v_A = 2\text{ m/s}$ and a sphere ($m_B = 5\text{ kg}$) at $x = 2.5\text{ m}$ moving at $v_B = -0.3\text{ m/s}$.

The time of their collision $t_1$ is found by: $$1.0 + 2t_1 = 2.5 - 0.3t_1 \implies

- 2.3t_1 = 1.5 \implies t_1 \approx

1. Analysis of the Initial Setup The table is horizontal ($\theta = 0^\circ$), so gravity does not aﬀect the motion along the table's surface.

- • Block (mass-0): $m_0 = 2.0$ kg, $x_0 = 1.0$ m, $v_0 = 2$ m/s.
- • Sphere (sphere-1): $m_s = 5.0$ kg, $x_s = 2.5$ m, $v_s = -0.3$ m/s.
- • Spring-Mass System (spring_mass-3): $m_1 = 1.0$ kg (at $3.0$ m), $m_2 = 1.0$

- 0.6522\text{ s}$$ The position of collision is $x_{coll1} =
- 1.0 + 2(0.6522) \approx 2.3043\text{ m}$

…. </think>

###### …. </think>

|Given the {scene description}, What is the velocity of the 2nd block in spring_mass-3 after 5.44 seconds?|
|---|

0.88 m/s 1.71 m/s

|0.8802 m/s|
|---|

Figure 2 | Overview of the SIM2REASON (Sim2Reason) pipeline. From left to right: we procedurally generate diverse physics scenes using a DSL, (a) compile them into MuJoCo simulations, and (b) record physically grounded state/force traces. (c–e) From these traces we automatically instantiate multiple types of question–answer pairs (numeric, reverse, and symbolic), and apply filtering to remove degenerate/shortcut questions and unstable simulation segments. (f) Finally, we post-train an LLM with RLVR on the resulting synthetic data and evaluate zero-shot sim-to-real transfer on real-world benchmarks (e.g., IPhO and other physics/math datasets).

We train LLMs using Reinforcement Learning (RL) on this synthetic data without incorporating any real-world physics QA pairs during the post-training phase. Evaluating our model across multiple rigorous benchmarks—including IPhO, JEE-Bench, PHYSICS and OlympiadBench—reveals consistent and meaningful performance gains, showcasing a robust sim-to-real transfer. We find that quality filtrating is critical to achieving these gains. For instance, simulatorgenerated questions often suffer from degeneracy, where problems are either trivially easy or computationally intractable. To address this, we implement a question pruning strategy that filters out these extremes, ensuring training compute is focused on useful samples that fall within the LLM’s solvable range.

Our results demonstrate that training solely on Sim2Reason data improves zero-shot performance on IPhO mechanics problems by 5–10 percentage points across 3B to 32B model scales. We observe similar gains on specialized benchmarks like JEEBench (+17.9% for 32B models) and PHYSICS, confirming that the model is not merely memorizing simulator dynamics but is developing a generalized capacity for multi-step physical reasoning. Furthermore, we find that the QA pairs generated by our framework serve as an effective benchmarking tool for foundation models. We observe a high correlation between model accuracy on our simulated questions and performance on real-world physics benchmarks, enabling scalable and automated testing across specific physical domains. Please refer to our project webpage, for code and video visualizations from SIM2REASON: https://sim2reason.github.io/

| | | |
|---|---|---|
| | | |

'<name>',pulls the string attached to

[Figure 12]

Numeric

[Figure 13]

it with a constant force of In a system called '<name>', <2> blocks are stacked on … with slope <15> degrees … blocks are <9> kg in bottom-to-top order.

Discarded QA Pair

MuJoCo

(Entity,Template) Pool

|?|
|---|

|A| |
|---|---|
| | |

###### …

'<name>' pulls the string attached to it with a constant velocity of <vx>1 m/s

[Figure 14]

In a system called '<name>', a block of mass <3> kg rests on a plane inclined at an angle of <45> degrees.

[Figure 15]

[Figure 16]

###### Reverse

[Figure 17]

|B|
|---|

|?| |
|---|---|
| | |

Ablation

Symbolic

Random Selection

[Figure 18]

[Figure 19]

Attributes Recording

[Figure 20]

Scene Connector

Accepted QA Pair

A = x B = ?

Scene Generation Simulation QA Generation Data Filter

Figure 3 | Overview of our synthetic data-generation pipeline. We procedurally generate simulatable scenes by randomly selecting and connecting DSL entities (Section 2.1), then simulate each scene in MuJoCo and record time-series data of key physical attributes (Section 2.2). From these traces we craft natural-language QA pairs in three formats (Section

- 2.3)-numeric, reverse, symbolic-and finally deduplicate and filter degenerate/shortcut-solvable questions before RL post-training (Section 2.4).

###### 2. Method

To train LLMs for physical reasoning, we first generate synthetic data using a physics simulator and then fine-tune the LLM on this synthetic data. Using MuJoCo Todorov et al. (2012) as our simulator, we generate question–answer pairs spanning a wide range of physical phenomena, broadly covering kinematics, rotational mechanics, orbital motion, variable-mass systems, and basic electromagnetism (e.g., a charged particle moving in the presence of time-varying fields).

The data generation pipeline (Figure 3) consists of 4 stages:

- 1. Scene Generation: Generating physically meaningful random scenes
- 2. Physics Simulation: Simulating scenes to record data
- 3. QA pair generation: Generating question-answer pairs from recorded data
- 4. Data filtration: Deduplicating and filtering degenerate qa pairs

###### 2.1. Scene Generation

To procedurally generate scenes in a structured and scalable manner, we design a domainspecific language (DSL) that isolates physically meaningful axes of randomization from those that do not fundamentally change the underlying reasoning (Figure 4). For example, changing the length of a pulley string typically does not affect the system’s dynamics, whereas changing the mass of a suspended block does.

Our DSL consists of three levels of abstraction: scene, entity, and body. Body is the most fundamental element. Each body has a name and a predefined set of parameters based on its type—for instance, the mass of a block or the radius of a sphere (see Appendix D for details). Additionally, for each body we define a template MuJoCo XML snippet and a template string that describes the body and its parameters.

DSL

scene: name: "Pulley System" entities:

- - name: "entity1" type: "MassWithFixedPulley" position: [0, 2, 0] parameters:

mass_type: "Mass" mass_values: [10]

- - name: "entity2" type: "MassWithMovablePulley" position: [0, 1, 0]

... connections:

- - tendon:
- - entity: "entity1" direction: "inner_to_outer"
- - entity: "entity2" direction: "outer_to_inner"

...

[Figure 21]

- Figure 4 | Example of our scene-generation DSL (top) and the corresponding simulator-rendered scene produced by compiling the DSL into simulator scene (bottom). The DSL composes scenes from reusable entities and bodies with explicit connection modes, enabling scalable procedural

generation while restricting randomization to physically meaningful parameters.

However, bodies cannot be arbitrarily connected—for instance, a mass block can be placed on a prism, but not vice versa. This motivates the next level of abstraction: an entity, which consists of a set of bodies connected in a specific, physically meaningful way. Each entity exposes well-defined connection points that specify how it can attach to other entities. We refer to Appendix F for a detailed list of entities.

The scene is formed by randomly selecting entities and connecting them. We generate the MuJoCo XML for a scene by concatenating the XML templates of its entities, each of which is in turn constructed by composing the XML templates of its bodies. This design allows us to generate simulatable scenes at scale without a human in the loop (Figure 4 in Appendix).

###### 2.2. Physics Simulation

To generate synthetic data, we simulate the generated scenes in MuJoCo and record key physical quantities for each body. We categorize bodies into either masses (proprioceptive quantities) or strings (tension and length); Appendix E lists all recorded quantities.

However, the recorded traces can contain unmodeled transitions—such as a block colliding with a pulley or falling off a plane—that lead to unpredictable dynamics. We detect these events by comparing the sliding-window mean and standard deviation. More specifically,

𝜇𝑡 = mean{𝑎𝑗}𝑡𝑗+=𝑡𝑤, 𝜎𝑡 = std{𝑎𝑗}𝑡𝑗+=𝑡𝑤,

truncate at 𝑡 if max

|𝑎𝑖 − 𝜇𝑡| ≥ 𝑘 𝜎𝑡.

𝑖∈{𝑡,...,𝑡+𝑤}

(1)

Here, 𝑎 denotes the recorded acceleration of a body, and 𝑘 is a threshold hyperparameter controlling how aggressively we flag spikes (smaller 𝑘 is more sensitive to spikes). We use 𝑘 = 5 during data generation.

An example of this pruning procedure is shown in Figure 9 in Appendix. We also extend the simulator to support variable-mass systems, Newtonian gravitation, and collisions with a

specified coefficient of restitution.

###### 2.3. QA Pair Generation

For a given simulatable scene, we convert its recorded time-series data into natural-language question–answer pairs. We first generate a scene description by concatenating the naturallanguage descriptions of its entities (themselves composed from body descriptions). We also describe inter-entity connections using reusable template strings for each connection mode.

To form a question, we randomly select a body, a recorded physical quantity, and a timestep. We generate questions in three ways, each requiring a different style of reasoning:

- • Numeric questions: Forward reasoning, e.g., “What is the velocity of block A at time 3s?”
- • Reverse questions: Inverse reasoning, where one scene parameter is masked (e.g., 𝑥), e.g., “What is the mass of block A if its velocity after 3s is 5m/s?”
- • Symbolic questions: Symbolic reasoning, where all numeric parameters are replaced by symbols, e.g., “What is the velocity of block A after time 𝑡?”

###### 2.4. Data Filtration

We filter the generated data to remove shortcut solutions, i.e., cases where a model can ignore part of the scene (or collapse a multi-body interaction into an oversimplified system) and still obtain the correct numeric answer (Figure 5). This is undesirable for RL training because it can reward incorrect physical reasoning and reinforce approximations.

To detect shortcut-solvable questions, we construct controlled “ablations” of each scene:

- • Entity-removal ablations: We treat a scene as a graph of entities and connections, generate sub-scenes by removing one entity at a time while preserving the connectivity of the remaining graph, and re-simulate these sub-scenes.
- • Joint-removal ablations: We generate variants in which individual joints/constraints are replaced by rigid “glued” components.

For a given question, if the ground-truth answer is unchanged between the original scene and any ablated variant, we discard the QA pair. This prunes questions whose solution does not actually depend on the purported multi-entity dynamics and can be solved by approximating the scene with an oversimplified setup. In practice, approximately 15% of generated QA pairs are filtered out by this procedure.

###### 2.5. RL Training

We post-train the LLM using reinforcement learning with verifiable rewards (RLVR). For each prompt 𝑥, we sample a group of 𝐺 responses {𝑦𝑖}𝐺𝑖=1 from the current policy 𝜋𝜃(· | 𝑥) and assign a scalar reward 𝑅(𝑥, 𝑦𝑖) based on final-answer correctness. We assign a positive reward when the model’s final answer lies within a 5% relative error of the simulator-recorded value; otherwise, the reward is zero. This tolerance accounts for the numerical approximations in the simulator, while penalizing incorrect physical reasoning. We optimize Group Sequence Policy Optimization (GSPO)Zheng et al. (2025a) with a reference policy 𝜋ref (the base Instruct model).

As is common in group-based RL, we compute group-relative advantages by normalizing rewards within each group (subtracting the group mean and dividing by the group standard

#### What is the acceleration of ?

- Figure 5 | Illustration of a shortcut solution. The correct answer depends on the coupled motion of the block 𝑚 and wedge 𝑀, but weaker models may collapse the dotted region into a single

body of mass 𝑀 + 𝑚 and still match the numeric answer. We filter QA pairs whose answers are invariant to such approximations.

deviation). The GSPO loss is a clipped, sequence-level policy-gradient objective:

###### ∑︁𝐺

1

min 𝜌𝑖𝐴ˆ𝑖, clip(𝜌𝑖,1 − 𝜀,1 + 𝜀)𝐴ˆ𝑖 (2)

LGSPO(𝜃) = −E𝑥,{𝑦

𝑖}

𝐺

𝑖=1

where 𝜌𝑖 = 𝜋𝜃(𝑦𝑖 | 𝑥)/𝜋old(𝑦𝑖 | 𝑥).

Finally, we incorporate DAPO-style dynamic sampling to improve training efficiency in sparsereward settings. Concretely, if a sampled prompt yields near-zero reward standard deviation across the group (leading to near-zero advantages), we resample additional prompts until the batch is filled with informative groups.Yu et al. (2025)

###### 3. Experiments

We evaluate our proposed SIM2REASON pipeline by post-training LLMs of various sizes with reinforcement learning (RL) on our synthetic dataset. We then test these resulting models on real-world reasoning benchmarks.

Datasets & Evaluation: Below we describe the datasets we use for training and evaluation.

- • Synthetic (SIM2REASON): We generate training questions on-the-fly using the proposed SIM2REASON pipeline; unless stated otherwise, all RL runs use this synthetic distribution. We use numeric QA mode as described in Section 2.3, for all our training runs, we compare against symbolic and reverse QA mode in our ablation section. Concretely, we train for 200 RL steps with batch size 32, so the model observes approximately 6,400 distinct question–answer pairs during post-training.
- • International Physics Olympiad (IPhO): We evaluate zero-shot transfer on a curated set of mechanics problems from the International Physics Olympiad. We collect and filter problems from 1967–2025 to form an evaluation set of 77 questions. For problems with diagrams, we provide figure captions generated from the original problem context using GPT-4o.
- • HCV (Concepts of Physics): We evaluate on a set of 512 mechanics problems curated from

- H. C. Verma’s Concepts of Physics (Vol.1). For problems with diagrams, we provide figure captions generated from the original problem context using GPT-4o.Verma (2017)

- • JEEBench: A collection of 515 problems from JEE–Advanced (India), covering physics, chemistry, and mathematics, and designed to stress multi-step quantitative reasoning. In our

0.23

Validation Accuracy on Synthetic Data

Response Length

8500

0.22

ValidationScore(mean@1)

ResponseLength(tokens)

8250

0.21

8000

0.20

7750

0.19

7500

0.18

7250

0.17

7000

0.16

6750

0.15

6500

0 20 40 60 80 100

20 40 60 80 100

Training Steps

Training Steps

7 Figure 6 | Validation accuracy (green) versus average response length (blue, in tokens) for Qwen3-30B-Instruct over RL post-training steps. Longer responses are strongly associated with higher validation accuracy, suggesting that post-training encourages more extensive intermediate reasoning.

evaluation, we restrict to text-only mechanics questions to avoid confounding gains from visual understanding. We follow the official evaluation pipeline from Arora et al. (2023)

- • OlympiadBench: A benchmark of high-difficulty STEM problems sourced from international and national science olympiads. Similar to other real-world evaluations in this section, we focus on text-only mechanics questions when applicable and report exact-match accuracy. We follow the official evaluation pipeline from He et al. (2024)
- • PHYSICS: A textbook-derived physics benchmark spanning a range of difficulty levels; only the test set is released publicly. We evaluate on the released test split and restrict to mechanics-related, text-only questions. We follow the official evaluation pipeline from Zheng et al. (2025b)
- • AIME 2025: We use problems from the 2025 American Invitational Mathematics Examination (AIME) as an out-of-domain math reasoning check. We evaluate using the LightEval Habib et al. (2023) pipeline and report mean@8 (mean accuracy over 8 sampled responses).AIME

(2025)

- • MATH 500: A 500-problem subset of the Hendrycks MATH dataset, which contains competitionstyle problems with final numeric or symbolic answers. We report exact-match accuracy.Hendrycks et al. (2021)

Models: We evaluate LLMs across multiple model sizes. Specifically, we use Qwen2.5 Instruct checkpoints at 3B, 7B, 14B, and 32B , and additionally include Qwen3-30B-Instruct as a stronger baseline. In our training setup, Qwen3-30B tends to produce substantially longer responses (~8k tokens on average) than comparably sized Qwen2.5 models (~1.5k tokens), which significantly increases RL training cost. Consequently, due to limited compute, we train Qwen3-30B for 100 RL steps, while all Qwen2.5 models are trained for 200 RL steps.

###### 3.1. Zero-shot generalization of SIM2REASON

In this section, we evaluate the generalization ability of our SIM2REASON pipeline. We posttrain LLMs of different sizes (3B–32B) using RL on our synthetic mechanics questions, and then evaluate the resulting checkpoints on held-out synthetic splits and multiple real-world benchmarks.

Table 1 | Performance of Qwen2.5 family Instruct models before and after RL on synthetic datasets, expressed in percentage. Improvements are shown in parentheses.

Model Synthetic Numeric Synthetic Symbolic HCV IPhO Mechanics Qwen3-30B 14.8% 8.8% 53.9% 35.6%

+ RL (synthetic) 17.4% (+2.6%) 8.0% (-0.8%) 59.0% (+5.1%) 40.0% (+4.4%) Qwen2.5-32B 8.9% 5.6% 50.6% 19.8%

+ RL (synthetic) 21.9% (+13.0%) 10.4% (+4.8%) 53.9% (+3.3%) 25.2% (+5.4%) Qwen2.5-14B 7.0% 5.6% 49.3% 16.07%

+ RL (synthetic) 17.0% (+10.0%) 10.4% (+4.8%) 51.7% (+2.4%) 20.45% (+4.4%) Qwen2.5-7B 7.7% 5.6% 44.5% 10.7%

+ RL (synthetic) 16.3% (+8.6%) 9.6% (+4.0%) 46.3% (+1.8%) 15.1% (+4.3%) Qwen2.5-3B 4.8% 3.2% 31.9% 5.68%

+ RL (synthetic) 12.5% (+7.7%) 9.4% (+6.2%) 39.5% (+7.6%) 13.15% (+7.5%)

Table 1 shows consistent improvements on IPhO Mechanics—up to 7 percentage points across model sizes—despite the fact that the post-training stage uses no real-world physics QA data. Notably, the gains persist even for stronger baselines: for example, Qwen3-30B-Instruct improves by +4.4 points on IPhO, suggesting that our synthetic RL signal provides benefits beyond what is already captured by scale and instruction tuning (Figure 6).

Although our default RL training distribution uses numeric questions, we find that Qwen2.5 models also improve on other reasoning modes (reverse and symbolic) on the synthetic evaluation splits (Table 1). This indicates that the post-trained models are learning reusable physical reasoning patterns, rather than overfitting to a single question template.

To further test sim-to-real transfer, Table 2 evaluates Qwen2.5-32B on additional real-world physics benchmarks (JEEBench, OlympiadBench, and PHYSICS) as well as out-of-domain math benchmarks (AIME 2025 and MATH 500). We observe consistent gains across all benchmarks. The largest improvement is on JEEBench (+17.9 points), which contains many mechanics questions closely aligned with the phenomena covered by our simulator. We also observe improvements on AIME and MATH, suggesting that training for physics reasoning also strengthens underlying algebraic and multi-step quantitative skills.

In this section, we take a deeper look at the improvements and broader implications of our framework. We first analyze the choice of our post-training training strategy (RL, SFT) and data composition, exploring how our synthetic data compares with existing post-training datasets such as DAPO 17k to improve reasoning. Subsequently, we propose an alternate use case of our framework: using the simulator itself as a scalable benchmarking tool. Finally, we perform a qualitative analysis of the model’s outputs to categorize the specific axes of improvement.

###### 3.2. Training Strategies for SIM2REASON

SIM2REASON can generate an effectively unbounded number of QA pairs from a physics simulator. A central question then is how to distill this simulator-derived supervision into the LLM in a way that (i) improves reasoning, and (ii) preserves the base model’s general capabilities, . We investigate two widely used post-training paradigms: (i) supervised fine-tuning (SFT) on high-quality demonstrations, and (ii) reinforcement learning with verifiable rewards (RLVR).

Table 2 | Mean accuracy of Qwen 2.5 32B Instruct on other real world benchmarks.

Benchmark Model Score JEEBench

Qwen2.5 32B 34.38%

+ RL (synthetic) 52.28%(+17.90%) PHYSICS

Qwen2.5 32B 39.42%

- + RL (synthetic) 43.09%(+3.67%)

OlympiadBench

Qwen2.5 32B 41.41%

- + RL (synthetic) 44.53%(+3.12%)

Qwen2.5 32B 10.83%

AIME 25

+ RL (synthetic) 12.5%(+1.67%) MATH 500

Qwen2.5 32B 78.4%

+ RL (synthetic) 82.8%(+4.4%)

Table 3 | Comparison of RL vs. SFT on 32B model performance.

Model (Qwen 32B) Synthetic IPhO Baseline 14.0% 19.8%

+ SFT 16.0% (+2.0%) 15.9% (-3.9%) + RL (Ours) 32.0% (+18.0%) 25.2% (+5.4%)

SFT. We construct SFT data of 200,000 question-answer pairs by rejection-sampling solutions from strong teacher models (GPT-4, o3, and o4-mini), and then fine-tune the LLM on the resulting trajectories. As shown in Table 3, SFT yields only modest in-distribution gains on our synthetic evaluation and substantially degrades out-of-distribution performance (e.g., -3.9% on IPhO Mechanics). We hypothesize that this is driven by a large KL shift from the base Instruct model, which can induce catastrophic forgetting during post-training. This failure mode is consistent with recent analyses showing that overly aggressive post-training updates can erase general reasoning skills when the optimization signal is narrow or distribution-shifted.Shenfeld et al. (2025)

RLVR. In contrast, RLVR directly optimizes task success using a sparse, verifiable reward (final-answer correctness), allowing the model to explore diverse solution strategies while staying closer to the base policy. Empirically, RLVR provides robust improvements both indistribution (synthetic) and out-of-distribution (IPhO and other real-world benchmarks), suggesting it is a more reliable way to distill simulator-derived supervision into generalizable reasoning skills.

###### 3.3. Ablations: QA format and data filtration

We ablate two design choices in our synthetic RL pipeline: (i) the question format used during post-training (Section 2.3), and (ii) whether we apply the shortcut-solution filtering described in Section 2.4. Unless stated otherwise, we report IPhO Mechanics accuracy for Qwen2.5-3B Instruct.

QA format: We compare training with numeric questions (our default) against reverse and

Table 4 | Ablations on (a) QA format and (b) Data filtration

(a) Improvements by each QA format during RL post-training.

(b) Effect of shortcut-solution filtering during data-generation.

Model (Qwen 3B) IPhO Baseline 5.68% + RL (reverse) 5.84% + RL (symbolic) 7.46% + RL (numeric) 13.15%

Model (Qwen 3B) IPhO Baseline 5.68% + RL (no filter) 7.14% + RL ( filtered) 13.15%

symbolic variants. Table 4a shows that numeric QA yields the strongest transfer to IPhO.

Shortcut filtering: We also test the impact of removing shortcut-solvable questions via scene ablations. As shown in Table 4b, shortcut filtering is critical: training without filtering yields substantially smaller gains than training on the filtered numeric distribution.

###### 3.4. Synthetic vs. Real-World Post-Training

The ablations above show that SIM2REASON benefits from careful data design—for example, filtering out shortcut-solvable questions materially improves transfer. In this section, we ask a broader question: how does simulator-generated post-training compare to real-world posttraining data? We study this from two complementary angles: first, by comparing against recent open-weight models such as Prime P1 that were post-trained directly on curated real-world physics QA corpora, and second, by comparing against DAPO-17K under a matched RL setup.

Table 5 | Comparison with open-weight post-trained models trained on real-world QA datasets.

Base Model Post-Trained Model IPhO Accuracy (%)

Qwen2.5-32B DAPO-32B 24.7 Qwen2.5-32B LIMO-32B 25.5 Qwen3-30B Prime P1 30B 38.6 Qwen3-30B Sim2Reason (Ours) 40.0

We first compare against recent open-weight post-trained models that were trained directly on real-world QA corpora (Table 5). Prime P1 Chen et al. (2025) 30B is trained on over 5,000 curated physics QA pairs from olympiads and textbooks, while DAPO-32B Yu et al. (2025) and LIMO-32B Ye et al. (2025) are trained on real-world math QA data. Despite using only simulator-generated synthetic data during post-training, Sim2Reason achieves 40.0% on IPhO, outperforming Prime P1 30B (38.6%), DAPO-32B (24.7%), and LIMO-32B (25.5%). This shows that simulator supervision can compete with, and even exceed, post-training on curated realworld corpora.

Unfortunately, there are currently no publicly available physics reasoning post-training datasets that are directly comparable to our setting for a matched data comparison. We therefore next compare against a strong public math RL dataset, DAPO-17K, released alongside the DAPO open-source RL system.Yu et al. (2025) DAPO-17K contains 17K curated mathematical problems designed for outcome-reward RL training at scale.

Table 6 shows that training on our SIM2REASON synthetic mechanics data yields substantially better IPhO transfer than training on DAPO-17K alone, despite DAPO-17K being an order of magnitude larger than our 1K-sample synthetic subset in this ablation. This suggests

that domain-aligned simulator data provides a higher-signal training distribution for physics reasoning than larger but generic math-only corpora.

Combining DAPO-17K with our synthetic data yields a further, albeit smaller, improvement over DAPO-17K alone, indicating partial complementarity: generic math data can strengthen broad quantitative skills, but physics-specific simulator supervision remains the primary driver of IPhO gains.

Table 6 | Comparison with a real-world post-training dataset.

Model (Qwen 3B) IPhO

Baseline 5.68 + RL DAPO-17K (Real) 9.98

+ RL Mixed: DAPO-17K (Real) + Synthetic 10.35 + RL Synthetic (Ours) 13.15

###### 3.5. Simulator as a benchmark

Beyond serving as a source of post-training supervision, SIM2REASON also enables a scalable benchmarking workflow for scientific reasoning. Measuring progress in physics reasoning is challenging because high-quality real-world evaluation sets are small, expensive to curate, and slow to expand (e.g., olympiad problems require expert selection and careful verification). In contrast, our simulator-driven pipeline can generate large numbers of mechanically grounded questions with automatically verifiable answers, enabling rapid iteration and fine-grained diagnostics across specific phenomena (e.g., pulleys, collisions, springs, rotation).

A key question is whether simulator accuracy predicts real-world reasoning. Figure 10 suggests it does: across models, synthetic accuracy correlates strongly with IPhO mechanics accuracy (Spearman 𝜌 = 0.79). This makes simulator-based evaluation a useful proxy for comparing models/ablations and for diagnosing strengths by stratifying results by scene type and physical quantity.

###### 3.6. Analysis of Capabilities

In this section, we first analyze the scalability of our pipeline as a data-generation framework, specifically, whether it can be extended to new scene types beyond those currently covered by our DSL. We then analyze the capabilities learned through RL post-training, focusing on three complementary questions: (i) robustness to harder problems, (ii) generalization to questions that cannot be directly simulated in our current environment, and (iii) qualitative changes in solution behavior.

###### 3.6.1. Scalability of the Pipeline

To study scalability, we identify three real-world mechanics questions (from F=ma, USAPhO, and JEE Advanced) that cannot be expressed using our current DSL, and ask an LLM agent to implement them in MuJoCo using two approaches: (1) direct generation of raw MuJoCo XML, and (2) generation within our DSL abstraction space. Direct XML generation succeeds in only 1 out of 3 cases (33%), with failures caused by incorrect spatial reasoning, joint configuration, or missing components. In contrast, DSL-based generation succeeds in all 3 cases (100%), requiring only minor corrections (Figure 30-32). These results suggest that, while direct simulator code is

difficult for LLMs to generate reliably, our DSL offers a more effective abstraction for extending the pipeline to new scenarios.

###### IPhO 2012 Q1: Focus On Sketches

z

A ball is thrown with a fixed initial speed 𝑣0 in a homogeneous gravitational field. The x-axis is horizontal, and the z-axis is vertical (opposing gravity 𝑔).

z₀

z = z₀ − kx²

(envelope)

By adjusting the launching angle, the ball can hit any target within a region defined by the parabola: 𝑧 ⩽ 𝑧0 − 𝑘𝑥2

reachable region

You can use this fact without proving it. Find the constants 𝑧0 and 𝑘 in terms of 𝑣0 and 𝑔.

x

O

√(z₀/k)

###### △ BASE MODEL

###### ✓ RL-FINETUNED MODEL

- Step 1: Write Equations of Motion The trajectory is given by: 𝑥 = 𝑣0 cos(𝜃)𝑡 𝑧 = 𝑣0 sin(𝜃)𝑡 − 21 𝑔𝑡2 Eliminating 𝑡, we get the trajectory equation: 𝑧 = 𝑥 tan(𝜃) − 2 𝑔𝑥2

𝑣2 0

cos2(𝜃)

- Step 2: Match Coefficients We compare this specific trajectory to the given boundary equation: 𝑧 = 𝑧0 − 𝑘𝑥2.

To find the constants, we assume the standard case for maximums, which is 𝜃 = 45◦.

- - At 45 degrees, tan(𝜃) = 1.
- - At 45 degrees, cos2(𝜃) = 0.5.

- Step 3: Solve for 𝑘 Looking at the 𝑥2 term in the trajectory equation:

- Step 1: Analyze the Physical Meaning The boundary conditions are:

- 1. The maximum possible height (𝑧max) where 𝑥 = 0.
- 2. The maximum possible range (𝑥max) where 𝑧 = 0.

- Step 2: Find 𝑧0 (The Peak) The maximum height is achieved by throwing the ball straight up (𝜃 = 90◦). Using energy conservation (𝑚𝑔ℎ = 21𝑚𝑣2): 𝑧0 = 𝑣02/(2𝑔)

- Step 3: Find 𝑘 (The Width) We know max range occurs at 45◦: 𝑥max = 𝑣02/𝑔

Now, substitute this point (𝑧 = 0, 𝑥 = 𝑣02/𝑔) into the boundary equation:

0 = 𝑧0 − 𝑘𝑥max2 0 = 𝑣

2 0

2𝑔 − 𝑘( 𝑣

2 0

𝑔 )2

CORRECT: Assumed the correct boundary condition.

- Step 4: Solve for 𝑘

Coefficient = 2 𝑔

𝑣2 0

###### cos2(𝜃) Substitute cos2(45) = 0.5: 𝑘 = 2 𝑔

𝑣2

0·0.5 𝑘 = 𝑔/𝑣02

𝑣4 0

2 0

𝑔2 = 𝑣

𝑘

2𝑔 𝑘 = 𝑣

ERROR: Assumed boundary is defined by the 45degree path.

2 0

2𝑔 · 𝑔2

= 2𝑔

𝑣2 0

𝑣4 0

Conclusion: 𝑧0 = 𝑣02/(2𝑔) 𝑘 = 𝑔/(2𝑣02)

Conclusion: 𝑧0 = 𝑣02/(2𝑔) 𝑘 = 𝑔/𝑣02

###### Predicted Answer: 𝑧0 = 𝑣02/2𝑔, 𝑘 = 𝑔/2𝑣02

Predicted Answer: 𝑧0 = 𝑣02/2𝑔, 𝑘 = 𝑔/𝑣02

Figure 7 | LLM answers before (left) and after (right) RL finetuning. Question adapted from IPhO 2012 Question 1 “Focus on sketches”.

This same abstraction also improves portability. Although our current implementation uses MuJoCo, scene generation operates at the level of entities and connections rather than raw

Table 7 | Detailed performance across difficulty levels on the PHYSICS benchmark.

Category Qwen 32B + RL (synthetic)

High School and Below 65.5% 68.3% (+2.8%) High School Olympiad 52.9% 54.0% (+1.1%) Undergraduate 47.9% 48.4% (+0.5%) Postgraduate 32.2% 37.8% (+5.6%)

simulator syntax. We therefore ask LLMs to port a subset of DSL entities from MuJoCo to NVIDIA Omniverse (Figure 33), and observe successful transfer for all entities supported by Omniverse’s physics engine. Together, these results suggest that broadening the coverage of SIM2REASON does not require rebuilding the entire pipeline from scratch: one can extend the DSL to new scenarios and re-implement entities for new simulator backends while preserving the same overall data-generation workflow.

###### 3.6.2. Capabilities Learned Through RL

Having established that the pipeline can scale to new scene types, we next analyze what RL posttraining teaches the model, first across difficulty levels and then under broader generalization.

Coverage Across Difficulty Levels. We evaluate robustness across difficulty tiers in the PHYSICS benchmark. As shown in Table 7, RL post-training on SIM2REASON improves performance at every tier.

Gains are modest at lower tiers (e.g., +2.8% at High School and Below) and largest at the Postgraduate tier (+5.6%), suggesting simulator-based RL particularly strengthens harder multi-step quantitative reasoning. We use Gemini 2.5 Flash as a verifier.

Generalization Beyond Simulation. A key question is whether the gains of Sim2Reason are limited to scenarios were explicitly model in MuJoCo. We find that improvements transfer to problems that are not directly covered by our current library of entities. In principle, many such problems could be simulated, but doing so can require bespoke entity design and scene construction tailored to that specific setting (e.g., adding specialized celestial-body interactions).

For example, the problem in Figure 8 involves a rocket taking off from a planet in the presence of a star. Accurately simulating this setup would require implementing additional entities logic with this exact case in mind. Nonetheless, the base Qwen2.5-32B-Instruct model fails to solve the problem in any of eight trials, whereas after RL on our synthetic data the success rate increases to 50% (4/8). This suggests that the post-trained model is learning transferable abstractions (e.g., formulating constraints and bookkeeping forces/energy), rather than merely overfitting to simulated scenes.

Qualitative Examples. To concretely illustrate these gains, we present comparative case studies across real-world problems. We observe improvements along several axes: arithmetic (reducing calculation errors; Figures 28, 29), physical reasoning (mapping text to correct equations and boundary conditions; Figures 7, 8, 27), and strategic planning (e.g., unit conversions and intermediate checks; Figure 26).

###### JEE ADVANCED 2017 PAPER 2

A rocket is launched normal to the surface of the Earth, away from the Sun, along the line joining the Sun and the Earth.

[Figure 22]

###### Parameters:

- • Mass of Sun = 300,000× Mass of Earth
- • Distance = 25,000× Radius of Earth
- • Escape Velocity (Earth) = 11.2 km/s

The minimum initial velocity (𝑣𝑆) in km/s required for the rocket to be able to leave the Sun-Earth system is closest to:

[A] 22 [B] 42 [C] 62 [D] 72

△ BASE MODEL

###### ✓ RL-FINETUNED MODEL

- Step 1: Calculate Escape Velocity from Sun Using the given ratios (𝑀Sun = 3 × 105𝑀𝐸, 𝑑 = 2.5 × 104𝑅𝐸):

𝑣sun =

√︂2𝐺𝑀Sun

𝑑 ≈ 54.78 km/s

ERROR: Correct ≈ 39 km/s

- Step 2: Calculate Total Velocity To escape both bodies, we sum velocities:

- Step 1: Analyze the System We need to escape the combined potential of Earth and Sun.

- - Earth Escape: 11.2 km/s
- - Sun Escape (at Earth’s distance): ≈ 42 km/s

- Step 2: Determine Effective Velocity Since the Sun’s gravitational well is significantly deeper, the escape velocity is dominated by the Sun.

###### 𝑣total = √︃𝑣sun2 + 𝑣earth2 Correct: Energy Addition

𝑣total = 11.2 + 54.78 ≈ 65.98 km/s

Because 𝑣sun(42) ≫ 𝑣earth(11), the result is approximately 𝑣sun. 𝑣total ≈ 42 km/s Conclusion: The minimum velocity is dictated by the Sun.

ERROR: Linear addition Treats gravitational potentials as independent additive quantities (physically wrong).

Conclusion: The calculated value (66 km/s) is closest to [C].

###### Predicted Answer: [B] 42 km/s

Predicted Answer: [C] 62 km/s

Figure 8 | LLM answers before (left) and after (right) RL finetuning. Question adapted from JEE Advanced 2017 Paper 2.

###### 4. Conclusion

We presented SIM2REASON, a simulator-driven pipeline that procedurally generates diverse physics scenes, converts simulated traces into verifiable QA pairs, and post-trains LLMs with RLVR. Across multiple real-world benchmarks (e.g., IPhO mechanics), models trained only on synthetic simulator supervision show consistent zero-shot sim-to-real gains, suggesting simulators are a scalable source of reasoning supervision.

A direct avenue for future work is to combine simulator-generated data with curated realworld QA to further improve robustness and coverage. More broadly, extending this approach beyond classical mechanics to other areas of physics (e.g., E&M, thermodynamics) and to other physical sciences is a promising direction.

###### Impact Statement

This work investigates training language models for physical reasoning using synthetic question– answer supervision generated from physics simulators. We expect the primary positive impact to be improved access to high-quality scientific tutoring and problem-solving tools, and a reduction in dependence on scraping internet QA data.

Potential risks include misuse of stronger reasoning models (e.g., to assist in harmful engineering) and over-reliance on simulator-generated supervision, which may encode modeling assumptions and failure modes that do not hold in the real world. To mitigate these issues, we emphasize evaluation on real-world benchmarks, report limitations of simulator fidelity and coverage, and encourage downstream deployments to include safeguards, monitoring, and domain-specific validation.

###### References

AIME. Aime problems and solutions. Website, 2025. URL https://artofproblemsolving. com/wiki/index.php/AIME_Problems_and_Solutions. Accessed: 2026-01-29.

- I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, R. Ribas, et al. Solving rubik’s cube with a robot hand. arXiv preprint arXiv:1910.07113, 2019.

D. Angelis, F. Sofos, and T. E. Karakasidis. Artificial intelligence in physical sciences: Symbolic regression trends and perspectives. Archives of Computational Methods in Engineering, page 1, 2023.

D. Arora, H. G. Singh, and Mausam. Have llms advanced enough? a challenging problem solving benchmark for large language models, 2023. URL https://arxiv.org/abs/2305.15074.

- S. L. Brunton, J. L. Proctor, and J. N. Kutz. Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proceedings of the National Academy of Sciences, 113(15):3932–3937, 2016. doi: 10.1073/pnas.1517384113. URL https://www.pnas.org/doi /abs/10.1073/pnas.1517384113.

- J. Chen, Q. Cheng, F. Yu, H. Wan, Y. Zhang, S. Zheng, J. Yao, Q. Zhang, H. He, Y. Luo, Y. Zhao, F. Wang, L. Sheng, C. Xie, Y. Zuo, Y. Li, W. Zeng, Y. Wu, R. Huang, D. Zhou, K. Chen, Y. Qiao, L. Bai, Y. Cheng, N. Ding, B. Zhou, P. Ye, and G. Cui. P1: Mastering physics olympiads with reinforcement learning, 2025. URL https://arxiv.org/abs/2511.13612.

DeepSeek-AI, D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang,

- X. Bi, X. Zhang, X. Yu, Y. Wu, Z. F. Wu, Z. Gou, Z. Shao, Z. Li, Z. Gao, A. Liu, B. Xue, B. Wang,

- B. Wu, B. Feng, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, D. Dai, D. Chen, D. Ji, E. Li,

- F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Bao, H. Xu, H. Wang, H. Ding,

- H. Xin, H. Gao, H. Qu, H. Li, J. Guo, J. Li, J. Wang, J. Chen, J. Yuan, J. Qiu, J. Li, J. L. Cai, J. Ni, J. Liang, J. Chen, K. Dong, K. Hu, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Zhao, L. Wang, L. Zhang, L. Xu, L. Xia, M. Zhang, M. Zhang, M. Tang, M. Li, M. Wang,

- M. Li, N. Tian, P. Huang, P. Zhang, Q. Wang, Q. Chen, Q. Du, R. Ge, R. Zhang, R. Pan, R. Wang,

- R. J. Chen, R. L. Jin, R. Chen, S. Lu, S. Zhou, S. Chen, S. Ye, S. Wang, S. Yu, S. Zhou, S. Pan, S. S. Li, S. Zhou, S. Wu, S. Ye, T. Yun, T. Pei, T. Sun, T. Wang, W. Zeng, W. Zhao, W. Liu, W. Liang,

- W. Gao, W. Yu, W. Zhang, W. L. Xiao, W. An, X. Liu, X. Wang, X. Chen, X. Nie, X. Cheng,
- X. Liu, X. Xie, X. Liu, X. Yang, X. Li, X. Su, X. Lin, X. Q. Li, X. Jin, X. Shen, X. Chen, X. Sun,

- X. Wang, X. Song, X. Zhou, X. Wang, X. Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. Zhang, Y. Xu,
- Y. Li, Y. Zhao, Y. Sun, Y. Wang, Y. Yu, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Piao, Y. Wang,

- Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Ou, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Xiong, Y. Luo, Y. You,

- Y. Liu, Y. Zhou, Y. X. Zhu, Y. Xu, Y. Huang, Y. Li, Y. Zheng, Y. Zhu, Y. Ma, Y. Tang, Y. Zha,

- Y. Yan, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Xie, Z. Zhang, Z. Hao, Z. Ma, Z. Yan, Z. Wu,
- Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Pan, Z. Huang, Z. Xu, Z. Zhang, and Z. Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

- N. Habib, C. Fourrier, H. Kydlícˇek, T. Wolf, and L. Tunstall. Lighteval: A lightweight framework for llm evaluation, 2023. URL https://github.com/huggingface/lighteval.

- C. He, R. Luo, Y. Bai, S. Hu, Z. L. Thai, J. Shen, J. Hu, X. Han, Y. Huang, Y. Zhang, J. Liu, L. Qi, Z. Liu, and M. Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024. URL https://arxiv.org/ abs/2402.14008.
- D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. URL https://arxi v.org/abs/2103.03874.

- B. Liu, L. Guertler, S. Yu, Z. Liu, P. Qi, D. Balcells, M. Liu, C. Tan, W. Shi, M. Lin, et al. Spiral: Self-play on zero-sum games incentivizes reasoning via multi-agent multi-turn reinforcement learning. arXiv preprint arXiv:2506.24119, 2025.

M. Raissi, P. Perdikaris, and G. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378:686–707, 2019. ISSN 0021-9991. doi: https://doi.org/10.1016/j.jcp.2018.10.045. URL https://www.sciencedirect.com/scie nce/article/pii/S0021999118307125.

P. A. Reinbold, L. M. Kageorge, M. F. Schatz, and R. O. Grigoriev. Robust learning from noisy, incomplete, high-dimensional experimental data via physically constrained symbolic regression. Nature communications, 12(1):3219, 2021.

- G. Sarch, S. Saha, N. Khandelwal, A. Jain, M. J. Tarr, A. Kumar, and K. Fragkiadaki. Grounded reinforcement learning for visual reasoning, 2025. URL https://arxiv.org/abs/2505.2 3678.

T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, and T. Scialom. Toolformer: Language models can teach themselves to use tools, 2023. URL https://arxiv.org/abs/2302.04761.

M. D. Schmidt and H. Lipson. Distilling Free-Form Natural Laws from Experimental Data. Science, 324:81–85, 2009. doi: 10.1126/science.1165893.

- Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- I. Shenfeld, J. Pari, and P. Agrawal. Rl’s razor: Why online reinforcement learning forgets less,

###### 2025. URL https://arxiv.org/abs/2509.04259.

- P. Shojaee, K. Meidani, S. Gupta, A. B. Farimani, and C. K. Reddy. Llm-sr: Scientific equation discovery via programming with large language models, 2025. URL https://arxiv.org/ abs/2404.18400.

F. Tajwar, Y. Jiang, A. Thankaraj, S. S. Rahman, J. Z. Kolter, J. Schneider, and R. Salakhutdinov. Training a generally curious agent. arXiv preprint arXiv:2502.17543, 2025.

E. Todorov, T. Erez, and Y. Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5026–5033. IEEE, 2012. doi: 10.1109/IROS.2012.6386109.

T. Trinh, Y. Wu, Q. V. Le, H. He, and T. Luong. Solving olympiad geometry without human demonstrations. Nature, 2024. doi: 10.1038/s41586-023-06747-5.

S.-M. Udrescu and M. Tegmark. Ai feynman: a physics-inspired method for symbolic regression,

2020. URL https://arxiv.org/abs/1905.11481.

H. C. Verma. Concepts of Physics: Part 1. Concepts of Physics. Bharati Bhawan Publishers & Distributors, Patna, India, 2017. ISBN 9788177091878.

Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: long papers), pages 13484–13508, 2023.

F. Wu, W. Xuan, H. Qi, X. Lu, A. Tu, L. E. Li, and Y. Choi. Deepsearch: Overcome the bottleneck of reinforcement learning with verifiable rewards via monte carlo tree search, 2025. URL https://arxiv.org/abs/2509.25454.

H. Xin, D. Guo, Z. Shao, Z. Ren, Q. Zhu, B. Liu, C. Ruan, W. Li, and X. Liang. Deepseekprover: Advancing theorem proving in llms through large-scale synthetic data. arXiv preprint arXiv:2405.14333, 2024.

A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Y. Ye, Z. Huang, Y. Xiao, E. Chern, S. Xia, and P. Liu. Limo: Less is more for reasoning, 2025.

URL https://arxiv.org/abs/2502.03387.

L. Yu, W. Jiang, H. Shi, J. Yu, Z. Liu, Y. Zhang, J. T. Kwok, Z. Li, A. Weller, and W. Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023.

- Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai, T. Fan, G. Liu, L. Liu, X. Liu, H. Lin, Z. Lin, B. Ma, G. Sheng, Y. Tong, C. Zhang, M. Zhang, W. Zhang, H. Zhu, J. Zhu, J. Chen, J. Chen, C. Wang, H. Yu, Y. Song, X. Wei, H. Zhou, J. Liu, W.-Y. Ma, Y.-Q. Zhang, L. Yan, M. Qiao, Y. Wu, and M. Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https://arxiv.org/abs/2503.14476.

- C. Zheng, S. Liu, M. Li, X.-H. Chen, B. Yu, C. Gao, K. Dang, Y. Liu, R. Men, A. Yang, J. Zhou, and J. Lin. Group sequence policy optimization, 2025a. URL https://arxiv.org/abs/2507.1 8071.

S. Zheng, Q. Cheng, J. Yao, M. Wu, H. He, N. Ding, Y. Cheng, S. Hu, L. Bai, D. Zhou, G. Cui, and

- P. Ye. Scaling physical reasoning with the physics dataset, 2025b. URL https://arxiv.or g/abs/2506.00022.
- Q. Zhu, D. Guo, Z. Shao, D. Yang, P. Wang, R. Xu, Y. Wu, Y. Li, H. Gao, S. Ma, et al. Deepseekcoder-v2: Breaking the barrier of closed-source models in code intelligence. arXiv preprint arXiv:2406.11931, 2024.

###### A. Related Work

Reinforcement Learning from Verifiable Feedback Recent work has explored Reinforcement Learning from Verifiable Rewards (RLVR) as a scalable alternative to human preference annotation for training reasoning-capable language models DeepSeek-AI et al. (2025); Shao et al. (2024); Yang et al. (2025); Yu et al. (2025). In RLVR, models are trained using automatically verifiable signals—such as exact-answer matching, program execution, theorem proving, or symbolic checks—to provide dense, objective reward signals for complex reasoning tasks Xin et al. (2024); Yang et al. (2025); Zhu et al. (2024). This paradigm has been successfully applied in domains such as mathematics, code generation, and formal reasoning, where correctness can be algorithmically verified. However, existing RLVR approaches rely on domains with deterministic and symbolic verification pipelines and are limited by the availability of structured ground truth problems and answers. In contrast, our work extends the RLVR paradigm to physical reasoning, where supervision is derived from physics simulation rather than question-answer pairs. By using simulators to generate verifiable outcomes and synthetic QA supervision, we enable RL-based training of LLMs in domains where formal verification might be infeasible, demonstrating zero-shot transfer to real-world physics benchmarks such as IPhO.

Symbolic Regression Symbolic regression aims to recover interpretable physical laws from data Angelis et al. (2023), using methods ranging from genetic programming Schmidt and Lipson (2009) to sparse regression Brunton et al. (2016) and neural approaches Raissi et al. (2019); Udrescu and Tegmark (2020). While these methods are appealing for their interpretability, purely data-driven symbolic regression is often most successful in relatively simple, low-dimensional settings and can become brittle when the data are noisy, incomplete, or high-dimensional without additional inductive bias or physical constraints Angelis et al. (2023); Reinbold et al. (2021). Recent work suggests that LLMs may help address some of these limitations by proposing candidate functional forms, programs, or symbolic expressions that guide the search over equations more effectively than brute-force enumeration alone Shojaee et al. (2025).

Synthetic Data training Synthetic data has emerged as a powerful alternative to manual annotation across several domains. In mathematics, AlphaGeometry procedurally generates large-scale synthetic geometry training data to solve Olympiad-level geometry problems Trinh et al. (2024). In robotics, Solving Rubik’s Cube with a Robot Hand shows that large-scale simulator training with automatic domain randomization can transfer complex manipulation policies to the real world Akkaya et al. (2019). In language-model post-training, methods such as MetaMath and Self-Instruct use LLMs to synthetically expand instruction or reasoning datasets Wang et al. (2023); Yu et al. (2023), while more recent approaches such as PAPRIKA and SPIRAL use synthetic interaction data or self-play to create scalable training curricula without relying exclusively on human-written supervision Liu et al. (2025); Tajwar et al. (2025).

Our work is complementary to these efforts but targets a distinct setting. Our setting is different from prior synthetic-data work in math and code, where the underlying tasks already come with clean symbolic structure and canonical forms of supervision. In physics, by contrast, the goal is often to answer natural-language questions about systems governed by continuous dynamics. We therefore propose a recipe for turning physics simulators into generators of structured post-training data—including English QA pairs with automatically verifiable answers—and show that this synthetic supervision alone yields zero-shot gains on real-world physics benchmarks.

- B. Domain-Specific Language and Timestep pruning strategy

We summarize the two additional components used to build training data. Figure 4 shows the YAML-based scene-generation DSL and an example MuJoCo rendering produced by compiling it to MuJoCo XML, while Figure 9 illustrates our timestep-pruning heuristic that removes unstable trace suffixes before QA generation.

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

- Figure 9 | Timestep pruning for simulation traces with unmodelled transitions. Left: MuJoCo scene snapshots at the start and at time 3s. Right: recorded time-series signals; when a

sliding-window deviation criterion flags an outlier (e.g., due to contact between block and pulley), we keep only the stable prefix (green) and discard the remainder before generating QA pairs.

C. Additional Results

- Figure 10 reports a correlation analysis across models/runs, showing that higher accuracy on our SIM2REASON synthetic questions tends to coincide with higher accuracy on IPhO mechanics. This supports using the synthetic QA suite as a lightweight proxy for real-world physics reasoning performance.

- D. Bodies and their parameters We define a list of bodies, along with their randomizable parameters in Table 8.
- E. Recorded physical quantities

During simulation, we log time-series data for each scene to enable question-answer pair generation. We group data into three categories: mass-related (body state and dynamics), string-related (length/tension), and contact (interaction forces). We list the recorded quantities for each category in Table 9.

Synthetic vs IPhO Performance

AccuracyinIPhO(%)

GPT 5 High

Gemini 2.5 Flash

42

40

38

GPT 5 Mini

36

GPT 5 Nano

Qwen 3 30B Instruct

34

32

Claude Sonnet 4.5

30

28

7 8 9 10 11 12 13

Accuracy in Synthetic data (%)

Figure 10 | Correlation between accuracy on SIM2REASON synthetic questions and IPhO mechanics questions.

###### F. Entities and their Connections

Here, we show a list of entities that we define (Figures 11–25). The randomizable parameters for each entity are visualized in the figures by their respective mathematical notations. The connection points and modes are also visualized as dotted lines.

mass_with_fixed_pulley consists of a fixed pulley with one side open for connection to other entities (represented by dotted line), and the other connected to a simple mass system. Below are the 3 variants of mass systems which are supported by this entity.

###### ENTITY VISUALIZATION

| | |
|---|---|
| | |

Figure 11 | Mass With Fixed Pulley

|Body<br><br>|Symbol(s)<br><br>|Description|
|---|---|---|
|Mass|𝑚<br><br>|Point mass / block mass.|
|Sphere<br><br>|𝑟, 𝑚|Sphere radius and mass.|
|Polygonal prism<br><br>|𝑛, 𝑟, ℎ, 𝑚|Number of sides, circumscribed radius, height, and mass.|
|Cylinder<br><br>|𝑟, ℎ, 𝑚<br><br>|Cylinder radius, height, and mass.|
|Disc<br><br>|𝑟, 𝑚|Disc radius and mass.|
|Bar<br><br>|𝑤, ℓ, ℎ, 𝑚|Bar width, length, height, and mass.|
|Hemisphere<br><br>|𝑟, 𝑚|Hemisphere radius and mass.|
|Bowl<br><br>|𝑟, ℎ𝑐, 𝑡, 𝑚|Bowl radius, cutting-plane height ℎ𝑐, shell thickness 𝑡 (if hollow), and mass.|
|Sphere with spherical hole<br><br>|𝑟, 𝑟ℎ, 𝑝ℎ, 𝑡, 𝑚<br><br>|Outer radius 𝑟, hole radius 𝑟ℎ, hole position 𝑝ℎ, shell thickness 𝑡 (if hollow), and mass.|
|Rocket<br><br>|𝑚dry, 𝑚0|Dry mass 𝑚dry and initial total mass 𝑚0.|
|Triangular prism<br><br>|𝛼𝐿, 𝛼𝑅, 𝑚<br><br>|Left/right face slopes (angles) and mass.|
|Plane<br><br>|𝛼<br><br>|Plane slope (incline angle).|
|Pulley<br><br>|𝑚|Pulley mass.|
|Spring–mass system|{𝑘𝑖}, {ℓ0,𝑖}, {𝑥𝑖}, {𝑚𝑖}<br><br>|Spring constants, natural lengths, mass positions, and masses connected by springs.|

Table 8 | Bodies used by the DSL and the corresponding randomizable parameters.

mass_with_movable_pulley consists of a movable pulley with both sides connected to one of the variants of mass_with_fixed_pulley (represented by dotted shapes 𝐸1 and 𝐸2), and the top is open for connection to other entities (represented by dotted line).

###### ENTITY VISUALIZATION

Figure 12 | Mass With Movable Pulley

|Category|Quantity<br><br>|Description|
|---|---|---|
|Mass Mass Mass Mass Mass Mass Mass Mass Mass Mass Mass Mass<br><br>|displacement com_offset velocity (6D) acceleration (6D) mass momentum (6D) net force (6D) kinetic_energy_linear kinetic_energy_angular potential_energy inertia em_potential_energy|Body displacement / position (in world frame).<br><br>Vector from body frame origin to center of mass.<br><br>Linear and angular velocity.<br><br>Linear and angular acceleration.<br><br>Body mass.<br><br>Linear and angular momentum.<br><br>Net force/torque (consistent with 𝐹 = 𝑚𝑎).<br><br>Translational kinetic energy.<br><br>Rotational kinetic energy.<br><br>Gravitational potential energy.<br><br>Inertia tensor.<br><br>Electromagnetic potential energy (when applicable).|
|Contact Contact|normal_force friction_force<br><br>|Normal contact force at interaction points. Tangential/frictional contact force.|
|String String String String|length<br><br>velocity<br><br>force<br><br>stiffness<br><br>|Current string length. Rate of change of string length. Tension force. Spring constant (for elastic strings/springs).|

Table 9 | Physical quantities recorded from MuJoCo for each simulated scene.

mass_with_reverse_movable_pulley is the reverse variant of mass_with_movable_pulley where the two connections of the pulley pull it up,

whereas in mass_with_movable_pulley the two connections of the pulley pull it down.

###### ENTITY VISUALIZATION

Figure 13 | Mass With Movable Pulley

two_side_mass_plane consists of a mass on plane which can be connected to other entities on either sides.

###### ENTITY VISUALIZATION

Figure 14 | Two Side Mass Plane

stacked_mass_plane consists of long mass blocks stacked on top of each other on a plane. Each of these mass blocks can be connected to other entities on either side.

###### ENTITY VISUALIZATION

Figure 15 | Stacked Mass Plane

directed_mass consists of mass block suspended from two fixed pulleys. The other ends of each of these pulleys can be connected to other entities.

###### ENTITY VISUALIZATION

Figure 16 | Directed mass

mass_prism_plane consists of a movable inclined plane and two mass blocks on either side of it. These mass blocks are connected to each other by a string.

###### ENTITY VISUALIZATION

Figure 17 | Mass Prism Plane

mass_box_plane consists of a large movable mass block and optional mass blocks on either face of it. These mass blocks are connected to each other by a string.

###### ENTITY VISUALIZATION

Figure 18 | Mass Box Plane

twoD_collision_plane consists of a large frictionless plane and a couple of spheres on top it, each given with some initial velocity.

###### ENTITY VISUALIZATION

|Top View<br><br>|
|---|

Figure 19 | TwoD Collision Plane

complex_collision_plane consists of a long frictionless plane and a couple of objects on top it, each given with some initial velocity. This setup is entirely 1D to lower complexity of the problems. Possible objects are sphere, block, fixed wall and spring blocks.

###### ENTITY VISUALIZATION

|Top View<br><br>|
|---|

Figure 20 | Complex Collision Plane

solar_system consists of a stationary star and a couple of planets revolving around it.

###### ENTITY VISUALIZATION

Figure 21 | Solar System

rocket_entity consists of a stationary planet and a rocket taking off of the planet. The rocket has a dry mass 𝑚0 and initial mass 𝑚. It burns fuel to propel itself, losing mass at a rate of 𝜇.

###### ENTITY VISUALIZATION

Figure 22 | Rocket Entity

rotation_entity consists of multiple 3D shapes attached to each other with rigid joints so that they move together. Additionally, they are attached to a pivot, allowing them to rotate around it due to gravity in a pendulum motion.

###### ENTITY VISUALIZATION

###### pivot

Figure 23 | Rotation Entity

rolling_entity consists of 3D shapes rolling on an inclined plane. We choose primitive 3D shapes such as spheres, cylinder and polygon cylinders. Additionally we also randomly cutout a shape from the body-for instance cutting a smaller sphere from a sphere results in a spherical shell. We automate this by using blender to generate arbitrary cutout shapes.

###### ENTITY VISUALIZATION

Figure 24 | Rolling Entity

em_entity consists of a moving charged particle in space in the presence of varying Electric ( 𝐸) and Magnetic fields ( 𝐵). These fields vary as a known function of the position, velocity of the particle and time.

###### ENTITY VISUALIZATION

|Top View<br><br>|
|---|

Figure 25 | Electro Magnetism Entity

###### IPhO 2005 Q1: An Ill Fated Satellite

A geosynchronous satellite of mass 𝑚 is in a circular orbit of radius 𝑟0 with velocity 𝑣0. An error causes the engine to fire, providing an instantaneous radial thrust Δ𝑣 directed towards Earth. We characterize this boost by the parameter:

Δ𝑣 𝑣0

𝛽 =

Task: Calculate the minimum boost parameter 𝛽esc needed for the satellite to escape Earth’s gravity.

###### △ BASE MODEL

###### ✓ RL-FINETUNED MODEL

- Step 1: Orbital Velocity 𝑣0 = √︁𝐺𝑀𝐸/𝑟0

- Step 2: Escape Condition Escape velocity: 𝑣esc = √︁2𝐺𝑀𝐸/𝑟0 Condition: 𝑣after ⩾ 𝑣esc

Since the thrust is directed radially inward, the effective velocity becomes reduced: 𝑣after = 𝑣0 − Δ𝑣

ERROR: Linear subtraction. Fails to treat velocity as a vector (radial ⊥ tangential).

- Step 3: Solve for 𝛽 √︁𝐺𝑀/𝑟 − 𝛽√︁𝐺𝑀/𝑟 ⩾ √︁2𝐺𝑀/𝑟 (1 − 𝛽) ⩾ √2 =⇒ 𝛽 ≈ −0.41 (Model confuses signs and algebra) 𝛽 ⩾ √2 − 1 ≈ 0.41

- Step 1: Energy Analysis

Initial Energy: 𝐸0 = −𝐺𝑀𝑚/2𝑟0 Escape Velocity needed: 𝑣esc = √︁2𝐺𝑀/𝑟0

- Step 2: Vector Addition

The kick is radial (Δ𝑣), perpendicular to the orbital velocity (𝑣0). The new velocity magnitude adds in quadrature:

𝑣new = √︃𝑣02 + (Δ𝑣)2

Correct: Quadrature Addition

- Step 3: Escape Condition 𝑣new ⩾ 𝑣esc √︃𝑣02 + (Δ𝑣)2 ⩾ √2𝑣0

Substitute√︃𝑣02 + 𝛽2𝑣02Δ⩾𝑣 √=2𝛽𝑣𝑣00:=⇒ √︁1 + 𝛽2 ⩾ √2

- Step 4: Solve for 𝛽 1 + 𝛽2 ⩾ 2 =⇒ 𝛽2 ⩾ 1 =⇒ 𝛽 ⩾ 1 Final Answer:

Final Answer:

|𝛽esc =<br><br>√<br><br>2 − 1 ≈ 0.41|
|---|

|𝛽esc = 1|
|---|

- Figure 26 | LLM answers before (left) and after (right) RL finetuning. Question adapted from IPhO 2005 Q1 “An Ill Fated Satellite”.

###### JEE Advanced 2023 Paper 1

A small mass (𝑚 = 0.10 kg) moving with initial speed 𝑢 = 5.00 m/s collides elastically with a bar (Mass 𝑀 = 1.00 kg, Length 𝐿 = 0.20 m) that is pivoted at one end. The collision occurs at a distance 𝐿/2 from the pivot.

###### Parameters:

- • 𝑚 = 0.10 kg; 𝑢 = 5.00 m/s
- • 𝑀 = 1.00 kg; 𝐿 = 0.20 m
- • 𝐼bar (about pivot) = 𝑀𝐿2/3

The velocity of the small mass (𝑣) after the collision is closest to: [A] 4.30 m/s [B] 5.00 m/s [C] −2.50 m/s [D] 2.15 m/s

###### △ BASE MODEL

###### ✓ RL-FINETUNED MODEL

- Step 1: Conservation of Angular Momentum 𝐿initial = 𝑚 · 𝑢 · (𝐿/2) 𝐿initial = 0.1 × 5.0 × 0.1 = 0.005

ERROR: Arithmetic (Should be 0.05)

𝐿final = 𝐼pivot𝜔 + 𝑚𝑣(𝐿/2) 0.005 = 0.00133𝜔 + 0.005𝑣

- Step 2: Conservation of Kinetic Energy Since collision is elastic, conserve energy. 𝐾𝐸initial = 0.5𝑚𝑢2 = 1.25 J

For the bar, we consider both rotation and translation: 𝐾𝐸bar = 0.5𝐼pivot𝜔2 + 0.5𝑀𝑣cm2

ERROR: Double counting energy

(Using 𝐼pivot includes energy of COM. Adding linear KE counts twice.)

- Step 3: Solve System The model attempts to fit the values... finds 𝑣 ≈ 4.30 m/s, but associates it with wrong option due to internal confusion. Predicted Answer: [B] 5.00 m/s

- Step 1: Conservation of Angular Momentum Calculate initial angular momentum correctly: 𝐿initial = 𝑚 · 𝑢 · (𝐿/2) 𝐿initial = 0.10 × 5.00 × 0.10 = 0.05

Correct: Arithmetic 𝐿final = 𝐼pivot𝜔 + 𝑚𝑣(𝐿/2)

- 0.05 = 0.0133𝜔 + 0.01𝑣

Step 2: Conservation of Kinetic Energy For a bar pivoting at the end, KE is purely rotational about pivot: 𝐾𝐸bar = 0.5𝐼pivot𝜔2

Correct: Energy counted correctly

Equation:

- 1.25 = 0.05𝑣2 + 0.00665𝜔2

- Step 3: Solve System Using Momentum equation and substituting into Energy... Solving yields 𝜔 ≈ 5.0 rad/s and 𝑣 = 5 − 1.33(5.0) ≈ 4.30 m/s. Predicted Answer: [A] 4.30 m/s

- Figure 27 | LLM answers before (left) and after (right) RL finetuning. Question adapted from JEE Advanced 2023 Paper 1.

###### IPhO 2018 Q1: LIGO-GW150914

Calculate the dimensionless coefficient 𝜉 for the power P emitted in gravitational waves by a binary system.

###### Formula:

###### ∑︁

2

d3𝑄𝑖𝑗 d𝑡3

𝐺 5𝑐5

𝐺 𝑐5

𝜇2𝐿4Ω6

P =

= 𝜉

𝑖,𝑗

Given Quadrupole Moments (where 𝑘 = 2Ω): 𝑄𝑖𝑖 = 𝜇𝐿22 (𝑎𝑖 + 𝑏𝑖 cos 𝑘𝑡) and 𝑄𝑖𝑗 = 𝜇𝐿22 𝑐𝑖𝑗 sin 𝑘𝑡 (for 𝑖 ≠ 𝑗) Parameters: 𝑏1 = 1, 𝑏2 = −1, 𝑏3 = 0; 𝑐12 = 𝑐21 = 1 (others 0). The value of 𝜉 is: [A] 1.6 [B] 3.2 [C] 6.4 [D] 12.8

###### △ BASE MODEL

###### ✓ RL-FINETUNED MODEL

- Step 1: Compute Third Time Derivatives We differentiate the quadrupole moments three times. 𝑄(𝑡) ∼ 12 cos(2Ω𝑡) First deriv: ∼ 12 · (2Ω) = Ω Second deriv: ∼ Ω · (Ω) = Ω2 Third deriv: ∼ Ω2 · (2Ω) = 2Ω3

ERROR: Failed chain rule (missing factor)

Resulting Coefficient: 𝐴 = 2

- Step 2: Square and Sum We square the coefficient found in Step 1. Square: 𝐴2 = 22 = 4

Summing over the non-zero components: Sum = 4 (from diagonal) +4 (from off-diagonal) = 8

- Step 3: Apply Formula P = 51 · Sum · (constants) 𝜉 = 85 = 1.6 Predicted Answer: [A] 1.6 (Incorrect)

- Step 1: Compute Third Time Derivatives We apply the chain rule consistently for the third derivative. 𝑄(𝑡) ∼ 12 cos(2Ω𝑡) First deriv: 12 · (−2Ω) sin(2Ω𝑡) Second deriv: 12 · (−4Ω2) cos(2Ω𝑡) Third deriv: 12 · (8Ω3) sin(2Ω𝑡)

Correct: Consistent Chain Rule

Resulting Coefficient: 𝐴 = 4

- Step 2: Square and Sum We square the coefficient found in Step 1. Square: 𝐴2 = 42 = 16

Summing over the components (using sin2 +cos2 = 1 identity): Sum = 16 (from diagonal) +16 (from off-diagonal) = 32

- Step 3: Apply Formula P = 15 · Sum · (constants) = 325 = 6.4 Predicted Answer: [C] 6.4 (Correct)

- Figure 28 | LLM answers before (left) and after (right) RL finetuning. Question adapted from IPhO 2018 Question 1 “LIGO-GW150914”.

###### IPhO 2013 Q1

A meteoroid from Comet Encke impacted Earth (the Maribo meteorite). Measurements show the meteorite formed shortly after the solar system’s birth.

In its orbit around the Sun, the minimum and maximum distances between Comet Encke and the Sun are:

𝑎min = 4.95 × 1010 m and 𝑎max = 6.16 × 1011 m

Calculate the orbital period 𝑡Encke of Comet Encke. Express the final answer in SI units.

###### △ BASE MODEL

###### ✓ RL-FINETUNED MODEL

- Step 1: Formula Use Kepler’s Third Law: 𝑇2 = (4𝜋2/𝐺𝑀) · 𝑎3
- Step 2: Calculate Semi-major Axis

𝑎 = (𝑎min + 𝑎max)/2 𝑎 = (4.95 × 1010 + 6.16 × 1011)/2 𝑎 = 33.275 × 1010 m

- Step 3: Calculate Period Using 𝜇 = 1.327 × 1020: 𝑇2 = (33.275 × 1010)3/1.327 × 1020 𝑇2 = 3.7074 × 1033/1.327 × 1020

- Step 1: Calculate Semi-major Axis

𝑎 = (𝑎min + 𝑎max)/2 𝑎 = (4.95 × 1010 + 61.6 × 1010)/2 𝑎 = 3.3275 × 1011 m

- Step 2: Unit Conversion Strategy Convert 𝑎 to Astronomical Units (AU) to simplify Kepler’s Law (𝑇2 = 𝑎3 in years/AU). 1 AU = 1.496 × 1011 m 𝑎𝑎𝑢 = 3.3275 × 1011/1.496 × 1011 ≈ 2.225 AU
- Step 3: Apply Kepler’s Law 𝑇2 = (2.225)3 ≈ 10.96

𝑇 ≈

√10.96 ≈ 3.31 years

- Step 4: Convert to SI 1 year = 3.154 × 107 seconds 𝑇 = 3.31 × 3.154 × 107 𝑇 ≈ 1.04 × 108 seconds Final Answer:

ERROR: Arithmetic. 𝑎3 ≈ 3.7 × 1034 (Order of magnitude off).

𝑇2 ≈ 2.794 × 1013 s2 𝑇 ≈ 5.285 × 106 s 𝑇 ≈ 61.37 days Final Answer:

|1.04 × 108 s|
|---|

###### (Correct)

|61.37 days|
|---|

(Incorrect)

- Figure 29 | LLM answers before (left) and after (right) RL finetuning. Question adapted from IPhO 2013 Question 1 “The Maribo Meteorite”.

###### F=MA 2024 Q17

[Figure 27]

Consider the following system of massless and frictionless pulleys, ropes, and springs. Initially, a block of mass 𝑚 is attached to the end of a rope, and the system is in equilibrium. Next the block is doubled in mass, and the system is allowed to come to equilibrium again. During the transition between these equilibria, how far does the end of the rope (where the block is suspended) move?

###### INVENTED ENTITIES AND NATURAL-LANGUAGE DESCRIPTIONS

- Entity 1: 3-point-movable-pulley A movable pulley of mass <mass:float> kg is used as a reusable transmission node. It exposes independent left and right branches and a center hook on the

[Figure 28]

<side:up/down> side.

- Entity 2: anchored spring

A spring with stiffness <k:float> N/m and natural length <length:float> m is anchored to a fixed support. The spring points toward <dir:up/down/left/right> and its movable endpoint mass is <m:float> kg. The same endpoint is exposed for connection to an external light string.

[Figure 29]

###### △ DIRECT XML ATTEMPT ✓DSL-GENERATED SIMULATION

[Figure 30]

[Figure 31]

- Figure 30 | Scaling up the DSL using LLMs. In this experiment, we task the LLM to extend our entity vocabulary by inventing new entities to support simulation of this question from F=MA

2024. We observe that while modern LLMs fail to simulate a target scene by generating raw simulator code (left), they can do so by extending our DSL with novel entities (right).

###### USA PHO 2019 B3

A bead of mass 𝑀 slides frictionlessly along a horizontal rail. It is attached to a rigid, massless rod of length 𝑅 with a ball of mass 𝑀 at the other end. The system is initially stationary with the ball directly above the bead (𝜃 = 0) before receiving an infinitesimal horizontal push. The rail only constrains the bead, allowing the rod and ball to pass through unhindered.

[Figure 32]

###### INVENTED ENTITIES AND NATURAL-LANGUAGE DESCRIPTIONS

- Entity 1: slider mass A carriage of mass <mass:float> kg is constrained to horizontal translation. The carriage has an exposed top pivot and side connectors for attaching strings.

[Figure 33]

- Entity 2: light rod pendulum

[Figure 34]

A rigid rod of length <length:float> m carries a bob of mass <mass:float> kg. The assembly rotates about a top pivot, initially tilted at an angle of

<slope:float> degrees from horizontal.

###### ✓DIRECT XML ATTEMPT ✓DSL-GENERATED SIMULATION

[Figure 35]

[Figure 36]

- Figure 31 | Scaling up the DSL using LLMs. In this experiment, we task the LLM to extend our entity vocabulary by inventing new entities to support simulation of this question from USA

PhO 2019. In this case, we observe that LLMs are able to simulate a target scene by generating raw simulator code (left), and by extending our DSL with novel entities (right).

###### JEE ADVANCED 2019 PAPER 2

A block of mass 2𝑀 is attached to a massless spring with spring constant 𝑘. This block is connected to two other blocks of masses 𝑀 and 2𝑀 using two massless pulleys and strings. The accelerations of the blocks are 𝑎1, 𝑎2, and 𝑎3 as shown in the figure. The system is released from rest with the spring in its unstretched state. The maximum extension of the spring is 𝑥0. Which of the following option(s) is/are correct? [𝑔 is the acceleration due to gravity. Neglect friction.]

[Figure 37]

INVENTED ENTITIES AND NATURAL-LANGUAGE DESCRIPTIONS

###### Entity 1: spring mass

A block of mass <mass:float> kg can slide on a plane inclined at <angle:float> degrees. Its left side is attached to a spring fixed to a wall. The spring has stiffness <k:float> N/m and natural length <length:float> m. The right side of the block is available for connection to another entity by a light string.

[Figure 38]

###### △ DIRECT XML ATTEMPT ✓DSL-GENERATED SIMULATION

[Figure 39]

[Figure 40]

- Figure 32 | Scaling up the DSL using LLMs. In this experiment, we task the LLM to extend our entity vocabulary by inventing new entities to support simulation of this question from JEE

Advanced 2019. We observe that while LLMs fail to simulate a target scene by generating raw simulator code (left), they can do so by extending our DSL with novel entities (right).

###### MUJOCO OMNIVERSE

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

- Figure 33 | We task LLMs to port a subset of our entities from MuJoCo simulator to Omniverse simulator, to evaluate the portability of our DSL across simulators. LLMs could successfully

transfer DSL entities across simulators, which could potentially help expand the scope of simulator-based synthetic data generation.

