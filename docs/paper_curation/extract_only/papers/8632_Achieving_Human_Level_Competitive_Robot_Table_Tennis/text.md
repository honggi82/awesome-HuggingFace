## Achieving Human Level Competitive Robot Table Tennis

# arXiv:2408.03906v3[cs.RO]1May2025

David B. D’Ambrosio1,∗, Saminda Abeyruwan1,∗, Laura Graesser1,∗, Atil Iscen1, Heni Ben Amor2, Alex Bewley2, Barney J. Reed2,†, Krista Reymann2, Leila Takayama2,§, Yuval Tassa2, Krzysztof Choromanski, Erwin Coumans, Deepali Jain, Navdeep Jaitly, Natasha Jaques, Satoshi Kataoka, Yuheng Kuang, Nevena Lazic, Reza Mahjourian, Sherry Moore, Kenneth Oslund, Anish Shankar, Vikas Sindhwani, Vincent Vanhoucke, Grace Vesom, Peng Xu, and Pannag R. Sanketi1

### Google DeepMind

1Primary contributors, ∗Corresponding authors (order randomized, equal contributions), 2Core contributors (Alphabetized), †Work done at Google DeepMind via Stickman Skills Center LLC, §Work done at Google DeepMind via Hoku Labs.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Fig. 1: Our table tennis robot playing against a professional coach. The green dots show the trajectory of the ball during the rally. The table tennis robot is a 6 DoF ABB 1100 arm mounted on top of two Festo linear gantries, enabling motion in the 2d plane. The x gantry, which moves side to side across the table, is 4m long and the y gantry, which moves towards and away from the table, is 2m long. A 3d printed paddle handle and paddle with short pips rubber is attached to the arm.

Abstract—Achieving human-level speed and performance on real world tasks is a north star for the robotics research community. This work takes a step towards that goal and presents the first learned robot agent that reaches amateur human-level performance in competitive table tennis. Table tennis is a physically demanding sport which requires human players to undergo years of training to achieve an advanced level of proficiency. In this paper, we contribute (1) a hierarchical and modular policy architecture consisting of (i) low level controllers with their detailed skill descriptors which model the agent’s capabilities and help to bridge the sim-toreal gap and (ii) a high level controller that chooses the low level skills, (2) techniques for enabling zero-shot sim-to-real including an iterative approach to defining the task distribution that is grounded in the real-world and defines an automatic curriculum, and (3) real time adaptation to unseen opponents. Policy performance was assessed through 29 robot vs. human matches of which the robot won 45% (13/29). All humans were unseen players and their skill level varied from beginner to tournament level. Whilst the robot lost all matches vs. the most advanced players it won 100% matches vs. beginners and 55% matches vs. intermediate players, demonstrating solidly amateur human-level performance. Videos of the matches can be viewed here1.

1See sites.google.com/view/competitive-robot-table-tennis

I. INTRODUCTION

Robot learning has made inspiring progress in recent years, resulting in robots that can cook [1], clean up [2], or even perform backflips [3]. While the capabilities of learned robot policies have increased dramatically, achieving humanlevel performance in terms of accuracy, speed and generality still remains a grand challenge in many domains. One such domain is table tennis – a physically demanding sport which requires human players to undergo years of training to achieve an advanced level of proficiency. Indeed, competitive matches are often breathtakingly dynamic, involving complex motion, rapid eye-hand coordination, and highlevel strategies that adapt to the opponent’s strengths and weaknesses. For a robotic table tennis system to mimic these abilities it needs high-speed motion, precise control, realtime decision-making and human-robot interaction. Thanks to these demanding requirements, table tennis provides a rich environment to advance robotic capabilities and has served as a benchmark for robotics research since the 1980s [4]. Numerous table tennis robots have been developed since and progress has been made on returning the ball to the opponent’s side [5], hitting to a target position [6], smashing [7],

cooperative rallying [8], and many other critical aspects of table tennis [9]. Yet no prior work has tackled the competitive game in which a robot plays a full game of table tennis against a previously unseen human opponent.

In this paper, we present the first learned robot agent that can play competitive table tennis at human level,

- as depicted in Figure 1. The robot uses a combination of techniques (known and novel) in order to acquire skills at different levels of abstraction. Table tennis players must be prepared to return balls across a wide variety of positions, speeds, and spins (i.e. angular velocities) and competitive players must know how to manipulate these factors to set up advantageous plays or exploit opponent weaknesses. Thus, there are two levels of play: the high level strategic decisions and the low level physical skills required to execute those strategies. This organization adds an additional layer of challenge to robotic sports where, unlike a purely strategic game like chess or go, the policy not only needs to decide the most advantageous move, but also needs to have the physical skills to perform it and may even have to choose a less strategically optimal action if it is not confident in successful execution. To address this challenge, we propose a hierarchical and modular policy architecture. Our system consists of multiple low-level skill policies and a highlevel controller that selects between them. Each low-level skill policy specializes in a specific aspect of table tennis, such as forehand topspin, backhand targeting, or forehand serve. Training is efficient — each skill builds on top of the same foundation policy for a given category (e.g. forehand, backhand), and once a good skill has been trained it can always be subsequently specialized. In addition to learning the policy itself, we collect and store information both offline and online about the strengths, weaknesses, and limitations of each low-level skill. The resulting skill descriptors provide the robot with important information regarding its abilities and shortcomings. In turn, a high-level controller, responsible for orchestrating the low-level skills, selects the optimal skill given the current game statistics, skill descriptors and the opponent’s capabilities.

At its core, our paper aims at scaling robot learning to complex physical tasks which may involve a human partner or adversary. There are two predominant paradigms in the robot learning community. Reinforcement Learning (RL) [10] is the preferred choice for dynamic control tasks, e.g., quadruped locomotion [11], [12], [13]. Due to its high sample complexity, RL is often first performed in simulation and later transferred to the real world, thereby speeding up training time. However, ensuring the distribution of simulated tasks is grounded in reality can be quite challenging. For example, in table tennis, the set of initial ball conditions at every hit that competitive play will induce is very different from a set of “all possible” initial ball conditions. By contrast, imitation learning (IL) [14] is the prevalent choice in tasks where demonstrations can be collected from an expert user or process (typically gathered through teleoperation or motion capture), e.g., manipulation [15], [16]. IL from real world data is anchored to examples that are known to solve

the desired task. Additionally it bypasses the challenges of simulating the task(s) and the inevitable sim-to-real gap. However, it typically requires multiple demonstrations for every single task [14] and may not cover critical parts of the state space. In this work, we introduce a hybrid training method that combines the best of both worlds. We collect a small amount of human-human play data to seed the initial task conditions. We then train an agent in simulation using RL and employ a number of techniques (known and novel) to deploy the policy zero-shot to real hardware. This agent plays with humans to generate more training task conditions and the training-deployment cycle is repeated. As the robot improves, the standard of play becomes progressively more complex whilst remaining grounded in real-world task conditions. This approach iteratively refines the skill repertoire based on real-world data. As the robot plays, it gathers data and gaps are revealed in its capabilities, which are then addressed through continued training in simulation. This hybrid sim-real cycle creates an automatic task curriculum and enables the robot’s skills to improve over time.

A final, yet critical, ingredient of our approach is the ability to rapidly adapt to the unseen human opponent’s capabilities and play style. [12] adapts to various environmental conditions for locomotion but not to humans or adversarial conditions. We enable rapid adaptation to the opponent by tracking in real-time the match statistics representing the robot’s and opponent’s strengths and weaknesses. Additionally, for each low-level skill we estimate preferences [10] online to augment the offline skill statistics. These two are used to derive the strategy. This online adaptation helps the controller to adapt to novel opponents and allows the robot to learn and refine its decision-making process, leading to improved robustness against a diverse set of human opponents.

The components described above ultimately lead to competitive gameplay at human level that humans actually enjoy playing with — the resulting policy is flexible, adaptable, extensible, and more interpretable than a monolithic system. In summary, in this paper we introduce the first robot learning system to achieve amateur human level performance in an interactive competitive sport against unseen human opponents demonstrated through a user study. To achieve this goal, we make the following technical contributions (1) a hierarchical and modular policy architecture, specifically the (i) low level controllers with their skill descriptors and (ii) a high level controller that chooses the low level skills; (2) techniques to enable zero-shot sim-to-real including an iterative approach to defining the training task distribution that is grounded in the real-world and defines an automatic curriculum, (3) real time adaptation to unseen opponents and (4) A user-study to test our model playing actual matches against unseen humans in physical environments.

II. METHOD A. Hardware, problem setting, and environment

Figure 1 depicts the physical robot. The table tennis robot is a 6 DoF ABB IRB 1100 arm mounted on top of two

Training (In Simulation)

###### Deployment (In Real)

Observation, reward

Action

###### Low level skill library (LLC)

###### Environment

###### Agent

Environment

###### Agent

Inference HLC

Forehand serve

MuJoCo physics + [9]

Vision Ball position

Selected LLC

50Hz, velocity control, & Joint space

Policy BGS

H-Values: Online Opponent model

Backhand serve

MoCap Paddle position & orientation

Action

Adaptation

Topspin adapter

Policy Skill descriptor Cache Control ﬂow Forehand/

Heuristics: Shortlist Frozen LLCs

Ball velocity Robot joint positions Robot fault status

Underspin adapter

Forehand rally

###### High level controller (HLC)

Style Selector: Forehand or Backhand

Backhand: Frozen Style Selector

Skill descriptor

Backhand rally

Referee

Real Inference

Adaptation

Observation, reward

- Fig. 2: Method overview. We train a skill library of low-level controllers (LLCs), including serving and rallying, and simto-sim adapters from a dataset of ball states. Using the same ball states, we train a high level controller (HLC) for style selection. The policies are trained purely in simulation (but using real ball states) using Blackbox Gradient Sensing (BGS) [9], [8]. The policies transfer zero-shot to the physical world. At deployment time, we freeze the style selector and skills. During inference HLC uses the style selector to select the side. The heuristics module shortlists the most effective skills. H-values (online opponent model) select the most preferred skill, and the skill executes the actions.

Festo linear gantries, enabling motion in the 2d plane. The x gantry, which moves side to side across the table, is 4m long and the y gantry, which moves towards and away from the table, is 2m long. A 3D printed paddle handle and paddle with short pips rubber [17] is attached to the ABB arm. A pair of Ximea MQ013CG-ON cameras operating at 125Hz capture images of the ball and these are used as input into a neural-perception system [9] which produces ball positions

system which tracks the ball, the motion capture system which tracks the human player paddle pose, a state machine that tracks the state of the game, and an observation module which provides data such as ball position and velocity, robot position, etc., to the policy. In addition, we also built a corresponding simulation environment built on top of the MuJoCo [19] physics engine. A detailed description of the basic system can be found in [9]. Below, we describe changes to this system that were made to enable real-time competitive play with humans.

- at the same frequency. We use a PhaseSpace motion capture system consisting of 20 cameras mounted around the play area to track the human opponent’s paddle.

B. Hierarchical agent architecture and training overview

We model table tennis as a single-agent sequential decision making problem in which the human opponent is modeled as part of the environment. We use the Markov Decision Process (MDP) [18] formalization. This consists of a of a 4-tuple (S, A, R, p), whose elements are the state space S, action space A, reward function R : S × A → R, and transition dynamics p : S × A → S. An episode (s0,a0,r0,...,sn,an,rn) is a finite sequence of s ∈ S, a ∈ A, r ∈ R elements, beginning with a start state s0 and ending when the environment terminates. An episode consists of a single incoming ball, which is hit and returned, beginning at the moment the opponent’s paddle contacts the ball and ends when either of the following conditions occur 1.) the robot returns the ball, 2) the ball goes out of play, or 3) the robot misses the ball. A ball return means that the robot hits the ball such that it bounces on the opponent’s side of the table without first bouncing on the robot’s side. The objective for the robot is to maximize the expected return rate over the ball distribution.

Our agent architecture and approach to training are both designed to address the numerous challenges presented by competitive table tennis with humans. The table tennis agent shown in Figure 2 consists of two levels of control which we refer to as the high level controller (HLC) and the low level controllers (LLCs). LLCs are policies representing different table tennis skills and are trained to produce joint velocity commands at 50Hz. For example LLCs may represent playing with a forehand style and hitting cross-court balls, playing backhand conservatively, or playing forehand to return underspin serves.

The HLC is responsible for selecting which LLC should be executed every incoming ball episode. The HLC does not have a fixed control frequency but instead is triggered to act once every time the opponent hits the ball. Within the HLC, there are six components that are combined to produce the choice of LLC; (1) Style: this is a policy trained to choose the play-style, forehand or backhand given the incoming ball, (2) Spin classifier: this classifier provides information about the

The real environment consists of the neural-perception

spin of the incoming ball identifying topspin or underspin, (3) LLC skill descriptors: these are a model of the agent’s own capabilities. They provide performance metadata for every LLC such as estimated return rate, ball hit velocity, and land position, conditioned on the specific incoming ball, (4) Match statistics about the opponent and robot’s performance, (5) Strategies: these take (1), (3) and (4) as input and output a shortlist of LLCs, and (6) LLC preferences (H-values): these estimate the performance of each LLC for the current player and are updated after every shot. The HLC combines (3), (5), and (6) to produce the final choice of LLC. The entire control flow within the HLC happens within 20ms.

We chose to train multiple, modular LLCs instead of a single monolithic LLC for a number of reasons: Avoiding catastrophic forgetting — once a good skill has been learned it is never forgotten, whilst still serving as an initialization point for further skill learning. Extensibility — new skills are straightforward to incorporate by adding a new LLC. Evaluation efficiency — which in turn speeds up experimental velocity. Once a low level skill has been tested in the real world, its capabilities are well understood and it does not have to be re-tested. In contrast a monolithic learned system will need to be tested on the full suite of expected capabilities every time the model weights change. Fast inference inference for each LLC takes 3ms on a CPU.

The LLCs and HLC style policy were trained iteratively, alternating between simulated training and zero-shot deployment in the real world during which human opponents play with the robot. The human-robot interactivity inherent in the task motivated doing all training entirely in simulation. Finetuning complex skills with humans in the loop in the real world is too time consuming to be feasible, especially since prior work [8] showed it took 6 hours of real-world fine tuning to train a policy to cooperate with a single human. This led to a substantial effort to narrow the sim-to-real gap.

C. LLC training

LLCs are meant to provide a library of skills that our HLC can deploy in its strategies. Our approach to training them can be summarized in three steps. (1) Train two generalist base policies, one for each main play style (forehand, backhand) and add this to the set of LLCs. (2) Specialize policies to different skills by adding reward function components and / or adjusting the training data mix before fine-tuning a new policy initialized from one of the existing LLCs. This is typically one of the generalist base policies but could be any policy in the LLC set. (3) Evaluate the new policies and assess if a policy exhibits the desired characteristics. For example, if a policy is trained to target a particular location on the table, calculate the average error between the ball landing position and target. If successful, add the policy to the set of LLCs.

Training algorithm All policies were trained in simulation with Blackbox Gradient Sensing (BGS) [8], an evolutionary strategies (ES) algorithm, on the task described in Section II-A. The training task distribution of initial ball states is sampled from a real world dataset, gathered

iteratively through multiple cycles of policy training and real world evaluations. Creating this dataset is discussed in detail in Section II-E. We chose BGS because we observed that it produced policies with relatively smooth actions, whereas policies trained with RL algorithms such as PPO [20] or SAC

- [21] produced noticeably jerkier actions. Additionally, it has been shown to have strong sim-to-real transfer performance [9]. We hypothesize that action smoothness and potentially less overfitting to the simulator are the main reasons why BGS-trained policies exhibit such good transfer.

Network architecture Each policy is a dilated-gated CNN

- [22] following the architecture in [23] with 10k parameters and an optional FILM adapter layer of 2.8k parameters designed to aid sim-to-real transfer (see Section II-E). The CNN contains 1D convolutions, convolving across timesteps.
- [23] found this accelerated learning and led to smoother outputs. The observation space is (8,16) consisting of 8 consecutive timesteps of ball position and velocity (6), robot joint position (8), and one-hot style; forehand or backhand

(2). The style component is an artifact of early experiments and likely could be removed without affecting performance. All policies output actions with dimension (8,) representing joint velocities at 50Hz. 8 timesteps is 0.14 seconds of history which was empirically determined to be sufficient to smooth out noise in the trajectory and give context to the current state.

Training generalist base policies To train for a particular style (forehand, backhand), each ball state in the dataset was annotated with forehand, backhand, or center based on where the ball trajectory intersected with the back of the table on the robot side. center was defined as +/- 0.2m around the center of the table, forehand as > 0.2, and backhand as < −0.2. Forehand policies were trained on only forehand + center balls, backhand on backhand + center. This created an overlap in the center where policies of either style are capable of returning the same balls. The policy was also rewarded for moving towards a reference pose (either forehand or backhand) at the beginning of the shot. Without such a reward we observed that the robot would sometimes employ a backhand pose to hit forehand balls even though it was less efficient. These base policies are important, not only to have a strong starting polices capable of returning a wide range of balls to branch from, but also to anchor play in specific styles for efficient returns. Base polices were trained for roughly 2.4 billion simulation steps across 6,000 parallel simulation workers.

Training specialists We experimented with the types of skills to train for based on advice from a table tennis coach and general game intuition, including targeting specific return locations, maximizing return velocity, and specializing to return serves of either topspin or underspin, fast balls, and lobs. We found that we did not need a specialist to handle lobs, and were unable to train a specialist on fast balls due to lack of data and hardware limitations. We therefore focused on developing serving, targeting and fast hitting specialists in addition to the generalists. Specialists were further trained for roughly 300-1200 million additional simulation steps

[Figure 6]

- Fig. 3: LLC training lineage. LLC x = ID of the LLC in the final system. The forehand (FH) and backhand (BH) LLCs were each developed from two independently trained generalists. One of the generalists was developed along with the dataset cycles, whilst the other was trained only after finalizing the dataset. Both the seed forehand generalists were deployed (LLC 0 and LLC 2) whilst for the backhand only one of the seed generalists was deployed (LLC 9).

depending on convergence.

Determining the total set of skill policies The final system contained 17 LLCs. 4 were specialized for returning serves, 13 for rallying. 11 played with a forehand style, 6 with a backhand style. Importantly, each policy had the same initial robot pose, enabling straightforward sequencing of LLC choices, since the initial robot pose will be indistribution for all LLCs. The training lineage along with brief descriptions of each LLC is shown in Figure 3. We kept training LLCs until we had covered our target set of skills — consistent generalists, targeting, fast hits, topspin and underspin serves. Beyond that it was desirable to have varied playstyles and to provide options to the HLC. ES training meant that policies trained with the same objective could exhibit different behaviors, and due the modular architecture, there was little downside in including additional LLCs. Thus, if we had a strong LLC, we included it, even if there was already an LLC covering that particular skill. Forehand training produced a wider set of viable LLCs and thus more were included.

D. The High Level Controller (HLC)

The HLC is responsible for making strategic decisions — e.g. where to return the ball, how fast to hit, how much risk to take. An overview of the decision-making process is given in Algorithm 1 and Figure 4. Currently the HLC is only capable of executing simple strategies and is an initial proof of concept of the entire system. However, the selection strategies sub-component is straightforward to replace with a more expressive implementation, which could even include a fully learned model. Next we give details the HLC components and how they are combined to produce an action.

1) Event-driven decisions: The HLC action is triggered by the opponent hitting the ball (i.e. an event external to the agent). One timestep after the opponent hits the ball, the

Algorithm 1 Pseudocode for HLC inference

- 1: procedure HLC ACT(B, S, πθSt, πθSp, oH, oSp, Λ, Γ, Ψ, H)

- 2:
- 3: Input:
- 4: B: [bx, by, bz, b˙x, b˙y, b˙z], ball state
- 5: S: is the ball a serve?
- 6: πθSt: style policy
- 7: πθSp: spin policy
- 8: oSt: πθSp observation, contains B
- 9: oSp: πθSp observation
- 10: Λ: LLCs
- 11: Γ: LLC skill descriptors
- 12: Ψ: set of LLC selection strategies
- 13: H: current preferences, one per LLC
- 14: Output:
- 15: λ: selected LLC
- 16:
- 17: st = πθSt(oSt) choose style
- 18: if S then select serving LLC
- 19: sp = πθSp(oSp) estimate spin type
- 20: λ = get serve llc(st, sp, Λ)

- 21: else select rallying LLC
- 22: get metrics from LLC skill descs.
- 23: γ = get llc metrics(Γ, B, st)

- 24: Λ∗ = [] LLC shortlist
- 25: R = [] prob. of returning ball B
- 26: for ψ ∈ Ψ do
- 27: get best LLC per strategy
- 28: λψ, Rψ = ψ(B, γ, st, Λ)
- 29: Λ∗ += λψ add λψ to shortlist
- 30: R += Rψ add return prob. Rψ to R
- 31: end for
- 32: estimate H values for selected LLCs
- 33: Hˆ = H[Λ∗] + R
- 34: p = softmax(Hˆ)
- 35: sample LLC proportional to prob. p
- 36: λ = sample(Λ∗, p)
- 37: end if
- 38:
- 39: return λ
- 40:
- 41: end procedure

Forehand LLC Descriptors

Heuristic Strategies

|Location|
|---|
|1.2] 1.2] 1.2] 1.2]<br><br>Land Location<br><br>[0.5, 1.2]|
|1.2] 1.2] 1.2]<br><br>[0.5, 1.2] [0.5, 1.2] [0.5, 1.2]<br><br>Land Location<br><br>[0.5, 1.2]|
|1.2] 1.2]<br><br>…<br><br>[0.5, 1.2] [0.5, 1.2] [0.5, 1.2]<br><br>[0.5, 1.2] [0.5, 1.2] [0.5, 1.2]<br><br>Land Location<br><br>[0.5,|
|[0.5, 1.2][0.5, 1.2][0.5,|
|[0.5, 1.2] …<br><br>[0.5, 1.2] [0.5, 1.2]<br><br>[0.5, [0.5,|
|[0.5, 1.2][0.5,|
|[0.5, 1.2] …<br><br>[0.5, [0.5, [0.5,|
|[0.5, …|

Ball State

Land Rate

Land

Speed

[0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] …

0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 …

[0.5, [0.5, [0.5, [0.5, [0.5, [0.5, [0.5, [0.5, [0.5,

6.5 6.5 6.5 6.5 6.5 6.5 6.5 6.5 6.5 …

Ball State

Land Rate

Speed

Exploit Weak Side

[0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] …

0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 …

6.5 6.5 6.5 6.5 6.5 6.5 6.5 6.5 6.5 …

Return Fast

Ball State

Land Rate

Speed

Random

[0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] …

0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 …

6.5 6.5 6.5 6.5 6.5 6.5 6.5 6.5 6.5 …

Ball State

Land Rate

Speed

[0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] …

0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 …

1.2] 1.2] 1.2] 1.2] 1.2] 1.2] 1.2] 1.2] 1.2]

6.5 6.5 6.5 6.5 6.5 6.5 6.5 6.5 6.5 …

Return Far

Player Skill

Style Policy: Forehand or Backhand

Ball State x,y,z,vx,vy,vz

Once per hit

###### Backhand LLC Descriptors

LLC Shortlist

Ball State

Land Rate

Land Location

Speed

[0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] …

0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 …

[0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] …

6.5 6.5 6.5 6.5 6.5 6.5 6.5 6.5 6.5 …

Ball State

Land Rate

Land Location

Speed

50Hz

If serve

[0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] …

0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 …

[0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] …

6.5 6.5 6.5 6.5 6.5 6.5 6.5 6.5 6.5 …

Ball State

Land Rate

Land Location

Speed

[0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] …

0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 …

[0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] …

6.5 6.5 6.5 6.5 6.5 6.5 6.5 6.5 6.5 …

Ball State

Land Rate

Land Location

Speed

[0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] [0,1,2,3,4,5] …

0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 …

[0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] [0.5, 1.2] …

6.5 6.5 6.5 6.5 6.5 6.5 6.5 6.5 6.5 …

Online LLC Preferences

###### Serving LLCs

Spin Classiﬁer

Forehand Top Spin

Forehand Back Spin

Backhand Top Spin

Backhand Back Spin

Robot Action

Chosen LLC

- Fig. 4: Once per ball hit, the HLC decides which LLC to return the ball with by first applying a style policy to the current ball state to determine forehand or backhand (in this example we demonstrate choosing forehand). If the ball is a serve it will attempt to classify the spin as topspin or underspin and pick the corresponding LLC. Otherwise it must determine which of the many rallying LLCs will perform best by finding the most similar ball state within the corresponding set of LLC skill tables and getting the return statistics. Heuristic strategies are applied to these statistics and produce a shortlist of candidate LLCs and the final LLC is chosen through a weighted selection. The chosen LLC will be queried at 50Hz with the current ball state to determine the robot actions.

HLC makes a decision that applies until the opponent hits the ball again. The robot only starts moving after the HLC has decided on the strategy for that ball. Waiting one step gives the policy sufficient information to make a decision. We also tried zero and three steps, but three steps did not give the robot enough time to react to faster balls and zero steps did not allow for an accurate estimation of ball velocity. The decision is only made once since switching LLCs mid-swing resulted in policies ending up in states that were outside of their training distribution (i.e. the robot arm was not where they expected it to be because the previous LLC moved it somewhere new), thus performing poorly.

C) and stack the latest 8 observations to form the observation. This observation space is an artifact of previous iterations of the HLC that required longer histories. For the current implementation that makes a single, early decision per ball a much smaller history would be preferable. The action space is (2,) representing a one-hot categorical choice between forehand and backhand.

To train the style policy, we first selected a general-purpose forehand and backhand LLC and froze their weights. Then, we selected all available ball states (including reflections) and trained the style policy to maximize the expected ball landing rate. Although we trained the style policy with rally ball states, we found that it generalized to serving ball states. Therefore, we used a single policy for both serving and rallying phases of the game.

2) Style policy: The style policy determines if the robot should return the ball with a forehand or backhand style. A naive heuristic would be to simply divide the table in half and choose a style based on which half the ball will end up on. However, such an approach neglects many strategic tradeoffs such as forehand shots being easier for the opponent to smash and the ambiguous nature of balls near the center. Additionally, noise in the real world and our inability to fully capture spin may mean our physics estimates are inaccurate. By learning a style policy the HLC can understand the strengths of individual LLCs and compensate for systematic inaccuracies, leading to better overall strategic decisions.

3) Spin classifier: The spin classifier is a binary classifier that determines if the incoming serve was hit by the humans as a topspin or an underspin. This is crucial for selecting the appropriate LLC for services because, unlike the rallying play, we found it very challenging to have a common policy that handled both topspin and underspin serves. To train the model, we built a dataset of paddle and ball states from the serving dataset (see Section II-E). Specifically, we record a history of the 6 timestamps of ball and paddle states directly before the paddle made contact with the ball. The observation space is (18,) (see Appendix VI-E for details), the policy is a 2-layer MLP and it outputs the probability that the incoming

The style policy architecture, similar to the LLC but with only 4.5k parameters, has a (8, 128) observation space. We flatten the LLC (8, 16) observation (described in Section II-

ball is topspin or underspin.

During inference, to increase the precision on underspin (the less common of the two spin types), we required that the classifier predict underspin on at least 4 out of 5 consecutive past queries to be deemed an underspin prediction overall.

4) LLC skill descriptors: To excel in interactive sports, it is crucial to understand one’s own capabilities. This motivated the development of LLC skill descriptors which provide detailed metrics to the HLC on the estimated performance of each LLC for a given incoming ball. They are the agent’s model of its own capabilities and together with a model of the opponent and current game play, are the foundation of all strategic decision making.

To create the descriptors, we evaluated each LLC in simulation on all 28k ball states averaged over ten repetitions, recording the following policy metadata:

- • Initial ball position and velocity.
- • Post-paddle median hit velocity (hit velocity).
- • Ball landing location and standard deviation on the opponent’s side.
- • Ball landing rate (land rate).

This metadata was used to construct lookup tables (we used KD-Trees [24]) with keys representing initial ball position and velocity. Given any ball in play, the table can be queried for information about the likely performance of each LLC were it to be selected by averaging performance of similar balls it has seen in the past.

While we observed high zero-shot transfer rates per LLC, there remained a sim-to-real gap. Hit rates in the real world were high, however ball return rates, whilst good, were lower than the > 80% return rates we typically observed in simulation. One common failure mode was the LLCs hitting the ball just over the edge of the table. This sim-to-real gap meant that building skill descriptors using only simulated data was likely to lead to errors.

To address this, we updated each LLC’s skill descriptor using real-world data. Four researchers played with the robot with the HLC set to randomly select an LLC in order to sample them roughly equally. This resulted in 91 - 257 real world ball throws per LLC. For each LLC and for each ball collected, the 25 nearest neighbors in the relevant LLCspecific tree were updated, weighting the simulated metrics and real world metrics for a single ball throw equally on the assumption that the real world data more accurately reflected expected performance.

5) Strategies and LLC shortlist: Every time the HLC acts, five hand-coded heuristics were used to generate a shortlist (one per heuristic) of the most promising LLC candidates, given the output from the style policy and information collected by the HLC about the opponent on their ability to return balls both in total and broken down by forehand, backhand, and center returns. This opponent information is persisted between games with the same opponent. Not all heuristics use all available information.

The set of heuristics we utilized are as follows;

• Random selection: Randomly select an LLC if its land-

ing rate exceeds 80%.

- • Prioritization of hit velocity: Select the top m LLCs with the fastest hit velocities, given that their landing rates are among the top n.
- • Prioritization of landing distance: Select the top m LLCs with the farthest landing positions from the initial ball state, given that their landing rates are among the top n.
- • Exploitation of opponent’s weak side (backhand or forehand): Select an LLC that targets the opponent’s weaker side.
- • Consideration of opponent’s overall skill: For strong opponents, we assume they can hit the ball from any position on the their side. If the opponent has a hit rate above 75% we select the farthest landing position for the given ball state on the assumption that this forces the opponent to work harder to return the ball. Otherwise, select the LLC with the highest landing rate.

From the shortlist we select the LLC that will be used to return the ball with weighted sampling (to make the robot less predictable) described below.

6) LLC preferences (H-value) & choosing an LLC: Another key aspect of playing competitive sports is understanding the opponent’s capabilities and being able to adapt in response. This motivated learning online preferences for each LLC which, as well as helping to bridge the remaining sim-to-real gap, provide a rudimentary model of the human opponent.

We learned a numerical preference for each LLC, H(LLC) ∈ R, based on the LLC’s online performance. The agent selects LLCs more often if their preference is higher. However, the preference itself has no connection to reward. Only the relative preference between LLCs matters. We used a simple gradient bandit algorithm [10]2 to learn these preferences, the pseudocode is given in Algorithm 2.

For a given ball, each LLC in the shortlist is associated with an offline return rate. We combined the offline return rate and the online preferences (H-values) to select an LLC. We found that combining learned H-values with information from the skill descriptor tables played an important role in improving performance. These H-values serve two major purposes. (1) Online sim-to-real correction; even though efforts were made through the offline updates to the skill descriptor tables, a sim-to-real gap remained, likely because the sample of real world balls used to update the tables was small and generated by a small number of players. Hvalues allow the policy to quickly switch away from poorperforming LLCs to more stable ones. (2) To learn playerspecific strengths and weaknesses; if the current opponent is able to easily send shots that one LLC struggles to return, the HLC can shift weight to another the opponent can less easily exploit.

Each time an LLC was selected the H-value was updated using the binary ball land signal as the reward function. For each new opponent, these values were initialized to a set of known baseline preferences, to ensure everyone

2Chapter 2, p37

played against the same initial agent. These preferences were updated and persisted across games for the same opponent.

E. Techniques for enabling zero-shot sim-to-real

There are two core challenges in simulating robotic table tennis. First, faithfully modeling the robot, paddle, and ball dynamics. High fidelity is required because advanced table tennis play involves manipulating ball angular velocity (i.e. spin) and due to the size and weight of the ball, components such as air friction and paddle material play a much larger role in dynamics than in typical robot tasks. Second, accurately modeling the task distribution, i.e. the distribution over initial states of real-world incoming ball trajectories toward the robotic player.

In contrast to prior work [8], the enhanced simulation components presented in this section facilitated a high degree of zero-shot transfer, obviating the need for real world finetuning of low-level policies.

1) Modeling ball and robot dynamics: We enhanced the simulation environment described in [9] and [8] by incorporating the MuJoCo physics engine [19], leveraging its advanced solid state fluid dynamics for ball trajectory simulation, refining model and system identification, and improving the representation of paddle rubber.

Our simulation utilized integrated-velocity actuators, stateful actuators with an activation state coupled with an integrator and a position-feedback mechanism. The activation state corresponds to the position actuator’s setpoint, and the control signal represents the velocity of this setpoint. System identification was performed for each actuator-joint pair to ascertain parameters such as position gain (N/rad), actuator damping (N/rad/s), friction loss, joint damping (Nm/(rad/s)), force limits (Nm), and armature inertia (kg m2). Our approach to system identification aligned with the methodology presented in [25].

We utilized MuJoCo’s ellipsoid-based stateless fluid model to simulate ball trajectories. We measured the Blunt drag coefficient and used the default value for the Slender drag coefficient, while setting the Angular drag coefficient to zero. The Kutta lift coefficient and Magnus lift coefficient were kept at their default values.

Furthermore, the paddle rubber was explicitly modeled using two orthogonal passive joints representing a springdamper system to approximate a rubber surface. Ballrubber contact solver parameters (softness, slip, friction) were determined empirically, while joint stiffness, damping, and armature were established through parameter sweeps optimizing for sim-to-real transfer. Analogously, ball-table contact solver parameters were also measured.

We observed a bimodal distribution in contact solver parameters for the paddle rubber restitution when we completed system ID for topspin and underspin ball contacts3. Underspin balls exhibit a damping coefficient of −103, while topspin balls have a damping coefficient of approximately −0. Consequently, during the topspin correction phase of

3https://mujoco.readthedocs.io/en/latest/modeling.html#csolver

[Figure 7]

[Figure 8]

Fig. 5: Sample training in simulation and zero-shot transfer to the hardware are shown side by side.

policy training (described in the following section), the simulator dynamically selects the appropriate solver parameters based on the ball’s pre-contact spin. This bimodality was not observed in the ball-table contact solver parameters.

In addition to modeling and domain randomizing observation noise and latency as described in [9], and [8], we randomized table and paddle damping, and friction parameters during training. We employed two shaping rewards, net height reward and a target for the last ABB joint at ball-paddle contact, to mitigate a sim-to-real gap observed due to robot returns overshooting the opponent’s side. This approach not only addressed the intended criteria but also promoted competitive robot returns.

2) Spin “correction” and sim-to-sim adapter layers: As discussed above, the paddle rubber physical parameters in simulation are bimodal, depending on whether there is a topspin or underspin ball. Therefore, directly deploying LLCs revealed a large sim-to-real gap for topspin balls. We developed two solutions to mitigate this issue: topspin correction and sim-to-real adapter layers.

For topspin correction, we fine-tuned an LLC in simulation, switching to topspin-related paddle parameters when the incoming ball had topspin. We also incorporated a net height reward, requiring the returned ball to cross the net at a certain height, and a target joint angle for ball contact. This technique successfully closed the sim-to-real gap in many specialized skills, and also increased the speed of robot returns, adding a competitive edge. However, the sim-toreal gap observed in generalized skilled policies for higher topspin remained despite the topspin correction.

To address the remaining gap, we augmented the topspincorrected policy with a thin FiLM layer [27] and trained the adapter using just the topspin balls. We learned a mapping FiLM(A|γ,β = f(ot)) = γ ∗ A + β, where A ∈ R8 is the

Number of balls

Dataset type Dataset All Forehand Center Backhand Fast Normal Slow Topspin Nospin Underspin Lob Rallying Initial 2,585 1036 785 764 478 1,982 125 972 1250 363 150

- Rallying Cycle 1 1,312 479 448 385 360 891 66 592 519 201 58
- Rallying Cycle 2 1,409 587 192 630 18 1357 35 1019 379 15 12
- Rallying Cycle 3 593 153 154 286 36 556 1 192 266 135 0
- Rallying Cycle 4 756 417 58 280 2 697 57 303 379 75 5
- Rallying Cycle 5 596 212 127 257 23 562 11 396 192 8 9
- Rallying Cycle 6 5,792 1,831 2,192 1,759 149 4,933 710 1,540 3,872 380 551
- Rallying Cycle 7 1,198 374 381 442 108 997 93 260 779 159 113 Rallying Final 14,241 5,089 4,337 4,803 1,174 11,975 1,098 5,274 7,633 1,336 898 Rallying Final+reflection 28,482 9,892 8,674 9,892 2,348 23,950 2,196 10,548 15,266 2,672 1,796

Serves Initial 858 350 175 318 - - - 5 796 57 -

- Serves Cycle 1 1,999 344 1,050 570 - - - 4 1,940 55 -
- Serves Cycle 2 512 161 253 90 - - - 1 453 58 Serves Final 3,369 855 1,478 978 - - - 10 3,189 170 -

TABLE I: The task distribution dataset was developed over multiple training cycles. Notably, cycle 6 substantially improved the coverage of slow and lob balls, whilst the majority of fast balls came from the initial data collection and 1st cycle.

|initial_human_play initial_ball_thrower<br><br>|
|---|

|cycle_1|
|---|

|cycle_6|
|---|

|serves|
|---|

|rallying<br><br>serves|
|---|

Fig. 6: Visualization of the task distribution dataset. TSNE [26] was used to project from 9-dimensional balls states to a 2-dimensional representation.

original action. FiLM learns a function f(ot) that outputs γ and β ∈ R8. The FiLM layer consists of 2.8k parameters and we train the adapter for 5k steps. This closed the simto-real gap while preserving underspin return ability. Similar techniques could be applied to heavy underpsin, and side spins, but we leave this for future work.

3) Iteratively grounding the training task in the real world: A seed dataset of 40 minutes of human vs. human play was collected along with 480 varied ball throws from a ball thrower. The perception system described in [9] was used to extract ball positions at 125Hz. The sequence of ball positions was segmented into trajectories consisting of single ball hits where the first ball position of a trajectory is when the ball enters play or immediately after a hit. We then employed an offline optimization process to extract the initial ball state —– position, velocity, and angular velocity —– from each trajectory, similar to the approach in [8], such that a simulated ball trajectory starting at that state matches the real ball trajectory as closely as possible. The output of this process is a dataset of initial ball states.

The initial data collection resulted in 2.6k initial ball states. Serves were excluded initially to simplify training. An independent initial serving dataset of 0.9k balls was gathered separately. We extracted ball states from the serving trajectories using optimization methods described in [28].

Policies were trained in simulation with the objective of

returning all balls in the dataset. During simulated training, we sampled a ball state from the dataset, added small random perturbations, and validated the resulting trajectory. We then initialized the MuJoCo internal state with the ball state and started an episode.

Our non-parametric approach to generating initial ball states — directly sampling them from the dataset — is substantially more effective than prior work [8], which used a uniform initial ball state distribution whose bounds were derived from real ball trajectories. The direct sampling approach improves sim-to-real transfer because it aligns the training distribution more closely with balls that are played by human players. [8] sampled values for each dimension of the ball state independently, whereas in reality position and velocity components are related. For example, a ball with high linear y velocity is unlikely to have high positive linear z velocity, nor is it likely to have high underspin. Thus independent sampling can create balls that are unrealistic in the real world or are not typically played by amateur humans. Our approach resolves this issue by preserving any empirical inter-relationships between different dimensions of the ball state. Additionally, since no training cycles were expended on unrealistic balls, model capacity was used more effectively, leading to faster training and higher return rates for the same model architecture and training algorithm.

The system was then deployed to the real-world and eval-

uated against human opponents. Following the same process outlined above, all evaluations were converted into another dataset of initial ball states, and automatically annotated by the system state machine with return — the ball was successfully returned, hit – the paddle made contact with the ball but it did not land on the opponent’s side, or miss ball — the paddle did not touch the ball. This dataset was then added to the initial dataset. Balls that were not returned (with annotations hit or miss) can optionally be overweighted in subsequent training cycles.

This iterative cycle of training models in simulation on the latest dataset, evaluating it in the real world, and using the annotated evaluation data to extend the dataset, can be repeated as many times as needed. We completed 7 cycles for rally balls and 2 cycles for serving balls over the course of 3 months with over 50 different human opponents, leading to a final dataset size of 14.2k initial ball states for rallies and 3.4k for serves. A summary of the dataset evolution is presented in Table I and Figure 6.

One advantage to this iterative approach to building the training task distribution is that if the policy is repeatedly evaluated against diverse opponents, gaps in capabilities are automatically identified and filled. As the agent’s skills improved new weaknesses were revealed whilst simultaneously generating training data to address it. We observed that after 7 cycles performance had not plateaued and we think further cycles could have continued to yield performance improvements.

Two further modifications to the training data distribution were important for boosting performance. (1) Reflecting the data along the y axis. This helped to correct a bias towards forehand play and doubled the final dataset size to 28k ball states. (2) Manually segmenting the dataset into 7 nonmutually exclusive categories — Fast, Normal speed, Slow, Topspin, No spin, Underspin, Lob. During training, balls were selected each episode by first sampling a category with a probability inversely proportional to the return rate of all balls within that category and then an initial ball state was sampled uniformly from within that category. This approach allowed us to focus on weak categories while still maintaining performance on “easier” balls within those categories and across all categories.

- 4) Deployment to hardware: Policies were trained in

simulation to return individual incoming balls, substantially simplifying training. To adapt these policies to play a full game of table tennis, we divide each point into sub-episodes [29] that mimic the training against individual balls: they start when the opponent’s paddle contacts the ball and end when the robot returns the ball or a point is scored by either player (i.e. the ball leaves play). After a sub-episode the robot and internal data structures of the real environment are reset, ensuring that the policy experienced the single episode semantics it saw in simulation. This step proved crucial for achieving high sim-to-real zero-shot transfer. To estimate paddle state, we employed a customized paddle equipped with motion capture capabilities.

III. EXPERIMENTS AND RESULTS A. User study design

To evaluate the skill level of our agent, we ran competitive matches against 29 table tennis players of varying skill levels – beginner, intermediate, advanced, and advanced+. To ensure that the robot played against a range of human table tennis players, we first ran a pre-study to assess their skill levels.

1) Pre-study: We recruited human players from a local population including a table tennis club, excluding players who were members of our research team or were under 18 years of age. In the pre-study, 59 volunteers played against a professional table tennis coach (who is also an author on this paper, henceforth referred to as ”the coach”), who rated each of them as a beginner, intermediate, advanced, and advanced+ player. Independently each person was asked to fill out a questionnaire containing a number of questions about their table tennis experience. The professional table tennis coach did not have access to the responses before rating the players, although the coach may have known some of the players from the local club.

Figure 7 presents the results of the questionnaire and shows clear differences between each of the four groups, validating the coach’s rating. Beginner and intermediate players have had little to no coaching and have played in almost no tournaments, however beginner players typically have less than a year’s experience and play less than monthly whereas intermediate players tend to have been playing for more than a year and play weekly or multiple times per week. Advanced and advanced+ players have all been playing for more than three years and have had coaching. Compared to advanced players, advanced+ players have been playing for longer and have competed in more tournaments.

Based on their skill levels, we invited a subset of players at different skill levels to return to participate in the full user study.

2) Full user study: A total of 29 user study participants came to our lab to play table tennis against the robot. Figure 8 gives the skill distribution.

Each participant was given a safety briefing and a practice session with the special paddle4 used for playing against this robot5. During the practice session, the participant played against the coach so they could get a feel for the paddle.

After the practice session with the coach, the participant walked to the table with the robot. The rules of the game were explained to the participant (details below). Members of the user study team stood by to run the robot and ensure the safety of the study participant.

The study participant then played three games against the robot and the coach served as the referee for these games. After completion of the three competitive games, we offered them an optional free play session with the robot (up to

- 4The paddle has motion capture LEDs for sensing paddle state and short pips rubber
- 5Due to human error, 1 beginner player was not given the option of a practice session.

How long have you played table tennis?

| |Beginner<br><br>Intermediate<br><br>Advanced<br><br>Advanced+| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

- 0
- 1
- 2
- 3
- 4
- 5
- 6

< 1 year 1-3 years 3-5 years > 5 years

How long have you had table tennis coaching?

Beginner

Intermediate

8

Advanced

Advanced+

6

4

2

0

Never < 6 months6-12 months 1-3 years > 5 years

How often do you play table tennis?

| |Beginner<br><br>Intermediate<br><br>Advanced<br><br>Advanced+| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

- 0
- 1
- 2
- 3
- 4
- 5
- 6

< Monthly Monthly Weekly > Weekly

Fig. 7: Participants’ experience playing table tennis by group.

Mean number of tournaments played

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

14

12

10

8

6

4

2

0

Beginner Intermediate Advanced Advanced+

- 5 minutes). Finally, we conducted a brief, semi-structured interview with the participant about their overall experience of playing table tennis against the robot.

3) User study metrics: The primary measure of agent performance was how well the robot scored in the matches against the human players. We also video recorded the games played against each of the human players and made them available on our website.

Beyond agent performance, we also wanted to understand what it was like for the human players to play against the robot. We administered questionnaires after each game, asking participants about what it was like to play table tennis against the robot. We also asked whether they had used any particular strategies to play against the robot in the game. At the end of the third game, the final questionnaire also asked them about their level of interest in playing with the robot again.

To get a behavioral assessment of how interested they were in playing again with the robot, we recorded how long they chose to play against the robot during the free play time.

B. Match rules

The table tennis matches between the human players and the robot largely followed the rules set forth by the International Table Tennis Federation (ITTF) [30]. In summary, the robot and human play three games, in which the first player to reach eleven points by a margin of two points (or a score of twenty points) wins the game. The player that wins the majority of games wins the match. Unlike a real “best-of-three” match, all three games were played to ensure

12 Number of participants

| |7<br><br>11<br><br>5<br><br>6| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

10

8

6

4

2

0

B I A A+

- Fig. 8: Number of participants by skill group. Human opponent skill level: B = Beginner, I = Intermediate, A = Advanced, A+ = Advanced+

consistent data among participants. The coach acted as a referee to determine scoring and rules violations. Human players were given a minimum two-minute break between matches to rest and fill out a short survey.

In table tennis, one player serves the ball to another by tossing the ball up and striking it so that it bounces on their side of the table and then the opponent’s. While the specific requirements of the serve limit the types of shots possible, the serving player nevertheless has a strong advantage [31] as they can control the initial conditions of the ball e.g. to exploit weaknesses in their opponent or to set themselves up for a strong shot. For this reason, standard table tennis rules rotate the serving player every two points. However, the robot in this study (as designed) cannot serve and therefore the human player always serves the ball to the robot. To compensate for this limitation the human cannot score points or lose points on the serve; the robot must return the ball and then points may be scored. This rule did result in some more skilled players repeatedly attempting risky serves near the edges of the robot’s capabilities to maximize their score, however we felt this was a necessary compromise to accommodate players of lower skill who may not be used to official serving rules. Overall we felt this method of serving struck the best balance between the human’s serving advantage and robot’s capabilities. Additionally, these rules ensure that matches progress beyond services and into the rallying component of the game, which involves a much broader set of skills than the service component. A smaller five player study was conducted with alternate serving rules and is discussed in Section III-G.

Two other limitations of the robot were accounted for. If the robot entered a protective stop state, then the point was considered a “let” (no one scores). Similarly if the ball was hit very high (roughly 2 meters above the table) the point would also be a let due to the limited field of view of the cameras. Applications of all rules were up to the referee’s discretion. Other limitations of the robot such as it’s inability to reach all the way to the net were deemed too complicated to accommodate for and thus no additional rules were applied.

C. Match results

Figure 9 breaks down various statistics of the matches between the humans and the robot. Overall the robot was solidly in the middle of the participants, winning 45% of

80 Robot win rate (%)

80 Robot games won (%)

80 Robot points won (%)

- 0.5
- 1.0

1.5

2.0

- 2.5
- 3.0 Num games to decide match

60 Active points per match

| |45 46<br><br>49| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| |66<br><br>38<br><br>34| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| |55<br><br>47<br><br>45| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| |2.0<br><br>2.7<br><br>2.2 2.2| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| |29.3<br><br>49.9<br><br>35.8 36.3| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

70

70

70

50

60

60

60

40

50

50

50

40

40

40

30

30

30

30

20

20

20

20

10

10

10

10

0

0

0

0.0

0

Matches Games Points

Game 1 Game 2 Game 3

Game 1 Game 2 Game 3

B I A A+

B I A A+

Matches won (%)

Games won (%)

Points won (%)

| |100<br><br>55<br><br>0 0| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| |95<br><br>100 100<br><br>86<br><br>55<br><br>100<br><br>27<br><br>36<br><br>7<br><br>0<br><br>20<br><br>0<br><br>6<br><br>17<br><br>0 0<br><br>B<br><br>I<br><br>A<br><br>A+| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| |72<br><br>79<br><br>72<br><br>67<br><br>50<br><br>62<br><br>43 45 34 32<br><br>40<br><br>30<br><br>34 36 35<br><br>31<br><br>B<br><br>I<br><br>A<br><br>A+| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

100

100

100

80

80

80

60

60

60

40

40

40

20

20

20

0

0

0

B I A A+

Overall Game 1 Game 2 Game 3

Overall Game 1 Game 2 Game 3

- Fig. 9: Match statistics. The robot won 45% of all matches. 100% against beginners, 55% against intermediate players and no matches against advanced players. This suggests the robot performs at an amateur intermediate level. Human opponent skill level: B = Beginner, I = Intermediate, A = Advanced, A+ = Advanced+

matches, 46% of games, and 49% of points. When we break down matches by skill level, a clear pattern emerges. The robot won 100% of matches against beginner opponents, 55% of matches against intermediate opponents, and no matches against more skilled opponents. The implication is that the robot’s skill level is intermediate; it can easily beat the previous skill level, is unable to win against higher skill levels and has roughly even odds to win against this skill level. That is not to say that the robot completely dominates or is dominated by other skills levels. Looking at the breakdown of points scored, the robot won 72% of points against beginners, 50% of points against intermediate players, and 34% of points against advanced and advanced+ players. Thus, the robot can still provide an interesting game to all levels of skills.

When playing against beginner and intermediate skill levels, there was an intriguing trend — the robot always won the first game (see Fig. 9, Games Won (%)). We hypothesize that during the first game, the human is getting used to the novel situation they find themselves in: playing a sport against a robot, using an unfamiliar paddle, pressure from the

< -60 -40 -20 0 20 40 60 80 >100 Estimated Initial Ball Angular Velocity (rad/s)

0.00

0.25

0.50

0.75

1.00

ReturnRate

UNDERSPIN TOPSPIN

- Fig. 10: Robot’s return rate vs different estimated spin levels during rally shows that the robot is better at returning incoming balls with topspin compared to balls with underspin. Data is aggregated across all matches.

competitive setting, and so on. In post-game surveys players commented that they were still getting used to the robot, with several players also suggesting that the robot was intimidating and loud, which could be a lesson for balancing highspeed performance and human comfort in HRI scenarios. Focusing on the intermediate players, by the second game, they appeared to have become more comfortable with the situation and the more skilled players had identified weak points in the robot’s policy they can exploit. Additionally, because the player may not have been playing optimally in the first game, the robot’s adaptation may not have keyed in on the correct features leading to lower game 2 performance by the robot. However, by the third game the robot was able to learn from the opponent’s play style and improved its win rate (also see Fig. 13).

D. Participant strategies and experience playing with robot

In addition to the quantitative match results, we also wanted to understand the qualitative side of this study; what was it actually like to play against a robot? Table tennis already has many so-called “robots” to aid in training, but these are essentially ball launchers whereas our system has the potential to be more dynamic, is better able to mimic the playstyles of a human, and carry on a full game.

After each game, players were asked to describe any strategies they employed against the robot. Before the study, we identified several weaknesses in the robot’s capabilities, most notably (1) difficulty dealing with large amounts of underspin, (2) very fast balls, (3) very low balls due to a hardcoded constraint that prevented the paddle from getting too close to the table, and (4) that the robot was physically unable to reach balls that landed very close to the net. Analyzing the post-game surveys we see that most players did not employ a specific strategy in game 1 or were mostly focused on probing the robot’s capabilities. During the second and third games, skilled players were able to identify gaps in the

Beginner Sentiment

Advanced Sentiment

Advanced+ Sentiment

Intermediate Sentiment

- 1
- 2
- 3
- 4
- 5

- 1
- 2
- 3
- 4
- 5

- 1
- 2
- 3
- 4
- 5

- 1
- 2
- 3
- 4
- 5

| |Annoying<br><br>Challenging<br><br>Easy<br><br>Engaging<br><br>Frustrating<br><br>Fun| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| || |
|---|
<br><br>Annoying<br><br>Challenging<br><br>Easy<br><br>Engaging<br><br>Frustrating<br><br>Fun| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| || |
|---|
<br><br>Annoying<br><br>Challenging<br><br>Easy<br><br>Engaging<br><br>Frustrating<br><br>Fun| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Annoying<br><br>Challenging<br><br>Easy<br><br>Engaging<br><br>Frustrating<br><br>Fun| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Sentiment

Sentiment

Sentiment

Sentiment

1 2 3

1 2 3

1 2 3

1 2 3

Game

Game

Game

Game

All Skill Level Sentiment

Games Where Robot Lost

Games Where Robot Won

Would you be interested in playing with this robot again?

- 1
- 2
- 3
- 4
- 5

- 1
- 2
- 3
- 4
- 5

- 1
- 2
- 3
- 4
- 5

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Annoying<br><br>Challenging<br><br>Easy<br><br>Engaging<br><br>Frustrating<br><br>Fun| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Annoying<br><br>Challenging<br><br>Easy<br><br>Engaging<br><br>Frustrating<br><br>Fun| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Annoying<br><br>Challenging<br><br>Easy<br><br>Engaging<br><br>Frustrating<br><br>Fun| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | |
|---|---|---|
| | | |
| | | |

20

Sentiment

Sentiment

Sentiment

#Responses

15

10

5

0

1 2 3

1 2 3

1 2 3

1 2 3 4 5

Game

Game

Game

(1=Definitely not, 5=Definitely yes)

- Fig. 11: Players were asked “To what degree do these words describe your experience with playing table tennis with this robot?”. These graphs show the mean player sentiment for each word across games out of a five point Likert scale with where 1 is “Strongly Disagree” and 5 is “Strongly Agree”. The final graph is histogram of responses to the question “Would you be interested in playing with this robot again?”

robot’s capabilities, which correlated with higher win rates: players that mentioned “downspin”, “backspin”,“chops”, or “underspin” (synonyms for weakness 1) in their game 2 and 3 comments were significantly more likely to have won their match (p < 0.05) and also to be of a higher skill level (p < 0.001). Only a few players mentioned ball speed (i.e. “fast”) or the robot’s reach (“net”). Beginner players were less able to identify holes in the policies, typically describing simple strategies like alternating forehand and backhand or hitting directly at the robot. Some intermediate level players were able to identify the gaps but were unable to exploit them.

To validate the players’ observation that the robot is weaker at underspin balls we measured the return rate of the robot against the players by estimated spin amounts. Figure 10, shows that the robot is more robust to balls with topspin compared to underspin. The robot’s return rate is reduced to 50% with some underspin, and reaches zero quickly against higher underspin values. On the other hand, the robot can handle topspin better, still returning 60% of the balls with topspin with estimated 80 (rad/s).

We also wanted to ensure that playing with the robot was actually something people would want to do. It is easy to imagine scenarios where a robot smashes every ball or repeats the same actions to not be very engaging, and so many of our design decisions (like sampling and learning from mistakes) were chosen not only to improve performance but also to create an enjoyable experience for the participants. Based on player feedback, we think this goal was achieved. Figure 11 shows that across all skill groups and win rates players agreed that playing with the robot was “fun” and “engaging” based on a five point Likert

scale. Novelty may play some role in this assessment, but the score tends to increase slightly over games and when players were offered additional time to freely play with the robot, 26/29 of them accepted and played for a mean of 4:06 and median of 5:00 out of a maximum of five minutes, implying that there is some lasting appeal to playing with the robot. Additionally, when asked “Would you be interested in playing with this robot again?”, on a scale of one to five, the average response was 4.87 and the median response was 5. Similarly players of all levels disagreed that playing with the robot was “annoying” or “frustrating” (frustrating also scored lower over time as players presumably adapted to the robot). Players gave middling scores to the robot being “easy” and “challenging”. When broken down by skill level, lower skilled players found the robot more challenging, but other sentiment metrics were largely consistent among groups. Post-study interviews supported these findings, with players calling the robot “dynamic”, “fun”, and “exciting.”

E. LLC Skills

LLCs provide diverse set of skills such as consistency, faster returns, targeting to specific part of the table or returning different types of balls. The LLCs used and their evaluations on hardware is represented in Table II. Based on the hardware evaluations, the land rates of LLCs vary between 0.23 and 0.75 per hit, and average hit velocities vary between 5.1 and 6.8 m/s. We can also see the lateral and horizontal diversity of the returned balls by looking at the x and y coordinates of the landing point for the returned balls. The difference between simulations and study comes from both sim-to-real gap as well as the difference in incoming balls. In simulations, we use a diverse set of incoming balls, but during the study, the players are competitive, sending

LLC Serving Style Policy Type Skill Ball Count Land Rate Hit Vel (y) Landing (x) Landing (y) Study Study Sim Study Sim Study Sim Study Sim

- 0 N FH B+A generalist 27 0.41 0.66 6.38 6.94 -0.29 -0.09 0.35 0.88
- 1 N FH B+A generalist 69 0.58 0.65 5.88 6.79 -0.12 -0.07 0.44 0.77
- 2 N FH B+A generalist 69 0.59 0.58 5.67 6.05 -0.21 -0.13 0.45 0.82
- 3 N FH B hit right 4 0.75 0.63 5.29 6.81 -0.20 -0.39 0.42 0.80
- 4 N FH B hit left 118 0.62 0.66 6.31 6.68 0.12 0.26 0.50 0.89
- 5 N FH B hit left 69 0.51 0.66 5.85 6.79 0.05 0.20 0.43 0.86
- 6 N FH B hit left 39 0.23 0.63 6.23 6.99 0.04 0.18 0.38 0.84
- 7 N FH B hit right, fast 87 0.33 0.73 6.83 7.69 -0.25 -0.29 0.34 0.80
- 8 N FH B hit left, fast 25 0.44 0.60 6.76 7.14 0.11 0.27 0.72 0.72
- 9 N BH B+A generalist 93 0.34 0.67 5.34 6.66 0.18 0.11 0.41 0.82
- 10 N BH B hit fast 349 0.41 0.70 5.79 7.31 0.10 -0.03 0.42 0.88
- 11 N BH B hit right 146 0.32 0.69 5.59 7.01 -0.10 -0.22 0.37 0.86
- 12 N BH B hit left 91 0.35 0.61 5.26 6.99 0.35 0.36 0.36 0.75

- 13 Y FH B topspin 696 0.75 N/A 5.89 N/A -0.09 N/A 0.53 N/A
- 14 Y BH B topspin 862 0.60 N/A 5.12 N/A 0.14 N/A 0.36 N/A
- 15 Y FH B underspin 134 0.48 N/A 6.72 N/A -0.37 N/A 0.64 N/A
- 16 Y BH B underspin 211 0.66 N/A 6.24 N/A 0.32 N/A 0.50 N/A

- TABLE II: Summary of low level skill policies (LLCs). Policy types: B = base policy, A = FiLM adapter layer. Ball Count refers to the number of real world ball throws per policy used to calculate average the land rate, hit velocity in y direction and ball land position. 280k simulated ball throws per LLC were used to calculate the sim metrics.

Generalist

FH Left

FH Right

0.6 0.4 0.2 0.0 0.2 0.4 0.6 X coordinate of the landing point for the returned balls.

- Fig. 12: Distributions of robot’s returned balls when using different LLCs that are trained for targeting to different edges of the table.

more challenging balls and higher percentage of underspin hits to win the game. The increase in underspin balls during the study also explains the lower hit velocity and lower landing points in the study compared to the simulations.

A subset of LLCs are specialized for targeting their returns to different parts of the table. Figure 12, shows a comparison of one of the non-targeting LLCs with targeting ones tested on the hardware against human players. The non-targeting forehand policy returns towards the center of the table, with a slight preference on the right side using diagonal returns. The targeting policies that are obtained by further training with a different reward function that shapes their returns closer to left or right side of the table. Overall, these LLCs diversify the portfolio of the choices for the HLC and afford strategic options such as exploiting a player’s weaker side or forcing the player to move from where they hit the ball.

- F. HLC strategy analysis

During each match, the HLC adapts to each opponent by learning numerical preferences (H-values) for the LLCs based on their online performance. We were interested in two questions about the adaptation component of the system; (1)

how much adaptation occurred during the matches? (2) Did the agent develop different strategies depending on the skill level of the opponent?

#### (1) How much adaptation occurred during the

matches? The change in H-values during a match measures the extent of adaptation. The more the H-values change, the greater the adaptation. Figure 13 shows the aggregated percentage change in H values after three games by player skill. For the forehand LLCs (ID 0 - 8) we consistently observe large changes in H-values of +/- 50% or more, and this trend holds across skill levels. In particular, the H-values for LLCs 0 (a generalist), 3 (a right targeting policy), and 8 (a fast hitting left targeting policy) changed the most. However for the backhand LLCs (ID 9 - 12) the change in H-values was much smaller and often just a few percentage points. This indicates the HLC adapted when it played a forehand style but not the backhand. Qualitatively this is consistent with the observation from the coach that the backhand play was not at the level of the forehand during the matches.

#### (2) Did the agent develop different strategies depending

on the skill level of the opponent? The H-values indicate a relative preference for a particular LLC. Since all matches began with the same initial H-values, the final H-values can be compared across skill groups to assess if the strategy differed. Table III presents these values and there was clearly some variation in strategy. For example, the H-values for LLCs 1, 2, 5, 8 and 9 all differ by a factor of ≈ 2 comparing the smallest and largest values across the player skill groups. Looking at the top LLCs per skill group (bolded), we can see that whilst there are a number of commonly favored LLCs (4, 7, 10 and 11), there are also some differences in strategy, most noticeably for beginners compared with the rest. LLCs 0, 1 and 9 were preferred for beginners whilst LLC 2 was favored for intermediate and advanced players. We can also see that the beginner skill group has the most LLCs with relatively high scores. This indicates that many LLCs are

###### I

###### A

A+

###### B

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

LLC:Forehand[0-8]&Backhand:[9-12]

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

150 100 50 0 50 100

0 50 100 150 200

200 100 0

50 0 50

%

%

%

%

- Fig. 13: The percentage change in preferences (H-values) after three games for each LLC per skill levels: B = Beginner, I

= Intermediate, A = Advanced, and A+ = Advanced+ skill levels.

###### LLC 0 1 2 3 4 5 6 7 8 9 10 11 12

Beginner 4.82 4.82 3.61 0.0 13.25 4.82 2.41 7.23 2.41 10.84 27.71 10.84 7.23 Intermediate 3.07 6.36 7.02 0.0 11.62 7.46 2.63 7.46 1.1 6.58 28.07 9.87 8.77 Advanced 4.28 3.74 6.95 1.07 12.3 8.02 2.14 8.02 1.6 5.88 26.74 12.83 6.42 Advanced+ 1.98 5.14 7.91 0.4 10.28 3.95 3.16 6.72 2.77 9.09 30.43 11.07 7.11

- TABLE III: LLC preferences for each skill level at the end of three games. Bold = top 3 forehand and top 2 backhand LLCs by skill level, including ties.

effective choices to use for beginners which is intuitive since this was the least skilled group.

- G. Serves

In order to assess the service component of the game, we conducted a smaller 5-person study with a different set of rules. In these games the serve alternated every two points between the human participant and the robot. During the participant’s serve, the participant could win or lose points as per normal ITTF rules. The only constraint was that serves should follow the Paralympic service rules [30]. During the robot’s “serve”, the participant physically served the ball but, as per the main study, could not win or lose points on the service. Only once the robot returned the ball and landed on the opponent side was the ball considered “in play”.

The match results are summarized in Table IV. The robot won only 20% matches (1/5), 33% (5/15) games, and 43% (117/271) points, indicating that the ability of the agent to return serves is at a lower level than its rallying skills, around that of a beginner.

We hypothesize this is due to two factors. First, we had strict collision avoidance protocols built into the agent to

Score (Human-Robot) Skill Level Winner RPLS⋆ (%) Game 1 Game 2 Game 3

Beginner Robot 43 7-11 10-12 4-11 Intermediate Human 23 11-4 11-5 9-11 Intermediate Human 42 11-4 11-8 11-6

Advanced Human 36 14-16 11-2 11-7 Advanced+ Human 36 11-9 11-5 11-6

- TABLE IV: Match details for games with points allowed on services. ⋆ RPLS = % of points lost by robot that were lost on services.

avoid the paddle colliding into the table. While this was necessary to protect the equipment, it posed a significant challenge in handling serves that stay very close to the table after the bounce, such as serves with high underspin. Thus the agent missed many underspin serves.

Additionally, we were not able to train a single LLC that could handle a broad range of spin, and in a serve, there is a higher chance of getting high spin compared to rallying. Thus, for serves we had to rely on a spin classifier along with specialist underspin and topspin LLCs to deal with the high spin serves.

Our spin classifier had high precision of 1.0 but low recall of 0.4 for underspin serves based on testing prior to the user study, meaning that numerous underspin serves were mis-identified as topspin. We also observed this to be the case during the user study. One contributing factor to this low accuracy was the motion capture system that tracks the paddle pose was unfortunately subject to frequent failures (for example two full matches had to be discarded due to persistent failures in this component of the agent). Additionally, even when functioning, the human player can temporarily obscure the paddle leading to transient interruptions in the availability of paddle state data.

Given the underlying capabilities of the specialized topspin and underspin serving LLCs, we conjecture that enabling the paddle to move closer to the table and improving spin classification of serves in real time would close most of the gap between the serving and rallying performance of the system. In hindsight, more domain randomization of contact parameters and observation noise specific to service data might also help close the gap in rallying and serve based games.

IV. RELATED WORK A. Robot sports

Since the early days of Artificial Intelligence, Chess and other competitive games have played a critical role in the development of new algorithms and technologies. Examples of highly-skilled AI agents can be found for backgammon [32], chess [33], [34], Go [35], Dota 2 [36], poker [37], and a large number of other games. In many of the these domains, agents have already achieved expert or super human-level performance by playing in a simulated or virtual environment against themselves or a human opponent. A more recent development are competitive games played between people and robotic agents in the real, physical world. In particular, robot sports games have gained substantial attention within the research community since they require fast-paced action generation in challenging, non-deterministic and partially-observable environments. A prominent example is the RoboCup competition [38] – a grand robotics challenge with the goal of fielding a fully autonomous robot team that can win a soccer match against the human World Cup champion team by 2050. RoboCup provides a comprehensive research domain that address realworld complexities like sensor fusion, reactive behavior learning, strategic acquisition, planning, multi-agent systems, context recognition, vision, strategic decision-making, motor control, and intelligent robot control in both simulated and hardware environments: e.g. [39], [40], [41], [42], [43]. Table tennis presents a research field of comparable intricacy, prompting us to devise strategies enabling direct competition with human opponents. Recently [25] demonstrated that RLtrained policies could learn a number of core soccer skills such as tackling, kick and chase, getting up from the ground, and blocking opponent shots in 1 vs. 1 robot vs. robot humanoid soccer. Other examples for robot sports include fencing [44], [45], skiing [46], tennis [47], catching [48] and badminton [49]. The majority of works and competitions in robot sports focus on either sub-aspects of the game or on playing in substantially simplified settings, rather than realistic and competitive matches with humans. There is one notable exception [50], which demonstrated impressive champion level performance in drone racing on a fixed track known in advance in a head-to-head race vs. human experts. By contrast, this work focuses on an interactive sport in which the robot must continuously respond to an unseen human’s actions instead of a head-to-head time trial.

In this paper, we focus playing a competitive game of table tennis against human players of different skill-levels and under realistic conditions. Early work on robot table tennis can be traced back to 1983 with Billingsley’s challenge – a competition based on a set of simplified rules that account for technological limitations of the time [4]. Since then, a variety of increasingly skilled table tennis agents have been proposed [51], [52], [53], [54], [55], [56], [57], [9]. A number of works have focused on aspects of action and motion generation, such as returning the ball [58], [5], [59], [60], [61], aiming to a specific target [6], [62], rallying [8], [63], [64], or

even smashing [7], [57], [65]. Conversely, other works have focused on aspects of state estimation inherent to the table tennis game, including, for instance, trajectory prediction [66], racket pose detection [67], spin detection [68], [69] or the identification of human strategies [70], [71], [72]. The architecture and framework proposed in this paper leverages a combination of these skills in order to tackle the full competitive game for the first time.

To date, the Omron Forpheus robot [73], [62] has the closest capabilities to the agent presented in this work, demonstrating sustained rallies on a variety of stroke styles with skilled human players. A key point of difference is that our agent learns the control policies and perception system, whereas the Forpheus agent uses a model-based approach. More specifically, Forpheus leverages rebound and aerodynamics models in order to identify the optimal configuration of the robot so as to return the ball to a target position. The Omron system represents a highly engineered system that cannot easily be customized to new players, environments, or paddles. Its main objective is to enable extended rallying with humans and, in turn, provide feedback to human player regarding their performance. In contrast to that, we aim to develop a trainable, adaptive robot agent that can play an engaging and skilled game of table tennis against a human player. To this end, we also perform a humansubject study to gauge the level of skill of the robot and its ability to engage the human partner in an interested and satisfying game. While there have been many demonstrations of robots playing table tennis against human players in the past, we believe this research is one of the first human-robot interaction studies to be conducted with full competitive matches against such a wide range of player skill levels.

B. Sim-to-real

Learning control policies in simulation using the reinforcement learning (RL) paradigm provides two main advantages. First, RL does not require any demonstrations of desired behavior, which is often time consuming and expensive to gather, and is sometimes infeasible. Second, learning through trial and error in simulation avoids any potential damage to the robot or environment that could occur as a result of random exploration. To be effective, simulated training needs to faithfully model both the real world physical dynamics and the task distribution. In table tennis, the relevant physical dynamics are the robot, paddle, table, and ball and the task is the expected distribution of incoming balls. Differences between the simulator and the real world are known as the sim-to-real gap [74].

Minimizing the sim-to-real gap for physical dynamics is widely recognized to be a challenging problem and an extensive body of research has been dedicated to closing it in different domains. Multiple techniques have been developed, including system identification [75], learning hybrid simulators [76], dynamics randomization [77] and learning from privileged information [78]. The agent presented in this paper utilises many of these techniques; system ID, dynamics randomization, simulated latency, automatic resets

and a policy architecture designed to encourage smooth control. For more details see [9]. Likewise, advances in physics engine development, MuJoCo’s [19] solid state fluid dynamics model for example, also contribute substantially to closing the sim to real gap. In addition, our agent utilizes task-conditioned physics parameters, in which some physical parameters are changed depending on the state of the environment, as well as an additional stage of simulated training to “spin correct” and our use of sim-to-real adapter layers.

The second contribution of this work in the sim-to-real domain relates to modeling the expected human behavior, i.e. the ball distribution that humans play. We improve upon the iterative approach from [8] in three main ways. First we create the initial bootstrap dataset from human vs. human play instead of single hits across the table, leading to an initial starting point at a much higher level of performance. Second, zero-shot policy transfer was also essential for preventing forgetting and preserving the ball distribution knowledge from simulation, since one human playing with the robot at any particular moment in the real world typically generates a much narrower ball distribution compared to the simulated distribution. This overcomes a key weakness of [8] which requires fine tuning in real which not only lead to forgetting but is very time consuming. Third, the human behavior model was also changed. A spin estimate was added to the initial ball state modeling ([8] only uses ball position and velocity). Then, given a set of initial ball states derived from real world data, instead of transforming the states into a 6-D hyper-cube from which simulated ball states were randomly sampled (as in [8]), we use a non-parametric dataset-based ball distribution. This trajectory generation approach led to much better alignment between simulated ball throws and real world human play as well as more efficient training.

C. Playing games in the physical world with humans

Being able to play with and adapt to a range of diverse human players remains an open problem. A line of research has focused on tackling this challenge in cooperative games, with the goal of building AI agents that can assist any human to perform a task. This is a long-standing problem in AI, and is known as ad-hoc team-play [79], or zero-shot coordination [80]. Initial attempts to first collect human data and use it to train agents that can coordinate well with humans showed a significant gap between the performance achieved with a simulated human proxy and real humans [81], in part because of the vast diversity of human styles and skill levels. Subsequent research has focused on using large simulated populations of self-play agents to train a policy that could coordinate well with a broad swath of players [82], [83], [84], [85]. Much of this work has focused on techniques for increasing the diversity of the simulated population [86], [87], [88], [89], [90], [91], [92], [87]. However, these works have focused on simulated games, and often do not even test with real humans. In contrast, our approach conducts extensive evaluation with real humans of diverse skill levels, in a highly dynamic, real-world setting.

D. Hierarchical robot policies

Control architectures and hierarchies play a critical role in robotics. The introduction of behavior-based robotics, for instance, is considered to be a seminal moment in robotics [93], [94]. In these frameworks, a set of low-level policies form the basic behavioral building blocks which are switched or combined in order to synthesize complex robot control patterns. An arbitration module – the arbiter – determines which individual behavior or sub-set of behaviors to activate at any given moment, based on environmental conditions. Traditionally, both the low-level policies, as well as the arbiter are carefully engineered by an human expert to achieve the intended effect. However, foreseeing the potential interplay between behaviors under a variety of conditions is challenging and can lead to undesirable local minima [95]. Accordingly, machine learning approaches have been proposed to automatically learn policies and arbitration modules. In [96] a hierarchical architecture was proposed in which each low-level policy votes for one or multiple possible next robot actions. The set of all votes collected from the policies is then combined by the higher-level arbiter to yield a final control signal which best satisfies the objectives of all policies. This architecture already used neural network controllers for low-level policies. A braininspired hierarchical neural network controller was proposed in [97] which leverages a combination of forward and inverse dynamics models for control. More recent approaches, such as HiREPS in [98], can learn both the arbitration module and the low-level policies jointly. To this end, HiREPS first estimates the probability that an action has been created by an individual policy. Thereafter the parameters of each lowerlevel policy are updated based on these probabilities.

A much earlier version of our system [99] demonstrated the effectiveness of a hierarchical approach to segment strategy, targeting, and execution for simulated robot table tennis, with a major difference being that the lower levels were model-based and were thus difficult to translate from simulation to the real world.

Our current work is closest in spirit to the work in [55] in which a gating network is learned to create mixtures of existing low-level policies. The gating network generates probabilities indicating the likelihood that a policy is the right one given the current context. The approach builds on an imitation learning framework and requires expert human demonstrations for each of the lower-level behaviors. In a similar vein, SayCan [100] uses learned value functions in order to determine if a low-level policy will be successful if executed from a given state. However, training SayCan is performed using both supervised and reinforcement learning. The overall objective of SayCan is to use a large language model for high-level planning which is the grounded via value functions and low-level policies in the real world. Our approach deviates from previous methodologies in that it builds upon instance based learning. Skill descriptors of the low-level policies are represented as KD-trees. Hence, new information can incrementally be added without retraining

the HLC. We leverage this ability to enable rapid sim to real transfer, i.e., we collect real-world data to improve the accuracy of skill descriptors. In addition, the HLC also continuously collects information about the current opponent, without the need for retraining. As a result, our approach enables real-time learning and is highly adaptive to the current environment and opponent.

V. LIMITATIONS

Our research demonstrates progress in training robots to play table tennis against human opponents. However, several limitations warrant further investigation: (1) Fast Balls: The robot struggles to react effectively to very fast balls due to multiple factors; lack of data, high system latency, and between-shot reset. Even with a high control frequency (50hz), the inherent latency in the system (∼100ms) significantly restricts the number of actionable decisions the robot can make within a given timeframe. This problem was exacerbated by the fact that the robot was reset to the same initial pose — close to the table, paddle facing forward — in between each ball hit to account for the single ball training. While increasing the control frequency could mitigate this issue, it reduces the available time per action for computation. (2) High Balls: The robot is unable to handle balls higher than ∼6ft above the table, as it goes outside the camera FOV. In practice, it was seldom a problem. (3) Low Balls: The robot’s inability to consistently handle very low balls is primarily attributed to collision avoidance protocols built into the system. These protocols, while essential for protecting the robot’s paddle, pose a significant challenge in handling balls that stay close to the table. This limitation was frequently exploited by skilled players by sending underspin balls that stay very low. (4) Short Pips Paddle: In this work, we chose a paddle that can impart spin, while still being possible to model within the simulator. The paddle doesn’t have the smooth rubber found on advanced table tennis paddles [17]. We found accurately modeling such advanced paddles extremely challenging given the highly nonlinear and multimodal nature of the paddle. This limitation was frequently noted by human participants and highlights the need for more advanced paddle modeling techniques. (5) Spin Detection: The policy is unable to accurately read extreme spins on an incoming ball. This limited the amount of spin that could be effectively handled, a fact that was exploited by a number of opponents. It was possible to train specialist LLC policies to handle different types of spin, so improvements in spin detection would likely directly translate to improved overall performance. (6) No Multi-ball Strategic Game Play: The robot’s gameplay is predominantly ”one ball at a time”, lacking the multi-ball strategic play observed in skilled human players. This limitation is attributed to the single-ball training approach. (7) Motion Capture Unreliability: The reliance on motion capture technology introduces a degree of unreliability, as occasional glitches or inaccuracies in motion tracking can negatively impact the robot’s performance. (8) Backhand The policy had weaker backhand play compared with the forehand. (9) Predictability The policy did vary

its placement however this could be improved further. Being less predictable in placement of certain balls was one of the top three areas for improvement as identified by the match referee (in addition to reading spin better and improving the backhand).

VI. CONCLUSIONS AND FUTURE WORK

In this work we presented a learned robot agent that achieves amateur human-level performance in competitive table tennis, a sport renowned for its dynamism and demanding skill requirements. Performance was assessed through 29 competitive matches vs. unseen human players with varied table tennis skills and who all reported to enjoy playing with the robot.

The limitations identified in Section V suggest many directions for future research in the area of robot table tennis. To address the latency constraints that hinder the robot’s reaction time to fast balls, we propose investigating advanced control algorithms and hardware optimizations. These could include exploring predictive models to anticipate ball trajectories or implementing faster communication protocols between the robot’s sensors and actuators. Additionally, training the policy to choose the reset pose, or simply removing betweenshot resets altogether, would give the policy more time and flexibility to react. The challenge of low balls could be tackled by developing more sophisticated collision detection and avoidance algorithms. These algorithms could classify different potential collisions with the table and in certain cases allow for the robot’s to move closer to the table whilst ensuring the safety of the paddle. The robot’s strategic capabilities can be improved by training on scenarios that last an entire rally to better capture the game state, potentially even exploring self-play techniques. Additionally, advanced and advanced+ players were able to find and exploit holes in the robot’s capabilities and mentioned this during interviews; we are hopeful that with our iterative learning method we could fill the gaps and adapt to these players with more training rounds, at least within the physical capabilities of the robot.

We also hope this research makes a useful contribution beyond robot table tennis. Four aspects have broader implications.

#### (1) Hierarchical policy architecture A crucial component

of performing well on complex real world tasks is having a good model of an agent’s capabilities. Our LLC skill descriptors are a novel approach to building such a model both in the level of context-specific detail they provide to a higher level policy, but also because they can be continuously updated online based on real world experience. Exploring ways to incorporate such a model into the increasingly popular hierarchical robot control systems is a fruitful area for future investigation.

#### (2) Enabling zero-shot sim-to-real via iterative real

world data collection Our agent used real world data to define the training tasks, whilst leveraging simulation to learn control policies. As a result it learned to solve realworld tasks whilst remaining data efficient, using only 17.5k

examples. There are many challenges to using simulation and RL to train generalist robot controllers that work in the real world; namely the difficulty in scaling simulator design to many tasks, the sim-to-real gap, and the difficulty in scaling RL training compared with state-of-the art supervised learning techniques. This work offers an approach to bridging the sim-to-real gap from a task distribution perspective, and physical dynamics modeling continues to improve thanks to active research in this area. Given this, it is worth considering how to better leverage simulators. Can they be used to train general purpose skill libraries for particular embodiments?

- (3) Real time adaptation: We enable rapid adaptation to

the opponent by tracking in real-time the match statistics representing the robot and opponent’s strengths and weaknesses. This online adaptation helps the controller to adapt to novel opponents and allows the robot to learn and refine its decision-making process as the game evolves, leading to improved robustness. It is applicable to any situation in which the deployment distribution differs from the training distribution and where a policy is choosing to execute one of multiple skills whilst receiving online feedback about its success rate.

- (4) System design: This work demonstrates that a (per-

haps surprisingly) high level of performance in table tennis can be achieved using relatively simple neural network architectures and training algorithms. The policy architectures in our agent are well-known, have existed for years, and have few parameters. This suggests the importance of system design in developing highly performant learned robotic controllers. Every aspect of the system went through multiple rounds of optimization and redesign. This played a central role in the robustness and sim-to-real performance of the controller sustained over hours of gameplay. Looking forward, in order to develop both highly capable and robust robot controllers for complex real world tasks, system design may be as important as the algorithms, policy architectures, and datasets.

This is the first robot agent capable of playing an interactive competitive sport with humans at human level and represents a milestone in robot learning and control. However, it is also only a small step towards a long-standing goal in robotics of achieving human-level performance on many useful real world skills. A lot of work remains in order to consistently achieve human-level performance on single tasks, and then beyond, in building generalist robots that are capable of performing many useful tasks, skillfully and safely interacting with humans in the real world.

ACKNOWLEDGMENT

We are very thankful to Jon Abelian, Justin Boyd, Omar Escareno, Gus Kouretas, Khem Holden, Utsav Malla, Sphurti More and Thinh Huu Nguyen for their tireless efforts helping us to maintain the system. We are so grateful to Jon Abelian, Omar Escareno, Tomas Jackson, Sphurti More, Thinh Huu Nguyen, Diego Reyes and April Zitkovich for all of their help running the user study. Thank you to Tianli Ding and Pierre Sermanet for their earlier work on the system in targeting.

Finally, we are grateful to Ken Caluwaerts, Nicolas Heess, Kanishka Rao, and Martin Riedmiller for their valuable feedback on an early draft of this paper and to the entire Google DeepMind Robotics team for their help and support.

REFERENCES

- [1] Z. Fu, T. Z. Zhao, and C. Finn, “Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation,” in arXiv, 2024.
- [2] J. Wu, R. Antonova, A. Kan, M. Lepert, A. Zeng, S. Song, J. Bohg, S. Rusinkiewicz, and T. Funkhouser, “Tidybot: personalized robot assistance with large language models,” Auton. Robots, vol. 47, no. 8, p. 1087–1102, nov 2023. [Online]. Available: https://doi.org/10.1007/s10514-023-10139-z
- [3] C. Li, M. Vlastelica, S. Blaes, J. Frey, F. Grimminger, and G. Martius, “Learning agile skills via adversarial imitation of rough partial demonstrations,” in Proceedings of The 6th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, K. Liu, D. Kulic, and J. Ichnowski, Eds., vol.

205. PMLR, 14–18 Dec 2023, pp. 342–352. [Online]. Available: https://proceedings.mlr.press/v205/li23b.html

- [4] J. Billingsley, “Robot ping pong,” Practical Computing, 1983.
- [5] Y. Huang, B. Sch¨olkopf, and J. Peters, “Learning optimal striking points for a ping-pong playing robot,” in 2015 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2015, pp. 4587–4592.
- [6] T. Ding, L. Graesser, S. Abeyruwan, D. B. D’Ambrosio, A. Shankar, P. Sermanet, P. R. Sanketi, and C. Lynch, “GoalsEye: Learning High Speed Precision Table Tennis on a Physical Robot,” in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022, pp. 10780–10787.
- [7] L. Chen, R. Paleja, and M. Gombolay, “Learning from suboptimal demonstration via self-supervised reward regression,” in Conference on robot learning. PMLR, 2021, pp. 1262–1277.
- [8] S. W. Abeyruwan, L. Graesser, D. B. D’Ambrosio, A. Singh, A. Shankar, A. Bewley, D. Jain, K. M. Choromanski, and P. R. Sanketi, “i-sim2real: Reinforcement learning of robotic policies in tight human-robot interaction loops,” in Conference on Robot Learning. PMLR, 2023, pp. 212–224.
- [9] D. B. D’Ambrosio, N. Jaitly, V. Sindhwani, K. Oslund, P. Xu, N. Lazic, A. Shankar, T. Ding, J. Abelian, E. Coumans, G. Kouretas, T. Nguyen, J. Boyd, A. Iscen, R. Mahjourian, V. Vanhoucke, A. Bewley, Y. Kuang, M. Ahn, D. Jain, S. Kataoka, O. E. Cortes, P. Sermanet, C. Lynch, P. R. Sanketi, K. Choromanski, W. Gao, J. Kangaspunta, K. Reymann, G. Vesom, S. Q. Moore, A. Singh, S. W. Abeyruwan, and L. Graesser, “Robotic Table Tennis: A Case Study into a High Speed Learning System,” in Proceedings of Robotics: Science and Systems, Daegu, Republic of Korea, July 2023.
- [10] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. The MIT Press, 2018.
- [11] K. Caluwaerts, A. Iscen, J. C. Kew, W. Yu, T. Zhang, D. Freeman, K.-H. Lee, L. Lee, S. Saliceti, V. Zhuang et al., “Barkour: Benchmarking animal-level agility with quadruped robots,” arXiv preprint arXiv:2305.14654, 2023.
- [12] A. Kumar, Z. Fu, D. Pathak, and J. Malik, “Rma: Rapid motor adaptation for legged robots,” 2021.
- [13] X. Cheng, K. Shi, A. Agarwal, and D. Pathak, “Extreme parkour with legged robots,” arXiv preprint arXiv:2309.14341, 2023.
- [14] T. Osa, J. Pajarinen, G. Neumann, J. A. Bagnell, P. Abbeel, J. Peters et al., “An algorithmic perspective on imitation learning,” Foundations and Trends® in Robotics, vol. 7, no. 1-2, pp. 1–179, 2018.
- [15] S. Stepputtis, J. Campbell, M. Phielipp, S. Lee, C. Baral, and H. Ben Amor, “Language-conditioned imitation learning for robot manipulation tasks,” Advances in Neural Information Processing Systems, vol. 33, pp. 13139–13150, 2020.
- [16] O. X.-E. Collaboration, A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, A. Tung, A. Bewley, A. Herzog, A. Irpan, A. Khazatsky, A. Rai, A. Gupta, A. Wang, A. Kolobov, A. Singh, A. Garg, A. Kembhavi, A. Xie, A. Brohan, A. Raffin, A. Sharma, A. Yavary,

- A. Jain, A. Balakrishna, A. Wahid, B. Burgess-Limerick, B. Kim,
- B. Sch¨olkopf, B. Wulfe, B. Ichter, C. Lu, C. Xu, C. Le, C. Finn,
- C. Wang, C. Xu, C. Chi, C. Huang, C. Chan, C. Agia, C. Pan, C. Fu, C. Devin, D. Xu, D. Morton, D. Driess, D. Chen, D. Pathak,

- D. Shah, D. B¨uchler, D. Jayaraman, D. Kalashnikov, D. Sadigh,
- E. Johns, E. Foster, F. Liu, F. Ceola, F. Xia, F. Zhao, F. V. Frujeri,
- F. Stulp, G. Zhou, G. S. Sukhatme, G. Salhotra, G. Yan, G. Feng,
- G. Schiavi, G. Berseth, G. Kahn, G. Yang, G. Wang, H. Su, H.-S. Fang, H. Shi, H. Bao, H. B. Amor, H. I. Christensen, H. Furuta,
- H. Walke, H. Fang, H. Ha, I. Mordatch, I. Radosavovic, I. Leal, J. Liang, J. Abou-Chakra, J. Kim, J. Drake, J. Peters, J. Schneider,

- J. Hsu, J. Bohg, J. Bingham, J. Wu, J. Gao, J. Hu, J. Wu, J. Wu,

- J. Sun, J. Luo, J. Gu, J. Tan, J. Oh, J. Wu, J. Lu, J. Yang,

- J. Malik, J. Silv´erio, J. Hejna, J. Booher, J. Tompson, J. Yang,

- J. Salvador, J. J. Lim, J. Han, K. Wang, K. Rao, K. Pertsch,
- K. Hausman, K. Go, K. Gopalakrishnan, K. Goldberg, K. Byrne,

- K. Oslund, K. Kawaharazuka, K. Black, K. Lin, K. Zhang, K. Ehsani,

- K. Lekkala, K. Ellis, K. Rana, K. Srinivasan, K. Fang, K. P. Singh,

- K.-H. Zeng, K. Hatch, K. Hsu, L. Itti, L. Y. Chen, L. Pinto, L. FeiFei, L. Tan, L. J. Fan, L. Ott, L. Lee, L. Weihs, M. Chen, M. Lepert, M. Memmel, M. Tomizuka, M. Itkina, M. G. Castro, M. Spero,

- M. Du, M. Ahn, M. C. Yip, M. Zhang, M. Ding, M. Heo, M. K. Srirama, M. Sharma, M. J. Kim, N. Kanazawa, N. Hansen, N. Heess,
- N. J. Joshi, N. Suenderhauf, N. Liu, N. D. Palo, N. M. M. Shafiullah,
- O. Mees, O. Kroemer, O. Bastani, P. R. Sanketi, P. T. Miller, P. Yin,
- P. Wohlhart, P. Xu, P. D. Fagan, P. Mitrano, P. Sermanet, P. Abbeel, P. Sundaresan, Q. Chen, Q. Vuong, R. Rafailov, R. Tian, R. Doshi,

- R. Mart’in-Mart’in, R. Baijal, R. Scalise, R. Hendrix, R. Lin, R. Qian,

- R. Zhang, R. Mendonca, R. Shah, R. Hoque, R. Julian, S. Bustamante, S. Kirmani, S. Levine, S. Lin, S. Moore, S. Bahl, S. Dass,
- S. Sonawani, S. Song, S. Xu, S. Haldar, S. Karamcheti, S. Adebola,

- S. Guist, S. Nasiriany, S. Schaal, S. Welker, S. Tian, S. Ramamoorthy,

- S. Dasari, S. Belkhale, S. Park, S. Nair, S. Mirchandani, T. Osa,
- T. Gupta, T. Harada, T. Matsushima, T. Xiao, T. Kollar, T. Yu,

- T. Ding, T. Davchev, T. Z. Zhao, T. Armstrong, T. Darrell, T. Chung, V. Jain, V. Vanhoucke, W. Zhan, W. Zhou, W. Burgard, X. Chen,

- X. Chen, X. Wang, X. Zhu, X. Geng, X. Liu, X. Liangwei, X. Li,
- Y. Pang, Y. Lu, Y. J. Ma, Y. Kim, Y. Chebotar, Y. Zhou, Y. Zhu, Y. Wu, Y. Xu, Y. Wang, Y. Bisk, Y. Dou, Y. Cho, Y. Lee, Y. Cui, Y. Cao, Y.-H. Wu, Y. Tang, Y. Zhu, Y. Zhang, Y. Jiang, Y. Li, Y. Li,

- Y. Iwasawa, Y. Matsuo, Z. Ma, Z. Xu, Z. J. Cui, Z. Zhang, Z. Fu, and
- Z. Lin, “Open X-Embodiment: Robotic learning datasets and RT-X models,” https://arxiv.org/abs/2310.08864, 2023.

- [17] “Globe 889 short pips-out table tennis rubber without sponge,” 2024, https://shorturl.at/DQTTV [Last Accessed: (08/01/2024)].
- [18] M. L. Puterman, Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.
- [19] E. Todorov, T. Erez, and Y. Tassa, “Mujoco: A physics engine for model-based control,” in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012, pp. 5026–5033.
- [20] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” CoRR, vol. abs/1707.06347, 2017.
- [21] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor,” in Proceedings of the 35th International Conference on Machine Learning. PMLR, 2018, pp. 1861–1870.
- [22] A. v. d. Oord, N. Kalchbrenner, O. Vinyals, L. Espeholt, A. Graves, and K. Kavukcuoglu, “Conditional image generation with pixelcnn decoders,” in Proceedings of the 30th International Conference on Neural Information Processing Systems, ser. NIPS’16. Red Hook, NY, USA: Curran Associates Inc., 2016, p. 4797–4805.
- [23] W. Gao, L. Graesser, K. Choromanski, X. Song, N. Lazic, P. Sanketi, V. Sindhwani, and N. Jaitly, “Robotic table tennis with model-free reinforcement learning,” in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020, pp. 5556–5563.
- [24] J. L. Bentley, “Multidimensional binary search trees used for associative searching,” Commun. ACM, vol. 18, no. 9, p. 509–517, sep 1975.
- [25] T. Haarnoja, B. Moran, G. Lever, S. H. Huang, D. Tirumala, J. Humplik, M. Wulfmeier, S. Tunyasuvunakool, N. Y. Siegel, R. Hafner, M. Bloesch, K. Hartikainen, A. Byravan, L. Hasenclever, Y. Tassa, F. Sadeghi, N. Batchelor, F. Casarini, S. Saliceti, C. Game, N. Sreendra, K. Patel, M. Gwira, A. Huber, N. Hurley, F. Nori, R. Hadsell, and N. Heess, “Learning agile soccer skills for a bipedal robot with deep reinforcement learning,” Science Robotics, vol. 9, no. 89, 2024.
- [26] L. van der Maaten and G. Hinton, “Visualizing data using t-sne,”

- Journal of Machine Learning Research, vol. 9, no. 86, pp. 2579– 2605, 2008.
- [27] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. C. Courville, “Film: Visual reasoning with a general conditioning layer,” in Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018, S. A. McIlraith and K. Q. Weinberger, Eds. AAAI Press, 2018, pp. 3942–3951.
- [28] B. Triggs, P. F. McLauchlan, R. I. Hartley, and A. W. Fitzgibbon, “Bundle adjustment - a modern synthesis,” in Proceedings of the International Workshop on Vision Algorithms: Theory and Practice, ser. ICCV ’99. London, UK, UK: Springer-Verlag, 2000, pp. 298– 372.
- [29] R. S. Sutton, J. Modayil, M. Delp, T. Degris, P. M. Pilarski, A. White, and D. Precup, “Horde: a scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction.” in AAMAS, L. Sonenberg, P. Stone, K. Tumer, and P. Yolum, Eds., 2011.
- [30] “The International Table Tennis Federation Statutes (effective 1st january 2024),” https://documents.ittf.sport/sites/default/files/public/ 2024-02/2024 ITTF Statutes clean version.pdf, ITTF, Accessed: 08/05/2024.

- [31] M. Katsikadelis, T. Pilianidis, and N. Mantzouranis, “The interaction between serves and match winning in table tennis players in the london 2012 olympic games,” in Book of abstracts of the 8th international table tennis federation sports science congress–the 3rd world congress of science and racket sports, 2013, pp. 77–79.
- [32] G. Tesauro, “Temporal difference learning and td-gammon,” Commun. ACM, vol. 38, no. 3, p. 58–68, mar 1995.
- [33] M. Campbell, A. Hoane, and F. hsiung Hsu, “Deep blue,” Artificial Intelligence, vol. 134, no. 1, pp. 57–83, 2002.
- [34] D. Silver, T. Hubert, J. Schrittwieser, I. Antonoglou, M. Lai, A. Guez, M. Lanctot, L. Sifre, D. Kumaran, T. Graepel, T. Lillicrap, K. Simonyan, and D. Hassabis, “A general reinforcement learning algorithm that masters chess, shogi, and go through self-play,” Science, vol. 362, no. 6419, pp. 1140–1144, 2018.
- [35] D. Silver, A. Huang, C. Maddison, A. Guez, L. Sifre, G. Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot,

- S. Dieleman, D. Grewe, J. Nham, N. Kalchbrenner, I. Sutskever,
- T. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, and D. Hassabis, “Mastering the game of go with deep neural networks and tree search,” Nature, vol. 529, pp. 484–489, 01 2016.

- [36] C. Berner, G. Brockman, B. Chan, V. Cheung, P. Debiak, C. Dennison, D. Farhi, Q. Fischer, S. Hashme, C. Hesse,

- R. J´ozefowicz, S. Gray, C. Olsson, J. Pachocki, M. Petrov, H. P. de Oliveira Pinto, J. Raiman, T. Salimans, J. Schlatter, J. Schneider,
- S. Sidor, I. Sutskever, J. Tang, F. Wolski, and S. Zhang, “Dota 2 with large scale deep reinforcement learning,” CoRR, vol. abs/1912.06680,

2019. [Online]. Available: http://arxiv.org/abs/1912.06680

- [37] N. Brown and T. Sandholm, “Superhuman ai for multiplayer poker,” Science, vol. 365, no. 6456, pp. 885–890, 2019.
- [38] H. Kitano, M. Asada, Y. Kuniyoshi, I. Noda, and E. Osawa, “Robocup: The robot world cup initiative,” in Proceedings of the First International Conference on Autonomous Agents, ser. AGENTS ’97. New York, NY, USA: Association for Computing Machinery, 1997, p. 340–347.
- [39] T. R¨ofer, T. Laue, A. Hasselbring, J. Lienhoop, Y. Meinken, and P. Reichenberg, “B-human 2022 – more team play with less communication,” in RoboCup 2022: Robot World Cup XXV, A. Eguchi, N. Lau, M. Paetzel-Pr¨usmann, and T. Wanichanon, Eds. Cham: Springer International Publishing, 2023, pp. 287–299.
- [40] P. Stone, R. S. Sutton, and G. Kuhlmann, “Reinforcement learning for robocup soccer keepaway,” Adaptive Behavior, vol. 13, no. 3, pp. 165–188, 2005.
- [41] S. Behnke, M. Schreiber, J. Stuckler, R. Renner, and H. Strasdat, “See, walk, and kick: Humanoid robots start to play soccer,” in 2006 6th IEEE-RAS International Conference on Humanoid Robots. IEEE, 2006, pp. 497–503.
- [42] V. Suriani, E. Musumeci, D. Nardi, and D. D. Bloisi, “Play everywhere: A temporal logic based game environment independent approach for playing soccer with robots,” in RoboCup 2023: Robot World Cup XXVI, C. Buche, A. Rossi, M. Sim˜oes, and U. Visser, Eds. Cham: Springer Nature Switzerland, 2024, pp. 3–14.

- [43] S. Wang, M. Neau, and C. Buche, “Robonlu: Advancing command understanding with a novel lightweight bert-based approach for service robotics,” in RoboCup 2023: Robot World Cup XXVI, C. Buche, A. Rossi, M. Sim˜oes, and U. Visser, Eds. Cham: Springer Nature Switzerland, 2024, pp. 29–41.
- [44] B. Yang, X. Xie, G. Habibi, and J. R. Smith, “Competitive physical human-robot game play,” in Companion of the 2021 ACM/IEEE International Conference on Human-Robot Interaction, 2021, pp. 242–246.
- [45] B. Yang, G. Habibi, P. Lancaster, B. Boots, and J. Smith, “Motivating physical activity via competitive human-robot interaction,” in Conference on Robot Learning. PMLR, 2022, pp. 839–849.
- [46] T. Petric, L. Peternel, A. Gams, B. Nemec, and L. Zlajpah, “Navigation methods for the skiing robot,” International Journal of Humanoid Robotics, vol. 10, 01 2012.
- [47] Z. Zaidi, D. Martin, N. Belles, V. Zakharov, A. Krishna, K. M. Lee, P. Wagstaff, S. Naik, M. Sklar, S. Choi, Y. Kakehi, R. Patil, D. Mallemadugula, F. Pesce, P. Wilson, W. Hom, M. Diamond, B. Zhao, N. Moorman, R. Paleja, L. Chen, E. Seraj, and M. Gombolay, “Athletic mobile manipulator system for robotic wheelchair tennis,” IEEE Robotics and Automation Letters, vol. 8, no. 4, p. 2245–2252, Apr. 2023. [Online]. Available: http://dx.doi.org/10.1109/LRA.2023.3249401
- [48] S. Abeyruwan, A. Bewley, N. M. Boffi, K. M. Choromanski, D. B. D’Ambrosio, D. Jain, P. R. Sanketi, A. Shankar, V. Sindhwani, S. Singh et al., “Agile catching with whole-body mpc and blackbox policy learning,” in Learning for Dynamics and Control Conference. PMLR, 2023, pp. 851–863.
- [49] S. Mori, K. Tanaka, S. Nishikawa, R. Niiyama, and Y. Kuniyoshi, “High-speed humanoid robot arm for badminton using pneumaticelectric hybrid actuators,” IEEE Robotics and Automation Letters, vol. 4, no. 4, pp. 3601–3608, 2019.
- [50] E. Kaufmann, L. Bauersfeld, A. Loquercio, M. Mueller, V. Koltun, and D. Scaramuzza, “Champion-level drone racing using deep reinforcement learning,” Nature, vol. 620, pp. 982–987, 08 2023.
- [51] R. L. Andersson, A robot ping-pong player. MIT press Cambridge, Massachusetts, 1988, vol. 988.
- [52] H. Hashimoto, F. Ozaki, and K. Osuka, “Development of a pingpong robot system using 7 degrees of freedom direct drive arm,” in IECON’87: Industrial Applications of Robotics & Machine Vision, vol. 856. SPIE, 1987, pp. 608–615.
- [53] J. Knight and D. Lowery, “Pingpong-playing robot controlled by a microcomputer,” Microprocessors and Microsystems - Embedded Hardware Design, 1986.
- [54] G. Schweitzer and J. Wen, “Where neural nets make sense in robotics,” in Prerational Intelligence: Adaptive Behavior and Intelligent Systems Without Symbols and Logic, Volume 1, Volume 2 Prerational Intelligence: Interdisciplinary Perspectives on the Behavior of Natural and Artificial Systems, Volume 3. Springer, 1994, pp. 530–560.
- [55] K. M¨ulling, J. Kober, O. Kroemer, and J. Peters, “Learning to select and generalize striking movements in robot table tennis,” The International Journal of Robotics Research, vol. 32, no. 3, pp. 263– 279, 2013.
- [56] J. Tebbe, Y. Gao, M. Sastre-Rienietz, and A. Zell, “A Table Tennis Robot System Using an Industrial KUKA Robot Arm,” GCPR, 2018.
- [57] D. B¨uchler, S. Guist, R. Calandra, V. Berenz, B. Sch¨olkopf, and J. Peters, “Learning to play table tennis from scratch using muscular robots,” IEEE Transactions on Robotics (T-RO), vol. 38, no. 6, pp. 3850–3860, 2022.
- [58] K. Muelling, J. Kober, and J. Peters, “Learning table tennis with a Mixture of Motor Primitives,” IEEE-RAS Humanoids, 2010.
- [59] O. Ko¸c, G. Maeda, and J. Peters, “Online optimal trajectory generation for robot table tennis,” Robotics and Autonomous Systems, vol. 105, pp. 121–137, 2018.
- [60] Y. Zhu, Y. Zhao, L. Jin, J. Wu, and R. Xiong, “Towards high level skill learning: Learn to return table tennis ball using montecarlo based policy gradient method,” in 2018 IEEE International Conference on Real-time Computing and Robotics (RCAR), 2018, pp. 34–41.
- [61] J. Tebbe, L. Krauch, Y. Gao, and A. Zell, “Sample-efficient reinforcement learning in robotic table tennis,” in 2021 IEEE international conference on robotics and automation (ICRA). IEEE, 2021, pp. 4171–4178.

- [62] C. Liu, Y. Hayakawa, and A. Nakashima, “Racket control for a table tennis robot to return a ball,” SICE Journal of Control, Measurement, and System Integration, vol. 6, pp. 259–266, 07 2013.
- [63] M. Matsushima, T. Hashimoto, and F. Miyazaki, “Learning to the robot table tennis task-ball control & rally with a human,” in SMC’03 Conference Proceedings. 2003 IEEE International Conference on Systems, Man and Cybernetics. Conference Theme - System Security and Assurance (Cat. No.03CH37483), vol. 3, 2003, pp. 2962–2969 vol.3.
- [64] Y. Sun, R. Xiong, Q. Zhu, J. Wu, and J. Chu, “Balance motion generation for a humanoid robot playing table tennis,” in 2011 11th IEEE-RAS International Conference on Humanoid Robots, 2011, pp. 19–25.
- [65] S. Guist, J. Schneider, H. Ma, L. Chen, V. Berenz, J. Martus, H. Ott, F. Gr¨uninger, M. Muehlebach, J. Fiene, B. Sch¨olkopf, and D. B¨uchler, “Safe & accurate at speed with tendons: A robot arm for exploring dynamic motion,” 2024. [Online]. Available: https://arxiv.org/abs/2307.02654
- [66] A. Nakashima, Y. Ogawa, C. Liu, and Y. Hayakawa, “Robotic table tennis based on physical models of aerodynamics and rebounds,” 2011 IEEE International Conference on Robotics and Biomimetics, 2011.
- [67] Y. Gao, J. Tebbe, J. Krismer, and A. Zell, “Markerless racket pose detection and stroke classification based on stereo vision for table tennis robots,” in 2019 Third IEEE International Conference on Robotic Computing (IRC), 2019, pp. 189–196.
- [68] P. Blank, B. H. Groh, and B. M. Eskofier, “Ball speed and spin estimation in table tennis using a racket-mounted inertial sensor.” in ISWC, S. C. Lee, L. Takayama, K. N. Truong, J. Healey, and T. Ploetz, Eds. ACM, 2017, pp. 2–9.
- [69] T. Gossard, J. Krismer, A. Ziegler, J. Tebbe, and A. Zell, “Table tennis ball spin estimation with an event camera,” 2024. [Online]. Available: https://arxiv.org/abs/2404.09870
- [70] K. Muelling, A. Boularias, B. Mohler, B. Sch¨olkopf, and J. Peters, “Learning strategies in table tennis using inverse reinforcement learning,” Biol. Cybern., vol. 108, no. 5, p. 603–619, oct 2014.
- [71] Z. Wang, K. M¨ulling, M. P. Deisenroth, H. B. Amor, D. Vogt, B. Sch¨olkopf, and J. Peters, “Probabilistic movement modeling for intention inference in human–robot interaction,” The International Journal of Robotics Research, vol. 32, no. 7, pp. 841–858, 2013.
- [72] Z. Wang, A. Boularias, K. Muelling, B. Sch¨olkopf, and J. Peters, “Anticipatory action selection for human-robot table tennis,” Artif. Intell., 2017.
- [73] A. Kyohei, N. Masamune, and Y. Satoshi, “The ping pong robot to return a ball precisely trajectory prediction and racket control for spinning balls,” 2019. [Online]. Available: https://api.semanticscholar.org/CorpusID:214698536
- [74] W. Zhao, J. P. Queralta, and T. Westerlund, “Sim-to-real transfer in deep reinforcement learning for robotics: a survey,” in 2020 IEEE Symposium Series on Computational Intelligence (SSCI), 2020, pp. 737–744.
- [75] N. Sontakke, H. Chae, S. Lee, T. Huang, D. W. Hong, and S. Hal, “Residual physics learning and system identification for sim-toreal transfer of policies on buoyancy assisted legged robots,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023, pp. 392–399.
- [76] Y. Jiang, T. Zhang, D. Ho, Y. Bai, C. K. Liu, S. Levine, and J. Tan, “Simgan: Hybrid simulator identification for domain adaptation via adversarial reinforcement learning,” in 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE Press, 2021, p. 2884–2890.
- [77] X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel, “Sim-toreal transfer of robotic control with dynamics randomization,” in 2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018, pp. 3803–3810.
- [78] K.-H. Lee, G. Ros, J. Li, and A. Gaidon, “Spigan: Privileged adversarial learning from simulation,” in International Conference on Learning Representations, 2018.
- [79] P. Stone, G. Kaminka, S. Kraus, and J. Rosenschein, “Ad hoc autonomous agent teams: Collaboration without pre-coordination,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 24, 2010, pp. 1504–1509.
- [80] H. Hu, A. Lerer, A. Peysakhovich, and J. Foerster, ““other-play” for zero-shot coordination,” in International Conference on Machine Learning. PMLR, 2020, pp. 4399–4410.

- [81] M. Carroll, R. Shah, M. K. Ho, T. L. Griffiths, S. A. Seshia, P. Abbeel, and A. D. Dragan, “On the utility of learning about humans for human-ai coordination,” CoRR, vol. abs/1910.05789,

2019. [Online]. Available: http://arxiv.org/abs/1910.05789

- [82] D. Strouse, K. McKee, M. Botvinick, E. Hughes, and R. Everett, “Collaborating with humans without human data,” Advances in Neural Information Processing Systems, vol. 34, pp. 14502–14515,

- 2021.

[83] R. Charakorn, P. Manoonpong, and N. Dilokthanakul, “Generating diverse cooperative agents by learning incompatible policies,” in The Eleventh International Conference on Learning Representations,

- 2022.

- [84] B. Cui, A. Lupu, S. Sokota, H. Hu, D. J. Wu, and J. N. Foerster, “Adversarial diversity in hanabi,” in The Eleventh International Conference on Learning Representations, 2022.
- [85] B. Sarkar, A. Shih, and D. Sadigh, “Diverse conventions for human-ai collaboration,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023.
- [86] C. Yu, J. Gao, W. Liu, B. Xu, H. Tang, J. Yang, Y. Wang, and Y. Wu, “Learning zero-shot cooperation with humans, assuming humans are biased,” in The Eleventh International Conference on Learning Representations, 2022.
- [87] R. Zhao, J. Song, Y. Yuan, H. Hu, Y. Gao, Y. Wu, Z. Sun, and W. Yang, “Maximum entropy population-based training for zero-shot human-ai coordination,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, 2023, pp. 6145–6153.
- [88] Z. Tang, C. Yu, B. Chen, H. Xu, X. Wang, F. Fang, S. S. Du, Y. Wang, and Y. Wu, “Discovering diverse multi-agent strategic behavior via reward randomization,” in International Conference on Learning Representations, 2020.
- [89] J. K. Pugh, L. B. Soros, and K. O. Stanley, “Quality diversity: A new frontier for evolutionary computation,” Frontiers in Robotics and AI, vol. 3, p. 40, 2016.
- [90] B. Tjanaka, M. C. Fontaine, J. Togelius, and S. Nikolaidis, “Approximating gradients for differentiable quality diversity in reinforcement learning,” in Proceedings of the Genetic and Evolutionary Computation Conference, 2022, pp. 1102–1111.
- [91] S. Wu, J. Yao, H. Fu, Y. Tian, C. Qian, Y. Yang, Q. Fu, and Y. Wei, “Quality-similar diversity via population based reinforcement learning,” in The Eleventh International Conference on Learning Representations, 2022.
- [92] A. Lupu, B. Cui, H. Hu, and J. Foerster, “Trajectory diversity for zero-shot coordination,” in International conference on machine learning. PMLR, 2021, pp. 7204–7213.
- [93] R. Brooks, “A robust layered control system for a mobile robot,” IEEE Journal on Robotics and Automation, vol. 2, no. 1, pp. 14–23, 1986.
- [94] R. C. Arkin, Behavior-Based Robotics. MIT Press, 1998.
- [95] C. W. Reynolds, “Steering behaviors for autonomous characters,” in Game Developers Conference, 1999, pp. 763–782.
- [96] J. K. Rosenblatt and C. E. Thorpe, A Behavior-based Architecture for Mobile Navigation. Boston, MA: Springer US, 1997, pp. 19–32.
- [97] M. Kawato, K. Furukawa, and R. Suzuki, “A hierarchical neuralnetwork model for control and learning of voluntary movement,” Biological cybernetics, vol. 57, pp. 169–185, 1987.
- [98] C. Daniel, G. Neumann, O. Kroemer, and J. Peters, “Hierarchical relative entropy policy search,” Journal of Machine Learning Research, vol. 17, no. 93, pp. 1–50, 2016.
- [99] R. Mahjourian, R. Miikkulainen, N. Lazic, S. Levine, and N. Jaitly, “Hierarchical policy design for sample-efficient learning of robot table tennis through self-play,” 2019. [Online]. Available: https://arxiv.org/abs/1811.12927
- [100] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, C. Fu, K. Gopalakrishnan, K. Hausman, A. Herzog, D. Ho,

- J. Hsu, J. Ibarz, B. Ichter, A. Irpan, E. Jang, R. J. Ruano, K. Jeffrey, S. Jesmonth, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, K.-H. Lee, S. Levine, Y. Lu, L. Luu, C. Parada, P. Pastor, J. Quiambao,
- K. Rao, J. Rettinghouse, D. Reyes, P. Sermanet, N. Sievers, C. Tan, A. Toshev, V. Vanhoucke, F. Xia, T. Xiao, P. Xu, S. Xu, M. Yan, and A. Zeng, “Do as i can and not as i say: Grounding language in robotic affordances,” in arXiv preprint arXiv:2204.01691, 2022.

- [101] N. Hansen, “The cma evolution strategy: A tutorial,” 2016. [Online]. Available: http://arxiv.org/abs/1604.00772

APPENDIX

- A. Contribution Statements

David B. D’Ambrosio1,∗: Worked on all parts of the system over the course of many years. Developed the policy architecture and training approach. Conceived, wrote, and edited this paper. Helped run and analyze the user study.

Saminda Abeyruwan1,∗: Worked on all parts of the system over the course of many years. Developed the policy architecture and training approach. Conceived, wrote, and edited this paper. Helped run and analyze the user study.

Laura Graesser1,∗: Worked on all parts of the system over the course of many years. Developed the policy architecture and training approach. Conceived, wrote, and edited this paper. Helped run and analyze the user study.

Atil Iscen1: Worked on targeting policies, sim-to-real, simulation modeling, results analysis. Helped run and analyze the user study.

Heni Ben Amor2: Investigated initial adapter and attenuation models, evaluated LLCs and spin classifiers, data collection, general advisor, paper writing.

Alex Bewley2: Led the design and implementation for the vision system. Built components for data infrastructure and model training. Camera configuration analysis.

Barney J. Reed2,†,: Expert table tennis advisor, coaching engineers, human data collection. Evaluator and referee for user study. Feedback on robot progress and skill level.

Krista Reymann2: Program Manager: organized operational support for the project; initiated and managed the user study and organized operational support during the study.

Leila Takayama2,§: Experiment design, metrics development, wrote methods section, wrote HRI related work.

Yuval Tassa2: Contributed to modeling and system identification of both robot and ball-flight dynamics.

Krzysztof Choromanski: Developed Blackbox optimization library and ES algorithms used for training table tennis policies over the course of many years.

Erwin Coumans: Helped setting up the initial physics simulations. Created initial virtual reality setup that allows to play tennis. Advise during research of simulation setup, URDF files, constraints setup.

Deepali Jain: Developed Blackbox optimization library and ES algorithms used for training table tennis policies over the course of many years.

Navdeep Jaitly: Conceived, designed, and led the initial stages of the project, built and sourced prototypes, foundational designs of systems like control and vision. Created initial vision pipeline and supervised algorithm development.

Natasha Jaques: Initial brainstorming and ideation for policy architecture design and opponent model and paper writing.

Satoshi Kataoka: Developed and maintained the custom robotics module orchestration system. Initial consultation on cameras and other infrastructure-related components.

Yuheng Kuang: Develops and advises on data infrastructure.

Nevena Lazic: Implemented the initial simulator and training algorithms.

Reza Mahjourian: Developed early hierarchical RL policy, defined the ball target skill, and built an initial version of the simulator.

Sherry Moore: Contributed to a software or hardware component that is in use today.

Kenneth Oslund: Wrote and integrated the control middleware layer, which provides a unified interface for the different types of robot hardware and connects the python environment layer to the manufacturer’s C++ driver.

Anish Shankar: Worked on earlier version of the system’s hardware and software infrastructure and system performance.

Vikas Sindhwani: Initiated and developed ES research agenda for table tennis and catching. Supported and advised on an ongoing basis.

Vincent Vanhoucke: Executive support and research direction.

Grace Vesom: Built the camera driver and an early version of the ball detection pipeline. Built camera calibration software and hardened camera hardware stability.

Peng Xu: Robot cell design and build, prototypes for generating data.

Pannag R. Sanketi1: Overall lead on the project (Managed the project + team). Set the vision and the research direction. Coded technical components for the project and wrote parts of the paper.

Corresponding Authors: Saminda Abeyruwan, David B. D’Ambrosio, Laura Graesser {saminda, ddambro, lauragraesser}@google.com

1Primary contributors ∗Corresponding authors (order randomized, equal contributions) 2Core contributors (Alphabetized) †Work done at Google DeepMind via Stickman Skills Center LLC §Work done at Google DeepMind via Hoku Labs.

- B. Simulation Details

- Table V contains the MuJoCo simulator parameters and
- Table VI contains the actuator parameters.

- C. Details on dataset creation

Figure 14 shows the evolution of the rallying task distribution dataset cycle by cycle. The 6th and 7th rally ball cycles were designed to close specific gaps in the agent’s capabilities;

- • Cycle 6: The initial cycles of data collection and evaluation had focused on intermediate level play. As a result, the system had overfit to intermediate players and did not perform well vs. beginners. In this cycle we targeted evaluations with 20+ beginner players and gathered in 5.8k initial ball states of beginner level play.
- • Cycle 7: This was the first cycle which incorporated serves. All previous cycles were focused on rallying. Evaluation were conducted with 20 players of varying levels resulting in 1.2k rallying initial ball states and 2k services. This not only expanded our serving data but we also observed that the typical first return from a human player after the service is meaningfully different from the first ball that a player hits if they start off rallying (i.e. hit the ball across the table without an initial bounce on their side). This gap in the dataset was only revealed once a system that could support serves was deployed.

The definitions of the 7 manual sub-categories of the dataset — Fast, Normal speed, Slow, Topspin, No spin, Underspin, Lob — are given below:

- • Fast: Forward velocity larger than 7m/s.
- • Normal Speed: Forward velocity between 7m/s and 3.5m/s.
- • Fast: Forward velocity less than 3.5m/s.
- • Topspin: Angular velocity (ωx) larger than 50 rad/s.
- • No Spin: Angular velocity (ωx) between 50 rad/s and

-25 rad/s.

- • Backspin: Angular velocity (ωx) less than -25 rad/s.
- • Fast: Forward velocity less than 5.1m/s and vertical velocity larger than 2.5m/s.

- D. LLC training details

Table VII describes the rewards used in simulation to train base LLCs. In addition reward the robot for successfully hitting the ball (1 and 2), additional rewards are provided to encourge the robot to move safely (3-8) and to reach a particular play style pose depending on which style is being trained (9 and 10). During the topspin correction phase, we employ two additional rewards. One of them is the net height reward (NHR), which requires the returned ball to cross the net at a specific height,

e−10∗|z−0.173|, if 0.173 ≤ z < 0.3 −1.1, otherwise

NHR(z) =

and a reward is given when the paddle reaches a target joint angle at the moment of ball contact. For the last

ABB axis, we define a target joint angle. This angle is set to -0.12 for forehands and 2.0 for backhands. The reward at the moment of contact is max(1.0 - min dist to target, 0). The specialists LLCs are fine tuned from the topspin corrected generalist policies to implicitly target a position on the opponent side with a tolerance radius of 0.1m.

We represent our policy using a three layer 1D fully convolutional gated dilated CNN with 10676 parameters. Details are given in Table VIII. The observation space is 2dimensional (timesteps x [ball position, ball velocity, robot joint position, style]) which is an (8 x 16) matrix. The networks outputs a vector (8,) representing joint velocities.

- Table IX presents the ES hyper-parameters used for both

simulated and real world training.

- Table X details the parameters used in the simulated sensor

latency model described above.

- Table XI details the variable used in domain randomization

and the ranges. E. HLC details

Algorithm 2 presents the pseudocode for the online (i.e. during a match) LLC preference update. Table XII compares waiting for 1 and 3 timesteps after the opponent hit to make a decision, and Table XIII compares a decisive HLC (i.e. select an LLC once and stick to it for the remainder of the shot) with an HLC that has the option to re-decide every k steps.

|Metric<br><br>|Wait 1 Step<br><br>|Wait 3 Steps|
|---|---|---|
|Num balls Hit Hit (cleared net) Hit (didn’t clear net) Land Miss|89 81% 66% 15% 39% 19%<br><br>|89 69%<br><br>34%<br>35% 25% 31%<br>|

- TABLE XII: HLC ablation: wait 1 vs wait 3 steps before making a decision

|Metric<br><br>|Decisive choice|Choose every k steps|
|---|---|---|
|Num balls Hit HLC wrong choice Indecisiveness Land Miss<br><br>|393 89% 5% 0% 64% 11%|273 75% 8% 12% 56% 12%<br><br>|

- TABLE XIII: HLC ablation: decisive (commit to first decision) first select LLC every k time steps. For these experiments, k = 1.

1) Spin classifier details: The features extracted at each timestamp for training the classifier are as follows

- • paddlez[t] − paddlez[t − 3]. (1 dimensional)
- • paddlenormal[t] − paddlenormal[t − 3] (3 dimensional)
- • paddlez[t] − ballz[t] (1 dimensional)
- • dist(paddle,ball),i.e.norm(paddlexyz,ballxyz) (1 dimensional)

The feature at each timestamp is 6-d, and we stack features at 3 timestamps preceding the hit, thus making the input

|Category|Property<br><br>|Value|
|---|---|---|
|Physics|Integrator|implicitfast|
| |Timestep (s)<br><br>|0.001|
|Fluid dynamics|density<br><br>|1.225|
| |viscosity<br><br>|1.8e-5|
| |wind|(0, 0, 0)|
| |fluidshape<br><br>|ellipsoid|
| |Blunt drag coefficient<br><br>|0.235|
| |Slender drag coefficient|0.25|
| |Angular drag coefficient<br><br>|0.0|
| |Kutta lift coefficient|1.0|
| |Magnus lift coefficient<br><br>|1.0|
|Contact (Ball-Table)|solref<br><br>|”-1000000 -17”|
| |friction|”0.1 0.1 0.005 0.0001 0.0001”|
| |solimp|”0.98 0.99 0.001 0.5 2”|
| |solreffriction<br><br>|”-0.0 -200.0”|
|Contact (Ball-Paddle)|solref/ topspin<br><br>|”-268072 -103”/ ”-268072 -0”|
| |friction<br><br>|”1.5 1.5 0.005 0.0001 0.0001”|
| |solimp<br><br>|”1.0 0.95 0.001 1.0 6”|
| |solreffriction<br><br>|”-0.0 -488.0”|
|Rubber<br><br>|stiffness|2e3|
| |damping<br><br>|1e0|
| |armature|1e-3|
| |mass<br><br>|1e-6|
| |damping|1e0|
|Actuator|<intvelocity/>| |

- TABLE V: MuJoCo simulator parameters.

|Joint|ctrlrange<br><br>|actrange<br><br>|forcerange|kp|kv<br><br>|frictionloss|damping|armature|
|---|---|---|---|---|---|---|---|---|
|SlideY|-2.0 2.0<br><br>|-1.8861 0.174|-4102 4102|1279178.0<br><br>|164636.0|0.0<br><br>|30.924<br><br>|0.002|
|SlideX|-2.0 2.0<br><br>|-1.8937 1.88<br><br>|-1499 1499<br><br>|248791.0|13157.0<br><br>|0.0<br><br>|255.683|27.097|
|Axis1<br><br>|-4.36 4.36<br><br>|-2.0 2.0|-100 100|85874.0<br><br>|1599|0.1|25<br><br>|0.01|
|Axis2|-4.36 4.36<br><br>|0.2 1.22<br><br>|-100 100|10669.0<br><br>|265<br><br>|0.1<br><br>|25<br><br>|0.01|
|Axis3<br><br>|-4.36 4.36<br><br>|0.0 0.9<br><br>|-60 60<br><br>|180694.0|2735<br><br>|0.1<br><br>|25|0.01|
|Axis4|-5.585 5.585<br><br>|-1.5708 1.5708<br><br>|-20 20<br><br>|21905.0|331<br><br>|0.1<br><br>|5|0.01|
|Axis5|-5.585 5.585<br><br>|-1.75 -0.65<br><br>|-25 25<br><br>|9000.0|211<br><br>|0.1<br><br>|3<br><br>|0.01|
|Axis6<br><br>|-7.33 7.33<br><br>|-0.7 2.3|-100 100<br><br>|5101.0<br><br>|66|0.1<br><br>|3<br><br>|0.01|

- TABLE VI: MuJoCo actuator parameters.

|Reward|Range<br><br>|Weight|Weighted max score|
|---|---|---|---|
|(1) State transition plus bonus for landing the ball<br><br>|[0, 2]|1<br><br>|2|
|(2) Bonus for hitting the ball and landing it on the opponent side of the table|[0, 1]|0.1<br><br>|0.1|
|(3) Episodic jerk reward (proxy for faulting in real)|[0, 1]<br><br>|0.3|0.3|
|(4) Episodic acceleration reward (proxy for faulting in real)|[0, 1]<br><br>|0.3|0.3|
|(5) Episodic velocity reward (proxy for faulting in real)<br><br>|[0, 1]|0.4|0.4|
|(6) Episodic joint angle reward (safety reward, aimed to prevent faulting in real )|[0, 1]|1<br><br>|1|
|(7) Safety reward, penalty for robot colliding with itself or table|[-1 * timesteps, 0]|1<br><br>|0|
|(8) Paddle height reward<br><br>|[-1 * timesteps, 0]|0.5<br><br>|0|
|(9) Style (initial pose) reward (for forehand)<br><br>|max(1-min(∥posei, pose∥), 0)|1<br><br>|1|
|(10) Style (initial pose) reward (for backhand)|max(2-min(∥posei, pose∥), 0)|1|2|
|Total| | |[5.1 - 6.1]|

TABLE VII: Rewards used in simulation to train base LLCs.

size 6 ∗ 3 = 18 for the model. We experimented with different features involving paddle and ball states, different history sizes and different subsampling strategies and found this feature vector to be the best performing one. The ground truth label for the feature is calculated using postprocessing and optimizing for the spin coefficients given the full trajectory [101], [28] of the ball. For data augmentation, we create samples from all the timestamps within a past window of 100ms of the actual hit and label it with the same label as the actual hit. The augmented data size is around 7500 samples.

The policy is a 2-layer MLP of hidden sizes (128, 64). F. User Study: Post match interview questions

In the final interview, we asked each participant: What are your first impressions of this robot? What was it like to play table tennis against this robot? How does playing against this robot compare to playing against other non-humans, e.g., ball throwers, walls? If you could use this robot in the future, what would you use it for? Did your experience playing with the robot differ between the match and free play at the end?

|initial_human_play initial_ball_thrower<br><br>|
|---|

|cycle_1|
|---|

|cycle_2|
|---|

|cycle_3|
|---|

|cycle_4|
|---|

|cycle_5|
|---|

|cycle_6|
|---|

|cycle_7|
|---|

|serves|
|---|

|rallying<br><br>serves|
|---|

|normal<br><br>fast<br><br>slow|
|---|

|topspin<br><br>nospin<br><br>underspin|
|---|

|not lob<br><br>lob|
|---|

- Fig. 14: Rows 1-2: Evolution of the rallying task distribution dataset plus the serving task dataset. There are three broad clusters — the initial dataset together with cycle 1 (r I,7), cycles 2 - 7 (r 2-7), and serves. Rows 3: Dataset clustered by different ball types. TSNE [26] was used to project from 9-dimensional balls states to a 2-dimensional representation

| |Layer|
|---|---|
|Parameter|1 2 3|
|Convolution dimension Number of filters Stride Dilation Activation function Padding<br><br>|1D 1D 1D 76 96 8<br><br>1 1 1<br>1 2 4<br><br><br>tanh tanh tanh valid valid valid|

TABLE VIII: CNN model architecture.

|Parameter Simulation Sim-to-Sim|
|---|
|Step size 0.00375 0.00125 Perturbation standard deviation 0.025 0.025 Number of perturbations 200 5 Number of rollouts per perturbation 15 3 Percentage to keep (top x% rollouts) 30% 60% Maximum environment steps per rollout 200 200 Use orthogonal perturbations True True Use observation normalization True True<br><br>|

TABLE IX: ES hyperparameters.

G. Match Results Details

1) Details on skill assessment:

| |Latencies (ms)|
|---|---|
|Component<br><br>|µ σ2|
|Ball observation ABB observation Festo observation ABB action<br><br>Festo action x-axis<br>Festo action y-axis<br>|40 8.2 29 8.2 33 9<br><br>71 5.7 56.1 12.3 84.0 12.3<br><br>|

- TABLE X: Sensor latency model parameters per component.

|Variable|Range|
|---|---|
|Table damping Paddle damping Paddle friction Table friction<br><br>|U(−1.0, 5.0) U(−5.0, −1.0) U(−0.29, 0.29) U(−0.05, 0.05)|

- TABLE XI: Variables and ranges for domain randomization.

During the matches the referee checked that the preskill assessment accurately reflected a player’s performance. The majority of players remained in the same category, 4 / 29 players changed category. 3 players with a pre-skill assessment of beginner were re-assessed to intermediate and one advanced player was re-assessed to advanced+. During the pre-skill assessment, the referee only had the opportunity to assess participants for a few minutes and it was not in a match setting, and so it was possible to mis-judge their

- 0.5
- 1.0

1.5

2.0

- 2.5
- 3.0 Num games to decide match

Matches won (%)

##### 60 Active points per match

| |2.2<br><br>2.8<br><br>2.2 2.2| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| |90<br><br>50<br><br>0 0| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| |35.0<br><br>50.5<br><br>34.8<br><br>37.6| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

100

50

80

40

60

30

40

20

20

10

0.0

0

0

B I A A+

B I A A+

B I A A+

Sets won (%)

Points won (%)

| |83<br><br>100<br><br>80<br><br>70<br><br>54<br><br>100<br><br>25<br><br>38<br><br>6<br><br>0<br><br>17<br><br>0<br><br>7<br><br>20<br><br>0 0<br><br>B<br><br>I<br><br>A<br><br>A+| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| |65<br><br>74<br><br>62<br><br>59 50<br><br>61<br><br>43<br><br>47<br><br>33<br><br>31<br><br>38<br><br>30<br><br>35<br><br>39<br><br>36<br><br>32<br><br>B<br><br>I<br><br>A<br><br>A+| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

100

100

80

80

60

60

40

40

20

20

0

0

Overall Game 1 Game 2 Game 3

Overall Game 1 Game 2 Game 3

- Fig. 15: Match statistics using the pre-study grouping. Human opponent skill level: B = Beginner, I = Intermediate, A = Advanced, A+ = Advanced+

| |Beginner<br><br>Intermediate<br><br>Advanced<br><br>Advanced+| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

< 1 year 1-3 years 3-5 years > 5 years

- 0
- 1
- 2
- 3
- 4
- 5
- 6 How long have you played table tennis?

Never < 6 months6-12 months 1-3 years > 5 years

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

How long have you had table tennis coaching?

Beginner

Intermediate

Advanced

Advanced+

| |Beginner<br><br>Intermediate<br><br>Advanced<br><br>Advanced+| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

< Monthly Monthly Weekly > Weekly

- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7

How often do you play table tennis?

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Beginner Intermediate Advanced Advanced+

0

2

4

6

8

10

Mean number of tournaments played

| |10<br><br>8<br><br>6<br><br>5| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

B I A A+

0

2

4

6

- 8

10

12 Number of participants

| |7.4<br><br>9.1<br><br>5.6<br><br>7.1| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

B I A A+

0

2

4

6

8

10

Match duration, active games (mins)

- Fig. 16: Top row: Participants’ experience playing table tennis by group using the pre-study grouping. Bottom row: Matches played by skill group and average match duration using the pre-study grouping. Human opponent skill level: B = Beginner, I = Intermediate, A = Advanced, A+ = Advanced+

skill level. Additionally, we think that a data entry mistake was made for the advanced player that was re-assessed to advanced+.

The during match (i.e. post re-assessment) skill levels were used for the results reported in the main paper because this more accurately reflects a player’s skill. Figure 15 presents match results using the pre-study assessment. All of the trends remain the same and none of the implications from

the main results are affected.

Figure 16 presents players’ table tennis experience by prestudy assessment skill group and the distinctions between groups are less clear compared with Figure 7. For example, there is a less clear distinction between beginner and intermediate players based on how long they have played table tennis and how often they play. This gives us an additional reason to think that the during match skill group assessment

Algorithm 2 Pseudocode for LLC preference update

- 1: procedure UPDATE H(M, H, R, Z, A, α, t¯ )

- 2:
- 3: Input:
- 4: M: latest batch of match statistics
- 5: H: current preferences, one per LLC
- 6: R¯: average reward
- 7: Z: mask, zero-array
- 8: A: action counts per LLC
- 9: α: step size
- 10: t: timestep
- 11: Output:
- 12: H∗: updated preferences, one per LLC
- 13:
- 14: P = prob. each LLC is selected
- 15: P = softmax(H)
- 16: Update the H values for each shot
- 17: H∗ = H
- 18: for m ∈ M do
- 19: R = 1 if robot returned ball else 0
- 20: R = get reward(m)

- 21: λ = get llc(m)

- 22: Update timestep
- 23: t += 1
- 24: Update action count
- 25: A[λ] += 1
- 26: Update average reward
- 27: R¯ += (R − R¯)/t
- 28: Z[λ] = 1
- 29: Update preferences
- 30: H∗ += α ∗ (R − R¯) ∗ (Z − P)
- 31: end for
- 32:
- 33: return H∗, A, R¯, t
- 34:
- 35: end procedure

is the most appropriate to use.

Score (Human-Robot) Skill Level Winner Game 1 Game 2 Game 3

- Beginner Robot 5-11 5-11 6-11 Beginner Robot 1-11 4-11 3-11 Beginner Robot 0-11 2-11 3-11 Beginner Robot 9-11 8-11 11-6

- Beginner Robot 0-11 1-11 4-11

Beginner Robot 6-11 8-11 3-11

- Beginner Robot 0-11 2-11 5-11

Intermediate Robot 6-11 11-9 8-11 Intermediate Human 6-11 11-4 11-8 Intermediate Human 10-12 11-6 11-9

- Intermediate Robot 4-11 13-15 12-14
- Intermediate Robot 5-11 10-12 11-2 Intermediate Robot 8-11 11-6 9-11 Intermediate Human 10-12 11-6 11-5 Intermediate Robot 8-11 11-9 6-11 Intermediate Human 3-11 11-8 11-7
- Intermediate Robot 5-11 6-11 12-10 Intermediate Human 9-11 11-2 11-6

- Advanced Human 11-3 6-11 11-4

- Advanced Human 11-8 11-7 11-7

Advanced Human 11-4 11-3 11-2

- Advanced Human 11-9 11-4 11-6 Advanced Human 11-2 11-8 11-5

Advanced+ Human 11-7 11-9 11-5 Advanced+ Human 11-3 11-2 11-2

- Advanced+ Human 11-2 11-1 12-10 Advanced+ Human 11-9 13-11 11-4 Advanced+ Human 7-11 11-9 11-5
- Advanced+ Human 11-3 11-5 11-4

TABLE XIV: Match details for games with the main rules.

