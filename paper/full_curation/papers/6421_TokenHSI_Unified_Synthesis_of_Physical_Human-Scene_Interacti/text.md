# arXiv:2503.19901v2[cs.CV]3Apr2025

## TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization

Liang Pan1,2 Zeshi Yang3 Zhiyang Dou2 Wenjia Wang2 Buzhen Huang4 Bo Dai2,5 Taku Komura2 Jingbo Wang1†

1 Shanghai AI Laboratory 2 The University of Hong Kong 3 Independent Researcher 4 Southeast University 5 Feeling AI https://liangpan99.github.io/TokenHSI

|Foundational HSI Skill|Long-horizon Task Completion|Terrain Shape Variation|
|---|---|---|

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

|Skill Composition|Object Shape Variation|
|---|---|

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Figure 1. Introducing TokenHSI, a unified model that enables physics-based characters to perform diverse human-scene interaction tasks. It excels at seamlessly unifying multiple foundational HSI skills within a single transformer network and flexibly adapting learned skills to challenging new tasks, including skill composition, object/terrain shape variation, and long-horizon task completion.

##### Abstract

Synthesizing diverse and physically plausible Human-Scene Interactions (HSI) is pivotal for both computer animation and embodied AI. Despite encouraging progress, current methods mainly focus on developing separate controllers, each specialized for a specific interaction task. This significantly hinders the ability to tackle a wide variety of challenging HSI tasks that require the integration of multiple skills, e.g. sitting down while carrying an object (see Fig. 1). To address this issue, we present TokenHSI, a single, unified transformer-based policy capable of multi-skill unification

†Corresponding author.

and flexible adaptation. The key insight is to model the humanoid proprioception as a separate shared token and combine it with distinct task tokens via a masking mechanism. Such a unified policy enables effective knowledge sharing across skills, thereby facilitating the multi-task training. Moreover, our policy architecture supports variable length inputs, enabling flexible adaptation of learned skills to new scenarios. By training additional task tokenizers, we can not only modify the geometries of interaction targets but also coordinate multiple skills to address complex tasks. The experiments demonstrate that our approach can significantly improve versatility, adaptability, and extensibility in various HSI tasks.

##### 1. Introduction

Generating diverse and life-like Human-Scene Interactions (HSI) through physics simulation is a fascinating yet challenging task for computer animation and embodied AI. In real-world scenarios, humans function as general-purpose agents, proficient in executing a wide variety of complex interaction tasks and adept at adapting to novel contexts. Driven by this, we aim to establish a unified controller for learning versatile human-scene interaction skills and to explore an effective method for adapting learned skills to new tasks and environments, thereby narrowing the gap between simulated characters and real-world humans.

Beyond methods [8, 23, 27, 72, 75, 99, 109, 116] that develop controllers solely focused on a single interaction task, some advancements [90, 106] have aimed to devise unified controllers with diverse skills. However, they have notable limitations in two main aspects: (1) Their controllers are primarily designed for interactions in static scenes, such as sitting or touching immovable objects, and therefore cannot be applied to dynamic scenarios that require manipulation skills, such as carrying those objects. This incomplete skill set results in limited applicability and diminishes the potential to accomplish tasks that require various synergies between diverse skills, such as composite tasks (e.g., sitting down while carrying an object) and long-term manipulation tasks by sequencing multiple skills. (2) These approaches focus more on in-domain settings and suffer from limited generalization capability in novel scenes, overlooking the adaptation of learned skills to novel scenarios. The process of directly fine-tuning pre-trained policies for new tasks is inefficient [111], thereby further constraining the adaptability of their controllers.

To address these challenges, we present TokenHSI, the first physics-based character control framework designed for unifying diverse HSI skills within a single network while maintaining flexible adaptability to novel scenarios. TokenHSI constructs a separate observation space by individually tokenizing the humanoid proprioception and multiple task states. During inference, the character is directed to perform a specific task by combining the proprioception token with the corresponding task token using the masking mechanism in the transformer encoder [92]. That is, the proprioception token is shared across tasks. This design choice not only enables the unified learning of multiple HSI skills but also encourages motor knowledge sharing to boost performance. Through multi-task training, our proprioception tokenizer can generalize to a wide range of character states. To demonstrate the effectiveness, we train TokenHSI to simultaneously learn four representative HSI skills using a single transformer network, including following, sitting, climbing, and carrying.

Transcending multi-skill unification, TokenHSI further excels at quickly adapting its learned skills to tackle new,

more challenging HSI tasks. Given that our transformer policy enables variable length inputs, we can introduce additional task tokenizers to adapt the pre-trained policy to new tasks and environments both robustly and efficiently. This is thanks to the effective generalization of our proprioception tokenizer which is trained across diverse tasks, and the reuse of prior task tokenizers relevant to new contexts via the masking mechanism. Although the concurrent works [29, 90] also utilize the transformer architecture with a shared proprioception tokenizer to develop a single controller, TokenHSI differs from these approaches since we not only unify diverse HSI tasks but also further unleash the model’s flexible and efficient adaptability to new tasks. Our framework facilitates the adaptation to novel task configurations by training only few additional parameters once the foundational skills are acquired, including task tokenizers, as well as adapter layers for the multilayer perceptron (MLP) based action head that predicts the actions. It eliminates the need to fine-tune the full parameters of the pre-trained policy, thereby enhancing the efficiency of the adaptation process. TokenHSI demonstrates significantly improved sample efficiency and performance compared to recent policy adaptation methods [110, 111].

We conduct extensive experiments on a variety of HSI tasks, including skill composition, object and terrain shape variation, and long-horizon task completion in complex environments; See an overview in Fig. 1. Our results demonstrate that TokenHSI, despite its simplicity and efficiency, significantly outperforms existing methods on these challenging HSI benchmarks. Our contributions can be summarized as follows:

- 1. We present TokenHSI, a novel transformer-based physical character controller that integrates versatile HSI skills within a single, unified policy.
- 2. Once trained, our approach enables flexible and efficient policy adaptation to novel HSI tasks, avoiding the full fine-tuning of the pre-trained policy.
- 3. We propose a dedicated tokenizer to encode proprioception, effectively facilitating both multi-task training and policy adaptation.

##### 2. Related Work

###### 2.1. Human-Scene Interaction

Recent advancements in human-scene interaction motion synthesis can be categorized into two main formulations: data-driven kinematic generation and physics-based character control. Traditional approaches have synthesized human-object interactions, such as grasping, using inverse kinematics [2, 19, 36, 42, 43] and optimal control [37]. Recently, a significant body of work has introduced datadriven methods [7, 13, 17, 34, 39, 40, 44, 45, 47, 48, 64, 81, 88, 95, 96, 112, 113, 121, 124, 126, 127, 129, 130] that

utilize large-scale datasets [4, 21, 26, 47, 62, 70, 107, 114] and advanced generative motion models [10, 15, 16, 20, 31, 32, 50, 52, 53, 65, 66, 87, 91, 93, 105, 108, 128, 132], to synthesize high-quality human-scene interaction motions. However, these methods often overlook the importance of the physical plausibility.

To address this, traditional physics-based methods utilize hand-crafted controllers to implement physics-based grasping skills [82, 131]. For instance, [131] constructed a hand-grasping database to generate grasping poses for PD controllers, enabling real-time physics-based hand-grasping synthesis. As the complexity of human-object interactions increases, trajectory optimization offers a versatile and effective approach for synthesizing complex skills [9, 54, 55, 58, 59, 73, 74, 120]. However, these optimization-based methods are typically offline and cannot be deployed in real-time applications. More recently, deep reinforcement learning (DRL) has emerged as a powerful framework for learning motor skills in physically simulated agents [3, 12, 14, 18, 27, 60, 76–80, 86, 89, 115, 119, 134] and realworld humanoid robots [11, 22, 28–30, 35, 38, 94, 123]. This approach has successfully synthesized numerous impressive human-object interaction skills that were previously unattainable, including basketball dribbling [57, 100], skateboarding [56], playing tennis [97, 125], solving a Rubik’s Cube with hands [1], using chopsticks [117], and interacting with everyday objects [5, 23, 27, 49, 51, 61, 68, 75, 99, 104, 106]. However, despite these significant advancements, existing methods still fall short of establishing a single, comprehensive controller for integrating diverse skills, such as climbing, contacting, and manipulation.

###### 2.2. Unified Character Controller

Beyond controllers designed for limited skills [27, 56, 78], recent works aim to develop unified controllers for a broader range of skills. Notably, studies [67, 101, 102] have enhanced the capabilities of physics-aware motion imitation, enabling support for large-scale reference motions and accommodating a variety of body shapes. To improve the flexibility of learned skills, [18, 69, 80, 86, 103, 118, 119] have established motion manifolds using physics simulations to model versatile and reusable skills while also training additional controllers for specific tasks. UniHSI [106] proposes a unified controller for various contact-based humanscene interaction tasks, utilizing automatic task generation through large language models (LLMs). More recently, [29, 90, 98] introduced a masking mechanism to train controllers that follow on-demand tracking targets. In contrast, our approach not only incorporates the masking mechanism to unify various HSI tasks across different categories—such as following, sitting, climbing, and carrying—but also explores effective methods to adapt the skills learned by our unified character controller to new tasks and environments.

##### 3. Methodology

###### 3.1. Overview

We focus on developing a physical HSI controller capable of unifying diverse interaction skills within a single network. Beyond versatility, the unified controller should be able to generalize learned skills to novel settings, enabling it to tackle more complex HSI tasks. Driven by these goals, we propose a transformer-based policy network, leveraging its support for variable length inputs to seamlessly incorporate an arbitrary number of tasks.

During foundational skill learning (Fig. 2 left), we standardize the observation lengths of different tasks using separate task tokenizers Ttask. Importantly, we adopt a proprioception tokenizer Tprop dedicated to processing the character state st, in contrast to existing methods [23, 27, 75, 84, 99, 106, 116] that rely on a joint character-goal state space. When training a specific task, we utilize the masking mechanism within the transformer encoder ϕ [92] to combine the proprioception token with the relevant task token. Using the shared ϕ and Tprop can effectively encourage motor knowledge sharing across tasks, thereby improving the multi-task training performance.

Once trained, we adapt the unified policy to novel tasks by reusing prior tokenizers Tprop, Ttask and adding additional task tokenizers Tnew if new task observations exist. To facilitate the adaptation, we integrate zero-initialized adapter layers ξA [33] into the action head module H. This transformer-based policy adaptation is efficient as the proprioception tokenizer Tprop trained across diverse tasks gains effective generalization to a wide range of character states. We achieve several challenging HSI tasks through the adaptation of foundation skills, including skill composition, object/terrain shape variation, and long-horizon task completion.

Sec. 3.2 provides the formulation for physics-based character control. Sec. 3.3 details the construction and training of the unified transformer policy for simultaneously learning multiple foundational HSI skills. Sec. 3.4 introduces our novel policy adaptation technique based on the flexible transformer architecture.

###### 3.2. Physics-based Character Control

Formulation. We use goal-conditioned reinforcement learning to formulate character control as a Markov Decision Process (MDP) defined by states, actions, transition dynamics, a reward function r, and a discount factor γ. The reward rt ∈ R is calculated by a style reward rtstyle [79] and a task reward rttask. The policy aims to maximize the accumulated discounted reward Tt=0 γtrt. We use the widely adopted Proximal Policy Optimization (PPO) algorithm [85] to train the control policy model.

Adapters

Discriminator

[Figure 15]

[Figure 16]

|𝑎𝑡|
|---|

[Figure 17]

[Figure 18]

[Figure 19]

###### Action Head

|Linear|
|---|

|Linear|
|---|

Linear

Linear

Linear

𝑎𝑡

𝐷

[Figure 20]

[Figure 21]

[Figure 22]

Condition

Attention Mask

|𝜙|
|---|

Pre-trained

###### Transformer Encoder

Transformer Encoder

|𝑙𝑡|
|---|

0 1 0 0

𝑒

𝑒

One-hot Label Learnable Embedding

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

| | |
|---|---|
|𝑔𝑡𝑛𝑒𝑤| |

𝑔𝑡𝑐

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Obs

[Figure 43]

[Figure 44]

|𝑠𝑡|
|---|

|𝑔𝑡𝑓|
|---|

|𝑠𝑡|
|---|

|𝑔𝑡𝑠|
|---|

|𝑔𝑡𝑚|
|---|

|𝑔𝑡𝑐|
|---|

Multi-task Env

Foundational Skill Learning Policy Adaptation

| |
|---|

| |
|---|

Adapters with Zero-initialization

Frozen Pre-trained Networks

[Figure 45]

[Figure 46]

Tokenizers

Trainable Networks

Figure 2. TokenHSI consists of two stages: (left) foundational skill learning and (right) policy adaptation. Through multi-task policy training, the proposed framework learns versatile interaction skills in a single transformer network. Theses learned skills can be flexibly adapted to more challenging HSI tasks by training the lightweight modules, e.g., Tnew, Tc, and ξA = {ξ0A, ξ1A}.

State and Actions. The state ot ≜ (st,gt) consists of the character state st and the goal state gt, which is recorded in the character’s local coordinate frame. We use PD controllers at each degree of freedom of the humanoid. The action at specifies joint target rotations for the PD controllers, resulting in a 32-dim action space.

###### 3.3. Foundational Skill Learning

Policy Architecture. We propose a scalable architecture that supports variable length inputs to enable multi-task training. As illustrated in Fig. 2 (left), our policy consists of four key parts: (1) Multiple tokenizers. Each tokenizer (Tf,Ts,Tm,Tc, or Tprop) standardizes its corresponding

observation (gtf,gts,gtm,gtc, or st) into a 64-dim feature. The tokenizers are implemented as separate MLPs with 3 hidden layers of [256,128,64] units. (2) A transformer encoder ϕ comprises 4 encoder layers [92], each containing 2 attention heads and a 512-dim feed-forward layer. It fuses all information through self-attention mechanism. (3) A 64dim learnable embedding e functions as the output token. (4) An action head H = {ξ0H,ξ1H,ξ2H} produces the final action at. It is also modeled by an MLP network, but with 3 hidden layers of [1024,512,32] units.

Foundational Skills. We consider four types of HSI tasks. Their task observations are formulated as: (1) gtf ∈ R2×10, the following task [25, 84, 98], tracking a target path described by ten 2D coordinates; (2) gts ∈ R38, the sitting task [27, 75, 106], approaching and sitting on a target object; (3) gtm ∈ R27, the climbing task [41], approaching

and climbing onto a target object; (4) gtc ∈ R42, the carrying task [23, 27], moving a box from its initial location to a target location. See Sec. B.2 of the supplementary for the detailed task designs.

Multi-task Policy Training. We train the transformer policy in a multi-task environment with flat ground. During resetting, the environment is assigned to a specific task based on a pre-defined probability distribution: 10% for the pathfollowing task and 30% each for the other three tasks. The assigned task’s reward function is then applied to compute the task reward rttask. We encode the active task using a one-hot label lt. The training objective is to obtain a multitask policy π(at|e,st,gtf,gts,gtm,gtc). Leveraging the current task label lt as an attention mask, we suppress features from non-target tasks in the self-attention mechanism. Critically, the character state st and the learnable embedding e persistently participate in all attention computations. Thus, the multi-task policy reduces to a single-task formulation π(at|e,st,gl

t ). To optimize the policy, we employ a value function V (st,gtf,gts,gtm,gtc), modeled by an MLP network with 4 hidden layers of [2048,1024,512,1] units. The irrelevant inputs will be padded to zeros, reducing the value function to V (st,gl

t

t ). Besides, we condition the motion discriminator D on the one-hot label lt to prevent the policy from learning motor skills unrelated to the current task. For instance, without the conditional D, the character may incorrectly produce squatting motions during pathfollowing. The discriminator is modeled by a three layer MLP with [1024,512,1] units. We run numerous environments in parallel [71] to achieve large-scale training.

t

###### 3.4. Transformer-based Policy Adaptation

In this section, we demonstrate how to adapt the pretrained policy to address diverse new HSI tasks. We validate our approach on three challenging tasks. As depicted in Fig. 2 (right), the adaptation process first freezes these components: Tprop, e, H, and ϕ to ensure that the adapted model remains effective for prior skills. Then, we introduce zero-initialized adapter layers ξA = {ξ0A,ξ1A} to the action head H to enhance the adaptation efficiency. We train a new motion discriminator D from scratch during each task. Next, we detail how we reuse pre-trained task tokenizers and introduce new tokenizers to learn these tasks. Detailed task designs are provided in Sec. B.3 in the supplementary. Skill Composition. We aim to create composite motions, such as sitting down while carrying a box (see Fig. 4 (b)). We reuse and freeze the task tokenizers for the sitting Ts and the carrying Tc tasks. Furthermore, we introduce an additional trainable task tokenizer Tnew to perceive the new task state gtnew, which is designed to contain the states of the target object and the box. During the training, the model gradually learns to coordinate the two learned skills to complete the composite task. We also apply this method to combine other learned skills with the carrying skill, respectively.

Object Shape Variation. Possessing the ability to interact with a variety of objects is an essential feature of an HSI controller. The pre-trained policy learns a box-carrying skill. Therefore, we aim to adapt it to new environments in which the boxes are replaced with irregular objects, such as chairs and tables. Since we can use the same observation information, we directly fine-tune the pre-trained task tokenizer Tc. After training, we obtain two additional finetuned tokenizers Tcchair and Tctable, enabling the unified model to generalize to more diverse object categories.

Terrain Shape Variation. The basic skills of our policy are trained on flat ground. However, there is usually complex terrain in the real world, such as stairs. As illustrated in Fig. 2 (right), we generalize the trajectory following and carrying skills to stairs environments. Similarly, we directly fine-tune the prior task tokenizer Tf or Tc. We further introduce an additional trainable task tokenizer Tnew to encode the height map [84].

Long-horizon Task Completion. Performing long-term tasks in complex environments challenges control models in achieving seamless skill transitions and collision avoidance. Existing methods [12, 46] tackle this problem by jointly fine-tuning multiple policies to achieve fluent transitions within the long skill chain. In contrast, our policy adaptation allows users to fine-tune only the lightweight task tokenizers. Additionally, we can effortlessly make the policy environment-aware by introducing a height tokenizer, which also facilitates the character’s ability to avoid environmental obstacles. During inference, we employ a Finite

State Machine (FSM) similar to [75] to enable automated task switching. Given a one-hot task label lt queried from the FSM, we use it as an attention mask to activate the corresponding task token and disable other tokens.

##### 4. Experiments

We conduct extensive experiments to evaluate foundational skill learning and policy adaptation. In Sec. 4.1, we evaluate the robustness of 4 basic skills learned by our unified policy. In Sec. 4.2, we demonstrate the efficiency advantage of our approach in adapting to complex HSI tasks. In Sec. E of the supplementary, we further demonstrate our approach’s extensibility through introducing novel out-of-domain skills.

###### 4.1. Evaluation on Foundational Skill Learning

Experimental Setup. We compare our unified multi-task policy with policies trained individually for each task. The AMP [79] framework is employed to train these specialists. Each policy is modeled by a four layer MLP with [2048,1024,512,32] units. All policies are trained with 4,096 parallel environments and 50k PPO [85] iteration steps. 512 trials are collected to calculate the success rate. We also measure the average error for all successful trials.

Follow. We declare path-following to be successful if the pelvis is within 30 cm (XY-planar distance) of the path endpoint. The error of a trial is calculated by

n

xpelvist − xtart

, where n is the episode length, and

1 n

2

t=1

xtart is the dynamically sampled target pelvis position. Both training and testing paths are procedurally generated [84].

Sit and Climb. The success criteria for both tasks require the pelvis to lie within a 20 cm radius sphere centered at the target location. Error is the minimal 3D distance between the pelvis and its target location throughout the trial. The object is spawned between 1 m and 5 m from the character with randomized orientation. The sitting task uses 49 training and 26 testing objects, while the climbing task utilizes 38 training and 26 testing objects.

Carry. The success criterion and error calculation are the same as above, with the reference point altered from the character’s pelvis to the box’s centroid. The box is placed between 1–9 m from the character, with randomized orientation and height. The target location is uniformly sampled from a 10 m × 10 m 2D horizontal (XY) plane. 9 boxes with different sizes are used for training and 9 for testing.

Results. Tab. 1 presents the quantitative results, indicating that compared to specialists, TokenHSI achieves higher success rates while maintaining comparable errors on all foundational skills. Our unified policy exhibits superior crosstask generalization performance, making it a versatile and flexible alternative to individual controllers.

Ablation on Shared Tprop. We answer the question: can using a shared proprioception tokenizer Tprop really boost

Task Method Success Rate (%) Error (cm)

|Follow<br><br>Specialist Ours (w/o Tprop) Ours<br><br>|98.7±0.5 6.5±0.0<br><br>99.3±0.3 9.7±0.2 99.7±0.0 9.3±0.1<br><br><br>|
|---|---|
|Sit<br><br>Specialist Ours (w/o Tprop) Ours<br><br>|98.2±2.0 5.6±0.0<br><br>98.7±0.4 5.6±0.1<br><br>99.6±0.2 5.6±0.2<br>|
|Climb<br><br>Specialist Ours (w/o Tprop) Ours<br><br>|99.7±0.1 2.4±0.2 99.5±0.2 3.1±0.8<br><br>99.8±0.1 2.7±0.3<br>|
|Carry<br><br>Specialist Ours (w/o Tprop) Ours<br><br>|83.1±5.0 5.1±0.2 90.9±3.3 6.0±0.5 92.2±6.7 4.2±0.6<br><br>|

Table 1. Quantitative comparison between our unified multi-task policy and specialist policies across four foundational HSI skills. Values are reported in the format of mean±std.

the performance? We design a variant of our approach, namely Ours (w/o Tprop), where the Tprop is discarded and the proprioception st is added into the input of each task tokenizer. We observe a general decrease in the success rate across all tasks, suggesting that using a shared Tprop indeed contributes to effective knowledge sharing across tasks.

###### 4.2. Evaluation on Policy Adaptation

We evaluate the policy adaptation on more challenging HSI tasks. Comparative experiments with existing methods demonstrate the high efficiency of our approach. TokenHSI achieves efficient policy adaptation through a simple and unified design, avoiding the need to train new policies from scratch, fully fine-tune prior models, or implement sophisticated architectures like AdaptNet [111] and CML [110].

###### 4.2.1. Skill Composition

Experimental Setup. We consider three possible combinations: Follow + Carry, Sit + Carry, Climb + Carry. For every composite task, the character should perform the core interaction task (e.g., sitting) while continuously carrying the box. Task success requires simultaneous fulfillment of: (1) core task completion; (2) box-carrying maintenance (carrying is deemed failed if the box’s lowest point is less than 20 cm from the ground). Please refer to Sec. 4.1 for the success criterion, error calculation, object dataset, and object initialization for each core task. The character is initialized in a standing pose while holding a box, which means the walk-to-pick-up stage is omitted.

Baselines. We first implement Scratch, which trains policies from scratch via AMP [79], to benchmark the difficulty of different tasks. Then, we adopt CML [110], the current SOTA approach for composite motion learning. Specifically, we apply the incremental learning scheme to adapt each core task’s specialist policy. Given that our approach uses two skill priors and CML only uses one, we further establish CML (dual) that can jointly coordinate two pretrained policies (one for the core task and another for the carrying task) to ensure a fair comparison. See Sec. C of the supplementary for the details of CML (dual). All base policies are pre-trained with 50k iterations in Sec. 4.1, except

Follow + Carry Sit + Carry Climb + Carry

Method

Succ. (%) Err. (cm) Succ. (%) Err. (cm) Succ. (%) Err. (cm)

|Scratch [79] CML [110] CML (dual)|98.1±0.7 12.5±0.1<br><br>97.6±0.8 9.9±0.0<br><br>99.1±0.2 10.0±0.0<br><br><br>|82.9±5.0 5.9±0.1 96.7±0.6 6.2±0.2 94.9±2.0 5.9±0.0<br><br>|26.8±37.8 5.7±4.8 68.3±21.9 4.8±0.5 51.3±40.2 9.1±7.3<br><br>|
|---|---|---|---|
|Ours (w/o Tprop) Ours<br><br>|99.2±0.5 11.5±0.2 99.1±0.4 10.8±0.3|82.0±4.2 6.9±0.9 96.3±1.1 5.5±0.0<br><br>|81.9±9.0 5.0±0.5 99.2±0.1 3.5±0.1<br><br>|

Table 2. Quantitative results across skill composition tasks.

Sit + Carry Climb + Carry

[Figure 47]

[Figure 48]

Success Rate (%) Success Rate (%)

Scratch

CML

CML (dual)

Ours (w/o )

[Figure 49]

Ours

Number of Iterations (1e2) Number of Iterations (1e2)

Figure 3. Learning curves comparing the efficiency on skill composition tasks using TokenHSI, policies trained from scratch [79], CML [110], and its improved version CML (dual). Colored regions denote mean values ± a standard deviation based on 3 models initialized with different random seeds.

the box-carrying policy, which is further trained to 100k iterations. This eliminates the performance gap between the carrying specialist (95.7%) and TokenHSI (97.4%). The training on skill composition exploits 4,096 environments and 5k iterations.

Results. We report quantitative results in Tab. 2. For the first two easier tasks–where even the Scratch attains 98.1% and 82.9% success rates–TokenHSI achieves near-optimal performance. Despite the drastic performance degradation observed in baselines as task difficulty escalates (Scratch: 80%+ → 26.8%, CML: 90%+ → 68.3%, and CML (dual): 90%+ → 51.3%), our approach still maintains a high success rate of 99.2% on the third challenging Climb + Carry task. In Fig. 3, we further show the learning curves of different methods to compare their convergence processes intuitively. TokenHSI exhibits superior efficiency and stability compared to all baselines, especially on the Climb + Carry task. We attribute these advantages to (1) the proprioception tokenizer Tprop trained across diverse tasks offers better generalization performance than the base policies (trained on a single task) used by CML and CML (dual); (2) the attention mechanism enables skill composition at an earlier stage (within the transformer encoder ϕ’s latent feature space), whereas CML-based approaches are limited to postcomposition in the action space. The synthesized composite motions are shown in Fig. 4 (a-c).

Ablation on Shared Tprop. We validate the importance of the shared Tprop on the policy adaptation. We adapt a base model trained without Tprop to learn composition tasks. During training, the additional Tnew observes (st,gtnew). The results show a significant performance (both robustness and efficiency) drop in the last two difficult tasks, which

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

###### (a) Skill Composition: follow + carry (b) Skill Composition: sit + carry (c) Skill Composition: climb + carry

[Figure 59]

[Figure 60]

[Figure 61]

###### (d) Object Shape Variation: irregular object carrying

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

(e) Terrain Shape Variation: box carrying across uneven terrain

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

(f) Terrain Shape Variation: trajectory following across uneven terrain

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

(g) Long-horizon Task Completion: performing a sequence of sub-tasks (following, carrying, climbing, and sitting) in a complex environment

- Figure 4. Through policy adaptation, TokenHSI can generalize learned foundational skills to more challenging scene interaction tasks.

Object Method Success Rate (%) Error (cm)

|Chair<br><br>Finetune AdaptNet [111] Ours|87.5±0.6 6.4±0.2<br><br>84.5±3.0 6.8±0.5<br><br>88.8±3.1 5.6±0.2<br><br><br>|
|---|---|
|Table<br><br>Finetune AdaptNet [111] Ours|83.4±1.6 6.0±0.1<br><br>82.4±3.9 6.4±0.3<br><br>83.6±1.6 6.3±0.2<br>|

Table 3. Quantitative results across object shape variation tasks.

[Figure 74]

[Figure 75]

Carry: Box-2-Table

Success Rate (%) Success Rate (%)

Finetune

AdaptNet

Ours

Carry: Box-2-Chair

Number of Iterations (2e2) Number of Iterations (5e2)

- Figure 5. Learning curves comparing the efficiency on object shape variation tasks using TokenHSI, full fine-tuning of pretrained policies, and AdaptNet [111].

object’s height axis < 30◦ from the ground’s vertical axis (Z) to filter out incorrect object placement poses. The object is placed on the ground between 1–5 m from the character, with randomized orientation. The target location is still uniformly sampled from a 10 m × 10 m 2D area.

Baselines. We first establish Finetune, which fine-tunes a specialized box-carrying policy. Then, we employ AdaptNet [111], the current SOTA network architecture for policy adaptation. The baselines’ base box-carrying training requires 100k iterations, versus 50k for TokenHSI, as detailed in Sec. 4.2.1. The training on object shape variation adopts 4,096 environments and 10k iterations.

Results. Tab. 3 shows that our approach surpasses all baselines in success rate. As shown in Fig 5, TokenHSI exhibits efficiency and stability advantages across most comparisons, except when compared with Finetune on the Chair category. Fine-tuning improves efficiency by overwriting all parameters, but it cannot retain prior skills. In contrast, our goal is skill-preserving adaptation. Under this requirement, TokenHSI significantly outperforms the fair baseline AdaptNet [111], validating our architectural superiority. The qualitative motions are presented in Fig. 4 (d).

further demonstrates that the design choice of modeling the proprioception as a separate token is necessary.

###### 4.2.2. Object Shape Variation

###### 4.2.3. Terrain Shape Variation

Experimental Setup. We aim to adapt Follow and Carry skills trained on flat ground to uneven terrain, which necessitates that the policy adaptation be flexible enough to incorporate new observations (e.g., height perception). We consider four types of terrain blocks: stairs up, stairs down,

Experimental Setup. We evaluate on two irregular object categories: Chair and Table, with the training-testing instance numbers being 63–27 and 21–9, respectively. The success criterion and error calculation are identical to the carrying task in Sec. 4.1. We add a constraint requiring the

Follow Carry

Method

Succ. (%) Err. (cm) Succ. (%) Err. (cm)

Scratch [79] 93.4±0.7 14.9±0.2 0 – AdaptNet [111] 92.4±0.5 12.3±0.2 63.4±0.6 5.9±0.3 Ours (w/o adapters) 63.0±1.1 22.5±0.4 10.8±1.5 7.7±0.1 Ours 96.0±0.4 11.8±0.0 74.0±2.5 5.9±0.2

Table 4. Quantitative results across terrain shape variation tasks.

Follow: Flat Ground-2-Terrain Carry: Flat Ground-2-Terrain

[Figure 76]

[Figure 77]

Success Rate (%) Success Rate (%)

Scratch

AdaptNet

Ours (w/oadapters)

###### Ours

Number of Iterations (1e2) Number of Iterations (1e3)

- Figure 6. Learning curves comparing the efficiency on terrain shape variation tasks using TokenHSI, Scratch [79], and AdaptNet [111]. We ablate the adapter layers during training.

obstacles, and flat ground, with an initialization probability of [0.35,0.35,0.2,0.1]. We follow [84, 98] to calculate a walkable map for character initialization.

Baselines. We first train policies π(at|st,gt,ht) from scratch, where gt is the corresponding task state and ht denotes the height map. We then employ AdaptNet [111] with the incorporation of the height map ht via its latent space injection. Similarly, we make TokenHSI height-aware during adaptation through training a new task tokenizer Tnew for processing ht, as illustrated in Fig. 2. The training on terrain shape variation uses 2,048 environments with 5k iterations for Follow and 50k iterations for Carry.

Results. As indicated by Tab. 4 and Fig. 6, our approach still maintains the efficiency advantage and outperforms all baselines in quantitative metrics. Compared to AdaptNet [111], TokenHSI supports both flexible input length and efficient learning, representing a significant advancement in policy adaptation for HSI. The generated animations are shown in Fig. 4 (e) and (f).

Ablation on Adapters. During training, we remove the adapter layers ξA to conduct ablation studies. The results demonstrate that attaching learnable bypasses to the action head H is critical for improving performance.

###### 4.2.4. Long-horizon Task Completion

In this section, we validate how TokenHSI facilitates longhorizon task completion in complex 3D scenes through policy adaptation. Directly executing multiple skills sequentially is prone to: (1) generating unseen transition states, which may cause subsequent skills to fail [12, 46]; (2) getting stuck by obstacles, since the application scenes are usually more cluttered than the training scenes, as illustrated in Fig. 7 (a). We design a long-horizon task containing these challenges. The task involves walking to pick up a box,

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

(a) Pre-trained Skills (b) Adapted Skills

Figure 7. Long-horizon task completion by sequentially executing (a) pre-trained skills and (b) adapted skills by our approach.

placing it near a platform, and sitting on a chair on the platform after climbing onto it using the box. We first introduce a new height map tokenizer to make all skills environmentaware, and then iteratively fine-tune each task tokenizer in the complex scene. After policy adaptation, TokenHSI successfully tackles the challenging long-horizon task, resulting in fluent skill execution and transition, as shown in Fig. 4 (g). Fig. 7 (b) also demonstrates that our adapted skills can correctly place the box in preparation for the next climbing skill. TokenHSI obviates the need to manually design transition states [109] and jointly fine-tune multiple policies [12, 46]. A more in-depth quantitative analysis is provided in Sec. D of the supplementary.

##### 5. Discussion and Limitations

In this work, we present TokenHSI to tackle the problem of unified synthesis of physical HSI animations. TokenHSI is a unified model that learns various HSI skills within a single transformer network and can flexibly generalize learned skills to novel tasks and environments through a simple yet efficient policy adaptation. We conduct extensive experiments to demonstrate that TokenHSI significantly improves versatility, adaptability, and extensibility in HSI.

The main limitation is that learning these skills requires engineering of reward functions, which involve tedious trial-and-error processes. However, this is a general problem for the goal-oriented RL framework. In the future, we should explore effective approaches using human data [83] or internet knowledge [6] to reduce the cost on reward engineering. Besides, the current long-horizon task completion is still non-autonomous. A simulated humanoid that can complete complex, long-term tasks in realistic environments without human guidance remains an open problem.

##### 6. Acknowledgments

We would like to thank Zhewen Zheng for his professional rendering techniques, which helped make appealing figures and videos in our paper. We also appreciate the anonymous reviewers for their constructive comments that improved the final version of this paper. This work is funded in part by the National Key R&D Program of China (2022ZD0160201), HKU Startup Fund, and Shanghai Artificial Intelligence Laboratory. This work is partly supported by the Innovation and Technology Commission of the HKSAR Government under the ITSP-Platform grant (Ref: ITS/335/23FP) and the InnoHK initiative (TransGP project). The research work was in part conducted in the JC STEM Lab of Robotics for Soft Materials funded by The Hong Kong Jockey Club Charities Trust.

##### References

- [1] Ilge Akkaya, Marcin Andrychowicz, Maciek Chociej, Mateusz Litwin, Bob McGrew, Arthur Petron, Alex Paino, Matthias Plappert, Glenn Powell, Raphael Ribas, et al. Solving rubik’s cube with a robot hand. arXiv preprint arXiv:1910.07113, 2019. 3
- [2] Yahya Aydin and Masayuki Nakajima. Database guided computer animation of human grasping using forward and inverse kinematics. Computers & Graphics, 23(1):145– 154, 1999. 2
- [3] Kevin Bergamin, Simon Clavet, Daniel Holden, and James Richard Forbes. Drecon: Data-driven responsive control of physics-based characters. ACM Transactions on Graphics (TOG), 38(6), 2019. 3
- [4] Bharat Lal Bhatnagar, Xianghui Xie, Ilya A Petrov, Cristian Sminchisescu, Christian Theobalt, and Gerard Pons-Moll. Behave: Dataset and method for tracking human object interactions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15935– 15946, 2022. 3
- [5] Jona Braun, Sammy Christen, Muhammed Kocabas, Emre Aksan, and Otmar Hilliges. Physically plausible full-body hand-object interaction synthesis. In 2024 International Conference on 3D Vision (3DV), pages 464–473. IEEE,

2024. 3

- [6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023. 8
- [7] Zhi Cen, Huaijin Pi, Sida Peng, Zehong Shen, Minghui Yang, Shuai Zhu, Hujun Bao, and Xiaowei Zhou. Generating human motion in 3d scenes from text descriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1855–1866, 2024. 2
- [8] Yu-Wei Chao, Jimei Yang, Weifeng Chen, and Jia Deng. Learning to sit: Synthesizing human-chair interactions via hierarchical control. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 5887–5895, 2021. 2

- [9] Penghui Chen, Yushi Wang, Changsheng Luo, Wenhan Cai, and Mingguo Zhao. Hifar: Multi-stage curriculum learning for high-dynamics humanoid fall recovery. arXiv preprint arXiv:2502.20061, 2025. 3
- [10] Rui Chen, Mingyi Shi, Shaoli Huang, Ping Tan, Taku Komura, and Xuelin Chen. Taming diffusion probabilistic models for character control. In ACM SIGGRAPH 2024 Conference Papers, pages 1–10, 2024. 3
- [11] Xuxin Cheng, Yandong Ji, Junming Chen, Ruihan Yang, Ge Yang, and Xiaolong Wang. Expressive whole-body control for humanoid robots. arXiv preprint arXiv:2402.16796,

2024. 3

- [12] Alexander Clegg, Wenhao Yu, Jie Tan, C Karen Liu, and Greg Turk. Learning to dress: Synthesizing human dressing motion via deep reinforcement learning. ACM Transactions on Graphics (TOG), 37(6), 2018. 3, 5, 8
- [13] Peishan Cong, Ziyi Wang, Zhiyang Dou, Yiming Ren, Wei Yin, Kai Cheng, Yujing Sun, Xiaoxiao Long, Xinge Zhu, and Yuexin Ma. Laserhuman: language-guided sceneaware human motion generation in free environment. arXiv preprint arXiv:2403.13307, 2024. 2
- [14] Jieming Cui, Tengyu Liu, Nian Liu, Yaodong Yang, Yixin Zhu, and Siyuan Huang. Anyskill: Learning openvocabulary physical skill for interactive agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 852–862, 2024. 3
- [15] Wenxun Dai, Ling-Hao Chen, Yufei Huo, Jingbo Wang, Jinpeng Liu, Bo Dai, and Yansong Tang. Real-time controllable motion generation via latent consistency model. 3
- [16] Wenxun Dai, Ling-Hao Chen, Jingbo Wang, Jinpeng Liu, Bo Dai, and Yansong Tang. Motionlcm: Real-time controllable motion generation via latent consistency model. In European Conference on Computer Vision, pages 390–408. Springer, 2024. 3
- [17] Christian Diller and Angela Dai. Cg-hoi: Contact-guided 3d human-object interaction generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19888–19901, 2024. 2
- [18] Zhiyang Dou, Xuelin Chen, Qingnan Fan, Taku Komura, and Wenping Wang. C· ase: Learning conditional adversarial skill embeddings for physics-based characters. In SIGGRAPH Asia 2023 Conference Papers, pages 1–11, 2023. 3
- [19] George ElKoura and Karan Singh. Handrix: animating the human hand. In Proceedings of the 2003 ACM SIGGRAPH/Eurographics Symposium on Computer Animation, pages 110–119, 2003. 2
- [20] Ke Fan, Junshu Tang, Weijian Cao, Ran Yi, Moran Li, Jingyu Gong, Jiangning Zhang, Yabiao Wang, Chengjie Wang, and Lizhuang Ma. Freemotion: A unified framework for number-free text-to-motion synthesis. In European Conference on Computer Vision, pages 93–109. Springer,

2024. 3

- [21] Huan Fu, Bowen Cai, Lin Gao, Ling-Xiao Zhang, Jiaming Wang, Cao Li, Qixun Zeng, Chengyue Sun, Rongfei Jia, Binqiang Zhao, et al. 3d-front: 3d furnished rooms with layouts and semantics. In Proceedings of the IEEE/CVF In-

- ternational Conference on Computer Vision, pages 10933– 10942, 2021. 3, 2
- [22] Zipeng Fu, Qingqing Zhao, Qi Wu, Gordon Wetzstein, and Chelsea Finn. Humanplus: Humanoid shadowing and imitation from humans. arXiv preprint arXiv:2406.10454,

2024. 3

- [23] Jiawei Gao, Ziqin Wang, Zeqi Xiao, Jingbo Wang, Tai Wang, Jinkun Cao, Xiaolin Hu, Si Liu, Jifeng Dai, and Jiangmiao Pang. Coohoi: Learning cooperative humanobject interaction with manipulated object dynamics. Advances in Neural Information Processing Systems, 37: 79741–79763, 2024. 2, 3, 4
- [24] F Sebastian Grassia. Practical parameterization of rotations using the exponential map. Journal of graphics tools, 3(3): 29–48, 1998. 1
- [25] Assaf Hallak, Gal Dalal, Chen Tessler, Kelly Guo, Shie Mannor, and Gal Chechik. Plamo: Plan and move in rich 3d physical environments. arXiv preprint arXiv:2406.18237,

2024. 4

- [26] Mohamed Hassan, Duygu Ceylan, Ruben Villegas, Jun Saito, Jimei Yang, Yi Zhou, and Michael J Black. Stochastic scene-aware motion prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11374–11384, 2021. 3, 2
- [27] Mohamed Hassan, Yunrong Guo, Tingwu Wang, Michael Black, Sanja Fidler, and Xue Bin Peng. Synthesizing physical character-scene interactions. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–9, 2023. 2, 3, 4
- [28] Tairan He, Zhengyi Luo, Xialin He, Wenli Xiao, Chong Zhang, Weinan Zhang, Kris M Kitani, Changliu Liu, and Guanya Shi. Omnih2o: Universal and dexterous humanto-humanoid whole-body teleoperation and learning. In 8th Annual Conference on Robot Learning, 2024. 3
- [29] Tairan He, Wenli Xiao, Toru Lin, Zhengyi Luo, Zhenjia Xu, Zhenyu Jiang, Jan Kautz, Changliu Liu, Guanya Shi, Xiaolong Wang, Linxi Fan, and Yuke Zhu. Hover: Versatile neural whole-body controller for humanoid robots. arXiv preprint arXiv:2410.21229, 2024. 2, 3
- [30] Tairan He, Jiawei Gao, Wenli Xiao, Yuanhang Zhang, Zi Wang, Jiashun Wang, Zhengyi Luo, Guanqi He, Nikhil Sobanbab, Chaoyi Pan, et al. Asap: Aligning simulation and real-world physics for learning agile humanoid wholebody skills. arXiv preprint arXiv:2502.01143, 2025. 3
- [31] Daniel Holden, Taku Komura, and Jun Saito. Phasefunctioned neural networks for character control. ACM Transactions on Graphics (TOG), 36(4):1–13, 2017. 3
- [32] Daniel Holden, Oussama Kanoun, Maksym Perepichka, and Tiberiu Popa. Learned motion matching. ACM Transactions on Graphics (ToG), 39(4):53–1, 2020. 3
- [33] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 3
- [34] Siyuan Huang, Zan Wang, Puhao Li, Baoxiong Jia, Tengyu Liu, Yixin Zhu, Wei Liang, and Song-Chun Zhu. Diffusionbased generation, optimization, and planning in 3d scenes. In Proceedings of the IEEE/CVF Conference on Computer

- Vision and Pattern Recognition, pages 16750–16761, 2023. 2
- [35] Tao Huang, Junli Ren, Huayi Wang, Zirui Wang, Qingwei Ben, Muning Wen, Xiao Chen, Jianan Li, and Jiangmiao Pang. Learning humanoid standing-up control across diverse postures. arXiv preprint arXiv:2502.08378, 2025. 3
- [36] Zhiyong Huang, Ronan Boulic, Nadia Magnenat Thalmann, and Daniel Thalmann. A multi-sensor approach for grasping and 3d interaction. In Computer graphics, pages 235–253. Elsevier, 1995. 2
- [37] Sumit Jain and C Karen Liu. Controlling physics-based characters using soft contacts. ACM Transactions on Graphics (TOG), 30(6):1–10, 2011. 2
- [38] Mazeyu Ji, Xuanbin Peng, Fangchen Liu, Jialong Li, Ge Yang, Xuxin Cheng, and Xiaolong Wang. Exbody2: Advanced expressive humanoid whole-body control. arXiv preprint arXiv:2412.13196, 2024. 3
- [39] Nan Jiang, Zimo He, Zi Wang, Hongjie Li, Yixin Chen, Siyuan Huang, and Yixin Zhu. Autonomous characterscene interaction synthesis from text instruction. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024. 2
- [40] Nan Jiang, Zhiyuan Zhang, Hongjie Li, Xiaoxuan Ma, Zan Wang, Yixin Chen, Tengyu Liu, Yixin Zhu, and Siyuan Huang. Scaling up dynamic human-scene interaction modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1737– 1747, 2024. 2
- [41] Niloofar Khoshsiyar, Ruiyu Gou, Tianhong Zhou, Sheldon Andrews, and M van de Panne. Partwisempc: Interactive control of contact-guided motions. In Computer Graphics Forum, page e15174. Wiley Online Library, 2024. 4
- [42] Junhwan Kim, Frederic Cordier, and Nadia MagnenatThalmann. Neural network-based violinist’s hand animation. In Proceedings Computer Graphics International 2000, pages 37–41. IEEE, 2000. 2
- [43] Yoshihito Koga, Koichi Kondo, James Kuffner, and JeanClaude Latombe. Planning motions with intentions. In Proceedings of the 21st annual conference on Computer graphics and interactive techniques, pages 395–408, 1994. 2
- [44] Jiye Lee and Hanbyul Joo. Locomotion-actionmanipulation: Synthesizing human-scene interactions in complex 3d environments. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9663– 9674, 2023. 2
- [45] Kyungho Lee, Seyoung Lee, and Jehee Lee. Interactive character animation by learning multi-objective control. ACM Transactions on Graphics (TOG), 37(6):1–10, 2018. 2
- [46] Youngwoon Lee, Joseph J Lim, Anima Anandkumar, and Yuke Zhu. Adversarial skill chaining for long-horizon robot manipulation via terminal state regularization. In 5th Annual Conference on Robot Learning, 2021. 5, 8
- [47] Jiaman Li, Jiajun Wu, and C Karen Liu. Object motion guided human motion synthesis. ACM Transactions on Graphics (TOG), 42(6):1–11, 2023. 2, 3
- [48] Jiaman Li, Alexander Clegg, Roozbeh Mottaghi, Jiajun Wu, Xavier Puig, and C Karen Liu. Controllable human-object

- interaction synthesis. In European Conference on Computer Vision, pages 54–72. Springer, 2024. 2
- [49] Jianan Li, Tao Huang, Qingxu Zhu, and Tien-Tsin Wong. Physics-based scene layout generation from human motion. In ACM SIGGRAPH 2024 Conference Papers, pages 1–10,

2024. 3

- [50] Ronghui Li, YuXiang Zhang, Yachao Zhang, Hongwen Zhang, Jie Guo, Yan Zhang, Yebin Liu, and Xiu Li. Lodge: A coarse to fine diffusion network for long dance generation guided by the characteristic dance primitives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1524–1534, 2024. 3
- [51] Yitang Li, Mingxian Lin, Zhuo Lin, Yipeng Deng, Yue Cao, and Li Yi. Learning physics-based full-body human reaching and grasping from brief walking references. arXiv preprint arXiv:2503.07481, 2025. 3
- [52] Jing Lin, Ailing Zeng, Shunlin Lu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, and Lei Zhang. Motion-x: A largescale 3d expressive whole-body human motion dataset. Advances in Neural Information Processing Systems, 36: 25268–25280, 2023. 3
- [53] Hung Yu Ling, Fabio Zinno, George Cheng, and Michiel Van De Panne. Character controllers using motion vaes. ACM Transactions on Graphics (TOG), 39(4):40–1, 2020. 3
- [54] C Karen Liu. Synthesis of interactive hand manipulation. In Proceedings of the 2008 ACM SIGGRAPH/Eurographics Symposium on Computer Animation, pages 163–171, 2008. 3
- [55] C Karen Liu. Dextrous manipulation from a grasping pose. ACM Transactions on Graphics (TOG), 28(3), 2009. 3
- [56] Libin Liu and Jessica Hodgins. Learning to schedule control fragments for physics-based characters using deep Qlearning. ACM Transactions on Graphics (TOG), 36(3),

2017. 3

- [57] Libin Liu and Jessica Hodgins. Learning basketball dribbling skills using trajectory optimization and deep reinforcement learning. ACM Transactions on Graphics (TOG), 37(4):1–14, 2018. 3
- [58] Libin Liu, KangKang Yin, Michiel Van de Panne, Tianjia Shao, and Weiwei Xu. Sampling-based contact-rich motion control. In ACM SIGGRAPH 2010 papers, pages 1–10.

2010. 3

- [59] Libin Liu, KangKang Yin, and Baining Guo. Improving sampling-based motion control. In Computer Graphics Forum, pages 415–423. Wiley Online Library, 2015. 3
- [60] Siqi Liu, Guy Lever, Zhe Wang, Josh Merel, SM Ali Eslami, Daniel Hennes, Wojciech M Czarnecki, Yuval Tassa, Shayegan Omidshafiei, Abbas Abdolmaleki, et al. From motor control to team play in simulated humanoid football. Science Robotics, 7(69):eabo0235, 2022. 3
- [61] Yun Liu, Bowen Yang, Licheng Zhong, He Wang, and Li Yi. Mimicking-bench: A benchmark for generalizable humanoid-scene interaction learning via human mimicking. arXiv preprint arXiv:2412.17730, 2024. 3
- [62] Yun Liu, Chengwen Zhang, Ruofan Xing, Bingda Tang, Bowen Yang, and Li Yi. Core4d: A 4d human-object-

- human interaction dataset for collaborative object rearrangement. arXiv preprint arXiv:2406.19353, 2024. 3
- [63] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multi-person linear model. ACM Trans. Graphics (Proc. SIGGRAPH Asia), 34(6):248:1–248:16, 2015. 1
- [64] Jintao Lu, He Zhang, Yuting Ye, Takaaki Shiratori, Sebastian Starke, and Taku Komura. Choice: Coordinated human-object interaction in cluttered environments for pick-and-place actions. arXiv preprint arXiv:2412.06702,

2024. 2

- [65] Shunlin Lu, Ling-Hao Chen, Ailing Zeng, Jing Lin, Ruimao Zhang, Lei Zhang, and Heung-Yeung Shum. Humantomato: Text-aligned whole-body motion generation. arXiv preprint arXiv:2310.12978, 2023. 3
- [66] Shunlin Lu, Jingbo Wang, Zeyu Lu, Ling-Hao Chen, Wenxun Dai, Junting Dong, Zhiyang Dou, Bo Dai, and Ruimao Zhang. Scamo: Exploring the scaling law in autoregressive motion generation model. arXiv preprint arXiv:2412.14559, 2024. 3
- [67] Zhengyi Luo, Jinkun Cao, Kris Kitani, Weipeng Xu, et al. Perpetual humanoid control for real-time simulated avatars. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10895–10904, 2023. 3
- [68] Zhengyi Luo, Jinkun Cao, Sammy Christen, Alexander Winkler, Kris Kitani, and Weipeng Xu. Omnigrasp: Grasping diverse objects with simulated humanoids. Advances in Neural Information Processing Systems, 37:2161–2184,

2024. 3

- [69] Zhengyi Luo, Jinkun Cao, Josh Merel, Alexander Winkler, Jing Huang, Kris M Kitani, and Weipeng Xu. Universal humanoid motion representations for physics-based control. In The Twelfth International Conference on Learning Representations, 2024. 3
- [70] Naureen Mahmood, Nima Ghorbani, Nikolaus F Troje, Gerard Pons-Moll, and Michael J Black. Amass: Archive of motion capture as surface shapes. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5442–5451, 2019. 3, 1, 2, 6
- [71] Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, et al. Isaac gym: High performance gpu-based physics simulation for robot learning. arXiv preprint arXiv:2108.10470, 2021. 4, 1
- [72] Josh Merel, Saran Tunyasuvunakool, Arun Ahuja, Yuval Tassa, Leonard Hasenclever, Vu Pham, Tom Erez, Greg Wayne, and Nicolas Heess. Catch & carry: reusable neural controllers for vision-guided whole-body tasks. ACM Transactions on Graphics (TOG), 39(4):39–1, 2020. 2
- [73] Igor Mordatch, Zoran Popovi´c, and Emanuel Todorov. Contact-invariant optimization for hand manipulation. In Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation, pages 137–144, 2012. 3
- [74] Igor Mordatch, Emanuel Todorov, and Zoran Popovi´c. Discovery of complex behaviors through contact-invariant optimization. ACM Transactions on Graphics (ToG), 31(4): 1–8, 2012. 3

- [75] Liang Pan, Jingbo Wang, Buzhen Huang, Junyu Zhang, Haofan Wang, Xu Tang, and Yangang Wang. Synthesizing physically plausible human motions in 3d scenes. In 2024 International Conference on 3D Vision (3DV), pages 1498–1507. IEEE, 2024. 2, 3, 4, 5
- [76] Soohwan Park, Hoseok Ryu, Seyoung Lee, Sunmin Lee, and Jehee Lee. Learning predict-and-simulate policies from unorganized human motion data. ACM Transactions on Graphics (TOG), 38(6), 2019. 3
- [77] Xue Bin Peng, Glen Berseth, KangKang Yin, and Michiel Van De Panne. Deeploco: Dynamic locomotion skills using hierarchical deep reinforcement learning. ACM Transactions on Graphics (TOG), 36(4), 2017.
- [78] Xue Bin Peng, Pieter Abbeel, Sergey Levine, and Michiel van de Panne. Deepmimic: Example-guided deep reinforcement learning of physics-based character skills. ACM Transactions on Graphics (TOG), 37(4), 2018. 3, 2
- [79] Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, and Angjoo Kanazawa. Amp: Adversarial motion priors for stylized physics-based character control. ACM Transactions on Graphics (ToG), 40(4):1–20, 2021. 3, 5, 6, 8, 1
- [80] Xue Bin Peng, Yunrong Guo, Lina Halper, Sergey Levine, and Sanja Fidler. Ase: Large-scale reusable adversarial skill embeddings for physically simulated characters. ACM Transactions On Graphics (TOG), 41(4):1–17, 2022. 3, 1
- [81] Huaijin Pi, Sida Peng, Minghui Yang, Xiaowei Zhou, and Hujun Bao. Hierarchical generation of human-object interactions with diffusion probabilistic models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15061–15073, 2023. 2
- [82] Nancy S Pollard and Victor Brian Zordan. Physically-based grasping control from example. In Proceedings of the 2005 ACM SIGGRAPH/Eurographics Symposium on Computer Animation, pages 311–318, 2005. 3
- [83] Ri-Zhao Qiu, Shiqi Yang, Xuxin Cheng, Chaitanya Chawla, Jialong Li, Tairan He, Ge Yan, Lars Paulsen, Ge Yang, Sha Yi, et al. Humanoid policy˜ human policy. arXiv preprint arXiv:2503.13441, 2025. 8
- [84] Davis Rempe, Zhengyi Luo, Xue Bin Peng, Ye Yuan, Kris Kitani, Karsten Kreis, Sanja Fidler, and Or Litany. Trace and pace: Controllable pedestrian animation via guided trajectory diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13756–13766, 2023. 3, 4, 5, 8, 2
- [85] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 3, 5
- [86] Agon Serifi, Ruben Grandia, Espen Knoop, Markus Gross, and Moritz B¨acher. Vmp: Versatile motion priors for robustly tracking motion on physical characters. In Computer Graphics Forum, page e15175. Wiley Online Library, 2024. 3
- [87] Yi Shi, Jingbo Wang, Xuekun Jiang, Bingkun Lin, Bo Dai, and Xue Bin Peng. Interactive character control with autoregressive motion diffusion models. ACM Transactions on Graphics (TOG), 43(4):1–14, 2024. 3

- [88] Sebastian Starke, He Zhang, Taku Komura, and Jun Saito. Neural state machine for character-scene interactions. ACM Transactions on Graphics, 38(6):178, 2019. 2
- [89] Jie Tan, Tingnan Zhang, Erwin Coumans, Atil Iscen, Yunfei Bai, Danijar Hafner, Steven Bohez, and Vincent Vanhoucke. Sim-to-real: Learning agile locomotion for quadruped robots. In Proceedings of Robotics: Science and Systems, 2018. 3
- [90] Chen Tessler, Yunrong Guo, Ofir Nabati, Gal Chechik, and Xue Bin Peng. Maskedmimic: Unified physics-based character control through masked motion inpainting. ACM Transactions on Graphics (TOG), 43(6):1–21, 2024. 2, 3
- [91] Guy Tevet, Sigal Raab, Brian Gordon, Yoni Shafir, Daniel Cohen-or, and Amit Haim Bermano. Human motion diffusion model. In The Eleventh International Conference on Learning Representations, 2023. 3
- [92] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 2, 3, 4
- [93] Weilin Wan, Zhiyang Dou, Taku Komura, Wenping Wang, Dinesh Jayaraman, and Lingjie Liu. Tlcontrol: Trajectory and language control for human motion synthesis. In European Conference on Computer Vision, pages 37–54. Springer, 2024. 3
- [94] Huayi Wang, Zirui Wang, Junli Ren, Qingwei Ben, Tao Huang, Weinan Zhang, and Jiangmiao Pang. Beamdojo: Learning agile humanoid locomotion on sparse footholds. arXiv preprint arXiv:2502.10363, 2025. 3
- [95] Jiashun Wang, Huazhe Xu, Jingwei Xu, Sifei Liu, and Xiaolong Wang. Synthesizing long-term 3d human motion and interaction in 3d scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9401–9411, 2021. 2
- [96] Jingbo Wang, Sijie Yan, Bo Dai, and Dahua Lin. Sceneaware generative network for human motion synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12206–12215, 2021. 2
- [97] Jiashun Wang, Jessica Hodgins, and Jungdam Won. Strategy and skill learning for physics-based table tennis animation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 3
- [98] Jingbo Wang, Zhengyi Luo, Ye Yuan, Yixuan Li, and Bo Dai. Pacer+: On-demand pedestrian animation controller in driving scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 718–728, 2024. 3, 4, 8
- [99] Wenjia Wang, Liang Pan, Zhiyang Dou, Zhouyingcheng Liao, Yuke Lou, Lei Yang, Jingbo Wang, and Taku Komura. Sims: Simulating human-scene interactions with real world script planning. arXiv preprint arXiv:2411.19921, 2024. 2, 3
- [100] Yinhuai Wang, Qihan Zhao, Runyi Yu, Ailing Zeng, Jing Lin, Zhengyi Luo, Hok Wai Tsui, Jiwen Yu, Xiu Li, Qifeng Chen, et al. Skillmimic: Learning reusable basketball skills from demonstrations. arXiv preprint arXiv:2408.15270,

2024. 3

- [101] Jungdam Won and Jehee Lee. Learning body shape variation in physics-based characters. ACM Transactions on Graphics (TOG), 38(6), 2019. 3

- [102] Jungdam Won, Deepak Gopinath, and Jessica Hodgins. A scalable approach to control diverse behaviors for physically simulated characters. ACM Transactions on Graphics (TOG), 39(4):33–1, 2020. 3
- [103] Jungdam Won, Deepak Gopinath, and Jessica Hodgins. Physics-based character controllers using conditional vaes. ACM Transactions on Graphics (TOG), 41(4):1–12, 2022. 3
- [104] Zhen Wu, Jiaman Li, Pei Xu, and C Karen Liu. Humanobject interaction from human-level instructions. arXiv preprint arXiv:2406.17840, 2024. 3
- [105] Lixing Xiao, Shunlin Lu, Huaijin Pi, Ke Fan, Liang Pan, Yueer Zhou, Ziyong Feng, Xiaowei Zhou, Sida Peng, and Jingbo Wang. Motionstreamer: Streaming motion generation via diffusion-based autoregressive model in causal latent space. arXiv preprint arXiv:2503.15451, 2025. 3
- [106] Zeqi Xiao, Tai Wang, Jingbo Wang, Jinkun Cao, Wenwei Zhang, Bo Dai, Dahua Lin, and Jiangmiao Pang. Unified human-scene interaction via prompted chain-of-contacts. In The Twelfth International Conference on Learning Representations, 2024. 2, 3, 4
- [107] Xianghui Xie, Jan Eric Lenssen, and Gerard Pons-Moll. Intertrack: Tracking human object interaction without object templates. arXiv preprint arXiv:2408.13953, 2024. 3
- [108] Yiming Xie, Varun Jampani, Lei Zhong, Deqing Sun, and Huaizu Jiang. Omnicontrol: Control any joint at any time for human motion generation. In ICLR, 2024. 3
- [109] Zhaoming Xie, Jonathan Tseng, Sebastian Starke, Michiel van de Panne, and C Karen Liu. Hierarchical planning and control for box loco-manipulation. Proceedings of the ACM on Computer Graphics and Interactive Techniques, 6(3):1– 18, 2023. 2, 8
- [110] Pei Xu, Xiumin Shang, Victor Zordan, and Ioannis Karamouzas. Composite motion learning with task control. ACM Transactions on Graphics (TOG), 42(4):1–16, 2023. 2, 6, 5
- [111] Pei Xu, Kaixiang Xie, Sheldon Andrews, Paul G Kry, Michael Neff, Morgan McGuire, Ioannis Karamouzas, and Victor Zordan. Adaptnet: Policy adaptation for physicsbased character control. ACM Transactions on Graphics (TOG), 42(6):1–17, 2023. 2, 6, 7, 8
- [112] Sirui Xu, Zhengyuan Li, Yu-Xiong Wang, and Liang-Yan Gui. Interdiff: Generating 3d human-object interactions with physics-informed diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14928–14940, 2023. 2
- [113] Sirui Xu, Yu-Xiong Wang, Liangyan Gui, et al. Interdreamer: Zero-shot text to 3d dynamic human-object interaction. Advances in Neural Information Processing Systems, 37:52858–52890, 2024. 2
- [114] Sirui Xu, Dongting Li, Yucheng Zhang, Xiyan Xu, Qi Long, Ziyin Wang, Yunzhi Lu, Shuchang Dong, Hezi Jiang, Akshat Gupta, Yu-Xiong Wang, and Liang-Yan Gui. Interact: Advancing large-scale versatile 3d human-object interaction generation. In CVPR, 2025. 3
- [115] Sirui Xu, Hung Yu Ling, Yu-Xiong Wang, and Liang-Yan Gui. Intermimic: Towards universal whole-body control for

- physics-based human-object interactions. arXiv preprint arXiv:2502.20390, 2025. 3
- [116] Xinyu Xu, Yizheng Zhang, Yong-Lu Li, Lei Han, and Cewu Lu. Humanvla: Towards vision-language directed object rearrangement by physical humanoid. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 2, 3
- [117] Zeshi Yang, Kangkang Yin, and Libin Liu. Learning to use chopsticks in diverse gripping styles. ACM Transactions on Graphics (TOG), 41(4):1–17, 2022. 3
- [118] Heyuan Yao, Zhenhua Song, Baoquan Chen, and Libin Liu. Controlvae: Model-based learning of generative controllers for physics-based characters. ACM Trans. Graph., 41(6),

2022. 3

- [119] Heyuan Yao, Zhenhua Song, Yuyang Zhou, Tenglong Ao, Baoquan Chen, and Libin Liu. Moconvq: Unified physicsbased motion control via scalable discrete representations. ACM Transactions on Graphics (TOG), 43(4):1–21, 2024. 3
- [120] Yuting Ye and C Karen Liu. Synthesis of detailed hand manipulations using contact sampling. ACM Transactions on Graphics (TOG), 31(4), 2012. 3
- [121] Hongwei Yi, Justus Thies, Michael J Black, Xue Bin Peng, and Davis Rempe. Generating human interaction motions in scenes with text control. In European Conference on Computer Vision, pages 246–263. Springer, 2024. 2
- [122] Ye Yuan, Shih-En Wei, Tomas Simon, Kris Kitani, and Jason Saragih. Simpoe: Simulated character control for 3d human pose estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7159–7169, 2021. 1
- [123] Chong Zhang, Wenli Xiao, Tairan He, and Guanya Shi. Wococo: Learning whole-body humanoid control with sequential contacts. arXiv preprint arXiv:2406.06005, 2024. 3
- [124] He Zhang, Yuting Ye, Takaaki Shiratori, and Taku Komura. Manipnet: Neural manipulation synthesis with a hand-object spatial representation. ACM Transactions on Graphics (TOG), 40(4), 2021. 2
- [125] Haotian Zhang, Ye Yuan, Viktor Makoviychuk, Yunrong Guo, Sanja Fidler, Xue Bin Peng, and Kayvon Fatahalian. Learning physically simulated tennis skills from broadcast videos. ACM Trans. Graph., 42(4), 2023. 3
- [126] Wanyue Zhang, Rishabh Dabral, Thomas Leimk¨uhler, Vladislav Golyanik, Marc Habermann, and Christian Theobalt. Roam: Robust and object-aware motion generation using neural pose descriptors. In 2024 International Conference on 3D Vision (3DV), pages 1392–1402. IEEE,

2024. 2

- [127] Xiaohan Zhang, Bharat Lal Bhatnagar, Sebastian Starke, Vladimir Guzov, and Gerard Pons-Moll. Couch: Towards controllable human-chair interactions. In European Conference on Computer Vision, pages 518–535. Springer, 2022. 2
- [128] Xiaohan Zhang, Bharat Lal Bhatnagar, Sebastian Starke, Ilya Petrov, Vladimir Guzov, Helisa Dhamo, Eduardo P´erez-Pellitero, and Gerard Pons-Moll. Force: Dataset and

- method for intuitive physics guided human-object interaction. arXiv e-prints, pages arXiv–2403, 2024. 3
- [129] Yan Zhang and Siyu Tang. The wanderings of odysseus in 3d scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20481– 20491, 2022. 2
- [130] Kaifeng Zhao, Yan Zhang, Shaofei Wang, Thabo Beeler, and Siyu Tang. Synthesizing diverse human motions in 3d indoor scenes. In Proceedings of the IEEE/CVF international conference on computer vision, pages 14738–14749,

2023. 2

- [131] Wenping Zhao, Jianjie Zhang, Jianyuan Min, and Jinxiang Chai. Robust realtime physics-based motion control for human grasping. ACM Transactions on Graphics (TOG), 32

(6), 2013. 3

- [132] Wenyang Zhou, Zhiyang Dou, Zeyu Cao, Zhouyingcheng Liao, Jingbo Wang, Wenjia Wang, Yuan Liu, Taku Komura, Wenping Wang, and Lingjie Liu. Emdm: Efficient motion diffusion model for fast and high-quality motion generation. In European Conference on Computer Vision, pages 18–38. Springer, 2024. 3
- [133] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5745–5753, 2019. 1
- [134] Qingxu Zhu, He Zhang, Mengting Lan, and Lei Han. Neural categorical priors for physics-based character control. ACM Transactions on Graphics (TOG), 42(6):1–16, 2023. 3

## TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization

### Supplementary Material

##### A. Simulated Character

Character Model Creation. We apply a custom simulated character model, with 32 degrees-of-freedom (DoF). This custom model is based on the character model used in AMP [79], which comprises 15 rigid bodies, 12 controllable joints, and 28 DoF, as depicted in Fig. A (a). While retaining most designs of AMP’s character model, we introduce three improvements:

- • (1) The 3D relative positions of the lower body joints, including the hips, knees, and ankles, are adjusted to match those in the SMPL [63] human body model configured with a neutral gender and default shape parameters.
- • (2) The collision shapes of the foot rigid bodies are modified from rectangular boxes to realistic foot meshes using the method proposed by SimPoE [122].
- • (3) The knee joints are upgraded from 1-DoF revolute joints to 3-DoF spherical joints.

An illustration of our custom character model is available in Fig. A (b). The primary motivation for building this custom model is two fold: the reference motion datasets are represented using SMPL parameters, and the kinematic structure of AMP’s character model is different from that of SMPL. Consequently, directly copying rotation parameters to retarget these motions onto AMP’s character model leads to unnatural lower body motions. In contrast, using our improved character model, which features a lower body structure consistent with SMPL, can significantly reduce retargeting errors and ensure more natural character motions.

The designed simulated character is used for most tasks, except those involving stairs terrain, as illustrated in Fig. 4 (e) and (f). This difference is due to inaccurate contact simulation between the meshed foot rigid bodies and the terrain in IsaacGym [71]. To address this issue, we revert the collision shapes of foot rigid bodies back to rectangular boxes, which are more simulation-friendly, as shown in Fig. A (c).

The Proprioception st and Action at. The proprioception st describes the simulated state of the character at each time step t. Following ASE [80], st is constructed using a set of features, including the positions, rotations, linear velocities, and angular velocities of all rigid bodies. All features are expressed in the character’s local coordinate frame, except for the root joint rotation, which is represented in the world coordinate frame. The 6D rotation representation [133] is employed. Notably, the root joint position is excluded from the proprioception. Combined, these fea-

[Figure 84]

[Figure 85]

[Figure 86]

(a) AMP (b) Ours (c) Ours (for terrain)

Figure A. Different simulated character models. Building on (a) AMP’s model, we devise two improved versions: (b) and (c), which are used for tasks on flat ground and tasks on stairs terrain, respectively.

tures define a 222D humanoid proprioception st ∈ R222. At each time step t, the control policy generates an action at, representing the target rotations for the PD controllers at each of the character’s degrees-of-freedom. The target rotations for 3D spherical joints are encoded using a 3D exponential map [24]. Our character model has ten 3-DoF spherical joints and two 1-DoF revolute joints (i.e., left and right elbows), resulting in a 32D action space at ∈ R32. No external forces are applied to any rigid body of the simulated character.

##### B. Tasks

In this section, we provide the implementation details about all tasks involved in this paper. Tab. A presents an overview of all 12 tasks, including 4 foundational HSI tasks, 3 skill composition tasks, 4 object/terrain shape variation tasks, and 1 long-horizon task. We begin by introducing the common settings shared across tasks in Section B.1. Taskspecific settings, such as task observations and reward functions, are detailed in the subsequent sections.

###### B.1. Preliminaries

Reference Motion Dataset. To encourage the character to perform tasks in a realistic and life-like manner, we manually construct a comprehensive reference motion dataset encompassing a wide variety of behavior categories associated with the four foundational HSI tasks. The dataset is divided into five distinct subsets:

• Loco: This subset includes 12 motion sequences from the AMASS [70] dataset, covering basic locomotion behaviors such as standing, walking, and turning around on flat ground. Since every task involves a walking stage, this subset is used for all tasks.

Num. of Num. of Obj. Reference Motion Dataset

Early Termination Condition

Task

Epis. Len. (s)

Task Tokens Train Test Loco Stair Climb Carry Sit Char. Fall Obj. Fall Path Dist. IET

|Follow Sit Climb Carry|1 1 1 1<br><br>|/ / 49 26 38 26 9 9<br><br>|✓<br><br>✓ ✓<br><br>✓ ✓<br><br>✓ ✓|10 10 10 20<br><br>|✓ ✓<br><br>✓ ✓ ✓ ✓ ✓|
|---|---|---|---|---|---|
|Follow + Carry Sit + Carry Climb + Carry<br><br>|3 3 3|/ + 5 / + 9 49 + 5 26 + 9 38 + 5 26 + 9<br><br>|✓ ✓<br><br>✓ ✓ ✓<br><br>✓ ✓ ✓<br><br>|10 10 10|✓ ✓ ✓<br><br>✓ ✓ ✓ ✓ ✓ ✓<br><br>|
|Obj. Shap. Var. (Chair) Obj. Shap. Var. (Table) Terr. Shap. Var. (Follow) Terr. Shap. Var. (Carry)<br><br>|1<br><br>1<br>2<br><br><br>2<br>|63 27<br><br>21 9 / / 9 9<br><br>|✓ ✓ ✓ ✓ ✓ ✓<br><br>✓ ✓ ✓<br><br>|20 20 10 20|✓ ✓ ✓ ✓ ✓ ✓<br><br>✓<br><br>|
|Long-horizon Task|5<br><br>|/ /|✓ ✓ ✓ ✓|40<br><br>|✓ ✓|

Table A. The overview of all 12 tasks implemented in this paper. Key settings for each task are summarized, including the number of task tokens, the construction of reference motion and object datasets, the episode length, and early termination conditions. The available termination conditions contain character fall, object fall, path distance, and interaction early termination (IET). A slash (/) indicates that the specific configuration is not applicable.

- • Stair: This subset is used for tasks on stairs terrain and consists of 20 motion sequences for ascending and descending stairs.
- • Climb: To support the training of the climbing task, we collect 11 motion sequences from the AMASS [70] dataset, where characters climb onto a high platform from the ground.
- • Carry: We collect the carrying motions from hybrid sources, with 17 sequences from the OMOMO [47] dataset and 4 sequences from the AMASS [70] dataset.
- • Sit: This subset consists of 20 sitting motions collected from the SAMP [26] dataset.

The usage of these five motion datasets in each task’s training process is summarized in Tab. A. For skill composition tasks, such as sitting down while carrying an object, no post-processing is applied to merge the two corresponding subsets (i.e., Carry and Sit) to obtain composite kinematic reference motions. Therefore, the reference motion dataset does not include any motions of sitting down while carrying a box. The policy learns these composite skills primarily through the guidance of task rewards. As the policy learns composite tasks, the style reward decreases while the task reward increases, leading to an overall increase in total reward and improved task completion.

Object Dataset. To make learned interaction skills effectively generalize to diverse unseen objects, we construct an object training dataset and a corresponding testing dataset to evaluate the generalization capabilities of these skills. The high-quality 3D object models are collected from the 3DFront [21] object dataset, while the 3D models of boxes are procedurally generated using Trimesh. The number of objects used for training and testing is described in Table A. We carefully ensure all test are conducted on the unseen objects.

Early Termination Condition. Early termination is an effective technique for improving the reinforcement learning (RL) training process by preventing negative samples from adversely affecting the policy gradient [78]. A fundamental and widely applicable termination condition is humanoid fall detection, which is utilized for all tasks in our implementation. To further facilitate the learning of dynamic object-carrying skills, we introduce a similar condition called object fall detection. If the object’s height drops below a specified threshold, the trial will be terminated. For the path following task, we adopt the path distance detection condition proposed in Trace&Pace [84]. If the 2D distance between the root of the simulated character and the target point on the trajectory at the current moment exceeds a specified threshold, the trial will be terminated. For tasks such as sitting and climbing, where the physical character enters a static interaction state with the object upon task completion, we introduce the Interaction Early Termination (IET) condition proposed by InterScene [75]. This effectively enhances smoothness and increases the success rate performance of these particular tasks.

###### B.2. Foundational HSI Tasks

B.2.1. Path Following Definition. This task requires the simulated character to move along a target 2D trajectory. We follow the prior work [84] to procedurally generate the trajectory dataset. A whole trajectory is formulated as τ = {xτ0.1,xτ0.2,...,xτT−0.1,xτT}, where xτ0.1 denotes a 2D waypoint of the trajectory τ at the simulation time 0.1s, and T is the episode length. According to Tab. A, the path following task’s episode length T is 10s. The character needs to follow this trajectory τ accurately.

Task Observation. At each simulation moment t second, we query 10 future waypoints {xτt ,xτt+0.1,...,xτt+0.8,xτt+0.9} in the future 1.0s from

the whole trajectory τ by linear interpolation. The sampling time interval is 0.1s. We use the 2D coordinates of the sampled waypoints as the task observation gtf ∈ R2×10. Task Reward. The task reward rtf calculates the distance between the current character 2D root position xtroot 2d and the desired target waypoint xτt :

rtf = exp − 2.0 xroott 2d − xτt 2 . (1)

###### B.2.2. Sitting

Definition. The task objective is for the character to move its root joint to a target 3D sitting position located on the object surface. The target position is placed 10 cm above the center of the top surface of the chair seat.

Task Observation. The sitting task observation gts ∈ R38 includes the 3D target sitting position ∈ R3 and the 3D information of the interacting object, i.e., the root position ∈ R3, the root rotation ∈ R6, and the 2D front-facing direction ∈ R2, as well as the positions of eight corner points on the object’s bounding box ∈ R3×8.

Task Reward. The sitting policy is trained by minimizing the distance between the character’s 3D root position xroott and the target 3D sitting position xtart . The task reward rts is defined as:

 

0.7 rtnear + 0.3 rtfar, xobjt 2d − xroott 2d > 0.5 0.7 rtnear + 0.3,otherwise

rts =



(2) rtfar = exp − 2.0 1.5 − d∗t · x˙roott 2d 2 (3)

rtnear = exp − 10.0 xtart − xroott 2 , (4) where xroott is the 3D coordinates of the character’s root, x˙roott 2d is the 2D linear velocity of the character’s root, xobjt 2d is the 2D position of the object root, d∗t is a horizontal unit vector pointing from xroott 2d to xobjt 2d, a · b represents vector dot product.

###### B.2.3. Climbing

Definition. In this work, we introduce a new contact-based interaction task similar to the sitting task. The goal is for the character to stand on a given object, placing its root joint at a target 3D climbing position. We place the target position 94 cm above the center of the top surface of the object.

Task Observation. The task observation gtm ∈ R27 includes the target root position ∈ R3 and the 3D coordinates of eight corner points on the object’s bounding box ∈ R3×8. Task Reward. This task is also optimized through minimizing the 3D distance between the character’s root xroott and its target location xtart . We formulate the task reward rtm, as follows:

 

0.5 rtnear + 0.2 rtfar, xobjt 2d − xroott 2d > 0.7 0.5 rtnear + 0.2 + 0.3 rtfoot,otherwise

rtm =



(5)

rtfar = exp − 2.0 1.5 − d∗t · x˙troot 2d 2 (6)

rtnear = exp − 10.0 xtart − xroott 2 (7) rtfoot = exp − 50.0 (xttar h − 0.94) − xtfoot h

2

, (8)

where xtart h denotes the height component of the 3D target root position, xtart , (xttar h − 0.94) represent the height of the top surface of the target object in the world coordinate, and xfoott h denotes the mean height of the two foot rigid bodies. The reward function rtfoot is introduced to encourage the character to lift its feet, which is applied when the character is close enough to the target object. We find it is crucial for the successful training of the climbing task.

B.2.4. Carrying Definition. The character is directed to move a box from a randomly initial 3D location xboxt init to a target 3D location xboxt tar. We use two thin platforms to support the box since the its initial and target heights are randomly generated. Task Observation. The task observation gtc ∈ R42 comprises the following properties of the target box:

- • Target location of the box ∈ R3
- • Root position ∈ R3
- • Root rotation ∈ R6
- • Root linear velocity ∈ R3
- • Root angular velocity ∈ R3
- • Positions of 8 corner points on the bounding box ∈ R3×8 Task Reward. We implement the multi-stage task reward function proposed by InterPhys [27]. The first stage aims to encourage the character to walk to the initial box. The corresponding reward rtc walk is defined as:

 

0.2, xobjt 2d − xroott 2d < 0.5 0.2 exp − 5.0 1.5 − d∗t · x˙troot 2d 2 ,

rtc walk =

(9)



otherwise

where d∗t is a horizontal unit vector pointing from xroott 2d to xobjt 2d, a · b represents vector dot product. The second stage is to encourage the character to pick up and move the box to its target location. We utilize two reward functions to achieve this stage, i.e., rtc carry to calculate the 3D distance between the box current root position xobjt and its target location xtart , and rtc pick to calculate the 3D distance between the box position xobjt and the mean 3D position of the character’s two hands xhand. We define rtc carry as follows:

 

0.2 rtnear + 0.2 rtfar,

rtc carry =

xobjt 2d − xtart 2d > 0.5 0.2 rtnear + 0.2,otherwise

(10)



2

(11)

rtfar = exp − 5.0 1.5 − d#t · x˙tobj 2d

2

rtnear = exp − 10.0 xtart − xobjt

, (12)

where xobjt is the 3D coordinates of the box’s root, x˙objt 2d is the 2D linear velocity of the box’s root, xobjt 2d is the 2D position of the object root, xtart 2d is the 2D coordinates of the box’s target location, d#t is a horizontal unit vector pointing from xobjt 2d to xtart 2d, a · b represents vector dot product. The task reward rtc pick to incentivize the character pick up the box using its hands, defined as follows:

 

0.0, xobjt 2d − xtroot 2d > 0.7 0.2 exp − 5.0 xobjt − xhandt

2

rtc pick =

(13)

,



otherwise

where xhandt denotes the mean 3D coordinates of the character’s two hands. Additionally, we further design a reward

function rtc put to incentivize the character to put down the box at its target location accurately, which is formulated as:

 

0.0, xobjt 2d − xtart 2d > 0.1 0.2 exp − 10.0 xobjt h − xtart h

2

rtc put =

(14)

, otherwise



where xtobj h denotes the hight of the current box and xtar

h

t

represents the height of the target placing position. Therefore, the total reward function rtc for training the carrying skill can be formulated as:

rtc = rtc walk + rtc carry + rtc pick + rtc put. (15)

###### B.3. Downstream HSI Tasks

In this section, we provide the details about how we implement the task observations and rewards used for training these more challenging HSI tasks.

Skill Composition. When learning the composite tasks using our policy adaptation, we reuse and freeze two relevant task tokenizers of foundational skills. Their task observations are illustrated in Sec B.2. Therefore, we mainly focus on describing how we construct the task configurations for these composite tasks.

Follow + Carry. The task observation gtf+c contains two parts: (1) the primary following task observation gtf ∈ R2×10 and (2) the revised carrying task observation gtc revised ∈ R39, which excludes the target location of the box ∈ R3 because the carrying task is no longer the primary task. The final composite task observation is gtf+c ∈ R2×10+39 that considers both the primary task states and the carrying box states. We define the task reward for this composite task rtf+c as follows:

 

0.0, xobjt 2d − xroott 2d > 0.7 0.5 rtf + 0.5 rtc pick,otherwise

rtf+c =

(16)



where the rtf is the same as Equ. 1 and the rtc pick is equal to Equ. 13.

Sit + Carry. The task observation gts+c also includes two parts: (1) the primary sitting task observation gts ∈ R38 and (2) the revised carrying task observation gtc revised ∈ R39, which have been introduced before. The final composite task observation is gts+c ∈ R38+39. We define the task reward for this composite task rts+c as follows:

 

0.0, xobjt 2d − xroott 2d > 0.7 0.7 rts + 0.3 rtc pick,otherwise

rts+c =

(17)



where the rts is the same as Equ. 2 and the rtc pick is equal to Equ. 13.

Climb + Carry. The task observation gtm+c also includes two parts: (1) the primary climbing task observation gtm ∈ R27 and (2) the revised carrying task observation gtc revised ∈ R39, which have been introduced before. The final composite task observation is gtm+c ∈ R27+39. We define the task reward for this composite task rtm+c as follows:

 

0.0, xobjt 2d − xroott 2d > 0.7 0.7 rtm + 0.3 rtc pick,otherwise

rtm+c =

(18)



where the rtm is the same as Equ. 5 and the rtc pick is equal to Equ. 13.

Object/Terrain Shape Variation. For object shape variation, we directly fine-tune the pre-trained box-carrying task tokenizer Tc. That is, the task observation is still gtc ∈ R42. And we reuse the box-carrying reward function Equ. 15. For terrain shape variation, we introduce an additional task tokenizer for perceiving the surrounding height map, which use 1024 sensor points to represent the height values in a 2×2 m2 square area centered at the humanoid root position. Thus, the new height map observation is gtnew ∈ R1024. We also reuse the task observation and reward function of the box-carrying task for terrain shape variation.

Long-horizon Task Completion. As illustrated in Fig. 4 (g), we sequence the four learned foundational skills to perform a long-horizon task in a complex environment. We reuse all observations of foundational skills and introduce a new height map observation gtnew ∈ R625, which utilizes 625 sensor points to observe the heights in a 1×1 m2 square area. We design a step-by-step task reward mechanism. For each step in the task sequence, we reuse the task reward from the corresponding task ((rtf,rts,rtm, or rtc)). Once a step is completed, the reward value for that sub-task is set to its maximal value, indicating the task have been accomplished. Then, the task reward for the next step in the sequence will be activated for reward calculation. Please refer to our publicly released code for more details.

##### C. Implementation Details of CML

In Sec. 4.2.1, we compare our transformer-based policy adaptation with CML [110] and CML (dual) on the skill composition tasks. The network structure of CML (dual) is an improved version based on the original CML framework.

CML [110] employs a hierarchical framework consisting of a pre-trained, fixed meta policy πmeta as the low-level controller and a newly introduced, trainable policy πnew as the high-level controller. The high-level policy πnew observes the humanoid proprioception st and the new task observation gtnew. The low-level policy πmeta observes the humanoid proprioception st and the base task observation gtbase. Take the Climb + Carry task as an example. We use a specialist policy trained on the climbing task as the πmeta(ametat |st,gtm), which possesses a joint character-goal state space. Then, we introduce a new policy πnew(anewt ,wtnew|st,gtm+c), which generates a new action anewt and a group of joint-wise weights wtnew ∈ R32, each value is ∈ [0,1]. The high-level policy πnew is trained to cooperate with the low-level policy πmeta to quickly learn the composite tasks. The composition process is conducted in the action space as follows:

at = anewt + wtnewametat , (19) which is called post-composition in the main paper.

However, the original CML framework supports only a single meta policy. To ensure a fair comparison, we develop CML (dual), an improved version that can simultaneously utilize two meta policies π1meta and π2meta. To handle the two sets of actions ameta

t and ameta

t , generated by the two meta policies, the high-level policy πnew outputs an additional set of weights. In this way, we obtain wnew

1

2

t and wnew

1

t . This results in the following post-composition process:

2

at = anewt + wnew

t ameta

t + wnew

t ameta

t , (20) where wnew

1

1

2

2

t and wnew

t are joint-wise weights applied to the two sets of meta actions, ameta

1

2

t and ameta

t , respectively. All weights are processed using sigmoid activations, transforming their values to [0,1].

1

2

##### D. Quantitative Evaluation on Long-horizon Task Completion

Experimental Setup. We first describe the construction of the long-horizon task shown in Fig. 4 (g). The long task comprises four sequential sub-tasks: follow a target trajectory → carry a box to its target location → climb onto the box → sit on a chair located on the high platform. Each subtask should have a sub-goal. The finite state machine monitors the task executing process using the spatial relationship between the reference point (the humanoid root joint or the

#### Long-term Task Completion

[Figure 87]

The completed sub-task count

Scratch

Finetune

Ours

Number of Iterations (1e2)

Figure B. Learning curves comparing the efficiency on longhorizon task completion using TokenHSI, Scratch [79], and iterative fine-tuning of multiple pre-trained specialist policies, namely Finetune.

box centroid) and the sub-goal. Each sub-goal is procedurally generated—follow: the trajectory is planned by A*; carry: the target box position is placed close to the platform using a rule-based method; climb and sit: the character’s target root position is pre-defined on the object geometry. We use the completed sub-task count as the evaluation metric. The maximal value is 4 in our case. We collect 512 trials to statistic the metrics.

Baselines. We compare our approach with two baseline methods: (1) Scratch [79]: training a policy to learn the whole long-term task from scratch; (2) Finetune [12, 46]: iterative fine-tuning multiple specialist policies in the environment to improve skill transitions and collision avoidance. We use our transformer policy as the policy architecture when conducting the experiment Scratch. Both Scratch and TokenHSI can observe height map. The difference between these two approaches is that our approach utilized pre-trained parameters. Due to the limited flexibility of MLP-based policies used by the experiment Finetune, we cannot make them environment-aware. The training adopts 1,024 parallel simulation environments and 3k PPO iterations.

Quantitative Results. Our method achieves the highest value of the completed sub-task count of 3.79 ± 0.14, significantly outperforming Scratch 0.82 ± 0.06 and Finetune 1.86 ± 0.02. We also illustrate the convergence curves in Fig. B, which shows that TokenHSI still maintains its efficiency advantage in the long-horizon task completion.

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

- (a)
- (b)

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Figure C. Qualitative results of new skills learned by our policy adaptation. (a) We first learn two out-of-domain interaction skills, i.e., pushing down a large object and walking to a target location while lifting up a box using two hands. (b) We then combine the new lifting skill with previously learned sitting and path-following skills. These results demonstrate the good extensibility of our transformer policy.

##### E. Extensibility

In the main paper, we mainly focus on adapting skills learned in the first stage (i.e., foundational skill learning) to address more challenging HSI tasks through policy adaptation. In this section, we want to evaluate the extensibility of our approach to more HSI skills. We attempt to answer two questions: (1) Can we insert out-of-domain skills into the pre-trained transformer policy? (2) Can we further combine the newly-added skills with previously learned foundational skills to create more compositional cases? The flexibility of the transformer policy allows us to explore these problems.

- Q1: Out-of-domain Skill Insertion. We consider two more types of manipulation skills, including pushing down a large object and walking to a target location while lifting up a box using two hands. To prepare the training, we collect the reference motions from the AMASS dataset [70], design the task observations, rewards, and other environmental configurations. During training, we introduce a randomly initialized task tokenizer Tnew and zero-initialized adapter layers to the action head H. The rest network parameters are frozen. For the pushing task, we declare it to be successful if the object falls. For the lifting task, we determine a testing trial to be successful if the pelvis is within 20 cm (XY-planar distance) of the target location while maintaining the box in a lifted position. As shown in Fig. C (a), our approach successfully synthesizes these out-of-domain manipulation skills. Specifically, the pushing task attains a success rate of 100% and the lifting task receives 80.6% ± 2.6.
- Q2: More Compositional Cases. Moreover, we combine the newly learned box-lifting skill with previous sitting and following skills. The training method is the same as skill composition. Through policy adaptation, we create more compositional cases shown in Fig. C (b). The success rates are 72.1% ± 2.8 and 91.1% ± 0.8, respectively.

