# arXiv:2512.22342v4[cs.RO]23Jan2026

## VL-LN Bench: Towards Long-horizon Goal-oriented Navigation with Active Dialogs

### Wensi Huang1,2∗, Shaohao Zhu2,3∗, Meng Wei2,4, Jinming Xu3, Xihui Liu4, Hanqing Wang2†, Tai Wang2†, Feng Zhao1†, Jiangmiao Pang2

Abstract—In most existing embodied navigation tasks, instructions are well-defined and unambiguous, such as instruction following and object searching. Under this idealized setting, agents are required solely to produce effective navigation outputs conditioned on vision and language inputs. However, realworld navigation instructions are often vague and ambiguous, requiring the agent to resolve uncertainty and infer user intent through active dialog. To address this gap, we propose Interactive Instance Goal Navigation (IIGN), a task that requires agents not only to generate navigation actions but also to produce language outputs via active dialog, thereby aligning more closely with practical settings. IIGN extends Instance Goal Navigation (IGN) by allowing agents to freely consult an oracle in natural language while navigating. Building on this task, we present the Vision Language–Language Navigation (VLLN) benchmark, which provides a large-scale, automatically generated dataset and a comprehensive evaluation protocol for training and assessing dialog-enabled navigation models. VL-LN comprises over 41k long-horizon dialog-augmented trajectories for training and an automatic evaluation protocol with an oracle capable of responding to agent queries. Using this benchmark, we train a navigation model equipped with dialog capabilities and show that it achieves significant improvements over the baselines. Extensive experiments and analyses further demonstrate the effectiveness and reliability of VL-LN for advancing research on dialog-enabled embodied navigation. Code and dataset can be found at https://0309hws.github.io/VLLN.github.io/.

I. INTRODUCTION

A practical navigation agent must handle vague tasks by both planning effectively and resolving ambiguities. Active dialog offers a natural solution, allowing the agent to clarify underspecified instructions and obtain cues for efficient navigation. To study this ability, we propose Interactive Instance Goal Navigation (IIGN; as shown in Fig. 1), which extends Instance-level Object Navigation (ION, named as IGN in our paper) [1]. In IIGN, the agent receives a basic ObjectNav [2] instruction (e.g.,“Search for the <category>”), which is insufficient to uniquely identify the target instance, and must consult an oracle via dialog. Since IIGN inherently poses exploration and disambiguation challenges, it provides an ideal testbed for examining the role of dialog in navigation.

To effectively investigate and address the proposed IIGN task, this paper aims to build an Agent–Oracle interaction benchmark for dialog-enabled navigation agent training and evaluation. Prior work [3] collected human–human dialogs in navigation to evaluate whether agents can understand and follow instructions. However, such efforts do not assess an agent’s ability to proactively ask targeted questions. More

1University of Science and Technology of China. 2Shanghai AI Laboratory. 3Zhejiang University. 4The University of Hong Kong.

∗ Equal Contribution. † Corresponding Author

recent approaches attempt to enable agents to generate questions, but they either limit the task to small-scale, room-level settings [4] or focus narrowly on instance descriptions with limited support for long-horizon exploration [5]. Moreover, existing methods generally lack large-scale training datasets, which constrains the development of agents capable of both active exploration and informative questioning.

In contrast to prior work, we propose the Vision Language–Language Navigation (VL-LN) benchmark, which provides both a large-scale, automatically generated training dataset and a comprehensive evaluation protocol for developing and assessing dialog-enabled navigation agents in long-horizon settings. The automated data collecting pipeline comprises three steps: (1) aggregating room-level region and instance attributes from MMScan [6] into unified houselevel annotations; (2) pairing target instances with feasible initial positions to instantiate episodes; and (3) collecting dialog-rich trajectories using a frontier-based navigator and a scripted oracle that answers questions to support exploration and disambiguation. Through this process, we curate the first large-scale training dataset of ∼ 41K dialog-augmented trajectories for IIGN. The VL-LN benchmark further incorporates an evaluation protocol with the scripted oracle, enabling the assessment of agents’ dialog generation capabilities without requiring human intervention. Together, the dataset and evaluation protocol establish a unified benchmark for training and evaluating agents’ abilities in both languagebased querying and navigation (LN) across IIGN and IGN.

We evaluate representative baselines on our benchmark and analyze emerging challenges. IGN performance remains substantially lower than ObjectNav, highlighting two key difficulties: efficient exploration and instance-level disambiguation. Training on our collected IIGN data yields a language-enabled agent that achieves state-of-the-art results on both IIGN and IGN, demonstrating the benefit of proactive querying. Nevertheless, the gains are not sufficient to bridge the disparity. A detailed error analysis reveals that the main bottleneck lies in image–attribute alignment, with 73% of failures caused by missing or misidentifying the target under detailed attributes. Moreover, in IIGN, the agent’s questioning is less efficient than humans’. Closing the gap needs better grounding, planning, and reasoning so the agent can identify the core ambiguity during exploration and pose maximally informative questions that efficiently reduce the candidate set. Our primary contributions are three-fold:

- • An automatic pipeline for generating house-level, longhorizon IIGN trajectories.
- • VL-LN, the first IIGN benchmark that provides a largescale dialog-augmented dataset and online evaluation

[Figure 1]

- Fig. 1. A case for the IIGN task. The oracle (top left) first gives a simple goal-oriented navigation instruction (“Search for the chair.”). The agent has to locate a specific instance of the given category (chair). The agent can ask three types of questions—attribute, route, and disambiguation—to progressively resolve ambiguity and locate the target (instance). The full description in the bottom right is the instruction given to the agent in the IGN task, which can locate the specific chair in this environment.

for both navigation and querying.

• Extensive experiments showing that active dialog improves agent performance on IGN and IIGN, achieving the best results while revealing remaining challenges.

II. RELATED WORKS A. Goal-oriented navigation

Goal-oriented navigation requires an agent to find a specified goal in an unknown environment [7]. In the text-guided setting, it is commonly divided into Object-goal Navigation (ObjectNav) [2], [8] and Instance-level Object Navigation (ION) [1]. Most prior work focuses on ObjectNav, where the instruction is unambiguous because any instance of the target category suffices. These approaches fall into two groups: training-based [9]–[13] and zero-shot [14]. Trainingbased methods typically use Imitation Learning [15] or Reinforcement Learning [16] to learn the observation-toaction mapping [11]–[13], and developing vision–language aligned embeddings [9], [10] either in a single end-to-end model [10]–[12] or a modular pipeline [17], [18]. Zeroshot methods instead combine classical frontier-based exploration [19] with the priors provided by large language [20], [21] or vision–language models [14], [22] to score frontiers and guide exploration without task-specific training.

IGN takes a step further by requiring the agent to locate a specific instance rather than any category member, which reflects another facet of real-world needs, where users often care about particular or personalized targets. [1], [23]. Recent instance goal navigation benchmarks [1], [23] show that disambiguation is a central challenge, and that learned exploration policies often struggle to backtrack and recover after moving in the wrong direction. To address this problem, IGN [1] builds an instance-level graph whose nodes encode color, material, and location features, and PSL [24] proposes Prioritized Semantic Learning for IGN, which enhances semantic understanding to better identify the target. Despite these advances, real environments often contain many lookalike instances, and textual descriptions alone frequently lack the specificity needed to differentiate them [5]. Unlike prior work that treats the agent as a passive recipient of instancelevel information, we enable the agent to proactively propose dialogs to refine the task specification and obtain targeted guidance, thus improving exploration efficiency.

B. Interactive Embodied Robotics

We study agent–human interaction where the human serves as a task assistant, providing detailed target descriptions or action suggestions.

Methods Early work measures uncertainty to decide whether the agent should act or ask for help; upon receiving a query, the human or a simulated oracle returns the next best action or a shortest path to the target [25], [26]. Another line of work returns images of candidate objects and asks the human to confirm the target [27]. These designs restrict interaction to a single mode, either asking for the next action or confirming the target. With the emergence of interactive platforms [28] and datasets [3], agents can ask free-form, natural-language questions. RMM [29] simulates oracle answers and estimates their effect on progress to learn which questions to ask. KNOWNO [30] uses conformal prediction to decide when to seek help from a language-model planner. More recently, AIUTA [5] allows an agent to identify the target instance through open-ended, template-free dialog. Despite these advances, existing methods underemphasize efficient exploration in instance-goal navigation; guidance is typically limited to target descriptions [5], [27] or a few rounds of local actions [25], [26], [29], [30]. Our approach supports both disambiguation questions and exploration-path questions, enabling guidance of long-term navigation.

Benchmarks We focus on natural-language interactive datasets. CVDN [3] collected human–human dialogs but lacks an oracle, making it unsuitable for online evaluation. DialFRED [4] and CoIN [5] introduce oracles for online evaluation: DialFRED’s oracle provides best actions in small, single-room scenes, while CoIN’s oracle only describes the target instance. As a result, these settings make it difficult to evaluate exploration and querying in the more practical house-scale environments. In addition, existing training datasets [3]–[5], are not large enough to support learning agents that both explore actively and converse effectively. We introduce VL-LN Bench, a large-scale dialog-augmented training set with a house-level oracle that answers both exploration and disambiguation queries, enabling efficient policy training and comprehensive evaluation. A detailed comparison of existing benchmarks is shown in Table I.

III. VL-LN BENCHMARK

This section first defines the Interactive Instance Goal Navigation (IIGN) task. We then present the VL-LN Benchmark, detailing the agent–oracle interaction in dialog-augmented trajectory generation and evaluation, the dataset construction for IIGN and dataset statistics, and the metrics employed to assess dialog quality.

A. Task Definition

The Interactive Instance Goal Navigation (IIGN) challenges a dialog-enabled embodied agent to locate a specific instance in an unfamiliar environment. It involves two active roles: an agent and an oracle. For each episode, the agent is randomly placed in an unknown environment [31] and given an ambiguous instruction providing only the target category (e.g., “Search for the chair”). At each step, the agent receives a visual observation ot and odometry lt from the environment, and chooses to either move from at ∈ A, where

A = {FORWARD(0.25m), LOOK-DOWN, LOOK-UP, ASK,

TURN-LEFT(30◦), TURN-RIGHT(30◦), STOP}

or query the oracle for guidance via free-form, open-ended natural-language interactions. The oracle is assumed to know all the information about the environment, such as: (1) the detailed attributes and location of the target instance, and (2) the global structure of the environment. The objective of the agent in IIGN is to locate the specified instance with minimal steps under a limited number of interactions.

- B. Agent–Oracle Interaction

In IIGN, the agent and oracle interact through templatefree, open-ended natural-language dialogs. The agent may ask any question, and the oracle responds based on privileged access to environment information. To illustrate the scope of possible queries, consider the task of retrieving a friend’s computer in a large house, given only the instruction “Search for my computer.” To act efficiently, the agent may first request Attribute information to resolve ambiguity among same-category items (e.g., “What color is it?”). During exploration, the agent may then seek Route guidance to avoid blind searching (e.g., “Where should I go?”). Finally, upon encountering a candidate, the agent may ask a Disambiguation question to confirm correctness (e.g., “Is this the right computer?”). A positive confirmation therefore indicates successful task completion.

We implement the oracle with GPT-4o and a set of deterministic rules to answer these questions. Upon receiving a query, the oracle first classifies it into one of the three types above. For Attribute questions, the oracle supplies GPT4o with instance-level metadata and returns the generated answer. For Route questions, the oracle converts the shortest path to natural-language guidance via the following rulebased procedure: leftmargin=*, itemsep=0pt, topsep=2pt

- 1) Compute a shortest path in Habitat-Sim from the agent’s current pose to the target; retain the first 4m.
- 2) Simplify the remaining route into waypoints at highcurvature turns or room transitions (e.g., living room

→ bedroom).

- 3) Anchor each waypoint to the nearest salient object to localize instructions (e.g., “when you reach the brown table, turn right”).
- 4) Render the sequence into natural language using predefined conjunctions.

For Disambiguation questions, the oracle answers “yes” if the target is centered in the current view and within 3m of the agent; otherwise it answers “no.”

- C. Dataset

As shown in Fig. 2, the training dataset with dialogaugmented trajectories is constructed through a carefully designed three-step pipeline that automatically scales to large numbers of dialog trajectories based on the required scene annotations.

1) Scenes Metadata Processing: We process the MP3D scene meta-annotations using the hierarchical labels from MMScan [6], which provide fine-grained descriptions at both object and region levels. Object-level annotations include spatial properties (geometry, pose) and attributes (category, appearance, material, state, functional use), while roomlevel annotations include each region’s function (e.g., dining

TABLE I COMPARISON WITH EXISTING INTERACTIVE INSTANCE GOAL NAVIGATION BENCHMARKS.

Training Dataset Attributes Instruction Dialog Support Dataset #Trajectories

#Dialogs (Q/A)

Scene Scale

Episode Length

Annotation Source

F. / P. Oracle Attr. Disamb. Route

CVDN [3] 7,000 2,050 house 25 (steps) Human ✗ / ✓ ✗ - - DialFRED [4] - 53,000 room - Human ✗ / ✓ ✓ ✓ ✗ ✓(dir.) CoIN [5] - - - - - ✗ / ✓ ✓ ✓ ✓ ✗ VL-LN (Ours) 41,891 95,559 house 22.5 (m) GPT+Rules ✓ / ✓ ✓ ✓ ✓ ✓(4m traj.)

Columns: F. (full instruction that uniquely identifies the target instance), P. (partial/ambiguous instruction), Attr. (attribute questions), Disamb. (disambiguation questions), dir. (oracle provides a direction from the agent’s current pose to the target instance), and 4 m traj. (oracle provides a detailed trajectory for the next 4m in natural language). VL-LN Bench offers larger scale and richer supervision—both in instruction and dialog types—supporting training for long-horizon interactive instance-goal navigation and more comprehensive evaluation.

[Figure 2]

- Fig. 2. Automatic pipeline for collecting dialog-augmented trajectories. We first aggregate room-level instance attributes into unified house-level annotations. We then pair each target instance with a start point to generate episodes. Finally, we collect dialog-augmented trajectories using a frontierbased exploration (FBE) agent that, with 90% probability, selects the frontier nearest to the previously chosen frontier and, with 10%, selects the frontier closest to the target (the “best frontier”). The attribute question is asked at the beginning of the trajectory, and the attribute is randomly chosen from one of the given attributes shown in the figure. The route question is asked when the best frontier is chosen. And the disambiguation question is proposed when an instance with the same category as the target is detected, the criterion of “detected” is that the GT semantic appears in the center of the image, and the instance is within 3 meters of the agent. The number following “#” indicates the corresponding number of cases. The ellipses indicate the potential inclusion of additional disambiguation, route, or attribute questions.

room, study, bathroom) and the objects contained in that region. Based on these annotations, we reconstruct houselevel meta-annotations into an instance dictionary and a region dictionary for each scene. The key difference from MMScan is that we merge room-level annotations into a single house-level index that covers all rooms in the scene. Moreover, we build a spatial-relation graph using Sr3D [32]. Each node corresponds to an instance, and edges connect an instance to nearby instances within 1m. Together with the house-level dictionaries, this graph provides robust relational

cues that help disambiguate instances even when they appear visually identical.

2) Episodes Generation: Each episode is defined by three core elements: an initial agent pose, a navigation instruction, and a set of target-instance viewpoints. The initial agent poses include starting poses from R2R-CE (72 scenes) [33] and our manually annotated 18 scenes. For the self-annotated scenes, we randomly sample navigable points, snap the agent to the selected location, and accept it as an initial pose only after verifying that the location is valid (on the navmesh,

collision-free, and within scene bounds). For every instance, we provide two instruction variants: a partial, categoryonly instruction mentioned above and a full description that uniquely identifies the target among all instances in the scene (e.g., “Locate the deep grey chair with black backrest, standing upright on a wooden floor near a computer and a tv in the bedroom.”). To create the full description, we leverage the house-level dictionaries and the spatial-relation graph to select discriminative attributes and relations for the instance, and then prompt GPT-4o [34] to generate a naturallanguage instruction. The full instruction supports a noninteractive instance goal navigation setting (i.e. IGN), while the partial instruction is used in IIGN. We generate a set of viewpoints for each instance, which serve as expected “Stop” locations. Based on empirically reasonable viewing distances, we expand each instance’s 3D bounding box by 0.6m in all directions and mark all navigable points within the expanded region as instance viewpoints. The episode is considered to be successful if the agent stops within 0.25 m of any viewpoint of the instance. Combining an initial agent pose with the instance’s instruction and its viewpoints defines an episode.

3) Training Trajectories Collection: The agent is equipped with an RGB–D camera and odometry for perception. Onboard, a frontier-based exploration (FBE) strategy and a ground-truth object detector operate continuously. The exploration policy selects the next frontier with a 90% probability as the one closest to the previously chosen frontier, and with the remaining 10% probability as the frontier nearest to the target location. Meanwhile, the ground-truth detector processes incoming images; once the target is identified, the agent navigates to it and terminates the episode. To encourage proactive assistance, we define question triggers during exploration. At the start of each episode, the agent asks a random Attribute question about the target (color, texture, material, shape, or placement). A Route question is triggered when the frontier that leads toward the target is selected. A Disambiguation question is asked when objects of the same category as the target are in view. For each question type, we predefine multiple semantically consistent question templates to enhance dialog diversity. All perceived sensor data and the full dialog history are logged to construct the VL–LN training dataset.

D. Statistics

Our dataset covers 112 object categories that can serve as navigation targets (excluding structural elements such as walls and ceilings), comprising 20476 object instances across 90 annotated MP3D scenes. For each scene, we sample a set of navigable start poses (mean 42 per scene). Pairing these starts with the annotated instances yields 333,319 episodes. Following the VLN-CE partitioning [33], we allocate 61/15/14 scenes to train/val/test, resulting in 246,433 training episodes, 86,386 validation episodes, and 500 test episodes. Shown in Tab. I, we collect 41,891 trajectories spanning all scenes and categories of IIGN, each coupling navigation with question–answer interactions. Fig. 3 summarizes the trajectory statistics. In addition, we collect 5,087

[Figure 3]

[Figure 4]

[Figure 5]

(a) (b)

fireplace

toilet

curtain

[Figure 6]

tv

[Figure 7]

cabinet

...

table

chair

Attr. 38.4%

Disamb. 38.5%

book

pillow

Route 23.1%

object

plant

picture

couch

(c) (d)

Fig. 3. Trajectory statistics of the VL–LN dataset. (a–b) Frequency histograms of per-episode path steps, and dialog turns; (c) frequency histogram of per-turn dialog length (tokens). Black lines denote smoothed density fits. (d) Nested donut of dialog data. Outer ring: target-instance category proportions; inner disk: question-type proportions (Attribute, Disambiguation, and Route question).

IGN and 23,774 ObjectNav trajectories generated by an FBE agent for the experiment.

E. Metric

In addition to the standard navigation related metrics—Success Rate (SR), Success Rate weighted by Path Length (SPL), Oracle Success Rate (OS), and Navigation Error (NE), we introduce the Mean Success Progress (MSP) metric to specifically evaluate dialog utility. Given a maximum dialog allowance of n turns, we first compute the baseline success rate without dialog, denoted as s0. Then measure the success rate under increasing dialog budgets, yielding s1,...,sn. For each budget, we calculate the success improvement relative to the baseline, i.e., (si − s0),0 < i ≤ n. The MSP score is defined as the mean of these improvements across all dialog budgets:

1 n

MSP =

n

(Si − S0). (1)

i=1

This metric captures two complementary aspects of dialog utility. First, it measures effectiveness by quantifying the average gains in navigation success attributable to dialog. Second, it reflects efficiency, since larger improvements achieved with fewer dialog turns increase the average score, whereas marginal improvements spread across many turns lower it. Consequently, MSP provides a balanced evaluation of both the usefulness and economy of dialog in enhancing task performance. In this paper, we set n = 5.

IV. EXPERIMENTS

In this section, we evaluate IIGN and IGN using both zeroshot and trained baselines, analyzing why instance navigation is inherently difficult and how dialog contributes to reducing

exploration failures and disambiguation errors, while highlighting remaining challenges in IIGN and providing some interesting findings.

A. Experimental Setup

Evaluated task In addition to IIGN, we benchmark Instance goal Navigation (IGN) without dialog, since most prior methods lack dialog capabilities. To enable the agent to identify the target instance, we provide the full instruction in the IGN setting, which is a complete and unambiguous description that uniquely specifies the target.

Baselines We evaluate five baselines: two zero-shot and three training-based methods. The zero-shot baselines are (i) a greedy frontier-based exploration (FBE) agent that repeatedly selects the nearest frontier and uses an openvocabulary detector built on Grounded SAM 2 [35] to detect the target instance, and (ii) VLFM [14], used as the released version. The learning-based baselines (VLLN-O, VLLNI, VLLN-D) are initialized from Qwen2.5-VL-7B-Instruct and trained following the InternVLA-N1 [36] procedure but with different data mixtures. All three include the VLN data from InternVLA-N1. VLLN-O additionally uses our ObjectNav data (23,774 trajectories), VLLN-I and VLLN-D further incorporate the filtered IGN data (11,661 trajectories): VLLN-I trained without dialog, whereas VLLN-D trained with dialogs.

Implementation Details The three learned baselines were trained on a 64×NVIDIA A800 GPU cluster. Each run took 50–59 hours (approximately 3200–3776 GPU-hours).

TABLE II RESULTS OF THE VL-LN BENCHMARKS. Task Method SR↑ SPL↑ OS↑ NE↓ MSP↑

FBE 8.4 4.74 25.2 11.84 – VLFM 10.2 6.42 32.4 11.17 – VLLN-O 14.8 10.36 47.0 8.91 – VLLN-I 14.2 8.18 47.8 9.54 – VLLN-D 20.2 13.07 56.8 8.84 2.76

IIGN

FBE 7.4 4.45 33.4 11.78 – VLFM 12.6 7.68 35.4 10.85 – VLLN-O 5.6 4.24 25.2 10.76 – VLLN-I 22.4 13.43 60.4 8.16 – VLLN-D* 21.8 14.33 54.8 8.56 VLLN-D 25.0 15.59 58.8 7.99 2.16

IGN

“—”: The model does not support dialog, hence MSP is not reported. “*”: The agent may ask a question, but the NPC does not respond.

TABLE III FAILURE COUNTS BY TYPE.

Detection

Method Task

Expl. ST WD Ambig.

VLLN-I 151 (35.2%) 159 (37.1%) 89 (20.7%) 30 (7.0%) VLLN-D 146 (36.6%) 145 (36.3%) 71 (17.8%) 37 (9.3%)

IIGN

VLLN-I 127 (32.7%) 143 (36.9%) 84 (21.6%) 34 (8.8%) VLLN-D 150 (40.0%) 124 (33.1%) 46 (12.3%) 55 (14.7%)

IGN

WD = Wrong Detection; Ambig. = Ambiguity; Expl. = Exploration Fail; ST = Stop Fail. Counts are computed over 500 test episodes; percentages in parentheses are the row-wise shares of total failures (sum to 100% within each row).

B. Result

Why Instance Goal Navigation is Hard. Tab. II shows that, even when trained on instance-goal data, performance remains far below prior goal-oriented benchmarks (e.g., VLLN-O: 59.3% SR on ObjectNav). We attribute this gap to two factors. First, long-horizon exploration: in ObjectNav, reaching any instance of a category (e.g., any of the seven “Chair” candidates in Fig. 1) suffices, whereas instance-goal navigation requires locating a specific instance, often necessitating substantially longer exploration. Second, agents often struggle to detect the target instance because attribute–image alignment is challenging; they may overlook or misalign attributes, leading to stops at same-category distractors on the true target.

Dialog Helps. Across both IIGN and IGN, VLLN-D achieves the best results, demonstrating the value of proactive querying while leaving room for further improvement. Dialog in the ambiguity setting (IIGN) is more effective than in the disambiguation setting (IGN) (MSP: 2.76 vs. 2.16). To better analyze the role of dialog during instance goal navigation, we categorize failures into three types: Exploration Fail, Detection Fail, and Stop Fail. Exploration Fail occurs when the agent keeps acting until the maximum number of steps without ever entering the oracle-success radius (OS = 0). Unlike prior goal-oriented navigation settings, we further split Detection Fail into Wrong Detection and Ambiguity: the former happens when the agent reaches the correct region (OS = 1) but fails to recognize the target instance, while the latter occurs when it stops at an instance of the same category as the target. Stop Fail denotes cases where the agent stops near the target (< 1m) but remains beyond the stop threshold (> 0.25m). As shown in Tab. III, when the agent is allowed to query an NPC, exploration failures decrease substantially: from 89 to 71 in IIGN and from 84 to 46 in IGN. This demonstrates that the agent can leverage information obtained through dialog to improve exploration efficiency. Notably, as shown in the supplementary video, the agent is able to ground naturallanguage short-route guidance into executable actions. We attribute it to co-training with VLN data, which, although different from goal-oriented navigation, teaches the model to follow natural-language route descriptions.

Moreover, dialog helps resolve ambiguity. In both IIGN and IGN settings, ambiguity-related failures decrease once the dialog is enabled. In IGN, the reduction from 143 to 124 primarily stems from disambiguation questions that guide the agent away from incorrect instances, as the attributes necessary to verify the target instance are already available, rendering additional attribute-related queries less informative. For IIGN, both attribute and disambiguation questions contribute to eliminating task ambiguity, make the ambiguity failures drop from 159 to 145 .

Key Challenges and Directions. To better understand IIGN’s challenges, we additionally evaluate four interaction settings on 100 episodes randomly sampled from the test set: Human - Human, Human - GPT, VLLN-D - Human, and VLLN-D - GPT. Results are reported in Tab. IV. First, we argue that image–attribute alignment is the primary bottleneck

in both the IGN and IIGN tasks. Even with full instructions and dialog capability, the success rate remains 25% (shown in Tab. II), and 73% of failures (shown in Tab. III) are due to detection errors. A promising direction is to train with hard negatives such as same-category objects with different attributes, which promotes instance-level discrimination and stronger attribute grounding. Zero-shot methods face the same challenge. Their performance hinges on detectors that can align detailed instance descriptions with visual evidence.

history to select maximally informative questions that either guide exploration or shrink the candidate set, placing stronger demands on grounding, planning, and reasoning.

C. Additional Insights

Reliability of evaluation and NPC The benchmark provides a reliable testbed for IIGN, as humans achieve high success rates. The GPT implemented NPC is also dependable, with the Human–GPT setting attaining performance comparable to Human–Human (91% vs. 93%), though at the cost of more dialog turns (9.72 vs. 2.04 on average). This discrepancy arises for two reasons: some queries fall outside the GPT’s knowledge or response schema, and users interacting with AI systems often seek additional confirmation (e.g., asking a disambiguation question even after the target has been identified). Although not flawless, the GPT implemented NPC is sufficient for evaluating IIGN, as performance in the VLLN-D – Human and VLLN-D – GPT settings is similar (16% vs. 17%).

TABLE IV CROSS-ROLE EVALUATION.

Navigator–NPC SR↑ SPL↑ OS↑ NE↓ Avg. turns

Human – Human 93 57.30 95 0.31 2.04 Human – GPT 91 49.88 94 0.69 9.72 VLLN-D – Human 16 12.63 55 7.02 1.54 VLLN-D – GPT 17 12.05 55 8.04 1.66

Why do humans fail? As shown in Fig. 4, two factors dominate in the Human-Human setting: (i) Referential ambiguity, where the expression (e.g., “the stairs”) does not uniquely identify the intended instance when multiple same-category objects are visible, causing mismatched grounding; and (ii) Partial observability, where the target is partially occluded so participant (Navigator) assumes a single candidate and commit without disambiguation, leading to a wrong choice. In the Human-GPT setting, we observe an additional failure mode, Exploration failure, where the human (Navigator) fails to complete the task before the step limit is reached. This typically occurs when the target is difficult to find or the scene is large and complex.

[Figure 8]

[Figure 9]

Search for the stairs Search for the table

| |
|---|

| |
|---|

Human (Navigator): Is this the target? Oracle: Yes, you are aligned correctly.

Human (Navigator): Are the stairs in front of me?

Human (Oracle): Yes, go straight ahead.

[Figure 10]

[Figure 11]

| |
|---|

| |
|---|

| |
|---|

Human (Navigator): STOP Human (Navigator): STOP

(a) Referential Ambiguity

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Search for the chair

Sensitivity to dialog turn budget To assess the effect of the dialog turn budget, we evaluate VLLN-D under varying turn limits.

Human (Navigator): Which way should I go? Human (Oracle): Turn around. Human (Navigator) keeps exploring

TABLE V EVALUATION UNDER DIFFERENT DIALOG-TURN LIMITS.

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

| |
|---|

| |
|---|

| |
|---|

Target Stop Here

Turn limit SR↑ SPL↑ OS↑ NE↓ Avg. turns

- 0 15.4 9.86 55.2 9.17 0.00
- 1 15.8 9.53 52.6 9.13 1.00
- 2 18.6 12.55 54.6 8.90 1.63
- 3 18.0 12.22 54.8 8.71 1.73
- 4 18.2 12.69 57.4 8.67 1.74 ∞ (5) 20.2 13.07 56.8 8.84 1.76

(c) Exploration Failure

(b) Partial Observability

Fig. 4. Failure cases. Green curves denote the geodesic shortest paths; blue curves are the navigator’s exploration trajectories; red shaded regions indicate the success zone around the target. (a) Referential ambiguity: within the same view, the navigator and the NPC refer to different instances, causing the navigator to stop at a wrong instance. (b) Partial observability: the navigator only observes a single candidate in the room and stops without disambiguating. (c) Exploration failure: despite continued interaction, the human navigator never finds the target.

As shown in Tab. V, SR and SPL generally increase as the dialog-turn budget grows, indicating the benefit of dialog. The largest gain occurs when the limit increases from 1 to 2 turns. With only one turn, the agent usually spends it on an initial attribute question and has no chance to ask the more informative follow-ups. The agent is also not prone to overquerying, which makes the approach practical: even with higher budgets, it asks fewer than two questions on average (1.63–1.76). This pattern is consistent with our training data. The average number of dialog turns in the training set is about 1 to 2, as shown in Fig. 3 (b). We hypothesize that both query efficiency and dialog frequency correlate with this training distribution.

Secondly, the agent’s questioning ability remains limited. Reliably disambiguating a target instance from samecategory distractors through dialog is still difficult. VLLND performs worse on IIGN (20.2%) than VLLN-D* on IGN (21.8%), indicating that dialog offers less guidance than a full instruction. We also find that a gap persists between agent-driven and human-driven proactive interaction. Human-Human reaches 93% SR with only about two questions on average, which suggests that a small number of wellchosen queries is sufficient to complete the task. Achieving such behavior requires the agent to exploit its observation

V. CONCLUSION

This paper investigates the Interactive Instance Goal Navigation (IIGN) task and introduces the VL-LN benchmark, which includes a long-horizon dataset comprising ∼ 41k automatically collected dialogue-augmented trajectories for training, along with an evaluation protocol involving an oracle for agent assessment. This benchmark enables agents to explore long-horizon environments and engage in meaningful dialogs. Our experiments demonstrate that incorporating active dialog significantly improves performance in both the IIGN and IGN tasks, achieving state-of-the-art results. Additionally, we highlight key challenges in the IIGN task and provide some interesting findings from our experiments.

REFERENCES

- [1] W. Li, X. Song, Y. Bai, S. Zhang, and S. Jiang, “Ion: Instancelevel object navigation,” in Proceedings of the 29th ACM international conference on multimedia, 2021, pp. 4343–4352.
- [2] D. Batra, A. Gokaslan, A. Kembhavi, O. Maksymets, R. Mottaghi, M. Savva, A. Toshev, and E. Wijmans, “Objectnav revisited: On evaluation of embodied agents navigating to objects,” 2020. [Online]. Available: https://arxiv.org/abs/2006.13171
- [3] J. Thomason, M. Murray, M. Cakmak, and L. Zettlemoyer, “Visionand-dialog navigation,” in Conference on Robot Learning. PMLR, 2020, pp. 394–406.
- [4] X. Gao, Q. Gao, R. Gong, K. Lin, G. Thattai, and G. S. Sukhatme, “Dialfred: Dialogue-enabled agents for embodied instruction following,” IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 10049– 10056, 2022.
- [5] F. Taioli, E. Zorzi, G. Franchi, A. Castellini, A. Farinelli, M. Cristani, and Y. Wang, “Collaborative instance object navigation: Leveraging uncertainty-awareness to minimize human-agent dialogues,” arXiv preprint arXiv:2412.01250, 2024.
- [6] R. Lyu, J. Lin, T. Wang, S. Yang, X. Mao, Y. Chen, R. Xu, H. Huang, C. Zhu, D. Lin, and J. Pang, “Mmscan: A multi-modal 3d scene dataset with hierarchical grounded language annotations,”

2025. [Online]. Available: https://arxiv.org/abs/2406.09401

- [7] I.-T. Ieong and H. Tang, “Multimodal perception for goal-oriented navigation: A survey,” arXiv preprint arXiv:2504.15643, 2025.
- [8] P. Anderson, A. Chang, D. S. Chaplot, A. Dosovitskiy, S. Gupta, V. Koltun, J. Kosecka, J. Malik, R. Mottaghi, M. Savva et al., “On evaluation of embodied navigation agents,” arXiv preprint arXiv:1807.06757, 2018.
- [9] T. Campari, P. Eccher, L. Serafini, and L. Ballan, “Exploiting scenespecific features for object goal navigation,” in European Conference on Computer Vision. Springer, 2020, pp. 406–421.
- [10] K. Yadav, R. Ramrakhya, A. Majumdar, V.-P. Berges, S. Kuhar, D. Batra, A. Baevski, and O. Maksymets, “Offline visual representation learning for embodied navigation,” in Workshop on Reincarnating Reinforcement Learning at ICLR 2023, 2023.
- [11] J. Ye, D. Batra, A. Das, and E. Wijmans, “Auxiliary tasks and exploration enable objectgoal navigation,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 16117–16126.
- [12] W. Xie, H. Jiang, S. Gu, and J. Xie, “Implicit obstacle map-driven indoor navigation model for robust obstacle avoidance,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 6785–6793.
- [13] H. Yin, X. Xu, L. Zhao, Z. Wang, J. Zhou, and J. Lu, “Unigoal: Towards universal zero-shot goal-oriented navigation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 19057–19066.
- [14] N. Yokoyama, S. Ha, D. Batra, J. Wang, and B. Bucher, “Vlfm: Visionlanguage frontier maps for zero-shot semantic navigation,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 42–48.
- [15] M. Zare, P. M. Kebria, A. Khosravi, and S. Nahavandi, “A survey of imitation learning: Algorithms, recent developments, and challenges,” IEEE Transactions on Cybernetics, 2024.
- [16] L. P. Kaelbling, M. L. Littman, and A. W. Moore, “Reinforcement learning: A survey,” Journal of artificial intelligence research, vol. 4, pp. 237–285, 1996.

- [17] J. Chen, G. Li, S. Kumar, B. Ghanem, and F. Yu, “How to not train your dragon: Training-free embodied object goal navigation with semantic frontiers,” arXiv preprint arXiv:2305.16925, 2023.
- [18] W. Cai, H. Wang, P. Liu, M. Wu, Z. Qian, and H. Dong, “Moddn: A coarse-to-fine attribute-based exploration agent for multiobject demand-driven navigation,” Advances in Neural Information Processing Systems, vol. 37, pp. 64176–64214, 2024.
- [19] B. Yamauchi, “A frontier-based approach for autonomous exploration,” in Proceedings 1997 IEEE International Symposium on Computational Intelligence in Robotics and Automation CIRA’97.’Towards New Computational Principles for Robotics and Automation’. IEEE, 1997, pp. 146–151.
- [20] B. Yu, H. Kasaei, and M. Cao, “L3mvn: Leveraging large language models for visual target navigation,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 3554–3560.
- [21] W. Cai, S. Huang, G. Cheng, Y. Long, P. Gao, C. Sun, and H. Dong, “Bridging zero-shot object navigation and foundation models through pixel-guided navigation skill,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 5228–5234.
- [22] A. Majumdar, G. Aggarwal, B. Devnani, J. Hoffman, and D. Batra, “Zson: Zero-shot object-goal navigation using multimodal goal embeddings,” Advances in Neural Information Processing Systems, vol. 35, pp. 32340–32352, 2022.
- [23] L. Barsellotti, R. Bigazzi, M. Cornia, L. Baraldi, and R. Cucchiara, “Personalized instance-based navigation toward user-specific objects in realistic environments,” Advances in Neural Information Processing Systems, vol. 37, pp. 11228–11250, 2024.
- [24] X. Sun, L. Liu, H. Zhi, R. Qiu, and J. Liang, “Prioritized semantic learning for zero-shot instance navigation,” in European Conference on Computer Vision. Springer, 2024, pp. 161–178.
- [25] K. Nguyen, D. Dey, C. Brockett, and B. Dolan, “Vision-based navigation with language-based assistance via imitation learning with indirect intervention,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 12527–12537.
- [26] T.-C. Chi, M. Shen, M. Eric, S. Kim, and D. Hakkani-Tur, “Just ask: An interactive learning framework for vision and language navigation,” in Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 03, 2020, pp. 2459–2466.
- [27] A. Majumdar, F. Xia, D. Batra, L. Guibas et al., “Findthis: Languagedriven object disambiguation in indoor environments,” in 7th Annual Conference on Robot Learning, 2023.
- [28] Q. Gao, G. Thattai, S. Shakiah, X. Gao, S. Pansare, V. Sharma, G. Sukhatme, H. Shi, B. Yang, D. Zhang et al., “Alexa arena: A user-centric interactive platform for embodied ai,” Advances in Neural Information Processing Systems, vol. 36, pp. 19170–19194, 2023.
- [29] H. R. Roman, Y. Bisk, J. Thomason, A. Celikyilmaz, and J. Gao, “Rmm: A recursive mental model for dialog navigation,” arXiv preprint arXiv:2005.00728, 2020.
- [30] A. Z. Ren, A. Dixit, A. Bodrova, S. Singh, S. Tu, N. Brown, P. Xu, L. Takayama, F. Xia, J. Varley et al., “Robots that ask for help: Uncertainty alignment for large language model planners,” arXiv preprint arXiv:2307.01928, 2023.
- [31] A. Szot, A. Clegg, E. Undersander, E. Wijmans, Y. Zhao, J. Turner, N. Maestre, M. Mukadam, D. S. Chaplot, O. Maksymets et al., “Habitat 2.0: Training home assistants to rearrange their habitat,” Advances in neural information processing systems, vol. 34, pp. 251– 266, 2021.
- [32] P. Achlioptas, A. Abdelreheem, F. Xia, M. Elhoseiny, and L. Guibas, “Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes,” in European conference on computer vision. Springer, 2020, pp. 422–440.
- [33] J. Krantz, E. Wijmans, A. Majumdar, D. Batra, and S. Lee, “Beyond the nav-graph: Vision-and-language navigation in continuous environments,” in European Conference on Computer Vision. Springer, 2020, pp. 104–120.
- [34] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford et al., “Gpt-4o system card,” arXiv preprint arXiv:2410.21276, 2024.
- [35] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. R¨adle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Doll´ar, and C. Feichtenhofer, “Sam 2: Segment anything in images and videos,” 2024. [Online]. Available: https://arxiv.org/abs/2408.00714
- [36] I.-N. Team, “InternVLA-N1: An open dual-system navigation foundation model with learned latent plans,” 2025.

