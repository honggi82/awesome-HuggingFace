# arXiv:2410.01273v3[cs.RO]8Aug2025

## CANVAS: Commonsense-Aware Navigation System for Intuitive Human-Robot Interaction

Suhwan Choi†1, Yongjun Cho†1, Minchan Kim†1, Jaeyoon Jung†1, Myunchul Joe1, Yubeen Park1, Minseo Kim2, Sungwoong Kim2, Sungjae Lee2, Hwiseong Park1, Jiwan Chung2, Youngjae Yu2 †Equal Contribution. 1MAUM.AI, 2Yonsei University. {milkclouds, cyjun0304, alan, jyjung}@maum.ai

[Figure 1]

Fig. 1: Humans often give abstract navigation directions using simple instruction, relying on the recipient’s commonsense to bridge the gaps. With CANVAS, robots can interpret and act on these directions like humans do, fostering a shared understanding of the environment. It shows how robots can use commonsense to translate vague human instructions into concrete actions, navigating across diverse settings in our COMMAND dataset, which we plan to open-source as valuable resources for imitation learning in commonsense-aware navigation.

Abstract—Real-life robot navigation involves more than just reaching a destination; it requires optimizing movements while addressing scenario-specific goals. An intuitive way for humans to express these goals is through abstract cues like verbal commands or rough sketches. Such human guidance may lack details or be noisy. Nonetheless, we expect robots to navigate as intended. For robots to interpret and execute these abstract instructions in line with human expectations, they must share a common understanding of basic navigation concepts with humans. To this end, we introduce CANVAS, a novel framework that combines visual and linguistic instructions for commonsense-aware navigation. Its success is driven by imitation learning, enabling the robot to learn from human navigation behavior. We present COMMAND, a comprehensive dataset with human-annotated navigation results, spanning over 48 hours and 219 km, designed to train commonsense-aware navigation systems in simulated environments. Our experiments show that CANVAS outperforms the strong rule-based system ROS NavStack across all environments, demonstrating superior performance with noisy instructions. Notably, in the orchard environment, where ROS NavStack records a 0% total success rate, CANVAS achieves a total success rate of 67%. CANVAS also closely aligns with human demonstrations and commonsense constraints, even in unseen environments. Furthermore, real-world deployment of CANVAS showcases impressive Sim2Real transfer with a total success rate of 69%, highlighting the potential of learning from human demonstrations in simulated environments for real-world applications.

Videos, datasets, and models: worv-ai.github.io/canvas

I. INTRODUCTION

Real-life robot navigation scenarios involve addressing complex objectives that extend far beyond simply reaching a destination. For example, an agricultural spraying robot must maximize field coverage [1], while a package delivery robot must adhere to road lanes and use crosswalks when transitioning between sidewalks. [2], [3] In both cases, robots need to optimize their movements while responding to the specific requirements of the scenario.

Humans typically communicate these scenario-specific goals through high-level guidance, such as verbal commands [4]–[6], rough sketches of the desired route [7], [8], or a combination of both [9]. While such guidance outlines the robot’s overall objectives, it often lacks the specificity required for precise execution. To convert these abstract and imprecise instructions into actionable navigation plans, robots need commonsense knowledge. In the context of robotics, commonsense refers to the general understanding humans naturally use to make decisions, covering aspects such as human desires, physics, and causality [10]. Robots must leverage this knowledge to flexibly adjust their paths, ensuring their decisions align with human expectations by adhering to commonsense constraints posed by the environment and the user’s true intentions.

In response to these challenges, we introduce CANVAS

(Commonsense-Aware NaVigAtion System), a novel framework for integrating abstract human instructions into robot navigation. Our approach utilizes both visual and linguistic inputs, such as rough sketch trajectories on map images or textual descriptions. These multimodal instructions are processed by a vision-language model that generates incremental navigation targets. By leveraging the commonsense knowledge embedded in pre-trained vision-language models [11]–[15], robots can develop a versatile understanding of commonsense navigation dynamics. Quantifying successful navigation behaviors using rewards [16]–[18] is particularly difficult in commonsense-aware navigation. Therefore, we employ imitation learning, enabling the robot to comprehend user intent behind noisy and imprecise instructions from human demonstrations. [19]

Additionally, we introduce COMMAND (COMMonsenseAware Navigation Dataset), a comprehensive dataset designed to train commonsense-aware navigation robots. The dataset features three simulated environments with distinct characteristics (office, street, and orchard). To facilitate imitation learning for instruction-following tasks, we provide 3,343 fully human-annotated navigation results from the simulated environments. Notably, COMMAND offers 48 hours of driving data, which is nearly three times longer than GoStanford [20], covering 219 km and thereby enriching the dataset’s diversity and scope. Furthermore, we propose two metrics to evaluate the commonsense adherence of navigation algorithms: Trajectory Deviation Distance (TDD) and Instruction Violation Rate (IVR).

Our results show that CANVAS consistently outperforms ROS NavStack [21] across all environments with noisy sketch instructions. Particularly in the challenging orchard environment, CANVAS navigates effectively, while NavStack fails due to its reliance on rule-based algorithms [22]. CANVAS ’s trajectory closely mirrors human demonstrations with fewer commonsense constraint violations, indicating a better understanding of human expectations. Despite being trained only on simulated data, CANVAS also excels in real-world scenarios, demonstrating strong Sim2Real transfer capabilities.

Our contributions are threefold:

- 1) We introduce CANVAS, a novel framework that allows humans to easily communicate with robots using multimodal inputs, ensuring that robots effectively achieve navigation goals, even when human instructions are vague or noisy.
- 2) We introduce COMMAND, a dataset for training commonsense-aware navigation robots, featuring 48 hours of driving data over 219 kilometers, with fully human-annotated sketch instruction and navigation outcomes.
- 3) We present extensive experiments demonstrating that CANVAS outperforms ROS NavStack in success rate, collision rate, trajectory deviation distance, and instruction violation rate. To support further research, we are open-sourcing CANVAS and COMMAND for imitation learning in commonsense-aware robot navigation.

II. RELATED WORK

- A. Robot Navigation

Historically, robot navigation systems were largely rulebased [22]–[24], relying on a set of predefined rules, as seen in frameworks like the ROS NavStack [21]. Following the successful application of deep learning to robotics, more flexible neural navigation approaches have emerged. Visual navigation models, such as NoMaD [25], ViNT [26], and GNM [27], utilize images as goal representations. However, since their high-level planning heavily relies on the topological graph with first-person visual observations, they cannot handle unvisited locations and are sensitive to environmental changes. [28] Vision-language navigation integrates visual information from sensors with language instructions [5], [6], [29]. However, the ambiguous nature of language instructions poses limitations on controlling detailed navigation routes. [30] Recently, LIM2N [9] enabled users to control robots through natural language and sketch trajectories, combining high-level goals with precise motion paths for more intuitive interaction. However, the system’s demand for highly accurate and detailed instructions, coupled with its vulnerability to missing details or small mistakes, reduces its usability for non-expert users and limits its effectiveness in a broader range of real-world applications [4].

Our proposed method, CANVAS, differentiates itself by addressing the challenge of interpreting abstract or noisy human instructions. CANVAS converts visual and linguistic instructions into detailed navigation actions, utilizing commonsense knowledge to fill in the gaps. This integration of commonsense enables CANVAS to dynamically adapt its navigation strategies across diverse contexts, resulting in enhanced task execution compared to ROS NavStack.

| |NoMaD [25]|NaVid [6]|LIM2N [9]<br><br>|CANVAS|
|---|---|---|---|---|
|Instruction Misleading (III-A.3) Custom Dataset Environment Scenes|Image ✗ ✗ Real Indoor, Outdoor|Language ✗ ✓ Real Indoor|Sketch, Language ✗ ✓ Real, Simulation Indoor|Sketch, Language ✓ ✓ Real, Simulation Indoor, Outdoor|

TABLE I: Comparison between various robot navigation methods.

- B. Imitation Learning in Robotics

Imitation Learning (IL) enables agents to learn tasks by mimicking expert demonstrations, eliminating the need for predefined rules or reward functions typically required in Reinforcement Learning (RL) [31]. By directly leveraging expert behavior, IL has proven particularly advantageous in situations where designing a reward function is challenging or exploration involves potential risks [32]. As a result, there has been a growing interest in applying IL to robot navigation [33], [34]. A key challenge in IL is modeling multimodal action distributions [35]. One solution is to quantize actions into discrete tokens, simplifying the action space [35]–[39]. Autoregressive prediction of quantized actions effectively reduces the complexity of modeling diverse and feasible action sequences.

[Figure 2]

Fig. 2: Data collection pipeline for COMMAND dataset. (a) First, we create diverse navigation environments and extract maps. (b) Then, human annotators sketch routes on the maps based on the guidelines. (c) Finally, we use teleoperation to collect human demonstrations. Red line shows the roughly sketched routes, while the blue line shows the human-demonstrated trajectory. FD refers to Frechet distance.

CANVAS builds upon this idea by converting continuous waypoints into 128 discrete waypoint tokens using K-means clustering [40], [41]. This approach enhances the ability of robots to model multimodal action distributions, enabling robust navigation strategies that adapt to diverse human instructions and environmental variations.

III. DATASET AND TASK

An interactive robot navigation framework should fulfill two key objectives: first, humans should be able to communicate desired routes and requirements intuitively; second, the robot should accurately interpret and execute those instructions. However, achieving these goals can be challenging. Simplifying communication for humans often complicates it for robots because humans naturally assume the listener shares their commonsense knowledge. This commonsense allows humans to infer meaning even when instructions are incomplete or imprecise but also causes robots to struggle without explicit and precise input.

To address this challenge, we introduce COMMAND—a comprehensive experiment suite designed to assess whether robots can use commonsense understanding to transform noisy or abstract human instructions into the most desired trajectory. COMMAND was collected in three distinct environments: office, street, and orchard, using NVIDIA Isaac Sim [42]. The data consists of high-quality sketch instruction labels and teleoperation data, all by human experts. COMMAND contains 48 hours of driving data covering a distance of 219 kilometers. In this section, we describe the dataset curation process and its resulting task definition.

A. Dataset

- A datapoint in COMMAND includes a canvas map, sketch

and language instructions, commonsense constraints, and teleoperation records. The canvas maps, linked to each environment, provide occupancy information and serve as a human-robot communication interface. We assume humans provide instructions in two modalities: sketch instructions S that consist of hand-drawn trajectories on maps for the robot to follow, and language instructions L that outline goals and related requirements. When collecting sketch instructions

S, we introduce both Precise and Misleading conditions to gather training data that enables our model to handle noisy sketch instructions more robustly. In addition to the instructions, we define a set of commonsense constraints C, which are derived from the navigation environment E and the language instructions L. These constraints help evaluate whether the robot exhibits appropriate navigation behaviors, such as using crosswalks. We also include human teleoperation records to optimize and evaluate the robot behavior against human actions. An overview of our data collection pipeline is illustrated in Figure 2, and the statistics are provided in Table II. We detail each step below.

- 1) Environments: COMMAND features three simulated

environments, each tailored to specific scenarios. The simulated environments include tasks such as coffee delivery in an office, package delivery on the street, and agricultural spraying in an orchard. An expert designer creates each simulated environment using Unreal Engine [43].

- 2) Canvas Map Extraction: The simulated environments

are subsequently exported to NVIDIA Isaac Sim, where occupancy maps are extracted programmatically.

- 3) Sketch Instruction Labeling: The data workers draw

sketch trajectories on the canvas maps following sketch guidelines manually crafted by the authors. When the Precise condition is included, the workers trace the most efficient route, closely following the guidelines. In contrast, when the Misleading condition is included, data workers are instructed to deliberately introduce noise by drawing trajectories that pass through walls or objects. All sketch instructions undergo manual inspection to ensure quality.

- 4) Teleoperation: The data workers are then provided

both the sketch and language instructions to teleoperate the virtual robot in NVIDIA Isaac Sim. We record front view images, canvas map images, and the human-demonstrated trajectory. After collecting the teleoperation data, we adopt the Frechet distance (FD) [44] to measure the discrepancy between the sketch trajectory and the human-demonstrated trajectory. This metric, which indicates noise in the sketch instructions, tends to be higher in the Precise condition and lower in the Misleading condition.

|Split<br><br>|Train|Test|
|---|---|---|
|Environment|Office<br><br>Street<br><br>Orchard Road Sidewalk|Office<br><br>Street<br><br>Orchard Gallery<br><br>Real Road Sidewalk Office<br><br>|
|Count Avg. Time Avg. Distance Avg. FD (P / M) % of Misleading|2,263 403 410 267<br><br>31s 57s 103s 172s 32.8m 80.4m 150.0m 191.6m 1.05 / 1.77 0.97 / 2.02 3.03 / 3.50 1.91 / 3.76 31% 51% 51% 40%|10 20 10 10 10 10 39s 56s 127s 182s 76s 30s 38.7m 91.0m 152.0m 232.88m 48.8m 16.0m 0.77 / 1.62 1.28 / 1.65 1.32 / 2.33 1.51 / 2.27 0.68 / 1.36 1.44 / 1.63 50% 50% 50% 50% 50% 50%|

TABLE II: Statistics for the train and test set. The train dataset includes 48 hours of driving data over 219 kilometers, while the test dataset consists of 1.6 hours. FD refers to Frechet distance, where (P / M) stands for Precise and Misleading. The unit of FD is meters.

𝑔 𝑝 𝑓

𝑋 (𝑡)

Front View

- B. Task Definition At each timestep t, the robot R manages two states: the

[Figure 3]

|[Figure 4]|
|---|

|ViT|
|---|

|[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>WaypointTokens<br><br>𝑤<br><br>𝑤<br><br>𝑤<br><br>𝑤|
|---|

|PDController|
|---|

|LLM|
|---|

𝑣

𝜏 𝜏 𝜏

WaypointTokens

PDController

𝜔

ViT

front view image Xf(t) and the robot’s hindsight trajectory [45] up to timestep t − 1, denoted as H(t). The front view image Xf(t) is captured by the robot’s camera, while H(t) is tracked through odometry to log the robot’s past positions. We combine the sketch instruction S and hindsight trajectory H(t) onto the same map to create the canvas map image Xc(t). At each step, the robot R generates an action y(t) = [w0,w1,w2,w3], which is a sequence of waypoints. This action is conditioned on the front view image Xf(t), the canvas map image Xc(t), and the language instruction L. Formally, the robot’s action is defined as:

𝑋 (𝑡)

Canvas Map

𝜏 𝜏 𝜏

LLM

|[Figure 9]|
|---|

|Waypoints|
|---|

𝐿

Language Instruction

|You must drive on the sidewalk|
|---|

Fig. 3: Overview of the CANVAS framework. It processes the front view image Xf(t), canvas map Xc(t), and language instruction L to generate waypoint tokens, which are passed to a PD controller to move the robot.

y(t) = R(Xf(t),Xc(t),L) At the end of each iteration, the hindsight trajectory is updated by appending p(t), which represents the robot’s position updated through predicted waypoints y(t), resulting in H(t + 1) = (p(1),p(2),...,p(t)). This update is reflected in the next canvas map image Xc(t+1), while the front view image Xf(t + 1) is also updated based on the robot’s new position. This process continues until the robot either reaches the destination or a maximum timestep t = T is reached.

with empirical testing showing that K=128 outperformed 32, 64, or 256. Fewer tokens can hinder precise actions like navigating narrow passages.

- B. Training

CANVAS is designed to generate actions as a sequence of waypoint tokens. A robot’s trajectory can be represented by N consecutive waypoints. During training, we minimize the negative log-likelihood loss, which is formulated as follows:

J(πθ) =

N

n=1

3

t=0

log πθ (wtn | Xf(t)n,Xc(t)n,Ln)

The model reframes the navigation as a classification problem, where it predicts the next waypoint based on the current state and given instructions. As explained in Related Work Section II-B, by treating navigation as a classification task, the model can manage multimodal distributions, enhancing both stability and accuracy in complex environments.

- C. Inference

IV. METHOD

To address the problem formulated in Section III, we introduce CANVAS, a navigation system designed to bridge the gap between abstract human instructions and concrete robot actions by leveraging commonsense understanding from pretrained vision-language models (VLMs) [46]. In this section, we provide an overview of the model architecture, as well as the training and inference processes.

A. Architecture

The model architecture is illustrated in Figure 3. We adopt a VLM denoted as πθ. The front view image Xf(t) and canvas map image Xc(t) are processed through a vision encoder gϕ(·). This results in two visual features, Zf = gϕ(Xf(t)) and Zc = gϕ(Xc(t)). A projector pϕ is used to project these visual features into the word embedding space, producing a sequence of visual tokens τv = pϕ(Zf,Zc).

During inference, the model-generated discrete waypoint tokens convert into continuous waypoints, which are then input into a Proportional-Derivative (PD) controller to produce linear and angular velocities (v,ω) for the robot’s actuators.

- A sequence of language tokens τl is also obtained from the language instruction L. Both the visual tokens τv and language tokens τl are then fed into the large language model denoted as fϕ(·), which outputs the waypoint tokens [w0,w1,w2,w3] = fϕ(τv,τl). Due to the reasons mentioned in Section II-B, we apply the simplest method, K-means clustering, to discretize continuous waypoints into tokens,

V. EXPERIMENTS In this section, we aim to answer the following questions:

- A. Can CANVAS handle a variety of commonsense-aware navigation tasks in simulated environments?
- B. Can CANVAS be transferred to a real-world environment in a zero-shot manner?

[Figure 10]

- Fig. 4: The left side of the figure compares CANVAS-L and CANVAS-S, showing CANVAS-L using the crosswalk despite a misleading sketch instruction. The right side compares CANVAS-L and NavStack, illustrating CANVAS-L avoiding small obstacles, such as rocks.

C. How much does leveraging the pre-trained weights of the VLM enhance CANVAS ’s performance?

- D. When NavStack fails, in what ways can CANVAS succeed?
- E. Is CANVAS fast enough for real-time navigation?

Experimental Setup. In COMMAND, successful navigation requires the robot to reach the target location without collisions, while also respecting commonsense constraints. As a result, four key metrics were used to evaluate performance: SR, CR, TDD, and IVR. Success Rate (SR) represents the proportion of successful episodes, while Collision Rate (CR) captures the proportion of episodes with collisions. Trajectory Deviation Distance (TDD) measures how closely the model follows human demonstrations, using the interquartile mean of Frechet distances. Finally, Instruction Violation Rate (IVR) assesses the proportion of episodes where human evaluators identified violations of commonsense constraints, such as keeping to the right lane or using crosswalks. TDD was calculated only for success cases, as including failure cases would skew the metric.

We compare CANVAS with ROS NavStack [21], a straightforward yet effective rule-based navigation system. For this system, we converted the sketch instructions into step-by-step, point-to-point inputs, but language instructions could not be accommodated. The same hyperparameters were used for all experiments with NavStack. We evaluate two variations of CANVAS. CANVAS-S modifies the original Idefics2 8B [46] by swapping the vision encoder for SigLIPL [47] and the text encoder for Qwen2-0.5B [48], reducing the model size from 8B to 0.7B to better accommodate real-world deployment. In contrast, CANVAS-L retains the original Idefics2 8B [46] architecture with its pre-trained weights. Both models were trained using LoRA [49] with r=256, a=512, and a dropout rate of 0.1. We use the AdamW optimizer [50] with a different learning rates of 2e-5 for the LLM and 5e-5 for the projector and vision encoder, a batch size of 32, and 5 training epochs. Each model utilizes 128 waypoint tokens. In the main experiments, CANVAS models were inferred using a single NVIDIA H100 GPU. All experiments were evaluated over three iterations for each test dataset, with randomized starting orientations. In the real-

world environment, SLAM with FAST-LIO2 [51] was used to find the robot’s current position.

Precise Misleading Total SR(↑) CR(↓) TDD(↓) SR(↑) CR(↓) TDD(↓) SR(↑)

Method

Seen Environment Office

NavStack 87% 13% 0.846m 0%* 100%* - -

- CANVAS-S 100% 0% 0.730m 87% 13% 0.843m 93% CANVAS-L 100% 0% 0.802m 100% 0% 0.753m 100%

Street (Road) NavStack 100% 0% 1.654m 0%* 100%* - -

- CANVAS-S 100% 0% 1.189m 100% 0% 1.075m 100% CANVAS-L 97% 3% 1.117m 97% 3% 1.236m 97%

Street (Sidewalk)

NavStack 53% 53% 1.450m 0%* 100%* - CANVAS-S 60% 40% 1.451m 47% 53% 2.379m 54% CANVAS-L 87% 13% 1.394m 53% 47% 1.839m 70% Orchard

NavStack 0% 87% - 0%* 100%* - CANVAS-S 73% 60% 1.561m 60% 33% 1.448m 67% CANVAS-L 67% 47% 1.759m 60% 53% 1.392m 64% Unseen Environment Gallery

NavStack 100% 0% 0.783m 0%* 100%* - CANVAS-S 87% 13% 0.773m 33% 66% 0.938m 60% CANVAS-L 100% 7% 0.9m 33% 66% 0.856m 67%

- TABLE III: Evaluation results on simulated environments. *: NavStack was not tested in the misleading scenario because it is not equipped to handle such situations.

|Environment<br><br>|Method<br><br>Precise Misleading IVR(↓) IVR(↓)|
|---|---|
|Street (Road)|NavStack 7% 100%*<br><br>CANVAS-S 0% 7% CANVAS-L 17% 30%<br><br>|
|Street (Sidewalk)|NavStack 7% 100%*<br><br>CANVAS-S 0% 26% CANVAS-L 0% 13%<br><br>|

- TABLE IV: Evaluation of violation rates for commonsense constraints in a street environment.

A. Results in the Simulated Environments

Seen Environments. Table III shows CANVAS’s performance in three seen environments: office, street (road, sidewalk), and orchard. Under precise instructions, CANVAS achieves similar SR and CR to NavStack in the office and street (road), where navigation is easier, indicating that CANVAS can learn the essential navigation behaviors effectively from human demonstrations. However, in more challenging

[Figure 11]

- Fig. 5: We developed a physical robot and created a realistic simulated environment that replicates real-world conditions.

environments like the street (sidewalk) and orchard, CANVAS significantly outperforms NavStack. A detailed analysis of CANVAS’s performance is provided in Section V-D.

Table IV compares the IVR between CANVAS and NavStack. In the street (road), commonsense constraints include lane adherence, while in the street (sidewalk), they involve crossing roads correctly and staying on the sidewalk. Unlike NavStack’s rule-based approach, CANVAS learns commonsense driving rules from human demonstrations, consistently achieves lower IVR even with misleading instructions.

Unseen Environment. We exclude the gallery environment during training to evaluate CANVAS’s performance in unseen settings. As demonstrated in Table III, CANVAS continues to show strong navigation capabilities, even in scenarios with noisy guidance.

- B. Results in the Real-World Environment

While COMMAND collects human demonstrations across a variety of simulated environments designed to resemble real-world conditions, a potential concern is whether these simulations fully capture the complexity of the real world. Therefore, it is important to demonstrate that the CANVAS’s effective navigation in simulation can extend to real-world environments. As shown in Figure 5, to assess its realworld performance, we tested CANVAS in an actual office environment that was used as the basis for the simulation. Despite being trained solely on simulated data, CANVAS demonstrated strong Sim2Real transfer capabilities, performing reliably in real-world scenario.

Method

Precise Misleading Total

SR(↑) SR(↑) SR(↑) NavStack 100% 0%* CANVAS-S 77% 60% 69% CANVAS-L 93% 33% 63%

TABLE V: Evaluation results on real environments.

- C. Ablation Study

We explore the importance of leveraging pre-trained weights from the VLM. As demonstrated in Table VI, these weights were crucial for CANVAS’s performance, especially in both unseen simulated and real-world settings. This indicates that the knowledge encapsulated in the pretrained VLMs offered a strong foundation for CANVAS to learn how to incorporate them in developing a generalizable understanding of driving dynamics.

|Environment<br><br>|Method<br><br>Precise Misleading Total SR(↑) SR(↑) SR(↑)|
|---|---|
|Seen - Office<br><br>|CANVAS-L 100% 100% 100% w/o Pre-training 100% 87% 93%<br><br>|
|Unseen - Gallery|CANVAS-L 100% 33% 67% w/o Pre-training 60% 40% 50%<br><br>|
|Real - Office|CANVAS-L 93% 33% 63% w/o Pre-training 73% 33% 53%<br><br>|

TABLE VI: Ablation study on the effect of VLM pre-training.

[Figure 12]

Fig. 6: We classify the failure cases of NavStack [21] in various simulated environments.

- D. Additional Case Study

We perform a qualitative analysis to examine the factors behind the success of CANVAS compared to NavStack [21]. Figure 6 highlights typical failure cases for NavStack, where the robot is unable to reach its destination. In 56% of these cases, failures result from stumbling over rocks in the orchard environment. Figure 4 compares CANVAS and NavStack in this primary failure scenario. The orchard has uneven terrain, and NavStack struggles to avoid small but hazardous obstacles like rocks because its limited perception cannot distinguish rocks from passable areas such as grass.

Tuning the hyperparameters of NavStack exacerbates the issue: setting the obstacle’s height threshold lower causes the system to misclassify grass as an obstacle, and the robot LiDAR sensor, which suffers from significant vibration, struggles to navigate with a small threshold. In contrast, CANVAS utilizes visual inputs from the camera to reliably detect unexpected obstacles and assesses their navigation risk based on learned experiences from demonstrations.

- E. Real-Time Navigation Feasibility Study

Finally, we evaluate the feasibility of deploying CANVAS in real-time applications. The NVIDIA AGX Orin, integrated into our robot, demonstrates on-device inference capabilities with an average latency of 400ms for CANVAS-S and 800ms for CANVAS-L, highlighting its potential for efficient realworld navigation tasks without substantial delays.

VI. CONCLUSION

We present CANVAS, a novel commonsense-aware navigation system that learns from human demonstrations through imitation learning. CANVAS allows intuitive human instructions using abstract sketches and natural language while leveraging commonsense reasoning to bridge the gap between vague human guidance and concrete robot actions. With the COMMAND dataset for imitation learning and pretrained vision-language models, CANVAS allows robots to understand implicit human intent and make decisions aligned

with human expectations. Experiments show that CANVAS outperforms ROS NavStack, a strong rule-based system, with higher success rates, fewer collisions, and better trajectory alignment with human demonstrations, all while adhering to commonsense constraints. Additionally, CANVAS exhibits strong performance in both unseen and real-world environments, highlighting its generalization capabilities. By opensourcing the COMMAND dataset and CANVAS, we hope to contribute to active research on imitation learning techniques for commonsense reasoning in robot navigation.

ACKNOWLEDGMENTS

This research was partly supported by GINT. The authors would like to thank GINT for providing feedback on the agricultural spraying robot.

REFERENCES

- [1] Z. Li, X. Liu, H. Wang, J. Song, F. Xie, and K. Wang, “Research on robot path planning based on point cloud map in orchard environment,” IEEE Access, 2024. 1
- [2] H. Wang, L. Zhang, Q. Kong, W. Zhu, J. Zheng, L. Zhuang, and X. Xu, “Motion planning in complex urban environments: An industrial application on autonomous last-mile delivery vehicles,” Journal of Field Robotics, vol. 39, no. 8, pp. 1258–1285, 2022. 1
- [3] Y. Du, N. J. Hetherington, C. L. Oon, W. P. Chan, C. P. Quintero, E. Croft, and H. Machiel Van der Loos, “Group surfing: A pedestrianbased approach to sidewalk robot navigation,” in 2019 International Conference on Robotics and Automation (ICRA), 2019, pp. 6518–

6524. 1

- [4] P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. S¨underhauf,

I. Reid, S. Gould, and A. van den Hengel, “Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 1, 2

- [5] C. Huang, O. Mees, A. Zeng, and W. Burgard, “Visual language maps for robot navigation,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023. 1, 2
- [6] J. Zhang, K. Wang, R. Xu, G. Zhou, Y. Hong, X. Fang, Q. Wu, Z. Zhang, and W. He, “Navid: Video-based vlm plans the next step for vision-and-language navigation,” arXiv preprint arXiv:2402.15852,

2024. 1, 2

- [7] M. Skubic, D. Anderson, S. Blisard, D. Perzanowski, and A. Schultz, “Using a hand-drawn sketch to control a team of robots,” Autonomous Robots, 2007. 1
- [8] F. Boniardi, A. Valada, W. Burgard, and G. D. Tipaldi, “Autonomous indoor robot navigation using a sketch interface for drawing maps and routes,” in 2016 IEEE International Conference on Robotics and Automation (ICRA), 2016. 1
- [9] W. Zu, W. Song, R. Chen, Z. Guo, F. Sun, Z. Tian, W. Pan, and J. Wang, “Language and sketching: An llm-driven interactive multimodal multitask robot navigation framework,” in 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024. 1, 2
- [10] J.-P. T¨oberg, A.-C. Ngonga Ngomo, M. Beetz, and P. Cimiano, “Commonsense knowledge in cognitive robotics: a systematic literature review,” Frontiers in Robotics and AI, 2024. 1
- [11] H. Chen, H. Tan, A. Kuntz, M. Bansal, and R. Alterovitz, “Enabling robots to understand incomplete natural language instructions using commonsense reasoning,” in 2020 IEEE International Conference on Robotics and Automation (ICRA), 2020. 2
- [12] K. Zhou, K. Zheng, C. Pryor, Y. Shen, H. Jin, L. Getoor, and X. E. Wang, “Esc: Exploration with soft commonsense constraints for zeroshot object navigation,” in Proceedings of the 40th International Conference on Machine Learning, 2023. 2
- [13] G. Zhou, Y. Hong, and Q. Wu, “Navgpt: Explicit reasoning in visionand-language navigation with large language models,” Proceedings of the AAAI Conference on Artificial Intelligence, 2024. 2
- [14] A. S. Chen, A. M. Lessing, A. Tang, G. Chada, L. Smith, S. Levine, and C. Finn, “Commonsense reasoning for legged robot adaptation with vision-language models,” arXiv preprint arXiv:2407.02666, 2024. 2

- [15] M. Kim, M. Kim, J. Bae, S. Choi, S. Kim, and B. Chang, “Exploiting semantic reconstruction to mitigate hallucinations in vision-language models,” in European Conference on Computer Vision. Springer, 2024, pp. 236–252. 2
- [16] E. Wijmans, A. Kadian, A. Morcos, S. Lee, I. Essa, D. Parikh, M. Savva, and D. Batra, “Dd-ppo: Learning near-perfect pointgoal navigators from 2.5 billion frames,” in International Conference on Learning Representations, 2020. 2
- [17] M. Yin, T. Li, H. Lei, Y. Hu, S. Rangan, and Q. Zhu, “Zero-shot wireless indoor navigation through physics-informed reinforcement learning,” in 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024. 2
- [18] H. Taheri and S. R. Hosseini, “Deep reinforcement learning with enhanced ppo for safe mobile robot navigation,” arXiv preprint arXiv:2405.16266, 2024. 2
- [19] Y. Ding, C. Florensa, M. Phielipp, and P. Abbeel, “Goal-conditioned imitation learning,” Advances in Neural Information Processing Systems, 2019. 2
- [20] N. Hirose, F. Xia, R. Mart´ın-Mart´ın, A. Sadeghian, and S. Savarese, “Deep visual mpc-policy learning for navigation,” IEEE Robotics and Automation Letters, 2019. 2
- [21] Open Robotics, “Github - ros-planning/navigation,” 2023, https:// github.com/ros-planning/navigation version: noetic, 1.17.3 accessed: 2024.09.15. 2, 5, 6
- [22] J. Christian Andersen, O. Ravn, and N. A. Andersen, “Autonomous rule-based robot navigation in orchards,” IFAC Proceedings Volumes,

2010. 2

- [23] F. Gul, W. Rahiman, and S. S. Nazli Alhady, “A comprehensive study for robot navigation techniques,” Cogent Engineering, 2019. 2
- [24] B. Liu, Z. Guan, B. Li, G. Wen, and Y. Zhao, “Research on slam algorithm and navigation of mobile robot based on ros,” in 2021 IEEE International Conference on Mechatronics and Automation (ICMA),

2021. 2

- [25] A. Sridhar, D. Shah, C. Glossop, and S. Levine, “Nomad: Goal masked diffusion policies for navigation and exploration,” in 2024 IEEE International Conference on Robotics and Automation (ICRA),

2024. 2

- [26] D. Shah, A. Sridhar, N. Dashora, K. Stachowicz, K. Black, N. Hirose, and S. Levine, “Vint: A foundation model for visual navigation,” in Proceedings of The 7th Conference on Robot Learning, 2023. 2
- [27] D. Shah, A. Sridhar, A. Bhorkar, N. Hirose, and S. Levine, “Gnm: A general navigation model to drive any robot,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023. 2
- [28] F. Blochliger, M. Fehr, M. Dymczyk, T. Schneider, and R. Siegwart, “Topomap: Topological mapping and navigation based on visual slam maps,” in 2018 IEEE International Conference on Robotics and Automation (ICRA), 2018. 2
- [29] J. Gu, E. Stefani, Q. Wu, J. Thomason, and X. Wang, “Vision-andlanguage navigation: A survey of tasks, methods, and future directions,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2022. 2
- [30] W. Wu, T. Chang, X. Li, Q. Yin, and Y. Hu, “Vision-language navigation: a survey and taxonomy,” Neural Computing and Applications,

2023. 2

- [31] E. Jang, A. Irpan, M. Khansari, D. Kappler, F. Ebert, C. Lynch, S. Levine, and C. Finn, “BC-z: Zero-shot task generalization with robotic imitation learning,” in 5th Annual Conference on Robot Learning, 2021. 2
- [32] M. Zare, P. M. Kebria, A. Khosravi, and S. Nahavandi, “A survey of imitation learning: Algorithms, recent developments, and challenges,” IEEE Transactions on Cybernetics, 2024. 2
- [33] H. Karnan, G. Warnell, X. Xiao, and P. Stone, “Voila: Visualobservation-only imitation learning for autonomous navigation,” in 2022 International Conference on Robotics and Automation (ICRA),

2022. 2

- [34] X. Pan, T. Zhang, B. Ichter, A. Faust, J. Tan, and S. Ha, “Zeroshot imitation learning from demonstrations for legged robot visual navigation,” in 2020 IEEE International Conference on Robotics and Automation (ICRA), 2020. 2
- [35] N. M. Shafiullah, Z. Cui, A. A. Altanzaya, and L. Pinto, “Behavior transformers: Cloning k modes with one stone,” Advances in neural information processing systems, 2022. 2
- [36] L. Metz, J. Ibarz, N. Jaitly, and J. Davidson, “Discrete sequen-

- tial prediction of continuous actions for deep rl,” arXiv preprint arXiv:1705.05035, 2017. 2
- [37] R. Dadashi, L. Hussenot, D. Vincent, S. Girgin, A. Raichuk, M. Geist, and O. Pietquin, “Continuous control with action quantization from demonstrations,” arXiv preprint arXiv:2110.10149, 2021. 2
- [38] Y. Chebotar, Q. Vuong, K. Hausman, F. Xia, Y. Lu, A. Irpan, A. Kumar, T. Yu, A. Herzog, K. Pertsch, et al., “Q-transformer: Scalable offline reinforcement learning via autoregressive q-functions,” in Conference on Robot Learning, 2023. 2
- [39] S. Lee, Y. Wang, H. Etukuru, H. J. Kim, N. M. M. Shafiullah, and L. Pinto, “Behavior generation with latent actions,” in Proceedings of the 41st International Conference on Machine Learning, 2024. 2
- [40] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al., “Rt-2: Visionlanguage-action models transfer web knowledge to robotic control,” arXiv preprint arXiv:2307.15818, 2023. 3
- [41] Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, et al., “Hydra-mdp: End-to-end multimodal planning with multi-target hydra-distillation,” arXiv preprint arXiv:2406.06978,

2024. 3

- [42] NVIDIA, “Isaac Sim - Robotics Simulation and Synthetic Data

— NVIDIA Developer,” https://developer.nvidia.com/isaac/sim, Accessed: 2024.09.15. 3

- [43] Epic Games, “Unreal engine,” https://www.unrealengine.com, [Accessed: 2024.09.15]. 3
- [44] B. Aronov, S. Har-Peled, C. Knauer, Y. Wang, and C. Wenk, “Fr´echet distance for curves, revisited,” in Algorithms–ESA 2006: 14th Annual

- European Symposium, Zurich, Switzerland, September 11-13, 2006. Proceedings 14, 2006. 3
- [45] J. Gu, S. Kirmani, P. Wohlhart, Y. Lu, M. G. Arenas, K. Rao, W. Yu, C. Fu, K. Gopalakrishnan, Z. Xu, P. Sundaresan, P. Xu, H. Su, K. Hausman, C. Finn, Q. Vuong, and T. Xiao, “Rt-trajectory: Robotic task generalization via hindsight trajectory sketches,” in The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024, 2024. 4
- [46] H. Laurenc¸on, L. Tronchon, M. Cord, and V. Sanh, “What matters when building vision-language models?” arXiv preprint arXiv:2405.02246, 2024. 4, 5
- [47] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 5
- [48] A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, F. Huang, et al., “Qwen2 technical report,” arXiv preprint arXiv:2407.10671, 2024. 5
- [49] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” in The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, 2022. 5
- [50] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019, 2019. 5
- [51] W. Xu, Y. Cai, D. He, J. Lin, and F. Zhang, “Fast-lio2: Fast direct lidar-inertial odometry,” IEEE Transactions on Robotics, 2022. 5

