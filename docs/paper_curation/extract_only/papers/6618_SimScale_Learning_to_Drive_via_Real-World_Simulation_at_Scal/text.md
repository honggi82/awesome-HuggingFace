# arXiv:2511.23369v3[cs.CV]10Apr2026

## SIMSCALE: Learning to Drive via Real-World Simulation at Scale

Haochen Tian1,2,3⋄ Tianyu Li2† Haochen Liu3 Jiazhi Yang2 Yihang Qiu2,3⋄ Guang Li3 Junli Wang1,3 Yinfeng Gao1,3 Zhang Zhang1§ Liang Wang1§ Hangjun Ye3 Tieniu Tan1 Long Chen3§ Hongyang Li2

- 1 MAIS, Institute of Automation, Chinese Academy of Sciences
- 2 OpenDriveLab at The University of Hong Kong 3 Xiaomi EV https://opendrivelab.com/SimScale

- 1. Real-World Data

- 2. Scalable Simulation Data

3. Sim-Real Co-training

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Real

Sim.

[Figure 5]

[Figure 6]

###### ANY

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

E2E Planner

[Figure 11]

[Figure 12]

[Figure 13]

…

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

- (a) Scalable Simulation Data and Sim-Real Co-training

Human-Expert Pseudo-Expert

[Figure 26]

(b) Real-World Benchmarks

- 1. Single-mode Regression

3. Vocabulary Scoring

- 2. Diffusion Policy

[Figure 27]

Robustness Generalization

27.5

84.2

32.6

85.9

Robustness Generalization

38.3

82.3

46.9 84.6

Robustness Generalization

24.4

81.5

30.2

84.4

[Figure 28]

[Figure 29]

[Figure 30]

Logged States Perturbed States

Real Sim-Real

reactive sim 1 reactive sim 2 reactive sim 3

Real Ego Sim. Ego

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Figure 1. Scaling up end-to-end planners by simulation. (a) We construct large-scale simulation data by perturbing ego trajectories, generating corresponding pseudo-expert demonstrations, and rendering multi-view observations in reactive environments. Combined with real-world data, this enables broad coverage of out-of-distribution states and supports sim–real co-training for any end-to-end planner.

- (b) Across three representative planner families, including regression, diffusion, and vocabulary scoring, sim-real co-training consistently produces synergistic improvements in robustness and generalization, demonstrating clear and predictable simulation scaling trends.

### Abstract

Achieving fully autonomous driving systems requires learning rational decisions in a wide span of scenarios, including safety-critical and out-of-distribution ones. However, such cases are underrepresented in real-world corpus collected by human experts. To complement for the lack of data diversity, we introduce a novel and scalable simulation framework capable of synthesizing massive unseen states upon existing driving logs. Our pipeline utilizes advanced neural rendering with a reactive environment to generate high-fidelity multi-view observations controlled by the perturbed ego trajectory. Furthermore, we

⋄ Work done while interning at Xiaomi Embodied Intelligence Team. † Project Lead. § Equal Advising. Primary contact to Haochen Tian tianhaochen2023@ia.ac.cn

develop a pseudo-expert trajectory generation mechanism for these newly simulated states to provide action supervision. Upon the synthesized data, we find that a simple cotraining strategy on both real-world and simulated samples can lead to significant improvements in both robustness and generalization for various planning methods on challenging real-world benchmarks, up to +8.6 EPDMS on navhard and +2.9 on navtest. More importantly, such policy improvement scales smoothly by increasing simulation data only, even without extra real-world data streaming in. We further reveal several crucial findings of such a sim-real learning system, which we term SimScale, including the design of pseudo-experts and the scaling properties for different policy architectures. Simulation data and code have been released at https://github.com/OpenDriveLab/

SimScale.

### 1. Introduction

Data scaling is recognized as a foundational principle in modern deep learning across various domains, including language, vision, and multimodal modeling, underpinning steady performance improvements as data sizes increase [5, 38, 72, 92]. In autonomous driving, end-to-end (E2E) planning learns to map raw observations to actions, offering a promising way to leverage large-scale driving data to enable the emergence of fully autonomous systems [29, 68, 98].

Nevertheless, real-world driving data from human expert demonstrations are dominated by common scenarios, while non-trivial cases, e.g., safety-critical, are underrepresented [9, 62, 84, 87]. Moreover, planners trained on such data are confined to human driving distribution and struggle to generalize to rare or unseen situations, leading to distribution shift and causal confusion in deployment [20, 56]. Consequently, scaling real-world data only is inefficient for achieving deploy-ready autonomous driving.

Simulation via neural rendering [40] can generate highfidelity driving scenarios and thus has the potential to produce out-of-distribution (OOD) states deviated from human demonstrations at scale, which is essential for closedloop planning [23, 71]. Therefore, scaling simulation data presents an attractive alternative to solely relying on realworld data. However, planners require feasible corresponding demonstrations to learn how to handle OOD states, while current simulation methods fail to generate such demonstrations effectively. Moreover, the impact of scaling simulation data lacks in-depth analysis. In this work, we aim to derive a systematic recipe for scaling simulation data from limited real-world scenarios in end-to-end planning.

To conduct comprehensive experiments and analyses, this study is structured along three key questions: (1) what constitutes effective simulation data, (2) how well planners benefit from it, and (3) whether this system scales predictably upon fixed real-world corpus.

To this end, we formulate a scalable simulation data generation framework that extends the expert distribution from existing real-world training data to bootstrap end-toend autonomous driving systems. We develop a 3D Gaussian Splatting (3DGS) [40] simulation data engine, which allows controlling the temporal ego and other agent states and rendering multi-view videos from the ego’s perspective. Concretely, we first sample a diverse set of plausible perturbations to the ego trajectory, maximizing coverage of the state space, e.g., off-center lane drifts, close interactions, among others. Then, we take the final state of each perturbed trajectory as the perturbed state and generate corresponding demonstrations using pseudo-experts of two forms in comparison. The first, recovery-based expert retrieves trajectories that steer the policy within the human trajectory manifold, resulting in human-like yet cautious behaviors. The second, privileged planner-based expert [16]

generates the trajectory that maximizes optimality, representing an exploratory strategy with lower realism. To enhance scalability and plausibility, the entire pipeline is executed in a reactive environment [76], where surrounding agents interact with the ego vehicle responsively.

To thoroughly assess the effect of simulation data, we consider three types of end-to-end planners with various model scales, i.e., LTF [12] for regression methods, DiffusionDrive [60] for diffusion-based planners, and GTRSDense [59] for vocabulary scoring. We employ a simple yet effective sim-real co-training strategy [67, 69] to maintain human driving distribution while mitigating visual domain degradation. Furthermore, by keeping a constant amount of real data and progressively adding simulation data via non-overlapping samples, we investigate how diverse planners can benefit from simulation data and the general scaling property. We utilize two real-world closed-loop benchmarks to evaluate the planners from multiple perspectives. navhard [6] focuses on unseen, challenging scenarios to assess the impact of OOD states on planners, while navtest [17] offers a broad set of diverse scenarios to test planners’ ability to handle varying situations.

The complete sim-real learning system, which comprises a scalable simulation data construction pipeline and an effective sim–real co-training strategy as shown in Fig. 1, is termed SimScale. Rigorous experiments uncover crucial findings enabled by SimScale—including, but not limited to, the following:

- • Scalable simulation with pseudo-expert unlocks the inherent potential of available real-world driving data.
- • Sim-real co-training improves both robustness and generalization synergistically in diverse end-to-end planners.
- • Exploratory expert and interaction environments improve the effectiveness of simulation data.
- • Planners with multimodal modeling capabilities exhibit more encouraging data scaling properties.

### 2. Methodology

We outline SimScale as follows. In Sec. 2.2, we briefly introduce our 3GDS simulation data engine supporting controllable multi-view video rendering. In Sec. 2.3, we propose a pseudo-expert scene simulation pipeline to generate diverse simulation data containing OOD states with feasible demonstrations. In Sec. 2.4, we demonstrate a scalable simreal co-training method with different end-to-end planners.

#### 2.1. Preliminary

End-to-end planning models take observations within a history window as input and output a predicted future trajectory. Each training scenario starts at a selected timestep and includes a history horizon of length T and a planning horizon of length H. The model processes the past T frames to

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Other Agent

Real Ego Sim. Ego

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

𝑇 + 2𝐻

𝑇 + 2𝐻

𝑇 + 2𝐻

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

𝑇 + 𝐻

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Sim.

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

𝑇 + 𝐻

𝑇 + 𝐻

𝑇 + 𝐻

[Figure 92]

𝑇

[Figure 93]

[Figure 94]

Real

Sim. Real

1. Recovery-based

2. Planner-based Sim.

[Figure 95]

reactive env.

(a) Trajectory Perturbation

(b) Pseudo-Expert Trajectory Generation

- Figure 2. Pseudo-expert scene simulation pipeline. (a) Trajectory perturbation on T to T + H, (b) reactive environment rollout, and pseudo-expert trajectory generation from T + H to T + 2H under recovery-based and planner-based strategies.

ensure the quality of generated simulation data.

predict the future H frames, resulting in a complete training sample spanning T + H timesteps.

#### 2.3. Pseudo-Expert Scene Simulation

#### 2.2. 3DGS Simulation Data Engine

Based on the data engine, we design a pseudo-expert scene simulation pipeline to generate diverse simulation data from existing real-world data, as shown in Alg. 1. The pipeline aims to produce feasible demonstrations from perturbed states paired with expert trajectories, as illustrated in Fig. 2. Reactive Scene Reconstruction. For each training clip d, we perform two simulations of duration H: one for exploring perturbed states at t = T, and the other for generating expert trajectories at t = T +H. In each simulation, ego trajectory a˜t:t+H is simulated by LQR [43], while other agents are modeled using the IDM [76] to interact with the ego, producing the corresponding future states s˜t:t+H. For the valid expert trajectories, the simulated ego–agent trajectories cross two stage (˜aT:T+2H,s˜T:T+2H) are then rendered into multi-view videos o˜T:T+2H using the data engine Φ. This decoupling of behavior simulation from sensor rendering enables reactive environments where other agents respond plausibly to the ego’s behaviors, thereby enhancing both the realism and diversity of the simulation data.

To reduce the domain gap between real-world data and generated observations from novel views in simulated scenarios, a photorealistic data engine is required. Built upon 3DGS [40] assets reconstructed from real-world datasets, our data engine Φ(Kt,Et,{xi,t,yi,t,θi,t}Ni=1) takes as input the camera intrinsics Kt and extrinsics Et at timestep t, along with the positions and yaw angles (x,y,θ) of a nonego vehicle i at the same timestep, and renders the corresponding RGB observations from the camera. The camera extrinsics Et can easily be obtained from the ego vehicle’s position and heading (x0,t,y0,t,θ0,t) with the ego-tocamera transformation, while the other camera parameters are directly adopted from the raw dataset.

Preprocessing. Following prior work [51], we apply exposure alignment across images captured simultaneously by multi-view cameras at time t, using projected LiDAR points at t as guidance. Moreover, with help of 3D bounding boxes provided by NAVSIM annotations, colored LiDAR points are divided into several groups, i.e., the static background and multiple vehicles, and further used as initialization for Gaussians to improve the reconstruction performance.

Trajectory Perturbation. At the first simulation step (t = T), we perturb the ego trajectory a˜t:t+H so that the ego reaches a new terminal state at t = T + H, which then serves as the start state for the next rollout (Fig. 2 (a)). Our objective is to sample diverse yet plausible states. For diversity, perturbations are drawn from a clustered humantrajectory vocabulary that densely covers the action space. For plausibility, we restrict perturbations to remain near human behavior by thresholding longitudinal/lateral shifts (rlon,rlat) and heading change |∆θ|, and by removing trajectories that are physically invalid (collisions, off-road, unstable renderings). We further apply spatially sparse sampling on trajectory endpoints using an interleaved grid with steps (δlon,δlat) to promote uniform coverage. Since reactive simulation is costly, we first filter infeasible trajectories

Block-wise Reconstruction. During scene reconstruction, increasing the number of images significantly raises computational cost and runtime. Thus, we perform reconstruction in a block-wise manner, where each block corresponds to a spatio-temporal range. Following previous work [51, 86], backgrounds and foregrounds are reconstructed as separate models using the per-timestep locations and orientations of 3D bounding boxes, resulting in a static background asset and multiple movable vehicle assets. These assets could then be placed at desired locations and headings specified by inputs, rendering sensor data at novel views. Blocks with low average PSNR in novel view synthesis are excluded to

Algorithm 1 Pseudo-Expert Scene Simulation Require: Training clip d = (ot,st,at)Tt=0+2H ∈ D, perturb

action sets Aper, sensor engine Φ, pseudo policy πexp, reward functions f, reactive simulation T .

- 1: Initialize simulation scene dataset Dsim ← ∅
- 2: for each perturbed action aper ∈ Aper do
- 3: for timestep t ∈ {T,T + H} do
- 4: 1.Pseudo Action Generation:
- 5: if t = T then a˜t:t+H ← aper;
- 6: else a˜t:t+H ← πexp(˜ot,s˜t).
- 7: end if
- 8: 2.Reactive State Simulation:
- 9: s˜t:t+H ← T (˜at:t+H,s˜t).
- 10: 3.Sensor Simulation:
- 11: o˜t:t+H ← Φ(˜st:t+H).
- 12: 4.Reward Filtering / Labeling:
- 13: r˜t:t+H ← f(˜at:t+H,s˜t:t+H).
- 14: if Filtered(˜rt:t+H,o˜t:t+H) then
- 15: Break
- 16: end if
- 17: end for
- 18: Append (˜ot,s˜t,a˜t,r˜t)Tt=0+2H to Dsim
- 19: end for
- 20: return Dsim

non-reactively and run reactive checks only on the sparsed set. This produces a set of dynamically and physically feasible perturbations aper ∈ Aper.

Pseudo-Expert Trajectory Generation. At the second simulation t = T + H, for each perturbed state, we employ a non-human expert, referring as the pseudo-expert πexp, to generate a feasible corresponding trajectory a˜t:t+H. Since the pseudo-expert is not perfect and a˜t:t+H will be used as supervision, we apply stricter filtering in the second stage simulation. In addition to physical constraints, traffic rules and vehicle kinematic limits are also enforced, following Eq. 6. To investigate which strategy best serves as supervision for end-to-end planners, we compared two strategies: a conservative recovery-based expert, and an exploratory planner-based expert, as detailed below.

- (1) Recovery-based Expert. Recovery steers the policy within the human manifold after perturbation. To ensure robustness, our recovery expert πexp retrieves a human trajectory from a large vocabulary Vh that best matches the ego vehicle’s logged state at time t = T +2H (Fig. 2 (b1)). For each candidate trajectory of horizon H, we summarize its initial and final poses with a compact matching vector:

###### m = [˜vtx,v˜ty,θ˜t,x˜t+H−1,y˜t+H−1,θ˜t+H−1]. (1)

Given the ego’s perturbed state with target vector mr, πexp

retrieves the closest human maneuver by:

∥m(a) − mr∥1. (2)

a˜t:t+H = arg min a∈Vh

This yields a human-like yet conservative fallback behavior, stabilizing under distributional drifts.

(2) Planner-based Expert. Following prior works [3, 97], we use a privileged planner P that leverages ground-truth states to generate reactive and optimized trajectory rollouts in simulation, as shown in Fig. 2 (b2). The planner-based expert πexp ← P is defined as: a˜t:t+H = P(˜st:t+H). Compared with the recovery policy, planner-based expert relies on rules or cost heuristics, trading human-likeness and realism of its behavior occasionally. Still, it offers strong optimization and diverse exploratory rollouts, enriching expert supervision beyond human data.

#### 2.4. Scalable Sim-Real Co-training

Strategies for Co-training. Sim-real co-training serves as a simple yet effective strategy [4, 67, 69, 87] that enables the integration of both real and simulated data for planning. In our approach, we randomly sample from a mixture of real-world data D and simulation data Dsim during training, aiming to preserve the human driving distribution while mitigating visual domain degradation caused by potential simulation artifacts, such as subtle rendering inconsistencies, temporal jitter, or unrealistic lighting and shadows [63, 64]. Our fully automated, scalable simulation data generation framework enables scaling the total training data by progressively adding non-overlapping simulation samples, while keeping the real data amount fixed.

Planners for Co-training. We aim to comprehensively evaluate the effectiveness of simulation data for end-toend planners. Modern end-to-end planning approaches can be broadly categorized into three representative paradigms: regression-based planners [12, 27], diffusion-based planners [60, 83], and vocabulary scoring-based planners [34, 56]. Accordingly, we consider representative models from each paradigm in our co-training experiments.

(1) Co-training with Pseudo-Expert Trajectory. Regression and diffusion-based planners rely on expert demonstrations. The co-training process can thus be formulated as:

sim) Lim a,πθ(ˆa|o) , (3)

E(a,o)∼(D∪ D

arg min

θ

here, Lim denotes the imitation loss; D and Dsim represent the real-world and generated simulation datasets, respectively; and A refers to expert trajectories, i.e., human-expert trajectories in D and pseudo-expert trajectories in Dsim. As for the vocabulary scoring-based planner, the learning objective has additional prediction of reward signals r that distill the evaluation metrics, e.g. EPDMS as Eq. 6:

E(a,o,r)∼(D∪D

sim) λLim + Lr r, πθ(ˆa|o) , (4)

arg min

θ

[Figure 96]

[Figure 97]

Closed [16] as the expert. Only pseudo-expert trajectories with decent sub-metrics of EPDMS are considered valid. Due to computational limits, for each type of pseudo-expert, we perform up to five random samplings of valid trajectories with no overlaps from navtrain split for rendering. Through multiple samplings, expert trajectory demonstrations are progressively accumulated. As shown in Fig. 3, the two pseudo-expert methods exhibit different success rates, with the planner-based expert achieving higher efficiency. In total, we generate 147K recovery-based simulation scenes and 237K planner-based simulation scenes. For more details, please refer to Sec. B.1 in the supplementary

237 K

Increment Data Cumulative Data

205 K

167 K

147 K

130 K

120 K

109 K

Real-World Data Size

80 K

65 K

45 K

Rounds Rounds

(a) Recovery-based Expert (b) Planner-based Expert

Benchmark and Metrics. We utilize two benchmarks in NAVSIMv2 [6] for end-to-end model evaluation, including navhard and navtest. navhard is the official twostage evaluation benchmark, which contains 244 challenging real-world scenarios in the first stage and corresponding 4,164 synthetic scenarios generated by 3DGS in the second stage. navtest is a one-stage evaluation benchmark, containing a large number of 12,146 real-world scenarios. navhard focuses on assessing the model’s closed-loop performance in safety-critical situations, while navtest emphasizes generalization across diverse driving conditions. The two benchmarks share a rule-based planning metric, EPDMS [47], with several sub-metrics:

- Figure 3. Simulation data statistics across multiple sampling rounds. (a) Recovery-based expert impose stronger constraints, leading to slower data accumulation than (b) Planner-based expert.

here, Lr denotes reward loss, λ is a weighting factor.

- (2) Co-training with Rewards Only. For the vocabulary scoring-based planner, when the reward signal is wellaligned, the expert is theoretically unnecessary, as shown in Eq. 4. The planner can explore directions to increase the reward without being restricted to a single expert trajectory. The co-training process can be formulated as:

[Lr]. (5)

E(a,o,r)∼D[λLim + Lr] + E(o,r)∼D

 

 

arg min

sim

θ

wmSm m∈Mavg wm

· m∈Mavg

, (6)

Sm

EPDMS =

We thus investigate purely reward-driven optimization for the scoring-based planner in simulation, to assess its potential for fully leveraging simulation data, and assess its potential for maximizing the utility of simulated data.

m∈Mpen

weighted average

penalties

where Sm is the sub-metric: penalty terms set Mpen includes No-at-fault Collisions (NC), Drivable Area Compliance (DAC), Driving Direction Compliance (DDC), and Traffic Light Compliance (TLC); weighted average terms set Mavg includes Time-to-Collision (TTC), Ego Progress (EP), Lane Keeping (LK), History Comfort (HC), and extended comfort (EC). Note that EPDMS in navhard further incorporates several modifications, e.g., two-stage aggregation, reactive traffic simulation, and the exclusion of penalties in cases where the human expert driver also fails.

### 3. Experiments

#### 3.1. Setup and Protocols.

Real-world Datasets. We use the navtrain split from NAVSIM [17], built upon the largest publicly available annotated driving dataset nuPlan [39], and comprises 100K interactive real-world scenarios.

Simulation Data Curation. We construct simulated scenarios on navtrain split. 3DGS models of blocks with PSNR < 27 in novel-view results are removed ensure reconstruction quality. For ego trajectory perturbation, we choose the plausible candidates from the vocabulary of 16,384 trajectories from [59], with EPDMS ≥ 0.8 and relative heading |∆θ| ≤ 20◦, where EPDMS is detailed in Eq. 6. During spatial sampling by trajectory endpoint, the longitudinal and lateral perturbation ranges are set as rlon = 20m,rlat = 2.0m, with step sizes δlon = 5m,δlat = 0.5m. For recovery-based trajectory generation, we use all human expert trajectories in navtrain split as the retrieval vocabulary. For planner-based trajectory generation, we select the SOTA rule-based planner in nuPlan [39], PDM-

Models and Training. To thoroughly validate our modelagnostic simulation data, we select one representative opensource model for each paradigm discussed in Sec. 2.4: the regression-based LTF [12], the diffusion-based DiffusionDrive [60], and the scoring-based GTRS-Dense [59]. We adopt their official implementations with two modifications: the input image resolution is unified to 2048 × 512, and the LiDAR inputs are removed to align with the navhard evaluation setting. All models are trained from scratch on NVIDIA H20-3e GPUs. To ensure fair comparison, each model follows the same training strategy, whether using only the navtrain split or augmented with simulation data. For more details, please refer to Sec. B.2 in supplementary.

Table 1. Performance on the NAVSIM-v2 navhard Leaderboard. PDM-Closed uses ground-truth symbolic inputs for planning, while other methods rely on sensor data. (∗: pseudo-expert supervision; †: reward scoring; S.: per-stage EPDM score.)

Method Backbone Sim. Stage NC↑ DAC↑ DDC↑ TLC↑ EP↑ TTC↑ LK↑ HC↑ EC↑ S.↑ EPDMS↑

- S 1
- S 2

94.4 88.1

98.8 90.6

100 96.3

99.5 98.5

100 100

93.5 83.1

99.3 73.7

87.7 91.5

36.0 25.4

-

PDM-Closed [16] - -

51.3

Regeression-based Planner

- S 1
- S 2

97.3 79.4

80.2 69.0

97.8 85.6

99.3 98.5

83.4 83.8

96.2 76.7

92.9 47.9

97.8 97.0

71.1 70.6

61.3 39.2

w/o

24.4

LTF [12] ResNet34

66.3+5.0 44.8+5.6 30.2 ↑24%

- S 1
- S 2

96.1 85.5

85.3 66.9

99.4 91.6

99.3 99.1

84.7 93.0

94.7 81.1

93.6 58.3

97.6 95.1

77.3 42.9

w/∗

Diffusion-based Planner

- S 1
- S 2

96.8 80.1

86.0 72.8

98.8 84.4

99.3 98.4

84.0 85.9

95.8 76.6

96.7 46.4

97.6 96.3

79.6 72.8

66.7 40.5

w/o

27.5

DiffusionDrive [60] ResNet34

67.5+0.8 46.8+6.3 32.6 ↑19%

- S 1
- S 2

97.4 86.4

88.7 72.1

99.3 92.9

99.3 98.5

82.8 92.1

96.9 80.6

98.0 60.8

97.3 95.4

59.6 31.9

w/∗

Scoring-based Planner

- S 1
- S 2

99.3 92.8

96.6 88.6

99.6 95.5

100 99.4

57.4 55.9

99.5 91.3

92.6 55.7

89.5 91.1

16.4 35.7

67.1 55.8

w/o

38.3

ResNet34

72.4+5.3 63.4+7.6 46.9 ↑22%

- S 1
- S 2

97.6 94.3

96.4 92.7

99.3 95.1

100 99.5

75.7 80.2

97.8 91.5

93.3 56.2

97.3 90.6

32.9 28.3

w/†

GTRS-Dense [59]

- S 1
- S 2

98.9 89.9

94.9 90.5

99.1 94.1

100 99.3

76.1 77.6

98.4 88.5

93.8 56.0

94.9 92.0

37.8 30.2

70.4 58.5

w/o

41.9

V2-99

72.5+2.1 65.4+6.9 48.0 ↑15%

- S 1
- S 2

99.3 94.9

97.6 94.3

99.4 94.5

100 99.3

71.5 79.0

99.6 92.6

96.0 57.8

95.8 93.4

30.7 30.9

w/†

Table 2. Performance on the NAVSIM-v2 navtest Leaderboard. (∗: pseudo-expert supervision; †: reward scoring.)

Method Backbone Sim. NC↑ DAC↑ DDC↑ TLC↑ EP↑ TTC↑ LK↑ HC↑ EC↑ EPDMS↑ Human Agent - - 100 100 99.8 100 87.4 100 100 98.1 90.1 90.3

Regression-based Planner LTF [12] ResNet34

w/o 97.7 94.0 99.3 99.8 87.2 96.7 95.5 98.3 82.9 81.5 w/∗ 98.3 95.6 99.6 99.8 87.1 97.5 97.2 98.3 88.2 84.4 +2.9

Diffusion-based Planner DiffusionDrive [60] ResNet34

w/o 98.4 95.5 99.5 99.8 87.5 97.5 96.9 98.4 87.7 84.2 w/∗ 98.5 97.1 99.6 99.8 87.4 97.8 98.0 98.3 87.5 85.9 +1.7

Scoring-based Planner

w/o 97.6 97.5 99.0 99.9 87.9 97.0 95.9 97.5 55.9 82.3 w/† 98.4 98.8 99.4 99.9 88.0 98.0 96.5 97.6 58.0 84.6 +2.3

ResNet34

GTRS-Dense [59]

w/o 97.6 98.5 99.5 99.9 89.5 97.2 96.8 97.2 57.2 84.0 w/† 97.6 98.9 99.6 99.9 89.9 97.1 97.2 97.7 58.7 84.6 +0.8

V2-99

#### 3.2. Leaderboard Results

Tab. 1 and Tab. 2 present the leaderboard results of SimScale sim-real co-training for the three planner paradigms on navhard and navtest, respectively.

Navhard Leaderboard. All models exhibit significant improvements in both Stage 1 and Stage 2. Notably, GTRSDense (V2-99) achieves a score of 48.0, establishing a new SOTA on navhard. These results demonstrate that incorporating simulation data with an extended distribution substantially enhances model robustness in challenging and unseen environments. Notably, weaker baseline models, i.e., LTF and DiffusionDrive, benefit the most, with gains exceeding 20%, indicating that sim-real co-training with simulation data effectively enables models to exploit dataset

knowledge better and unlock their latent learning potential. Navtest Leaderboard. All models show consistent improvements of up to +2.9 points, demonstrating stronger performance under large-scale and diverse conditions. The quantitative results above highlight that our simulation data is model-agnostic, and that our general sim-real co-training achieves a synergistic optimization of robustness and generalization, which is essential for reliable closed-loop deployment in the real world [9, 62].

#### 3.3. Ablation and Data Scaling Analysis

Data Scaling Curves of Different Planners. Due to the lack of prior work studying the scaling behavior of simulation data under a fixed amount of real-world data, we model

[Figure 98]

[Figure 99]

[Figure 100]

Recovery-based Expert Planner-based Expert Reward Scoring

Inflection Point

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

###### EPDMS

EPDMS

EPDMS

EPDMS

[Figure 105]

[Figure 106]

𝑁 𝑁 𝑁 𝑁

Samples Samples Samples Samples

+ Sim. Data

###### + Sim. Data

+ Sim. Data

###### + Sim. Data

−0.54×(𝑙𝑜𝑔𝑁) +5.74×𝑙𝑜𝑔𝑁 − 14.66 −0.05×(𝑙𝑜𝑔𝑁) +0.64×𝑙𝑜𝑔𝑁 − 1.48 −0.01×(𝑙𝑜𝑔𝑁) +1.47×𝑙𝑜𝑔𝑁 − 3.73

[Figure 107]

−0.62×(𝑙𝑜𝑔𝑁) +6.48×𝑙𝑜𝑔𝑁 − 16.63

[Figure 108]

[Figure 109]

[Figure 110]

−0.15×(𝑙𝑜𝑔𝑁) +1.62×𝑙𝑜𝑔𝑁 − 4.01 −0.25×(𝑙𝑜𝑔𝑁) +2.69×𝑙𝑜𝑔𝑁 − 6.78

[Figure 111]

−0.23×(𝑙𝑜𝑔𝑁) +2.44×𝑙𝑜𝑔𝑁 − 6.17 −0.02×(𝑙𝑜𝑔𝑁) +0.33×𝑙𝑜𝑔𝑁 − 0.83

[Figure 112]

[Figure 113]

[Figure 114]

- −0.38×(𝑙𝑜𝑔𝑁) +4.12×𝑙𝑜𝑔𝑁 − 10.73
- −0.39×(𝑙𝑜𝑔𝑁) +4.20×𝑙𝑜𝑔𝑁 − 10.93

(a) LTF-ResNet34 | 56M (b) DiffusionDrive-ResNet34 | 61M (c) GTRS-Dense-ResNet34 | 67M (d) GTRS-Dense-2_99 | 83M

- Figure 4. Scaling dynamics across different planners and pseudo-expert trajectories. We visualize how simulation data scale and supervision signals influence the driving performance of various planners, where the infection point indicates learning plateau.

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

(a) off-center lane drift

Front

[Figure 122]

(b) near collision

(c) leaving the drivable area (d) cutting in

Front-Left

Front Front-Right Front

Front Right

Front-Right

Pseudo-Expert Trajectory Perturbed Trajectory

[Figure 123]

[Figure 124]

Front Front-Right

[Figure 125]

[Figure 126]

- Figure 5. Qualitative results of the simulation scenes on navtrain. Four representative simulation scenarios are shown, each mirroring a typical real-world OOD scene, with synthetic front-view and auxiliary key-view images provided.

proposed in Sec. 2.4. Some classical trends can be observed in Fig. 4. For instance, comparing planner-based and reward-scoring settings in Fig. 4 (c) and (d), larger models exhibit more pronounced data scaling trends under the same amount of data. Additional interesting and meaningful observations are highlighted below.

the relationship between performance and the total data size (simulation + real) using a log-quadratic function:

###### S(N) = alog2(N) + blog(N) + c, (7)

where S(N) denotes the planner performance with total data size N, and a, b, c are parameters fitted via nonlinear least squares:

Pseudo-Expert Should Be Exploratory. For all planners, the scaling curves under recovery-based setting converge earlier and achieve lower performance compared to plannerbased. The recovery-based expert always steers to the human log, limiting diversity as the simulation data grows from the same real scenarios. In contrast, the planner-based expert explores a broader range of possibilities and even produces feasible solutions in challenging situations. Consequently, the recovery-based expert only exhibits advantages in small-data regimes (Fig. 4 (d)), likely because its trajectory distribution is human-like and better aligned with real-world data, making it easier to learn. In most cases, data scaling yields diminishing returns for recovery-based methods compared to planner-based ones, and performance may even degrade as visual artifacts accumulate and begin to dominate. These observations highlight the importance of exploratory behavior for pseudo-experts, which enhances

M

Si − S(Ni;a,b,c) 2, (8)

(a∗,b∗,c∗) = arg min a,b,c

i=1

where Si is the observed performance at total data size Ni, and M is the number of data points. If the data scaling trend exists, the quadratic coefficient a approaches zero, degenerating the model to a linear log relation; otherwise, the curve exhibits a parabolic shape with a visible saturation point. We evaluate four planners under 2 pseudoexperts following simulation data scaling in Fig. 3 and select EPDMS in navhard as S. The scaling curves and fitted log-quadratic functions are shown in Fig. 4, with the residual standard deviation depicted as an error band. Additionally, we conduct extra experiments using reward scoring only in planner-based simulation data for GTRS-Dense,

Table 3. The effect of expert with simulated reward scoring on navhard using GTRS-Dense. (S1/2:per-stage EPDM scores.)

###### Backbone Real Expert Sim. Expert S1 ↑ S2 ↑ EPDMS ↑

✓ ✓ 67.1 55.8 38.3 ✓ 66.8 54.9 37.6

ResNet34

✓ ✓ ✓ ✓ 70.8 63.9 46.1

###### ✓ ✓ ✓ 72.4 63.4 45.9

✓ ✓ 70.4 58.5 41.9 ✓ 68.1 55.6 38.8

V2-99

✓ ✓ ✓ ✓ 71.7 65.5 47.7

###### ✓ ✓ ✓ 72.5 65.4 48.0

the value of simulation data under scaling.

Multi-modality Modeling Sparks the Scaling. Although the regression-based LTF and diffusion-based DiffusionDrive have comparable model sizes (56M vs 61M), they exhibit markedly different scaling properties for the plannerbased in Fig. 4 (a) and (b). For LTF, performance saturates and starts to degrade when the simulation-to-real ratio reaches 1:1, whereas DiffusionDrive exhibits an approximately linear improvement. This is due to the gradually increasing diversity of demonstrations from the same real scenario, which introduces an effectively multi-modal supervision problem. Single-mode regression struggles to model multi-peak distributions, leading to mode confusion and performance degradation, while diffusion models can capture multimodality, making them amenable to optimization under diverse supervision [11]. Since real-world autonomous driving is inherently a multi-peak problem [34],

- our simulation-scaling results underscore the importance of multimodal modeling for scalable real-world end-to-end autonomous driving.

Reward is All You Need. In Fig. 4 (c) and (d), for the scoring-based GTRS-Dense planner, reward signals alone, without expert trajectories in simulation, yields even better performance. To further analyze this, we conduct rewardonly training on real-world data only, which instead leads to performance degradation, as shown in Tab. 3. These results indicate that with sufficient expert supervision to stabilize the optimization direction, the reward guidance is better. The model benefits from the feedback from rewards during its exploration and interaction within the environment [14]. Effect of Reactive Simulation. To isolate the effect of reactive traffic, we compare reward-scoring GTRS-Dense on navhard using non-reactive vs. reactive simulation data (Tab. 4). Two rounds of non-reactive sampling yield 141K trajectories, i.e., 21K more valid samples due to fewer collision rates, yet provide no performance improvement in EPDMS. When reactive simulation reaches the third round, it produces 167K samples yet delivers consistent and significant EPDMS gains across both model sizes. These results

Table 4. The effect between non-reactive vs. reactive data simulation on navhard using GTRS-Dense, across sampling rounds.

EPDMS↑

Type #Round #Sim.

ResNet34 V2-99

Non-Reactive 2 141K 43.7 45.6

- Reactive 2 120K 44.4 46.7

- Reactive 3 167K 45.0 47.9

indicate that reactive agent dynamics enhance the realism and diversity of traffic interactions, thereby increasing the effectiveness of simulation data. Detailed scaling curves are shown in Sec. C.3 in the supplementary material.

#### 3.4. Qualitative Results of the Simulation Scenes

We present qualitative visualizations from the simulation data in Fig. 5, showcasing four representative OOD scenarios used to train the policy. These scenarios mirror typical real-world driving challenges where learned policies tend to struggle, including (a) off-center lane drift, (b) near collision, (c) departure, and (d) cutting in cases. Each scenario is illustrated with a top-down view showing the pseudo-expert trajectory as supervision, and the deviating Perturbed Trajectory as history actions, alongside the synthetic front-view as sensory input to the policy. For instance, scenario (b) requires the policy to adaptively avoid the collision in short horizon. See supplementary Sec. C.2 for extended results.

### 4. Conclusion

In this paper, we introduce SimScale, a sim–real learning system that reveals how scalable simulation can amplify the value of real-world datasets for end-to-end autonomy. For the simulation data pipeline, we first generate pseudo-expert demonstrations from potential OOD states by ego perturbation within reactive environments. Toward the real-world simulation, the associated high-fidelity multi-view observations are rendered with 3DGS engine. Upon the simulated data, sim-real co-training produces synergistic improvements in robustness and generalization for various planners on challenging real-world benchmarks, up to +8.6 EPDMS on navhard and +2.9 navtest. Remarkably, the sim–real system scales clearly and predictably with increased simulation while keeping the real-world corpus fixed. We further uncover that exploration and interaction contribute to more effective simulation, and that multi-modal planners strengthen their scaling behavior. We hope SimScale will inspire the community to further explore real-world simulation for data scaling.

We provide additional experiments in the supplementary material, such as multi-expert ensemble and scaling with varying real data. It also provides extended discussions on related work, limitations, and broader impact.

### Acknowledgments

This study is supported by the National Key R&D Program of China (2022ZD0117901), the National Natural Science Foundation of China (Grant No. 62373355, 62236010), and the Beijing Natural Science Foundation (Grant L252033). This work is in part supported by the JC STEM Lab of Autonomous Intelligent Systems funded by The Hong Kong Jockey Club Charities Trust. We also appreciate the gener-

- ous research sponsor from Xiaomi. We extend our gratitude to Naiyan Wang, Yunsong Zhou,

Hanxue Zhang, and the rest of the members from OpenDriveLab and Xiaomi Embodied Intelligence Team for their profound support.

### References

- [1] Hassan Abu Alhaija, Jose Alvarez, Maciej Bala, Tiffany Cai, Tianshi Cao, Liz Cha, Joshua Chen, Mike Chen, Francesco Ferroni, Sanja Fidler, et al. Cosmos-transfer1: Conditional world generation with adaptive multimodal control. arXiv preprint arXiv:2503.14492, 2025. 1, 5
- [2] Mustafa Baniodeh, Kratarth Goel, Scott Ettinger, Carlos Fuertes, Ari Seff, Tim Shen, Cole Gulino, Chenjie Yang, Ghassen Jerfel, Dokook Choe, et al. Scaling laws of motion forecasting and planning–a technical report. arXiv preprint arXiv:2506.08228, 2025. 2
- [3] Jens Beißwenger. Pdm-lite: A rule-based planner for carla leaderboard 2.0. Univ. T¨ubingen, 2024. 4
- [4] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alex Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: Visionlanguage-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023. 4
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020. 2
- [6] Wei Cao, Marcel Hallgarten, Tianyu Li, Daniel Dauner, Xunjiang Gu, Caojun Wang, Yakov Miron, Marco Aiello, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, and Kashyap Chitta. Pseudosimulation for autonomous driving. In CoRL, 2025. 2, 5, 1, 6

- [7] Sergio Casas, Abbas Sadat, and Raquel Urtasun. Mp3: A unified model to map, perceive, predict and plan. In CVPR,

2021. 1

- [8] David Charatan, Sizhe Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In CVPR,

2024. 6

- [9] Li Chen, Penghao Wu, Kashyap Chitta, Bernhard Jaeger, Andreas Geiger, and Hongyang Li. End-to-end autonomous driving: Challenges and frontiers. TPAMI, 2024. 2, 6, 1
- [10] Ziyu Chen, Jiawei Yang, Jiahui Huang, Riccardo de Lutio, Janick Martinez Esturo, Boris Ivanovic, Or Litany, Zan Gojcic, Sanja Fidler, Marco Pavone, Li Song, and Yue Wang. Omnire: Omni urban scene reconstruction. In ICLR, 2025. 1
- [11] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. RSS, 2025. 8
- [12] Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. TPAMI, 2022. 2, 4, 5, 6, 1, 3
- [13] Daphne Cornelisse, Aarav Pandya, Kevin Joseph, Joseph Su´arez, and Eugene Vinitsky. Building reliable sim driving agents by scaling self-play. arXiv preprint arXiv:2502.14706, 2025. 2, 6
- [14] Marco Cusumano-Towner, David Hafner, Alex Hertzberg, Brody Huval, Aleksei Petrenko, Eugene Vinitsky, Erik Wijmans, Taylor Killian, Stuart Bowers, Ozan Sener, et al. Robust autonomy emerges from self-play. In ICML, 2025. 8, 2, 6
- [15] Chenxu Dang, Sining Ang, Yongkang Li, Haochen Tian, Jie Wang, Guang Li, Hangjun Ye, Jie Ma, Long Chen, and Yan Wang. Drivefine: Refining-augmented masked diffusion vla for precise and robust driving. arXiv preprint arXiv:2602.14577, 2026. 1
- [16] Daniel Dauner, Marcel Hallgarten, Andreas Geiger, and Kashyap Chitta. Parting with misconceptions about learning-based vehicle motion planning. In CoRL, 2023. 2, 5, 6, 3
- [17] Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, and Kashyap Chitta. Navsim: Data-driven nonreactive autonomous vehicle simulation and benchmarking. In NeurIPS, 2024. 2, 5
- [18] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. CARLA: An open urban driving simulator. In CoRL, 2017. 1
- [19] Lan Feng, Yang Gao, Eloi Zablocki, Quanyi Li, Wuyang Li, Sichao Liu, Matthieu Cord, and Alexandre Alahi. Rap: 3d rasterization augmented end-to-end planning. arXiv preprint arXiv:2510.04333, 2025. 2
- [20] Hao Gao, Shaoyu Chen, Bo Jiang, Bencheng Liao, Yiang Shi, Xiaoyang Guo, Yuechuan Pu, Haoran Yin, Xiangyu Li, Xinbang Zhang, Ying Zhang, Wenyu Liu, Qian Zhang,

- and Xinggang Wang. Rad: Training an end-to-end driving policy via large-scale 3dgs-based reinforcement learning. In NeurIPS, 2025. 2, 1
- [21] Yinfeng Gao, Deqing Liu, Qichao Zhang, Yupeng Zheng, Haochen Tian, Guang Li, Hangjun Ye, Long Chen, Da-Wei Ding, and Dongbin Zhao. Learning from mistakes: Posttraining for driving vla with takeover data. arXiv preprint arXiv:2603.14972, 2026.
- [22] Yinfeng Gao, Qichao Zhang, Deqing Liu, Zhongpu Xia, Guang Li, Kun Ma, Guang Chen, Hangjun Ye, Long Chen, Da-Wei Ding, and Dongbin Zhao. Perlad: Towards enhanced closed-loop end-to-end autonomous driving with pseudo-simulation-based reinforcement learning. arXiv preprint arXiv:2603.14908, 2026. 1
- [23] Mitchell Goff, Greg Hogan, George Hotz, Armand du Parc Locmaria, Kacper Raczy, Harald Sch¨afer, Adeeb Shihadeh, Weixing Zhang, and Yassine Yousfi. Learning to drive from a world model. In CVPR, 2025. 2
- [24] Ke Guo, Haochen Liu, Xiaojun Wu, Jia Pan, and Chen Lv. ipad: Iterative proposal-centric end-to-end autonomous driving. arXiv preprint arXiv:2505.15111, 2025. 1
- [25] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR,

2016. 2, 4

- [26] Shengchao Hu, Li Chen, Penghao Wu, Hongyang Li, Junchi Yan, and Dacheng Tao. St-p3: End-to-end visionbased autonomous driving via spatial-temporal feature learning. In ECCV, 2022. 1
- [27] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In CVPR, 2023. 4, 1
- [28] Xin Huang, Eric M Wolff, Paul Vernaza, Tung Phan-Minh, Hongge Chen, David S Hayden, Mark Edmonds, Brian Pierce, Xinxin Chen, Pratik Elias Jacob, et al. Drivegpt: Scaling autoregressive behavior models for driving. arXiv preprint arXiv:2412.14415, 2024. 2
- [29] Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung, Jingwei Ji, Kristy Choi, Di Huang, Tong He, Paul Covington, Benjamin Sapp, et al. Emma: End-to-end multimodal model for autonomous driving. TMLR, 2025. 2, 1
- [30] Bernhard Jaeger, Daniel Dauner, Jens Beißwenger, Simon Gerstenecker, Kashyap Chitta, and Andreas Geiger. Carl: Learning scalable planning policies with simple rewards. In CoRL, 2025. 2, 6
- [31] Xiaosong Jia, Penghao Wu, Li Chen, Jiangwei Xie, Conghui He, Junchi Yan, and Hongyang Li. Think twice before driving: Towards scalable decoders for end-to-end autonomous driving. In CVPR, 2023. 1
- [32] Xiaosong Jia, Junqi You, Zhiyuan Zhang, and Junchi Yan. Drivetransformer: Unified transformer for scalable end-toend autonomous driving. In ICLR, 2025. 1
- [33] Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In ICCV, 2023.

- [34] Bo Jiang, Shaoyu Chen, Hao Gao, Bencheng Liao, Qian Zhang, Wenyu Liu, and Xinggang Wang. VADv2: Endto-end autonomous driving via probabilistic planning. In ICLR, 2026. 4, 8, 1
- [35] Junzhe Jiang, Nan Song, Jingyu Li, Xiatian Zhu, and Li Zhang. Realengine: Simulating autonomous driving in realistic context. arXiv preprint arXiv:2505.16902, 2025. 1
- [36] Max Jiang, Yijing Bai, Andre Cornman, Christopher Davis, Xiukun Huang, Hong Jeon, Sakshum Kulshrestha, John Lambert, Shuangyu Li, Xuanyu Zhou, et al. Scenediffuser: Efficient and controllable driving simulation initialization and rollout. In NeurIPS, 2024. 1, 6
- [37] Siwen Jiao, Kangan Qian, Hao Ye, Yang Zhong, Ziang Luo, Sicong Jiang, Zilin Huang, Yangyi Fang, Jinyu Miao, Zheng Fu, et al. EvaDrive: Evolutionary adversarial policy optimization for end-to-end autonomous driving. arXiv preprint arXiv:2508.09158, 2025. 1
- [38] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361,

2020. 2, 1

- [39] Napat Karnchanachari, Dimitris Geromichalos, Kok Seang Tan, Nanxiang Li, Christopher Eriksen, Shakiba Yaghoubi, Noushin Mehdipour, Gianmarco Bernasconi, Whye Kit Fong, Yiluan Guo, et al. Towards learning-based planning: The nuplan benchmark for real-world autonomous driving. In ICRA, 2024. 5, 2, 6
- [40] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. TOG, 2023. 2, 3
- [41] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015. 2
- [42] Youngwan Lee and Jongyoul Park. Centermask: Real-time anchor-free instance segmentation. In CVPR, 2020. 2, 4
- [43] Norman Lehtomaki, NJAM Sandell, and Michael Athans. Robustness results in linear-quadratic gaussian based multivariable control designs. TAC, 2003. 3
- [44] Bohan Li, Jiazhe Guo, Hongsi Liu, Yingshuang Zou, Yikang Ding, Xiwu Chen, Hu Zhu, Feiyang Tan, Chi Zhang, Tiancai Wang, et al. Uniscene: Unified occupancycentric driving scene generation. In CVPR, 2025. 6
- [45] Derun Li, Jianwei Ren, Yue Wang, Xin Wen, Pengxiang Li, Leimeng Xu, Kun Zhan, Zhongpu Xia, Peng Jia, Xianpeng Lang, et al. Finetuning generative trajectory model with reinforcement learning from human feedback. arXiv preprint arXiv:2503.10434, 2025. 1
- [46] Hongchen Li, Tianyu Li, Jiazhi Yang, Haochen Tian, Caojun Wang, Lei Shi, Mingyang Shang, Zengrong Lin, Gaoqiang Wu, Zhihui Hao, et al. Plannerrft: Reinforcing diffusion planners through closed-loop and sample-efficient fine-tuning. arXiv preprint arXiv:2601.12901, 2026. 2
- [47] Kailin Li, Zhenxin Li, Shiyi Lan, Yuan Xie, Zhizhong Zhang, Jiayi Liu, Zuxuan Wu, Zhiding Yu, and Jose M Alvarez. Hydra-mdp++: Advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820, 2025. 5

- [48] Quanyi Li, Zhenghao Peng, Lan Feng, Qihang Zhang, Zhenghai Xue, and Bolei Zhou. Metadrive: Composing diverse driving scenarios for generalizable reinforcement learning. TPAMI, 2022. 1
- [49] Qifeng Li, Xiaosong Jia, Shaobo Wang, and Junchi Yan. Think2drive: Efficient reinforcement learning by thinking with latent world model for autonomous driving (in carlav2). In ECCV, 2024. 2
- [50] Shihao Li, Naisheng Ye, Tianyu Li, Kashyap Chitta, Tuo An, Peng Su, Boyang Wang, Haiou Liu, Chen Lv, and Hongyang Li. Optimization-guided diffusion for interactive scene generation. arXiv preprint arXiv:2512.07661, 2025. 6
- [51] Tianyu Li, Yihang Qiu, Zhenhua Wu, Carl Lindstr¨om, Peng Su, Matthias Nießner, and Hongyang Li. MTGS: Multi-traversal gaussian splatting. arXiv preprint arXiv:2503.12552, 2025. 3, 6
- [52] Xiaofan Li, Yifu Zhang, and Xiaoqing Ye. Drivingdiffusion: layout-guided multi-view driving scenarios video generation with latent diffusion model. In ECCV, 2024. 1
- [53] Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, et al. Drivevla-w0: World models amplify data scaling law in autonomous driving. arXiv preprint arXiv:2510.12796, 2025. 1
- [54] Yingyan Li, Yuqi Wang, Yang Liu, Jiawei He, Lue Fan, and Zhaoxiang Zhang. End-to-end driving with online trajectory evaluation via bev world model. arXiv preprint arXiv:2504.01941, 2025. 1
- [55] Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, et al. ReCogDrive: A reinforced cognitive framework for end-to-end autonomous driving. In ICLR,

2026. 1

- [56] Zhenxin Li, Kailin Li, Shihao Wang, Shiyi Lan, Zhiding Yu, Yishen Ji, Zhiqi Li, Ziyue Zhu, Jan Kautz, Zuxuan Wu, et al. Hydra-mdp: End-to-end multimodal planning with multi-target hydra-distillation. arXiv preprint arXiv:2406.06978, 2024. 2, 4
- [57] Zhenxin Li, Shihao Wang, Shiyi Lan, Zhiding Yu, Zuxuan Wu, and Jose M Alvarez. Hydra-next: Robust closed-loop driving with open-loop training. In ICCV, 2025. 1
- [58] Zhenxin Li, Wenhao Yao, Zi Wang, Xinglong Sun, Jingde Chen, Nadine Chang, Maying Shen, Jingyu Song, Zuxuan Wu, Shiyi Lan, et al. Ztrs: Zero-imitation end-to-end autonomous driving with trajectory scoring. arXiv preprint arXiv:2510.24108, 2025. 1
- [59] Zhenxin Li, Wenhao Yao, Zi Wang, Xinglong Sun, Joshua Chen, Nadine Chang, Maying Shen, Zuxuan Wu, Shiyi Lan, and Jose M Alvarez. Generalized trajectory scoring for end-to-end multimodal planning. arXiv preprint arXiv:2506.06664, 2025. 2, 5, 6, 1, 3, 4
- [60] Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. DiffusionDrive: Truncated diffusion model for end-to-end autonomous driving. In CVPR,

2025. 2, 4, 5, 6, 1, 3

- [61] Haochen Liu, Zhiyu Huang, Wenhui Huang, Haohan Yang, Xiaoyu Mo, and Chen Lv. Hybrid-prediction integrated planning for autonomous driving. TPAMI, 2025. 1
- [62] Haochen Liu, Tianyu Li, Haohan Yang, Li Chen, Caojun Wang, Ke Guo, Haochen Tian, Hongchen Li, Hongyang Li, and Chen Lv. Reinforced refinement with self-aware expansion for end-to-end autonomous driving. arXiv preprint arXiv:2506.09800, 2025. 2, 6, 1
- [63] Kunhao Liu, Ling Shao, and Shijian Lu. Novel view extrapolation with video diffusion priors. arXiv preprint arXiv:2411.14208, 2024. 4
- [64] Xi Liu, Chaoyi Zhou, and Siyu Huang. 3dgs-enhancer: Enhancing unbounded 3d gaussian splatting with viewconsistent 2d diffusion priors. In NeurIPS, 2024. 4
- [65] William Ljungbergh, Adam Tonderski, Joakim Johnander, Holger Caesar, Kalle Astr¨˚ om, Michael Felsberg, and Christoffer Petersson. Neuroncap: Photorealistic closedloop safety testing for autonomous driving. In ECCV, 2024. 1
- [66] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 2
- [67] Abhiram Maddukuri, Zhenyu Jiang, Lawrence Yunliang Chen, Soroush Nasiriany, Yuqi Xie, Yu Fang, Wenqi Huang, Zu Wang, Zhenjia Xu, Nikita Chernyadev, Scott Reed, Ken Goldberg, Ajay Mandlekar, Linxi Fan, and Yuke Zhu. Sim-and-real co-training: A simple recipe for visionbased robotic manipulation. In RSS, 2025. 2, 4
- [68] Alexander Naumann, Xunjiang Gu, Tolga Dimlioglu, Mariusz Bojarski, Alperen Degirmenci, Alexander Popov, Devansh Bisla, Marco Pavone, Urs Muller, and Boris Ivanovic. Data scaling laws for end-to-end autonomous driving. In CVPR, 2025. 2, 1
- [69] NVIDIA, Nikita Cherniadev Johan Bjorck andFernando Casta˜neda, Xingye Da, Runyu Ding, Linxi ”Jim” Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. GR00T N1: An open foundation model for generalist humanoid robots. arxiv preprint arxiv:2503.14734, 2025. 2, 4
- [70] OpenScene Contributors. OpenScene: The largest up-todate 3d occupancy prediction benchmark in autonomous driving. https://github.com/OpenDriveLab/ OpenScene, 2023. 6
- [71] Alexander Popov, Alperen Degirmenci, David Wehr, Shashank Hegde, Ryan Oldja, Alexey Kamenev, Bertrand Douillard, David Nist´er, Urs Muller, Ruchi Bhargava, et al. Mitigating covariate shift in imitation learning for autonomous vehicles using latent space generative world models. arXiv preprint arXiv:2409.16663, 2024. 2
- [72] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,

- Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 1
- [73] Yang Shi, Yuhao Dong, Yue Ding, Yuran Wang, Xuanyu Zhu, Sheng Zhou, Wenting Liu, Haochen Tian, Rundong Wang, Huanqian Wang, et al. Realunify: Do unified models truly benefit from unification? a comprehensive benchmark. arXiv preprint arXiv:2509.24897, 2025. 6
- [74] Wenchao Sun, Xuewu Lin, Yining Shi, Chuang Zhang, Haoran Wu, and Sifa Zheng. Sparsedrive: End-to-end autonomous driving via sparse scene representation. In ICRA,

2025. 1

- [75] Tianyi Tan, Yinan Zheng, Ruiming Liang, Zexu Wang, Kexin Zheng, Jinliang Zheng, Jianxiong Li, Xianyuan Zhan, and Jingjing Liu. Flow matching-based autonomous driving planning with advanced interactive behavior modeling. In NeurIPS, 2025. 6
- [76] Martin Treiber, Ansgar Hennecke, and Dirk Helbing. Congested traffic states in empirical observations and microscopic simulations. Physical review E, 2000. 2, 3, 1, 6
- [77] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards realworld-drive world models for autonomous driving. In ECCV, 2024. 1
- [78] Xinshuo Weng, Boris Ivanovic, Yan Wang, Yue Wang, and Marco Pavone. Para-drive: Parallelized architecture for real-time autonomous driving. In CVPR, 2024. 1
- [79] Penghao Wu, Xiaosong Jia, Li Chen, Junchi Yan, Hongyang Li, and Yu Qiao. Trajectory-guided control prediction for end-to-end autonomous driving: A simple yet strong baseline. In NeurIPS, 2022. 1
- [80] Yanhao Wu, Haoyang Zhang, Tianwei Lin, Lichao Huang, Shujie Luo, Rui Wu, Congpei Qiu, Wei Ke, and Tong Zhang. Generating multimodal driving scenes via nextscene prediction. In CVPR, 2025. 6
- [81] Chengen Xie, Bin Sun, Tianyu Li, Junjie Wu, Zhihui Hao, XianPeng Lang, and Hongyang Li. Latentvla: Efficient vision-language models for autonomous driving via latent action prediction. arXiv preprint arXiv:2601.05611, 2026. 1
- [82] Ziyang Xie, Zhizheng Liu, Zhenghao Peng, Wayne Wu, and Bolei Zhou. Vid2sim: Realistic and interactive simulation from video for urban navigation. In CVPR, 2025. 1
- [83] Zebin Xing, Xingyu Zhang, Yang Hu, Bo Jiang, Tong He, Qian Zhang, Xiaoxiao Long, and Wei Yin. Goalflow: Goaldriven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In CVPR, 2025. 4, 1
- [84] Runsheng Xu, Hubert Lin, Wonseok Jeon, Hao Feng, Yuliang Zou, Liting Sun, John Gorman, Kate Tolstaya, Sarah Tang, Brandyn White, et al. Wod-e2e: Waymo open dataset for end-to-end driving in challenging long-tail scenarios. arXiv preprint arXiv:2510.26125, 2025. 2
- [85] Zhiyuan Xu, Bohan Li, Huan-ang Gao, Mingju Gao, Yong Chen, Ming Liu, Chenxu Yan, Hang Zhao, Shuo Feng, and Hao Zhao. Challenger: Affordable adversarial driving video generation. arXiv preprint arXiv:2505.15880, 2025.

- [86] Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, and Sida Peng. Street gaussians: Modeling dynamic urban scenes with gaussian splatting. In ECCV, 2024. 3, 1
- [87] Jiazhi Yang, Kashyap Chitta, Shenyuan Gao, Long Chen, Yuqian Shao, Xiaosong Jia, Hongyang Li, Andreas Geiger, Xiangyu Yue, and Li Chen. Resim: Reliable world simulation for autonomous driving. In NeurIPS, 2025. 2, 4
- [88] Jiawei Yang, Jiahui Huang, Yuxiao Chen, Yan Wang, Boyi Li, Yurong You, Apoorva Sharma, Maximilian Igl, Peter Karkus, Danfei Xu, et al. Storm: Spatio-temporal reconstruction model for large-scale outdoor scenes. In ICLR,

2025. 6

- [89] Yan Yang, Haochen Tian, Yang Shi, Wulin Xie, Yi-Fan Zhang, Yuhao Dong, Yibo Hu, Liang Wang, Ran He, Caifeng Shan, et al. A survey of unified multimodal understanding and generation: Advances and challenges. Authorea Preprints, 2025. 6
- [90] Yifan Ye, Jun Cen, Jing Chen, and Zhihe Lu. Self-evolved imitation learning in simulated world. arXiv preprint arXiv:2509.19460, 2025. 6
- [91] Junqi You, Xiaosong Jia, Zhiyuan Zhang, Yutao Zhu, and Junchi Yan. Bench2drive-r: Turning real world data into reactive closed-loop autonomous driving benchmark by generative model. arXiv preprint arXiv:2412.09647, 2024. 1
- [92] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In CVPR, 2022. 2, 1
- [93] Chris Zhang, Sourav Biswas, Kelvin Wong, Kion Fallah, Lunjun Zhang, Dian Chen, Sergio Casas, and Raquel Urtasun. Learning to drive via asymmetric self-play. In ECCV,

2024. 2

- [94] Huanyu Zhang, Chengzu Li, Wenshan Wu, Shaoguang Mao, Yifan Zhang, Haochen Tian, Ivan Vuli´c, Zhang Zhang, Liang Wang, Tieniu Tan, et al. Scaling and beyond: Advancing spatial reasoning in mllms requires new recipes. arXiv preprint arXiv:2504.15037, 2025. 6
- [95] Huanyu Zhang, Wenshan Wu, Chengzu Li, Ning Shang, Yan Xia, Yangyu Huang, Yifan Zhang, Li Dong, Zhang Zhang, Liang Wang, et al. Latent sketchpad: Sketching visual thoughts to elicit multimodal reasoning in mllms. arXiv preprint arXiv:2510.24514, 2025. 6
- [96] YiFan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans? In ICLR, 2025. 1
- [97] Zhejun Zhang, Alexander Liniger, Dengxin Dai, Fisher Yu, and Luc Van Gool. End-to-end urban driving by imitating a reinforcement learning coach. In ICCV, 2021. 4
- [98] Yupeng Zheng, Zhongpu Xia, Qichao Zhang, Teng Zhang, Ben Lu, Xiaochuang Huo, Chao Han, Yixian Li, Mengjie Yu, Bu Jin, et al. Preliminary investigation into data scaling laws for imitation learning-based end-to-end autonomous driving. arXiv preprint arXiv:2412.02689, 2024. 2, 1
- [99] Yinan Zheng, Ruiming Liang, Kexin ZHENG, Jinliang Zheng, Liyuan Mao, Jianxiong Li, Weihao Gu, Rui Ai,

- Shengbo Eben Li, Xianyuan Zhan, and Jingjing Liu. Diffusion-based planning for autonomous driving with flexible guidance. In ICLR, 2025. 6
- [100] Hongyu Zhou, Longzhong Lin, Jiabao Wang, Yichong Lu, Dongfeng Bai, Bingbing Liu, Yue Wang, Andreas Geiger, and Yiyi Liao. Hugsim: A real-time, photo-realistic and closed-loop simulator for autonomous driving. arXiv preprint arXiv:2412.01718, 2024. 1
- [101] Jiawei Zhou, Linye Lyu, Zhuotao Tian, Cheng Zhuo, and Yu Li. Safemvdrive: Multi-view safety-critical driving video synthesis in the real world domain. arXiv preprint arXiv:2505.17727, 2025. 1
- [102] Yunsong Zhou, Michael Simon, Zhenghao Peng, Sicheng Mo, Hongzi Zhu, Minyi Guo, and Bolei Zhou. Simgen: Simulator-conditioned driving scene generation. In NeurIPS, 2024. 1
- [103] Yunsong Zhou, Naisheng Ye, William Ljungbergh, Tianyu Li, Jiazhi Yang, Zetong Yang, Hongzi Zhu, Christoffer Petersson, and Hongyang Li. Decoupled diffusion sparks adaptive scene generation. In ICCV, 2025. 1, 6

## SIMSCALE: Learning to Drive via Real-World Simulation at Scale Supplementary Material

##### A. Related Work 1

- A.1. End-to-End Autonomous Driving . . . . . . . . 1
- A.2. Scene Simulation for Driving. . . . . . . . . . . 1
- A.3. Data Scaling for Driving. . . . . . . . . . . . . . 1
- A.4. Comparison with Online RL . . . . . . . . . . . 2

##### B. Extended Implementation Details 2

- B.1. Simulation Data Curation . . . . . . . . . . . . . 2
- B.2. Models and Training . . . . . . . . . . . . . . . . 2

##### C. Extended Experimental Results 4

- C.1. Detailed Leaderboard Results . . . . . . . . . . 4
- C.2. Complete Qualitative Results. . . . . . . . . . . 4
- C.3. Scaling of Reactive vs. Non-Reactive . . . . . 4

##### D. Additional Experimental Results 4

- D.1. Multi-Expert Ensemble. . . . . . . . . . . . . . . 4
- D.2. Effect of Scenarios beyond Scaling . . . . . . . 5
- D.3. Effect of Simulation Visual Fidelity . . . . . . 5
- D.4. Scaling with Varying Scales of Real Data. . . 5

##### E. Limitations and Broader Impact 6

- E.1. Pseudo-Expert . . . . . . . . . . . . . . . . . . . . 6
- E.2. Scene Simulation. . . . . . . . . . . . . . . . . . . 6
- E.3. Self-Play . . . . . . . . . . . . . . . . . . . . . . . . 6
- E.4. Societal Impact . . . . . . . . . . . . . . . . . . . . 6

##### F. License of Assets 6

We outline the supplementary material as follows. In Sec. A, we first provide additional discussions of related work. Sec. B presents further implementation details regarding data curation and models. In Sec. C, we report extended experimental analyses and qualitative results corresponding to the main experiments (Sec. 3). In Sec. D, we further present additional experimental studies. Finally, Sec. E discusses the limitations and broader impacts, and Sec. F lists the licenses of all utilized assets.

### A. Related Work

#### A.1. End-to-End Autonomous Driving

End-to-end system maps directly from raw sensor inputs to planning [9]. Early works adopt regression-based planning and gradually shift from extra task branches [12, 31, 79] to unifying perception, prediction, and planning under joint supervision [27, 32, 33, 61, 78]. Recent work has moved toward generative approaches. Diffusion-based systems [15, 45, 55, 60, 83] are framed as a conditional denoising process, enabling diverse and high-fidelity trajectories.

Concurrently, trajectory scoring has emerged as an efficient alternative, ranking candidate trajectories under spline curves [7, 26], discretized tokens [20, 34], clustered human trajectories [57, 59], or predicted proposals [24, 54, 74, 81]. Moreover, it offers a natural interface for reward- or costbased optimization [20–22, 37, 55, 58, 62], enabling reinforced improvements. Still, supervision on logged end-toend data limits training to the expert’s open-loop distribution, causing compromised learning in the drifted state of sensory data. We address this by introducing a scalable simulation framework that reactively generates end-to-end pairs using pseudo-experts or rewards, allowing end-to-end systems across any above paradigms to bootstrap extra supervision from existing training data.

#### A.2. Scene Simulation for Driving

Scene simulation, including traffic behavior simulation and sensor simulation, has long been a key topic in autonomous driving research. For traffic simulation, existing works [6, 20, 100] leverage rule-based planners like IDM [76], or diffusion-based generators [36, 103] to simulate plausible interaction of traffic agents. For sensor simulation, traditional graphics-based simulators [18, 48] suffer from a significant sim-to-real gap, which limits the real-world deployment of trained planners. Recent data-driven efforts follow two directions: some approaches [1, 52, 77, 85, 91, 101, 102] attempt to generate sensor data in unseen scenarios via video generation models conditioned on 3D bounding boxes, HD maps or BEV maps. Other works focus on scene reconstruction [6, 10, 35, 65, 82, 86, 100] to build photorealistic simulators for novel-view synthesis, using techniques like Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). Although these methods achieve impressive visual results, most are primarily designed for closed-loop evaluation or visual augmentation. Our work aims to explore how to use traffic and sensor simulation to generate realistic scenes with feasible demonstrations for planner training, and its impacts on planner performances.

#### A.3. Data Scaling for Driving

Recent data scaling laws have driven major advances in foundation models [38, 72, 92, 96], but their impact on autonomous driving planning remains underexplored. Prior researches [29, 68, 98] demonstrate that increasing realworld data from thousands to millions of driving logs improves end-to-end planner performance, though improvements diminish. Dense supervision from video predictions improves this data scaling efficiency [53]. For planners operating on abstract BEV representations, industry efforts

[2, 28] demonstrate clear benefits from scaling real-world data and model capacity, while self-play [13, 14, 93] scales reinforcement learning via massive simulation to achieve strong zero-shot sim-to-real transfer. However, most existing approaches rely on costly real-world collection or traffic simulation, which expands only in abstract state spaces rather than raw sensory domains. 3D rasterization provides a lightweight remedy, but suffers from information loss [19]. In contrast, we investigate scaling properties of planning directly through large-scale 3DGS sensory simulation, which bridges abstract traffic simulation and realworld perception, offering a scalable and realistic alternative to real-world data collection.

- A.4. Comparison with Online RL

[Figure 127]

[Figure 128]

[Figure 129]

Action

[Figure 130]

RL

Update x M

[Figure 131]

[Figure 132]

IL / RL

Perturb x N

Policy K

Human State

[Figure 133]

x N

Policy K+1

ANY Policy

Pseudo Expert

- (a) 3DGS-based Online RL
- (b) SimScale Simulation Pipeline

[Figure 134]

Figure 6. Learning paradigm comparison of e2e autonomous driving between 3DGS-based Online RL and SimScale.

Online reinforcement learning (RL) [30, 46, 49] in traffic simulators learn through exploration and feedback. Sensor simulation serves as a bridge to real-world environments, enabling real-world RL for autonomous driving, i.e., 3DGSbased Online RL [20], as shown in Fig. 6 (a). In contrast, the proposed SimScale introduces an alternative framework, as illustrated in Fig. 6 (b), that generates OOD states with expert demonstrations at scale, enabling real-world simulations to support both imitation learning (IL) and reward learning (i.e. Offline RL) across arbitrary planners.

- B. Extended Implementation Details

- B.1. Simulation Data Curation

Tab. 5 summarizes the detailed terms with notation and value used in the simulation data curation process to sup-

Table 5. Simulation Data Curation Pipeline Configurations.

Term Notation Value 3DGS Data Engine Peak signal-to-noise ratio of GS PSNR < 27 Trajectory Perturbation Clustered human traj vocabulary Vc 16, 384 Relative heading range to log ∆θ ±20◦ Longitudinal shift range rlon ±20m Lateral shift range rlat ±2m Longitudinal sampling step δlon 5m Lateral sampling step δlat 0.5m Reward filterer RperEPDMS ≥ 0.8 Pseudo-Expert Trajectory Generation Reward filterer RexpEPDMS SSm=1,∀m̸=EP

EP>0.5

Recovery-based Whole human traj vocabulary Vh 103, 288 Planner-based Privileged planner P PDM-Closed [16]

###### Table 6. Model and Training Hyperparameters.

Hyperpara. LTF [12] DiffusionDrive [60] GTRS-Dense [59] Model Configuration Sensors 3 × Cam. 3 × Cam. 3 × Cam. Resolution 2048 × 512 2048 × 512 2048 × 512 Horizon 4s 4s 4s Frequency 2Hz 2Hz 10Hz Backbone R34 [25] R34 [25] R34 [25] / V99 [42] Parameters 56M 61M 67M / 83M Aux. Tasks Det. Seg. Det. Seg. None Training Configuration GPUs 8 × H20 8 × H20 32 × H20 Epochs 100 100 50 Total BS 512 512 352 Initial LR 2.83 × 10−4 6 × 10−4 4 × 10−4 Schedule Constant Cosine Decay Cosine Decay Optimizer Adam [41] AdamW [66] AdamW [66]

plement Sec. 3.1 curation pipeline in the main paper. For pseudo-expert trajectory generation, we discard infeasible candidates to ensure valid supervision. Specifically, all submetrics of EPDMS must be satisfied, with SEP relaxed: EP ≥ 0.5, preventing biased driving styles. Qualitative results during curation are shown in Sec. C.2

#### B.2. Models and Training

Tab. 6 provides the detailed model and training hyperparameters that complement Sec. 3.1 implementation protocol in the main paper. We follow the NAVSIM [17] default setup and use only the train logs [39] of navtrain for both real-data training and sim–real co-training. All planners share identical settings across the real or sim-real training setups, and each model is trained to saturation to ensure the validity of the subsequent scaling analyses.

LTF. [12] A regression-based planner that directly regresses future waypoints upon fused sensory latents. It employs

Table 7. Detailed Results on navhard. PDM-Closed uses ground-truth symbolic inputs for planning, while other methods rely on sensor data. (recovery / planner: recovery-based / planner-based expert; reward: reward-only scoring; S.: per-stage EPDM score.)

Method Backbone Sim. Stage NC↑ DAC↑ DDC↑ TLC↑ EP↑ TTC↑ LK↑ HC↑ EC↑ S.↑ EPDMS↑

S 1 S 2

94.4 88.1

98.8 90.6

100 96.3

99.5 98.5

100 100

93.5 83.1

99.3 73.7

87.7 91.5

36.0 25.4

-

PDM-Closed [16] - -

51.3

Regeression-based Planner

S 1 S 2

97.3 79.4

80.2 69.0

97.8 85.6

99.3 98.5

83.4 83.8

96.2 76.7

92.9 47.9

97.8 97.0

71.1 70.6

61.3 39.2

w/o

24.4

S 1 S 2

96.4 88.9

78.4 71.1

98.9 91.8

99.8 99.0

80.5 77.3

96.2 85.5

92.7 53.8

97.6 96.8

78.2 47.5

###### 60.7 46.3

LTF [12] ResNet34

recovery

29.8

66.3 44.8

- S 1
- S 2

96.1 85.5

85.3 66.9

99.4 91.6

99.3 99.1

84.7 93.0

94.7 81.1

93.6 58.3

97.6 95.1

77.3 42.9

planner

30.2 Diffusion-based Planner

- S 1
- S 2

96.8 80.1

86.0 72.8

98.8 84.4

99.3 98.4

84.0 85.9

95.8 76.6

96.7 46.4

97.6 96.3

79.6 72.8

66.7 40.5

w/o

27.5

###### 69.4 41.7

- S 1
- S 2

97.2 82.4

88.4 67.7

99.1 89.1

99.8 98.6

83.9 89.0

96.0 77.6

96.7 53.8

97.6 95.2

76.9 46.8

DiffusionDrive [60] ResNet34

recovery

30.4

- S 1
- S 2

97.2 82.4

88.7 72.1

99.3 92.9

99.3 98.5

82.8 92.1

96.9 80.6

98.0 60.8

97.3 95.4

59.6 31.9

67.5 46.8

planner

32.6 Scoring-based Planner

- S 1
- S 2

99.3 92.8

96.6 88.6

99.6 95.5

100 99.4

57.4 55.9

99.5 91.3

92.6 55.7

89.5 91.1

16.4 35.7

67.1 55.8

w/o

38.3

- S 1
- S 2

97.6 92.0

94.2 88.1

99.3 94.3

99.6 98.6

76.7 83.6

97.6 89.8

94.9 58.1

97.1 91.4

37.8 25.9

69.8 59.3

recovery

43.0

ResNet34

- S 1
- S 2

98.7 94.2

94.2 92.1

99.7 95.0

100 99.0

74.3 79.9

98.2 91.8

95.8 58.9

96.7 91.0

34.7 32.9

###### 70.8 63.9

planner

46.1

72.4 63.4

- S 1
- S 2

97.6 94.3

96.4 92.7

99.3 95.1

100 99.5

75.7 80.2

97.8 91.5

93.3 56.2

97.3 90.6

32.9 28.3

reward

46.9

GTRS-Dense [59]

- S 1
- S 2

98.9 89.9

94.9 90.5

99.1 94.1

100 99.3

76.1 77.6

98.4 88.5

93.8 56.0

94.9 92.0

37.8 30.2

70.4 58.5

w/o

41.9

###### 73.4 63.1

- S 1
- S 2

99.1 95.0

98.2 90.8

99.4 94.6

100 99.4

71.9 75.5

99.1 93.5

95.6 57.4

95.8 93.7

32.4 36.8

recovery

46.4

V2-99

- S 1
- S 2

99.3 95.6

97.1 91.9

99.9 95.0

100 98.9

67.2 76.7

99.3 93.7

94.0 61.9

94.4 90.8

23.6 38.0

###### 71.7 65.5

planner

47.7

- S 1
- S 2

99.3 94.9

97.6 94.3

99.4 94.5

100 99.3

71.5 79.0

99.6 92.6

96.0 57.8

95.8 93.4

30.7 30.9

72.5 65.4

reward

48.0

Table 8. Detailed Results on navtest. (recovery / planner: recovery-based / planner-based expert; reward: reward-only scoring)

Method Backbone Sim. NC↑ DAC↑ DDC↑ TLC↑ EP↑ TTC↑ LK↑ HC↑ EC↑ EPDMS↑ Human Agent - - 100 100 99.8 100 87.4 100 100 98.1 90.1 90.3

Regression-based Planner

w/o 97.7 94.0 99.3 99.8 87.2 96.7 95.5 98.3 82.9 81.5 recovery 97.9 95.1 99.5 99.8 87.7 97.2 97.2 98.3 87.1 83.6 planner 98.3 95.6 99.6 99.8 87.1 97.5 97.2 98.3 88.2 84.4

LTF [12] ResNet34

Diffusion-based Planner

w/o 98.4 95.5 99.5 99.8 87.5 97.5 96.9 98.4 87.7 84.2 recovery 98.4 96.7 99.6 99.8 87.6 97.5 97.5 98.3 87.1 85.4 planner 98.5 97.1 99.6 99.8 87.4 97.8 98.0 98.3 87.5 85.9

DiffusionDrive [60] ResNet34

Scoring-based Planner

w/o 97.6 97.5 99.0 99.9 87.9 97.0 95.9 97.5 55.9 82.3 recovery 98.2 97.9 99.5 99.9 87.2 97.8 96.3 97.6 56.5 83.4 planner 97.8 98.3 99.5 99.9 89.2 97.3 97.3 96.9 55.1 84.0 reward 98.4 98.8 99.4 99.9 88.0 98.0 96.5 97.6 58.0 84.6

ResNet34

GTRS-Dense [59]

w/o 97.6 98.5 99.5 99.9 89.5 97.2 96.8 97.2 57.2 84.0 recovery 98.3 98.7 99.4 99.9 88.9 97.9 97.1 97.6 55.9 84.5 planner 98.7 99.0 99.5 99.8 86.8 98.3 95.9 97.9 58.5 84.5 reward 97.6 98.9 99.6 99.9 89.9 97.1 97.2 97.7 58.7 84.8

V2-99

[Figure 135]

Non-Reactive Sim. Data Reactive Sim. Data

pretrained ResNet34 [25] as image encoder and is trained for 100 epochs on 8 GPUs with a total batch size of 512 and a learning rate of 2.83 × 10−4.

[Figure 136]

[Figure 137]

EPDMS

EPDMS

DiffusionDrive. [60] A multi-modal, DETR-style generative planner that iteratively denoises diverse trajectories using anchor-conditioned truncated diffusion. Each anchor adaptively queries the encoded image features. It employs pretrained ResNet34 [25] as image encoder and applies truncated diffusion with 20 clustered anchors. It is trained for 100 epochs on 8 GPUs with a total batch size of 512 and a learning rate of 6 × 10−4.

+ Sim. Data 𝑁Samples + Sim. Data 𝑁Samples

(a) GTRS-Dense-ResNet34 | 67M (b) GTRS-Dense-V2_99 | 83M

Figure 7. Scaling dynamics under reactive and non-reactive simulation using GTRS-Dense across model sizes.

the full scaling curves, demonstrating that reactive simulation data consistently achieves superior scaling performance across model sizes.

GTRS-Dense. [59] A scoring-based, DETR-style planner that ranks a dense vocabulary of clustered human trajectories using multiple supervised scoring heads. Inputs are augmented with random spatial shifts. It employs pretrained ResNet34 [25] or V2-99 [42] as image encoder with an action space of 16,384 trajectories. It is trained for 50 epochs on 32 GPUs with a total batch size of 352 and a learning rate of 4 × 10−4. We use a vocabulary of 16,384 trajectories for training. At inference time, 8,192 trajectories are used for navhard, while the full 16,384 trajectories are used for navtest.

### D. Additional Experimental Results

#### D.1. Multi-Expert Ensemble

Table 9. The effect of multi-expert ensemble on navhard using GTRS-Dense. (S1/2:Per-stage EPDM scores; recovery / planner: recovery-based / planner-based expert; reward: reward scoring only; ∗:multi-expert ensemble)

### C. Extended Experimental Results

###### Backbone Sim Expert S1 ↑ S2 ↑ EPDMS ↑

#### C.1. Detailed Leaderboard Results

67.1 55.8 38.3

In the main paper Sec. 3.2, Tab. 1 and Tab. 2 report only the best results in navhard and navtest of each planner cotraining (highlighted in blue rows ). Supplementary Tab. 7 and Tab. 8 provide full results across different pseudoexperts and the reward-only scoring method. The best results remain highlighted in blue rows , allowing direct comparison with the main paper. Planner-based experts generally outperform recovery-based experts, while scoringbased planner benefits most from reward-only method.

✓ recovery 69.8 59.3 43.0 ✓ planner 70.8 63.9 46.1 ✓ reward 72.4 63.4 46.9

ResNet34

###### ✓ ∗ 72.8 65.1 47.7

70.4 58.5 41.9

✓ recovery 73.4 63.1 46.4 ✓ planner 71.7 65.5 47.7 ✓ reward 72.5 65.4 48.0

V2-99

✓ ∗ 74.8 67.6 50.9 ∗ ✓ ∗ 75.4 66.4 50.4

#### C.2. Complete Qualitative Results

We show extended visualizations of our simulated data in Fig. 10 and Figs. 11–13, which demonstrate the high fidelity and pseudo-expert generation of our data generation pipeline. More OOD simulation scenes are shown in Fig 14. Recovery-based Expert. As shown in Fig. 10, all simulation trajectories successfully converge back to the human expert’s final waypoint.

Different pseudo-experts exhibit distinct behavioral characteristics, e.g., the recovery-based expert is more conservative, whereas the planner-based expert is more exploratory. This raises the question of whether these complementary properties can be leveraged jointly. Therefore, we conduct a simple ensemble study, multi-expert enseble as we call it, to examine their potential complementarity on navhard, as shown in Tab. 9. The scoring-based GTRS-Dense planner is chosen because it enables a straightforward ensemble: the predicted sub-scores of each trajectory in the vocabulary can be directly averaged across the three models (recoverybased, planner-based, reward-only scoring).

Planner-based Expert As shown in Figs. 11–13, the simulated feasible trajectories under perturbed states exhibit substantial variation and broader coverage across diverse traffic scenarios, including the Y-branch intersection, the dense urban avenue, and the narrow local road.

#### C.3. Scaling of Reactive vs. Non-Reactive

Ensemble Gains from Expert Diversity. As shown in Tab. 9, this simple multi-expert ensemble yields consistent

In the main paper, Tab. 4 reports only a subset of simulation rounds for GTRS-Dense. Supplementary Fig. 7 provides

[Figure 138]

[Figure 139]

stationary_in_traffic

traversing_pickup_dropoffnear_trafficcone_on_driveable

stationarylow_magnitude_speed on_pickup_dropoff

medium_magnitude_speed

traversing_traffic_light_intersection

near_pedestrian_at_pickup_dropoff near_pedestrian_on_crosswalk

near_long_vehicle on_traffic_light_intersection

Proportion

on_stopline_stop_sign

EPDMS

low_magnitude_speed

Avg 49.2 Avg 61.7

on_stopline_crosswalktraversing_intersectionhigh_magnitude_speed

Avg 49.2

on_intersection

traversing_crosswalk

EPDMS

Scenario Tags

(a) EPDMS Distribution of EgoMLP on Sim. and Real Data. (b) EPDMS of EgoMLP over the Top-20 Most Frequent Scenarios in Sim Data

Figure 8. EgoMLP EPDMS on simulation and real data. (a) EPDMS distribution under simulation versus real-world data. (b) Scenario tags ranked by EPDMS. The figures reveal the distribution shift and scenario characteristics of simulation data.

improvements for both backbones, achieving +0.8 and +2.9 EPDMS on ResNet34 and V2-99 compared with rewardonly scoring, respectively, for navhard. Further enlarging the ensemble to all six models yields no additional EPDMS gains. These indicate that, for end-to-end planning, different pseudo-experts provide strong complementary benefits—often exceeding the gains brought by different backbone structures.

with PSNR ≥ 27 consistently yields higher EPDMS under different data scales and co-training strategies. These results indicate that simulation data with higher visual fidelity helps reduce the visual gap between simulation and realworld observations, thereby improving the effectiveness of simulation data.

[Figure 140]

#### D.4. Scaling with Varying Scales of Real Data

#### D.2. Effect of Scenarios beyond Scaling

Only Real Sim-Real

[Figure 141]

[Figure 142]

Beyond data scaling, simulation data exhibits an out-ofdistribution shift relative to real-world data. We provide a preliminary analysis of these characteristics by evaluating an EgoMLP planner, trained on navtrain, on both navtest and simulation data, as shown in Fig. 8. (1) Fig. 8 (a): Sim data shows a lower EPDMS distribution than real data, confirming a higher safety-critical level. (2) Fig. 8 (b): Low EPDMS scores across nuPlan scenario tags identify challenging simulation scenarios, facilitating future study on their distinct impact, e.g., increasing their proportions during simulation.

|+ 11.5%|
|---|

|+ 16.7%|
|---|

|+ 14.9%|
|---|

|+ 9.2%|
|---|

EPDMS

###### EPDMS

|+ 16.7%|
|---|

|+ 10.8%|
|---|

|+ 22.4%|
|---|

|+ 12.1%|
|---|

Samples

Samples

1 x 105 2 x 105 5 x 105 10 x 105

1 x 105 2 x 105 5 x 105 10 x 105

+ Real Data

+ Real Data

(a) GTRS-Dense-ResNet34 | 67M (b) GTRS-Dense-V2_99 | 83M

Figure 9. Scaling simulation with varying real data. Simulation data are scaled by corresponding real data scenario tokens and fixed sim-real data ratio

#### D.3. Effect of Simulation Visual Fidelity

To assess the practical utility of our approach, we study how simulation data affects performance as the amount of available real-world data scales up. Based on GTRS-Dense with reward-only scoring sim-real co-training, we conduct experiments with different real data from navtrain (10K, 20K, 50K, 100K). Inspired by [1], simulation data are generated from identical scenarios of corresponding real data tokens, and we fix the sim-real data ratio (five sampling rounds) for each experiment.

Table 10. Effect of simulation visual fidelity on navhard using GTRS-Dense, across sampling rounds and co-training strategies.

Backbone PSNR Pseudo-Expert Reward-Only Scoring

#R 1 #R 2 #R 1 #R 2

< 27 41.0 41.1 40.8 41.6 ≥ 27 41.9 41.2 41.3 42.9

ResNet34

< 27 44.4 44.5 43.7 44.8 ≥ 27 45.4 46.1 44.9 46.3

V2-99

Sustained Simulation Gains Across Real Data Scales. As shown in Fig. 9, our simulation data provides the most significant gains when real data is scarce (10K), achieving +22.4% EPDMS on ResNet34 and +12.1% EPDMS on V2-99 for navhard. As the amount of real data increases (from 10K to 100K), these gains remain consistently high, without noticeable narrowing. This demonstrates that our simulation data can effectively complement limited real-

We retrain GTRS-Dense with new simulation data of PSNR < 27 (∼10K×2 rounds) and compare it with the matched samples randomly selected from simulations with PSNR ≥ 27. The former represents lower reconstruction quality and is more likely to yield simulations with reduced visual fidelity. As shown in Tab. 10, simulation data

world data and still offers substantial potential to further exploit and amplify the value of real data even when it is abundant.

### E. Limitations and Broader Impact

#### E.1. Pseudo-Expert

Despite its effectiveness, our pseudo-expert generation pipeline has limitations. Current rule-based trajectory perturbations are static. A potential enhancement is the selfevolving approach [90], using pretrained planners to iteratively explore the simulator and collect recovery data. Additionally, the privileged BEV planner we use is rulebased with limited performance, causing some degradation in comfort metrics (HC, EC in Tab. 7) and failing in extremely corner cases. More advanced learning-based BEV planners [30, 75, 99] could further improve generation efficiency and realism.

#### E.2. Scene Simulation

For traffic behavior simulation in our decoupled scene simulation paradigm, other agents are controlled by IDM [76], which enables interaction but limits scenario diversity. Diffusion-based traffic generators [36, 50, 103] offer a promising improvement. For sensor simulation, feedforward GS [8, 88] can improve reconstruction efficiency, while adding modalities like LiDAR [44] provides complementary modality information. Furthermore, unified world models [73, 80, 89, 94, 95] can serve as a substitute for both behavior and sensor simulation.

#### E.3. Self-Play

Self-play [13, 14] allows the ego and surrounding agents to share learned policies instead of relying on separate traffic behavior simulators, enabling behaviors to co-evolve through interaction. With sensor simulation, it can support end-to-end policy learning and potentially improve robustness in long-tail scenarios, though it remains limited by interaction cost and simulation fidelity.

#### E.4. Societal Impact

Despite promising improvements, pseudo-expert scene simulation still has room for quality and efficiency enhancements for real-world deployment. Co-training may be affected by unrealistic simulation visuals and distributional differences in real-world corner cases, which could lead to potential risks. We hope SimScale will inspire both academia and industry in driving and robotics to leverage real-world simulation to address rare corner cases in data scaling, advancing fully autonomous systems that are robust and generalizable. In addition, we will provide the community with a fully open-source simulation dataset and training framework to promote academic research on autonomous driving in simulation.

### F. License of Assets

All training and evaluation are performed on data from publicly licensed datasets [6, 39, 70]. The real-world data engine is based on the MTGS codebase [51] under the Apache-2.0 license. We adopt publicly available end-to-end planners, including GTRS [59] (Apache-2.0) and DiffusionDrive [60] and LTF [12] (MIT).

| |
|---|

Pseudo-Expert Trajectory Perturbed Trajectory

Perturbed State

Log End Point Real Ego Sim. Ego

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Sim.3

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

Sim.2 Sim.1

Front-Left Front Front-Right

Real

[Figure 160]

[Figure 161]

[Figure 162]

Front-Left Front Front-Right

- Sim. 1

- Sim. 2

- Sim. 3 Front-Left Front Front-Right

[Figure 163]

[Figure 164]

[Figure 165]

Front-Left Front Front-Right

[Figure 166]

[Figure 167]

[Figure 168]

###### Figure 10. Qualitative results of recovery-based expert with real and simulation data.

| |
|---|

Pseudo-Expert Trajectory Perturbed Trajectory Perturbed State

Real Ego Sim. Ego

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

- Sim.2

[Figure 174]

[Figure 175]

[Figure 176]

- Sim.3

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Sim.1

[Figure 182]

Real Front-Left Front Front-Right

[Figure 183]

[Figure 184]

- Sim. 1 Front-Left Front Front-Right

- Sim. 2 Front-Left Front Front-Right

- Sim. 3 Front-Left Front Front-Right

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

###### Figure 11. Qualitative results of planner-based expert with real and simulation data. (Y-branch intersection)

| |
|---|

Pseudo-Expert Trajectory Perturbed Trajectory Perturbed State

Real Ego Sim. Ego

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

Sim.3 Sim.2

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

Real Front-Left Front Front-Right Sim.1

[Figure 202]

[Figure 203]

[Figure 204]

- Sim. 1 Front-Left Front Front-Right

- Sim. 2 Front-Left Front Front-Right

- Sim. 3 Front-Left Front Front-Right

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

###### Figure 12. Qualitative results of planner-based expert with real and simulation data. (dense urban avenue)

| |
|---|

Pseudo-Expert Trajectory Perturbed Trajectory Perturbed State

Real Ego Sim. Ego

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Sim.2

Sim.1

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

Sim.3

Real Front-Left Front Front-Right

[Figure 223]

[Figure 224]

- Sim. 1 Front-Left Front Front-Right

- Sim. 2 Front-Left Front Front-Right

- Sim. 3 Front-Left Front Front-Right

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

###### Figure 13. Qualitative results of planner-based expert with real and simulation data. (narrow local road)

| |
|---|

Pseudo-Expert Trajectory Perturbed Trajectory

Sim. Ego

[Figure 229]

[Figure 230]

Front-Left Front Front-Right

Sim.

[Figure 231]

[Figure 232]

Sim.

Front-Left Front Front-Right

[Figure 233]

[Figure 234]

Sim. Front-Left Front Front-Right

[Figure 235]

[Figure 236]

Sim. Front-Left Front Front-Right

[Figure 237]

[Figure 238]

Sim. Front-Left Front Front-Right

###### Figure 14. Additional qualitative results of the simulation scenes on navtrain.

