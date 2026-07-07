# arXiv:2606.15133v1[cs.RO]13Jun2026

## DragMesh-2: Physically Plausible Dexterous Hand-Object Interaction with Articulated Objects

#### Tianshan Zhang∗ Yijia Duan∗ Yanjun Li∗ Zeyu Zhang∗† Hao Tang‡ School of Computer Science, Peking University

∗Equal contribution. †Project lead. ‡Corresponding author: bjdxtanghao@gmail.com.

Abstract: Dexterous interaction with articulated objects is important for household, assistive, and humanoid manipulation, where multi-finger hands can provide compliant contact patterns beyond parallel-jaw grasping. However, articulatedobject manipulation differs from static-object manipulation: the target part cannot be directly actuated, and its motion must emerge through sustained physical hand– handle contact. This makes the transition from object-centric articulated generation to hand-driven dexterous hand–object interaction non-trivial, since geometric trajectory replay or open-loop execution does not model the contact dynamics required to move the articulated part. Moreover, policies trained only for task completion under fixed dynamics can overfit nominal contact loads, especially without tactile or force feedback, and may degrade when the contact load changes. To address these challenges, we present DragMesh-2, a contact-driven framework for dexterous interaction with articulated objects that extends articulated interaction from object-centric generation to hand-driven dexterous hand–object interaction, where articulated motion must arise through physical contact. We further propose PICA, a physically informed contact-aware training mechanism that injects physical signals into policy learning without tactile or force feedback, improving robustness and task success under changing contact loads. Finally, we conduct systematic evaluation across multiple damping conditions and articulatedobject categories to study robustness under contact-load variation, and provide a pure-geometry dexterous interaction resource to support future loco-manipulation and humanoid hand–object interaction research. Across seven GAPartNet objects, DragMesh-2 achieves stronger robustness under contact-load variation than the compared methods while maintaining high task success across damping conditions. Code: https://github.com/AIGeeksGroup/DragMesh-2. Website: https://aigeeksgroup.github.io/DragMesh-2.

Keywords: Dexterous Manipulation, Articulated Object Manipulation, HandObject Interaction

### 1 Introduction

Dexterous hand interaction with articulated objects [1, 2, 3, 4, 5, 6] is a central problem in robot manipulation and is important in household robotics, assistive systems, and humanoid manipulation settings. Compared with parallel-jaw grippers, dexterous hands provide more compliant multifinger contact patterns [7, 8, 9]. In recent years, understanding and interacting with articulated objects [10, 11, 12, 13, 14, 15, 16, 17] has become an important research topic in robotics and 3D intelligence. Existing work has mainly focused on articulated structure modeling, motion constraint reasoning, and articulated motion generation. Our previous work, DragMesh 1 [18], showed that explicit articulation priors can convert user interaction into articulated motion that follows kinematic constraints, enabling object-centric articulated interaction.

However, unlike static objects, articulated objects cannot be directly controlled. Their motion must emerge through sustained hand–object contact, making the transition from object-centric generation to realistic hand–object interaction (HOI) non-trivial. Naive strategies based on geometric trajectory replay, open-loop execution, or direct state control often fail to capture contact dynamics. More importantly, existing reinforcement learning methods are typically trained under fixed dynamics and optimize task completion as the sole objective. Without tactile or force feedback, policies tend to overfit nominal dynamics and rely on dynamics shortcuts rather than learning stable contact behaviors. As a result, a policy that succeeds under nominal damping often degrades rapidly when contact loads change, such as under increased damping. In other words, success under nominal dynamics does not necessarily imply stable contact behavior.

To address these challenges, we propose DragMesh-2, a contact-driven framework for dexterous interaction with articulated objects. DragMesh-2 formulates articulated-object manipulation as a problem that must be completed through real hand–handle interaction: articulated parts cannot be directly controlled, and their motion can only arise through physical interaction between the dexterous hand and the articulated structure. Building on this framework, we further introduce PICA (Physically Informed Contact-Aware) training to improve robustness under changing contact loads without requiring additional force sensing. PICA explicitly injects physically informed signals into policy learning through contact-aware constraints and dynamics randomization, mitigating action saturation, contact detachment, and overfitting to a single dynamics condition. We further combine PICA with temporal contact-response modeling to improve policy representations of changing interaction states. In summary, the main contributions of this work include:

- • We introduce DragMesh-2, a contact-driven framework that extends DragMesh-style articulated interaction from object-centric motion generation to dexterous hand–object interaction. In DragMesh-2, the policy controls only the hand, the target joint has no action channel, and articulated motion must emerge through sustained physical hand–handle contact.
- • We propose PICA, a physically informed contact-aware training mechanism that injects observable physical signals into policy learning, including contact maintenance, detachment risk, action-boundary regularization, damping variation, and temporal contact response. PICA shifts learning from task-progress-only optimization toward contact-conditioned interaction, improving robustness under contact-load changes without tactile or force feedback.
- • We construct a systematic evaluation protocol across multiple damping conditions and articulated-object categories, together with a pure-geometry dexterous interaction dataset. The protocol evaluates not only task success but also contact-aware diagnostics such as action saturation and detachment, while the generated trajectories provide geometry-guided grasp initialization, task-scale normalization, and tracking references for reproducible contact-driven interaction.

### 2 Related Work

Articulated Object Understanding and Manipulation. Articulated objects are common in human environments, but their constrained part motions make perception and manipulation more challenging than rigid-object interaction. Existing studies address articulated object understanding through part-level perception, pose estimation, and shape representation [19, 20, 21, 22], joint parameter prediction [23, 24, 25], and interactive simulation platforms [26, 27]. For manipulation, prior methods either infer articulation models for downstream planning [28] or learn actionable representations and policies from observations [29, 30, 31]. These efforts, however, mainly target object- or scene-level manipulation with mobile manipulators, grippers, or simplified end-effectors, rather than multifinger contact-rich interaction with articulated parts. Dexterous interaction with articulated objects remains less explored, as it requires physically compatible coordination between multi-finger contacts and moving object parts.

Dexterous Hand-Object Manipulation. Classical dexterous manipulation methods modeled multifinger coordination through contact mechanics, grasp stability, and force closure [32, 33]. Although

theoretically grounded, they require accurate geometry and contact models, which restricts their applicability to diverse objects and uncertain dynamics. Reinforcement learning reduces reliance on explicit modeling by learning high-DoF control policies directly from interaction, and has achieved strong performance on rigid-object in-hand manipulation [34, 35, 36]. Since such policies often require costly exploration, imitation learning and teleoperation-based methods introduce demonstrations and hand-object datasets as behavioral priors [37, 38, 39, 40]. More recent work extends these efforts from rigid-object settings to articulated-object settings, introducing dexterous manipulation benchmarks and hand-object interaction datasets [2, 41]. However, many of these learning pipelines are still primarily evaluated by task progress or success, whereas physically compatible interaction also requires stable contacts, limited interpenetration, and coordinated motion between fingers and articulated parts.

Physics-Grounded Manipulation Learning. Vision-based teleoperation has enabled dexterous manipulation from visual observations [42, 43], while domain-randomized policy learning has demonstrated autonomous dexterous control without tactile sensing [34]. In these approaches, contact and dynamics variations are handled through data diversity or domain randomization, without explicitly representing physical variation as a deploy-time adaptation variable. Sim-to-real adaptation methods instead condition policies on dynamics parameters or latent environment factors inferred from recent interaction history during deployment [44, 45]. Such factors are typically lowdimensional and global, making them less suited for capturing the local, state-dependent responses that vary with handle geometry, finger configuration, contact state, and part motion. This motivates PICA to use short-horizon interaction history as a task-level signal for contact-rich articulated manipulation.

Reliable execution of such policies further demands regularized action generation, since saturated joint commands can break contact and destabilize articulated motion. Reward penalties couple action regularization with the task reward and require manual weight tuning. Constrained reinforcement learning [46, 47] formalizes the separation between task objectives and control constraints through Constrained MDPs and Lagrangian optimization. Motivated by this formulation, PICA introduces separate action-bound and contact-preserving regularization terms alongside the task reward.

### 3 Method

#### 3.1 Contact-Driven Task Formulation

DragMesh-2 defines a contact-driven pulling task for a 51-DoF SMPL-X hand, including 6 virtual wrist DoFs and 45 finger joints. The policy controls only the hand. The object joint has no action channel, so the target part can move only through hand–handle contact. For each GAPartNet object, the target joint is selected as the object DoF with the largest motion range in a geometry-guided reference trajectory. The success threshold is set by the relative motion range of that trajectory,

qdone = qmintraj + ρ(qmaxtraj − qmintraj), (1) and task progress is normalized by the same object-specific range,

qto − qstart qgoal − qstart

. (2)

pt = max 0,

This definition makes drawers, sliders, and doors comparable without using a fixed absolute joint displacement.

The observation contains hand joint positions and velocities, handle pose, relative palm–handle geometry, target-joint state, and task-scale features derived from progress and remaining distance to the success threshold. It does not include RGB, depth, point clouds, force, or tactile signals. The action is a 51-dimensional increment to the hand PD target, clipped to the joint limits before execution. The reference trajectory is used only to initialize the expert grasp, define the target motion scale, and provide a non-learned tracking baseline; it is not replayed as an object-control command and does not provide expert action labels.

[Figure 1]

Figure 1: Architecture of DragMesh-2.

#### 3.2 Physically Informed Contact-Aware Learning

The reference policy instantiates the PICA signal mechanism with PPO [48], not as a new RL algorithm, but as a controlled policy for the benchmark. A history token combines the current PD tracking error and previous action,

ht = [et,at−1], et = qtPD − qth, (3)

and a GLA encoder[49] maps the recent token block to a contact-history feature. A causal-window auxiliary head predicts observable contact responses from this feature:

yt = qto − qto−K, max

dτ, ⊮ max

τ∈[t−K,t]

τ∈[t−K,t]

∥eτ∥2 . (4)

dτ > ddetach , max

τ∈[t−K,t]

The four targets describe recent object response, maximum palm–handle distance, detachment risk, and tracking stress. PICA incorporates these physical signals in both the environment and policy levels. At the environment level, the reward augments task progress with contact maintenance, action regularization, detachment handling, and successful termination:

rt = rtask + rdist + ract + rtime + rdetach + rsuccess + rbound + rcontact. (5)

Here rdist, rbound, and rcontact encourage contact maintenance and suppress unsafe separation or saturated control, while the detachment term is triggered only after the hand has entered and then leaves the effective contact range.

At the policy level, PICA constrains the temporal representation through contact-response prediction:

L = LPPO + cvLV + cbLbounds + wauxLaux. (6) The auxiliary loss updates the temporal encoder to predict recent object response, maximum palm– handle distance, detachment risk, and tracking stress. Thus, PICA augments PPO with observable physical proxies that bias policy learning away from nominal-success shortcuts and toward contactconditioned interaction. Full coefficient values, network settings, and inference parameters are provided in Appendix A.

Evaluation reports task success and progress together with physical diagnostics. clip099 is the fraction of rollout steps whose maximum action magnitude exceeds 0.99, and detach_proxy

is the detachment-failure rate. For damping set B = {×1,×2,×4} and execution mode m ∈ {det,stoch}, robustness is summarized as

1 |B| b∈B

S¯m =

Sb,m, Smworst = min b∈B

Sb,m. (7)

The disaggregated per-damping success values remain important because averages can hide failure under strong damping.

#### 3.3 Dataset

The reference contact trajectories are drawn from a dataset that we generate heuristically, without any learning, directly from GAPartNet geometry. For each articulated object, a geometry-guided procedure reads the part, handle, and joint-mobility annotations together with a SMPL-X hand model, and emits a phased interaction trajectory—approach, grasp, drag, and release—whose wrist and finger motion is geometrically consistent with the target joint (full procedure in Appendix B, Algorithm 1). Each trajectory is stored as a JSON file of per-frame wrist poses and finger configurations, so the dataset is independent of any policy or physics backend and can be regenerated for any GAPartNet object that carries the required annotations.

The dataset contains 277 trajectories over 7 GAPartNet categories (Table 1). Its distribution follows that of GAPartNet and is dominated by StorageFurniture, the largest articulated-object category. Within DragMesh-2 the dataset plays three roles: it initializes the expert grasp state and target motion scale for the contact-driven reinforcement-learning task, it defines the non-learned trajectorytracking baseline, and we release it as a pure-geometry interaction resource for future contact-rich and whole-body loco-manipulation research.

### 4 Experiments

We evaluate on a benchmark of 7 GAPartNet objects spanning three categories (Dishwasher, StorageFurniture, Microwave) and two joint types (5 revolute doors and 2 prismatic drawers). All episodes start from the expert grasp state, and the target part can be opened only through hand– handle contact (Figure 3); the reference contact trajectories that provide this initialization come from our heuristically generated dataset (Section 3.3, Contribution 3). Each (method, object, damping, mode) cell uses 20 episodes. Deterministic execution uses the Gaussian mean, while stochastic execution samples from the learned policy. The damping multipliers ×1, ×2, and ×4 measure nominal performance, mild contact-load shift, and strong out-of-distribution (OOD) contact-load shift, respectively.

Table 1: Heuristic trajectory dataset.

Category #Traj. StorageFurniture 256 TrashCan 7 Dishwasher 5 Refrigerator 4 Oven 3 Microwave 1 TableObject 1 Total 277

We compare against a trajectory-tracking replay reference, a GT-part-pose parallel-jaw primitive, and four learned baselines: state-only PPO, flat-history PPO, GRU-PPO, and Transformer-PPO. We further ablate the two physical-structure components of PICA by removing either the physical signals or the GLA temporal encoder. Beyond task success, the contact-aware protocol logs the action-saturation ratio and detachment-failure rate (Appendix A.2); these diagnostics, training hyperparameters, and the full baseline taxonomy are reported in Appendix C.

Main comparison. Figure 2 and Table 2 report the main comparison, and four findings stand out. First, the trajectory-tracking reference reaches 1.00 deterministic success at ×1 on all seven objects, confirming the reference trajectory genuinely drives the target part through contact rather than replaying object states, but its average drops to 0.71 at ×2 and ×4 as two objects (45936 and 7310) lose contact once damping rises—open-loop replay alone is not OOD-robust. Second,

GT Parallel-Jaw

State-only PPO

Flat-history PPO

GLA w/o Phys

GRU-PPO

Transformer-PPO

PICA w/o GLA

PICA (Ours)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

###### (A) Deterministic

(B) Stochastic

1.00

0.75

SuccessRate

0.50

0.25

0.00

×1 ×2 ×4 Damping multiplier

×1 ×2 ×4 Damping multiplier

- Figure 2: Success rate across damping multipliers (×1, ×2, ×4) under deterministic (A) and stochastic (B) execution, averaged over 7 GAPartNet objects (20 episodes each). We compare a non-learned GT-part-pose parallel-jaw primitive with seven learned methods, including recurrent (GRU-PPO) and Transformer baselines; the primitive is deterministic, so its value is identical across execution modes. Adding physical signals and the GLA encoder yields consistent gains, and PICA attains the highest mean success in all six mode×damping settings and retains the highest absolute success under strong damping (0.56 deterministic at ×4, versus 0.27 for state-only PPO and 0.09 for Transformer-PPO).

the GT-part-pose parallel-jaw primitive succeeds on only one object (0.14 mean) and is dampinginvariant, showing that a geometric primitive cannot substitute for closed-loop dexterous contact control even when the part pose is known. Third, among learned policies PICA attains the best mean in every damping/mode column: deterministic success goes from 0.89 at ×1 to 0.56 at ×4, versus 0.27 for state-only PPO, 0.32 for flat-history PPO, 0.30 for a GRU policy, and 0.09 for a Transformer policy over the same observation. Fourth, adding richer temporal encoders alone does not close the gap: GRU and Transformer baselines, like the GLA-only ablation (Table 3, 0.36 at ×4), all lag PICA by at least 0.13 at ×4; the win is therefore the combination of physical signals with temporal contact-response modeling, not the temporal encoder by itself. The per-object numbers also reveal substantial heterogeneity—no single method dominates every instance—which we revisit

- as a limitation.

Ablation. Table 3 isolates the two physical-structure components of PICA. Using only the GLA temporal encoder (without the physical signals) reaches 0.65 deterministic success at ×1 and 0.36

- at ×4; using only the physical signals (with a flat-history encoder) reaches 0.75 and 0.43; combining the two reaches 0.89 and 0.56. The full model exceeds either component by at least 0.13 at ×4, and the components help along different axes—the physical signals contribute more under nominal damping while the temporal encoder helps more under stochastic mid-damping—so they are complementary rather than redundant.

Nominal success masks saturation collapse. To show that nominal success alone is misleading, we vary only the training length of the base policy on a single object, before any contact-stabilization fine-tuning (Table 4). Deterministic nominal (×1) success rises from 0.90 to 1.00 as training extends from 150 to 500 epochs, yet strong-damping (×4) success collapses from 0.55 to 0.10 while the action-saturation proxy clip099 climbs toward 1.0. Longer training therefore buys nominal success by driving the policy into a saturated, low-robustness regime—directly motivating a protocol that reports OOD damping and saturation alongside success, and checkpoint selection by OOD robustness rather than by training reward.

Additional studies. Appendix C reports further studies. Extended fine-tuning and damping-range expansion do not yield stable additional OOD gains, indicating that robustness under strong contact

- Table 2: Per-object success on 7 GAPartNet objects across damping multipliers (×1 nominal, ×2 mild, ×4 OOD). Each cell is deterministic/stochastic success (mean over 20 episodes). Trajectory tracking is an open-loop replay reference (deterministic) and the parallel-jaw primitive is deterministic, so both have no stochastic value. The best learned-policy deterministic Avg per damping is in bold.

Per-object success (deterministic/stochastic)

Method Damp

12583

Dishwasher door

45261

StorageFurn. drawer

45661

StorageFurn. door

45936

StorageFurn. door

46440

StorageFurn. drawer

48513

StorageFurn. door

7310

Microwave

door Avg

Traj. tracking

- ×1 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
- ×2 1.00 1.00 1.00 0.00 1.00 1.00 0.00 0.71 ×4 1.00 1.00 1.00 0.00 1.00 1.00 0.00 0.71

Parallel-Jaw

- ×1 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.14
- ×2 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.14 ×4 1.00 0.00 0.00 0.00 0.00 0.00 0.00 0.14

State-only PPO

- ×1 1.00/0.95 0.10/0.20 0.00/0.00 1.00/0.20 0.95/0.90 1.00/0.85 0.00/0.00 0.58/0.44
- ×2 0.95/0.50 0.10/0.10 0.00/0.00 0.05/0.10 1.00/0.95 0.95/0.80 0.00/0.00 0.44/0.35 ×4 0.00/0.05 0.10/0.10 0.00/0.00 0.00/0.00 0.85/0.85 0.95/0.80 0.00/0.00 0.27/0.26

Flat-history PPO

- ×1 0.10/0.10 1.00/0.90 0.05/0.45 0.25/0.25 0.65/0.20 0.00/0.05 0.95/0.45 0.43/0.34
- ×2 0.00/0.00 1.00/0.70 0.40/0.35 0.00/0.00 0.65/0.05 0.00/0.20 0.45/0.05 0.36/0.19 ×4 0.00/0.00 1.00/0.80 0.50/0.30 0.00/0.00 0.75/0.20 0.00/0.20 0.00/0.00 0.32/0.21

GRU-PPO

- ×1 0.60/0.15 0.20/0.10 1.00/0.90 0.00/0.00 1.00/1.00 0.50/0.75 0.30/0.15 0.51/0.44
- ×2 0.00/0.00 0.00/0.05 1.00/0.95 0.00/0.00 1.00/0.85 0.00/0.00 0.30/0.05 0.33/0.27 ×4 0.00/0.00 0.00/0.05 1.00/1.00 0.00/0.00 1.00/0.90 0.00/0.00 0.10/0.00 0.30/0.28

Transformer-PPO

- ×1 1.00/0.65 0.00/0.00 0.00/0.00 0.00/0.00 0.45/0.20 0.85/0.70 0.15/0.20 0.35/0.25
- ×2 0.90/0.25 0.00/0.05 0.00/0.00 0.00/0.00 0.40/0.25 0.15/0.10 0.15/0.15 0.23/0.11 ×4 0.00/0.00 0.00/0.00 0.00/0.00 0.00/0.00 0.60/0.25 0.00/0.00 0.00/0.00 0.09/0.04

PICA (Ours)

- ×1 1.00/1.00 1.00/1.00 0.85/0.90 0.90/0.55 1.00/0.75 0.95/0.85 0.55/0.70 0.89/0.82
- ×2 1.00/0.95 0.90/0.90 0.80/0.70 0.65/0.45 1.00/0.70 0.70/0.80 0.55/0.55 0.80/0.72 ×4 0.95/0.95 1.00/0.95 0.60/0.65 0.25/0.00 0.85/0.15 0.10/0.30 0.15/0.00 0.56/0.43

- Table 3: Per-object ablation of the two physical-structure components across damping multipliers. Each cell is deterministic/stochastic success. w/o PICA keeps the GLA temporal encoder but drops the physical signals; w/o GLA keeps the physical signals with a flat-history encoder. The best deterministic Avg per damping is in bold.

Per-object success (deterministic/stochastic)

45936

46440

48513

45261

45661

7310

12583

StorageFurn. door

StorageFurn. door

StorageFurn. drawer

StorageFurn. drawer

StorageFurn. door

Microwave

Dishwasher door

door Avg

Method Damp

- ×1 1.00/1.00 1.00/1.00 0.00/0.40 0.35/0.50 0.85/0.40 0.45/0.70 0.90/0.75 0.65/0.68
- ×2 1.00/1.00 1.00/1.00 0.05/0.35 0.25/0.30 0.45/0.40 0.40/0.70 0.80/0.70 0.56/0.64 ×4 0.95/0.75 1.00/0.90 0.00/0.20 0.05/0.05 0.00/0.05 0.10/0.30 0.45/0.20 0.36/0.35

w/o PICA (GLA only)

- ×1 1.00/1.00 0.95/0.85 0.00/0.00 1.00/0.95 1.00/1.00 0.45/0.65 0.85/0.95 0.75/0.77
- ×2 1.00/0.85 0.85/0.85 0.00/0.00 1.00/0.70 1.00/1.00 0.50/0.35 0.65/0.25 0.71/0.57 ×4 0.05/0.00 0.90/0.95 0.00/0.00 0.80/0.40 1.00/1.00 0.25/0.15 0.00/0.00 0.43/0.36

w/o GLA (PICA only)

- ×1 1.00/1.00 1.00/1.00 0.85/0.90 0.90/0.55 1.00/0.75 0.95/0.85 0.55/0.70 0.89/0.82
- ×2 1.00/0.95 0.90/0.90 0.80/0.70 0.65/0.45 1.00/0.70 0.70/0.80 0.55/0.55 0.80/0.72 ×4 0.95/0.95 1.00/0.95 0.60/0.65 0.25/0.00 0.85/0.15 0.10/0.30 0.15/0.00 0.56/0.43

PICA (Ours)

load requires richer contact interfaces rather than longer optimization. We also report rollout-level diagnostics and the contact-aware saturation and detachment metrics that motivate the protocol.

### 5 Limitations and Future Work

DragMesh-2 still has clear limitations, which also point to concrete next steps. First, even with the PICA signals, the learned policy relies on a position-increment action interface and tends toward action saturation under strong contact load: success drops from 0.89 at ×1 to 0.56 at ×4, and per-object results remain heterogeneous, so no single policy dominates every instance. Because

Approach Grasp Open

[Figure 2]

[Figure 3]

[Figure 4]

HardwareSimulation

[Figure 5]

[Figure 6]

[Figure 7]

- Figure 3: Qualitative illustration of contact-driven articulated interaction. Top: a hardware feasibility example showing approach, grasp, and opening through hand–handle contact. Bottom: a simulated rollout of the reference policy on a StorageFurniture instance from the dataset. The hardware example is included only as a qualitative illustration; all quantitative evaluations are conducted in simulation.

the observation channel provides no force or tactile feedback, contact state can only be inferred indirectly from kinematic error, which appears insufficient for stable light pulling at high damping. A natural next step is therefore to enrich the contact interface, including wrist force or torque outputs and contact-force or tactile feedback, so that the policy can regulate grip force directly rather than push to the actuator boundary.

Second, our task isolates contact-driven pulling from an expert grasp state and controls a floating dexterous hand. The reference contact trajectories, however, are full hand–object motion clips that are geometrically consistent with the target joint dynamics, so they extend naturally beyond the hand alone. A promising direction is to couple this upper-body contact interaction with wholebody control, using the dataset as a motion-scale prior for humanoid loco-manipulation, where balance and locomotion must be coordinated with the same physically plausible contact behavior studied here.

Table 4: Training-length study of the base policy.

Base Succ. ×1 Succ. ×4 clip099 ×4

150 ep 0.90 0.55 0.90 200 ep 0.90 0.50 0.97 300 ep 1.00 0.10 0.99 500 ep 1.00 0.10 0.99

### 6 Conclusion

We presented DragMesh-2, a contact-driven framework for dexterous hand–articulated-object interaction, where the target part moves only through physical hand–handle contact. Extending DragMesh 1 from object-centric articulated interaction to hand-driven physical interaction, DragMesh-2 shows that nominal task success does not guarantee stable contact behavior, since policies trained only for task progress can degrade sharply under contact-load shifts. PICA improves robustness by adding physically informed training signals, dynamics randomization, and temporal contact-response modeling without force or tactile feedback. Across seven GAPartNet objects under nominal, moderate, and out-of-distribution damping, DragMesh-2 achieves stronger robustness than competing methods, and we release a pure-geometry interaction dataset for future whole-body loco-manipulation and humanoid HOI.

#### Acknowledgments

If a paper is accepted, the final camera-ready version will (and probably should) include acknowledgments. All acknowledgments go at the end of the paper, including thanks to reviewers who gave useful comments, to colleagues who contributed to the ideas, and to funding agencies and corporate sponsors that provided financial support.

### References

- [1] J. Li, Y. Zhu, Y. Xie, Z. Jiang, M. Seo, G. Pavlakos, and Y. Zhu. Okami: Teaching humanoid robots manipulation skills through single video imitation. In Conf. Robot Learn., 2024.
- [2] C. Bao, H. Xu, Y. Qin, and X. Wang. Dexart: Benchmarking generalizable dexterous manipulation with articulated objects. In IEEE Conf. Comput. Vis. Pattern Recog., 2023.
- [3] Z. Jing, S. Yang, J. Ao, T. Xiao, Y. Jiang, and C. Bai. Humanoidgen: Data generation for bimanual dexterous manipulation via llm reasoning. arXiv preprint arXiv:2507.00833, 2025.
- [4] Y. Lu, Y. Tian, Z. Yuan, X. Wang, P. Hua, Z. Xue, and H. Xu. H3DP: Triply-hierarchical diffusion policy for visuomotor learning. arXiv preprint arXiv:2505.07819, 2025.
- [5] M. Lee, D. K. Kim, J. K. Bandi, M. Smith, A. Liao, A.-a. Agha-mohammadi, and S. Omidshafiei. Stageact: Stage-conditioned imitation for robust humanoid door opening. arXiv preprint arXiv:2509.13200, 2025.
- [6] Y. Wang, Y. Miao, W. Zhao, W. Yang, Z. Wang, J. Pajarinen, L. Van Gool, D. P. Paudel, J. Kannala, X. Wang, et al. Paws: Perception of articulation in the wild at scale from egocentric videos. arXiv preprint arXiv:2603.25539, 2026.
- [7] Q. Liu, Y. Cui, Q. Ye, Z. Sun, H. Li, G. Li, L. Shao, and J. Chen. Dexrepnet: Learning dexterous robotic grasping network with geometric and spatial hand-object representations. In IEEE/RSJ Int. Conf. Intell. Robots Syst., 2023.
- [8] Z. Xu, Y. Wang, B. Abbatematteo, J. Preechayasomboon, S. Chan, N. Colonnese, and A. H. Memar. Contact-grounded policy: Dexterous visuotactile policy with generative contact grounding. In Robot. Sci. Syst., 2026.
- [9] Y. Cui, Y. Zhang, L. Tao, Y. Li, X. Yi, and Z. Li. End-to-end dexterous arm-hand vla policies via shared autonomy: Vr teleoperation augmented by autonomous hand vla policy for efficient data collection. arXiv preprint arXiv:2511.00139, 2025.
- [10] M. Gao, Y. Pan, H.-a. Gao, Z. Zhang, W. Li, H. Dong, H. Tang, L. Yi, and H. Zhao. Partrm: Modeling part-level dynamics with large cross-state reconstruction model. In IEEE Conf. Comput. Vis. Pattern Recog., 2025.
- [11] Y. Liu, B. Jia, R. Lu, J. Ni, S.-C. Zhu, and S. Huang. Building interactable replicas of complex articulated objects via gaussian splatting. In Int. Conf. Learn. Represent., 2025.
- [12] L. Le, J. Xie, W. Liang, H.-J. Wang, Y. Yang, Y. J. Ma, K. Vedder, A. Krishna, D. Jayaraman, and E. Eaton. Articulate-Anything: Automatic modeling of articulated objects via a visionlanguage foundation model. In Int. Conf. Learn. Represent., 2025.
- [13] R. Wu, X. Wang, L. Liu, C. Guo, J. Qiu, C. Li, L. Huang, Z. Su, and M.-M. Cheng. DIPO: Dual-state images controlled articulated object generation powered by diverse data. Adv. Neural Inform. Process. Syst., 38:108665–108689, 2026.
- [14] Y. Wang, X. Zhang, R. Wu, Y. Li, Y. Shen, M. Wu, Z. He, Y. Wang, and H. Dong. Adamanip: Adaptive articulated object manipulation environments and policy learning. In Int. Conf. Learn. Represent., 2025.

- [15] R. Wu, Y. Zhao, K. Mo, Z. Guo, Y. Wang, T. Wu, Q. Fan, X. Chen, L. Guibas, and H. Dong. VAT-mart: Learning visual action trajectory proposals for manipulating 3d ARTiculated objects. In Int. Conf. Learn. Represent., 2022.
- [16] Y. Wang, Z. Wang, M. Nakura, P. Bhowal, C.-L. Kuo, Y.-T. Chen, Z. Erickson, and D. Held. Articubot: Learning universal articulated object manipulation policy via large scale simulation. In Robot. Sci. Syst., 2025.
- [17] Z. Zhao, Y. Li, W. Li, Z. Qi, L. Ruan, Y. Zhu, and K. Althoefer. Tac-Man: Tactile-informed prior-free manipulation of articulated objects. IEEE Transactions on Robotics (T-RO), 2025.
- [18] T. Zhang, Z. Zhang, and H. Tang. Dragmesh: Interactive 3d generation made easy. arXiv preprint arXiv:2512.06424, 2025.
- [19] H. Geng, H. Xu, C. Zhao, C. Xu, L. Yi, S. Huang, and H. Wang. GAPartNet: Cross-category domain-generalizable object perception and manipulation via generalizable and actionable parts. In IEEE Conf. Comput. Vis. Pattern Recog., 2023.
- [20] X. Li, H. Wang, L. Yi, L. J. Guibas, A. L. Abbott, and S. Song. Category-level articulated object pose estimation. In IEEE Conf. Comput. Vis. Pattern Recog., 2020.
- [21] J. Mu, W. Qiu, A. Kortylewski, A. Yuille, N. Vasconcelos, and X. Wang. A-SDF: Learning disentangled signed distance functions for articulated shape representation. In Int. Conf. Comput. Vis., 2021.
- [22] L. Zhang, M. Mei, A. Wang, X. Meng, Y. Zhong, X. Song, L. Liu, R. Wang, Z. He, and C. Lu. DICArt: Advancing category-level articulated object pose estimation in discrete state-spaces. arXiv preprint arXiv:2602.19565, 2026.
- [23] X. Wang, B. Zhou, Y. Shi, X. Chen, Q. Zhao, and K. Xu. Shape2Motion: Joint analysis of motion parts and attributes from 3D shapes. In IEEE Conf. Comput. Vis. Pattern Recog., 2019.
- [24] A. Jain, R. Lioutikov, C. Chuck, and S. Niekum. ScrewNet: Category-independent articulation model estimation from depth images using screw theory. In IEEE Int. Conf. Robot. Autom., 2021.
- [25] Z. Jiang, C.-C. Hsu, and Y. Zhu. Ditto: Building digital twins of articulated objects from interaction. In IEEE Conf. Comput. Vis. Pattern Recog., 2022.
- [26] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, and G. State. Isaac Gym: High performance GPU-based physics simulation for robot learning. In NeurIPS Datasets and Benchmarks, 2021.
- [27] F. Xiang, Y. Qin, K. Mo, Y. Xia, H. Zhu, F. Liu, M. Liu, H. Jiang, Y. Yuan, H. Wang, L. Yi, A. X. Chang, L. J. Guibas, and H. Su. SAPIEN: A simulated part-based interactive environment. In IEEE Conf. Comput. Vis. Pattern Recog., 2020.
- [28] R. Buchanan, A. Röfer, J. Moura, A. Valada, and S. Vijayakumar. Online estimation and manipulation of articulated objects. arXiv preprint arXiv:2601.01438, 2026.
- [29] K. Mo, L. J. Guibas, M. Mukadam, A. Gupta, and S. Tulsiani. Where2Act: From pixels to actions for articulated 3D objects. In Int. Conf. Comput. Vis., 2021.
- [30] Z. Xu, Z. He, and S. Song. UMPNet: Universal manipulation policy network for articulated objects. IEEE Robot. Autom. Lett., 7(2):2447–2454, 2022.
- [31] B. Eisner, H. Zhang, and D. Held. FlowBot3D: Learning 3D articulation flow to manipulate articulated objects. In Robot. Sci. Syst., 2022.

- [32] A. Bicchi. On the closure properties of robotic grasping. Int. J. Robot. Res., 14(4):319–334, 1995.
- [33] A. M. Okamura, N. Smaby, and M. R. Cutkosky. An overview of dexterous manipulation. In IEEE Int. Conf. Robot. Autom., 2000.
- [34] M. Andrychowicz, B. Baker, M. Chociej, R. Józefowicz, B. McGrew, J. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, J. Schneider, S. Sidor, J. Tobin, P. Welinder, L. Weng, and W. Zaremba. Learning dexterous in-hand manipulation. Int. J. Robot. Res., 39(1):3–20, 2020.
- [35] A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, and S. Levine. Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. In Robot. Sci. Syst., 2018.
- [36] T. Chen, J. Xu, and P. Agrawal. A system for general in-hand object re-orientation. In Conf. Robot Learn., 2021.
- [37] Y. Qin, Y.-H. Wu, S. Liu, H. Jiang, R. Yang, Y. Fu, and X. Wang. DexMV: Imitation learning for dexterous manipulation from human videos. In Eur. Conf. Comput. Vis., 2022.
- [38] L. Yang, K. Li, X. Zhan, F. Wu, A. Xu, L. Liu, and C. Lu. OakInk: A large-scale knowledge repository for understanding hand-object interaction. In IEEE Conf. Comput. Vis. Pattern Recog., 2022.
- [39] Y. Qin, W. Yang, B. Huang, K. Van Wyk, H. Su, X. Wang, Y.-W. Chao, and D. Fox. AnyTeleop: A general vision-based dexterous robot arm-hand teleoperation system. In Robot. Sci. Syst., 2023.
- [40] Y. Liu, Y. Liu, C. Jiang, K. Lyu, W. Wan, H. Shen, B. Liang, Z. Fu, H. Wang, and L. Yi. HOI4D: A 4D egocentric dataset for category-level human-object interaction. In IEEE Conf. Comput. Vis. Pattern Recog., 2022.
- [41] Z. Fan, O. Taheri, D. Tzionas, M. Kocabas, M. Kaufmann, M. J. Black, and O. Hilliges. ARCTIC: A dataset for dexterous bimanual hand-object manipulation. In IEEE Conf. Comput. Vis. Pattern Recog., 2023.
- [42] A. Handa, K. Van Wyk, W. Yang, J. Liang, Y.-W. Chao, Q. Wan, S. Birchfield, N. D. Ratliff, and D. Fox. DexPilot: Vision-based teleoperation of dexterous robotic hand-arm system. In IEEE Int. Conf. Robot. Autom., 2020.
- [43] Y. Qin, H. Su, and X. Wang. From one hand to multiple hands: Imitation learning for dexterous manipulation from single-camera teleoperation. IEEE Robot. Autom. Lett., 7(4):10873–10881, 2022.
- [44] W. Yu, J. Tan, C. K. Liu, and G. Turk. Preparing for the unknown: Learning a universal policy with online system identification. In Robot. Sci. Syst., 2017.
- [45] A. Kumar, Z. Fu, D. Pathak, and J. Malik. RMA: Rapid motor adaptation for legged robots. In Robot. Sci. Syst., 2021.
- [46] J. Achiam, D. Held, A. Tamar, and P. Abbeel. Constrained policy optimization. In Int. Conf. Mach. Learn., 2017.
- [47] C. Tessler, D. J. Mankowitz, and S. Mannor. Reward constrained policy optimization. In Int. Conf. Learn. Represent., 2019.
- [48] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [49] S. Yang, B. Wang, Y. Shen, R. Panda, and Y. Kim. Gated linear attention transformers with hardware-efficient training. In Int. Conf. Mach. Learn., 2024.

### A Additional Method Details

This appendix gives implementation-level details omitted from the main text. Section 3 contains the task definition, core PICA signals, and evaluation metrics; the appendix expands only the observation and control representation, reward terms, temporal encoder, auxiliary supervision, optimization settings, additional diagnostics, and implementation parameters.

- A.1 Observation and Control Details

Because the environment does not provide explicit force or tactile sensors, DragMesh 2 is intrinsically a partially observable Markov decision process. The kinematic state at a single frame is insufficient to infer the contact impedance of the hand–object system, load variation, or potential detachment trends. The protocol therefore uses a state-only observation that describes the visible geometric and joint state at the current frame, while the policy approximates the implicit physical state from recent control history. No RGB, depth, point cloud, or semantic segmentation input is used.

The state at time t contains hand joint positions and velocities, the handle pose, the relative hand– handle geometry, the object joint state, and task-scale features derived from the target joint position:

st = qth, q˙th, xhandlet , rthandle, xhandlet − xpalmt , dt, qto, q˙to, ϕ(qto) . (8)

Here qth,q˙th are 51-dimensional hand joint positions and velocities, xhandlet ,rthandle are the handle position and orientation, dt is the distance from palm center to handle center, and qto,q˙to are the object joint position and velocity. The task-scale features ϕ(qto) contain relative task progress, the remainder to the success threshold, and the object-level motion scale ∆q = qgoal − qstart. These features are deterministic functions of qto and the task boundary defined by the trajectory. Keeping them explicit helps learn a value function shared across doors, drawers, and sliders with different motion ranges.

The policy outputs a 51-dimensional continuous action at for incremental control of the virtual wrist joint and finger joints. The action is first clipped to [−1,1] and then scaled by α into a local increment of the hand PD target:

∆qth = αat, qt,htarget = clip(qth + ∆qth,qhmin,qhmax). (9) The target is sent to a position PD controller, while the object joint is affected only through simulated contact. Since no policy channel directly drives the object joint, the target part can only be opened through contact between the hand and the handle. This incremental action representation reduces the difficulty of high-dimensional hand control while preserving the role of contact dynamics in producing object motion. Action scaling, control frequency, and the inference-time execution mode are listed in the appendix.

- A.2 Evaluation-Protocol Details

The main text defines the contact-aware metrics and robustness summary. The appendix records the only additional protocol detail: the non-learned trajectory-tracking baseline feeds each nextframe hand pose from the reference trajectory as the hand PD target, while the object joint state is never replayed. The target part is still driven only through hand–handle contact, so this baseline tests whether the reference motion can induce physical opening rather than reproduce stored object states.

- A.3 Reference Policy: Physical Signal Mechanism (PICA)

- A.3.1 Physical-Plausibility Reward

The reference policy treats contact maintenance and action regularity as differentiable training targets. In addition to the task-progress reward, the policy is subject to contact maintenance, action

magnitude, and termination constraints. The palm–handle distance dt defines a weak contact maintenance term:

rdist = −wdist dt + wnear exp(−κdt). (10) The task-progress term is the per-step increment of the target joint:

rtask = wtask (qto − qto−1). (11) The action and time costs are

ract = −wact mean(a2t), rtime = −wtime. (12)

If the palm has previously approached the handle and the palm–handle distance later exceeds ddetach before the task is done, the episode is declared a detachment failure and incurs a one-shot penalty; if the target joint reaches the success threshold a one-shot success bonus is granted. The detachment criterion only triggers after the policy has been within effective contact range, which acts as a contact gate. The policy must remain on the contact manifold and cannot evade later action costs by releasing contact or moving away from the handle. Success, detachment failure, and exceeding the maximum episode length each terminate the episode.

To suppress brittle pulling driven by saturated actions, the reference policy adds action-boundary and contact-distance regularizers:

rbound = −wbound mean max(|at| − asat,0)2 , (13)

rcontact = −wcontact max(dt − dsafe,0)2. (14) The total reward is

rt = rdist + rtask + ract + rtime + rdetach + rsuccess + rbound + rcontact. (15)

The four explicit physical constraints, namely saturation gating rbound, contact-distance regularization rcontact, detachment gating rdetach, and the damping randomization described next, together constitute the PICA signal mechanism. The reward signal of the policy is driven not only by nominal task progress, but also by whether the task is completed under contact-maintaining and actionregularized conditions. Coefficient values are listed in the appendix.

For fine-tuning, two optional contact-stabilization rewards serve as ablation modules. The first, ARAM, is an adaptive version of rbound; in high-impedance stalled states, it imposes additional penalties on high-magnitude actions, so the policy cannot bypass contact resistance through sustained saturated pulling. The second, Reconfig, encourages small-amplitude hand reconfiguration when the policy stalls in contact, allowing the policy to re-establish effective contact rather than persist with a failing pulling pose. These two modules are used in the diagnostic ablations as the ARAM, Reconfig, and combined Both fine-tunes; they do not change the DragMesh 2 task definition, the observation interface, or the evaluation protocol.

#### A.3.2 Damping Randomization

Damping randomization tests whether the policy relies on a single nominal dynamics setting. At each environment reset, a damping scale for the target object joint is drawn uniformly from a specified interval and applied to the nominal damping. This perturbation exposes the policy during training to pulling responses under varying resistance, reducing dependence on a single dynamics setting. The default training interval is [1.0,2.0]; friction randomization is not used. Evaluation uses higher damping multipliers, including ×4, to construct OOD dynamics tests.

#### A.3.3 Contact-History Temporal Encoder

Contact-rich dexterous manipulation is strongly time-dependent. The hand pose and object joint state at a single frame do not fully reveal whether contact is stable, whether the hand is sliding off the handle, or whether recent actions are producing object response. The reference policy therefore

includes recent control history in the policy state. Each history token consists of the hand PD tracking error and the previous action:

ht = [et, at−1], et = qtPD − qth. (16) The full history block is

Ht = [ht−L+1, ..., ht]. (17) This sequence reflects observable physical response during contact. When the hand is impeded by reaction forces from the handle, the tracking error grows. When strong control produces no targetjoint progress, the action–error pattern in the history exposes high impedance or invalid contact. When the palm drifts away from the handle, the change in contact distance combined with the action response signals impending detachment.

The reference policy uses a two-branch actor–critic architecture. The state st is encoded by an MLP, while the history block Ht is projected by a linear layer into a token representation and fed to a Gated Linear Attention temporal encoder. Compared with a history-concatenated MLP or standard self-attention, the gating in GLA is more sensitive to abrupt phase transitions in contact dynamics, such as transient impacts or sliding. By default, the output corresponding to the last history token is used as the contact-history feature:

zthist = GLA(Ht)L. (18) After normalization, this temporal feature is concatenated with the current state feature to form the fused representation zt. The actor head outputs the Gaussian policy mean µt from zt, and the critic head outputs the state value V (st,Ht) from the same fused representation. Network widths, head counts, history length, and policy standard deviation are listed in the appendix.

#### A.3.4 Causal-Window Contact-Response Auxiliary Supervision

- A temporal encoder does not automatically learn a representation consistent with contact dynamics: under a training loop driven only by task reward, it can equally well degrade into a shortcut representation under the nominal dynamics. The reference policy therefore applies causal-window

physical auxiliary supervision to zthist, requiring this feature to recover observable contact-response signals from recent history. These signals serve as an implicit time-domain representation of contact impedance: they do not measure contact forces directly, but they characterize how the hand–object system responds to control inputs through object response, contact distance, and hand tracking error.

The environment maintains a buffer of the most recent K +1 physics steps of palm–handle distance and target object joint position. The auxiliary head receives only zthist and predicts

yt = qto − qto−K, max

∥eτ∥2 . (19)

dτ, ⊮ max

dτ > ddetach , max

τ∈[t−K,t]

τ∈[t−K,t]

τ∈[t−K,t]

The four channels denote, respectively, the recent object joint response, the maximum palm–handle distance in the window, the detachment-risk proxy, and the maximum tracking stress in the causal window. Here eτ = qτPD − qτh, and maxτ∈[t−K,t]∥eτ∥2 measures the largest tracking residual between PD target and hand state in the window, which can be read, in a compliant-control sense, as an observable proxy for contact load. The auxiliary targets are passed only to the auxiliary head as training supervision. When the actor, critic, and GLA backbone process the observation, these target channels are explicitly removed from the input and never serve as additional state features for policy decisions.

DragMesh 2 uses causal windows rather than single-step differences to construct auxiliary targets. Single-step physical quantities in rigid-body contact are susceptible to high-frequency noise from the contact solver, and single-step transients cannot represent the slow degradation of contact quality, such as gradual palm slip before a failure flag is triggered. Windowed object response, maximum contact distance, and the detachment-risk proxy provide a smoother and more stable impedance representation, making the auxiliary supervision a more reliable constraint for guiding the temporal encoder toward contact-response representations.

The auxiliary loss uses a channel-weighted mean-squared error:

Laux = ωq ℓ(qresponse) + ωd ℓ(dmax) + ωb ℓ(bdetach) + ωe ℓ(Etrack). (20)

This loss updates both the auxiliary head and the GLA temporal encoder. Its purpose is not to reward any specific action, but to encourage the temporal encoder to represent whether actions produce object response, whether contact is maintained, and whether the hand is under tracking stress. The actor and critic therefore receive a temporally encoded representation with explicit physical meaning. The channel weights and the window length are listed in the appendix.

#### A.3.5 Policy Optimization

The reference policy is optimized with PPO using GAE for advantage estimation. PPO serves as the optimization vehicle rather than a contribution. The training objective combines the clipped policy loss, the value loss, the actor output-boundary loss, and the physical auxiliary loss:

L = LPPO + cv LV + cb Lbounds + waux Laux. (21)

Lbounds acts directly on the actor output distribution and discourages the policy mean from approaching the action boundary, while the environment-side rbound penalizes saturated actions after execution. The auxiliary weight waux follows a linear warmup, so the policy first acquires basic control through the task reward and is then constrained by contact-response prediction. All training hyperparameters, reward coefficients, action scaling, history length, and inference modes are listed in the appendix.

The components in the reference-policy layer, namely the contact-regularized reward, damping randomization, GLA contact-history encoding, and causal-window auxiliary supervision, jointly instantiate the PICA signal mechanism. The training objective explicitly contains observable physical proxies, the temporal representation is constrained toward implicit states consistent with contact response, and the policy is trained to behave stably under varied contact loads. This instantiation is one possible implementation; the specific network structure, reward coefficients, and optimization hyperparameters are not the main contribution of this paper. They serve to test whether the defined task and protocol can distinguish contact-conditioned behavior from unstable shortcut behavior.

### B Reference-Trajectory Generation and Dataset Construction

This section details the geometry-guided procedure that produces the reference contact trajectories and the released dataset (Section 3.3); it uses only GAPartNet geometry and annotations, with no learning.

Target-part selection. For a given GAPartNet object, the system selects the target part according to the semantic annotation: parts whose category contains slider or drawer are preferred; otherwise door is chosen; if neither matches, the last annotated part is used as a fallback. The handle annotation nearest the front face of the target part is then chosen; its oriented bounding box gives the center ch, long axis lh, outward normal nh, short axis sh, and thickness dh. If no explicit handle annotation exists, the front face of the target part is used as a fallback to construct the contact geometry.

Wrist and finger poses. The wrist orientation Rw is constructed from lh and nh so that the palm faces the interaction region and the fingers wrap along the long axis of the handle. Since control is anchored to the wrist coordinate frame while contact occurs near the palm, the wrist position compensates for the palm-center offset opalm:

xw = ch − Rwopalm. (22)

The finger configurations include an open pose qopen, a pre-grasp pose qpre, and a force-closure grasp pose qgrasp, with qgrasp adjusted to the handle thickness dh.

Approach phase. The pre-grasp position is offset along the outward normal: xpre = xw + δnh, (23)

and the wrist is linearly interpolated from xpre to xw over Tapproach steps, while the fingers move from qopen to qpre:

xt = (1 − αt)xpre + αtxw, qt = (1 − αt)qopen + αtqpre, (24) with αt = t/Tapproach. Grasp phase. The wrist pose is held fixed and the fingers close from qpre to qgrasp. Drag phase. For a prismatic joint, the interaction center translates along the outward normal of the handle:

ct = ch + αtdnh, (25) with d a preset drag distance. For a revolute joint, the interaction center rotates about the joint axis aj and the joint origin oj:

ct = oj + R(αtθ,aj)(ch − oj), (26) where θ is a preset rotation angle. The wrist position is recomputed from the updated interaction center at each step; in the revolute case, the wrist orientation rotates synchronously so the palm tracks the handle.

Release phase. After the drag completes, the fingers gradually open to qopen and the wrist retracts along the final outward normal.

The full pseudocode is given in Algorithm 1. The generator has no learned parameters and depends only on the GAPartNet geometry and mobility annotations, so it runs directly on any articulated object equipped with both geometry and mobility annotations. The phase durations Tapproach, Tgrasp, Tdrag, Trelease, the geometric offset δ, the drag distance d, and the rotation angle θ are set adaptively to the object type.

### C Additional Experimental Results

This appendix reports diagnostic results that are omitted from the compact main text. Section 4 gives the main success table; the appendix focuses on additional qualitative visualizations, rollout-level behavior, strong-damping diagnostics, extended fine-tuning, damping-range expansion, and ablations. All multi-episode cells use the expert-grasp initialization and 20 episodes per deterministic or stochastic execution mode unless stated otherwise. These diagnostic studies are auxiliary to the seven-object main comparison; they explain failure modes and checkpoint-selection effects, but they do not replace the aggregate success and ablation tables in Section 4.

In the checkpoint-diagnostic subsections, the base policy denotes a PICA reference policy trained with the full physical-signal mechanism on the diagnostic object. Base (N ep) denotes its checkpoint after N epochs. The contact-stabilization modules ARAM and Reconfig (Appendix A.3.1) are applied on top of the base policy for a further 50 epochs, giving the ARAM, Reconfig, and Both fine-tunes. SELECTED and OVERTRAINED are used only for single-rollout visualization diagnostics: SELECTED is base (150ep) followed by the Both fine-tune, while OVERTRAINED continues that fine-tune well beyond it.

Training-curve scope. We omit raw training reward curves from the appendix, because reward scales are not identical once contact regularizers, auxiliary terms, and fine-tuning modules are introduced. The baseline evidence should therefore remain evaluation-side success, progress, clip099, and detach_proxy across damping conditions. If additional curves are needed for diagnostics, they should compare evaluation metrics rather than raw rewards; only within the same training recipe is a reward-versus-OOD-success curve directly interpretable.

Algorithm 1: Geometry-Guided Interaction Trajectory Generation Input: Articulated object o; GAPartNet annotations A; mobility annotations M; hand model

H; motion parameters Θ = {δ,d,θ,Tapproach,Tgrasp,Tdrag,Trelease}. Output: Reference contact trajectory τ = {(xt,Rt,qt,sobjt )}Tt=1. /* Scene and annotation initialization. */

- 1 Initialize a physics scene with object o and hand model H;
- 2 Load part bounding boxes, categories, link names from A and joint information from M; /* Target part, handle and mobility selection. */
- 3 p⋆ ← SELECTTARGETPART(A);
- 4 h⋆ ← FINDASSOCIATEDHANDLE(p⋆,A);
- 5 (ch,lh,nh,sh,dh) ← DECOMPOSEHANDLEBBOX(h⋆);
- 6 m ← QUERYMOBILITYTYPE(p⋆,M); /* Wrist pose and hand shape. */
- 7 Rw ← COMPUTEWRISTORIENTATION(lh,nh);
- 8 xw ← COMPUTEWRISTPOSITION(ch,Rw);
- 9 xpre ← xw + δ nh;
- 10 qopen ← OPENHANDPOSE();
- 11 qpre ← PRESHAPEHANDPOSE();
- 12 qgrasp ← FORCECLOSUREPOSE(dh);
- 13 τ ← ∅; /* Approach: wrist moves from pre-grasp to grasp; fingers shape up. */
- 14 τ ← τ ∪ EXECUTEPHASE(xpre,xw,qopen,qpre,Tapproach); /* Grasp: wrist fixed; fingers close to force closure. */
- 15 τ ← τ ∪ EXECUTEPHASE(xw,xw,qpre,qgrasp,Tgrasp); /* Drag: follow target joint axis through contact. */
- 16 for t = 1 to Tdrag do

- 17 α ← t/Tdrag;
- 18 if m is revolute then

- 19 (ct,Rt) ← ROTATEAROUNDJOINT(ch,Rw,M,α);
- 20 else

- 21 (ct,Rt) ← TRANSLATEALONGNORMAL(ch,Rw,nh,α);

- 22 xt ← COMPUTEWRISTPOSITION(ct,Rt);
- 23 τ ← τ ∪ STEPANDRECORD(xt,Rt,qgrasp); /* Release: open fingers and retract wrist along outward normal. */

- 24 τ ← τ ∪ RELEASEANDRETRACT(cfinal,Rfinal,qgrasp,qopen,Trelease);
- 25 Persist τ as the reference contact trajectory used by the DragMesh 2 environment (initial state writing), the trajectory tracking baseline, and the motion-scale reference of the evaluation protocol;
- 26 return τ;

#### C.1 Qualitative Simulation and Hardware Visualizations

Figures 4 and 6 provide additional visual evidence using frames not shown in the main text. The simulation snapshots illustrate that the same contact-driven formulation covers both prismatic drawers and revolute doors. Figure 4 shows approach, grasp, and drag stages for three object instances, and Figure 5 shows additional terminal-stage examples. For Figure 5, each render replays the stored right-hand trajectory state using the simulator’s 51-DoF floating SMPL-X right-hand model: 3 virtual wrist-translation DoFs, 3 wrist-rotation DoFs, and 45 finger DoFs. The hardware frames are extracted from the supplementary stage video and are included as a qualitative feasibility check. They are not pooled with the 20-episode simulation statistics and do not constitute a separate realworld quantitative benchmark.

Approach Grasp Drag

[Figure 8]

[Figure 9]

[Figure 10]

46440 StorageFurn. drawer

[Figure 11]

[Figure 12]

[Figure 13]

12583 Dishwasher door

[Figure 14]

[Figure 15]

[Figure 16]

7310 Microwave

door

- Figure 4: Additional simulated approach, grasp, and drag snapshots on Three generated object trajectories.

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

40147 StorageFurn. slider drawer

44962 StorageFurn. slider drawer

48513 StorageFurn. hinge door

102996 TrashCan slider drawer

103008 TrashCan hinge door

- Figure 5: Additional terminal-stage renders showing category and joint-type diversity in the generated trajectory dataset.

#### C.2 Rollout-Level Diagnostics

To interpret the contact behavior behind the numerical metrics, we compare single-rollout visualizations for two checkpoints: SELECTED denotes the base policy (150ep) followed by a 50-epoch Bothmodule fine-tune, and OVERTRAINED denotes a checkpoint obtained by continuing that fine-tune well beyond it. Only one rollout per object, execution mode, and damping setting is saved. Thus, this section describes behavior patterns and is not pooled into the multi-episode success statistics.

Table 5: Single-rollout diagnostics for representative checkpoints.

Checkpoint Object Success Mean Prog. Best Prog. Steps L2

SELECTED 7310 5/6 0.451 0.564 33.3 3.624 SELECTED 45936 3/6 0.285 0.597 66.0 4.132 OVERTRAINED 7310 0/6 0.175 0.288 6.0 3.351 OVERTRAINED 45936 2/6 0.199 0.548 36.8 3.104

The video evidence is consistent with the numerical summary. On 7310, SELECTED succeeds in 5 of 6 rollouts and, even when it fails under strong damping, maintains a relatively continuous pulling pose. The failure mode resembles a correctly aligned pulling direction with insufficient

Approach Grasp Open

[Figure 22]

[Figure 23]

[Figure 24]

Figure 6: Hardware stage snapshots extracted from the supplementary stage video.

contact output to overcome the elevated damping load. The trend on 45936 is similar: SELECTED completes some rollouts under nominal and mid damping, and under ×4 it fails with relatively long episode durations, indicating that the policy remains in contact interaction rather than detaching immediately.

The OVERTRAINED checkpoint shows a different pattern. It fails on every rollout of 7310, with a mean episode length of only 6.0 steps and a best progress of 0.288. It does not apply small and steady pulling actions, but rapidly enters a failure or abnormal contact state. The video shows that OVERTRAINED develops visible perturbations after failing to pull the part open, and these perturbations do not translate into target-joint progress. This behavior is consistent with the low mean action L2, short episode length, and low progress. On 45936, OVERTRAINED succeeds only under nominal damping and degrades sharply at higher damping, indicating that subsequent fine-tuning does not improve robustness to contact load and may instead weaken the stable temporal pulling strategy.

This visualization diagnostic supports a finer claim than success alone. The dominant failure mode of SELECTED is a largely correct contact direction and pulling intent with insufficient sustained output, whereas OVERTRAINED fails to maintain stable contact response under resistance and switches to ineffective perturbation. These visualizations are used only to interpret the diagnostic logs; the main paper’s quantitative claims remain the seven-object simulation results in Section 4.

#### C.3 OOD Damping Robustness

The ×4 damping setting is analyzed separately on diagnostic object 45936 because it most directly exposes stability under strong load. Under deterministic execution, the base policy (150ep), the base policy (200ep), the Reconfig fine-tune, and the Both-module fine-tune all retain ×4 success in the 0.50–0.55 range, while the base policy (300ep) and the base policy (500ep) drop to roughly 0.10 with clip099 approaching 0.99. This indicates that the base policy undergoes a relatively rapid OOD ×4 collapse between epochs 200 and 300 and then settles into a high-saturation, lowrobustness regime. Longer training is therefore not sufficient for physical robustness and can push the policy toward a highly saturated nominal-dynamics solution. Under stochastic execution, the Both-module fine-tune is the only variant in the available logs to reach 0.55 at ×4, suggesting that the combined fine-tuning is more stable when dynamics perturbation and sampling noise act together.

#### C.4 Limits of Extended Fine-Tuning

The OOD-robustness collapse in the base policy occurs between epochs 200 and 300. A natural question is whether the 50 epochs of Both-module fine-tuning fall in an early-stopping window or whether the Both-module fine-tune is still under-trained. Starting from the Both-module fine-tune (total epoch 200) and keeping the training recipe fixed, we continue training for another 200 epochs to total epoch 400 and reevaluate checkpoints at epoch 250, epoch 300, and epoch 399 on object 45936. This diagnostic uses a separate controlled re-evaluation on object 45936 with 20 episodes per cell and is intended only to analyze within-run fine-tuning trends rather than replace the sevenobject main comparison.

- Table 6: Object-45936 per-checkpoint diagnostics at ×4 damping. Det and Stoch rows are reported separately.

Variant Mode Success ×4 Prog. ×4 Return ×4 clip099 ×4 detach ×4

base (150 ep) Det 0.55 0.398 138.0 0.896 0.45 base (200 ep) Det 0.50 0.425 137.3 0.972 0.50 base (300 ep) Det 0.10 0.206 19.5 0.986 0.90 base (500 ep) Det 0.10 0.236 27.8 0.993 0.90 ARAM (50 ep) Det 0.25 0.382 88.6 0.969 0.75 Reconfig (50 ep) Det 0.50 0.439 141.1 0.981 0.50 Both (50 ep) Det 0.55 0.449 151.1 0.971 0.45

base (150 ep) Stoch 0.30 0.334 83.7 0.936 0.70 base (200 ep) Stoch 0.10 0.249 31.3 0.967 0.90 base (300 ep) Stoch 0.15 0.318 56.5 0.939 0.85 base (500 ep) Stoch 0.30 0.201 48.5 0.926 0.70 ARAM (50 ep) Stoch 0.15 0.337 61.8 0.984 0.85 Reconfig (50 ep) Stoch 0.30 0.383 96.4 0.986 0.70 Both (50 ep) Stoch 0.55 0.420 143.5 0.973 0.45

- Table 7: Extending fine-tuning beyond the Both fine-tune. The training-reward-best checkpoint at epoch 399 still lies in the degraded regime, while clip099 drifts monotonically upward.

Checkpoint Det ×1 Det ×2 Det ×4 Stoch ×1 Stoch ×2 Stoch ×4 clip099 ×4 Both fine-tune (200 ep) 1.00 0.85 0.00 0.95 0.70 0.20 continued (250 ep) 1.00 0.25 0.10 0.90 0.10 0.10 0.598 continued (300 ep) 1.00 0.60 0.00 0.95 0.20 0.05 0.732 continued (399 ep, reward-best) 0.90 0.30 0.00 0.95 0.15 0.10 0.869

From the training curves, extended training appears stable: success_mean stays near 1.0 throughout epochs 200–400, and reward_mean rises slowly from roughly 255 to 260. The OOD evaluation shows the opposite trend. Deterministic ×2 success drops immediately from 0.85 for the Bothmodule fine-tune to 0.25 at epoch 250, partially recovers to 0.60 at epoch 300, and falls back to 0.30 at epoch 399. Stochastic ×2 likewise degrades from 0.70 to the 0.10–0.20 range. Deterministic ×4 remains near the 0.00–0.10 floor, and extended training provides no stable improvement. At the same time, clip099 at ×4 grows monotonically over epochs (0.598 → 0.732 → 0.869), indicating slow saturation of the action distribution.

ARAM delays, but does not block, the drift toward saturated actions. Across 200 additional epochs, the Both-module fine-tune reproduces the OOD collapse that the base policy undergoes between epochs 200 and 300, only stretched to roughly four times the time scale. Even the checkpoint automatically saved by the training framework as the best by training reward (epoch 399) lies in the degraded regime, confirming that checkpoint selection by training reward alone is insufficient for OOD robustness. This supports OOD-based early stopping, rather than monotone training reward, as a protocol-level principle for checkpoint selection.

#### C.5 Limits of Damping-Distribution Expansion

The OOD ×4 collapse raises a second question: can fine-tuning broaden the training damping range so that the policy encounters strong-damping samples earlier and learns more stable contact behavior? Starting from the Both-module fine-tune, we broaden the damping scale interval from [1.0,2.0] to [1.0,4.0] while keeping all reward terms, action-boundary regularizers, network structure, and training hyperparameters fixed. We then fine-tune for 25 epochs on object 45936 to obtain the broadened-damping fine-tune. The two rows are evaluated under the same controlled setting on object 45936 and are used only to analyze the effect of broadening the damping range.

Broadening the training damping range yields no stable improvement at ×4: deterministic success rises only from 0.00 to 0.05, while ×2 degrades noticeably (deterministic 0.85 → 0.50, stochastic 0.70 → 0.35) and stochastic ×4 regresses overall. An additional check of intermediate checkpoints

Table 8: Broadening the training damping range during fine-tuning.

Variant Det ×1 Det ×2 Det ×4 Stoch ×1 Stoch ×2 Stoch ×4

Both fine-tune (controlled re-eval) 1.00 0.85 0.00 0.95 0.70 0.20 Both + damping [1, 4] (25 ep) 1.00 0.50 0.05 0.95 0.35 0.00

shows that one intermediate epoch transiently reaches 0.25 deterministic ×4 success, but its ×2 success drops in parallel to roughly 0.55, and the effect is not reproduced in later epochs. Earlier exploration with the narrower range [1.0,2.5] combined with delayed ARAM, and with the more aggressive range [2.0,4.0], produces sharper degradation at ×4 and ×2 and is not listed in the main table.

Broadening the training damping range alone provides limited benefit under the current incrementalposition control and reward-shaping framework. The contact-interface capability required at ×4, namely stable light pulling under sustained high load, may lie beyond what the present 51dimensional position-increment action channel and force-free observation channel can represent. Further improvement in OOD ×4 robustness therefore requires changes to the task or policy interfaces, such as adding a wrist force or torque output dimension on the action side, introducing contact force or torque feedback on the observation side, or separating light pulling and heavy pulling into different contact modes at the policy level via mode switching or expert mixtures.

#### C.6 Ablation Analysis

This diagnostic ablation analysis examines the effects of training duration, damping randomization, and the two fine-tuning modules on object 45936. Comparing the base policy at 150, 200, 300, and 500 epochs shows that additional training improves success under nominal and mid damping but does not monotonically improve robustness under strong damping. The OOD ×4 degradation concentrates between epochs 200 and 300: the base policy at 300ep already nearly matches the 500ep checkpoint on ×4 success (0.10) and clip099 (0.986 vs. 0.993). The collapse is therefore not a slow over-saturation but a relatively rapid behavioral transition. Continued fine-tuning shows the same pattern under ARAM, stretched to roughly four times the time scale. The direction does not reverse: between epoch 200 and epoch 400, deterministic ×2 success degrades from 0.85 to the 0.30 range while clip099 rises monotonically. These results support the observation that optimizing only the task reward selects high-saturation, low-robustness contact policies, so the protocol must report action saturation and detachment alongside success and use OOD-based criteria rather than the training reward for checkpoint selection.

Broadening the training damping distribution alone does not resolve the OOD ×4 failure. The extended-damping fine-tuning variant improves deterministic ×4 only from 0.00 to 0.05 and introduces significant degradation at ×2. Section C.5 gives the detailed analysis. Within the incrementalposition control and reward-shaping framework, adjusting the training damping interval alone provides limited benefit. Further improvement at ×4 requires new mechanisms at the action channel, contact observation, or policy level rather than only a different training distribution.

The three fine-tune variants indicate different roles for the physical modules. The ARAM finetune improves deterministic nominal-damping success but yields no robustness gain at ×4, suggesting that constraining high-effort actions alone does not solve contact recovery under strong damping. The Reconfig fine-tune raises deterministic ×2 success to 0.95 and preserves higher ×4 progress, suggesting that contact reconfiguration helps recovery under moderate resistance. The Both-module fine-tune reaches 0.55 success and 0.420 progress under stochastic ×4 while keeping detach_proxy at 0.45, the most stable stochastic strong-damping result in the available logs. The two fine-tuning modules are therefore not simple substitutes: ARAM acts more directly on high-effort action behavior, while Reconfig is closer to contact recovery. Their combination is most useful when dynamics perturbation and sampling noise occur together.

#### C.7 Summary of Diagnostic Findings

The diagnostics support a narrower interpretation of the appendix experiments. PICA’s contact regularizers, dynamics randomization, and temporal auxiliary supervision should be read as a coupled protocol: GLA provides temporal capacity, but stable contact response requires pairing that capacity with explicit contact-maintenance and action-regularization signals. In the object-45936 diagnostic logs, the combined ARAM/Reconfig fine-tune gives the strongest stochastic ×4 result among the diagnostic variants considered here, while temporal encoding alone does not prevent saturation and detachment shortcuts.

Across extended fine-tuning and damping-range expansion, task reward and nominal success are not reliable checkpoint selectors for strong-load robustness. Longer training can preserve nominal performance while weakening ×2/ × 4 robustness, and broadening the damping range alone gives limited ×4 benefit while degrading mid-damping behavior. These trends motivate reporting success together with clip099, detach_proxy, progress, and damping-conditioned performance, rather than treating a single nominal score as sufficient.

#### C.8 Relationship Between Physical Diagnostics and Temporal Encoding

The central claim that follows from these empirical results is that, for contact-driven articulatedobject manipulation, physical diagnostics, the robustness summary, and temporal encoding must be designed jointly into the protocol. Temporal encoding alone can let the policy reach a nominaldynamics shortcut faster; physical shaping alone attains some robustness but lacks fine-grained modeling of historical contact response. The policy is more likely to extract a representation consistent with contact state only when the task protocol explicitly specifies contact-maintaining behavior and auxiliary supervision guides the temporal encoder to predict observable responses under such behavior. The experiments above provide progressive evidence for this position.

#### C.9 Limitations

Two caveats apply to the numbers reported here. First, each cell aggregates 20 episodes, so the standard error of a success estimate is approximately 0.10; small differences under strong damping, such as 0.15 versus 0.30 at ×4, should be read as the same order rather than as strictly ranked, and our conclusions rely on aggregate trends across objects, damping multipliers, and execution modes rather than on any single cell.

Second, because the observation channel provides no force or tactile signal, the policy infers contact state only indirectly from kinematic error. Under strong damping, this is the main caveat behind the residual light-pulling failures in the diagnostic results.

#### C.10 Future Work

Future work can extend the action and observation interfaces toward force-aware control and richer contact-response supervision. Within the strong-load regime, useful directions include separating light and heavy pulling into distinct contact modes and expanding the auxiliary-supervision targets to quantities such as contact normals or sliding velocity. Together with the whole-body locomanipulation extension outlined in the main text, these directions define a path toward contact-rich articulated manipulation that remains contact-stable under wider dynamics.

### D Implementation Details and Inference Settings

This appendix collects the implementation parameters omitted from the main text. These values support reproducibility and are not methodological contributions; they are placed here so the main text retains the task definition, evaluation protocol, and the key diagnostic conclusions.

#### D.1 Task and Control Parameters

- Table 9 lists the task and control constants shared by all policies: the success-threshold fraction, action scaling, episode length, and the temporal-encoder and network dimensions.

Table 9: Task, control, and network parameters.

Symbol Value Meaning

ρ 0.5 success-threshold fraction on the reference motion range α 0.05 action-to-PD-target increment scale Tmax 300 maximum episode length L 16 contact-history window length K 5 causal auxiliary prediction window length history token dim 102 dim of [tracking error, previous action] GLA heads 4 attention heads in the GLA encoder token embedding dim 128 history-token projection dimension MLP hidden dims [512, 512, 256] current-state encoder hidden sizes activation ELU MLP activation

D.2 Reward and Termination Parameters

- Table 10 lists the reward weights, contact thresholds, and one-shot bonuses and penalties used by the reference policy.

Table 10: Reward and termination parameters.

Symbol Value Meaning

wdist 0.25 linear penalty on palm–handle distance wnear 0.10 near-distance contact-keep bonus κ 5.0 decay coefficient of the near-distance bonus wtask 250 weight on target-joint progress wact 0.002 weight on action energy penalty wtime 0.05 per-step time cost ddetach 0.10m detachment-failure threshold rdetach −50 one-shot detachment penalty rsuccess +100 one-shot success reward asat 0.90 soft action-saturation threshold wbound 20 weight on action-boundary regularizer dsafe 0.08m soft contact-distance threshold wcontact 0.5 weight on contact-distance regularizer

D.3 Auxiliary-Supervision Parameters

- Table 11 lists the channel weights of the causal-window auxiliary loss and the warmup schedule of its overall weight.

Table 11: Auxiliary-supervision parameters.

Symbol Value Meaning ωq 1.0 weight on recent object joint response loss

- ωd 1.0 weight on max palm–handle distance loss ωb 0.5 weight on detachment-risk proxy loss
- ωe 0.5 weight on max tracking stress loss waux 0 → 0.01 linear warmup range

#### D.4 PPO Training Parameters

- Table 12 lists the PPO optimization hyperparameters; they follow standard on-policy settings and are not tuned per object.

Table 12: PPO training hyperparameters.

Hyperparameter Value

parallel environments 64 rollout length 32 discount γ 0.99 GAE λ 0.95 learning rate 3 × 10−4 PPO clip 0.2 minibatch epochs 5 value-loss coefficient 1.0 entropy coefficient 0 actor bounds-loss coeff. 0.01

#### D.5 Damping Randomization and Inference

During training, the damping scale factor on the target object joint is sampled uniformly from [1.0,2.0] by default. Extended damping experiments replace this interval with [1.0,4.0], [1.0,2.5], or [2.0,4.0] to analyze the boundary of training-distribution variation; friction randomization is not enabled in the current experiments.

At evaluation, the model checkpoint is fixed and rollouts are conducted at ×1, ×2, and ×4 damping. Deterministic execution uses the policy mean and stochastic execution samples from the learned Gaussian policy. Each quantitative cell uses 20 episodes by default; single-rollout visualizations are reserved for behavioral diagnostics and are not pooled into the multi-episode statistics. Checkpoint selection should not rely on training reward alone but should combine OOD damping evaluation, the action-saturation ratio, and the detachment-failure rate.

