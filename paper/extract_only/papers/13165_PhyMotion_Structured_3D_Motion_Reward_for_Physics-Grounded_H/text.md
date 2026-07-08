# arXiv:2605.14269v1[cs.CV]14May2026

## PhyMotion: Structured 3D Motion Reward for Physics-Grounded Human Video Generation

###### Yidong Huang1∗ Zun Wang1∗ Han Lin1 Dong-Ki Kim2 Shayegan Omidshafiei2

Jaehong Yoon3 Jaemin Cho4,5 Yue Zhang1 Mohit Bansal1 1UNC Chapel Hill 2FieldAI 3NTU Singapore 4AI2 5Johns Hopkins University https://phy-motion.github.io/

#### Abstract

Generating realistic human motion is a central yet unsolved challenge in video generation. Whilereinforcementlearning(RL)-basedpost-traininghasdrivenrecent gains in general video quality, extending it to human motion remains bottlenecked by a reward signal that cannot reliably score motion realism. Existing video rewards primarily rely on 2D perceptual signals, without explicitly modeling the 3D body state, contact, and dynamics underlying articulated human motion, and often assign high scores to videos with floating bodies or physically implausible movements. To address this, we propose PhyMotion, a structured, fine-grained motion reward that grounds recovered 3D human trajectories in a physics simulator and evaluates motion quality along multiple dimensions of physical feasibility. Concretely, we recover SMPL body meshes from generated videos, retarget them onto a humanoid in the MuJoCo physics simulator, and evaluate the resulting motion along three axes: kinematic plausibility, contact and balance consistency, and dynamic feasibility. Each component provides a continuous and interpretable signal tied to a specific aspect of motion quality, allowing the reward to capture which aspects of motion are physically correct or violated. Experiments show that PhyMotion achieves stronger correlation with human judgments than existing reward formulations. These gains carry over to RL-based post-training, where optimizing PhyMotion leads to larger and more consistent improvements than optimizing existing rewards, improving motion realism across both autoregressive and bidirectional video generators under both automatic metrics and blind human evaluation (+68 Elo gain). Ablations show that the three axes provide complementary supervision signals, while the reward preserves overall video generation quality with only modest training overhead.

#### 1 Introduction

Recent video generation models [Ho et al., 2022, OpenAI, 2024, Kong et al., 2024, Team Wan et al., 2025] produce increasingly photorealistic outputs, yet generating realistic human motion remains one of the most significant open challenges. While scene textures, lighting, and camera motion have reached impressive fidelity [Zheng et al., 2025, Liu et al., 2024b, Gao et al., 2025], modeling human motion remains fundamentally challenging due to the tight coupling between body articulation and physical constraints. As a result, generated videos frequently exhibit artifacts like floating feet, limb penetrations, anatomical distortions, and movements that defy basic laws of physics [Zheng et al.,

- 2025, Yang et al., 2026]. As human-centric content constitutes one of the most important application domains, from entertainment and education to virtual communication [Zhu et al., 2023, Sui et al.,
- 2026], improving human motion quality is a pressing goal. ∗Equal contribution.

Preprint.

Prompt: A person is stretching on a yoga mat. Generated Video (Baseline)

[Figure 1]

(b) Our PhyMotion Rewards (Explicit 3D Physics Space)

(a) Existing Rewards (Implicit 2D Visual Space)

(1) 3D Human Reconstruction

[Figure 2]

[Figure 3]

[Figure 4]

- (2) Physics Simulator

Excessive contact impulse Floating body

- (3) Decomposed Motion Scores

[Figure 5]

Vision Encoder

General Reward Score: 0.85

VLM Classifier Score: 0.78 Score: 1.0

Abrupt joint acceleration

### ✗

Kinematic Score: 0.75 Dynamic Score: 0.35 Contact Score: 0.67

High score on physically impossible motions

✓Frame-LevelError Physically-Grounded Fine-GrainedScore

✓ ✓

(c) RL Post-Training with Our 3D Reward Prompt

Video Generator

Candidate Videos

Physics Simulator

Our 3D Rewards

Our Video

3D Human Reconstruction

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Policy Optimization

[Figure 10]

[Figure 11]

Our Video

✓ Stable Motion ✓ Physics-Consistent Dynamics ✓ Realistic Contact Behavior

Figure 1: Overview of PhyMotion. Existing metrics operate in 2D pixel space (VLMs [Liu et al., 2025b], frame-level classifiers [Zheng et al., 2025]), producing misleading scores that are structurally blind to articulated motion. We recover SMPL meshes from generated videos and place them in a physics simulator, yielding decomposed, physically-grounded motion scores. The structured reward drives RL-based post-training (bottom), consistently improving human motion quality.

A promising direction is reinforcement learning (RL)-based post-training, which has recently demonstrated clear gains on general visual quality and text alignment [Black et al., 2024, Xue et al., 2025, Zhang et al., 2026]. The success of RL, however, hinges on the reward signal, and for human motion, existing rewards remain inadequate. Specifically, they broadly fall into three categories, each with a distinct failure mode: (i) frame-level 2D classifiers [Huang et al., 2024a, Zheng et al., 2025, Motamed et al., 2026] detect localized artifacts but miss trajectory-level failures; (ii) vision-language evaluators [Meng et al., 2024, Chow et al., 2025, Hong et al., 2025] provide coarse semantic judgments unsuitable as dense RL rewards; and (iii) general preference rewards [Bansal and Grover, 2024, Liu et al., 2025b, He et al., 2024, Ma et al., 2025] can hallucinate quality from appearance or prompt alignment rather than motion plausibility. Despite their differences, these rewards primarily rely on pixel-space perceptual representations and learned visual features. While effective for appearance quality and semantic alignment, we empirically observe that they still struggle to reliably capture key aspects of articulated human motion and physical feasibility, such as kinematic consistency, anatomical constraints, and contact dynamics (see Sec. 3.1). Consequently, they can assign high scores to videos with clear physical violations (Fig. 1(a) and Fig. 2).

To address this limitation, we propose PhyMotion, a structured, fine-grained motion reward for human video generation that grounds 3D human trajectories in a physics simulator to evaluate motion quality across multiple dimensions of physical feasibility (Fig. 1(b)). We lift each generated video into a structured 3D representation by recovering an SMPL [Loper et al., 2015] body mesh and grounding it in a physically consistent human model in the MuJoCo simulator [Todorov et al., 2012]. This enables access to structured physical signals (e.g., contact consistency, joint-level kinematics, and motion-driving torques) that are not explicitly captured in pixel-based representations, allowing evaluation based on whether the motion satisfies physical constraints, which directly underlie many

visually implausible artifacts. From these observables, we derive three structured feasibility scores: kinematic (joint consistency), contact/balance (interaction correctness), and dynamic (force and motion consistency). Each score targets a distinct failure mode identified in Sec. 3.1. To test whether our physics-grounded scores better capture perceived motion quality, we conduct a pairwise human study in which raters compare pairs of generated videos according to motion realism, and measure how well each metric predicts these preferences. We compare PhyMotion against prior 2D perceptual metrics and learned video reward models. We further use the same scores as a dense and interpretable reward for RL-based post-training (Fig. 1(c)), examining whether optimizing physical feasibility leads to improved human motion generation. Because the reward decomposes into physically meaningful components, it provides transparency into which aspects of motion are optimized, enabling fine-grained diagnosis of motion failures and revealing improvements across different physical dimensions.

Experiments demonstrate that PhyMotion serves as both a reliable motion evaluator and an effective reward for RL-based video post-training. As an evaluator, PhyMotion achieves 80% average pairwise agreement with human judgments and the highest aggregate Spearman correlation (ρ=0.376) on 1,200 video pairs, across multiple aspects of motion quality including body structure, balance, and motion naturalness. This substantially outperforms existing perceptual, preference-based, and physics-aware reward baselines, which typically achieve 50–66% agreement and only weak Spearman correlations (ρ = 0–0.25). As a reward, PhyMotion further enables effective RL-based posttraining across both autoregressive [Zhu et al., 2026] and bidirectional [Zhang et al., 2025] video generators. Additionally, our reward also improves scores on external evaluators, including VBench metrics [Huang et al., 2024a], VideoAlign [Liu et al., 2025b], and VideoPhy-PC [Bansal and Grover, 2024], by an average of 7.1%, while achieving consistent gains across all three physical feasibility dimensions. Human preference evaluation further confirms these improvements: our post-trained model achieves the highest Elo scores on body structure, balance, motion naturalness, and overall preference, outperforming all baselines, including the larger Wan2.2 14B model. Meanwhile, our ablation studies show that each reward component improves its corresponding physical dimension, while the combined reward achieves the best overall trade-off across all dimensions. Finally, we further show that training with PhyMotion preserves general video generation quality on VBench and VBench-2.0 [Huang et al., 2024a, Zheng et al., 2025], while introducing only modest overhead through pipelined reward computation.

#### 2 Related Work

RL-based post-training and video reward models. Reinforcement learning has become a central paradigm for post-training generative models [Ouyang et al., 2022, Christiano et al., 2017, Shao et al., 2024b, Guo et al., 2025]. For diffusion, DDPO [Black et al., 2024], DPOK [Fan et al.,

- 2024], and Diffusion-DPO [Wallace et al., 2024] optimize policy gradients or preference objectives on reverse sampling, while DanceGRPO [Xue et al., 2025], Flow-GRPO [Liu et al., 2025a], and DiffusionNFT [Zheng et al., 2026] extend on-policy RL to video generators; recent work further scales RL to distilled autoregressive video models [Zhang et al., 2026, Wang et al., 2026a, Lu et al.,
- 2025, He et al., 2025]. The reward itself is typically a learned scalar over general human preferences, e.g., HPSv3 [Ma et al., 2025], VideoReward [Liu et al., 2025b], VisionReward [Xu et al., 2024a], and VideoScore [He et al., 2024], or an aggregate of VBench-derived feature-model scores [Huang

- et al., 2024a,b]. Such rewards are structurally agnostic to articulated human motion and provide no dimension-specific feedback. In contrast, we derive the reward from a structured 3D body model, enabling the RL optimization signal to incorporate explicit physical priors on human kinematics, balance, and dynamics.

Human motion in video generation and evaluation. A growing line of work improves human motion by directly modifying the generator or incorporating motion priors. Pose-conditioned methods [Wang

- et al., 2024b, Ma et al., 2024, Xu et al., 2024b, Hu, 2024, Zhou et al., 2024, Zhu et al., 2024, Shao et al., 2024a, Wang et al., 2025, 2026b] condition on external 2D poses or 3D SMPL priors, while VideoJAM [Chefer et al., 2025] and EchoMotion [Yang et al., 2026] co-predict motion representations with video. However, these methods typically require architectural changes, external pose conditioning, or additional motion data supervision. On the evaluation side, beyond distribution-level statistics [Salimans et al., 2016, Heusel et al., 2017, Unterthiner et al., 2019], recent benchmarks [Huang et al., 2024a,b, Zheng et al., 2025, Liu et al., 2024a, 2023, Ling et al., 2025] decompose evaluation

###### Ours VBench VideoAlign VideoPhy

(a) Text Prompt: A person in athletic gear swings a sledgehammer with both hands.

[Figure 12]

[Figure 13]

Generated Video

[Figure 14]

SMPL Mesh

Kinematic: -1.92

Contact: -0.07

Dynamic: -1.34

- (b) Text Prompt: A woman performs a powerful flying kick.

- (c) Text Prompt: A single person is performing a baseball pitch.

[Figure 15]

[Figure 16]

Generated Video

Previous Metrics VBench: 1.43 VideoAlign: 0.84 VideoPhy: 0.8

[Figure 17]

SMPL Mesh

Kinematic: -1.24

Contact: -2.27

Dynamic: -1.91

Our Metrics: Dynamic: -1.86 Kinematic: -1.71 Contact: -0.16 Our avg: -1.24

Previous Metrics VBench: 1.13 VideoAlign: 0.48 VideoPhy: 0.8

[Figure 18]

[Figure 19]

Generated Video

Our Metrics: Dynamic: -1.84 Kinematic: -2.84 Contact: 0.34 OurOuravg:Metrics:

[Figure 20]

VBench: 0.97 VideoAlign: -0.34 VideoPhy: 0.8

SMPL Mesh

Kinematic: -1.71

Contact: -0.16

Dynamic: -1.86

Our Metrics: Dynamic: -1.91 Kinematic: -1.24 Contact: -2.27 Our avg:

- Figure 2: PhyMotion identifies distinct physical failure modes overlooked by existing 2D metrics. Each example shows a generated video and its recovered SMPL trajectory. Colored SMPL frames highlight the dominant motion errors: kinematic (articulated-body inconsistency), contact (unstable body–environment interaction), and dynamic (physically infeasible motion). All reported metric values are quartile-normalized z-scores.

into interpretable dimensions, but still rely largely on perceptual video cues, providing only indirect evidence of articulated motion quality. VLM-based evaluators [Bansal and Grover, 2024, Huang

- et al., 2025, Meng et al., 2024, Wang et al., 2024c, Sun et al., 2024, Hong et al., 2025] probe physical commonsense and motion understanding, but remain limited for fine-grained human articulation. Related 3D motion evaluators such as MBench [Lin et al., 2026] and MotionCritic [Wang et al., 2024a] assess motion quality directly, but are not designed as rewards for synthetic videos. Our work connects generation and evaluation through a structured 3D motion reward for human video generation. Without modifying the generator architecture, we apply the reward for RL post-training, where each component operates in 3D space, provides continuous supervision, and targets specific motion failure modes.

#### 3 Physics-Grounded 3D Motion Evaluation

We present our evaluation in two parts: we first characterize the failure modes of human motion that are fundamentally unidentifiable from pixel observations (Sec. 3.1). We then introduce a physics-grounded evaluation protocol that recovers 3D motion along three feasibility axes (Sec. 3.2).

###### 3.1 The Failure Modes of 2D Motion Evaluation

Fig. 2 illustrates a key limitation of existing 2D motion evaluators, such as VBench [Huang et al., 2024a, Zheng et al., 2025], VideoAlign [Liu et al., 2025b], and VideoPhy [Bansal and Grover, 2024]. Although these metrics can assign high scores based on perceptual quality, text alignment, or visual commonsense, the generated videos may still contain physically implausible human motion. These failures require reasoning about human bodies’ physical quantities beyond RGB appearance: whether the 3D body configuration is anatomically valid over time, whether the body is properly supported by ground contacts, and whether the motion can be produced by plausible forces and torques. Therefore, we identify three classes of motion failures that 2D evaluators are structurally unable to detect.

Kinematic inconsistency over time. Frame-level visual metrics may judge individual poses as plausible, but fail to detect whether the recovered body configuration is anatomically valid in 3D.

Such metrics can miss abnormal joint configurations, unrealistic joint velocities, and self-penetrations that become clear only after reconstructing the articulated body. As shown in Fig. 2(a), although the frames appear to show a swing, the reconstructed 3D body reveals that the hand penetrates the hip, a self-penetration error that existing 2D rewards fail to detect (e.g., VideoPhy assigns a high normalized z-score of 0.8).

Physically inconsistent contact. 2D evaluators often score a video high even when the body is floating, sliding, penetrating the ground, losing balance, or changing contact state inconsistently over time. As shown in Fig. 2(b), the RGB frames appear to show a powerful flying kick, but the recovered 3D trajectory reveals an implausible support pattern: the body becomes airborne toward the end of the video. Such contact and balance failures are difficult for 2D rewards to detect (e.g. VideoAlign assigns a high normalized z-score of 1.21).

Dynamically infeasible motion. A video can appear smooth and semantically correct while still requiring physically implausible forces to execute. As shown in Fig. 2(c), the frames depict a reasonable baseball pitch, but the recovered 3D motion indicates that the body would require excessive joint torques to reproduce the observed pose transition. Such dynamic failures are difficult for existing

- 2D metrics to detect because they do not test whether the motion can be produced by a physically valid human body (e.g. VBench assigns a high normalized z-score of 1.43).

3.2 PhyMotion: Physics-Grounded 3D Rewards

To address the failure modes in Sec. 3.1, we convert each generated video into a physically interpretable 3D motion representation. Given a video {It}Tt=1 at frame rate f, we recover an SMPL-X [Pavlakos et al., 2019] trajectory with GVHMR [Shen et al., 2024]. We denote the recovered pose by qt, the

- 3D body joints by Xt = {xt,j}Jj=1, and the joint angular velocity by vt,j = f(qt+1,j − qt,j). We retarget the motion to a MuJoCo [Todorov et al., 2012] human model with explicit mass, inertia, joint limits, and contact geometry, and run inverse dynamics [Winter, 2009] to estimate joint torques τt

and ground reaction forces FGRFt . We then score each video along three complementary axes. This conversion makes the failure modes in Sec. 3.1 directly measurable. As shown in Fig. 2, the recovered

- 3D trajectory lets us identify whether a video fails due to kinematic inconsistency, implausible contact and balance, or dynamically infeasible motion.

Kinematic feasibility. Kinematic feasibility measures whether the recovered body motion is smooth and anatomically valid. We combine three normalized violations: angular-velocity violation vvel, computed by thresholding ∥vt,j∥2 against a clean-motion tolerance; self-penetration violation vspen, computed as the fraction of frames with intersecting non-adjacent mesh triangles; and joint-limit violation vlim, computed as the fraction of joints whose angles fall outside the valid MuJoCo range. The final score is Fkin = 1 − 13(vvel + vspen + vlim), where higher values indicate smoother motion with fewer anatomical violations.

Contact feasibility. Contact feasibility measures whether the body interacts with the ground plausibly. We infer binary foot–ground contacts ct,k ∈ {0,1} for each foot k ∈ {L,R} from foot height and velocity, where L and R denote the left and right foot. We compute four normalized violations: foot sliding vslip, which measures displacement of a contacted foot; ground penetration vgpen, which measures how far the foot moves below the floor; foot floating vfloat, which flags frames where neither foot is in contact while the body does not follow a plausible ballistic trajectory; and balance violation vbal is the fraction of frames where the projected center of mass falls outside the support polygon of contacting feet. The score is Fcon = 1 − 14(vslip + vgpen + vfloat + vbal).

Dynamic feasibility. Dynamic feasibility measures whether the recovered motion can be replayed by a physically plausible human body. Using inverse dynamics in MuJoCo, we estimate the forces required to reproduce the trajectory and compute three scores: sτ penalizes unrealistically large joint torques, sGRF penalizes excessive ground contact forces whose magnitude exceeds a maximum plausible threshold Fmax, and smet penalizes motions with unusually high joint effort, measured by the torque–velocity work proxy Cmet = t,j |τt,jvt,j|. The final score is Fdyn = 13(sτ +sGRF +smet), with higher values indicating more physically realizable motion.

Together, these axes cover articulation quality, environment interaction, and physical realizability, producing continuous and interpretable metrics for both evaluation and reward-based post-training. We provide the detailed reward definitions and implementation specifications in Appendix D.

#### 4 RL Post-Training with PhyMotion

The evaluation metrics in Sec. 3.2 provide decomposed signals for different human motion failure modes. To use them as an optimization signal, we aggregate the three PhyMotion scores into a single motion reward:

1 3

(Fkin(v) + Fcon(v) + Fdyn(v)). (1) This reward is then used as the optimization target in the policy-learning objective below.

Rmotion(v) =

Problem formulation. We frame the video generation model as a policy πθ that maps a text prompt c to a generated video v. Our goal is to fine-tune πθ to maximize the expected reward R(v) while staying close to the base reference policy πref via a KL penalty:

θ(·|c) R(v) − λKL πθ ∥πref , (2)

Ec∼D, v∼π

max

θ

where λ controls the strength of the KL regularization to prevent mode collapse and preserve general video generation capability. The prompt distribution D is a curated set of human-motion-specific prompts covering diverse actions and scenarios (details in Sec. 5.2).

Policy optimization. We adopt the forward-process RL formulation of DiffusionNFT [Zheng et al., 2026], following Astrolabe [Zhang et al., 2026]. Given a generated video v0 ∼ πθ(· | c) with normalized reward r˜ ∈ [0,1], a noisy version vt is constructed at a randomly sampled timestep t. Using the current velocity predictor vθ and the old predictor vθ

, implicit positive and negative policies are defined via interpolation:

old

old − β vθ, (3) where β controls the interpolation strength. The policy loss contrasts these implicit policies against the target forward velocity vtarget:

+ β vθ, v− = (1 + β)vθ

v+ = (1 − β)vθ

old

Lpolicy = r˜∥v+ − vtarget∥22 + (1 − r˜)∥v− − vtarget∥22. (4)

This trajectory-free formulation requires only clean generated samples and avoids backpropagating through the reverse sampling chain, enabling memory-efficient and solver-agnostic training.

Reward integration. The reward Rmotion(v) in Equation (2) follows the equally weighted aggregation defined in Equation (1). Because the reward is decomposed into three independent axes, we can diagnose per-dimension improvement during training and detect reward hacking early. For example, if one axis improves disproportionately at the expense of others, the decomposition makes this immediately visible.

#### 5 Experiments

We organize our experiments around two central questions. First, we evaluate whether our physicsgrounded 3D metrics better align with human judgments of synthetic human video quality than existing video metrics (Sec. 5.1). Second, we use the same metrics as rewards for RL post-training and evaluate whether they improve human motion video generation (Sec. 5.2).

###### 5.1 Human Alignment Results of PhyMotion

Experiment setup. We compare against existing evaluators, including VBench [Huang et al., 2024a], VBench-2.0 [Zheng et al., 2025], VideoAlign [Liu et al., 2025b], and VideoPhy [Bansal and Grover, 2024]. To construct the annotation set, we sample prompts from the Motion-X dataset [Lin et al., 2023] and generate videos under the same prompt using six baseline video models: Wan-2.1

- 1.3B, Wan-2.2 5B, Wan-2.2 14B [Team Wan et al., 2025], Causal Forcing-1.3B [Zhu et al., 2026], EchoMotion-5B [Yang et al., 2026], and FastWan [Zhang et al., 2025]. For each sample, we form video pairs from different models based on the same prompt, so that annotators compare motion quality under identical text conditions rather than across different conditions. From these candidate pairs, we randomly select 1,200 comparisons for annotation. Each annotation presents two anonymized videos side by side with randomized left/right order, together with the shared text prompt. We recruit six annotators and ask them to judge which video is better along three criteria (body structure, balance,

Body Structure

Balance

Motion Naturalness

| |
|---|

| |
|---|

| |
|---|

90

Ours Existing metrics

AgreementwithHuman(%)

82.9

81.5

81.1

81.0

80.0

79.6

79.4

80

78.0

78.0

68.3

69.0

68.2

70

66.1

66.7

66.0

66.6 66.3

66.3

65.7

63.6

64.9

63.1

62.9

60.5

61.2

59.0

58.9

60

52.3

51.9

Random

51.6

50

40

32.2

31.0

30.6

30

Kin Contact DynamicVB-AnatomyVB-AestheticVB-SmoothVB-DynDegVB-T-Flicker VA-MQ VA-VQ VPhy-PC

- Figure 3: Agreement with human judgments for motion quality. Our metrics achieve the highest agreement compared with perceptual (VBench / VBench2) and learned (VideoAlign, VideoPhy) video metrics across three human-evaluation questions: body structure, balance, and motion naturalness.

- Table 1: Correlation with human judgments for motion quality. We report Spearman’s rank correlation (ρ) between automatic metric scores and human judgments across three questions. Best results are in bold, and second-best results are underlined.

Question VBench/VBench2 VideoAlign VideoPhy PhyMotion (Ours) Mot. Sm. Dyn. Deg. Aes. Q. Hum. An. Temp. Fl. VA-MQ VA-VQ VP-PC Kin. Con. Dyn. Overall

Body Structure +0.248 +0.236 +0.175 −0.135 +0.273 +0.155 +0.045 +0.168 +0.369 +0.290 +0.367 +0.391 Balance +0.238 +0.246 +0.171 −0.118 +0.235 +0.175 +0.053 +0.142 +0.314 +0.337 +0.316 +0.333 Motion Naturalness +0.260 +0.274 +0.208 −0.149 +0.278 +0.156 +0.067 +0.108 +0.375 +0.281 +0.389 +0.402 All Questions +0.248 +0.252 +0.185 −0.135 +0.262 +0.161 +0.055 +0.138 +0.353 +0.290 +0.358 +0.376

and motion naturalness) corresponding to the failure modes in Sec. 3.1. For each criterion, annotators choose among “Video A”, “Video B”, or “Tie”, allowing ambiguous cases to be excluded from decisive metric-alignment comparisons. Additional annotation details are provided in Appendix C.

Quantitative results. As shown in Fig. 3 and Table 1, our physics-grounded metrics achieve the strongest and most consistent alignment with human judgments under both pairwise agreement and Spearman’s rank correlation (ρ). In the agreement results, kinematic feasibility performs best on body structure (82.9%), while contact and dynamic feasibility remain consistently high across all criteria (78–81%); in contrast, existing perceptual and learned metrics mostly remain in the 50–66% range, with Dynamic Degree near chance. The same trend holds in correlation: overall feasibility (Rmotion) achieves the highest aggregate correlation (+0.376), outperforming the strongest existing metric, VBench2 Human Anatomy (+0.262), as well as VideoAlign-MQ (+0.161) and VideoPhy-PC (+0.138). The individual components also align with their intended diagnostic roles: kinematic feasibility is strongest for body-structure judgments, contact feasibility is strongest for balance, and dynamic feasibility is highly correlated with motion naturalness, providing non-redundant diagnostic signals that serve as stable, interpretable rewards for RL-based post-training (Sec. 5.2).

- 5.2 RL Post-Training Results of PhyMotion

- 5.2.1 Experiment Setup

Weapplyourstructured3Dmotionrewardtotwodistilled4-stepWan1.3Bbackbones: FastWan[Zhang et al., 2025] (bidirectional) and Causal Forcing [Zhu et al., 2026] (autoregressive). We recover SMPL motion with GVHMR [Shen et al., 2024] and compute physical feedback in MuJoCo [Todorov et al., 2012]. We fine-tune with LoRA adapters [Hu et al., 2022] using RL post-training; full hyperparameters are provided in Appendix E. Training uses 8 NVIDIA A100 80GB GPUs for 330 steps, taking 2.5 days. We compare against EchoMotion [Yang et al., 2026], Causal Forcing 1.3B [Zhu et al., 2026], FastWan 1.3B [Zhang et al., 2025], and Wan models including Wan 2.1 1.3B, Wan 2.2 5B, and Wan 2.2 14B [Team Wan et al., 2025]. To compare against VLM-based reward optimization, we also fine-tune the Causal Forcing 1.3B model with the VideoAlign Motion Quality (MQ) reward under the same training setting. We evaluate on two prompt sets: a human-motion prompt set of 22,471

- Table 2: Main results on human-motion video generation. We compare open-source baselines, a VLM-reward-trained baseline, and our structured 3D reward training. We report VBench perceptual metrics, VideoAlign scores, VideoPhy Physical Commonsense, and our structured 3D reward scores. All metrics are higher-is-better. Bold: best; underline: second best.

VBench/VBench2 VideoAlign VideoPhy PhyMotion (Ours)

Method

Mot. Sm. Dyn. Deg. Aes. Q. Hum. An. Temp. Fl. VA-MQ VA-VQ VP-PC Kin. Con. Dyn. Overall Open-source baselines

Wan2.2 14B [Team Wan et al., 2025] 0.9892 0.393 0.5141 0.943 0.9857 1.133 0.557 3.97 0.881 0.733 0.912 0.842 Wan2.2 5B [Team Wan et al., 2025] 0.9846 0.598 0.4622 0.823 0.9803 1.069 0.514 3.91 0.880 0.710 0.913 0.834 EchoMotion 5B [Yang et al., 2026] 0.9850 0.683 0.4756 0.836 0.9777 1.044 0.565 3.74 0.888 0.702 0.918 0.836 Wan 1.3B [Team Wan et al., 2025] 0.9868 0.700 0.4758 0.834 0.9787 1.213 0.621 3.86 0.862 0.705 0.904 0.824 Causal Forcing 1.3B [Zhu et al., 2026] 0.9895 0.631 0.5391 0.886 0.9819 1.241 0.575 4.06 0.927 0.739 0.948 0.871 FastWan 1.3B [Zhang et al., 2025] 0.9847 0.928 0.5035 0.876 0.970 1.307 0.680 3.90 0.908 0.734 0.916 0.853

VLM-reward trained baseline VideoAlign-MQ [Liu et al., 2025b] 0.9904 0.570 0.5229 0.910 0.9835 1.549 0.912 4.12 0.916 0.753 0.946 0.878

PhyMotion (Ours) trained with Causal Forcing 1.3B

###### 0.9956

###### 0.5589

###### 0.951

###### 0.9957

###### 0.982

###### 0.809

###### 0.988

###### 0.902

0.411

1.313

0.720

4.29

(+0.6%)

(-34.9%)

(+3.7%)

(+7.3%)

(+1.4%)

(+5.8%)

(+25.2%)

(+5.7%)

(+5.9%)

(+9.4%)

(+4.2%)

(+3.5%)

###### 4.39

###### 0.982

0.9927

0.420

0.5218

0.867

0.9928

1.318

0.687

0.805

0.952

0.913

FastWan 1.3B

(+0.8%)

(-54.7%)

(+3.6%)

(-1.0%)

(+2.4%)

(+0.8%)

(+1.0%)

(+12.6%)

(+8.1%)

(+9.7%)

(+3.9%)

(+7.0%)

prompts derived from Motion-X [Lin et al., 2023], and the standard VBench and VBench-2.0 prompt sets [Huang et al., 2024a, Zheng et al., 2025]. We report three metric groups: VBench/VBench-2.0 metrics, VLM-based rewards (VideoAlign-MQ/VQ [Liu et al., 2025b] and VideoPhy-PC [Bansal and Grover, 2024]), and our PhyMotion from Sec. 3.2.

###### 5.2.2 Quantitative and Qualitative Results

Quantitative results on automatic metrics. Table 2 shows that our structured 3D reward improves human-motion quality beyond the metrics used for optimization. Compared with the Causal Forcing 1.3B base model, our RL-post-trained model improves all external metrics except VBench Dynamic Degree, including VBench motion smoothness, aesthetic quality, human action, temporal flickering, VideoAlign, and VideoPhy. Notably, it improves VideoAlign video quality by +25.2% and VideoPhy physical commonsense by +5.7%, and even outperforms larger 5B/14B reference models on most metrics.

As expected, these gains are also reflected in our PhyMotion rewards, with improvements of +3.5% in overall feasibility. The same trend also holds for FastWan 1.3B: after post-training with PhyMotion, the model improves motion smoothness, aesthetic quality, temporal flickering, VideoAlign, VideoPhy, and all PhyMotion scores, including a +7.0% gain in overall feasibility. The only decreased external metric, VBench dynamic degree, measures the magnitude of pixel-level motion rather than motion plausibility. Similarly, prior work also observes that encouraging large motion amplitude can introduce jitter and reduce motion realism [An

###### Winning fraction heatmap

100%

|[Figure 21]| |
|---|---|
| | |
| | |
| | |

[Figure 22]

51 (n=76)

58 (n=72)

67 (n=69)

71 (n=67)

81 (n=71)

81 (n=73)

Ours 1.3B

49 (n=76)

55 (n=66)

69 (n=65)

76 (n=90)

74 (n=101)

74 (n=106)

Wan2.2 14B

75%

42

Causal Forcing 1.3B

- (n=72)

45 (n=66)

44 (n=40)

71 (n=56)

71 (n=62)

72 (n=73)

33 (n=69)

31 (n=65)

56 (n=40)

59 (n=56)

71 (n=50)

74 (n=69)

29 (n=67)

24 (n=90)

29 (n=56)

41 (n=56)

58 (n=79)

52 (n=72)

19 (n=71)

26 (n=101)

29 (n=62)

29 (n=50)

42 (n=79)

54 (n=74)

19

- (n=73)

Awinrate(vsB)

FastWan 1.3B

50%

Wan 1.3B

25%

EchoMotion 5B

26 (n=106)

28 (n=73)

26 (n=69)

48 (n=72)

46 (n=74)

Wan2.2 5B

0%

Ours1.3BWan2.214BCausalForcing1.3BFastWan1.3BWan1.3BEchoMotion5BWan2.25B

- et al., 2026, Bhowmik et al., 2026]. Furthermore, compared with the VLM-reward-trained VideoAlign-MQ baseline, our method achieves higher VideoPhy physical commonsense and stronger structured 3D feasibility across all dimensions, suggesting that explicit 3D physical rewards provide a more targeted training signal than generic VLM-based video rewards.

Figure 4: Winning fraction heatmap of different models. Each cell shows the win rate between models on the overall human preference evaluation, with ties counted as 0.5.

Quantitative results on human preference evaluation. To verify that improvements in automatic metrics translate to human-perceived motion quality, we further conduct a human preference evaluation across representative baseline models. We randomly sample 1,487 video pairs and ask human annotators the same preference questions as in Sec. 5.1. As shown in Table 3, our RL-post-trained model achieves the highest overall human preference Elo score, outperforming all baselines, including the much larger Wan2.2 14B model.

- Table 3: Human preference evaluation. We report Elo ratings computed from pairwise human preferences. Higher is better.

Model Body Bal. Motion All Q Wan2.2 5B 1376 1388 1384 1383 EchoMotion 5B 1386 1403 1374 1387 Wan 1.3B 1429 1440 1411 1427 FastWan 1.3B 1526 1521 1528 1525 Causal 1.3B 1562 1546 1553 1553 Wan2.2 14B 1600 1593 1618 1604 Ours 1.3B 1620 1610 1632 1621

Table 4: Ablation results on rewards. VB, VA, and VP denote compact summary metrics for VBench average, VideoAlign average, and VideoPhy Physical Commonsense, respectively.

Reward VB VA VP Kin. Con. Dyn. Ours Average 0.782 0.979 4.12 0.954 0.763 0.943 0.883 Only Kin. 0.767 0.904 3.86 0.963 0.730 0.951 0.881 Only Con. 0.773 0.897 3.74 0.911 0.772 0.917 0.867 Only Dyn. 0.766 0.915 3.91 0.948 0.734 0.958 0.880

- Table 6: Reward computation overhead. We report the per-video reward computation time and the effective training overhead introduced by each reward module.

HPSv3 VideoAlign-MQ VideoPhy-PC PhyMotion (Ours)

Time / video (s) 4.72 0.25 0.20 2.80 Effective overhead 35% 0.2% 0.2% 7%

Beyond aggregate Elo rankings, we additionally compute pairwise human preference win rates between models. For each model pair, we count a win as 1, a loss as 0, and a tie as 0.5, and report the average win rate between models. As shown in Fig. 4, our PhyMotion optimized model consistently wins against all compared baselines in direct head-to-head comparisons, including the much larger Wan2.2 14B model. This indicates that the human preference improvement of PhyMotion is not driven by a single favorable matchup, but is broadly consistent across diverse baseline comparisons.

Qualitative comparison. Fig. 5 provides qualitative comparisons across diverse human-motion prompts, including (a) martial arts, (b) multi-human dancing, (c) soccer, (d) handstand, (e) figure skating, and (f) floor exercise / side lifting. Compared with baseline models, our model better preserves the intended action structure over time. In challenging motions such as handstand, figure skating, and floor exercises (Fig. 5d–f), baseline models often exhibit unstable body support, floating feet or hands, implausible limb bending, or sudden pose discontinuities. In contrast, our model more consistently maintains physically plausible motions.

###### 5.2.3 Additional Analysis

We further analyze PhyMotion from three perspectives: the contribution of each reward component, whether reward post-training preserves general-domain video generation ability, and the computational overhead of our reward during training.

Effect of individual reward components. We ablate each reward component by training singlereward variants from the same Causal Forcing base model, using identical hyperparameters and 120 training steps for all variants. As shown in Table 4, each single-reward variant primarily improves its corresponding feasibility dimension. However, optimizing any single component alone does not yield the best overall performance. In contrast, the full reward achieves the highest overall feasibility while remaining competitive across all individual dimensions, indicating that jointly optimizing kinematic, contact, and dynamic feasibility produces a better-balanced human-motion prior.

Generalization to broader prompts. To verify that our human-centric reward does not over-specialize the model, we further evaluate on the standard VBench and VBench-

Table 5: Generalization to general prompts. Gray rows denote reference models at larger scales.

- 2.0 prompt sets. Since our model is initialized from Causal Forcing 1.3B, we use Causal Forcing 1.3B as the primary fair comparison and also report larger Wan2.2 models as references. As shown in Table 5, our model maintains comparable average VBench performance to the base model (0.654 vs. 0.657) and slightly improves on VBench-2.0 (0.462 vs. 0.458). These results suggest that PhyMotion improves human-centered physical plausibility without substantially sacrificing general video generation quality. We provide per-dimension comparisons in the Appendix B.

Model VBench VBench-2.0

Wan 2.2 5B 0.607 0.403 Wan 2.2 14B 0.657 0.510

Wan 1.3B 0.630 0.456 Causal Forcing 1.3B 0.657 0.458 Ours 0.654 0.462

(a) A young woman in traditional martial arts attire performs a powerful Kung Fu kick.

(b) A group of people in traditional Korean attire perform Ganggangsullae.

[Figure 23]

[Figure 24]

EchoMotion-5BCausalForcing-1.3BWan-1.3BOurs-1.3BEchoMotion-5BCausalForcing-1.3BWan-1.3BEchoMotion-5BOurs-1.3BCausalForcing-1.3BWan-1.3BOurs-1.3B

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

❌ Extra leg

❌ Extra arm

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

❌ Excessive hip torque

❌ Sudden extra-person appearance

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

❌ Floating body

❌ Person disappears

(c) A single person kicking a soccer ball in front of the goal.

(d) A person lunges forward, plants their hands, kicks their legs upward and down.

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

❌ Over-twisted legs

❌ Backward-facing head

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

❌ Extra arm

❌ Unphysical ground force

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

❌ Twisted body

❌ Over-twisted legs

(e) A young woman glides backward on ice skates, performing a fluid pivoting turn. (f) A person performs a side-lying leg lift exercise.

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

❌ Extra leg

❌ Extra head

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

❌ Excessive hip angular velocity

❌ Opposite-facing feet

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

❌ Missing arm

❌ Unconnected arm

- Figure 5: Qualitative comparison on diverse human-motion prompts. We compare our model against baseline models across challenging motion categories. Baseline models often produce artifacts, while our model maintains more physically plausible motion.

Reward computation overhead. Finally, we measure both standalone reward-evaluation latency and effective training overhead for PhyMotion and representative reward models, including HPSv3 [Ma et al., 2025], VideoAlign-MQ [Liu et al., 2025b], and VideoPhy-PC [Bansal and Grover, 2024]. HPSv3 is a widely used human-preference reward model, while VideoAlign-MQ and VideoPhy-PC measure motion quality and physical commonsense, respectively. For standalone latency, we generate a fixed batch of B=8 FastWan videos at 480×832, run each reward once for warm-up and five times for timing, and report the steady-state per-video average. However, standalone latency does not directly determine training overhead, because our training loop pipelines video sampling and reward computation: while the model samples the next batch, the reward for the previous batch is computed in parallel. Therefore, only the non-overlapped portion contributes to the end-to-end training time. We measure the effective overhead by comparing the epoch time of each reward

reward epoch −Tepochconst

configuration against a constant-reward baseline: Overhead = T

Tepochconst . As shown in Table 6, PhyMotion takes 2.80s per video as a standalone reward computation, but introduces only 7% effective training overhead because most of its computation is hidden behind video sampling. In

contrast, HPSv3 leaves a larger non-overlapped critical path and introduces 35% overhead. These results indicate that the proposed structured 3D reward is practical for RL post-training.

#### 6 Conclusion

We introduce PhyMotion, a structured, fine-grained motion reward for physics-grounded human video generation. Instead of relying only on perceptual or learned video-level signals, PhyMotion lifts generated videos into 3D human motion, grounds them in a physics simulator, and evaluates motion feasibility through kinematic, contact/balance, and dynamic consistency. Experiments show that PhyMotion aligns better with human judgments than existing video rewards, and our post-training experiment also shows that PhyMotion serves as an effective RL post-training objective, improving automatic metrics performance and human preference scores while preserving general video quality across different video-generation backbones.

#### 7 Acknowledgements

This work was supported by ONR Grant N00014-23-1- 2356, ARO Award W911NF2110220, and NSF-AI Engage Institute DRL-2112635. The views contained in this article are those of the authors and not of the funding agency.

#### References

Zhaochong An, Orest Kupyn, Théo Uscidda, Andrea Colaco, Karan Ahuja, Serge Belongie, Mar Gonzalez-Franco, and Marta Tintore Gazulla. Vggrpo: Towards world-consistent video generation with 4d latent reward. arXiv preprint arXiv:2603.26599, 2026.

Hritik Bansal and Aditya Grover. VideoPhy: Evaluating physical commonsense for video generation. In International Conference on Machine Learning (ICML), 2024.

Aritra Bhowmik, Denis Korzhenkov, Cees G. M. Snoek, Amir Habibian, and Mohsen Ghafoorian. Moalign: Motion-centric representation alignment for video diffusion models. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/ forum?id=OR0ySm4l9h.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In International Conference on Learning Representations (ICLR), 2024.

Hila Chefer, Uriel Singer, Amit Zohar, Yuval Kirstain, Adam Polyak, Yaniv Taigman, Lior Wolf, and Shelly Sheynin. VideoJAM: Joint appearance-motion representations for enhanced motion generation in video models. arXiv preprint arXiv:2502.02492, 2025.

Wei Chow, Jiageng Mao, Boyi Li, Daniel Seita, Vitor Guizilini, and Yue Wang. PhysBench: Benchmarking and enhancing vision-language models for physical world understanding. In International Conference on Learning Representations (ICLR), 2025.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. DPOK: Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems (NeurIPS), 2024.

Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Haoran He, Yuxiao Ye, Jie Liu, Jiajun Liang, Zhiyong Wang, Ziyang Yuan, Xintao Wang, Hangyu Mao, Pengfei Wan, and Ling Pan. GARDO: Reinforcing diffusion models without reward hacking. arXiv preprint arXiv:2512.24138, 2025.

Xuan He, Dongfu Jiang, Ge Zhang, Max Ku, Achint Soni, Sherman Siu, Haonan Chen, Abhranil Chandra, Ziyan Jiang, Aaran Arulraj, Kai Wang, Quy Duc Do, Yuansheng Ni, Bohan Lyu, Yaswanth Narsupalli, Rongqi Fan, Zhiheng Lyu, Yuchen Lin, and Wenhu Chen. VideoScore: Building automatic metrics to simulate fine-grained human feedback for video generation. arXiv preprint arXiv:2406.15252, 2024.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Wenyi Hong et al. MotionBench: Benchmarking and improving fine-grained video motion understanding for vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. InInternational Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

Li Hu. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8153–8163, 2024.

Yidong Huang, Zun Wang, Han Lin, Dong-Ki Kim, Shayegan Omidshafiei, Jaehong Yoon, Yue Zhang, and Mohit Bansal. Planning with sketch-guided verification for physics-aware video generation. arXiv preprint arXiv:2511.17450, 2025.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024a.

Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen, Ying-Cong Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503, 2024b.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. HunyuanVideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Jing Lin, Ailing Zeng, Shunlin Lu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, and Lei Zhang. Motion-X: A large-scale 3D expressive whole-body human motion dataset. Advances in Neural Information Processing Systems (NeurIPS), 2023.

Jing Lin, Ruisi Wang, Junzhe Lu, Ziqi Huang, Guorui Song, Ailing Zeng, Xian Liu, Chen Wei, Wanqi Yin, Qingping Sun, Zhongang Cai, Lei Yang, and Ziwei Liu. The quest for generalizable motion generation: Data, model, and evaluation. In International Conference on Learning Representations (ICLR), 2026.

Xinpeng Ling et al. VMBench: A benchmark for perception-aligned video motion generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-GRPO: Training flow matching models via online RL. arXiv preprint arXiv:2505.05470, 2025a.

Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, Xiaohong Liu, Fei Yang, Pengfei Wan, Di Zhang, Kun Gai, Yujiu Yang, and Wanli Ouyang. Improving video generation with human feedback. In Advances in Neural Information Processing Systems (NeurIPS), 2025b.

Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. EvalCrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024a.

Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22139–22149, 2024b.

Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, and Lu Hou. FETV: A benchmark for fine-grained evaluation of open-domain text-to-video generation. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multi-person linear model. ACM Transactions on Graphics (Proc. SIGGRAPH Asia), 34

(6):248:1–248:16, 2015.

Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, et al. Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678, 2025.

Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: pose-guided text-to-video generation using pose-free videos. In Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence and Fourteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’24/IAAI’24/EAAI’24. AAAI Press, 2024. ISBN 978-1-57735-887-9. doi: 10.1609/aaai.v38i5.28206. URL https://doi.org/10.1609/aaai.v38i5.28206.

Yuhang Ma, Yunhao Shui, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. HPSv3: Towards widespectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsense-based benchmark for video generation. arXiv preprint arXiv:2410.05363, 2024.

Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles? In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 948–958, 2026.

OpenAI. Sora: Creating video from text. https://openai.com/sora, 2024. https://openai.

com/sora.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems (NeurIPS), 35:27730–27744, 2022.

Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3D hands, face, and body from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training GANs. In Advances in Neural Information Processing Systems (NeurIPS), 2016.

Ruizhi Shao, Youxin Pang, Zerong Zheng, Jingxiang Sun, and Yebin Liu. Human4DiT: Free-view human video generation with 4D diffusion transformer. arXiv preprint arXiv:2405.17405, 2024a.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024b.

Zehong Shen, Huaijin Pi, Yan Xia, Zhi Cen, Sida Peng, Zechen Hu, Hujun Bao, Ruizhen Hu, and Xiaowei Zhou. World-grounded human motion recovery via gravity-view coordinates. In SIGGRAPH Asia Conference Proceedings, 2024.

Kewei Sui, Anindita Ghosh, Inwoo Hwang, Bing Zhou, Jian Wang, and Chuan Guo. A survey on human interaction motion generation. International Journal of Computer Vision, 134(3):113, 2026.

Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2VCompBench: A comprehensive benchmark for compositional text-to-video generation. arXiv preprint arXiv:2407.14505, 2024.

Team Wan, Ang Wang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5026–5033. IEEE, 2012. doi: 10.1109/IROS.2012.6386109.

Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. FVD: A new metric for video generation. In ICLR Workshop on Deep Generative Models for Highly Structured Data, 2019.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8228–8238, 2024.

Haoru Wang, Wentao Zhu, Luyi Miao, Yishu Xu, Feng Gao, Qi Tian, and Yizhou Wang. Aligning human motion generation with human perceptions. arXiv preprint arXiv:2407.02272, 2024a.

Tan Wang, Linjie Li, Kevin Lin, Yuanhao Zhai, Chung-Ching Lin, Zhengyuan Yang, Hanwang Zhang, Zicheng Liu, and Lijuan Wang. DisCo: Disentangled control for realistic human dance generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9326–9336, 2024b.

Yiping Wang, Xuehai He, Kuan Wang, Luyao Ma, Jianwei Yang, Shuohang Wang, Simon Shaolei Du, and Yelong Shen. Is your world simulator a good story presenter? A consecutive events-based benchmark for future long video generation. arXiv preprint arXiv:2412.16211, 2024c.

Zehan Wang, Tengfei Wang, Haiyu Zhang, Xuhui Zuo, Junta Wu, Haoyuan Wang, Wenqiang Sun, Zhenwei Wang, Chenjie Cao, Hengshuang Zhao, et al. WorldCompass: Reinforcement learning for long-horizon world models. arXiv preprint arXiv:2602.09022, 2026a.

Zun Wang, Jaemin Cho, Jialu Li, Han Lin, Jaehong Yoon, Yue Zhang, and Mohit Bansal. Epic: Efficient video camera control learning with precise anchor-video guidance. arXiv preprint arXiv:2505.21876, 2025.

Zun Wang, Han Lin, Jaehong Yoon, Jaemin Cho, Yue Zhang, and Mohit Bansal. Anchorweave: Worldconsistent video generation with retrieved local spatial memories. arXiv preprint arXiv:2602.14941, 2026b.

David A Winter. Biomechanics and motor control of human movement. John wiley & sons, 2009. Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan,

Shen Yang, Qunlin Jin, Shurun Li, et al. VisionReward: Fine-grained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059, 2024a.

Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. MagicAnimate: Temporally consistent human image animation using diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1481–1490, 2024b.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, and Ping Luo. DanceGRPO: Unleashing GRPO on visual generation. arXiv preprint arXiv:2505.07818, 2025.

Yuxiao Yang, Hualian Sheng, Sijia Cai, Jing Lin, Jiahao Wang, Bing Deng, Junzhe Lu, Haoqian Wang, and Jieping Ye. EchoMotion: Unified human video and motion generation via dual-modality diffusion transformer. In International Conference on Learning Representations (ICLR), 2026.

Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhengzhong Liu, and Hao Zhang. Fast video generation with sliding tile attention. arXiv preprint arXiv:2502.04507, 2025.

Songchun Zhang, Zeyue Xue, Siming Fu, Jie Huang, Xianghao Kong, Yue Ma, Haoyang Huang, Nan Duan, and Anyi Rao. Astrolabe: Steering forward-process reinforcement learning for distilled autoregressive video models. arXiv preprint arXiv:2603.17051, 2026.

Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu. VBench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. DiffusionNFT: Online diffusion reinforcement with forward process. In International Conference on Learning Representations (ICLR), 2026.

Jingkai Zhou, Benzhi Wang, Weihua Chen, Jingqi Bai, Dongyang Li, Aixi Zhang, Hao Xu, Mingyang Yang, and Fan Wang. RealisDance: Equip controllable character animation with realistic hands. arXiv preprint arXiv:2409.06202, 2024.

Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressivediffusiondistillationdonerightforhigh-qualityreal-timeinteractivevideogeneration. arXiv preprint arXiv:2602.02214, 2026.

Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Zilong Dong, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3D parametric guidance. In European Conference on Computer Vision (ECCV), pages 145–162, 2024.

Wentao Zhu, Xiaoxuan Ma, Dongwoo Ro, Hai Ci, Jinlu Zhang, Jiaxin Shi, Feng Gao, Qi Tian, and Yizhou Wang. Human motion generation: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(4):2430–2449, 2023.

#### Appendix

#### A Additional Qualitative Results A.1 Qualitative Analysis of Individual Reward Components

We further provide qualitative examples to illustrate the diagnostic role of each PhyMotion submetric. As shown in Fig. 6, each example is selected by filtering for videos that score poorly on one target submetric while scoring relatively well on the other two. This isolates failure cases primarily captured by a single component.

The examples show that the three submetrics capture complementary artifacts. Kinematic feasibility Fig. 6(a) detects articulated-body errors such as self-penetration, extra limbs and impossible joint angles. Contact/balance feasibility Fig. 6(b) captures unstable support, balance violations, and foot sliding. Dynamic feasibility Fig. 6(c) identifies motions requiring implausible forces and torques, such as excessive horizontal ground force. These results support the need for a decomposed reward: each component provides an interpretable signal for a distinct class of physical failures.

#### B Per-category Comparison on General Video Benchmarks.

In Fig. 7, the category-level results show that our model remains competitive across broad perceptual, temporal, and compositional dimensions, while retaining strong performance on human- and motionrelated categories such as subject consistency, motion smoothness, temporal flickering, human action, and human anatomy.

#### C Human Study Detail

This appendix describes the human preference studies used to evaluate whether PhyMotion aligns with human judgments of generated human-motion quality. We conduct two pairwise-comparison studies: one for validating our automatic feasibility metrics against human preferences, and one for ranking post-trained and baseline video generation models. Both studies were conducted with non-author participants including students and researchers in related areas. Participants compared short generated videos under the same text prompt and judged them along three axes: body structure, balance, and motion naturalness.

###### C.1 Task Design

Each annotation example consists of two short videos generated from the same text prompt by two different models. The videos are displayed side by side and played in sync. Model identities are hidden from annotators, and the left/right order is randomized independently for each pair.

For each video pair, annotators answer three preference questions, corresponding to the main perceptual dimensions studied in this paper:

- • Body structure: Which video shows a more anatomically correct human body? Annotators are asked to consider limb consistency, body-part distortions, missing or extra limbs, and self-intersections.
- • Balance: Which video shows more physically plausible body support? Annotators are asked to consider whether the person appears stably supported by the contacting feet, hands, or other body parts.
- • Motion naturalness: Which video shows motion that looks more like a real human action? Annotators are asked to consider temporal smoothness, realistic timing, and the absence of teleporting, flickering, or floating artifacts.

For each question, annotators choose one of three options: Video A, Video B, or Tie. The default option is Tie, so unanswered questions do not introduce a preference toward either side. The same instruction text is shown to every annotator, and a screenshot of the annotation interface is shown in Fig. 8.

- (a) Kinematic Errors A person stands still in a wide shot, repeatedly extending both arms forward and back

[Figure 65]

[Figure 66]

❌ Hands self-penetration

[Figure 67]

❌ Extra arm A person lies and repeatedly opening and closing the top knee

[Figure 68]

[Figure 69]

❌ Impossible joint angle A person lies and repeatedly opening and closing the top knee

[Figure 70]

- (b) Contact/Balance Errors

[Figure 71]

❌ Balance violation A figure skater performs a Biellmann spin on ice

[Figure 72]

[Figure 73]

❌ Balance violation person does inclined sit-ups on a tilted bench

[Figure 74]

[Figure 75]

❌ Foot sliding A person tiptoes sideways

[Figure 76]

- (c) Dynamic Errors

[Figure 77]

❌ Left-right leg identity swap A person dribbles a basketball on a court

[Figure 78]

[Figure 79]

❌ Excessive joint torque A person mimes rolling downhill while standing in place

[Figure 80]

[Figure 81]

❌ Excessive horizontal ground force A fit individual performs a dynamic jumping jack motion

[Figure 82]

[Figure 83]

Kinematic: -0.84 Contact: +0.64 Dynamic: +0.38

[Figure 84]

[Figure 85]

A person lies and repeatedly opening and closing the top knee

[Figure 86]

❌ Extra arm Kinematic: -1.17 Contact: +0.40 Dynamic: +0.60

[Figure 87]

[Figure 88]

[Figure 89]

A person lies and repeatedly opens and closes the top knee

[Figure 90]

❌ Impossible joint angle Kinematic: -1.28 Contact: +0.69 Dynamic: +0.63

[Figure 91]

[Figure 92]

[Figure 93]

- (b) Contact/Balance Errors A figure skater performs a Biellmann spin on ice

[Figure 94]

[Figure 95]

❌ Center of mass outside root Kinematic:+0.84 Contact: -0.85 Dynamic: +0.36

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

A person does inclined sit-ups on a tilted bench

[Figure 100]

❌ Center of mass outside root Kinematic: +0.70 Contact: -1.39 Dynamic: +0.62

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

A person tiptoes sideways

[Figure 105]

❌ Foot sliding Kinematic: +0.16 Contact: -1.34 Dynamic: +0.32

[Figure 106]

[Figure 107]

[Figure 108]

- (c) Dynamic Errors A person dribbles a basketball on a court

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

❌ Left-right leg identity swap Kinematic:+0.08 Contact: +0.21 Dynamic: -1.75

A person mimes rolling downhill while standing in place

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

❌ Excessive joint torque Kinematic: +0.32 Contact: +0.24 Dynamic: -1.71

A fit individual performs jumping jacks

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

###### ❌ Excessive force to the ground Kinematic: +0.15 Contact: +0.17 Dynamic: -1.83

- Figure 6: Qualitative examples selected by individual PhyMotion submetrics. Each row shows a generated video that scores poorly on one target submetric while scoring relatively well on the other two. The examples show complementary failures captured by kinematic feasibility, contact/balance feasibility, and dynamic feasibility. All scores are reported in z-scores after normalization.

###### C.2 Study Composition

We form the dataset using a wide variety of video models including Causal Forcing 1.3B [Zhu et al., 2026], FastWan 1.3B [Zhang et al., 2025], Wan 2.1 1.3B, Wan2.2 5B, Wan2.2 14B [Team Wan et al., 2025], and EchoMotion 5B [Yang et al., 2026]. Each pair contains two videos generated from the same prompt by different models. We filter out unsafe videos, nearly identical pairs, and pairs where both videos already have very high feasibility scores, since these provide little discriminative signal. After filtering incomplete sessions and low-coverage variants, the final Elo analysis uses 1,487 vote rows per question, or 4,461 axis-level annotations in total. Ties are counted as 0.5, and Elo ratings are computed with base rating 1500 and K = 32.

| | |
|---|---|
| | |

Wan 1.3B

Wan2.2 5B

Wan2.2 14B

Causal Forcing 1.3B

Ours 1.3B

Wan 1.3B

Wan2.2 14B

Ours 1.3B

Wan2.2 5B

Causal Forcing 1.3B

(a) VBench per-dimension comparison.

(b) VBench-2.0 per-dimension comparison.

- Figure 7: Per-category comparison on general video benchmarks. We show normalized radar charts for VBench and VBench-2.0 dimensions. Our model remains competitive across general perceptual, temporal, and compositional categories, while showing strong performance on humanand motion-related dimensions. These results support that PhyMotion improves human-centered physical plausibility without substantially sacrificing general video generation quality.

- Table 7: Inter-annotator agreement on the human pilot study. For each question, we report the mean pairwise agreement between human annotators on overlapping video pairs.

Question Body Structure Balance Motion Naturalness Mean Pairwise Agreement 95.9% 91.4% 96.9%

###### C.3 Inter-Annotator Agreement

We also compute inter-annotator agreement on overlapping video pairs in the pilot study. Since some comparisons are ambiguous, we treat Tie as compatible with either Video A or Video B; a disagreement is counted only when two annotators make hard opposite choices, i.e., one selects Video A and the other selects Video B. As shown in Table 7, annotators achieve high agreement across all three questions, suggesting that the criteria are consistently understood.

###### C.4 Human Study Significance

Correlation with human judgments. In addition to the correlation values reported in the main paper, we report the statistical significance of the metric–human alignment results in Table 8.

Human preference evaluation. To complement the compact human preference results in the main paper, we report the full per-question Elo ratings with bootstrap uncertainty in Table 9.

#### D Detailed Definition of PhyMotion Metrics

This appendix provides the detailed computation of the three PhyMotion feasibility scores used in the main paper: kinematic feasibility, contact feasibility, and dynamic feasibility. The goal of these metrics is to convert a generated human video into interpretable physical signals that can be used both for evaluation and as rewards for RL post-training.

Given a generated video v = {It}Tt=1 with frame rate f, we first recover an SMPL-X body trajectory using GVHMR. Let qt denote the recovered body pose at frame t, Xt = {xt,j}Jj=1 denote the 3D body joints, and Mt denote the body mesh. For SMPL-X, we use J = 55 joints and a mesh with

[Figure 124]

- Figure 8: Human preference annotation interface. Annotators compare two anonymized videos generated from the same prompt and answer three pairwise preference questions on body structure, balance, and motion naturalness.

V = 10,475 vertices and F = 20,908 triangular faces. We then retarget the recovered motion to a MuJoCo human model and compute physics-related quantities such as joint torques and ground reaction forces. We write ∆t = 1/f.

For foot-related metrics, we define two foot-sole vertex sets, VL and VR, corresponding to the left and right feet. These sets are precomputed from the SMPL-X template by selecting the lowest vertices on each foot. In our experiments, all metrics are computed from the GVHMR-recovered camera-frame trajectory, with f = 16.

###### D.1 Kinematic Feasibility

Kinematic feasibility measures whether the reconstructed human body is anatomically valid and temporally smooth. It penalizes three common artifacts: joints moving too fast, body parts intersecting each other, and unstable limb geometry. The final kinematic score is high when the motion has smooth joints, little self-penetration, and a stable body structure.

Joint angular velocity. Generated videos sometimes contain abrupt limb snapping, where a joint rotates too quickly between adjacent frames. We approximate the angular velocity of joint j at frame t by finite differences:

ωt,j = f dQ(qt+1,j,qt,j), where dQ(·,·) measures the distance between two joint rotations, such as the geodesic distance on SO(3). We then compute the fraction of joint-frame pairs whose angular velocity exceeds a plausible joint-specific threshold:

T−1

J

1 (T − 1)J

1 ωt,j > ωjmax .

vvel =

t=1

j=1

- Table 8: Correlation with human judgments and statistical significance. We report Spearman’s rank correlation (ρ) between automatic metric scores and human judgments across three questions, together with significance levels. Best results are in bold, and second-best results are underlined.

Metric Group Metric Body Structure Balance Motion Naturalness All Questions

VBench / VBench2

Aesthetic +0.248 ± 0.032 +0.238 ± 0.033 +0.260 ± 0.032 +0.248 ± 0.018 Motion +0.236 ± 0.035 +0.246 ± 0.034 +0.274 ± 0.034 +0.252 ± 0.019 Temporal +0.175 ± 0.032 +0.171 ± 0.035 +0.208 ± 0.034 +0.185 ± 0.020 Dynamic −0.135 ± 0.033 −0.118 ± 0.035 −0.149 ± 0.034 −0.135 ± 0.019 Anatomy +0.273 ± 0.032 +0.235 ± 0.033 +0.278 ± 0.033 +0.262 ± 0.018

VideoAlign

Motion +0.155 ± 0.035 +0.175 ± 0.036 +0.156 ± 0.034 +0.161 ± 0.020 Visual +0.045 ± 0.035 +0.053 ± 0.035 +0.067 ± 0.034 +0.055 ± 0.020

VideoPhy Physics +0.168 ± 0.033 +0.142 ± 0.035 +0.108 ± 0.035 +0.138 ± 0.020

PhyMotion(Ours)

Kinematic +0.369 ± 0.031 +0.314 ± 0.033 +0.375 ± 0.031 +0.353 ± 0.018 Contact +0.290 ± 0.032 +0.337 ± 0.033 +0.281 ± 0.032 +0.290 ± 0.018 Dynamic +0.367 ± 0.030 +0.316 ± 0.032 +0.389 ± 0.030 +0.358 ± 0.017 Overall +0.391 ± 0.030 +0.333 ± 0.032 +0.402 ± 0.030 +0.376 ± 0.017

- Table 9: Detailed human preference Elo ratings. We provide the full Elo ratings corresponding to the human preference evaluation summarized in the main paper. Ratings are computed from pairwise human preferences on body structure, balance, motion naturalness, and all questions. Higher is better, and uncertainty denotes bootstrap standard deviation.

Model Body Struct. Balance Motion Overall Ours 1620 ± 34 1610 ± 34 1632 ± 39 1621 ± 11 Wan2.2 14B 1600 ± 34 1593 ± 33 1618 ± 40 1604 ± 13 Causal-Forcing 1.3B 1562 ± 35 1546 ± 33 1553 ± 39 1553 ± 8 FastWan 1.3B 1526 ± 33 1521 ± 32 1528 ± 35 1525 ± 4 Wan 1.3B 1429 ± 37 1440 ± 33 1411 ± 39 1427 ± 15 EchoMotion 5B 1386 ± 36 1403 ± 35 1374 ± 38 1387 ± 14 Wan2.2 5B 1376 ± 35 1388 ± 34 1384 ± 39 1383 ± 6

Self-penetration. Another common failure is body-part penetration, such as a hand passing through the torso or a leg intersecting the body. For each frame, we detect intersections between non-identical mesh triangles using a BVH-based triangle-intersection test. Let F be the set of mesh triangles. We compute the percentage of intersecting triangle pairs:

(f1,f2) ∈ F2 : f1 ∩ f2 ̸= ∅, f1 ̸= f2 F

spent = 100 ·

.

We then average this quantity over time:

1 T

spen =

T

spent.

t=1

Because clean SMPL meshes can have small soft-body overlaps, we treat roughly 2% self-penetration as a normal baseline and regard values above 20% as severe. The normalized self-penetration violation is

spen − 2 18

,0,1 .

vspen = clip

Joint-limit violation. Generated videos may contain anatomically impossible poses, such as a knee bending backward, a hip twisting too far, or an arm rotating beyond a plausible range. To detect these failures, we compare each reconstructed joint angle with the valid joint range defined by the MuJoCo human model. Let qt,j denote the pose parameter of joint j at frame t, and let [qjmin,qjmax] denote its valid range. We define the joint-limit violation as the fraction of joint-frame pairs that fall outside this

range:

1 TJ

vlim =

T

J

1 qt,j < qjmin ∨ qt,j > qjmax .

t=1

j=1

For multi-dimensional joints, the violation is computed per degree of freedom and then averaged across all joint dimensions. This term penalizes poses that may look locally plausible in RGB frames but require anatomically invalid joint configurations in 3D.

Finally, the kinematic feasibility score is

Fkin(v) = 1 −

1 3

(vvel + vspen + vlim).

Higher Fkin indicates smoother and more anatomically plausible body motion.

###### D.2 Contact Feasibility

Contact feasibility measures whether the body interacts with the ground in a physically plausible way. It penalizes foot sliding, ground penetration, floating feet, and poor balance.

For each foot k ∈ {L,R}, let ut,v denote the 3D position of mesh vertex v at frame t. We compute the foot height using the lowest sole vertex:

u(t,vz).

ht,f = min v∈Vf

We also compute the average foot-sole position:

1 |Vf| v∈V

pt,f =

f

ut,v.

A foot is considered to be in contact with the ground if it is both close to the ground and nearly stationary:

ct,k = 1[ht,f < 0.02 ∧ ∥p˙t,f∥2 < 0.05].

Foot sliding. If a foot is in contact with the ground, it should not slide significantly. We therefore measure the amount of foot displacement while contact is active:

slipt,f = ct,k · ∥p˙t,f∥2 · ∆t. The average foot-sliding violation is

T

- 1

- 2T

vslip =

slipt,f.

t=1 k∈{L,R}

Ground penetration. Generated or reconstructed feet can sometimes go below the ground plane. We measure penetration depth as

gpent,f = max(0,−ht,f), and average it over both feet and all frames:

T

- 1

- 2T

vgpen =

gpent,f.

t=1 k∈{L,R}

Foot floating. A common artifact is that the root body moves while the feet do not make plausible contact with the ground, producing floating or whipping feet. To capture this, we compare the foot motion relative to the root with the root motion itself:

ρt,f = ∥(pt,f −˙ xt,0)∥2 ∥x˙t,0∥2 + ε

,

where xt,0 denotes the pelvis/root joint. We flag a frame if the foot moves implausibly slowly relative to the root, or implausibly fast:

ρt,f < 0.6 or ρt,f > 1.75. The floating violation is the fraction of frames where such implausible foot-root motion is detected:

- 1

- 2T

vfloat =

T

1[ρt,f < 0.6 ∨ ρt,f > 1.75].

t=1 k∈{L,R}

In implementation, we additionally include sequence-level checks for sustained non-contact and whole-body floating, which help detect cases where both feet remain airborne with an irregular trajectory.

Balance. A physically plausible standing or walking person should keep the projected center of mass close to the support region formed by the contacting feet. We approximate the center of mass as a weighted average of body joints, giving the pelvis three times the weight of other joints:

1 j wj

Ct =

J−1

wjxt,j, w0 = 3, wj>0 = 1.

j=0

Let Pt denote the support polygon, defined as the convex hull of the contacting ankle positions in the ground plane. We compute the distance from the projected center of mass Ct(xy) to this support polygon:

1.0, if there is no support contact, mine∈∂P

dt =

diste(Ct(xy)), otherwise. The balance violation is the normalized average distance:

t

1 T

vbal =

T

clip(dt,0,0.5) 0.5

.

t=1

We use this continuous balance measure instead of a binary stable/unstable indicator because binary thresholds often saturate and provide less useful training signal.

The final contact feasibility score is

1 4

Fcon(v) = 1 −

(vslip + vgpen + vfloat + vbal),

where vslip and vgpen are normalized to [0,1] using fixed thresholds. Higher Fcon indicates more plausible body-ground interaction.

###### D.3 Dynamic Feasibility

Dynamic feasibility measures whether the recovered motion could be produced by a physically plausible human body. It penalizes motions that require excessive ground reaction forces, excessive joint torques, or unusually large mechanical effort.

Ground reaction force. We estimate the ground reaction force needed to produce the observed center-of-mass acceleration. Let m = 70kg be the body mass and g = 9.81m/s2 be gravity. Using Newton’s second law, we estimate

  

  .

- mC¨t(x)
- mC¨t(y)

FtGRF =

m(g + C¨t(z))

We penalize vertical forces larger than 3 times body weight:

T

1 T

vgrfvert =

1 (FtGRF)(z) > 3mg .

t=1

We also penalize horizontal forces larger than 0.5 times body weight:

T

1 T

vgrfhoriz =

###### 1 (FtGRF)(x)2 + (FtGRF)(y)2 > 0.5mg .

t=1

The ground-reaction-force score is

- 1

- 2

vgrfvert + vgrfhoriz .

sGRF = 1 −

Joint torque. We estimate the torque required to drive each joint using a simple segment-level inertia model:

###### τt,j = Ij∥x¨t,j∥2,

where Ij is the approximate inertia for joint j. In our implementation, we use Ij = 1.0kg m2 for all joints.

For each joint type, we compute the fraction of frames where the estimated torque exceeds a plausible limit:

T

1 T

vtorque(j) =

1 τt,j > τmax(j) .

t=1

We use torque limits of 200, 300, 400, and 200Nm for the ankle, knee, hip, and spine, respectively. The torque score is

J

1 J

vtorque(j) .

sτ = 1 −

j=1

Mechanical effort. Even if a motion does not exceed a single force or torque threshold, it may still require unrealistically large total effort. We therefore compute a mechanical-work proxy by multiplying joint torque with joint velocity:

J−1

T

τt,j∥x˙t,j∥2∆t.

MET =

t=1

j=0

We convert this value into a normalized score:

MET 10000

smet = max 0,1 −

.

The final dynamic feasibility score is

1 3

Fdyn(v) =

(sτ + sGRF + smet).

Higher Fdyn indicates that the motion requires more plausible forces, torques, and total effort.

- D.4 Overall Reward The final PhyMotion reward is the average of the three feasibility axes:

1 3

Rmotion(v) =

(Fkin(v) + Fcon(v) + Fdyn(v)).

All violation terms are clipped or normalized to [0,1], so higher scores consistently indicate more physically plausible human motion. This decomposition also makes the reward interpretable: kinematic feasibility captures body-shape and joint-motion artifacts, contact feasibility captures body-ground interaction failures, and dynamic feasibility captures physically infeasible forces and motions.

- Table 10: Training hyperparameters for PhyMotion training on Causal Forcing-1.3B and FastWan1.3B. For the training time of FastWan, part of the GPU was running another job, so the reported time is longer than expected.

Setting Causal Forcing-1.3B FastWan-1.3B Backbone & reward Base model Causal Forcing-1.3B (chunkwise) FastWan-1.3B (bidirectional) Dataset MotionX (21,348 training prompts) MotionX (21,348 training prompts) Resolution / frames 480 × 832, 45 frames 480 × 832, 45 frames LoRA

Rank r 256 256 Scale α 256 256 Dropout 0 0

Optimization Optimizer AdamW AdamW Learning rate 1 × 10−5 1 × 10−5 (β1, β2) (0.9, 0.999) (0.9, 0.999) Adam ϵ 1 × 10−8 1 × 10−8 Weight decay 1 × 10−4 1 × 10−4 Gradient clipping ∥g∥2 ≤ 1.0 ∥g∥2 ≤ 1.0 Mixed precision bf16 bf16

NFT objective KL interpolation β 0.1 0.1 KL coefficient λ 1 × 10−4 1 × 10−4 Advantage normalization per-prompt, |A| ≤ 5 per-prompt, |A| ≤ 5

Rollout / sampling Denoising step list {1000, 750, 500, 250} {999, 856, 599} num_frame_per_block 3 N/A (bidirectional, full-sequence) KV-cache enabled n/a (bidirectional) Guidance scale 1.0 (post-distillation) 1.0 (post-distillation) Samples per prompt 24 24 Rollout batches per epoch 16 16

EMA EMA on LoRA params yes yes EMA start step 0 0

Schedule & hardware Per-rank micro-batch size 3 3 Gradient accumulation steps 16 16 Reported step 330 330 GPUs 8× A100 80GB 8× A100 80GB Wall-clock ∼ 60.9 hours ∼ 66 hours (crowded GPU)

#### E Training Details

We provide the full training and rollout configuration used for our method in Table 10. Unless otherwise specified, all experiments use the same configuration. We report the checkpoint at training step 330.

##### F Licenses for Existing Assets We show the licenses of all assets we use in Table 11

- Table 11: Licenses of external assets, models, datasets, codebases, and evaluation tools used in this work.

Asset / Method Website / Source License Wan series [Team Wan et al., 2025] https://github.com/Wan-Video/

Apache-2.0 Causal Forcing [Zhu et al., 2026] https://github.com/thu-ml/

Wan2.2

Apache-2.0 FastWan [Zhang et al., 2025] https://github.com/hao-ai-lab/

Causal-Forcing

Apache-2.0 EchoMotion [Yang et al., 2026] https://github.com/D2I-ai/

fastvideo

CC BY-NC 4.0 VideoAlign [Liu et al., 2025b] https://github.com/

EchoMotion

MIT VideoPhy[BansalandGrover,2024] https://github.com/

KlingAIResearch/VideoAlign

MIT GVHMR [Shen et al., 2024] https://github.com/zju3dv/

Hritikbansal/videophy

Educational, research, and non-profit use only

GVHMR

Motion-X [Lin et al., 2023] https://github.com/

Non-commercial scientific research only

IDEA-Research/Motion-X

Apache-2.0 Astrolabe [Zhang et al., 2026] https://github.com/

VBench / VBench-2.0 [Huang et al., 2024a, Zheng et al., 2025]

https://github.com/Vchitect/ VBench

Apache-2.0

franklinz233/Astrolabe

SMPL [Loper et al., 2015] https://smpl.is.tue.mpg.de/ SMPL-Body Creative Commons License / CC BY 4.0

MuJoCo [Todorov et al., 2012] https://github.com/

Apache-2.0

google-deepmind/mujoco

