## Humanoid-GPT: Scaling Data and Structure for Zero-Shot Motion Tracking

Zekun Qi1,2,∗, Xuchuan Chen1,2,∗, Dairu Liu2,∗, Chenghuai Lin2,∗, Yunrui Lian1,2, Sikai Liang2,3, Zhikai Zhang1,2, Yu Guan1,2, Jilong Wang2, Wenyao Zhang2,3, Xinqiang Yu2, He Wang2,4,†, Li Yi1,5,†

1Tsinghua University 2Galbot Inc. 3Shanghai Jiao Tong University 4Peking University 5Shanghai Qi Zhi Institute ∗Equal Contribution, †Corresponding author

# arXiv:2606.03985v1[cs.RO]2Jun2026

We introduce Humanoid-GPT, a GPT-style Transformer with causal attention trained on a billion-scale motion corpus for whole-body control. Unlike prior shallow MLP trackers constrained by scarce data and an agility–generalization trade-off, Humanoid-GPT is pre-trained on a 2B-frame retargeted corpus that unifies all major mocap datasets with large-scale in-house recordings. Scaling both data and model capacity yields a single generative Transformer that tracks highly dynamic behaviors while achieving unprecedented zero-shot generalization to unseen motions and control tasks. Extensive experiments and scaling analyses show that our model establishes a new performance frontier, demonstrating robust zero-shot generalization to unseen tasks while simultaneously tracking highly dynamic and complex motions.

Date: June 3, 2026 Code: https://github.com/GalaxyGeneralRobotics/Humanoid-GPT/ Project Page: https://qizekun.github.io/Humanoid-GPT/

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

### Humanoid-GPT

[Figure 5]

Data Frame Scale

2.0B

Generative Pre-Training Scalable Architecture

[Figure 6]

19.5M

7.2M

[Figure 7]

[Figure 8]

PD Target

[Figure 9]

AMASS MotionX++ Ours

###### Large Scale Mocap & Human Video

[Figure 10]

[Figure 11]

Causal Transformer

80M/ 22M

Model Param.

1.1M 0.25M

. . .

Observation Sequence

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

BeyondMimic TWIST Ours

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Realtime Teleoperation & Dance without finetuning Zero-shot Dance, Jump, Bend over, Kungfu

Figure 1 Overview of Humanoid Generative Pre-Training for Zero-Shot Motion Tracking. We introduce Humanoid-GPT, a Transformer-based humanoid tracker trained on an unprecedented 2B-frame retargeted corpus, whose scaling in data and model capacity yield a universal controller that zero-shot tracks highly dynamic and unseen human motions beyond the agility and generalization limits of prior MLP trackers. The figure illustrates the real-time whole-body control and zero-shot dance generation capabilities of Humanoid-GPT.

1

- 1 Introduction

Artificial General Intelligence (AGI) for embodied agents is ultimately a generalization problem: a humanoid should execute robust whole-body behaviors under unseen tasks, styles, and environments [32, 3, 12, 13]. In language and vision, the most reliable path to generalization has been scale—larger data, larger models, and carefully designed training objectives [32, 28, 1, 13, 35, 30, 15]. Scaling is not only a recipe for better average performance; it often unlocks new capabilities and predictable trends [38].

Humanoid motion tracking has not followed this trajectory. Current trackers are typically shallow MLPs trained on small motion corpora. Even widely used datasets [24, 9, 17] contain only on the order of 104 trajectories (about 7.2M frames). This mismatch in scale creates a persistent failure mode: agility and generalization trade off. Trackers that excel on indomain agile motions often break on unseen styles, while trackers that generalize modestly tend to underfit complex dynamics and lose sharpness in tracking. Recent results make this tension clear: BeyondMimic [19] and ASAP [11] track agile motions well but do not generalize zero-shot to unseen movements; TWIST [42] and UniTracker [41] generalize better but struggle on highly dynamic actions.

We argue that this trade-off is not fundamental. It is a symptom of insufficient scale and mismatched training design. Simply adding more motion clips to the same pipeline is not enough. When the scale increases by orders of magnitude, three questions become decisive: ❶ What data should we train on, and how do we process the large, noisy data? ❷ What model structure matches the online tracking constraint and continues to improve with scale? ❸ What training recipe remains stable when the dataset grows from millions to billions of frames?

This paper answers these questions and presents Humanoid-GPT, a universal, online humanoid motion tracker built around the science of scaling.

Science of Scale. We construct a motion corpus at a new regime for tracking. We aggregate all widely available mocap sources, including Lafan1 [9], AMASS [24], Motion-X++ [43], PHUMA [16], and MotionMillion [7], and we add a large internally captured dataset for real-world coverage. After strict filtering, segmentation, and augmentation, we obtain

- 2B G1-retargeted motion frames/tokens, over 200× larger than prior tracker training sets. This scale forces changes that smaller systems can ignore: we redesign key reward components and re-tune sensitive

Table 1 Comparison of Humanoid-GPT with related works.

Method Low-level Tracker Agile Zero-shot #Frames

HumanPlus [8] Transformer × × 7.2M OmniH2O [10] MLP × × 7.2M ASAP [11] MLP ✓ × GMT [4] MoE-MLP ✓ × 6.0M UniTracker [41] MLP ✓ × 7.2M BumbleBee [37] Transformer ✓ × 7.2M TWIST [42] MLP × ∼ 9.2M Any2Track [46] MLP ✓ × 9.1M SONIC [23] MLP ✓ ✓ 100M Humanoid-GPT (ours) Transformer ✓ ✓ 2.0B

hyperparameters to keep training stable. Crucially, we provide the first systematic evidence that videoestimated motion can materially improve tracking when the model and the training set are scaled appropriately.

Modern Structure for Online Tracking. Motion tracking for control is inherently causal: at test time the policy cannot access future observations. Many existing trackers still rely on non-causal modeling choices or capacity-limited MLPs. We instead adopt a scalable Transformer with GPT-style causal attention. The model predicts per-joint PD targets with causal temporal attention, which aligns with the deployment constraint by design. This structure also scales cleanly with data and model size, unlike shallow MLPs and non-causal variants that saturate early.

Balanced Diversity Matters. More data does not automatically mean better generalization. In large motion corpora, common styles dominate and rare but important behaviors vanish in the long tail. We introduce Harmonic Motion Embedding (HME) as a representation learning tool that measures and organizes motion diversity directly from raw motion. HME enables diversity-aware, distribution-balanced sampling during training. Our analysis shows a simple but powerful insight: diversity and balance are both necessary. Diversity without balance still overfits frequent modes; balance without diversity caps capability.

Results and scaling laws. With these ingredients, Humanoid-GPT substantially improves both agility and zero-shot generalization. We further derive a scaling law for humanoid motion tracking that relates performance to data scale and model capacity, offering a concrete roadmap for future general-purpose whole-body control.

Compared with prior work that either applies Transformer controllers on limited motion hours [8] or scales MLP-based policies on hundreds of millions of frames [23], Humanoid-GPT is, to our knowledge, the first system that (i) distills a large library of RL motion experts into a single GPT-style tracker, (ii)

b) Scaling up Motion Experts

a) Data Curation

[Figure 21]

[Figure 22]

###### ~300 Experts

[Figure 23]

[Figure 24]

[Figure 25]

Humanoid GPT

𝒈𝒕 MotionExperts

[Figure 26]

𝒂 𝒕

###### ~2B

clips sampling

𝒔𝒕𝒑𝒓𝒊𝒗.

[Figure 27]

Frame

MotionX++

RL

[Figure 28]

Curated Motion Data

~20M

AMASS

~7M

[Figure 29]

LAFAN1

Filtering + Retargeting

c) Scaling up Action Transformer

[Figure 30]

freq. HME

temporal parallel training

[Figure 31]

embedding & clustering

amp.

Curated Motion Data

Online Retargeting

BC

[Figure 32]

𝒂𝒕

Causal Transformer

[Figure 33]

ref. qpos

###### . . .

History Sequence

𝒈𝒕

. . .

𝒆𝒕 𝑯

𝒆𝒕 𝑯 𝟑

𝒆𝒕

𝒔𝒕

- Figure 2 Overview of Humanoid-GPT. The system consists of three stages: (a) data curation and processing, (b) training PPO-based motion experts on clusters with keypoint-level rewards, and (c) distilling all experts into a single Transformer-based generalist policy via parallel DAgger supervision. The resulting Humanoid-GPT can take unseen or online retargeted motions as reference inputs and track them in a fully zero-shot manner.

2.2 Learning Human Motion Tracking

trains on a curated 2B-frame corpus, and (iii) systematically characterizes how data scale, model scale, and diversity balance jointly govern zero-shot agile motion tracking on real humanoid hardware.

Physics-based tracking aims to produce temporally coherent, dynamically feasible whole-body control from reference motions. Early works [21, 22, 6, 44] establish the paradigm of coupling imitation with contact-aware stability in simulation, while subsequent pipelines [8, 10, 5, 14, 40, 39, 18, 29, 42, 45] extend to real-world deployment on specific platforms.

#### 2 Related work

- 2.1 Large-scale Motion Data

Recent efforts shift toward generalization. GMT [4] employs Mixture-of-Experts with adaptive sampling; UniTracker [41] adopts a CVAE-based teacherstudent framework—both improving coverage but remaining constrained by limited motion scale. SONIC [23] scales to 100M frames with an MLP controller, yet MLP capacity saturates as data grows. HumanPlus [8] introduces a Transformer controller but trains it with standard PPO, missing the parallelism advantage inherent to Transformers.

Large-scale motion datasets have become essential for learning generalizable human motion tracking. Early datasets [9, 24, 17] offered high-quality but studio-constrained motions, limiting diversity. With video-based reconstruction and large-scale synthetic generation, recent datasets greatly expand motion coverage, incorporating diverse activities, styles, and subjects with multimodal supervision [20, 43, 7, 25]. More recently, [16] provides physically consistent motions with contact modeling, joint constraints, and reduced foot-sliding, offering stability benefits over purely kinematic sources. Together, these increasingly diverse and physically grounded datasets supply richer motion variety and stronger physical priors, forming key foundations for unified and robust human motion tracking systems.

As summarized in Table 1, existing methods either rely on curated motion sets or architectures that do not scale gracefully. Humanoid-GPT reframes tracking as GPT-style sequence modeling: we distill hundreds of RL experts into a causal Transformer trained on 2B frames, achieving strong zero-shot generalization where similarly-sized MLPs plateau.

#### 3 Scaling up Humanoid Motion Data

We first collect and curate large-scale human motion data to ensure the fidelity of motion dynamics, followed by retargeting these motions to the humanoid joint space.

- 3.1 Data Curation

Constructing a high-quality motion dataset is essential for ensuring the fidelity and diversity of motion dynamics in zero-shot humanoid motion tracking. Existing datasets [24, 9, 20, 25] often contain limited categories of motion capture sequences or exhibit inconsistencies in physical plausibility and spatial alignment, which constrain their generalization to complex whole-body tracking scenarios. Recent advances in large-scale motion generation [7] and physically grounded motion filtering [16] have introduced abundant and high-quality motion priors, significantly extending the coverage of motion distributions. To fully exploit these available resources, we curate a large-scale motion corpus by aggregating AMASS [24], LAFAN1 [9], MotionMillion [7], and PHUMA [16], shown in Fig. 2(a), encompassing a wide spectrum of human activities that serve as the foundation for Humanoid-GPT.

After assembling the datasets into a unified corpus, we employ an off-the-shelf motion retargeting framework [2] to map each human motion sequence into the 29-DoFs joint space of the Unitree-G1 humanoid. During this process, we further filter out sequences involving explicit object interactions—such as sitting on chairs, swimming, or stair climbing—to ensure the resulting motions are compatible with the humanoid’s actuation in a plain scene. To further enrich temporal variability and improve robustness to motion speed, we apply motion time-warping augmentation by uniformly accelerating and decelerating every sequence, expanding the dataset to approximately five times its original size. This yields a clean, physically consistent, and diverse dataset suitable for downstream reinforcement-learning based expert training.

- 3.2 Harmonic Motion Embedding

To balance the trade-off between motion coverage and training efficiency, we partition the full motion corpus into multiple clusters and train each expert on a specific motion subset. To cluster motions directly in the latent space, we propose a novel embedding representation called Harmonic Motion Embedding (HME). Concretely, we first train several Periodic Autoencoders [33] on different data partitions to extract per-joint periodic amplitudes and frequencies from

each motion sequence. For each sequence, we then aggregate the mean and standard deviation of these joint-level harmonic features to obtain its HME vector, yielding a compact and descriptive embedding for the entire corpus. Finally, we apply K-Means clustering over all HME embeddings using pairwise distances as the similarity metric, producing roughly 300 motion clusters. Each cluster contains about 1k–2k sequences, offering strong intra-cluster consistency while preserving broad coverage of the global motion distribution.

#### 4 Scalable Generative Tracker

We present the Humanoid-GPT framework. Built via a two-stage pipeline: RL-trained motion experts followed by transformer distillation, our model enables humanoids to track arbitrary human motions without any finetuning.

4.1 Training Motion Experts

To enable diverse motion priors for the HumanoidGPT, we train multiple motion experts to collectively cover the dynamic distribution present in the dataset.

On each cluster, we train a PPO-based policy to track all the sequences inside the cluster, presenting in Fig 2(b). The policy is formulated as π : G × S  → A, which maps the input reference joints and humanoid proprioceptive observations to low-level motor actions. At each time step t, the policy receives the current privileged robot state spriv.t along with the target reference pose qtref extracted from the motion clip. The state spriv.t encodes per-joint positions and velocities, the root’s angular velocity, projected gravity, and the previous control action. The policy outputs per-joint actions at, which are converted into actuator torques through a PD controller. The motion tracking objective is to drive the robot’s state to match the target pose gt = qtref while maintaining balance and dynamic stability. To enforce physically grounded tracking, the reward is computed at the body keypoint level, including position and velocity consistency terms for critical parts of the body (e.g., arms, hips, feet, pelvis). Let K denote the set of tracked body keypoints. For each keypoint k∈K at time t, let eposk,t ∈R3 and evelk,t∈R3 denote the position and velocity residuals between the humanoid and the reference motion, and let θk,t be the rotation error induced by the SO(3) log map. With positive keypoint weights wk and scaling factors αpos,αrot,αvel,

|-88<br>-87<br>-86<br>-85<br>-84<br>-83<br>-82<br>-81<br>-80<br><br><br>0.098 0.1 0.102 0.104 0.106 0.108 0.11 0.112 0.114 0.116<br><br>Log-volume<br><br>Geometric-std<br><br>| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | |Dive action a|rsity of mplitude| | | | |
| | | | | | | | | |
| |PHUM|A| | | | | | |
| | | | | |Our|Trainin|g data| |
|Acti|on| | | | | | | |
|space b|readth| | |AMASS|+LAFAN|1| | |
| | |AMAS|S| | | | | |
| | | | | | | | | |
<br><br>better diversity|
|---|

Log-volume

- Figure 3 Comparison of dataset diversity in the HME embedding space. Each bubble represents a dataset, where the horizontal and vertical axes denote gstd and log-volume respectively, and the bubble size reflects the relative amount of motion clips. Upper-right bubbles indicate broader coverage and higher diversity.

the abstract keypoint reward is formulated as Rkpt(t) = Rpos(t) + Rrot(t) + Rvel(t) + Rpenal(t), Rpos(t) =

wk exp −αpos∥eposk,t ∥1 ,

k∈K

wk exp(−αrotθk,t),

Rrot(t) =

k∈K

wk exp −αvel∥evelk,t∥1 .

Rvel(t) =

k∈K

(1) The exponential form softly penalizes deviations in position, orientation, and velocity across all body keypoints, and Rpanel(t) consists of several penalties like self-contacts and smoothness, promoting a globally accurate yet locally stable motion tracking.

During training, we randomly sample short motion clips as tracking targets and evaluate each expert using root pose error, velocity error, and stable tracking duration. These metrics ensure convergence toward physically consistent motion reproduction within each cluster. After training, only experts that achieve high-fidelity and long-horizon stability are retained, forming a diverse library of motion priors that provides Humanoid-GPT with a physically grounded initialization across heterogeneous motion regimes.

- 4.2 Building Zero-shot Foundational Tracker

The motion experts trained above can accurately reproduce physically-grounded motions within their own clusters, but tend to degrade sharply when encountering out-of-distribution motion targets. To bridge the gaps across motion domains and consolidate their specialized knowledge, we introduce a

distillation stage illustrated in Fig.2(c) that transfers the behaviors of all experts T into a single unified policy, adopting the DAgger [31] framework to distill the knowledge of all motion experts into a single generalist policy.

To distill expert behaviors efficiently, we reformulate the distillation process as a sequence modeling problem and employ a Transformer [36]-based generalist tracker Gθ. At each timestep t, the input token embedding et is constructed by concatenating the current proprioceptive state st and the target reference pose qtref from the motion clip. A sequence of length H containing such tokens {et−H+1,et−H+2,...,et} is fed into the Transformer with a temporal causal mask, allowing the model to capture long-horizon dependencies and temporal consistency across the trajectory. After a forward pass, actions at all output positions will be supervised by the corresponding history of teacher ti’s output, empowering the model to be trained on DAgger feedback efficiently over multiple timesteps in a single pass, shown in eq.(2). We use SmoothL1Loss as our loss function L.

ti(spriv.t−k ,gt−k)

aˆt−H+1:t =

concat

k∈[−H+1,0]

ti∈T

l = L(Gθ(et−H+1:t),aˆt−H+1:t)

(2)

During inference, we maintain a queue of maximal H history tokens as the input of transformer and use the output located in the last position as the current control target.

This design of our Humanoid-GPT model naturally exploits the Transformer’s inherent strengths of parallel sequence supervision and autoregressive temporal predicting, Moreover, because tokens at different positions attend to varying amounts of historical context, the trained model implicitly learns position-invariant temporal prediction, enabling it to output stable and physically consistent control targets even at the beginning of an episode, where historical information is scarce.

#### 5 Experiments

In this section, we aim to answer the following core questions that arise from our exploration of the scaledup humanoid motion tracker and the emergence of zero-shot generalization in Humanoid-GPT:

Scaling Law in Humanoid Tracking.

How does the zero-shot tracking ability of HumanoidGPT scale with the amount of training data and

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

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

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

###### Figure 4 Real-world experiments for our Humanoid-GPT. All motions illustrated are excluded from training to verify generalization capability. Our method can track diverse, complex and high-dynamic motion in a zero-shot manner.

- Table 2 Comparison of backbone architectures and scaling effects. Larger datasets and higher-capacity Transformers consistently improve stability and zero-shot tracking accuracy across all metrics.

Backbone #Train Tokens #Model Params.(M) SR ↑ MPJPE ↓ MPJVE ↓ RootVelErr ↓ MPKPE ↓

MLP (3-layer) 2M 0.25M 76.89 0.1191 0.6081 0.2304 100.49 TCN (8-layer) 2M 0.65M 81.48 0.0885 0.5716 0.2266 79.75

Humanoid-GPT-S 2M 5.7M 83.26 0.0853 0.5492 0.2049 62.65 Humanoid-GPT-S 20M 5.7M 86.02 0.0802 0.5210 0.1868 46.49 Humanoid-GPT-B 200M 22.1M 88.27 0.0793 0.5076 0.1820 44.78 Humanoid-GPT-B 2B 22.1M 90.43 0.0768 0.4891 0.1756 41.49 Humanoid-GPT-L 2B 80.4M 92.58 0.0735 0.4820 0.1785 40.99

model capacity? Does enlarging the motion corpus or Transformer parameters yield predictable improvements in stability and generalization?

Diversity and Balance Drives Generalization.

How can we quantify the diversity of motion data, and how does such diversity contribute to zero-shot tracking performance?

Powerful Structure Matters.

How does the architectural choice, such as the Transformer versus other alternatives, affect the model’s ability to capture long-horizon dynamics and generalize across unseen motions?

We systematically design experiments to address these questions, analyzing scaling trends, diversity–generalization relationships, and architectural contributions to robust zero-shot humanoid tracking.

- 5.1 Experiment Settings

We evaluate our method in both simulation and real-world settings, using the 29-DoF Unitree-G1 as the humanoid platform for tracking the target motion in all experiments. In simulation, we employ MuJoCo[34] as the physics engine to quantitatively assess the performance of different data and model variants under controlled conditions. For real-world evaluation, we adopt an online motion-retargeting pipeline that continuously converts the motion of a MoCap actor into the G1’s joint space, which serves as the online reference trajectories for our HumanoidGPT to track.

5.2 Analysis on Data Diversity

We first evaluate how the diversity of motion data influences the generalization ability of our tracker. We compare three datasets with an increasing amount of clips: the commonly used AMASS [24], the extended AMASS+LAFAN [9], recent PHUMA[16], and our curated large-scale dataset described in Sec. 3.1, which aggregates AMASS, LAFAN1, MotionMillion, and

PHUMA to cover a substantially broader range of human dynamics. To quantitatively measure dataset diversity in the latent space, we compute the geometric mean standard deviation (gstd) and the logarithmic volume of the covariance ellipsoid (log-volume) based on the HME embeddings introduced in Sec 3.1. To ensure that the size of each dataset does not bias the diversity estimation, we uniformly sample 10,000 embeddings from each dataset for evaluation. Given an embedding matrix X = [x1,x2,...,xN]⊤ ∈ RN×D, we compute the covariance Σ and the two diversity indicators are then defined as

 ,

  1

D

gstd = exp

log σj

(3)

D

j=1

log-volume = 12 log det(Σ + ϵI),

where σj denotes the standard deviation of the j-th embedding dimension and ϵ is a small regularization term ensuring numerical stability. Higher values indicate that the dataset spans a larger and more uniformly distributed region in the latent manifold. As shown in Fig. 3, our curated dataset exhibits both higher embedding scale and broader latent coverage, with an approximately 4−5 increase in logvolume compared with AMASS. This result highlights that richer motion diversity substantially expands the latent coverage of the motion manifold, providing stronger priors for robust zero-shot humanoid tracking.

5.3 Evaluation in Simulation

We first evaluate Humanoid-GPT in the MuJoCo simulation to systematically analyze the effects of data and model scaling on zero-shot tracking performance. This controlled setup allows us to precisely measure stability, fidelity, and generalization across diverse motion categories before transferring to the real world.

Setup. We construct multiple training configurations by varying both the size of the motion corpus and

the capacity of the Transformer backbone. Specifically, we sample teachers from clusters of 10k, 50k, 100k, and 300k motion clips for data scaling, and employ Transformer models with different parameters for model scaling, resulting in Humanoid-GPT-small, Humanoid-GPT, Humanoid-GPT-Large. Each configuration is trained with identical DAgger distillation settings to ensure fair comparison. We test all variants above in the AMASS-test split in [24] following [21], which is an unseen subset during training.

Compared methods. We compare the following baseline policies, which represent the strongest publicly available humanoid trackers at the time of writing and are all based on MLP-style low-level controllers trained on around 6–9M motion frames (see Tab. 1):

- • GMT [4]: A Mixture-of-Experts (MoE) tracker trained on a subset of AMASS [24] motions, where each expert specializes in a particular motion pattern and the gating network selects appropriate experts to maintain physically consistent wholebody tracking.
- • TWIST [42]: A whole-body imitation policy distilled from the TWIST teleoperation system, designed for responsive human-in-the-loop control on Unitree humanoids and trained on a large corpus of teleoperated demonstrations covering everyday and dynamic behaviors.
- • Any2Track [46]: A general tracker trained on AMASS [24] and LAFAN1 [9], which emphasizes robustness to perturbations by incorporating dynamics-adaptive control objectives and strong disturbance randomization during training.

For all three methods, we use the authors’ released implementations and checkpoints and evaluate them under the same simulation and retargeting protocol as our Humanoid-GPT models to ensure a fair comparison.

Metrics. We report three quantitative metrics: ❶: Tracking Success Rate (SR), which measures the proportion of trajectories that can be stably tracked without falling. ❷: Mean per-Joint Position Error (MPJPE) (rad) as the average position error of all joints. ❸: Mean per-Joint Velocity Error (MPJVE) (rad/s) as the average angular velocity error of all joints. ❹: Root Velocity Error (RootVelErr) (m/s) as the average linear velocity error of the humanoid’s base, and ❺: Mean per-Keypoint Position Error (MPKPE) (mm) as the average error of keypoint position among the sequence.

Results. As shown in Table 2, the Transformer-based

[Figure 58]

7.48 ms

CPU

ONNX 6.04 ms

[Figure 59]

5.29 ms

GPU ONNX

4.36 ms

[Figure 60]

0.92 ms

TensorRT & Cache

0.58 ms

[Figure 61]

0.65 ms

C++ COMM

[Figure 62]

Max Mean

0.39 ms

[Figure 63]

3.32 ms 2.79 ms

TWIST CPU ONNX

Figure 5 Comparison of inference latency among different optimization methods. Our final optimization reaches about 5 times faster than TWIST.

Humanoid-GPT exhibits clear scaling laws: enlarging both the motion corpus and the model capacity yields consistent and substantial gains in tracking accuracy and stability. The largest Humanoid-GPT-L model trained on 2B tokens achieves the best performance across nearly all metrics. MLP and TCN baselines also benefit from scaling but reveal two critical limitations. First, data scaling saturates: while larger models eventually reach competitive success rates (e.g., TCN-L achieves 89.05% at 2B tokens), the gains from 200M to 2B are marginal compared to Humanoid-GPT’s continued improvement. Second, larger models overfit on small data: when trained on only 2M tokens, MLP-L (75.25% SR) performs worse than MLP-S (76.89% SR), and TCN-L (79.85% SR) underperforms TCN-S (81.48% SR). This overfitting diminishes with more data, but the MPKPE gap remains significant—even the best baseline (TCN-L at 56.15mm) lags behind HumanoidGPT-S (43.25mm) by 30%. These results demonstrate that while MLP/TCN can achieve reasonable success rates with sufficient data, Transformers offer superior tracking precision and more efficient scaling.

5.4 Real-world Evaluation

To further validate the zero-shot generalization of Humanoid-GPT, we deploy the distilled generalist tracker on the real Unitree-G1 humanoid. Using several pre-recorded dancing sequences, which are entirely unseen during training and consist of highly dynamic full-body motions involving rapid limb coordination and frequent contact transitions. Despite their difficulty, our tracker reproduces these motions in real time without any task-specific fine-tuning, demonstrating strong zero-shot transfer from simu-

- Table 3 Real-world tracking accuracy on four unseen dancing motions. For each motion clip, we record both the target and executed joint configurations and compute MPJPE/MPJVE over the entire sequence to evaluate tracking precision and temporal consistency. Remarkably, the real-world performance closely matches the results obtained in simulation, demonstrating that Humanoid-GPT achieves strong zero-shot transfer and maintains high-fidelity whole-body tracking even under real-world dynamics.

Can Do Can Go! Gokuraku Joudo HuoYuanJia/Fearless PokerFace MPJPE ↓ MPJVE ↓ MPJPE ↓ MPJVE ↓ MPJPE ↓ MPJVE ↓ MPJPE ↓ MPJVE ↓

Backbone

GMT [4] 0.1087 1.2560 0.1098 1.2865 0.0921 0.7054 0.0994 0.8217 TWIST [42] 0.1253 1.1637 0.1162 1.2731 0.1079 0.7821 0.1047 0.8893 Any2Track [46] 0.1039 1.1828 0.1136 1.2366 0.0956 0.6410 0.0928 0.8641

Humanoid-GPT-S 0.1024 1.0572 0.1180 1.2362 0.0825 0.6209 0.0903 0.8476 Humanoid-GPT-B 0.0974 0.9813 0.1075 1.2257 0.0858 0.6158 0.0856 0.7325

lation to the real world. Quantitative motor-sensor analysis in Table 3 further confirms consistent tracking stability and smooth torque regulation, showing that the physically grounded control learned in simulation effectively transfers to hardware.

We also evaluate Humanoid-GPT in an online wholebody teleoperation setting, where a live MoCap stream is continuously retargeted to the G1’s joint space. Without any additional calibration or adaptation, the tracker directly drives the physical robot to follow the actor’s movements in real time. As shown in Fig. 4, the humanoid successfully imitates diverse actions like squatting, stepping, turning, leaning, and expressive arm motions, while maintaining balance and fluid transitions. These results demonstrate unprecedented zero-shot whole-body tracking in the real world, highlighting that motion knowledge distilled from simulation-trained experts and large-scale data seamlessly transfers to embodied execution.

- 5.5 Additional Visualization

In this section, we provide additional real-robot examples, as shown in Fig. 6, including both teleoperation demonstrations and zero-shot dancing. As a powerful zero-shot tracker, Humanoid-GPT can execute a wide range of complex behaviors, such as playing basketball, collaboratively carrying boxes with a human partner, and even rolling over and standing up from the ground. We also showcase more iconic dance routines, where motions are directly captured from videos and retargeted to the G1 space; these sequences are not included in our training set.

- 5.6 Engineering Optimization

To satisfy the strict real-time requirements of wholebody humanoid control, we carefully optimized the deployment pipeline to ensure that scaling up the model size does not compromise inference latency. During deployment, the entire model is exported to the ONNX[26] and compiled a compute graph using

TensorRT[27]. Illustrated in Fig. 5, we also developed a C++-based streaming pipeline to further reduce the communication latency for online teleoperation. These optimizations significantly reduce both computing and memory-access costs. As a result, the final deployed controller achieves an end-to-end inference latency of under 1.5ms on a single NVIDIA RTX 4090 GPU. This demonstrates that, despite scaling the model to a substantially larger parameter count, careful engineering and hardware-aware optimization allow the system to maintain real-time performance demanded by full-body humanoid control.

#### 6 Scaling Laws

The main paper qualitatively demonstrates monotonic scaling trends. Here we provide a more formal analysis of scaling laws and quantify the relationship between dataset diversity and generalization.

- 6.1 Data Scaling Up

We vary the number of training tokens T ∈ {2M,20M,200M,2B} using the Humanoid-GPT-B architecture. For each T, we sample a subset of the 2B-frame corpus without overlap across subsets. As shown in Fig. 7. We also observe that the marginal gains decrease slightly between 200M and 2B tokens, suggesting the onset of a data-limited regime for the current model capacity.

- 6.2 Model Scaling Ability

We evaluate the scalability of our model by comparing a Transformer-B architecture with an MLP of comparable parameter size, both trained on 2B tokens, with results summarized in Fig. 8. We observe that the Transformer continues to improve steadily as training progresses, whereas the MLP saturates early, which demonstrates the scalability of Humanoid-GPT.

[Figure 64]

[Figure 65]

[Figure 66]

Help with moving

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Play Basketball

[Figure 73]

[Figure 74]

[Figure 75]

Turn over

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Get up

[Figure 83]

[Figure 84]

[Figure 85]

Boxing

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Michael Jackson Dance

[Figure 92]

[Figure 93]

[Figure 94]

Gokuraku Joudo Dance

- Figure 6 Additional Real-world experiments for our Humanoid-GPT. All motions illustrated are excluded from training to verify generalization capability. Our method can track diverse, complex and high-dynamic motion in a zero-shot manner, especially various dance motions.

[Figure 95]

[Figure 96]

Zero-Shot MPJPE

[Figure 97]

0.166

0.128

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

0.105

[Figure 103]

0.094

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Data Scale

2M 20M 200M

2B

Figure 9 Data distribution visualization.

- Figure 7 Data Scaling up Curve on Zero-shot Performance.

Training Loss

50K 100K 150K 200k

Training Steps

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

MLP Transformer

0.09 0.06

- Figure 8 Model Scalability Comparison.

Ayoub, Megha Goel, George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Anaïs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lucas Gonzalez, Misha Khalman, Jakub Sygnowski, and et al. Gemini: A family of highly capable multimodal models. CoRR, abs/2312.11805, 2023.

- [2] Joao Pedro Araujo, Yanjie Ze, Pei Xu, Jiajun Wu, and C Karen Liu. Retargeting matters: General motion retargeting for humanoid motion tracking. arXiv preprint arXiv:2510.02252, 2025.
- [3] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel HerbertVoss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. 2020.
- [4] Zixuan Chen, Mazeyu Ji, Xuxin Cheng, Xuanbin Peng, Xue Bin Peng, and Xiaolong Wang. Gmt: General motion tracking for humanoid whole-body control. arXiv preprint arXiv:2506.14770, 2025.
- [5] Xuxin Cheng, Yandong Ji, Junming Chen, Ruihan Yang, Ge Yang, and Xiaolong Wang. Expressive whole-body control for humanoid robots. arXiv preprint arXiv:2402.16796, 2024.
- [6] Antonin Dallard, Mehdi Benallegue, Fumio Kanehiro, and Abderrahmane Kheddar. Synchronized human-humanoid motion imitation. IEEE Robotics and Automation Letters, 8(7):4155–4162, 2023.
- [7] Ke Fan, Shunlin Lu, Minyue Dai, Runyi Yu, Lixing Xiao, Zhiyang Dou, Junting Dong, Lizhuang Ma, and Jingbo Wang. Go to zero: Towards zero-shot motion generation with million-scale data. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13336–13348, 2025.
- [8] Zipeng Fu, Qingqing Zhao, Qi Wu, Gordon Wet-

- 7 Conclusion & Future Work

We introduced Humanoid-GPT, a GPT-style humanoid motion tracker built by scaling both data and model capacity. Through billion-frame motion curation, clustered expert training, and Transformerbased distillation, our system achieves unified agility, stability, and zero-shot generalization. Experiments in simulation and on the real Unitree-G1 show strong transfer without fine-tuning, enabling reliable realtime whole-body imitation. Future work includes incorporating richer modalities such as contacts, vision, or language, and extending the framework to interactive or multi-agent scenarios. We also see potential in coupling Humanoid-GPT with longerhorizon planning or VLA-style instruction toward more general-purpose embodied foundation models.

References

[1] Rohan Anil, Sebastian Borgeaud, Yonghui Wu, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Slav Petrov, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy P. Lillicrap, Angeliki Lazaridou, Orhan Firat, James Molloy, Michael Isard, Paul Ronald Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem

- zstein, and Chelsea Finn. Humanplus: Humanoid shadowing and imitation from humans. 270:2828– 2844, 2024.
- [9] Félix G Harvey, Mike Yurick, Derek Nowrouzezahrai, and Christopher Pal. Robust motion in-betweening. ACM Transactions on Graphics (TOG), 39(4):60–1, 2020.
- [10] Tairan He, Zhengyi Luo, Xialin He, Wenli Xiao, Chong Zhang, Weinan Zhang, Kris Kitani, Changliu Liu, and Guanya Shi. Omnih2o: Universal and dexterous human-to-humanoid whole-body teleoperation and learning. arXiv preprint arXiv:2406.08858, 2024.
- [11] Tairan He, Jiawei Gao, Wenli Xiao, Yuanhang Zhang, Zi Wang, Jiashun Wang, Zhengyi Luo, Guanqi He, Nikhil Sobanbab, Chaoyi Pan, et al. Asap: Aligning simulation and real-world physics for learning agile humanoid whole-body skills. arXiv preprint arXiv:2502.01143, 2025.
- [12] Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander Madry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, Alex Nichol, Alex Paino, Alex Renzin, Alex Tachard Passos, Alexander Kirillov, Alexi Christakis, Alexis Conneau, Ali Kamali, Allan Jabri, Allison Moyer, Allison Tam, Amadou Crookes, Amin Tootoonchian, Ananya Kumar, Andrea Vallone, Andrej Karpathy, Andrew Braunstein, Andrew Cann, Andrew Codispoti, Andrew Galu, Andrew Kondrich, Andrew Tulloch, Andrey Mishchenko, Angela Baek, Angela Jiang, Antoine Pelisse, Antonia Woodford, Anuj Gosalia, Arka Dhar, Ashley Pantuliano, Avi Nayak, Avital Oliver, Barret Zoph, Behrooz Ghorbani, Ben Leimberger, Ben Rossen, Ben Sokolowsky, Ben Wang, Benjamin Zweig, Beth Hoover, Blake Samic, Bob McGrew, Bobby Spero, Bogo Giertler, Bowen Cheng, Brad Lightcap, Brandon Walkin, Brendan Quinn, Brian Guarraci, Brian Hsu, Bright Kellogg, Brydon Eastman, Camillo Lugaresi, Carroll L. Wainwright, Cary Bassin, Cary Hudson, Casey Chu, Chad Nelson, Chak Li, Chan Jun Shern, Channing Conger, Charlotte Barette, Chelsea Voss, Chen Ding, Cheng Lu, Chong Zhang, Chris Beaumont, Chris Hallacy, Chris Koch, Christian Gibson, Christina Kim, Christine Choi, Christine McLeavey, Christopher Hesse, Claudia Fischer, Clemens Winter, Coley Czarnecki, Colin Jarvis, Colin Wei, Constantin Koumouzelis, and Dane Sherburn. Gpt-4o system card. CoRR, abs/2410.21276, 2024.
- [13] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Ku-

- mar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, and Ilge Akkaya. Openai o1 system card. CoRR, abs/2412.16720, 2024.
- [14] Mazeyu Ji, Xuanbin Peng, Fangchen Liu, Jialong Li, Ge Yang, Xuxin Cheng, and Xiaolong Wang. Exbody2: Advanced expressive humanoid wholebody control. arXiv preprint arXiv:2412.13196, 2024.
- [15] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloé Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross B. Girshick. Segment anything. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 3992–4003. IEEE, 2023.
- [16] Kyungmin Lee, Sibeen Kim, Minho Park, Hyunseung Kim, Dongyoon Hwang, Hojoon Lee, and Jaegul Choo. Phuma: Physically-grounded humanoid locomotion dataset. arXiv preprint arXiv:2510.26236, 2025.
- [17] Jiaman Li, Jiajun Wu, and C Karen Liu. Object motion guided human motion synthesis. ACM Transactions on Graphics (TOG), 42(6):1–11, 2023.
- [18] Yixuan Li, Yutang Lin, Jieming Cui, Tengyu Liu, Wei Liang, Yixin Zhu, and Siyuan Huang. Clone: Closed-loop whole-body humanoid teleoperation for long-horizon tasks. arXiv preprint arXiv:2506.08931, 2025.
- [19] Qiayuan Liao, Takara E Truong, Xiaoyu Huang, Guy Tevet, Koushil Sreenath, and C Karen Liu. Beyondmimic: From motion tracking to versatile humanoid control via guided diffusion. arXiv preprint arXiv:2508.08241, 2025.

- [20] Jing Lin, Ailing Zeng, Shunlin Lu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, and Lei Zhang. Motion-x: A large-scale 3d expressive whole-body human motion dataset. Advances in Neural Information Processing Systems, 36:25268–25280, 2023.
- [21] Zhengyi Luo, Jinkun Cao, Kris Kitani, Weipeng Xu, et al. Perpetual humanoid control for real-time simulated avatars. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10895–10904, 2023.
- [22] Zhengyi Luo, Jinkun Cao, Josh Merel, Alexander Winkler, Jing Huang, Kris Kitani, and Weipeng Xu. Universal humanoid motion representations for physics-based control. arXiv preprint arXiv:2310.04582, 2023.
- [23] Zhengyi Luo, Ye Yuan, Tingwu Wang, Chenran Li, Sirui Chen, Fernando Castañeda, Zi-Ang Cao, Jiefeng Li, David Minor, Qingwei Ben, et al. Sonic: Supersizing motion tracking for natural humanoid whole-body control. arXiv preprint arXiv:2511.07820, 2025.
- [24] Naureen Mahmood, Nima Ghorbani, Nikolaus F Troje, Gerard Pons-Moll, and Michael J Black. Amass: Archive of motion capture as surface shapes. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5442–5451, 2019.
- [25] Jiageng Mao, Siheng Zhao, Siqi Song, Tianheng Shi, Junjie Ye, Mingtong Zhang, Haoran Geng, Jitendra Malik, Vitor Guizilini, and Yue Wang. Learning from massive human videos for universal humanoid pose control. arXiv preprint arXiv:2412.14172, 2024.
- [26] Microsoft Corporation. ONNX Runtime, 2024. https://onnxruntime.ai. High-performance inference engine for ONNX models.
- [27] NVIDIA Corporation. NVIDIA TensorRT, 2024. https://developer.nvidia.com/tensorrt. Version 10.0.
- [28] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. 2022.
- [29] Yuzhe Qin, Wei Yang, Binghao Huang, Karl Van Wyk, Hao Su, Xiaolong Wang, Yu-Wei Chao, and Dieter Fox. Anyteleop: A general visionbased dexterous robot arm-hand teleoperation system. arXiv preprint arXiv:2307.04577, 2023.
- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning

- transferable visual models from natural language supervision. volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 2021.
- [31] Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pages 627–635. JMLR Workshop and Conference Proceedings, 2011.
- [32] Konstantinos I. Roumeliotis and Nikolaos D. Tselikas. Chatgpt and open-ai models: A preliminary review. Future Internet, 15(6):192, 2023.
- [33] Sebastian Starke, Ian Mason, and Taku Komura. Deepphase: Periodic autoencoders for learning motion phase manifolds. ACM Transactions on Graphics (ToG), 41(4):1–13, 2022.
- [34] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 5026–5033. IEEE, 2012.
- [35] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971, 2023.
- [36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, pages 5998–6008, 2017.
- [37] Yuxuan Wang, Ming Yang, Weishuai Zeng, Yu Zhang, Xinrun Xu, Haobin Jiang, Ziluo Ding, and Zongqing Lu. From experts to a generalist: Toward general whole-body control for humanoid robots. CoRR, abs/2506.12779, 2025. doi: 10.48550/ARXIV.2506.12779. https://doi.org/10. 48550/arXiv.2506.12779.
- [38] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022.
- [39] Weiji Xie, Jinrui Han, Jiakun Zheng, Huanyu Li, Xinzhe Liu, Jiyuan Shi, Weinan Zhang, Chenjia Bai, and Xuelong Li. Kungfubot: Physics-based humanoid whole-body control for learning highlydynamic skills. arXiv preprint arXiv:2506.12851, 2025.
- [40] Lujie Yang, Xiaoyu Huang, Zhen Wu, Angjoo Kanazawa, Pieter Abbeel, Carmelo Sferrazza, C Karen Liu, Rocky Duan, and Guanya Shi. Omniretarget: Interaction-preserving data generation for

- humanoid whole-body loco-manipulation and scene interaction. arXiv preprint arXiv:2509.26633, 2025.
- [41] Kangning Yin, Weishuai Zeng, Ke Fan, Minyue Dai, Zirui Wang, Qiang Zhang, Zheng Tian, Jingbo Wang, Jiangmiao Pang, and Weinan Zhang. Unitracker: Learning universal whole-body motion tracker for humanoid robots. arXiv preprint arXiv:2507.07356, 2025.
- [42] Yanjie Ze, Zixuan Chen, JoÃG, o Pedro AraÃšjo, Ziang Cao, Xue Bin Peng, Jiajun Wu, and C Karen Liu. Twist: Teleoperated whole-body imitation system. arXiv preprint arXiv:2505.02833, 2025.
- [43] Yuhong Zhang, Jing Lin, Ailing Zeng, Guanlin Wu, Shunlin Lu, Yurong Fu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, and Lei Zhang. Motion-x++: A large-scale multimodal 3d whole-body human motion dataset. arXiv preprint arXiv:2501.05098, 2025.
- [44] Zhikai Zhang, Yitang Li, Haofeng Huang, Mingxian Lin, and Li Yi. Freemotion: Mocap-free human motion synthesis with multimodal large language models. In European Conference on Computer Vision, pages 403–421. Springer, 2024.
- [45] Zhikai Zhang, Chao Chen, Han Xue, Jilong Wang, Sikai Liang, Yun Liu, Zongzhang Zhang, He Wang, and Li Yi. Unleashing humanoid reaching potential via real-world-ready skill space. arXiv preprint arXiv:2505.10918, 2025.
- [46] Zhikai Zhang, Jun Guo, Chao Chen, Jilong Wang, Chenghuai Lin, Yunrui Lian, Han Xue, Zhenrong Wang, Maoqi Liu, Jiangran Lyu, et al. Track any motions under any disturbances. arXiv preprint arXiv:2509.13833, 2025.

#### A Summary of Contributions

Science of Scale: We are the first motion tracker with zero-shot ability trained on 2BFrame data. Our training set is over 200× larger than prior trackers. Scaling by two orders of magnitude requires redesigning the reward and re-tuning key hyperparameters, and it enables the first systematic evidence that video-esti data materially benefits tracking.

Modern Structure: We use a scalable causal Transformer for motion tracking. Online tracking cannot access future observations; causal modeling fits this constraint and scales better than MLP and non-causal variants.

Balanced Diversity Matters: We introduce HME Representation Learning to apply diversity-aware, distribution-balanced sampling in motion tracking. We find that diversity and balance are both critical for a general tracker.

#### B Additional Ablation Studies

- B.1 Number of Experts and Cluster Granularity

We next study the effect of the number of motion experts and the granularity of clusters produced by the Harmonic Motion Embedding (HME) representation. We vary the number of clusters C ∈ {128,256,384,512,1024} while keeping the total training corpus fixed, leading to different numbers of experts. For each configuration, we train a distilled Humanoid-GPT-B model on the corresponding expert set and evaluate on the AMASS test split.

As indicated in Fig. 10, extremely coarse clustering (e.g., 128 experts) leads to experts that cover overly heterogeneous motion patterns, which harms teacher tracking fidelity. Overly fine granularity (e.g., 1024 experts) increases training cost with conflicting guidelines for the students. The configuration with roughly C ≈ 384 experts offers the best balance between diversity, per-cluster coherence, and compute.

- B.2 History Length of Transformer

Compared with MLPs, Transformers not only offer greater scalability, but more importantly provide substantially enhanced temporal modeling. MLP-based policies typically condition on only a single historical frame, whereas Transformers are inherently designed for sequence modeling. In Fig. 10, we present the effect of varying sequence length on model performance. All experiments use a Base-sized model with default

hyperparameters under a controlled single-factor setting. As shown in the figure, performance continues to improve even with a history of 64 frames. However, due to the quadratic increase in computation with sequence length, we adopt 32 historical frames as the default setting.

- B.3 Environment Number for DAgger Rollout

As we scaled up the training data, we found that the number of environments in DAgger also needed to increase accordingly, as shown in Fig. 10. We ultimately adopted 32K environments. We hypothesize that this is due to the large number of reference motions, where using too few environments may lead to overfitting and forgetting.

C Implementation and Reproducibility Details

This section expands the method details that could not fit in the main paper. We include RL hyperparameters, DAgger schedules, and full compute accounting.

- C.1 RL Expert Training Details

Environment and episode setup. For each skill cluster, we train a PPO expert in MuJoCo using randomized episode lengths ranging from Tmin to Tmax frames (typically 600–1200). The control loop runs at 50 Hz, and the reference motion is downsampled to match this frequency. Episodes terminate upon detecting a fall, encountering excessive joint-limit violations, or reaching the time-out horizon. All hyperparameters and reward weights involved in PPO training are listed in Table 7 and Table 6.

To enhance robustness and improve generalization across diverse motion clusters, we apply a set of domain randomization strategies during training. These perturbations are injected into the MuJoCo environment at the beginning of each episode and occasionally throughout rollout generation, ensuring that the learned policy remains stable under variations in dynamics, sensing, and execution conditions. As shown in Table 4, we randomize:

- • Terrain properties, including floor friction, maximum terrain height, and procedural noise parameters for terrain generation (noise scale, octaves, persistence, and lacunarity).
- • External forces, where both the interval between force injections and the magnitude of the applied

Success Rate

Success Rate

Success Rate

90.4 90.6

88.7 89.5 90.0 90.2 90.4

89.9 90.4 90.1

90.1

89.7

89.5

[Figure 120]

89.0

[Figure 121]

[Figure 122]

88.4

[Figure 123]

[Figure 124]

Cluster Num

History 4 8 16 32 64 Length

Envs 2048 4096 8192 16384 32768 Num

128 256 384 512 1024

- Figure 10 Ablation studies for Humanoid-GPT.

Table 4 Domain Randomizations.

##### Item Random range

Terrains

Floor friction U(0.3, 2.0) Max terrain height 0.3

Noise scale U(10.0, 16.0)

Noise octaves U(5.0, 8.0) Noise persistence U(0.3, 0.5) Noise lacunarity U(2.0, 4.0)

External Forces

##### Interval range U(5.0, 10.0) Velocity magnitude range U(0.1, 1.0)

Physical Property Changes

DoF friction scaling U(0.5, 2.0) Armature scaling U(1.0, 1.05) Torso CoM position change U(−0.15, 0.15) Torso mass change U(−3.0, 6.0) Default DoF position jittering U(−0.05, 0.05)

velocity perturbations are sampled from uniform distributions.

• Physicalpropertyvariations, including DoF friction scaling, armature scaling, torso center-of-mass shifts, torso mass perturbations, and per-DoF position jittering at the beginning of each rollout.

These domain randomization settings are applied uniformly across all PPO expert training environments. For the DAgger stage, we use the same environment configuration and identical randomization scheme, ensuring that the aggregated demonstrations capture the full distribution of dynamic variations encountered by the expert policy.

- C.2 DAgger Distillation Schedule

We follow a standard DAgger loop for Behaviour Cloning (BC):

1. Initialize both the expert teacher and the student policy within the simulation environment.

- Table 5 Hyperparameter settings for training motion experts. Hyperparameter Value

Env Numbers 32768

Batch size 1024 Discount factor γ 0.97 GAE parameter λ 0.95

Clipping parameter ϵ 0.2

Policy network size [512, 256, 128] Critic network size [512, 256, 128] Learning rate 3 × 10−4 Entropy coefficient 0.01

Optimizer Adam Training iteration per expert 3B

- Table 6 Reward weights of different terms. Term Weight

lowerbody keypoints wk 1.5 upperbody keypoints wk 0.75 keypoint position αpos 1.0

keypoint orientation αrot 2.0 keypoint linear velocity αvel 0.03

- Table 7 Hyperparameter settings for DAgger BC. Hyperparameter Value

Env Numbers 32768

Batch size 32768 Gradient Clipping 1.0

Learning rate 1 × 10−4 Num Layers 12

Channel dims 256/384/768 Optimizer AdamW Training iteration 200k

- 2. At iteration i, roll out the student policy and query the expert for the corresponding target action using the same state.
- 3. Train the student to match the expert’s action, and then update the environment using the student’s executed action.

Table 8 Approximate compute breakdown.

Stage Hardware Total GPU hours Fraction of total (%)

PPO experts (∼ 384 experts) RTX 4090 12,000 75% Distillation (Humanoid-GPT-S/B/L) H100 3,000 25%

Total — 15,000 100%

- Figure 11 T-SNE distribution Visualization for our dataset.

[Figure 125]

Blue: AMASS Brown: LAFAN Light blue: All

In practice, we fix the maximal history length H in Eq. (2) to 32 and maintain history buffers for both the teacher’s actions and the student’s observations. To avoid mode collapse when some experts cover only partial behavior distributions, the batch size used for Behaviour Cloning is kept no smaller than the number of experts.

- C.3 Compute Cost Breakdown

The main paper reports a total compute budget of roughly 15,000 GPU hours. Here we detail the breakdown between expert training and Transformer distillation shown in Table 8.

We emphasize that, once trained, only the distilled Humanoid-GPT policy is required at deployment time. The expert library is used solely as a trainingtime teacher and can be discarded afterwards.

- C.4 T-SNE feature distribution

We visualize the feature distribution via t-SNE in Fig. 11. Our data covers a substantially broader region than AMASS+LAFAN1.

C.5 Deployment Details and Latency Measurements

For completeness, we provide the exact configuration used to obtain the latency results in Fig. 5 of the main paper:

- • Inference hardware: Single NVIDIA RTX 4090 GPU, CPU: Intel Core i9-14900KF.
- • ONNX export: FP32 weights with CUDA.
- • TensorRT optimization: Engine built with optimized kernels for causal attention and fused MLPs.
- • Control loop: End-to-end closed-loop frequency of 50 Hz, including sensor read, inference, PD computation, and actuation commands.

The complete deployment stack will be released as configuration files and scripts, enabling reproducible real-time control on G1-like humanoids.

