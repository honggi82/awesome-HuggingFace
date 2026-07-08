# arXiv:2506.09937v2[cs.RO]30Oct2025

## SAFE: Multitask Failure Detection for Vision-Language-Action Models

Qiao Gu1,2,3 Yuanliang Ju1,2,3 Shengxiang Sun1,2 Igor Gilitschenski1,2,3 Haruki Nishimura4 Masha Itkina4 Florian Shkurti1,2,3 1University of Toronto (UofT), 2UofT Robotics Institute, 3Vector Institute, 4Toyota Research Institute (TRI) q.gu@mail.utoronto.ca

#### Abstract

While vision-language-action models (VLAs) have shown promising robotic behaviors across a diverse set of manipulation tasks, they achieve limited success rates when deployed on novel tasks out of the box. To allow these policies to safely interact with their environments, we need a failure detector that gives a timely alert such that the robot can stop, backtrack, or ask for help. However, existing failure detectors are trained and tested only on one or a few specific tasks, while generalist VLAs require the detector to generalize and detect failures also in unseen tasks and novel environments. In this paper, we introduce the multitask failure detection problem and propose SAFE, a failure detector for generalist robot policies such as VLAs. We analyze the VLA feature space and find that VLAs have sufficient highlevel knowledge about task success and failure, which is generic across different tasks. Based on this insight, we design SAFE to learn from VLA internal features and predict a single scalar indicating the likelihood of task failure. SAFE is trained on both successful and failed rollouts, and is evaluated on unseen tasks. SAFE is compatible with different policy architectures. We test it on OpenVLA, π0, and π0-FAST in both simulated and real-world environments extensively. We compare SAFE with diverse baselines and show that SAFE achieves state-of-the-art failure detection performance and a favorable trade-off between accuracy and detection time using conformal prediction. More qualitative results and code can be found at the project webpage: https://vla-safe.github.io/.

#### 1 Introduction

Recently, scaling up robot manipulation datasets has enabled the development of large visionlanguage-action (VLA) models, which are generalist manipulation policies that can follow language instructions and accomplish a wide range of tasks [1–6]. However, when VLAs are directly deployed on unseen tasks without collecting additional demonstrations and finetuning the model, they still suffer from limited success rates and a wide range of failure modes. This has been demonstrated by evaluations in recent work [2, 4, 7]: while VLAs achieve success rates of 80–90% on seen tasks, their performance on unseen tasks drops to 30–60% out of the box. Therefore, to safely and reliably deploy VLA policies in the real world, it is important to promptly detect their potential failures.

Most existing failure detection methods train a separate failure detector for each task, and evaluate the detector only on that task [8–17]. While these methods work well for specialist policies, they do not suit generalists like VLAs. VLAs are designed to accomplish diverse tasks and may frequently encounter novel task instructions and unseen environments during deployment. In such cases, it is impractical to exhaustively collect rollouts and train a failure detector for every new task. Some recent works introduce task-generic failure detectors, but they either require sampling multiple actions [18]

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

or need to query a large VLM [19, 20], which poses significant inference overhead for VLAs in the real world. This motivates the need for an efficient and multitask failure detector that can generalize to unseen tasks zero-shot and detect failures in a timely manner during the on-policy rollout of the VLA.

In this paper, we focus on the multitask failure detection problem. This setting evaluates the failure detection performance of a VLA policy without collecting rollouts or finetuning the failure detector on unseen tasks. To our knowledge, such multitask failure detection capabilities for VLAs have not been shown in the literature. To tackle this problem, we study the internal features of VLAs and find that they capture high-level knowledge about task success and failure. As shown in Fig. 1, failed rollouts occupy a distinct region (“failure zone”) in the VLA feature space, and this separation remains consistent across different tasks.

Based on this insight, we introduce SAFE, a ScAlable Failure Estimation method that scales across diverse tasks for generalist policies like VLAs. SAFE takes in a VLA’s internal features and regresses

- them to a single scalar indicating the likelihood of failure. By training on successful and failed rollouts of multiple tasks, SAFE learns to identify task-generic representations for failure detection. To determine the threshold for failure detection, we adopt the functional conformal prediction (CP) [8, 21] framework and calibrate the prediction band on the seen tasks. We conduct failure detection

experiments on OpenVLA [2], π0 [4] and π0-FAST [5], in both simulation and the real world. For evaluation, we adapt diverse baseline failure detection methods from both the LLM literature [22, 23] and the robot learning literature [8, 18] onto VLAs. SAFE and baselines are evaluated on both training (Seen) tasks and a set of held-out (Unseen) tasks. Experiments show that SAFE outperforms other existing baselines and achieves the best trade-off between accuracy and timeliness for failure detection. The contributions of our paper can be summarized as follows:

- • We analyze the VLA feature space and show that, across different task instructions and environments, the internal features of the VLA distinctly separate successful and failed rollouts.
- • We propose SAFE, a multitask failure detector designed for generalist robot policies. By operating on latent features, training on multiple tasks, and using conformal prediction methods, SAFE shows generalization capabilities in detecting failures on unseen tasks.
- • We evaluate SAFE and diverse baselines on several recent large VLA models in both simulation and the real world. Experiments show that SAFE outperforms baselines and achieves state-of-the-art (SOTA) performance.

#### 2 Related Work

##### 2.1 Vision-Language-Action Models

Recent advances in large-scale machine learning and the availability of extensive robot demonstration datasets have paved the way for VLA models [1–4, 6, 7, 24, 25]. These generalist robotic policies are initialized from pretrained large-scale VLMs [26–28], and thus inherit the ability to understand diverse semantic concepts from both images and language. They are augmented with an action head that produces continuous control signals, through per-step binning [1, 2, 7, 24], diffusion networks [3, 4, 29, 30], or frequency-space tokenization [5]. These VLAs are then trained on vast robotic datasets covering a wide array of tasks [31–33]. As a result, VLAs can successfully perform familiar tasks in new environments and even tackle previously unseen tasks when provided with novel language instructions. Nevertheless, significant variability in real-world deployments and the challenging domain gaps between training and testing environments continue to hinder VLA performance. Most state-of-the-art VLA models achieve success rates between 30% and 60% when evaluated out-of-the-box on real robots with unseen task instructions [2, 4, 31]. These limitations highlight the need for robust multitask failure detection methods tailored to generalist VLA models.

##### 2.2 Failure Detection in Robot Manipulation

Monitoring failures is critical when deploying robotic policies in real-world environments, as even minor errors can result in hazardous conditions [34–36]. The literature on failure detection in robot learning can be broadly divided into unsupervised out-of-distribution (OOD) detection [8–11, 17] and

supervised failure detection [9, 12–16, 37]. OOD detection-based methods treat successful executions as the in-domain baseline and consider any deviation from this norm as a failure. However, the assumption that any unseen scenario constitutes a failure is overly restrictive for generalist VLAs, which may frequently encounter unseen tasks at test time. These unseen tasks are likely different from the in-domain training data but should not be simply treated as failures. Our proposed method, SAFE, falls within the supervised failure detection category, leveraging both successful and failed rollouts to train a failure classifier. Unlike existing methods that train and calibrate separate classifiers per task, SAFE uses a single unified failure detector and works effectively on generalist policies like VLAs. Some recent works have explored multitask failure detection by designing action consistency scores [18] or instruction-finetuning a VLM [19, 20], but they require either sampling multiple actions or querying a large VLM, which poses significant overhead for controlling robots in real time.

Recently, FAIL-Detect [8] conducted a systematic evaluation of various failure detection methods, including OOD detection-based approaches [9, 38, 39], smoothness-based techniques [40], and consistency-based strategies [18]. Their experiments indicate that the best performance was achieved by LogpZO, which learns a proxy for the likelihood of the data in the observation embedding space using flow matching [8]. However, their evaluation is limited to only single-task policies, and our evaluation in the multitask setting shows that their best-performing LogpZO method suffers from overfitting to the training tasks.

##### 2.3 Uncertainty Quantification for LLM

Although LLMs and VLMs have demonstrated remarkable understanding and generative capabilities across various tasks, they are prone to producing hallucinated responses [41–43]. Numerous methods have been developed for uncertainty quantification (UQ) in LLMs/VLMs. Token-level uncertainty quantification methods estimate uncertainty by analyzing the probability distribution over each generated token to assess the likelihood of an entire response [44–46]. Semantic-similarity methods generate multiple responses to the same query and evaluate their semantic alignment [23, 47, 48]; a higher variance among responses typically signals low confidence. Since vision-language-action models (VLAs) share the generative nature and transformer architecture of LLMs/VLMs, we adapt these UQ methods to VLAs as promising baselines and evaluate their performance on failure detection. Note that these UQ baselines are used as a proxy for failure detection, assuming that when a policy becomes uncertain about its actions, it will have a higher probability of failing the task. Recent research has also explored the internal latent space of LLMs for hallucination detection [49–55]. These methods train a classifier on internal latent features to distinguish between truthful and hallucinated outputs, paralleling supervised failure detection techniques in robotics. This approach has proven to be simple, efficient, and effective for UQ in LLMs. In our study, we investigate its application to large VLA policies and observe promising performance in robotic tasks.

#### 3 Problem Formulation

This work aims to detect when a robot policy fails during task execution. Specifically, we develop a multi-task failure detector that performs well when generalist VLAs encounter novel tasks at inference time. At timestep t, a VLA is given an input observation ot, consisting of RGB images, natural language instruction, and current robot state, and outputs a control signal At = [at,at+1,...,at+H−1], which is a chunk of actions for the next H timesteps. The first H′ (H′ ≤ H) actions in At are executed, and then the VLA replans a new action sequence At+H′ at time t + H′. We denote the internal embedding vector within the VLA model at time t as et. Some VLAs [1, 2, 4, 5] also decode a series of m tokens Wt = [wt1,...,wtm] before converting them into the actual action vector. To train and evaluate failure detection models, we run the VLA on different tasks in simulation or the real world, collect the rollout trajectory τi = {(ot,et,Wt,At)}t=0,H′,...,nH′ with time duration T = nH′, and annotate each rollout with a failure label yi (yi = 1 if the robot fails to accomplish the task and yi = 0 if the robot succeeds). Note that for training, we only use the trajectory-level annotation yi, and do not require knowing the exact timestep when the policy starts to fail. A failure detector receives the rollout information up to time t and predicts a failure score st, indicating the likelihood of task execution failure at time t. If st exceeds a threshold δt, a failure flag is raised, and

- then either the task execution is aborted or a human monitor will step in and take over the control. In this work, we use conformal prediction [58] to calibrate the threshold δt.

[Figure 1]

[Figure 2]

###### Different tasks Same “failure zone”

- 3 2

- 4

- 3 2

- 4

6

6

5

5

4

4

3 2

3 2

1

1

5

5

1

1

6

6

[Figure 3]

Failed Rollouts Successful Rollouts

|𝑡 = 𝑇|
|---|

𝑡 = 0

(a) t-SNE of policy latent features, colored by task success

(b) t-SNE of policy latent features, colored by task ID.

Failed rollout: “Turn on the stove and put the moka pot on it”

Successful rollout: “Turn on the stove and put the moka pot on it”

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

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

[Figure 26]

[Figure 27]

1 2 3 4 5 6

###### 1 2 3 4 5 6

Failure Occurs

(c) Successful and failed rollouts on a task

- Figure 1: The internal features of a VLA capture high-level information about task success and failure. When the VLA is failing, the features, even those from different tasks, fall into the same “failure zone”. This motivates SAFE, an efficient multitask failure detector that is based on VLA

internal features and can generalize to unseen tasks. Plot (a) visualizes the latent features of π0-FAST on LIBERO-10 [56] using t-SNE [57]. For successful rollouts, features are colored in blue. For failed rollouts, features follow a blue-to-red gradient based on timestep progression, with red corresponding

- to later timesteps that often coincide with failure. Plot (b) visualizes the same set of t-SNE features, colored by task ID. In (c), we show two example rollouts over time and mark their corresponding projected features in (a) and (b).

In experiments, we split all tasks into seen and unseen subsets, where rollouts from seen tasks are used for training Dtrain and validation Deval-seen, and all rollouts from unseen tasks Deval-unseen are reserved for testing the cross-task generalization ability of failure detectors. Failure detectors are trained on Dtrain, and evaluated on Deval-seen for hyperparameter tuning and in-domain performance, and tested on Deval-unseen for out-of-distribution generalization.

#### 4 Method

##### 4.1 Visual Analysis on VLA Latent Space

VLAs process multi-modal inputs and extract rich semantic information in their internal feature space. We hypothesize that these features also capture the high-level and abstract knowledge about task execution success/failure, by separating features from successful/failed rollouts into different regions. We study this hypothesis by visualizing the VLA features in Fig. 1, where we plot the internal features from π0-FAST[5] when running the LIBERO-10 benchmark [56]. Fig. 1(a) demonstrates that when the VLA is failing, its internal features are grouped in the same region in the feature space (“failure zone”). Comparing Fig. 1(a) and Fig. 1(b), we can further see that although the features are extracted from different tasks with various instructions, objects and environments, when the VLA fails, its features fall in the same “failure zone”. Fig. 1(c) further illustrates how VLA’s features evolve in the feature space when VLA progresses temporally. From Fig. 1(c), we can see that failure rollout initially stays out of the “failure zone” when it progresses normally, and when the robot mistakenly drops the pot in the middle of execution and starts to fail, it steps into the “failure zone”. On the contrary, for the successful rollout, its features always stay out of the “failure zone”.

This visual analysis shows that the VLA’s internal features for succeeding and failing task executions are well separated in the feature space, and this separation is general across different tasks. Furthermore, during task execution, the features reflect how well the VLA performs on the current tasks in a

1. Extract latent features

2. Learning the failure score

3. Calibrate failure detection

from VLA models

predictor, SAFE

threshold and deploy

| | |
|---|---|
| |Failure<br><br>score 𝑠𝑡|
| | |

Prediction from calibration rollouts

Failure detection threshold

SAFE-MLP

Action:𝑨𝑡

|𝑠1|
|---|

|𝑠2|
|---|

|𝑠3|
|---|

|𝑠𝑇|
|---|

[Figure 28]

[Figure 29]

Functional conformal prediction

| | |
|---|---|
|𝑠1| |

| | |
|---|---|
|𝑠2| |

| | | | |
|---|---|---|---|
|+| |𝑠𝑇| |
| | | | |

|+|
|---|

|+|
|---|

+

ǁ

ǁ

𝑠ǁ3

ǁ

Decoder

|𝒆𝑡|
|---|

MLP

MLP

MLP

MLP

VLA Model

| | |
|---|---|
|𝒆1| |

| | |
|---|---|
|𝒆2| |

| | |
|---|---|
|𝒆3| |

| | |
|---|---|
|𝒆𝑇| |

|Time 𝑡|
|---|

Detect failures on test rollouts

Encoder

SAFE-LSTM

[Figure 30]

|𝑠1|
|---|

|𝑠2|
|---|

|𝑠3|
|---|

|𝑠𝑇|
|---|

|𝑙𝑡|
|---|

LSTM

LSTM

LSTM

LSTM

𝒐𝑡

Observation Instruction

| | |
|---|---|
|𝒆1| |

| | |
|---|---|
|𝒆2| |

| | |
|---|---|
|𝒆3| |

| | |
|---|---|
|𝒆𝑇| |

- Figure 2: The proposed failure detector, SAFE, has three major components: (1) SAFE extracts the latent feature from the last layer of a VLA model; (2) SAFE sequentially processes the latent feature and predicts a failure score, using an MLP or an LSTM backbone; and (3) SAFE determines a time-varying threshold using functional conformal prediction (CP) on a hold-out calibration set. If the predicted score exceeds the threshold during testing, SAFE confidently detects a failure.

timely manner. Inspired by this observation, we design SAFE, which uses the internal features of VLAs for failure detection. An overview of the proposed method is shown in Fig. 2.

##### 4.2 Failure Detection by Feature Probing

We design SAFE to learn the abstract information from the VLA’s internal features and determine whether the task execution is failing. We extract the VLA’s hidden state vectors from the final layer, before being decoded to token logits [2, 5] or a velocity field [4]. We ablate different ways to aggregate the internal features into a single embedding vector e, and select the best one based on Deval-seen performance. Please refer to Appendix for details on VLA feature extraction.

The failure detector f(e0:t) takes as input the VLA’s features e0:t = {e1,...,et} up to the current timestep t, and is trained to predict st. We explore the two backbone designs for SAFE: a multi-layer perceptron (fMLP) and an LSTM [59] (fLSTM). Both models are designed to be simple (only one or two layers), to avoid overfitting and improve generalization ability on unseen tasks. For fMLP, we use an MLP g(·) to project et into a single scalar for each timestep t independently and accumulate the outputs as the failure score, i.e. fMLP(e0:t) = τ=1,...,t σ(g(eτ)), where σ(·) is a sigmoid function and therefore 0 < st < t. To train the MLP model, we apply an L1 loss on all timesteps to push up the scores for failed rollouts and push down those for successful ones. Specifically, LMLP = i [yi t(t − st) + (1 − yi) t st], where index i iterates over all data points in Dtrain.

For fLSTM, we use an LSTM model to sequentially process the input stream of VLA’s features e0:t and project the hidden state vector of LSTM into a scalar score. Specifically, fLSTM(e0:t) = σ(LSTM(e0:t)), where a sigmoid function σ(·) is applied to normalize the output score s.t. 0 ≤ st ≤ 1. To train the LSTM model, we apply a binary cross entropy loss on all timesteps, i.e. LLSTM = i t [yi log(st) + (1 − yi)log(1 − st)].

##### 4.3 Threshold Selection by Conformal Prediction

When the predicted failure score st exceeds the time-varying threshold δt, we raise a failure flag. To determine δt in a principled way, we adopt the functional conformal prediction (CP) framework [21]. Functional CP constructs a time-varying prediction band by leveraging the distribution of st observed in successful rollouts within a calibration set. Under the exchangeability assumption [60] and given a user-specified significance level α, the CP band guarantees that, for a new successful rollout, its st will lie within this band at all times t with probability 1 − α. Conversely, if the score of a test rollout exceeds the band at time t, we can declare a failure with nominal confidence 1 − α.

Formally, given a time series of any scalar score st and a user-specified significance level α ∈ (0,1), functional CP gives a distribution-free prediction band Cα. Following Xu et al. [8], we adopt the one-sided time-varying CP band formulation, where Cα is a set of intervals {[lowert,uppert] : t =

Table 1: Failure detection results on simulation benchmarks, measured by area under ROC (ROCAUC). “-” indicates that the failure detection method does not apply. Entries with gray background indicate the failure detection methods that sample 10 actions per inference timestep, while others use only 1 action. The first and second best-performing methods are colored in red and orange, respectively. Results are averaged over 3 random seeds with different splits of seen and unseen tasks. SAFE achieves the highest averaged ROC-AUC over all simulation benchmarks.

VLA Model OpenVLA π0-FAST π0 π0∗ Average

Benchmark LIBERO LIBERO LIBERO SimplerEnv Eval Task Split Seen Unseen Seen Unseen Seen Unseen Seen Unseen Seen Unseen

|Token Unc.<br><br>|Max prob. Avg prob. Max entropy Avg entropy|50.25 53.83<br><br>44.05 51.58<br><br>52.94 53.09<br><br>45.27 50.03<br><br><br>|61.32 69.44 52.46 58.04 46.69 62.96 50.93 58.63|- -<br><br>- -<br><br>- -<br><br>- -<br><br><br>|- -<br>- -<br>- -<br>- -<br>|55.79 61.64<br><br>48.26 54.81<br><br>49.81 58.03<br><br><br>48.10 54.33|
|---|---|---|---|---|---|---|
|Embed. Distr.<br><br>|Mahalanobis dist. Euclidean dist. k-NN Cosine dist. k-NN PCA-KMeans [9] RND [39] LogpZO [8]|62.03 58.85<br><br>66.00 55.23<br><br>67.09 69.45<br><br><br>57.18 55.10 52.57 46.88 61.57 52.91<br><br>|93.56 83.79 92.04 84.12 92.09 84.64 68.46 57.12 88.67 81.57 91.52 83.07<br><br>|77.12 74.31 75.64 70.73<br><br>75.76 70.31 64.92 60.35 71.92 69.44<br>76.80 73.23<br>|88.42 52.84<br><br>89.73 68.41<br><br>90.19 71.32<br><br><br>66.88 61.19 85.07 65.89 88.79 74.66<br><br>|80.28 67.45<br><br>80.85 69.62<br>81.28 73.93 64.36 58.44 74.56 65.95 79.67 70.97<br>|
|Sample Consist.<br><br>|Action total var. Trans. total var. Rot. total var. Gripper total var. Cluster entropy|62.76 65.43 55.33 58.99 47.85 55.30 61.84 64.48 50.16 51.44<br><br>|76.95 74.50 78.21 80.03 80.87 77.29 76.82 74.42 80.22 80.53<br><br>|77.20 75.18 49.38 54.71 52.94 61.06 77.19 75.19 76.19 72.12<br><br>|68.41 67.94 63.27 55.90 58.07 62.10<br><br>69.16 69.29 68.25 73.66<br><br><br>|71.33 70.76 61.55 62.41 59.93 63.94 71.25 70.84 68.71 69.44<br><br>|
|Action Consist.|STAC [18]<br><br>STAC-Single<br><br>|- -<br><br>- -<br><br><br>|83.07 85.31 85.46 81.16<br><br>|46.55 47.91 68.46 69.39<br><br>|60.74 62.21 68.71 70.40<br><br>|63.45 65.14 74.21 73.65<br><br>|
|SAFE (Ours)<br><br>|SAFE-LSTM SAFE-MLP|70.24 72.47 72.68 73.47<br><br>|92.98 84.48 90.06 80.44<br><br>|76.98 71.09 73.50 73.27|88.85 80.11<br><br>89.50 84.82<br>|82.26 77.04 81.43 78.00<br><br>|

1,...,T}, where lowert = −∞ and uppert = µt +ht, with a time-varying mean µt and a bandwidth ht. This band is calibrated on successful rollouts in Deval-seen. Under mild assumptions [61, 62], for any new successful rollout, st < µt +ht holds for all t = 1,...,T with probability 1−α. Intuitively, this gives a guarantee that the false positive rate of the failure detector (a failure flag is raised at any time during a successful rollout) is at most α. We use uppert as the failure flag threshold δt, and more details about functional CP can be found in Appendix.

#### 5 Experiments

##### 5.1 Evaluation Benchmarks

LIBERO [56]: The LIBERO benchmark has been widely adopted for evaluating VLA models in simulation [2, 4–6]. Among the LIBERO task suites, the LIBERO-10 suite consists of 10 long-horizon tasks with diverse objects, layouts, and instructions, and is considered the most challenging one. Therefore, we use LIBERO-10 in our experiments and test OpenVLA [2], π0 [4] and π0-FAST [5] on it. We adopt the model checkpoints that are finetuned on the LIBERO benchmark and released by their authors. In experiments, 3 out of 10 tasks are randomly picked and reserved as unseen tasks.

SimplerEnv [63]: SimplerEnv provides a high-fidelity simulation environment for manipulation policies, which are replicas of the demonstration data from RT-series [1, 7, 31] and BridgeData V2 [33]. On SimplerEnv, we test pretrained π0 models from a reproduction [64], which we denote as π0∗ in this paper. We train and evaluate the failure detection methods on the Google Robot embodiment [1] and on the WidowX embodiment [33], respectively. We exclude the “pick up coke” task because π0∗ rarely fails on it (success rate at 98%). This leaves 4 tasks for each embodiment, among which 3 tasks are seen and 1 task is unseen.

Real-world Franka Experiments: We deploy the π0-FAST-DROID checkpoint [4, 5]1 on a Franka Emika Panda Robot. This checkpoint has been finetuned on the DROID dataset [32], and we do not further collect demonstrations or finetune the VLA model. We design 13 tasks and collect 30 successful and 30 failed rollouts for each task. The real-robot setup and example rollouts are visualized in Fig. 3. In experiments, 3 tasks out of 13 are randomly selected as unseen tasks.

Real-world WidowX Experiments: We also deploy the OpenVLA model pretrained on the “Open-X Magic Soup++” dataset [2] on a WidowX robot manipulator in our lab. With this setup, we collected

1https://github.com/Physical-Intelligence/openpi

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

“pick up the lid and place it on the pot”

Franka Emika Panda Robot

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

“pick up the marker and place it in the cup”

ZED 2 Camera ZED Mini Camera

[Figure 40]

“lift AAA battery”

RealSense D435 Camera

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

WidowX 250 Robot

“put the carrot on plate”

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

- Figure 3: Illustration of real-world experiment setup (left) and example rollouts collected (right).

a total of 532 rollouts on the 8 lifting and pick-and-place tasks, including 244 successful and 288 failed rollouts. In this experiment, 2 tasks out of 8 are randomly selected as unseen tasks.

##### 5.2 Uncertainty Quantification Baselines

Estimating uncertainty in generated responses has been widely used to detect truthfulness or hallucination in LLMs [22, 23, 43, 65]. For VLAs, uncertainty in the generated actions may indicate a lack of ability to solve the given task, and thus correlates with task failures. Therefore, we first adapt the UQ methods from the LLM literature to VLAs and use them as failure detection baselines.

Token uncertainty-based methods aggregate the predictive uncertainty from each generated token. These methods are efficient, as they only require a single forward inference. Given the generated tokens Wt = [wt1,...,wtm], we denote the probability of sampling the token wti as pi and the entropy over the distribution of the ith token as Hi. We adopt the token-based uncertainty estimation methods used by Huang et al. [22] as follows:

Token max prob.: maxi(−log pi); Token avg prob.: −m1 i log pi; Token max entropy: maxi Hi; Token avg entropy: m1 i Hi.

Sample consistency-based methods estimate uncertainty as the inconsistency within multiple generated sentences [22, 23, 65]. For VLA models, the output actions are continuous vectors, and we can measure inconsistency by their variance. Specifically, at time t, given K sampled actions At = {Akt }k=1,...,K, we measure the uncertainty as the total variation over the set of vectors: action total var. = trace(cov(At)). Similarly, we also compute variation for the translational (trans. total var.), rotational (rot. total var.), and gripper control (gripper total var.) components of At.

Furthermore, inspired by semantic entropy [23], we define cluster entropy as entropy(cluster({Akt }k=1,...,K)), where cluster(·) generates an integer set, containing k cluster labels for the k actions and entropy(·) measures the entropy of the integer set. UQ methods based on sample consistency are shown to perform better for LLM [22, 23], but they necessitate multiple inferences, which may not be practical for large VLAs that control robots in real time.

##### 5.3 Failure Detection Baselines

Embedding Distance: We compare to baselines that directly use the distances in the feature space as failure scores. Specifically, instead of training a neural network, all VLA embeddings from Dtrain are stored in the two feature sets, Esucc and Efail, containing all VLA embeddings from successful and failed rollouts, respectively. During evaluation on Deval-seen and Deval-unseen, failure scores are

OpenVLA LIBERO

###### 0-FAST LIBERO

1.0

0.9

| |
|---|

| |
|---|

Method

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

0.9

Max prob

| |
|---|

0.8

BalancedAccuracy

BalancedAccuracy

Avg prob

0.8

| |
|---|

Max entropy

0.7

| |
|---|

Avg entropy

0.7

| |
|---|

Mahalanobis dist.

0.6

Euclidean dist. k-NN

0.6

| |
|---|

Cosine dist. k-NN

| |
|---|

0.5

0.5

PCA-KMeans

RND

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 Average Detection Time (Normalized)

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 Average Detection Time (Normalized)

###### 0 LIBERO

LogpZO

* 0 SimplerEnv

Action total var. Trans. total var. Rot. total var.

0.80

0.8

0.75

BalancedAccuracy

| |
|---|

BalancedAccuracy

Gripper total var.

0.70

0.7

Cluster entropy

0.65

| |
|---|

| |
|---|

| |
|---|

| |
|---|

STAC

| |
|---|

| |
|---|

0.60

| |
|---|

0.6

STAC-Single

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.55

SAFE-LSTM

| |
|---|

| |
|---|

SAFE-MLP

0.50

0.5

GT Avg Fail Time

0.45

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 Average Detection Time (Normalized)

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 Average Detection Time (Normalized)

- Figure 4: In all simulation experiments, the proposed SAFE-LSTM and SAFE-MLP perform better than or on par with the best baselines. The plots show the variation of balanced accuracy (bal-acc)

with respect to average detection time (T-det) on Deval-unseen, under different significance levels α used for functional CP. Good failure detection methods should detect policy failures both accurately (high bal-acc) and proactively (lower T-det), and thus place curves towards the top left in each plot. Note that baselines in gray require multiple action samples.

computed as st = d(et,Esucc) − d(et,Efail), where d(·,·) measures the distance between a single vector and a set of vectors. Intuitively, if et is far from Esucc and close to Efail, it’s more likely to fail. Following recent works [10, 18, 66], we ablate different types of distance, including Mahalanobis distance, and Euclidean and Cosine distance averaged over k-Nearest Neighbors of et. We also compare to the PCA-KMeans distance measure from Liu et al. [9].

Learned OOD Detector: Following Xu et al. [8], we adopt LogpZO, the best-performing method, and RND [39], a strong baseline, for OOD detection–based failure detectors. Both methods use a neural network fOOD(·) to model the embedding distribution from successful rollouts and return an OOD score for a new embedding. We adapt them to learn from both successful and failed rollouts by training two models, fsuccOOD(·) and ffailOOD(·), on Esucc and Efail respectively. Similar to embedding distance baselines, the failure score is computed as st = fsuccOOD(et) − ffailOOD(et).

Action Consistency: STAC [18] detects policy failures by measuring the statistical distance on the overlapping segment of two consecutive predicted action chunks. As it requires sampling multiple actions from the policy ([18] uses 256 actions), it compromises real-time operation for real robots, because unlike relatively small diffusion policy networks, large VLAs are not optimized for parallel inference2. Therefore, we only test STAC in the simulation experiments with 10 sampled actions. We also adopt STAC-Single, a real-time version of STAC, which computes action inconsistency using only one sample from each inference timestep. Since OpenVLA only outputs one-step immediate action (H = 1), STAC and STAC-single do not apply to it.

##### 5.4 Evaluation Protocol

We consider two types of evaluation. The first type evaluates how well st separates the successful and failed rollouts across all possible selections of δt. Following the evaluation protocol widely adopted in the LLM UQ literature [22, 49, 53, 68], we use the area under the ROC curve (ROC-AUC) metric. Furthermore, because a failure flag is raised whenever st exceeds δt, a successful rollout (ground truth negative) becomes a false positive whenever st > δt, and remains a true negative only if st ≤ δt for all time. Therefore, we consider the max-so-far score s¯t = maxτ=1,...,t sτ and compute the ROC-AUC metric based on s¯T, the maximum failure score throughout the entire rollout.

2π0 is 152% slower and π0-FAST is 221% slower to generate 10 action samples compared to 1 sample, tested on a single NVIDIA RTX 3090 GPU, with vmap optimization and JiT compilation in Jax [67]. For comparison, SAFE methods only add negligible overhead (<1ms, or <1% of the inference time of π0 and π0-FAST).

|[Figure 49]<br><br>[Figure 50]|
|---|

|[Figure 51]<br><br>[Figure 52]|
|---|

|[Figure 53]<br><br>[Figure 54]|
|---|

|[Figure 55]<br><br>[Figure 56]|
|---|

|[Figure 57]<br><br>[Figure 58]|
|---|

|[Figure 59]<br><br>[Figure 60]|
|---|

|[Figure 61]<br><br>[Figure 62]|
|---|

|[Figure 63]<br><br>[Figure 64]|
|---|

|[Figure 65]<br><br>[Figure 66]|
|---|

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

|(a) 𝜋0-FAST. “Pick up book and place in back compartment of caddy”|
|---|

|(c) 𝜋0∗. “Put the carrot on the plate”|
|---|

(b) OpenVLA. “Put both alphabet soup and tomato sauce in basket”

- Figure 5: Failures detected by SAFE-LSTM align well with the actual robot failures, as shown in the corresponding camera observations from simulation experiments. The blue-shaded areas show the

functional CP band Cα. Once failure scores exceed Cα, a failure flag is raised. In (a), the π0-FAST policy misses the insertion, and its actions become unstable after that. In (b) and (c), OpenVLA and

π0∗ miss the grasp but still proceed to the placing action, causing a failure detection. Note that these tasks are not seen when training SAFE-LSTM.

The second type of evaluation utilizes δt = uppert calibrated by functional CP in Section 4.3. By setting the significance level α, we get a decisive positive/negative detection for each rollout. Following

related works [8, 18], we consider the following metrics: true positive rate (TPR), false positive rate (FPR), balanced accuracy (bal-acc), and averaged detection time (T-det), where Bal-Acc = TPR+TNR2 . T-det is the relative timestep where st > δt for the first time (if st never exceeds δt, T-det becomes 1), averaged over all ground truth failed rollouts.

- 6 Results

##### 6.1 How well do failure detectors distinguish failures from successes?

In Table 1 and Fig. 6 (a), we report the ROC-AUC metric based on s¯T, in simulation and realworld experiments, respectively. With a higher ROC-AUC metric, a failure detector achieves higher accuracy averaged over all possible thresholds. The tables show that Token Unc. methods have poor performance, which is aligned with findings in the LLM literature [22, 23]. On the other hand, the Sample Consist. and STAC [18] methods, which require multiple action samples, perform better and even achieve the best performance on unseen tasks in π0-FAST LIBERO (STAC) and π0 LIBERO (Gripper total var.). However, as these methods require multiple action samples, they cause significant overhead for VLA models and thus are not currently practical for real robots. Embed. Distr. methods perform well, achieving the best performance in two simulation benchmarks (π0 and π0-FAST) and are the second best in the real world. This demonstrates that a VLA’s internal features are informative about task execution success/failure. The proposed SAFE methods perform better or on par with the best baselines, consistently in all settings. Averaged across simulation benchmarks, SAFE-MLP and SAFE-LSTM have similar performance, both outperforming the best baseline by 4-5% on unseen tasks, while still achieving the best performance on seen tasks. For the real-robot experiments, on both π0-FAST+Franka and OpenVLA+WidowX rollouts, SAFE-MLP achieves the best performance and SAFE-LSTM performs closely with the best baseline (Mahala. dist. and Euclid. k-NN). Comparing SAFE with Embed. Distr. methods, we attribute the success of SAFE to its stronger ability to extract high-level abstract information from raw feature vectors through learned neural networks.

##### 6.2 What is the trade-off between detection accuracy and time using functional CP?

- In Fig. 4, we use Deval-seen to calibrate the functional CP band Cα and evaluate on Deval-unseen. By varying the user-specified α, we can adjust the conservativeness of the failure detectors and obtain a trade-off between accuracy (bal-acc) and detection time (T-det). A good failure detector should detect failures both accurately (higher bal-acc) and promptly (lower T-det), and thus have the curve rise toward the top-left corner in the plots of Fig. 4. As we can see from Fig. 4, the proposed SAFE-MLP and SAFE-LSTM perform the best on OpenVLA+LIBERO and π0+SimplerEnv benchmarks, and are

|[Figure 73]<br><br>|[Figure 74]<br><br>[Figure 75]| |
|---|---|---|
|(c) “put cup to the upright position” 𝑠𝑡 plateaus after task success| | |
|[Figure 76]<br><br>|[Figure 77]<br><br>[Figure 78]| |

|[Figure 79]<br><br>|[Figure 80]<br><br>[Figure 81]| |
|---|---|---|
|(b) “pick up marker and place it in cup” 𝑠𝑡 plateaus after task success| | |
|[Figure 82]<br><br>|[Figure 83]<br><br>[Figure 84]| |

π0-FAST Franka OpenVLA WidowX Method Seen Unseen Seen Unseen

Max prob. 53.74 48.59 50.77 54.25 Avg prob. 51.60 47.30 48.94 44.36 Max entropy 59.23 53.50 51.88 49.19 Avg entropy 50.67 46.08 47.72 53.84 Mahala. dist. 75.54 53.93 82.37 70.00 Euclid. k-NN 80.35 60.27 72.01 53.64 Cosine k-NN 80.23 59.51 74.76 65.88 PCA-KMeans 49.98 51.03 75.62 47.22 RND 62.00 45.83 66.68 47.67 LogpZO 64.43 52.24 62.94 51.32 STAC-Single 45.24 38.01 – –

SAFE-LSTM 77.27 58.70 84.29 71.80 SAFE-MLP 86.76 64.16 89.11 88.42

(d) “put both carrot and ball in the bowl”

(e) “pick up knife and put it on plate”

Policy gives zero action and execution is frozen

Knife is dropped after picked up

(a) Failure Detection ROC-AUC

Figure 6: SAFE-MLP achieves the best failure detection performance in real-world experiments with both π0-FAST Franka and OpenVLA WidowX. Plot (a) presents quantitative results, while (b–e) show qualitative examples from SAFE-MLP on the real robot. ROC-AUC values are averaged over five random seeds with different task splits.

on par with the best baseline on the other two benchmarks. We also manually annotate the ground truth (GT) failure timesteps (when a human thinks that failure happens or intervention is needed) for failed rollouts, and plot them as blue vertical lines in Fig. 4. Comparing SAFE’s performance with the GT fail time, we can see that SAFE can detect failures with high accuracy in the early stages of rollouts and potentially before the failure happens. This early detection allows early intervention for policy failures before they get stuck in execution or cause harm to the real-world environment.

##### 6.3 What failure modes are detected, and do they align with human judgment?

- In Fig. 5 and Fig. 6(b-d), we visualize rollouts with the failure scores detected by SAFE. Fig. 5 demonstrates common failure modes in simulation, including imprecise insertion, oscillatory motions, and missed grasps. Two successful rollouts on the real robot are shown in Fig. 6(b-c), where failure scores stop increasing after task completion. For the failed rollouts, the failure flag is raised after the policy is frozen (Fig. 6d) or the object slips out of the gripper (Fig. 6e). This aligns well with human intuition. Please refer to Appendix for video illustrations.

##### 6.4 How efficient and practical it is to deploy SAFE?

SAFE uses a 1-2 layer MLP or LSTM and poses a minimal (less than 1%) computational overhead at runtime. For example, SAFE-LSTM contains 2.3 million parameters and introduces an additional 0.73 ms of inference time. This is negligible compared to large VLA models. For instance, pi0 has 3.3 billion parameters and an inference time of 149 ms. SAFE only requires access to the latent features of VLA models and is compatible with any white-box robot policies based on neural networks. However, SAFE does require deploying the policy and collecting successful and failed rollouts to train the failure detector before it can detect failures.

#### 7 Conclusion

In this paper, we introduce the multitask failure detection problem for generalist VLA policies, where failure detectors are trained only on seen tasks and evaluated on unseen tasks. We analyze VLA’s internal feature space and find that the internal features are separated for successful and failed rollouts. Based on this observation, we propose SAFE, a simple and efficient failure detection method by operating on the VLA’s internal features. SAFE is evaluated on multiple VLAs in both simulation and the real world, and compared with diverse baselines. Experiments show that SAFE achieves SOTA results in failure detection, and aligns with human intuition.

Limitations: Most recent VLAs have shown capabilities in handling diverse modalities, controlling diverse embodiments, and learning latent actions from non-robotic action-less video data [69, 70]. This paper only considers multitask failure detection for manipulation tasks, and it is not clear how well the failure detectors generalize across embodiments, sim2real or to action-less videos. Besides, SAFE only uses features from the last layer, and how to effectively aggregate information across multiple layers of a VLA remains an open question for future work.

#### Acknowledgments and Disclosure of Funding

The authors were partially supported by the Toyota Research Institute (TRI) and NSERC Discovery Grant. The authors thank Blerim Abdullai, Sebastian Aegidius, Jasper Gerigk, Ruthrash Hari, and Wei-Cheng Tseng for helpful discussions and feedback. The authors also thank the anonymous reviewers and area chairs for their constructive feedback.

#### References

- [1] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.
- [2] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. OpenVLA: An open-source vision-language-action model. In Pulkit Agrawal, Oliver Kroemer, and Wolfram Burgard, editors, Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pages 2679–2713. PMLR, 06–09 Nov 2025.
- [3] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Charles Xu, Jianlan Luo, Tobias Kreiman, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An open-source generalist robot policy. In Proceedings of Robotics: Science and Systems, Delft, Netherlands, 2024.
- [4] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo

Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

- [5] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. FAST: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.
- [6] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.
- [7] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. RT-2: Vision-languageaction models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.
- [8] Chen Xu, Tony Khuong Nguyen, Emma Dixon, Christopher Rodriguez, Patrick Miller, Robert Lee, Paarth Shah, Rares Ambrus, Haruki Nishimura, and Masha Itkina. Can we detect failures without failure data? uncertainty-aware runtime failure detection for imitation learning policies. arXiv preprint arXiv:2503.08558, 2025.
- [9] Huihan Liu, Yu Zhang, Vaarij Betala, Evan Zhang, James Liu, Crystal Ding, and Yuke Zhu. Multi-task interactive robot fleet learning with visual world models. In 8th Annual Conference on Robot Learning (CoRL), 2024.
- [10] Rohan Sinha, Amine Elhafsi, Christopher Agia, Matthew Foutter, Ed Schmerling, and Marco Pavone. Real-time anomaly detection and reactive planning with large language models. In Robotics: Science and Systems, 2024.
- [11] Josiah Wong, Albert Tung, Andrey Kurenkov, Ajay Mandlekar, Li Fei-Fei, Silvio Savarese, and Roberto Martín-Martín. Error-aware imitation learning from teleoperation data for mobile manipulation. In Aleksandra Faust, David Hsu, and Gerhard Neumann, editors, Proceedings of the 5th Conference on Robot Learning, volume 164 of Proceedings of Machine Learning Research, pages 1367–1378. PMLR, 08–11 Nov 2022.

- [12] Huihan Liu, Shivin Dass, Roberto Martín-Martín, and Yuke Zhu. Model-based runtime monitoring with interactive imitation learning. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 4154–4161. IEEE, 2024.
- [13] Cem Gokmen, Daniel Ho, and Mohi Khansari. Asking for help: Failure prediction in behavioral cloning through value approximation. In IEEE International Conference on Robotics and Automation (ICRA), pages 5821–5828. IEEE, 2023.
- [14] Annie Xie, Fahim Tajwar, Archit Sharma, and Chelsea Finn. When to ask for help: Proactive interventions in autonomous reinforcement learning. Advances in Neural Information Processing Systems, 35:16918–16930, 2022.
- [15] Rohan Sinha, Edward Schmerling, and Marco Pavone. Closing the loop on runtime monitors with fallback-safe mpc. In Conference on Decision and Control (CDC), pages 6533–6540. IEEE, 2023.
- [16] Alec Farid, David Snyder, Allen Z. Ren, and Anirudha Majumdar. Failure prediction with statistical guarantees for vision-based robot control. In Robotics: Science and Systems, 2022.
- [17] Anirudha Majumdar, Mohit Sharma, Dmitry Kalashnikov, Sumeet Singh, Pierre Sermanet, and Vikas Sindhwani. Predictive red teaming: Breaking policies without breaking robots. arXiv preprint arXiv:2502.06575, 2025.
- [18] Christopher Agia, Rohan Sinha, Jingyun Yang, Ziang Cao, Rika Antonova, Marco Pavone, and Jeannette Bohg. Unpacking failure modes of generative policies: Runtime monitoring of consistency and progress. In Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pages 689–723. PMLR, 2025.
- [19] Jiafei Duan, Wilbert Pumacay, Nishanth Kumar, Yi Ru Wang, Shulin Tian, Wentao Yuan, Ranjay Krishna, Dieter Fox, Ajay Mandlekar, and Yijie Guo. AHA: A vision-language-model for detecting and reasoning over failures in robotic manipulation. arXiv preprint arXiv:2410.00371, 2024.
- [20] Yuqing Du, Ksenia Konyushkova, Misha Denil, Akhil Raju, Jessica Landon, Felix Hill, Nando de Freitas, and Serkan Cabi. Vision-language models as success detectors. In Conference on Lifelong Learning Agents, pages 120–136. PMLR, 2023.
- [21] Jacopo Diquigiovanni, Matteo Fontana, Simone Vantini, et al. The importance of being a band: Finite-sample exact distribution-free prediction sets for functional data. STATISTICA SINICA, 1:1–41, 2024.
- [22] Yuheng Huang, Jiayang Song, Zhijie Wang, Shengming Zhao, Huaming Chen, Felix Juefei-Xu, and Lei Ma. Look before you leap: An exploratory study of uncertainty measurement for large language models. arXiv preprint arXiv:2307.10236, 2023.
- [23] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In The Eleventh International Conference on Learning Representations, ICLR, 2023.
- [24] Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3D-VLA: A 3d vision-language-action generative world model. arXiv preprint arXiv:2403.09631, 2024.
- [25] Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, Hang Li, and Tao Kong. Vision-language foundation models as effective robot imitators. In The Twelfth International Conference on Learning Representations, ICLR, 2024.
- [26] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

- [27] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024.
- [28] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic VLMs: Investigating the design space of visually-conditioned language models. In International Conference on Machine Learning, pages 23123–23144. PMLR, 2024.
- [29] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In Kostas E. Bekris, Kris Hauser, Sylvia L. Herbert, and Jingjin Yu, editors, Robotics: Science and Systems, 2023.
- [30] Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Zhibin Tang, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, et al. TinyVLA: Towards fast, data-efficient vision-language-action models for robotic manipulation. IEEE Robotics and Automation Letters, 2025.
- [31] Open X-Embodiment Collaboration. Open x-embodiment: Robotic learning datasets and RT-X models. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903, 2024. doi: 10.1109/ICRA57147.2024.10611477.
- [32] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, et al. DROID: A large-scale in-the-wild robot manipulation dataset. In Robotics: Science and Systems, 2024.
- [33] Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe HansenEstruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning, pages 1723–1736. PMLR, 2023.
- [34] Rohan Sinha, Apoorva Sharma, Somrita Banerjee, Thomas Lew, Rachel Luo, Spencer M Richards, Yixiao Sun, Edward Schmerling, and Marco Pavone. A system-level view on out-ofdistribution data in robotics. arXiv preprint arXiv:2212.14020, 2022.
- [35] Rajesh Natarajan, Santosh Reddy, Subash Chandra Bose, HL Gururaj, Francesco Flammini, and Shanmugapriya Velmurugan. Fault detection and state estimation in robotic automatic control using machine learning. Array, 19:100298, 2023.
- [36] Quazi Marufur Rahman, Peter Corke, and Feras Dayoub. Run-time monitoring of machine learning for robotic perception: A survey of emerging trends. IEEE Access, 9:20067–20075, 2021.
- [37] Trevor Ablett, Filip Mari´c, and Jonathan Kelly. Fighting failures with fire: Failure identification to reduce expert burden in intervention-based learning. arXiv preprint arXiv:2007.00245, 2020.
- [38] Chen Xu, Xiuyuan Cheng, and Yao Xie. Normalizing flow neural networks by jko scheme. Advances in Neural Information Processing Systems, 36:47379–47405, 2023.
- [39] Nantian He, Shaohui Li, Zhi Li, Yu Liu, and You He. ReDiffuser: Reliable decision-making using a diffuser with confidence estimation. In International Conference on Machine Learning, pages 17921–17933. PMLR, 2024.
- [40] Sivakumar Balasubramanian, Alejandro Melendez-Calderon, Agnes Roby-Brami, and Etienne Burdet. On the analysis of movement smoothness. Journal of neuroengineering and rehabilitation, 12:1–11, 2015.
- [41] Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. In The Twelfth International Conference on Learning Representations, ICLR, 2024.
- [42] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Comput. Surv., 55(12):248:1–248:38, 2023.

- [43] Ola Shorinwa, Zhiting Mei, Justin Lidard, Allen Z Ren, and Anirudha Majumdar. A survey on uncertainty quantification of large language models: Taxonomy, open research challenges, and future directions. arXiv preprint arXiv:2412.05563, 2024.
- [44] Chen Ling, Xujiang Zhao, Wei Cheng, Yanchi Liu, Yiyou Sun, Xuchao Zhang, Mika Oishi, Takao Osaki, Katsushi Matsuda, Jie Ji, Guangji Bai, Liang Zhao, and Haifeng Chen. Uncertainty decomposition and quantification for in-context learning of large language models. In 2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics, 2024. URL https://openreview.net/forum?id=Oq1b1DnUOP.
- [45] Yijun Xiao and William Yang Wang. On hallucination and predictive uncertainty in conditional language generation. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 2734–2744, 2021.
- [46] Andrey Malinin and Mark J. F. Gales. Uncertainty estimation in autoregressive structured prediction. In 9th International Conference on Learning Representations, ICLR, 2021.
- [47] Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. Generating with confidence: Uncertainty quantification for black-box large language models. Trans. Mach. Learn. Res., 2024, 2024. URL https://openreview.net/forum?id=DWkJCSxKU5.
- [48] Jiuhai Chen and Jonas Mueller. Quantifying uncertainty in answers from any language model and enhancing their trustworthiness. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5186–5200, 2024.
- [49] Amos Azaria and Tom Mitchell. The internal state of an llm knows when it’s lying. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 967–976, 2023.
- [50] Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inferencetime intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36:41451–41530, 2023.
- [51] Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision. In The Eleventh International Conference on Learning Representations, ICLR, 2023.
- [52] Xuefeng Du, Chaowei Xiao, and Sharon Li. HaloScope: Harnessing unlabeled llm generations for hallucination detection. Advances in Neural Information Processing Systems, 37:102948– 102972, 2024.
- [53] Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, and Yarin Gal. Semantic entropy probes: Robust and cheap hallucination detection in llms. arXiv preprint arXiv:2406.15927, 2024.
- [54] Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. INSIDE: llms’ internal states retain the power of hallucination detection. In The Twelfth International Conference on Learning Representations, ICLR, 2024.
- [55] Seongheon Park, Xuefeng Du, Min-Hsuan Yeh, Haobo Wang, and Yixuan Li. How to steer llm latents for hallucination detection? arXiv preprint arXiv:2503.01917, 2025.
- [56] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.
- [57] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008.
- [58] Anastasios N Angelopoulos and Stephen Bates. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv preprint arXiv:2107.07511, 2021.
- [59] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735–1780, 1997.

- [60] Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. Algorithmic learning in a random world, volume 29. Springer, 2005.
- [61] Chen Xu and Yao Xie. Conformal prediction for time series. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(10):11575–11587, 2023. doi: 10.1109/TPAMI.2023. 3272339.
- [62] Chen Xu and Yao Xie. Sequential predictive conformal inference for time series. In International Conference on Machine Learning, pages 38707–38727. PMLR, 2023.
- [63] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Oier Mees, Karl Pertsch, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, et al. Evaluating real-world robot manipulation policies in simulation. In 8th Annual Conference on Robot Learning, 2024.
- [64] Allen Z. Ren. open-pi-zero: Re-implementation of pi0 vision-language-action (vla) model from physical intelligence. https://github.com/allenzren/open-pi-zero, 2025. Version 0.1.1, released January 27, 2025. Accessed April 6, 2025.
- [65] Yashvir S Grewal, Edwin V Bonilla, and Thang D Bui. Improving uncertainty quantification in large language models via semantic embeddings. arXiv preprint arXiv:2410.22685, 2024.
- [66] Julia Nitsch, Masha Itkina, Ransalu Senanayake, Juan Nieto, Max Schmidt, Roland Siegwart, Mykel J Kochenderfer, and Cesar Cadena. Out-of-distribution detection for automotive perception. In IEEE International Intelligent Transportation Systems Conference (ITSC), pages 2938–2943. IEEE, 2021.
- [67] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/jax-ml/jax.
- [68] Mingjian Jiang, Yangjun Ruan, Prasanna Sattigeri, Salim Roukos, and Tatsunori Hashimoto. Graph-based uncertainty metrics for long-form language model outputs. arXiv preprint arXiv:2410.20783, 2024.
- [69] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny

Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0.5: a visionlanguage-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

- [70] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. GR00T N1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.
- [71] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, ICLR, 2023.
- [72] Joe H Ward Jr. Hierarchical grouping to optimize an objective function. Journal of the American statistical association, 58(301):236–244, 1963.
- [73] Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Schölkopf, and Alexander Smola. A kernel two-sample test. The Journal of Machine Learning Research, 13(1):723–773, 2012.
- [74] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun, editors, International Conference on Learning Representations, ICLR, 2015.
- [75] Isaac Gibbs and Emmanuel Candes. Adaptive conformal inference under distribution shift. Advances in Neural Information Processing Systems, 34:1660–1672, 2021.
- [76] Tianlong Wang, Xianfeng Jiao, Yinghao Zhu, Zhongzhi Chen, Yifan He, Xu Chu, Junyi Gao, Yasha Wang, and Liantao Ma. Adaptive activation steering: A tuning-free llm truthfulness improvement method for diverse hallucinations categories. In Proceedings of the ACM on Web Conference 2025, pages 2562–2578, 2025.

### Appendix

#### A Potential Societal Impact

This work advances the safety and reliability of VLAs through multitask failure detection, which can reduce unintended behaviors during robot deployment. However, the proposed framework could be misused in surveillance or fully autonomous systems with limited human oversight. Additionally, biases or privacy issues may arise from training data collected during robot interactions. These risks can be mitigated through responsible data handling, transparency in model release, and maintaining human oversight in downstream applications.

#### B Experiment Details

##### B.1 Vision-Language-Action Models

We conduct experiments on 3 state-of-the-art large VLA models: OpenVLA [2], π0 [4] and π0FAST [5]. Given the internal feature vectors E ∈ Rn×d

′

produced by a VLA model, where dimension n corresponds to different token positions, diffusion steps, etc. and d is the feature dimension, we aggregate E into a single fixed-dimensional feature vector e ∈ Rd before inputting to the proposed SAFE models. In this paper, we consider and ablate the following ways of feature aggregation:

- • First: take the first vector along the dimension n, e = E1;
- • Last: take the last vector along the dimension n, e = En;
- • Mean: take average over the dimension n, e = n1 ni=1 Ei;

- • First&Last: concatenate the first and the last vector, e = concat(E0,En) ∈ R2d

′

.

Both OpenVLA and π0-FAST first predict a sequence of discrete tokens and then convert them into continuous actions. We take feature vectors before being decoded into the output tokens from

′

the last transformer block as E ∈ Rn×d

, and therefore n corresponds to the number of generated tokens. We ablate the aggregation method along this dimension and denote the aggregation method as aggtoken. For π0-FAST, we additionally ablate using the feature vectors before (“encoded”) and after (“pre-logits”) the final RMS normalization layer as E.

Differently, π0 (and π0∗) outputs action vectors by flow matching [71], and we take the feature vectors before being projected into the velocity field. Suppose π0 predict an action chunk of horizon H and performs k flow matching steps, the internal features become E ∈ RH×k×d and we perform the aggregation process along the H dimension and the k dimension separately to get the final embedding vector e ∈ Rd

′

. The aggregation methods are denoted as agghori and aggdiff along these two dimensions, respectively.

For all VLA models, we ablate different methods of aggregating the hidden features E into a single feature vector e and select the best method according to Deval-seen performance. The detailed ablation results are shown in Section B.8.

OpenVLA and π0∗ use MIT License; π0 and π0-FAST use Apache-2.0 license.

##### B.2 SAFE Failure Detector

SAFE-LSTM uses an LSTM model with 1 layer and a hidden dimension of 256, and an additional linear layer is used to project the hidden states of LSTM into a single scalar st. SAFE-MLP uses a multi-layer perceptron with 2 layers and a hidden dimension of 256. Since successes and failures from the generated rollouts are imbalanced, the losses on positive (failed) and negative (successful) rollouts are weighted by their inverse class frequency. We also apply an L2 regularization loss on the model weights to reduce overfitting, and this loss is weighted by λreg and optimized together with the failure score learning loss LLSTM or LMLP. λreg are determined by grid search.

##### B.3 Failure Detection Baselines

For the cluster entropy baseline, we use agglomerative clustering with the ward linkage criterion [72]. The distance threshold is denoted as δ and decided by grid search.

For the STAC baseline [18], we use the Maximum Mean Discrepancy (MMD) distance measure [73] with radial basis function kernels, which was reported to have the best performance by [18]. The bandwidth of the RBF kernel is 1.

For all baselines except for RND [39] and LogpZO [8], we ablate one version that only considers the failure score computed from the current timestep (“cumsum=False”) and another that uses the cumulative sum (cumsum) of the failure scores over time (“cumsum=True”).

For RND and LogpZO, we use the original implementation provided by the authors 3 and do not accumulate scores. In [8], RND and LogpZO are trained to model the distribution of (encoded) observations ot and predicted actions At. In this work, we adapt them to model the distribution of VLA’s internal embeddings et.

Note that as π0 (and π0∗) does not output discrete tokens, token uncertainty-based baselines do not apply. And for OpenVLA, H = H′ = 1 and thus the STAC [18] and STAC-Single do not apply.

##### B.4 Conformal Prediction

We follow [8, 21] for CP band construction. Please refer to Section. B in the Appendix of [8] for a detailed formulation. Specifically, in our experiments, we use the adaptive modulation function (Equation 2 in the Appendix of [8]), which models the non-extreme behaviors of the functional data.

##### B.5 Benchmark Details

LIBERO [56]: We adopt the LIBERO-10 task suite, which contains the most diverse set of objects, environments, and instructions among the 4 LIBERO task suites, and therefore LIBERO-10 is considered the most challenging task suite. LIBERO-10 contains 10 tasks with 50 rollouts in each task. We use the initial conditions for all rollouts as specified and provided by the author4. To test VLA models on LIBERO, we adopt the trained model weights provided by the respective authors and do not further finetune them. On LIBERO-10, OpenVLA achieves a success rate of 53.7%, π0-FAST achieves 60.2%, and π0 achieves 85.2%. For evaluation, 3 out of 10 tasks are unseen, and within seen tasks, 60% of rollouts are used for Dtrain and the remaining 40% for Deval-seen.

Note that the LIBERO simulator stops the rollout execution when the robot finishes the task (considered a success) or a maximum rollout length is reached (considered a failure). Therefore, in the generated rollouts, failed ones always have the maximum length, but successful ones are shorter. This could result in an unfair advantage for some of the compared failure detectors (if a failure detector simply learns to count the time elapsed, i.e., st = t, it will achieve perfect failure detection since failed rollouts have a fixed and longer duration). To ensure a fair comparison, for evaluation in Table 1, we compute the minimum rollout length for each task and use that as T for that task. The failure detection performance (in ROC-AUC) is then determined based on sT, where T is the same for all successful and failed rollouts within each task.

LIBERO benchmark uses the MIT license.

SimplerEnv [63]: SimplerEnv carefully identifies and reduces the domain gap between the simulation and the real-world demonstration data, and provides simulated environments that highly resemble the demonstration data from RT-series [1, 7, 31] (with the Google Robot embodiment) and BridgeData V2 [33] (with the WidowX embodiment). They show that models pretrained on real-world datasets can also accomplish similar tasks in SimplerEnv without finetuning, and their performance in simulation matches that in the real world.

On this benchmark, we adopt the pretrained model checkpoints of π0∗ [64]. Note that π0∗ model checkpoints are trained separately on the Google Robot embodiment and the WidowX embodiment, which results in two model checkpoints that have different internal feature spaces. Therefore, all

- 3https://github.com/CXU-TRI/FAIL-Detect
- 4https://github.com/Lifelong-Robot-Learning/LIBERO/tree/master/libero/libero/init_files

- Table 2: List of tasks used in π0∗ + SimplerEnv benchmark.

Embodiment Task ID Environment Name π0∗ Success Rate (%)

|Google Robot Google Robot Google Robot Google Robot<br><br>|1<br>2<br>3<br>4<br>|google_robot_move_near_v0<br><br>google_robot_open_drawer<br><br>google_robot_close_drawer<br><br>google_robot_place_apple_in_closed_top_drawer<br><br>|77 50 80 40|
|---|---|---|---|
|WidowX WidowX WidowX WidowX|1<br><br>2<br><br>3<br><br>4<br><br><br>|widowx_carrot_on_plate widowx_put_eggplant_in_basket widowx_spoon_on_towel widowx_stack_cube<br><br>|44 88 79 43|

- Table 3: List of tasks used in the real-world Franka experiments.

|Task<br><br>|Instruction<br><br>|Rollout Length T|
|---|---|---|
|1|close the door|300|
|2|close the drawer<br><br>|200|
|3|pick up the ball and place it in the bowl|400|
|4<br><br>|pick up the knife and put it on the plate<br><br>|350|
|5<br><br>|pick up the lid and place it on the pot|400|
|6<br><br>|pick up the lid from the pot and place it on the table<br><br>|400|
|7<br><br>|pick up the marker and place it in the cup|400|
|8|place the green block on the yellow block<br><br>|350|
|9|place the pink cup to the right of the blue cup<br><br>|300|
|10<br><br>|press the button<br><br>|200|
|11<br><br>|put both the carrot and the ball in the bowl|500|
|12|put the cup to the upright position|500|
|13<br><br>|unfold the cloth<br><br>|500|

failure detectors are trained and evaluated on each embodiment separately as well. All reported evaluation metrics are computed separately for each embodiment and then averaged. In Table 2, we list the tasks used for failure detection on SimplerEnv. We generate 100 rollouts for each task with random initial configurations, and the success rates of π0∗ on each task are also listed in Table 2. A rollout stops after the maximum number of allowed timesteps have passed, regardless of task success or failure. Within each embodiment, 1 out of 4 tasks is unseen, and within the seen tasks, 66% of the rollouts are in Dtrain and the remaining 33% in Deval-seen.

SimplerEnv benchmark uses the MIT license.

Real-world experiments with Franka robot: In Table 3, we list the tasks used in the real-world experiments. For each task, we set a number of timesteps T allowed for one rollout, and all rollouts of the same task are terminated after the same T timesteps regardless of task success or failure. In Fig. 7, we further visualize some example successful and failed rollouts from the real-world experiments.

Real-world experiments with WidowX robot: We also tested the OpenVLA model pretrained on the “Open-X Magic Soup++" dataset on a real WidowX robot arm in our lab. In this experiment, we collected a total of 532 rollouts on the following 8 tasks (listed in Table 4), with 244 successes and 288 failures. Each task has roughly the same number of rollouts.

##### B.6 Benchmark Statistics

In the Table 5, we summarize the statistics on the number of tasks and rollouts collected for each benchmark and how they are split into training and evaluation subsets. We note that as SAFE is designed for multitask failure detection, it is trained on only a limited set of training tasks and rollouts and can generalize to new tasks without further collecting rollouts. While SAFE does require hundreds of rollouts from multiple tasks during training, when handling new tasks, SAFE becomes more efficient than existing task-specific failure detectors (like [8, 18]) that require collecting rollouts for training and calibration for every new task encountered.

Table 4: List of tasks used in the real-world experiments.

|Task<br><br>|Instruction|
|---|---|
|1<br><br>|Lift AAA Battery|
|2<br><br>|Lift Eggplant|
|3<br><br>|Lift Red Bottle|
|4<br><br>|Lift Blue Cup|
|5<br><br>|Put Blue Cup on Plate|
|6|Put the Red Bottle into Pot|
|7|Put the Carrot on Plate|
|8<br><br>|Put the Red Block into the Pot|

|Benchmark|Number of Tasks Seen Unseen Total<br><br>|Number of rollouts Train Eval Seen Eval Unseen Total|
|---|---|---|
|LIBERO π0∗ SimplerEnv, Google Robot π0∗ SimplerEnv, WidowX Octo SimplerEnv Real Franka Real WidowX|7 3 10<br><br>2 2 4 2 2 4<br><br>9 3 12<br><br>10 3 13<br><br><br>6 2 8<br><br>|210 140 150 500 198 102 100 400 198 102 100 400 594 306 300 1200 450 150 180 780 250 133 149 532|

- Table 5: Benchmark statistics for how tasks and rollouts are split into different subsets. Note that as we preformed multiple runs with different random seeds for all experiments, each run will use different set of tasks for seen and unseen subsets.

##### B.7 Training Details

We use Adam optimizer [74] with β1 = 0.9, β1 = 0.999, ϵ = 10−8, and a learning rate (lr) determined by grid search. The SAFE models are trained for 1000 epochs with batch size 512. Note that each rollout is considered as one data point and thus batch size of 512 translates to training on (at most) 512 rollouts in each iteration. All training and evaluation are done on a single NVIDIA A100 40GB GPU. Since SAFE uses small networks (MLP or LSTM with 1 or 2 layers), the typical training time for one model is less than one minute.

##### B.8 Hyperparameter Tuning

To determine the hyperparameters for the proposed SAFE and baselines, we perform a grid search over them and select the ones with the highest failure detection performance (ROC-AUC) on the Deval-seen split. In Table 9, Table 10, and Table 11, we report the hyperparameters we have searched over and the values with the best performance. Note that for the real-world experiments, we fix the et to be the “pre-logits” with “Mean” aggregation.

#### C Additional Results

##### C.1 Feature Visualization and Analysis

We perform the feature analysis similar to Section 4.1 and Fig. 1 on other benchmarks and show the plots in Fig. 8, Note that in this feature analysis process, the t-SNE algorithm was performed on the VLA’s embeddings without any learning. Therefore, the feature dimension reduction process is unsupervised and does not know about task successes or failures.

Comparing the plots in Fig. 1 and Fig. 8, we can see that the embedding spaces from VLAs are different from each other, which corresponds to the different failure patterns presented by the VLAs. For π0-FAST (Fig. 1) and π0 on LIBERO (Fig. 8a and b), when task execution fails, the embeddings fall into the same region (“failure zone”). This corresponds to the major failure mode of π0-FAST trained on the LIBERO dataset, where the predicted actions At become unstable and the robot arms move to weird configurations and out of the observation frame. For OpenVLA on LIBERO (Fig. 8c and d). we observe that for most failed rollouts, the robot freezes at or shakes around certain

configurations during the middle of task execution. Such failed rollouts result in features very close to each other, which corresponds to small blobs of red dots in Fig. 8c.

Despite the different appearances of the embedding spaces from the above benchmarks, their successful and failed rollouts are separable in the feature space. This is aligned with the high performance of the proposed SAFE and the embedding-based baseline methods. Moreover, although the embeddings of the failed rollouts from OpenVLA are spread over the space and do not form a unified “failure zone”, SAFE is still able to learn to separate task failures from successes (possibly by extracting the correlations that are not visualized by t-SNE) and generalize well to unseen tasks, as reported in Table 1.

However, the visualized embeddings of π0-FAST on the real Franka robot (Fig. 8e and f) are different, where embeddings from successful and failed rollouts are not easily separable through the t-SNE visualization. We hypothesize that because the tasks we used for real-world experiments are more diverse, their failures do not have a unified semantic meaning, and thus the embeddings are not clearly separated in the visualization. This explains the limited performance of all failure detection methods as reported by Fig. 6, where ROC-AUC is at most 64 on Deval-unseen. Nevertheless, SAFE-MLP still outperforms all baselines on both seen and unseen splits in this evaluation.

##### C.2 Conformal Prediction Results

We use functional CP [8, 21] to determine the time-varying thresholds δt for failure detection. By varying the significance level α used in functional CP, we can adjust the conservativeness of failure detection and get different performance. In Fig. 9, we plot the change of TNR (True Negative Rate),

TPR (True Positive Rate) and Bal-acc (Balanced Accuracy, TNR+2TPR) w.r.t. α. From Fig. 9, we can observe that while the metrics do vary with the α, choosing α = 0.15 (or in general, between 0.05

and 0.2) performs well across the board. We have also chosen α to be 0.15 for most qualitative results and analyses reported in the paper.

Note that we calibrate the CP bands on successful rollouts (negative data points), and thus if the assumptions used in CP (st are sampled i.i.d.) hold, the TNR rate is lower bounded by and close to 1 − α (the gray dashed line in the TNR plots in Fig. 9) [21]. However, as the multitask failure detection problem requires detecting failures on tasks that are not in the training or the calibration sets, we need to calibrate CP bands on Deval-seen and then evaluate them on Deval-unseen. Therefore, the i.i.d assumption may not hold, and TNR may deviate from the gray dashed line.

From Fig. 9, we can see that on OpenVLA+LIBERO and π)∗+SimplerEnv benchmarks, the TNR curves obtained by SAFE are close to the gray dashed line 1 − α, while those on the other 3 benchmarks are lower than 1 − α. A similar phenomenon is also observed for the baseline methods: none of the TNR curves obtained from the baselines consistently conform to the 1 − α curve across all benchmarks. We attribute this to the challenging nature of the multitask failure detection problem, where the failure scores for calibration and evaluation may not come from the same distribution. Nevertheless, we still adopt the functional CP as a principled method to determine the time-varying failure detection threshold δt. Moreover, from Fig. 9, we can see that SAFE can achieve higher TPR and result in fewer false negatives compared to the baselines. This is crucial for safety-critical environments, where a missing failure (false negative) can be much more catastrophic than a false alarm (false positive).

##### C.3 Failure Detection Time

As mentioned in Section 6.2, we manually label when the failure happens for the failed rollouts. The labeling process is based on video recordings after all rollouts are collected and no interventions were done during the task execution. While the exact times of failure are clear for some failure modes (e.g. dangerous actions, breaking objects), they can be ambiguous and hard to annotate for other failure modes. For example, a policy may freeze in the middle of task execution, and after that either recovering from it or getting stuck indefinitely can be possible. In another case, a policy may repeatedly try grasping the object but keep missing the grasp until timeout, and it’s hard to determine a single point of failure. To handle such cases, we instruct the human annotators to pick the time where they think intervention is needed and they should take over control to prevent an execution failure. In practice, for the above ambiguous failure modes, we annotate the failures after the policy gets stuck by a few seconds or re-tries the grasping action a few times. For some rollouts that look

- Table 6: Performance on the OpenVLA+LIBERO benchmark using different numbers of training tasks.

# Training Tasks 1 3 5 7 Eval Task Split Seen Unseen Seen Unseen Seen Unseen Seen Unseen

Mahalanobis 40.21 52.75 58.00 52.31 57.68 50.78 62.03 58.85 Euclid. k-NN 49.74 63.76 61.66 67.02 59.14 67.11 66.00 55.23 Cosine. k-NN 53.27 60.76 65.39 65.64 67.46 70.57 67.09 69.45 PCA-KMeans 60.39 40.58 61.18 52.87 61.50 53.06 57.18 55.10 RND 29.29 50.32 54.46 47.39 56.71 49.15 52.57 46.88 LogpZO 61.75 56.17 52.89 50.49 65.99 56.60 61.57 52.91 SAFE-LSTM 50.88 52.25 68.85 63.31 70.70 66.31 70.24 72.47 SAFE-MLP 54.34 63.76 67.86 67.03 69.32 68.17 72.68 73.47

very plausible but do not succeed due to the time limit, the failure time is annotated as the end of the rollout. Note that we annotate only the failed rollouts and not the successful ones, even though they may also show subtle signs of failure in the middle.

In Fig. 10, we compare the times of failure detected by the proposed SAFE-MLP model and a human annotator. From Fig. 10, we can see that for both π0 and π0-FAST models, SAFE-MLP can detect failures before they happen (as identified by a human). When used for π0-FAST deployed on LIBERO, SAFE-MLP can forecast failures well in advance and even predict 40% of the failures after the first timestep.

Furthermore, from Fig. 10a and Fig. 10c, we can see that the blue curves jump up on the right edge of the plots. This means that the human annotator does not think these rollouts are failures until the very last moment, where the VLA model is probably on the right track and fails only due to timeout. We think such failures are also hard for failure detectors to detect, and it explains the low performance of all failure detectors on these benchmarks.

##### C.4 Result Variance

In Table 8, we report the standard deviation for all results in Table 1 and Fig. 6 left. Note that for the repeated runs, not only are they using different random seeds, also the tasks are split differently into the seen and the unseen subsets. Since different tasks have different difficulties for failure detection, it is normal to see large standard deviations in Table 8. From Table 8, we can see that the proposed SAFE methods achieve high averaged performance with relatively low standard deviations compared to the baselines, across all evaluation benchmarks. This signifies the strong and also stable performance of SAFE.

#### D Additional Ablation Studies

##### D.1 Number of Training Tasks

As SAFE learns to distinguish failures from successes from training rollouts, the diversity of failure modes and the number of tasks in the training data have an effect on the failure detection performance. To quantify this effect, we conduct an experiment varying the number of seen tasks that are used in training. Note that different tasks typically also have different failure modes, and in this way, we are also ablating the diversity of failure modes.

- In Table 6, we report the failure detection ROC-AUC on the OpenVLA+LIBERO benchmark, trained on different numbers of tasks. While the number of seen tasks is ablated, all experiments use the same set of unseen tasks for evaluation, and performance on unseen tasks is comparable. All numbers are averaged over 3 random seeds. Experiments with 7 tasks for training match the setting reported in the paper. Training-free methods do not depend on training tasks and are not shown.

- Table 6 shows that for most methods, with more tasks used for training, the performance on unseen tasks gets better. The proposed SAFE-MLP performs well in all settings and can also achieve good performance when fewer (3 or 5) tasks are used for training.

- Table 7: Comparison of model performance across different visual encoders and architectures. Method LSTM MLP Eval Task Split Seen Unseen Seen Unseen

DINOv2 76.93 56.96 76.20 59.46 CLIP 76.77 52.71 77.88 59.77 DINOv2+CLIP 77.09 59.65 76.36 58.43 VLA (Ours) 77.27 58.70 86.76 64.16

##### D.2 Features from Foundation Models

- In Table 7, we ablate SAFE-MLP and SAFE-LSTM using DINOv2 features, CLIP features, DINOv2 and CLIP concatenated (DINOv2+CLIP), and the VLA last-layer features (VLA; our main method). DINOv2 and CLIP features are extracted from the observation images, and this experiment is conducted over the real-world Franka rollouts, following the same setting as reported in the paper. Numbers are averaged ROC-AUC on the Seen and Unseen tasks.

The best performing method in the above table is the SAFE-MLP method based on VLA last-layer features, where VLA features outperform other feature types by a large margin. We think that this is because VLA feature space learns high-level information about the tasks, and thus can more easily distinguish failures from successes than general pretrained models. Similar findings were also reported in related works like [18].

#### E Additional Discussions

##### E.1 Comparing Failure Detection, Uncertainty Quantification and OOD Detection

Failure detection, uncertainty quantification and OOD detection are three closely connected concepts with subtle differences. SAFE learns to model the probability of failures and detect failures of VLA policies, but it achieves this not through uncertainty quantification (UQ) or OOD detection. Here, we provide a detailed discussion comparing these three concepts.

Failure detection is the task of detecting failures when a robot is performing certain tasks. SAFE learns the likelihood of failure through training on a set of successful and failed rollouts. SAFE-LSTM is trained by BCE loss and outputs a normalized score indicating the probability of failure of VLA. The output of SAFE-MLP is not normalized and thus not a probability. However, output scores from both SAFE-LSTM and SAFE-MLP are calibrated through functional Conformal Prediction CP and can be used for failure detection with theoretical guarantees.

Uncertainty quantification (UQ) measures a VLA’s uncertainty in its outputs and can be used as a proxy for failure detection. In our experiments, the token uncertainty baselines and sample consistency baselines are inspired by LLM/VLM literature and designed based on UQ. Methods in this category are typically training-free, but they only show limited success according to our experiments.

OOD detection-based failure detection methods treat successful rollouts as normal execution conditions and assume that deviations from this norm lead to a higher chance of failure. In our experiments, the embedding distribution-based baselines are designed to detect policy failure based on OOD detection. Methods in this category can work without failed rollouts. In our experiments, we adapted them to take in both successes and failures, and they showed strong performance. Please see Section 2.2 in our paper and also [8] for more comprehensive discussions on these methods.

Uncertainty quantification methods have been widely used for LLM/VLM hallucination detection (see Section 2.3), and OOD detection-based methods have been shown to be effective for failure detection in robotics policies (see Section 2.2). Therefore, we think it’s appropriate to use them as baselines.

Different from existing works based on UQ or OOD detection, SAFE directly learns to detect failures from a history of observations and the language instruction specifying the desired task without using

uncertainty or OOD detection as the proxy measurement. Experiments show that this direct learning regime used by SAFE is more effective and achieves better performance than other methods.

#### F Potential Future Works

##### F.1 Extending Beyond the Last-layer features

In this paper, we maximize the simplicity and transferability when designing SAFE. By only taking feature vectors at the last layer, the proposed method can be easily integrated into any VLA models with minimal implementation changes and no finetuning on the VLAs themselves.

Further fusing or aggregating deep features from multiple layers can also benefit failure detection and is a promising future direction. Related works have shown potential in this direction. For example, [55] proposed Truthfulness Separator Vector (TSV), which is injected in the LLM latent features in the middle of the transformer and is optimized to better separate the hallucinated and truthful responses in the final token feature space. We think a similar technique can also be developed for VLA failure detection. However, this would require a special design and implementation for each VLA model (as some VLA models output discrete tokens, and others use flow matching to output continuous actions, there may not be a single design that can be applied to all VLA architectures), reducing its transferability. We leave the development of such methods as a promising future work. Using the latent feature from an intermediate transformer block may also be a promising future direction. As shown by [49] and [53], LLM latent features from different layers have different performance on hallucination detection, and the best one may not be the last layer. However, exactly which layer works the best may depend on the model and require extensive ablation experiments to find out. For example, as reported by [49], for the OPT-6.7B model, the 20th layer works the best, but for LLAMA2-7B, the 16th layer works the best. To locate the best layer, [49] has to perform a grid search over each LLM tested. On the contrary, in our setting, VLA users can avoid such grid search experiments and simply choose the final layer for failure detection. Therefore, we think that precisely finding the layer that works the best for VLA failure detection is outside the scope of this paper, but it would be very interesting to explore for future work.

##### F.2 Adaptive Thresholding by Online Conformal Prediction

The proposed SAFE and baselines can be extended to online or adaptive conformal prediction (CP) [75]. In such a framework, rollout results are observed one-by-one and compared to the prediction results, and the significance level α is adjusted for each individual task if the prediction is wrong. However, when VLA policies are deployed, they may constantly meet novel environments and customized task instructions, and may rarely repeat the same task. In such a case, it may be less pratical to develop adaptive CP band for each specific task. Therefore, in this work, we focus on the performance of failure detectors when they are directly deployed on a novel task, and have never seen or repeated the task before. In this setting, offline CP is more appropriate. Nevertheless, we believe online CP is an interesting extension to our work, and we leave it as an important future work.

##### F.3 Using Detected Failures for Behavior Improvement

SAFE focuses on detecting failures accurately and in a timely manner, which enables either the robot to abort potentially dangerous actions or a human monitor to step in and take over control. How to learn a recovery policy or how to improve the VLAs themselves are important areas for future work. In this work, we focused on detecting failures of multitask VLAs reliably and in real-time, which is a crucial stepping stone towards autonomous recovery (e.g., with a fallback policy) and policy improvement (e.g., through interactive imitation learning).

We think it is possible to use the findings from this paper to further develop methods for steering or improving VLA behaviors. For example, we show that the embeddings for successful and failed rollouts are separated in the latent space, so it is possible to learn a steering vector that manipulates the latent activations of a VLA and changes its output actions, as done in [50] and [76]. However, different from the stylization or hallucination reduction tasks for LLMs, robot manipulation involves multistep closed-loop interaction between the policy and the environment, which greatly complicates the relationship between VLA outputs and task execution successes. Therefore, how to improve

###### Table 8: Mean and standard deviation of failure detection ROC-AUC on all benchmarks. This table complements the results from Table 1 and Fig. 6 left.

VLA Model OpenVLA π0-FAST π0 π0∗ π0-FAST Benchmark LIBERO LIBERO LIBERO SimplerEnv Real Franka Eval Task Split Seen Unseen Seen Unseen Seen Unseen Seen Unseen Seen Unseen

|Max prob. Avg prob. Max entropy Avg entropy|50.25±2.51 53.83±6.32<br><br>44.05±1.26 51.58±1.82<br><br>52.94±4.36 53.09±7.68<br><br>45.27±1.78 50.03±3.18<br><br><br>|61.32±9.57 69.44±13.61 52.46±3.44 58.04±5.64 46.69±13.33 62.96±19.62 50.93±1.22 58.63±3.47|- -<br><br>- -<br><br>- -<br><br>- -<br><br><br>|- -<br>- -<br>- -<br>- -<br>| |53.74±3.46 48.59±3.00 51.60±3.12 47.30±4.32 59.23±3.06 53.50±3.15 50.67±3.96 46.08±4.79|
|---|---|---|---|---|---|---|
|Mahalanobis dist. Euclidean dist. k-NN Cosine dist. k-NN PCA-KMeans [9] RND [39] LogpZO [8]<br><br>|62.03±5.11 58.85±4.16<br><br>66.00±2.33 55.23±10.05<br>67.09±2.74 69.45±6.14 57.18±2.04 55.10±1.16 52.57±4.56 46.88±4.92 61.57±3.62 52.91±5.79<br>|93.56±2.32 83.79±7.18 92.04±2.39 84.12±6.47 92.09±1.70 84.64±4.90 68.46±4.92 57.12±10.44<br><br>88.67±3.05 81.57±8.67 91.52±2.39 83.07±7.17<br><br>|77.12±8.57 74.31±12.64 75.64±6.20 70.73±16.69<br><br>75.76±6.16 70.31±16.84 64.92±8.90 60.35±19.93 71.92±7.02 69.44±19.39<br>76.80±9.12 73.23±11.64<br>|88.42±2.82 52.84±31.97<br><br>89.73±3.08 68.41±9.22<br><br>90.19±4.05 71.32±12.02<br><br><br>66.88±5.10 61.19±14.76<br><br>85.07±4.04 65.89±6.52<br><br>88.79±4.92 74.66±14.96| |75.54±4.07 53.93±5.06 80.35±5.36 60.27±4.79 80.23±5.12 59.51±5.76 51.91±4.20 49.86±6.19 62.00±5.44 45.83±5.10 64.43±7.82 52.24±3.68|
|Action total var. Trans. total var. Rot. total var. Gripper total var. Cluster entropy<br><br>|62.76±1.66 65.43±2.50 55.33±2.06 58.99±5.13 47.85±2.88 55.30±4.38 61.84±2.67 64.48±3.05 50.16±2.36 51.44±1.01|76.95±7.22 74.50±12.19<br><br>78.21±4.09 80.03±9.11 80.87±5.85 77.29±8.71 76.82±7.10 74.42±12.13<br><br>80.22±7.37 80.53±8.65<br><br>|77.20±5.65 75.18±5.08 49.38±9.95 54.71±7.57 52.94±7.56 61.06±10.60 77.19±5.66 75.19±5.08 76.19±4.31 72.12±1.04|68.41±10.81 67.94±15.97<br><br>63.27±7.17 55.90±19.19<br><br>58.07±10.41 62.10±9.39<br><br>69.16±9.50 69.29±14.77<br><br><br>68.25±9.03 73.66±16.03<br><br>| |- -<br>- -<br>- -<br>- -<br>- -<br>|
|STAC [18] STAC-Single<br><br>|- -<br><br>- -<br><br><br>|83.07±4.61 85.31±6.71 85.46±6.55 81.16±8.63|46.55±8.90 47.91±20.94<br><br>68.46±5.10 69.39±8.22<br><br>|60.74±13.89 62.21±16.72 68.71±7.06 70.40±8.76| |- 45.24±3.68 38.01±9.81|
|SAFE-LSTM SAFE-MLP|70.24±1.49 72.47±5.55 72.68±2.38 73.47±5.39<br><br>|92.98±2.62 84.48±7.29 90.06±2.82 80.44±5.72<br><br>|76.98±5.34 71.09±6.94 73.50±7.43 73.27±11.85<br><br>|88.85±6.30 80.11±10.49<br>89.50±4.49 84.82±8.12<br>| |77.27±5.82 58.70±4.37 86.76±2.64 64.16±5.88|

VLAs through activation steering is a challenging and open research question beyond the scope of our paper.

###### Table 9: Grid-searched and best-performing hyperparameters (in bold text) for OpenVLA+LIBERO (left) and π0-FAST+LIBERO (right).

|Method|HParams<br><br>|Values|
|---|---|---|
|Max prob. Avg prob. Max entropy Avg entropy<br><br>|cumsum cumsum cumsum cumsum|True False True False True False True False<br><br>|
|Mahalanobis dist.<br><br>Euclidean dist. k-NN<br><br>Cosine dist. k-NN<br><br>PCA-KMeans<br><br>RND LogpZO|aggtoken<br><br>Feat<br><br>cumsum aggtoken Feat<br><br>cumsum<br><br>k<br><br>aggtoken<br><br>Feat<br><br>cumsum<br><br>k<br><br>aggtoken<br><br>Feat<br><br>cumsum<br><br>clusters<br><br>dim<br><br>aggtoken<br><br>Feat<br><br>aggtoken<br><br>Feat|First Last Mean<br><br>Encoded Pre-logits<br><br>True False<br><br>First Last Mean<br><br>Encoded Pre-logits<br><br>True False<br><br>1 5 10<br><br>First Last Mean<br><br>Encoded Pre-logits<br><br>True False<br><br>1 5 10<br><br>First Last Mean<br><br>Encoded Pre-logits<br><br>True False<br><br>16 32 64<br><br>32 64 128<br><br>First Last Mean<br><br>Encoded Pre-logits<br><br>First Last Mean<br><br>Encoded Pre-logits<br><br>|
|Action total var. Trans. total var. Rot. total var. Gripper total var. Cluster entropy<br><br>|cumsum cumsum cumsum cumsum cumsum δ|True False True False True False True False True False 0.01 0.05 0.1 0.2 0.5 1 2 5<br><br>|
|STAC STAC-Single<br><br>|cumsum cumsum<br><br>|True False True False|
|SAFE-LSTM<br><br>SAFE-MLP<br><br>|aggtoken Feat lr<br><br>λreg aggtoken Feat lr λreg<br><br>|First Last Mean<br><br>Encoded Pre-logits 3e-5 1e-4 3e-4 1e-3 1e-3 1e-2 1e-1<br><br>First Last Mean Encoded Pre-logits 3e-5 1e-4 3e-4 1e-3 1e-3 1e-2 1e-1|

|Method|HParams<br><br>|Values|
|---|---|---|
|Max prob. Avg prob. Max entropy Avg entropy<br><br>|cumsum cumsum cumsum cumsum|True False True False True False True False<br><br>|
|Mahalanobis dist. Euclidean dist. k-NN<br><br>Cosine dist. k-NN<br><br>PCA-KMeans<br><br>RND LogpZO<br><br>|aggtoken cumsum aggtoken cumsum k<br><br>aggtoken cumsum k<br><br>aggtoken cumsum clusters dim<br><br>aggtoken aggtoken|First Last Mean<br><br>True False<br><br>First Last Mean<br><br>True False<br><br>1 5 10<br><br>First Last Mean<br><br>True False<br><br>1 5 10<br><br>First Last Mean<br><br>True False<br><br>16 32 64<br><br>32 64 128<br><br>First Last Mean First Last Mean<br><br>|
|Action total var. Trans. total var. Rot. total var. Gripper total var. Cluster entropy|cumsum cumsum cumsum cumsum cumsum δ<br><br>|True False True False True False True False True False 0.01 0.05<br><br>|
|SAFE-LSTM<br><br>SAFE-MLP<br><br>|aggtoken lr<br><br>λreg aggtoken lr λreg|First Last Mean<br><br>1e-4 3e-4 1e-3<br><br>1e-3 1e-2 1e-1 1<br><br>First Last Mean<br><br>1e-4 3e-4 1e-3<br><br><br>1e-3 1e-2 1e-1 1|

###### Table 10: Grid-searched and best-performing hyperparameters (in bold text) for π0+LIBERO (left) and π0∗+SimplerEnv (right).

|Method<br><br>|HParams|Values|
|---|---|---|
|Mahalanobis dist.<br><br>Euclidean dist. k-NN<br><br>Cosine dist. k-NN<br><br>PCA-KMeans<br><br>RND LogpZO<br><br>|agghori aggdiff cumsum<br><br>agghori aggdiff cumsum<br><br>k<br><br>agghori aggdiff cumsum<br><br>k agghori aggdiff cumsum clusters dim<br><br>agghori aggdiff agghori aggdiff|First Last First&Last First Last First&Last True False<br><br>First Last First&Last First Last First&Last True False<br><br>1 5 10<br><br>First Last First&Last First Last First&Last True False<br><br>1 5 10<br><br>First Last First&Last First Last First&Last True False<br><br>16 32 64<br><br>32 64 128<br><br>First Last First&Last First Last First&Last First Last First&Last<br><br>First Last First&Last<br><br>|
|Action total var. Trans. total var. Rot. total var. Gripper total var. Cluster entropy|cumsum cumsum cumsum cumsum cumsum δ<br><br>|True False True False True False True False True False 0.01 0.05 0.1 0.2 0.5 1 2 5<br><br>|
|STAC STAC-Single|cumsum cumsum<br><br>|True False True False|
|SAFE-LSTM<br><br>SAFE-MLP<br><br>|agghori aggdiff lr<br><br>λreg agghori aggdiff lr λreg|First Last First&Last First Last First&Last 1e-5 3e-5 1e-4 3e-4 1e-3<br><br>1e-3 1e-2 1e-1<br><br>First Last First&Last First Last First&Last 1e-5 3e-5 1e-4 3e-4 1e-3<br><br>1e-3 1e-2 1e-1|

|Method<br><br>|HParams|Values|
|---|---|---|
|Mahalanobis dist.<br><br>Euclidean dist. k-NN<br><br>Cosine dist. k-NN<br><br>PCA-KMeans<br><br>RND LogpZO<br><br>|agghori aggdiff cumsum<br><br>agghori aggdiff cumsum<br><br>k<br><br>agghori aggdiff cumsum<br><br>k agghori aggdiff cumsum clusters dim<br><br>agghori aggdiff agghori aggdiff<br><br>|First Last Mean First&Last First Last Mean First&Last True False First Last Mean First&Last First Last Mean First&Last True False 1 5 10 First Last Mean First&Last First Last Mean First&Last True False 1 5 10 First Last Mean First&Last First Last Mean First&Last True False 16 32 64 32 64 128<br><br>First Last Mean First&Last First Last Mean First&Last First Last Mean First&Last First Last Mean First&Last|
|Action total var. Trans. total var. Rot. total var. Gripper total var. Cluster entropy|cumsum cumsum cumsum cumsum cumsum δ<br><br>|True False True False True False True False True False 0.01 0.05 0.1 0.2 0.5 1 2 5<br><br>|
|STAC STAC-Single<br><br>|cumsum cumsum|True False True False<br><br>|
|SAFE-LSTM<br><br>SAFE-MLP<br><br>|agghori aggdiff lr<br><br>λreg agghori aggdiff lr λreg|First Last Mean First&Last<br><br>First Last Mean First&Last<br><br>1e-4 3e-4 1e-3<br><br>1e-3 1e-2 1e-1 1<br><br>First Last Mean First&Last<br><br>First Last Mean First&Last<br><br>1e-4 3e-4 1e-3<br><br><br>1e-3 1e-2 1e-1 1|

###### Table 11: Grid-searched and best-performing hyperparameters (in bold text) for π0-FAST on realworld rollouts.

|Method<br><br>|HParams|Values|
|---|---|---|
|Max prob. Avg prob. Max entropy Avg entropy<br><br>|cumsum cumsum cumsum cumsum<br><br>|True False True False True False True False|
|Mahalanobis dist. Euclidean dist. k-NN<br><br>Cosine dist. k-NN PCA-KMeans<br><br>|cumsum cumsum k<br><br>cumsum k cumsum clusters dim<br><br>|True False True False 1 5 10<br><br>True False 1 5 10 True False 16 32 64 32 64 128|
|STAC-Single|cumsum|True False|
|SAFE-LSTM SAFE-MLP<br><br>|lr λreg lr λreg|1e-4 3e-4 1e-3 3e-3<br><br>1e-3 1e-2 1e-1<br><br>1e-4 3e-4 1e-3 3e-3<br><br><br>1e-3 1e-2 1e-1|

“place the green block on the yellow block”

|①|[Figure 85]<br><br>[Figure 86]<br><br>②|③|[Figure 87]<br><br>[Figure 88]<br><br>④|
|---|---|---|---|

Success Criteria:

Pick up the green block and place it on top of the yellow block.

|[Figure 89]<br><br>①|[Figure 90]<br><br>②|③|[Figure 91]<br><br>[Figure 92]<br><br>④|
|---|---|---|---|

###### Failed Reason:

The robot struggle to pick up the green block in Frame ②, then doesn’t place it on the yellow block in Frame ④.

“put the cup to the upright position”

|[Figure 93]<br><br>①|②|[Figure 94]<br><br>③|[Figure 95]<br><br>[Figure 96]<br><br>④|
|---|---|---|---|

Success Criteria:

Grasp the edge or handle of a cup lying horizontally on the table, then rotating it to place it upright.

|[Figure 97]<br><br>①|②|[Figure 98]<br><br>③|[Figure 99]<br><br>[Figure 100]<br><br>④|
|---|---|---|---|

###### Failed Reason:

In Frame ③, it grasp the cup’s edge but it slip, and in Frame ④, it struggle to pick it up again.

“unfold the cloth”

|①|[Figure 101]<br><br>②|[Figure 102]<br><br>[Figure 103]<br><br>③|[Figure 104]<br><br>④|
|---|---|---|---|

Success Criteria:

Pick up a folded cloth and unfold it by spreading it out flat on the table.

|[Figure 105]<br><br>①|[Figure 106]<br><br>②|③|[Figure 107]<br><br>[Figure 108]<br><br>④|
|---|---|---|---|

###### Failed Reason:

The robot can’t grasp the corner of the cloth in Frame ②③④ and gets stuck.

“close the door”

|[Figure 109]<br><br>①|[Figure 110]<br><br>[Figure 111]<br><br>②|③|[Figure 112]<br><br>④|
|---|---|---|---|

Success Criteria:

Close the door by pushing it until it is fully shut tightly, ensuring no gap remains.

|[Figure 113]<br><br>①|[Figure 114]<br><br>②|[Figure 115]<br><br>[Figure 116]<br><br>③|④|
|---|---|---|---|

Failed Reason:

In Frame ④, the door latch does not engage, leaving the door not fully shut.

###### “pick up the marker and place it in the cup”

|[Figure 117]<br><br>①|[Figure 118]<br><br>②|③|[Figure 119]<br><br>[Figure 120]<br><br>④|
|---|---|---|---|

###### Success Criteria:

Pick up the marker from the table and place it inside the cup, ensuring it stays in the cup.

|[Figure 121]<br><br>①|[Figure 122]<br><br>②|③|[Figure 123]<br><br>[Figure 124]<br><br>④|
|---|---|---|---|

###### Failed Reason:

The robot can’t pick up the marker then gets stuck in Frame ②③④.

Figure 7: Example successful and failed rollouts from real-world experiments.

[Figure 125]

[Figure 126]

(a) π0+LIBERO, colored by task success (b) π0+LIBERO, colored by task ID

[Figure 127]

[Figure 128]

(c) OpenVLA+LIBERO, colored by task success (d) OpenVLA+LIBERO, colored by task ID

[Figure 129]

[Figure 130]

(e) π0-FAST+Franka, colored by task success (f) π0-FAST+Franka, colored by task ID

Figure 8: t-SNE plots of VLA’s internal features, from different evaluation benchmarks.

OpenVLA LIBERO

1.0

| |
|---|

TNR:TrueNegativeRate

0.8

0.6

| |
|---|

0.4

0.2

0.0

0.0 0.1 0.2 0.3 0.4 0.5 Significance Level

###### 0-FAST LIBERO

1.0

| |
|---|

TNR:TrueNegativeRate

0.8

0.6

0.4

| |
|---|

0.2

0.0

0.0 0.1 0.2 0.3 0.4 0.5 Significance Level

###### 0 LIBERO

1.0

TNR:TrueNegativeRate

0.8

| |
|---|

0.6

0.4

0.2

0.0

0.0 0.1 0.2 0.3 0.4 0.5 Significance Level

###### * 0 SimplerEnv

1.0

| |
|---|

TNR:TrueNegativeRate

0.8

| |
|---|

0.6

| |
|---|

| |
|---|

0.4

0.2

0.0

0.0 0.1 0.2 0.3 0.4 0.5 Significance Level

0-FAST Franka

1.0

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | | | |
|---|---|---|---|---|
| | | | | |

TNR:TrueNegativeRate

0.8

0.6

0.4

0.2

0.0

0.0 0.1 0.2 0.3 0.4 0.5 Significance Level

OpenVLA LIBERO

1.0

| |
|---|

TPR:TruePositiveRate

0.8

0.6

| |
|---|

| |
|---|

0.4

| |
|---|

0.2

0.0

0.0 0.1 0.2 0.3 0.4 0.5 Significance Level

###### 0-FAST LIBERO

1.0

TPR:TruePositiveRate

0.8

0.6

0.4

0.2

0.0

0.0 0.1 0.2 0.3 0.4 0.5 Significance Level

###### 0 LIBERO

1.0

TPR:TruePositiveRate

0.8

| |
|---|

0.6

0.4

0.2

0.0

0.0 0.1 0.2 0.3 0.4 0.5 Significance Level

###### * 0 SimplerEnv

1.0

TPR:TruePositiveRate

0.8

| |
|---|

| |
|---|

0.6

| |
|---|

| |
|---|

0.4

| |
|---|

0.2

0.0

0.0 0.1 0.2 0.3 0.4 0.5 Significance Level

0-FAST Franka

1.0

TPR:TruePositiveRate

0.8

| |
|---|

0.6

| |
|---|

0.4

0.2

0.0

0.0 0.1 0.2 0.3 0.4 0.5 Significance Level

OpenVLA LIBERO

0.9

| |
|---|

| |
|---|

| |
|---|

Bal-acc:BalancedAccuracy

| |
|---|

| |
|---|

0.8

| |
|---|

| |
|---|

| |
|---|

0.7

| |
|---|

0.6

0.5

0.0 0.1 0.2 0.3 0.4 0.5 Significance Level

###### 0-FAST LIBERO

1.0

| |
|---|

| |
|---|

Bal-acc:BalancedAccuracy

0.9

0.8

0.7

| |
|---|

0.6

0.5

0.0 0.1 0.2 0.3 0.4 0.5 Significance Level

Method

Max prob

Avg prob

###### 0 LIBERO

Max entropy

Avg entropy

0.8

Bal-acc:BalancedAccuracy

| |
|---|

Mahalanobis dist.

| |
|---|

Euclidean dist. k-NN

| |
|---|

0.7

Cosine dist. k-NN

| |
|---|

PCA-KMeans

0.6

RND

| |
|---|

LogpZO

Action total var. Trans. total var. Rot. total var.

0.5

0.4

Gripper total var.

0.0 0.1 0.2 0.3 0.4 0.5 Significance Level

Cluster entropy

STAC

STAC-Single

* 0 SimplerEnv

SAFE-LSTM

SAFE-MLP

|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | | | |
|---|---|---|---|---|
| | | | | |

0.8 Bal-acc:BalancedAccuracy

GT Avg Fail Time

0.7

0.6

0.5

0.4

0.3

0.0 0.1 0.2 0.3 0.4 0.5 Significance Level

0-FAST Franka

Bal-acc:BalancedAccuracy

0.65

0.60

0.55

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

0.50

| |
|---|

0.45

0.0 0.1 0.2 0.3 0.4 0.5 Significance Level

- Figure 9: Additional failure detection results using δt obtained by functional CP. These plots show TNR (left column), TPR (middle column), and Bal-acc (right column) w.r.t. the significance level α, for each evaluation benchmark. These plots are obtained with random seed= 0.

1.0

TP

1.0

TP FN

GT Positives

Detectedfailuretime

0.8

0.8

Cumulativefailures

(proportion)

(relative)

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

0.0 0.5 1.0 GT failure time (relative)

0.0 0.2 0.4 0.6 0.8 1.0 Time (relative)

(a) π0 on LIBERO benchmark.

1.0

1.0

TP FN

Detectedfailuretime

0.8

0.8

Cumulativefailures

(proportion)

(relative)

0.6

0.6

0.4

0.4

0.2

0.2

TP

0.0

GT Positives

0.0

0.0 0.5 1.0 GT failure time (relative)

0.0 0.2 0.4 0.6 0.8 1.0 Time (relative)

(b) π0-FAST on LIBERO benchmark.

1.0

TP

1.0

GT Positives

Detectedfailuretime

0.8

0.8

Cumulativefailures

(proportion)

(relative)

0.6

0.6

TP FN

0.4

0.4

0.2

0.2

0.0

0.0

0.0 0.5 1.0 GT failure time (relative)

0.0 0.2 0.4 0.6 0.8 1.0 Time (relative)

(c) π0-FAST on the real Franka robot.

- Figure 10: Comparison between detected and ground truth (GT) failure w.r.t time. On the left column, we plot the cumulative number of true failures (true positives) detected by SAFE-MLP (red) and a human annotator (blue), w.r.t. elapsed time in each rollout. The right column shows the time of failures detected by SAFE-MLP (y-axis) and a human annotator (x-axis) for each rollout, where failures missed by the detector (false negatives) are plotted in blue crosses. Experiments are done with seed 0 and functional CP with significance level α = 0.15.

