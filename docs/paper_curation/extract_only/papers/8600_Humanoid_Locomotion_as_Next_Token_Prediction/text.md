## Humanoid Locomotion as Next Token Prediction

# arXiv:2402.19469v1[cs.RO]29Feb2024

Ilija Radosavovic1 Bike Zhang1 Baifeng Shi1 Jathushan Rajasegaran1 Sarthak Kamat1 Trevor Darrell1 Koushil Sreenath1 Jitendra Malik1

### Abstract

We cast real-world humanoid control as a next token prediction problem, akin to predicting the next word in language. Our model is a causal transformer trained via autoregressive prediction of sensorimotor trajectories. To account for the multi-modal nature of the data, we perform prediction in a modality-aligned way, and for each input token predict the next token from the same modality. This general formulation enables us to leverage data with missing modalities, like video trajectories without actions. We train our model on a collection of simulated trajectories coming from prior neural network policies, model-based controllers, motion capture data, and YouTube videos of humans. We show that our model enables a full-sized humanoid to walk in San Francisco zero-shot. Our model can transfer to the real world even when trained on only 27 hours of walking data, and can generalize to commands not seen during training like walking backward. These findings suggest a promising path toward learning challenging real-world control tasks by generative modeling of sensorimotor trajectories.

### 1. Introduction

The last decade of artificial intelligence (AI) has shown that large neural networks trained on diverse datasets from the Internet can lead to impressive results across different settings. The core enablers of this wave of AI have been large transformer models (42) trained by generative modeling of massive quantities of language data from the Internet (29, 8, 30, 31, 4). By predicting the next word, these models acquire rich representations of language that can be transferred to downstream tasks (29), perform multi-task learning (30, 31), and learn in a few-shot manner (4).

Are such modeling techniques exclusive to language? Can we learn powerful models of sensory and motor representa-

1University of California, Berkeley. Correspondence to: Ilija Radosavovic <ilija@berkeley.edu>.

tions in a similar fashion? Indeed, we have seen that we can learn good representations of high-dimensional visual data by autoregressive modeling (6) and related masked modeling approaches (13). While there has been positive signal on learning sensorimotor representations in the context of manipulation (32), this area remains largely unexplored.

In this paper, we cast humanoid control as data modeling of large collections of sensorimotor trajectories. Like in language, we train a general transformer model to autoregressively predict shifted input sequences. In contrast to language, the nature of data in robotics is different. It is high-dimensional and contains multiple input modalities. Different modalities include sensors, like joint encoders or inertial measurement units, as well as motor commands. These give rise to sensorimotor trajectories which we view as the sentences of the physical world. Adopting this perspective suggests a simple instantiation of the language modeling framework in the robotic context. We tokenize the input trajectories and train a causal transformer model to predict shifted tokens. Importantly, we predict complete input sequences, including both sensory and motor tokens. In other words, we are modeling the joint data distribution as opposed to the conditional action distribution.

This has several benefits. First, we are training the neural network to predict more bits of information and consequently acquire a richer model of the world. Second, we can leverage noisy or imperfect trajectories that may contain suboptimal actions. Third, we can generalize our framework to learning from trajectories with missing information.

Our core observation is that if a trajectory is incomplete, i.e., some of the sensory or motor information is missing, we can still learn from it by predicting whatever information is present and replacing the missing tokens with learnable mask tokens. The intuition is that if the model has learned to make good predictions, even in the absence of information, it will have acquired a better model of the physical world. A very important source of such data are human videos from the Internet. Namely, we can observe human movement in videos but we do not get access to the motor commands or complete sensory inputs. We demonstrate that our method can learn from such data sources effectively.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Figure 1: A humanoid that walks in San Francisco. We deploy our policy to various locations in San Francisco over the course of one week. Please see our project page for videos. We show that our policy can walk over different surfaces including walkways, concrete, asphalt, tiled plazas, and sanded roads. We find that our policy follows omnidirectional velocity commands well and enables deployment in a challenging city environment like San Francisco.

To validate our method, we apply it to the challenging task of real-world humanoid locomotion. We use the full-sized Digit humanoid robot developed by Agility Robotics. We first collect a dataset of sensorimotor trajectories in simulation. These include complete trajectories from a neural network policy trained with reinforcement learning (33), as well as incomplete trajectories from three different sources: (i) Agility Robotics controller based on model predictive control, (ii) motion capture of humans, and (iii) YouTube videos of humans. We reconstruct human videos by using computer vision techniques and retarget both motion capture and YouTube trajectories via inverse kinematics. We then train a transformer model to autoregressively predict trajectories. At test time, we execute the actions autoregressively and ignore the sensory predictions.

We demonstrate that our policy can be deployed in the real world zero-shot and walk on different surfaces. Specifically, deploy our model across a range of different locations in San Francisco over the course of one week. Please see

- Figure 1 for examples and our project page for videos. To quantitatively evaluate different aspects of our approach, we perform an extensive study in simulation. We find that our autoregressive policies trained from offline data alone are comparable to the state-of-the-art approaches that use reinforcement learning (33) in tested settings. We further find that our approach can readily benefit from incomplete trajectories and has favorable scaling properties.

These findings suggest a promising path toward learning challenging real-world robot control tasks by generative modeling of large collections of sensorimotor trajectories.

### 2. Related Work

Generative modeling. The study of data has been extensive, ranging from Shannon’s foundational work (37) to the recent era of large language models. Various such models emerged over the last decade. Notable such models includes, GAN (12) and Diffusion models (39, 16) for generating pixels, LSTM (17) and GPT (29) for generating language tokens. These models have been adopted for other modalities as well (27, 11, 43). Among these, autoregressive transformer models became the front runner, due to the impressive scaling behaviours (19) and ability to learn from in-context examples (3). This behavior is even shown to extend to other modalities such as pixels (6), languagepixels (36), and language-pixels-audio (21). We explore autoregressive generative models in the context of real-world humanoid locomotion.

Transformers in robotics. Following the success of transformer models (42) in natural language processing (29, 8, 30, 3) and computer vision (9, 13), over the last few years, there has been an increased interested in using transformer

models in robotics. We have seen several works showing that transformers can be effective with behavior cloning. For example, (38) learns multi-task transformer policies with language, and (2) trains language-conditioned manipulation policies from large-scale data. (10) trains language models with embodied data. We have also seen that transformer policies can be effective for large-scale reinforcement learning (33). (32) learns sensorimotor representations with masked prediction. (1) trains goal-conditioned policies are learned from demonstrations. Likewise, we share the goal of using transformer models for robotics but focus on autoregressive modeling of diverse trajectories for real-world humanoid locomotion.

Humanoid locomotion. Mastering the ability for robots to walk has been a long-standing challenge in robotics. In the past several decades, roboticists have built a variety of humanoid robots (20, 15, 26, 40, 7) to explore humanlike locomotion skills. Stable locomotion behaviors have been achieved through model-based control approaches (34, 18), and optimization-based methods further enable highly dynamic humanoid motions (22). Although significant progress has been made with these strategies and combining them with learning (5), learning-based approaches are gaining attention for their ability to improve and adapt to a wide range of environments. Recently, we have seen that a purely learning based approach trained with large-scale reinforcement learning in simulation can enable real-world humanoid locomotion (33). Like in prior work, our model is a causal transformer. Unlike prior work, we perform autoregressive modeling instead of reinforcement learning.

### 3. Approach

In this section, we assume that a dataset D of sensorimotor trajectories T is given and describe our approach below.

#### 3.1. Objective

Each sensorimotor trajectory is a sequence of sensory observations and actions: T = (o1,a1,o2,a2,...,oT,aT). We first tokenize the trajectory into K tokens to obtain t = (t1,t2,t3,...,tK). Our goal is to train a neural network to model the density function p(t) autoregressively:

p(t) =

K

p(tk|tk−1,...,t1) (1)

k=1

We train our model by minimizing the negative loglikelihood over our trajectory dataset:

##### L =

−log p(t) (2)

t∈D

Data Training Deployment

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Neural network policy Model based controller

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

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Motioncapture Internetvideos Transformer

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

- Figure 2: Humanoid locomotion as next token prediction. We collect a dataset on trajectories from various sources, such as from neural network policies, model-based controllers, human motion capture, and YouTube videos of humans. Then we use this dataset to train a transformer policy by autoregressive modeling of observations and actions. Our transformer allows a humanoid to walk zero-shot on various terrains around San Francisco. Please see our project page for video results.

#### 3.4. Joint training

We assume a Gaussian distribution with constant variance and train a neural network to minimize the mean squared error between the predicted and the ground truth tokens:

We have two options for training on collections that contain diverse trajectories in terms of noise levels or modality subsets. We can either train jointly with all data at once, including complete and incomplete trajectories. Alternatively, we can first pre-train on noisy and incomplete trajectories. This can be viewed as providing a good initialization for then training on complete trajectories. We find that both approaches work comparably in our setting and opt for joint training in the majority of the experiments for simplicity.

K

1 K

( tk − tk)2 (3)

L =

k=1

Instead of regressing the raw token values, we could quantizing each dimension into bins or perform vector quantization. However, we found the regression approach to work reasonably well in practice and opt for it for simplicity.

#### 3.5. Model architecture

#### 3.2. Missing modalities

Our model is a vanilla transformer (42). Given the trajectories from either complete or incomplete data, we first tokenize the trajectories into tokens. We learn separate linear projection layers for each modality but shared across time. To encode the temporal information we use positional embeddings. Let’s assume oi ∈ Rm and ai ∈ Rn, then:

In the discussion so far we have assumed that each trajectory is a sequence of observations and actions. Next, we show how our framework can be generalized to sequences with missing modalities, like trajectories extracted from human videos that do not have actions. Suppose we are given a trajectory of observations without the actions T = (o1,o2,...,oT). Our key insight is that we can treat a trajectory without actions like a regular trajectory with actions masked. Namely, we can insert mask tokens [M] to obtain T = (o1,[M],o2,[M],...,oT,[M]). This trajectory now has the same format as our regular trajectories and thus can be processed in a unified way. We ignore the loss for the predictions that correspond to the masked part of inputs. Note that this principle is not limited to actions and applies to any other modality as well.

ti = concat(oi,ai), (4) h0i = Wti, (5)

where W ∈ Rd×(m+n) is a linear projection layer to project concatenated observation and action modalities to d dimensional embedding vector. The superscript 0 indicates the embedding at 0-th layer, i.e., the input layer. When action is unavailable, we use a mask token [M] ∈ Rn to replace ai, and [M] is initialized as a random vector and learned endto-end with the whole model. The model takes the sequence of embedding vectors H0 = {h01,h02,...,h0t} as input.

#### 3.3. Aligned prediction

Rather than predicting the next token in a modality-agnostic way, we make predictions in a modality-aligned way. Namely, for each input token we predict the next token of the same modality. Please see Figure 3 for diagrams.

The transformer architecture contains L layers, each consisting of a multi-head self-attention module and an MLP module. Assume the output of the layer l is Hl, then the

Training with complete data

Training with missing data

Transformer

Transformer

M

M M M

- Figure 3: A general framework for training with different data sources. Our data modeling allows us to train our transformer with multiple modes of training. In the case of observation-action pairs being available, we train our transformer to predict the next pair of observation-action. When there is no action data available, with MoCap and internet data, we only train our transformer to predict the next observations by masking the actions with a mask token. These two models of training allow our model to utilize both types of data, and this enables us to scale our training in terms of data.

layer l + 1 output is computed as follows:

H˜l = LayerNorm(Hl) (6) H˜l = H˜l + MHSA(H˜l) (7)

Hl+1 = H˜l + MLP(H˜l) (8)

Here, the multi-head self-attention has causal masking, where the token only attends to itself and the past tokens. Once the tokens are processed through all the layers, we project the embedding to predicted states and actions, by learning a linear projection layer W ∈ R(m+n)×d:

ti+1 = WhLi (9) oi+1 = ( ti+1)0:m (10) ai+1 = ( ti+1)m:(m+n) (11)

Then we train the transformer with the objective in (3). In the cases where the token is masked, we do not apply any losses. We train our transformer with both types of data, as shown in Figure 3. This allows us to use various sources of data, thus enabling scaling in terms of data.

#### 3.6. Model inference

At inference time, our transformer model will always have access to observation-action pairs. In this setting, we apply our transformer model autoregressively for each observationaction pair token. By conditioning on past observations and actions, we predict the next actions (or observation-action pair) and execute the action. Then we take the observations from the robot and discard the predicted observations. We use the observed observation and predicted action as the next set of tokens and concatenate them with past pairs to predict the next observation-action pair.

### 4. Dataset

Our approach motivates building a dataset of trajectories for training our model. Our dataset includes trajectories from different sources: (i) neural network policies, (ii) modelbased controllers, (iii) human motion capture, and (iv) human videos from YouTube. An illustration of different data sources is shown in Figure 4. We describe each in turn next.

#### 4.1. Neural network trajectories

As the first source of training trajectories, we use a neural network policy trained with large-scale reinforcement learning (33). Specifically, this policy was trained with billions of samples from thousands of randomized environments in Isaac Gym (25). We run this policy in the Agility Robotics’ simulator and collect 10k trajectories of 10s each on flat ground, without domain randomization. Each trajectory is conditioned on a velocity command sampled from a clipped normal distribution as follows: linear velocity forward [0.0,1.0] m/s, linear velocity sideways [−0.5,0.5] m/s, and turning angular velocity [−0.5,0.5] rad/s.

Since we have access to the data generation policies, we are able to record complete observations as well as the exact actions that the model predicted. We use this set as our source of complete sensorimotor trajectories that have complete observations as well as ground truth actions.

#### 4.2. Model-based trajectories

As the second source of trajectories, we use the model-based controller developed by Agility Robotics. It is the controller that is deployed on the Digit humanoid robot and available in the Agility Robotics’ simulator as well. We collect two sets of 10k trajectories of walking on a flat ground of 10s each. In both cases, we sample the velocity commands

[Figure 41]

Neural Net Controller Model based Controller MoCap Internet Videos

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

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

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

[Figure 85]

- Figure 4: Training dataset. To train our model, we construct a dataset of trajectories coming from four different sources. (i) neural network policy: provides trajectories with complete observations and actions. (ii) model-based controller: produces trajectories without actions. (iii) motion capture of humans: does not contain actions and is approximately retargeted onto the robot. (iv) internet videos of humans: noisy human poses are first reconstructed via 3D reconstruction and then approximately retargeted onto the robot.

as follows: linear velocity forward [−1.0,1.0] m/s, linear velocity sideways [−1.0,1.0] m/s, and turning angular velocity [−1.0,1.0] rad/s. We use the default model-based configurations for one set and randomize the leg length, step clearance, and bounciness of the floor for the other.

The optimization variables include q, ˙q. For constraints, (12b) is the Euler integration of posture q, (12c) constrains the range of q and ˙q to their admissible sets Q and V. In the cost function, φtraj tracks keypoint locations from human trajectories, and φreg represents the regularization costs, such as joint velocity minimization and smoothness.

As this controller outputs joint torques, which are not consistent with our joint position action space. We only record the observations without the actions. This data serves as a source of trajectories with reasonably good observations from the same morphology but without the actions.

#### 4.4. Trajectories from YouTube videos

Internet videos of people doing various activities are potentially a vat source of data for learning human locomotion. However, the raw pixels have no information about the state and actions of the human. To recover this, we first we run a computer vision tracking algorithm PHALP (35) to extract human trajectories in 3D. This provides an estimate of the 3D joints of the human body SMPL (23) parameters and a noisy estimate of the human joints in the world coordinates. We use the human body joint positions to retarget the motion to the humanoid robot using the inverse kinematics optimization described above. Once we retarget the motion from the Internet videos to humanoid trajectories, we filter the trajectories with the low optimization cost. Note that the scale of this data comes with the cost of being noisy.

[Figure 86]

[Figure 87]

#### 4.3. Human motion capture trajectories

As the next source of trajectories, we use the motion capture (MoCap) recordings of humans from the KIT dataset (28) distributed via the AMASS repository (24). This data was recorded using optical marker-based tracking in a laboratory setting. The dataset consists of ∼4k trajectories. We use a subset of ∼1k standing, walking, and running trajectories.

In addition to not containing the ground truth actions, the MoCap trajectories come with an additional challenge: different morphology. Namely, MoCap trajectories capture human keypoint positions in 3D. In order to use these trajectories for training a robot, we solve an inverse kinematics problem to find the corresponding robot poses.

### 5. Experiments

We evaluate our approach on the challenging task of humanoid locomotion. We perform outdoor experiments on real hardware and systematic evaluations in simulation.

We formulate an inverse kinematics optimization problem:

N

φtraj[t] + φreg[t] (12a)

min

q[t],q˙[t]

#### 5.1. Experimental setup

t=1

˙q[t + 1] + ˙q[t] 2

s.t. q[t + 1] = q[t] +

dt, (12b) q ∈ Q, ˙q ∈ V (12c)

Robot platform. Digit is a humanoid robot platform developed by Agility Robotics. It is a full-sized humanoid that is 1.6m tall and weighs 45 kilograms. It has 30 degrees of freedom of which 20 are actuated. Due to its high dimensionality and four-bar linkage structure, it is challenging

where q is the robot state in the generalized coordinates, and N and dt are the optimization horizon and sampling time.

[Figure 88]

- Figure 5: Comparison to state of the art, trajectory adherence. The robot is commanded to walk starting from the origin with a fixed heading command of 0.5 m/s and varying yaw commands in [−0.4,0.4] rad/s. We plot the desired (dotted) and actual (solid) trajectories for our policy and a reinforcement-learning trained policy (RL).

[Figure 89]

[Figure 90]

Figure 6: Tracking error comparisons. We measure the tracking error of our policy against a state-of-the-art benchmark (left), as well as the improvement produced by complementing action-labeled RL trajectories with action-free trajectories (right).

to optimize fast which makes it particularly interesting for learning approaches that can learn efficiently from trajectory collections like ours.

Model. Our model has a hidden size of 192 dimensions, with 4 layers of self-attention layers and MLP layers. Each self-attention has 4 heads. We use LayerNorm before each attention layer and ReLU activation after the MLP layer. We use a BatchNorm layer to process the input before the transformer model. When predicting a token at time k, to keep the context length at a reasonable size, we only keep the past 16 steps in input. In Section 5.9, we show the model is able to scale up to more parameters and longer context length and achieve higher performance.

#### 5.2. Real-world deployment

We begin by reporting the results of deploying our policy in the real world. Specifically, we evaluate deploying our robot at various locations in San Francisco over the course of one week. Please see Figure 1 for examples and project page for videos. We find that our policy is able to walk over a variety of surfaces including walkways, concrete, asphalt, tiled plazas, and dirt roads. Note that the deployment in a large city environment, like San Francisco, is considerably more challenging than in constrained environments. The city environment is much more crowded, less patient, and not forgiving. This makes the error tolerance low and requires a policy that works consistently well.

#### 5.3. Evaluation Metrics

We evaluate locomotion policies with two metrics: tracking error and prediction error. Tracking error measures how accurately the robot follows a specific locomotion command. The prediction error is the next token prediction loss measured on a separate set of validation data. We introduce two

metrics with details as follows and show that two metrics can consistently predict locomotion performance.

Tracking error. In all experiments, the robot starts from rest in a simulated environment and is issued a constant natural walking command consisting of a desired heading velocity sampled in [0.35,0.70] m/s, angular velocity sampled in [−0.4,0.4] rad/s, and zero lateral velocity. We compute x∗(t), the ideal robot base position trajectory that fully satisfies the velocity command v∗(t) at all time steps. To measure the accuracy of command tracking, we define the position tracking error as T1 Tt=0 ∥x(t) − x∗(t)∥. We use the MuJoCo simulator (41) for evaluations, and all trajectories last for a duration of 10 seconds.

Prediction error. Since the model is trained with the next token prediction, we test the prediction error on a set of validation data that is separated from training data and contains state-action trajectories collected from the RL policy. This is similar to the language modeling evaluation for large language models (14). We test both state and action prediction errors and add them together as the final error metric.

#### 5.4. Comparison to the state of the art

Trajectory Adherence. We compare our policy to a neuralfig:tracking network controller trained with reinforcement learning (RL) (33). Figure 5 presents a visual comparison of the trajectory adherence of our controller against these state-of-the-art baselines. Starting with a robot at the origin, we plot the actual trajectory of the robot with eleven different yaw commands selected from {0.00,±0.05,±0.10,±0.20,±0.30,±0.40} rad/s. For each policy, we jointly plot the desired and actual path traced by the robot base. Our model exhibits superior tracking to the RL controller at all turning speeds, and has near-perfect tracking for straight-line walking.

Pos.TrackingError(m)

0.28 0.30 0.32 0.34 0.36 0.38

r=0.87

1.3 1.2 1.1 1.0 0.9

Prediction Loss (10−2)

Figure 7: Prediction error correlates with performance. We plot the tracking error and prediction error for 14 models. The prediction error linearly correlates with task tracking error with r = 0.87, which means lower prediction loss likely indicates more accurate command following.

Quantitative Evaluation. In Figure 6, left, we repeat the above comparison to the RL controller (N = 245), with the full range of heading and yaw velocities mentioned in Section 5.3. We plot the mean position tracking error, binned by the commanded angular yaw. While both models have lower tracking errors at lower yaw, ours consistently outperforms the baseline RL policy. This is an interesting result, since our model was trained on next token prediction on trajectories produced by this very policy.

#### 5.5. Prediction error correlates with performance

We collect 14 models trained with different training recipes, model architectures, data size and types, and test tracking error and prediction error for each one of them. We plot the tracking and prediction errors of all the models into a single scatter plot, as shown in Figure 7. We can see that tracking and prediction error are highly correlated with Pearson coefficient r = 0.87, which means models with lower prediction error on the validation set likely follow different commands with higher accuracy. This suggests that the prediction error is predictive task performance.

#### 5.6. Gait quality

In humanoid locomotion, the smoothness in the robot’s gait is contingent on the rhythmic functioning of its actuated knee joints. One way to measure this is a phase portrait, which is a parametric plot of a joint’s generalized position and velocity over time. Patterns in the plot can reveal information about the type of movement the joint is undergoing. For example, a cyclic pattern may indicate repetitive motion, while irregular patterns might suggest complex or varied movements, such as stumbling. In Figure 8, we command the robot to walk forward at 0.5 m/s, and plot the associated phase portrait of its left knee joint. Notice that our policy re-

[Figure 91]

- Figure 8: Gait quality. We command the robot with a heading velocity of 0.5 m/s and plot the resulting phase portrait of the left knee joint. Compared to the RL policy, our policy features fewer irregularities and a smoother, cyclic gait.

tains the overall shape of the RL policy while having fewer aberrations. This supports our qualitative assessment of the more regularized behavior seen on our policy.

- 5.7. Generalization to unseen commands

We find that our policy also extrapolates new skills such as walking backward, which was not included in the actionlabeled training data. As Figure 9 illustrates, by prompting our controller with negative values for the heading command, we find that the robot naturally performs backward walking at speeds up to 0.5 m/s without falling.

- 5.8. Training with action-free data

One of the benefits of our approach is that it can be applied to trajectories from diverse sources, including missing information like actions in the case of human videos from YouTube. In Figure 6, right, we compare the performance of training only with complete trajectories to joint training on both complete and incomplete trajectories. We observe that including incomplete trajectories consistently leads to better performance. This is a promising signal for scaling our approach to a large collection of diverse trajectories.

[Figure 92]

[Figure 93]

[Figure 94]

- Figure 9: Unseen commands. Our policy is able to follow backward commands at test time, unseen during training.

[Figure 95]

Figure 10: Scaling studies. We find that our approach scales with the number of trajectories in the training dataset (left), context length (middle), and larger models (right).

#### 5.9. Scaling studies

Training data. In Figure 10, left, we study the scaling of our model’s performance by increasing the size of the training dataset. We find that training on more trajectories reduces position tracking error, which is a positive signal for increased performance when training on larger datasets.

Context length. We study the effect of increasing the number of tokens used in the context window of the transformer policy, varying it between 16, 32, and 48 steps in Figure 10 middle. Larger context windows produce better policies, which suggests that our generative policy performs a form of in-context adaptation that improves with scale.

Model size. We compare models with increasing number of parameters (1M, 2M, 8M) by varying the embedding dimension (144, 192, 384), number of attention heads (3, 4, 12), and number of transformer blocks (4, 4, 6) respectively. Tracking error monotonically decreases with model size.

#### 5.10. Ablation studies

Concatenated vs. separate tokens. For the input of transformer, we can either concatenate observation and action at each step into a single token, or embed them into two separate tokens. We compare these two choices in Table 1a. We can see that concatenation has lower prediction error while separating tokens has lower tracking error. Overall these two perform comparably while using separate tokens doubles the input length and introduces computation overhead.

Modality-aligned vs. non-aligned prediction. When we use separate tokens for observation and actions as input, we can either predict oi+1 from oi and ai+1 from ai, which aligns modality between prediction and input, or we can predict oi+1 from ai and ai+1 from oi+1, which does not have alignment. From Table 1b, we can see that modality alignment has clearly better performance than no alignment. We suspect this is because, at t-th step during inference,

when predicting action of (t + 1)-th step, since there is no alignment, we need to first predict oi+1 and use this prediction as input to predict ai+1. If the predicted oi+1 is not accurate compared to real oi+1 (which is used to predict ai+1 during training), there will be a discrepancy between test and training data which will cause error in action prediction.

Joint training vs. staged training. Given both complete data with action and incomplete data without action, we can either jointly train on both data as described in Section 3, or we can first pre-train the model on all the data with state prediction only, then fine-tune the model on complete data with action prediction. We compare these two approaches in Table 1c. We observe no significant difference between these two, which indicates that pre-training on state prediction then fine-tuning on action prediction also gives a reasonable locomotion policy.

State-action prediction vs. action-only prediction. We compare the performance of our policy when trained with only predicting actions, versus when trained with predicting both states and actions. The results in Table 1d show that the state-action prediction improves model performance on trajectory tracking. We hypothesize that the additional learning signal enables the model to learn richer representations of the world that are beneficial for the locomotion task.

### 6. Discussion

We present a self-supervised approach for real-world humanoid locomotion. Our model is trained on a collection of sensorimotor trajectories, which come from prior neural network policies, model-based controllers, human motion capture, and YouTube videos of humans. We show that our model enables a full-sized humanoid to walk in the realworld zero-shot. These findings suggest a promising path toward learning challenging real-world robot control tasks by generative modeling of large collections of trajectories.

Track Err. Pred. Err.

Track Err. Pred. Err.

Concat 0.310 0.88 Separate 0.299 0.98

Align 0.299 0.98 Non-align 0.338 1.05

(a) Concatenated vs. separate tokens for states and action. Two modeling designs have comparable performance while concatenating state and action gives shorter input length and faster inference.

(b) Alignment vs. non-alignment of states or actions for next token prediction. Prediction with aligned modality performs better on bothe tracking error and next token prediction error.

Track Err. Pred. Err.

Joint training 0.310 0.88 Staged training 0.311 -

(c) Joint vs. staged training on data with and without actions. Staged training which pre-trains on state prediction and finetunes on action prediction has similar performance as joint training.

Track Err. Pred. Err.

State-action 0.305 0.97 Action-only 0.335 -

(d) State-action vs. action-only prediction. Predicting both states and actions leads to lower tracking error than only predicting action as in vanilla behavior cloning.

Table 1: Ablations on different design choices in modeling and training. For each ablation we compare the average tracking error on a set of commands, as well as the next token prediction error on the test set. For a fair comparison, we do not report next token prediction error for models that only predict actions.

### Acknowledgements

This work was supported in part by DARPA Machine Common Sense program, ONR MURI program (N00014-21-12801), NVIDIA, Hong Kong Centre for Logistics Robotics, The AI Institute, and BAIR’s industrial alliance programs. We thank Saner Cakir and Vikas Ummadisetty for help with the inverse kinematics simulation experiments.

### References

- [1] Bousmalis, K., Vezzani, G., Rao, D., Devin, C., Lee, A. X., Bauza, M., Davchev, T., Zhou, Y., Gupta, A., Raju, A., et al. Robocat: A self-improving foundation agent for robotic manipulation. arXiv:2306.11706, 2023.
- [2] Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., et al. Rt-1: Robotics transformer for real-world control at scale. arXiv:2212.06817, 2022.
- [3] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. In NeurIPS, 2020.
- [4] Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. NeurIPS, 2020.

- [5] Castillo, G. A., Weng, B., Zhang, W., and Hereid, A. Robust feedback motion policy design using reinforcement learning on a 3d digit bipedal robot. In IROS, 2021.
- [6] Chen, M., Radford, A., Child, R., Wu, J., Jun, H., Luan, D., and Sutskever, I. Generative pretraining from pixels. In ICML, 2020.
- [7] Chignoli, M., Kim, D., Stanger-Jones, E., and Kim, S. The mit humanoid robot: Design, motion planning, and control for acrobatic behaviors. In Humanoids, 2021.
- [8] Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HCT, 2019.
- [9] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2020.
- [10] Driess, D., Xia, F., Sajjadi, M. S., Lynch, C., Chowdhery, A., Ichter, B., Wahid, A., Tompson, J., Vuong, Q., Yu, T., et al. Palm-e: An embodied multimodal language model. arXiv:2303.03378, 2023.
- [11] Engel, J., Agrawal, K. K., Chen, S., Gulrajani, I., Donahue, C., and Roberts, A. Gansynth: Adversarial neural audio synthesis. arXiv:1902.08710, 2019.

- [12] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. Generative adversarial nets. In NeurIPS, 2014.
- [13] He, K., Chen, X., Xie, S., Li, Y., Doll´ar, P., and Girshick, R. Masked autoencoders are scalable vision learners. arXiv:2111.06377, 2021.
- [14] Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.
- [15] Hirai, K., Hirose, M., Haikawa, Y., and Takenaka, T. The development of honda humanoid robot. In ICRA, 1998.
- [16] Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In NeurIPS, 2020.
- [17] Hochreiter, S. and Schmidhuber, J. Long short-term memory. Neural computation, 1997.
- [18] Kajita, S., Kanehiro, F., Kaneko, K., Yokoi, K., and Hirukawa, H. The 3d linear inverted pendulum mode: A simple modeling for a biped walking pattern generation. In IROS, 2001.
- [19] Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv:2001.08361, 2020.
- [20] Kato, I. Development of wabot 1. Biomechanism, 1973.
- [21] Kondratyuk, D., Yu, L., Gu, X., Lezama, J., Huang, J., Hornung, R., Adam, H., Akbari, H., Alon, Y., Birodkar, V., et al. Videopoet: A large language model for zero-shot video generation. arXiv:2312.14125, 2023.
- [22] Kuindersma, S. Recent progress on atlas, the world’s most dynamic humanoid robot, 2020. URL https: //youtu.be/EGABAx52GKI.
- [23] Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., and Black, M. J. Smpl: A skinned multi-person linear model. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, 2023.
- [24] Mahmood, N., Ghorbani, N., Troje, N. F., Pons-Moll, G., and Black, M. J. AMASS: Archive of motion capture as surface shapes. In ICCV, 2019.
- [25] Makoviychuk, V., Wawrzyniak, L., Guo, Y., Lu, M., Storey, K., Macklin, M., Hoeller, D., Rudin, N., Allshire, A., Handa, A., et al. Isaac gym: High performance gpu-based physics simulation for robot learning. In NeurIPS, 2021.

- [26] Nelson, G., Saunders, A., Neville, N., Swilling, B., Bondaryk, J., Billings, D., Lee, C., Playter, R., and Raibert, M. Petman: A humanoid robot for testing chemical protective clothing. Journal of the Robotics Society of Japan, 2012.
- [27] Oord, A. v. d., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A., and Kavukcuoglu, K. Wavenet: A generative model for raw audio. arXiv:1609.03499, 2016.
- [28] Plappert, M., Mandery, C., and Asfour, T. The KIT motion-language dataset. Big Data, 2016.
- [29] Radford, A., Narasimhan, K., Salimans, T., and Sutskever, I. Improving language understanding by generative pre-training. 2018.
- [30] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. 2019.
- [31] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In ICML, 2021.
- [32] Radosavovic, I., Shi, B., Fu, L., Goldberg, K., Darrell, T., and Malik, J. Robot learning with sensorimotor pre-training. In CoRL, 2023.
- [33] Radosavovic, I., Xiao, T., Zhang, B., Darrell, T., Malik, J., and Sreenath, K. Real-world humanoid locomotion with reinforcement learning. arXiv:2303.03381, 2023.
- [34] Raibert, M. H. Legged robots that balance. MIT press, 1986.
- [35] Rajasegaran, J., Pavlakos, G., Kanazawa, A., and Malik, J. Tracking people by predicting 3d appearance, location and pose. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2740–2749, 2022.
- [36] Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., and Sutskever, I. Zero-shot text-to-image generation. In ICML, 2021.
- [37] Shannon, C. E. Prediction and entropy of printed english. Bell system technical journal, 1951.
- [38] Shridhar, M., Manuelli, L., and Fox, D. Perceiveractor: A multi-task transformer for robotic manipulation. In CoRL, 2022.
- [39] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015.

- [40] Stasse, O., Flayols, T., Budhiraja, R., Giraud-Esclasse, K., Carpentier, J., Mirabel, J., Del Prete, A., Sou`eres, P., Mansard, N., Lamiraux, F., et al. Talos: A new humanoid research platform targeted for industrial applications. In Humanoids, 2017.
- [41] Todorov, E., Erez, T., and Tassa, Y. Mujoco: A physics engine for model-based control. In IROS, 2012.
- [42] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In NeurIPS, 2017.
- [43] Wu, J., Zhang, C., Xue, T., Freeman, B., and Tenenbaum, J. Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling. In NeurIPS, 2016.

