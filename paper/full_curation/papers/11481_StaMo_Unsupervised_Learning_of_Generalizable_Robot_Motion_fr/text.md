## arXiv:2510.05057v2[cs.RO]12Apr2026

[Figure 1]

### StaMo: Unsupervised Learning of Generalizable Robot Motion from Compact State Representation

Mingyu Liu1,2,∗ Jiuhe Shu1,∗ Hui Chen1 Zeju Li1 Canyu Zhao1 Jiange Yang3 Shenyuan Gao4 Hao Chen1,† Chunhua Shen1,5,† 1State Key Laboratory of CAD & CG, Zhejiang University 2Shanghai Innovation Institute 3Nanjing University 4HKUST 5Ant Group

mingyuliu@zju.edu.cn, chunhuashen@zju.edu.cn

#### Abstract

A fundamental challenge in embodied intelligence is learning state representations that are both compact and expressive for world modeling and decision making, yet existing methods often remain either redundant or miss task-critical information. We propose an unsupervised approach that learns a highly compressed two-token state representation using a lightweight encoder and a pre-trained Diffusion Transformer (DiT) decoder, capitalizing on its strong generative prior. Our representation is efficient, interpretable, and integrates seamlessly into existing VLA-based models, improving performance by 11.6% on LIBERO and 31% in real-world task success rate with minimal inference overhead. More importantly, we find that the difference between these tokens, obtained via latent interpolation, naturally represents the motion, which can be further decoded into executable robot actions. This emergent capability reveals that our representation captures dynamics without explicit motion supervision. We name our method StaMo for its ability to learn generalizable robotic Motion from compact State representation, which is encoded from static images, challenging the prevalent dependence on learning robotic motions with complex temporal modeling and video data. Our learned representations also enhance policy co-training, outperforming prior methods by 10.4% with improved interpretability. Moreover, our approach scales effectively across diverse data sources, including real-world robot data, simulation, and human egocentric video. Project page at https://aim-uofa.github.io/StaMo/.

#### 1. Introduction

“What we observe as static is merely dynamic equilibrium.”

— Richard Feynman, The Feynman Lectures on Physics

∗Equal contribution. †Corresponding authors.

Learning reusable and generalizable representations is a cornerstone of intelligent robotics systems. While visual representations in VLAs retain rich perceptual details for multimodal fusion, state representations for world modeling and intermediate reasoning serve a different purpose: they are designed to enable efficient future prediction and bridging visual planning with action execution. This role demands two key properties: First, it must be highly compact. Unlike rich visual embeddings used for perception, a state representation’s proximity to action generation demands a focus on actionable information to ensure efficient future prediction. Second, it must be expressive despite its compactness. Common low-dimensional approaches, such as trajectories [47] or flow fields [19, 50], often fail on this front; they capture basic motion but lack the semantic richness to encode goal states, interaction dynamics, or structured spatial relationships.

Building on these principles, we propose a compact state representation learned using a Diffusion Autoencoder finetuned on robotics data. Our method compresses an image into a highly compact latent state (as few as 2 tokens of 1024 dimensions). To ensure the representation is rich, we initialize its decoder from a powerful DiT model pretrained on internet data [18], reasoning that the ability to reconstruct pixels accurately requires an implicit understanding of key state information like robot poses and object interactions. This representation integrates seamlessly into existing VLM architectures, extensive experiments show that our method significantly enhances model performance with negligible overhead to inference speed, achieving improvements of + 11.6% in LIBERO and + 31% in real-world deployments.

Interestingly, from this compact state representation, we make an important discovery: robot motion (which is also known as latent action) naturally emerges from the state representation space. By simply performing a linear interpolation between the latent encodings of start and goal

[Figure 2]

[Figure 3]

[Figure 4]

Static Compressor Training Motion Interpolating

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

VAE

Image Space

[Figure 13]

DiT

T

Latent DINOv2

Compact Latent Space

VAE

[Figure 14]

Compressor

[Figure 15]

+Noise

[Figure 16]

|DiT|
|---|

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

# StaMo

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Long-Context

[Figure 36]

|[Figure 37]|
|---|

[Figure 38]

[Figure 39]

Decoder-Free

|[Figure 40]|
|---|

[Figure 41]

###### Action

High-Efficiency

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

###### MLLM

Real World

Ego-Centric

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

|[Figure 52]|
|---|

[Figure 53]

[Figure 54]

Compressor Encoder

[Figure 55]

[Figure 56]

[Figure 57]

”Put the milk into the Basket”

[Figure 58]

DINOv2

[Figure 59]

Simulation Scalable Training

Efficient World Modeling

Figure 1. An overview of our StaMo framework. Our method efficiently compresses and encodes robotic visual representations, enabling the learning of a compact state representation. Motion naturally emerges as the difference between these states in the highly compressed token space. This approach facilitates efficient world modeling and demonstrates strong generalization, with the potential to scale up with more data.

a sufficiently expressive state representation from individual frames such that the simple difference between two states naturally encapsulates a meaningful latent action? Our work demonstrates this is not only feasible but also highly advantageous. Compared to traditional videobased models, our state-centric approach is more trainingefficient and avoids the representational ambiguity caused by motion variations in video. The emergent latent actions exhibit strong generalization and seamless sim-to-real transferability. We validate this effectiveness through extensive co-training experiments, where our latent actions achieve superior performance over traditional and more complex latent action learning approaches.

observations, we can generate smooth, plausible, and dynamically consistent motion trajectories. This observation provides an alternative to the dominant paradigm of learning latent actions from video data [11, 15, 52, 53]. While using video seems intuitive as action is inherently temporal, it presents significant challenges: it demands complex, computationally expensive temporal models and often produces ambiguous, coarse-grained actions. This is because high-variance motion within video clips causes models to learn averaged-out representations, making per-frame learning both inefficient and representationally flawed.

These challenges motivated us to revisit the fundamental motivation for using video data, if the objective is to capture latent actions through frame-wise changes, why should we commit to learning complex motion extractors from suboptimal state representations? We therefore raise a more fundamental question: rather than explicitly modeling motion from sequences, can we learn

The captivating properties of state and motion ultimately form the foundation of our method, dubbed StaMo, which learns generalizable Motion from compact State representation. While the state is static, the motion is dynamic, a harmonious balance is elegantly achieved in our approach. We

hope our method would shed new light on future research. Our contributions are summarized as follows:

capabilities. Inspired by recent unified generative and comprehension models, a promising direction is to enable large models to reason with images—predicting future visual states while inferring actions or language, thereby creating a mutually reinforcing process. Several works [12, 27, 45, 55] have explored this idea. However, these approaches either require decoding full images during inference or rely on overly redundant state representations, limiting their generalization to novel scenes. In contrast, our method employs a more compact representation that improves model performance with minimal impact on inference speed. Furthermore, our representation exhibits strong generalization ability, enabling adaptation to unseen scenes without requiring further fine-tuning of the encoder.

- • We propose StaMo, a novel framework that encodes a compact state representation from static images, from which motion naturally emerges.
- • Our representation can be efficiently utilized for world modeling and serves as an intermediate representation that bridges vision-language models (VLMs) and action expert modules. It can be seamlessly integrated into existing VLM-based frameworks, delivering improved performance with minimal inference overhead.
- • Our motion extraction approach offers enhanced flexibility, strong generalization capability, and excellent transferability. It can be effectively utilized for co-training downstream models and facilitating goal-image planning tasks.
- • Comprehensive simulation and real-world experiments, along with extensive visualizations, demonstrate the effectiveness of our approach, which can be readily scaled up with additional data.

##### 2.3. Latent Action Learning

Robot learning scalability is constrained by scarce and heterogeneous robotic data, prompting the use of large-scale, action-free internet videos to instill foundational physical and operational knowledge for improved generalization and data efficiency [33]. Numerous studies have explored learning discrete [9–11, 15, 41, 53] or continuous [20, 52] latent actions from unlabeled data and have demonstrated their effectiveness across a wide range of downstream tasks. However, most of these approaches rely on carefully designed model architectures and depend on extracting frames from continuous videos, which introduces limitations such as sensitivity to frame sampling intervals, potential temporal biases, and poorly interpretable, blurry action representations. In contrast, our approach reveals that actionable representations can inherently emerge from large-scale natural image collections—challenging the conventional belief that latent actions must be learned from video sequences. We further demonstrate that such image-emergent actions exhibit superior interpretability and stronger generalization capabilities.

#### 2. Related Works

##### 2.1. Robot Representation Learning

Previous robotics research on visual representations has often faced a fundamental trade-off: methods excelling at motion representation, like latent actions [6, 11, 14, 20, 53], flow [19, 51], or trajectories [47], typically lack a rich, compact state representation. Conversely, approaches that encode detailed state information from raw images [5, 45] or dense features [26, 32, 34, 40, 49, 55] are often too highdimensional and redundant to effectively represent motion via simple differences. StaMo, as shown in Figure 2, overcomes this dichotomy by achieving a unique balance. It learns to compress and encode a robot’s visual state into a highly compact token space that is both expressive enough for complex tasks and minimal enough that motion can be elegantly and powerfully represented as the difference between two states. This unified approach not only enables efficient world modeling but also demonstrates superior generalization and scalability, providing a robust foundation for future robotic systems.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

###### Where is StaMo？

[Figure 65]

[Figure 66]

HighCompactnessLow

State Motion

[Figure 67]

[Figure 68]

[Figure 69]

Action

[Figure 70]

State Motion

Compressor Compressor

[Figure 71]

[Figure 72]

[Figure 73]

Frame N Frame M

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

##### 2.2. World Modeling in VLA

Motion Extractor

StaMo

State Motion

[Figure 79]

Latent Action Encoder Encoder

[Figure 80]

Benefiting from powerful vision-language models (VLMs) [3, 4, 16] pre-trained on large-scale internet data, vision-language-action (VLA) models have demonstrated significant potential in generalizing across manipulation tasks. Typically, VLAs [7, 8, 13, 22, 24, 30, 42, 43, 46, 48, 58] map visual information and language instructions into the robot’s action space through end-to-end training. A natural extension is to endow the model with world modeling

Flow/Trajectory /OpticFlow

Dense Visual Feature

Image

[Figure 81]

Low High

Expressivity

Figure 2. Where is StaMo? This figure visualizes how different robotic representations fall on the spectrum of expressiveness versus compactness. StaMo uniquely occupies the ideal position, offering both a rich, expressive state representation and the ability to model motion from a highly compact space.

#### 3. Method

##### 3.1. Compress Image into Compact State

A straightforward approach to state representation is to employ pretrained image encoders to process observations, using their output features as latent states. However, this strategy yields prohibitively large feature maps (e.g., 256×1024), which contain significant redundant information and impose limitations on real-time execution and longhorizon planning in robotics. While using the single [CLS] token from the large feature maps substantially reduces dimensionality, this representation is often too coarse to facilitate effective and precise task execution. To resolve this trade-off between dimensionality and representational fidelity, we employ a Diffusion Autoencoder. We fine-tune this architecture specifically for robotic settings, enabling it to act as a powerful state compressor. This approach successfully compresses the high-dimensional observation into a highly compact latent state of as few as two 1024dimensional tokens, capturing essential information while remaining computationally efficient.

To this end, as illustrated in Figure 1, the encoder of our Diffusion Autoencoder, denoted as E, consists of a pretrained DINOv2 model [35] serving as a frozen feature extractor, which is followed by a transformer-based compressor, to map the input observations into a sequence of compact token representations. The decoder D, is a Diffusion Transformer (DiT) [37] conditioned on these tokens to reconstruct the original image observations. Our implementation is built upon the Stable Diffusion 3, where only the transformer-based compressor and the DiT decoder are trained. The DINOv2 encoder remains frozen throughout the training process. We employ the same Flow Matching [2, 28, 31] objective used in base model training to optimize the model, which minimizes the loss between the velocity predicted by D and the target velocity u:

z0 = τ(x0), LDAE = Ez

(1)

0,t∥D(zt,E(x0),t) − u(zt)∥22,

where x0 is the input image observation and τ is the VAE encoder of pretrained diffusion model to produce latent z0. zt = (1 − σt)z0 + σtϵ is the linear interpolation between pure noise ϵ and z0. After training, The trained encoder E is able to yield the state representation s.

##### 3.2. Interpolating Motion from States

A key advantage of our learned representation is its capacity to serve a dual role, representing both state and motion. This duality resolves a fundamental trade-off inherent in conventional robotics representations, which we analyze comparatively in Figure 2. On one hand, traditional motion representations such as end-effector poses, optical flow, trajectories, or recent latent action models, are typically low-

dimensional and compact. Their strength lies in capturing dynamics, often through simple differencing, making them effective for representing actions. However, this compactness comes at the cost of expressivity: they lack the rich visual context necessary to reconstruct a plausible or detailed state. On the other hand, prevailing state representations, derived from encoding raw image observations or supplementary inputs like depth and segmentation maps, capture rich semantic and geometric details. While highly expressive, these high-dimensional features are computationally burdensome and fail to intrinsically encode the dynamics or motion required to transition between states. Our approach elegantly bridges this gap. By defining motion as the vector difference between consecutive compact state tokens at = st+1 − st, we achieve a representation that is both compact and expressive, unifying the description of “what the world looks like” and “how the world changes”. The effectiveness of representing motions as the differences between states is demonstrated in our experiments in Section 4.6.

##### 3.3. StaMo for Efficient World Modeling

Existing VLA models, such as our OpenVLA baseline, typically operate as reactive policies. They learn a direct mapping from the current visuolinguistic context to a sequence of low-level actions (e.g., 7-DoF end-effector controls). While effective, this paradigm does not explicitly compel the agent to reason about the physical consequences of its actions or how the world will change as a result. We hypothesize that endowing the agent with a predictive world model to anticipate future visual states will improve the future-planning capability of model. This auxiliary task of predicting “what happens next” should, in turn, regularize the policy and improve the quality of the primary action prediction task.

The compact state representations produced by our StaMo encoder E are ideally suited for this purpose, which provide a concise yet rich summary of the environment for the model to reason about the future. To this end, we integrate the compact representations of StaMo into the OpenVLA architecture and train the model to jointly predict the next state and the corresponding action. Technically, we achieve this by attaching a lightweight MLP head to OpenVLA’s autoregressive backbone, which is tasked specifically with predicting the subsequent state representation. The model is optimized with a composite loss function that balances action generation and future-state prediction:

Ltotal = λactionLaction + λfuture(Lmse(spred, sgt)

+ L1(spred, sgt)), (2)

where Laction is the standard cross-entropy loss for nexttoken prediction, anchoring the model to its primary control task. The world model objective combines Mean Squared Error (MSE) and L1 losses to enforce accurate regression

of the ground-truth future state. In our experiments, we set λaction = λfuture, reflecting our hypothesis that learning to predict is as crucial as learning to act. Our experimental results validate this hypothesis, demonstrating that compelling the VLA to predict future states significantly improves task success rates, as detailed in Section 4.2.

latent space, the decoded images exhibit excellent continuity and plausible motion. Furthermore, we demonstrate the results of motion transfer. By taking the difference between tokens in the latent space, the resulting latent motion proves to be highly transferable—both in sim-to-sim, sim-to-real, and real-to-sim scenarios. This indicates that the learned motion is not scene-specific but possesses strong generalization capabilities.

##### 3.4. StaMo for Latent Motions Co-Training

While Section 4.1 provides a qualitative visualization of the emergent motion, the abstract nature of our latent motion representation makes direct quantitative comparison challenging. To rigorously assess its effectiveness, we employ a policy co-training strategy. This approach evaluates the utility of our motion representation on downstream tasks by jointly training a policy model on a small set of actionlabeled robot data alongside a larger set of action-less video data. We infer a latent motion, mt, for each pair of consecutive video frames by calculating the difference between their compact state representations encoded by our model E(·) : mt = E(ot+1) − E(ot), This process generates pseudo-action labels for the video data, creating a unified dataset where our emergent motions and ground-truth robot actions are learned jointly within a single policy model. This approach unlocks the potential to learn from vast and diverse video sources without requiring explicit action labels.

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Reconstruction

PSNR: 27.83 SSIM: 0.9183 PSNR: 27.79 SSIM: 0.9261 PSNR: 21.96 SSIM: 0.8504 PSNR: 23.444 SSIM: 0.8508

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

RealSimulation

MotionTransferMotionInterpolation

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

Start Frame 1 2 3 4 5 6 End Frame

[Figure 106]

[Figure 107]

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

Sim2Sim Sim2Real Real2Sim

Figure 3. Reconstruct images using our StaMo encoder with as few as two 1024-dimensional tokens. The first row shows the ground truth, and the second row shows the predicted results, with corresponding PSNR and SSIM metrics listed below. The results demonstrate that StaMo can preserve high image fidelity and structural similarity even under extremely compressed state representations.

We tested this co-training framework in Section 4.5, our method leads to substantial performance gains compared with previous method. The results confirm that the motions emerging from StaMo serve as an effective and robust proxy for true actions, providing a scalable pathway to enhance policy learning by leveraging the wealth of unlabeled visual data.

##### 4.2. World Modeling Results

In this section, we evaluate the application of our StaMo representation for efficient world modeling. A key advantage of our approach is its ability to integrate seamlessly into existing VLM-based frameworks. We demonstrate this by conducting experiments on two strong baselines: OpenVLA and its successor, OpenVLA-OFT. We specifically chose OpenVLA-OFT because it introduces architectural improvements like parallel decoding and action chunking, which result in a significantly longer action horizon. This requires the model to predict a state representation at a more distant future step, making it a more challenging and relevant benchmark for world modeling.

#### 4. Experiments

##### 4.1. Qualitative Analysis of StaMo

StaMo demonstrates the ability to perform high-quality reconstructions of robotic manipulation images with strong generalization capabilities. It achieves excellent results on both in-domain and out-of-domain data. Quantitative reconstruction results are shown in Table 1, and some qualitative results are presented in Figure 3. More results can be found in Appendix. For main experiments, we use DROID [23] and LIBERO [29] as our training data.

To validate our approach, we retrained both OpenVLA and OpenVLA-OFT, replacing their original vision encoders with our StaMo encoder and training them to predict StaMo states. We denote these modified architectures as OpenVLA* and OpenVLA-OFT*; the original, unmodified models are presented without the asterisk. For our OpenVLA-OFT* variant, we introduced a minor architectural modification, using only the third-person camera view and removing the wrist camera input. We evaluate the performance of our StaMo representation on the LIBERO benchmark to demonstrate its superior capability in assist-

Table 1. Reconstruct Performance comparison of different datasets using our StaMo encoder.

Dataset libero 10 libero 90 libero goal libero object libero spatial Droid Maniskill(OOD)

PSNR (dB) 25.5194 27.2470 24.6467 27.0011 25.9683 20.2492 22.1673 SSIM 0.8909 0.8962 0.8926 0.9105 0.8984 0.7346 0.8824

Figure 3 also showcases StaMo’s powerful motion interpolation ability. By linearly interpolating the tokens in the

ing VLA manipulation through enhanced world modeling. For each task, we conduct 1,000 evaluation rollouts, and all results are presented as percentage success rates.

- Table 2. Performance comparison of different methods (%).

Method Spatial Object Goal Long Average

DP [17] 78.3% 92.5% 68.3% 50.5% 72.4% Octo-Base [42] 78.9% 85.7% 84.6% 51.1% 75.1% Spatial-VLA [39] 88.2% 89.9% 78.6% 55.5% 78.1% CoT-VLA [56] 87.5% 91.6% 87.6% 69.0% 83.9% π0-FAST [38] 96.4% 96.8% 88.6% 60.2% 85.5% WorldVLA [12] 87.6% 96.2% 83.4% 60.0% 81.8% UniVLA [45] 95.4% 98.8% 93.6% 94.0% 96.5%

OpenVLA [24] 84.7% 88.4% 79.2% 53.7% 76.5% OpenVLA* + DINOv2 Feature 88.6% 90.4% 83.5% 61.4% 80.9% OpenVLA* + StaMo state 92.3% 94.8% 88.1% 75.2% 87.6% OpenVLA* + StaMo motion 93.1% 95.1% 87.4% 76.9% 88.1%

OpenVLA-OFT [25] 93.7% 94.2% 89.7% 91.3% 92.2% OpenVLA-OFT* + DINOv2 Feature 94.1% 95.0% 91.2% 93.0% 93.3% OpenVLA-OFT* + StaMo state 96.8% 98.9% 95.0% 96.3% 96.8% OpenVLA-OFT* + StaMo motion 95.3% 97.1% 94.1% 92.9% 94.9%

Both the StaMo state and motion representations can serve as predictive targets for a world model, and their performance is detailed in Table 2. Our findings indicate that the optimal choice of representation depends on the policy’s action horizon. For the standard OpenVLA, which predicts actions over a short, single-step interval, the motion representation yields better performance. This is because the latent motion is conceptually analogous to the delta end-effector (EE) pose used in frameworks like LIBERO, providing a direct and incremental target. Conversely, for OpenVLA-OFT, which plans over a longer execution step, the state representation is recommended, as it acts as a stable goal-conditioning signal.

4.3. Inference Speed Comparison

A key advantage of StaMo lies in its utilization of two highly compressed tokens to assist VLM-based models with world modeling. During the inference stage, it obviates the need to decode full images; the model only needs to predict the tokens themselves. This approach significantly enhances the efficiency of the world modeling process. We conducted a comparative analysis of the model’s inference frequency before and after integrating the StaMo representation, and also benchmarked its speed against other world modeling methods. Our findings indicate that StaMo introduces significantly less inference overhead. The comparison of different model’s inference frequency is presented in

- Table 3.
- 4.4. Zero-Shot Transfer to New Environment

Due to the compact nature of the StaMo representation, which focuses on manipulation-relevant information rather than extraneous visual details, our model exhibits superior generalization capabilities. We validated this in Table 4 by

Table 3. Inference Speed Comparison.The table presents a comparison of inference speeds for different methods, with all models deployed on a single RTX 4090 GPU. We benchmark against other world model-based VLA approaches, such as UniVLA [45] and WorldVLA [12]. As the results show, our method achieves a higher inference frequency compared to previous methods, including directly using DINOv2 features.Our approach introduces minimal computational overhead compared to the original baseline model.

Method Frequency (Hz) ↑

WorldVLA [12] 2.27 UniVLA [45] 2.65 OpenVLA [24] 4.16 OpenVLA + DINOv2 Feature 2.86 OpenVLA + StaMo 4.02 OpenVLA-OFT [25] 18.24 OpenVLA-OFT + DINOv2 Feature 11.42 OpenVLA-OFT + StaMo 17.82

Table 4. Zero-Shot Performance on Maniskill [21].

Method Pick Cube Push Cube Stack Cube Pull Cube Tool

OpenVLA [24] 28% 4% 18% 16% UniVLA [45] 32% 7% 22% 20% WorldVLA [12] 30% 10% 17% 22% OpenVLA + StaMo state 41% 32% 48% 36%

taking the model trained on the LIBERO benchmark and applying it directly to the ManiSkill [21] environment. Crucially, all methods were deployed in this zero-shot transfer setting. We evaluated each task over 100 trials, with results presented as percentage success rates. The outcomes demonstrate that our approach achieves stronger performance than competing models under these conditions.

##### 4.5. Policy Co-training Results

In Table 5, we detail the results of our policy co-training. Our approach utilizes a DIT-based model on the RDT architecture, trained on a per-task dataset of 10 robot trajectories and 40 video trajectories with StaMo-generated pseudo-actions. We evaluated the final model checkpoint for each task across 20 trials, averaging the success rate over three full runs. StaMo significantly outperforms both LAPA [53] and ATM [47], suggesting that its pseudo-action labels are more faithful to the original actions.

Table 5. Performance comparison of RDT with different data configurations (%).

Method Spatial Object Goal Long Average

RDT (all Real) 91.6% 93.3% 86.7% 73.3% 86.2% RDT (1 Real) 71.7% 70.0% 66.7% 43.3% 62.9% RDT (1Real+ 4ATM [47]) 83.3% 81.7% 71.7% 56.7% 73.4% RDT (1Real+ 4LAPA [53]) 80.0% 76.7% 75.0% 65.0% 74.2% RDT (1Real + 4StaMo) 90.0% 91.6% 86.7% 70.0% 84.6%

##### 4.6. Action Linear Probing Results

A central claim of our work is that an effective latent action can be formulated as the simple vector difference between two latent states encoded by StaMo. To quantitatively validate the quality and utility of these emergent latent actions, we employ a linear probing protocol [1], a standard methodology for evaluating the efficacy of learned representations, to measure how much explicit, predictable information about the motion is contained within our latent action formulation.

We first construct a dataset by randomly sampling tuples from a large collection of robot trajectories. Each sampled tuple is of the form (In,In+k,An), where In is a starting image, In+k is the goal image after a horizon of k steps, and An = (an,...,an+k−1) is the sequence of ground-truth robot actions executed between them. Each image pair is then encoder into latent action, ∆z, using our frozen StaMo encoder, E:

∆z = E (In+k) − E (In). (3)

We then train a lightweight Multi-Layer Perceptron (MLP) to predict the action sequence An from the latent action representation ∆z. The performance is measured by the Mean Squared Error (MSE) between the predicted action sequence Aˆ n and the ground-truth sequence An. This process can be formally described as follows.

MSE An,Aˆn = MSE(An,MLP(∆z)). (4)

A low LP-MSE score provides strong quantitative evidence that the vector difference between our latent states serves as a highly informative and linearly separable representation for robotic motions.

[Figure 118]

- Figure 4. Linear Probing MSE results. We compare our method against three baselines. Our method consistently achieves the lowest MSE across all horizons.

To validate our approach, we benchmark our latent action representation against several baselines. The first set of baselines tests the importance of our structured latent space against alternatives operating on raw data: Pooled Delta Image (using pixel-wise differences) and Delta DINOv2 Features (using feature-space differences). The sec-

ond baseline, LAPA [54], contrasts our deterministic statedifference formulation with a state-of-the-art autoregressive model that generates latent action tokens.

For a fair comparison, all representations are used to train an identical lightweight MLP. We evaluate nongenerative methods across 1, 2, 4, and 8-step prediction horizons, while LAPA is assessed at its native 1-step horizon. Once trained, the MLP can directly map a current observation and a goal image to an executable action trajectory on the test set. The results in Figure 4 confirm the superiority of our method.

##### 4.7. Goal-Conditioned Task Planning

Once the linear head is trained, it becomes capable of translating abstract representations into concrete, executable actions. We evaluated the goal-conditioned task planning capabilities of StaMo and DINO-WM [57] using the ManiSkill benchmark. ManiSkill was specifically selected because it was not included in the pre-training datasets for either StaMo or DINO-WM. For this comparison, we trained only the linear head while keeping the encoders of both StaMo and DINO-WM frozen. Table 6 presents the performance comparison, and further implementation details are available in the appendix. We evaluated each task over 100 trials, with results presented as percentage success rates.

Table 6. Goal-Conditioned Task Planning.

Method Pick Cube Push Cube Stack Cube Pull Cube Tool

DINO-WM [57] 12% 8% 4% 18% StaMo 22% 32% 16% 38%

##### 4.8. Real World Experiments

Experiment Setup Our real-world experimental benchmark comprises six manipulation tasks (three short-horizon and three long-horizon) to comprehensively evaluate the effectiveness of our StaMo representation in facilitating world modeling and decision-making across a spectrum of task complexities, as shown in Figure 5. For each task, we collected 50 human demonstrations and evaluated performance over 20 trials. The results, presented as percentage success rates, are shown in Table 7.

##### 4.9. Ablation Study 4.9.1. Ablation of Token Dimension

We ablate the hidden dimension of StaMo’s 2-token latent state (trained on LIBERO datasets, evaluated via PSNR/SSIM). As shown in Table 8, across 256/512/1024 dimensions. This indicates hidden dimension has limited impact on performance, as the pre-trained DiT decoder’s generative prior offsets potential information loss in lower dimensions.

Table 7. Performance comparison on short and long horizon tasks.Best results are in bold and second-best results are underlined.

Short horizon tasks Long horizon tasks

Pick up the [toyname]

put the toy into the basket

Open the drawer

put all cups into the basket

put the toy into the drawer

Stack all cups

All Average

Method

Average

Average

Spatial-VLA [39] 0.45 0.50 0.20 0.38 0.35 0.20 0.20 0.25 0.32 WorldVLA [12] 0.55 0.20 0.15 0.30 0.25 0.40 0.15 0.26 0.28 UniVLA [45] 0.55 0.30 0.35 0.40 0.35 0.30 0.10 0.25 0.33 OpenVLA [24] 0.35 0.30 0.25 0.30 0.20 0.30 0.15 0.20 0.25 OpenVLA + StaMo state 0.60 0.55 0.50 0.60 0.55 0.55 0.45 0.52 0.56 OpenVLA-OFT + StaMo state 0.70 0.65 0.60 0.65 0.70 0.65 0.55 0.63 0.64

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

FR3 Arm

RealSense D435 Camera

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

Pick up the [toyname] Open the drawer

UMI Gripper

Put the [toyname] into the basket

Put all cups into the basket

Put the toy into drawer

Stack all cups

###### Real World Robot Setting

- Figure 5. Real World Setting and Tasks. We designed a benchmark of six real-world robotics tasks, spanning both short- and long-horizon challenges, to evaluate the effectiveness of our representation for learning world models.

Table 8. Reconstruction performance with 2 tokens under different hidden dimensions using our StaMo encoder.

Figure 6. Scaling Performance. Performance of our model can be scale with more data, including human ego-centric data.“S” means Simulation data only,“D” represents DROID [23] data, “O” represents OXE [36] data, “Ego” means Ego-centric data.

hidden dim Datasets libero 10 libero 90 libero goal libero object libero spatial

PSNR (dB) 26.5969 28.5858 27.5070 31.2779 25.7993 SSIM 0.9082 0.8948 0.9327 0.9478 0.9118

256

PSNR (dB) 27.3322 29.5773 26.2000 31.0129 24.6279 SSIM 0.9180 0.9127 0.9194 0.9444 0.9009

512

PSNR (dB) 27.0418 29.4810 26.4807 31.2653 25.9964 SSIM 0.9186 0.9124 0.9194 0.9480 0.9155

cal importance of pre-training on large-scale natural image datasets. Such pre-training provides a crucial prior, which is highly beneficial for learning a more effective representation of the robotics data distribution.

1024

###### 4.9.2. Ablation of Additional Training Data

A natural question following our initial validation is the scalability of our approach. To explore this, we investigated the effect of the StaMo training data volume and diversity on real-world task performance. We compared several data configurations: (1) simulation data only, (2) simulation + DROID [23] data (our main setting), (3) simulation + DROID + OXE [36] data, and (4) simulation + DROID + OXE + egocentric human (Ego) data [44]. The results, clearly depicted in Figure 6, confirm that our model is highly scalable. This verifies that incorporating larger volumes of data and greater task diversity, such as varied robot embodiments from OXE and human-centric data, steadily improves StaMo’s performance capabilities.

Table 9. Performance comparison between using pretrained diffusion decoder and training from scratch.

Method Spatial Object Goal Long Average

Train from Scratch 90.3% 92.7% 86.5% 73.4% 85.7% w. Pretrain Diffusion Decoder 92.3% 94.8% 88.1% 75.2% 87.6%

#### 5. Conclusion

In this work, we introduced StaMo, a framework that extracts reusable latent action vectors from static images via a transformer-based tokenizer and a diffusion decoder. We demonstrate that linear interpolation in this latent space produces smooth, physically plausible motions, and that these vectors generalize zero-shot to unseen senarios, suggesting that large-scale visual models implicitly learn a linearized dynamics manifold. Our approach offers a pathway toward scalable unsupervised skill discovery from diverse visual data, bridging the gap between static perception and dynamic action.

###### 4.9.3. Ablation of Pretrained Diffusion Decoder

We compared the performance of using a pre-trained diffusion decoder against training one entirely from scratch. The decoder trained from scratch required a significantly longer time to converge and ultimately was unable to match the same level of performance as its pre-trained counterpart. As shown in Table 9. This finding demonstrates the criti-

#### Acknowledgments

This work was supported in part by the Zhejiang Provincial Pioneering Science and Technology Program (2025), and by the National Natural Science Foundation of China (No. 62576315).

#### References

- [1] Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644, 2016. 7
- [2] Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571, 2022. 4
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 3
- [4] Lucas Beyer, Andreas Steiner, Andr´e Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024. 3
- [5] Homanga Bharadhwaj, Debidatta Dwibedi, Abhinav Gupta, Shubham Tulsiani, Carl Doersch, Ted Xiao, Dhruv Shah, Fei Xia, Dorsa Sadigh, and Sean Kirmani. Gen2act: Human video generation in novel scenarios enables generalizable robot manipulation. arXiv preprint arXiv:2409.16283,

2024. 3

- [6] Johan Bjorck, Fernando Casta˜neda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025. 3
- [7] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi0: A visionlanguage-action flow model for general robot control. corr, abs/2410.24164, 2024. doi: 10.48550. arXiv preprint ARXIV.2410.24164. 3
- [8] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 3
- [9] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In ICML, 2024. 3
- [10] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xu Huang, Shu Jiang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.
- [11] Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li.

- Univla: Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111, 2025. 2, 3
- [12] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, et al. Worldvla: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025. 3, 6, 8
- [13] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024. 3
- [14] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024. 3
- [15] Yi Chen, Yuying Ge, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, and Xihui Liu. Moto: Latent motion token as the bridging language for robot manipulation. arXiv preprint arXiv:2412.04445, 2024. 2, 3
- [16] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024. 3
- [17] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, page 02783649241273668, 2023. 6
- [18] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 1

- [19] Chongkai Gao, Haozhuo Zhang, Zhixuan Xu, Zhehao Cai, and Lin Shao. Flip: Flow-centric generative planning as general-purpose manipulation world model. arXiv preprint arXiv:2412.08261, 2024. 1, 3
- [20] Shenyuan Gao, Siyuan Zhou, Yilun Du, Jun Zhang, and Chuang Gan. Adaworld: Learning adaptable world models with latent actions. arXiv preprint arXiv:2503.18938, 2025. 3
- [21] Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, Xinyue Wei, Yunchao Yao, Xiaodi Yuan, Pengwei Xie, Zhiao Huang, Rui Chen, and Hao Su. Maniskill2: A unified benchmark for generalizable manipulation skills. In International Conference on Learning Representations, 2023. 6
- [22] Huy Ha, Pete Florence, and Shuran Song. Scaling up and distilling down: Language-guided robot skill acquisition. In Conference on Robot Learning, pages 3766–3777. PMLR,

2023. 3

- [23] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti,

- Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024. 5, 8
- [24] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 3, 6, 8
- [25] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025. 6
- [26] Peiyan Li, Yixiang Chen, Hongtao Wu, Xiao Ma, Xiangnan Wu, Yan Huang, Liang Wang, Tao Kong, and Tieniu Tan. Bridgevla: Input-output alignment for efficient 3d manipulation learning with vision-language models. arXiv preprint arXiv:2506.07961, 2025. 3
- [27] Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. arXiv preprint arXiv:2503.00200, 2025. 3
- [28] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 4
- [29] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023. 5
- [30] Mingyu Liu, Zheng Huang, Xiaoyi Lin, Muzhi Zhu, Canyu Zhao, Zongze Du, Yating Wang, Haoyi Zhu, Hao Chen, and Chunhua Shen. Bridge thinking and acting: Unleashing physical potential of vlm with generalizable action expert. arXiv preprint arXiv:2510.03896, 2025. 3
- [31] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 4
- [32] Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, Vincent-Pierre Berges, Tingfan Wu, Jay Vakil, et al. Where are we in the search for an artificial visual cortex for embodied intelligence? Advances in Neural Information Processing Systems, 36:655–677, 2023. 3
- [33] Robert McCarthy, Daniel CH Tan, Dominik Schmidt, Fernando Acero, Nathan Herr, Yilun Du, Thomas G Thuruthel, and Zhibin Li. Towards generalist robot learning from internet video: A survey. arXiv preprint arXiv:2404.19664, 2024. 3
- [34] Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3m: A universal visual representation for robot manipulation. arXiv preprint arXiv:2203.12601, 2022. 3
- [35] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 4
- [36] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open

- x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024. 8
- [37] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 4

- [38] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for visionlanguage-action models. arXiv preprint arXiv: 2501.09747,

2025. 6

- [39] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yani Ding, Zhigang Wang, Jiayuan Gu, Bin Zhao, Dong Wang, and Xuelong Li. Spatialvla: Exploring spatial representations for visual-language-action model. ROBOTICS,

2025. 6, 8

- [40] Ilija Radosavovic, Baifeng Shi, Letian Fu, Ken Goldberg, Trevor Darrell, and Jitendra Malik. Robot learning with sensorimotor pre-training. In Conference on Robot Learning, pages 683–693. PMLR, 2023. 3
- [41] Dominik Schmidt and Minqi Jiang. Learning to act without actions. arXiv preprint arXiv:2312.10812, 2023. 3
- [42] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024. 3, 6
- [43] Kaijun Wang, Liqin Lu, Mingyu Liu, Jianuo Jiang, Zeju Li, Bolin Zhang, Wancai Zheng, Xinyi Yu, Hao Chen, and Chunhua Shen. Odyssey: Open-world quadrupeds exploration and manipulation for long-horizon tasks. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 18602–18610, 2026. 3
- [44] Xiaofeng Wang, Kang Zhao, Feng Liu, Jiayu Wang, Guosheng Zhao, Xiaoyi Bao, Zheng Zhu, Yingya Zhang, and Xingang Wang. Egovid-5m: A large-scale video-action dataset for egocentric video generation. arXiv preprint arXiv:2411.08380, 2024. 8
- [45] Yuqi Wang, Xinghang Li, Wenxuan Wang, Junbo Zhang, Yingyan Li, Yuntao Chen, Xinlong Wang, and Zhaoxiang Zhang. Unified vision-language-action model. arXiv preprint arXiv:2506.19850, 2025. 3, 6, 8
- [46] Yating Wang, Haoyi Zhu, Mingyu Liu, Jiange Yang, HaoShu Fang, and Tong He. Vq-vla: Improving visionlanguage-action models via scaling vector-quantized action tokenizers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11089–11099, 2025. 3
- [47] Chuan Wen, Xingyu Lin, John So, Kai Chen, Qi Dou, Yang Gao, and Pieter Abbeel. Any-point trajectory modeling for policy learning. arXiv preprint arXiv:2401.00025, 2023. 1, 3, 6
- [48] Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Zhibin Tang, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, et al. Tinyvla: Towards fast, data-efficient vision-

- language-action models for robotic manipulation. IEEE Robotics and Automation Letters, 2025. 3
- [49] Tete Xiao, Ilija Radosavovic, Trevor Darrell, and Jitendra Malik. Masked visual pre-training for motor control. arXiv preprint arXiv:2203.06173, 2022. 3
- [50] Mengda Xu, Zhenjia Xu, Yinghao Xu, Cheng Chi, Gordon Wetzstein, Manuela Veloso, and Shuran Song. Flow as the cross-domain manipulation interface. In CoRL. 1
- [51] Mengda Xu, Zhenjia Xu, Yinghao Xu, Cheng Chi, Gordon Wetzstein, Manuela Veloso, and Shuran Song. Flow as the cross-domain manipulation interface. arXiv preprint arXiv:2407.15208, 2024. 3
- [52] Jiange Yang, Yansong Shi, Haoyi Zhu, Mingyu Liu, Kaijing Ma, Yating Wang, Gangshan Wu, Tong He, and Limin Wang. Como: Learning continuous latent motion from internet videos for scalable robot learning. arXiv preprint arXiv:2505.17006, 2025. 2, 3
- [53] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al. Latent action pretraining from videos. arXiv preprint arXiv:2410.11758, 2024. 2, 3, 6
- [54] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al. Latent action pretraining from videos. arXiv preprint arXiv:2410.11758, 2024. 7
- [55] Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, He Wang, Zhizheng Zhang, et al. Dreamvla: a vision-languageaction model dreamed with comprehensive world knowledge. arXiv preprint arXiv:2507.04447, 2025. 3
- [56] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1702–1713, 2025. 6
- [57] Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. Dino-wm: World models on pre-trained visual features enable zero-shot planning. arXiv preprint arXiv:2411.04983,

2024. 7

- [58] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023. 3

