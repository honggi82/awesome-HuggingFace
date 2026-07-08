## Chain of World: World Model Thinking in Latent Motion

Fuxiang Yang1,2* Donglin Di2 Lulu Tang3,6 Xuancheng Zhang2 Lei Fan4 Hao Li2 Wei Chen2 Tonghua Su1,5† Baorui Ma2‡

1Harbin Institute of Technology 2Li Auto 3Beijing Academy of Artificial Intelligence (BAAI) 4University of New South Wales 5Chongqing Research Institute of HIT 6Peking University

hityangfx@foxmail.com, donglin.ddl@gmail.com, lulutang @outlook.com xczhang.thu@gmail.com, lei.fan1@unsw.edu.au, {lihao43, chenwei10}@lixiang.com thsu@hit.edu.cn, mabaorui2014@gmail.com

# arXiv:2603.03195v1[cs.CV]3Mar2026

### Abstract

[Figure 1]

[Figure 2]

| |
|---|

[Figure 3]

[Figure 4]

E

…

Text

Text

[Figure 5]

Vision-Language-Action (VLA) models are a promising path toward embodied intelligence, yet they often overlook the predictive and temporal-causal structure underlying visual dynamics. World-model VLAs address this by predicting future frames, but waste capacity reconstructing redundant backgrounds. Latent-action VLAs encode frame-to-frame transitions compactly, but lack temporally continuous dynamic modeling and world knowledge. To overcome these limitations, we introduce CoWVLA (Chainof-World VLA), a new “Chain of World” paradigm that unifies world-model temporal reasoning with a disentangled latent motion representation. First, a pretrained video VAE serves as a latent motion extractor, explicitly factorizing video segments into structure and motion latents. Then, during pre-training, the VLA learns from an instruction and an initial frame to infer a continuous latent motion chain and predict the segment’s terminal frame. Finally, during co-fine-tuning, this latent dynamic is aligned with discrete action prediction by jointly modeling sparse keyframes and action sequences in a unified autoregressive decoder. This design preserves the world-model benefits of temporal reasoning and world knowledge while retaining the compactness and interpretability of latent actions, enabling efficient visuomotor learning. Extensive experiments on robotic simulation benchmarks show that CoWVLA outperforms existing world-model and latentaction approaches and achieves moderate computational efficiency, highlighting its potential as a more effective VLA pretraining paradigm. The project website can be found at https://fx-hit.github.io/cowvla-io.

[Figure 6]

VLM VLM

[Figure 7]

| |
|---|

| |
|---|

[Figure 8]

[Figure 9]

…

[Figure 10]

[Figure 11]

| |
|---|

[Figure 12]

[Figure 13]

(a) World Model (b) Latent Action

| | |
|---|---|
| | |

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

…

Text

[Figure 18]

[Figure 19]

[Figure 20]

motion

[Figure 21]

VLM

[Figure 22]

[Figure 23]

E

[Figure 24]

| |
|---|

[Figure 25]

…

[Figure 26]

| |
|---|

structure

(c) CoWVLA

Figure 1. Comparison of VLA pretraining strategies. (a) World Model: It predicts future visual frames, leading to redundant background reconstruction. (b) Latent Action: It learns the frame-toframe transition using a visual encoder E, but lacks temporally continuous reasoning. (c) CoWVLA: Our method first uses a video encoder E to decompose each video segment into motion and structure latents, and then trains the VLM to infer latent motion and predict the terminal frame of the segment given the instruction and the initial frame.

### 1. Introduction

Embodied intelligence aims to build agents that can perceive, understand, and act in the physical world. VisionLanguage-Action (VLA) models represent a significant step toward this goal, unifying multimodal perception and motor control into end-to-end transformers [3, 24, 34, 61]. While effective at mapping visual observations and language instructions directly to actions for many tasks, standard VLAs lack the future prediction capabilities that humans possess, which has spurred interest in enriching them with predictive

*Work done during an internship at Li Auto. †Corresponding author. ‡Project leader and corresponding author.

world models [1, 5].

A prominent approach integrates world models into VLAs by predicting future visual frames to explicitly model environmental dynamics, as illustrated in Figure 1 (a). Methods such as WorldVLA [7], UniVLA [50], and FlowVLA [58] typically built on large-scale autoregressive transformers, learn to anticipate future states and thus benefit action policy learning. While effective, this paradigm has fundamental limitations. It requires modeling entire visual frames containing substantial static and redundant background pixels, leading to near-trivial pixel replication rather than focusing on meaningful motion and dynamic change. Furthermore, quantizing images [15] into discrete tokens results in excessively long sequences and severe training inefficiency when multiple frames are used.

From a cognitive standpoint, such frame prediction is misaligned with how humans model the world: we reason about motion and interactions rather than rebuilding every pixel in memory. This observation raises an important question: can we build a more compact, abstract, and dynamic form of world modeling? The latent action paradigm [6, 11, 12, 54] offers compelling inspiration. As shown in Figure 1 (b), it encodes frame-to-frame transitions as latent actions, which serve as abstract motion carriers for world modeling, enabling large-scale pretraining using the pseudo-action labels built from videos.

However, we identify two critical limitations in the current latent action paradigm compared to world models. First, world models perform temporally continuous dynamic modeling, whereas existing latent actions often focus only on the change between two frames [6, 12, 54]. Second, world models, through future frame prediction, learn generalizable knowledge for task execution and common sense about the world. In contrast, latent actions only encode “how to move”, but lack an understanding of what is moving, where the motion happens, or how the scene should evolve after the motion.

To address these limitations, we propose Chain-of-World VLA (CoWVLA), which establishes a new paradigm that unifies the advantages of both approaches, as shown in Figure 1 (c). Our key insight is that effective world modeling requires both the compactness of motion representations and the temporal continuity and world knowledge of frame prediction. We argue that it is possible to extract continuous and compact motion representations from video clips, suggesting the need for a model capable of decoupling the content structure and motion in videos. Such motion representations serve as carriers for perceiving essential dynamic changes and further enable the model to reason about keyframes after temporal evolution, thereby preserving crucial visual landmarks.

Specifically, our approach employs a pretrained video VAE as a latent motion extractor, which explicitly disen-

tangles each video segment into structure and motion representations, providing compact and interpretable supervision for downstream visuomotor learning. We then train a unified VLA decoder through two stages. During the pretraining stage, the model learns to infer latent dynamics and predict the terminal frame of a video segment given the instruction and initial frame, thereby establishing a dynamicsaware world prior in the latent motion space. During the subsequent co-fine-tuning stage, this prior is further aligned with discrete action prediction by jointly modeling sparse keyframes and action sequences in a unified autoregressive manner. This design combines the interpretability and compactness of latent motion with the temporal reasoning and world knowledge of world models, achieving efficient and robust visuomotor learning without reconstructing redundant intermediate frames.

In summary, our contributions are as follows:

- • We present CoWVLA, establishing the “Chain-of-World” paradigm that unifies world modeling and latent action learning through continuous latent-motion sequences and terminal keyframe prediction.
- • We introduce a structure-motion disentangled latent prior that yields interpretable, continuous, and effective dynamic representations.
- • We conduct extensive experiments demonstrating that CoWVLA achieves state-of-the-art performance across multiple benchmarks, surpassing existing world-model and latent-action approaches.

### 2. Related Work

Vision-Language-Action Models. Deep learning has been widely applied in various industrial scenarios, such as visual anomaly detection [16, 17]. Recent vision-languageaction (VLA) models have rapidly advanced toward directly generating actions from visual and language inputs within a unified framework [3, 18, 22, 24, 25, 34–36, 41, 46, 61]. RT-2 [61] pioneered this direction by treating robotic control as a sequence modeling problem, fine-tuning a pretrained vision-language model on robotic data to output discretized action tokens. This approach was scaled up by RT-X [34], which demonstrated the benefits of joint training across diverse robot platforms and tasks. OpenVLA [24, 25] further democratized this effort with an opensource implementation. FAST [35] introduced a unified frequency-domain formulation for discretizing actions, enhancing temporal correlation in discrete control. Meanwhile, another line of research explores continuous trajectory generation [3, 13, 21, 28]. They leverage diffusion or flow-matching models to generate continuous, highfrequency action sequences. However, most existing methods primarily focus on action space modeling, with limited capability to capture how the environment evolves.

##### World Models for Robotics. World models are com-

Latent Motion Extractor

Pre-training Co-fine-tuning

[Figure 27]

[Figure 28]

[Figure 29]

…

𝑍

𝑍

𝑣

𝑣

𝐴

𝑣 𝐴 …

MLP MLP 𝐴

𝑣

Decoder

Transformer Decoder Transformer Decoder

structure

𝑧 𝑧

motion

𝑧

| | |
|---|---|
| | |

| | |
|---|---|
| | |

𝑇 𝑣 𝑄 𝑣

𝑇 𝑣 𝑄

𝐴 𝑣 𝐴 …

𝐴

Avg. height Avg. width

Inference

| |𝑧|
|---|---|
| | |

Q-Former

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

motion Text Vision Action

Stack Green Cube

𝑞

| | |
|---|---|
| | |

Encoder

[Figure 34]

[Figure 35]

[Figure 36]

…

Motion Query

𝑇 𝑣 𝑄 𝐴 𝑣 𝐴

𝑣 𝑣 𝑣

Latent Motion

- Figure 2. Overview of the CoWVLA framework. CoWVLA consists of two core components: a latent motion extractor and a VLA decoder. The latent motion extractor, implemented as a video VAE, disentangles each video segment into a structure latent zs and two

directional motion latents zmh and zmw, which are concatenated into a unified latent motion vector zm. The VLA decoder performs unified autoregressive modeling over multimodal sequences. During pre-training, the model takes the instruction and initial frame as input, and uses a learnable motion query Q to predict the latent motion zˆm while reconstructing the terminal frame of the video segment. During co-fine-tuning, the input expands into alternating keyframe–action pairs; Q continues to aggregate temporally continuous latent dynamics, guiding multi-step action generation under sparse visual observations.

the latent action representations inevitably encode static appearance and contextual details. While TLA [6] mitigates this issue by decoupling task relevance, an ideal latent space should explicitly separate structure from motion, producing cleaner and more interpretable action representations.

monly employed to capture environment states and their future evolution, and have been widely applied in areas such as autonomous driving [48, 51], image and video generation [5, 14, 31, 42, 45, 53], and robotics [1, 7, 19, 38, 50, 56]. When combined with VLA models, most approaches [7, 8, 50, 52, 57, 58] rely on predicting future visual states to provide implicit world knowledge and demonstrate improved performance in robotic manipulation. UVA [29] further jointly optimizes video prediction and action prediction using diffusion models, enhancing both visual reasoning and control inference efficiency. However, these methods require reconstructing full visual frame sequences, leading to high computational cost and heavy resource consumption.

Video Compression and Decoupling. Recent methods in video representation learning have increasingly focused on compressing visual information into disentangled latent spaces that separately encode spatial structure and temporal motion [26, 40, 49, 53, 55]. The design of our latent motion space is inspired by these advances. Models like CMD [55] and VidTwin [49] have successfully disentangled overall content and dynamic information in a highly compressed latent space. This factorization provides a compact, continuous, and meaningful representation of how scenes evolve. While these models were developed for video generation, we are the first to hypothesize and demonstrate that their pretrained latent motion space can serve as a powerful dynamic prior for a robotic world model.

Latent Actions for Robotics. Latent-action methods learn a compact latent transition between two frames to model environment dynamics. LAPA [54] introduces a three-stage framework (including latent action quantization, latent pretraining, and action fine-tuning), leveraging largescale pseudo-action supervision to improve learning of realworld robotic control. MoTo [12] follows this paradigm with enhancements in motion quantization and real action quality. TLA [6] further disentangles task-relevant and taskirrelevant motion factors. However, these approaches generally restrict latent action modeling to frame pairs, limiting their ability to capture long-range temporal dynamics. Although Villa-X [11] extends latent actions to multi-frame settings, it still generates one latent action per local frame pair, resulting in limited temporal consistency. Moreover,

### 3. Method

#### 3.1. Overall Framework

We consider a robotic manipulation task that involves executing a sequence of actions conditioned on a language instruction and visual observations. The instruction is denoted as T. The raw action sequence is A1:t = {a1,...,at}. To enable discrete sequence modeling, the action sequence A1:t is partitioned into consecutive chunks of fixed length

la, i.e., A1:t = Nj=1 Aj, Aj = {a(j−1)l

a}, and each chunk Aj is then quantized into a discrete token sequence Ajq, using the FAST [35] algorithm. The raw corresponding visual observation sequence is represented as V1:t = {v1,...,vt}, where each frame vi ∈ RH×W×3. We extract the first frame of each action chunk as a keyframe: V˜ = {v˜j}Nj=1 = {v(j−1)l

a+1,...,ajl

a+1}Nj=1, where

each v˜j is subsequently quantized into a visual token v˜qj using VQGAN [15]. Additionally, a learnable motion query token Q ∈ RD

Q is introduced as a world dynamics query, whose hidden representation summarizes past context and provides a future dynamics-aware conditioning signal for generating subsequent vision or action tokens.

The overall framework consists of two models and three training stages. The first model is the latent motion extractor (video VAE paradigm), which encodes a video subsequence V1:f into an intermediate latent z ∈ Rd

z×f×h×w, and decomposes it into a structural feature zs and two directional motion features zmh and zmw. The two motion components are concatenated to form a unified latent motion vector zm ∈ RD

m, providing the ground-truth supervision. The second model is the VLA decoder (Transformerdecoder paradigm), which performs unified autoregressive next-token prediction across modalities. During pretraining, the input sequence is organized as [T,vq1,Q,vqf]. The final hidden representation corresponding to the query token Q, obtained from the VLA decoder, is fed into an MLP to predict the latent motion zˆm. This stage enables the model to infer latent dynamics and future observations from language and the initial visual input. During co-finetuning, we use alternating keyframes and action tokens, e.g., [T,v˜q1,Q,A1q,v˜q2,A2q,...]. The model continues to predict a latent motion vector zˆm at Q position. As a result, the model maintains explicit dynamics reasoning under sparse keyframe observations and generates stable multi-step actions from compact latent representations.

#### 3.2. Latent Motion Extractor

To encode temporal dynamics in a compact latent space, we adopt a pretrained video variational autoencoder [49] as the latent motion extractor. As illustrated in Figure 2, the extractor achieves structure–motion disentanglement through two dedicated branches. Given a video segment V1:f, the encoder produces a latent tensor z ∈ Rd

z×f×h×w. The structure branch employs a Q-Former [27] module with a set of learnable queries {qi}ni=1q to aggregate global semantics and low-frequency dynamics along the temporal dimension, yielding zs ∈ Rd

s×nq×hs×ws,nq ≤ f. The motion branch operates along spatial dimensions: several convolutional layers reduce the dimension of z and produce z′ ∈ Rd

m×f×hm×wm. Then, spatial averaging µ(·) is applied independently along the height and width axes to extract directional motion embeddings: zmh = µh(z′) ∈

m×f×hm. These two motion components are concatenated and flattened to form a unified latent motion representation: zm ∈ RD

Rd

m×f×wm,zmw = µw(z′) ∈ Rd

,Dm = f ×dm ×(hm +wm). In the decoder stage, the three latent components (zs,zmh ,zmw) are upsampled through convolutional and MLP layers to the same spatial and temporal size, summed together, and then fed into the decoder to reconstruct Vˆ 1:f. The training objective follows the original VAE design [49], combining reconstruction loss Lrec, perceptual loss Lp, adversarial loss LGAN, and KL-divergence regularization loss LKL to preserve temporal consistency and visual realism:

m

Lvae = Lrec + λpLp + λGANLGAN + λKLLKL. (1)

Through explicit structure–motion disentanglement and mild adaptation, the extractor yields a compact, interpretable, and transferable latent representation well-suited for robotic scenarios, providing effective supervision for downstream VLA pre-training and co-fine-tuning.

#### 3.3. Pre-training to Think in Latent Motion

The pre-training stage aims to align language and initial visual observations with latent motion representations, enabling the model to reason about continuous temporal dynamics in the latent space and predict the terminal frame of the video segment. Given a continuous video segment V1:f = {v1,...,vf}, the latent motion extractor produces a latent motion supervision signal zm. Its first and last frames are quantized into discrete visual tokens, denoted as vq1 and vqf, respectively. Based on this, we organize the input sequence to the VLA decoder as: [T,vq1,Q,vqf], where T denotes the instruction, vq1 represents the initial observation, Q is a learnable motion query token, and vqf corresponds to the visual state that would be reached after applying the underlying motion from v1 through zm. During the forward pass, the hidden state at the query position is fed to an MLP to predict the latent motion zˆm.

To prevent information leakage, causal masking is applied so that Q only attends to {T,vq1} while being masked from vqf. The training objective contains latent motion supervision and terminal-frame visual consistency:

Lpretrain = ∥zˆm − zm∥22 +

CE(ˆvqx,vqx), (2)

x∈{1,f}

where the first term enforces that the latent representation extracted at Q accurately summarizes the continuous motion from v1 to vf, while the second ensures that the model forms a coherent prediction of the resulting future state. Through this stage, the model learns to infer latent temporal dynamics directly from language and the initial frame, thus establishing a dynamics-aware prior for subsequent action modeling.

- 3.4. Co-Fine-Tuning for Aligning Latent Dynamics with Action Policies

After the pre-training stage establishes a dynamics-aware prior in the latent motion space, the co-fine-tuning stage further aligns latent motion reasoning with discrete action modeling in a unified autoregressive framework, enabling stable multi-step control under sparse keyframe observations. Given a continuous video sequence V1:f and its corresponding action sequence A1:f, we extract N = f/la keyframes and quantize them into visual tokens: V˜ q = {v˜q1,...,v˜qN}, where v˜qj = vq(j−1)la+1. We further quantize the action sequence using FAST [35]: A1:f −−−→FAST {A1q,...,ANq }. The input sequence adopts a “single-Q for the full window” design: [T, v˜q1, Q, A1q, v˜q2, A2q, ..., ANq ], where the query token Q appears only once after the first keyframe and serves as a latent dynamics aggregator for the entire temporal horizon. The decoder autoregressively predicts both action and visual tokens; the hidden state at Q is passed through an MLP to produce a single latent motion vector zˆm, enforcing consistency between latent dynamics and subsequent predictions. As in pre-training, causal masking prevents Q from attending to future keyframes and actions, compelling the model to reason over latent dynamics rather than directly peeking at future states.

The co-fine-tuning objective consists of three terms:

Lfinetune =

N

j=1

CE A ˆ jq, Ajq + λ1 ∥zˆm − zm(V1:f)∥22

+ λ2

N

j=1

CE v ˆ˜qj, v˜qj .

(3) Here, zm(V1:f) is a continuous latent motion supervision signal produced by the pretrained extractor. The first term ensures accurate execution of discrete actions. The second term encourages the latent representation at the query token to faithfully capture the continuous dynamics from v1 to vf. The third term anchors motion predictions to sparse visual checkpoints, maintaining consistent state transitions driven by the predicted dynamics.

- 4. Experiments

- 4.1. Benchmarks

LIBERO. The LIBERO [32] benchmark is designed for studying knowledge transfer in multitask and lifelong robot learning, requiring both declarative knowledge about objects and spatial relations and procedural knowledge about motion and behaviors. It contains four task suites: LIBERO-Spatial emphasizes spatial reasoning by placing a bowl based on its location, LIBERO-Object focuses on

object recognition via picking and placing distinct objects, LIBERO-Goal tests procedural learning with varying task goals under fixed objects, and LIBERO-Long contains ten long-horizon tasks with diverse objects, layouts, and goals.

SimplerEnv. SimplerEnv [30] is a collection of manipulation evaluation environments for common real-world robot setups, showing strong correlation with real-robot performance. It enables assessing the transferability and generalization of models trained on real-world video data. We evaluate on four tasks using a 7-DoF WidowX robotic arm.

#### 4.2. Implementation Details

Our latent motion extractor is built upon a pretrained video VAE (VidTwin [49]) and is further fine-tuned on a robotcentric dataset consisting of 237k videos (details provided in the appendix). Each video segment is uniformly sampled to 16 frames and resized to 224 × 224. The structure latent zs has a shape of 4 × 16 × 7 × 7, while the directional motion embeddings zmh and zmw have shapes of 8 × 16 × 7. The motion latent dimension is Dm = 1792.

The backbone of our VLA model follows the design of UniVLA [50] and is based on the 8.5B-parameter VLM Emu3 [47]. Visual observations are quantized into discrete tokens using VQGAN [15], while actions are partitioned into chunks and discretized into tokens using the FAST algorithm [35]. During the pre-training stage, we trained the model using the aforementioned 237k videos with pretrained Emu3 initialization. From each video, we extracted a frame sequence of length f = 16, where the first and last frame tokens supervise visual modeling, and the latent motion extracted from VidTwin provides supervision. We trained using a batch size of 256 for 10k steps. During the co-fine-tuning stage, we initialized from the pretrained checkpoint and trained on the benchmark-specific datasets. For the LIBERO benchmark, we used the mixed data from the four task suites curated by OpenVLA [24], including both third-person and wrist-mounted views. We trained the model with a batch size of 128 for 8k iterations, resized all images to 200×200, set the action chunk length to la = 10, and used λ1 = 0.1 and λ2 = 0.01. For SimplerEnv, we trained the model on the Bridge V2 dataset [43] with a batch size of 128 for 12k iterations. Single-view images were resized to 256 × 256, the action chunk length is set to la = 5, and we used λ1 = 0.1 and λ2 = 0. In the co-fine-tuning stage, we set N = 2, where two visual observations and two corresponding ground-truth action chunks were used.

Further training details and supplementary results are provided in the appendix.

#### 4.3. Comparison with SOTA Methods

We compared CoWVLA against three representative categories of methods: VLA baselines (OpenVLA [24], SpatialVLA [36], CogACT [28], DiTA [21], π0 [3],

Table 1. Comparison of different methods on the LIBERO [32] and SimplerEnv-WidowX [30] benchmarks. The best and the second-best values for each metric are bold and underlined respectively.

LIBERO SimplerEnv-WidowX SPATIAL OBJECT GOAL LONG Avg. Stack Block Put Carrot Put Spoon Put Eggplant Avg.

Model

OpenVLA [24] 0.849 0.884 0.792 0.537 0.765 0.000 0.000 0.000 0.041 0.010 SpatialVLA [36] 0.882 0.899 0.786 0.555 0.781 0.292 0.250 0.167 1.000 0.427 CogACT [28] 0.960 0.874 0.868 0.846 0.887 0.150 0.508 0.717 0.675 0.513 Dita [21] 0.842 0.963 0.854 0.638 0.824 – – – – –

π0 [3] 0.968 0.988 0.958 0.852 0.942 0.167 0.000 0.291 0.625 0.401 π0-FAST [35] 0.964 0.968 0.886 0.602 0.855 0.108 0.219 0.291 0.666 0.483 GR00T N1 [2] 0.944 0.976 0.930 0.906 0.939 0.167 0.458 0.625 0.208 0.495

###### w/ Latent Actions

LAPA [54] – – – – – 0.542 0.458 0.708 0.583 0.573 villa-X [11] 0.975 0.970 0.915 0.745 0.901 0.613 0.463 0.779 0.646 0.625 TLA [6] 0.965 0.968 0.956 0.920 0.952 0.028 0.556 0.528 0.806 0.480

w/ World Model WorldVLA [7] 0.856 0.890 0.826 0.590 0.791 – – – – – CoT-VLA [57] 0.875 0.916 0.876 0.690 0.811 – – – – – UniVLA [50] 0.960 0.992 0.932 0.914 0.950 0.292 0.625 0.833 1.000 0.687 FlowVLA [58] 0.932 0.950 0.916 0.726 0.881 0.625 0.625 0.708 1.000 0.740

Ours 0.972 0.978 0.946 0.928 0.956 0.625 0.667 0.792 0.958 0.760

Table 2. Evaluation of VAE-Reconstructed Videos and downstream fine-tuning performance on SimplerEnv-WidowX [30].

Reconstruction Metrics Simulation Evaluation PSNR↑ SSIM↑ LPIPS↓ Stack Block Put Carrot Put Spoon Put Eggplant Average

Model

Pretrain 32.7 0.923 0.122 0.458 0.750 0.792 0.917 0.729 Finetune 33.4 0.934 0.123 0.625 0.667 0.792 0.958 0.760

tions directly, (ii) frame-to-frame latent transitions, and (iii) pixel/token-level future frames. They collectively represent the main paradigms in current VLA pretraining and provide strong and fair comparison points. The results are shown in Table 1.

Origin Recon M.Recon S.Recon

[Figure 37]

[Figure 38]

[Figure 39]

𝒕𝟏

Overall, our CoWVLA achieves SOTA performance with superior cross-domain robustness. We observe that TLA achieves a strong 0.952 on LIBERO but significantly drops to 0.480 on SimplerEnv, while FlowVLA is strong on SimplerEnv (0.740) but noticeably weaker on LIBERO (0.881). UniVLA shows a more balanced performance (0.950/0.698). In contrast, CoWVLA achieves 0.956/0.760 on the two benchmarks, outperforming UniVLA on both and demonstrating higher absolute performance and greater cross-domain stability.

[Figure 40]

[Figure 41]

[Figure 42]

𝒕𝟐

- Figure 3. Visualization of the disentangled motion and structure

latents. We select two frames (t1 and t2) and show the original (Orig.) and reconstructed (Recon.) frames. “M. Recon.” and “S. Recon.” denote the reconstructions obtained by decoding only the motion latent or only the structure latent, respectively. The structure latent preserves the global scene layout, whereas the motion latent captures motion and fine-grained temporal details.

#### 4.4. Latent Motion Analysis

In this subsection, we analyze the effectiveness of the proposed disentangled latent space from three perspectives: the separation of structure and motion factors, the improved adaptiveness of the motion latent after fine-tuning on robot data, and the enhanced capability of modeling future dynamics. These results collectively verify that our latent

π0-FAST [35], GR00T-N1 [2]), latent-action approaches (LAPA [54], villa-X [11], TLA [6]), and world-model approaches (WorldVLA [7], CoT-VLA [57], UniVLA [50], FlowVLA [58]). These methods respectively model: (i) ac-

[Figure 43]

motion representation provides a clearer physical prior and stronger action reasoning ability.

[Figure 44]

𝑡 𝑡 𝑡 𝑡 𝑡

[Figure 45]

|Structure Latent|
|---|

Effective decoupling of structure and motion latent. As shown in Figure 3, we reconstruct frames using only the motion latent (M. Recon.) or only the structure latent (S. Recon.). The structure latent preserves global scene layout and object appearance, whereas the motion latent captures robot arm trajectories and fine-grained temporal dynamics. Figure 4 provides additional evidence through crossreconstruction. Since motion cues are subtle in individual frames, we visualize the pixel-wise differences, which highlight the motion-affected regions and show that injecting the motion latent alters only the dynamic parts while keeping the static structure intact. These visualizations demonstrate that our latent space effectively separates content structure and dynamic information, providing a more interpretable representation for downstream visuomotor reasoning.

+

Motion Latent

Cross Recon

[Figure 46]

[Figure 47]

- Figure 4. Cross-reconstruction visualization. We extract the structure latent from the static video in the first row and the motion latent from the robot-arm motion video in the second row. By combining the two latents, we reconstruct the video shown in the third row. We compute the difference between the cross-reconstructed frames and the static frames to highlight the changed regions, which correspond to the robot arm’s motion.

(b) predict one goal frame (c) predict 𝑧 and one terminal frame

(a) predict five frames

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

𝑧

[Figure 54]

[Figure 55]

[Figure 56]

𝑧

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

- Figure 5. Comparative visualization of future-frame prediction strategies. There are two tasks demonstrated: i) pick up the black bowl from the table center and place it on the plate, and ii) sweep into a pile. (a) The world-model approach predicts five future frames. (b) The single-goal-frame approach predicts one goal frame. (c) Our method reasons through a learned motion latent zm, producing more reasonable and instruction-aligned frames.

[Figure 62]

Fine-tuning on robot data improves motion latent quality. As presented in Table 2, fine-tuning the latent motion extractor on robot data not only improves reconstruction quality (higher PSNR and SSIM) but also boosts downstream performance. In the SimplerEnv-WidowX evaluation, the average task success rate increases from 0.729 to 0.760. This confirms that motion latents adapted to the robot domain contain higher-quality dynamic cues that benefit policy learning.

Motion latent enhances dynamic modeling for future frame prediction. As illustrated in Figure 5, we visualize the future frame predictions under different pretraining strategies. From top to bottom, the tasks in each subfigure are: i) pick up the black bowl from the table center and place it on the plate, ii) sweep into a pile. Worldmodel-based approaches reconstruct redundant background pixels and therefore struggle to focus on interactive motion, while single-goal-frame prediction lacks supervision of temporal evolution and often produces unstable goal frames. This leads both strategies to easily generate results with no changes, such as Figure 5 (b) Task i. In contrast, our model leverages the motion latent as a “chain of world” during reasoning, achieving physically plausible future states that align more closely with the instructions.

#### 4.5. Ablation and Efficiency Analysis

In this section, we conduct an in-depth analysis of key modules, hyperparameter settings, and training efficiency. Experiments in Table 3 and Table 4 adhere to a unified dataset and training configuration, with a batch size of 256 for 10k steps during the pre-training phase and a batch size of 128 for 8k steps during the co-fine-tuning phase. In Table 3, we provide a unified comparison of the effectiveness of latent action, world model, and our proposed method. In Table 4, we analyze the effect of the loss weighting ratio between the latent motion loss (λ1) and the visual token loss (λ2)

on task success rates during the co-fine-tuning strategy. In addition, we analyze the pre-training cost and task success rate of different methods in Figure 6. The main conclusions are as follows.

i) Our latent motion modeling significantly outperforms existing latent action methods. The “Latent Action” part of Table 3 compares several baselines. The “w/o LA” variant, which skips pre-training and fine-tunes directly on LIBERO data, achieves the lowest average success rate

Table 3. Ablation study on the LIBERO [32] benchmark.

Config Variant Spatial Object Goal Long Average

w/o LA 0.622 0.146 0.694 0.328 0.448 LAPA style 0.718 0.852 0.804 0.488 0.716 villa-X style 0.840 0.904 0.834 0.668 0.812 structure latent 0.856 0.898 0.822 0.692 0.817 motion latent 0.916 0.932 0.886 0.774 0.877

Latent Action

UniVLA Style 0.958 0.978 0.932 0.898 0.942 CoT-VLA style 0.942 0.964 0.950 0.838 0.924

World Model

motion 0.960 0.980 0.922 0.882 0.936 motion & cot 0.948 0.974 0.958 0.906 0.947

Ours

(0.448). “LAPA style” (0.716) and “villa-X style” (0.812) both outperform the “w/o LA” variant, with “villa-X style” achieving stronger performance by modeling richer multiframe information. Our method separates the latent into a “structure latent” (0.817) capturing content and texture, and a “motion latent” (0.877) encoding dynamic information. Modeling with the cleaner motion notably improves task success rate.

- ii) World model methods show stronger overall perfor-

mance than latent action methods. In the “World Model” part of Table 3, both “UniVLA style” (pretrained with six frames) and “CoT-VLA style” (pretrained with initial and target frames) achieve higher success rates (0.942 and

- 0.924, respectively) than those methods in the “Latent Action” category. Notably, “UniVLA style”, which uses more frames, performs better, indicating that world model methods have a distinct advantage in temporal modeling and learning knowledge of environmental evolution.

- iii) Our method achieves superior performance to la-

tent action and world models. The “Ours” part in Table 3 presents two configurations of our method. Both use latent motion supervision during pre-training and set λ1 = 0.1,λ2 = 0 during fine-tuning (i.e., using only real action and latent motion losses). The “motion” configuration does not use the final frame vf during pre-training and achieves a success rate of 0.936. In contrast, the “motion & cot” configuration adds supervision from vf during pretraining and improves the success rate to 0.947. This yields two conclusions: first, introducing latent motion during the fine-tuning phase effectively guides the inference of real actions; second, introducing vf as an evolutionary target during pre-training significantly enhances the model’s perception and understanding of environmental evolution.

- iv) Balancing latent motion and visual token losses dur-

ing co-fine-tuning further improves performance. Table 4 presents an ablation study on the loss weights λ1 (latent motion) and λ2 (visual token) during the co-fine-tuning stage, based on the same pretrained model. First, we fix λ2 = 0 to analyze the impact of λ1. When λ1 = 0 (no latent motion loss), the success rate is only 0.872. As λ1 increases from 0.1 to 1.0, the success rate improves from 0.936 to

- 0.945, indicating that the guiding effect of latent motion

Table 4. Ablation study of loss weights on the LIBERO [32] benchmark.

- λ1 λ2 Spatial Object Goal Long Average

- 0.0 0.0 0.922 0.962 0.862 0.742 0.872

- 0.1 0.0 0.960 0.980 0.922 0.882 0.936 1.0 0.0 0.958 0.970 0.950 0.902 0.945 0.1 0.05 0.954 0.972 0.944 0.914 0.946

- 0.1 0.01 0.970 0.964 0.958 0.926 0.955

- 1.0 0.01 0.970 0.956 0.934 0.922 0.946

12 10 8 6 4

Training Speed (s/iter)

0.70

0.75

0.80

0.85

0.90

0.95

SuccessRate

UniVLA

CoT-VLA

LAPA

villa-X

Ours (motion & cot)

Ours (motion)

Faster

Higher

GPU Memory 76 GB 40 GB 32 GB

- 30 GB

42 GB

- 31 GB

Figure 6. Comparison of pre-training efficiency and task performance on LIBERO [32] across different methods. Blue and orange circles denote world-model and latent-action baselines, respectively, while green circles denote our configurations. Circle size indicates training-time GPU memory usage. Our method balances pre-training efficiency and performance, achieving a higher success rate with moderate computational efficiency.

is strengthening. Next, we introduce the visual token loss

- λ2. By comparing (λ1 = 0.1,λ2 = 0.05) at 0.946 and (λ1 = 0.1,λ2 = 0.01) at 0.955, we find that the weight for visual token prediction should not be too high. Then we

tune λ1 = 1.0 and λ2 = 0.01, achieving an average success rate of 0.946. This proves that simultaneously introducing latent motion (λ1 = 0.1) and a low-weighted visual token prediction (λ2 = 0.01) during the fine-tuning phase most effectively guides the inference of real actions.

v) Our method balances pre-training efficiency and performance. As shown in Figure 6, we compare several methods from Table 3 in terms of training speed, GPU memory usage, and task success rate (batch size = 4 per GPU). UniVLA is the slowest and most memory-intensive, while LAPA is the fastest but less successful. Our method has two configurations: “motion” without vf achieves the secondfastest speed and slightly lower performance than UniVLA, and “motion & cot” with vf achieves a better balance of efficiency and performance, surpassing UniVLA in both.

### 5. Conclusion

In this work, we presented CoWVLA, which for the first time integrates the temporal reasoning capability of world models with a disentangled latent motion representation, enabling world modeling directly in a structure–motion separated latent space. By introducing the Chain-of-World paradigm, our method predicts a continuous latent motion chain and a terminal keyframe from the instruction and initial observation, compactly capturing temporal evolution and physical dynamics without reconstructing intermediate pixels. Extensive experiments on LIBERO and SimplerEnv benchmarks demonstrate that CoWVLA outperforms both world-model and latent-action approaches, while offering improved dynamic consistency and visuomotor grounding, thereby providing a more efficient pretraining route toward general-purpose robotic manipulation.

Limitations. Despite its promising results, our approach still has limitations. The latent motion space remains dependent on the quality and domain coverage of the pretrained video VAE, which may introduce distribution mismatch in new environments. Moreover, the model relies on a large VLA backbone and substantial computational resources. We believe exploring more lightweight and scalable architectures, as well as further enhancing the coupling between latent dynamics and action learning, will broaden the applicability of our method to real-world robotics.

### 6. Acknowledgments

This work was supported by the National Natural Science Foundation of China (Grant No. 62277011), Project of Chongqing MEITC (Grant No. YJX-2025001001009), and CAAI-CANN Open Fund, developed on OpenI Community.

### References

- [1] Mido Assran, Adrien Bardes, David Fan, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025. 2, 3
- [2] Johan Bjorck, Fernando Casta˜neda, Nikita Cherniadev, Xingye Da, Runyu Ding, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025. 6
- [3] Kevin Black, Noah Brown, Danny Driess, et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 1, 2, 5, 6
- [4] Anthony Brohan, Noah Brown, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 1
- [5] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, et al. Genie: Generative interactive environments. In ICML, 2024. 2, 3

- [6] Qingwen Bu, Yanting Yang, Jisong Cai, et al. Learning to act anywhere with task-centric latent actions. In RSS, 2025. 2, 3, 6
- [7] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, et al. Worldvla: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025. 2, 3, 6
- [8] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024. 3
- [9] Lili Chen, Shikhar Bahl, and Deepak Pathak. Playfusion: Skill acquisition via diffusion from language-annotated play. In CoRL, pages 2012–2029, 2023. 1
- [10] Lawrence Yunliang Chen, Simeon Adebola, and Ken Goldberg. Berkeley UR5 demonstration dataset, 2024. 1
- [11] Xiaoyu Chen, Hangxing Wei, Pushi Zhang, Chuheng Zhang, Kaixin Wang, et al. villa-X: enhancing latent action modeling in vision-language-action models. arXiv preprint arXiv:2507.23682, 2025. 2, 3, 6
- [12] Yi Chen, Yuying Ge, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, and Xihui Liu. Moto: Latent motion token as the bridging language for robot manipulation. In ICCV, 2025. 2, 3, 6
- [13] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, et al. Diffusion policy: Visuomotor policy learning via action diffusion. In RSS, 2023. 2
- [14] Donglin Di, He Feng, Wenzhang Sun, Yongjia Ma, Hao Li, Wei Chen, Lei Fan, Tonghua Su, and Xun Yang. Dh-facevid1k: A large-scale high-quality dataset for face video generation. In ICCV, pages 12124–12134, 2025. 3
- [15] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, pages 12873–12883, 2021. 2, 4, 5
- [16] Lei Fan, Dongdong Fan, Zhiguang Hu, Yiwen Ding, Donglin Di, Kai Yi, Maurice Pagnucco, and Yang Song. Manta: A large-scale multi-view and visual-text anomaly detection dataset for tiny objects. In CVPR, pages 25518–25527, 2025. 2
- [17] Lei Fan, Junjie Huang, Donglin Di, Anyang Su, Tianyou Song, Maurice Pagnucco, and Yang Song. Salvaging the overlooked: Leveraging class-aware contrastive learning for multi-class anomaly detection. In ICCV, pages 21419– 21428, 2025. 2
- [18] Chongkai Gao, Zixuan Liu, Zhenghao Chi, Junshan Huang, Xin Fei, Yiwen Hou, Yuxuan Zhang, Yudi Lin, Zhirui Fang, and Lin Shao. VLA-OS: Structuring and dissecting planning representations and paradigms in vision-languageaction models. In NeurIPS, 2025. 2
- [19] Shenyuan Gao, Siyuan Zhou, Yilun Du, Jun Zhang, and Chuang Gan. Adaworld: Learning adaptable world models with latent actions. In ICML, 2025. 3
- [20] Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, Xinyue Wei, Yunchao Yao, et al. Maniskill2: A unified benchmark for generalizable manipulation skills. arXiv preprint arXiv:2302.04659, 2023. 1

- [21] Zhi Hou, Tianyi Zhang, Yuwen Xiong, Haonan Duan, et al. Dita: Scaling diffusion transformer for generalist visionlanguage-action policy. In ICCV, 2025. 2, 5, 6
- [22] Physical Intelligence, Kevin Black, Noah Brown, et al. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025. 2
- [23] Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian Ibarz, et al. Scalable deep reinforcement learning for vision-based robotic manipulation. In CoRL, pages 651–673, 2018. 1
- [24] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, et al. OpenVLA: An open-source vision-language-action model. In CoRL, 2024. 1, 2, 5, 6
- [25] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. In RSS, 2025. 2
- [26] Jaihyun Lew, Jooyoung Choi, Chaehun Shin, Dahuin Jung, and Sungroh Yoon. Disentangled motion modeling for video frame interpolation. In AAAI, pages 4607–4615, 2025. 3
- [27] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, pages 19730–19742. PMLR, 2023. 4
- [28] Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, et al. CogACT: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024. 2, 5, 6
- [29] Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. In RSS, 2025. 3
- [30] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Oier Mees, Karl Pertsch, et al. Evaluating real-world robot manipulation policies in simulation. In CoRL, 2024. 5, 6, 1, 2, 4
- [31] Bin Lin, Yunyang Ge, Xinhua Cheng, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024. 3
- [32] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In NeurIPS, 2023. 5, 6, 8, 1, 3
- [33] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for languageconditioned policy learning for long-horizon robot manipulation tasks. RA-L, 7(3):7327–7334, 2022. 1, 2, 4
- [34] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, et al. Open xembodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration. In ICRA, pages 6892– 6903, 2024. 1, 2
- [35] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, et al. Fast: Efficient action tokenization for vision-languageaction models. arXiv preprint arXiv:2501.09747, 2025. 2, 4, 5, 6
- [36] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, et al. Spatialvla: Exploring spatial representations for visual-language-action model. In RSS, 2025. 2, 5, 6
- [37] Erick Rosete-Beas, Oier Mees, Gabriel Kalweit, Joschka Boedecker, and Wolfram Burgard. Latent plans for taskagnostic offline reinforcement learning. In CoRL, pages 1838–1849, 2023. 1

- [38] Sandeep Routray, Hengkai Pan, Unnat Jain, Shikhar Bahl, and Deepak Pathak. Vipra: Video prediction for robot actions. In ICLR, 2026. 3
- [39] Rutav Shah, Roberto Mart´ın-Mart´ın, and Yuke Zhu. Mutex: Learning unified policies from multimodal task specifications. arXiv preprint arXiv:2309.14320, 2023. 1
- [40] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, et al. Motioni2v: Consistent and controllable image-to-video generation with explicit motion modeling. In ACM SIGGRAPH, pages 1–11, 2024. 3
- [41] Shibo Sun, Xue Li, Donglin Di, Mingjie Wei, Lanshun Nie, Wei-Nan Zhang, Dechen Zhan, Yang Song, and Lei Fan. Llapa: A vision-language model framework for counterfactual-aware procedural planning. In ACM MM, pages 5020–5029, 2025. 2
- [42] Zhenhong Sun, Junyan Wang, Zhiyu Tan, Daoyi Dong, Hailan Ma, Hao Li, and Dong Gong. Eggen: Image generation with multi-entity prior learning through entity guidance. In ACM MM, pages 6637–6645, 2024. 3
- [43] Homer Rich Walke, Kevin Black, Tony Z Zhao, et al. Bridgedata v2: A dataset for robot learning at scale. In CoRL, pages 1723–1736, 2023. 5, 1, 4
- [44] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 3
- [45] Junyan Wang, Zhenhong Sun, Zhiyu Tan, Xuanbai Chen, Weihua Chen, Hao Li, Cheng Zhang, and Yang Song. Towards effective usage of human-centric priors in diffusion models for text-based human image generation. In CVPR, pages 8446–8455, 2024. 3
- [46] Kun Wang, Xiao Feng, Mingcheng Qu, and Tonghua Su. Hmvla: Hyperbolic multimodal fusion for vision-languageaction models. arXiv preprint arXiv:2602.02533, 2026. 2
- [47] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 5, 1
- [48] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In CVPR, pages 14749–14759, 2024. 3
- [49] Yuchi Wang, Junliang Guo, Xinyi Xie, Tianyu He, Xu Sun, and Jiang Bian. Vidtwin: Video vae with decoupled structure and dynamics. In CVPR, pages 22922–22932, 2025. 3, 4, 5, 1
- [50] Yuqi Wang, Xinghang Li, Wenxuan Wang, Junbo Zhang, Yingyan Li, Yuntao Chen, Xinlong Wang, and Zhaoxiang Zhang. Unified vision-language-action model. In ICLR,

2026. 2, 3, 5, 6, 1

- [51] Julong Wei, Shanshuai Yuan, Pengfei Li, Qingda Hu, Zhongxue Gan, and Wenchao Ding. Occllama: An occupancy-language-action generative world model for autonomous driving. arXiv preprint arXiv:2409.03272, 2024. 3
- [52] Hongtao Wu et al. Unleashing large-scale video generative pre-training for visual robot manipulation. In ICLR, 2024. 3
- [53] Jialong Wu, Shaofeng Yin, Ningya Feng, Xu He, Dong Li, Jianye Hao, and Mingsheng Long. iVideoGPT: Interactive

- videogpts are scalable world models. In NeurIPS, pages 68082–68119, 2024. 3
- [54] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, et al. Latent action pretraining from videos. In ICLR, 2025. 2, 3, 6
- [55] Sihyun Yu, Weili Nie, De-An Huang, Boyi Li, Jinwoo Shin, and Anima Anandkumar. Efficient video diffusion models via content-frame motion-latent decomposition. In ICLR,

2024. 3

- [56] Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, et al. Dreamvla: A vision-language-action model dreamed with comprehensive world knowledge. In NeurIPS, 2025. 3
- [57] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, et al. Cot-VLA: Visual chain-ofthought reasoning for vision-language-action models. In CVPR, pages 1702–1713, 2025. 3, 6
- [58] Zhide Zhong, Haodong Yan, Junfeng Li, Xiangchen Liu, Xin Gong, et al. Flowvla: Visual chain of thought-based motion reasoning for vision-language-action models. arXiv preprint arXiv:2508.18269, 2025. 2, 3, 6, 1
- [59] Gaoyue Zhou, Victoria Dean, Mohan Kumar Srirama, Aravind Rajeswaran, Jyothish Pari, Kyle Hatch, Aryan Jain, Tianhe Yu, Pieter Abbeel, Lerrel Pinto, et al. Train offline, test online: A real robot learning benchmark. arXiv preprint arXiv:2306.00942, 2023. 1
- [60] Yifeng Zhu, Abhishek Joshi, Peter Stone, and Yuke Zhu. Viola: Imitation learning for vision-based manipulation with object proposal priors. In CoRL, pages 1199–1210, 2023. 1
- [61] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In CoRL, pages 2165–2183, 2023. 1, 2

## Chain of World: World Model Thinking in Latent Motion Supplementary Material

### 1. Implementation Details

#### 1.1. Datasets

We collected high-quality robot manipulation data for finetuning the Latent Motion Extractor (LME) and training the VLA, with the datasets summarized in Table 1. Most of the data comes from the OXE [34] dataset, and we additionally include the Calvin [33] and Libero [32] simulation datasets. For LME fine-tuning, we use only episode frames. In the VLA pre-training stage, we use both episode frames and text instructions. Following UniVLA [50], we adopt different sampling intervals for each dataset to ensure that the temporal gap between keyframes is approximately one second. We then uniformly sample 16 frames from the continuous frames covered by six keyframes for pre-training. Throughout this stage, only third-person view data is used, excluding wrist-camera views.

During the VLA co-fine-tuning stage, we train on the benchmark-specific training sets using text instructions, frames, and actions. For example, the BridgeV2 dataset [43] is used for the SimplerEnv-Bridge evaluation [30], while the Libero [32] evaluation uses the mixed data of four Libero task suites processed by OpenVLA [24]. In addition, the appendix includes extended experiments using the Fractal dataset [4] for the SimplerGoogle Robot [30] evaluation and the Calvin dataset [33] for the Calvin evaluation, covering both ABCD→D and ABC→D task settings. Across the co-fine-tuning experiments, Bridge and Google Robot training use only thirdperson views, while Libero and Calvin use both thirdperson and wrist views.

#### 1.2. Training Details

For LME fine-tuning, we start from the VidTwin [49] pretrained model and fine-tune it on the video data from the datasets listed in Table 1. We use 4 A800 GPUs with a per-GPU batch size of 4, randomly sampling 16 frames per video. Each frame is resized to 224×224. The KL loss weight is set to 1e-6, and the reconstruction loss is reduced using the mean over all elements rather than the default reduction over the batch dimension only. We randomly sample 1000 videos from the training set as a validation set and select the checkpoint with the lowest reconstruction loss. The final model corresponds to the checkpoint trained for one epoch plus 20k iterations.

For VLA pre-training, we initialize from the 8.5B Emu3 [47] pretrained checkpoint and train on the datasets in Table 1. The training is performed on 32 A800 GPUs with a per-GPU batch size of 8. Image observations are re-

Table 1. Training datasets.

Dataset Name Count Berkeley Autolab Ur5 [10] 892 Bridgev2 [43] 24879 Cmu Play Fusion [9] 576 Fractal [4] 65530 Kuka [23] 84202 Maniskill [20] 30029 Taco Play [37] 3242 Toto [59] 899 Utaustin Mutex [39] 1500 Viola [60] 135 Calvin [33] 22966 Libero [32] 1693 Total 236543

sized to 256×256. We use the first and last frames of each video clip together with one learnable motion query, and the maximum sequence length is set to 2500 tokens. We train for 10k iterations in total, which takes roughly 24 hours.

For VLA co-fine-tuning, we follow the evaluation protocols from UniVLA [50] for each benchmark. We load the checkpoint from the VLA pre-training stage and train with 16 A800 GPUs, using a batch size of 8 per GPU and fullparameter fine-tuning. The maximum sequence length is set to 3200 tokens. For SimplerEnv-Windowx [30], we use BridgeV2 [43] data with images resized to 256×256 and train for 12k iterations. For SimplerEnv-Google Robot [30], Fractal [4] images are resized to 240×192, and training continues for 16k iterations. For Libero [32], images are resized to 200×200, and training runs for 8k iterations. For Calvin [33], third-person views are resized to 200×200 and wrist views to 80×80, with training conducted for 12k iterations. The per-iteration training time across these configurations is similar; for example, Libero training takes about 25 hours for 8k iterations. Overall, each configuration requires roughly one to two days of training.

#### 1.3. Interpretation of the World Model and the Latent Motion Chain

Our method combines a world model formulation with latent action modeling. The world model component consists of two stages: pre-training and co-fine-tuning. During pre-training, the world model is not action-conditioned. This follows the representation adopted by UniVLA [50] and FlowVLA [58], where the world model predicts future

Table 2. Long-horizon robotic manipulation evaluation on the CALVIN [33] benchmark. Methods marked with † are from our reimplementation.

Tasks Completed in a Row

Avg. Len ↑ 1 2 3 4 5

Method Task

0.988 0.934 0.883 0.829 0.764 4.398 Ours 0.972 0.939 0.894 0.859 0.809 4.473 TLA [6]

UniVLA† [50]

ABCD→D

0.955 0.858 0.754 0.669 0.565 3.800 Dita [21] 0.945 0.825 0.728 0.613 0.500 3.610 UniVLA† [50] 0.972 0.902 0.826 0.741 0.661 4.102 Ours 0.968 0.912 0.844 0.779 0.708 4.211

ABC→D

Table 3. Evaluation on SimplerEnv-Google Robot [30] across various manipulation tasks.

Model Pick Move Drawer Place Average

OpenVLA [24] 0.180 0.563 0.630 0.000 0.343 SpatialVLA [36] 0.860 0.779 0.574 0.090 0.576 MoTo [12] 0.740 0.604 0.431 0.000 0.444 villa-X [11] 0.987 0.750 0.593 0.056 0.597 UniVLA [50] 0.870 0.565 0.194 0.167 0.449

Ours 0.923 0.676 0.428 0.407 0.609

- 0.95
- 1.00

SuccessRate

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

5 10 15 20 25

Action Chunk Size (la)

0.55

0.60

0.65

0.70

0.75

0.80

0.85

0.90

0.95

1.00

Spatial Object

Goal Long Average

Figure 1. Sensitivity analysis of N and la on LIBERO.

environment evolution given a language instruction and an initial state, rather than explicit actions. During the co-finetuning stage, we introduce an action-conditioned formulation: p(vt+1 | vt,At).

Our latent motion does not explicitly perform multi-step rollouts. Instead, it provides a continuous and decoupled motion encoding over a temporal window, which can be interpreted as an implicit motion chain.

- 2. Additional Results

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.90

0.85

0.80

0.75

0.70

0.65

0.60

0.55

1 2 3 4 5

Sparse Keyframes (N)

#### 2.1. Analysis of keyframes and action chunk size

We evaluate the number of sparse keyframes N ∈ {1,2,3,4,5} and action chunk sizes la ∈{5,10,20,25} on LIBERO to understand the temporal granularity required by latent motion reasoning. As shown in Figure 1, both hyperparameters exhibit a clear inverted-U trend. The best performance is achieved at (N = 2,la = 10), corresponding

to a ∼20-frame (≈2s) temporal horizon.

When using only one keyframe (N = 1), performance drops significantly across all suites, especially on longhorizon tasks, indicating that the latent motion becomes under-constrained. Increasing N to 2 provides sufficient visual anchoring and yields the largest improvement. However, further increasing N gradually degrades performance. With dense observations, the model can rely on shortterm visual matching instead of inferring motion dynamics, weakening the benefit of latent temporal reasoning.

A similar phenomenon appears for action chunk size. Small chunks (la = 5) reduce temporal abstraction and make the policy closer to step-wise imitation. Large chunks (la ≥ 20) introduce high uncertainty in future evolution, particularly harming the long-horizon tasks. The intermediate chunk size (la = 10) achieves the best trade-off between predictability and abstraction.

Overall, the results suggest that the proposed model performs best when sparse observations provide partial constraints while still requiring the model to infer continuous evolution. This supports our design motivation: the latent motion token serves as a dynamics aggregator over a medium temporal window rather than dense frame tracking or one-step prediction.

#### 2.2. Comparison with other Video VAE

To further analyze the role of latent motion representations, we replace VidTwin with the VAE from Wan 2.1 [44] and

Table 4. Comparison between our latent motion representation and Wan 2.1 VAE latent z on LIBERO.

Variant Pre-training Co-fine-tuning Spatial Object Goal Long Average

Ours latent motion + terminal frame + latent motion 0.948 0.974 0.958 0.906 0.947 Wan2.1 VAE [44] latent z + terminal frame + latent z 0.938 0.950 0.922 0.868 0.920

[Figure 63]

[Figure 64]

Figure 2. Cross-Recon visualization on LIBERO [32]. The first six columns show temporally sampled frames from three rows: Structure (top), Motion (middle), and Cross-Recon (bottom). The Cross-Recon videos are generated by combining the static appearance from the Structure video with the motion representation extracted from the Motion video, revealing the transferred motion patterns. Each CrossRecon frame is overlaid with a motion heatmap to highlight dynamic regions. The last column presents three summary maps: motion heatmaps obtained by averaging and maximizing per-frame absolute differences between Cross-Recon and Structure, and the end-effector trajectory estimated from the motion regions.

conduct a controlled comparison. Specifically, we use the latent z extracted by the Wan 2.1 VAE as auxiliary supervi-

[Figure 65]

[Figure 66]

Figure 3. Cross-Recon visualization on SimplerEnv [30] and Bridgev2 [43].

sion during both pre-training and co-fine-tuning.

The Wan 2.1 VAE is trained on large-scale video data and therefore incorporates rich generic video priors. As shown in Table 4, this variant achieves an average success rate of 0.920 on LIBERO. While competitive, it remains inferior to our latent motion design (0.947).

#### 2.3. CALVIN

Calvin [33] is an open-source simulated benchmark built on PyBullet, designed for learning long-horizon, languageconditioned robotic manipulation tasks. It provides a tabletop simulation environment containing 23 types of manip-

ulation skills, such as lifting, pushing, rotating, and object relocation. These skills must be executed in sequence to complete multi-step tasks, introducing substantial uncertainty and randomness, which makes Calvin a highly challenging evaluation benchmark. The dataset includes a large number of expert demonstrations and is organized into multiple subsets. In our experiments, we use the ABCD→D and ABC→D subsets, and during training, we only utilize demonstrations that include natural language descriptions of the actions. Following the official evaluation protocol, all tests consist of 1000 episodes, each containing a sequence of five sub-tasks specified by natural language instructions.

Cluster 1 Cluster 2 Cluster 3 Cluster 4

[Figure 67]

- (a) Unsupervised clustering of latent-motion trajectories.
- (b) Representative video clips sampled from motion clusters.

Cluster 1 Cluster 2

Cluster 3 Cluster 4

[Figure 68]

[Figure 69]

Figure 4. Visualization of latent-motion clusters and corresponding video examples. (a) Unsupervised clustering results of clip-level motion trajectories. Each subplot shows the average 2D motion trajectory (obtained from the first two PCA components of the accumulated framewise motion deltas) for one cluster. (b) Representative video examples from clusters. Cluster 1 and 2 correspond to monotonic downwardlike or upward-like motions, whereas Cluster 3 and 4 exhibit rightward-like or leftward-like behaviors.

Task: pick up the chocolate pudding and place it in the basket

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

𝑧

Task: pour

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

𝑧

Task: open fridge

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

𝑧

Task: put the banana inside the drawer

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

𝑧

(c) predict 𝑧 and one terminal frame

(b) predict one goal frame

(a) predict five frames

Figure 5. Comparative visualization of future-frame prediction strategies.

[Figure 94]

Figure 6. An Intel RealSense camera and a Realman RM75B robot.

The main results are presented in Table 2. Our method achieves an average success length of 4.473 on the ABCD→D task and 4.211 on the ABC→D task. For a fair comparison, we reproduced UniVLA [50] using the training sets listed in Table 1, and followed a fine-tuning setup with 16 A800 GPUs and a per-GPU batch size of 8. Under the same training configuration, our approach outperforms UniVLA [50].

#### 2.4. SimplerEnv-Google Robot

We also evaluate our method on the SimplerEnv-Google Robot benchmark. The evaluation primarily follows the visual matching protocol, which assesses the alignment between real and simulated visual appearances by overlaying real-world images onto simulated backgrounds and adjusting the textures of foreground objects and the robot within the simulator. This benchmark includes four tasks: pick coke can, move near, open/close drawer, and place in closed drawer.

The main results are shown in Table 3. Our method achieves an average success rate of 0.609, outperforming UniVLA [50], villa-x [11], MoTo [12], and other baselines. Here, UniVLA refers to our reproduction. Our method surpasses UniVLA on all four tasks and shows a particularly large improvement on the place in closed drawer task.

#### 2.5. More Visualization

We provide extended visualizations for the latent motion analysis presented in Section 4.4, with the main results shown in Figures 2, 3, 4, and 5.

Effective decoupling of structure and motion latents. Figures 2 and 3 analyze representative samples from the Libero and Bridge datasets. The first six columns display temporally sampled frames from three rows: Structure (top), Motion (middle), and Cross-Reconstruction (bottom). The Cross-Recon videos are synthesized by combining the static appearance from the Structure video with the motion representation extracted from the Motion video,

thereby revealing transferred motion patterns. Each CrossRecon frame is overlaid with a motion heatmap to highlight dynamic regions. The final column summarizes three diagnostic maps: motion heatmaps computed by averaging and maximizing the per-frame absolute differences between Cross-Recon and Structure, as well as the end-effector trajectory estimated from the activated motion regions. As shown, the highlighted areas consistently follow the movement of the robot arm in the Motion video. In the video results, these regions fluctuate over time; for clarity in static visualization, we display aggregated highlights in the figures.

We further analyze the distribution of motion latents, as shown in Figure 4. To derive an interpretable trajectory representation from high-dimensional motion latents, we first extract per-frame motion features from each video clip and accumulate framewise differences to obtain a temporal sequence describing the overall motion trend of the clip. These sequences are then resampled to a fixed length across all clips and standardized globally. We subsequently apply PCA to the sequence features and take the first two principal components as a 2D trajectory for each clip. This representation preserves the dynamic structure encoded in the latent space while enabling clear comparison across clips.

Figure 4 (a) shows unsupervised clustering of all motion trajectories in the 2D PCA space. To obtain cluster-level canonical shapes, we temporally align trajectories within each cluster via resampling and plot their mean curves along with 95% confidence intervals. Distinct trajectory patterns emerge across clusters—such as monotonic rises, two-stage reversals, and multi-phase back-and-forth motions—indicating that the model’s motion latent captures high-level motion semantics. To further validate the semantic consistency within each cluster, we randomly sample two video clips per cluster and visualize three uniformly sampled frames from each clip, as shown in Figure 4 (b). The clips within the same cluster exhibit highly similar motion trends in appearance, confirming that the structure of the motion-latent space yields meaningful discrimination among different action patterns.

Motion latent enhances dynamic modeling for future frame prediction. As shown in Figure 5, we further visualize future frame predictions under different pretraining strategies. From top to bottom, the examples correspond to four tasks: i) pick up the chocolate pudding and place it in the basket, ii) pour, iii) open the fridge, and iv) put the banana inside the drawer. In Figure 5 (a), worldmodel-based approaches suffer from reconstructing redundant background pixels, which can draw attention away from critical interactions and motion cues. As a result, the predicted future frames sometimes remain nearly unchanged, such as in tasks (ii) and (iii). Figure 5 (b) shows that predicting only the target frame often leads to unsta-

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

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

[Figure 118]

(a) collected data (b) real-world deployment

Figure 7. Comparison between data collection and real-world deployment during testing.

ble generation due to the absence of intermediate evolution steps: in task (i), the target frame nearly collapses back to the initial frame, and in task (iii), only one door of the fridge is generated. In contrast, our method leverages the motion latent zm as a chain-of-thought for motion, providing stronger guidance for future-frame prediction. The generated final frames align more accurately with the intended task instructions.

### 3. Real-Robot Experiments

Experimental Setup. As shown in Figure 6, we use the Realman RM75B robot, which is equipped with 7 degrees of freedom and a single gripper. An Intel RealSense camera is used to capture RGB images. We set up a cup-grasping experiment and collected a total of 127 episodes, consisting of 65,382 frames with corresponding actions. Each episode contains an average of 515 frames, corresponding to approximately 20 seconds in the real world. The dataset

mainly includes grasping cups of four different colors, with the number of episodes per color as follows: red 31, blue 39, yellow 24, and purple 33. Figure 7 (a) shows some collected data.

During training, all images are cropped and resized to 256×256. The action chunk size is set to 10. We train the model for 2k steps using 16 GPUs with a per-GPU batch size of 8. The data were collected in the afternoon and evening and then used for model training. Testing was conducted the following day. As shown in Figure 7, the lighting conditions have some differences between data collection compared and during real-world deployment. We found that the model was still able to correctly execute instructions under different lighting conditions. Figure 7 (b) shows in the first two rows two test cases: grasping a red/purple cup and placing it on a plate. Their background lighting differs from the training data, but the model is still able to execute the tasks successfully.

