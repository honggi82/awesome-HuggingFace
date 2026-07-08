# arXiv:2405.15223v3[cs.CV]31Oct2024

## iVideoGPT: Interactive VideoGPTs are Scalable World Models

Jialong Wu1∗, Shaofeng Yin1,2∗, Ningya Feng1, Xu He3, Dong Li3, Jianye Hao3,4, Mingsheng Long1

1School of Software, BNRist, Tsinghua University, 2Zhili College, Tsinghua University 3Huawei Noah’s Ark Lab, 4College of Intelligence and Computing, Tianjin University

wujialong0229@gmail.com, ysf22@mails.tsinghua.edu.cn, mingsheng@tsinghua.edu.cn

### Abstract

World models empower model-based agents to interactively explore, reason, and plan within imagined environments for real-world decision-making. However, the high demand for interactivity poses challenges in harnessing recent advancements in video generative models for developing world models at scale. This work introduces Interactive VideoGPT (iVideoGPT), a scalable autoregressive transformer framework that integrates multimodal signals—visual observations, actions, and rewards—into a sequence of tokens, facilitating an interactive experience of agents via next-token prediction. iVideoGPT features a novel compressive tokenization technique that efficiently discretizes high-dimensional visual observations. Leveraging its scalable architecture, we are able to pre-train iVideoGPT on millions of human and robotic manipulation trajectories, establishing a versatile foundation that is adaptable to serve as interactive world models for a wide range of downstream tasks. These include action-conditioned video prediction, visual planning, and modelbased reinforcement learning, where iVideoGPT achieves competitive performance compared with state-of-the-art methods. Our work advances the development of interactive general world models, bridging the gap between generative video models and practical model-based reinforcement learning applications. Code and pre-trained models are available at https://thuml.github.io/iVideoGPT.

### 1 Introduction

Recent years have witnessed remarkable advancements in generative models of multimodal contents, including text [1], audio [9], and images [22], with video generation now emerging as a new frontier [11]. A particularly significant application of these generative video models, learned in an unsupervised way on diverse Internet-scale data, is to construct predictive world models [53, 28] at scale. These world models are expected to accumulate commonsense knowledge about how the world works, enabling the prediction of potential future outcomes (e.g., visual observations and reward signals) based on the actions of agents. By leveraging these world models, agents employing model-based reinforcement learning (RL) can imagine, reason, and plan inside world models [20, 29], thus acquiring new skills more safely and efficiently with a handful of trials in the real world.

Despite the fundamental connection, significant gaps remain between generative models for video generation and visual world models for agent learning. One primary challenge is achieving the best of both interactivity and scalability. In model-based RL, world models predominantly utilize recurrent network architecture. This design naturally allows the transition of observations or latent states conditioned on actions in each step, facilitating interactive behavior learning [29, 80, 34]. However, these recurrent models mostly focus on games or simulated environments with simple

∗Equal Contribution

38th Conference on Neural Information Processing Systems (NeurIPS 2024).

[Figure 1]

[Figure 2]

[Figure 3]

Video Prediction

Human Manipulation Trajectories

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

###### iVideoGPT

Visual Planning

Robot Manipulation Trajectories

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Visual Model-based RL

- Figure 1: Practical applications of iVideoGPT, which is designed to scale, allowing pre-training on millions of human and robotic manipulation trajectories. This results in a single, versatile foundation of interactive world models, adaptable to a wide range of downstream tasks.

visuals and have limited capability to model complex, in-the-wild data at scale [48, 81]. On the other hand, Internet-scale video generative models [37, 7, 11] can synthesize realistic long videos that are controllable via text descriptions [109] or future action sequences [101] at the beginning of generation. Although suitable for high-level planning [19], their trajectory-level interactivity does not provide sufficient granularity needed by agents to intervene step-by-step during the simulation to learn precise basic skills efficiently. This dilemma naturally raises the question:

How can we leverage the advancements in scalable video generative models for developing interactive visual world models?

In this work, we explore world models that are both interactive and scalable within a GPT-like autoregressive transformer framework [90, 75]. Pioneering efforts have been made recently through diffusion models [102] and masked generative models [12]. Nevertheless, utilizing autoregressive transformers offers distinct advantages such as seamless integration with the established Large Language Model (LLM) ecosystem [110] and greater flexibility in handling diverse conditions without the need for specific architectural modifications like adapter modules [77, 107]. We present Interactive VideoGPT (iVideoGPT), a scalable world model architecture that incorporates multimodal signals, including visual observations, actions, and rewards, in an interactively autoregressive manner. Unlike multimodal LLMs that discretize visual observations into tokens frame-by-frame using image tokenizers [55], a key innovation of iVideoGPT for enhancing scalability is to learn compressive tokenization for each observation conditioned on rich contextual observations, achieving an asymptotic 16× reduction in token sequence length. We highlight that more compact video tokenization could not only facilitate more efficient training and generation but also enhance video quality. This is achieved by decoupling context from dynamics, allowing the model to focus on predicting the motion of objects while maintaining temporal consistency within the scene [99].

We demonstrate a series of practical applications of iVideoGPT for visual robotic manipulation,

- as illustrated in Figure 1. Mirroring the two-phase approach popularized by LLMs, our method involves pre-training followed by domain-specific adaptation. During pre-training, iVideoGPT is scalable for action-free video prediction across a mixture of over one million robotic and human manipulation trajectories [70, 25]. The pre-trained iVideoGPT serves as a single, adaptable foundation of interactive world models for various downstream tasks, such as action-conditioned video prediction [21, 16], visual planning [86], and visual model-based RL [105]. Additionally, we showcase the pre-trained transformer’s preliminary zero-shot video generation capability without fine-tuning, requiring only tokenizer adaptation for unseen domains. We further explore a variant of iVideoGPT for goal-conditioned video prediction, underscoring the flexibility of sequence modeling. The main contributions of this work can be summarized as follows:

- • We introduce Interactive VideoGPT (iVideoGPT), an autoregressive transformer architecture for scalable world models, which features compressive tokenization for visual observations.
- • We pre-train iVideoGPT on a large-scale dataset comprising millions of robotic and human manipulation trajectories and adapt it to domain-specific tasks. The pre-trained models have been publicly available to encourage further research.
- • Extensive experiments covering video prediction, visual planning, and visual model-based RL demonstrate that iVideoGPT can simulate accurate and realistic experiences and provide competitive performance compared with state-of-the-art methods.

𝑜 :  𝑜 𝑜 𝑜

𝑜 𝑜 𝑜

…

…

…

| | | | |
|---|---|---|---|
|Recurrent|model| | |

R

Transformer or Diffusion model

Autoregressive Transformer

…

𝑎 𝑎 𝑎 𝑎 :   

… …

…

𝑎 𝑎 𝑎

𝑜 𝑜 tokens / latent features 𝑜

(a) Recurrent world models (b) Video generation models (c) Interactive video prediction models

- Figure 2: Conceptual comparison among architectures, illustrated using a single context frame

(T0 = 1) for simplicity. (a) Recurrent architectures for world models like Dreamer [29] and MuZero [80] provide step-level interactivity but limited scalability. (b) Recent video generation advancements like VideoGPT [101] and Stable Video Diffusion [8, 7] use non-causal temporal modules that can only offer trajectory-level interactivity. (c) Our model utilizes an autoregressive transformer that separately maps each step into a sequence of tokens, achieving both scalability and interactivity.

### 2 Problem Formulation

A world model is an internal model learned by the agent to simulate the environment. This environment is typically modeled as a partially observable Markov decision process (POMDP) (S,O,ϕ,A,p,r,γ). At each step, st ∈ S represents the underlying state of the environment, and ot = ϕ(st) is the observation received by the agent, only providing incomplete information of st. After taking an action at ∈ A, p(st+1|st,at) defines the transition probability from state st to st+1. The agent also receives immediate rewards rt+1 = r(st,at), and aim to learn a policy π such that at ∼ π(o1:t) maximizing the γ-discounted accumulated rewards Ep,π[ t γt−1rt].

While world models can be learned from many types of data, video is one modality that is taskagnostic, widely available, and embeds broad knowledge that can be learned in a self-supervised way. Thus, we formulate learning world models for visual control as an interactive video prediction problem [102, 12] where O = RH×W×3 is the space of video frames2. Concretely, given a short history visual observations of T0 frames o1:T

, at each step t = T0,...,T − 1, the agent takes an action at based on its policy and previous imagined observations, and then the world model need to approximate and sample the transition p(ot+1,rt+1 | o1:t,aT

0

0:t) to feedback the agent.

As depicted in Figure 2, a majority of advanced video generation models [101, 8, 104], including VideoGPT, can not deal with the interactive video prediction problem because they design non-causal modules fusing information along the temporal dimension, lacking the ability for causal, intermediate action control during generation (see extended discussion in Appendix C.2). Existing world models in the literature of MBRL [29, 80], such as Dreamer, utilize recurrent architecture but lack scalability.

### 3 Interactive VideoGPT

In this section, we introduce Interactive VideoGPT, a scalable world model architecture with great flexibility to integrate multimodal signals, including visual observations, actions, rewards, and other potential sensory inputs. At its core, iVideoGPT consists of a compressive tokenizer to discretize video frames and an autoregressive transformer predicting subsequent tokens (Section 3.1). This model can acquire common knowledge of motions and interactions in various scenes through pretraining on diverse human and robotic manipulation videos (Section 3.2) and then effectively transfer to downstream tasks incorporating additional modalities (Section 3.3).

#### 3.1 Architecture

Compressive tokenization. Transformers particularly excel in operating over sequences of discrete tokens. VQGAN [22] is a commonly used visual tokenizer that converts from raw pixels to discrete tokens. Instead of using an image tokenizer to discretize each frame independently [55, 63, 27], leading to rapidly increasing sequence lengths, or using a 3D tokenizer that compresses videos

2Due to this connection, we use the terms "video frame" and "visual observation" interchangeably.

[Figure 17]

[Figure 18]

𝑜 𝑟 𝑜 𝑟 𝑜 𝑟

|[Figure 19]|
|---|

|[Figure 20]|
|---|

|[Figure 21]|
|---|

|[Figure 22]|
|---|

[Figure 23]

Reward head

Reward head

Reward head

Detokenize

[S] [S] [S]

Decoder

Decoder

| | | | |
|---|---|---|---|
| | | | |

[Figure 24]

iVideoGPT

Encoder

Encoder

[S] [S] [S]

| | |
|---|---|
|[Figure 25]| |

[Figure 26]

Tokenize

[Figure 27]

|[Figure 28]|
|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

[Figure 32]

Action embed.

Action embed.

Action embed.

𝑜 : 

𝑜 𝑎 𝑜 𝑎 𝑜 𝑎 𝑜

𝑜

- (a) Compressive tokenization (b) Interactive prediction with Transformers

Figure 3: Architecture of iVideoGPT, simplified to show only a single context frame (T0 = 1). (a) Compressive tokenization utilizes a conditional VQGAN that discretizes future frames conditioned on context frames to handle temporal redundancy, significantly reducing the number of video tokens.

- (b) An autoregressive transformer integrates multimodal signals—visual observations, actions, and rewards—into a sequence of tokens, enabling interactive agent experiences through next-token prediction. Actions and rewards are optional and not included in action-free video pre-training.

spatiotemporally at the expense of interactivity [101, 104], we propose to tokenize videos with a novel conditional VQGAN consisting of dual encoders and decoders {(Ec,Dc),(Ep,Dp)}. As illustrated in Figure 3a, initial context frames o1:T

, rich in contextual information, are independently

0

tokenized and reconstructed through N tokens: zt(1:N) = Ec(ot),oˆt = Dc(zt) for t = 1,...,T0. In contrast, due to the temporal redundancy between context and future frames, only essential dynamics information, such as the position and pose of moving objects, needs to be encoded. This is achieved using a conditional encoder and decoder, which require a far smaller number of n tokens (n ≪ N):

zt(1:n) = Ep(ot|o1:T

##### ) for t = T0 + 1,...,T. (1)

##### ),oˆt = Dp(zt|o1:T

0

0

We implement this conditioning mechanism using cross-attention between multi-scale feature maps (see details in Appendix A.1). Overall, the proposed tokenizer is trained with the following objective:

T0

T

)), (2)

LVQGAN(ot;Ec(·),Dc(·)) +

LVQGAN(ot;Ep(·|o1:T

##### ),Dp(·|o1:T

Ltokenizer =

0

0

t=1

t=T0+1

where LVQGAN(o;E,D) is a combination of a L1 reconstruction loss, a commitment loss [89], a perceptual loss [44], and optionally an adversarial loss [22].

There are primarily two benefits of the proposed tokenization. First, it significantly reduces the sequence length of tokenized videos, which grows linearly with the number of frames but at a much smaller rate n. In this work, we set N = 16 × 16 and n = 4 × 4, resulting in an asymptotic reduction of 16×, facilitating faster rollouts for model-based planning and reinforcement learning. Second, by conditional encoding, transformers predicting subsequent tokens can maintain temporal consistency of the context much easier and focus on modeling essential dynamics information [99]. We discuss the assumptions and limitations of our tokenization in Section 6.

Interactive prediction with Transformers. After tokenization, the video is flattened into a sequence of tokens: x = (z1(1),...,z1(N),[S],z2(1),...,z2(N),...,[S],zT(1)

##### 0+1,...,zT(n)

0+1,...) with a length of L = (N + 1)T0 + (n + 1)(T − T0) − 1. Special slot tokens [S] are inserted to delineate frame boundaries and facilitate the integration of extra low-dimensional modalities such as actions (see Section 3.3 for details). As Figure 3b, a GPT-like autoregressive transformer is utilized for interactive video prediction through next-token generation frame-by-frame. In this work, we take the model size of GPT-2 [76] but adopt the LLaMA architecture [87] in order to embrace the latest innovations for LLM architecture, applying pre-normalization using RMSNorm [106], SwiGLU activation function [83], and rotary positional embeddings [85].

#### 3.2 Pre-Training

Large language models can gain extensive knowledge from Internet text in a self-supervised way via next-word prediction. Similarly, the action-free video pre-training paradigm for world models [81, 99, 62] involves video prediction as a pre-training objective, providing Internet-scale supervision with physical world knowledge absent in LLMs. We pre-train iVideoGPT on this generic objective, applying a cross-entropy loss to predict subsequent video tokens:

Lpre-train = −

L i=(N+1)T0+1

log p(xi|x<i), (3)

where L is the total sequence length and (N + 1)T0 + 1 marks the first token index of the frames to be predicted. Notably, we do not train iVideoGPT to generate context frames, making its capacity focus on dynamics information, as previously discussed.

Pre-training data. While there are numerous videos available on the Internet, due to computational limitations, we specifically pre-train iVideoGPT for the robotic manipulation domain. We leverage a mixture of 35 datasets from the Open X-Embodiment (OXE) dataset [70] and the SomethingSomething v2 (SSv2) dataset [25], totaling 1.4 million trajectories (see Appendix A.2 for details). OXE is a diverse collection of robot learning datasets from a variety of robot embodiments, scenes, and tasks. These datasets are highly heterogeneous but can be easily unified in the action-free video prediction task. To further enhance the diversity, we also include SSv2, a dataset of humanobject interaction videos, as previous work has demonstrated knowledge transfer from these human manipulation videos for learning a world model for robotic manipulation tasks [99, 62].

Flexibility of sequence modeling. A sequence of tokens provides a flexible way to specify tasks, inputs, and outputs [60, 76]. To preliminarily showcase this flexibility, we introduce a variant of iVideoGPT for goal-conditioned video prediction: p(oT

,oT), where the model predicts a video sequence reaching a specified goal observation oT. This is simply achieved by rearranging the frame sequence as o˜1:T = (oT,o1,o2,...,oT−1) while keeping the architecture and training procedure consistent as above (see details in Appendix A.2). Qualitative results of goal-conditioned prediction are shown in Figure 4, with further exploration left for future work3.

0+1:T|o1:T

0

#### 3.3 Fine-Tuning

Action conditioning & reward prediction. Our architecture is also designed to flexibly incorporate additional modalities for learning interactive world models, as illustrated in Figure 3b. Actions are integrated by linear projection and adding to the slot token embeddings. For reward prediction, instead of learning independent reward predictors, we add a linear head to the last token’s hidden state of each observation. This multi-task learning approach can enhance the model’s focus on task-relevant information, thereby improving prediction accuracy for control tasks [57]. We use a mean-squared error loss for reward prediction in addition to the cross-entropy loss in Eq. (3).

Tokenizer adaptation. We choose to update the full model, including the tokenizer, for downstream tasks, finding this strategy more effective than parameter-efficient fine-tuning methods [39]. This is likely due to the limited diversity of our pre-trained data compared to Internet-scale images, which, while extensive, may also not adequately cover specific real-world applications like robotics. Minimal literature explores adapting a VQGAN tokenizer to domain-specific data. As our tokenization is designed for decoupling dynamics information from context conditions, we hypothesize that while our model may encounter unseen objects like different robot types in downstream tasks, the fundamental knowledge of physics—such as motions and interactions—learned by the transformer from diverse scenes is commonly shared. This hypothesis is supported by our experiments transferring iVideoGPT from mixed pre-training data to the unseen BAIR dataset [21], where the pre-trained transformer can zero-shot generalize to predict natural motions, requiring only the tokenizer to be fine-tuned for unseen robot grippers (see Figure 8). This property is particularly important for scaling GPTlike transformers to large sizes, enabling lightweight alignment across domains while keeping the transformer intact. We leave an in-depth analysis of tokenizer adaptation for future work.

3Unless otherwise specified, action- and goal-free video prediction is used as the default pre-training objective to obtain pre-trained models for all experiments.

###### Open X-Embodiment (action-free)

t = 0 t = 2 t = 4 t = 6 t = 9 t = 12 t = 15

t = 0 t = 2 t = 4 t = 6 t = 9 t = 12 t = 15

[Figure 33]

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

Ground truth

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

(context)

(context)

Predicted

RoboNet t = 0 t = 2 t = 3 t = 5 t = 7 t = 9 t = 11

t = 0 t = 2 t = 3 t = 5 t = 7 t = 9 t = 11

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

Ground truth

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

(context)

(context)

Predicted

VP2 t = 0 t = 2 t = 3 t = 5 t = 7 t = 9 t = 11

t = 0 t = 2 t = 3 t = 5 t = 7 t = 9 t = 11

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Ground truth

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

(context)

(context)

Predicted

###### Open X-Embodiment (goal-conditioned)

t = 0 t = 1 t = 2 t = 3 t = 4 t = 5 t = 7 t = 9 t = 10 t = 11 t = 12 t = 13 t = 14 t = 15

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Ground truth

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

(context) (goal)

Predicted

- Figure 4: Qualitative evaluation: video prediction results of iVideoGPT on Open X-Embodiment, RoboNet, and VP2. Zoom in for details. Extended examples can be found in Appendix B.1.

### 4 Experiments

In this section, we evaluate iVideoGPT in three different control-relevant settings and compare its performance with prior state-of-the-art methods. We demonstrate that iVideoGPT is versatile to provide competitive performance across a range of tasks (Section 4.1, 4.2, and 4.3) and conduct in-depth analysis to understand the tokenization and prediction ability, data efficiency, model scaling, and computational efficiency (Section 4.4). Experimental details can be found in Appendix A.

#### 4.1 Video Prediction

Setup. The BAIR robot pushing dataset [21] consists of 43k training and 256 test videos, where we predict 15 frames from a single initial frame, a standard protocol of prior works. The RoboNet dataset [16] contains 162k videos across 7 robotic arms. Following prior works, we use 256 videos for testing, predicting 10 frames from two frames. Notably, RoboNet overlaps with our pre-training data OXE, from which we have carefully filtered test videos. We compare against a variety of video prediction models, including variational [91, 98, 4], diffusion [93], masked [104, 27], and autoregressive models [101], across four metrics: FVD [88], PSNR [40], SSIM [97], and LPIPS [108].

Results. As shown in Table 1, iVideoGPT provides competitive performance compared to stateof-the-art methods, MAGVIT [104] for BAIR and FitVid [4] for RoboNet, while achieving both interactivity and scalability in its architecture. Initially pre-trained action-free, our model flexibly allows for action-conditioning, which notably improves FVD for BAIR by almost 20%. Although primary experiments are at a low resolution of 64×64, iVideoGPT can be easily extended to 256×256 for RoboNet. We highlight that MaskViT, a prior method leveraging per-frame tokenization, suffers from temporal inconsistency and flicker artifacts in VQGAN reconstructions. Our model, which employs compressive tokenization conditioned on consistent contextual information, improves this and significantly outperforms MaskViT. For qualitative results, refer to Figure 4.

#### 4.2 Visual Planning

Setup. VP2 is a control-centric benchmark [86] that evaluates video prediction models for visual model-predictive control (MPC) [24, 20] across four Robosuite [117] and seven RoboDesk tasks [47].

Table 1: Video prediction results on the BAIR robot pushing and RoboNet datasets. We report the mean and standard deviation for each metric calculated over three runs. "-" marks that the value is not reported in the original papers. LPIPS and SSIM scores are scaled by 100 for convenient display.

BAIR [21] FVD↓ PSNR↑ SSIM↑ LPIPS↓ action-free & 64×64 resolution

RoboNet [16] FVD↓ PSNR↑ SSIM↑ LPIPS↓ action-conditioned & 64×64 resolution

VideoGPT [101] 103.3 - - MaskViT [27] 93.7 - - FitVid [4] 93.6 - - MCVD [93] 89.5 16.9 78.0 MAGVIT [104] 62.0 19.3 78.7 12.3 iVideoGPT (ours) 75.0±0.20 20.4±0.01 82.3±0.05 9.5±0.01

MaskViT [27] 133.5 23.2 80.5 4.2 SVG [91] 123.2 23.9 87.8 6.0 GHVAE [98] 95.2 24.7 89.1 3.6 FitVid [4] 62.5 28.2 89.3 2.4 iVideoGPT (ours) 63.2±0.01 27.8±0.01 90.6±0.02 4.9±0.00

action-conditioned & 256×256 resolution

action-conditioned & 64×64 resolution

MaskViT [27] 211.7 20.4 67.1 17.0 iVideoGPT (ours) 197.9±0.66 23.8±0.00 80.8±0.01 14.7±0.01

MaskViT [27] 70.5 - - iVideoGPT (ours) 60.8±0.08 24.5±0.01 90.2±0.03 5.0±0.01

iVideoGPT (ours) FitVid SVG' MCVD MaskViT Struct-VRNN Simulator

1.0

0.8

SuccessRate

0.6

0.4

0.2

0.0

Robosuite push Flat block Open drawer Open slide Blue button Green button Red button Upright block Aggregate

- Figure 5: Visual MPC results on the VP2 benchmark. We report the mean and min/max performance of iVideoGPT over 3 control runs. On the right, we show the mean scores averaged across all tasks except flat block due to low simulator performance, normalized by the performance of the simulator.

Each environment’s training dataset includes noisy scripted interaction trajectories. Following the protocol from the original benchmark paper, we trained iVideoGPT on 5k trajectories for Robosuite and 35k for RoboDesk, comparing our models with established baselines.

Results. Figure 5 presents the success rates of iVideoGPT compared to baseline models. While Tian et al. [86] observed that excellent perceptual metrics do not always correlate with effective control performance, iVideoGPT outperforms all baselines in two RoboDesk tasks with a large margin and achieves comparable average performance to the strongest model, SVG′ [91]. In Appendix C.3, we analyze iVideoGPT’s suboptimal performance on the open slide task, which is attributed to both limitations of discretization in our model and imperfect built-in reward design of the benchmark.

#### 4.3 Visual Model-based Reinforcement Learning

backpropagate

Dreamer

𝜋 𝜋 𝑣

Setup. We conduct experiments on six robotic manipulation tasks of varying difficulty from MetaWorld [105]. Leveraging iVideoGPT as interactive world models, we have developed a model-based RL method adapted from MBPO [42], which augments the replay buffer with synthetic rollouts to train a standard actor-critic RL algorithm (see Appendix A.5 for the pseudo-code). Our implementation builds upon DrQ-v2 [103], a state-of-the-art visual model-free RL method. We also compare against a state-of-the-art model-based RL algorithm, DreamerV3 [32], with and without world model pre-training [81].

…

Enc

Env

𝒟

Latent Imagination

###### MBPO

(ours)

Env

𝜋 𝑣

Enc

𝒟

iVideoGPT

World model rollouts

Standard model-free RL

Figure 6: Powerful iVideoGPTs enable a simple yet performant MBRL algorithm, decoupling model rollouts and policy learning.

Button Press Topdown Wall

Plate Slide

Hammer

100

100

100

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

SuccessRate(%)

80

80

80

MetaWorld Aggregated

60

60

60

100

||DrQ-v2<br><br>DreamerV3<br><br>DreamerV3 (w/ PT)<br><br>MBPO (w/o PT, ours)<br><br>MBPO (w/ PT, ours)|
|---|
| | | | |
|---|---|---|---|---|
| | | | | |

40

40

40

DrQ-v2

DreamerV3

80

DreamerV3 (w/ PT)

20

20

20

SuccessRate(%)

MBPO (w/o PT, ours)

MBPO (w/ PT, ours)

0

0

0

60

0 1 2 3 4 5 6

0 2 4 6 8 10

0 2 4 6 8 10

Environment Steps (× )

Environment Steps (× )

Environment Steps (× )

Door Lock

Handle Pull Side

Coffee Push

40

100

100

100

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

SuccessRate(%)

80

80

80

20

60

60

60

0

40

40

40

0.0 0.2 0.4 0.6 0.8 1.0

Normalized Environment Steps

20

20

20

0

0

0

0 5 10 15

0 5 10 15 20

0 10 20 30 40

Environment Steps (× )

Environment Steps (× )

Environment Steps (× )

- Figure 7: Visual model-based RL on Meta-world. (Left) Aggregated results report interquartile mean and 95% confidence interval (CI) [2] across a total of 30 runs over six tasks. (Right) Individual results for each task, report mean and 95% CI across five runs, measuring success rates over 20 evaluation episodes. PT denotes pre-training.

Results. Figure 7 shows that our model-based algorithm not only remarkably improves the sample efficiency over its model-free counterpart but also matches or exceeds the performance of DreamerV3. To our knowledge, this reports the first successful application of MBPO to visual continuous control tasks. These results highlight the opportunity, with powerful world models, to eliminate the need for latent imagination—a common strategy used in advanced MBRL systems to train policies on rollouts of latent states within world models [29, 80] (see comparison in Figure 6). Our development of performant MBRL algorithms decouples model and policy learning, where iVideoGPT simply serves as a drop-in replacement of the environment. This can substantially simplify the design space, thereby greatly enhancing the practicality and effectiveness of MBRL algorithms in real-world applications.

Comparison to recurrent world models. We argue that recurrent world models lack the capacity for large-scale pre-training on real-world data—a crucial capability for modern foundation models. To validate this, we pre-train DreamerV3 XL (200M parameters, comparable to iVideoGPT) on the same dataset. As shown in Figure 11 in the Appendix, DreamerV3 fails to capture natural robot dynamics, yielding low-quality, blurred predictions. Further evaluation on the Meta-World benchmark in Figure 7 reveals that DreamerV3 cannot benefit from such ineffective pre-training.

4.4 Model Analysis

Zero-shot prediction. We first analyze the zero-shot video prediction ability of large-scale pretrained iVideoGPT on the unseen BAIR dataset. Interestingly, we observe in the second row of

- Figure 8 that iVideoGPT, without fine-tuning, predicts a natural movement of a robot gripper—albeit a different one from our pre-training dataset. This indicates that while, due to insufficient diversity of pre-training data, our model has a limited ability of zero-shot generalization to completely unseen robots, it effectively separates scene context from motion dynamics. In contrast, with an adapted tokenizer, the transformer that is not fine-tuned itself successfully transfers the pre-trained knowledge and predicts movements for the new robot type in the third row, providing a similar perceptual quality as the fully fine-tuned transformer in the fourth row. Quantitative results can be found in Figure 9a.

Few-shot adaptation. Large-scale pre-trained models have proven effective, especially in datascarce scenarios. Figure 9a shows iVideoGPT’s performance when fine-tuned with various sizes of action-free BAIR trajectories. We observe that pre-training offers minimal benefits when full downstream data is available, yet the advantages become significant under data scarcity (with 100 or 1,000 trajectories). We also adapt iVideoGPT using 1,000 action-conditioned BAIR trajectories, achieving an FVD of 82.3. The fast adaptation ability with a handful of data is particularly crucial in model-based RL. As shown in Figure 7, world models trained from scratch may generate inaccurate predictions, thereby degenerating the sample efficiency that is vital for model-based agents.

Model scaling. All previous experiments are conducted using an iVideoGPT with 12 transformer layers and 768-dimensional hidden states (138M parameters). To initially investigate the scaling

###### BAIR (action-free)

t = 0 t = 1 t = 3 t = 5 t = 7 t = 9 t = 11 t = 13 t = 15

t = 2 t = 4 t = 6 t = 8 t = 10 t = 12 t = 14

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

Ground truth

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Predicted (w/o fine-tuning)

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

Predicted (tokenizer fine-tuned)

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

Predicted (full fine-tuned)

- Figure 8: Zero-shot prediction by pre-trained transformer in iVideoGPT. Without fine-tuning, the transformer predicts natural movements of a different robot gripper using the pre-trained tokenizer (second row) but accurately predicts for the correct gripper type with an adapted tokenizer (third row).

(a) Few-shot adaptation (b) Model scaling

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | |4x4 tokenizer 0.180 LPIPS| |Training Memory|GPU (GB)| |
| | |1.45s Time| | | | |
| | |10.6GB Mem|.| | | |
| | | | | | | |
| | |Compressive<br><br>0.059 LPIPS<br>1.46s Time<br>|tokenizer (Ours) 16x16 tok| |enizer| |
| | | | | | | |
| | |22.3GB Mem|. 0.036 22.8s Training| |LPIPS Time<br><br>OOM| |

(c) Tokenization efficiency

- Figure 9: Model analysis. (a) Video prediction results with various fine-tuning strategies and data sizes on BAIR. (b) Validation losses for the 138M and 436M transformer models on the pre-training dataset. (c) Computational efficiency and reconstruction quality of different tokenizers.

behavior of our model, we trained a larger iVideoGPT with 24 layers and 1024-dimensional hidden states (436M parameters). Figure 9b illustrates the validation loss curves on the pre-trained dataset. It shows that (1) the validation loss (perplexity) continues to decrease regardless of model size, and (2) increasing the model size accelerates the loss decrease. These results align with our expectation that larger model sizes and increased computation [48] can build more powerful iVideoGPTs.

Tokenization efficiency. We evaluate the effectiveness of our compressive tokenization by comparing it against standard VQGAN tokenizers that independently convert each frame into 16 × 16 and 4 × 4 tokens. We train three tokenizers from scratch on RoboNet for the same number of steps. As Figure 9c, the tokenizer with 4 × 4 tokens suffers from low reconstruction quality due to its insufficient capacity. Our proposed tokenization method slightly compromises reconstruction quality compared to the standard 16 × 16 tokenizer but can provide more consistent contextual information, which is beneficial for video prediction tasks. More importantly, it significantly enhances computational efficiency with a significantly fewer amount of tokens, which greatly saves time and memory, allowing us to scale the model size with fewer costs (see quantitative results in Appendix B.5).

Context-dynamics decoupling. Our tokenizer is designed with a bottleneck of much fewer tokens, focusing only on capturing necessary dynamics information for future frames while sharing contextual information with initial frames to reconstruct raw pixels. To explicitly visualize this decoupling of context and dynamics information, we drop cross-attention blocks to context frames in the decoder when reconstructing future frames. The results in Figure 10 show that the decoder can still reproduce the movement trajectories accurately but with minimal contextual information. This visualization supports the explanation of our model’s generalization capability shown in Figure 8.

Goal-conditioned prediction. In Figure 4, we also showcase video prediction generated by goalconditioned iVideoGPT, pre-trained on massive human and robotic videos (Section 3.2). Unlike action-free prediction, which often results in trajectories diverging from the ground truth, the goalconditioned model produces more accurate paths to reach specified goals. We believe this highlights the potential of leveraging the flexibility of a unified sequence modeling paradigm.

t = 0 t = 2 t = 4 t = 6 t = 8 t = 10 t = 12 t = 14 t = 0 t = 2 t = 4 t = 6 t = 8 t = 10 t = 12 t = 14

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

Reconstruction

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

Reconstruction (w/o cross-attention in decoder)

- Figure 10: Context-dynamics decoupling in our compressive tokenization. By removing crossattention from future frames to context frames, the decoder can still reconstruct a trajectory that moves in the same way as the original, but the visual context is almost entirely missing.

### 5 Related Work

World models for visual control. Learning general world models in visual domains remains a significant challenge in model-based reinforcement learning. A straightforward method involves learning action-conditioned video prediction models [69, 45]. Advanced model-based RL algorithms [29, 31, 32, 80, 34, 33] utilize latent imagination for more efficient and accurate rollouts but complicate themselves by tightly coupling model and policy learning. We show that this complexity can be reduced with powerful world models that have accumulated generalizable knowledge beyond specific tasks. Recent efforts to facilitate this include leveraging scalable architectures like transformers [63] and pre-training from large-scale data [99, 62]. Of particular relevance to our work are UniSim [102] and Genie [12], which have developed extensively trained world models with diffusion and masked models, respectively, though neither is publicly available. Our work distinguishes itself by utilizing a generic autoregressive transformer framework, advancing the flexibility of scalable world models.

Video generation and prediction. Recent developments in Internet-scale video generation models now enable the synthesis of realistic videos conditioned on class labels, text descriptions, and initial frames—the last one also known as the video prediction problem. Various models have been developed, including deterministic RNNs [84, 96], variational autoencoders [18, 3, 30, 4], diffusion [38, 11], masked [104, 27], and autoregressive models [101, 50, 55]. However, most recent works do not treat video prediction as a dynamics modeling problem and perform spatiotemporal compression [101, 8], thus providing limited interactivity to serve as world models. We achieve both compressive tokenization and interactivity by context-aware representation, employing cross-attention mechanisms with minimal inductive bias. This method diverges from previous techniques that rely on motion vectors [43] or optical flows [52] and offers a more generic form of video tokenization.

### 6 Discussion

We introduced Interactive VideoGPT (iVideoGPT), a generic and efficient world model architecture that leverages a scalable autoregressive transformer to integrate multimodal signals into a sequence of tokens, providing an interactive agent experience via next-token prediction. iVideoGPT has been pre-trained on millions of human and robotic manipulation trajectories and adapted to a wide range of downstream tasks. As a powerful foundation of world models, it enables accurate and generalizable video prediction as well as simplified yet performant model-based planning or reinforcement learning.

Limitations and future work. While iVideoGPT marks significant progress, there is substantial room for improvement. We found limited diversity in publicly available robotic data, including the large-scale Open X-Embodiment dataset, and initiated efforts to transfer knowledge from human videos [25]. We believe iVideoGPT should be pre-trained on more extensive data [26] to bridge knowledge between humans and robots. This also requires iVideoGPT to incorporate more modalities, such as multi-view observations, proprioceptive robot states, and actions, within the unified formulation beyond action-free video prediction. Specifically, to process high-dimensional visual observations, our compressive tokenization assumes that initial frames provide sufficient contexts for future frames, which works for low-level control tasks as model-based agents often foresee tens of steps, but may falter in scenarios with long videos and significant camera motion. This issue can be mitigated by keyframe extraction [51] but leaves an important future avenue of exploration. Finally, extending to more complex real-robot tasks is essential, as the benefits of model scaling to even larger sizes remain unobserved in this work within visually simple simulation for downstream control tasks.

### Acknowledgements

We would like to thank many colleagues, in particular Yuchen Zhang, Lanxiang Xing, and Haixu Wu, for their valuable discussion. This work was supported by the National Natural Science Foundation of China (U2342217 and 62022050), the BNRist Project, the Huawei Innovation Fund, and the National Engineering Research Center for Big Data Software.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C Courville, and Marc Bellemare. Deep reinforcement learning at the edge of the statistical precipice. In NeurIPS, 2021.
- [3] Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan, Roy H Campbell, and Sergey Levine. Stochastic variational video prediction. In ICLR, 2018.
- [4] Mohammad Babaeizadeh, Mohammad Taghi Saffar, Suraj Nair, Sergey Levine, Chelsea Finn, and Dumitru Erhan. Fitvid: Overfitting in pixel-level video prediction. arXiv preprint arXiv:2106.13195, 2021.
- [5] Shikhar Bahl, Russell Mendonca, Lili Chen, Unnat Jain, and Deepak Pathak. Affordances from human videos as a versatile representation for robotics. In CVPR, 2023.
- [6] Suneel Belkhale, Yuchen Cui, and Dorsa Sadigh. Hydra: Hybrid robot actions for imitation learning. In CoRL, 2023.
- [7] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [8] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023.
- [9] Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, et al. AudioLM: a language modeling approach to audio generation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023.
- [10] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. RT-1: Robotics transformer for real-world control at scale. In RSS, 2023.
- [11] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024.
- [12] Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In ICML, 2024.
- [13] Lili Chen, Shikhar Bahl, and Deepak Pathak. Playfusion: Skill acquisition via diffusion from languageannotated play. In CoRL, 2023.
- [14] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In RSS, 2023.
- [15] Zichen Jeff Cui, Yibin Wang, Nur Muhammad Mahi Shafiullah, and Lerrel Pinto. From play to policy: Conditional behavior generation from uncurated robot data. In ICLR, 2023.
- [16] Sudeep Dasari, Frederik Ebert, Stephen Tian, Suraj Nair, Bernadette Bucher, Karl Schmeckpeper, Siddharth Singh, Sergey Levine, and Chelsea Finn. RoboNet: Large-scale multi-robot learning. In CoRL, 2019.

- [17] Shivin Dass, Jullian Yapeter, Jesse Zhang, Jiahui Zhang, Karl Pertsch, Stefanos Nikolaidis, and Joseph J. Lim. CLVR jaco play dataset, 2023.
- [18] Emily Denton and Rob Fergus. Stochastic video generation with a learned prior. In ICML, 2018.
- [19] Yilun Du, Mengjiao Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Brian Ichter, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B Tenenbaum, et al. Video language planning. In ICLR, 2024.
- [20] Frederik Ebert, Chelsea Finn, Sudeep Dasari, Annie Xie, Alex Lee, and Sergey Levine. Visual foresight: Model-based deep reinforcement learning for vision-based robotic control. arXiv preprint arXiv:1812.00568, 2018.
- [21] Frederik Ebert, Chelsea Finn, Alex X Lee, and Sergey Levine. Self-supervised visual planning with temporal skip connections. In CoRL, 2017.
- [22] Patrick Esser, Robin Rombach, and Björn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, 2021.
- [23] Yunhai Feng, Nicklas Hansen, Ziyan Xiong, Chandramouli Rajagopalan, and Xiaolong Wang. Finetuning offline world models in the real world. In CoRL, 2023.
- [24] Chelsea Finn and Sergey Levine. Deep visual foresight for planning robot motion. In ICRA, 2017.
- [25] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The "something something" video database for learning and evaluating visual common sense. In ICCV, 2017.
- [26] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In CVPR, 2022.
- [27] Agrim Gupta, Stephen Tian, Yunzhi Zhang, Jiajun Wu, Roberto Martín-Martín, and Li Fei-Fei. Maskvit: Masked visual pre-training for video prediction. In ICLR, 2023.
- [28] David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. In NeurIPS, 2018.
- [29] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In ICLR, 2020.
- [30] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In ICML, 2019.
- [31] Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. In ICLR, 2021.
- [32] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.
- [33] Nicklas Hansen, Hao Su, and Xiaolong Wang. TD-MPC2: Scalable, robust world models for continuous control. In ICLR, 2024.
- [34] Nicklas Hansen, Xiaolong Wang, and Hao Su. Temporal difference learning for model predictive control. In ICML, 2022.
- [35] Minho Heo, Youngwoon Lee, Doohyun Lee, and Joseph J. Lim. Furniturebench: Reproducible real-world benchmark for long-horizon complex manipulation. In RSS, 2023.
- [36] Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Ian Osband, et al. Deep q-learning from demonstrations. In AAAI, 2018.
- [37] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [38] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In NeurIPS, 2022.
- [39] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In ICLR, 2022.

- [40] Quan Huynh-Thu and Mohammed Ghanbari. Scope of validity of psnr in image/video quality assessment. Electronics letters, 44(13):800–801, 2008.
- [41] Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, and Chelsea Finn. BC-z: Zero-shot task generalization with robotic imitation learning. In CoRL, 2021.
- [42] Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. When to trust your model: Model-based policy optimization. In NeurIPS, 2019.
- [43] Yang Jin, Zhicheng Sun, Kun Xu, Kun Xu, Liwei Chen, Hao Jiang, Quzhe Huang, Chengru Song, Yuliang Liu, Di Zhang, Yang Song, Kun Gai, and Yadong Mu. Video-lavit: Unified video-language pre-training with decoupled visual-motional tokenization. In ICML, 2024.
- [44] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In ECCV, 2016.
- [45] Lukasz Kaiser, Mohammad Babaeizadeh, Piotr Milos, Blazej Osinski, Roy H Campbell, Konrad Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, et al. Model-based reinforcement learning for atari. In ICLR, 2020.
- [46] Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian Ibarz, Alexander Herzog, Eric Jang, Deirdre Quillen, Ethan Holly, Mrinal Kalakrishnan, Vincent Vanhoucke, et al. Qt-opt: Scalable deep reinforcement learning for vision-based robotic manipulation. In CoRL, 2018.
- [47] Harini Kannan, Danijar Hafner, Chelsea Finn, and Dumitru Erhan. Robodesk: A multi-task reinforcement learning benchmark. https://github.com/google-research/robodesk, 2021.
- [48] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.
- [49] Minchan Kim, Junhyek Han, Jaehyung Kim, and Beomjoon Kim. Pre-and post-contact policy decomposition for non-prehensile manipulation with zero-shot sim-to-real transfer. In IROS, 2023.
- [50] Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Rachel Hornung, Hartwig Adam, Hassan Akbari, Yair Alon, Vighnesh Birodkar, et al. Videopoet: A large language model for zero-shot video generation. In ICML, 2024.
- [51] Didier Le Gall. Mpeg: A video compression standard for multimedia applications. Communications of the ACM, 34(4):46–58, 1991.
- [52] Guillaume Le Moing, Jean Ponce, and Cordelia Schmid. Ccvs: Context-aware controllable video synthesis. In NeurIPS, 2021.
- [53] Yann LeCun. A path towards autonomous machine intelligence. preprint posted on openreview, 2022.
- [54] Michelle A Lee, Yuke Zhu, Krishnan Srinivasan, Parth Shah, Silvio Savarese, Li Fei-Fei, Animesh Garg, and Jeannette Bohg. Making sense of vision and touch: Self-supervised learning of multimodal representations for contact-rich tasks. In ICRA, 2019.
- [55] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with blockwise ringattention. arXiv preprint arXiv:2402.08268, 2024.
- [56] Corey Lynch, Ayzaan Wahid, Jonathan Tompson, Tianli Ding, James Betker, Robert Baruch, Travis Armstrong, and Pete Florence. Interactive language: Talking to robots in real time. IEEE Robotics and Automation Letters, 2023.
- [57] Haoyu Ma, Jialong Wu, Ningya Feng, Chenjun Xiao, Dong Li, Jianye Hao, Jianmin Wang, and Mingsheng Long. Harmonydream: Task harmonization inside world models. In ICML, 2024.
- [58] Ajay Mandlekar, Jonathan Booher, Max Spero, Albert Tung, Anchit Gupta, Yuke Zhu, Animesh Garg, Silvio Savarese, and Li Fei-Fei. Scaling robot supervision to hundreds of hours with roboturk: Robotic manipulation dataset through human reasoning and dexterity. In IROS, 2019.
- [59] Tatsuya Matsushima, Hiroki Furuta, Yusuke Iwasawa, and Yutaka Matsuo. Weblab xArm Dataset, 2023.
- [60] Bryan McCann, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. The natural language decathlon: Multitask learning as question answering. arXiv preprint arXiv:1806.08730, 2018.

- [61] Oier Mees, Jessica Borja-Diaz, and Wolfram Burgard. Grounding language with visual affordances over unstructured data. In ICRA, 2023.
- [62] Russell Mendonca, Shikhar Bahl, and Deepak Pathak. Structured world models from human videos. In RSS, 2023.
- [63] Vincent Micheli, Eloi Alonso, and François Fleuret. Transformers are sample efficient world models. In ICLR, 2023.
- [64] Matthias Minderer, Chen Sun, Ruben Villegas, Forrester Cole, Kevin P Murphy, and Honglak Lee. Unsupervised learning of object structure and dynamics from videos. In NeurIPS, 2019.
- [65] Soroush Nasiriany, Tian Gao, Ajay Mandlekar, and Yuke Zhu. Learning and retrieval from prior data for skill-based imitation learning. In CoRL, 2022.
- [66] Anh Nguyen, Jason Yosinski, and Jeff Clune. Deep neural networks are easily fooled: High confidence predictions for unrecognizable images. In CVPR, 2015.
- [67] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Charles Xu, Jianlan Luo, Tobias Kreiman, You Liang Tan, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An open-source generalist robot policy. In RSS, 2024.
- [68] Jihoon Oh, Naoaki Kanazawa, and Kento Kawaharazuka. X-Embodiment U-Tokyo PR2 Datasets, 2023.
- [69] Junhyuk Oh, Xiaoxiao Guo, Honglak Lee, Richard Lewis, and Satinder Singh. Action-conditional video prediction using deep networks in atari games. In NeurIPS, 2015.
- [70] Abhishek Padalkar, Acorn Pooley, Ajinkya Jain, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anikait Singh, Anthony Brohan, et al. Open x-embodiment: Robotic learning datasets and rt-x models. In ICRA, 2024.
- [71] Abhishek Padalkar, Gabriel Quere, Antonin Raffin, João Silvério, and Freek Stulp. A guided reinforcement learning approach using shared control templates for learning manipulation skills in the real world. Research square preprint rs-3289569/v1, 2023.
- [72] Abhishek Padalkar, Gabriel Quere, Franz Steinmetz, Antonin Raffin, Matthias Nieuwenhuisen, João Silvério, and Freek Stulp. Guiding reinforcement learning with shared control templates. In ICRA, 2023.
- [73] Suraj Patil, William Berman, Robin Rombach, and Patrick von Platen. amused: An open muse reproduction. arXiv preprint arXiv:2401.01808, 2024.
- [74] Gabriel Quere, Annette Hagengruber, Maged Iskandar, Samuel Bustamante, Daniel Leidner, Freek Stulp, and Joern Vogel. Shared Control Templates for Assistive Robotics. In ICRA, 2020.
- [75] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.
- [76] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.
- [77] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.
- [78] Erick Rosete-Beas, Oier Mees, Gabriel Kalweit, Joschka Boedecker, and Wolfram Burgard. Latent plans for task agnostic offline reinforcement learning. In CoRL, 2022.
- [79] Amrita Sawhney, Steven Lee, Kevin Zhang, Manuela Veloso, and Oliver Kroemer. Playing with food: Learning food item representations through interactive exploration. In Experimental Robotics: The 17th International Symposium, pages 309–322. Springer, 2021.
- [80] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588(7839):604–609, 2020.
- [81] Younggyo Seo, Kimin Lee, Stephen L James, and Pieter Abbeel. Reinforcement learning with action-free pre-training from videos. In ICML, 2022.
- [82] Rutav Shah, Roberto Martín-Martín, and Yuke Zhu. MUTEX: Learning unified policies from multimodal task specifications. In CoRL, 2023.

- [83] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.
- [84] Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-Kin Wong, and Wang-chun Woo. Convolutional lstm network: A machine learning approach for precipitation nowcasting. In NeurIPS, 2015.
- [85] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [86] Stephen Tian, Chelsea Finn, and Jiajun Wu. A control-centric benchmark for video prediction. In ICLR, 2023.
- [87] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [88] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.
- [89] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. In NeurIPS, 2017.
- [90] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.
- [91] Ruben Villegas, Arkanath Pathak, Harini Kannan, Dumitru Erhan, Quoc V Le, and Honglak Lee. High fidelity video prediction with large stochastic recurrent neural networks. In NeurIPS, 2019.
- [92] Jörn Vogel, Annette Hagengruber, Maged Iskandar, Gabriel Quere, Ulrike Leipscher, Samuel Bustamante, Alexander Dietrich, Hannes Hoeppner, Daniel Leidner, and Alin Albu-Schäffer. Edan - an emg-controlled daily assistant to help people with physical disabilities. In IROS, 2020.
- [93] Vikram Voleti, Alexia Jolicoeur-Martineau, and Chris Pal. Mcvd-masked conditional video diffusion for prediction, generation, and interpolation. In NeurIPS, 2022.
- [94] Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao, Philippe Hansen-Estruch, Quan Vuong, Andre He, Vivek Myers, Kuan Fang, Chelsea Finn, and Sergey Levine. Bridgedata v2: A dataset for robot learning at scale. In CoRL, 2023.
- [95] Yixuan Wang, Zhuoran Li, Mingtong Zhang, Katherine Driggs-Campbell, Jiajun Wu, Li Fei-Fei, and Yunzhu Li. D3fields: Dynamic 3d descriptor fields for zero-shot generalizable robotic manipulation. arXiv preprint arXiv:2309.16118, 2023.
- [96] Yunbo Wang, Haixu Wu, Jianjin Zhang, Zhifeng Gao, Jianmin Wang, S Yu Philip, and Mingsheng Long. Predrnn: A recurrent neural network for spatiotemporal predictive learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2):2208–2225, 2022.
- [97] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.
- [98] Bohan Wu, Suraj Nair, Roberto Martin-Martin, Li Fei-Fei, and Chelsea Finn. Greedy hierarchical variational autoencoders for large-scale video prediction. In CVPR, 2021.
- [99] Jialong Wu, Haoyu Ma, Chaoyi Deng, and Mingsheng Long. Pre-training contextualized world models with in-the-wild videos for reinforcement learning. In NeurIPS, 2023.
- [100] Ge Yan, Kris Wu, and Xiaolong Wang. UCSD Kitchens Dataset. August 2023.
- [101] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021.
- [102] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. In ICLR, 2024.
- [103] Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto. Mastering visual continuous control: Improved data-augmented reinforcement learning. In ICLR, 2022.
- [104] Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, et al. MAGVIT: Masked generative video transformer. In CVPR, 2023.

- [105] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In CoRL, 2020.
- [106] Biao Zhang and Rico Sennrich. Root mean square layer normalization. In NeurIPS, 2019.
- [107] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023.
- [108] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.
- [109] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. In ICLR, 2024.
- [110] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023.
- [111] Gaoyue Zhou, Victoria Dean, Mohan Kumar Srirama, Aravind Rajeswaran, Jyothish Pari, Kyle Hatch, Aryan Jain, Tianhe Yu, Pieter Abbeel, Lerrel Pinto, Chelsea Finn, and Abhinav Gupta. Train offline, test online: A real robot learning benchmark. In ICRA, 2023.
- [112] Yifan Zhou, Shubham Sonawani, Mariano Phielipp, Heni Ben Amor, and Simon Stepputtis. Learning modular language-conditioned robot policies through attention. Autonomous Robots, pages 1–21, 2023.
- [113] Yifan Zhou, Shubham Sonawani, Mariano Phielipp, Simon Stepputtis, and Heni Amor. Modularity through attention: Efficient training and transfer of language-conditioned policies for robot manipulation. In CoRL, 2023.
- [114] Xinghao Zhu, Ran Tian, Chenfeng Xu, Mingyu Ding, Wei Zhan, and Masayoshi Tomizuka. Fanuc manipulation: A dataset for learning-based manipulation with fanuc mate 200id robot. 2023.
- [115] Yifeng Zhu, Abhishek Joshi, Peter Stone, and Yuke Zhu. Viola: Imitation learning for vision-based manipulation with object proposal priors. In CoRL, 2022.
- [116] Yifeng Zhu, Peter Stone, and Yuke Zhu. Bottom-up skill discovery from unsegmented demonstrations for long-horizon robot manipulation. IEEE Robotics and Automation Letters, 7(2):4126–4133, 2022.
- [117] Yuke Zhu, Josiah Wong, Ajay Mandlekar, Roberto Martín-Martín, Abhishek Joshi, Soroush Nasiriany, and Yifeng Zhu. robosuite: A modular simulation framework and benchmark for robot learning, 2022.

### A Implementation and Experimental Details

The main hyperparameters of our experiment are detailed in Tables 2, 3, and 5. In this section, we provide a comprehensive explanation of all experimental details.

#### A.1 Architecture

Table 2: Hyperparameters of iVideoGPT architectures.

VQGAN Low-resolution High-resolution

Parameters 114M 310M Resolution 64 × 64 256 × 256 Down blocks 3 5 Down layers per block 2 2 Down channels [128, 256, 512] [128, 256, 256, 512, 768] Mid block attention False False Up blocks 3 5 Up layers per block 3 3 Up channels [512, 256, 128] [768, 512, 256, 256, 128] Embedding dim 64 64 Codebook size 8192 8192 Norm GroupNorm GroupNorm Norm group 32 32 Activation SiLU SiLU Max cross-att. resolution 16 32

Transformer Small Medium

Parameters 138M 436M Layers 12 24 Heads 12 16 Hidden dim 768 1024 Feedforward dim 3072 4096 Dropout 0.1 0.1 Activation SiLU SiLU

Tokenizier. As illustrated in Figure 3, we use a conditional VQGAN for compressive tokenization. This comprises two encoder-decoder pairs: (Ec,Dc) for context frames (referred to as the context encoder-decoder) and (Ep,Dp) for future frames (referred to as the prediction encoder-decoder). Both pairs share the same architecture (detailed in Table 2), but the prediction encoder-decoder has a tighter bottleneck, focusing solely on encoding dynamic information. Specifically, it uses a 4 × 4 convolution to downsample 16 × 16 embeddings into 4 × 4 before looking up the codebook. Consequently, the prediction encoder-decoder needs to be conditioned on the features of the context encoder-decoder to incorporate rich contextual information. This conditioning is implemented via a multi-scale cross-attention mechanism, similar to ContextWM [99].

The intuition behind the multi-scale cross-attention across feature maps is as follows: the context encoder extracts contextual features at varying levels of abstraction, while the prediction encoder uses cross-attention to adaptively filter out contextual information and distill dynamics information. During decoding, the prediction decoder blocks employ cross-attention to retrieve contextual information at corresponding levels, facilitating the gradual reconstruction of the scene. This framework enhances the model’s ability to understand and manipulate complex scenes by focusing on dynamic changes, rather than being overwhelmed by irrelevant visual details.

Specifically, at the end of each encoder block, let Fcl ∈ Rc×h×w be the feature map of a context frame, and Fpl ∈ Rc×h×w be the feature map of a future frame. Before being processed by the next

Table 3: Hyperparameters of iVideoGPT training and evaluation.

Low-resolution (64 × 64) High-resolution (256 × 256) VQGAN Pre-train BAIR RoboNet VP2 Pre-train RoboNet

GPU days 17 2 8 4 16 9 Training steps 1 × 106 2 × 105 6 × 105 2 × 105 2.5 × 105 1.5 × 105 Disc. start - - - - - 1 × 105 Batch size 64 64 64 64 32 32 Sequence length 16 16 12 12 16 12 Context frames 2 1 2 2 2 2 Sampled future frames 6 7 6 6 6 6 Learning rate 5 × 10−4 1 × 10−4 1 × 10−4 1 × 10−4 5 × 10−4 1 × 10−4 LR Schedule Constant

- Weight decay 0.0 Grad clip 1.0 Warmup steps 500 Loss balancing Equal weights Optimizer AdamW Mixed precision bf16 Transformer Pre-train BAIR RoboNet VP2 Pre-train RoboNet

GPU days 19 1.5 10 3 9 26 Training steps 7 × 105 1 × 105 6 × 105 2 × 105 3.5 × 105 5 × 105 Batch size 64 64 64 64 16 32 Sequence length 16 16 12 12 16 12 Context frames 2 1 2 2 2 2 Learning rate 1 × 10−4 LR Schedule Cosine

- Weight decay 0.01 Grad clip 1.0 Warmup steps 5000 Loss balancing N/A or equal weights Optimizer AdamW Mixed precision bf16

Sampling temperature 1.0 Sampling top-k 100

block, Fpl is augmented with Fcl as follows:

Fcl+1 = EncBlocklc+1(Fcl) Fpl+1 = EncBlocklp+1(Augment(Fpl,Fcl))

(4)

This is achieved by performing cross-attention between the 2hw positions of the feature maps: Flatten: Q = Norm Reshape Fpl + PosEmbQ ∈ Rhw×c

K = V = Norm Reshape Fcl + PosEmbKV ∈ Rhw×c Cross-Attention: R = Attention QWQ,KWK,V WV ∈ Rhw×c

Residual-Connection: Augment(Fpl,Fcl) = SiLU Fpl + Reshape(R) ∈ Rc×h×w.

(5)

To reduce memory usage, we apply the cross-attention mechanism only when the feature map size is below a certain threshold (16 × 16 for a 64 × 64 original resolution and 32 × 32 for a 256 × 256 resolution). This mechanism is symmetrically performed across the context and prediction decoder.

Since attention mechanisms can flexibly handle varied input lengths, the conditioning mechanism can be easily extended to accommodate different numbers of context frames. Each context frame is

independently processed by the context encoder and decoder, and their feature maps are concatenated to serve as inputs for cross-attention in the prediction encoder and decoder.

Our VQGAN for 256 × 256 resolution is initialized from the pretrained model from aMUSEd4 [73]. We do not use discriminators for 64 × 64 resolution, effectively converting the VQGAN into a vanilla VQVAE with an additional perceptual loss.

Transformer. We flatten a video into a sequence of tokens:

x = (z1(1),...,z1(N),[S1],z2(1),...,z2(N),...,[S2],zT(1)

0+1,...,zT(n)

0+1,...), (6)

where we use two types of slot tokens [S1] and [S2] before the start of context frames and future frames, respectively. Context and future frames do not share token IDs, resulting in a transformer vocabulary of 16,386 tokens: the first 8,192 for context frames, the next 8,192 for future frames, and the last two for slot tokens. We adopt the autoregressive transformer architecture from LLaMA [87], but instantiate it to smaller models matching the size of GPT-2. We considered two model sizes, listed in Figure 2. Most of our experiments utilize a 138M parameter transformer, while preliminary scaling analysis is conducted using a 436M model.

#### A.2 Action-free Video Pre-training

Data mixture. We pre-train iVideoGPT using 35 datasets from the Open X-Embodiment Dataset (OXE) [70] and Something-Something-v2 (SSv2) [25]. To construct our training dataset from OXE, we implement a filtering and weighting process similar to Octo [67]. Initially, we exclude datasets lacking image streams and those derived from mobile robots. Subsequently, datasets exhibiting excessive repetition or possessing low image resolutions are eliminated. The remaining datasets were categorized as either "large" or "small," and each was assigned a weight based on its size and diversity. We select 1% of samples from each subset as validation data and use the rest for training. For SSv2, we manually select 95 classes with clear motion trends from the original 174 video classes as our pre-training data with a weighting of 15%. We use the official splits of SSv2 for training and validation. For a comprehensive breakdown of the mixture, refer to Table 4.

Training details. During training, we sample sequences of frames by first randomly selecting a training video and then uniformly sampling a segment of a specified length and step size, i.e., neighboring frames in the segment are spaced a certain number of steps apart in the original video. We observe that datasets are collected at different frequencies. To maintain consistency, we adjust sampling with varied step sizes, aligning each with its respective dataset frequency, as listed in Table 4. For tokenizer training, the initial frames of the segment are used as context frames, and from the remaining frames, we randomly sample a subset as future frames to reduce memory requirements and increase batch size. For transformer training, we use the full segment of frames. The number of frames in minibatches for each dataset is detailed in Table 3. We use a mixture of OXE and SSv2 for training the tokenizer to ensure visual diversity, while only OXE is used for training the transformer. For data augmentation, we apply random resized crop and color jitter, ensuring consistency across the sequence. During both tokenizer and transformer training, we blend different losses with equal weights. Unless specified otherwise, we follow the same implementation details when fine-tuning iVideoGPT on downstream tasks.

Goal-conditioned prediction. To train a goal-conditioned variant of iVideoGPT on the same dataset, we first fine-tune the previously obtained tokenizer using two randomly sampled frames as context for 550k training steps. Then, we train a transformer from scratch with the rearranged frame segment o˜1:T = (oT,o1,o2,...,oT−1) for 1 million steps. The architecture and training procedures remain consistent with the above setup.

License. The Open X-Embodiment dataset follows the Apache license. RoboNet is licensed under Creative Commons Attribution 4.0, while BAIR follows Creative Commons BY 4.0. The Something-Something-V2 dataset is subject to the Data License Agreement.

4https://github.com/huggingface/amused under openrail++ license

Table 4: iVideoGPT pre-training data mixture from the Open X-Embodiment [70] and SomethingSomething-V2 [25] datasets.

Dataset Num of trajectories Step size Sampling weight Fractal (RT-1) [10] 87,212 1 12.8% Bridge [94] 28,935 2 12.8% BC-Z [41] 43,264 3 12.8% RoboNet [16] 82,649 1 12.8% Kuka [46] 580,392 3 8.5% Language Table [56] 442,226 3 4.2% Stanford MaskViT [27] 9,200 1 4.2% UIUC D3Field [95] 768 1 2.2% Taco Play [78, 61] 3,603 5 0.5% Jaco Play [17] 1,085 3 0.5% Roboturk [58] 1,995 3 0.5% Viola [115] 150 7 0.5% Toto [111] 1,003 10 0.5% Columbia Cairlab Pusht Real [14] 136 3 0.5% Stanford Kuka Multimodal Dataset [54] 3,000 7 0.5% Stanford Hydra Dataset [6] 570 3 0.5% Austin Buds Dataset [116] 50 7 0.5% NYU Franka Play Dataset [15] 456 1 0.5% Furniture Bench Dataset [35] 5,100 3 0.5% UCSD Kitchen Dataset [100] 150 1 0.5% UCSD Pick and Place Dataset [23] 1,355 1 0.5% Austin Sailor Dataset [65] 240 7 0.5% UTokyo PR2 Tabletop Manipulation [68] 240 3 0.5% UTokyo Xarm Pick and Place [59] 102 3 0.5% UTokyo Xarm Bimanual [59] 70 3 0.5% KAIST Nonprehensile [49] 201 3 0.5% DLR SARA Pour [72] 100 3 0.5% DLR SARA Grid [71] 107 3 0.5% DLR EDAN Shared Control [92, 74] 104 3 0.5% ASU Table Top [113, 112] 110 4 0.5% UTAustin Mutex [82] 1,500 7 0.5% Berkeley Fanuc Manipulation [114] 415 3 0.5% CMU Playing with Food [79] 174 3 0.5% CMU Play Fusion [13] 576 2 0.5% CMU Stretch [5, 62] 135 3 0.5% Something-Something-V2 [25] 120,581 1 15.0% Total 1,417,954 - 100.0%

Algorithm 1 Model-Based Policy Optimization (MBPO), adapted from [42]

- 1: Initialize actor-critic πϕ,vψ, world model pθ
- 2: Initialize real replay buffer Dreal with random policy
- 3: Initially train model pθ on Dreal
- 4: Initialize imagined replay buffer Dimag with random rollouts using pθ
- 5: for N steps do
- 6: // Training
- 7: if model update step then
- 8: Update world model pθ on a mini-batch from Dreal
- 9: end if
- 10: Update actor-critic πϕ,vψ with model-free objectives on a mini-batch from Dimag ∪ Dreal
- 11: // Data collection
- 12: if model rollout step then
- 13: Sample a mini-batch of ot uniformly from Dreal
- 14: Perform k-step model rollout starting from ot using policy πϕ; add to Dimag
- 15: end if
- 16: Take action in environment according to πϕ; add to Dreal
- 17: end for

#### A.3 Video Prediction

Evaluation metrics. We evaluate our model across four different metrics5: Structural Similarity Index Measure (SSIM) [97], Peak Signal-to-noise Ratio (PSNR) [40], Learned Perceptual Image Patch Similarity (LPIPS) [108] and Fréchet Video Distance (FVD) [88]. Following prior works [3, 91, 4, 104], we account for the stochastic nature of video prediction by sampling 100 future trajectories per test video and selecting the best one for the final PSNR, SSIM, and LPIPS scores. For FVD, we use all 100 samples.

#### A.4 Visual Planning

We use the official repository6 to evaluate our model on the VP2 benchmark. The reported baseline results are provided by the authors of the benchmark. For the Robosuite tasks, a cost below 0.05 is considered a success.

#### A.5 Visual Model-based RL

Environments. Meta-world [105], following MIT License, is a benchmark of 50 robotic manipulation tasks. We select six tasks for our experiments: Button Press Topdown Wall, Plate Slide, Hammer, Door Lock, Handle Pull Side, and Coffee Push. We set the maximum episode length to 200 environment steps with an action repeat of 2 and a frame stack of 3 across these tasks and adjust the number of training steps to match the varying difficulty levels. During experiments, we observed that high rewards do not consistently correlate with high success rates in the original Meta-world implementation. This discrepancy presents a challenge to the learning stability of agents. To address this, we introduce an additional bonus for task success rbonus = 10.0 alongside the original task reward rtask:

r = rtask + rbonus · Itask success. (7)

Moreover, Meta-world features hard-exploration tasks, resulting in significant variance in the learning curves, which poses challenges to the accurate evaluation of RL algorithm performance. To mitigate this issue, we initialize the replay buffer of all compared methods, with 5 successful demonstration trajectories for all tasks except Door Lock. This strategy, commonly used to accelerate reinforcement learning [36], helps stabilize the training process and provides a more reliable evaluation.

- 5We use public implementations of metrics: https://github.com/francois-rozet/piqa under MIT license for SSIM and PSNR, https://github.com/richzhang/PerceptualSimilarity under BSD-2Clause license for LPIPS, and https://github.com/universome/stylegan-v under NVidia license for FVD.
- 6https://github.com/s-tian/vp2

Table 5: Hyperparameters of model-based RL with iVideoGPT. Model-based RL Hyperparameter Value

Init rollout batch size 640 Interval 200 env. steps Batch size 32 Horizon 10

Model rollout

Init training steps 1000 Tokenizer training interval 40 env. steps Transformer training interval 10 env. steps Batch size 16 Sequence length 12 Context frames 2 Sampled future frames (tokenizer) 5 Learning rate 1 × 10−4 Weight decay 0 Optimizer Adam

Model training

Model-based RL Real data ratio 0.5

Action-free t = 0 t = 2 t = 4 t = 6 t = 8 t = 10 t = 12 t = 14 t = 0 t = 2 t = 4 t = 6 t = 8 t = 10 t = 12 t = 14 Prediction

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

iVideoGPT (252M)

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

Dreamer XL (200M)

- Figure 11: Qualitative evaluation on the Open X-Embodiment dataset for Dreamerv3-XL pre-trained on the same dataset as ours.

Implementation Details. We have developed a simple model-based RL algorithm using iVideoGPT as a world model within the MBPO [42] framework, with DrQ-v2 [103] as the base actor-critic RL algorithm. Please refer to Algorithm 1 for the pseudo-code. Our implementation is based on the official DrQ-v2 code7, using the same hyperparameters and architecture for actor-critic learning. Hyperparameters specific to model-based RL are listed in Table 5. We use a symlog transformation [32] for reward prediction in iVideoGPT.

Baselines. To compare our method with DreamerV3, which lacks native pre-training support, we use APV [81]—a method enabling action-free pre-training on DreamerV2—as a baseline, modified to incorporate DreamerV3 features. We pre-train this model on the same dataset as iVideoGPT.

### B Extended Experimental Results

#### B.1 Qualitative Evaluation

We present additional examples of video predictions by iVideoGPT on various datasets in Figures 12, 13, 15, 16, 17, 18, and 19. We also include an additional showcase of zero-shot predictions by the pre-trained transformer in iVideoGPT in Figure 14, supplementing Figure 8 of the main text.

Additionally, we showcase prediction examples from the large-scale pre-trained DreamerV3-XL on the Open X-Embodiment dataset in Figures 11.

#### B.2 Human Study

Numerical metrics like FVD don’t always align with human-judged visual quality. To address this, we conduct a human user study on the prediction results of various models. Due to the lack of official

7https://github.com/facebookresearch/drqv2 under MIT License

###### Open X-Embodiment (action-free)

t = 0 t = 1 t = 3 t = 5 t = 7 t = 9 t = 11 t = 13 t = 15

t = 2 t = 4 t = 6 t = 8 t = 10 t = 12 t = 14

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

Ground truth

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

(context)

Predicted

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

Ground truth

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

(context)

Predicted

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

Ground truth

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

(context)

Predicted

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

Ground truth

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

(context)

Predicted

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

Ground truth

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

(context)

Predicted

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

Ground truth

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

(context)

Predicted

- Figure 12: Additional qualitative evaluation on the Open X-Embodiment dataset for action-free video prediction.

pretrained models for most baselines, we are only able to compare iVideoGPT with VideoGPT [101] and MCVD [93] on the action-free BAIR dataset. We generate videos using these three models from the test set and ask users to label preferences between two randomly sampled videos based on the physical naturalness and feasibility of robot-object interactions. A total of 386 annotations are collected from 9 participants. The results in Figure 20 demonstrate that iVideoGPT is preferred by human annotators more.

- B.3 Visual Planning Quantitative results on the VP2 benchmark are reported in Table 6.
- B.4 Visual Model-based RL

Comparison to FitVid-based world models. Although FitVid [4] is originally designed for the video prediction task and has not been used as world models for MBRL, we have implemented a baseline using FitVid by replacing iVideoGPT in our implementation. To predict rewards, we add an MLP head on top of FitVid’s latent states, parallel to the image decoder. As shown in Figure 21, MBPO with iVideoGPT outperforms FitVid on 5 out of 6 tasks and performs comparably on the remaining one. We also qualitatively observe that FitVid’s imagined trajectories are blurrier compared to ours, which hinders its ability to simulate real environments accurately and may hurt MBRL performance.

- B.5 Computational Efficiency

We report training and inference time and memory usage with various tokenizers in Table 7 and 8, respectively. Our proposed compressive tokenization provides significant memory savings during training and faster rollouts during generation. We note that although we use a more complicated

###### Open X-Embodiment (goal-conditioned)

t = 0 t = 1 t = 2 t = 3 t = 4 t = 5 t = 6 t = 7 t = 8 t = 9 t = 10 t = 11 t = 12 t = 13 t = 14 t = 15

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

Ground truth

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

(context) (goal)

Predicted

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

Ground truth

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

(context) (goal)

Predicted

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

Ground truth

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

(context) (goal)

Predicted

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

Ground truth

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

(context) (goal)

Predicted

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

Ground truth

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

(context) (goal)

Predicted

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

Ground truth

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

(context) (goal)

Predicted

- Figure 13: Additional qualitative evaluation on the Open X-Embodiment dataset for goal-conditioned video prediction.

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

t = 0 t = 1 t = 3 t = 5 t = 7 t = 9 t = 11 t = 13 t = 15

Ground truth

Predicted (w/o fine-tuning)

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

t = 2 t = 4 t = 6 t = 8 t = 10 t = 12 t = 14

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

Predicted (tokenizer fine-tuned)

BAIR (action-free)

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

Predicted (full fine-tuned)

- Figure 14: Additional zero-shot prediction by the pre-trained transformer in iVideoGPT, supplementing Figure 8 of the main text.

###### BAIR (action-conditioned)

t = 0 t = 1 t = 3 t = 5 t = 7 t = 9 t = 11 t = 13 t = 15

t = 2 t = 4 t = 6 t = 8 t = 10 t = 12 t = 14

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

Ground truth

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

(context) Predicted

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

Ground truth

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

(context) Predicted

###### Figure 15: Additional qualitative evaluation on the BAIR dataset, given future actions.

###### RoboNet (action-conditioned)

t = 0 t = 1 t = 2 t = 3 t = 4 t = 5 t = 6 t = 7 t = 8 t = 9 t = 10 t = 11

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

Ground truth

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

(context)

Predicted

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

Ground truth

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

(context)

Predicted

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

Ground truth

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

(context)

Predicted

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

Ground truth

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

(context)

Predicted

###### Figure 16: Additional qualitative evaluation on the RoboNet dataset, highlighting accurate movements of the pushed objects.

RoboNet (action-conditioned, 256x256 resolution)

t = 0 t = 1 t = 2 t = 3 t = 4 t = 5 t = 6 t = 7 t = 8 t = 9 t = 10 t = 11

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

Ground truth

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

(context)

Predicted

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

Ground truth

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

(context)

Predicted

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

Ground truth

[Figure 885]

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

(context)

Predicted

###### Figure 17: Additional qualitative evaluation on the RoboNet dataset, in high resolution (256 × 256).

###### VP2 (RoboDesk)

t = 0 t = 1 t = 2 t = 3 t = 4 t = 5 t = 6 t = 7 t = 8 t = 9 t = 10 t = 11

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

Ground truth

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

(context)

Predicted

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

[Figure 928]

Ground truth

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

[Figure 938]

(context)

Predicted

###### VP2 (RoboSuite)

t = 0 t = 1 t = 2 t = 3 t = 4 t = 5 t = 6 t = 7 t = 8 t = 9 t = 10 t = 11

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

[Figure 948]

[Figure 949]

[Figure 950]

Ground truth

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

[Figure 955]

[Figure 956]

[Figure 957]

[Figure 958]

[Figure 959]

[Figure 960]

(context)

Predicted

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

[Figure 965]

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

Ground truth

[Figure 973]

[Figure 974]

[Figure 975]

[Figure 976]

[Figure 977]

[Figure 978]

[Figure 979]

[Figure 980]

[Figure 981]

[Figure 982]

(context)

Predicted

###### Figure 18: Additional qualitative evaluation on the VP2 benchmark.

###### Meta-world (action-conditioned & reward prediction)

t = 0 t = 1 t = 2 t = 3 t = 4 t = 5 t = 6 t = 7 t = 8 t = 9 t = 10 t = 11

[Figure 983]

[Figure 984]

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

[Figure 991]

[Figure 992]

[Figure 993]

[Figure 994]

Ground truth

[Figure 995]

[Figure 996]

[Figure 997]

[Figure 998]

[Figure 999]

[Figure 1000]

[Figure 1001]

[Figure 1002]

[Figure 1003]

[Figure 1004]

(context)

Predicted

[Figure 1005]

[Figure 1006]

[Figure 1007]

[Figure 1008]

[Figure 1009]

[Figure 1010]

[Figure 1011]

[Figure 1012]

[Figure 1013]

[Figure 1014]

[Figure 1015]

[Figure 1016]

Ground truth

[Figure 1017]

[Figure 1018]

[Figure 1019]

[Figure 1020]

[Figure 1021]

[Figure 1022]

[Figure 1023]

[Figure 1024]

[Figure 1025]

[Figure 1026]

(context)

Predicted

[Figure 1027]

[Figure 1028]

[Figure 1029]

[Figure 1030]

[Figure 1031]

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

[Figure 1036]

[Figure 1037]

[Figure 1038]

Ground truth

[Figure 1039]

[Figure 1040]

[Figure 1041]

[Figure 1042]

[Figure 1043]

[Figure 1044]

[Figure 1045]

[Figure 1046]

[Figure 1047]

[Figure 1048]

(context)

Predicted

[Figure 1049]

[Figure 1050]

[Figure 1051]

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

[Figure 1058]

[Figure 1059]

[Figure 1060]

Ground truth

[Figure 1061]

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

[Figure 1069]

[Figure 1070]

(context)

Predicted

###### Figure 19: Additional qualitative evaluation on Meta-world tasks. True and predicted rewards are labeled at the top left corner. Zoom in for details.

iVideoGPT win 50.7%

Draw 24.6% Draw 30.1%

VideoGPT win

iVideoGPT vs. VideoGPT

- 24.7%

MCVD win

- 25.2%

iVideoGPT win 44.7%

iVideoGPT vs. MCVD

VideoGPT win MCVD vs. VideoGPT 24.0%

MCVD win 46.3%

Draw 29.7%

- Figure 20: Human study. Videos generated by three models, VideoGPT, MCVD, and iVideoGPT, on the action-free BAIR dataset are presented to human users, who label their preferences based on the physical naturalness and feasibility of robot-object interactions.

- Table 6: Quantitative results on the VP2 benchmark, reporting mean, min, and max performance over various control runs.

Tasks

Success rate

iVideoGPT (ours)

FitVid SVG′ MCVD MaskViT

StructVRNN

Simulator

Robosuite push

mean 0.7833 0.6760 0.7980 0.7733 0.8260 0.5540 0.9350 max 0.7950 0.7900 0.8400 0.7900 0.8500 0.6000 0.9500 min 0.7750 0.6400 0.7600 0.7400 0.7900 0.5000 0.9200

Flat block

mean 0.0333 0.0917 0.0200 0.0500 0.0400 0.0467 0.1333 max 0.0417 0.1333 0.0333 0.0667 0.1000 0.1333 0.1333 min 0.0250 0.0667 0.0000 0.0333 0.0000 0.0000 0.1333

Open drawer

mean 0.3750 0.2533 0.1667 0.1167 0.0400 0.0267 0.7667 max 0.3917 0.3333 0.2667 0.1333 0.1000 0.1000 0.7667 min 0.3500 0.1333 0.0667 0.1000 0.0000 0.0000 0.7667

Open slide

mean 0.1611 0.3533 0.5733 0.1833 0.0867 0.1267 0.7167 max 0.1917 0.4000 0.7333 0.2000 0.1667 0.2333 0.7333 min 0.1250 0.2667 0.4667 0.1667 0.0333 0.0667 0.7000

Blue button

mean 0.9556 0.9400 0.9733 0.9500 0.9467 0.8667 1.0000 max 0.9833 0.9667 1.0000 1.0000 0.9667 0.9000 1.0000 min 0.9333 0.8667 0.9333 0.9000 0.9333 0.8000 1.0000

Green button

mean 0.8250 0.8400 0.8133 0.8333 0.6400 0.6800 0.9667 max 0.8667 0.9000 0.9000 0.8333 0.7000 0.8000 0.9667 min 0.7833 0.7667 0.7667 0.8333 0.6000 0.5667 0.9667

Red button

mean 0.9222 0.5867 0.7600 0.7333 0.2400 0.3067 0.9000 max 0.9333 0.6333 0.8667 0.7333 0.3333 0.3333 0.9000 min 0.9000 0.5000 0.6333 0.7333 0.1333 0.2333 0.9000

Upright block

mean 0.4472 0.5133 0.4867 0.5667 0.6200 0.3333 0.9000 max 0.4667 0.5667 0.6667 0.6000 0.7333 0.3667 0.9000 min 0.4250 0.5000 0.4000 0.5333 0.5000 0.3000 0.9000

tokenizer design, it is not the bottleneck of generation time. Additionally, although we use more tokens for context frames compared to the 4 × 4 tokenizer, generation time is primarily influenced by the number of forward passes of the autoregressive transformer, which remains the same.

- Table 7: Training efficiency of the transformer with various tokenizers, measured on 40G A100 GPUs with a per-device batch size of 16.

Tokenizer Speed (#iters/sec) Memory (GB)

4 × 4 3.10 10.6 16 × 16 N/A OOM Ours 2.62 22.3

- Table 8: Generation efficiency with various tokenizers, measured on an RTX 4090 GPU with a batch size of 1.

Tokenizer Tokenize (sec) Generation (sec) Detokenize (sec) Memory (GB)

4 × 4 0.27 1.13 0.05 1.98 16 × 16 0.26 22.5 0.04 1.90

Ours 0.29 1.11 0.06 2.33

Button Press Topdown Wall

Plate Slide

Hammer

100

100

100

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

SuccessRate(%)

80

80

80

60

60

60

40

40

40

DrQ-v2

20

20

20

MBPO (w/ FitVid)

MBPO (w/ iVideoGPT, ours)

0

0

0

0 1 2 3 4 5 6

0 2 4 6 8 10

0 2 4 6 8 10

Environment Steps (× )

Environment Steps (× )

Environment Steps (× )

Door Lock

Handle Pull Side

Coffee Push

100

100

100

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

SuccessRate(%)

80

80

80

60

60

60

40

40

40

20

20

20

0

0

0

0 5 10 15

0 5 10 15 20

0 10 20 30 40

Environment Steps (× )

Environment Steps (× )

Environment Steps (× )

- Figure 21: Visual model-based RL on Meta-world, comparing to an additional baseline using FitVid

- as world models.

### C Extended Discussions

#### C.1 Differences with IRIS

Discrete tokenization and autoregressive transformers are prevalent in contemporary deep learning due to their simplicity and generality. iVideoGPT generally shares this architecture with IRIS [63], but possesses distinguishing features, summarized as follows:

- • Pre-training and fine-tuning paradigm: iVideoGPT is designed for a paradigm that involves pre-training on large-scale videos and fine-tuning on various downstream tasks. In contrast, IRIS focuses solely on MBRL with Transformer-based world models trained from scratch in the Atari domain.
- • Efficient tokenization: iVideoGPT proposes novel compressive tokenization to significantly reduce the number of tokens, saving time and memory (see Table 7 and 8), while IRIS uses per-frame tokenization.
- • Flexible action-conditioning design: iVideoGPT employs slot tokens with optional additive action embeddings to support both action-free pre-training and action-conditioned finetuning, while IRIS strictly treats discrete Atari actions as tokens.
- • Off-policy MBRL implementation: iVideoGPT uses an off-policy RL algorithm while IRIS performs on-policy learning. On-policy learning needs a large number of model rollouts, which, combined with inefficient tokenization, results in 7 days for 100k-environment-step training. In comparison, iVideoGPT only needs ∼4 hours.

#### C.2 Differences with VideoGPT

We elaborate on the the difference between the tokenizer in VideoGPT [101] and ours, and how they impact interactivity.

VideoGPT uses a VQVAE for video that relies on a series of 3D convolutions to downsample across space and time. For example, it downsamples original pixels from 16 × 64 × 64 to discrete tokens of 8 × 32 × 32 or 4 × 16 × 16, depending on the downsampling ratio. The key issue is that this non-causal downsampling over the temporal dimension results in each token containing information from a window of frames. As a result, the entire video of a fixed length can only be reconstructed after VideoGPT generates all tokens. As shown in Figure 2, VideoGPT only allows the input of future action sequences at the beginning of prediction, preventing an agent from interactively determining its actions based on predicted observations. In contrast, our tokenizer discretizes video frames separately, using a conditional mechanism to handle temporal redundancy, enabling frame-by-frame video generation and allowing for intermediate action intervention.

Moreover, our tokenizer’s novel design, with its cross-attention mechanism, is more efficient in handling temporal redundancy, converting videos into significantly fewer tokens (L = 511 with N = 256,n = 16,T = 16,T0 = 1 as stated in Section 3.1). In contrast, VideoGPT finds that using a larger downsampling ratio than a token size 8 × 32 × 32, results in worse performance.

#### C.3 Failure Case Analysis for Visual Planning

Our model performs sub-optimally on the RoboDesk open slide task from the VP2 benchmark. In this section, we investigate the underlying causes through case studies, attributing the performance issues to limitations in both our model and the benchmark.

Inaccurate model prediction. Despite achieving excellent overall video prediction metrics, such as mean square error and perceptual loss, on the validation set for the open slide task, our model predicts wrong outcomes on a few trajectories. We visualize these trajectories in Figure 22 and find that while the observation is limited to 64 × 64 resolution, the task of opening the slide requires the model to capture subtle changes, particularly whether the robot’s gripper has made contact with the slide handle. Actually, even humans struggle to discriminate this detail with low-resolution inputs. Due to this uncertainty, the model may incorrectly predict a sequence of imprecise actions as successful. This overconfidence [66] can be exacerbated in the process of model predictive control, which samples a large number of action candidates and selects the "best" one according to the model. Our analysis provides an explanation to the observation by Tian et al. [86] that overall excellent perceptual metrics do not always correlate with effective control performance, as the worst-case scenarios are critical in model-predictive control.

Furthermore, we hypothesize that our two-stage architecture of tokenization and prediction can exacerbate the aforementioned uncertainty, as discrete tokenization inevitably results in some loss of information from the observations. This hypothesis is supported by the fact that end-to-end models like SVG′ [91] and FitVid [4] perform significantly better than two-stage models, including ours and others like MaskViT [27], which uses a visual tokenizer, and Struct-VRNN [64], which employs a keypoint-based representation.

We anticipate that training and evaluating our model at a higher resolution, such as 256 × 256, could mitigate these issues and enhance control performance. However, we currently conduct experiments

- at a lower resolution to ensure a fair comparison with other models.

Imperfect built-in reward design. We observe that no current model in the VP2 benchmark consistently outperforms other models across all tasks, and iVideoGPT is no exception. Beyond models’ inaccuracies in prediction for severely out-of-distribution (OOD) actions, our analysis of this inconsistent performance also reveals flaws in the benchmark’s built-in reward design.

In VP2, scores for sampled actions are estimated mainly by a learned classifier that assesses task success based on model-predicted frames. This classifier, trained by the VP2 authors, appears to lack robustness and is easily fooled by OOD inputs, assigning high rewards to low-quality or unlikely-tosucceed predicted trajectories (see examples in Figure 23). Such an imperfect reward function likely contributes to the mixed results observed on this benchmark, with iVideoGPT even outperforming the oracle simulator in one task. Addressing visual planning with imperfect rewards is an independent research problem and beyond the scope of this paper.

###### VP2: RoboDesk (Failure Cases)

t = 0 t = 1 t = 2 t = 3 t = 4 t = 5 t = 6 t = 7 t = 8 t = 9 t = 10 t = 11

[Figure 1071]

[Figure 1072]

[Figure 1073]

[Figure 1074]

[Figure 1075]

[Figure 1076]

[Figure 1077]

[Figure 1078]

[Figure 1079]

[Figure 1080]

[Figure 1081]

[Figure 1082]

Ground truth

[Figure 1083]

[Figure 1084]

[Figure 1085]

[Figure 1086]

[Figure 1087]

[Figure 1088]

[Figure 1089]

[Figure 1090]

[Figure 1091]

[Figure 1092]

(context)

Predicted

[Figure 1093]

[Figure 1094]

[Figure 1095]

[Figure 1096]

[Figure 1097]

[Figure 1098]

[Figure 1099]

[Figure 1100]

[Figure 1101]

[Figure 1102]

[Figure 1103]

[Figure 1104]

Ground truth

[Figure 1105]

[Figure 1106]

[Figure 1107]

[Figure 1108]

[Figure 1109]

[Figure 1110]

[Figure 1111]

[Figure 1112]

[Figure 1113]

[Figure 1114]

(context)

Predicted

[Figure 1115]

[Figure 1116]

[Figure 1117]

[Figure 1118]

[Figure 1119]

[Figure 1120]

[Figure 1121]

[Figure 1122]

[Figure 1123]

[Figure 1124]

[Figure 1125]

[Figure 1126]

Ground truth

[Figure 1127]

[Figure 1128]

[Figure 1129]

[Figure 1130]

[Figure 1131]

[Figure 1132]

[Figure 1133]

[Figure 1134]

[Figure 1135]

[Figure 1136]

(context)

Predicted

- Figure 22: Failure case analysis on the RoboDesk open slide task from the VP2 benchmark, where, likely due to the low resolution of observations, our model fails to discriminate between subtle changes, particularly whether the robot’s gripper has made contact with the slide handle.

[Figure 1137]

[Figure 1138]

[Figure 1139]

[Figure 1140]

[Figure 1141]

[Figure 1142]

[Figure 1143]

[Figure 1144]

[Figure 1145]

[Figure 1146]

[Figure 1147]

Failure trajectory with Higher reward (-7179.45)

t = 0 t = 2 t = 4 t = 6 t = 8 t = 10 t = 11

[Figure 1148]

[Figure 1149]

Reasonable trajectory with Lower reward (-7308.77)

- Figure 23: Imperfect built-in reward in VP2 benchmark. A learned reward model can assign high rewards to predicted transitions that are less likely to succeed, which can mislead optimizers in model-predictive control.

### D Computational Resources

We implement iVideoGPT in PyTorch, using the diffusers8 and transformers9 libraries. Our models are trained and evaluated on an A100 and RTX 3090 GPU cluster. Each experiment utilizes 4 GPUs in parallel, with 16 data loader workers per device. GPU days required for training are reported in Table 2. Experiments at 64 × 64 resolution can be conducted with 24 GB of GPU memory per device, while 256 × 256 resolution requires 40 GB. The Open X-Embodiment dataset is particularly large, occupying about 5TB of disk space.

### E Broader Impact

World models advance the development of autonomous machine intelligence, particularly through the valuable visual insights offered by videos. However, their full potential remains untapped without scalable and interactive architectures capable of distilling vast amounts of commonsense knowledge from multimodal data. This paper, we believe, takes an important step by introducing a flexible framework with a specific focus on the robotic manipulation domain. Our results may pave the way for higher-quality world models applicable across diverse domains, enhancing performance in control tasks of embodied intelligence. Despite the benefits, designing and training these models is challenging, requiring substantial computational power and increasing the carbon footprint. Using

- 8https://github.com/huggingface/diffusers under Apache License
- 9https://github.com/huggingface/transformers under Apache License

underdeveloped, inaccurate world models for autonomous control could lead to risky actions and potential physical damage, which can be mitigated by developing uncertainty-aware models to prevent uncertain actions. Conversely, the advancement of these models could lead to job displacement in sectors relying on manual control tasks. Additionally, the underlying techniques for world models can be misused to generate synthetic videos that mimic real events or people. However, since our model is merely a research prototype, trained only with robotic and human manipulation data and relatively small in scale, we do not anticipate immediate negative societal impacts such as deepfakes or job displacement.

