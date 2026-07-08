# arXiv:2412.04445v4[cs.RO]16Oct2025

## Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos

Yi Chen1∗, Yuying Ge2†, Weiliang Tang3, Yizhuo Li1,2, Yixiao Ge2, Mingyu Ding4, Ying Shan2, Xihui Liu1† 1The University of Hong Kong, 2ARC Lab, Tencent PCG, 3The Chinese University of Hong Kong, 4University of California, Berkeley https://chenyi99.github.io/moto/

[Figure 1]

[Figure 2]

###### PERFORMANCE ON SIMPLER

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Latent Motion Next Latent Motion Token Prediction

[Figure 11]

[Figure 12]

###### Tokenizer

0.75

SUCCESSRATE

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

[Figure 28]

0.55

[Figure 29]

[Figure 30]

##### Moto-GPT

[Figure 31]

0.35

[Figure 32]

[Figure 33]

[Figure 34]

Pick Move Drawer Overall

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

#### Videos

[Figure 53]

S

[Figure 54]

[Figure 55]

###### PERFORMANCE ON CALVIN

[Figure 56]

Tokenized Motion Trajectory

Text Initial Frame

[Figure 57]

0.95

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

SUCCESSRATE

Motion Priors

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

###### I have learned about:

[Figure 69]

0.55

- • Motion Semantics.
- • Trajectory Rationality.
- • Future Anticipation.
- • …

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Act

[Figure 75]

0.15

1 2 3 4 5

Moto w/o Motion Token Moto

Figure 1. The overview of Moto, which utilizes Latent Motion Tokens as a bridging “language” for autoregressive pretraining on video data. The Moto-GPT pre-trained through next motion token prediction learns a wealth of motion-related prior knowledge from videos, which can be seamlessly transferred to enhance downstream robot manipulation tasks with significant performance gains.

###### Abstract

Recent developments in Large Language Models (LLMs) pre-trained on extensive corpora have shown significant success in various natural language processing (NLP) tasks with minimal fine-tuning. This success offers new promise for robotics, which has long been constrained by the high cost of action-labeled data. We ask: given the abundant video data containing interaction-related knowledge available as a rich “corpus”, can a similar generative pretraining approach be effectively applied to enhance robot learning? The key challenge is to identify an effective representation for autoregressive pre-training that benefits robot manipulation tasks. Inspired by the way humans learn new skills through observing dynamic environments, we propose that effective robotic learning should emphasize motion-related knowledge, which is closely tied to low-level

*Part of the work done during internship at Tencent ARC Lab. †Corresponding Authors.

actions and is hardware-agnostic, facilitating the transfer of learned motions to actual robot actions. To this end, we introduce Moto, which converts video content into latent Motion Token sequences by a Latent Motion Tokenizer, learning a bridging “language” of motion from videos in an unsupervised manner. We pre-train Moto-GPT through motion token autoregression, enabling it to capture diverse visual motion knowledge. After pre-training, Moto-GPT demonstrates the promising ability to produce semantically interpretable motion tokens, predict plausible motion trajectories, and assess trajectory rationality through output likelihood. To transfer learned motion priors to real robot actions, we implement a co-fine-tuning strategy that seamlessly bridges latent motion token prediction and real robot control. Extensive experiments show that the fine-tuned Moto-GPT exhibits superior robustness and efficiency on robot manipulation benchmarks, underscoring its effectiveness in transferring knowledge from video data to downstream visual manipulation tasks.

###### 1. Introduction

Recent advancements in Natural Language Processing (NLP) have stemmed from successful autoregressive pretraining on large text corpora via next-word prediction [5, 17, 43, 45, 50]. Pre-trained Large Language Models (LLMs) have shown exceptional performance across various downstream NLP tasks after fine-tuning on smaller datasets. This success opens new opportunity for robotics, which has been limited by the high costs of action-labeled data. Given the abundance of interaction-rich video data [1, 57], we ask: Can we leverage autoregressive pre-training on video data to improve robot learning?

The main challenge is finding an appropriate representation for autoregressive pre-training on video data that effectively captures prior knowledge for robot manipulation. Pioneering research in video pre-training for robotics primarily focused on static frames, emphasizing frame-level visual details [8, 18, 54]. However, humans learn skills by observing dynamic environments, focusing on changes in state—what we term motion. Thus, we argue that effective autoregression for robotics should prioritize motion-related knowledge, which aligns closely with low-level robot actions and is hardware-agnostic, facilitating the transfer of learned motions to actual robot actions through fine-tuning.

In this work, we introduce Moto, which utilizes Latent Motion Tokens as a bridging “language” to model visual motions between video frames in an unsupervised manner. As illustrated in Fig. 1, we first train a discrete Latent Motion Tokenizer to produce compact latent motion tokens that capture dynamics between video frames without external supervision. We then pre-train Moto-GPT using a GPTbased architecture to predict the next latent motion token, absorbing motion priors from videos. These learned priors are subsequently transferred to enhance robot manipulation performance through a co-fine-tuning strategy.

Specifically, as shown in Fig. 2, the Latent Motion Tokenizer encoder employs a VQ-VAE-based architecture [51] to compress two successive video frames into discrete tokens. By regularizing the decoder to reconstruct the second frame from the first frame and the tokens, the tokenizer is trained to effectively capture the changes between video frames, which often arise from motion. Once the tokenizer is trained, we obtain latent motion tokens of every two consecutive frames in a video clip and concatenate them into a sequence to represent the motion trajectory. Subsequently, Moto-GPT is pre-trained on these sequences by predicting the next token based on the initial frame and corresponding language instruction. After this pre-training phase, MotoGPT is capable of generating plausible trajectories by predicting latent motion tokens autoregressively.

To adapt Moto-GPT for downstream robot manipulation tasks, we concatenate action query tokens with latent motion token chunk at each time step for co-fine-tuning on

action-labeled robot data. The action query tokens are processed by a learnable module to predict low-level actions, while the motion tokens are fine-tuned using the original next-token prediction mechanism. This co-fine-tuning strategy effectively transfers abstract intentions in learned motion priors into precise action execution, allowing the model to utilize the inherent knowledge of the pre-trained MotoGPT for successful manipulation.

We conduct extensive experiments to validate our claims from various perspectives: (1) Latent Motion Tokens as an Interpretable Motion Language: Experiments show that latent motion tokens encapsulate compact and expressive representations of visual motions and exhibit promising cross-embodiment transfer ability (even from human to robot). (2) Pre-trained Moto-GPT as a Useful Motion Prior Learner: Results indicate that pre-trained Moto-GPT achieves promising outcomes in predicting plausible motion trajectories and assessing the rationality of robot trajectories based on output likelihood. (3) Fine-tuned Moto-GPT as an Effective Robot Policy: The fine-tuned Moto-GPT demonstrates significant performance gains over counterparts trained from scratch without motion priors, especially with limited training data. The performance can be further boosted with human video pre-training, highlighting the potential of our approach in transferring motion knowledge learned from Internet-scale videos to robot manipulation.

In summary, our contributions are as below:

- • Introduction of Latent Motion Tokens, which model visual motions between video frames in an unsupervised manner, serving as a bridging “language” for autoregressive pre-training to enhance robot learning.
- • Pre-training of Moto-GPT through next latent motion token prediction on video data, enabling the model to learn useful motion priors without requiring action annotations.
- • Implementation of a co-fine-tuning strategy to successfully transfer learned motion priors to actual robot manipulations, with the fine-tuned model showing competitive performance on robotic benchmarks.
- • We conduct systematic experiments and analyses to validate each training stage of our approach and the effectiveness of the resulting robot policy. Our open-source code provides clear guidance for future exploration.

###### 2. Related Work

Vision-Language-Action Models. Recent studies have utilized transformers as vision-language-action (VLA) architectures to generate robot actions from observations and language instructions [4, 25, 30, 47]. VLA model pretraining has gained traction, with approaches either finetuning policy models from powerful vision-language models pre-trained on large image-text datasets [15, 32, 62] or training generalist models on diverse robot data with action labels [2, 14, 27, 41, 52]. Our work enhances VLA mod-

|Δ𝑥 = [0.1,−0.2,0], Δ𝜃 = [10°,25°,−7°] , Δ𝑔𝑟𝑖𝑝 = 1| |
|---|---|
| | |

[Figure 76]

De-tokenize

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

…

…

[Figure 81]

[Figure 82]

[Figure 83]

###### Pre-training

Co-fine-tuning

ViT Decoder

[Figure 84]

[Figure 85]

Action Head

39 17 … 63 48

39 17 … 63 48

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

…

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

…

…

…

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

>>>>>>>>

[Figure 108]

[Figure 109]

Learned

Moto-GPT

Moto-GPT

[Figure 110]

[Figure 111]

Motion

MLP

Reconstruct

Priors

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

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

>>>>>>>>

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Tokenize

[Figure 134]

…

###### …

…

…

39 17 … 63 48

…

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

VQ Codebook

inserted during fine-tuning stage

[S]

[S]

39 17 … 63 48

39 17 … 63 48

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

…

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Latent Motion

M-Former

T5

ViT

Tokenize

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

ViT Encoder

| | | | |
|---|---|---|---|
|[Figure 155]| | | |
| |𝑜0| | |

|[Figure 156]| | |
|---|---|---|
| |𝑜1| |

|[Figure 157]| | |
|---|---|---|
| |𝑜𝑇| |

[Figure 158]

Task: Place pepsi can upright.

[Figure 159]

[Figure 160]

Text

Start Token Action Query

𝑜𝑡−1 𝑜𝑡

Latent Motion

… …

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

Initial Frame

Tokenizer

[Figure 166]

Latent Motion Token

Figure 2. Overview of Moto’s three training stages: (1) The Latent Motion Tokenizer encodes key visual motions between video frames into compact latent tokens in an unsupervised manner using pure video data. (2) Moto-GPT is pre-trained with autoregressive motion token prediction to learn motion priors from video-instruction pairs. (3) Moto-GPT is co-fine-tuned on action-labeled trajectories to predict robot actions based on the output of learnable action query tokens while maintaining the next-motion-token prediction objective.

els through generative pre-training on video data, providing richer interaction details. Beyond VLA models, several studies improve robot manipulation by incorporating multi-perspective views and depth information [7, 35, 59], and employing techniques like action chunking and policy diffusion for precision [11, 22, 26]. Additionally, some works [19, 34] decompose high-level language instructions into latent skills via auxiliary training objectives.

Robot Learning from Videos. Early works [37, 42] used contrastive learning with egocentric videos to enhance visual representations. Some studies [3, 16, 28, 29, 33] generate videos as intermediate plans for low-level control. Recent research [8, 23, 54] has focused on generative video pre-training for end-to-end policy models. Escontrela et al. [18] pre-train autoregressive video prediction models for reinforcement learning. These works primarily use pixel values or patch-level tokens. In contrast, our approach targets latent motion tokens, emphasizing key visual motions. Some studies build world models through actionconditioned video generation [21, 55, 56], aiding reinforcement learning. Genie [6] introduces unsupervised learning of latent actions from large-scale videos for 2D gaming simulators. Similar latent action concepts are explored in LAPO [49] and DynaMo [12], though LAPO is limited to video games and DynaMo focuses on visual representation learning. Concurrently, LAPA [58] is pre-trained to predict one-step future latent action, while IGOR [10] uses latent actions as intermediate goals. Our work differs by pre-training an end-to-end policy model to autoregressively predict a trajectory of latent motion tokens for future video clips, providing a more natural modeling of sequential visual motions from videos. Additionally, we conduct a rigorous analyses to validate the effectiveness of policy and each training stage, offering clear guidance for future research.

###### 3. Methodology

###### 3.1. Overview

Moto utilizes autoregressive generative pre-training on latent motion token sequences to learn motion priors from videos, followed by co-fine-tuning on action-labeled data for robot control. As illustrated in Fig. 2, Moto consists of three stages: 1) unsupervised training of the Latent Motion Tokenizer, 2) pre-training of the generative model MotoGPT, and 3) co-fine-tuning for robot action policy. In Sec. 3.2, we detail the Latent Motion Tokenizer, which encodes visual dynamics into quantized latent motion tokens. We also describe the training procedures for Moto-GPT, including motion token autoregressive pre-training in Sec. 3.3 and supervised co-fine-tuning in Sec. 3.4. Implementation details can be found in the Supplementary Material.

###### 3.2. Latent Motion Tokenizer

The Latent Motion Tokenizer, as shown in Fig. 2, learns a latent “language” to capture essential visual motions between successive video frames1 in an unsupervised manner. The architecture follows a standard auto-encoder design for motion tokenization and detokenization. The tokenization employs an M-Former, a multi-layer transformer that extracts motion features from the last-layer patch features of the current frame ot and the preceding frame ot−1 using a frozen pre-trained ViT encoder [24]. We concatenate 8 learnable query embeddings with these patch features as additional input to the M-Former, where the queries interact through self-attention layers. The output query features are then processed by a VQ codebook with a vocabulary size of 128 to produce discrete latent motion tokens.

1To ensure significant visual differences, we down-sample the original video by a certain rate.

For de-tokenization, we use a ViT Decoder for image reconstruction, which takes the linearly embedded patches of ot−1 and recovers the pixel values for ot based on the latent motion tokens. An MLP projects the concatenated quantized embeddings of the latent motion tokens into a compact embedding (1 token), which is added to each input patch embedding. This conditional embedding acts as an information bottleneck between the encoder and decoder, enabling the ViT Decoder to capture nuanced changes between frames and accurately transform ot−1 into ot.

The components of the Latent Motion Tokenizer are jointly optimized using the standard VQ-VAE objective [51], which includes reconstruction loss, vector quantization loss, and commitment loss. We specifically use the MSE loss between the output pixel values from the ViT Decoder and the ground-truth pixel values of ot as the reconstruction loss. Once trained, the Latent Motion Tokenizer is frozen to produce unified sequential motion representations for videos through “bi-frame” tokenization. Additionally, with the initial observation and specified latent motion tokens, the decoder can function as a “simulator” to generate rollouts for visualizing environmental changes.

###### 3.3. Motion Token Autoregressive Pre-training

With the Latent Motion Tokenizer, Moto-GPT is allowed to learn about diverse visual motions from videos, using latent motion tokens as a bridging language. As shown in Fig. 2, Moto-GPT is pre-trained with a next-motion-token prediction objective. For a video clip [o0,o1,...,oT], we derive a chunk of latent motion tokens for each pair of consecutive frames, concatenating them chronologically to form a sequence. Moto-GPT employs a GPT-style transformer for autoregression on these motion token trajectories. Additionally, we prepend the text features from the instruction and the visual features from the initial video frame as input prompts. The pre-training objective maximizes the likelihood of the ground-truth latent motion token sequence given the language instruction and the initial video frame:

M

Lmotion = −

i=1

log P(mi|l,v,m<i;Θ), (1)

where l and v are text and visual features from the frozen pre-trained T5 [46] and ViT [24] models, respectively. m<i represents the latent motion tokens preceding the current token mi, and Θ denotes the trainable model parameters. Here, M = K∗T, where K is the number of tokens for motion between successive frames and T is the video length.

###### 3.4. Co-fine-tuning for Robot Manipulation

After pre-training, Moto-GPT can anticipate future trajectories by generating latent motion tokens based on language instructions and initial observations. This process resem-

###### Pick-Place Close Disassembly

[Figure 167]

[Figure 168]

[Figure 169]

place the banana into the pan close the laptop pull the blue stick out of the yellow base

Figure 3. Illustration of real-world evaluation tasks.

bles the policy inference of real robots if we take the codebook of latent motion tokens as an abstract action space. However, a gap remains in achieving precise robot control.

To address this, during fine-tuning, we introduce special action query tokens into Moto-GPT’s input, enabling the generation of real robot actions through a flexible action head, as illustrated in the right part of Fig. 2. Specifically, N query tokens are added after the latent motion token chunk at each time step, where N corresponds to the number of robot actions occurring between two video frames. The fine-tuning stage follows the same causal mask mechanism as pre-training in general. Nevertheless, the latent motion tokens do not attend to the newly inserted action query tokens to stay consistent with the pre-training setting. Besides, we randomly mask 50% of the attention from action query tokens to latent motion tokens, allowing knowledge transfer while reducing dependency on ground-truth conditions. This also improves inference efficiency, enabling direct queries to Moto for real actions without generating latent motion tokens. This can be achieved by using padding tokens as placeholders for latent action tokens, blocking attention from action query tokens to these placeholders.

An MLP-based action head projects the output hidden state of each action query token into the real robot action space. We apply Smooth-L1 loss for continuous action components, such as positional (∆x) and rotational (∆θ) displacements, and Binary Cross Entropy (BCE) loss for binary components, like the gripper’s open/close state (∆grip)2. The total action loss Laction is defined as:

Laction = L(∆x) + L(∆θ) + L(∆grip) (2)

We retain the training objective for latent motion token prediction to ensure Moto-GPT retains the motion priors learned from videos. Thus, the overall loss function for the fine-tuning stage is:

Lft = Lmotion + Laction (3)

###### 4. Benchmarks and Datasets

More details of experimental settings can be found in the supplementary material.

2The action space may vary with different robot embpdiments. For example, the Google Everyday Robot uses a continuous value for gripper extension, necessitating Smooth-L1 loss for ∆grip.

###### Initial Frame : forward : backward : down : left, forward :right, forward

|[Figure 170]|
|---|

|[Figure 171]|
|---|

|[Figure 172]|
|---|

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

|[Figure 176]|
|---|

|[Figure 177]|
|---|

|[Figure 178]|
|---|

|[Figure 179]|
|---|

|[Figure 180]|
|---|

|[Figure 181]|
|---|

###### [69,35,34,36,108,117,101] [61,8,48,90,108,60,39,118] [62,81,108,20,41,60,19,64] [68,119,41,60,123,101,39,41] [34,60,93,25,11,13,72,117]

- Figure 4. Interpretability of latent motion tokens. Each row displays reconstructed frames from the same initial frame using different latent motion tokens, while each column shows frames reconstructed from the same latent motion tokens with varying initial frames. The latent motion tokens exhibit consistent (see columns) and discriminative (see rows) semantics, despite being trained in an unsupervised manner.

[93,11,86,64,111,16,100,0] [16,13,111,60,37,25,42,121] [84,103,47,116,113,2,99,55] [71,72,79,36,80,0,70,107] [81,103,54,96,100,92,9,24] [39,112,22,33,60,68,32,62]

Imitation Video

|[Figure 182]|
|---|

Demonstration Video

|[Figure 183]|
|---|

|[Figure 184]|
|---|

- Initial Frame A
- Initial Frame B

|[Figure 185]|
|---|

- Figure 5. Video imitation generation via latent motion tokens, where a sequence of motion tokens extracted from a demonstration video are decoded into a new video. This generated video is based on a different initial frame while preserving the original movement semantics.

SIMPLER [31]. We focus on three tasks with the Google Everyday Robot: “pick coke can”, “move near”, and “open/close drawer”. For pre-training, we leverage a subset of Open-X-Embodiment (OXE) [52], consisting of 109k trajectory videos [4, 9, 13, 36, 38, 40, 44, 48, 53, 60, 61]. Fine-tuning is performed using 73k action-labeled expert trajectories from the RT-1 Robot-Action dataset [4].

CALVIN (ABC−→D) [39]. We assess long-horizon task completion with the Franka Emika Panda robot, requiring consecutive completion of 5 out of 34 tasks in each trial in a zero-shot setting. Pre-training involves all play videos from environments A, B, and C, with 35% (18k videos) containing language annotations. Fine-tuning uses 18k expert trajectories with action labels from the same environments. Testing is conducted in the unseen environment D.

Real-world Robot Experiments. We conduct real-world evaluations with a FANUC LR Mate 200iD robot on three tasks: “pick-place banana”, “close laptop”, and “disassembly” (Fig. 3). Pre-training utilizes OXE data, and we collect 90 teleoperated demonstrations (30 per task) for finetuning. Each task is tested 10 times with randomized object positions. Generalizability is evaluated in two scenarios: (i)

Novel Object, altering the object’s color, texture, and shape; (ii) Visual Distractor, introducing irrelevant objects.

###### 5. Experiments

To comprehensively evaluate the effectiveness of Moto, we study three key experimental questions:

- • Q1 (Interpretability): Does the Latent Motion Tokenizer learn interpretable latent motion tokens that effectively represent visual motions from videos?
- • Q2 (Motion Priors): Does Moto-GPT gain meaningful prior knowledge of motion trajectories through autoregressive pre-training on latent motion token sequences?
- • Q3 (Performance): Can the motion priors be transferred to enhance policy performance in robot manipulation benchmarks through efficient fine-tuning?

###### 5.1. Latent Motion Token as an Interpretable Motion Language

The Latent Motion Tokenizer effectively reconstructs authentic next frames using ground-truth latent motion tokens, capturing key dynamics between initial and subse-

Pick apple from top drawer and place on counter

|[Figure 186]|
|---|

Initial Frame

|[Figure 187]|
|---|

Place apple into top drawer

|[Figure 188]|
|---|

- Figure 6. Visualization of video trajectories generated from a sequence of latent motion tokens, which are predicted by the pretrained Moto-GPT given different language instructions.

- Table 1. Video classification accuracy with varied representations.

Video Representation Semantic Acc.

Initial frame 0.292 Initial frame repeated by 8 times 0.283 Initial frame + 7 subsequent frames 0.828 Initial frame + 7 latent motion token chunks 0.797

quent frames, as evidenced by supplementary examples. This demonstrates its ability to represent fine-grained motion details, with its decoder acting as a reliable simulator for visualizing environmental changes. Fig. 4 highlights the controllability and consistency of latent motion tokens. Different token chunks produce varied motion orientations and scales, while identical chunks yield consistent effects across different starting observations with varied robot embodiments. By concatenating token chunks from consecutive frames, motion trajectories can be represented sequentially, akin to natural language, enabling contextualized motion generation across diverse initial observations (Fig. 5). This underscores the potential of latent motion tokens as a unified language for imitation learning.

Quantitatively, Table 1 shows the semantic interpretability of latent motion tokens. A video classifier using initial frame ViT patch features and concatenated latent motion tokens for seven subsequent frames achieves 79.7% accuracy in predicting semantic labels for 34 CALVIN tasks. This performance is comparable to using ViT features for all eight frames, despite reducing input features from 196 to 8 tokens per frame, confirming that latent motion tokens provide a compact, expressive, and interpretable representation of visual motions linked to high-level semantics.

###### 5.2. Moto-GPT as a Useful Motion Prior Learner

Moto-GPT is pre-trained through autoregression on videos using latent motion tokens, enabling it to predict motion trajectories from initial observations and diverse language prompts, as shown in Fig. 6. The top-k token prediction accuracy and the visualization of predicted video trajectories

success failure random

3.5

LogLikelihood()

4.0

4.5

5.0

20 40 60 80

Sequence Step

Figure 7. Moto-GPT distinguishes successful, failed, and random robot trajectories using log-likelihoods, enabling effective assessment of trajectory rationality and potential reward signals.

for more complex actions (e.g., “rotation” and “stacking”) are provided in the supplementary material. This demonstrates Moto-GPT’s ability to acquire prior knowledge essential for robot action inference based on human instructions. Additionally, latent motion tokens allow Moto-GPT to interpret trajectory videos as compact token sequences and evaluate their rationality through the autoregressive likelihood (Eq. 3.3). Fig. 7 illustrates the potential of using Moto’s log-likelihoods as a reward signal for trajectory videos, indicating how well a trajectory aligns with MotoGPT’s distribution and measuring the temporal consistency of behavior. To assess this, we collected 98 video triplets in CALVIN using the baseline policies and a random policy. Each triplet consists of three types of trajectory videos originating from the same environment state. The averaged loglikelihoods for each trajectory type at each sequence step, shown in Fig. 7, clearly differentiate successful trajectories from failures and random attempts.

###### 5.3. Moto-GPT as an Effective Robot Policy

Overall Performance. After fine-tuning, Moto-GPT3 was evaluated on the SIMPLER and CALVIN benchmarks, demonstrating promising results as shown in Tables 2 and 3. It significantly outperforms Moto w/o Motion Token, which is trained from scratch without latent motion tokens, underscoring the effectiveness of transferring motion priors learned from videos to enhance robot manipulation tasks. Despite having only 98M parameters for the GPT backbone and no access to action labels during pre-training, MotoGPT performs comparably to larger models like RT-2-X (PaLI-X 55B) and OpenVLA (Prismatic-7B) on SIMPLER. It also maintains competitiveness against OpenVLA (finetuned), which is further fine-tuned specially on the RT-1 Robot-Action trajectories, despite its pre-training data already containing action labels from this dataset. Moto-GPT

3Moto-GPT is referred to as Moto in the following tables and figures.

- Table 2. SIMPLER evaluation results of models pre-trained on Open-X-Embodiment [52] datasets. The “Overall” column reports the success rate averaged across the sub-tasks of all task types.

Method

Pick Coke Can Move Near Open / Close Drawer Overall Horizontal Vertical Standing Average Average Open Close Average Average

- RT-1-X [4] 0.820 0.330 0.550 0.567 0.317 0.296 0.891 0.597 0.534
- RT-2-X [62] 0.740 0.740 0.880 0.787 0.779 0.157 0.343 0.250 0.607 Octo-Base [41] 0.210 0.210 0.090 0.170 0.042 0.009 0.444 0.227 0.169 OpenVLA [27] 0.270 0.030 0.190 0.163 0.462 0.194 0.518 0.356 0.248 OpenVLA (fine-tuned) [27] 0.470 0.080 0.540 0.363 0.542 0.102 0.361 0.231 0.349

Moto 0.820 0.500 0.900 0.740 0.604 0.130 0.732 0.431 0.614 Moto w/o Motion Token 0.600 0.190 0.740 0.503 0.554 0.000 0.796 0.398 0.480

- Table 3. Comparison of models adopting different pre-training techniques on CALVIN (ABC−→D). Avg. Len. is a comprehensive metric indicating the average number of tasks accomplished in a row across 1,000 trial sequences. “Static RGB” and “Gripper RGB” denote the RGB images from a static camera or a gripper view, respectively. “Proprio” is short for the proprioceptive robot state.

Tasks competed in a row (1000 chains) 1 2 3 4 5 Avg. Len.

Model Observation Space

SuSIE [3] Static RGB 0.870 0.690 0.490 0.380 0.260 2.69 RoboFlamingo [32] Static RGB + Gripper RGB 0.824 0.619 0.466 0.331 0.235 2.47 MT-R3M [54] Static RGB + Gripper RGB + Proprio 0.529 0.234 0.105 0.043 0.018 0.93 GR-1 [54] Static RGB + Gripper RGB + Proprio 0.854 0.712 0.596 0.497 0.401 3.06

Moto Static RGB 0.897 0.729 0.601 0.484 0.386 3.10 Moto w/o Motion Token Static RGB 0.779 0.555 0.380 0.256 0.167 2.14

70

Moto

Moto w/o Motion Token

60

50

SuccessRate(%)

40

30

20

10

0

Pick-Place Close Disassembly Avg. Visual Distractor

Novel Object

Figure 8. Evaluation results in the real-world environment.

also generalizes well in the unseen CALVIN environment, outperforming baseline models that use various pre-training strategies (see supplementary material for detailed descriptions about baselines). Notably, Moto-GPT relies solely on RGB images from a static camera, achieving competitive results compared to GR-1, which is pre-trained to predict future pixel values and uses gripper RGB and proprioceptive states as additional inputs. Our findings suggest that focusing on motion dynamics rather than frame-level details is a more effective approach for learning from videos.

Real-world Experiments. We additionally test Moto-GPT on three real-world tasks. As shown in Fig. 8, Moto-GPT consistently outperforms Moto w/o Motion Token on these tasks, improving the average success rate from 23.33% to

85

| |55.4%<br><br>60.4%<br><br>78.3%<br><br>+5.0%<br><br>+17.9%| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

80

75

SuccessRate(%)

70

65

60

55

50

Moto w/o Motion Token Moto (OXE) Moto (OXE+SSV2)

Figure 9. With additional human video (SSV2) pre-training, Moto (OXE+SSV2) significantly outperforms both Moto w/o Motion Token and Moto (OXE) on the Move Near task in SIMPLER.

60%. For generalizability evaluation, we add visual distractors (Visual Distractor) or change the appearance of manipulated objects (Novel Object). Moto-GPT still enhances the average performance by 20% and 30% under varying conditions. This further demonstrates the robustness of MotoGPT in real-world deployment.

Learning from Human Videos. We expand our approach to Internet-scale human videos, demonstrating their potential for large-scale robotics pre-training. Using SSV2 [20] capturing simple human hand movements, we filter out less motion-relevant videos, retaining 105k human videos for a preliminary study. These are combined with 109k OXE

###### left, up up down (small) down (large) right

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

###### [99,57,43,41,86,15,90,76] [21,76,80,122,64,0,103,23] [112,28,111,86,22,34,41,32] [104,118,37,29,71,50,55,42] [6,95,104,96,66,46,79,6]

Figure 10. Visualization of motion transfer from human to robot videos via latent motion tokens.

Moto Moto w/o Motion Token

| |
|---|
| |
| |
| |
| |
| |

0.8

SuccessRate

0.6

0.4

0.2

0.0

1% 5% 10% 50% 100%

Proportion of Fine-tuning Data

Figure 11. Task success rate of models fine-tuned with different proportions of data on CALVIN (ABC−→D).

robot videos to train the Latent Motion Tokenizer and pretrain Moto-GPT. Fig. 9 shows that adding human videos further enhances Moto-GPT’s performance compared with Moto (OXE) which is only pre-trained on OXE videos. Additionally, latent motion tokens can act as a unified “language”s to translate human motions into semantically aligned robot actions (Fig. 10). This enables not only pretraining with human videos but also in-context robot learning guided by online human demonstrations. Future work will improve model architectures and incorporate more diverse human videos to tackle complex manipulation tasks.

Data Efficiency. Moto-GPT is pre-trained exclusively on videos, bypassing the need for supervised data with action labels. This enables pre-training on large, easily accessible video datasets, followed by fine-tuning with smaller actionlabeled trajectories for policy adaptation. Fig. 11 shows that Moto-GPT fine-tuned with varying amounts of labeled data consistently outperforms its variant trained from scratch without latent motion tokens, especially with limited data. For instance, Moto-GPT achieves a 52.5% success rate with just 1% of labeled data, compared to 0% for the variant. This demonstrates Moto-GPT’s efficiency in action adaptation and its potential to improve robot manipulation tasks through large-scale video pre-training.

Ablations on Policy Fine-tuning Methods. In Fig. 12, we assess Moto’s co-fine-tuning strategy. Moto-IML and

Moto Moto-IML Moto-DM Moto w/o Motion Token

| |
|---|

0.8

| |
|---|

| |
|---|

SuccessRate

0.6

0.4

0.2

0.0

1 2 3 4 5

Tasks Completed in a Row

Figure 12. Ablations of Moto-GPT on CALVIN (ABC−→D).

Moto-DM use the same pre-training as Moto-GPT but differ in fine-tuning: Moto-IML excludes the loss term for latent motion token prediction, while Moto-DM omits latent motion tokens from the input entirely. Compared to Moto w/o Motion Tokens, which is trained from scratch, both Moto-IML and Moto-DM show improved performance due to pre-training motion priors, yet they still lag behind MotoGPT. This highlights the importance of retaining latent motion tokens in the sequence, allowing action query tokens to transfer knowledge through direct attention. Furthermore, co-fine-tuning for latent motion token prediction helps preserve the learned motion priors in Moto-GPT.

###### 6. Conclusion and Discussion

We present Moto, a method that utilizes latent motion tokens as a “language” interface to bridge generative pretraining on video data with precise robot control. By learning motion-related priors from videos without the need for hardware-specific action labels, Moto effectively translates learned motions into actual robot actions. We demonstrate that Moto can not only be used for pre-training on robot videos, but also be extended to Internet-scale human videos and enable cross-embodiment knowledge transfer. Future research should aim to scale up the pre-training dataset and optimize fine-tuning to enhance performance across a broader range of robotic applications.

###### Acknowledgement

The research work described in this paper was conducted in the JC STEM Lab of Autonomous Intelligent Systems funded by The Hong Kong Jockey Club Charities Trust.

###### References

- [1] Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan L Yuille, Trevor Darrell, Jitendra Malik, and Alexei A Efros. Sequential modeling enables scalable learning for large vision models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22861–22872, 2024. 2
- [2] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom,

Karol Hausman, Brian Ichter, et al. π0: A vision-languageaction flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 2

- [3] Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya, Homer Rich Walke, Chelsea Finn, Aviral Kumar, and Sergey Levine. Zero-shot robotic manipulation with pre-trained image-editing diffusion models. In The Twelfth International Conference on Learning Representations, 2024. 3, 7, 2
- [4] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 2, 5, 7, 1
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, pages 1877–1901. Curran Associates, Inc.,

2020. 2

- [6] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024. 3
- [7] Qingwen Bu, Jia Zeng, Li Chen, Yanchao Yang, Guyue Zhou, Junchi Yan, Ping Luo, Heming Cui, Yi Ma, and Hongyang Li. Closed-loop visuomotor control with generative expectation for robotic manipulation. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024. 3
- [8] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024. 2, 3

- [9] Lawrence Yunliang Chen, Simeon Adebola, and Ken Goldberg. Berkeley UR5 demonstration dataset. https://sites.google.com/view/berkeley-ur5/home. 5, 1
- [10] Xiaoyu Chen, Junliang Guo, Tianyu He, Chuheng Zhang, Pushi Zhang, Derek Cathera Yang, Li Zhao, and Jiang Bian. Igor: Image-goal representations are the atomic control units for foundation models in embodied ai. arXiv preprint arXiv:2411.00785, 2024. 3
- [11] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, page 02783649241273668, 2023. 3
- [12] Zichen Cui, Hengkai Pan, Aadhithya Iyer, Siddhant Haldar, and Lerrel Pinto. Dynamo: In-domain dynamics pretraining for visuo-motor control. Advances in Neural Information Processing Systems, 37:33933–33961, 2025. 3
- [13] Shivin Dass, Jullian Yapeter, Jesse Zhang, Jiahui Zhang, Karl Pertsch, Stefanos Nikolaidis, and Joseph J. Lim. Clvr jaco play dataset, 2023. 5, 1
- [14] Ria Doshi, Homer Rich Walke, Oier Mees, Sudeep Dasari, and Sergey Levine. Scaling cross-embodied learning: One policy for manipulation, navigation, locomotion and aviation. In 8th Annual Conference on Robot Learning, 2024. 2
- [15] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. In International Conference on Machine Learning, pages 8469–8488. PMLR,

2023. 2

- [16] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. Advances in Neural Information Processing Systems, 36, 2024. 3
- [17] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 2

- [18] Alejandro Escontrela, Ademi Adeniji, Wilson Yan, Ajay Jain, Xue Bin Peng, Ken Goldberg, Youngwoon Lee, Danijar Hafner, and Pieter Abbeel. Video prediction models as rewards for reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 2, 3
- [19] Divyansh Garg, Skanda Vaidyanath, Kuno Kim, Jiaming Song, and Stefano Ermon. Lisa: Learning interpretable skill abstractions from language. Advances in Neural Information Processing Systems, 35:21711–21724, 2022. 3
- [20] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The” something something” video database for learning and evaluating visual common sense.

- In Proceedings of the IEEE international conference on computer vision, pages 5842–5850, 2017. 7
- [21] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations, 2020. 3
- [22] Siddhant Haldar, Zhuoran Peng, and Lerrel Pinto. BAKU: An efficient transformer for multi-task policy learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 3
- [23] Haoran He, Chenjia Bai, Ling Pan, Weinan Zhang, Bin Zhao, and Xuelong Li. Learning an actionable discrete diffusion policy via large-scale actionless video pre-training. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 3
- [24] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 3, 4
- [25] Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar, Yuke Zhu, and Linxi Fan. Vima: robot manipulation with multimodal prompts. In Proceedings of the 40th International Conference on Machine Learning, pages 14975– 15022, 2023. 2
- [26] Tsung-Wei Ke, Nikolaos Gkanatsios, and Katerina Fragkiadaki. 3d diffuser actor: Policy diffusion with 3d scene representations. In ICRA 2024 Workshop—Back to the Future: Robot Learning Going Probabilistic, 2024. 3
- [27] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 2, 7
- [28] Po-Chen Ko, Jiayuan Mao, Yilun Du, Shao-Hua Sun, and Joshua B. Tenenbaum. Learning to act from actionless videos through dense correspondences. In The Twelfth International Conference on Learning Representations, 2024. 3
- [29] Peiyan Li, Hongtao Wu, Yan Huang, Chilam Cheang, Liang Wang, and Tao Kong. Gr-mg: Leveraging partially annotated data via multi-modal goal conditioned policy. arXiv preprint arXiv:2408.14368, 2024. 3
- [30] Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, et al. Cogact: A foundational visionlanguage-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650,

2024. 2

- [31] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, et al. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024. 5, 1
- [32] Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang,

- Huaping Liu, Hang Li, and Tao Kong. Vision-language foundation models as effective robot imitators. In The Twelfth International Conference on Learning Representations, 2024. 2, 7
- [33] Junbang Liang, Ruoshi Liu, Ege Ozguroglu, Sruthi Sudhakar, Achal Dave, Pavel Tokmakov, Shuran Song, and Carl Vondrick. Dreamitate: Real-world visuomotor policy learning via video generation. arXiv preprint arXiv:2406.16862,

2024. 3

- [34] Zhixuan Liang, Yao Mu, Hengbo Ma, Masayoshi Tomizuka, Mingyu Ding, and Ping Luo. Skilldiffuser: Interpretable hierarchical planning via skill abstractions in diffusion-based task execution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16467– 16476, 2024. 3
- [35] Fanfan Liu, Feng Yan, Liming Zheng, Chengjian Feng, Yiyang Huang, and Lin Ma. Robouniview: Visual-language model with unified view representation for robotic manipulation. arXiv preprint arXiv:2406.18977, 2024. 3
- [36] Jianlan Luo, Charles Xu, Xinyang Geng, Gilbert Feng, Kuan Fang, Liam Tan, Stefan Schaal, and Sergey Levine. Multistage cable routing through hierarchical imitation learning. arXiv pre-print, 2023. 5, 1
- [37] Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy Zhang. VIP: Towards universal visual reward and representation via value-implicit pre-training. In The Eleventh International Conference on Learning Representations, 2023. 3
- [38] Ajay Mandlekar, Jonathan Booher, Max Spero, Albert Tung, Anchit Gupta, Yuke Zhu, Animesh Garg, Silvio Savarese, and Li Fei-Fei. Scaling robot supervision to hundreds of hours with roboturk: Robotic manipulation dataset through human reasoning and dexterity. In 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1048–1055. IEEE, 2019. 5, 1
- [39] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for languageconditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3): 7327–7334, 2022. 5, 1
- [40] Oier Mees, Jessica Borja-Diaz, and Wolfram Burgard. Grounding language with visual affordances over unstructured data. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), London, UK,

2023. 5, 1

- [41] Oier Mees, Dibya Ghosh, Karl Pertsch, Kevin Black, Homer Rich Walke, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An opensource generalist robot policy. In First Workshop on VisionLanguage Models for Navigation and Manipulation at ICRA 2024, 2024. 2, 7
- [42] Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3m: A universal visual representation for robot manipulation. In Conference on Robot Learning, pages 892–909. PMLR, 2023. 3, 2
- [43] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini

- Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022. 2
- [44] Jyothish Pari, Nur Muhammad Shafiullah, Sridhar Pandian Arunachalam, and Lerrel Pinto. The surprising effectiveness of representation learning for visual imitation, 2021. 5, 1
- [45] Alec Radford. Improving language understanding by generative pre-training. 2018. 2
- [46] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 4
- [47] Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio G´omez Colmenarejo, Alexander Novikov, Gabriel Barth-maron, Mai Gim´enez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, Tom Eccles, Jake Bruce, Ali Razavi, Ashley Edwards, Nicolas Heess, Yutian Chen, Raia Hadsell, Oriol Vinyals, Mahyar Bordbar, and Nando de Freitas. A generalist agent. Transactions on Machine Learning Research,

2022. Featured Certification, Outstanding Certification. 2

- [48] Erick Rosete-Beas, Oier Mees, Gabriel Kalweit, Joschka Boedecker, and Wolfram Burgard. Latent plans for task agnostic offline reinforcement learning. 2022. 5, 1
- [49] Dominik Schmidt and Minqi Jiang. Learning to act without actions. arXiv preprint arXiv:2312.10812, 2023. 3
- [50] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2
- [51] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 2, 4
- [52] Quan Vuong, Sergey Levine, Homer Rich Walke, Karl Pertsch, Anikait Singh, Ria Doshi, Charles Xu, Jianlan Luo, Liam Tan, Dhruv Shah, Chelsea Finn, Max Du, Moo Jin Kim, Alexander Khazatsky, Jonathan Heewon Yang, Tony Z. Zhao, Ken Goldberg, Ryan Hoque, Lawrence Yunliang Chen, Simeon Adebola, Gaurav S. Sukhatme, Gautam Salhotra, Shivin Dass, Lerrel Pinto, Zichen Jeff Cui, Siddhant Haldar, Anant Rai, Nur Muhammad Mahi Shafiullah, Yuke Zhu, Yifeng Zhu, Soroush Nasiriany, Shuran Song, Cheng Chi, Chuer Pan, Wolfram Burgard, Oier Mees, Chenguang Huang, Deepak Pathak, Shikhar Bahl, Russell Mendonca, Gaoyue Zhou, Mohan Kumar Srirama, Sudeep Dasari, Cewu Lu, Hao-Shu Fang, Hongjie Fang, Henrik I Christensen, Masayoshi Tomizuka, Wei Zhan, Mingyu Ding, Chenfeng Xu, Xinghao Zhu, Ran Tian, Youngwoon Lee, Dorsa Sadigh, Yuchen Cui, Suneel Belkhale, Priya Sundaresan, Trevor Darrell, Jitendra Malik, Ilija Radosavovic, Jeannette Bohg, Krishnan Srinivasan, Xiaolong Wang, Nicklas Hansen, YuehHua Wu, Ge Yan, Hao Su, Jiayuan Gu, Xuanlin Li, Niko Suenderhauf, Krishan Rana, Ben Burgess-Limerick, Federico Ceola, Kento Kawaharazuka, Naoaki Kanazawa, Tatsuya Matsushima, Yutaka Matsuo, Yusuke Iwasawa, Hiroki Furuta, Jihoon Oh, Tatsuya Harada, Takayuki Osa, Yujin

- Tang, Oliver Kroemer, Mohit Sharma, Kevin Lee Zhang, Beomjoon Kim, Yoonyoung Cho, Junhyek Han, Jaehyung Kim, Joseph J Lim, Edward Johns, Norman Di Palo, Freek Stulp, Antonin Raffin, Samuel Bustamante, Jo˜ao Silv´erio, Abhishek Padalkar, Jan Peters, Bernhard Sch¨olkopf, Dieter B¨uchler, Jan Schneider, Simon Guist, Jiajun Wu, Stephen Tian, Haochen Shi, Yunzhu Li, Yixuan Wang, Mingtong Zhang, Heni Ben Amor, Yifan Zhou, Keyvan Majd, Lionel Ott, Giulio Schiavi, Roberto Mart´ın-Mart´ın, Rutav Shah, Yonatan Bisk, Jeffrey T Bingham, Tianhe Yu, Vidhi Jain, Ted Xiao, Karol Hausman, Christine Chan, Alexander Herzog, Zhuo Xu, Sean Kirmani, Vincent Vanhoucke, Ryan Julian, Lisa Lee, Tianli Ding, Yevgen Chebotar, Jie Tan, Jacky Liang, Igor Mordatch, Kanishka Rao, Yao Lu, Keerthana Gopalakrishnan, Stefan Welker, Nikhil J Joshi, Coline Manon Devin, Alex Irpan, Sherry Moore, Ayzaan Wahid, Jialin Wu, Xi Chen, Paul Wohlhart, Alex Bewley, Wenxuan Zhou, Isabel Leal, Dmitry Kalashnikov, Pannag R Sanketi, Chuyuan Fu, Ying Xu, Sichun Xu, brian ichter, Jasmine Hsu, Peng Xu, Anthony Brohan, Pierre Sermanet, Nicolas Heess, Michael Ahn, Rafael Rafailov, Acorn Pooley, Kendra Byrne, Todor Davchev, Kenneth Oslund, Stefan Schaal, Ajinkya Jain, Keegan Go, Fei Xia, Jonathan Tompson, Travis Armstrong, and Danny Driess. Open xembodiment: Robotic learning datasets and RT-x models. In Towards Generalist Robots: Learning Paradigms for Scalable Skill Acquisition @ CoRL2023, 2023. 2, 5, 7, 1
- [53] Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao, Philippe HansenEstruch, Quan Vuong, Andre He, Vivek Myers, Kuan Fang, Chelsea Finn, and Sergey Levine. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning (CoRL), 2023. 5, 1
- [54] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. In The Twelfth International Conference on Learning Representations, 2024. 2, 3, 7
- [55] Jialong Wu, Shaofeng Yin, Ningya Feng, Xu He, Dong Li, Jianye HAO, and Mingsheng Long. ivideoGPT: Interactive videoGPTs are scalable world models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 3
- [56] Sherry Yang, Yilun Du, Seyed Kamyar Seyed Ghasemipour, Jonathan Tompson, Leslie Pack Kaelbling, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. In The Twelfth International Conference on Learning Representations, 2024. 3
- [57] Sherry Yang, Jacob Walker, Jack Parker-Holder, Yilun Du, Jake Bruce, Andre Barreto, Pieter Abbeel, and Dale Schuurmans. Video as the new language for real-world decision making. arXiv preprint arXiv:2402.17139, 2024. 2
- [58] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al. Latent action pretraining from videos. arXiv preprint arXiv:2410.11758, 2024. 3
- [59] Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3d-vla:

- A 3d vision-language-action generative world model. arXiv preprint arXiv:2403.09631, 2024. 3
- [60] Gaoyue Zhou, Victoria Dean, Mohan Kumar Srirama, Aravind Rajeswaran, Jyothish Pari, Kyle Hatch, Aryan Jain, Tianhe Yu, Pieter Abbeel, Lerrel Pinto, Chelsea Finn, and Abhinav Gupta. Train offline, test online: A real robot learning benchmark. In 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023. 5, 1
- [61] Yifeng Zhu, Abhishek Joshi, Peter Stone, and Yuke Zhu. Viola: Imitation learning for vision-based manipulation with object proposal priors. 6th Annual Conference on Robot Learning (CoRL), 2022. 5, 1
- [62] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023. 2, 7

## Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos

### Supplementary Material

###### 7. Details on Experiment Setup

###### 7.1. Benchmarks

SIMPLER. On the SIMPLER benchmark, we focus on three tasks concerning the Google Everyday Robot embodiment: Pick Coke Can, Move Near, and Open/Close Drawer, as illustrated in Fig. 13. The “Pick Coke Can” task involves grasping and lifting the empty coke can in three different orientations: horizontal laying, vertical laying, and standing. The “Move Near” task places 3 (out of 8) objects in a triangle pattern on the tabletop and instructs the robot to move a designated source object near another object as the target. The “Open/Close Drawer” task requires the robot to open or close a specific drawer from the top, middle, or down position of a cabinet.

###### Pick Coke Can Move Near Open / Close Drawer

[Figure 204]

[Figure 205]

[Figure 206]

Figure 13. Illustration of the evaluation tasks in SIMPLER [31].

CALVIN (ABC−→D). The CALVIN benchmark uses a Franka Emika Panda robot embodiment. It evaluates longhorizon task completion with unconstrained language instructions. In each trial, the robot should accomplish 5 (out of 34) tasks in a row. There are four different environments (A, B, C, D), each containing a desk with a sliding door, a drawer, differently colored blocks, a button that toggles an LED, and a switch controlling a lightbulb. As shown in Fig. 14, the environments differ in the textures of the desk, and the positions of all static elements including the sliding door, the drawer, the LED button, and the lightbulb switch. We conduct experiments under the most challenging ABC−→D setting, i.e., training on data from environments A, B, and C while testing in D.

Real-world Robot Experiments. We design three tasks for real-world evaluation: Pick-Place Banana (picking up a banana from the tabletop and placing it in a pan), Close Laptop (pushing the laptop’s lid until it is completely closed), and Disassembly (removing the stick

Training

[Figure 207]

[Figure 208]

move the switch to turn on the light bulb push the button

Env A Env B

Test

[Figure 209]

[Figure 210]

place the red block in the slider push the blue block to the left

Env C Env D

Figure 14. Illustration of the four different environments in CALVIN, adapted from the original figure in Mees et al. [39].

that is assembled in the slot). All tasks are executed with a FANUC LR Mate 200iD robot arm, as shown in Fig. 3. Each task is tested for 10 times, with the initial positions of objects randomized. For generalization evaluation, we consider two scenarios: (i) Novel Object: we change the color, texture, and shape of the manipulated object; (ii) Visual Distractor: we add irrelevant objects as distractors.

###### 7.2. Datasets

Training Latent Motion Tokenizer & Pre-training Moto-GPT. On the SIMPLER benchmark, we use a subset of Open-X-Embodiment (OXE) [52] to train the Latent Motion Tokenizer and pre-train Moto-GPT, including 109k real-world trajectory videos from RT-1 Robot Action [4], Bridge [53], Task Agnostic Robot Play [40, 48], Jaco Play [13], Cable Routing [36], RoboTurk [38], NYU VINN [44], Austin VIOLA [61], Berkeley Autolab UR5 [9], and TOTO [60] datasets across various embodiments. On CALVIN, we use all the play videos from environments A, B, and C to train the Latent Motion Tokenizer. 35% data (including 18k trajectory videos) with language annotations is used for autoregressive pre-training. The real-world experiments also use OXE data for pre-training.

Fine-tuning Moto-GPT. On the SIMPLER benchmark, we use the 73k action-labeled real-world expert trajectories from RT-1 Robot Action [4] to fine-tune the policy model. On the CALVIN benchmark, we use the 18k demonstration trajectories with language annotations and action labels from environments A, B, and C for fine-tuning. For realworld experiments, we collect 90 demonstration trajectories (30 for each task) with teleoperation for fine-tuning.

Note that all the pre-training and fine-tuning data for SIMPLER come from the real world instead of simulation environments. This setting aims to study the model’s transfer ability between real and simulation scenarios. On the other hand, the ABC-only setting for CALVIN training data aims to evaluate the model’s zero-shot generalization capability to the unseen environment D.

###### 7.3. Compared Models

SIMPLER. On the SIMPLER benchmark, we mainly compare Moto-GPT with five representative models pretrained with Open-X-Embodiment datasets:

- • RT-1-X [4] uses a transformer backbone to output tokenized actions with a FiLM EfficientNet to fuse language and 6 history images into token inputs.
- • RT-2-X [62] adapts the pre-trained large vision-language model (VLM), PaLI-X (55B), into a robot policy by casting tokenized actions into text tokens.
- • Octo-Base [41] employ a transformer architecture to process language and image tokens, with a diffusion-based action head to produce actions.
- • OpenVLA [27] builds on a pre-trained Prismatic-7B VLM backbone for robot action prediction.
- • OpenVLA (fine-tuned) further fine-tunes OpenVLA on the RT-1 Robot Action dataset [4], despite its actionlabeled pre-training data already contains this dataset.

CALVIN. On the CALVIN benchmark, we select the following baseline models that leverage pre-training strategies to improve robot manipulation performance:

- • SuSIE [3] pre-trains an image editing model to generate the goal image, which is fed into a low-level policy for action prediction.
- • RoboFlamingo [32] is a robot policy model adapted from OpenFlamingo, a large VLM pre-trained on extensive vision-language corpus.
- • GR-1 [54] pre-trains a GPT-style transformer to directly predict the pixel values of a single-step future observation for each input observation.
- • MT-R3M [54] is a variation of GR-1, which leverages the pre-trained robot visual encoder R3M [42] to encode observation images.

Table 4. Implementation details of the Latent Motion Tokenizer.

Component Parameter Value

num queries 8 num layers 4 hidden size 768 num heads 12

M-Former

patch size 16 num layers 12 hidden size 768 num heads 12

ViT Decoder

num codes 128 latent dim 32

VQ Codebook

Ablations of Moto-GPT. We study the following variations of Moto-GPT as optional baselines:

- • Moto w/o Motion Token shares the same backbone with Moto-GPT but is trained from scratch on action-labeled robot data without latent motion tokens.
- • Moto-IML undergoes the same pre-training stage as Moto-GPT. It keeps latent motion tokens in the input sequence but ignores the next-motion-token-prediction loss during the fine-tuning stage.
- • Moto-DM is pre-trained in the same way as Moto-GPT but completely discards latent motion tokens in the input sequence during fine-tuning.

###### 8. Training Details

###### 8.1. Latent Motion Tokenizer

The implementation details for the trainable modules of the Latent Motion Tokenizer are summarized in Table 4. We use the hyperparameters listed in Table 5 to train this model on four 40G GPUs. To facilitate the learning of latent motion tokens, we downsample the original videos in the training dataset, ensuring that the visual motion between frames is sufficiently distinct. Specifically, for videos from the OXE data, we sample one frame every three frames (i.e., ∆t = 3) for videos from the RT-1 Robot-Action subset. The sampling rates for videos from other OXE subsets are adjusted based on the ratio of their fps to that of RT-1 RobotAction videos. For human videos, ∆t is empirically set to 6. We train the Latent Motion Tokenizer for 350k steps. For videos from the CALVIN dataset, we adopt a sampling rate of one frame every five frames (∆t = 5) and train the model for 150k steps. For real-world robot experiments, we fine-tune the Latent Motion Tokenizer pre-trained on OXE videos for another 500 steps on the newly collected realworld trajectory videos.

Table 5. Training hyperparameters for Latent Motion Tokenizer.

Parameter Value batch size 256 optimizer AdamW lr max 1e-4 lr schedule cosine decay weight decay 1e-4 warmup steps 1000

- Table 6. Implementation details of Moto-GPT.

Component Parameter Value

GPT backbone

num layers 12 hidden size 768 num heads 12

Action Head

num layers 2 hidden size 384

- Table 7. Training hyperparameters for Moto-GPT.

Parameter Value batch size 512 optimizer AdamW lr max 1e-4 lr schedule cosine decay weight decay 1e-4 warmup epochs 1

###### 8.2. Moto-GPT

We present the implementation details of Moto-GPT in Table 6, where the Action Head is included only during the fine-tuning phase. Moto-GPT handles a maximum video length of three frames, and the video downsampling rate applied during both the pre-training and fine-tuning stages is consistent with the rate used for training the Latent Motion Tokenizer. When fine-tuning Moto-GPT across different benchmarks, the number of action query tokens inserted after the latent motion tokens at each time step varies. Specifically, for the SIMPLER benchmark, we insert three action query tokens, whereas for the CALVIN benchmark, we insert five. For pre-training, Moto-GPT is trained for 10 epochs using eight 40G GPUs, with the relevant hyperparameters outlined in Table 7. The hyperparameters for fine-tuning remain consistent with those used during pretraining, with the exception of the number of epochs. During fine-tuning, Moto-GPT is trained for three epochs on the RT1-Robot-Action dataset and 18 epochs on the CALVIN dataset, utilizing four 40G GPUs. For real-world experiments, we start with the same pre-trained checkpoint of Moto-GPT as adopted for the SIMPLER benchmark. We

further pre-train it with a combination of OXE videos and the 90 newly collected trajectory videos for 5 epochs. Then we fine-tune it with real robot action labels for 20 epochs.

###### 9. Details of Experiments

Next Frame

Next Frame (ground-truth)

Initial Frame

(reconstructed)

|[Figure 211]|
|---|

[Figure 212]

[Figure 213]

|[Figure 214]|
|---|

[Figure 215]

[Figure 216]

Figure 15. Qualitative examples of reconstruction results, where discrete motion tokens obtained from the Latent Motion Tokenizer based on the initial and next frame, are fed into the decoder along with the initial frame to reconstruct the target frame.

Table 8. Top-K motion token prediction accuracy of Moto-GPT in predicting ground-truth latent motion tokens from a 128-size codebook on the validation splits of the pre-training datasets.

Dataset Top-5 Top-10 Top-20

Oepn-X-Embodiment 0.521 0.698 0.853 Calvin (ABC−→D) 0.298 0.518 0.768

###### 10. Limitations & Future Work

This paper introduces Moto, a novel method that uses latent motion tokens as a “language” interface to bridge generative pre-training on video data with precise robot control. Moto opens several exciting avenues for future work.

Firstly, Moto demonstrates the feasibility of learning a unified language to interpret diverse visual dynamics from videos, eliminating the need for hardware-specific action labels. The latent motion trajectories tokenized from videos provide a rich resource for models to learn motion priors closely related to low-level actions. While we currently mainly use robot videos to train the Latent Motion Tokenizer, the learned latent motion tokens demonstrate the potential to produce consistent visual motions across varied contexts and embodiments. Notably, our preliminary

Initial Frame

Rotate the pink block 90 degrees to the right

|[Figure 217]|
|---|

|[Figure 218]|
|---|

Initial Frame Stack blocks on top of each other.

|[Figure 219]|
|---|

|[Figure 220]|
|---|

Figure 16. Predicted video trajectories by the pre-trained Moto-GPT for CALVIN tasks reflecting delicate robot actions.

experiments with SSV2 videos show promising results in human-to-robot motion transfer via latent motion tokens. We believe a similar approach could be applied to model more complex human motions, enabling robots to learn a wealth of world knowledge from Internet-scale videos.

Besides, the Moto-GPT pre-trained on videos tokenized into latent motion token sequences and fine-tuned on action-labeled trajectories, effectively transfers motion priors learned from (even human) videos to actual robot action prediction. This is particularly beneficial in low-resource scenarios. Future work could scale up pre-training video data and optimize fine-tuning to improve model performance on downstream robot tasks further.

Moreover, while Moto is primarily utilized to enhance imitation learning for robot manipulation tasks, it shows potential as a reward model for measuring trajectory rationality and as a vivid environment simulator. Future research could explore Moto’s use in improving the robustness of reinforcement learning agents and extending its application to a wider range of robotic tasks, such as navigation and locomotion, to develop a more versatile robot action policy.

