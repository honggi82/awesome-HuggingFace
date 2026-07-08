# arXiv:2501.09781v2[cs.CV]5Mar2025

[Figure 1]

[Figure 2]

## VideoWorld: Exploring Knowledge Learning from Unlabeled Videos

Zhongwei Ren1,2, Yunchao Wei1†, Xun Guo2, Yao Zhao1, Bingyi Kang2, Jiashi Feng2, Xiaojie Jin2†‡

1Beijing Jiaotong University 2ByteDance Seed https://VideoWorld.github.io/

#### Abstract

This work explores whether a deep generative model can learn complex knowledge solely from visual input, in contrast to the prevalent focus on text-based models like large language models (LLMs). We develop VideoWorld, an auto-regressive video generation model trained on unlabeled video data, and test its knowledge acquisition abilities in video-based Go and robotic control tasks. Our experiments reveal two key findings: (1) video-only training provides sufficient information for learning knowledge, including rules, reasoning and planning capabilities, and (2) the representation of visual change is crucial for knowledge acquisition. To improve both the efficiency and efficacy of this process, we introduce the Latent Dynamics Model (LDM) as a key component of VideoWorld. Remarkably, VideoWorld reaches a 5-dan professional level in the Video-GoBench with just a 300-million-parameter model, without relying on search algorithms or reward mechanisms typical in reinforcement learning. In robotic tasks, VideoWorld effectively learns diverse control operations and generalizes across environments, approaching the performance of oracle models in CALVIN and RLBench. This study opens new avenues for knowledge acquisition from visual data, with all code, data, and models open-sourced for further research.

#### 1. Introduction

The next token prediction training paradigm has endowed large language models (LLMs) [2, 3, 7, 13, 19, 20, 29, 30, 56, 60, 67] with remarkable world knowledge and intelligence, enabling them to help address complex tasks that require reasoning [22, 23, 32, 45–47, 63, 74], planning ahead [26, 51, 53, 62], and decision-making [10, 11, 17, 36]. However, language alone cannot fully capture all forms of knowledge or encompass the vast information present in the real world. In nature, biological organisms acquire knowl-

Work done when Zhongwei and Xun interned at ByteDance Seed. †Correspondence to Xiaojie Jin <jinxiaojie@bytedance.com> and Yunchao Wei <yunchao.wei@bjtu.edu.cn>. ‡ Project lead.

[Figure 3]

Figure 1. VideoWorld explores learning knowledge from unlabeled videos, ranging from task-specific rules to high-level reasoning and planning capabilities. Compared to other learning methods: reinforcement learning (RL), supervised learning (SL) and text-based learning, it offers three advantages: 1) better generalization with unified visual representation for various tasks and interfaces, 2) lower manual annotation burden, and 3) learning richer real-world information than text description.

edge primarily through visual information, rather than relying solely on language. For instance, gorillas and other primates learn vital skills like foraging and social interactions mainly through visual observation, mimicking adult behaviors without relying on language [9, 25, 64].

Most existing research has focused on learning knowledge from texts or labels [1, 55], with relatively little attention to learning from pure visual signals. Some studies, such as [16], have explored using video data to train models for robot manipulation, but they still rely heavily on language instructions. Moreover, these tasks are often limited to single commands, without requiring complex reasoning or planning. This raises an important question: can an AI model learn knowledge* solely from visual input, akin to how a gorilla learns from its environment?

In this work, we take an initial step toward exploring

*Following prior works in AI and knowledge representation [5, 49] which view ‘knowledge’ as extending beyond factual information, we use ‘knowledge’ to broadly refer to a model’s learned rules, reasoning, and planning abilities necessary for task completion. For better readability and clarity, these terms are used interchangeably in specific contexts.

1

knowledge learning from raw video data by leveraging the next token prediction paradigm. To this end, we construct two experiment environments: Go and robotic manipulation (as shown in Fig. 1) to collect purely visual training data. The Go game serves as an ideal testbed for assessing a model’s ability to learn rules, reasoning, and planning: it not only has well-defined rules but also demands complex reasoning about the current state to determine the best move and forward planning to outmaneuver an opponent. Moreover, it disentangles low-level details (e.g., appearance and texture) from higher-level knowledge, making it particularly suitable for our investigation. In parallel, we also assess the model’s capability to understand robotic manipulation rules and planning, trained solely on raw video data, for control tasks in the CALVIN [40] and RLBench [28] benchmarks.

We begin our investigation with a basic video generation model comprising a VQ-VAE [15, 57] and an autoregressive transformer. The raw task execution videos, collected from the environments described above, serve as the sole source of training data and hence the only source of knowledge. We convert video frames into discrete tokens using VQ-VAE and, similar to large language models (LLMs), train an auto-regressive transformer on these tokens under the next-token (or next-frame) prediction paradigm. During testing, the model generates new frames based on previous ones, and task-specific operations—such as moves in Go or robotic actions—are inferred from the newly generated frames (see Sec. 3.2 for details).

Through this approach, we observe two key findings:

- 1. The model can learn basic knowledge from raw videos. This is evidenced by its ability to learn Go rules and policies and fundamental robotic operations.
- 2. The representation of visual change is crucial for knowledge learning. While videos contain sufficient information for task completion, redundant representations of visual changes related to key decisions and actions hinder learning efficiency. A compact representation is essential for enhancing the model’s learning efficiency and knowledge acquisitions. Building on the observations above, we propose the La-

tent Dynamics Model (LDM), which enhances both the efficiency and effectiveness of video learning while providing a mechanism to probe the model’s learned knowledge. The LDM compresses the future visual changes into a set of latent codes to serve as the compact representation of multistep visual context. This allows the model to predict both video frames and latent code during training, improving its ability to capture and reason about diverse visual information, such as object interactions and scene dynamics.

These learned representations significantly enhance the model’s learning efficiency and reasoning capability. As shown in Fig. 2, the video generation model with LDM

[Figure 4]

Figure 2. Comparison of prediction targets. “State”, “Video” and “Video w/ LDM” refer to three different prediction targets: a state sequence (e.g., labeled positions of moves in Go), a raw video sequence, and a video sequence augmented with latent codes representing future visual changes (this approach is adopted by VideoWorld). “Action-Value” denotes the score for each move in the game, with details provided in Sec. 4.2. By combining rich video information with a compact representation of visual changes, VideoWorld enables more effective learning.

achieves superior training efficiency and demonstrates a 5dan† professional level of performance against the RL agent KataGO‡ [65]—an impressive accomplishment, given that this level is challenging even for most human players and our model achieves this with only 300 million parameters and solely through visual observation, without using search or reward learning techniques typical in RL [52, 65].

In the CALVIN and RLBench robotic scenarios, VideoWorld successfully performs various control tasks with promising data scaling behavior. It also demonstrates generalization capabilities across multiple environments, a challenge for RL methods due to environmental distinctions. Notably, it achieves performance close to the oracle models trained with ground-truth action labels. Above results highlight the model’s ability to master complex tasks and acquire knowledge effectively.

Despite the promising results, we acknowledge that learning knowledge from unlabeled videos is still in its early stage. Improved visual representations and large-scale pretraining are likely to further enhance the model’s learning capabilities. In summary, we make following contributions: • We explore for the first time whether video generation

models can learn sophisticated knowledge and observe two key findings: i) merely observing videos suffices to learn complex tasks, and ii) compact representations of visual changes greatly enhance knowledge learning.

†dan denotes the ranking system in Go, with higher levels indicating greater skill. 5-dan can already represent a highly trained human player.

‡KataGO is an open-source implementation of AlphaGo and is arguably the strongest accessible Go agent.

- • We propose VideoWorld, which leverages a latent dynamics model to represent multi-step future visual changes, thereby boosting both the efficiency and effectiveness of knowledge acquisition.
- • We construct Video-GoBench, a large-scale video-based Go dataset for model training and evaluation, facilitating future research on knowledge learning from pure videos.

#### 2. Related Work

##### 2.1. Video Generation

Existing video generation methods typically utilize Variational Autoencoders (VAEs) [31, 58] to compress raw video data, and then perform large-scale generative pre-training in the compressed latent space based on diffusion [24, 54] or auto-regressive [12, 72] paradigms. Thanks to the scaling properties of the Transformer models [44, 59], the performance of video generation models can be continually improved, making significant progress in tasks such as text-tovideo [6, 69, 75] and image-to-video [4, 21] generation.

Recent works have begun to explore broader uses for video generation models. [68] indicates that video generation models can be planners, agents, and environment simulators with the help of in-context learning [61], planning [27], and reinforcement learning [43]. UniPi [16] proposes that representing policies using text-conditioned video generation models enables effective learning of general-purpose decision-making systems.

However, existing works are limited to executing single, simple commands and heavily rely on language instructions. This work pioneers exploring AI models’ ability to learn sophisticated reasoning knowledge solely from vision.

##### 2.2. Learning Knowledge

Learning knowledge in complex scenarios is crucial for intelligent agents, involving tasks like policy learning, planning, and decision-making, which are essential in applications such as game agents and autonomous driving. Existing works have achieved remarkable success in reasoning tasks using language models with techniques like chain-ofthought [63, 70] and reinforcement learning [33, 43]. Additionally, some studies focus on planning and reasoning within well-annotated, pre-defined environments. For example, ChessGPT [18] and Othello-GPT [35] explore reasoning in chess using GPT-like models trained on annotated game data. Similarly, ChessBench [48] introduces a dataset of 10 million annotated chess games to evaluate transformer models on planning tasks.

Existing methods typically rely on explicit state or action labels. In contrast, our work aims to explore knowledge learning in an unsupervised manner, driven purely by visual signals. Specifically, we investigate whether video generation alone can learn complex knowledge required for

long-term, high-complexity tasks, without labeled data.

##### 2.3. Latent Actions

The concept most similar to our latent dynamics model is the latent action model. LAPO [50] learns latent actions unsupervised from videos, using an inverse dynamics model (IDM) to predict latent actions and a forward dynamics model (FDM) to ensure predictive consistency. Similarly, Genie [8] extracts latent actions through a causal latent action model and uses both video tokens and latent actions to predict the next frame. Additionally, LAPA [71] is an unsupervised pretraining method that teaches Vision-LanguageAction (VLA) models to learn discrete latent actions from videos without human-annotated labels.

In contrast, our work goes beyond action learning by showing that latent actions derived from unlabeled video data can support tasks requiring complex reasoning and planning. More importantly, we investigate how multi-step latent actions representing future visual changes enhance knowledge acquisition, especially in long-horizon tasks demanding extended planning and decision-making.

#### 3. VideoWorld 3.1. Basic Video Generation Framework

Problem formulation. We define the video generation process for learning knowledge in the context of a specific task as a tuple G = ⟨X,A,ρ⟩, where X is the observation space, A is the action space, and ρ is a video generator. An offline dataset D = {xn1:T

}Nn=1 consists of N video demonstrations within this environment, with Tn frames in each video sequence. For brevity, we omit the subscript n in the following text. Our goal is to train a generator ρ(xt+1|x1:t) that models the conditional distribution of the next frame xt+1, given the history of observations up to time step t within observation space X.

n

Since the video frame sequences generated by ρ inherently contain the necessary information for task progression, we can directly learn a task-relevant mapping function π that converts generated video frames into actions, Specifically, π(·|x1:t+1) : X → A, where π is a policy that maps the generated frames at time step t+1 to appropriate action in A. This allows us to leverage the structure of generated videos to learn and execute the task without relying on explicit action annotations.

Basic Framework. We focus on using an auto-regressive video generator to instantiate ρ. The basic framework includes a VQ-VAE encoder-decoder, and an auto-regressive transformer. The encoder converts video frames into discrete tokens, which the transformer uses for next-token prediction during training. During inference, the transformer generates discrete tokens for the next frame, which are subsequently converted back to pixel space by the decoder.

[Figure 5]

- Figure 3. Overview of the proposed VideoWorld model architecture. (Left) Overall architecture. (Right) The proposed latent dynamics model (LDM). First, LDM compresses the visual changes from each frame to its subsequent H frames into a set of latent codes. Then, an auto-regressive transformer seamlessly integrates the output of LDM with the next token prediction paradigm.

In our implementation, the VQ-VAE is instantiated using a custom MAGVIT-v2 [73], equipped with an FSQ quantizer [41], while the transformer is implemented using the Llama architecture [56]. This setup allows us to generate high-quality frames, enabling effective modeling of visual sequences for diverse knowledge learning tasks.

##### 3.2. Learning with Latent Dynamic Model

Compared to a state sequence (e.g., move positions in Go), video provides essential visual information for understanding tasks—for instance, local patterns among stones in Go or environments in robotics. In preliminary experiments, we find that the basic framework can already learn fundamental knowledge solely from videos (see Sec. 5.3). However, as shown in Fig. 2, its learning efficiency still lags behind that of models trained on states. We attribute this to the inefficient representation of visual changes tied to critical decisions and actions. For example, while moves in Go can be encoded by just a few positional tokens in a state sequence, raw video requires significantly more tokens after passing through a vision encoder. This discrepancy adversely affects both learning efficiency and performance. Therefore, we develop VideoWorld, which integrates rich visual information with a compact representation of visual

change for more effective video learning.

Latent Dynamics Model. Video encoding typically requires hundreds or thousands of VQ tokens to capture the full range of visual information, leading to sparse embedding of knowledge across these tokens. To enhance efficiency, we introduce a latent dynamics model that uses query embeddings to represent visual changes across multiple frames. For example, multi-step board changes in Go or continuous actions in robotics exhibit strong temporal correlations. By compressing these multi-step changes into compact embeddings, we not only increase the compactness of policy information but also encode guidance for forward planning.

The model employs a MAGVITv2-style [73] causal encoder-decoder, while intentionally omitting temporal downsampling to preserve detail in each frame. For a video clip x1:T, we sample each frame xt along with the subsequent H frames, denoted as xt+1:t+H. If fewer than H steps remain, we apply replication padding. In Fig. 3, the encoder first extracts visual feature maps ft:t+H in a causal manner. Importantly, these features are not quantized, allowing them to retain detailed temporal information.

Next, we define a set of attention blocks and corresponding learnable embeddings {qh}Hh=1. Each query qh,

[Figure 6]

- Figure 4. UMAP projection [34] of the learned latent code on the Go (Left) and CALVIN (right) training set. Each point represents the continuous (pre-quantization) latent code generated by the LDM. In Go examples, odd steps represent white’s moves, and even steps represent black’s moves. We visualize the latent codes of black moves in steps 2/4/6. The legend shows examples of common patterns learned for new black moves. For clarity, these moves are highlighted on the board with added colors and lines to indicate new patterns. On the right, we visualize the latent codes of the robotic arm’s movement along the X/Y/Z axes at intervals of 1, 5, and 10 frames. Points are color-coded by displacement range, with purple and red indicating the maximum displacement in opposite directions along each axis.

via the attention mechanism, captures change information in ft:t+h, yielding a continuous latent representation z˜th, which is then quantized with a discrete codebook by FSQ [41]. This quantized representation serves as an information bottleneck, preventing the LDM from learning shortcuts (e.g. trivially copying ft+h to zth).

Finally, the decoder uses the feature map ft and the latent change embeddings {zth}Hh=1 to predict the subsequent frames xˆt+1:t+H in a causal manner. The training objective of LDM is minimize the ℓ2 distance between xt+h and xˆt+h. For further details on LDM, please refer to the Supp.

By sequentially encoding changes from xt to xt+H using multiple embeddings, we achieve a compact and informative representation of temporal dynamics, which is crucial for long-term reasoning and planning tasks. As shown

- in Fig. 4, this model can learn meaningful embeddings that capture both short- and long-term dependencies in visual sequences.

Auto-regressive Transformer. The auto-regressive transformer is employed as the video generator, and it seamlessly integrates the output of the LDM with the next token prediction paradigm. This integration offers several benefits, including improved modeling of long-term dependencies, enhanced reasoning capabilities, and more efficient learning of complex visual patterns over time.

For each video x1:T, we first get latent codes

{zth}T,Ht=1,h=1 using the latent dynamics model. These latent codes, along with discretized video frames are combined

into a sequence for auto-regressive prediction. The codebook used by the video decoder is distinct from the one used in the latent dynamics model, and the vocabulary of the auto-regressive transformer is the union of both. This allows the transformer to leverage both the fine-grained visual details captured by the video encoder-decoder and the compact, task-relevant embeddings produced by LDM, enabling it to generate both visually coherent frames and maintain the underlying temporal dynamics captured by the LDM.

Mapping to Task Operation. During inference, at each time step t, we use the transformer to auto-regressively generate the latent codes {zˆth}Hh=1 and the predicted frame xˆt+1. To convert the generated content into actionable decisions for specific tasks, we further train an Inverse Dynamics Model (IDM) π like [16]. IDM consists of several MLP layers and is trained independently from the video generator, using a small amount of video action label data.

In the basic framework, the IDM takes the current frame xt and the predicted frame xˆt+1 to generate the corresponding action: π(·|xt,xˆt+1). When incorporating the LDM, the IDM is extended to take both the predicted frame and the latent codes, resulting in: π(·|xt,xˆt+1,{zˆth}Hh=1). This allows the IDM to leverage the rich temporal representations encoded by LDM, enhancing its ability to produce

more temporally consistent and accurate actions. For additional details, please refer to the Supp. A.

#### 4. Video-GoBench

Our goal is to explore whether a deep generative model can learn complex knowledge from raw videos, including welldefined rules, advanced planning and reasoning. We prioritize tasks that emphasize high-level planning and reasoning while minimizing dependence on low-level visual details like shape and texture.

To this end, we introduce Video-GoBench, a purely visual Go benchmark, with a simple yet effective visual design to capture rule learning, reasoning, and planning, and enable direct evaluation through online gameplay. By simplifying visuals, we minimize distractions from intricate textures and dynamics, thereby allowing us to focus solely on assessing knowledge acquisition abilities. Additionally, the simulated Go environment allows for the seamless conversion of large-scale, state-based gameplay data into video sequences, supporting efficient and scalable evaluation.

##### 4.1. Dataset Generation

To construct Video-GoBench, we collect 10 million 9x9 Go game records, including 3.2 million self-play training examples from KataGo [65] and 7.8 million from humanmatch games from OGS [42]. For the human data, we re-annotate the moves based on the original board states using KataGo to provide the optimal next moves for each state. The dataset comprises about 400 million unique board states, each converted into 256-pixel images. Additionally, we curated a test set of 1,000 matches, with each move reannotated by KataGo. To improve its ability to benchmark model generalization on new boards, we removed opening moves—given the limited diversity in early-game positions—resulting in 56,000 unique board states.

###### 4.2. Evaluation We use the following metrics to evaluate our models:

- • Legal rate: The percentage of legal moves generated by the model on the test set, indicating the model’s ability to understand and adhere to the basic rules of Go.
- • Game playing strength (Elo): Following [48], we evaluate models through an internal tournament. Eight agents, as shown in Tab. 1, play 400 games per pair, totaling 11.2K games. Elo ratings are calculated using BayesElo [14] with a default confidence of 0.5. Among the agents are three versions of the KataGo engine, finetuned to 1-dan, 5-dan, and 9-dan human skill levels, to showcase the model’s strategic levels in relation to human play. We anchor the KataGo 9-dan model’s Elo rating at 2700, aligning with the standard human 9-dan score.
- • Action accuracy: Percentage of test instances where the model selects the best move made by KataGO (oracle).

[Figure 7]

Figure 5. Illustration of playing against KataGO and UMAP projection [34] of the predicted latent code. Our model plays as black. The generated latent code is visualized through the LDM decoder and new stones in the visualization are marked with colors to match the legend. The visualization serves as a probe, indicating that the model shows signs of forward planning.

• Action-Value: During tournament matches, we record the board state at each move and use KataGo-9d to annotate the action-values, which are the expected cumulative rewards for all possible moves in each state, serving as oracle values. We then calculate the ratio of the action-value of each move made by the model to the oracle action-value. This metric measures the model’s ability to evaluate moves. An action-value of 100% indicates that the model’s move matches the level of KataGo-9d.

#### 5. Experiment

##### 5.1. Implementation Details

Our auto-regressive transformer is randomly initialized, with its encoder initially trained on the target dataset for reconstruction and then frozen. Trainable parameters including the transformer layers, projection layers, and output layers. The default vocabulary size for both LDM and transformer encoders is 64,000, corresponding to FSQ levels [8,8,8,5,5,5]. For Go and CALVIN environments, we increase the number of downsampling layers in each encoder, compressing each frame to 4x4. We use a frame length of 6 for Go and 10 for CALVIN. We train and test in the CALVIN ABCD→D split. RLBench shares the same training settings as CALVIN, with 20k trajectorie data generated by custom scripts. The training uses the AdamW [37] optimizer with a learning rate of 0.0003 and no weight decay. The batch size is set to 256 for Go and 32 for CALVIN, requiring approximately 4 and 2 days of training on 8 A100 GPUs, respectively.

Idx Agent Train w/o Search Input Legal rate (%) Action-Value (%) Best Action Acc. (%) Tournament Elo

- 1 KataGO-human-1d RL State 100 67.6 64.5 2019±23
- 2 KataGO-human-5d RL State 100 83.5 83.7 2253±20
- 3 KataGO-human-9d (Oracle) RL State 100 100 100 2700

- 4 Transformer 300M SL State 99.8 79.7 87.2 2308±21

- 5 Transformer 300M SL Video 99.6 59.7 58.9 1998±38

- 6 VideoWorld 50M (ours) SL Video 99.5 73.9 80.9 2093±25
- 7 VideoWorld 150M (ours) SL Video 99.7 82.0 86.7 2218±23

- 8 VideoWorld 300M (ours) SL Video 99.7 83.7 88.1 2317±25

- Table 1. Comparison on Video-GoBench. KataGO-human-9d represents the highest human level and serve as the Oracle for best actions.

Agents Input/Ouput

Task Success Rate (%) Push Open/Close Turn on/off

MCIL [39] Video/Lab. Action 33.0 38.7 41.2 HULC [38] Video/Lab. Action 65.8 80.9 85.3 Transformer (Oracle) Video/Lab. Action 75.4 95.3 96.2 Transformer Video 17.3 24.1 19.2

VideoWorld Video 56.2 75.4 72.1 VideoWorld (+10k data)† Video 65.3 81.2 79.3 VideoWorld (+30k data)‡ Video 72.7 91.0 93.8

- Table 2. Comparison on CALVIN benchmark. “Lab.” means annotated labels. All models have 300M parameters. † and ‡ denote using an additional 10k and 30k CALVIN trajectories for training, respectively.

Agents

CALVIN RLBench Push Open/Close Turn on/off Microwave Fridge

Transformer (Oracle) 61.3 79.5 78.0 72.1 69.0 Transformer 6.5 13.0 15.6 12.0 10.9 VideoWorld 56.0 74.8 74.5 67.1 62.5

- Table 3. Results of joint training on CALVIN and RLBench.

##### 5.3. Key Findings with The Basic Framework

Model can learn basic knowledge from raw videos. As shown in Tab. 1 (idx 5) and Tab. 2 (line two), using the basic framework to train on Go videos and robotic videos enables the model to master the rules of Go, and learn basic robotic operation skills. On the Go test set, the model demonstrates almost 100% legality, indicating that it has mastered rules including prohibitions against repeated moves, suicide moves, and more advanced rules like ko fights. Additionally, it achieves nearly a 50% accuracy rate in predicting the best moves, showing a moderate understanding of Go strategy. In robotic scenarios, the model also demonstrates a certain ability to complete tasks. Compared to Go, handling more complex appearance without relying on pre-training imposes higher demands on the model’s generation quality. Representation of visual change is crucial. While the basic framework can learn from video, we find that using more compact representations of visual changes yields significantly better performance and efficiency. In Go, a state sequence encodes each move and its corresponding positional token to represent changes on the board; in contrast, raw video captures additional visual details beyond these key actions, thus requiring a substantially larger number of tokens. Although the state-based approach is deficient in representing local patterns and shapes among stones—an important property in Go—its compactness leads to higher learning efficiency and better results. As shown in Tab. 1, training with these state sequences significantly improves performance across all metrics (idx 4 vs. 5) under the same data budget. Fig. 2 further demonstrates that the state representation converges faster. In robotic scenarios, although fully representing the environment with state sequences may not be feasible, adding action labels for each frame still markedly boosts task completion rates. These findings underscore the importance of a compact representation of visual change for effective video learning.

##### 5.2. Benchmarks

We evaluate VideoWorld on three benchmarks: VideoGoBench, CALVIN and RLBench. Go tests long-horizon reasoning and complex forward planning, while the latter two involve more intricate visual information. For Go, we use the evaluation metric detailed in Sec. 4.2. In CALVIN benchmark, our model controls a Franka Emika Panda robot with a parallel-jaw gripper in an environment with a desk, openable drawer, colored blocks, and an LED and light bulb. We evaluate three tasks: Push Blocks, Open/Close Drawer, and Turn On/Off Light. Each task is specified by an instruction label, such as “go push the red block right”, which is provided to the transformer as a condition for video generation. In RLBench benchmark, the model controls the same robotic arm, but the environment and camera views differ from CALVIN. We evaluate two tasks: Close Microwave and Close Fridge. Object positions in both environments are randomly generated for each execution. We sample 500 tasks randomly and record the success rate.

##### 5.4. Results with LDM

Based on the above observations, we introduce the Latent Dynamics Model (LDM) to enhance the efficiency and ef-

Compression length H

Go Act-Value Act-Acc.

Compression length H

CALVIN Push Open/Close Turn On/Off

Code Index

Go Act-Value Act-Acc. None 73.9 80.9

baseline 47.5 44.3 1 70.3 77.0 3 72.5 80.6 5 73.9 80.9

baseline 12.7 20.8 15.6 1 33.7 53.6 67.3 5 46.8 66.1 69.6

- 1 46.2 42.1

- 2 69.5 76.6

- 3 72.1 80.6

###### 10 50.3 71.1 69.7

All 45.8 43.7 (c) Intervene latent codes with different index LDM codebook size

(a) Different Compression length on Go.

(b) Different Compression length on CALVIN.

Data source

Data amount

Go Act-Value Act-Acc.

Go CALVIN

Act-Value Act-Acc. Push Open/Close Turn On/Off

Human† 6.8M 34.6 34.8 Human 6.8M 71.5 75.7 KataGo 3.2M 70.8 77.5 All 10M 73.9 80.9

729 65.5 71.1 12.9 20.0 16.0 15,625 69.3 72.8 20.5 24.1 36.3 64,000 73.9 80.9 50.3 71.1 69.7

262,144 50.1 53.2 29.8 30.0 31.7 (d) Different codebook size of LDM.

(e) Data source ablation. † means human data without re-annotation.

Table 4. Ablations and analysis. We conduct all experiments based on our 50M model.

Prediction Target

Go CALVIN

Act-Value Act-Acc. Push Open/Close Turn on/off

video 47.5 44.3 12.7 20.8 15.6 code 73.0 78.6 47.2 70.0 65.1

code/video 73.9 80.9 50.3 71.1 69.7

Table 5. Latent code prediction only with 50M parameters.

fectiveness of video learning, resulting VideoWorld model. Results on Go. In Tab. 1, we compare VideoWorld with three official KataGo models calibrated to human skill levels: 1-dan, 5-dan, and 9-dan. The 1-dan model represents players with a foundational understanding of Go, while the 5-dan and 9-dan levels indicate progressively advanced skills, with 9-dan serving as an Oracle at the highest human level. Notably, all these models are RL-based, using search and reward mechanisms.

We evaluate VideoWorld at three parameter scales: 50M, 150M, and 300M. All models show strong generalization on novel boards. Even the smallest model, VideoWorld50M surpasses KataGo-1d (Elo 2093 vs. 2019), while VideoWorld-300M outperforms the human-level KataGo5d (Elo 2317 vs. 2253), demonstrating VideoWorld’s strong capability for mastering complex tasks. Compared to its counterpart without LDM, VideoWorld shows a significant improvement (Elo 2317 vs. 1998), underscoring the benefits of learning with latent representations.

Furthermore, we observe that performance across all metrics improves consistently with model size, suggesting that VideoWorld’s knowledge learning capabilities continue to scale effectively, with potential for further enhancements as model size increases.

Results on CALVIN. Tab. 2 presents the results of the models on the CALVIN benchmark. VideoWorld, relying solely on observed videos and does not use action labels, achieves performance close to that of models supervised with real action labels. This demonstrates that our proposed LDM effectively supports video-based knowledge learning, even

in more visually complex scenarios. Furthermore, we use a supervised learning agent [66] to generate an additional 30k trajectory data in the CALVIN environment, which we incorporate into VideoWorld’s training. As shown in Tab. 2, VideoWorld demonstrates data scaling capabilities, further approaching oracle performance as data volume increases. This demonstrates the potential of our method for application in larger-scale training.

Generalization across multiple environments. To further validate VideoWorld’s generalization across different environments, we introduce two additional tasks from RLBench, which are visually distinct from CALVIN: closing microwave and closing fridge. For each task, we generate 10k trajectories and train VideoWorld on these data combined with CALVIN. We then evaluate the task success rate in both environments. As shown in Tab. 3, VideoWorld generalizes well across two settings, simultaneously mastering skills in two environments and approaching oracle performance. Reinforcement learning methods often struggle to generalize across different environments, as they heavily rely on task-specific states/action/rewards, etc. This also demonstrates the potential of our method to serve as a general knowledge learner. See Supp. A for more details.

##### 5.5. Understanding Learned Knowledge with LDM

The latent representation learned in LDM provides valuable insights into the knowledge learning process of VideoWorld. Below we offer an in-depth analysis of what our model learns through latent representations.

LDM learns patterns in the training set. As shown in Fig. 4, the latent codes on the training set capture both shortand long-term dependencies, demonstrating the model’s ability to represent knowledge at different temporal scales. In the Go scenario, salient regions in the latent codes correspond to common move patterns, indicating that the model effectively embeds multi-step strategies into a compressed space, hence aiding decision-making and reasoning. Similarly, in the robotics scenario, the clustering of latent codes

across steps reveals key dynamic dependencies over various time ranges, thus benefiting diverse manipulation tasks.

LDM enables forward planning during testing. We examine the role of codes during inference. The visualization

- in Fig. 5 shows that codes from different steps group by output positions, suggesting that VideoWorld models longrange changes progressively, similar to human forwardplanning. The visualization also includes imagination of the opponent’s moves, achieving a high average action-value of 71.2% and action accuracy of 74.3%. This indicates that, at each step, VideoWorld considers long-term changes in the game situation within the latent space, enabling it to make strategic moves with a long-term perspective.

Similar findings are observed in the robotic scenario. We visualize the predicted latent codes during inference across different tasks in Fig. 6. Here, H = 9, meaning the transformer generates 9 latent codes per time step, corresponding to 9 prediction steps. As shown, the latent codes for different prediction steps are grouped by task type, indicating that they capture task-relevant dynamics. Codes for steps 1–4 show greater overlap, likely because they focus on fine-grained displacements shared across tasks. In contrast, steps 5–9 show more distinct separation by task type, highlighting the model’s ability to progressively capture longrange changes specific to each task.

LDM generates causally interrelated codes. To further investigate the impact of the learned latent codes, we conduct an intervention experiment in Tab. 4c. We replace latent codes at different time steps with random tokens and observe the effect on performance. Intervening with the first code has the greatest effect, likely due to the causal dependencies learned among the codes. This aligns with intuition: altering the first code, which represents the immediate next optimal decision, affects all future decisions over a longer horizon. These results highlight the importance of the order and structure of latent codes for effective reasoning.

##### 5.6. More Ablations

Horizon length in latent dynamics model. Tab. 4a shows the effects of varying compression lengths during LDM training. We use a default compression length of 5 steps for Go and 10 steps for CALVIN. When we vary the compression length while keeping the codebook size constant, we observe that with a compression length of 1 step (similar to pseudo action labels), both Go and CALVIN improvement significantly over the baseline. For Go, optimal performance is at 5 steps, but further increasing the length causes the LDM training to fail to converge. We attribute this to the exponential growth in possible board position variations, which overwhelms the codebook and causes training instability. In contrast, the simpler dynamic changes in CALVIN environment allow for longer compression lengths.

Latent codebook size. Tab. 4d examines the impact of

[Figure 8]

Figure 6. Illustration of robotic manipulation and UMAP projection of the predicted latent code during inference. Latent codes are visualized through the LDM decoder. The UMAP projection illustrates the 9 predicted latent codes (i.e. H = 9) across different tasks, with each point color-coded by task type. Visualizations with a yellow background show the model’s actual robotic arm control during inference, while those with a green background represent the model’s next-frame predictions during training.

LDM’s codebook size on knowledge learning. By adjusting the FSQ level, we test four different codebook sizes to evaluate how this affects the model’s ability to learn effective policies. In the Go scenario, we find that even a smaller vocabulary yields significant improvements; however, this is not the case for the robotic scenario. This discrepancy likely arises because the robotic scenario involves more motion and appearance information, requiring a larger vocabulary to capture essential details. When the vocabulary size becomes too large, LDM training struggles to converge, leading to a noticeable performance drop in both scenarios.

Data quality. Tab. 4e examines the effect of data quality on learning. Video-GoBench dataset has two types of data: human matches and KataGO self-play matches. These data sources have different distributions, and their policy content varies, potentially influencing the learning process. We

[Figure 9]

(a) Capture opponent’s stones

[Figure 10]

(b) Sacrificing short-term gains to capture more opponent stones.

Figure 7. Visualizations of learned Go Strategies. our model captures opponent stones using the squeeze tactic and self-sacrifice tactic. New black stones are red, new white stones are blue.

[Figure 11]

- Figure 8. Visualizations of performing CALVIN tasks.

[Figure 12]

- Figure 9. Visualizations of performing RLBench tasks.

find that using the original human data significantly impacts the final performance, as it includes players of varying skill levels, highlighting the importance of data quality.

Latent code prediction only. To demonstrate the necessity of jointly predicting the latent code {zˆth}Hh=1 and next frame xˆt+1 given x1:t, we remove the supervision for the next frame xˆt+1 during training. In this setup, frames are only used as input, and only the latent codes are subject to the CE loss. In this case, we retrain an IDM that only receives latent code inputs. As shown in Tab. 5, for both Go and robotic scenarios, using codes alone significantly improves performance, and incorporating next-frame prediction further enhances it. We hypothesize this is because next-frame prediction enhances the model’s understanding of the environment and helps generate more accurate codes.

##### 5.7. Visualizations

We provide more visualizations of VideoWorld performing Go and robotic tasks in Fig. 7 and Fig. 8, respectively. The Go visualizations are from matches between our 300M VideoWorld and KataGo-5d. With comparable Elo scores, these matches fairly demonstrate of how the model applies its learned knowledge. In Fig. 7a, our model captures opponent stones using the squeeze tactic. In Fig. 7b, VideoWorld deliberately sacrifices a single stone, prompting the opponent to capture it, thereby creating an opportunity to capture more of the opponent’s stones. This highlights the model’s ability to prioritize long-term planning over shortterm gains. For the robotic scenario, we show the model’s actual control results for the robotic arm, demonstrating its understanding of manipulation tasks and ability to execute them effectively.

#### 6. Conclusion

In this work, we take an initial step toward exploring knowledge learning from raw video data using the next token prediction paradigm. We conduct systematic experiments in custom Video Go and robotic simulation environments. We present two key findings: i) The model can master the rules of Go and learn fundamental robotic operations, and ii) The representation of visual change is crucial for knowledge learning. Based on these findings, we propose a latent dynamics model (LDM) to boost the efficiency and effectiveness of knowledge acquisition. Although applying this approach to real-world scenarios still faces challenges such as high-quality video generation and generalization, we believe that video generation models have the potential to serve as general knowledge learners and ultimately function as the artificial brain capable of thinking and acting in the real world.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 1

- [2] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023. 1
- [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 1
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3
- [5] Ronald J Brachman. Knowledge Representation and Reasoning. Morgan Kaufman/Elsevier, 2004. 1
- [6] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators.

2024. 3

- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 1
- [8] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024. 3
- [9] Richard W Byrne and Anne E Russon. Learning by imitation: A hierarchical approach. Behavioral and brain sciences, 21(5):667–684, 1998. 1
- [10] Dingyang Chen, Qi Zhang, and Yinglun Zhu. Efficient sequential decision making with large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 9157–9170, Miami, Florida, USA, 2024. Association for Computational Linguistics. 1
- [11] Lu Chen, Yuxuan Huang, Yixing Li, Yaohui Jin, Shuai Zhao, Zilong Zheng, and Quanshi Zhang. Alignment between the decision-making logic of llms and human cognition: A case study on legal llms. arXiv preprint arXiv:2410.09083, 2024. 1
- [12] Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In Proceedings of the 37th International Conference on Machine Learning, pages 1691–1703. PMLR, 2020. 3
- [13] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul

- Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240): 1–113, 2023. 1
- [14] R´emi Coulom. Whole-history rating: A bayesian rating system for players of time-varying strength. In Computers and Games, 2008. 6
- [15] Yilun Du, Mengjiao Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Joshua B. Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Vector quantization. IEEE Assp Magazine, 1(2):4–29, 1984. 2
- [16] Yilun Du, Mengjiao Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Joshua B. Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. In NeurIPS, 2023. 1, 3, 5
- [17] Eva Eigner and Thorsten H¨andler. Determinants of llmassisted decision-making. arXiv preprint arXiv:2402.17385,

2024. 1

- [18] Xidong Feng, Yicheng Luo, Ziyan Wang, Hongrui Tang, Mengyue Yang, Kun Shao, David Mguni, Yali Du, and Jun Wang. Chessgpt: Bridging policy learning and language modeling. Advances in Neural Information Processing Systems, 36, 2024. 3
- [19] Xin Gu, Guang Chen, Yufei Wang, Libo Zhang, Tiejian Luo, and Longyin Wen. Text with knowledge graph augmented transformer for video captioning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18941–18951, 2023. 1
- [20] Xin Gu, Heng Fan, Yan Huang, Tiejian Luo, and Libo Zhang. Context-guided spatio-temporal video grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18330–18339, 2024. 1
- [21] Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Pengfei Wan, Di Zhang, Yufan Liu, Weiming Hu, Zhengjun Zha, et al. I2v-adapter: A general image-to-video adapter for diffusion models. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 3
- [22] Kunyang Han, Yibo Hu, Mengxue Qu, Hailin Shi, Yao Zhao, and Yunchao Wei. Rose: Revolutionizing open-set dense segmentation with patch-wise perceptual large multimodal model, 2024. 1
- [23] Alex Havrilla, Yuqing Du, Sharath Chandra Raparthy, Christoforos Nalmpantis, Jane Dwivedi-Yu, Maksym Zhuravinskyi, Eric Hambro, Sainbayar Sukhbaatar, and Roberta Raileanu. Teaching large language models to reason with reinforcement learning. arXiv preprint arXiv:2403.04642,

2024. 1

- [24] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [25] William Hoppitt and Kevin N Lala. Social learning: an introduction to mechanisms, methods, and models. In Social learning. Princeton University Press, 2013. 1
- [26] Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In International conference on machine learning, pages 9118–9147. PMLR, 2022. 1

- [27] Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In International conference on machine learning, pages 9118–9147. PMLR, 2022. 3
- [28] Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J. Davison. Rlbench: The robot learning benchmark & learning environment, 2019. 2
- [29] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023. 1
- [30] Siyu Jiao, Hongguang Zhu, Jiannan Huang, Yao Zhao, Yunchao Wei, and Humphrey Shi. Collaborative vision-text representation optimizing for open-vocabulary segmentation,

2024. 1

- [31] Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 3
- [32] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022. 1
- [33] Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, et al. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint arXiv:2309.00267, 2023. 3
- [34] McInnes Leland, Healy John, and Melville James. Umap: Uniform manifold approximation and projection. The Journal of Open Source Software, 2018. 5, 6
- [35] Kenneth Li, Aspen K Hopkins, David Bau, Fernanda Vi´egas, Hanspeter Pfister, and Martin Wattenberg. Emergent world representations: Exploring a sequence model trained on a synthetic task. In The Eleventh International Conference on Learning Representations, 2023. 3
- [36] Ollie Liu, Deqing Fu, Dani Yogatama, and Willie Neiswanger. Dellma: A framework for decision making under uncertainty with large language models. arXiv preprint arXiv:2402.02392, 2024. 1
- [37] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. 6
- [38] Corey Lynch and Pierre Sermanet. Language conditioned imitation learning over unstructured data, 2021. 7
- [39] Oier Mees, Lukas Hermann, and Wolfram Burgard. What matters in language conditioned robotic imitation learning over unstructured data, 2022. 7
- [40] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for languageconditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters (RA-L), 7(3):7327–7334, 2022. 2
- [41] Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: Vq-vae made simple. In ICLR, 2024. 4, 5
- [42] OGS. Online-go. 2024. 6

- [43] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022. 3
- [44] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 3

- [45] Mengxue Qu, Yu Wu, Wu Liu, Xiaodan Liang, Jingkuan Song, Yao Zhao, and Yunchao Wei. Rio: A benchmark for reasoning intention-oriented objects in open environments,

2023. 1

- [46] Mengxue Qu, Xiaodong Chen, Wu Liu, Alicia Li, and Yao Zhao. Chatvtg: Video temporal grounding via chat with video dialogue large language models, 2024.
- [47] Zhongwei Ren, Zhicheng Huang, Yunchao Wei, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin. Pixellm: Pixel reasoning with large multimodal model, 2024. 1
- [48] Anian Ruoss, Gr´egoire Del´etang, Sourabh Medapati, Jordi Grau-Moya, Li Kevin Wenliang, Elliot Catt, John Reid, Cannada A. Lewis, Joel Veness, and Tim Genewein. Amortized planning with large-scale transformers: A case study on chess, 2024. 3, 6
- [49] Stuart J Russell and Peter Norvig. Artificial intelligence: a modern approach. Pearson, 2016. 1
- [50] Dominik Schmidt and Minqi Jiang. Learning to act without actions. arXiv preprint arXiv:2312.10812, 2023. 3
- [51] Wentao Shi, Xiangnan He, Yang Zhang, Chongming Gao, Xinyue Li, Jizhi Zhang, Qifan Wang, and Fuli Feng. Enhancing long-term recommendation with bi-level learnable large language model planning. arXiv preprint arXiv:2403.00843,

2024. 1

- [52] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature, 529(7587):484–489, 2016. 2
- [53] Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2998–3009, 2023. 1
- [54] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. 3
- [55] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1
- [56] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste

- Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. 1, 4
- [57] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In NeurIPS, 2017. 2
- [58] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 3
- [59] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 3
- [60] Jiacong Wang, Bohong Wu, Haiyong Jiang, Zhou Xun, Xin Xiao, Haoyuan Guo, and Jun Xiao. World to code: Multimodal data generation via self-instructed compositional captioning and filtering. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 4608–4623, 2024. 1
- [61] Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, and Tiejun Huang. Images speak in images: A generalist painter for in-context visual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6830–6839, 2023. 3
- [62] Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Ma, Yitao Liang, and Team CraftJarvis. Describe, explain, plan and select: interactive planning with large language models enables open-world multi-task agents. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 34153–34189, 2023. 1
- [63] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 1, 3
- [64] Andrew Whiten, Victoria Horner, and Frans BM De Waal. Conformity to cultural norms of tool use in chimpanzees. Nature, 437(7059):737–740, 2005. 1
- [65] David J Wu. Accelerating self-play learning in go. arXiv preprint arXiv:1902.10565, 2019. 2, 6
- [66] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation, 2023. 8
- [67] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 1
- [68] Sherry Yang, Jacob Walker, Jack Parker-Holder, Yilun Du, Jake Bruce, Andre Barreto, Pieter Abbeel, and Dale Schuurmans. Video as the new language for real-world decision making. arXiv preprint arXiv:2402.17139, 2024. 3
- [69] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 3
- [70] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of

- thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36, 2024. 3
- [71] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al. Latent action pretraining from videos. arXiv preprint arXiv:2410.11758, 2024. 3
- [72] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10459–10469, 2023. 3
- [73] Lijun Yu, Jos´e Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, and Lu Jiang. Language model beats diffusion – tokenizer is key to visual generation, 2024. 4
- [74] Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, xJia Deng, Boji Shan, Huimin Chen, Ruobing Xie, Yankai Lin, et al. Advancing llm reasoning generalists with preference trees. arXiv preprint arXiv:2404.02078, 2024. 1
- [75] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023. 3

[Figure 13]

## VideoWorld: Exploring Knowledge Learning from Unlabeled Videos

### Supplementary Material

#### A. Implementation Details

Training details. We present the detailed training configurations of latent dynamics model and auto-regressive transformer in Tab. A.1.

Latent dynamics model. The decoder can be divided into three parts based on three functions: i) using an encoder to causally extract image features; ii) using learnable embeddings to extract change information from the extracted features; iii) using a decoder to causally reconstruct video frames based on image feature of the initial frame and the learned embeddings. We present the PyTorch-style pseudocodes for the overall LDM and each part in Alg. 1.

Robotics action execution. Our model generates a set of latent codes {zˆth}Hh=1 and next frame prediction xˆt+1 based on language instruction and input sequence x1:t. We feed {zˆth}Hh=1, xˆt+1 and xt into the Inverse Dynamics Model to obtain corresponding action. The inferred action is executed in an open-loop manner: after predicting each action, we input it into the environment engine to obtain a new observation xt+1, which is then appended to the input sequence for the next prediction cycle. In Fig. 6, we visualize the model’s next-frame predictions and the actual control results of the robotic arm. We find that the model’s next-frame predictions align with the task’s execution intent and effectively control the robotic arm to complete the tasks. We visualize the latent codes at different time steps during testing and analyze them in Sec. 5.5.

Inverse dynamics model. The training objective of IDM is to predict the control signals for the robotic arm, represented as a 7-dimensional vector that includes the arm’s displacement along the XYZ axes, Euler angles, and the gripper’s open/close state. Specifically, IDM uses a ResNet-18 to process the predicted frame, applying a global average pooling layer to the penultimate feature map to obtain a feature vector of size (1, 512). Simultaneously, a shared MLP transforms the feature vectors of each latent code into a size of (H, 512), where H, as defined in Sec. 3.2, represents the number of learnable embeddings or prediction steps in LDM. Then, the features from the latent codes and the generated frames are concatenated into a vector of size (H + 1, 512), which is then passed through another MLP with dimensions (512, 7) followed by a global average pooling layer to produce the final 7 control signals. IDM is trained using AdamW with a learning rate of 1e-4 for a total of 1 million steps, with mean squared error as the objective.

RLBench evaluation. In Sec. 5.4, we test VideoWorld’s ability to perform tasks in two different environments:

|Config<br><br>|LDM|AR Transformer|
|---|---|---|
|optimizer base learning rate weight decay optimizer monmentum batch size learning rate schedule warmup iterations max iterations augmentations Training Loss Training Target|AdamW<br><br>5.4e-5<br><br>0.01<br><br>β1, β2=0.5,0.9<br><br>16<br><br>WarmipDecayLR<br><br>3e+4 1e+5 None L2 loss<br><br>Reconstruction<br><br>|AdamW 3.0e-5<br><br>0<br><br>β1, β2=0.9,0.98 256(Go),32(CALVIN) WarmipDecayLR 3e+4<br><br>1e+6 None Cross Entropy loss Next token pred.<br>|

)

Table A.1. Training configurations for the latent dynamics model (LDM) and auto-regressive (AR) transformer.

[Figure 14]

(a) Distribution of board state count in training set

[Figure 15]

(b) Repetition rate during online battle.

Figure A.1. Dataset statistics. Best viewed digitally.

CALVIN and RLBench. For CALVIN, we maintain the original task settings. For RLBench, we generate 20k trajectory data using a script. RLBench uses the same robotic arm and action space as CALVIN, but the environment appearance and task settings differ significantly. We use the combined RLBench and CALVIN datasets to train both the LDM and Transformer, while IDM is trained separately in each environment. Fig. 9 shows VideoWorld performing robotic tasks in RLBench environment.

Algorithm 1: Pseudo codes of LDM.

# Inputs: video:The first frame and its subsequent H frames [1+H, h, w, 3];

# Variables: ldm q: Learnable embeddings for H time spans [H,C]; image pe:position embedding of image features;

# Functions: CrossAttention();MLP(); up scale(); down scale(); FSQ(); Causal3DCNN()

- 1 def encoder(video): # video:video sequences.[1+H,h,w,3] # The encoder consists of a set of encoder

layers, composed of Causal3DCNN and down scale layers.

- 2 f = Causal3DCNN(video);# f:[1+H,h,w,C]
- 3 for layer in encoder layers: # process and downsample features using

Causal3DCNN and down scale.

- 4 f = layer(f) # Capture dynamic changes in video.

- 5 z = ldm qformer(f)

- 6 return z, f[:0];# f:[1+H,h’,w’,C]. z:[H,C]

- 7 def ldm qformer(f): # f:features of each frame.[1+H, h’, w’, C]

- 8 q list = []

- 9 for h in range(H):

- 10 query = ldm q[h];# query:[1,C]

- 11 f h = f[:(1+h)];# f h:[1,h’,w’,C]

- 12 key = f h + image pe;# key:[1,h’,w’,C]

- 13 q h = CrossAttention(q=query, k=key,

v=f h);# q h:[1,C]

- 14 q h = MLP(q h);# q h:[1,C]

- 15 q list.append(q h)

- 16 q list = stack(q list);# q list:[H,C]

- 17 return q list

- 18 def decoder(z, first f): # z:ldm query embeddings that captures change

information in video frames.[H, C]; first f:image features of first frame.[1, h’, w’, C]

# The decoder consists of a set of decoder layers, composed of Causal3DCNN and up scale layers.

- 19 z = repeat interleave(z, h’, dim=-2)

- 20 z = repeat interleave(z, w’, dim=-1)

- 21 rec video = cat(first f, z); # [1+H, h’, w’, C]

- 22 for layer in decoder layers: # process and upsample features using

Causal3DCNN and up scale.

- 23 rec video = layer(rec video)

- 24 rec video = Causal3DCNN(rec video)

- 25 return rec video;# rec video:[1+H, h, w, C] # Main Function

- 26 def latent dynamics model(video): # video:video sequences # extract change information from video frames

- 27 z, first f = encoder(video)

- 28 z = FSQ(z); # quantize z using FSQ # LDM training stage
- 29 if is train: # obtain reconstructed video frames.

- 30 rec video = decoder(z, first f) # we train ldm using mse loss.

- 31 loss = MSE(rec video, video)

- 32 return loss

- 33 return z

extensive references for the model to learn from. Moreover, due to the combinatorial explosion effect, the proportion of repeated board states—those encountered during inference that also appear in the training set—decreases sharply as the game progresses when our model competes against reinforcement learning agents. This ensures that good performance cannot rely on memorizing training scenarios, highlighting the model’s generalization ability.

Board state count. The Video-GoBench training set contains approximately 400M unique board states. We analyze their distribution based on move numbers, i.e., the number of stones each state contains. As shown in Fig. A.1a, the data features a diverse range of board states, primarily concentrated within the first 100 moves, with significantly fewer states beyond that.

Board state repetition rate. We collect 400 game records between our model and KataGo-9d, calculating the overlap rate of board states with the training set across different move numbers. As shown in Fig. A.1b, the repetition rate drops sharply as the games progress. By move 30, it reaches zero, with all games continuing beyond this point. This eliminates the possibility of relying on memory alone to achieve high performance.

#### B. Details on Video-GoBench

In this section, we systematically analyze Video-GoBench. The benchmark includes many unique board states, offering

