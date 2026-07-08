## Scaling Behavior Cloning Improves Causal Reasoning: An Open Model for Real-Time Video Game Playing

Yuguang Yue† Irakli Salia† Samuel Hunt† Chris Green† Wenzhe Shi† Jonathan J Hunt†

# arXiv:2601.04575v2[cs.AI]28Jan2026

Abstract

Behavior cloning has seen a resurgence as scaling model and data sizes demonstrate strong performance. In this work, we introduce an open recipe for training a video game playing foundation model designed for inference in realtime on a consumer GPU. We release all data (8300+ hours of high quality human gameplay), training and inference code, and pretrained checkpoints under an open license. Empirically, we show that our best model achieves performance competitive with human players across a variety of 3D games1. We use this recipe to investigate the scaling laws of behavior cloning, with a focus on causal reasoning. In a controlled toy setting, we first demonstrate that increasing training data and network depth leads to the model learning a more causal policy. We then validate these findings at scale, analyzing models up to 1.2 billion parameters. We observe that the causal improvements seen in the toy domain hold true as model size and training steps increase.

1. Introduction

Artificial intelligence (AI) has been applied to game playing since its inception (Turing, 1953). When models are allowed to interact directly with an environment, reinforcement learning has achieved remarkable success, including superhuman performance in complex games (Berner et al., 2019; Vinyals et al., 2019). However, such systems are typically tailored to a single game, as they rely on carefully engineered training

†Player2, USA. Correspondence to: Yuguang Yue <yuguang@elefant.gg>. Preprint. January 30, 2026.

1All data, code, model checkpoints and videos of game playing are available from the accompanying website https: //elefant-ai.github.io/open-p2p/.

environments and substantial manual design of reward functions, limiting their generality and scalability.

Behavior cloning (BC), by contrast, is a simple and long-standing approach to policy learning that formulates control as supervised learning from state–action pairs (Pomerleau, 1989; Bain & Sammut, 1995). Because it learns solely from collected datasets, behavior cloning has the potential to generalize across diverse game environments without requiring environmentspecific reward engineering. Nevertheless, BC is known to suffer from two fundamental challenges: distributional shift (Ross et al., 2011) and causal confusion (De Haan et al., 2019), both of which can severely degrade performance.

In this paper, we train a single model capable of playing a range of 3D video games using raw image observations and producing keyboard and mouse actions in real time on a consumer-grade GPU. Our approach leverages BC with a large-scale dataset collected across a diverse set of games. We release all training and inference code, as well as the dataset (an example is shown in Figure 1), under open licenses. We demonstrate that the model can play simple games that do not require complex planning with a high level of competence.

We study the scaling laws of BC models under dataconstrained regimes by training four model sizes ranging from 150M to 1.2B parameters across five dataset size ranges of 6%, 12%, 25%, 50%, and 100% of the full training data. We observe a clear power-law relationship between test loss and dataset size.

Causal confusion occurs due to non-causal correlations in the data, which can result in the policy learning to predict the action using these non-causal correlates. An example, taken from De Haan et al. (2019), is that the policy may learn to apply the brakes when it sees the brake lights since these are highly correlated with braking. Obviously, such a non-causal policy performs poorly.

Using both a simple toy example with a rigorous causality metric and empirical results with an approxi-

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

w a s d

w a s d

w a s d

w a s d

w a s d

keyboardmouse action

Navigate through the hotel shortcut towards the parking lot and Quarter Odis.

Advance towards the special forces base via the scrap yard.

Move towards the dirt road, passing the tank truck and parking lot.

text instruction

Enter the convenience store.

Activate the VANT scorestreak.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

w a s d

w a s d

w a s d

w a s d

w a s d

keyboardmouse action

Deploy and control the Predator Missile scorestreak.

Switch to the minigun.

Enter the garage. Activate the VANT scorestreak. Switch to the minigun.

text instruction

- Figure 1. Example gameplay sequence with aligned action and text annotations. For visual clarity, we only show the frames where a text annotation is initialized, keyboard actions are simplified to WASD inputs, and mouse clicks are omitted; The highlighted key means the key is pressed, and the arrow indicates mouse movement in the x and y directions.

mate causality score, we show that in our setting, scaling both model and dataset size leads to models that more reliably attend to causal signals. These results suggest that a practical approach to mitigating causality issues in behavior cloning is to scale up model capacity alongside data size and diversity.

- 2. Gameplay Dataset

- 2.1. Annotated data

We collect a large-scale, high-quality dataset of human gameplay spanning a diverse set of popular 3D video games. The complete list of games is provided in Appendix A.1. Gameplay is recorded by experienced players who are instructed to capture only active gameplay segments (e.g., excluding lobby or waiting periods). Annotators use a variety of hardware, monitor sizes and resolutions, mouse sensitivities, and gameplay styles, which provide a diverse set of gameplay videos. We do not enforce balanced data distribution across games.

All gameplay videos are recorded at 20 frames per second (FPS), following prior work (Baker et al., 2022).

For each frame, we capture the raw screen pixels oi together with the corresponding keyboard and mouse actions ai. A gameplay trajectory is represented as a sequence

[o1,a1,o2,a2,...,oT,aT],

After filtering for quality (see Appendix A.3 for details), the dataset comprises over 8,300 hours of highquality human gameplay, corresponding to around 600 million image–action pairs. The distribution of recording hours across games is shown in Appendix A.2.

2.1.1. Text-Annotated Data

To enable text-conditioned policy learning, we augment the gameplay data with text annotations. These annotations provide text instructions that the model is trained to follow over temporal windows ranging from several seconds to a few minutes.

Text annotations were generated retrospectively using a commercial VLM. The VLM was prompted to review gameplay segments and infer plausible instructions that could have guided the human player at specific timestamps. This task is inherently challenging, as it requires inferring the player’s underlying intent from recorded behavior and does not admit a unique mapping. While VLMs are not currently capable of playing games at a high level (Zhang et al., 2025), we found that they produce good-quality instructional descriptions when analyzing gameplay videos offline, as shown in Figure 1.

A key challenge in this process is that commercial VLMs typically operate on temporally compressed videos (often around 1 Hz), which is insuﬀicient for text annotation in fast-paced games. To address this limitation, we deliberately prompt the VLM to generate temporally extended, goal-oriented instructions (e.g., “move toward the skull gate”) rather than instantaneous commands (e.g., “turn left”). We further design the prompt to suppress repetitive, high-frequency events, such as shooting, by annotating such behaviors only once per contiguous segment. Finally, the VLM is prompted to generate start and end timestamps, at second-level precision, for each instruction, which are then aligned with the original video recordings.

A single unified prompt is used across all games to

ensure scalability and avoid per-game customization. The full prompt is provided in Appendix A.4.

- 2.1.2. Correction Data

A common challenge of behavior cloning is distributional shift, where the state distribution encountered during online deployment deviates from that of the training data. Inspired by DAgger (Ross et al., 2011), we mitigated this issue by collecting human correction data.

To collect correction trajectories, we deploy a trained policy to interact with the game environment while a human annotator monitors its behavior. When the policy encounters out-of-distribution situations (e.g., becoming stuck or exhibiting degenerate behavior), the annotator temporarily takes control to guide the agent back to a valid state. Control is then returned to the policy. For behavior cloning, we used only the human actions and masked out the loss for actions taken by the agent. These correction trajectories are mixed with the original annotated data during training and constitute less than 1% of the total annotated dataset.

- 2.1.3. Simple Benchmark Environments

In addition to commercial games, we collected data from two lightweight 3D environments we designed for automated evaluation: Hovercraft and Simple-FPS (see Appendix A.5). These environments are fully programmatic, allowing precise control over game state and diﬀiculty, which provides a fair and controlled benchmark for comparing model performance (see Section 4).

- 2.2. Unlabeled Data

In addition to annotated gameplay data, we curated a large corpus of unlabeled gameplay videos from public sources (Fan et al., 2022; Baker et al., 2022). Details of the unlabeled dataset and its usage are provided in Appendix B.

- 3. Policy Model

We present a multimodal action policy model, which we refer to as Pixels2Play (P2P). P2P is a textconditioned policy that takes visual observations and optional textual instructions as input and outputs lowlevel keyboard and mouse actions. The model builds upon the transformer-based policy architecture introduced in Yue et al. (2025), but introduces substantial extensions to support text conditioning and to improve online performance by explicitly conditioning the backbone transformer on ground-truth action tokens dur-

ing training.

A primary design constraint is that the model must operate in real time (20 Hz) on high-end consumer GPUs (e.g. NVIDIA RTX 5090), enabling deployment on end-user hardware. To satisfy this requirement, we trained a lightweight, decoder-only transformer backbone from scratch rather than fine-tuning a large pretrained VLM, as explored in prior work (Kim et al., 2024; Tan et al., 2025). This design offers two advantages: (i) it enables a custom image tokenization pipeline that produces a small number of image tokens, allowing the model to attend to longer temporal histories, which is important for games that require long-term memorization; and (ii) the reduced model size and custom architecture ensures fast inference and compatibility with compilation and optimization techniques.

We evaluated several image encoders, including MAGVIT-v2 (Yu et al., 2023; Luo et al., 2024), DINOv2 (Oquab et al., 2023), IBQ (Shi et al., 2025), and CosmosTokenizer (Agarwal et al., 2025). We found that EﬀicientNet (Tan & Le, 2019), following Pearce & Zhu (2022), provides the best trade-off between representation quality and computational eﬀiciency. Specifically, we use the first six layers of EﬀicientNet, followed by a linear projection into a small set of visual tokens (typically Ni ∈ {1,..., 4}). Unfreezing the image encoder during training consistently improves performance compared to freezing it (Appendix C.1). The model operates directly on the resized raw pixel inputs (192 pixels × 192 pixels).

Much prior work focused on single-game agents and adopted reduced action spaces that can be modeled as a single categorical variable (Baker et al., 2022; Pearce & Zhu, 2022). In contrast, our setting requires a unified policy capable of operating across many games, necessitating a much larger action space comprising the full keyboard and mouse input domain. We allowed up to four simultaneous key presses and two concurrent mouse actions, making treating the action as a single one-hot combinatorial approach impractical.

To address this, we modeled actions autoregressively. We used Na = 8 tokens for each action: 4 keyboard tokens, 2 mouse tokens (x, y movement), and 2 mouse button tokens. To avoid increasing the token count of the main transformer, we introduced a lightweight action decoder. The backbone policy transformer outputs a single latent action token ain, which is then decoded autoregressively by the action decoder into the full action specification (Figure 2a). This design allows the policy transformer to perform a single forward pass per timestep during inference, yielding approximately

###### Output

[Figure 11]

Action Decoder

Action Decoder

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Policy Transformer

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Input

| |
|---|

Special token (learned token)

Text token Image token

Predicted action token True action token

(a) (b)

- Figure 2. (a) Architecture of P2P. The core policy transformer and action decoder are both decoder-only transformers. Each timestep begins with a text token ti. Since many frames do not contain a text annotation there is a default text token tnull used on these frames. This is followed by image token(s) from video frame oi followed by a learnable “reasoning’’ token ki that grants the model extra computation. The policy transformer then outputs a single action prediction token ain. We refer to the reasoning token and the action prediction token as special tokens in the architecture. A smaller transformer, the action decoder, then auto-regressively transforms and samples the single action prediction token into the

full action space. Then the true action tokens ai are input so that ain at time i + 1 can attend to the true action tokens from time i. (b) Attention mask used in our transformer policy (green denotes 1 and gray 0). This custom mask ensures the action prediction token ain at time i cannot attend to the ground truth action at time i. Note that no other tokens attend to ain to stabilize the training process.

a 5× speedup in real-time execution compared to directly predicting all action tokens.

For text embeddings, we use EmbeddingGemma (Vera et al., 2025) to encode textual inputs and apply mean pooling over token embeddings to obtain a single text representation. Text embeddings are precomputed, and the text tokenizer is frozen during training.

At each timestep, the policy transformer consumes image tokens (oi), a text-conditioning token (ti), groundtruth action tokens (ai), and a single action prediction token (ain). We further introduce an optional “thinking” token k, which provides the model with an additional reasoning step prior to action prediction. We refer to the reasoning token and the action prediction token as special tokens in Figure 2a. The resulting token count per timestep is Ni + Na + 3. Each token is augmented with a learned type embedding indicating its role (image, text, reasoning, ground-truth action, or action prediction). Rotary positional embeddings (Su et al., 2024) are applied at every transformer layer. During inference, we employ key–value caching with sliding-window attention to bound memory usage.

Because ground-truth action tokens are provided as input during training, we design a custom causal attention mask to prevent information leakage. The action prediction token ain is prohibited from attending to ground-truth action tokens at the same timestep, en-

suring causality. Other tokens (image, text, reasoning, and ground-truth actions) may attend to each other within the same timestep, but are restricted from attending to ain tokens from previous timesteps to avoid training instability (Figure 2b).

Behavior cloning often suffers from causal confusion (De Haan et al., 2019). One particular case of this, which becomes worse at higher frequencies, is that the model learns to copy previous actions instead of attending to visual inputs (Wen et al., 2020). Although this issue can be avoided by excluding ground-truth actions from the policy inputs, we find that conditioning the policy on ground-truth action tokens produces more human-like behavior than conditioning on visual observations alone. In particular, the policy learns to sustain actions over multiple frames before switching, more closely resembling human gameplay dynamics. Qualitative differences in behavior are illustrated at https://elefant-ai.github.io/open-p2p/.

We observe pronounced causal confusion when directly predicting action tokens without an action decoder and when we only had a small amount of training data. In contrast, introducing an action decoder and scaling the dataset size substantially mitigates this issue. A detailed analysis is provided in Section 5.1.

- 3.0.1. Leveraging unlabeled data

Since unlabeled data from publicly available resources are far more abundant than annotated data, it is desirable to leverage such unlabeled data for training. Details on leveraging unlabeled data, including the training procedure and some preliminary experimental results, are provided in Appendix D.8.

- 3.1. Mitigating the Training–Inference Gap

Early experiments revealed a substantial performance gap between offline evaluation and online deployment. We traced this gap to discrepancies between training and inference inputs arising from video recording, compression, and resizing. For practical latency and storage considerations, videos undergo two lossy processing steps during data collection: (1) compression and upload on the annotator side, and (2) compressing and resizing to 192 × 192 for model processing. During inference resizing occurs but no compression takes place.

We observed a substantial divergence in model outputs when using uncompressed raw frames versus resized training frames, which leads to deceptively strong training metrics but poor online performance. We measured this gap by collecting a small number of uncompressed videos. We then compressed the videos using differing compression options and compared the trained model probabilities on the uncompressed versus compressed output. Although compression at the annotation stage is unavoidable (which causes the irreducible gap), we find that the choice of color space during resizing plays a critical role: RGB encoding yields a smaller training–inference gap than YUV encoding (Figure 3a). Unfortunately, NVIDIA hardware encoders support only the YUV color space; therefore, we adopt a mixture of QP values ranges from 6 to 18 to balance encoding speed and encoding quality.

We also found that two different resizing functions were used between inference and training code paths that, while visually indistinguishable, contributed to the training–inference gap. The training code used a PyTorch function, while inference used a Rust function. We modified the code to ensure a bitwise identical resizing function was used for both training data and inference, and this mitigated the issue.

Data augmentation also substantially reduces the training–inference gap. We applied mild spatial transformations, color perturbations, Planckian jitter, ISO noise, random gaussian or motion blur, sharpening, and translation during training. As shown in Figure 3b, these augmentations significantly reduced the discrepancy and improved online performance. Con-

sequently, all experiments in this work employ data augmentation. We believe further improvements to reduce the training-inference gap via targeted data augmentation are an interesting area for future work. The detailed parameters are shown in Table 3.

[Figure 12]

- (a)

[Figure 13]

- (b)

Figure 3. (a) Gap induced by video compression without data augmentation. We measured this gap by comparing model outputs on raw frames (inference) and resized frames (training). An irreducible gap arises from lossy video compression during data collection. The gap was smaller with RGB than YUV encoding and increases as compression quality degrades (lower file size). The x-axis qp denotes the quantization parameter, where larger values indicate lower quality. (b) Data augmentation mitigates this gap when reasonable compression quality is used.

Finally, discretization of the mouse action space can lead to overly aggressive or overly conservative mouse movements. Following Pearce & Zhu (2022), we discretized mouse actions using quantile-based bins, which provide fine resolution near zero but coarse resolution in the tails. To improve robustness, we fit truncated normal distributions to the x and y axes of the mouse action space using undiscretized training data. At inference time, the policy’s predicted discrete mouse action defines the upper and lower bounds, from which we sample the final mouse action using the fitted

truncated normal distribution. The mean and standard deviation estimated from the mouse movements data are µˆx = 0,µˆy = 0,σˆx = 96,σˆy = 22. Empirically, this approach yields smoother control and better online performance.

- 4. Evaluation

We trained policy models at four parameter scales (150M, 300M, 600M, and 1.2B) and compared their performances. More technical details of the training parameters are in Appendix D.1.

We evaluated our models using (i) the scores from controlled programmatic environments, (ii) human preference evaluations on real games, and (iii) quantitative analyses of scaling behavior on test loss and causality. Although several game benchmarks exist (Zhang et al., 2025; Park et al., 2025; Xu et al., 2025; Tomilin et al., 2023), they primarily focus on text-heavy 2D games or sandbox-style 3D environments, which differ substantially from the real-time, first-person gameplay data used in our training setup.

Textual input is used exclusively for the instructionfollowing evaluation and is not provided in any other evaluation setting.

Camera settings can significantly impact performance when playing games in real time, as overly high sensitivity makes it diﬀicult for the model to adjust to small movements. Detailed camera configurations are provided in Appendix D.2.

- 4.1. Simple Programmatic Environment

We first evaluate our models in two programmatic environments: Hovercraft and Simple-FPS. These environments were implemented by us in Godot (Holfeld, 2023), allowing full control over map layouts and difficulty settings, which enables fair and reproducible comparisons across model variants.

For Hovercraft, we measure the time (in seconds) required for the agent to complete a full loop. For Simple-FPS, we report the number of hits on the enemy minus the number of hits received. We also report end-to-end inference latency measured on a single RTX 5090 GPU. We compared models of varying sizes trained on the full labeled dataset. Each model was evaluated 16 times in the same environment, and we report the mean and standard deviation. The results are summarized in Table 1. Overall, the largest

2Perplexity of the keyboard was used as test loss because some games do not require mouse actions so mouse perplexity was more noisy.

Model Size Hovercraft ↓ Simple-FPS ↑ FPS ↑

##### 150M 41.1 ± 8.8 25.3 ± 6.5 80 300M 41.7 ± 8.2 25.1 ± 5.6 64 600M 38.1 ± 4.2 26.5 ± 10.4 62 1.2B 36.8 ± 2.9 26.6 ± 3.7 40

Table 1. Performance on Godot-based programmatic environments across model sizes. For Hovercraft, the score is the time (in seconds) required to complete a loop; for Simple-FPS, the score is the number of enemy hits minus the number of hits received. Each model is evaluated 16 times per environment, and we report the mean ± standard deviation. FPS denotes the achievable inference throughput (frames per second) on an RTX 5090 GPU.

model achieves the highest mean score with the lowest variance.

4.2. Human Evaluation in Real Environments

Because our policy is designed for human-like real-time interaction in real video games, it is challenging to evaluate. We used human evaluation on gameplay videos generated by the models. We evaluate performance across four games: two single-player titles (DOOM and Quake from the Steam platform) and two multiplayer games (Roblox Be-a-Shark and Roblox Hypershot). For DOOM and Quake, we manually divide each game into three checkpoints. The model is initialized from each checkpoint and run for one to two minutes or until it reaches the subsequent checkpoint. To reduce evaluation variance, we run the model three times for each checkpoint in DOOM and Quake. Together with Be-a-Shark and Hypershot, this results in a total of 20 evaluation videos for each model.

Human evaluators assessed model quality by counting the occurrences of the following issues during gameplay: (1) colliding with walls; (2) shooting into the air; (3) missing targets (including items or enemies); (4) exhibiting non–human-like behavior (e.g., repeating loops or moving backward); (5) remaining idle; and (6) camera shaking or jittering. For each gameplay video, evaluators recorded the number of occurrences of each issue and normalized the counts by the video length. Lower values indicate better performance. Evaluators were blinded to the model that generated the video.

Figure 4a presents quantitative preference comparisons across model sizes, with the underlying rubric scores reported in Appendix D.4. Larger models are consistently preferred over smaller ones, in agreement with the numerical results in Section 4.1.

4.2.1. Instruction-following

We evaluated the model’s ability to follow text instructions. We performed this evaluation in the Quake en-

[Figure 14]

- (a)

[Figure 15]

- (b)

Figure 4. (a) Human preference comparisons across model sizes. Blue and gray bars indicate preferences for each model, while the centered numbers denote the percentage of ties. Each model is evaluated on 20 game checkpoints, with gameplay trajectories recorded. (b) Instructionfollowing comparison. Each model is evaluated from the same maze checkpoint with and without an instruction (“press the red button”), with five runs per condition. We report the success rate of completing the maze.

vironment, where experiments can reliably start from the same checkpoint. We selected a maze scenario in which the player must press three red buttons on the wall to unlock a door (see Appendix D.5 for details).

Without textual input, the model often fails to press all three buttons. This failure mode is expected, as the policy primarily imitates expert trajectories, and the action of not pressing a button introduces only subtle deviations in behavior unless explicitly emphasized by instruction. We evaluated all models on this maze over five runs and compared their success rates. As shown in Figure 4b, providing the text instruction “press the red button” substantially increases the success rate compared to the no-text baseline for all the models, demonstrating that the model actively conditions its behavior on text input.

We note that the model can currently follow only simple instructions that are similar to those seen during training, due to the limited quantity and diversity of textual instructions in the training data. We expect this limitation to be alleviated by scaling up text instructions in both diversity and volume.

[Figure 16]

Figure 5. Lowest test loss versus dataset size for the 1.2B model. As might be expected, we find the test loss fits a power-law curve closely.

5. Causality and scaling laws

We used our working recipe for model training and architecture to investigate the relationship between model and dataset size and both the test loss and the causal behavior of the model. We focused on the data constrained regime, which is becoming an increasing focus in other modalities (Muennighoff et al., 2025; Kim et al., 2025; Pearce et al., 2024). In this regime, we train each model/data size for multiple epochs until overfitting or a lack of further improvement is observed.

We used a static subset of approximately 500M frames (about 7,000 hours of gameplay). Models are trained on fractions of this dataset (100%,50%,25%,12%,6%) across four parameter scales (1.2B,600M,300M,150M).

Following Kaplan et al. (2020), we fitted the empirical scaling relationship

L(D) = L∞ + (

)α ,

Dc D

where L∞ denotes the irreducible loss, D is the number of training frames, and Dc and α are fitted constants.

For the 1.2B model, we estimate L∞ = 1.111, α = 0.2336, and Dc = 17, as shown in Figure 5. Scaling curves for all model sizes are reported in Appendix D.6.

In Appendix D.6 Figure 14, we provide the full test loss trajectories as a function of training steps for all combinations of model scale and dataset size. These results show that, in general, larger models achieve lower test loss, and increasing the amount of training data consistently reduces test loss. Moreover, larger models benefit more from additional data in data-abundant regimes (e.g., when using 50% or 100% of the training

data).

[Figure 17]

[Figure 18]

(a) (b)

- Figure 6. (a) A toy environment we used to investigate causality in behavior cloning. The observation contains both a causally informative feature (is an obstacle present) and a correlated but non-causal feature (is the brake light on from the previous frame). (b) We find that increasing the depth of the network improves the speed of learning a causally correct solution. We also find that for all nonlinear networks, an approximately causality correct solution is found using SGD. Despite the fact that a optimal linear policy exists, we find that SGD makes no progress towards learning a solution with a randomly initialized linear network.

- 5.1. Causality Analysis

- 5.1.1. Toy problem

Before presenting the results on the large behavior cloning models, we first construct a simple environment to better understand the relationship between causality and network depth. As we will show, the results demonstrate surprising commonality with the scaled up models.

In the toy environment, the observation consists of two binary features (Figure 6a) and three distractor features of random noise. The action is a single binary choice (brake, don’t brake). The binary features indicate when an obstacle is present and whether the brake was applied in the previous step (i.e. brake light). The optimal policy is to brake whenever an obstacle is present and to ignore all other components of the observation. However, because obstacles persist for multiple frames, there is a strong correlation between the brake light feature and braking.

We train simple neural networks consisting of a linear (with bias) network and multilayer perceptrons (MLPs) of differing depths with ReLU non-linearity. All networks have a final sigmoid, so that the output can be interpreted as p(a|s). All networks are trained using stochastic gradient descent (SGD), a batch size of 1, and a fixed learning rate. The training data consists only of optimal behavior.

We evaluate the causality of the learned policies as c = log p(a = 1|so) − log p(a = 1|sb) where so is an observation with an obstacle and no brake light, and sb is an observation with no obstacle but the brake light on. The optimal causal policy should brake for

the obstacle but not for the brake light so coptimal = ∞. Note that this measure is logarithmic.

The causally optimal policy can be trivially represented with a linear policy. Despite this, we find that training a randomly initialized linear policy using SGD results in no progress towards learning a causal policy. Adding non-linearity is necessary for the SGD optimization to make progress towards a causally correct solution. We find that increasing the number of layers improves the speed of learning a causally optimal solution (Figure 6b).

5.2. Scaled Causality

In practical settings, such as video game playing, it is much more diﬀicult to have an explicit measure of the causal correctness of a model. We therefore focus on one particular pain-point: in action-conditioned models, predictions can be influenced by two sources: the visual input (frame sequence) and the ground-truth action history. While effective decision-making should largely be causally driven by image observationsanalogous to human perception—models often exhibit a “lazy” behavior, relying disproportionately on the statistical priors of the action sequence rather than the visual evidence.

To quantify this tendency, we propose a causality score that measures the model’s reliance on the input frame sequence. A higher score indicates a stronger causal dependence of the model’s output on frame input. We analyze this score across different model sizes and dataset scales as a function of the number of training frames. Our results show that, as in the toy example, larger, deeper models generally achieve higher causality scores, and that causality increases with additional training. Furthermore, increasing the number of unique training frames leads to higher causality scores, except in extremely data-limited regimes.

We compute the causality score by measuring changes in the model’s output distribution using the KL divergence between predictions produced from the original input sequence and from a perturbed version of the same sequence. Formally,

### ∑B

### ∑C

(

)

score(f) =

f(ob,c,ab,c)||f(˜ob,c,ab,c)

KL

,

c=1

b=1

where f denotes the policy model, o is the original image sequence, a is the original action sequence, o˜ is the perturbed image sequence, B is the batch size, and C is the number of evenly sized temporal chunks.

Given an input sequence, we partition it into C equallength chunks and randomly perturb individual image

[Figure 19]

- Figure 7. Causality score as a function of dataset size and model size. Except in the low-data regime (30M), the causality score generally increases with larger models and larger training datasets.

frames with a probability p. The model output at the end of each chunk is then compared between the original and perturbed inputs. Only image frames are perturbed; the action sequence remains unchanged. To ensure that perturbations remain semantically meaningful rather than degenerate noise, each perturbed frame is replaced with a frame drawn from a different scene within the same game. This is implemented by swapping frames within the batch, preserving gamelevel context while inducing perturbations. We use C = 10,p = 0.5,B = 32 for all evaluations.

For fair comparison across models, we compute causality scores using the checkpoint with the lowest test loss for each model. As shown in Figure 7, causality scores increase with both model size and dataset size, except in the extremely data-limited regime (e.g., 30M frames). The standard deviation of the score is on the order of 10−4 and does not affect the relative ranking of models.

We additionally report causality scores as a function of training steps in Appendix D.7, where we observe a consistent increase throughout training. Notably, causality scores continue to rise even in overfitting regimes, suggesting that both causality scores and test loss should be considered jointly for interpretation.

- 6. Related Work

A large body of prior work studied vision language action (VLA) models for robotic control. Many approaches fine-tune pretrained vision–language models (VLMs) to generate robot actions (Driess et al., 2025; Kim et al., 2024; Pertsch et al., 2025; Intelligence et al., 2025; Zitkovich et al., 2023; Wang et al., 2025a; Chen et al., 2025; Zhou et al., 2025; Shukor et al., 2025; Bjorck et al., 2025), while others train VLA models from scratch using robot-centric video and instruc-

tion data (Brohan et al., 2022; Zheng et al., 2025). These methods have demonstrated strong performance in both simulated and real-world robotic tasks.

A parallel line of work focuses on learning action policies in video game environments. Many studies trained agents within a single environment or testbed, such as Minecraft (Li et al., 2025; Hafner et al., 2025; Fan et al., 2022; Baker et al., 2022; Lifshitz et al., 2023; Wang et al., 2024) or Counter-Strike: Global Offensive (Pearce & Zhu, 2022; Durst et al., 2024) and Genshin Impact (Tan et al., 2025). Other approaches leverage large language models (LLMs) or vision language models (VLMs) to guide or improve gameplay through planning, code generation, or tool use (Yang et al., 2024; Li et al., 2025; Wang et al., 2023; Ma et al., 2023).

In multi-game settings, Reed et al. (2022) demonstrated a generalist agent capable of playing a variety of Atari games, while Wiedemer et al. (2025) leveraged VLMs for zero-shot understanding and control. More recent works train general action models across diverse video game environments (Raad et al., 2024; Bolton et al., 2025; Wang et al., 2025b). However, these systems do not release code or datasets, making reproduction and further research challenging for the broader community.

A contemporaneous piece of work (Magne et al., 2025) also explored training a single model across multiple video games, primarily focusing on controller-based console games (e.g., Xbox). In contrast, our work targets keyboard–mouse based PC games, which involve different interaction modalities and higher-frequency real-time control, making our setting complementary to prior efforts.

7. Conclusion

In this work, we present a large-scale, high-quality video dataset annotated with both actions and text, and introduce a real-time policy model designed to leverage this data for online gameplay in real-world environments. We discuss key technical considerations for effectively training such models and evaluate their performance through both qualitative and quantitative analyses.

Furthermore, we study how model performance scales with dataset size and model capacity, examining both scaling laws and causal behavior. Our results show that larger models achieve lower test loss and higher causality scores in data-abundant regimes. This suggests that one approach to issues of causality in behavior cloning may be simply scaling both model size and

dataset size and diversity.

References

Agarwal, N., Ali, A., Bala, M., Balaji, Y., Barker, E., Cai, T., Chattopadhyay, P., Chen, Y., Cui, Y., Ding, Y., et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

Bain, M. and Sammut, C. A framework for behavioural cloning. In Furukawa, K., Michie, D., and Muggleton, S. (eds.), Machine Intelligence 15. Oxford University Press, 1995.

Baker, B., Akkaya, I., Zhokov, P., Huizinga, J., Tang, J., Ecoffet, A., Houghton, B., Sampedro, R., and Clune, J. Video pretraining (vpt): Learning to act by watching unlabeled online videos. Advances in Neural Information Processing Systems, 35:24639– 24654, 2022.

Berner, C., Brockman, G., Chan, B., Cheung, V., Dębiak, P., Dennison, C., Farhi, D., Fischer, Q., Hashme, S., Hesse, C., et al. Dota 2 with large scale deep reinforcement learning. arXiv preprint arXiv:1912.06680, 2019.

Driess, D., Springenberg, J. T., Ichter, B., Yu, L., LiBell, A., Pertsch, K., Ren, A. Z., Walke, H., Vuong, Q., Shi, L. X., et al. Knowledge insulating visionlanguage-action models: Train fast, run fast, generalize better. arXiv preprint arXiv:2505.23705, 2025.

Durst, D., Xie, F., Sarukkai, V., Shacklett, B., Frosio, I., Tessler, C., Kim, J., Taylor, C., Bernstein, G., Choudhury, S., et al. Learning to move like professional counter-strike players. In Computer Graphics Forum, volume 43, pp. e15173. Wiley Online Library, 2024.

Fan, L., Wang, G., Jiang, Y., Mandlekar, A., Yang, Y., Zhu, H., Tang, A., Huang, D.-A., Zhu, Y., and Anandkumar, A. Minedojo: Building open-ended embodied agents with internet-scale knowledge. Advances in Neural Information Processing Systems, 35:18343–18362, 2022.

Hafner, D., Yan, W., and Lillicrap, T. Training agents inside of scalable world models. arXiv preprint arXiv:2509.24527, 2025.

Holfeld, J. On the relevance of the godot engine in the indie game development industry. arXiv preprint arXiv:2401.01909, 2023.

Bjorck, J., Castañeda, F., Cherniadev, N., Da, X., Ding, R., Fan, L., Fang, Y., Fox, D., Hu, F., Huang, S., et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

Intelligence, P., Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., et al. π0. 5: A vision-language-action model with open-world generalization. arxiv 2025. arXiv preprint arXiv:2504.16054, 2025.

Bolton, A., Lerchner, A., Cordell, A., Moufarek, A., Bolt, A., Lampinen, A., Mitenkova, A., Hallingstad, A. O., Vujatovic, B., Li, B., et al. Sima 2: A generalist embodied agent for virtual worlds. arXiv preprint arXiv:2512.04797, 2025.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

Chen, P., Bu, P., Wang, Y., Wang, X., Wang, Z., Guo, J., Zhao, Y., Zhu, Q., Song, J., Yang, S., et al. Combatvla: An eﬀicient vision-language-action model for combat tasks in 3d action role-playing games. arXiv preprint arXiv:2503.09527, 2025.

De Haan, P., Jayaraman, D., and Levine, S. Causal confusion in imitation learning. Advances in neural information processing systems, 32, 2019.

Kim, K., Kotha, S., Liang, P., and Hashimoto, T. Pre-training under infinite compute. arXiv preprint arXiv:2509.14786, 2025.

Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al. Openvla: An opensource vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

Li, Z., Xie, Y., Shao, R., Chen, G., Jiang, D., and Nie, L. Optimus-2: Multimodal minecraft agent with goal-observation-action conditioned policy. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 9039–9049, 2025.

Lifshitz, S., Paster, K., Chan, H., Ba, J., and McIlraith, S. Steve-1: A generative model for text-tobehavior in minecraft. Advances in Neural Information Processing Systems, 36:69900–69929, 2023.

Luo, Z., Shi, F., Ge, Y., Yang, Y., Wang, L., and Shan, Y. Open-magvit2: An open-source project toward democratizing auto-regressive visual generation. arXiv preprint arXiv:2409.04410, 2024.

Ma, Y. J., Liang, W., Wang, G., Huang, D.-A., Bastani, O., Jayaraman, D., Zhu, Y., Fan, L., and Anandkumar, A. Eureka: Human-level reward design via coding large language models. arXiv preprint arXiv:2310.12931, 2023.

Magne, L., Awadalla, A., Wang, G., Xu, Y., Belofsky, J., Hu, F., Kim, J., Schmidt, L., Gkioxari, G., Kautz, J., Yue, Y., Choi, Y., Zhu, Y., and Fan, L. J. Nitrogen: An open foundation model for generalist gaming agents. Preprint, NVIDIA / MineDojo, December 2025. URL https://nitrogen.minedojo.org/ assets/documents/nitrogen.pdf.

Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., et al. Mixed precision training. arXiv preprint arXiv:1710.03740, 2017.

Muennighoff, N., Rush, A. M., Barak, B., Scao, T. L., Piktus, A., Tazi, N., Pyysalo, S., Wolf, T., and Raffel, C. Scaling data-constrained language models. Journal of Machine Learning Research, 26(53):1– 66, 2025. URL http://jmlr.org/papers/v26/24-1000. html.

Nvidia, J. B., Castaneda, F., Cherniadev, N., Da, X., Ding, R., Fan, L., Fang, Y., Fox, D., Hu, F., Huang, S., et al. Gr00t n1: An open foundation model for generalist humanoid robots. ArXiv, abs/2503.14734, 2025.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Park, D., Kim, M., Choi, B., Kim, J., Lee, K., Lee, J., Park, I., Lee, B.-U., Hwang, J., Ahn, J., et al. Orak: A foundational benchmark for training and evaluating llm agents on diverse video games. arXiv preprint arXiv:2506.03610, 2025.

Pearce, T. and Zhu, J. Counter-strike deathmatch with large-scale behavioural cloning. In 2022 IEEE Conference on Games (CoG), pp. 104–111. IEEE, 2022.

Pearce, T., Rashid, T., Bignell, D., Georgescu, R., Devlin, S., and Hofmann, K. Scaling laws for pretraining agents and world models. arXiv preprint arXiv:2411.04434, 2024.

Pertsch, K., Stachowicz, K., Ichter, B., Driess, D., Nair, S., Vuong, Q., Mees, O., Finn, C., and Levine, S. Fast: Eﬀicient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

Pomerleau, D. A. Alvinn: An autonomous land vehicle in a neural network. In Advances in Neural Information Processing Systems, volume 1. Morgan Kaufmann, 1989.

Raad, M. A., Ahuja, A., Barros, C., Besse, F., Bolt, A., Bolton, A., Brownfield, B., Buttimore, G., Cant, M., Chakera, S., et al. Scaling instructable agents across many simulated worlds. arXiv preprint arXiv:2404.10179, 2024.

Reed, S., Zolna, K., Parisotto, E., Colmenarejo, S. G., Novikov, A., Barth-Maron, G., Gimenez, M., Sulsky, Y., Kay, J., Springenberg, J. T., et al. A generalist agent. arXiv preprint arXiv:2205.06175, 2022.

Ross, S., Gordon, G., and Bagnell, D. A reduction of imitation learning and structured prediction to noregret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pp. 627–635. JMLR Workshop and Conference Proceedings, 2011.

Rybakov, O., Chrzanowski, M., Dykas, P., Xue, J., and Lanir, B. Methods of improving llm training stability. arXiv preprint arXiv:2410.16682, 2024.

Schmidt, D. and Jiang, M. Learning to act without actions. arXiv preprint arXiv:2312.10812, 2023.

Shi, F., Luo, Z., Ge, Y., Yang, Y., Shan, Y., and Wang, L. Scalable image tokenization with index backpropagation quantization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16037–16046, 2025.

Shukor, M., Aubakirova, D., Capuano, F., Kooijmans, P., Palma, S., Zouitine, A., Aractingi, M., Pascal, C., Russi, M., Marafioti, A., et al. Smolvla: A visionlanguage-action model for affordable and eﬀicient robotics. arXiv preprint arXiv:2506.01844, 2025.

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Tan, M. and Le, Q. Eﬀicientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pp. 6105– 6114. PMLR, 2019.

Tan, W., Li, X., Fang, Y., Yao, H., Yan, S., Luo, H., Ao, T., Li, H., Ren, H., Yi, B., et al. Lumine: An open recipe for building generalist agents in 3d open worlds. arXiv preprint arXiv:2511.08892, 2025.

Tomilin, T., Fang, M., Zhang, Y., and Pechenizkiy, M. Coom: A game benchmark for continual reinforcement learning. Advances in Neural Information Processing Systems, 36:67794–67832, 2023.

Turing, A. M. Digital computers applied to games. Faster than thought, 1953.

Vera, H. S., Dua, S., Zhang, B., Salz, D., Mullins, R., Panyam, S. R., Smoot, S., Naim, I., Zou, J., Chen, F., et al. Embeddinggemma: Powerful and lightweight text representations. arXiv preprint arXiv:2509.20354, 2025.

Vinyals, O., Babuschkin, I., Czarnecki, W. M., Mathieu, M., Dudzik, A., Chung, J., Choi, D. H., Powell, R., Ewalds, T., Georgiev, P., et al. Grandmaster level in starcraft ii using multi-agent reinforcement learning. nature, 575(7782):350–354, 2019.

Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., and Anandkumar, A. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

- Wang, Y., Ding, P., Li, L., Cui, C., Ge, Z., Tong, X., Song, W., Zhao, H., Zhao, W., Hou, P., et al. Vla-adapter: An effective paradigm for tinyscale vision-language-action model. arXiv preprint arXiv:2509.09372, 2025a.
- Wang, Z., Cai, S., Mu, Z., Lin, H., Zhang, C., Liu, X., Li, Q., Liu, A., Ma, X. S., and Liang, Y. Omnijarvis: Unified vision-language-action tokenization enables open-world instruction following agents. Advances in Neural Information Processing Systems, 37:73278–73308, 2024.

Wang, Z., Li, X., Ye, Y., Fang, J., Wang, H., Liu, L., Liang, S., Lu, J., Wu, Z., Feng, J., et al. Gametars: Pretrained foundation models for scalable generalist multimodal game agents. arXiv preprint arXiv:2510.23691, 2025b.

Wen, C., Lin, J., Darrell, T., Jayaraman, D., and Gao, Y. Fighting copycat agents in behavioral cloning from observation histories. Advances in Neural Information Processing Systems, 33:2564–2575, 2020.

Wiedemer, T., Li, Y., Vicol, P., Gu, S. S., Matarese, N., Swersky, K., Kim, B., Jaini, P., and Geirhos, R. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.

Xu, S., Deng, W., Zhou, Y., and Zhong, F. Probe by gaming: A game-based benchmark for assessing conceptual knowledge in llms. arXiv preprint arXiv:2505.17512, 2025.

Yang, J., Dong, Y., Liu, S., Li, B., Wang, Z., Tan, H., Jiang, C., Kang, J., Zhang, Y., Zhou, K., et al. Octopus: Embodied vision-language programmer from environmental feedback. In European conference on computer vision, pp. 20–38. Springer, 2024.

Ye, S., Jang, J., Jeon, B., Joo, S., Yang, J., Peng, B., Mandlekar, A., Tan, R., Chao, Y.-W., Lin, B. Y., et al. Latent action pretraining from videos. arXiv preprint arXiv:2410.11758, 2024.

Yu, L., Lezama, J., Gundavarapu, N. B., Versari, L., Sohn, K., Minnen, D., Cheng, Y., Birodkar, V., Gupta, A., Gu, X., et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.

Yue, Y., Green, C., Hunt, S., Salia, I., Shi, W., and Hunt, J. J. Pixels to play: A foundation model for 3d gameplay. In 2025 IEEE Conference on Games (CoG), pp. 1–4. IEEE, 2025.

Zhang, A. L., Griﬀiths, T. L., Narasimhan, K. R., and Press, O. Videogamebench: Can vision-language models complete popular video games? arXiv preprint arXiv:2505.18134, 2025.

Zheng, R., Wang, J., Reed, S., Bjorck, J., Fang, Y., Hu, F., Jang, J., Kundalia, K., Lin, Z., Magne, L., et al. Flare: Robot learning with implicit world modeling. arXiv preprint arXiv:2505.15659, 2025.

Zhou, Z., Zhu, Y., Zhu, M., Wen, J., Liu, N., Xu, Z., Meng, W., Peng, Y., Shen, C., Feng, F., et al. Chatvla: Unified multimodal understanding and robot control with vision-language-action model. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 5377– 5395, 2025.

Zitkovich, B., Yu, T., Xu, S., Xu, P., Xiao, T., Xia, F., Wu, J., Wohlhart, P., Welker, S., Wahid, A., et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pp. 2165–2183. PMLR, 2023.

- A. Dataset

- A.1. Game List We list all the games we collected in the annotated dataset:

- • Roblox: Blade Ball
- • Roblox: Death Ball
- • Roblox: Be a Shark
- • Roblox: Be a Snake
- • Roblox: Be a Tornado
- • Roblox: Natural Disaster Survival
- • Roblox: Rivals
- • Roblox: Slap Battle
- • Roblox: A Dusty Trip
- • Roblox: Hypershot
- • Roblox: Evade
- • Roblox: Murderers vs. Sheriffs

- • Msdos: Quake
- • Msdos: Need for Speed
- • Msdos: DOOM
- • Msdos: DOOM II
- • DOOM Eternal
- • Eval: Hovercraft
- • Eval: Simple FPS
- • Left 4 Dead 2
- • Call of Duty Mobile
- • Call of Duty: Black Ops

- II Zombies

• Call of Duty: Black Ops

- III Zombies

- • Goat Simulator

- • Goat Simulator 3
- • Helldivers 2
- • Need for Speed: Hot Pursuit
- • Need for Speed: Most Wanted
- • Need for Speed: Carbon
- • Need for Speed: Underground 2
- • Fortnite
- • House Flipper
- • House Flipper 2
- • PowerWash
- • Euro Truck Simulator 2

- • Warhammer: Vermintide
- • Warhammer: Vermintide 2
- • Saints Row: The Third
- • Saints Row IV
- • Resident Evil 5: Mercenaries
- • Resident Evil Revelations 2: Raid
- • Grand Theft Auto V
- • Grand Theft Auto: San Andreas
- • Minecraft

- A.2. Labeled Data The labeled data distribution regarding game types can be found at Figure 8

- Figure 8. Distribution of games in the annotated dataset.
- A.3. Quality filter

Each recorded video undergoes a two-stage quality assurance process. First, we apply automated filtering based on the following criteria: (1) no single key is held for more than 60% of the total frames to avoid hardware or recording artifacts; (2) no more than six keys are pressed simultaneously at any time; (3) a minimum level of interaction is maintained, ensuring that actions are taken throughout the sequence; (4) a lightweight action policy model (described in Section 3) is evaluated on the full trajectory, and videos with likelihoods below a predefined threshold are flagged.

All videos flagged by these checks are reviewed by human inspectors. In addition, we randomly sampled videos for manual inspection using an adaptive strategy based on each annotator’s prior contributions to a given game: annotators with fewer accepted, recorded hours for that game were sampled with a higher probability, while those with more extensive prior submissions were sampled less frequently.

In this approach, we ensured that the recorded videos were of high quality, expert gameplay.

- A.4. Text Annotation Here we show the prompt we used for commercial VLM to do text annotations The prompt used for video text annotation is You are an expert gaming tutor and video analyst.￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿

- I. MACRO INSTRUCTIONS￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿

- A. Definition

- A MACRO INSTRUCTION is an event that is explicitly listed in the ”GAME-FOCUS RULE” table (Section II) or selects the ’players next objective or area and passes **all three** checks:

- 1. Changes the long-term goal or destination ￿( 1 new room, zone, target, phase, or tactic).

- 2. Without it the player would wander or be unsure what to pursue next.

- 3. When applicable, cites a **distinct visual landmark** beginners can recognise (door colour, sign, glowing platform, giant tree, scoreboard icon, billboard, unique prey, ball aura, etc.).

- B. Start & finish timestamps

- • ‘”start”‘ = first frame where the objective becomes clear.

- • ‘”end”‘ = frame where the objective is satisfied; use ‘null‘ if unfinished when the video ends.￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿

- II. GAME-FOCUS RULE￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿ When deciding whether an event is *strategic enough* to log as a macro, follow the table below. Skip events that ’dont match the focus for that game family. | Game family | Focused macro events (examples) | |--------------------------------|---------------------------------------------------------------------------------------------------| | **Growth-Arena** (Roblox *Be a Snake*, *Be a Shark*, *Be a Tornado*) | • Switching strategy to avoid larger

snake/shark/tornado or ambush a smaller snake/shark/tornado (”**Eat the small snake**; **avoid the big shark

**”). Use the numbers on top of each snake/shark/tornado’s head to indicate the size. • Activating or chasing a power-up landmark “(Dash to the **gold lightning-bolt boost”**). |

| **FPS** (*Doom*, *Doom2*, *Quake*, *Call of Duty Series*, *Left 4 dead*, etc) | • Reaching or unlocking key areas “(Collect the **red skull key** from the ”altar, ”press switch on the wall to open the door”). • Navigating path choices “(Enter the **bronze-framed doorway with a skull carving**, **move to the entry with red mark ”**”). • Securing major gear required to finish the level “(Take the **rocket launcher on the blood altar”**). • Use switch to open the door “(Press the switch on the wall to open the ”door). • Switch weapons or obtain weapons (”Use axe”, ”Use gun”). • Pay attention to what the weapon the player is holding, if it was switched then note it down (”switch to gun”)|

| **Arena Ball / Dodgeball** (Roblox *Blade Ball*, *Death Ball*, etc) | • Choosing a safe position or platform “( Jump onto the **neon cube just outside the danger zone”**). • Engaging the ball intentionally “(Charge the ** glowing ball with a spin-slash”**). • Triggering or acquiring a special ability landmark “(Grab the **purple deflect power-up circle”**). |

| **Racing** (*Need for Speed*, etc.) | • Major route decision or shortcut “(Drift through the **billboard shortcut into the alley”**). • Scheduled pit stop or repair landmark “(Pit in at the **blue-neon service lane”**). • Switching race tactics “(Switch to a **draft-and-slingshot strategy behind the lead car”**). |

| **Survival** (*Natural Disaster Survival*, etc.) | • Major route decision to escape or survive “(Run to the ** elevator shaft”**).|

If a game ’isnt listed, map it to the closest family.￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿

- III. GAME-IGNORE RULE￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿ When deciding whether an event is **NOT** *strategic enough* to log as a macro, follow the table below.

| Game family | SKIP macro events (examples) | |--------------------------------|---------------------------------------------------------------------------------------------------| | **Growth-Arena** (Roblox *Be a Snake*, *Be a Shark*, *Be a Tornado*) | • Eating small prey from the map (e.g.,

”Eat the small fog”) | | **Survival** (*Natural Disaster Survival*, etc.) | • IGNORE the DISASTER WARNING caption.|

If a game ’isnt listed, map it to the closest family.￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿

- IV. EVENT-FREQUENCY RULES￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿

- A. Global cool-down ￿ 8 s between macros (unless a genuine strategy shift appears sooner).

- B. Continuous-action filter (Growth-Arena, racers)

- • Record a macro only when the agent changes *overall tactic* or *target landmark* (giant pellet, boss prey, pit …stop).

- • Ignore repetitive minor events.￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿

- V. DIVERSITY RULE￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿ After accuracy is guaranteed, vary verbs, landmark phrases, sentence forms. Try to avoid using the same instruction, use diverse description to describe the same event. Never invent events just to add —varietyaccuracy outranks diversity.￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿

- VI. AVOID RULE￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿ Avoid using the same instruction to describe the same event, try to be as diverse in wording as possible. Avoid using directions in the instruction, such as turn right to xxx, turn left to xxx, etc.￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿

- VII. ON-SCREEN TEXT RULE￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿ Capture any explicit text hint/objective that appears on screen. Quote it exactly once and log it as a macro if it meets the four checks.￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿

- VIII. WORKFLOW￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿

- 1. Watch the video once; identify the game and its win condition(s).

- 2. **Narrative** – third-person past tense, focus on strategy and playstyle.

- 3. **Macro list** – array of objects: { ”start”: ”MM:SS”, ”end”: ”MM:SS | null”, ”instruction”: ”<imperative sentence>” }￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿

- IX. OUTPUT (JSON only – no markdown)￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿￿

{ ”narrative”: ”<string>”, ”macro_instructions”: [

{

”start”: ”00:12”, ”end”: ”00:25”, ”instruction”: ”Enter the bronze-framed doorway with a skull carving”

}, {

”start”: ”00:34”, ”end”: ”00:57”, ”instruction”: ”Take the corridor lit by flickering red lights”

}, {

”start”: ”00:01”, ”end”: ”00:02”, ”instruction”: ”Switch to axe”

}, {

”start”: ”00:52”,

”end”: ”00:54”, ”instruction”: ”Run through without engaging with enemy”

}, {

”start”: ”01:01”, ”end”: ”01:02”, ”instruction”: ”Change the weapon from gun to fist”

},

] }

- A.5. Simple Environment We constructed some simple games for automated testing with Godot.

[Figure 20]

[Figure 21]

(a) Hovercraft environment: the car is looping in a fixed racing road. We evaluate the model by measuring how much time it takes to finish one loop

(b) Simple-FPS environment: a simple fps in a static map. We evaluate the model by counting the number of hit enemies minus the hits taken.

- B. Unlabeled dataset

- B.1. Unlabeled data distribution

In addition to annotated gameplay data, we curate a large corpus of unlabeled gameplay videos from public sources (Fan et al., 2022; Baker et al., 2022). While such data are abundant, they present several challenges: highly variable quality, interleaving with non-game content, different resolutions and frame rates, and the absence of corresponding action labels.

We curate unlabeled gameplay trajectories using a two-stage filtering pipeline based on commercial VLMs. In the first stage, videos are filtered using metadata signals such as titles, descriptions, topics, and thumbnails (when available). A VLM is queried to assess relevance to a predefined set of game-related queries. In the second stage, the full video content is processed to segment and remove non-gameplay intervals. To balance cost and filtering accuracy, this step is performed on temporally downsampled video.

We run queries covering a broad range of popular game titles to collect diverse gameplay footage. The distribution of video hours in the resulting unlabeled dataset is shown in Figure 10.

Note that we did not release this dataset as we do not hold the copyright.

Figure 10. Distribution of games in the unlabeled dataset.

- C. Policy Model

- C.1. Image Tokenizer

- Figure 11 compares training and validation perplexity when the image tokenizer is either frozen or trained jointly with the policy model. Jointly training the image tokenizer consistently yields lower perplexity on both training and validation sets, indicating that adapting the visual representation is critical for achieving strong downstream policy performance.

[Figure 22]

[Figure 23]

(a) Training perplexity with frozen vs. unfrozen tokenizer

(b) Validation perplexity with frozen vs. unfrozen tokenizer

- Figure 11. Effect of training the image tokenizer jointly with the policy model. As might expected, training the image tokenizer during model training significantly reduce the training and validation perplexity.

- D. Evaluation

- D.1. Training details All models were trained using automatic mixed-precision training (Micikevicius et al., 2017), with model parameters stored in float32 and activations in bfloat16, except for RMSNorm layers, which were computed in float32. Common techniques such as z-loss and K,Q norm (Rybakov et al., 2024) were applied to stabilize the training. Training was performed on 8×NVIDIA H100 GPUs. All inference experiments are conducted on a single NVIDIA RTX 5090 GPU; an additional NVIDIA RTX 5080 GPU was used for game rendering to ensure inference ran at a consistent speed. Unless otherwise specified, all experiments share the same set of hyperparameters, which are reported in Appendix D.3. We apply data augmentation during training, which we find to be indispensable for mitigating the training–inference gap.
- D.2. Camera setup We list the in-game camera setup below:

- • Roblox: Camera sensitivity set to 0.52.
- • Quake: Mouse sensitivity set to 3.5; smoothing disabled.
- • DOOM: Smoothing set to 2×; look sensitivity 22%, move sensitivity 22%.

- D.3. Hyperparameters

Table 2 summarizes the hyperparameters used for training the policy model. Batch sizes are chosen to maximize GPU utilization for each model scale. We perform a limited sweep over learning rates {1×10−4,3×10−4,3×10−5} on the full dataset and select the best-performing value for each model size, which is then fixed for all remaining experiments. We also list the hyperparameter for data augmentations at Table 3.

- D.4. Human Evaluation

We report the raw human evaluation scores for DOOM and Quake checkpoints in Table 4. For each model, three checkpoints are evaluated, and each checkpoint is run three times, resulting in 18 total runs.

Human evaluators assessed model quality by counting the occurrences of the following issues during gameplay: (1) colliding with walls (Wall); (2) shooting into the air (Shoot Air); (3) missing targets, including items or enemies (Miss); (4) non–human-like behavior, such as repeating loops or moving backward (Non-human Move); (5) remaining idle (Idle); and (6) camera shaking or jitter (Jitter). Counts were normalized by video length and multiplied by 100 to obtain the final score, with lower values indicating better performance.

Parameter Value Batch size 5 (1.2B), 8 (600M), 8 (300M), 10 (150M) History length 200 Frame resolution 192 × 192 Text tokenizer EmbeddingGemma (Vera et al., 2025) Image tokenizer EﬀicientNet-B0 Number of image tokens 1 Transformer Backbone Number of layers 10 (150M), 20 (300M), 9 (600M), 22 (1.2B) Hidden dimension 1024 (150M, 300M), 2048 (600M, 1.2B) Query heads 16 Key–value heads 16 Action Decoder

Number of layers 3 Query heads 8 Key–value heads 8 Hidden dimension Same as transformer hidden dimension

Optimizer (AdamW) Learning rate 1 × 10−4 (150M, 300M, 600M), 3 × 10−5 (1.2B) Weight decay 1 × 10−4

- β1 0.9
- β2 0.999 Table 2. Architecture and training hyperparameters across model scales.

Table 3. Data augmentation parameters.

Augmentation p Parameters Spatial transform 1.0 Rotation angle ∼ U(−3◦,+3◦) Color perturbation 0.25 Brightness, contrast, saturation, hue shifts ∼

U(0.0,0.2) Planckian jitter 0.25 Illumination preset ∈ {6,12,18,24} ISO noise 0.1 Color shift ∼ U(0.01,0.2); intensity ∼ U(0.1,0.6) Random blur 0.2 Gaussian: k ∈ [3,7],σ ∈ [0.1,2.0]; or Motion: k ∈

[3,7], angle ∈ [0◦,360◦], direction ∈ [−1,1] (Gaussian w.p. 0.5)

Sharpening 0.15 Sharpness factor ∼ U(0.5,1.5) Translation 0.25 Horizontal shift up to 3% of image width

Tasks are denoted by the capital letter of the game, followed by the checkpoint number and repetition index (e.g., D1-1 denotes the first run of the first checkpoint in DOOM). Note that Be a Shark and Hypershot are not included in this rubric, as their online environments are dynamic and cannot be reliably controlled across runs.

Task Model Wall Shoot Air Miss Non-human Move Idle Jitter Duration Score

- D1-1

1.2B 1 - - - - - 28 3.57 600M 1 - - - - - 24 4.17 300M - - - 4 3 - 102 6.86 150M 3 - 1 1 - - 109 4.59

- D1-2

1.2B 1 1 - - - - 54 3.70 600M - 1 - 1 1 - 74 4.05 300M - - 1 - - - 30 3.33 150M 2 - 2 - - 1 91 5.49

1.2B 2 - - 1 - - 92 3.26 600M 3 - 1 - - - 59 6.78 300M - - - - 1 1 62 3.23 150M - 1 1 - - - 38 5.26

- D1-3

- D2-1

1.2B 1 - - - - - 29 3.45 600M - - - 1 1 - 50 4.00 300M 3 1 1 - 1 1 83 8.43 150M - - 1 - - - 28 3.57

- D2-2

1.2B 2 - - - - - 28 7.14 600M - 1 - - - - 28 3.57 300M 3 1 - - - - 84 4.76

- 150M 1 - - - - - 21 4.76

D2-3

1.2B 6 - - 1 - - 128 5.47 600M 2 - - - - - 58 3.45 300M 1 1 - 1 - - 84 3.57

- 150M 2 - 1 1 - - 32 12.50

- D3-1

1.2B - - 2 2 1 - 54 9.26 600M 3 1 - 3 - 1 125 6.40 300M 5 1 - 1 1 - 125 6.40 150M 3 1 1 3 1 - 118 7.63

- D3-2

1.2B 1 1 - 3 - - 102 4.90 600M 1 - - 1 - 1 45 6.67 300M 1 - 1 2 1 - 95 5.26 150M 3 - 1 - - 2 96 6.25

- D3-3

1.2B 2 - - - - - 54 3.70 600M 4 1 1 3 - - 140 6.43 300M 3 - - 2 1 - 128 4.69 150M 2 - 1 - 1 - 66 6.06

- Q1-1

1.2B 1 - 1 - - - 41 4.88 600M - 2 - - - 1 58 5.17 300M 4 - - 3 - - 60 11.67 150M - - - 1 - - 27 3.70

- Q1-2

1.2B - 1 - - - - 34 2.94 600M 2 - - - - - 33 6.06 300M - 1 - - - - 25 4.00 150M 1 2 - 3 - - 85 7.06

- Q1-3

1.2B 1 1 - - - - 43 4.65 600M 1 - - - - - 26 3.85 300M 2 1 - 4 - - 83 8.43 150M 3 1 - 1 - - 56 8.93

1.2B - 1 1 1 - - 57 5.26 600M - 1 1 3 - - 49 10.20 300M 2 - 2 2 - - 66 9.09 150M - - 1 - - - 27 3.70

- Q2-1

- Q2-2

1.2B 1 1 3 - - - 59 8.47 600M - 1 1 - - - 32 6.25 300M 1 1 - 2 - - 42 9.52 150M - 3 - 2 - - 42 11.90

- Q2-3

1.2B 1 - - - - - 51 1.96 600M - - - - - - 21 0.00 300M 1 - - 1 - - 33 6.06 150M - - - 3 - - 32 9.38

- Q3-1

1.2B - - - - - - 18 0.00 600M - - - - - - 13 0.00 300M - - 1 - - - 16 6.25 150M 2 - - 1 - - 42 7.14

- Q3-2

1.2B - - - - - - 10 0.00 600M 1 - - - - - 23 4.35 300M - - 1 2 - - 30 10.00 150M - 1 - - - - 22 4.55

- Q3-3

1.2B - - - 1 - - 26 3.85 600M - 1 - - - - 24 4.17 300M 1 - - - - - 26 3.85 150M 1 - - - - - 13 7.69

- D.5. Text Instruction Checkpoints

Figure 12 shows the checkpoints used for evaluating text-conditioned behavior in Quake. In the maze, there are three red buttons and the player needs to press all of them to open the door.

[Figure 24]

Figure 12. Selected frames from the Quake maze. Three buttons appear along the path, all of which must be pressed to open the door.

- D.6. Scaling Laws

- Figure 13 presents scaling-law fits for all four model sizes (150M, 300M, 600M, and 1.2B). Across all configurations, test loss closely follows a power-law relationship with respect to the number of training frames. Larger models consistently achieve lower test loss across dataset sizes when dataset size is relatively large.

[Figure 25]

- Figure 13. Scaling-law curves relating test loss to the number of training frames. All four models are fitted a power-law curve between the data size and test loss, the data exhibits a strong fit to the power-law curve.
- Figure 14 further breaks down test loss as a function of training frames across different dataset fractions and model sizes.

[Figure 26]

[Figure 27]

(a) 100% of the dataset (b) 50% of the dataset

[Figure 28]

[Figure 29]

(c) 25% of the dataset (d) 12% of the dataset

[Figure 30]

(e) 6% of the dataset

- Figure 14. Test loss as a function of training frames across dataset sizes and model scales. The larger model can leverage the data better when the dataset size is larger, and in general the larger models achieve lower test loss when trained with similar number of frames.

- D.7. Causality Analysis

Figure 15 shows causality scores as a function of training steps for different model and dataset sizes. Because causality scores consistently increase over the course of training, the 600M model trained on the 30M dataset can exhibit a higher causality score than the same model trained on the 500M dataset at certain training steps. However, when we instead select model checkpoints based on the lowest test loss and then evaluate causality, the resulting trends align with those discussed in Section 5.1.

0 1 2 3 4 5 6 7 8

Number of Frames 1e9

0.006

0.008

0.010

0.012

0.014

Score

Model / Dataset Size

Model:150M, Data:500M Model:600M, Data:500M Model:1.2B, Data:500M

Model:1.2B, Data:30M

Model:600M, Data:30M Model:150M, Data:30M

Model:150M, Data:250M Model:600M, Data:250M Model:1.2B, Data:250M

Figure 15. Causality scores as a function of training steps for different model and dataset sizes. Because causality scores generally increase during training, a 600M model trained on 30M samples can exhibit higher causality than the same model trained on 500M samples at intermediate checkpoints. When selecting checkpoints based on lowest test loss, however, the resulting causality trends are consistent with those reported in Section 5.1.

- D.8. Pretraining with Unlabeled Data

Unlabeled gameplay videos are orders of magnitude more abundant than human annotated demonstrations. We therefore leverage an inverse dynamics model (IDM) to convert unlabeled videos into additional training data.

- D.8.1. Inverse Dynamics Model

Two classes of IDMs have been explored in prior work: real-action models that directly predict keyboard and mouse actions (Baker et al., 2022), and latent-action models that infer abstract action codes subsequently mapped to real actions (Schmidt & Jiang, 2023; Ye et al., 2024; Nvidia et al., 2025). For simplicity, we adopt the realaction formulation.

Formally, the IDM predicts the action at time t conditioned on the surrounding image sequence:

(

)

a˜t ∼ pIDM

at | o1,o2,...,oT

.

The IDM shares the same architecture as the policy model (Figure 2a), with two key differences. First, it does not condition on text or ground-truth action tokens, since environment dynamics are independent of text inputs

and the IDM must predict all action labels in a single forward pass rather than autoregressively. Second, the IDM is non-causal and is allowed to attend to future frames, which improves action imputation accuracy. This architectural alignment ensures that improvements to the policy model transfer directly to the IDM.

The IDM was trained using cross-entropy loss on the labeled dataset. To ensure robustness to the diversity of unlabeled data, we applied extensive data augmentation during training. Once trained, the IDM was used to impute actions for the unlabeled gameplay dataset.

- D.8.2. Evaluation of pretrained model

We trained a 600M model that leverages unlabeled data to study the benefits of large-scale pretraining. We refer to this model as pretrained-600M, which is obtained through a three-stage procedure: (1) training an inverse dynamics model (IDM) on the full labeled dataset; (2) using the trained IDM to generate pseudo-labels for an unlabeled dataset that is approximately 10× larger than the annotated dataset, followed by pretraining the 600M policy model on this pseudo-labeled data for one epoch; and (3) fine-tuning the model on the full labeled dataset.

Evaluating the quality of the IDM presents a nontrivial challenge. Since ground-truth action labels are only available for the annotated dataset, quantitative evaluation of the IDM can only be performed on this data, on which the IDM is also trained. As a result, standard evaluation metrics tend to overestimate performance and do not fully reflect the IDM’s effectiveness in its intended deployment setting. In practice, the IDM is primarily used to generate pseudo-labels for unlabeled gameplay videos, whose distribution differs substantially from that of the labeled data.

To account for this distributional gap, we complement quantitative evaluation with manual inspection. Specifically, we sample unlabeled videos and assess the plausibility and temporal consistency of the generated pseudolabels using human judgment. This qualitative evaluation provides an additional sanity check on whether the IDM produces reasonable action annotations when applied to out-of-distribution data.

- D.8.3. Simple Programmatic Environment

Similar to the evaluation of other models, we show the evaluation on Godot environment of the 600M pretrained model at Table 5. There is incremental changes in the scores when compared to its 600M counterpart.

Model Size Hovercraft ↓ Simple-FPS ↑ FPS ↑ Pretrained 600M 38 24 61

Table 5. Performance on Godot-based programmatic environments across model sizes for pretrained 600M model. There is only incremental change when comparing with the 600M scores from Table 1.

- D.8.4. Quantitative Eval

Here, we compare the test loss of the pretrained 600M model with that of a 600M model trained solely on labeled data. As shown in Figure 16, the pretrained 600M model achieves substantially lower test loss than the model trained only on labeled data when both are trained on the same number of frames.

[Figure 31]

- Figure 16. Comparison of test loss between pretrained and label-only 600M models. The pretrained 600M model achieves substantially lower test loss than a model trained solely on labeled data when trained on the same number of frames.

- D.8.5. Human Evaluation in Real Environment

We compare gameplay videos generated by the pretrained 600M model and the 600M model trained using labeled data only. Although the pretrained model achieves lower test loss, its online performance is not significantly preferred over the non-pretrained model.

We hypothesize that this gap arises because pretraining incorporates a large amount of unrelated video data, which may introduce atypical or environment-specific movements when deployed in a particular game. Such behaviors can appear less human-like, which is a critical factor in human preference during evaluation.

