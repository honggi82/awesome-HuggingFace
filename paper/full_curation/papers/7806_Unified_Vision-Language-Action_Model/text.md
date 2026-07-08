# Unified Vision-Language-Action Model

##### Yuqi Wang1∗ Xinghang Li2 Wenxuan Wang1,2 Junbo Zhang3 Yingyan Li1 Yuntao Chen4 Xinlong Wang2 Zhaoxiang Zhang1

1 CASIA 2 BAAI 3 THU 4 HKISI Project page: https://robertwyq.github.io/univla.github.io

arXiv:2506.19850v1[cs.CV]24Jun2025

#### Unified Vision-Language-Action Model

[Figure 1]

[Figure 2]

Multi-modal output

Action output

###### CALVIN

[Figure 3]

[Figure 4]

LIBERO

Multi-modal Model

Language Model

[Figure 5]

[Figure 6]

Interleaved Video Static Image

BRIDGE

VIT

Unified Post-train Tasks

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[0.1,-0.1,0.1,0.02,0.01,0.2,0.8]

Instruction: locate the position of the fork

| |
|---|

Instruction: pick sponge from top drawer and place on counter

Instruction: Put spoon on the towel

[Figure 15]

[0.1,0.2,0.1,0.01,0.01,0.0,0.5]

[Figure 16]

[Figure 17]

[x1,y1,x2,y2]

[x, y, z, roll, pitch, yaw, gripper]

Vision Supervision Action Supervision

Text Supervision

Policy Learning

Perception Grounding World Model

- Figure 1: We present UniVLA, a unified vision-language-action model. Unlike prior VLA approaches that typically rely on an extra vision encoder to extract image features and generate only action outputs, UniVLA represents vision, language, and action as discrete tokens within a unified autoregressive framework. This unified modeling paradigm enables multi-modal outputs and supports a wide range of tasks—such as text-supervised perception grounding, vision-supervised world modeling, and action-supervised policy learning—within a single architecture. The unified token-based design further allows UniVLA to effectively leverage large-scale multimodal data, particularly video, for scalable and generalizable learning. UniVLA achieves new state-of-the-art results on CALVIN, LIBERO, and SimplerEnv-Bridge, significantly surpassing existing methods.

### Abstract

Vision-language-action models (VLAs) have garnered significant attention for their potential in advancing robotic manipulation. However, previous approaches predominantly rely on the general comprehension capabilities of vision-language models (VLMs) to generate action signals, often overlooking the rich temporal and causal structure embedded in visual observations. In this paper, we present UniVLA, a unified and native multimodal VLA model that autoregressively models vision, language, and action signals as discrete token sequences. This formulation enables flexible multimodal tasks learning, particularly from large-scale video data. By incorporating world modeling during post-training, UniVLA captures causal dynamics from videos, facilitating effective transfer to downstream policy learning—especially for long-horizon tasks. Our approach sets new state-of-the-art results across several widely used simulation benchmarks, including CALVIN, LIBERO, and Simplenv-Bridge, significantly surpassing previous methods. For

∗Work done during an internship at BAAI. Corresponding author

Preprint. Under review.

example, UniVLA achieves 95.5% average success rate on LIBERO benchmark, surpassing π0-FAST’s 85.5%. We further demonstrate its broad applicability on real-world ALOHA manipulation and autonomous driving.

### 1 Introduction

Developing agents capable of perceiving, reasoning, and acting in the physical world has long been a central objective of artificial intelligence. Recent advances in vision-language-action (VLA) models [8, 56, 42, 6], grounded in the powerful generalization capabilities of vision-language models (VLMs) [57, 38, 3, 66, 30], have demonstrated impressive performance across a wide range of robotic manipulation tasks, and are increasingly being adapted to generalist humanoid robots [5, 20] that demand broader embodied intelligence. However, most existing VLA approaches [42, 6] follow a language-centric paradigm: visual observations are first projected into a semantic space, and action policies are subsequently derived based on these representations. This late-fusion strategy, while beneficial for semantic understanding and generalization, limits the formation of deeply coupled cross-modal representations and impedes the learning of temporal and causal dependencies across the perception-action loop. This raises a central question: Can vision, language, and action be jointly modeled within a unified representation space to facilitate tighter cross-modal integration and more effective policy learning?

While appealing in theory, unified modeling presents two key challenges. First, vision, language, and action are inherently heterogeneous modalities: vision comprises high-dimensional, continuous spatial signals; language conveys abstract, discrete semantics; and actions involve temporally ordered sequences with causal dependencies. Second, the perception-to-action pipeline is inherently dynamic and causal, yet existing VLA models [8, 42, 6] often adopt static, language-centric paradigms that merely learn the mapping from static image to action. These models fail to capture the dynamic nature of real-world interactions, thereby limiting their ability to leverage the rich temporal information from videos for training.

To address the above challenges, we introduce UniVLA, a novel framework for unified vision–language–action learning. As illustrated in Figure 1, we propose a unified framework that supports both multimodal and multi-task learning. At the modality level, vision, language, and action signals are all transformed into discrete tokens and modeled using a shared vocabulary. This unified token representation allows for joint learning across modalities, fostering deeper cross-modal understanding and integration. Building upon the unified framework, we adopt an autoregressive, Markov chain-based sequence modeling approach, where observations and actions are interleaved. This structure naturally incorporates causal dependencies, enabling the model to reason over temporal dynamics rather than treating perception and action as isolated tasks. By integrating the world model paradigm during training, we leverage large-scale robotics videos for self-supervised learning, allowing the model to capture environment dynamics in a temporally consistent and causally grounded manner. Remarkably, we find that post-training with world models significantly enhances policy learning, particularly for long-horizon and out-of-distribution tasks.

Experiments across multiple simulation benchmarks, including CALVIN [54], LIBERO [49], and SimplerEnv [48], demonstrating clear performance improvements over existing methods. Our model incorporates world model learning during post-training, enabling it to effectively capture visual dynamics from large-scale videos. This strategy significantly enhances both data efficiency and training efficiency in downstream policy learning, and allows for rapid adaptation to novel robotic tasks. Beyond policy learning, we demonstrate the model’s multimodal output capabilities, including spatial reasoning and visual prediction, highlighting its versatility. Furthermore, we extend our approach to autonomous driving scenarios for broader applicability. These results underscore the potential of our unified VLA model as an alternative and promising direction for generalist embodied intelligence.

Our contributions are summarized as follows:

- • We propose UniVLA, the first unified vision–language–action (VLA) model that encodes vision, language, and action as discrete tokens within a shared vocabulary, jointly modeling them through autoregressive sequence learning. This approach offers a novel architecture alternative to the existing VLA paradigm, facilitating more integrated cross-modal modeling and enabling large-scale video-based training.

- • Our unified sequence modeling framework supports a broad range of multimodal tasks. By investigating various post-training strategies, we demonstrate that world models can effectively learn temporal dynamics from video data, substantially enhancing performance and improving both data and training efficiency in downstream policy learning—particularly in long-horizon and out-of-distribution scenarios.
- • Our model achieves state-of-the-art performance on several simulated benchmarks (CALVIN, LIBERO, and SimplerEnv-Bridge) and introduces an open-source VLA method supporting large-scale video training. We further explore its capabilities across various modalities, including spatial reasoning and video prediction, and demonstrate its effective transfer to driving scenarios, highlighting its potential for generalist embodied intelligence.

### 2 Related Works

##### 2.1 Vision-Language-Action Models

Recent vision-language-action (VLA) models have demonstrated strong task performance across diverse robots and tasks [8, 64, 22, 42, 76, 12, 6, 78, 50, 41, 37]. These models leverage pre-trained vision-language models (VLMs) to enhance understanding and generalization, further fine-tuned on large-scale robotic datasets for low-level control. Currently, VLA models can be categorized into two paradigms based on their output space: pure action prediction and visual-guided action prediction.

Pure action prediction. Recent efforts have extended vision-language models (VLMs) to incorporate action modalities, enabling direct action prediction from visual and language inputs. A prominent example is RT-2[8], which learns from both internet-scale and robotic data to generate discrete actions autoregressively, showcasing strong generalization and semantic grounding. Building upon this, RT-H[2] introduces hierarchical actions to facilitate data sharing across tasks. OpenVLA[42] scales this paradigm with a 7B-parameter open-source model trained on 970k real-world demonstrations spanning diverse datasets. To enhance spatial reasoning, SpatialVLA[59] integrates spatial representations into the action modeling process. Beyond architecture scaling, new action modeling techniques have also emerged. π0 [6] incorporates flow matching to improve action learning efficiency, while FAST [58] introduces a unified frequency-domain formulation for discretizing actions.

Visual-guided action prediction. These studies leverage the power of visual pretraining, typically based on a policy-as-video formulation, by predicting future visual signals and subsequently decoding them into actions. SuSIE [7] predicts key future frames and derives actions through inverse dynamics. UniPi [24] generates videos from text instructions, extracting actions from the frames. GR series [70, 12, 44] leverages video pretraining for general policy learning. PAD [31] uses diffusion models to simultaneously learn future images and actions. LAPA [73] proposes to learn latent actions between images with VQ-VAE from action-free internet-scale videos. Track2Act [4] extracts point tracks from diverse web videos to guide the learning of interaction plans.

Both approaches have their strengths and weaknesses. The first, focused on action prediction, integrates well with Vision-Language Models (VLMs) but lacks spatial understanding and visual prediction capabilities. The second, incorporating visual generation, requires separating generative and action prediction models, limiting the full potential of VLMs. Our work unifies these approaches, combining video generation pretraining with the strengths of VLMs to propose a native multimodal model with significant future potential.

##### 2.2 World Models for Robotics

World models [32, 33, 43] have gained widespread attention for their ability to capture and reason about the dynamics of the physical world. They have emerged as a cornerstone in a range of domains, including interactive video generation [10, 11], autonomous driving [35, 69, 67, 26], and robotics [24, 71, 72]. Recent advances in robotics increasingly focus on general-purpose controllable video generation to simulate realistic and diverse robot-environment interactions. Visual Foresight [25] leverages action-conditioned video prediction with model-predictive control, enabling robots to plan manipulation tasks by forecasting future visual observations. UniSim [72] builds a “universal simulator” trained on diverse visual datasets, capable of visualizing the effects of both highlevel instructions (e.g., “open the drawer”) and low-level controls in novel scenes. RoboDreamer [80] learns a compositional world model by factorizing video generation, facilitating the synthesis of

𝐿 𝐿 ⋯ 𝐿 𝐿

Text Tokens

###### Pre-train VL-Align

[Figure 18]

## Autoregressive Transformer

Vision Tokens

Post-train World Model

[Figure 19]

𝐿 𝐿 𝐿 ⋯ 𝐿 𝐿

Action Tokens

Fine-tune Policy Learning

[Figure 20]

[Figure 21]

Task

[Figure 22]

[Figure 23]

Pick carrot ⋯

[Figure 24]

Causal Multimodal Sequence

- Figure 2: Overview of the UniVLA framework. Our model unifies information from different modalities into a discrete interleaved sequence, which is modeled using an autoregressive Transformer. To enable unified modeling, images are discretized using vector-quantized (VQ) encoders, while actions are transformed into the frequency domain and discretized via Discrete Cosine Transform (DCT) encoding. This causal multimodal sequence naturally preserves the temporal dynamics and causality required for real-world tasks. The model builds upon a pretrained vision-language model and follows a two-stage training strategy: (1) a post-training phase that adopts world-model training on large-scale datasets without requiring actions, and (2) a fine-tuning phase that interleaves actions into the sequence, enabling policy learning on downstream tasks.

novel action sequences. DREMA [1] replicates scene dynamics and structure by integrating Gaussian Splatting with physics simulation. VLP [23] enables long-horizon visual planning by combining text-to-video generation with vision-language models as heuristic evaluators. DayDreamer [71] extends Dreamer [34] to real-world robotic platforms, while UVA [45] proposes a joint video-action latent space to decouple video and action generation, achieving high accuracy and efficiency in policy inference. AdaWorld [27] extracts latent actions from videos in a self-supervised manner and builds an autoregressive world model conditioned on these latent actions.

- 3 Unified Vision-Language-Action Model

In this section, we present the design of UniVLA, as illustrated in Figure 2. Unlike previous VLA models [42, 6] that rely on ViT [21] for image encoding, our approach adopts an encoder-free architecture, converting all modalities into discrete tokens and learning them autoregressively. The overall design is simple yet effective, demonstrating strong scalability.

Our unified paradigm has two key aspects: first, it unifies the learning of multiple modalities, integrating various modality tokens into a shared representation space and employing a transformer for autoregressive learning; Second, it unifies sequence modeling across tasks through the natural interleaving of modalities, facilitating the seamless combination of tasks such as video generation, visual grounding, and action learning. In the following sections, we will introduce the method from the perspectives of Unified Multimodal Model and Unified Multimodal Sequence Modeling.

##### 3.1 Unified Multimodal Model

As illustrated in Figure 2, our method unifies language, vision, and action modalities by converting each into discrete tokens and concatenating them into a single multimodal sequence L. Specifically, Lt, Lv, and La denote the discrete token sequences for language, vision, and action, respectively, all drawn from a shared vocabulary. The superscript indicates the temporal step, with tokens interleaved across modalities to preserve temporal alignment.

For example, in the robotic manipulation task, a textual instruction is provided only at the beginning, followed by a naturally interleaved sequence of visual observations and actions. The language and vision tokenizers adopt the same design as Emu3 [68]; visual observations are discretized using a VQ tokenizer [77], while actions are encoded using FAST [58]. To clearly demarcate modality boundaries, we employ special tokens—boi (begin of image), eoi (end of image), boa (begin of action), and eoa (end of action)—to encapsulate image and action tokens, respectively.

Action Modeling We follow FAST [58] and apply the Discrete Cosine Transform (DCT) to convert continuous action sequences into discrete action tokens. Specifically, given an action sequence within a time window, we define La at a given time step as a sequence of action tokens [T1,...,Tn]. The raw action sequence A1:H = {a1,a2,...,aH} spans a window of size H, where each action at is a d-dimensional vector. The FAST action tokenizer encodes A1:H into a discrete token sequence [T1,...,Tn], with n tokens drawn from a vocabulary of size |V |. Similar to natural language processing, action sequences can vary in token length, resulting in a variable-length (n) discrete representation.

Training Objective Since all modality signals are transformed into discrete tokens, the training objective is simplified to a standard next-token prediction task using cross-entropy loss. To accommodate different task formats, we selectively include specific tokens in the loss computation, ensuring compatibility and flexibility across diverse tasks.

##### 3.2 Unified Multimodal Sequence Modeling

As shown in Figure 2, our multimodal sequence representation naturally captures the temporal dynamics and causal structure inherent in task execution. The embodied planning problem can be formulated as a Markov Decision Process (MDP), a general mathematical framework for decisionmaking in partially stochastic environments. For example, in the task of picking a carrot, the instruction and current observation inform the action; this action alters the environment, leading to a new observation that subsequently guides the next action. Building on this interleaved Markovian formulation, we unify a variety of tasks within a shared sequence modeling framework, and present the task-specific modeling strategies in the following.

World Model (Post-train) Within the MDP framework, a world model aims to learn the dynamics of the environment by modeling the transition function P(st+1|st,at). The learned world model enables agents to simulate future trajectories, plan actions, and reason about consequences without direct interaction with the environment. Specifically, in the context of robotic tasks, we treat the language instructiom as a general form of action. Given the current observation L1v and the instruction L1t, the world model need to predict future visual content. In this setting, we use the loss computed solely from the vision tokens as the supervisory signal, enabling the model to generate visual predictions conditioned on the given instruction and observed state. Sequence Sv formulation is as follows:

Sv = {L1t,L1v,L2v,...,Ltv} (1)

Policy Learning (Fine-tune) Policy learning enables the agent to determine optimal actions based on both current observations and prior states, thereby effectively guiding task execution. In this setting, we employ a loss function computed solely from the action tokens. The sequence Sa representing the interactions over time is formulated as:

Sa = {L1t,L1v,L1a,L2v,L2a,...,Ltv,Lta} (2)

- As illustrated in Figure 2, in this interleaved format, we adopt a two-stage training paradigm for robotic tasks. The model is initialized with a vision-language (VL) aligned checkpoint, endowing it with basic vision-language capabilities. The post-training stage leverages a world model objective to capture video dynamics, treating world modeling as a general visual learning task. Building upon the learned world model, the fine-tuning stage focuses on action learning to refine task-specific behaviors. We observe that incorporating the world model significantly enhances the efficiency and effectiveness of policy learning.

### 4 Experiments

##### 4.1 Dataset

CALVIN. CALVIN [54] is a simulated benchmark tailored for evaluating long-horizon, languageconditioned robotic manipulation. It comprises four simulated environments (A, B, C, and D), each containing demonstration trajectories collected via human teleoperation. The benchmark encompasses 34 distinct manipulation tasks with a total of 1,000 unique language instructions. Performance is

measured by the average number of successfully completed sub-tasks within a sequence. Standard evaluation protocols include the ABC→D and ABCD→D settings, which test a model’s ability to generalize to unseen environments and compositions of long-horizon tasks.

LIBERO. The LIBERO benchmark [49] is a comprehensive suite for lifelong robotic manipulation, comprising four task suites with 10 tasks and 50 human demonstrations each. These suites are designed to evaluate different generalization abilities: LIBERO-Spatial tests spatial reasoning by varying layouts with fixed objects; LIBERO-Object assesses object-level generalization with varying objects in a fixed scene; LIBERO-Goal targets goal-conditioned behavior by varying task goals; and LIBERO-Long (LIBERO-10) features long-horizon, compositional tasks with diverse objects, layouts, and goals, challenging temporal and compositional reasoning.

SimplerEnv. SimplerEnv [48] serves as a simulation benchmark designed to evaluate the transferability and generalization capabilities of models trained on real-world video data. It incorporates diverse manipulation setups across both the WidowX and Google Robot platforms, encompassing variations in lighting conditions, object textures, color distributions, and camera viewpoints.

##### 4.2 Implementation Details

The model adopts a purely autoregressive Transformer architecture with 8.5 billion parameters, identical to Emu3 [68]. Images are tokenized using a VQ-based image encoder with a spatial compression factor of 8. For action encoding, we use the relative differences between consecutive frames. We first apply 1st and 99th percentile normalization, and then utilize the FAST tokenizer [58], which has a vocabulary size of 1024 and replaces the final 1024 token IDs of the language tokenizer.

Post-training Stage. In the post-training stage, we leverage large-scale robot-centric video datasets to study the effects of various post-training strategies on downstream policy learning. The model is initialized with pre-trained weights from the first stage of Emu3 [68]. We curate a total of 622K videos from existing robotics datasets (details provided in the appendix), and identify the world model as the most effective post-training approach. During training, supervision is applied solely on the vision tokens. The model is trained for 30K steps with a batch size of 64.

Fine-tuning Stage. During fine-tuning, the model is initialized with weights from the post-training stage and trained using a two-frame interleaved vision-action sequence with an action chunk size of 10. A cosine annealing learning rate schedule is applied, starting at 8 × 10−5, and the loss is computed solely over action tokens. For the CALVIN benchmark, RGB observations from both third-person (200 × 200) and wrist-view (80 × 80) cameras are used. Training is conducted on A100 GPUs with a batch size of 192 for 8k steps. For the LIBERO benchmark, third-person and wrist-view RGB images (both at 200×200) are used to train a unified model with a batch size of 192 for 8k steps. A single model is evaluated across four task suites. For the SimplerEnv benchmark, single-view RGB observations are used with input resized to 256 × 256. Training is conducted on the Bridge-WidowX setup using a batch size of 128 for 20k steps, with an action chunk size of 5.

Additional implementation details on the post-training strategy, real-robot fine-tuning procedures, and autonomous driving experiments are provided in the appendix.

##### 4.3 Main Results

In this section, we evaluate our method on three simulation benchmarks: CALVIN (long-horizon tasks), LIBERO (diverse generalization), and SimplerEnv (real-to-sim manipulation). Our approach consistently achieves state-of-the-art performance across all settings.

CALVIN Simulation Evaluation. Table 1 presents the experimental results in the CALVIN benchmark. Our method achieves the highest performance on both the ABC→D and ABCD→D tasks, significantly outperforming previous approaches and demonstrating strong capabilities in multi-task learning and long-horizon planning.

LIBERO Simulation Evaluation. Following [75], we report the average success rate over 500 episodes for each task suite (Spatial, Object, Goal, Long). As shown in Table 2, UniVLA achieves the best overall performance across all LIBERO benchmark suites, with particularly significant gains on long-horizon tasks—improving the previous state of the art from 69.0% to 94.0%. Compared to π0 [58], our method demonstrates superior performance on long-horizon tasks.

##### Table 1: Long-horizon robotic manipulation evaluation on the CALVIN benchmark.

###### Method Task Tasks Completed in a Row Avg. Len ↑

1 2 3 4 5

MCIL [52] ABCD→D 0.373 0.027 0.002 0.000 0.000 0.40 RT-1 [9] ABCD→D 0.844 0.617 0.438 0.323 0.227 2.45 Robo-Flamingo [47] ABCD→D 0.964 0.896 0.824 0.740 0.660 4.09 GR-1 [70] ABCD→D 0.949 0.896 0.844 0.789 0.731 4.21 UP-VLA [74] ABCD→D 0.962 0.921 0.879 0.842 0.812 4.42 RoboVLMs [46] ABCD→D 0.967 0.930 0.899 0.865 0.826 4.49 UniVLA ABCD→D 0.985 0.961 0.931 0.899 0.851 4.63

MCIL [52] ABC→D 0.304 0.013 0.002 0.000 0.000 0.31 Robo-Flamingo [47] ABC→D 0.824 0.619 0.466 0.331 0.235 2.47 SuSIE [7] ABC→D 0.870 0.690 0.490 0.380 0.260 2.69 GR-1 [70] ABC→D 0.854 0.712 0.596 0.497 0.401 3.06 UP-VLA [74] ABC→D 0.928 0.865 0.815 0.769 0.699 4.08 RoboVLMs [46] ABC→D 0.980 0.936 0.854 0.778 0.704 4.25 Seer-Large [63] ABC→D 0.963 0.916 0.861 0.803 0.740 4.28 UniVLA ABC→D 0.989 0.948 0.890 0.828 0.751 4.41

##### Table 2: Comparison of different methods on the LIBERO benchmark.

###### Method SPATIAL OBJECT GOAL LONG Average

DP* [17] 78.3% 92.5% 68.3% 50.5% 72.4% Octo [62] 78.9% 85.7% 84.6% 51.1% 75.1% OpenVLA [42] 84.9% 88.4% 79.2% 53.7% 76.5% SpatialVLA [59] 88.2% 89.9% 78.6% 55.5% 78.1% CoT-VLA [75] 87.5% 91.6% 87.6% 69.0% 81.1% π0-FAST [58] 96.4% 96.8% 88.6% 60.2% 85.5% UniVLA 95.4% 98.8% 93.6% 94.0% 95.5%

SimplerEnv Simulation Evaluation. Table 3 summarizes the performance across various manipulation policies on the Bridge-WidowX setup. Our approach demonstrates a significant improvement over prior methods, raising the average success rate from 42.7% to 69.8%. In particular, it shows marked improvements on previously difficult tasks, including stack block, put carrot and put spoon.

##### Table 3: Evaluation on SimplerEnv-WidowX across various manipulation tasks.

Put Spoon on Towel Put Carrot on Plate Stack Green on Yellow Block Put Eggplant in Yellow Basket Overall Grasp Success Grasp Success Grasp Success Grasp Success Success

Model

RT-1-X [8] 16.7% 0.0% 20.8% 4.2% 8.3% 0.0% 0.0% 0.0% 1.1% Octo-Base [56] 34.7% 12.5% 52.8% 8.3% 31.9% 0.0% 66.7% 43.1% 16.0% Octo-Small [56] 77.8% 47.2% 27.8% 9.7% 40.3% 4.2% 87.5% 56.9% 29.5% OpenVLA [42] 4.1% 0.0% 33.3% 0.0% 12.5% 0.0% 8.3% 4.1% 1.0% RoboVLMs [46] 70.8% 45.8% 33.3% 20.8% 54.2% 4.2% 91.7% 79.2 37.5% SpatialVLA [59] 20.8% 16.7% 29.2% 25.0% 62.5% 29.2% 100% 100% 42.7% UniVLA 83.3% 83.3% 74.0% 66.7% 95.8% 33.3% 100.0% 95.8% 69.8%

##### 4.4 In-Depth Analysis

In this section, we provide an in-depth analysis within our unified framework, which may offer key insights for the design of future VLA models. We first analyze how post-training enhances downstream policy learning in terms of both performance (Table 4) and training efficiency (Table 5), highlighting the potential of world models as a general post-training strategy for robotics. We then investigate that even without post-training stage, incorporating visual prediction loss (Table 6a) and historical context (Table 6b) still contributes positively to policy learning.

Effectiveness of World Model Post-Training. Table 4 investigates the effects of different posttraining strategies on downstream policy learning across various simulation benchmarks. The results reveal that, due to inconsistencies in the action space across tasks, action-only learning exhibits low transferability, leading to a negative impact on performance. In contrast, most post-training approaches significantly enhance policy learning, highlighting the crucial role of visual learning in transferability. Among these, the world model post-training approach yields the most substantial

- Table 4: Effectiveness of World Model Post-Training. We compare different post-training strategies by fine-tuning only with action prediction on the downstream benchmarks.

Post-training Stage Generalization Long-horizon Strategy Sequence Supervision LIBERO SimplerEnv-WidowX LIBERO-Long CALVIN

48.5 0.0 17.4 1.46 action prediction T,I,A action 43.9 (-4.6) 0.0 10.6 (-6.8) 0.52(-0.94)

text-to-image T,I vision 69.8 (+21.3) 6.3 (+6.3) 55.8 (+38.4) 3.79 (+2.33) video prediction I1,...,It vision 78.9 (+30.4) 17.7 (+17.7) 80.8 (+63.4) 3.59 (+2.13)

world model T,I1,...,It vision 94.2 (+45.7) 64.6 (+64.6) 89.2 (+71.8) 4.61 (+3.15)

gains, enhancing both generalization and long-horizon planning capabilities. A comparison with text-to-image (T2I) training emphasizes the importance of modeling temporal dynamics in video data, while contrasting with video-only training highlights the essential role of textual guidance in state transitions. Notably, this world model training requires no action annotations, enabling scalable learning from large-scale video data and providing a promising direction for future VLA research.

Data and Training Efficiency. Table 5 shows that post-training substantially enhances downstream policy learning efficiency. On the CALVIN benchmark (Table 5a), our method achieves higher success rates using only 10% of the fine-tuning data, outperforming prior approaches such as GR-1 [70] and RoboVLMs [46]. In addition, Table 5b highlights improved training efficiency, as the model rapidly converges with fewer fine-tuning iterations. The Simpler-Env results further demonstrate the effectiveness of world-model-based post-training for efficient policy adaptation across diverse robotic setups. While similar effects are observed in latent-action methods [73, 16, 27], our world model offers a simpler paradigm without latent actions, achieving better transferability.

- Table 5: Post-training enables data-efficient and training-efficient downstream policy learning. (a) Data efficiency comparison.

(b) Training efficiency comparison.

Method Data CALVIN

RT-1 [9] 10% 0.34 MT-R3M [55] 10% 0.61 HULC [53] 10% 1.11 GR-1 [70] 10% 2.00 RoboVLMS [46] 10% 2.52

UniVLA (w/o post-train) 10% 0.15 UniVLA 10% 3.19

###### Fast convergence (CALVIN)

Training Iters 2k 4k 8k

w/o post-train 0.37 0.82 1.46 w/ post-train 4.21 4.56 4.61

Fast adaptation (SimplerEnv-Bridge) Method Batch size Iters Success

RoboVLMs [46] 128 50k 37.5 UniVLA 128 12k 64.6

##### Table 6: Ablation study on the visual prediction and historical context in policy learning. (a) Effectiveness of visual prediction.

###### (b) Effectiveness of history context.

###### Post-train Visual prediction CALVIN LIBERO

✓ 4.61 94.2 ✓ 4.42 88.7 1.46 48.5

Observations Avg. Len ↑ History Window Current + History

0 1 + 0 4.26

- 10 1 + 1 4.61

- 10 1 + 2 4.43 20 1 + 2 4.47

Effectiveness of Visual Prediction. While post-training proves effective, it is also crucial that the model demonstrates strong performance without relying on it. As shown in Table 6a, our findings indicate that, even without post-training, fine-tuning with visual loss supervision—leveraging the autoregressive nature of the model—naturally integrates world model learning into the policy learning process. This approach leads to a significant improvement in the model’s performance.

Effectiveness of History Context. History context—comprising past observations and actions—provides valuable guidance for robot planning. In this section, we investigate the appropriate length of the history window during the fine-tuning stage. As shown in Table 6b, our ablation study on the CALVIN benchmark examines the impact of varying history window lengths. Incorporating a history window significantly improves performance (from 4.26 to 4.61). However, extending the window beyond a certain length yields diminishing returns, suggesting that recent observations carry the most predictive value, consistent with the Markov property in sequential planning.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Put the yellow and white mug in the microwave and close it

Turn on the stove and put the moka pot on it

| |
|---|

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Spatial Grounding: Locate the [objects]

Visual Prediction: Pick up the can on the table

- Figure 3: Multimodal capabilities of UniVLA. Top: Action outputs for executing long-horizon tasks in the LIBERO benchmark. Bottom: Visual predictions and spatial grounding demonstrating the model’s spatiotemporal understanding. The red box marks the current observation; green boxes indicate predicted object detections.

##### 4.5 Multimodal Capability

As illustrated in Figure 3, we qualitatively showcase the model’s ability to interleave multiple modalities—action, language, and vision—within a unified framework. This design enables policy learning for embodied control, spatial reasoning through language output, and future state prediction via visual output, highlighting the model’s capacity for generalizable multimodal understanding.

##### 4.6 Broader Applications

End-to-end Learning for Autonomous Driving. To further explore the potential of our method, we perform a preliminary transfer to the autonomous driving domain by finetuning the model on the NAVSIM benchmark. Notably, our method is a pure autoregressive, token-based framework, modeling the driving task as causal sequence prediction over discretized multimodal tokens. Despite using only front-view camera inputs—without relying on BEV representations or multi-sensor fusion—our model achieves powerful performance on the NAVSIM test set. Notably, the current performance is not pretrained on driving videos but is only fine-tuned on downstream policy benchmarks. These results highlight the strong potential of our method for broader real-world applications.

##### Table 7: Broader applications of UniVLA for end-to-end autonomous driving on the NAVSIM. MC: Multi Camera. L: LiDAR. FC: Front Camera.

###### Method Model Input NC↑ DAC↑ EP↑ TTC↑ C↑ PDMS↑

Human – – 100.0 100.0 87.5 100.0 99.9 94.8 Ego Status MLP – Ego State 93.0 77.3 62.8 83.6 100.0 65.6 VADv2 [15] BEV-based MC 97.9 91.7 77.6 92.9 100.0 83.0 UniAD [36] BEV-based MC 97.8 91.9 78.8 92.9 100.0 83.4 Transfuser [18] BEV-based MC&L 97.7 92.8 79.2 92.8 100.0 84.0 UniVLA Auto-regressive FC 96.9 91.1 76.8 91.7 96.7 81.7

### 5 Conclusion

In this paper, we present UniVLA, a unified framework for vision–language–action modeling that bridges heterogeneous modalities through a shared token space and models them autoregressively. The proposed unified design facilitates deeper cross-modal integration and inherently supports flexible multimodal tasks. By leveraging a world model trained to capture dynamics and causality from videos, we observe significant improvements in downstream policy learning, both in terms of performance and efficiency. Extensive simulation experiments further demonstrate the model’s strong generalization ability, efficient policy learning, and broad applicability across diverse domains. These findings highlight the great potential of our method as a new paradigm for vision–language–action modeling.

Limitations and Future Work. Due to limited computational resources, our investigation into posttraining scalability is still in its early stages. Nonetheless, initial results are promising and indicate potential for scaling to larger video datasets. Furthermore, while the unified multimodal framework exhibits strong capabilities in cross-modal learning, further research is needed to fully integrate it with reinforcement learning paradigms, enabling more robust and adaptive policy learning.

### References

- [1] Leonardo Barcellona, Andrii Zadaianchuk, Davide Allegro, Samuele Papa, Stefano Ghidoni, and Efstratios Gavves. Dream to manipulate: Compositional world models empowering robot imitation learning with imagination. arXiv preprint arXiv:2412.14957, 2024. 4
- [2] Suneel Belkhale, Tianli Ding, Ted Xiao, Pierre Sermanet, Quon Vuong, Jonathan Tompson, Yevgen Chebotar, Debidatta Dwibedi, and Dorsa Sadigh. Rt-h: Action hierarchies using language. arXiv preprint arXiv:2403.01823, 2024. 3
- [3] Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024. 2
- [4] Homanga Bharadhwaj, Roozbeh Mottaghi, Abhinav Gupta, and Shubham Tulsiani. Track2act: Predicting point tracks from internet videos enables diverse zero-shot robot manipulation. In European Conference on Computer Vision, 2024. 3
- [5] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025. 2
- [6] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo

Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 2, 3, 4

- [7] Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya, Homer Walke, Chelsea Finn, Aviral Kumar, and Sergey Levine. Zero-shot robotic manipulation with pretrained image-editing diffusion models. arXiv preprint arXiv:2310.10639, 2023. 3, 7
- [8] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-languageaction models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818,

2023. 2, 3, 7

- [9] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 7, 8, 16
- [10] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024. 3
- [11] Haoxuan Che, Xuanhua He, Quande Liu, Cheng Jin, and Hao Chen. Gamegen-x: Interactive open-world game video generation. arXiv preprint arXiv:2411.00769, 2024. 3
- [12] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024. 3
- [13] Lawrence Yunliang Chen, Simeon Adebola, and Ken Goldberg. Berkeley UR5 demonstration dataset. https://sites.google.com/view/berkeley-ur5/home. 16
- [14] Lili Chen, Shikhar Bahl, and Deepak Pathak. Playfusion: Skill acquisition via diffusion from language-annotated play. In Conference on Robot Learning, pages 2012–2029. PMLR, 2023. 16
- [15] Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243, 2024. 9
- [16] Yi Chen, Yuying Ge, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, and Xihui Liu. Moto: Latent motion token as the bridging language for robot manipulation. arXiv preprint arXiv:2412.04445, 2024. 8

- [17] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, page 02783649241273668, 2023. 7
- [18] Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence, 45(11):12878–12895, 2022. 9
- [19] Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, et al. Navsim: Data-driven nonreactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems, 37:28706–28719, 2024. 18
- [20] Pengxiang Ding, Jianfei Ma, Xinyang Tong, Binghong Zou, Xinxin Luo, Yiguo Fan, Ting Wang, Hongchao Lu, Panzhong Mo, Jinxin Liu, et al. Humanoid-vla: Towards universal humanoid control with visual integration. arXiv preprint arXiv:2502.14795, 2025. 2
- [21] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. 4
- [22] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. In International Conference on Machine Learning, pages 8469–

8488. PMLR, 2023. 3

- [23] Yilun Du, Mengjiao Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Brian Ichter, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B Tenenbaum, et al. Video language planning. ICLR, 2024. 4
- [24] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. Advances in neural information processing systems, 36:9156–9172, 2023. 3
- [25] Chelsea Finn and Sergey Levine. Deep visual foresight for planning robot motion. In 2017 IEEE international conference on robotics and automation (ICRA), pages 2786–2793. IEEE,

2017. 3

- [26] Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. arXiv preprint arXiv:2405.17398, 2024. 3
- [27] Shenyuan Gao, Siyuan Zhou, Yilun Du, Jun Zhang, and Chuang Gan. Adaworld: Learning adaptable world models with latent actions. arXiv preprint arXiv:2503.18938, 2025. 4, 8
- [28] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The" something something" video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, pages 5842– 5850, 2017. 16
- [29] Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, Xinyue Wei, Yunchao Yao, Xiaodi Yuan, Pengwei Xie, Zhiao Huang, Rui Chen, and Hao Su. Maniskill2: A unified benchmark for generalizable manipulation skills. In International Conference on Learning Representations, 2023. 16
- [30] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 2
- [31] Yanjiang Guo, Yucheng Hu, Jianke Zhang, Yen-Jen Wang, Xiaoyu Chen, Chaochao Lu, and Jianyu Chen. Prediction with action: Visual policy learning via joint denoising process. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 3

- [32] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018. 3
- [33] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603, 2019. 3
- [34] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In International conference on machine learning, pages 2555–2565. PMLR, 2019. 4
- [35] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023. 3
- [36] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17853–17862, 2023. 9
- [37] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. \pi_ {0.5}: a visionlanguage-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025. 3
- [38] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024. 2
- [39] Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian Ibarz, Alexander Herzog, Eric Jang, Deirdre Quillen, Ethan Holly, Mrinal Kalakrishnan, Vincent Vanhoucke, et al. Scalable deep reinforcement learning for vision-based robotic manipulation. In Conference on robot learning, pages 651–673. PMLR, 2018. 16
- [40] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024. 16
- [41] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025. 3
- [42] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 2, 3, 4, 7, 16
- [43] Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1):1–62, 2022. 3
- [44] Peiyan Li, Hongtao Wu, Yan Huang, Chilam Cheang, Liang Wang, and Tao Kong. Gr-mg: Leveraging partially-annotated data via multi-modal goal-conditioned policy. IEEE Robotics and Automation Letters, 2025. 3
- [45] Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. arXiv preprint arXiv:2503.00200, 2025. 4
- [46] Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, Hanbo Zhang, and Huaping Liu. Towards generalist robot policies: What matters in building vision-language-action models. arXiv preprint arXiv:2412.14058, 2024. 7, 8, 16
- [47] Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, et al. Vision-language foundation models as effective robot imitators. In ICLR, 2024. 7
- [48] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, et al. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024. 2, 6

- [49] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023. 2, 6, 16
- [50] Jiaming Liu, Hao Chen, Pengju An, Zhuoyang Liu, Renrui Zhang, Chenyang Gu, Xiaoqi Li, Ziyu Guo, Sixiang Chen, Mengzhen Liu, et al. Hybridvla: Collaborative diffusion and autoregression in a unified vision-language-action model. arXiv preprint arXiv:2503.10631,

2025. 3

- [51] Jianlan Luo, Charles Xu, Fangchen Liu, Liam Tan, Zipeng Lin, Jeffrey Wu, Pieter Abbeel, and Sergey Levine. Fmb: a functional manipulation benchmark for generalizable robotic learning. The International Journal of Robotics Research, page 02783649241276017, 2023. 16
- [52] Corey Lynch and Pierre Sermanet. Language conditioned imitation learning over unstructured data. arXiv preprint arXiv:2005.07648, 2020. 7
- [53] Oier Mees, Lukas Hermann, and Wolfram Burgard. What matters in language conditioned robotic imitation learning over unstructured data. IEEE Robotics and Automation Letters, 7(4):11205–11212, 2022. 8
- [54] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3):7327–7334, 2022. 2, 5, 16
- [55] Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3m: A universal visual representation for robot manipulation. arXiv preprint arXiv:2203.12601, 2022. 8
- [56] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Charles Xu, Jianlan Luo, Tobias Kreiman, You Liang Tan, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An open-source generalist robot policy. In Proceedings of Robotics: Science and Systems, Delft, Netherlands,

2024. 2, 7

- [57] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 2
- [58] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025. 3, 4, 5, 6, 7
- [59] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025. 3, 7
- [60] Erick Rosete-Beas, Oier Mees, Gabriel Kalweit, Joschka Boedecker, and Wolfram Burgard. Latent plans for task-agnostic offline reinforcement learning. In Conference on Robot Learning, pages 1838–1849. PMLR, 2023. 16
- [61] Rutav Shah, Roberto Martín-Martín, and Yuke Zhu. Mutex: Learning unified policies from multimodal task specifications. arXiv preprint arXiv:2309.14320, 2023. 16
- [62] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024. 7
- [63] Yang Tian, Sizhe Yang, Jia Zeng, Ping Wang, Dahua Lin, Hao Dong, and Jiangmiao Pang. Predictive inverse dynamics models are scalable learners for robotic manipulation. arXiv preprint arXiv:2412.15109, 2024. 7
- [64] Quan Vuong, Sergey Levine, Homer Rich Walke, Karl Pertsch, Anikait Singh, Ria Doshi, Charles Xu, Jianlan Luo, Liam Tan, Dhruv Shah, et al. Open x-embodiment: Robotic learning datasets and rt-x models. In Towards Generalist Robots: Learning Paradigms for Scalable Skill Acquisition@ CoRL2023, 2023. 3

- [65] Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe HansenEstruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning, pages 1723–1736. PMLR, 2023. 16
- [66] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 2
- [67] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-drive world models for autonomous driving. In European Conference on Computer Vision, pages 55–72. Springer, 2024. 3
- [68] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 4, 6
- [69] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14749–14759, 2024. 3
- [70] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. In The Twelfth International Conference on Learning Representations,

2024. 3, 7, 8

- [71] Philipp Wu, Alejandro Escontrela, Danijar Hafner, Pieter Abbeel, and Ken Goldberg. Daydreamer: World models for physical robot learning. In Conference on robot learning, pages 2226–2240. PMLR, 2023. 3, 4
- [72] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 1(2):6, 2023. 3
- [73] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al. Latent action pretraining from videos. ICLR, 2025. 3, 8
- [74] Jianke Zhang, Yanjiang Guo, Yucheng Hu, Xiaoyu Chen, Xiang Zhu, and Jianyu Chen. Up-vla: A unified understanding and prediction model for embodied agent. arXiv preprint arXiv:2501.18867, 2025. 7
- [75] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. arXiv preprint arXiv:2503.22020, 2025. 6, 7
- [76] Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3d-vla: A 3d vision-language-action generative world model. In International Conference on Machine Learning, pages 61229–61245. PMLR, 2024. 3
- [77] Chuanxia Zheng, Tung-Long Vuong, Jianfei Cai, and Dinh Phung. Movq: Modulating quantized vectors for high-fidelity image generation. Advances in Neural Information Processing Systems, 35:23412–23425, 2022. 4
- [78] Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daumé III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. arXiv preprint arXiv:2412.10345, 2024. 3
- [79] Gaoyue Zhou, Victoria Dean, Mohan Kumar Srirama, Aravind Rajeswaran, Jyothish Pari, Kyle Hatch, Aryan Jain, Tianhe Yu, Pieter Abbeel, Lerrel Pinto, et al. Train offline, test online: A real robot learning benchmark. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 9197–9203. IEEE, 2023. 16

- [80] Siyuan Zhou, Yilun Du, Jiaben Chen, Yandong Li, Dit-Yan Yeung, and Chuang Gan. Robodreamer: Learning compositional world models for robot imagination. arXiv preprint arXiv:2404.12377, 2024. 3
- [81] Yifeng Zhu, Abhishek Joshi, Peter Stone, and Yuke Zhu. Viola: Imitation learning for visionbased manipulation with object proposal priors. In Conference on Robot Learning, pages 1199–1210. PMLR, 2023. 16

Appendix

- A Implementation Details

Post-training Stage We began by selecting several high-quality robotics datasets for post-training, as summarized in Table 8. To account for differences in data collection frequencies across datasets, we applied dataset-specific frame sampling intervals to ensure that the temporal gap between keyframes is approximately one second. We further filtered out video sequences containing fewer than six frames, as well as those lacking corresponding text instructions. Due to the large number of videos from the Kuka [39] dataset, we randomly retained 100k videos to prevent it from dominating the overall training data.

##### Table 8: Post-training datasets.

###### Dataset Source Data Type Raw Videos Used Videos Interval

RT-1 [9] Real Text, Video, Action 87212 84084 3 BridgeV2 [65] Real Text, Video, Action 60064 28083 5 DROID [40] Real Text, Video, Action 275997 145641 15 Kuka [39] Real Text, Video, Action 580392 100000 3 TOTO [79] Real Text, Video, Action 902 899 20 Taco Play [60] Real Text, Video, Action 3242 3242 5 FMB [51] Real Text, Video, Action 8611 7876 5 Berkeley autolab ur5 [13] Real Text, Video, Action 896 896 5 VIOLA [81] Real Text, Video, Action 135 135 15 Cmu Play Fusion [14] Real Text, Video, Action 576 576 10 Utaustin Mutex [61] Real Text, Video, Action 1500 1500 10

CALVIN [54] Sim Text, Video, Action 22966 22966 5 LIBERO [49] Sim Text, Video, Action 3386 3386 10 ManiSkill2 [29] Sim Text, Video, Action 30213 193273 10

SSV2 [28] Real Text, Video 220847 220847 1

For the experiments in Table 4, to ensure a fair comparison of different post-training strategies, all models are trained on the same dataset (excluding SSV2 [28], which does not contain action annotations), with only the post-training strategy varied. For the action prediction task, we organize the input as (T,I,A), where T denotes the text instruction, I the image observations, and A the action sequence. During training, only the action tokens A are supervised in the loss computation. For the text-to-image task, the input is organized as (T,I), where T denotes the input text and I denotes the target image. During training, the loss is only computed on the vision tokens corresponding to I. For the video prediction task, the input is organized as (I1,...,It), where I denotes the video frame. During training, the loss is computed on the vision tokens. For the world model task, the input is organized as (T,I1,...,It), where T denotes the input text, I denotes the video frame. During training, the loss is computed on the vision tokens.

During training, the observations are resized to 256×256, using six frames as input, with the maximum sequence length set to 6400. We perform full-parameter training for 50k steps using 32 A100 GPUs (40GB), which takes approximately 4–5 days.

Simulation Finetuning The training setup is described in the main paper. We adopt full-parameter training, and for evaluation, we follow the testing protocols of OpenVLA [42] and RoboVLMs [46] across various benchmarks. By default, our model is trained using video-format sequences; however, it also supports fine-tuning with image-format sequences. In the ablation study evaluating the effect of visual prediction, when post-training is not applied, the visual token weight is set to 0.5 while the action token weight is set to 1.0, in order to maintain balance between the two modalities.

Real-robot Finetuning For real-world evaluation, we conduct experiments on the ALOHA platform, using images captured from three perspectives: cam high, wrist left, and wrist right. The real-robot is controlled using end-effector (EE) pose. All input images are resized to a resolution of 128×128. The model outputs a 14-dimensional action vector. The action chunk size is set to 20.

For each task, we train for 8k steps with a batch size of 256. The learning rate is set to 5 × 10−5, and all other settings remain consistent with the above. We also leverage world model pretraining, using video-based post-training on a collected real-aloha dataset (Table 9). Interestingly, this post-training provides substantial benefits even when transferring to real-robot execution.

### B Real-Robot Experiments

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

###### Wrist left Camera high Wrist right

- Figure 4: Real-world setup of the AgileX Cobot Magic dual-arm robot. The system is equipped with three RGB cameras for visual observation: one mounted on the left wrist, one on the right wrist, and one positioned above for a high-angle view.

B.1 ALOHA Experimental Setup

The robotic platform used in this paper is AgileX Cobot Magic V2.0, a dual-arm robot. As shown in Figure 4, the robot is equipped with two arms and three camera views, enabling it to perform a variety of manipulation tasks. For example, Figure 5 illustrates a range of manipulation tasks collected from real-world scenarios.

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

- Figure 5: Real-world task examples. These include diverse tasks such as wiping a whiteboard, organizing tableware, making a burger, and plugging in a connector.

Real-World Task Collection Table 9 provides a summary of the real-world data collected from the physical robot, recorded at an actual frequency of 30 Hz. A total of 8 tasks were included, with each task collecting approximately 500 trajectories on average. During preprocessing, static frames at the beginning and end of each trajectory were filtered out.

Table 9: Real-world task trajectories.

Fold Clothes

Make Hamburger 528 500 500 500 496 500 500 640

Clear Desk

Store Glasses

Food Packing

Pour Water

Clean Blackboard

Insert Plug

Data Processing To reduce redundancy and improve training efficiency, we select keyframes based on thresholding the changes in recorded action joint values. For each selected sequence, the action chunk is normalized by subtracting the joint values of the first frame.

### C Autonomous Driving Experiments

NAVSIM Setup The NAVSIM dataset [19], resampled from OpenScene to emphasize challenging scenarios, is currently one of the most established end-to-end evaluation benchmarks in the autonomous driving domain. The dataset is divided into two parts: Navtrain and Navtest, comprising 1,192 scenarios for training and validation, and 136 scenarios for testing.

For model training, the input images are resized to a resolution of 512×288. We follow the standard training setup, using the current image frame and ego status to predict trajectories for the next 8 frames. Both the action and ego status are encoded using the fast tokenizer.

