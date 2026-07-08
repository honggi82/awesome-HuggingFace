# arXiv:2312.00589v2[cs.CV]3Jul2024

## Merlin: Empowering Multimodal LLMs with Foresight Minds

En Yu1⋆, Liang Zhao2⋆, Yana Wei3, Jinrong Yang3, Dongming Wu4, Lingyu Kong5, Haoran Wei2, Tiancai Wang2, Zheng Ge2, Xiangyu Zhang2, and Wenbing Tao1⋆⋆

1 Huazhong University of Science and Technology

- 2 MEGVII Technology
- 3 ShanghaiTech University

4 Beijing Institute of Technology 5 University of Chinese Academy of Sciences {yuen,wenbingtao}@hust.edu.cn

Object Tracking

###### Video Relation

###### Video Referring

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

| | |
|---|---|
| | |
| | |

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Which object does the person use to stir the detergent in the bowl?

Please track this person and panda.

What is this player doing in this video?

[Figure 15]

[Figure 16]

This player is participating in a soccer match and is dribbling the ball.

[Figure 17]

[Figure 18]

[Figure 19]

The person uses the spoon to stir the detergrent in the bowl.

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

What is this player in the first frame going to do next?Analyse based on the trajectory.

Does the vehicle's movement entail

any risks? The ego car's correct course of action is?

FutureReasoning

[Figure 24]

[Figure 25]

Based on the trajectory of this player,it is evident he is readying to make a

There is a pedestrian crossing the street in the middle of the crosswalk. The ego car should be cautious and prepared to stop or slow down when necessary to avoid colliding with the pedestrian. The ego car's correct course of action is to be patient and

[Figure 26]

[Figure 27]

basketball 3-point shot next.

[Figure 28]

What are the chances of the player making this 3-point shot?

[Figure 29]

[Figure 30]

There is a defender planning to interfere the shot. Therefore, the

wait for the pedestrian to cross the

[Figure 31]

| | |
|---|---|
| | |

street before proceeding.

[Figure 32]

shooting percentage might decrease.

Fig. 1: Demo cases presentation of Merlin. Here we showcase several main capabilities of our built Multimodal Large Language Model (MLLM). Notably, in the dialogue, the words marked with colors correspond to the trajectory outputs of the targets in the image. To save space, we highlight them using the same colors.

Abstract. Humans can foresee the future based on present observations, a skill we term as foresight minds. However, this capability remains under-explored within existing MLLMs, hindering their capacity to understand intentions behind subjects. To address this, we integrate the future modeling into MLLMs. By utilizing the trajectory, a highly structured representation, as a learning objective, we aim to equip the

⋆ Equal Contribution ⋆⋆ Corresponding Author

model to understand spatiotemporal dynamics. Inspired by the learning paradigm of LLMs, we first propose Foresight Pre-Training (FPT) that jointly learns various tasks centered on trajectories, enabling MLLMs to predict entire trajectories from a given initial observation. Then, we propose Foresight Instruction-Tuning (FIT) that requires MLLMs to reason about potential future events based on predicted trajectories. Aided by FPT and FIT, we build an unified MLLM named Merlin that supports complex future reasoning. Experiments show Merlin’s foresight minds with impressive performance on both future reasoning and visual comprehension tasks. Project page: https://ahnsun.github.io/merlin. Keywords: Multimodal Large Language Model · Future Reasoning

### 1 Introduction

Human beings can predict future events or outcomes based on current observations, known in neuroscience theory as predictive processing [21]. In this paper, we refer to this ability as foresight minds, which involves the use of past experiences, knowledge, sensory information, and probabilistic reasoning to generate expectations about future events. In the artificial intelligence (AI) domain, the capability to predict future events is an important topic towards the realization of artificial general intelligence (AGI).

Recent advancements in Multimodal Large Language Models (MLLMs), such as GPT-4V [50] and Bard [2], have shown significant potential in image understanding and logical reasoning. Despite these achievements, these models struggle to foresee future events based on current image observations. Even provided with additional observations, like sequences of multiple frames, the current MLLM models still struggle to adequately analyze and infer specific target behaviors, such as predicting object movements or interactions (shown in Figure 2). On the contrary, human can reason the future to some extent based on the observed current state [5,56], which shows powerful foresight minds.

To mitigate this existing deficiency in MLLMs, we start from dividing human’s process of foreseeing the future into a two-stage system [32,56]: (1) observing the dynamic clues of the subject and then (2) analyzing the behavior pattern and reasoning what might happen according to the observation. For instance, while watching a basketball game, people will first observe the moving players on the court, and then forecast the specific player’s forthcoming actions, e.g., shooting, slam-dunking, or passing, by analyzing the current states and movement patterns of the players. Compare this system to current MLLMs, we find that MLLMs can complete the second stage well, thanks to the powerful logical reasoning ability of LLM [52,72]. Therefore the key challenge is the first stage. That is, how to make MLLM acquire correctly spatiotemporal dynamics from the multi-image observation?

Explicitly modeling next frames (e.g., reconstructing next frames [12,75]) can be a straightforward way. However, it can be hard to directly extract dynamic clues from the redundant visual information [27], especially from video sequences.

###### Failure case of GPT-4V

[Figure 33]

What is this player in the first frame going to

do next?Analyse based on the trajectory.

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Based on the player’s position within the court, their body language suggest they might be considering taking a shot.

√

[Figure 38]

Is there any player trying to disturb this shot?

[Figure 39]

[Figure 40]

In the provided frames, there does not appear to be a

defender trying to contest the shot.

##### Fig. 2: Failure case of GPT-4V about future reasoning.

It is necessary to construct a suitable learning objective to assist MLLM in obtaining dynamic clues about the specific subjects. To this end, we point out that trajectory, as a highly structured representation, is a good learning objective which can link the temporal contexts between the past and the future.

Based on this insight, we propose to model the future to empower existing MLLMs with “foresight minds”. Following the modern learning paradigm of LLMs, our future reasoning learning process includes two stages: (1) Foresight Pre-Training (FPT), a paradigm that causally models the temporal trajectories, which interleave with multi-frame images. The model starts with the initial observation of one or multiple subjects in the first frame as the query and then is required to predict the whole trajectory. Notably, we introduce various tasks containing richly labeled data [20,28,33,60,66,79], including object detection, object tracking, etc., to perform multitask learning. And samples from these tasks are properly formatted to ensure coordinated pre-training. (2) Future InstructionTuning (FIT), then, considers the trajectory modeling bestowed by FPT as a bridge in the logical chain of future reasoning. Simply put, when querying an MLLM, it must articulate its reasoning in conjunction with the trajectory for each object referenced. This method, as a form of Trajectory Chain-of-Thought, effectively narrows the gap between trajectory perception and predictive future reasoning, thereby fully unleashing model’s foresight minds.

Aided by the above future modeling technologies, we provide Merlin6, a novel and unified MLLM capable of handling inputs and outputs of spatial coordinates or tracklets from single image or multiple frames. Moreover, Merlin is adept at performing inductive reasoning about future events based on current observational results. To demonstrate this, we provide several real dialogues between users and Merlin, as displayed in the Figure 1. Unlike the previous MLLMs [43,83,86] which only supported interaction with a single image, Merlin not only provides a richer multi-image interaction, but also on this basis, is capable of executing unique and powerful future reasoning.

We construct a new future reason benchmark to evaluate Merlin’s logical reasoning and future prediction abilities. The results, which significantly surpass

- 6 Merlin is a legendary character in the tales of King Arthur, renowned as a powerful wizard and a wise counselor in the Arthurian legends. He is depicted as having the power to foresee future events and has a deep understanding of fate and destiny.

previous baselines [10,41,43,76], demonstrate Merlin’s stunning performance in future reasoning. We further reveal Merlin’s exceptional performance in general visual understanding. Through analysis in scenarios such as VQA (Visual Question Answering) [25,30], comprehensive understanding [44,78], and hallucination [39], we unexpectedly discovered that our proposed novel paradigm of future learning aids MLLMs in gaining a deeper understanding of images. We believe this brings new insights for the training of future MLLMs.

### 2 Related Work

#### 2.1 Large Language Models

Large Language Models (LLMs) have gained significant attention due to their capabilities in language generation and logical reasoning. Pioneering models like BERT [16], GPT-2 [54], and T5 [55] laid the groundwork, but GPT-3 [8], the first model with a 175 billion parameter size, made notable strides, demonstrating strong zero-shot performance. An emergent ability, wherein model size scaling results in significant language capability improvements, was also observed in LLMs. This was further facilitated by InstructGPT [51] and ChatGPT [49] using Reinforcement Learning with Human Feedback (RLHF) on GPT-3. These advancements led to what’s called LLMs’ “iPhone moment”. Following GPT’s success, several open-source LLMs, including OPT [82], LLaMA [68], and GLM [80], have been proposed, showing similar performance to GPT-3. Models like Alpaca [67] and Vicuna [11] illustrate the application of these LLMs, using a self-instruct framework to construct excellent dialogue models.

#### 2.2 Multimodal Large Language Models

The advancements in LLMs [49,68,69] have projected a promising path towards artificial general intelligence (AGI). This has incited interest in developing multimodal versions of these models. Current Multi-modal Large Language Models (MLLMs) harness the potential of cross-modal transfer technologies. These models consolidate multiple modalities into a unified language semantic space, and then employ autoregressive language models as decoders for language outputs. Models like Flamingo [1] have adopted a gated cross-attention mechanism, trained on billions of image-text pairs, to align visual and linguistic modalities, showing impressive performance on few-shot learning tasks. Similarly, BLIP-

- 2 [38] introduced the Q-Former to align visual features more effectively with language space. The LLaVA series [41,43] further enhanced this process by using simply a MLP in place of the Q-Former and designing a two-stage instructiontuning procedure. Apart from creating general MLLMs, techniques have also been developed for visual-interactive multimodal comprehension, involving the precise tuning of referring instructions [10,81,84]. Furthermore, another interesting direction in MLLM research involves integrating MLLMs for cross-modal generation [18,22,34] by using text-to-image models such as Stable Diffusion.

Response:

Judging by his respective trajectories and movements: Track1<Id1>Frame1:[50,66,115,191];Frame2:[65,60,124,205];Frame3:[62,70,136,214]. This player will be tackled by another player in a white jersey:Track2<Id2>Frame1: [35,58,119,203];Frame2:[13,52,113,200];Frame3:[1,72,101,205]</Id2>. And both players will fall to the ground.

|[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>Trajectory Visualization|
|---|

Judging by his ...

###### a video clip tell me what will happen

...

[Figure 45]

[Figure 46]

Large Language Model

Given a video clip tell me what will happen to this player .

[Figure 47]

[Figure 48]

[Figure 49]

What will happen to this player?

|| |
|---|
|
|---|

[Figure 50]

[Figure 51]

Location:Frame1:[20,46,145,191] Appearance: The red player Action: Running on the rugby field

Projector

Vision Encoder

Show me the trajectory please.

First frame

Multiple Images Contexts Initial Observation

User Prompt

- Fig. 3: Overall pipeline of Merlin. The architecture of Merlin consists of three main components: (1) an image encoder, (2) a large language model, and (3) a modality-align projector. Bottom: The diverse input format that supports multiple-image contexts, initial observation and the specific user prompt. Top: The model response including the predicted trajectory and the future reasoning.

### 3 Metholodgy

In this section, we introduce the design details of Merlin, encompassing its overall architecture (Section 3.1), and the two-stage training pipeline of foresight learning, including Foresight Pre-Training (Section 3.2) and Foresight Instruction Tuning (Section 3.3).

#### 3.1 Overall Architecture

Merlin is designed to unlock the foresight minds based on observations from single images and multi-frame video clips. In order to accomplish this, images and videos are comprehensively represented through a series of visual tokens, which are then integrated into the language sequence that can be comprehended by Large Language Models (LLMs) in a unified framework. Specifically, Merlin consists of an image encoder, a decoder-only LLM, and a modality alignment block as illustrated in Figure 3. Following prevalent practice [10,41,43,86], we opt for the pre-trained CLIP [53] ViT-L/14 [19] as the visual encoder and Vicuna-

- 7B v1.5 [11] as the large language decoder. For more details, please refer to our supplementary materials.

To provide enough visual information and details, the input images are resized to a resolution of 448×448. At this juncture, the visual encoder iteratively attends to (448/14)2 uniformly divided image patches, yielding 1024 encoded tokens. Considering the limited context length of LLMs and addressing the substantial computational challenges posed by high resolution and multi-frame context modeling, we simply utilize a 2D convolution to achieve both dimension projection and token aggregation [7,57].

We choose 2D convolution over 1D linear layers [10,41,43] or cross-attention layers [4,38,86] as connector for the following reasons: (1) 2D convolution clusters local visual tokens on a spatial scale [24], effectively achieving a one-step transformation from spatial to channel information; (2) The good convergence properties [31,63] of 2D convolution compared with cross-attention lay a solid foundation for foresight learning in a two-step training approach.

#### 3.2 Foresight Pre-Training

Generative Pre-Training (GPT) [8,49,50] serves as the cornerstone of this generation’s Language Models (LLMs). Through learning to predict next token, the model efficiently condenses data, thereby yielding emergent forms of intelligence [71]. In this context, a very natural approach to enhance the model’s perception of the dynamic clues across multiple frames is to explicitly model the next frame (or image). However, due to the high redundancy in multi-frame visual information, the truly next-frame prediction remains a significant challenge to date. A better approach at this juncture is to implicitly model high semantic information in the label space (such as categories, bounding boxes) on a frameby-frame basis. Temporally, this label information forms a trajectory, a highly structured representation. Causally modeling the trajectory in conjunction with each frame of image helps to connect the past and present in time, thus enabling the model to perceive the future.

To this end, we propose the Foresight Pre-Training, a way of causally modeling the trajectories interleaved with multi-frame images, to empower the MLLM with the capacity of perceive the dynamic clues, and ultimately achieving future reasoning. Specifically, given a video clip including several frames, we first give the model the observation of the first frame, then we require the model to predict the whole trajectory of the concerned subject in this video conditioned on the initial observation. Notably, the observation of the first frame can be the description or simple position of the concerned object. Formally,

P(Y |X) ∼ P(Y |{X1,X2,...},Ofirst), (1)

where Xi denotes the ith frame and Ofirst is the first frame observation, Y refers to the trajectory of the subject in Ofirst within the frame sequence. The observation and the raw frames will be regarded as the condition to prompt MLLM to predict the trajectory.

Data Construction. We first aggregate all valuable multimodal information from diverse data resources and then properly organize them for multi-task foresight pre-training. Specifically, for each sample instance I, we first collect its multimodal information including consecutive multi-frame images {X1,X2,...}, subject observations from the first frame Ofirst, and subject trajectory Y constructed from all frames. Formally,

I = {{X1,X2,...},Ofirst,Y }. (2)

We categorize observations of one subject of the first frame into three types: location description, appearance description and action description. Then we

###### An Example of FPT dialogue

###### Interleaved Multiple Images: Initial Observation:

###### Randomly choice

Given a video clip including:

Location: Frame1:[562, 342, 926, 561]

[Figure 52]

[Figure 53]

[Figure 54]

Appearance: A panda on the right side.

\n,frame2: \n,frame3: \n

frame1:

| |
|---|

Action: A panda is lying on the ground.

Dialogue: Question:

Interleaved Multiple Images, can you tell me the trajectory according to the initial observation ? To respond correctly, utilize the specified <Idi>Frame t:[xmin,ymin,xmax,ymax]</Idi> format.

###### Answer:

Its trajectory is <Id1>Frame1:[562,342,926,561]; Frame2: [576,334,960,568];Frame3:[632,366,979,589]</Id1>.

- Fig. 4: One example to illustrate the multi-modality pretraining dataset. The top block shows the provided contexts including the multiple images contexts and initial observation (box, appearance and action) about the subject to prompt the LLM. The bottom block shows the dialogue including question and answer.

randomly select one of these observations of a particular subject in the first frame as the query object.

To better unleash the powerful generative modeling capacity of LLM, we construct this query process as a type of conversation. Here is an example of the constructed data shown in Figure 4. In this case, we want to query the subject — the panda on the right — with the randomly select observation, and expect the answer with the movement trajectory of this panda across multiple frames. To model this process, we convert the query to question and trajectory to answer with proper natural language for embellishment.

Overall, the aforementioned process of dialogization roughly follows these three principles: (1) Precise definition of task prompts and answer formats. In particular, we use a task prompt to tell MLLM what specific task to do (detect or track), and also specified the answer format with accurate descriptions in each question. In this way, different types of tasks can be flexibly organized together without compromising the general language ability. (2) Clear indication of multimodal information. Concretely, for each group of image tokens, we add a special frame indicator in front of then, i.e., frame1:<image> and frame2:<image>, so as to help MLLM better focus on the corresponding image. (3) Interleaving of frames and observations. For the same identity, we interleave the frames in which it appears with its positional observations, and enclose them with two ID tokens (i.e. <Idi> and </Idi>) to construct a trajectory. We believe that this interleaved organization helps in generatively training to model causality within the trajectory, while the ID tokens ensures that the model can distinguish among different identity objects.

Training Details. The objective in this stage is to initially endow MLLMs with the capacity of modeling the spatiotemporal dynamics across multi-frame images, while ensuring that its general language capabilities do not diminish. Previous practices [4,42,43] typically conducting a separate modality alignment training phase following a multi-task pre-training stage, which however, com-

plicates the training process and data construction. In this paper, we directly incorporate both of them into one stage, and unfreeze all modules during pretraining. This is because that we believe the MLLMs are sufficiently powerful to concurrently handle the learning of general multimodal capabilities and multitask specific abilities under proper guidance. Furthermore, we mix a large amount of image-text pairs and rich-annotated conversation data (formatted according to the above method) from diverse data sources [20,28,33,48,61,64,66,79] to conduct multi-task learning. In doing so, not only endows the model with foresight minds but also ensures its multimodal alignment.

#### 3.3 Foresight Instruction Tuning

Althought Foresight Pre-Training equips the model with the ability to observe dynamic clues across multiple frames, it still falls short of true foresight minds. This is because models typically struggle to effectively transform such observations into successful future reasoning [70,85].

Recent work [52, 85] has highlighted that Chain-of-Thought (CoT) [72] is crucial in bridge the gap between the observations and actions of MLLMs with theory of mind [58,70]. Meanwhile, several prior studies [10,83] have also demonstrated that prompts indicating position (such as bounding boxes or points) a principle analogous to CoT — can concentrate an MLLM’s attention on the relevant area, leading to more accurate dialogues and reducing the likelihood of visual hallucination. Drawing inspiration from these findings, we conduct the Foresight Instruction Training (FIT) building upon the foundation of FPT to further enhance the model’s future reasoning capability. In specific, building on the trajectory generating powered by FPT, we further union the trajectories to generatively rationalize the forthcoming events. Mathematically,

P(Z|X,Y ) ∼ P(Z|{X1,X2,...},Ofirst,Y ), (3)

where Z refers to the future observation which is deduced from observations in each frame. It can be actions, events, trends, or simply likelihoods. In this context, multi-frame images, in conjunction with the first subject observation, and the trajectory of the same subject across all frames, serve as the union condition to prompt MLLM to causally predict the future. This way, akin to a Trajectory Chain-of-Thought, effectively bridges the gap between trajectory perception and predictive future reasoning, thereby fully unleashing model’s foresight minds.

Data Construction. The specific data construction method is similar to FPT, but on this basis, we also deduce a future observation Z from the information across multiple frames and append it after the trajectory in the answer. Formally,

I = {{X1,X2,...},Ofirst,Y,Z}. (4) Practically, in this paper, we constitute future observations based on multi-

frame, multi-target action descriptions combined with human priors, and further

process them with GPT-4 [50] to ultimately form reasonable future inferences. More details are provided in the supplementary materials.

Figure 3 provides an illustrative example of FIT, when a user questions Merlin about the future of a player in red attire, Merlin initially presents the observed trajectory of the concerned player, followed by the trajectory of another player in white. Using these trajectories, Merlin deduces that the player in white is likely to tackle the one in red, resulting in both players falling to the ground.

Training Details. We freeze the vision encoder and keep the convolutional projector and the LLM unfreezed in this stage. On this basis, we primarily adopt the open-source instruction tuning datasets, i.e., LLaVA-665K [41], for building the basic ability for multi-round visual-language conversation. For further unleash the foresight minds of model, we first uniformly sample a certain number of multitask dialogues in FPT, in order to maintain the model’s capacity of modeling the dynamic clues across multi-frame images. In addiction, we also sample data from three specific scenario datasets [40,47,73] and construct around 60K FIT conversations based on the aforementioned data construction process.

### 4 Experiment

#### 4.1 Experimental Settings

Datasets. For the foresight pre-training (FPT) stage, we first use 10M imagetext pairs sampled from LAION400M [59] to ensure multimodal alignment. On this basis, we gather various open-source datasets with rich annotations to conduct multi-task learning, including (1) object detection datasets: Object365 [61] and OpenImage [35]; (2) tracking datasets: LaSOT [20], GOT10K [28], MOT17 [48], DanceTrack [66] and SOMPT22 [64]; (3) grounding dataset: RefCOCO [33]; (4) object relation dataset: VCR [79]. For these data, as described in Section 3.2), we apply strict task definitions and format specifications, and re-organize them in the form of interleaved frames and observations. Ultimately, we obtain approximately 5M question-answer data, which are mixed with 10M paired data for foresight pre-training.

For the foresight instruction-tuning (FIT) stage, we mix approximately 730K conversation data, including (1) open-source instruction-tuning data LLaVA665K [41], which integrates a series of VQA datasets [65] and multi-round conversation datasets [43]; (2) around 30K FIT multi-frame conversations constructed from three specific scenarios including MultiSports [40], TITAN [47] and STAR [73] based on the data construction method described in Section 3.3; (3) nearly 40K randomly sampled FPT multi-task data. For more details of the datasets, please refer to the supplementary materials.

Implementation Details. As outlined in Section 3.1, Merlin utilizes the CLIPViT-L/14 [53] as its vision encoder for image encoding and the open-source Vicuna-7B v1.5 [11] for foresight decoding. Between them, a 3 × 3 convolution layer with padding set to 1 and a stride of 2 is employed for both dimension projection and token aggregating. During the foresight pre-training, we optimize

- Table 1: The Effectiveness of Prediction Reasoning. We mainly select 5 metrics from MMBench develop and test set, respectively, including OL: Object localization (Prediction), PPR: Physical property reasoning, FR: Function reasoning, IR: Identity reasoning, and FP: Future prediction. Avg. denotes the average score. The best and second-best performances are shown in bold font and underlined respectively.

Prediction Reasoning (Dev.) Prediction Reasoning (Test) Avg. OL PPRFR IR FP Avg. OL PPRFR IR FP

Method LLM Size

InstructBLIP [14] 13B 42.0 14.8 30.7 56.8 88.9 19.0 44.4 5.7 24.0 67.3 92.7 32.4 MiniGPT-4 [86] 13B 43.3 28.4 30.7 49.4 86.7 21.4 48.9 21.0 35.0 67.3 90.2 31.1 OpenFlamingo [3] 7B 5.28 2.5 10.7 8.6 2.2 2.4 11.5 2.9 14.0 9.3 11.0 20.3 MMGPT [44] 7B 19.5 1.2 24.0 9.9 60.0 2.4 16.8 3.8 13.0 12.1 52.4 2.7 MiniGPT-4 [86] 7B 26.8 7.4 14.7 19.8 80.0 11.9 27.9 8.6 13.0 29.9 61.0 27.0 InstructBLIP [14] 7B 34.8 6.2 17.3 51.9 84.4 14.3 39.0 2.9 17.0 52.3 78.0 44.6 LLaVA [43] 7B 38.7 8.6 25.3 53.1 77.8 28.6 39.7 13.3 35.0 48.6 82.9 18.9 mPLUG-Owl [76] 7B 41.0 18.5 18.7 66.7 86.7 14.3 45.9 16.2 23.0 59.8 91.5 39.2 Shikra [10] 7B 51.5 32.1 30.7 63.0 88.9 42.9 60.0 27.6 50.0 70.1 92.7 59.5 Kosmos-2 [29] 1.6B 54.4 38.3 33.3 56.8 91.1 52.4 58.2 40.4 30.0 65.4 89.0 66.2 LLaVA-1.5 [41] 7B 59.6 43.252.0 71.6 93.3 38.1 - - - - - -

Merlin (Ours) 7B 64.4 42.0 54.7 72.897.854.8 66.5 41.351.0 83.097.659.7

all parameters of the model, setting the learning rate to 5e − 5 and training for one epoch. In the instruction tuning stage, we freeze the visual encoder and finetune the parameters of the projector and LLM. In both stages, we train Merlin using the AdamW [46] optimizer and a cosine annealing scheduler [45] as the learning rate scheduler. The entire training process is conducted on 64 NVIDIA A800 GPUs, with approximately 12 hours required for pre-training and 3 hours for instruction-tuning. Additional implementation details can be found in the supplementary materials.

#### 4.2 Properties Evaluation of Foresight Minds

In this section, we mainly verify the foresight minds (future reasoning) of Merlin from two aspects, i.e., prediction reasoning and identity association ability, where the former focuses on forecasting and reasoning location, events or behavior based on image observation, and the latter focuses on the model’s ability to establish subject identity associations across multiple frames to obtain dynamic clues for future reasoning.

Prediction Reasoning. To evaluate this ability, we probe this ability based on the several sub-tasks of MMBench [44]. MMBench provides a comprehensive evaluation system to assess various capabilities of MLLM, with some metrics focusing on the model’s prediction and reasoning capabilities. To this end, we pick out these metrics to establish this new future reasoning benchmark and compare Merlin with the existing SOTA models. As shown in Table 1, Merlin achieves the best overall performance (64.4 average score on the development set and 66.5 average score on the test set). Moreover, it obtains the best in 8/10

- Table 2: Comparison on main tracking benchmarks. Notably, the original LLaVA-1.5 [41] model was incapable of performing tracking tasks. Therefore, we utilized the model configuration of LLaVA-1.5 and trained a version of the model with the same dataset as Merlin for the fair comparsion.

Table 3: Comparison with SOTA methods on main MLLM benchmarks. For VQA tasks, we mainly choose GQA [30] and VisWiz [26] to evaluate the model; For general evaluation, we mainly choose MMBench [44] and MM-Vet [78]. †Includes using in-house data that is not publicly accessible.

LaSOT GOT10k

Method Success Pnorm P AO SR0.5 SR0.75

Specialist Models

SiamFC [6] 33.6 42.0 33.9 34.8 35.3 9.8 ATOM [15] 51.5 - - 55.6 63.4 40.2 SiamRPN++ [37] 49.6 56.9 49.1 51.8 61.8 32.5 SiamFC++ [74] 54.4 62.3 54.7 59.5 69.5 47.9

Generalist Models

LLaVA-1.5 [41] 19.4 16.5 12.8 23.5 20.2 9.7 Merlin (Ours) 39.8 40.2 38.1 51.4 55.9 42.8

VQA Task Generalist Method GQA VisWiz MMBd MMBt MM-Vet BLIP-2 [38] 41.0 19.6 - - 22.4 InstructBLIP [14] 49.2 34.5 36.0 33.9 26.2 Shikra [10] - - 58.8 60.2 IDEFICS-9B [36] 38.4 35.5 48.2 45.3 IDEFICS-80B [36] 45.2 36.0 54.5 54.6 Qwen-VL† [4] 59.3 35.2 38.2 32.2 Qwen-VL-Chat† [4] 57.5 38.9 60.6 61.8 LLaVA-1.5 [41] 62.0 50.0 64.3 59.5 30.5 Merlin (Ours) 60.5 50.4 66.2 65.5 34.9

indicators and ranks second in all other indicators, which favorably demonstrates Merlin’s strong predcition and reasoning ability.

Identity Association. We examine this ability by evaluating the performance of object-tracking tasks, which can comprehensively demonstrate object association and prediction capabilities. To this end, we evaluate Merlin in existing mainstream tracking benchmarks, i.e., LaSOT [20] and GOT10K [28]. It is worth noting that Merlin is the first MLLM that can also carry out tracking tasks. As shown in Table 2, Merlin achieves comparable performance with expert models and even outperforms on some metrics. Notably, we only sample a small amount of tracking data to train Merlin instead of the full amount of data, which means LLM exhibits significant potential in handling temporal tasks, possibly because tracking, as a temporal task, can be viewed as a casually frame-level autoregressive task.

#### 4.3 General Comprehension

In order to showcase the general multi-modal ability, we further benchmark Merlin on various VQA benchmarks and recent benchmarks proposed for evaluating the comprehensive capabilities of MLLMs.

Visual Question Answering (VQA). We first evaluate Merlin on several mainstream VQA benchmarks to reflect the perceptual abilities of MLLMs in understanding image content. As shown in Table 3, Merlin achieves competitive performance compared with existing advanced MLLMs in the selected VQA benchmarks (VQA). The results indicate that Merlin possesses strong image understanding and question-answering capabilities.

Synthetica MLLM Benchmarks. Recently, several benchmarks have been proposed to evaluate the comprehensive performance of MLLMs, encompassing diverse finer-grained scenarios including visual perception, object recognition, optical character recognition (OCR), future reasoning, and so on. In this part,

##### Table 4: Zero-shot object hallucination evaluation on the COCO validation set. “Yes” represents the proportion of positive answers that the model outputs.

Random Popular Adversarial

Method LLM Size

Accuracy F1-Score Yes Accuracy F1-Score Yes Accuracy F1-Score Yes

LLaVA [43] 13B 64.12 73.38 83.26 63.90 72.63 81.93 58.91 69.95 86.76 MiniGPT-4 [86] 13B 79.67 80.17 52.53 69.73 73.02 62.20 65.17 70.42 67.77 InstructBLIP [14] 13B 88.57 89.27 56.57 82.77 84.66 62.37 72.10 77.32 73.03 Shikra [10] 13B 86.90 86.19 43.26 83.97 83.16 45.23 83.10 82.49 46.50 MultiModal-GPT [23] 7B 50.10 66.71 99.90 50.00 66.67 100.00 50.00 66.67 100.00 mPLUG-Owl [76] 7B 53.97 68.39 95.63 50.90 66.94 98.57 50.67 66.82 98.67 LLaVA [43] 7B 72.16 78.22 76.29 61.37 71.52 85.63 58.67 70.12 88.33 LLaVA-1.5 [41] 7B 83.29 81.33 - 81.88 80.06 - 78.96 77.57 Qwen-VL [4] 7B 84.73 82.67 - 84.13 82.06 - 82.26 80.37 -

Merlin (Ours) 7B 91.58 91.66 49.38 89.53 89.56 50.27 84.10 84.95 55.63

we select several mainstream MLLM benchmarks to evaluate Merlin. As shown in Table 3, We present performance in accuracy on benchmarks including MMVet [78] and MMBench [44]. On MMBench, we report results on the both development and test sets. The results show that Merlin significantly outperforms comparative methods, even though many methods utilized a substantial amount of in-house data for pre-training, or employed several times more parameters. This implies that, while introducing foresight minds into MLLMs, we not only preserved their original visual capabilities but even further enhanced their overall level of visual perception.

#### 4.4 Object Hallucination

Hallucination presents a significant challenge in existing MLLMs. This term describes the phenomenon where the generated textual content exhibits inconsistencies when compared to its corresponding image content. In this section, we present the experiments from the Polling-Based Object Probing Evaluation (POPE [39]). As demonstrated in Table 4, Merlin surpasses recent SOTA methods with clear margins. More specifically, Merlin achieves optimal performance in all metrics across three scenarios: Random, Popular and Adversarial, with improvements of up to 5 points compared to the highly competitive baseline Shikra [10]. Surprisingly, in multiple scenarios, the ’yes’ rate of Merlin is quietly close to 50%, demonstrating its extraordinary visual perception capabilities.

We analyze this success largely owing to the proposed foresight learning (FPT and FIT). By enabling the model to learn the dynamic correspondence between trajectories across multiple images, the model has gained a more precise ability to attend to relevant object (trajectories) contexts in the image, which helps to better avoid misidentification and misalignment of irrelevant targets. We believe that this result will provide new thinking about addressing the issue of hallucinations in MLLM.

- Table 5: Ablation study of the proposed strategies in Merlin. (ITP: Image-text pair data, ITD: instruction-tuning data). We mainly report the AO score of GOT10k and the average score of future reasoning.

Pre-Training Inst.-Tuning GOT10K Prediction Rea. ITP FPT-Data ITD FIT-Data AO Averagedev

- ✗ ✗ - 59.5

- ✗ - 60.7

✗ 15.5 52.8 ✗ 51.4 61.2 51.4 64.4

- Table 6: Ablation studies of the model settings including resolution, vision encoder and projector of Merlin.

|Exp<br><br>|Resolution Projector Visual Encoder<br><br>|Tokens Num<br><br>|Prediction Rea. Got-10K|
|---|---|---|---|
|➊ ➋ ➌ ➍<br><br>|448x Conv2d unfrozen 336x Conv2d unfrozen 336x MLP unfrozen 448x Conv2d frozen|256 256 576 256<br><br>|64.4 51.4<br><br>59.8 47.3 58.1 23.5<br>60.8 28.4<br>|

#### 4.5 Ablative Analysis of FPT & FIT

As introduced in Section 3.2 and Section 3.3, FPT serves as the pre-training strategy to enable MLLM to encapsulate dynamic information across frames by predicting the trajectory of the next frame. FIT is designed to activate the ability of foresight minds in a way of Trajectory CoT during instruction finetuning. To further explore the effect of FPT and FIT, we conduct an ablation study based on the established future reasoning benchmark and tracking dataset GOT10K [28]. As shown in Table 5, we mainly report the average overlap (AO) of GOT10K and the average score of future reasoning in the development set.

The results show that both FPT and FIT training strategies contribute to the improvement of the metrics. Combining both FPT and FIT, Merlin achieves the best performance which proves the effectiveness of the proposed strategies. Furthermore, we can also observe that the lack of image-text pair data during the pre-training stage considerably hampers the model’s general ability. This phenomenon supports our perspective that, during the comprehensive pre-training phase, the integration of image-text pair data is essential for maintaining modality alignment and preventing a decline in combined capabilities.

#### 4.6 Ablative Analysis of Model Configuration

The configuration of model architecture for large-scale models is also a focal point of interest for researchers. In this subsection, we specifically investigate the impact of Merlin’s model configuration on performance. As depicted in Table 6, we focus on examining the effects of model input resolution, the visual encoder

[Figure 55]

[Figure 56]

[Figure 57]

###### This player <Id1> Frame1: [461, 278, 651, 976]; Frame2: [458, 254, 609, 968]; Frame3: [442, 254, 607, 918] </Id1> is

dribbling the ball while being pursued by the opponents.

- Fig. 5: Attention map visualization. To facilitate the observation, we map the attention between the box responses and the visual tokens of each frame for visualization.

of the model, and the model’s projector on the ultimate performance of Merlin. From the experimental outcomes, we can draw the following conclusions:

- (i) High-resolution input is more conducive to visual perception and understanding tasks (row ➊ and ➋), particularly for tasks that require precise localization, such as detection and tracking.
- (ii) The primary contribution of Conv2d is the ability to compress the number of tokens efficiently and elegantly, which is crucial for supporting high-resolution images. In contrast, MLPs cannot compress tokens. This high token count hinders the training with multiple images. Moreover, more visual tokens does not improve performance in future reasoning tasks (row ➊ and ➌). We speculate that an increased number of visual tokens may lead to the sparsity of supervision.
- (iii) During the pre-training phase, the visual encoder should be unfrozen (row

➊ and ➍), which is beneficial for modal alignment and the expansion of the fine-grained spatial information. Similar conclusion is also claimed in [9].

#### 4.7 Visualization Analysis

In this subsection, we visualize the attention map of Merlin to further substantiate the effectiveness of utilizing the proposed strategies. As shown in Figure 5, we select the output attention map of the middle-level layers of LLM for visualization. We can observe that the word embedding of the output trajectory coordinates can attend to the corresponding object from different frames correctly. This visualization results further prove that the trajectory representation is a good interface to enable MLLM to establish the alignment between the language description and the multi-images dynamic visual contexts. Furthermore, this effectively explains why Merlin possesses a more powerful comprehensive visual capability and a greatly lower level of hallucination compared to previous baselines. Indeed, the trajectory-driven foresight learning allows the large language model to read images more profoundly!

### 5 Limitation and Conclusion

This study highlighted an obvious deficiency in Multimodal Large-Language Models (MLLMs), specifically their ability to predict future events or outcomes based on current observations, referred as “foresight minds”. To address this,

we serve as the first to point out that trajectory, as a highly structured representation, is a good learning objective to assist MLLM in obtaining dynamic information from the image observations. Based on this insight, we introduced a unique training method including Foresight Pre-Training (FPT) and Foresight Instruction-Tuning (FIT). By synergizing FPT and FIT, we created Merlin, a unified MLLM that effectively understands and outputs spatial coordinates or tracklets from single images or multiple frames. Merlin excels at a range of traditional vision-language tasks while demonstrating powerful future reasoning capacities. Despite the substantial advancements made by Merlin, there still are some limitations, particularly in processing long sequential videos and more comprehensive future reasoning evaluation. We aspire for Merlin to guide the enhancement of more advanced MLLMs in the future.

### Acknowledgements

This work was supported by the National Natural Science Foundation of China under Grant 62176096 and Grant 61991412.

### A Appendix

In this supplementary material, we offer additional information about Merlin due to the paper’s page limit of pages. Specifically, Section B provides in-depth insights into the dataset we constructed, including its data sources and how it was created. Section C delves into the data formats of the proposed FPT and FIT, as explained in the main manuscript. Section D offers a more detailed explanation of the training approach. Section E expands on the discussion of the proposed method, including its limitations and future directions. Finally, Section F offers additional experimental results and demo visualizations.

### B Dataset Details

In Section 3 of the manuscript, we explained how we created the Foresight PreTraining (FPT) and Foresight Instruction-Tuning (FIT) datasets. Now, in this section, we go into greater detail about how we collected and built the dataset. To start, we provide an overview of our collected data in Table 7, and then dive into the step-by-step process of how it was constructed.

Image-text pairs. We mainly collect image-text pairs dataset from the opensourced dataset, LAION-400M [59]. LAION-400M is a high-quality dataset with CLIP-filtered 400 million image-text pairs. In this paper, we random sample 10M image-text pairs of LAION-400M for the Foresight Pre-Training.

Detection datasets. Detection datasets are crucial for improving the model’s ability to understand space during its initial training phase. In our research, we used datasets from various publicly available sources like Object365 [60],

- Table 7: All training data. Cap.:Captioning, Ref.:Referring (including REC, REG and Referring Tracking), Det.: Detection, Track:Tracking (including single object tracking (SOT) and multiple object tracking (MOT)), Rea.:Reasoning, Dia.:Dialogue.

* means that the data is only used in the SFT stage.

##### Task Data Size Task description example Cap. LAION 10M –

RefCOCO 200K Refer to the supplied image and box coordinates for a short response to the question.

Ref.

MeViS 180K Provide the trajectory coordinate of the target according to the sentence describes.

Object365 3.2M When submitting your answer, maintain the

Det.

OpenImage 800K CrowdHuman 20K

category:[xmin,ymin,xmax,ymax] structure consistently.

GOT10K 100K

LaSOT 15K MOT17 10K Sompt 5K DanceTrack 25K SportsMOT 20K BDD100K 120K

For the trajectories included in the answer, please use the format <Idi>Frame t: [xmin,ymin,xmax,ymax]</Idi>.

Track

Rea. VCR 250K In response, account for any relevant ob-

ject locations, denoted by [x0,y0,x1,y1]. Dia.*

LLaVA-Ins 665K –

Merlin-chat 30K

OpenImage [35], and CrowdHuman [62]. We processed these datasets in the following ways:

- (1) Extract all objects in each image along with their categories and bounding boxes coordinates.
- (2) Remove too small objects (smaller than 1/32 of the image size).
- (3) Randomly select fixed number (n = 15) of categories if image has more categories than a set limit.
- (4) Compose the detection data format as the following: cat1:<box>,<box>,<box>;cat2:<box>;... . As shown in Table 7, we also created special output prompts that guide the

model to respond in a precise format, as detailed in our study.

Tracking datasets. Building tracking data is a key part of our Foresight PreTraining (FPT) method, as we’ve discussed in our main paper. FPT is designed to causally modeling the trajectory in conjunction with each frame of the image to empower the model to acquire foresight minds. The tracking data naturally includes trajectory information about movement paths, making it ideal

###### Examples of Data Formats

###### Image-Text Pairs

<im_start>LAION400M/Key-094857437.jpg<im_end>\nGo Compare Car Insurance Groups by Moneysupermarket On The Forbes Innovative Growth.<eos>

###### General Tracking

###### General Detection

Given frame1:<im_start>dancetrack0004/img1/00000001.jpg<im_end>\n,frame2: <im_start>dancetrack0004/img1/00000006.jpg<im_end>\n,track person<Id1>Frame1:

<im_start>Object365/v2/patch16/objects365_v2_00900001.jpg<im_end>\ndetect all objects.Maintain strict adherence to the format category:[xmin,ymin,xmax,ymax] when presenting your answer. Answer:Person:[0,374,472,1000],[559,382,902,766];Necklace: [148,593,341,651];Mirror:[387,119,1000,836].<eos>

- [497,401,655,918]</Id1>,person<Id2>Frame1:[325,351,433,933]</Id2>.Keep your response consistent with the class<Idi>Frame t:[xmin,ymin,xmax,ymax]</Idi> format.Answer:person<Id1>Frame1:[497,401, 655,918];Frame2:[506,491,671,912]</ Id1>,person<Id2>Frame1: [325,351,433,933];Frame2: [295,409,445,975]</Id2>. Referring Tracking

Given frame1:<im_start>MeViS/6084240e75fa/00055.jpg<im_end>\n,frame2:<im_start> MeViS/6084240e75fa/00060.jpg<im_end>\n and frame3:<im_start>MeViS/6084240e75fa/ 00060.jpg<im_end>\n,point out the trajectory of the panda rolling around.Your trajectory details should strictly align with the Tracki<Idi>Frame t:[xmin,ymin,xmax,ymax]</Idi> format in your submission. Answer:Track1<Id1>Frame1:[483,658,275,340]; Frame2:

- [498,714, 225,283];Frame3:[481,725,223,269]</Id1>.

###### Open-Vocabulary Detection

<im_start>Object365/v1/patch8/objects365_v1_00420917.jpg<im_end>\ndetect parking meter,shelf and bakset.When composing your answer, be sure to consistently utilize the category:[xmin,ymin, xmax,ymax] structure. Answer:Shelf:[190,247,242,392],[266,293, 318,401]; Bakset:[4,213,10,254] ,[9,210,146,251].<eos>

###### Referring Expression Comprehension

<im_start>COCO_train2014_000000310289.jpg<im_end>\ntell me the position of the giant doughnut with white icing and red , white , and blue sprinkles. Utilize the [xmin,ymin, xmax,ymax] format in your answer. Answer:[523,621,816,937].<eos>

Given frame1:<im_start>MeViS/8e78a5a2978a/00014.jpg<im_end>\n,frame2:<im_start> MeViS/8e78a5a2978a/00015.jpg<im_end>\n and frame3:<im_start>MeViS/8e78a5a2978a/ 00020.jpg<im_end>\n,tell me what this object<Id1>frame:1:[340,347,407,391];frame:2: [362,329,428,365];frame:3:[412,310,503,389]</Id1> is doing?Given the image frames and trajectory coordinates, please respond succinctly to the question. Answer: This is a person leaning towards the left while riding motorized-bicycle

###### Referring Expression Generation

<im_start>COCO_train2014_000000106755.jpg<im_end>\ntell me what can you see in this region[131,238,675,578]? Craft a concise reply using the image and box coordinates you have at hand. Answer:I can see a guy with his back turned to us.<eos>

###### Visual Reasoning

<im_start>VCR/lsmdc_3038_ITS_COMPLICATED/3038_ITS_COMPLICATED_00.08.14.616-00.08.17.313@0.jpg<im_end>\nKindly review the image and provide a thorough answer to my question.Why is [337,5,452,928] smiling at [0,9,434,889]? Answer: Because [337,5,452,928] has spinach in her teeth .<eos>

###### Merlin-Chat

Given a video clip including frame1:<img_start>MultiSports/basketball/v_4r8QL_wglzQ_c009/frame81.jpg<img_end>/n,frame2:<img_start>MultiSports/basketball/v_4r8QL_wglzQ_c009/ frame84.jpg<img_end>/n and frame3:<img_start>MultiSports/basketball/v_4r8QL_wglzQ_c009/frame87.jpg<img_end>/n,what observations can you make from the video clip?

In this video clip, a captivating basketball game is visible, where athletes are fully engrossed in a highly intense contest. This competition, marked by skillful plays and fervent energy, is actively unfolding right before our eyes.

Could you tell me what this player<Id1>Frame1:[401, 506, 479, 662]</Id1>, might be contemplating as their next move?Maintain consistency in your answer by using the object<Idi>Frame

t:[xmin,ymin,xmax,ymax]</Idi> format for trajectories.

Given the trajectory and action of this player<Id1>Frame1:[401, 506, 479, 662];Frame2:[374, 495, 457, 669];Frame3:[353, 493, 414, 680];Frame4:[316, 484, 405, 675]</Id1>, it is evident he is readying to make a basketball pick-and-roll defensive next.

Which player is he planning to defense? This player is planning to avoid the dribble of the player<Id2>Frame1:[265, 491, 316, 675];Frame2:[257, 480, 312, 659]</Id2>.

- Fig. 6: Data format visualization. In our training, we use datasets that involve multiple tasks. To illustrate how these datasets are structured, we’ve chosen an example from each one. It’s important to mention that all information about boxes has been adjusted to a standard range of 1000. In the examples, questions are shown in black text, answers in blue, and negative samples in red.

for our FPT task. We gathered a variety of open-source tracking data, including Single Object Tracking (SOT) and Multi-Object Tracking (MOT) datasets like GOT10K [28], LaSOT [20], MOT17 [48], Sompt [64], DanceTrack [66], SportsMOT [13], BDD100K [77], MeViS [17]), etc. We follow the steps below to pre-process the tracking data.

- (1) Random sample 3 to 5 frames at a certain frame interval (gap = 3,4 or 5) to form a video clip each time.
- (2) Extract all trajectories with their category, identity and bounding boxes in each video clip.
- (3) Remove the trajectory containing too small objects (smaller than 1/32 of the image size).
- (4) Select the initial observation (location, appearance or action in the first frame) as the trajectory query.
- (5) Compose the tracking data format as the following: query,cat1<Idi>Frame1:<box>;Frame2:<box>;...</Idi> .

Similar to constructing detection data, we also adopt output format prompts to guide the model to provide answers in the desired trajectory format.

Visual Reasoning datasets. Visual reasoning dataset is constructed to enhance the common sense reasoning ability of the model. In this work, we mainly collect the VCR dataset and adopt the same processing method as Shikra [10].

Referring datasets. For referring dataset, we mainly collect from RefCOCO [33]

and MeViS [17] to construct image referring and video referring datasets, respectively. For image referring, we simply extract the description and the corresponding bounding box from each image. For video referring, we primarily extract pairs of trajectories and trajectory descriptions from MeViS.

Dialogue datasets. To facilitate the model’s ability to achieve long conversations and follow user instructions, we utilized the open-sourced LLaVA-665K instruction tuning dataset [41]. Additionally, we created a smaller instruction tuning dataset (30K entries) using our Trajectory Chain-of-Thought (T-CoT) method within Foresight Instruction Tuning. This dataset focuses on three specific contexts: MultiSports [40], TITAN [47], and STAR [73]. MultiSports catalogs multi-person sports actions with spatial and temporal details. TITAN encompasses street scenes with varied labels like vehicle states, pedestrian age groups, and specific pedestrian actions. STAR offers real-world video data with symbolic descriptions and logic-based analysis.

To create the T-CoT conversation data, we leveraged GPT-4 to generate rich feature reasoning dialogues using the trajectories and actions from these datasets. We guided GPT-4 to interpret video clips through these trajectories and their associated descriptions or actions, then instructed it to formulate Q&A pairs. These questions were designed to derive answers from available information. This process resulted in 30K QA pairs, incorporating trajectory coordinates in both questions and answers. We plan to expand this dataset, which we refer to as Merlin-chat, in the future.

### C Data Format Details

To assist readers in comprehending the specific nature of the data we’ve constructed, this section includes visualizations of the training data format. As illustrated in Figure 6, we present an exhaustive list of all data formats used, encompassing both pretraining and instruction tuning phases. Notably, we have incorporated negative samples (highlighted in bold red) into our question-answer pairs. This addition is designed to teach the model the skill of responding with a negative answer when appropriate, thereby bolstering its ability to resist producing hallucinatory responses.

### D Training Details

In this section, we report the detailed training hyperparameter settings of Merlin in Table 8.

- Table 8: Training hyperparameters of Merlin. The hyperparameter placed in the middle indicates that this hyperparameter is used in both stages.

Configuration Pre-training Supervised Fine-tuning

ViT init. OpenAI-CLIP-L/14 Merlin FPT LLM init. Vicuna-7B-v1.5 Merlin FPT Projection init. random Merlin FPT Image resolution 4482 4482 ViT sequence length 2048 2048 LLM sequence length 2048 2048 Optimizer AdamW Optimizer hyperparameter β2 = 0.95, eps = 1e−8 Peak learning rate 5e−5 Minimum learning rate 0 ViT learning rate decay 0.9 0 ViT Drop path rate 0 Learning rate schedule cosine decay Weight decay 0.05 Gradient clip 1.0 Training steps 7k 3k Warm-up steps 70 90 Global batch size 2048 256 Gradient Acc. 8 1 Numerical precision bfloat16 Optimizer sharding ✓ Activation checkpointing ✗ Model parallelism ✗ Pipeline parallelism ✗

- Table 9: Comparisions between Merlin with (w) and without (w/o) the precise task description. We minaly report the AO score of GOT10K and the average score of Future Reasoning.

Setting Metrics Precise Task description GOT10K Future Rea

- 1 w 51.4 64.4
- 2 w/o 28.4 62.4

### E More Discussions

Limitations and Future Work. While Merlin demonstrates exceptional foresight capabilities, it is not without its limitations in its current form. A significant constraint is the inability to support long-range video sequences (exceeding

- 8 frames), which hinders its capacity to model extended motion information. This is primarily due to Merlin’s reliance on an image encoder rather than a video encoder, resulting in an excessive number of vision tokens for the LLM

Table 10: Ablation studies of trajectory CoT

|Exp|Pretrain data SFT data Trajectory CoT<br><br>|Prediction Reasoning|
|---|---|---|
|➊ ➋ ➌|FPT-data LLaVA-665k w/o FPT-data FIT-data w/o FPT-data FIT-data w<br><br>|61.2 63.0 64.4|

to process. Addressing this, the development of a more efficient tokenizer for long-range videos emerges as a crucial area for future research. Moreover, this paper introduces a novel benchmark for future reasoning, building upon the existing MMBench framework. Currently, there is no comprehensive and accurate benchmark to evaluate future reasoning abilities thoroughly. Therefore, exploring the creation of a more robust and comprehensive future reasoning benchmark represents another significant avenue for future investigation.

### F More Experiments

In this section, we aim to provide more experimental results including ablation studies (shown in Table 9 and Table 10) and qualitative analysis (shown in Figure 7, 8 and 9).

Effect of Precise Task Description. In the method section of main paper, we emphasize the importance of precise task descriptions, which can prevent conflicts between multiple task learning and damage to general language abilities. The performance corresponding to the models with and without the precise task description is reported in Table 9. We can observe that the model without the precise task description behaves significantly poorer, especically the tracking performance (51.4% - 28.4%). The results further prove that precise task description is an important option for multi-task learning in existing MLLM.

The effectiveness of trajectory CoT (T-CoT). As shown in Tab. 10, despite the fact that FPT has endowed the model with the capability for future reasoning, the actual performance of prediction reasoning remains limited. However, with the incorporation of the T-CoT, there has been a significant enhancement in the model’s reasoning abilities. This aligns with the mainstream viewpoint on the CoT, which posits that the model’s capabilities need to be progressively guided through a step-by-step reasoning process. Furthermore, T-CoT enables the model to visually analyze the image by forecasting the trajectory, which provides richer dynamic clues for the following reasoning.

### References

1. Alayrac, J., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., Ring, R., Rutherford, E., Cabi, S., Han, T., Gong, Z., Samangooei, S., Monteiro, M., Menick, J.L., Borgeaud, S., Brock, A.,

###### General Detection

[Figure 58]

[Figure 59]

Please detect all objects.Ensure your response adheres strictly to the format category:[xmin,ymin,xmax,ymax].

[Figure 60]

###### Person:[259, 601, 343, 1000];Giraffe:[460, 001, 836, 985],[672, 306, 822, 755], [845, 359, 1000, 812].

###### OV-Detection

[Figure 61]

[Figure 62]

Detect helmet,gloves and hockey stick.When submitting your answer, maintain

the category:[xmin,ymin,xmax,ymax] structure consistently.

[Figure 63]

Helmet:[416, 022, 520, 225],[556, 066, 628, 233];gloves:[275, 335, 385, 547],[453, 445, 530, 560],[595, 348, 658, 435];hockey stick:[443, 445, 965, 898],[000, 698, 093, 745],[453, 258, 925, 445].

###### REC

[Figure 64]

[Figure 65]

A man sitting on the right bottom corner.Provide the bounding box coordinate

of the region this sentence describes.

[Figure 66]

[610, 400, 1000, 820]

###### REG

[Figure 67]

[Figure 68]

What can you see in this region [158, 623, 400, 999]? Refer to the supplied image and box coordinates for a short response to the question.

[Figure 69]

In this region, I can see a bird flying in front of a castle.

###### Image Relation

[Figure 70]

[Figure 71]

What is the difference between this girl[180, 643, 266, 944] and this girl[387, 662, 468, 946]?

The main difference between the two girls is their position in the image. The girl on the left, [180, 643, 266, 944], is standing on the left side of the image, while the girl on the right, [387, 662, 468, 946], is standing on the right side. This difference in positioning might be due to the girls' roles in the performance or the composition of the image.

[Figure 72]

- Fig. 7: More conversation visualization using Merlin in image-level tasks. We showcase additional examples where Merlin adeptly handles various image-level tasks including general detection, open vocabulary detection (OV-Detection), referring expression comprehension (REC), referring expression generation (REG), and relation reasoning.

###### General Tracking

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Given frame1:<image>,frame2:<image>, frame3:<image>,track person<Id1>Frame1:[329, 238, 420, 633]</Id1> and person<Id2> Frame1:[491, 235, 570, 628]</Id2>.To respond correctly, utilize the specified class<Idi>Frame t:[xmin,ymin,xmax,ymax]</Idi> format.

[Figure 77]

###### person<Id1>Frame1:[329, 238, 420, 633];Frame2:[310, 212, 400, 645];Frame3:[271, 202, 378, 648]</Id1>, person<Id2>Frame1:[491, 235, 570, 628];Frame2:[495, 247, 572, 637];Frame3:[481, 258, 553, 635]</Id2>.

###### Video REC

| |
|---|

[Figure 78]

[Figure 79]

[Figure 80]

| |
|---|

[Figure 81]

Given frame1:<image>,frame2:<image> and frame3:<image>,track the black cat in this video clip.Use the specified Tracki<Idi>Frame t:[xmin,ymin,xmax,ymax]</Idi> format for all trajectories in your reply.

[Figure 82]

###### There is one black cat.Track1<Id1>frame:1:[098, 181, 600, 662];frame:2:[196, 000, 642, 800];frame:3:[068, 115, 623, 786]</Id1>.

###### Video REG

|[Figure 83]|
|---|

[Figure 84]

[Figure 85]

[Figure 86]

Given frame1:<image>,frame2:<image> and frame3:<image>,what is the hulkbuster<Id1>Frame1:[000,001,682,999];Frame2:[500, 275, 999, 999];Frame3:[167, 001, 698, 999]</Id1> doing in this video clip?

The Hulkbuster<Id1>Frame1:[000,001,682,999];Frame2:[500, 275, 999, 999];Frame3:[167, 001, 698, 999]</Id1> is a robotic suit that is part of the Marvel Cinematic Universe. In the video clip, the Hulkbuster is engaged in a fight with another character, possibly the Hulk or another opponent. The suit is designed to be a powerful and versatile weapon, capable of dealing significant damage to

[Figure 87]

its enemies. The Hulkbuster's appearance and actions in the clip suggest that it is in the midst of a battle, using its advanced

technology and strength to take on its opponent.

###### Fig. 8: More conversation visualization using Merlin in video-level tasks. We showcase additional examples where Merlin adeptly handles various video-level tasks including general tracking, video referring expression comprehension (Video REC) and video referring expression generation (Video REG).

###### Future Reasoning

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

According to frame1<image>, frame2<image> and frame3<image>, track person and human-used objects present in the video along with a caption about what the person is doing. When detailing person and human-used objects’ trajectories in your response, adhere to the class<Idi>Frame t:[xmin,ymin,xmax,ymax]</Idi> format.

[Figure 92]

###### There is one person<Id1>frame:1:[392, 001, 827, 998];frame:2:[392, 001, 827, 998];frame:3:[342, 001, 742, 998]</Id1> and one trash can<Id2>frame:1:[332, 608, 485, 983];frame:2:[332, 608, 485, 983];frame:3:[228, 163, 419, 536]</Id2>. The person is holding a trash can and is standing in a room.

[Figure 93]

What might this person do next immediately?

[Figure 94]

The man is standing in a room and appears to be emptying the trash can. The man might next put the trash can on the floor or a designated trash receptacle.

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

According to frame1<image>, frame2<image> and frame3<image>, Explain Cyclist<Id1>Frame1:[280, 436, 346, 721]</Id1>'s route and include a short caption. For the trajectories included in the answer, please use the format class<Idi>Frame t:[xmin,ymin,xmax, ymax]</Idi>.

[Figure 99]

###### There is one Cyclist<Id1>Frame1:[280, 436, 346, 721];Frame2:[183, 425, 300, 771];Frame3:[071, 402, 235, 829]</Id1>. Cyclist's route: From the left side of the road, the cyclist is riding a bicycle down the road.

[Figure 100]

If there is an ego car on the road, this picture is the one seen inside the ego car. How should the ego car have?

[Figure 101]

If there is an ego car on the road. The ego car should be prepared to stop or slow down if necessary, and be cautious of the cyclist's movements.

##### Fig. 9: More future reasoning conversation visualization using Merlin.

Nematzadeh, A., Sharifzadeh, S., Binkowski, M., Barreira, R., Vinyals, O., Zisserman, A., Simonyan, K.: Flamingo: a visual language model for few-shot learning. In: NeurIPS (2022) 4

- 2. Anil, R., Dai, A.M., Firat, O., Johnson, M., Lepikhin, D., Passos, A., Shakeri, S., Taropa, E., Bailey, P., Chen, Z., et al.: Palm 2 technical report. arXiv preprint arXiv:2305.10403 (2023) 2
- 3. Awadalla, A., Gao, I., Gardner, J., Hessel, J., Hanafy, Y., Zhu, W., Marathe, K., Bitton, Y., Gadre, S., Sagawa, S., et al.: Openflamingo: An open-source

- framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390 (2023) 10
- 4. Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., Zhou, J.: Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966 (2023) 6, 7, 11, 12
- 5. Bates, C., Battaglia, P.W., Yildirim, I., Tenenbaum, J.B.: Humans predict liquid dynamics using probabilistic simulation. In: CogSci (2015) 2
- 6. Bertinetto, L., Valmadre, J., Henriques, J.F., Vedaldi, A., Torr, P.H.: Fullyconvolutional siamese networks for object tracking. In: ECCV. pp. 850–865 (2016) 11
- 7. Bolya, D., Fu, C., Dai, X., Zhang, P., Feichtenhofer, C., Hoffman, J.: Token merging: Your vit but faster. In: ICLR. OpenReview.net (2023) 5
- 8. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. Advances in neural information processing systems 33, 1877–1901 (2020) 4, 6
- 9. Chen, B., Xu, Z., Kirmani, S., Ichter, B., Driess, D., Florence, P., Sadigh, D., Guibas, L., Xia, F.: Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. arXiv preprint arXiv:2401.12168 (2024) 14
- 10. Chen, K., Zhang, Z., Zeng, W., Zhang, R., Zhu, F., Zhao, R.: Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195

(2023) 4, 5, 6, 8, 10, 11, 12, 18

- 11. Chiang, W.L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J.E., Stoica, I., Xing, E.P.: Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. https://lmsys.org/blog/2023-0330-vicuna/ (2023) 4, 5, 9
- 12. Cholakov, R., Kolev, T.: Transformers predicting the future. applying attention in next-frame and time series forecasting. arXiv preprint arXiv:2108.08224 (2021) 2
- 13. Cui, Y., Zeng, C., Zhao, X., Yang, Y., Wu, G., Wang, L.: Sportsmot: A large multiobject tracking dataset in multiple sports scenes. arXiv preprint arXiv:2304.05170

(2023) 17

- 14. Dai, W., Li, J., Li, D., Tiong, A.M.H., Zhao, J., Wang, W., Li, B., Fung, P., Hoi, S.: Instructblip: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500 (2023) 10, 11, 12
- 15. Danelljan, M., Bhat, G., Khan, F.S., Felsberg, M.: Atom: Accurate tracking by overlap maximization. In: CVPR. pp. 4660–4669 (2019) 11
- 16. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805

(2018) 4

- 17. Ding, H., Liu, C., He, S., Jiang, X., Loy, C.C.: Mevis: A large-scale benchmark for video segmentation with motion expressions. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2694–2703 (2023) 17, 18
- 18. Dong, R., Han, C., Peng, Y., Qi, Z., Ge, Z., Yang, J., Zhao, L., Sun, J., Zhou, H., Wei, H., et al.: Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499 (2023) 4
- 19. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: ICLR. OpenReview.net (2021) 5

- 20. Fan, H., Lin, L., Yang, F., Chu, P., Deng, G., Yu, S., Bai, H., Xu, Y., Liao, C., Ling, H.: Lasot: A high-quality benchmark for large-scale single object tracking. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5374–5383 (2019) 3, 8, 9, 11, 17
- 21. Friston, K.: The free-energy principle: a unified brain theory? Nature reviews neuroscience 11(2), 127–138 (2010) 2
- 22. Ge, Y., Ge, Y., Zeng, Z., Wang, X., Shan, Y.: Planting a seed of vision in large language model. arXiv preprint arXiv:2307.08041 (2023) 4
- 23. Gong, T., Lyu, C., Zhang, S., Wang, Y., Zheng, M., Zhao, Q., Liu, K., Zhang, W., Luo, P., Chen, K.: Multimodal-gpt: A vision and language model for dialogue with humans. arXiv preprint arXiv:2305.04790 (2023) 12
- 24. Goyal, A., Bengio, Y.: Inductive biases for deep learning of higher-level cognition. CoRR abs/2011.15091 (2020) 6
- 25. Gurari, D., Li, Q., Stangl, A.J., Guo, A., Lin, C., Grauman, K., Luo, J., Bigham, J.P.: Vizwiz grand challenge: Answering visual questions from blind people. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3608–3617 (2018) 4
- 26. Gurari, D., Li, Q., Stangl, A.J., Guo, A., Lin, C., Grauman, K., Luo, J., Bigham, J.P.: Vizwiz grand challenge: Answering visual questions from blind people. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3608–3617 (2018) 11
- 27. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16000–16009 (2022) 2
- 28. Huang, L., Zhao, X., Huang, K.: Got-10k: A large high-diversity benchmark for generic object tracking in the wild. IEEE transactions on pattern analysis and machine intelligence 43(5), 1562–1577 (2019) 3, 8, 9, 11, 13, 17
- 29. Huang, S., Dong, L., Wang, W., Hao, Y., Singhal, S., Ma, S., Lv, T., Cui, L., Mohammed, O.K., Liu, Q., et al.: Language is not all you need: Aligning perception with language models. arXiv preprint arXiv:2302.14045 (2023) 10
- 30. Hudson, D.A., Manning, C.D.: Gqa: A new dataset for real-world visual reasoning and compositional question answering. In: CVPR (2019) 4, 11
- 31. Ioffe, S., Szegedy, C.: Batch normalization: Accelerating deep network training by reducing internal covariate shift. In: ICML. JMLR Workshop and Conference Proceedings, vol. 37, pp. 448–456. JMLR.org (2015) 6
- 32. Kay, K.N., Naselaris, T., Prenger, R.J., Gallant, J.L.: Identifying natural images from human brain activity. Nature 452(7185), 352–355 (2008) 2
- 33. Kazemzadeh, S., Ordonez, V., Matten, M., Berg, T.: Referitgame: Referring to objects in photographs of natural scenes. In: Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). pp. 787–798 (2014) 3, 8, 9, 18
- 34. Koh, J.Y., Fried, D., Salakhutdinov, R.: Generating images with multimodal language models. arXiv preprint arXiv:2305.17216 (2023) 4
- 35. Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Kamali, S., Popov, S., Malloci, M., Kolesnikov, A., et al.: The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International Journal of Computer Vision 128(7), 1956–1981 (2020) 9, 16
- 36. Laurençon, H., Saulnier, L., Tronchon, L., Bekman, S., Singh, A., Lozhkov, A., Wang, T., Karamcheti, S., Rush, A.M., Kiela, D., et al.: Obelics: An open web-scale filtered dataset of interleaved image-text documents. In: Thirty-seventh Conference

- on Neural Information Processing Systems Datasets and Benchmarks Track (2023) 11
- 37. Li, B., Wu, W., Wang, Q., Zhang, F., Xing, J., Yan, J.: SiamRPN++: Evolution of siamese visual tracking with very deep networks. In: CVPR. pp. 4282–4291 (2019) 11
- 38. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597 (2023) 4, 6, 11
- 39. Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W.X., Wen, J.R.: Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355 (2023) 4, 12
- 40. Li, Y., Chen, L., He, R., Wang, Z., Wu, G., Wang, L.: Multisports: A multi-person video dataset of spatio-temporally localized sports actions. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 13536–13545 (2021) 9, 18
- 41. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744 (2023) 4, 5, 6, 9, 10, 11, 12, 18
- 42. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744 (2023) 7
- 43. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning (2023) 3, 4, 5, 6, 7, 9, 10, 12
- 44. Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281 (2023) 4, 10, 11, 12
- 45. Loshchilov, I., Hutter, F.: Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983 (2016) 10
- 46. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (2019) 10
- 47. Malla, S., Dariush, B., Choi, C.: Titan: Future forecast using action priors. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11186–11196 (2020) 9, 18
- 48. Milan, A., Leal-Taixé, L., Reid, I., Roth, S., Schindler, K.: Mot16: A benchmark for multi-object tracking. arXiv preprint arXiv:1603.00831 (2016) 8, 9, 17
- 49. OpenAI: Chatgpt. https://openai.com/blog/chatgpt/ (2023) 4, 6
- 50. OpenAI: Gpt-4 technical report (2023) 2, 6, 9
- 51. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P.F., Leike, J., Lowe, R.: Training language models to follow instructions with human feedback. In: NeurIPS (2022) 4
- 52. Pi, R., Gao, J., Diao, S., Pan, R., Dong, H., Zhang, J., Yao, L., Han, J., Xu, H., Zhang, L.K.T.: Detgpt: Detect what you need via reasoning. arXiv preprint arXiv:2305.14167 (2023) 2, 8
- 53. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PMLR (2021) 5, 9
- 54. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al.: Language models are unsupervised multitask learners. OpenAI blog 1(8), 9 (2019) 4

- 55. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research 21(1), 5485–5551 (2020) 4
- 56. Ramnani, N., Miall, R.C.: A system in the human brain for predicting the actions of others. Nature neuroscience 7(1), 85–90 (2004) 2
- 57. Ryoo, M.S., Piergiovanni, A.J., Arnab, A., Dehghani, M., Angelova, A.: Tokenlearner: What can 8 learned tokens do for images and videos? CoRR abs/2106.11297 (2021) 5
- 58. Sap, M., Bras, R.L., Fried, D., Choi, Y.: Neural theory-of-mind? on the limits of social intelligence in large lms. In: EMNLP. pp. 3762–3780. Association for Computational Linguistics (2022) 8
- 59. Schuhmann, C., Vencu, R., Beaumont, R., Kaczmarczyk, R., Mullis, C., Katta, A., Coombes, T., Jitsev, J., Komatsuzaki, A.: Laion-400m: Open dataset of clipfiltered 400 million image-text pairs. arXiv preprint arXiv:2111.02114 (2021) 9, 15
- 60. Shao, S., Li, Z., Zhang, T., Peng, C., Yu, G., Zhang, X., Li, J., Sun, J.: Objects365: A large-scale, high-quality dataset for object detection. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 8430–8439 (2019) 3, 15
- 61. Shao, S., Li, Z., Zhang, T., Peng, C., Yu, G., Zhang, X., Li, J., Sun, J.: Objects365: A large-scale, high-quality dataset for object detection. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 8430–8439 (2019) 8, 9
- 62. Shao, S., Zhao, Z., Li, B., Xiao, T., Yu, G., Zhang, X., Sun, J.: Crowdhuman: A benchmark for detecting human in a crowd. arXiv preprint arXiv:1805.00123

(2018) 16

- 63. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. In: ICLR (2015) 6
- 64. Simsek, F.E., Cigla, C., Kayabol, K.: Sompt22: A surveillance oriented multipedestrian tracking dataset. In: European Conference on Computer Vision. pp. 659–675. Springer (2022) 8, 9, 17
- 65. Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., Rohrbach, M.: Towards vqa models that can read. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8317–8326

(2019) 9

- 66. Sun, P., Cao, J., Jiang, Y., Yuan, Z., Bai, S., Kitani, K., Luo, P.: Dancetrack: Multi-object tracking in uniform appearance and diverse motion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20993–21002 (2022) 3, 8, 9, 17
- 67. Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., Hashimoto, T.B.: Stanford alpaca: An instruction-following llama model. https: //github.com/tatsu-lab/stanford_alpaca (2023) 4
- 68. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., Lample, G.: Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023) 4
- 69. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.: Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023) 4

- 70. Ullman, T.D.: Large language models fail on trivial alterations to theory-of-mind tasks. CoRR abs/2302.08399 (2023) 8
- 71. Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., Chi, E.H., Hashimoto, T., Vinyals, O., Liang, P., Dean, J., Fedus, W.: Emergent abilities of large language models. Trans. Mach. Learn. Res. 2022 (2022) 6
- 72. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems 35, 24824–24837 (2022) 2, 8
- 73. Wu, B., Yu, S., Chen, Z., Tenenbaum, J.B., Gan, C.: Star: A benchmark for situated reasoning in real-world videos. In: Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2) (2021) 9, 18
- 74. Xu, Y., et al.: Siamfc++: Towards robust and accurate visual tracking with target estimation guidelines. In: AAAI. pp. 140–148 (2020) 11
- 75. Yan, W., Zhang, Y., Abbeel, P., Srinivas, A.: Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157 (2021) 2
- 76. Ye, Q., Xu, H., Xu, G., Ye, J., Yan, M., Zhou, Y., Wang, J., Hu, A., Shi, P., Shi, Y., et al.: mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178 (2023) 4, 10, 12
- 77. Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V., Darrell, T.: Bdd100k: A diverse driving dataset for heterogeneous multitask learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2636–2645 (2020) 17
- 78. Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., Wang, L.: Mmvet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490 (2023) 4, 11, 12
- 79. Zellers, R., Bisk, Y., Farhadi, A., Choi, Y.: From recognition to cognition: Visual commonsense reasoning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6720–6731 (2019) 3, 8, 9
- 80. Zeng, A., Liu, X., Du, Z., Wang, Z., Lai, H., Ding, M., Yang, Z., Xu, Y., Zheng, W., Xia, X., et al.: Glm-130b: An open bilingual pre-trained model. arXiv preprint arXiv:2210.02414 (2022) 4
- 81. Zhang, S., Sun, P., Chen, S., Xiao, M., Shao, W., Zhang, W., Chen, K., Luo, P.: Gpt4roi: Instruction tuning large language model on region-of-interest. arXiv preprint arXiv:2307.03601 (2023) 4
- 82. Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X.V., et al.: Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068 (2022) 4
- 83. Zhao, L., Yu, E., Ge, Z., Yang, J., Wei, H., Zhou, H., Sun, J., Peng, Y., Dong, R., Han, C., et al.: Chatspot: Bootstrapping multimodal llms via precise referring

- instruction tuning. arXiv preprint arXiv:2307.09474 (2023) 3, 8

84. Zhao, L., Yu, E., Ge, Z., Yang, J., Wei, H., Zhou, H., Sun, J., Peng, Y., Dong, R., Han, C., et al.: Chatspot: Bootstrapping multimodal llms via precise referring

- instruction tuning. arXiv preprint arXiv:2307.09474 (2023) 4

- 85. Zhou, P., Madaan, A., Potharaju, S.P., Gupta, A., McKee, K.R., Holtzman, A., Pujara, J., Ren, X., Mishra, S., Nematzadeh, A., Upadhyay, S., Faruqui, M.: How far are large language models from agents with theory-of-mind? CoRR abs/2310.03051 (2023) 8
- 86. Zhu, D., Chen, J., Shen, X., Li, X., Elhoseiny, M.: Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv preprint arXiv:2304.10592 (2023) 3, 5, 6, 10, 12

