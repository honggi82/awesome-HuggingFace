## arXiv:2512.10675v2[cs.RO]6Jan2026

[Figure 1]

# Evaluating Gemini Robotics Policies in a Veo World Simulator

Gemini Robotics Team, Google DeepMind1

Webpage: veo-robotics.github.io

Generative world models hold significant potential for simulating interactions with visuomotor policies in varied environments. Frontier video models can enable generation of realistic observations and environment interactions in a scalable and general manner. However, the use of video models in robotics has been limited primarily to in-distribution evaluations, i.e., scenarios that are similar to ones used to train the policy or fine-tune the base video model. In this report, we demonstrate that video models can be used for the entire spectrum of policy evaluation use cases in robotics: from assessing nominal performance to out-of-distribution (OOD) generalization, and probing physical and semantic safety. We introduce a generative evaluation system built upon a frontier video foundation model (Veo). The system is optimized to support robot action conditioning and multi-view consistency, while integrating generative image-editing and multi-view completion to synthesize realistic variations of real-world scenes along multiple axes of generalization. We demonstrate that the system preserves the base capabilities of the video model to enable accurate simulation of scenes that have been edited to include novel interaction objects, novel visual backgrounds, and novel distractor objects. This fidelity enables accurately predicting the relative performance of different policies in both nominal and OOD conditions, determining the relative impact of different axes of generalization on policy performance, and performing red teaming of policies to expose behaviors that violate physical or semantic safety constraints. We validate these capabilities through 1600+ real-world evaluations of eight Gemini Robotics policy checkpoints and five tasks for a bimanual manipulator.

##### 1. Introduction

Generalist robot policies demand generalist evaluation. The very feature of generalist policies that makes them appealing — that they can be instructed via natural language to perform a variety of useful tasks in a wide range of environments — poses a fundamental technical challenge in evaluating their reliability, generalization, and safety. Conducting hardware evaluations that are sufficiently broad to cover both nominal and edge-case scenarios is typically impractical, especially when the goal is to compare multiple policies in order to glean frequent insights for training. When the objective is to evaluate safety, hardware evaluation is often simply infeasible.

As an example, consider how one might evaluate the semantic safety (Sermanet et al., 2025) of a generalist policy, i.e., its ability to obey commonsense safety constraints in open-domain environments (Fig. 1; bottom). Setting up real-world scenes that probe vulnerabilities of a policy in the “long tail" of such constraints — that sharp objects may break computer screens, that a piece of plastic should not be placed on a stove, that broken glass should not be left on the floor, and so on — can endanger the robot, its environment, and humans.

While simulation presents one promising avenue towards such evaluation (Li et al., 2024; Liu et al., 2023; Pumacay et al., 2024), traditional physics-based simulators pose several challenges. First, a wide range of realistic assets (e.g., laptop, sharp objects, etc.) need to be curated or created. Second, accurately simulating these assets can be very challenging, especially with non-rigid objects

1See Authors section for full author list.

© 2026 Google DeepMind. All rights reserved

###### Nominal OOD Safety

[Figure 2]

[Figure 3]

[Figure 4]

ⓥ ⓥ

###### Initial frame Policy rollout Prediction

[Figure 5]

[Figure 6]

[Figure 7]

###### World model

ⓥ ⓥ

[Figure 8]

“Close the laptop lid”

Generated frames Unsafe

- Figure 1 | Top: We present an evaluation system based on video prediction to predict nominal performance, OOD generalization, and safety. Bottom: Our “world model” predicts potentially unsafe behavior of a policy.

or humans. Third, closing the visual gap between simulation and real-world observations can involve a months-long iterative process that requires significant human expertise (e.g., careful green screening; Badithela et al. (2025); Li et al. (2024)) and effort.

In this report, we demonstrate the capability of video models to serve as generalist evaluators for generalist policies. Frontier video models offer an alternate way to simulate the world that holds the key to the challenges highlighted above. They have the potential to simulate a wide variety of different asset categories and their complex behaviors using a unified recipe. By leveraging web-scale video datasets and highly expressive generative architectures (Agarwal et al., 2025; Blattmann et al., 2023; Brooks et al., 2024; Veo Team, 2025), they can produce outputs that are both photorealistic and physically realistic. However, realizing this potential has historically remained elusive due to artifacts in closed-loop action-conditioned generation, the difficulty of simulating contact dynamics, and the requirement for multi-view consistency in modern policy architectures.

We present a video modeling-based evaluation system capable of supporting the full spectrum of policy evaluation use cases in robotics, from in-distribution evaluation, to out-of-distribution (OOD) generalization, to red teaming for safety. Building upon state-of-the-art video generation models (Veo Team, 2025), we achieve action-conditioned, multi-view consistent video simulation that is both photorealistic and responsive to fine-grained robot control. The integration of generative editing techniques allows for the creation of realistic and diverse variations of real-world scenes to simulate novel objects, visual backdrops, and safety-critical elements without requiring physical setup.

We validate predictions from our video model across 1600+ real-world trials with eight generalist policy checkpoints and five tasks. Our results demonstrate the ability to preserve the base capabilities of the underlying video foundation model while achieving the necessary fidelity for rigorous robotic evaluation. Specifically, we demonstrate:

- 1. Accurate prediction of relative performance and rankings of robot policies in pick-and-place

- tasks that are within the domain of the system’s training data.
- 2. Accurate prediction of the relative degradation caused by different axes of generalization (e.g., scene objects, visual background, etc.; Gao et al. (2025a)) for a given policy, and accurate prediction of the relative performance of different checkpoints along different generalization axes.
- 3. Predictive red teaming (Majumdar et al., 2025) for safety: by rolling out policies in edited scenes that involve safety-critical elements, the system discovers potential vulnerabilities without requiring hardware evaluations.

While we are still in the early days of video modeling for robotics (see Sec. 7 for challenges and limitations), this report demonstrates a path towards scalable evaluation of generalization and safety of robot policies in video-simulated worlds.

##### 2. Method Overview

In this section, we describe the video generation model used for policy evaluation, including the pretrained video model and how this pretrained model is finetuned on robot-specific data.

Model Architecture. We use the Veo2 text-to-video model (van den Oord and Roman, 2024) as our base model. Veo is built using a latent diffusion architecture. It first uses autoencoders to compress spatio-temporal data into smaller, more efficient latent representations. A transformer-based denoising network is then trained to remove noise from these latent vectors. To generate a video, the model iteratively applies this denoising network to a random noise input, refining it into the final video output (Veo Team, 2025).

Training Data & Curation. The model is trained on a large dataset of videos, images, and associated annotations (Veo Team, 2025). These text captions are generated at different levels of detail using multiple Gemini (GeminiTeam et al., 2025) models. This data undergoes a rigorous preparation process as part of the model’s construction. The pretraining data for Veo is filtered for quality and to remove unsafe content and personally identifiable information. The pretraining data is "semantically deduplicated" to prevent the model from overfitting or memorizing specific training examples. Please refer to the Veo tech report (Veo Team, 2025) for additional information.

Action Conditioning. We finetune the pretrained Veo2 model on a large-scale robotics dataset consisting of diverse tasks that cover a broad range of manipulation skills across a multitude of scenes. This fine-tuned robotic video generation model can be conditioned on a current image observation of the scene and a sequence of future robot poses, and can predict a sequence of future images that correspond to the future robot poses and observations. Fig. 2 (top) shows an example of rendered poses overlaid over the video generated using these poses as conditioning.

Multi-View Generation. In order to mitigate the effect of partial observations, we tile the four observations across four cameras in our setup, including the top-down view, the side view, and the left and right wrist view. We finetune Veo2 to generate the tiled future frames conditioned on the initial frame and future robot poses. Fig. 2 (bottom) shows an example of a multi-view video frame generated using the model.

##### 3. Evaluating Policies in Nominal Scenarios

We begin by using the fine-tuned Veo (Robotics) model for evaluating policies in nominal (i.e., in-distribution) scenarios involving tasks, instructions, objects, distractors, and visual backgrounds that are similar to the training data used for policies and for fine-tuning the video model.

[Figure 9]

[Figure 10]

[Figure 11]

###### ⓥ ⓥ

[Figure 12]

ⓥ ⓥ

ⓥ ⓥ

- Figure 2 | Top: Video generation is conditioned on the initial scene image and a sequence of commanded robot poses. The figure shows the rendered poses overlaid on top of the generated video frames. Bottom: Multi-view consistent video generation for the four robot cameras.

###### 3.1. Experimental Setup

Tasks. We use five tasks for the ALOHA 2 bimanual platform (Aldaco et al., 2024; Zhao et al., 2024) shown in Fig. 3 for policy evaluation. For each task, we vary the initial positions of objects, the identity and location of distractor objects in the scene, and the visual backdrop behind the table (which varies based on the particular robot that the policy is executed on). In addition, we evaluate instruction generalization with the following variations:

- • Rephrasing the instruction, e.g., “pick the red grapes (top right) and put them in the grey box (top left compartment)" instead of "put the top right red grapes into the top left compartment of the grey box".
- • Typographical errors in the instruction, e.g., “put the brwn bar into the top pckt of the lnch bag" instead of "put the brown bar into the lunch bag’s top pocket".
- • A different language that the instruction is provided in, e.g., “coloque las uvas verdes de la parte superior izquierda en el compartimento derecho de la caja gris" instead of "put the top left green grapes into the right comppartment of the grey box".
- • Different levels of specificity in the instruction, e.g., “pick up the top right red grapes and place them in the top left container of the grey box" instead of "put the top right red grapes into the top left compatment of the grey box".

In total, we consider 80 scene-instruction combinations for evaluating policies and use a binary success metric for scoring.

Policies. We train end-to-end vision-language-action (VLA) policies based on the Gemini Robotics On-Device (GROD) model. Starting from a powerful VLM backbone, GROD is trained on a large-scale teleoperated robot action dataset collected over 12 months from a fleet of ALOHA 2 robots (Zhao et al.,

- 2024). This dataset consists of real-world expert robot demonstrations, covering scenarios with varied manipulation skills, objects, task difficulties, episode horizons, and dexterity requirements. GROD is trained to predict a 1-second action chunk with continuous actions at 50 Hz; we use a combination of

“Put the top right red grapes into the top left compartment of the grey box”

“Put the top left green grapes into the right compartment of the grey box”

“Put the lego into the lego bag”

[Figure 13]

[Figure 14]

[Figure 15]

“Put the brown bar into the lunch bag’s top pocket”

“Put banana in bowl with a handover”

[Figure 16]

[Figure 17]

- Figure 3 | A set of five nominal tasks used in our analysis to evaluate different VLA policy checkpoints.

asynchronous policy execution and on-device optimizations to run the policy on a single GPU with minimal latency. For more details on the training data and a comprehensive evaluation of the policy model, see the Gemini Robotics technical report (GeminiRoboticsTeam et al., 2025) and the GROD announcement (Parada, 2025).

###### 3.2. Results

We compare predictions made by the Veo (Robotics) model with real-world paired evaluations for the 80 scene-instruction combinations presented in Sec. 3.1. For each initial scene, we condition the closed-loop video rollout with the first frame from the robot’s four cameras along with the task instruction. Each episode consists of an 8-second rollout, which is scored with the binary success metric by human evaluators. Fig. 4 compares real-world success rates with predictions for eight variants of the GROD policy described in Sec. 3.1. We observe that Veo (Robotics) is able to accurately rank the different policies by their performance. In addition, there is a strong linear correlation between predicted and actual success rates. We note that the absolute values of predicted success rates are lower than their real counterparts (see Sec. 7 for a discussion).

[Figure 18]

MMRV = 0.03 Pearson = 0.88

Realsuccessrate

Veo (Robotics) success rate

Figure 4 | Generalist policy performance in our actionconditioned video model correlates strongly with realworld performance in nominal scenarios.

In order to quantitatively evaluate predic-

tions from Veo (Robotics), we present two metrics in Fig. 4. First, the mean maximum rank violation (MMRV) metric (Li et al., 2024) compares the consistency of policy rankings between real outcomes

and predictions. Given 𝑛 policies 𝜋1,...,𝜋𝑛 and corresponding success rates 𝑅1real,...,𝑅𝑛real from real-world evaluations and predicted success rates 𝑅1pred,...,𝑅𝑛pred, the MMRV is defined as:

1 𝑛

∑︁𝑛

RankViolation(𝑖,𝑗), (1)

MMRV :=

max

1≤𝑗≤𝑛

𝑖=1

where RankViolation(𝑖,𝑗) := |𝑅𝑖real − 𝑅𝑗real| · [︁(𝑅𝑖pred < 𝑅𝑗pred) ̸= (𝑅𝑖real < 𝑅𝑗real)]︁. (2)

The MMRV has range [0,1], with lower values indicating greater rank consistency. Second, we compute the Pearson coefficient to quantify the linear correlation between predicted and real success rates.

##### 4. Evaluating Policies In Out-Of-Distribution Scenarios

###### Original Edited Prediction

###### Policy rollout

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

###### World model

###### ⓥ ⓥ

[Figure 25]

“Put banana in bowl” Wrong object

Generated frames

“Put pink brush in bowl”

- Figure 5 | To evaluate generalist policies in OOD scenarios, we generate edited versions of nominal scenes using NanoBanana and use it as the first frame for video generation.

[Figure 26]

Replicated real scenes

Synthetic scenes via image editing

[Figure 27]

Background Small distractor Large distractor Object

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Background Small distractor Large distractor Object

[Figure 32]

[Figure 33]

ⓥ ⓥ ⓥ ⓥ

- Figure 6 | We generate OOD scenes corresponding to four axes of generalization using generative image-editing (top), and create equivalent real-world scenes to evaluate generalist policies.

Next, we present results for policy evaluation in out-of-distribution (OOD) evaluations. We edit a nominal RGB observation from the robot’s overhead camera to reflect a change in a given factor of interest (e.g., adding a new object to be manipulated, changing the visual backdrop, or adding a distractor; see Fig. 5). We use Gemini 2.5 Flash Image (a.k.a. NanoBanana) to generate this edited scene using a language description of the desired change (GeminiTeam et al., 2025). We also edit the

|ⓥ|
|---|

|ⓥ|
|---|

[Figure 34]

[Figure 35]

- Figure 7 | Multi-view synthesis: A fine-tuned Veo2 model takes an edited overhead observation (top left) as input and synthesizes observations from three different viewpoints.

task instruction for the robot accordingly; for example, in Fig. 5, the instruction is updated to “put pink brush in bowl with handover” instead of “put banana in bowl with handover”.

The edited single-view overhead observation is used to generate a multi-view observation to fill in the robot’s other camera views. This “multi-view synthesis” is performed using a version of Veo2 that is fine-tuned to predict multi-view images from a single-view image. Fig. 7 shows an example of this process. We roll out the policy we want to evaluate using the Veo (Robotics) model with the edited observations and language instruction as input. The rollout is then scored for success or failure.

Evaluation. For OOD evaluations, we considered four axes of generalization visualized in Fig. 6:

- • Background. We add a cloth colored red, green, or blue to each scene.
- • Small Distractor. We add a novel distractor to the scene. In particular, we consider plushies (soft toys) that were unseen in the policies’ training data: ‘purple octopus’, ‘green turtle’, ‘penguin’, ‘yellow duck’, ‘pink axolotl’. The objects are approximately 3-4 inches in size, and are shown in Appendix A. In each of the five tasks described in Sec. 3.1, we add one of the five distractors.
- • Large Distractor. We also consider larger distractors in the form of 10-12 inch sized plushies: ‘polar bear’, ‘golden retriever’, ‘teddy bear’, ‘bighorn sheep’, and ‘dolphin’. These objects are visualized in Appendix A.
- • Object. We add a novel object that needs to be manipulated. In particular, we consider the following objects that were unseen during policy training: ‘toy elephant figurine’, ‘yellow and black toy jeep’, ‘pink plastic kitchen brush with a handle’, ‘blue teacup’, ‘blue and green checkered zipper pouch’. The objects are shown in Appendix A. In each of the five tasks described in Sec. 3.1, we add one of the five novel objects and change the instruction so that the robot needs to manipulate the new object instead of the object in the original task (e.g., see Fig. 5).

In order to validate predictions made by the video model, we replicate the edited scenes as closely as possible in the real world. Fig. 6 shows examples of scenes generated via image editing and their real-world replicated counterparts. We use five policy checkpoints for OOD evaluations.

###### 4.1. Comparing Axes Of Generalization For a Given Policy

First, we consider a single policy — which we refer to as Policy A — with the strongest performance in nominal scenarios, and compare the impact that each axis of generalization has on performance (Gao et al., 2025a). Fig. 8 compares predictions made by Veo (Robotics) with real-world success rates. First, we observe that we can accurately rank the different axes of generalization by difficulty. In particular, Veo (Robotics) predicts both small and large distractors to have the least impact on performance, while changing the background is predicted to have a larger impact, and changing the object is predicted to have the largest impact. These predictions are validated by the real-world evaluations with an MMRV of 0.06. Moreover, we can also predict the relative values of the performance degradation induced by each axis of generalization: there is a strong linear correlation (Pearson = 0.86) between predicted and real success rates. Similar to the results in Sec. 3, the absolute values of predicted success rates are lower than real success rates.

[Figure 36]

Realsuccessrate

MMRV = 0.06 Pearson = 0.86

Veo (Robotics) success rate

Figure 8 | Performance of a single policy checkpoint across different generalization axes.

In addition to quantitative predictions about success rates under different conditions, evaluations in a video model can also yield qualitative insights into failure modes of policies. As an example, visual inspection of videos generated for Policy A in the ‘Object’ condition demonstrates that a significant portion of failures are due to incorrect instruction following: when instructed to manipulate an unfamiliar object, the policy steers to a more familiar one instead. This is shown in Fig. 5, where the policy is instructed to put the pink brush in the bowl, but approaches the banana. Such qualitative insights could be leveraged to improve policy training, e.g., by guiding additional data collection.

###### 4.2. Comparing Policies Along Each Axis Of Generalization

Next, we demonstrate the ability to compare different policies along each axis of generalization. Fig. 9 presents real-world success rates (as measured by hardware evaluations in OOD conditions shown in Fig. 6) with predictions made using the video model. Each plot compares different policies for a given axis of generalization (background, small/large distractor, object). We find that predicted success rates are strongly correlated to the real-world success rates, especially for the background

Background Small distractor Large distractor Object

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

MMRV = 0.0 Pearson = 0.91

MMRV = 0.10 Pearson = 0.86

MMRV = 0.14 Pearson = 0.77

MMRV = 0.15 Pearson = 0.56

Realsuccessrate

Veo (Robotics) success rate

Veo (Robotics) success rate Veo (Robotics) success rate Veo (Robotics) success rate

- Figure 9 | In OOD evaluation scenarios across four different axes of generalization, Veo (Robotics) remains predictive of policy performance (Pearson co-efficient) and relative ordering (MMRV).

#### Veo

“Quick, grab the red block!” Unsafe outcome

|[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>ⓥ ⓥ ⓥ<br><br>|[Figure 44]<br><br>ⓥ|
|---|---|

Replicated real scene

|[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]|[Figure 48]|
|---|---|

Veo

“Close the laptop” Unsafe outcome

|[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>ⓥ ⓥ ⓥ<br><br>|[Figure 52]<br><br>ⓥ|
|---|---|

### Replicated real scene

|[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]|[Figure 56]|
|---|---|

- Figure 10 | Examples of potentially unsafe behaviors discovered by red-teaming Gemini Robotics policies in the Veo (Robotics) simulator, and replicated in the real-world.

and distractor axes. For object generalization, all policies exhibit low success rates and it is thus more challenging to distinguish them.

- 5. Red Teaming Policies For Safety We demonstrate how the Veo (Robotics) model enables red teaming for safety (Majumdar et al.,

- 2025): by generating synthetically edited scenes with safety-relevant elements, we can mine for potentially unsafe behaviors exhibited by a policy. We focus on scenarios where the robot needs to demonstrate semantic understanding of the scene in order to execute a given command safely. In addition, we are interested in scenarios where safety cannot simply be determined from the initial scene and instruction (e.g., instructing the robot to grab a bottle of bleach and pour it onto an electronic device). In such scenarios, we do not need to roll out the policy in order to determine safety, and such unsafe instructions can be filtered out by a safety layer that is policy agnostic. Instead, we focus on scenarios where rolling out the policy is necessary for determining safety. We describe our process for generating such scenarios below.

Scenario Generation. In order to evaluate safety-relevant behavior, we generated a set of scenarios for an ALOHA tabletop environment. Each scenario consists of a paired user request and a corresponding image. To ensure these scenarios were non-trivial and suitable for safety evaluation, we used Gemini 2.5 Pro as a critic to filter the generated set. A scenario was retained only if it satisfied all of the following properties:

- 1. Contains a Hazard: The scene includes at least one object that is potentially unsafe for the robot to manipulate (e.g., a sharp knife, a hot object, a full glass).
- 2. Requires Multimodal Reasoning: The task’s safety constraints can only be resolved by using both the image and the user request. Neither modality alone provides sufficient information.
- 3. Features Ambiguity: The scenario successfully introduces one or more of the following ambiguous situations:

- • Object Ambiguity: The user request is imprecise (e.g., "pick up the bottle") in a scene with multiple candidate objects, at least one of which is unsafe.
- • Destination Ambiguity: The request specifies an imprecise drop-off location (e.g., "put it on the counter") where at least one plausible destination is unsafe (e.g., too close to an edge or a human).
- • Trajectory/Affordance Ambiguity: An object requires a specific grasp point (e.g., a knife handle) or trajectory (e.g., keeping a cup upright) for safe manipulation.
- • Human Interaction: A human is present in the workspace, and an incorrect robot action could pose a direct risk of injury.

Results. Fig. 10 shows examples of unsafe behaviors found by our pipeline for Policy A. For the instruction “Quick, grab the red block!", the robot moves its gripper towards the block and makes contact with the human hand. For the instruction “close the laptop", the robot closes the laptop without moving the scissors away, potentially breaking the laptop’s screen. We also replicated these scenarios with real-world props, and found that the unsafe behaviors predicted by the video model are observed in these experiments. The project website has additional examples of scenarios with unsafe behaviors.

The safety scenarios in Fig. 10 demonstrate the power of generative methods for policy evaluation. Conducting real-world tests without jeopardizing the robot, its environment, or humans can be very challenging or simply infeasible. While a limited amount of testing can be performed with real-world assets, these are necessarily not fully representative in terms of realism and coverage. Large-scale testing in silico combined with careful small-scale testing on hardware can help discover unsafe behaviors and test various mitigation strategies.

##### 6. Related work

Offline Evaluation. Scalable and predictive evaluation for robot policies has been an open area of investigation in the literature, especially as resource requirements for statistically meaningful performance measurements of multitask robot policies expand to hundreds of thousands of expensive real-world evaluation trials (Brohan et al., 2023). One approach to measuring policy performance without real-world rollouts has been to directly evaluate robot policies in a physics simulation. Numerous manipulation benchmarks (Liu et al., 2023; Pumacay et al., 2024; Wang et al., 2025) have proposed standardized simulation environments encompassing sets of robot tasks defined by initial conditions and success criteria alongside simulated training datasets of expert trajectories, aiming to provide a fair evaluation for studying the performance and generalization capabilities of policy learning methods when training on and evaluating in simulation. Recently, Li et al. (2024) evaluated various

manipulation policy checkpoints, trained only on real robot datasets, on a set of tuned simulation environments which are curated based on initial conditions from real-world evaluations. These real-to-sim environments curated specifically for evaluation (or training) can be sourced directly from real-world environments and potentially improved with more data (Badithela et al., 2025; Torne et al.,

- 2024). Such real-to-sim evaluations are nascent for learning-based robot manipulation but have seen substantial predictive signal for other robotic applications such as autonomous driving (Dosovitskiy et al., 2017). While physics simulations may provide useful structural priors and grounding which are useful for contact-rich manipulation, physics simulations are difficult to tune and expensive to scale to many types of initial conditions and object sets, such as challenging objects like deformables or liquids.

Video Generation Models. In contrast, data-driven video generation models provide an alternative approach to in silico policy evaluation. Du et al. (2023) show how a fine-tuned video generation model can generate robot policy rollouts conditioned on a high-level language instruction, while action-conditioned world models have demonstrated that generative video models can not only follow coarse language conditioning but also low-level robot actions expressed as explicit (NVIDIA, 2025; Russell et al., 2025) or latent actions (1XW, 2025; Bruce et al., 2024). Recent works (Guo et al., 2025; Quevedo et al., 2025) show that such action-conditioned world models can be used to evaluate policies trained only on real-world data on a variety of in-distribution training tasks, providing both relative and absolute signal on expected real-world policy performance. In addition, our work studies the effect of various distribution shifts, ranging from visual and semantic generalization to safety-critical red-teaming initial condition changes. Similar to our work, Majumdar et al. (2025) use image editing to generate variations of nominal scenes along different axes of generalization and make predictions about policy performance. However, these predictions are made using a heuristic approach based on anomaly detection given only the first (edited) frames of episodes; in contrast, we simulate policies for entire episodes using an action-conditioned video model.

Evaluating Safety. There is a large body of work on evaluating physical safety for robotic systems such as autonomous vehicles (Favaro et al., 2023, 2025; Gao et al., 2025b). However, evaluating policies for semantic safety — the long tail of commonsense constraints that generalist robots operating in human-centered environments should satisfy — has only recently received attention. Initial work in this area include text-only benchmarks that evaluate the abilities of large language models to reason about commonsense safety constraints (Bianchi et al., 2023; Zhang et al., 2023). Multi-modal benchmarks that assess safety of vision-language models have also been developed (Zhang et al., 2024). Sermanet et al. (2025) proposed the ASIMOV benchmark, which is a large-scale collection of datasets that ground scenarios in real-world scenes and injury reports from hospitals. ASIMOV-2.0 (Jindal et al.,

- 2025) expands the benchmark to include videos and physical constraint reasoning. These benchmarks have been used to evaluate the Gemini Robotics embodied reasoning models (GeminiRoboticsTeam et al., 2025; Team et al., 2025). All evaluation benchmarks highlighted above are non-interactive in nature — text, images, or videos are provided as input to a language model in order to assess safety. In contrast, the work presented in this report provides a way to assess closed-loop safety of the policy. This is critical in settings where safety cannot simply be inferred from the initial scene and task instruction, and where actions that the robot takes at one time-step have implications for safety in future time-steps. Concurrent work (Wayve, 2025) in the context of autonomous driving provides a complementary demonstration of the power of world modeling and scene editing for evaluating safety.

[Figure 57]

[Figure 58]

[Figure 59]

###### ⓥ ⓥ ⓥ

- Figure 11 | An example of unrealistic video generation: a novel object appears spontaneously while the gripper is interacting with a different object.

##### 7. Discussion

This report demonstrates the viability of action-conditioned video models for the full suite of policy evaluation applications in robotics: from in-distribution evaluations, to out-of-distribution generalization, to safety. We demonstrate that by training video models on large-scale robotics datasets, we obtain a powerful simulator capable of generating photorealistic and consistent predictions from multiple viewpoints. Our results confirm that state-of-the-art video models, combined with generative image editing, enable the creation of effectively infinite scene variations to probe policy capabilities.

While the results reported here represent a significant milestone, our analysis highlights specific areas for continued development. First, simulating contact-rich interactions, particularly with small objects, remains a challenge. Fig. 11 illustrates an instance of hallucination where an object appears spontaneously during interaction; additional examples of generation artifacts are provided on the project website. We anticipate that scaling diverse interaction data in future iterations will directly address these fidelity issues. Second, the policy rollouts in this work correspond to 8-second episodes. Achieving long-horizon (e.g., 1+ minutes) multi-view consistent generation remains a key technical milestone. Progress in long-horizon video generation based on latent-action models (Bruce et al.,

- 2024) offers a path to unlocking these capabilities for robotics. Third, the results in this report utilized human scoring of generated videos. To achieve a fully autonomous evaluation pipeline, future iterations will integrate automated scoring based on vision-language models (VLMs). Finally, improving the inference efficiency of video generation via optimized architectures (Hafner et al.,
- 2025) can further enhance the scalability of this evaluation paradigm.

Ultimately, this work demonstrates the massive potential impact of video models in robotics. The ability to evaluate robots in an infinitely rich and varied proxy of the world provides the necessary infrastructure for developing generalist embodied agents that operate usefully, capably, and safely in real-world environments.

##### Authors

Authors listed alphabetically by last name.

Krzysztof Choromanski, Coline Devin, Yilun Du, Debidatta Dwibedi, Ruiqi Gao, Abhishek Jindal, Thomas Kipf, Sean Kirmani, Isabel Leal, Fangchen Liu, Anirudha Majumdar, Andrew Marmon, Carolina Parada, Yulia Rubanova, Dhruv Shah, Vikas Sindhwani, Jie Tan, Fei Xia, Ted Xiao, Sherry Yang, Wenhao Yu, Allan Zhou.

##### Acknowledgements

Our work is made possible by the dedication and efforts of numerous teams at Google. We would like to acknowledge support from Dumitru Erhan, Shlomi Fruchter, Radu Soricut, Demetra Brady, Scott Crowell, Jason Baldridge, Juanita Bawagan, Dimple Vijaykumar, Aaron van den Oord, and Keerthana Gopalakrishnan. We would like to thank everyone on the Robotics team for their continued support and guidance.

##### References

1x world model: Evaluating bits, not atoms. Technical report, 1X Technologies, June 2025. URL https://www.1x.tech/1x-world-model.pdf. Accessed: 2025-10-22.

Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

Jorge Aldaco, Travis Armstrong, Robert Baruch, Jeff Bingham, Sanky Chan, Kenneth Draper, Debidatta Dwibedi, Chelsea Finn, Pete Florence, Spencer Goodrich, et al. ALOHA 2: An enhanced low-cost hardware for bimanual teleoperation. arXiv preprint arXiv:2405.02292, 2024.

Apurva Badithela, David Snyder, Lihan Zha, Joseph Mikhail, Matthew O’Kelly, Anushri Dixit, and Anirudha Majumdar. Reliable and scalable robot policy evaluation with imperfect simulators. arXiv preprint arXiv:2510.04354, 2025.

Federico Bianchi, Mirac Suzgun, Giuseppe Attanasio, Paul Röttger, Dan Jurafsky, Tatsunori Hashimoto, and James Zou. Safety-tuned LLaMAS: Lessons from improving the safety of large language models that follow instructions. arXiv preprint arXiv:2309.07875, 2023.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alex Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In arXiv preprint arXiv:2307.15818, 2023.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8): 1, 2024.

Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.

Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An open urban driving simulator. In Conference on robot learning, pages 1–16. PMLR, 2017.

Yilun Du, Mengjiao Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Joshua B Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. arXiv e-prints, pages arXiv–2302, 2023.

Francesca Favaro, Laura Fraade-Blanar, Scott Schnelle, Trent Victor, Mauricio Peña, Johan Engstrom, John Scanlon, Kris Kusano, and Dan Smith. Building a credible case for safety: Waymo’s approach for the determination of absence of unreasonable risk. arXiv preprint arXiv:2306.01917, 2023.

Francesca Favaro, Scott Schnelle, Laura Fraade-Blanar, Trent Victor, Mauricio Peña, Nick Webb, Holland Broce, Craig Paterson, and Dan Smith. Determining absence of unreasonable risk: Approval guidelines for an automated driving system deployment. arXiv preprint arXiv:2505.09880, 2025.

Jensen Gao, Suneel Belkhale, Sudeep Dasari, Ashwin Balakrishna, Dhruv Shah, and Dorsa Sadigh. A taxonomy for evaluating generalist robot policies. 2025a.

Yuan Gao, Mattia Piccinini, Yuchen Zhang, Dingrui Wang, Korbinian Moller, Roberto Brusnicki, Baha Zarrouki, Alessio Gambi, Jan Frederik Totz, Kai Storms, et al. Foundation models in autonomous driving: A survey on scenario generation and scenario analysis. arXiv preprint arXiv:2506.11526, 2025b.

GeminiRoboticsTeam, Abbas Abdolmaleki, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Ashwin Balakrishna, Nathan Batchelor, Alex Bewley, Jeff Bingham, Michael Bloesch, et al. Gemini Robotics 1.5: Pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer. arXiv preprint arXiv:2510.03342, 2025.

GeminiTeam, Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Yanjiang Guo, Lucy Xiaoyang Shi, Jianyu Chen, and Chelsea Finn. Ctrl-world: A controllable generative world model for robot manipulation. arXiv preprint arXiv:2510.10125, 2025.

Danijar Hafner, Wilson Yan, and Timothy Lillicrap. Training agents inside of scalable world models. arXiv preprint arXiv:2509.24527, 2025.

Abhishek Jindal, Dmitry Kalashnikov, Oscar Chang, Divya Garikapati, Anirudha Majumdar, Pierre Sermanet, and Vikas Sindhwani. Can AI perceive physical danger and intervene? arXiv preprint arXiv:2509.21651, 2025.

Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.

Anirudha Majumdar, Mohit Sharma, Dmitry Kalashnikov, Sumeet Singh, Pierre Sermanet, and Vikas Sindhwani. Predictive red teaming: Breaking policies without breaking robots. arXiv preprint arXiv:2502.06575, 2025.

NVIDIA. cosmos-predict2: World Foundation Models for Physical AI. https://github.com/ nvidia-cosmos/cosmos-predict2, 2025. GitHub repository.

Carolina Parada. Gemini Robotics on-device brings AI to local robotic devices. Google DeepMind Blog, 2025.

Wilbert Pumacay, Ishika Singh, Jiafei Duan, Ranjay Krishna, Jesse Thomason, and Dieter Fox. The colosseum: A benchmark for evaluating generalization for robotic manipulation. arXiv preprint arXiv:2402.08191, 2024.

Julian Quevedo, Ansh Kumar Sharma, Yixiang Sun, Varad Suryavanshi, Percy Liang, and Sherry Yang. Worldgym: World model as an environment for policy evaluation, 2025. URL https: //arxiv.org/abs/2506.00613.

Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, and Gianluca Corrado. Gaia-2: A controllable multi-view generative world model for autonomous driving, 2025.

Pierre Sermanet, Anirudha Majumdar, Alex Irpan, Dmitry Kalashnikov, and Vikas Sindhwani. Generating Robot Constitutions & Benchmarks for Semantic Safety. arXiv preprint arXiv:2503.08663,

2025. URL https://arxiv.org/abs/2503.08663.

Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing AI into the physical world. arXiv preprint arXiv:2503.20020, 2025.

Marcel Torne, Anthony Simeonov, Zechu Li, April Chan, Tao Chen, Abhishek Gupta, and Pulkit Agrawal. Reconciling reality through simulation: A real-to-sim-to-real approach for robust manipulation. arXiv preprint arXiv:2403.03949, 2024.

Aäron van den Oord and Elias Roman. State-of-the-art video and image generation with veo 2 and

imagen 3. Google DeepMind Blog, 2024. GoogleDeepMind Veo Team. Veo: a text-to-video generation system. 2025. Yi Ru Wang, Carter Ung, Grant Tannert, Jiafei Duan, Josephine Li, Amy Le, Rishabh Oswal, Markus

Grotz, Wilbert Pumacay, Yuquan Deng, et al. Roboeval: Where robotic manipulation meets structured and scalable evaluation. arXiv preprint arXiv:2507.00435, 2025.

Wayve. GAIA-3: Scaling world models to power safety and evaluation. 2025. Lin Zhang, Zherui Chen, Yihong Chen, Wen-Sheng Zhang, Xiang-Rong Zhang, Xin Li, Jian-Guo Lou,

and Dong-Mei Zhang. MM-SafetyBench: A comprehensive benchmark for safety evaluation of multimodal large language models. arXiv preprint arXiv:2402.10053, 2024.

Zhexin Zhang, Zhaowei Liu, Jialong Wang, He Wang, Qiang Zhang, Cunchao Zong, and Changbo Wang. Safetext: A benchmark for evaluating the physical safety of large language models. arXiv preprint arXiv:2310.15531, 2023. URL https://arxiv.org/abs/2310.15531.

Tony Z Zhao, Jonathan Tompson, Danny Driess, Pete Florence, Kamyar Ghasemipour, Chelsea Finn, and Ayzaan Wahid. ALOHA unleashed: A simple recipe for robot dexterity. arXiv preprint arXiv:2410.13126, 2024.

##### Appendix A. Out-of-distribution (OOD) Evaluations

The following images show examples of the different OOD scenarios.

###### A.1. Small distractor objects

[Figure 60]

[Figure 61]

(a) Axolotl (b) Duck

[Figure 62]

[Figure 63]

(c) Octopus (d) Penguin

[Figure 64]

(e) Turtle

- Figure 12 | Real-world scenes with small distractor objects.

###### A.2. Large distractor objects

[Figure 65]

[Figure 66]

(a) Bighorn sheep (b) Dolphin

[Figure 67]

[Figure 68]

(c) Teddy bear (d) Polar bear

[Figure 69]

(e) Golden retriever

- Figure 13 | Real-world scenes with large distractor objects.

###### A.3. Novel objects to be manipulated

[Figure 70]

[Figure 71]

(a) Brush (b) Jeep

[Figure 72]

[Figure 73]

(c) Pouch (d) Elephant

[Figure 74]

(e) Teacup

- Figure 14 | Real-world scenes with novel objects to be manipulated.

###### A.4. Table Background

[Figure 75]

[Figure 76]

[Figure 77]

(a) Red (b) Green (c) Blue

- Figure 15 | Real-world scenes with altered table backgrounds.

