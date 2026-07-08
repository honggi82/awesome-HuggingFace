# QUANTIPHY: A Quantitative Benchmark Evaluating Physical Reasoning Abilities of Vision-Language Models

[Figure 1]

Li Puyin1,∗, Tiange Xiang1,∗, Ella Mao1,∗, Shirley Wei1, Xinye Chen1, Adnan Masood2 Li Fei-Fei1,†, Ehsan Adeli1,†

1Stanford University, 2UST

*Equal First Authorship †Equal Last Authorship

Understanding the physical world is essential for generalist AI agents. However, it remains unclear whether state-ofthe-art vision perception models (e.g., large VLMs) can reason physical properties quantitatively. Existing evaluations are predominantly VQA-based and qualitative, offering limited insight into whether these models can infer the kinematic quantities of moving objects from video observations. To address this, we present QuantiPhy, the first benchmark designed to quantitatively measure a VLM’s physical reasoning ability. Comprising more than 3.3K video–text instances with numerical ground truth, QuantiPhy evaluates a VLM’s performance on estimating an object’s size, velocity, and acceleration at a given timestamp, using one of these properties as an input prior. The benchmark standardizes prompts and scoring to assess numerical accuracy, enabling fair comparisons across models. Our experiments on state-of-the-art VLMs reveal a consistent gap between their qualitative plausibility and actual numerical correctness. We further provide an in-depth analysis of key factors like background noise, counterfactual priors, and strategic prompting and find that state-of-the-art VLMs lean heavily on pre-trained world knowledge rather than faithfully using the provided visual and textual inputs as references when reasoning kinematic properties quantitatively. QuantiPhy offers the first rigorous, scalable testbed to move VLMs beyond mere verbal plausibility toward a numerically grounded physical understanding.

## arXiv:2512.19526v1[cs.AI]22Dec2025

[Figure 2]

[Figure 3]

Dataset: https://huggingface.co/datasets/PaulineLi/QuantiPhy-validation Project: https://quantiphy.stanford.edu/ Code: https://github.com/Paulineli/QuantiPhy Correspondence to: {puyinli, xtiange, eadeli}@stanford.edu

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

m/s²

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

pixel/s² m

[Figure 23]

[Figure 24]

pixel

[Figure 25]

[Figure 26]

?

[Figure 27]

m/s;m/s² m/s;m/s² m/s;m/s²

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

m m m

[Figure 39]

Figure 1. On a crowded city street, a bird’s nest falls from a branch, a car rushes by, an eagle flits over a building, and a person walks in a crosswalk — the real world is full of complex physical motion. To enable AI to understand and navigate this environment, it is essential for generalist embodied systems to reason about physical properties quantitatively. Because objects obey common laws of physics, their kinematic properties (such as size, velocity, and acceleration) are interrelated. This interdependence makes it possible for visual AI to systematically reason about these properties with respect to available priors. In this work, we present QUANTIPHY, the first benchmark to evaluate the reasoning ability of AI models on quantitative kinematic inference tasks.

#### 1. Introduction

Understanding the physical world has long been a challenge for artificial intelligence. Humans inhabit a world governed by physical laws. From an apple falling from a tree to the trajectory of a thrown ball, we have developed mathematical tools to measure and calculate physical attributes quantitatively. This quantitative understanding forms the foundation for all modern scientific advances.

General visual intelligence systems, such as VisionLanguage Models (VLMs), are developed in a manner that differs significantly from that of humans. They are trained to fit vast amounts of real-world data, which implicitly contain the abstract physical principles behind visual observations. Thus, assessing the numerical accuracy of VLMs’ reasoning about physical properties is a necessary next step. It is crucial for deploying applications such as embodied AI [12, 42], AR/VR [18, 26, 48–50], and autonomous driving [40].

Reasoning about quantitative physical properties from visual data can be challenging. For example, advanced methods like FoundationPose [44] require extensive prior information, including color, depth, object meshes, and camera parameters, to accurately localize an object in 3D space. Unfortunately, most of these priors are unavailable in “inthe-wild” captures. These challenges make it natural to ask whether large VLMs can leverage their rich implicit priors to reason end-to-end about precise kinematic and geometric properties. Moreover, a robust evaluation of VLM reasoning has the potential to help assess and improve the physical realism of videos created by generative models.

Studies of VLM physical understanding are not new. A variety of related benchmarks exist, spanning kinematics [58] and dynamics [52] to relationships [11] and scene understanding [54]. However, almost all existing benchmarks are Visual Question Answering (VQA)-based and qualitative. In this paradigm, ground-truth answers are effectively constrained by the prompt, and models are typically evaluated with multiple-choice questions. However, this VQA paradigm does not provide fine-grained evaluations of physical understanding. For example, if a model is asked to infer the size of a car from a video (ground truth: 3 meters), the incorrect answers 3.1 meters and 31 meters would be treated

- as equally wrong in a multiple-choice format. Quantitatively, however, the 31-meter answer is 10× worse. To truly push VLMs toward real-world applications, it is crucial to capture this numerical gap.

Our contributions in this work are four-fold: (I) We propose a new quantitative paradigm for evaluating the physical reasoning ability of VLMs, moving beyond the limitations of qualitative VQA. (II) We define a kinematic inference task for VLM physical reasoning that explicitly targets the challenge of understanding dynamics in videos: since kinematic

properties of an object, such as size, velocity, and acceleration, are mutually correlated quantities, our task formalizes how visual agents could, in principle, transform a single physical prior into a family of numerically grounded predictions in real-world units that are useful for understanding and acting in the physical world. (III) We present QUANTIPHY, the first benchmark to systematically evaluate a VLM’s quantitative reasoning on object kinematic properties in videos, spanning 2D/3D motion, static and dynamic priors, and diverse scene conditions, together with a standardized metric, prompting protocol, and leaderboard over 21 state-of-the-art models. (IV) We provide a detailed analysis of factors affecting VLM reasoning, including scene complexity, video availability, counterfactual priors, and chain-of-thought prompting. Summarizing the analysis, we conclude with one interesting finding: when estimating kinematic quantities, existing VLMs hallucinate by relying heavily on pre-trained world knowledge while hardly inferring from the actual reference video and text.

#### 2. Related Work

###### 2.1. Benchmarks for VLM Physical Understanding

For over a decade, researchers have recognized the need for AI to comprehend real-world physics. A series of benchmarks and evaluation protocols has been developed to assess the qualitative physical reasoning abilities of VLMs. Early efforts primarily focused on evaluating a model’s capacity to describe, predict, and explain basic physical events, such as collisions, falling, and rebounding in controlled synthetic environments [6, 58]. Subsequent works expanded this scope to include reasoning about inherent physical properties of objects, such as mass, friction, elasticity, and deformability [41]. More recently, comprehensive benchmarks like PhysBench [11] and STAR [45] have been proposed to assess broader aspects of physical reasoning. These include object relationships, physical scene understanding, complex physics-based dynamics, and physics-situated actions in more realistic or diverse environments [19, 60], aiming to simulate the real-world reasoning demands that embodied AI systems will face.

However, despite these advances, the vast majority of existing benchmarks adopt the VQA framework [3, 39, 51, 57]. In this framework, models are asked to provide multiplechoice selections or descriptive explanations of physical events. One relevant work, VSI-Bench [54], provides preliminary studies on basic spatial understanding using numerical metrics, but is limited to static objects, focusing more on the models’ perceptual ability than on emergent reasoning. Following that, Super-VSI [55] was recently proposed, demonstrating that VLMs empowered with the ability for numerical spatial understanding have strong potential in empowering embodied AI. Therefore, a need still exists for

[Figure 40]

Figure 2. Sample examples from QUANTIPHY, illustrating the four core task combinations defined by Dimensionality {2D, 3D} and Physical Prior {Static, Dynamic}, as described in section 3.1. Our collected data is diverse in nature, and each video is paired with multiple (prior, question, ground truth) triplets. Please see the supplementary materials for more data examples.

benchmarking VLM’s ability to perform quantitative reasoning over the geometric and kinematic properties of moving objects (e.g., estimating size, velocity, or acceleration in metric terms).

###### 2.2. Physical Reasoning Models

Although AI models have achieved impressive results on general tasks, their ability to reason about the physical world remains limited. For VLMs, state-of-the-art models such as ChatGPT-4o and Gemini-1.5 Pro achieve only around 60% accuracy on the PhysBench benchmark, far below humanlevel performance (∼95%) [11], indicating persistent challenges in physical reasoning. Generative models also struggle. Recent work shows that generated videos often violate basic physical laws [7], as these models are more sensitive to low-level visual cues (e.g., color, shape) than to underlying physical properties [21]. In embodied AI, recent systems have demonstrated basic physical reasoning abilities. They can interact with the environment [4], but their reasoning remains largely qualitative, lacking a precise and quantitative understanding of physical dynamics. These gaps underscore the need for benchmarks and models that transcend qualitative judgment and move toward accurate quantitative reasoning in physical contexts.

###### 2.3. Physical Understanding and Reasoning

In computer vision, techniques such as optical flow [8, 20], combined with object detection and tracking models like YOLO [33] and ByteTrack [59], enable accurate analysis of object motion and are widely used in broad applications [15, 36]. These methods demonstrate that quantitative information (e.g., velocity, displacement) can be reliably inferred from video inputs. Meanwhile, research in cognitive science shows that humans possess strong abilities to reason about scale, from microscopic to astronomical, by leveraging relational cues and prior knowledge [35]. Even in visually unfamiliar or simulated environments, people can extract physical rules and perform causal and predictive reasoning about object motion and interactions [1, 14, 23]. This highlights the potential for robust physical reasoning ability to emerge in large AI models from visual inputs, a crucial step toward grounded quantitative understanding.

#### 3. Methods

We introduce QUANTIPHY, the first benchmark designed to quantitatively evaluate VLMs’ physical reasoning ability on moving objects. In subsection 3.1, we first define the basic object kinematic properties to be assessed in QUANTIPHY, and present key statistics of it. Then, subsection 3.2 provides

details on the construction of QUANTIPHY, focusing on data collection.

###### 3.1. Overview of QUANTIPHY

The task of kinematic inference. The purpose of QUANTIPHY is to evaluate whether VLMs can utilize prior knowledge to reason about objects’ kinematic properties with numerical accuracy, which forms the foundation for a more sophisticated understanding of physics. Specifically, we focus on the translational movements of various objects.1

Given a video, we provide the VLM with a single physical prior for a source object (from the set {Sworld,Vtworld,Aworldt }, in real-world units) as textual input. The model is then prompted to quantitatively determine requested kinematic properties for a target object (which may be the same as or different from the source object) in world space.

We distinguish pixel space, where quantities are measured in pixels ([pixel], [pixel/s], [pixel/s2]), from world space, where they are expressed in physical units (e.g., [m], [m/s], [m/s2]). Consider a video capturing the translational movement of an object with a fixed camera. At any time t, the object’s location in pixel space, Xpixelt , can be obtained from the frames. From the discrete trajectory, we compute pixel-space velocity and acceleration via finite differences:

pixel t+dt−Xpixelt

pixel t+2dt−2Xpixelt+dt+Xpixelt

Vtpixel ≈ X

dt ; Apixelt ≈ X

dt2 .

The video thus defines the kinematics only in pixel units. To relate them to world space, we assume an unknown scalar scale factor γ > 0 (with units [world length/pixel]) such that, along the motion direction, Sworld = γSpixel, Vtworld = γVtpixel, Aworldt = γ Apixelt , where Spixel is an object size measured in pixels and Sworld the same size in a physical unit. When a single prior in world space is provided (object size Sworld, velocity Vtworld, or acceleration Aworldt at some time t), together with the corresponding pixel-space quantity from the video, γ is determined, and any other kinematic property in world space follows by rescaling its pixel-space counterpart.

See Figure 2 for examples of the kinematic tasks included in the benchmark. The performance of VLMs is measured by the numerical error between the VLM’s prediction and the annotated ground truth.

Benchmark setup. For a comprehensive evaluation of the kinematic movements above, QUANTIPHY is designed to include video-question pairs categorized along three primary

1We do not include rotational movements in this work; we defer this to future work and provide a discussion in section 6.

[Figure 41]

Figure 3. QUANTIPHY Statistics. The collected data and curated QA pairs are among four main setups with further breakdowns.

axes. We first describe the two axes that define the core reasoning task:

- • Dimensionality: {2D, 3D}. 2D movement assumes the object moves strictly in the x-y plane with no change in depth relative to the camera. 3D movement includes the z-axis, resulting in varying depth, which is intrinsically more challenging2.
- • Physical prior: {Static, Dynamic}. The Static prior indicates the provision of the object size Sworld, which is constant throughout the video. While Dynamic prior indicates velocity Vtworld or acceleration Aworldt at a given timestamp t.

Together, these two axes divide the Benchmark into four task categories: 2D-Static, 2D-Dynamic, 3D-Static, and 3D-Dynamic.3

Data statistics. Similar to [11, 54], data in QUANTIPHY is organized into triplets of (question/prompt, video, numerical ground truth). Each video may be paired with multiple questions and corresponding ground truth annotations. To ensure diversity, the four data setups were collected from various sources, yielding a total of 569 unique videos and 3355 questions. With proper post-processing, our collected videos typically have a duration of 2-3 seconds, occupying approximately 115MB of disk storage, making the benchmark suitable for use across various hardware settings. See Figure 3 for a detailed breakdown of the data statistics.

2We provide additional depth prior on 3D scenes. 3In the graphs and tables, we denote the four kinematic inference task

categories as 2S (2D-Static), 2D (2D-Dynamic), 3S (3D-Static), and 3D (3D-Dynamic) for short.

[Figure 42]

##### Data Collection Data Annotation Prior & Question QuantiPhy

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

ScenefileCamdepthPixelruler

[Figure 53]

[Figure 54]

- P: 𝑺 of pepperoni
- Q: 𝑽𝒕of cutter

Scraping Blender

Internet

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

- 2D Data

- 3D Data

[Figure 60]

- P: 𝑨𝒕 of cutter
- Q: 𝑺of cash money

[Figure 61]

[Figure 62]

(no depth variation)

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Simulation Lab

[Figure 69]

[Figure 70]

- P: 𝑨𝒕 of boat
- Q: 𝑺of pier

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

- P: 𝑺 of beach house
- Q: 𝑽𝒕of boat

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

- P: 𝑨𝒕 of PingPong + 𝑫
- Q: 𝑽𝒕of PingPong

Collection

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

(with depth variation)

[Figure 86]

- P: 𝑺 of bottle + 𝑫
- Q: 𝑨𝒕of PingPong

- Figure 4. The construction of QUANTIPHY proceeds in three sequential stages. First, we collect diverse raw videos from three different sources. Additionally, we segment these videos with solid plain background (described in subsection 3.2). Second, we obtain high-quality annotations, employing distinct labeling methods tailored to each data source to accurately capture the object’s physical properties. Finally, we formulate the benchmark tasks by associating each video with multiple (prior, question, ground truth) triplets. Each triplet is then categorized as either 2D or 3D, depending on the object’s movement relative to the camera.

###### 3.2. Benchmark Construction

out-of-distribution scenarios, we leverage the internet as a source for in-the-wild data. However, not all internet data is suitable for our benchmark, as our benchmark specifically requires consistent coverage of moving objects captured by a relatively static camera, along with reference objects of known dimensions (e.g., a standard coin) to serve as priors. Hence, we narrowed our scraping scope to websites hosting high-quality monocular videos and then manually inspected them to select those meeting our stringent requirements.

In QUANTIPHY, we constructed a large-scale dataset containing diverse video data from multiple sources.

Data collection. Our data combines simulated and realworld sources for controllability and practical applicability. The data collection process is visualized in Figure 4.

• Blender simulation: Blender allows us to render scenes that are both visually realistic and physically plausible, encompassing 2D and 3D motion. Simulating object movements in Blender provides complete control over the environment and guarantees precise ground-truth annotations. This control allows us to probe VLM’s reasoning capabilities by placing objects with different motion types within a single scene. It also enables us to study model robustness by holding motions fixed while systematically varying the background and visual noise. To enhance data diversity, we include simulations of scenes that are difficult or impossible to record in the real world, ranging from microscopic scales (e.g., red blood cells in vessels) to astronomical scales (e.g., galaxy movements).

We provide a detailed description of the data collection process in the supplementary materials Appendix B.

Additional segmented data. We apply SAM [32] to segment moving objects against solid backgrounds, doubling our dataset without additional annotation while enabling controlled analysis of background complexity effects.

Data annotation and question design. Blender simulations provide direct extraction of sizes, velocities, and accelerations via automation scripts. We construct Blender scenes using open-source 3D assets from Sketchfab [37] and BlenderKit. Automation scripts were developed for each scene to extract accurate, time-specific physical properties for any given object.

- • Lab capturing: Reconstructing objects in 4D (3D space + time) lets us annotate real-world object motion using a multi-view stereo setup. Our captures feature a wide diversity of movements, including free fall, sliding down slopes, pendulum motion, and bouncing. We also varied the physical properties of the objects, including those that are rollable and deformable. Further details on our setup can be found in the supplementary materials.
- • Internet scraping: To extend our dataset to more

However, data annotation is non-trivial for the other two sources. For lab captures, we utilize metric depth directly from depth cameras combined with multi-view stereo to achieve a full 4D reconstruction of the scene. We then apply per-pixel segmentation masks to outline the objects of interest in each video. For each video, we select a primary camera and use the object’s metric depth from that viewpoint as the depth prior provided to the VLM. Object movements

are then computed in world coordinates. For internet data, since the videos are monocular and lack multi-view information, we manually annotate sizes and displacements in pixel space and use reference objects with known priors to obtain the mapping to world scale.

We provide a detailed description of the data annotation process in the supplementary materials Appendix D.

Prompt design. As mentioned, the input to the VLM consists of a video paired with a single physical prior. For simplicity and effective integration, this prior is provided as a textual description (see Figure 2 for examples). In addition, we include textual cues in the prompt, such as ‘analyze the video and calculate the answer carefully’. We also add output constraints, instructing the model to ‘output only the numerical answer and unit’ in its final response.

We provide a detailed description of the prompt design process in the supplementary materials Appendix F.

#### 4. Evaluation on QUANTIPHY

###### 4.1. Evaluation Setup

Benchmark models. With different architecture designs, training data and protocals, different VLMs may have varying ability of physical understanding from videos. For better comprehensiveness, we include the evaluations of a total number of 21 state-of-the-art VLMs and variants in this work, consisting of 6 proprietary models include ChatGPT-

###### 5.1 [31], ChatGPT-5 [30], Gemini-2.5 Flash [16], Gemini2.5 Pro [17], Grok-4.1 [46], and Claude-4.5 Sonnet [2] and 15 open-sourced models (as listed in Table 1). The code supports multiple providers (OpenAI, Gemini, xAI, Anthropic, and Replicate API) with provider-specific parameters. Temperature is typically 0–0.1 for deterministic outputs. Token limits vary: OpenAI models use up to 10,000 tokens due to longer thinking steps, and open-sourced models with around 500–2,048.

Human studies. To complement our model evaluation and establish a reference point for human-level performance on quantitative physical reasoning, we conducted a survey study. Participants watched 18 videos and answered 1–3 quantitative kinematic questions using the same priors and task definitions as our VLM evaluation.

Importantly, humans and VLMs receive fundamentally different forms of input. VLMs operate directly on pixelaccurate video tensors, while human participants rely on visual perception, intuition, and coarse approximations.

Metric design. All tasks in QUANTIPHY require models to output numerical values. Following VSI-Bench [54], we use Mean Relative Accuracy (MRA) as the primary metric to measure the proximity between model predictions

and ground-truth answers. While accuracy based on exact matching is a simple baseline but too brittle for continuous, noisy physical measurements in QUANTIPHY, MRA evaluates whether the relative error falls below a set of tolerance thresholds and averages the resulting accuracies, offering a more calibrated and robust notion of when models are “accurate enough” for physically grounded reasoning (see subsection A.2 in the supplementary material for further discussion).

Concretely, we consider a set of confidence thresholds C = {0.1,0.2,...,0.9,0.95}, and define MRA for a prediction yˆ with ground truth y as

1 10 θ∈C

MRA =

|yˆ − y| |y|

< 1 − θ ,

where (·) is the indicator function. Intuitively, larger θ corresponds to a stricter tolerance 1 − θ on the relative error, and MRA averages accuracy across a spectrum of such tolerances, providing a more informative measure than a single-threshold accuracy.

Our benchmark organizes questions into four kinematic categories, 2D-Static, 2D-Dynamic, 3D-Static, and 3D-Dynamic. For each model and each category, we compute the category-level score by averaging the questionlevel MRA over all questions in that category for which the model produces a valid numerical answer. The overall score of a model is then obtained as the unweighted mean of its four category-level scores.

A practical challenge is that not all VLMs consistently output sensible numerical predictions for our reasoning tasks. For every video–question pair, we query a model up to five times with the same prompt, stopping early if any response contains a parseable numerical value. If none of the five responses yields a valid number, we regard the model as failing to answer that question.

###### 4.2. Main Results

Table 1 summarizes performance on QUANTIPHY across four task types (2D-Static, 2D-Dynamic, 3D-Static, and 3D-Dynamic). Overall, we find that quantitative kinematic inference remains challenging for current VLMs: even the best systems do not yet reach human performance, despite in principle having access to more precise information.

Human baseline. Human annotators achieve an average MRA of 55.6 across all categories, with scores between 50.0 (2D-Static) and 59.1 (2D-Dynamic). This range is consistent with the fact that humans do not have direct access to pixel-level measurements and must instead rely on coarse visual estimation (e.g., counting grid lines or comparing to

familiar objects), but still demonstrates that the tasks are solvable with reasonable accuracy.

Proprietary models. ChatGPT-5.1 attains the highest overall score with 53.1 MRA, followed by Gemini-2.5-Pro

- at 49.6. ChatGPT-5.1 slightly surpasses humans on the 2D-Dynamic category, yet none of them surpass the human average. Other closed models such as GPT-5 and Claude Sonnet 4.5 are substantially weaker, with overall MRA around 32.6 and 22.8, respectively.

Open-weight models. Open-weight models exhibit a wide performance spread. The best open-weight system, Qwen3-VL-Instruct-32B, reaches 46.0 overall MRA, with strong 2D-Dynamic and 3D-Dynamic scores, followed by InternVL-3.5-30B (40.7) and Qwen3-VL-Instruct-8B (38.8). These models are clearly below the top proprietary ones, but already comparable to mid-tier closed models. Smaller open-weight models such as Phi-4 Multimodal and SmolVLM-256M still achieve non-trivial MRA, yet remain far from both human and large-model performance.

Scaling effects. To better understand the role of model scale, we compare models within the same family across different parameter sizes. Within the Qwen3-VL family, average MRA increases from 29.0 at 2B parameters to 38.8 at 8B and further to 46.0 at 32B. A similar trend is observed for InternVL, where InternVL-3.5-30B (40.7) substantially outperforms its 8B (35.4) and 2B (25.0) variants. Notably, scaling benefits are most pronounced on dynamic categories (2D-Dynamic and 3D-Dynamic), suggesting that larger models are better able to integrate temporal information for quantitative inference. However, these gains exhibit diminishing returns and do not close the gap to either top proprietary models or human performance, indicating that scale alone is insufficient for faithful physical reasoning.

Gap to super-human performance. By collecting results from both humans and VLMs, we observe an interesting pattern. Importantly, human performance does not represent the theoretical ceiling for QUANTIPHY. While humans must rely on coarse visual approximations, an ideal agent with precise frame-level access to pixel coordinates could recover the world–pixel scale and compute target quantities exactly. In theory, VLMs should be capable of significantly outperforming humans by leveraging this exact pixel information to perform precise algebraic operations. The fact that the best current systems cluster around 50% MRA, comparable to or below human baselines, suggests they still fundamentally under-utilize visual precision and physical priors, leaving robust and accurate quantitative reasoning for kinematic inference tasks as an open challenge.

#### 5. Dissecting Quantitative Reasoning in VLMs

Beyond directly evaluating VLMs as black boxes on kinematic inference tasks, we next take an inside look at how they arrive at their quantitative conclusions. Specifically, we investigate three aspects: (i) how scene context, such that background complexity and the number of objects, modulates task difficulty; (ii) the extent to which VLMs faithfully use the provided video and priors, rather than relying on memorized world knowledge; and (iii) whether structured prompting with step-by-step guidance can systematically improve kinematic inference.

###### 5.1. Effect of Scene Context

For a more fine-grained analysis, we further categorize each video-question pair along a third axis describing the visual environment: Scene Difficulty ∈ {SX,MX,SS,MS,SC,MC}. The first letter indicates whether there is a single (S) or multiple (M) moving objects. The second letter indicates the background: a solid plain color (X)4, a simple but textured scene (S), or a visually complex scene (C).

Figure 5 summarizes MRA across these categories. Overall, we observe that background complexity has only a mild effect on model’s performance on the task. Models perform slightly better in the SAM-denoised condition than in the simple-texture (S) background, suggesting that removing irrelevant clutter reduces distractions and stabilizes quantitative estimates. Interestingly, performance in visually complex scenes (C) is above the other two background conditions for most models. A plausible explanation is that realistic backgrounds provide additional reference cues (e.g., tiles, windows, or road markings) that help the model infer scale and motion.

In contrast, the number of objects in the scene exhibits a clearer trend: setups with multiple objects (MX, MS, MC) consistently yield higher MRA than their single object counterparts (SX, SS, SC). Having more objects gives the model extra reference targets (e.g., another ball, a ruler-like structure), which can be used as implicit comparison standards for both size and speed.

These observations highlight that VLMs robustly benefit from richer background information and relational structure in the scene.

###### 5.2. Do VLMs Use Videos and Priors Faithfully?

QUANTIPHY is designed to assess whether VLMs can truly understand physical events captured by visual sensors. It is therefore crucial to investigate whether these systems genuinely comprehend the provided visual signals, rather than

4Some X-type scenes are constructed by segmenting the target object with SAM2 and compositing it onto a uniform background, i.e., a fully “denoised” variant of the original video.

###### Kinematic Categories

Models Size

Avg.

2S 2D 3S 3D Proprietary models

ChatGPT-5.1 [31] – 46.3 56.2 51.5 58.3 53.1 Gemini-2.5 Pro [17] – 44.8 57.5 42.4 53.7 49.6

[Figure 87]

[Figure 88]

Gemini-2.5 Flash [16] – 40.3 53.2 43.6 57.4 48.6 Grok 4.1 (Fast Reasoning) [46] – 39.4 49.5 42.4 48.6 45.0

[Figure 89]

ChatGPT-5 [30] – 36.6 35.0 25.9 33.1 32.6 Claude Sonnet 4.5 [2] – 19.6 23.0 19.6 29.1 22.8

Open-weight models Qwen3-VL-Instruct-32B [5] 32B 35.8 51.6 43.2 53.4 46.0

[Figure 90]

InternVL-3.5-30B [10] 30B 36.7 45.4 38.6 42.0 40.7 Qwen3-VL-Instruct-8B [5] 8B 26.0 47.8 35.1 46.3 38.8

[Figure 91]

[Figure 92]

InternVL-3.5-8B [10] 8B 27.3 41.8 34.4 38.3 35.4

Molmo-7B [13] 7B 30.0 43.1 24.4 36.6 33.5 Phi-4-Multimodal-Instruct [29] 5.6B 33.4 42.3 25.4 28.4 32.4

Qwen3-VL-Instruct-2B [5] 2B 27.1 39.0 17.6 32.1 29.0 SmolVLM-Instruct [27] 0.26B 31.6 34.4 20.0 27.8 28.5 InternVL-3.5-2B [10] 2B 25.0 31.1 16.6 27.4 25.0

VILA-7B [24] 7B 23.0 29.8 14.4 23.0 22.6 CogVLM2 Video [43] 12B 19.4 28.7 12.7 27.9 22.2

Phi-3-Mini-128K-Instruct-3.8B [28] 3.8B 17.3 14.7 19.5 18.6 17.5

LLaVA-13B [25] 13B 14.4 22.1 8.0 16.5 15.2 MiniCPM-V 4.5 [56] 8B 27.6 26.3 0.4 0.0 13.6

Fuyu-8B [22] 8B 9.5 14.7 9.5 16.2 12.5 Human Baseline – 50.0 59.1 55.2 57.9 55.6

Table 1. Evaluation results on QUANTIPHY. We report Mean Relative Accuracy (MRA %) on four kinematic categories (2S, 2D, 3S, 3D) and their average. Dark cell marks the best overall model and light cell marks the best open-weight model.

Video + Prior Prior only Counterfactual CoT

56.1 39.0 15.4 27.7 60.9 46.1 29.9 49.8

- 49.8 36.1 14.4 22.4 47.5 44.3 31.6 39.5 34.2 50.8 29.6 53.7 25.4 16.6 11.6 25.9

- 50.1 37.2 34.0 23.1 45.4 – 12.1 17.6 40.5 24.9 12.0 21.0

- 37.0 19.3 29.7 18.2

- 39.8 – 14.7 15.9

- 40.0 20.1 9.2 23.5 34.9 28.2 25.2 25.6

- 38.9 – 14.3 17.8 32.7 25.1 22.5 21.5 31.8 – 14.1 10.0

- 28.5 – 9.5 26.4 11.1 10.3 8.4 7.2 20.2 – 13.9 14.4

- 29.7 – 19.9 24.1 14.3 – 9.0 21.1

– – – –

Table 2. Extensive results on an analysis subset. We report Mean Relative Accuracy (MRA) in %. Rows follow the same model order as in Table 1.

merely producing plausible guesses. Here, we re-evaluate all models on a controlled subset of 1612D video-prior pairs,5 using two complementary probes.

Key finding: VLMs rely more on learned prior knowledge than visual inputs for physical reasoning. We first compare the default video+prior setting against a prior-only setting in which the video is removed but the prompt (including the physical prior, object description, and question) is kept unchanged. Hypothetically, without the reference video, we would expect an evident performance drop, since VLMs would be forced to guess; even if such guesses are reasonable, they may not correspond to the specific instance. However, across most models we observe only modest differences between the two conditions: on the 161-pair subset, MRA in the prior-only setting is often close to, and sometimes only slightly below, the video+prior setting, even for strong systems such as ChatGPT-5.1, Gemini 2.5 Pro/Flash, and Qwen3-VL-Instruct-32B. In other words, models can already obtain reasonably high scores by relying on their internal prior knowledge about typical object sizes and speeds, with limited added value from the actual video frames. This suggests that, in our tasks, many VLMs behave less like visual measurers and more like powerful guessers conditioned on textual hints.

5Due to resource constraints, the experiments in subsection 5.2 and subsection 5.3 are run on a 2D-only subset of 161 instances, and MRA values therefore differ slightly from those in Table 1.

Key finding: VLMs (mostly) do not reason but memorize. To further test whether models faithfully utilize the conditioning prior and video to infer the target physical quantity, we perform a counterfactual analysis. For each of the 161 instances, we construct a family of counterfactual prompts by multiplying the original physical prior by a scalar factor α ∈ {0.001,0.01,0.1,0.2,5,50,100,200,500,700}, while keeping the video and question unchanged. Hypothetically, if a model is capable of correct kinematic reasoning, its prediction should scale accordingly, tracking the counterfactual ground truth ycf = α · y. As a result, we would expect no, or at least only modest, performance drop in this test. However, in practice, we find that even the best models obtain very low scores in this setting: most models’ MRA drops by 80%, and even the strongest model drops by 70%. Despite being given a numerically precise but altered prior, the outputs remain close to the original physical magnitudes implied by real-world experience, rather than those dictated by the provided priors.

Taken together, these results lead to a consistent conclusion: existing VLMs are not yet input-faithful quantitative reasoners. They only weakly exploit pixel-level information in videos, and they do not reliably condition on the exact numerical priors provided in the prompts. Instead, their quantitative kinematic inferences are dominated by internal, pre-trained world knowledge, with visual evidence and explicit priors acting more as soft hints than hard constraints.

[Figure 93]

- Figure 5. Effect of scene context. We plot the MRA (%) scores for all benchmark models on different categories, sorted in descending order according to their average MRA performance.

###### 5.3. Do Structured Prompts Help VLMs Reason?

Based on the above observation that VLMs rely more heavily on text prompts, we therefore further investigate whether a more structured “chain-of-thought” reasoning pattern can improve quantitative physical predictions. Instead of directly asking for the target quantity, we decompose each question into a four-step chain:

- (I) Pixel-level source property: “What is [the ground-truth object’s property] in pixels?”
- (II) Scale estimation: “What is the proportional relationship between pixels and [a kinematic scale]?”
- (III) Pixel-level target property: “What is [the inference target’s property] in pixels?”
- (IV) World-level target property: “What is [the inference target’s real-world property]?”

We query each model sequentially on the same analysis subset as in subsection 5.2. If the model produces a parseable numerical answer at a given step, we append both the question and the extracted answer as additional context for the next step. If the model fails to generate a valid number at a step, subsequent prompts omit that step and its answer. We evaluate models’ final answers with MRA.

As shown in Table 2, chain-of-thought (CoT) prompting is less helpful than anticipated. Among the 21 models we examined, only three show any improvement, with ChatGPT5 and Fuyu-8B exhibiting a noticeable increase in MRA. For the remaining 19 models, including several strong openweight systems, performance under our CoT protocol is worse than under direct zero-shot prompting, sometimes by a large margin. In other words, under our setup, explicitly spelling out pixel measurement, scale estimation, and rescaling does not systematically improve current VLMs’ quantita-

tive reasoning. Many models appear unable to reliably solve the intermediate numeric subproblems, so decomposing the task mainly amplifies and propagates early errors.

Our analysis suggests that existing VLMs can exploit visual cues, priors, and structured prompts to reason quantitatively, but do so in a rather brittle and often inconsistent manner. QUANTIPHY thus provides not only a benchmark for aggregate performance, but also a diagnostic tool for probing where quantitative physical reasoning succeeds, fails, and how it might be improved. We provide additional studies in the supplementary material Appendix A to further support our findings.

#### 6. Discussion and Future Work

Summary of Findings. Our key finding is that state-of-theart VLMs have not yet established a reliable link between visual observations and quantitative physical facts. This disconnect manifests as a critical lack of input faithfulness: although models process video inputs, our extensive studies show that they hardly infer kinematic properties from the actual pixel-level information. Instead, they exhibit a strong reliance on parametric priors, often overriding explicit user inputs and visual evidence in favor of memorized world knowledge. Consequently, current systems act more as approximate guessers” based on semantic context rather than precise visual measurers,” limiting their reliability for real-world embodied agents.

Limitations. Our study has several limitations that suggest avenues for future work. The dataset focuses exclusively on translational movement, omitting rotational dynamics, and utilizes a fixed camera perspective, which simplifies the task compared to real-world scenarios with dynamic viewpoints. Additionally, we examined only rigid objects, excluding soft bodies and deformable materials. Finally, our dataset is relatively simplified, featuring isolated movements rather than complex, multi-object interactions.

Conclusion and Future Work. These limitations highlight promising directions for future research. First, a more comprehensive video dataset is needed to evaluate VLMs on more diverse physics, incorporating the complexities we omitted: rotational dynamics, deformable objects, varied camera perspectives, and complex multi-body interactions. Second, our findings can inform new VLM training methodologies, such as physics-informed objectives or specialized pre-training on physics-rich data. Ultimately, this research aims to advance the development of generalist embodied AI agents capable of sophisticated reasoning and interaction with the physical world.

#### Acknowledgment

The authors thank Juze Zhang and Heng Yu for their help with lab data capturing. This work was partially funded by the NIH Grant R01AG089169, Stanford HAI Hoffman-Yee Award, Stanford HAI graduate fellowship, and UST.

#### References

- [1] Kelsey R Allen, Kevin A Smith, and Joshua B Tenenbaum. Rapid trial-and-error learning with simulation supports flexible tool use and physical reasoning. Proceedings of the National Academy of Sciences, 117(47):29302–29310, 2020. 3
- [2] Anthropic. Claude 4.5 sonnet. https://claude.ai/,

2025. Accessed: 2025-11-14. Large language model. 6, 8, 31

- [3] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425– 2433, 2015. 2
- [4] Alisson Azzolini, Hannah Brandon, Prithvijit Chattopadhyay, Huayu Chen, Jinju Chu, Yin Cui, Jenna Diamond, Yifan Ding, Francesco Ferroni, Rama Govindaraju, et al. Cosmos-reason1: From physical common sense to embodied reasoning. arXiv preprint arXiv:2503.15558, 2025. 3
- [5] Shuai Bai and et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 8, 31
- [6] Anton Bakhtin, Laurens van der Maaten, Justin Johnson, Laura Gustafson, and Ross Girshick. PHYRE: A New Benchmark for Physical Reasoning, 2019. 2
- [7] Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. arXiv preprint arXiv:2503.06800, 2025. 3
- [8] Steven S. Beauchemin and John L. Barron. The computation of optical flow. ACM computing surveys (CSUR), 27(3):433– 466, 1995. 3
- [9] Zhongang Cai, Yubo Wang, Qingping Sun, Ruisi Wang, Chenyang Gu, Wanqi Yin, Zhiqian Lin, Zhitao Yang, Chen Wei, Hui En Pang, Xuanke Shi, Kewang Deng, Xiaoyang Han, Zukai Chen, Jiaqi Li, Xiangyu Fan, Hanming Deng, Lewei Lu, Bo Li, Ziwei Liu, Quan Wang, Dahua Lin, and Lei Yang. Holistic evaluation of multimodal llms on spatial intelligence. arXiv preprint arXiv:2508.13142, 2025. 4
- [10] Zhe Chen and et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic capabilities. arXiv preprint arXiv:2312.14238, 2024. 8, 31
- [11] Wei Chow, Jiageng Mao, Boyi Li, Daniel Seita, Vitor Guizilini, and Yue Wang. PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding, 2025. 2, 3, 4
- [12] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, et al. Palm-e: An embodied multimodal language model. 2023. 2

- [13] Allen Institute for AI. Molmo: A family of open visionlanguage models. AI2 Blog, 2025. 8, 31
- [14] Tobias Gerstenberg, Noah D Goodman, David A Lagnado, and Joshua B Tenenbaum. A counterfactual simulation model of causal judgments for physical events. Psychological review, 128(5):936, 2021. 3
- [15] Andrea Giachetti, Marco Campani, and Vincent Torre. The use of optical flow for road navigation. IEEE transactions on robotics and automation, 14(1):34–48, 2002. 3
- [16] Google. Gemini 2.5 flash [large language model]. https:// deepmind.google/models/gemini/flash/, 2025. Accessed: 2025-11-14. 6, 8, 31
- [17] Google. Gemini 2.5 pro [large language model]. https: //deepmind.google/models/gemini/pro/, 2025. Accessed: 2025-11-14. 6, 8, 31
- [18] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022. 2
- [19] Madeleine Grunde-McLaughlin, Ranjay Krishna, and Maneesh Agrawala. Agqa: A benchmark for compositional spatio-temporal reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11287–11297, 2021. 2
- [20] Berthold KP Horn and Brian G Schunck. Determining optical flow. Artificial intelligence, 17(1-3):185–203, 1981. 3
- [21] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How Far is Video Generation from World Model: A Physical Law Perspective,

2024. 3

- [22] Adept AI Labs. Fuyu-8b: A smaller, faster multimodal model. Adept AI Blog, 2023. 8, 31
- [23] Shiqian Li, Kewen Wu, Chi Zhang, and Yixin Zhu. On the learning mechanisms in physical reasoning. Advances in Neural Information Processing Systems, 35:28252–28265,

2022. 3

- [24] Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. arXiv preprint arXiv:2312.07533, 2023. 8, 31
- [25] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485,

2023. 8, 31

- [26] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding. Advances in Neural Information Processing Systems, 36:46212–46244, 2023. 2
- [27] Francesco Marafioti and et al. Smolvlm: Efficient multimodal ai, 2025. As summarized by Emergent Mind. Refers to underlying technical papers. 8, 31
- [28] Microsoft. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024. 8, 31
- [29] Microsoft. Empowering innovation: The next generation of the phi family. Microsoft Azure Blog, 2025. 8, 31

- [30] OpenAI. Chatgpt, version 5.1 [large language model]. https://chat.openai.com/, 2025. Accessed: 202511-13. 6, 8, 31
- [31] OpenAI. ChatGPT version 5.1 [large language model]. https://chatgpt.com/, 2025. Accessed: 2025-11-14. 6, 8, 31
- [32] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 5, 20
- [33] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 779–788, 2016. 3
- [34] Tianhe Ren, Qing Jiang, Shilong Liu, Zhaoyang Zeng, Wenlong Liu, Han Gao, Hongjie Huang, Zhengyu Ma, Xiaoke Jiang, Yihao Chen, et al. Grounding dino 1.5: Advance the” edge” of open-set object detection. arXiv preprint arXiv:2405.10300, 2024. 21
- [35] Ilyse Resnick, Alexandra Davatzes, Nora S Newcombe, and Thomas F Shipley. Using relational reasoning to learn about scientific phenomena at unfamiliar scales. Educational Psychology Review, 29(1):11–25, 2017. 3
- [36] Keattisak Sangsuwan and Mongkol Ekpanyapong. Videobased vehicle speed estimation using speed measurement metrics. IEEE Access, 12:4845–4858, 2024. 3
- [37] Sketchfab, Inc. Sketchfab. https://sketchfab.com/,

2025. Accessed: 2025-11-14. 5

- [38] Deepak* Sridhar, Kartikeya* Bhardwaj, Jeya Pradha Jeyaraj, Nuno Vasconcelos, Ankita Nayak, and Harris Teague. Video reasoning without training, 2025. 4
- [39] Makarand Tapaswi, Yukun Zhu, Rainer Stiefelhagen, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. Movieqa: Understanding stories in movies through question-answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4631–4640, 2016. 2
- [40] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289, 2024. 2
- [41] Hsiao-Yu Tung, Mingyu Ding, Zhenfang Chen, Daniel M Bear, Chuang Gan, Joshua B Tenenbaum, Daniel L K Yamins, Judith Fan, and Kevin A Smith. Physion++: Evaluating Physical Scene Understanding that Requires Online Inference of Different Physical Properties. 2
- [42] Quan Vuong, Sergey Levine, Homer Rich Walke, Karl Pertsch, Anikait Singh, Ria Doshi, Charles Xu, Jianlan Luo, Liam Tan, Dhruv Shah, et al. Open x-embodiment: Robotic learning datasets and rt-x models. In Towards Generalist Robots: Learning Paradigms for Scalable Skill Acquisition@ CoRL2023, 2023. 2
- [43] Wenyi Wang and et al. Cogvlm2: A versatile visual language model for both image and video understanding. arXiv preprint arXiv:2406.11438, 2024. 8, 31

- [44] Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchfield. Foundationpose: Unified 6d pose estimation and tracking of novel objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17868–17879,

2024. 2, 29

- [45] Bo Wu, Shoubin Yu, Zhenfang Chen, Joshua B Tenenbaum, and Chuang Gan. Star: A benchmark for situated reasoning in real-world videos. arXiv preprint arXiv:2405.09711, 2024. 2
- [46] xAI. Grok-4.1. https://x.ai/grok, 2025. Accessed: 2025-12. 6, 8
- [47] xAI. Grok 4.1 API Documentation. xAI, 2025. 31
- [48] Tiange Xiang, Adam Sun, Scott Delp, Kazuki Kozuka, Li Fei-Fei, and Ehsan Adeli. Wild2avatar: Rendering humans behind occlusions. arXiv preprint arXiv:2401.00431, 2023. 2
- [49] Tiange Xiang, Adam Sun, Jiajun Wu, Ehsan Adeli, and Li Fei-Fei. Rendering humans from object-occluded monocular videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3239–3250, 2023.
- [50] Tiange Xiang, Kuan-Chieh Wang, Jaewoo Heo, Ehsan Adeli, Serena Yeung, Scott Delp, and Li Fei-Fei. Neuhmr: Neural rendering-guided human motion reconstruction. In 2025 International Conference on 3D Vision (3DV), pages 1518–1528. IEEE, 2025. 2
- [51] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786,

2021. 2

- [52] Xinrun Xu, Pi Bu, Ye Wang, B¨orje F. Karlsson, Ziming Wang, Tengtao Song, Qi Zhu, Jun Song, Zhiming Ding, and Bo Zheng. DeepPHY: Benchmarking Agentic VLMs on Physical Reasoning. arXiv preprint arXiv:2508.05405, 2025. 2
- [53] Jihan Yang, Shusheng Yang, Anjali Gupta, Rilyn Han, Li FeiFei, and Saining Xie. Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 4
- [54] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025. 2, 4, 6
- [55] Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis Brown, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, et al. Cambrian-s: Towards spatial supersensing in video. arXiv preprint arXiv:2511.04670, 2025. 2
- [56] Yuan Yao, Tianyu Yu, and et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.10032, 2024. 8, 31
- [57] Kexin Yi, Jiajun Wu, Chuang Gan, Antonio Torralba, Pushmeet Kohli, and Josh Tenenbaum. Neural-symbolic vqa: Disentangling reasoning from vision and language understanding. Advances in neural information processing systems, 31, 2018. 2

- [58] Kexin Yi, Chuang Gan, Yunzhu Li, and Pushmeet Kohli. CLEVRER: COLLISION EVENTS FOR VIDEO REPRESENTATION AND REASONING. 2020. 2
- [59] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. In European conference on computer vision, pages 1–21. Springer, 2022. 3
- [60] Zhicheng Zheng, Xin Yan, Zhenfang Chen, Jingzhou Wang, Qin Zhi Eddie Lim, Joshua B Tenenbaum, and Chuang Gan. Contphy: Continuum physical concept learning and reasoning from videos. arXiv preprint arXiv:2402.06119, 2024. 2

### QUANTIPHY: A Quantitative Benchmark Evaluating Physical Reasoning

[Figure 94]

### Abilities of Vision-Language Models Supplementary Material

#### Contents

- A. More Studies and Results 2

- A.1. Additional Case Studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
- A.2. Metric Design Justification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- A.3. Model MRA Distribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

- B. Dataset Construction Guidelines 6

- B.1. General Principles . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- B.2. Video Types . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- B.2.1. Video Categories Definition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- B.2.2. Quantitative breakdown of video types. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

- C. Details of Data Collection 9

- C.1. Blender Simulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- C.1.1. Blender Toolkits and Asset Sources . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- C.1.2. Two Motion Simulation Types . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- C.1.3. Blender Videos Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

- C.2. Lab Capturing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- C.3. Internet Scraping . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- C.4. Segmented Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- C.5. Quality Control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- C.6. Ethical Considerations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- D. Details of Data Annotation 22

- D.1. Blender Simulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- D.1.1. Size . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- D.1.2. Displacement or Path. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- D.1.3. Velocity and Acceleration. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- D.1.4. Depth and Distance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- D.2. Lab Data Annotation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- D.3. Internet Data Annotation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- D.4. Segmented Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

- E. Vision-Language Models 31
- F. Prompt Design 31
- G. Answer Retrieval and Parsing 34
- H. Human Study Details. 34

- H.1. Participants . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- H.2. Task Construction and Experimental Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- H.3. Evaluation Metric . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- H.4. Results and Observations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36

- I. Sketchfab Model Sources 36

#### A. More Studies and Results

###### A.1. Additional Case Studies

To better understand how Vision-Language Models (VLMs) solve kinematic inference tasks beyond aggregate scores, we conduct a qualitative case study on the top-performing model, ChatGPT-5.1, using its Thinking mode in the standard user-facing interface. For each selected video–text pair, we repeatedly query the model and inspect the toolaugmented chain-of-thought until we obtain representative traces that are syntactically well-formed and numerically valid.6 We then analyze both the final numerical answers and the intermediate reasoning steps.7

Overall, we observe a sharp contrast between successful instances, where the model follows a textbook-like “measure pixels → apply prior → compute target” pipeline, and failure modes, where it largely ignores the video and the provided prior, and instead falls back to pre-trained world knowledge or generic heuristics. Below we discuss four representative cases.

- Case 1: Faithful pixel–prior reasoning. Figure 6 shows a 2D scene with a yellow car moving laterally. The model is asked two questions: (i) given that the car’s length is 5.67m, what is its speed at 2.0s; and (ii) what is the car’s width in meters. In this instance, ChatGPT-5.1’s chain-ofthought closely matches the intended reasoning procedure. The model first identifies the relevant frames around t = 2.0s, uses OpenCV-style tools to obtain bounding boxes, and explicitly treats the longer side of the box (135 px) as the car’s length in pixel space. It then calibrates a pixel-to-meter scale from the given length prior (5.67 m), and computes the width as

width ≈

58 135 × 5.67m ≈ 2.44m,

which is close to the ground truth width and achieves high relative accuracy. Here the model behaves as an input-faithful visual measurer: it grounds both the prior and the kinematic target in pixel space and performs the correct proportional reasoning. When this pipeline is followed, the resulting numerical answers are often near the ground truth.

- Case 2: Counterfactual prior breaks faithfulness. In the second case (Figure 7), we reuse the same video but multiply

- 6We prompt the model multiple times and observe substantial variability: on the same instance, some runs produce accurate and well-structured reasoning, while others fail to parse the question or ignore key inputs. We therefore collect several responses per instance and manually select representative traces that are syntactically coherent and numerically valid for detailed analysis. A similar instability is also present in our API-based evaluation, even with temperature fixed to 0; in the main benchmark, we address this by running multiple trials and recording a failure rate.
- 7We emphasize that this analysis is diagnostic rather than evaluative: we study one specific model’s internal behavior to illustrate broader patterns of (un)faithful quantitative reasoning.

Video

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Text Prompt

You are an expert analyst specializing in physics measurements. Based on the provided context and ques on, provide ONLY the numerical answer with units. No explana on or reasoning needed. Format your response as: [value] [unit] Example: 2.5 cm Be as accurate as possible with measurements and calcula ons. Please give me an es mated answer even if you are not sure. Ques on: Given that length of the yellow car = 5.67m, what is the speed of the yellow car at 2.0s in m/s?

[Figure 101]

GPT-5.1 Thinking

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

GPT-5.1 Answer

###### Ground Truth

Relative Accuracy

5.2 m/s 5.07

1

Text Prompt

Ques on: Given that length of the yellow car = 5.67m, what is the width of the yellow car in meters?

[Figure 107]

###### GPT-5.1 Thinking

Calculating car length and width I’m trying to calculate the car’s length and width from the bounding box. The car's length aligns along the x-axis, so I’ll use the wider dimension in pixels (135 px) to represent its actual length of 5.67 m. For the width, I’ll compute its proportion: 58/135 * 5.67 m. Analyzed

[Figure 108]

###### GPT-5.1 Answer

###### Ground Truth

Relative Accuracy

2.436 m 2.43

1

Figure 6. Case 1: Faithful pixel–prior reasoning.

the car-length prior by a counterfactual factor of 1000, changing the input to “length of the yellow car = 5670m.” The task is again to infer speed at 2.0s and the car’s width. In its Thinking trace, the model explicitly notes that “5670m” is an implausible car length and expresses confusion. Crucially, instead of continuing to rely on pixel measurements and the (counterfactual) prior, it effectively abandons the video and the numeric input. For the width question, it switches to a generic heuristic, assuming a “typical car’s width-to-length ratio” and hallucinating a plausible-looking width independent of the actual scene. The final width prediction happens to have high relative accuracy (close to 0.9), but this success is not input-faithful; it arises from pre-trained knowledge about cars rather than from the specific video or the given prior. This case highlights a key risk that purely

###### Text Prompt (Video Ablated)

Video

[Figure 109]

[Figure 110]

You are an expert analyst specializing in physics measurements. Based on the provided context and ques on, provide ONLY the numerical answer with units. No explana on or reasoning needed. Format your response as: [value] [unit] Example: 2.5 cm Be as accurate as possible with measurements and calcula ons. Please give me an es mated answer even if you are not sure. Ques on: Given that length of the yellow car = 5.67m, what is the speed of the yellow car at 2.0s in m/s?

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Text Prompt

[Figure 115]

You are an expert analyst specializing in physics measurements. Based on the provided context and ques on, provide ONLY the numerical answer with units. No explana on or reasoning needed. Format your response as: [value] [unit] Example: 2.5 cm Be as accurate as possible with measurements and calcula ons. Please give me an es mated answer even if you are not sure. Ques on: Given that length of the yellow car = 5670m, what is the speed of the yellow car at 2.0s in m/s?

###### GPT-5.1 Answer

###### Ground Truth

Relative Accuracy

12 m/s 5.07

0

Text Prompt

Ques on: Given that length of the yellow car = 5.67m, what is the width of the yellow car in meters?

[Figure 116]

[Figure 117]

GPT-5.1 Thinking

###### GPT-5.1 Answer

###### Ground Truth

Relative Accuracy

1.8 m 2.43

0.7

[Figure 118]

[Figure 119]

Figure 8. Case 3: Video ablation reveals reliance on priors.

Case 4: Strong gravitational prior overrides counterfactual physics. The fourth case (Figure 9) involves a Blendersimulated basketball scene that visually resembles a realistic indoor court, but with counterfactual physics. The ball’s acceleration is time-varying and close to 1m/s2, rather than standard gravity. The model is asked for the ball’s acceleration at 0.5s and its speed at 1.5s, given the ball’s diameter as a prior. Here, ChatGPT-5.1 completely ignores both the video and the non-standard trajectory implied by the simulation. It directly outputs the canonical gravitational acceleration 9.8m/s2, and for speed simply multiplies g by time (i.e., 9.8 × 1.5 = 14.7m/s), leading to relative accuracy equals to 0 on both queries. No pixel measurements or scale computations appear in the Thinking trace. This illustrates how strong pre-trained physical priors (e.g., “objects fall with acceleration g”) can dominate the model’s behavior, even when they contradict the actual visual input and the provided prior.

[Figure 120]

GPT-5.1 Answer

###### Ground Truth

Relative Accuracy

9655 m/s 5070

0

Text Prompt

Ques on: Given that length of the yellow car = 5670m, what is the width of the yellow car in meters?

[Figure 121]

###### GPT-5.1 Thinking

Calcula ng yellow car width from length The yellow car‘s length is given as 5670m, which seems unusually large — maybe it’s to scale? Assuming a typical car‘s width-to-length ra o… Thought for 5s Done

[Figure 122]

###### GPT-5.1 Answer

###### Ground Truth

Relative Accuracy

2268 m 2430

0.9

Figure 7. Case 2: Counterfactual prior breaks faithfulness.

outcome-based metrics can judge an answer as “good,” while the underlying reasoning ignores the provided evidence.

Discussion. These four cases collectively sharpen our main quantitative findings.

- Case 3: Video ablation reveals reliance on priors. Case 3 (Figure 8) uses a video-ablation setting. The model receives only the text prompt (including “length of the yellow car

- • When everything works, ChatGPT-5.1 can execute an impressive, tool-augmented pipeline that does read pixel trajectories, apply the physical prior, and compute accurate kinematic quantities.
- • However, this behavior is fragile. As soon as the prior becomes counterfactual, the video is ignored, or the underlying physics departs from familiar regimes, the model quickly reverts to pre-trained world knowledge or rough heuristics, often ignoring the provided inputs.
- • High numerical accuracy does not guarantee input-faithful reasoning. Specifically, Case 2 demonstrates that a model can get the “right” answer for the wrong reasons, while Cases 3 & 4 show that it can stick to canonical physical constants even when the scene violates them.

= 5.67m”), without access to the video. When asked for the car’s speed at 2.0s, ChatGPT-5.1 produces an answer (12m/s) that is far from the ground truth, confirming that motion estimation is difficult without visual evidence. However, when asked for the car’s width (still without video), the model outputs a numerically reasonable value with relatively high accuracy (relative accuracy ≈ 0.7). Since no pixel information is available in this ablated setting, this behavior can only be explained by the model’s internal prior over typical car dimensions. Combined with Case 1 and 2, this suggests a pattern that even when video is available, much of the “size” inference can be driven by pre-trained world knowledge rather than by explicit pixel measurements.

These observations suggest that improving VLMs’ quan-

Video

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

Text Prompt

You are an expert analyst specializing in physics measurements. Based on the provided context and ques on, provide ONLY the numerical answer with units. No explana on or reasoning needed. Format your response as: [value] [unit] Example: 2.5 cm Be as accurate as possible with measurements and calcula ons. Please give me an es mated answer even if you are not sure. Ques on: Given that the diameter of the basketball = 0.282m, what is the accelera on of the basketball at  me 0.5s in m/s^2?

[Figure 130]

###### GPT-5.1 Thinking

(No Thinking Process Presented)

[Figure 131]

###### GPT-5.1 Answer

###### Ground Truth

Relative Accuracy

9.8 m/s 1.183

0

Text Prompt

Question: Given that the diameter of the basketball = 0.282m, what is the speed of the basketball at time 1.5s in m/s?

[Figure 132]

###### GPT-5.1 Thinking

(No Thinking Process Presented)

[Figure 133]

###### GPT-5.1 Answer

###### Ground Truth

Relative Accuracy

14.7 m/s 1.033

0

- Figure 9. Case 4: Strong gravitational prior overrides counterfactual physics.

titative physical reasoning will require not only better aggregate performance, but also mechanisms that encourage faithful use of visual evidence and explicit numerical priors, rather than letting powerful, but sometimes misleading, pretrained world knowledge dominate the inference process.

###### A.2. Metric Design Justification

In QUANTIPHY, we adopt Mean Relative Accuracy (MRA) as the primary evaluation metric for quantitative physical inference tasks with continuous outputs. Each prediction yˆ is compared to the ground truth y using its relative error |yˆ − y|/y, and awarded partial credit if the error falls below confidence thresholds θ ∈ {0.5,0.55,...,0.95}. The final MRA score is the average of binary accuracies across these thresholds

1 10 θ∈C

MRA =

|yˆ − y| |y|

< 1 − θ , (1)

where C = {0.5,0.55,...,0.95} is the set of confidence thresholds.

While one could alternatively compute continuous relative error (e.g., mean relative error or mean absolute percentage error), we prefer MRA for several practical and conceptual reasons.

Discrete but calibrated. MRA discretizes accuracy into a finite set of thresholds, offering interpretable feedback on how often the model is “close enough” under increasing demands. Rather than penalizing deviations proportionally (which can be dominated by outliers), MRA provides a graded scale of correctness, similar in spirit to the mAP@IoU8 metrics in object detection.

Robust to ambiguity and noise. Many video-based physical inferences involve semantic or visual uncertainty. For example, estimating a person’s height may vary depending on whether hair or shoes are included; estimating a cup’s diameter may depend on whether the inner or outer rim is used. MRA tolerates such ambiguity by granting full credit when answers fall within a reasonable margin.

Moreover, measurement noise is often unavoidable:

- • Temporal aliasing: limited frame rates restrict temporal resolution for computing velocities and accelerations;
- • Motion blur: fast-moving objects introduce visual uncertainty during measurement;
- • Imprecise priors: even real-world scale references (e.g., credit card dimensions) may not be perfectly visible or aligned.

MRA accommodates these natural imperfections more flexibly than a regression-style loss.

Supported by precedent. The MRA metric was introduced by Yang et al. [53] in VSI-Bench for evaluating numerical answers in visual–spatial reasoning tasks. Some follow-up work [9, 38] adopted the same evaluation to benchmark multimodal models in physical and spatial settings. These works motivate MRA as a stable and discriminative way to capture proximity between predicted and ground-truth values, especially when scale varies across examples. Our use of MRA continues this design choice, ensuring comparability while improving robustness.

In sum, MRA balances informativeness, robustness, and interpretability. It is well-suited for evaluating VLMs on physically grounded, numerically sensitive tasks like those in QUANTIPHY, where small deviations are acceptable, but large errors are unacceptable regardless of scale.

###### A.3. Model MRA Distribution

Figure 10 visualizes the MRA distributions for each VLM. Each subplot shows a density curve with mean (gray solid) and median (blue dashed) lines, along with summary statistics (mean, std, median, Q25, Q75) displayed to the right.

The top-performing models, including ChatGPT-5.1, Gemini-2.5 Pro, Gemini-2.5 Flash, Qwen3-VL-instruct-32B,

8mAP@IoU: Mean Average Precision (mAP) calculated at specific Intersection over Union (IoU) thresholds.

[Figure 134]

###### Figure 10. Distribution of MRA by model. One caveat to note is that the Avg. MRA in Table 1 reflects the mean MRA across inference task categories for each model (i.e., the average MRA of 2D-Static, 2D-Dynamic, 3D-Static, and 3D-Dynamic). In contrast, the mean in this distribution plot represents the average MRA at the individual-question level for each model.

and Grok-4.1-Fast-Reasoning, show notably higher densities around moderate to high MRA values above 0.5, with means and medians clustering around 0.4–0.6. Their smoother, more concentrated curves indicate both higher accuracy and more consistent performance across the evaluation set.

- A second tier of models, including InternVL-3.5-30B,

Qwen3-VL-instruct-8B, Molmo-7B, and ChatGPT-5, exhibits slightly lower means and medians, generally around 0.3–0.4. Their broader distributions with heavier tails indicate greater variability that these models produce some strong outputs but also more low-scoring cases. The alignment of medians and means suggests errors are not extremely skewed.

Mid-tier models such as Qwen-3-VL-instruct-2B, Phi-4Multimodal-Instruct, SmolVLM-Instruct, CogVLM2 Video, Claude-4.5 Sonnet, and VILA-7B show means of 0.2–0.3. Their distributions are heavily weighted toward low MRA values with thin right tails, indicating that while they occasionally achieve moderate scores, they rarely reach the performance levels of top-tier systems.

The weakest-performing models, for example, Phi3-Mini-128K-Instruct-3.8B, LLaVA-13B, Fuyu-8B, and MiniCPM-V 4.5-8B, show distributions highly concentrated near zero. With means below 0.15 and medians around 0.0, these models fail to produce meaningful MRA performance in most cases.

Notably, many distributions have substantial mass centered around zero. An MRA of zero indicates either the model output zero as an answer or failed to produce proper numerical output. We also observed that some model APIs are unstable and produce errors over time, potentially due to API server error, running environment, internet traffic, batch size variations, etc. We abstract away from these in this paper and simply treat these failed cases as observations with MRA equal to zero.

Overall, modern frontier proprietary models cluster around substantially higher and more consistent MRA values, mid-tier models show moderate capability with noticeable variability, and smaller or older models yield predominantly low scores. The density patterns and comparative statistics reveal a clear performance gap between state-of-the-art systems and lightweight or earlier-generation models.

- B. Dataset Construction Guidelines

###### B.1. General Principles

Our data collection and curation follow several general principles to ensure that QUANTIPHY is ethically sourced, physically well-defined, and suitable for quantitative evaluation.

Copyright and ethics. We carefully avoid copyright and

ethical issues throughout all stages of data collection and processing. All videos, 3D assets, and simulation resources are either open-source, licensed for research use, or explicitly verified to pose no known copyright conflicts before inclusion. When raw videos contain personally identifiable information (e.g., human faces, license plates), we apply blurring or masking to anonymize the content.

Video selection criteria. To make quantitative kinematic inference well-posed and to reduce confounding factors, we enforce the following constraints on all collected videos.

- • Static camera in world coordinates. The camera remains fixed in the world frame during each clip. This avoids entangling camera motion with object motion, which would otherwise introduce additional ambiguity and noise into the inference problem.
- • At least one rigid object undergoing translational motion. Each video contains at least one rigid object whose dominant motion is translation. This requirement ensures that we can formulate well-defined kinematic inference tasks. Non-rigid objects and purely rotational motions are left out of the current benchmark and deferred to future work.9
- • Planar motion for 2D tasks. For 2D instances, the target object and the reasoning target are constrained to move in a plane parallel to the image plane, i.e., the depth relative to the camera remains (approximately) constant over time. This assumption guarantees a consistent mapping between pixel displacement and world-space distance within each clip, making the 2D kinematic inference problem welldefined.

Video–text record schema. Each annotated instance in QUANTIPHY is represented as a structured video–text record. Table 3 shows a representative example.

Each record contains the following fields:

- 1. video id. A unique identifier for the underlying video.

- 2. video source. The data source from which the video was obtained (e.g., simulation, lab, or internet).

- 3. video type. A four-letter code encoding the configuration of the task. The four characters denote, in order: (i) the type of physical prior (Size, Velocity, or Acceleration), (ii) whether the reasoning task is 2D or 3D, (iii) whether there is a single (S) or multiple (M) moving objects, and (iv) the background type, that is, plain (X), simple (S), or complex (C). More details are included in subsection B.2.

9In this work, we adopt a relaxed definition of rigid objects: we consider an object “rigid” if its motion can be consistently approximated by a stable center of mass across frames. This includes some entities that may exhibit slight non-rigid deformation (e.g., a flying bird or a walking person), as long as their motion remains locally trackable and structurally coherent.

Property Example value

video id simulation 0032 video source simulation video type A3MC fps 30 inference type DD question What is the acceleration of the orange car at 1.0s in m/s2? ground truth prior gravity acc = 9.8 m/s2 depth info t=1s, distance ball camera = 13.80 m;

t=2s, distance ball camera = 13.80 m;

- t=1.5s, distance orange camera = 10.18 m;

- t=2s, distance orange camera = 10.40 m;

t=2.5s, distance green camera = 4.32 m;

- t=3s, distance green camera = 6.91 m

ground truth posterior 2.86

Table 3. Example of a single video–text record in QUANTIPHY.

- 4. fps. The frame rate of the video, used to convert frame indices into time and to compute velocities/accelerations consistently.
- 5. inference type. A two-letter code indicating whether the prior and the inference target are static or dynamic over time: S denotes a static quantity, and D denotes a time-dependent (dynamic) one. The first letter corresponds to the prior, and the second to the posterior.

- 6. question. The natural-language prompt presented to the VLM. We ensure that each question explicitly specifies the physical unit (e.g., m, cm/s, m/s2) and, for velocity or acceleration, clearly indicates whether the query concerns an instantaneous quantity at a given timestamp or an average quantity over an interval.
- 7. ground truth prior. The physical prior provided to the model, formatted as a positive numeric value with unit (e.g., gravity acc = 9.8 m/s2). We enforce consistent formatting to simplify parsing and downstream use.

- 8. depth info. Depth annotations used only for 3D reasoning tasks. This field contains depth values (in metric units) for the prior object and, when needed, for the inference target at one or more timestamps. Depth information is designed so that, in principle, the depth of the inference target can be recovered from the provided entries. The formatting mirrors that of ground truth prior.

- 9. ground truth posterior. The numeric groundtruth answer to the kinematic inference question, represented as a positive scalar without unit (the unit is part of the question text).

Balance and diversity. Finally, we design the dataset to be both balanced across task types and diverse in content.

• Fine-grained video types. Using the four-letter video -

type code, we partition all clips into 36 fine-grained categories. We ensure that each category contains at least four videos, so that every configuration is represented nontrivially.

- • Balanced core task categories. We strive to keep the four core inference task categories (2D-Static, 2D-Dynamic, 3D-Static, 3D-Dynamic) approximately balanced in terms of the number of videos and associated questions, enabling fair comparison across conditions.
- • Rich scenes and motion patterns. We cover a broad spectrum of spatial scales and motion types. Scenes range from astronomical (e.g., planetary motion), to everyday macroscopic settings (e.g., traffic, sports), to microscopic phenomena (e.g., cells and bacteria). Motion patterns include uniform motion, accelerated and decelerated linear motion, projectile motion, pendulum-like oscillations, and centripetal motion, among others. This diversity is crucial for probing whether VLMs’ quantitative reasoning generalizes beyond narrow, highly stylized scenarios.

###### B.2. Video Types

###### B.2.1. Video Categories Definition

As described in the section , we assign each video a fourcharacter code that encodes its physical prior, dimensionality, object setting, and background type.

First character (S / V / A). The first character specifies which physical prior the video is constructed to probe. For a designated object in the scene, we use

- • S = size
- • V = velocity
- • A = acceleration

Thus, the first character takes one of {S,V,A}, indicating whether the ground-truth physical quantity is size, velocity, or acceleration for that object, as illustrated in Figure 11.

Second character (2 / 3). The second character indicates whether the video is 2-Dimensional(2D) or 3Dimensional(3D).

- • 2D = planar video
- • 3D = volumetric video

More precisely, the second character is set to 2 when, at every frame, the distances from the moving object, the groundtruth reference object, and every object referenced in the inference question to the camera are exactly equal, so they occupy a single depth layer with no relative depth or parallax among them in the rendered image, and to 3 otherwise. Video examples are shown in Figure 12.

Third character (S / M). The third character distinguishes between “single-object” and “multiple-object” settings, relative to the queried object(s) rather than the sheer number of objects visible in the scene:

- • S = single-object
- • M = multiple-object

As shown in Figure 13, this label depends on how many objects the viewer needs to reason about. We use S (singleobject) when the reasoning process only involves one object. Other objects may appear, but they only serve as background or generic distractors. We use M (multiple-object) when answering any of the questions requires reasoning about two or more objects, for example by comparing their sizes or speeds, or by using one object as an explicit reference for another.

Fourth character (X / S / C). The fourth character’s examples are illustrated in the Figure 14. encodes the complexity of the background.

- • X = plain background, typically a single uniform RGB color with essentially no texture, clutter, or noise;
- • S = simple background, which may contain mild lighting or shading variations but remains visually uncluttered;
- • C = complex background, with rich textures, multiple visible objects, more intricate lighting, and substantial visual “noise”.

The boundary between “simple” and “complex” backgrounds is somewhat subjective, but during dataset construction we deliberately separated these categories and designed scenes so that their visual difference is as clear as possible.

Putting these components together, for example, a code such as A3MC indicates that the video (i) targets the acceleration prior (A), (ii) is rendered as a 3D video (3), (iii)

is labeled as a multiple-object setting (M) because at least one inference question asks about an object different from the one whose acceleration prior is defined, and (iv) uses a complex background.

###### B.2.2. Quantitative breakdown of video types.

To complement the qualitative description above, we now provide a quantitative summary of how clips are distributed across the four-character codes. The benchmark contains 569 videos in total, of which 328 are 2D and 241 are 3D, yielding an approximately 4:3 split between planar and volumetric setups (about 58% 2D and 42% 3D). This ensures that both 2D and 3D configurations are substantially represented rather than the dataset being dominated by a single family of scenes. Table 4 reports the corresponding counts for each individual code and for the 2D versus 3D groups. By construction, combining three physical priors (S / V / A), two dimensionalities (2 / 3), two object settings (S / M), and three background types (X / S / C) yields 36 distinct four-character codes, and all 36 appear in the dataset. For 2D videos, the codes are A2SX, A2SS, A2SC,

A2MX, A2MS, A2MC, S2SX, S2SS, S2SC, S2MX, S2MS, S2MC, V2SX, V2SS, V2SC, V2MX, V2MS, V2MC; for 3D videos, the codes are A3SX, A3SS, A3SC, A3MX, A3MS, A3MC, S3SX, S3SS, S3SC, S3MX, S3MS, S3MC, V3SX, V3SS, V3SC, V3MX, V3MS, V3MC. Each code is instantiated by at least 4 clips, so even the smallest categories have non-trivial support. The largest corpus (V2MC in our current dataset) contains 51 clips. Across all 36 codes, per-code counts range from 4 to 51 clips, with the majority lying between 5 and 35. This distribution avoids both extremely rare “one-off” configurations and a few overwhelmingly frequent ones. The precise per-code counts are given in Table 4. Table 4 also details how each video is obtained. The Blender column counts fully synthetic clips rendered directly in Blender from our own scenes, contributing 300 videos. The Internet column counts 72 clips sourced from existing online footage that we curate and annotate. The Captured column counts 112 clips that we record ourselves (for example, using handheld cameras or screen recordings). The Segmented column counts 85 clips created by segmenting foreground objects from source footage and compositing them into new backgrounds. For each four-character code, the Total column gives the number of clips that realize that configuration. Taken together, these statistics in Table 4 show that the dataset spans all intended configurations, with each category populated by multiple clips and supported by a mix of blender-rendered, internet-sourced, captured, and segmented videos.

[Figure 135]

- Figure 11. Examples of S/V/A scene.

[Figure 136]

- Figure 12. Examples of 2/3 scene.

#### C. Details of Data Collection

###### C.1. Blender Simulation

###### C.1.1. Blender Toolkits and Asset Sources

Asset sources and selection. We construct Blender scenes using 3D assets sourced from online repositories, primarily

BlenderKit and Sketchfab. We deliberately use these two libraries for complementary purposes: BlenderKit mainly provides complex themed environments that serve as background layouts, whereas Sketchfab mainly provides rigged and animated foreground objects whose motions can be reused with minimal manual keyframing.

[Figure 137]

- Figure 13. Examples of S/M scene.

[Figure 138]

- Figure 14. Examples of X/S/C scene.

BlenderKit for complex themed environments. BlenderKit is a community-driven asset library that is tightly integrated into Blender’s interface and offers a large collection of render-ready models, materials, High Dynamic Range Images(HDRIs), brushes, and complete scenes that can be searched and inserted directly from within

Blender. In our pipeline, we primarily rely on BlenderKit for complex “themed” environments, such as indoor rooms, streets, architectural spaces, and other richly cluttered layouts at both small and large spatial scales. These pre-built scenes typically include coherent lighting, materials, and background geometry (for example, furniture, buildings, and

2D/3D Video Type Blender Internet Captured Segmented Total 2D/3D Total

|2D|A2SX 0 0 0 11 11 A2SS 10 6 0 0 16 A2SC 10 5 0 0 15 A2MX 6 0 1 8 15 A2MS 14 0 1 0 15 A2MC 17 2 1 0 20 S2SX 0 0 0 10 10 S2SS 10 5 0 0 15 S2SC 7 8 0 0 15 S2MX 8 0 0 8 16 S2MS 11 3 0 0 14 S2MC 16 20 0 0 36 V2SX 2 0 0 9 11 V2SS 10 7 0 0 17 V2SC 10 8 0 0 18 V2MX 13 0 0 7 20 V2MS 13 0 0 0 13 V2MC 43 8 0 0 51<br><br>|
|---|---|
| |328|

|3D<br><br>|A3SX 2 0 6 1 9 A3SS 2 0 9 0 11 A3SC 3 0 7 0 10 A3MX 1 0 4 2 7 A3MS 3 0 4 0 7 A3MC 22 0 4 0 26 S3SX 1 0 8 2 11 S3SS 2 0 8 0 10 S3SC 2 0 8 0 10 S3MX 0 0 9 13 22 S3MS 1 0 10 0 11 S3MC 24 0 10 0 34 V3SX 3 0 2 0 5 V3SS 2 0 2 0 4 V3SC 3 0 2 0 5 V3MX 2 0 5 14 21 V3MS 3 0 5 0 8 V3MC 24 0 6 0 30|
|---|---|
| |241|

Total 300 72 112 85 569 569

Table 4. Statistics of videos.

vegetation), which allows us to quickly instantiate diverse indoor and outdoor environments without having to model every object or compose every layout from scratch. This substantially reduces the authoring effort for environment design while still giving us control over camera placement, object insertion, and motion trajectories.

Sketchfab for rigged and animated foreground objects.

Sketchfab is a large community platform for hosting and distributing 3D content, including many rigged and animated models across categories such as animals, vehicles, and articulated characters. In our work, we mainly use Sketchfab to obtain foreground objects that already come with a skeleton rig and a small set of reusable animation clips. Typical examples include wing-flapping cycles for flying birds (such as eagles spreading and flapping their wings) and swimming

cycles for fish with realistic body undulation. Instead of manually keyframing these motions, we can directly reuse or lightly retarget the provided animations in our scenes. This makes it much easier to populate environments with moving agents whose motion is visually plausible, while significantly reducing the time spent on low-level animation authoring.

Licensing and reuse. For both BlenderKit and Sketchfab, we restrict ourselves to assets whose licenses explicitly allow reuse and modification in works. These licensing choices ensure that all assets in our dataset are used in a copyrightcompliant way and that future researchers can reconstruct our pipeline using the same publicly available resources.

###### C.1.2. Two Motion Simulation Types

The Blender-generated portion of our dataset contains two complementary categories of motion, reflecting the major paradigms of movement in computer graphics and physicsbased animation.

- (1) Keyframed Motion. Examples of this category are shown in Figure 16. This category includes humans, animals, and other objects whose motion is defined using rigged skeletons or keyframed animation curves shown in Figure 17. Because these trajectories are authored manually rather than produced through physical simulation, they are visually plausible but not physically constrained.

A key implication is that the resulting motion does not necessarily obey real-world physical laws. For example, in the lunar-walking scene (Figure 22), the astronaut’s pushoff, airtime, and landing motion are shaped by artist-edited animation curves. Although the motion is loosely inspired by reduced lunar gravity, we do not compute the animation using the Moon’s gravitational constant nor derive the trajectory from force-based simulation. These sequences should therefore be interpreted as perceptually reasonable approximations of motion rather than physically calibrated ground truth.

- (2) Physics-Driven / Force-Based Motion. Examples of this category are shown in Figure 15. This category consists of rigid bodies (shown in Figure 18) moving directly under Newtonian dynamics. Their trajectories are generated by applying explicit forces (e.g., gravity, impulses) inside a physics engine or by specifying analytical kinematic profiles (e.g., constant velocity, uniform acceleration, projectile motion). These clips yield clean and physically interpretable motion, allowing us to provide exact ground-truth displacement, velocity, and acceleration.

For all videos in this category, we explicitly state the forces or kinematic parameters involved, and these values constitute part of the known prior information for each question.

Clarifying Prior Knowledge vs. Visual Assumptions. In an era where videos may originate from real capture, simulation software, procedural animation, or generative models, visual motion alone does not guarantee adherence to physical laws. Accordingly, VLMs should reason strictly based on the prior conditions we provide, rather than relying on pretrained assumptions about how objects “should” move.

When prior information explicitly specifies a force or acceleration (e.g., “the object accelerates at 2.5m/s2”), that value serves as authoritative ground truth. When such information is not given, the model should not infer physical constants, such as Earth’s gravity, only from the visual appearance of the motion. Because our dataset includes diverse simulated and animated sequences, correct reasoning requires using only the provided priors, not assumed real-world physics.

###### C.1.3. Blender Videos Construction

Quantitative Overview. In the Blender subset, we start from approximately 125 distinct base 3D models and 81 base scenes. From these bases, we generate 312 unique Blender video clips. Beyond simply increasing the size of the dataset, Blender enables us to construct diverse scenes and controlled trajectories of objects in simplified environments. Because these scenes are generated procedurally, we can precisely specify object sizes, positions, and kinematics via scripts and directly compute accurate ground-truth physical quantities for each clip. As a result, the Blender subset is not only large, but also covers a broad spectrum of scene scales and motions under tight experimental control.

Scene Construction. We construct a diverse collection of synthetic scenes to broaden the range of physical situations represented in the benchmark. Specifically, we design both everyday indoor spaces (e.g., rooms with furniture and household objects, see Figure 19) and outdoor spaces (e.g., natural environments with varied terrain, vegetation, and water, see Figure 20). These scenes differ in layout, depth structure, and illumination, so that models must handle physical reasoning under heterogeneous visual conditions rather than overfitting to a single canonical setting.

Beyond such everyday environments, we also include scenes that are difficult or impossible to capture, measure, or systematically manipulate in the real world. Examples include microscopic settings (e.g., red blood cells moving through a vessel, see Figure 21), extraterrestrial scenarios (e.g., astronaut motion on the Moon, see Figure 22), and more.

Starting from base assets and scenes, we explicitly control geometry, lighting, and background complexity to generate families of videos that fall into different categories in our taxonomy. For instance, in the red-blood-cell scene (Figure 23), we derive a 3MX-style variant by removing the

[Figure 139]

- Figure 15. Physics-driven example. A bowling ball collides with pins under Newtonian simulation. The motion and resulting trajectories arise directly from rigid-body dynamics and elastic collisions, making this clip representative of our force-based motion category.

[Figure 140]

- Figure 16. Keyframed example. The floating swim ring follows a manually authored animation curve rather than buoyancy, drag, or wind forces. Its trajectory is visually plausible but not physically derived, representing our keyframed motion category.

vessel wall, discarding all but two target cells, and replacing the background with a uniform RGB field. Using the same procedure, as shown in the Figure 24 we construct simplebackground versions of more complex scenes by simplifying clutter while keeping coarse structural cues (e.g., horizon lines, major surfaces, object structures) and realistic lighting.

It is important to note that, although we present several examples that appear to reuse the same underlying scene. In practice, not all plain-background videos and simplebackground videos are reused across conditions. In addition, we also include videos that are uniquely constructed to appear only in the plain-background condition or only in the simple-background condition. This design choice is intended to increase the diversity of the video data. Reusing identical base scenes improves the efficiency of dataset construction,

but it may also introduce potential biases or unwanted correlations when evaluating VLMs, so we deliberately balance such reuse with the creation of novel, non-paired scenes.

For the objects in each scene, we specify target object sizes, velocities, and accelerations using real-world statistics gathered from online references (e.g., typical dimensions and speeds of vehicles, animals, and more). Whenever possible, moving objects are modeled at real-world scale; in rare cases where real-world-scale objects yield motion trajectories that are barely discernible in the rendered videos, we apply uniform scaling to increase perceptual visibility while preserving the underlying physical relationships.

An example is the “ice cube falling into a cup” scene illustrated in the Figure 25. If we model the cup and cube at their real-world dimensions, the cube traverses the cam-

[Figure 141]

- Figure 17. Keyframed animation-curve example. The animation curve controlling the swim ring in Figure 16. The ring moves forward along a manually authored trajectory, while small randomized perturbations in translation and rotation are added to imitate the visual appearance of floating motion.

[Figure 142]

- Figure 18. Physics-driven rigid-body example. Rigid-body simulation of objects such as the bowling pins reacting to the impact of an incoming ball in Figure 15. The pins’ motion, tipping, and scattering are governed entirely by Newtonian dynamics and collision responses within the physics engine.

the priors specified in the prompt, or whether they instead rely predominantly on generic pre-trained knowledge and commonsense expectations about how such objects “should” behave.

Object Motion Design and Implementation. As discussed in subsubsection C.1.2, our motion design follows two main paradigms: keyframed motion and physics-driven motion. The construction of physics-driven scenes is illustrated in Figure 15. For the keyframed cases, in addition to manual editing in the Graph Editor, we also use scripts to automate the animation process.

We distinguish two main classes of scripted motion: (i) analytic one-dimensional motion along a single axis, and (ii) curve-following motion along human-designed paths.

For analytic 1D motion, we explicitly encode standard kinematic equations and bake the resulting trajectories as keyframes. Given an object with initial world-space position x0, a target duration T, and a desired acceleration a along the −Y axis, our script iterates over frames f and computes the physical time

f − START FRAME FPS

t =

###### .

The corresponding displacement is computed using the constant-acceleration formula

###### s(t) = 21at2.

era frustum in essentially a single frame, making its falling trajectory almost invisible to observers and thus providing little signal for quantitative evaluation. To address this, we uniformly enlarge both the cup and the ice cube by a factor of 10. After scaling, the cube remains visible from frame 11 to 22, and the rendered video reveals a clear multi-frame trajectory. This controlled rescaling not only makes the motion measurable but also allows us to probe whether VLMs recover physical quantities from the observed kinematics given

- 1 disp_y = 0.5 * ACCEL * (t ** 2)
- 2 new_y = start_loc.y - disp_y
- 3 obj.location = Vector((start_loc.x, new_y,

→ start_loc.z))

- 4 obj.keyframe_insert(data_path="location",

→ index=-1, frame=f)

This directly realizes s(t) = 21at2 in world coordinates. For constant-velocity motion, we instead use the linear rela-

[Figure 143]

###### Figure 19. Examples of indoor scene.

[Figure 144]

###### Figure 20. Examples of outdoor scene.

[Figure 145]

###### Figure 21. Examples of microscopic scene.

[Figure 146]

###### Figure 22. Examples of extraterrestrial scene.

[Figure 147]

- Figure 23. Examples of building X background scene.

[Figure 148]

- Figure 24. Examples of building S background scene.

tion:

s(t) = vt, implemented via

- 1 disp_y = VEL * t
- 2 new_y = start_loc.y - disp_y

while keeping the same frame loop and keyframe insertion logic. After all keyframes are written, we programmatically set each F-curve’s interpolation mode to LINEAR, overriding Blender’s default B´ezier interpolation to ensure that the frame-to-frame displacement matches the analytically specified velocity or acceleration profile rather than being inadvertently smoothed or distorted.

For more complex motions along curved paths, we first author an approximate trajectory in Blender (e.g., by manually keyframing a car following a road or a particle moving along a spiral), and then reparameterize this path in Python to obtain physically interpretable kinematics. Concretely, we sample the object’s world-space position at each frame of the original animation:

- 1 positions = []
- 2 for f in range(frame_start, frame_end + 1):
- 3 scene.frame_set(f)
- 4 pos = obj.matrix_world.translation.copy()
- 5 positions.append(pos)

We then compute the cumulative arc length along this polyline.

- 1 distances = [0.0]
- 2 for i in range(1, len(positions)):
- 3 d = (positions[i] - positions[i-1]).length
- 4 distances.append(distances[-1] + d)
- 5 total_length = distances[-1]
- 6 duration = (frame_end - frame_start) / fps

We then derive a desired kinematic profile in terms of traveled distance s(t) along the curve. For constant-speed motion, we set v = L/T with L = total length and use

s(t) = vt, implemented as

1 target_dist = speed * t

which corresponds to s(t)=vt. For constant-acceleration motion along the same curve, we instead use the quadratic form

s(t) = v0t + 21at2,

with user-specified v0 and a. In both cases, for each frame we find the segment [i − 1,i] such that

distances[i − 1] ≤ s(t) ≤ distances[i], and linearly interpolate between the sampled positions:

- 1 ratio = (target_dist - distances[i-1]) /

→ (distances[i] - distances[i-1])

- 2 new_loc = positions[i-1].lerp(positions[i],

→ ratio)

[Figure 149]

###### Figure 25. Examples of scaled scene.

- 3 obj.location = new_loc
- 4 obj.keyframe_insert(data_path="location",

→ frame=f)

Finally, as in the analytic case, all resulting keyframes are set to linear interpolation. This arc-length reparameterization scheme lets us reuse authored trajectories while imposing well-defined kinematic profiles (constant velocity, constant acceleration, or other time–distance schedules) in physical units. Together with the untouched native animations, these scripted motions give our benchmark a wide spectrum of motion patterns, from simple 1D acceleration to complex, irregular dynamics, under tight quantitative control.

Rendering. All Blender-generated videos in QUANTIPHY are rendered with the Cycles and EEVEE path-tracing engine using a physically based workflow. Videos use multiple spatial resolutions, including 1920×1080 (16:9), 1080×1080

(1:1), 480×960 (vertical), as well as several additional intermediate sizes, so that models see both landscape, square, and portrait-style content. The temporal sampling is likewise heterogeneous: frame rates in the benchmark include 24, 25, 30, 33, 60, and 120fps. Frames are exported via FFmpeg as MP4 files with H.264 compression and a constant frame rate, without any interpolation or stabilization, so each frame aligns exactly with the simulated timeline and ground-truth annotations.

Camera parameters are also varied to increase visual diversity. Intrinsics cover a range of focal lengths from wide to normal and telephoto views, and extrinsics place the camera at different heights, offsets, and azimuth/elevation angles around the scene, producing both frontal and oblique perspectives on object motion. Lighting setups combine different environment maps and local light sources, and objects are assigned a broad set of physically based materials (e.g., diffuse,

glossy, metallic, translucent), leading to diverse shading, reflection, and contrast conditions. Together, these choices give QUANTIPHY wide coverage over resolutions, frame rates, viewpoints, and appearance statistics, while the underlying physical trajectories and numeric ground truth remain precisely controlled.

###### C.2. Lab Capturing

To further complement the diversity of QUANTIPHY with real-world data, a major part of our dataset comes from a customized motion capture system in a lab environment. Due to the nature of real-world data, lab captured videos contribute only to the category of 3D.

MoCap setup. We use four Orbbec Femto Mega cameras to construct our customized motion capture system. The specifications of the cameras are shown in Table 5.

We designed two different camera setups and object arrangements in the MoCap system. The first setup covers a smaller spatial range, with the main camera placed close to the objects. In this setup, we capture motions of small objects such as a tennis ball, book, and ping pong ball moving on a desk. The second setup covers a larger range and is used to capture larger-scale motions such as basketball bouncing and accelerated motion of a trash bin. For both setups, we initialize the camera extrinsics via multi-view calibration with a checkerboard.

Collection workflow. For efficient video capture, we developed a tool with a user-friendly interface that supports synchronized recording across all four cameras. We collect videos based on a list of predefined questions. For each question, we set up the scene and objects without moving the cameras and then perform specific object motions according to the question. Three people were involved in the lab data collection workflow: one person was responsible for operating the tool (starting and pausing capture), and two people were responsible for setting up the scenes and executing the motions.

Unlike Blender simulation, capturing real-world data makes it difficult to directly manipulate specific scene parameters. We therefore manually configured the scenes to meet target conditions, such as the slope angle or the frequency of pendulum motion, using careful measurements within a set tolerance. For motions that require a controlled initial velocity, we used a motor running at s constant speed to pull the object and provide the desired initial velocity.

Post-processing. After capturing the raw videos, we obtain .bag files for each camera that contain the binary recordings. We follow the official camera SDK to decode the raw data. From this decoding, we obtain: (1) intrinsics for each camera, (2) timestamped RGB streams for each camera, (3) timestamped depth streams for each camera, and (4) relative

extrinsics from the depth cameras to the RGB cameras. We then apply coordinate transformations to reproject the depth data into the image coordinates of the RGB videos, which allows pixel-wise depth values to be read consistently.

With aligned image coordinates, it becomes possible to read out specific depth values for target objects (with respect to the main camera). A straightforward approach is to use segmentation masks for the target object and average the depth values over the masked pixels. However, this automated approach did not work well in practice because the depth information from the camera is often incomplete and ambiguous, especially at pixels with high motion, due to hardware limitations. As a result, we could not reliably automate depth extraction for the target objects using only segmentation masks.

To ensure high-quality depth measurements, we developed a UI-based tool that overlays the transformed depth map on top of the RGB frames. The tool allows users to click on the frame to read out exact depth values from the main camera at the selected pixels. An illustration of this UI-based tool is shown in Figure 26.

List of objects used. Table 6 lists all physical objects used in our controlled lab videos, together with their measured dimensions. These objects serve either as priors (with known size) or as inference targets in our kinematic tasks. All measurements are taken with a ruler or caliper in metric units before filming.

###### C.3. Internet Scraping

Our “internet” split in QUANTIPHY consists of real-world videos captured by commodity cameras, and is constructed from two sources that we treat under a unified category: (i) open-source online video platforms, contributing 42 clips; and (ii) videos recorded by the authors using smartphone rear cameras in everyday indoor and outdoor environments, contributing 30 clips. All of these videos are direct camera recordings of natural scenes, and thus closely reflect the statistics and imperfections of the physical world.

Why only 2D inference from internet data. Unlike our simulation and controlled-lab settings, internet videos do not come with calibrated depth, camera intrinsics, or precise object geometry. In particular, reliable metric depth is almost impossible to obtain for arbitrary internet footage. For this reason, we restrict internet videos to 2D kinematic inference tasks. For each selected clip, we manually construct a pixel ruler, measure pixel-level size/position trajectories of the objects of interest, and then use an approximate scale factor—derived from obvious real-world references in the scene (e.g., gravity g = 9.8 m/s2, the length of a credit card, lane width on a road, or the speed of an airport conveyor belt)—to convert these pixel measurements into world-space

Depth Technology Time of Flight Wavelength 850nm Depth Range *0.25–5.46m (depending on depth mode) Depth Resolution/FPS Up to 1024 × 1024@15 fps (WFOV), 640 × 576@30 fps (NFOV) Depth FOV H 120◦ V 120◦ (WFOV), H 75◦ V 65◦ (NFOV) RGB Resolution/FPS Up to 3840 × 2160@25 fps RGB FOV H 80◦ V 51◦ Processing NVIDIA Jetson Nano IMU Supported

Table 5. Specifications of the cameras used in our customized motion capture system.

[Figure 150]

Figure 26. The users’ interface of the tool we have developed for obtaining depth value.

kinematic quantities. While this procedure yields reasonably accurate ground truth, it is inherently less precise than the annotation pipelines used for our simulation and lab data. Consequently, we intentionally keep the proportion of internet data moderate in the overall benchmark.

Videos from open-source platforms. We choose opensource video platforms primarily because they avoid copyright issues, offer relatively high image quality, and provide diverse content. However, videos that satisfy the three screening criteria in subsection B.1 (static camera, at least one rigid object undergoing translational motion, and planar motion for 2D tasks) are relatively rare. Beyond these core constraints, we additionally require that each candidate clip contain at least one visually obvious physical prior that can be reasonably assumed or measured (e.g., gravity,

a credit-card-sized object, standard lane width, or known conveyor-belt speed). These additional constraints further narrow the pool of usable videos. There is currently no offthe-shelf automatic pipeline for this selection process, so all internet clips are hand-picked by project members, who visually inspect candidates for compliance with our physical and annotation requirements. Representative examples are shown in Figure 27.

Author-recorded videos. Because suitable clips on opensource platforms are scarce, we complement them with 30 videos recorded by the authors using smartphone rear cameras in a variety of everyday scenes, including parking lots, road traffic, bedrooms, and indoor/outdoor sports venues. During recording, we enforce a fixed camera viewpoint and ensure that at least one rigid object exhibits predominantly

Objects Dimensions

2 green tennis balls d = 6.7 cm 2 pink tennis balls d = 6.7 cm 2 white ping pong balls d = 4 cm 1 purple yoga ball d = 52.2 cm 1 soccer ball d = 17.5 cm 1 basketball d = 23.2 cm 1 red plastic ball d = 7 cm 1 orange plastic ball d = 5.7 cm 1 small whiteborad (slope) 30.8 cm x 23.2 cm x 0.5 cm 1 large whiteborad 81 cm x 60 cm x 1 cm 1 trashbin 34.5 cm x 89.1 cm 59.9 cm 1 tape 10.1 cm x 10.1 cm x 4.8 cm 1 white food box 19.5 cm x 6.0 cm x 7.2 cm stuffed toy 1 14.5 cm x 11 cm x 8 cm stuffed toy 2 29.5 cm x 15.0 cm x 10.0 cm stuffed toy 3 8.1 cm x 6.4 cm x 19.6 cm 1 toy cookie 8.0 cm x 8.0 cm x 1.8 cm 1 cosmetic jar 7.2 cm x 7.2 cm x 6.0 cm 1 glass jar 6.8 cm x 6.8 cm x 8.5 cm 1 white cup 8.6 cm x 8.6 cm x 14.5 cm 1 pink water bottle 6.7 cm x 6.7 cm x 19.5 cm 1 marker 13.5 cm x 0.8 cm x 0.8 cm 1 black pen 14.5 cm x 1. 2 cm x 1.2 cm 1 green notebook 18.6 cm x 8.8 cm x 1.5 cm 1 white-covered book 12.5 cm x 19.5 cm x 2.2 cm 1 pencil case 6.7 cm x 7.2 cm x 21 cm 1 credit card 85.6 mm x 53.98 mm 1 deck of poker card 9.8 cm x 6.3 cm x 1.8 cm

Table 6. List of physical objects and their measured dimensions used in our lab-captured videos.

translational motion, so that the criteria in subsection B.1 are satisfied. In many such settings, precise physical quantities (e.g., vehicle speed, basket height) cannot be directly measured. We therefore annotate these videos using the same pixel-ruler and approximate-scale procedure as for online platform videos, again restricting them to 2D inference tasks. Example frames from these author-recorded clips are shown in Figure 28.

Privacy and anonymization. For all internet videos, whether sourced from open platforms or recorded by the authors, we manually inspect frames for sensitive personal information. Whenever faces, license plates, or other identifying details appear, we apply blurring or masking before including the clip in the dataset. This ensures that our internet data respect both copyright and privacy constraints while still providing realistic real-world scenarios for quantitative kinematic inference.

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

- Figure 27. Examples of videos from open-source platforms.

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

- Figure 28. Examples of self-recorded videos with identity removed.

###### C.4. Segmented Data

In QUANTIPHY, we aim to benchmark the capabilities of VLMs on videos with diverse background types. Video backgrounds often contain contextual information that may either aid or hinder the inference of a target object’s physical properties. To comprehensively study this impact, we design experiments that isolate the target object by completely removing the background. Specifically, we compare the original videos against processed versions where the background has been completely denoised and removed using a segmentation model.

State-of-the-art semantic segmentation models, such as SAM 2 [32], have demonstrated robust capabilities in tracking and segmenting arbitrary targets across various tasks. In

[Figure 159]

Figure 29. The users’ interface of the segmentation tool we have developed.

our experiments, we employ SAM 2 as the backbone segmenter, utilizing a hybrid prompting strategy. To automate the pipeline, we leverage Grounding DINO 1.5 [34] to localize target objects using textual descriptions derived from our video-question pairs. The bounding boxes generated by Grounding DINO serve as box prompts for SAM 2, enabling automatic segmentation.

For complex scenes with multiple objects where the automated pipeline may fail, we incorporate manual intervention by providing point-based prompts to SAM 2. To streamline this workflow, we developed a custom UI-based tool that allows multiple annotators to label objects efficiently, thereby scaling the segmentation process. Figure 29 illustrates the user interface of this tool, and Figure 30, Figure 31, Figure 32 provides exemplary frames of the segmented videos for reference.

###### C.5. Quality Control

The data collection process for QUANTIPHY is highly diverse and complex. Since the data originates from heterogeneous sources with varying characteristics, establishing a universal automated protocol for quality assurance is challenging. To address this, we incorporated an additional manual review stage for all candidate data.

To maximize data quality, we manually excluded videos exhibiting excessive motion blur, severe occlusion of the target object, or objects that are difficult to model. Furthermore, we removed videos containing identifiable human subjects to ensure ethical compliance.

Following this review process, approximately 3% of the Blender data and 30% of the lab data were discarded, while for the videos scraped from the internet only 72 clips were retained.

###### C.6. Ethical Considerations

In compiling the QUANTIPHY dataset, comprising both simulated Blender environments and web-sourced videos, we strictly adhere to all relevant copyright and licensing regulations. For specific data assets requiring attribution, we provide full acknowledgments in the supplementary materials.

We implemented strict privacy measures throughout the data collection and annotation phases to ensure that no personally identifiable information (PII) is retained in the dataset. Relevant Institutional Review Board (IRB) documentation is available upon request. Furthermore, QUANTIPHY complies with ethical guidelines regarding content

safety; we have rigorously screened the data to exclude biased or harmful content while prioritizing diversity to foster fairness and inclusivity.

#### D. Details of Data Annotation

###### D.1. Blender Simulation

We annotate five basic types of physical quantities: size, displacement over a given time interval, velocity, acceleration, and depth and distance.

###### D.1.1. Size

Blender size annotations are extracted directly from Blender’s internal measurement readout to guarantee numerical accuracy. For most objects, size is obtained from the object’s axis-aligned bounding box dimensions in world space. In other special cases, the dimensions of objects may change from frame to frame.

For example, as shown in the Figure 33, articulated humans height changes while they move, thus we explicitly measure three configurations within each clip:

- 1. Rest standing height: the height in a rest “T-pose” frame, where the character stands upright and fully extended.
- 2. Minimum apparent height during walking: the smallest height observed over the walking segment of the clip.
- 3. Maximum apparent height during walking: the largest height observed over the same segment.

All heights are obtained by selecting the corresponding frames in Blender and measuring the vertical extent in world coordinates. In practice, these three measurements are usually very close (often differing by only a few centimeters), but we still record all of them to keep the annotation protocol precise and consistent.

Similarly, for other rigged objects (such as birds and insects), we track their width frame by frame over the flight segment of the clip and only record the smallest and largest width observed. See details in Figure 34. This is because, unlike the human case, these assets may not admit a clear, static “rest standing” pose in which the animal is fully extended. As a result, we do not define a separate rest measurement for flying objects.

Correspondingly, the inference questions and prior ground truth are phrased with the same level of precision. For human figures, we explicitly distinguish between the standing height at rest and the minimum and maximum height observed while walking. For flying animals, we clearly refer to the minimum or maximum width attained during flight.

###### D.1.2. Displacement or Path.

All displacement- or path-related labels are computed from object trajectories in Blender world space using internal custom Python scripts.

Time–frame alignment across scenes. In Blender, each scene specifies a render range with scene.frame start and scene.frame end, and scene.frame start is not required to be 0. We define time t = 0 to occur at the render start frame, regardless of its numeric index. More precisely, if a scene has frame rate f frames per second and render start frame fstart, then the first frame of the exported video (timestamp t = 0 s) is Blender frame fstart, the frame at time t = 1/f s is frame fstart + 1, and so on.

When a question specifies a time interval [tstart,tend] in seconds, we convert it to Blender frame indices by istart = round fstart+f tstart , iend = round fstart+f tend .

This convention works uniformly for all scenes, including those whose render ranges begin at non-zero frame numbers (e.g., frame start = 20, 40, 60, ...). In code, the conversion has the following form:

- 1 scene = bpy.context.scene
- 2 fps = scene.render.fps
- 3 f_start = scene.frame_start

→ # may be 0, 20, 40, ...

- 4
- 5 def time_to_frame(t_sec: float) -> int:
- 6 return round(f_start + t_sec * fps)

When an interval is specified directly in frames, we simply take (istart,iend) as given and interpret the corresponding times as

iend − fstart f

istart − fstart f

, tend =

###### ,

tstart =

so that t = 0 always aligns with the first frame of the rendered video, irrespective of the absolute Blender frame index.

At any frame i, the scripts query the world-space position of the target object (e.g., the animated rigid body) via

- 1 scene.frame_set(i)
- 2 loc = obj.matrix_world.translation.copy()

→ # (x, y, z) in world space

yielding a vector p(i) = (x(i),y(i),z(i)) ∈ R3.

Displacement annotations. For displacement labels, we consider a time interval [tstart,tend] or frame interval [istart,iend]. The scripts record the world-space locations

pstart = p(tstart) = p(istart), pend = p(tend) = p(iend), and form the displacement vector

∆p = pend − pstart = (∆x,∆y,∆z).

From this vector we derive two scalar quantities.

[Figure 160]

- Figure 30. Examples of segmented blender data. After segmentation, we can replace the background image freely.

[Figure 161]

- Figure 31. Examples of segmented lab data. After segmentation, we can replace the background image freely.

[Figure 162]

- Figure 32. Examples of segmented internet data. After segmentation, we can replace the background image freely.

- 1. The full 3D displacement

D3D = ∥∆p∥2 = ∆x2 + ∆y2 + ∆z2,

- 2. The planar displacement in a 2D plane, obtained by projecting the motion onto that plane. For any chosen 2D coordinate plane with axes (u,v), we define the planar displacement as

D2D = (∆u)2 + (∆v)2.

In our benchmark we primarily use the three canonical planes: the horizontal ground plane

DXY = (∆x)2 + (∆y)2, and the two vertical planes DXZ = (∆x)2 + (∆z)2, DYZ = (∆y)2 + (∆z)2.

A compact version of the core computation is:

[Figure 163]

###### Figure 33. Examples of human height measuring.

[Figure 164]

###### Figure 34. Examples of flying animal measuring.

- 1 def displacement_world(obj, frame_start,

→ frame_end, scene):

- 2 # sample world-space positions at start and

→ end

- 3 loc_start = get_world_location_at_frame(obj,

→ frame_start, scene)

- 4 loc_end = get_world_location_at_frame(obj,

→ frame_end, scene)

- 5
- 6 # displacement vector
- 7 disp_vec = loc_end - loc_start
- 8 dx, dy, dz = disp_vec.x, disp_vec.y,

→ disp_vec.z

- 9
- 10 # 3D displacement and horizontal (XY)

→ displacement

- 11 D_3D = disp_vec.length #

→ sqrt(dx*dx + dy*dy + dz*dz)

- 12 D_XY = (dx**2 + dy**2) ** 0.5 #

→ sqrt(dx*dx + dy*dy)

- 13
- 14 return {
- 15 "frame_start": frame_start,
- 16 "frame_end": frame_end,
- 17 "loc_start": loc_start,
- 18 "loc_end": loc_end,
- 19 "dx": dx, "dy": dy, "dz": dz,
- 20 "D_3D": D_3D,
- 21 "D_XY": D_XY,
- 22 }

For each annotated interval, we store:

- • the frame indices (istart,iend);
- • the corresponding times (tstart,tend), computed relative to the render start frame as above;
- • the start and end world-space coordinates pstart,pend;
- • the displacement components (∆x,∆y,∆z);
- • the two scalar values D3D and DXY in Blender units.

- 13 disp = loc - prev_loc

→ # p_i - p_{i-1}

- 14 vel_vec = disp / dt

→ # v_i

- 15 speed = vel_vec.length

→ # s_i

- 16
- 17 write_per_frame_entry(frame, loc, vel_vec,

→ speed)

- 18 prev_loc = loc
- 19 prev_frame = frame

In the question text, we explicitly state whether “displacement” refers to the full 3D displacement D3D or to a planar displacement D2D in a specified plane (e.g., horizontal DXY or vertical DXZ / DYZ). The posterior ground-truth answer for that question is taken from the corresponding stored value.

For each frame i in the chosen range, this loop automatically prints one line of kinematic data into the text block, containing:

###### D.1.3. Velocity and Acceleration.

- • the frame index i and its timestamp ti (relative to the render start frame);
- • the world-space position pi;
- • the full 3D velocity vector vi;
- • the scalar speed si.

Velocity measurement and uniform-speed diagnostics. Using the time–frame convention introduced above, let pi ∈ R3 denote the world-space origin of the target object at frame i, obtained by querying obj.matrix world.translation after calling scene.frame set(i). For two consecutive frames i − 1 and i with

Any question that refers to “the speed at t seconds” or “the velocity at frame i” is answered by mapping the queried time to a frame index using the same time–frame conversion and then reading off the corresponding per-frame entry from this table, without any manual measurement or additional approximation.

timestamps ti−1 and ti, we define the temporal spacing

∆ti = ti − ti−1,

and the instantaneous 3D velocity and scalar speed at frame i as

pi − pi−1 ∆ti

In addition to per-frame velocities, we use the sequence of speeds {si} to characterize simple motion regimes at the clip level. Let Iv be the index set of frames for which a speed value is defined (all frames except the very first one), and let Nv = |Iv|. We compute the mean speed

, si = ∥vi∥2,

vi =

where vi = (vx,i,vy,i,vz,i) is the full 3D velocity vector in world coordinates and

si = ∥vi∥2 = vx,i2 + vy,i2 + vz,i2

1 Nv i∈I

s¯ =

###### si

is its Euclidean norm. All operations are performed in Blender world space and use the true frame rate, so that positions pi, velocities vi, and speeds si are expressed in a self-consistent system of physical units determined by the underlying scene scale.

v

and the maximum relative deviation from this mean,

 

0, if s¯ = 0, max

δspeed =

|si − s¯| s¯



, otherwise.

The Blender analysis script computes these quantities frame by frame and writes them into a text block in Blender’s Text Editor:

i∈Iv

Given a user-specified tolerance τv (parameter UNIFORM SPEED TOLERANCE, e.g., τv = 0.01), we classify the clip as:

- 1 prev_loc = None
- 2 prev_frame = None
- 3
- 4 for frame in range(frame_start, frame_end + 1):
- 5 scene.frame_set(frame)
- 6 loc = obj.matrix_world.translation.copy() #

→ p_i

- 7
- 8 if prev_loc is None:
- 9 vel_vec = Vector((0.0, 0.0, 0.0))
- 10 speed = 0.0
- 11 else:
- 12 dt = (frame - prev_frame) / fps

- • no effective motion if s¯ ≈ 0 (numerically s¯ = 0, indicating that the object is essentially stationary);
- • approximately uniform speed if s¯ > 0 and δspeed ≤ τv;
- • non-uniform speed otherwise (speed fluctuations exceed the tolerance).

This logic matches the following code fragment:

- 1 avg_speed = sum(speeds) / len(speeds)
- 2 max_dev = max(abs(s - avg_speed) for s in

→ speeds)

→ # delta t_i

time to a frame index and reading off the corresponding ai entry.

- 3 max_rel_dev = max_dev / avg_speed if avg_speed !=

→ 0 else 0.0

- 4
- 5 if avg_speed == 0:
- 6 # almost no motion
- 7 ...
- 8 else:
- 9 if max_rel_dev <= UNIFORM_SPEED_TOLERANCE:
- 10 # approximately uniform speed
- 11 ...
- 12 else:
- 13 # not uniform-speed
- 14 ...

To summarize acceleration behaviour over the entire clip, we perform an analogous diagnostic analysis on the set of defined accelerations. Let Ia be the index set of frames for which ai is defined (starting from the third frame), and let Na = |Ia|. We compute the mean acceleration

1 Na i∈I

###### ai

a¯ =

a

and the maximum absolute deviation from this mean,

Acceleration measurement and acceleration diagnostics. Starting from the third frame, we derive a scalar acceleration sequence that describes how quickly the object’s speed changes over time. Given the per-frame speeds {si} defined above, and the same temporal spacings ∆ti = ti − ti−1, we define for all frames i ≥ 2

∆maxa = max i∈Ia

|ai − a¯|.

If |a¯| is larger than a small absolute threshold εmin (parameter MIN ABS ACCEL), we also measure the maximum relative deviation

 

si − si−1 ∆ti

0, if |a¯| ≤ εmin, ∆maxa |a¯|

ai =

,

δaccel =



, otherwise.

so that ai measures the finite-difference rate of change of speed between frames i−1 and i. The script computes these values alongside the speeds:

Given a user-specified tolerance τa (parameter UNIFORM ACCEL TOLERANCE), the script then classifies the acceleration regime as:

- 1 prev_speed = None
- 2 prev_frame = None
- 3
- 4 for frame in range(frame_start, frame_end + 1):
- 5 scene.frame_set(frame)
- 6 loc = obj.matrix_world.translation.copy()
- 7
- 8 if prev_frame is None:
- 9 dt = 0.0
- 10 speed = 0.0
- 11 accel = None
- 12 else:
- 13 dt = (frame - prev_frame) / fps
- 14 disp = loc - prev_loc
- 15 vel_vec = disp / dt
- 16 speed = vel_vec.length
- 17
- 18 if prev_speed is None:
- 19 accel = None
- 20 else:
- 21 accel = (speed - prev_speed) / dt #

→ a_i

- 22
- 23 write_per_frame_entry(frame, speed, accel)
- 24 prev_loc = loc
- 25 prev_speed = speed
- 26 prev_frame = frame

- • near-zero acceleration if |a¯| ≤ εmin; in this case the overall acceleration is negligible and the clip is better interpreted as approximately constant-speed (or almost static);
- • approximately uniformly accelerated if |a¯| > εmin and δaccel ≤ τa; the sign of a¯ further distinguishes uniform acceleration (a¯ > 0) from uniform deceleration (a¯ < 0);
- • irregular acceleration otherwise, meaning that the accelerations fluctuate significantly around their mean.

This classification logic is implemented as:

- 1 avg_accel = sum(accels) / len(accels)
- 2 max_dev_a = max(abs(a - avg_accel) for a in

→ accels)

- 3 if abs(avg_accel) > MIN_ABS_ACCEL:
- 4 max_rel_dev_a = max_dev_a / abs(avg_accel)
- 5 else:
- 6 max_rel_dev_a = 0.0
- 7
- 8 if abs(avg_accel) <= MIN_ABS_ACCEL:
- 9 # acceleration is very small -> closer to

→ uniform speed / almost static

- 10 ...
- 11 else:
- 12 if max_rel_dev_a <= UNIFORM_ACCEL_TOLERANCE:
- 13 if avg_accel > 0:
- 14 # approximately uniformly

→ accelerating

- 15 ...
- 16 elif avg_accel < 0:

For each frame i in the valid range, this produces a scalar acceleration value ai (undefined at the first two frames), which we store in the same text table. Any question that refers to “the acceleration at frame i” or to “the acceleration over a given interval” is answered by mapping the queried

- 17 # approximately uniformly

→ decelerating

- 18 ...
- 19 else:
- 20 # numerically close to zero
- 21 ...
- 22 else:
- 23 # not uniformly accelerated
- 24 ...

These acceleration diagnostics never modify the underlying frame-level values {ai}, just as the velocity diagnostics never modify {si} or {vi}. Instead, they provide a principled, threshold-based way to tag each clip as approximately constant-speed, uniformly accelerated, uniformly decelerated, or irregular, allowing us to phrase velocity- and acceleration-related questions at a level of precision that matches the actual motion regime present in each clip.

###### D.1.4. Depth and Distance

All geometric quantities used in our benchmark are computed directly inside Blender in world coordinates via a Python script that evaluates the scene frame by frame. The script reads the world-space transforms of selected entities, computes Euclidean distances, and logs a per-frame table into a Blender Text Editor text block. From this table, we derive two kinds of quantities: (i) depth metadata between objects and the camera, and (ii) inter-object distances in 3D and in projected 2D planes.

Depth metadata (object–camera). We extract depth values in Blender only as auxiliary geometric metadata, not appearing in the inference question text or supervised targets. For any object o with world-space position

pt(o) = (xot,yto,zto) ∈ R3

and the active camera object with world-space origin

pcamt = (xcamt ,ytcam,ztcam) ∈ R3, at time t, we define the depth as the 3D Euclidean distance

ddeptht (o) = pt(o) − pcamt 2

= (xot − xcamt )2 + (yto − ytcam)2 + (zto − ztcam)2.

During annotation, we treat ddeptht (o) as a time series and often materialize it at a small set of shared reference time points (e.g., t1 = 1s, t2 = 2s). Thus, for a moving object we record pairs such as

ddeptht

(o), ddeptht

(o) ,

2

1

and we use the same time points for other relevant entities in the scene (e.g., additional moving objects or candidate target

objects). This temporal alignment ensures that when we expose depth values as priors, they are always comparable across objects at exactly the same timestamps. For static objects, the depth is constant over the clip, so the values at different time points are trivially identical.

Crucially, we never use these depth values as evaluation targets. We do not ask questions of the form “What is the distance between object o and the camera at time t?”, and we do not treat ddeptht (o) as ground truth in any task. Instead, depth is used only as internal geometric metadata and, in some infer questions, as numeric priors that can help a VLM reason about the relative 3D configuration of the scene. Even in those infer questions, we do not necessarily expose the camera–target depth of the queried object itself; rather, we expose a subset of aligned depth values for selected objects. All exposed depth values are manually reviewed to ensure that, together with the visual information, they are logically sufficient to infer the correct answer.

3D inter-object distances. We use the same Blender script to annotate 3D distances between pairs of entities. In our benchmark, these entities are either (i) two moving objects, or (ii) a moving object and a static reference object in the scene. We do not treat object–camera distances as interobject ground truth; camera-related distances only appear as depth metadata as described above.

Let pt(a) = (xat ,yta,zta) and pt(b) = (xbt,ytb,ztb) denote the world-space positions of entities a and b at time t. The 3D object–object distance is

d3tD(a,b) = pt(a) − pt(b) 2

= (xat − xbt)2 + (yta − ytb)2 + (zta − ztb)2.

For objects whose dimensions do not change over time, we take pt(o)to be the Blender object origin in world coordinates. For articulated skeletal models (e.g., humans), transformations are defined on a set of bones rather than a single rigid body. In these cases, we first enumerate all bones in the armature and select a semantically stable reference joint (for humans, typically the pelvis/hip bone) as the anchor. All distances involving that character are then defined using the world-space position of this reference bone, which provides a temporally stable and semantically meaningful notion of “where the character is”. For some non-humanoid articulated assets or composite rigs where no single point or bone has a clear semantic interpretation as the “character center” (e.g., certain vehicles or multi-part machines), we instead introduce an auxiliary helper object. Concretely, we create an Empty object h rigidly parented to the rig root, snap its origin to the root in world space, and log its world-space position pt(h) at each frame. In those scenes, all distances involving the articulated asset are computed with respect to

pt(h) rather than a specific bone. This dual strategy (bonebased anchors for humanoid characters and helper-object anchors for other articulated assets) keeps the distance annotations consistent while accommodating the diversity of rig structures in our Blender scenes.

- 7 def world_pos(obj_or_bone):
- 8 """Return world-space translation of an

→ object or a specific bone."""

- 9 # For regular objects:
- 10 if hasattr(obj_or_bone, "matrix_world"):
- 11 return obj_or_bone.matrix_world.to_trans⌋

→ lation()

- 12 # For pose bones (e.g.,

→ armature.pose.bones["Hips"]):

- 13 return obj_or_bone.matrix.to_translation()
- 14
- 15 for f in range(f_min, f_max + 1):
- 16 scene.frame_set(f)
- 17
- 18 # Example: distance between entities A and B
- 19 p_a = world_pos(entity_a)
- 20 p_b = world_pos(entity_b)
- 21
- 22 # Full 3D distance in world coordinates
- 23 dist_3d = (p_a - p_b).length
- 24
- 25 # Example planar distance in the XY plane

→ (horizontal separation):

- 26 # other planes (XZ, YZ) are obtained

→ analogously in the full script.

- 27 p_a_xy = Vector((p_a.x, p_a.y, 0.0))
- 28 p_b_xy = Vector((p_b.x, p_b.y, 0.0))
- 29 dist_xy = (p_a_xy - p_b_xy).length
- 30
- 31 # Timestamp (seconds), assuming t = 0 at

→ f_min

- 32 t = (f - f_min) / fps
- 33
- 34 # Log one line (frame, time, 3D distance,

→ XY-plane distance, ...)

- 35 write_to_text_block(f, t, dist_3d, dist_xy)

Planar (2D) distances. Some tasks explicitly constrain distance reasoning to a 2D projection, for example “horizontal distance in the ground plane” or “distance in the vertical cross-section”. To support such tasks, we also annotate planar distances by projecting 3D positions onto a chosen coordinate plane Π ∈ {XY,XZ,YZ}. We define the projections

ΠXY(x,y,z) = (x,y), ΠXZ(x,y,z) = (x,z), ΠYZ(x,y,z) = (y,z).

The planar distance between a and b at time t is then

dplanet (a,b) = Π pt(a) − Π pt(b) 2.

Whether a distance question is treated as 2D or 3D in our benchmark is fully determined by (i) the infer question and (ii) the geometric category of the underlying video (2D vs. 3D sequence). Each video is assigned to a 2D or 3D split based on how it is generated and how its geometry is intended to be interpreted.

- For videos in the 2D split, distances are interpreted in a

single plane by construction. If a question in such a video simply asks for “the distance between objects a and b”, the ground-truth target is the planar distance dplanet (a,b), and the task is labeled as 2D.

- For videos in the 3D split, we have full world-space geom-

The same script is run with different choices of entity -

a and entity b to (i) record object–camera distances as depth metadata and (ii) generate object–object distance annotations. In the code listing above, we show the XY-plane case for concreteness. In practice, this block is used as a template: when planar distances in other coordinate planes (XZ or YZ) are needed, we simply modify the projection lines accordingly (e.g.,Vector((p a.x, p a.z, 0.0)) for XZ or Vector((0.0, p a.y, p a.z)) for YZ) and rerun the script for that scene.

etry. If the infer question explicitly restricts reasoning to a plane (e.g., “horizontal distance” or “vertical distance”), we again use the planar distance dplanet (a,b) and label the task as 2D. Otherwise, distance questions on 3D videos default to the full 3D Euclidean distance dt(a,b) defined above, and these tasks are labeled as 3D.

Blender implementation and per-frame logging. All depth and distance quantities above are computed by a unified Blender Python script. For a specified frame range [fmin,fmax], the script iterates over frames, queries worldspace positions, computes distances, and writes a formatted table into a Blender Text Editor text block. The core of the script is:

All measurements are derived directly from Blender’s world-space transforms, so the numeric values used in our questions and annotations match the underlying scene geometry exactly. Depth values are stored only as auxiliary geometric metadata or optional numeric priors, whereas interobject distances serve as the ground truth posterior.

###### D.2. Lab Data Annotation

Unlike Blender-based simulations, lab captures do not provide annotations that can be directly read out from software. Nevertheless, with our multi-stereo camera setup, we are able to reconstruct both the 3D scene and the 3D geometry of the target objects.

- 1 import bpy
- 2 from mathutils import Vector
- 3
- 4 scene = bpy.context.scene
- 5 fps = scene.render.fps
- 6

To obtain annotations for lab captures, we first attempted to use off-the-shelf 3D reconstruction models such as FoundationPose [44] to assist with annotation. However, in our experiments, we found that even state-of-the-art AI models struggled to localize objects stably in world coordinates and frequently failed under occlusions.

Therefore, we instead rely on traditional geometry-based reconstruction with calibrated camera poses. We use the main camera together with manually selected metric depth values to recover the objects’ world coordinates. Concretely, we use the UI-based annotation tool introduced earlier (see Figure 26) to manually click on the center of the target object across a sequence of frames in the video. The tool records the 2D coordinates of each click in image space together with the corresponding metric depth from the main camera. Using the calibrated camera intrinsics, we then back-project the annotated object centers into the world coordinate system.

To obtain static priors such as object size, we manually measure the shape and dimensions of each object to reduce measurement error. To obtain dynamic priors such as object velocities and accelerations, we annotate the object center across at least 5 adjacent frames (for smoothing purpose) and compute the instantaneous velocity and acceleration magnitudes from the resulting sequence of world coordinates, assuming a constant frame rate. Formally, the computation of dynamic priors in lab captured videos are:

xworldk+1 − xworldk ∆t

vkworld ≈

,

xworldk+1 − 2xworldk + xworldk−1 ∆t2

aworldk ≈

,

where ∆t is determined from the frame rate of the captured videos.

Note that, based on our predefined questions and the priors required at specific time stamps, not all frames of the lab videos need to be annotated. This greatly reduces the annotation workload and helps prevent error accumulation over time.

When the object is occluded in the main camera view, we instead estimate its world coordinates by averaging the positions obtained by transforming its 2D annotations from the other three cameras into the world coordinate system.

###### D.3. Internet Data Annotation

For internet videos, we must derive kinematic ground truth from raw pixels rather than from simulator logs or calibrated sensors. To keep the process systematic, we adopt a threestage annotation workflow: (i) metadata and task specification, (ii) pixel-level measurement with a custom video tool, and (iii) conversion from pixel-space kinematics to real-world quantities.

###### Metadata and task specification. Each collected internet

clip is first assigned a unique identifier (e.g., internet 0001). Based on the scene content, annotators determine the physical prior available in the video (e.g., gravity g = 9.8m/s2, the length of a credit card, lane width on a road, or conveyor-belt speed), and label the corresponding video type code (e.g., S2MC, A2SS) as described in subsection B.2. We then identify all feasible inference targets in the clip (e.g., vehicle speed, ball acceleration, object size) and enumerate kinematic inference questions that can be solved from the chosen prior. For each question, we annotate the inference type (static vs. dynamic posterior) and write the final textual question so that the requested quantity, unit, and time reference (instantaneous vs. average) are explicit.

Pixel-level measurement tool. To read off pixel-space trajectories, we build a small annotation tool in Python using OpenCV. The script takes as input the path to a video file and interactively queries the annotator for the correct frame rate. It first shows the FPS detected by OpenCV, then allows the annotator to either accept it or manually enter a more reliable value (e.g., 24/25/30/60fps). The tool verifies or manually counts the total frame number to avoid metadata errors, and reports basic properties such as resolution and duration.

After loading, the script launches an interactive player (Figure 35). Each frame is overlaid with its index and timestamp, Frame: k/N | Time: tk s, and a play/pause status indicator. A trackbar at the bottom allows direct jumping to any frame; dragging the slider or entering a frame index updates the display immediately. Annotators can also step frame-by-frame with the left/right arrow keys or play the sequence in real time with the space bar. In addition to these controls, we use a standard OpenCV mouse callback to display the current pixel coordinates (x,y) of the cursor, which facilitates precise measurement of distances and object positions.

(3) Measuring pixel-space kinematics. Given the verified frame rate f (in fps), we set the time step as ∆t = 1/f. Annotators use the player to locate the relevant frames for each question: for instantaneous quantities we jump to the frame closest to the target timestamp; for average quantities we select a sequence of frames spanning the interval of interest. Along the main motion direction, we record (i) the object’s pixel length Spixel (e.g., bumper-to-bumper for a car), and (ii) the pixel coordinate xpixelk of the object’s reference point at discrete times tk = k∆t.

From these measurements we compute pixel-space kinematics using finite differences. For a 1D trajectory {xpixelk }Kk=0, the (approximate) velocity and acceleration

[Figure 165]

frame informa on and  mestamp

[Figure 166]

jump to a speciﬁed frame

pixel coordinates

Figure 35. Pixel-level measurement tool.

at time tk are



Sworld Spixel

, if size prior is given |Vtworld

xpixelk+1 − xpixelk ∆t



Vkpixel ≈

| |Vtpixel

,

, if velocity prior is given |Aworldt

0

γ =

|

xpixelk+1 − 2xpixelk + xpixelk−1 ∆t2

0

Apixelk ≈

| |Apixelt

.



, if acceleration prior is given.

0

|

0

where t0 is the timestamp at which the prior is defined. Once s is determined, any kinematic quantity of the in-

When motion is measured in two image-plane directions, we apply the same formulas component-wise to (xpixelk ,ykpixel). In this way we obtain, for every prior object and inference target, the relevant pixel-space size Spixel, velocity Vkpixel, and acceleration Apixelk .

ference target can be expressed in world units by a simple rescaling:

(Starget)world = γ (Starget)pixel, (Vtarget)worldt

= γ (Vtarget)pixelt

, (Atarget)worldt

(4) Converting to real-world kinematics. Assuming planar motion (subsection B.1), a single scalar scale factor γ > 0 with units [world length / pixel] suffices along the motion direction. Depending on which physical prior is available for a given clip, we estimate γ via

k

k

= γ (Atarget)pixelt

.

k

k

These world-space values are then used as the ground-truth priors and posteriors in QUANTIPHY.

###### D.4. Segmented Data

Segmenting out target objects from the videos only changes the background and does not alter their original physical properties. Therefore, we reuse the original annotations of the videos for the segmented data without any modification.

#### E. Vision-Language Models

We evaluate a diverse suite of 21 VLMs, spanning proprietary closed-source APIs, open-source models hosted via Replicate, and open-source models deployed by our own.

Proprietary Models. We select flagship models from four major providers, including standard multimodal capabilities and those utilizing inference-time reasoning.

- • OpenAI. We evaluate two models: GPT-5.1 [31] (gpt-5.1-2025-11-13) and GPT-5 [30] (gpt-5-2025-08-07), both are flagship multimodal models. The latter is a “thinking model” that leverages Chain-of-Thought (CoT) processing for complex reasoning tasks. Both are accessed via the OpenAI API, accepting video inputs as sequences of base64-encoded frames to enable temporal understanding.
- • Google. We evaluate Gemini 2.5 Pro [17] (gemini-2.5-pro), a high-capacity model capable of processing large context windows of interleaved images and text, and Gemini 2.5 Flash [16] (gemini-2.5-flash), a lightweight variant optimized for low-latency tasks. Both models accept base64 inline data and are optimized for complex reasoning across modalities.
- • Anthropic. We utilize Claude Sonnet 4.5 [2] (claude-sonnet-4-5-20250929). This model accepts a structured list of content blocks (text and base64encoded images). It is characterized by its detailed explanatory capabilities, often providing extensive textual rationale alongside numerical answers. (See Figure 37 for raw response examples).
- • xAI. We evaluate Grok 4.1 (Fast Reasoning) [47], a model that combines rapid inference with advanced reasoning capabilities, designed to handle complex multimodal tasks with reduced latency while maintaining high accuracy.

Open-source Models. We also evaluate 15 distinct opensource models, representing a spectrum of architectures, parameter sizes, and input modalities to systematically assess architectural variations and scaling effects.

• We include LLaVA-13B [25], a foundational baseline combining a Vicuna LLM with a CLIP encoder; VILA7B [24], pre-trained on interleaved image-text data; and Qwen3-VL-8B [5], the latest iteration of Alibaba’s visionlanguage series.

- • To examine scaling effects across model sizes, we deploy and evaluate the Qwen3-VL-Instruct series at three scales: 2B, 8B, and 32B parameters [5], enabling direct comparison of performance improvements with increased capacity within a single architecture family.
- • Similarly, we assess the InternVL-3.5 series across three parameter scales: 2B, 8B, and 30B [10], providing additional insights into how architectural choices interact with model scale for vision understanding.
- • We evaluate Fuyu-8B [22], which utilizes a simplified architecture processing raw image patches without a separate visual encoder.
- • We test Molmo-7B [13] and SmolVLM [27] (1.6B), both designed for high-efficiency reasoning on consumer hardware.
- • Phi-4 Multimodal [29] and the smaller Phi-3-Mini-128KInstruct [28], both are Microsoft suite and process text interleaved with image lists.
- • Finally, to assess temporal analysis capabilities, we include MiniCPM-V 4.5 [56] and CogVLM2-Video [43]. Distinct from frame-based approaches, these models accept direct video file inputs for native temporal processing.

In total, our evaluation encompasses 21 models (6 proprietary and 15 open-source), providing comprehensive coverage of the current state-of-the-art in multimodal reasoning across different scales, architectures, and deployment paradigms.

#### F. Prompt Design

For quantitative evaluation, we use a constrained generation strategy designed for precise numerical outputs. The system prompt explicitly restricts the output space, instructing the model to “provide ONLY the numerical answer with units” and emphasizing “No explanation or reasoning needed.” The prompt structure includes the visual input, system instructions, and ground truth priors and/or depth info, followed by the specific query and a post-prompt reinforcement. (See Figure 6 for the Text Prompt template).

To ensure reproducibility, we enforce deterministic generation where possible. We set temperature=0 (greedy decoding) for all models supporting this parameter. For hosted models via Replicate where explicit temperature control is unavailable, we utilize the default inference parameters recommended by the model maintainers.

The input sequence for all models is programmatically structured as [Video Frames][System Prompt][Ground Truth Prior and/or Depth Info][Question][Post-prompt]. Detailed examples are provided in Figure 36.

• [Video Frames]. This segment contains the full sequence of frames extracted from the source video, base64-

- 21Whatisthelengthoftheslopeincm?0.6,ballacc.4.6m/s0s,dist.slope_far1.8540m,dist.slope_near1.7010mtt=====
- 22Whatisthebasketball’sdistancefromthecameraat1.2sinmeters?acc.9.8m/s1.0s,dist.ball1.4010m,1.2s,dist.ball1.5640mtt=====
- 23Whatisthebasketball’sdistancefromthecameraat1.2sinmeters?acc.9.8m/s1.0s,dist.ball1.4010m,1.2s,dist.ball1.5640mtt=====
- 24Whatisthevelocityoftheballattheendoftheslopeincm/s?1.4,ballacc.4.05m/s1.0s,dist.ball1.7790m,dist.slope_far_end1.8150mtt=====
- 25Whatisthevelocityofthesoccerballat1.5sinm/s?gravityacc.9.8m/s1.5s,dist.soccer_ball1.4020mt===
- 26Whatisthevelocityofthetennisballat3.4s?gravityacc.9.8m/sdist.near_person2.002m,dist.far_person3.130m===
- 27Whatistheaccelerationoftheredballat0.2sincm/s?diameteroftheredball7cm0.2s,dist.book1.0995m,0.6s,dist.book1.1510mtt=====
- 28Whatistheshoulderbreadthofthepersonincm?gravityacc.9.8m/s=
- 29Whatisthebasketball’sdiameterincm?acc.9.8m/s1.0s,dist.ball1.4010m,1.2s,dist.ball1.5640mtt=====

- 210Whatisthevelocityofthetoyat2.6sincm/s?gravityacc.9.8m/s=
- 211Whatisthevelocityofthetoyat2.6sincm/s?gravityacc.9.8m/s=
- 212Whatisthediameterofthetennisballincm?gravityacc.9.8m/sdist.near_person2.002m,dist.far_person3.130m===

#QuestionGTPriorGTDepthInfo

- 214Whatistheaccelerationoftheastronautattime1sinm/s?heightoftheastronaut2.2m1s,dist.astronaut23.137mt===
- 215Whatistheaccelerationofthetennisballattime1.5sinm/s?diameterofthetennisball0.17m=
- 216Whatisthediameter(width)oftheplantinmeters?acc.ofthemugcupbefore0.5s9.8m/s=
- 217Whatisthevelocityoftheballattheendoftheslopeincm/s?1.4,ballacc.4.05m/s1.0s,dist.ball1.7790m,dist.slope_far_end1.8150mtt=====
- 218Whatisthediameterofthesoccerballincm?gravityacc.9.8m/sdist.near_person2.002m,dist.far_person3.130m===
- 219Whatisthedistancebetweentwopeopleincm?gravityacc.9.8m/s0.60s,dist.near_person1.5800m,dist.far_person3.2370mt====
- 220Whatistheheightofthepersonincm?gravityacc.9.8m/s=
- 221Whatistheheightofthetoyincm?gravityacc.9.8m/s=
- 222Whatisthewidthoftheboxincm?gravityacc.9.8m/sdist.near_person2.002m,dist.far_person3.130m===
- 223Whatistheheightoftheyellowtoyincm?gravityacc.9.8m/s0.60s,dist.jar1.9735m,dist.carrot2.6300mt====
- 224Whatisthebasketball’sdiameterincm?acc.9.8m/s1.0s,dist.ball1.4010m,1.2s,dist.ball1.5640mtt=====

- 25Whatistheheightofthewomaninyellowinmeters?walkingvelocity1.25m/s=
- 26Whatisthelengthofthespeedboatinmeters?sailboatlength12m=

13Whatisthelengthofthewoodblockincm?rulercalibre1cm=

- 227Whatisthelengthofthestringincm?gravityacc.9.8m/s0.3s,dist.ball1.7060m,0.6s,dist.ball1.3440mtt=====
- 228Whatisthedisplacementoftherollingobjectfrom0sto0.6sincm?0.6,ballacc.4.6m/s0s,dist.ball1.8180m,0.6s,dist.ball1.7288mttt======
- 229Whatisthebasketball’sdistancefromthecameraat1.8sinmeters?acc.9.8m/s0.0s,dist.ball1.3210m,1.8s,dist.ball1.5835mtt=====
- 230Whatisthevelocityofthetennisballat3.4s?gravityacc.9.8m/sdist.near_person2.002m,dist.far_person3.130m===
- 231Whatisthatball’saccelerationat0.6sinm/s?gravityacc.9.8m/s0.3s,dist.ball1.7060m,0.6s,dist.ball1.3440mtt=====
- 232Whatisthewidthofthedeskincm?1.5,ballacc.3.0m/s0s,dist.cup1.2100m,dist.desk_left0.8690mtt=====
- 233Whatisthelengthoftheslopeincm?1.5,ballacc.3.0m/s0s,dist.slope_far1.3450m,dist.slope_near1.1040mtt=====
- 234Whatisthelengthoftheboxincm?gravityacc.9.8m/sdist.near_person2.002m,dist.far_person3.130m===

- Table 36. Selected examples of text inputs and answers (part 1). Question is the natural-language prompt presented to the VLM. GT Prior is the physical prior provided to the model. GT Depth Info is the depth annotation used for 3D reasoning tasks. GT Posterior is the numeric ground-truth answer to the kinematic inference question. See Table 3 for detailed explanation. Raw Response shows the corresponding VLM output for each model, and Parsed Value shows the parsed value extracted from the Raw Response.

- 1A3MSDS30.8Lookingatthevideoframes,Icanseeatable. . .Length0.1530m15.30cm15.30Claude-4.5Sonnet==
- 2A3SSDS1.5651.5640m1.564Claude-4.5Sonnet
- 3A3SSDS1.5651.56401.564CogVLM2Video[]
- 4A3SSDD208.76velocity_at_the_end_of_the_slopeunitNoneCogVLM2Video[][]
- 5A3SSDD2.79681.5m/s\nInthegivenvideo,thesoccerballisﬂying. . .at1.5m/s.1.5Fuyu-8B
- 6A3MCDD542.822.002m/s\nIntheimage,. . .Therefore,itsvelocityis2.002m/s.2.002Fuyu-8B

#VideoTypeInferenceTypeGTPosteriorRawResponseParsedValueModel

- 214S3SSSD3.6040m/s0ChatGPT-5
- 215S2SSSD6.1299.8m/s9.8InternVL-3.5-30B

27S3SCSD1086.090.0cm/s0Gemini-2.5Flash

- 8A2MCDS3939.8cm39.8Gemini-2.5Flash
- 9A3SSDS23.223.9cm23.9Gemini-2.5Pro
- 10A2MSDD3570cm/s0Gemini-2.5Pro
- 11A2MXDD3572549cm/s2549ChatGPT-5.1→
- 12A3MCDS6.76.5cm6.5ChatGPT-5.1

13S2MCSS9.910cm10ChatGPT-5

- 18A3MXDS17.50.310.280.28LLaVA-13B[][]
- 19A3MXDS3253.2370m3.237LLaVA-13B[][]
- 20A2MCDS170<think>So,let’stryto. . .maybetheperson’sheightisaround170cm.170MiniCPM-V4.5-8B
- 21A2MXDS14thinkOkay,soIneedto. . .IfIcandeterminethetimeitNoneMiniCPM-V4.5-8B<>
- 22A3MXDS7.20.000000000000000000000 . . .0Molmo-7B
- 23A3MSDS1415.2cm15.2Molmo-7B
- 24A3SSDS23.223.9cmend23.9Phi-3-Mini-128K-Instruct||<>
- 25V2MCDS1.71Cannotbedeterminedfromgiveninformation.NonePhi-3-Mini-128K-Instruct
- 26S2MCSS4.8ValuemissingduetoinsufﬁcientinformationNonePhi-3-Mini-128K-Instruct[]
- 27A3SSDS3034.02cm34.02Phi-4-Multimodal-Instruct
- 28A3MCDS207.922.8cm22.8Phi-4-Multimodal-Instruct→
- 29A3SCDS1.58351.5835m1.5835Qwen3-VL-Instruct-8B
- 30A3MCDD542.820.0m/s0Qwen3-VL-Instruct-8B
- 31A3SSDS4.9390.000000000000000000000 . . .0VILA-7B
- 32A3MXDS857.91.2100m1.21VILA-7B
- 33A3MSDS30.81345013450SmolVLM-Instruct
- 34A3MCDS19.51.500000000000000041.5SmolVLM-Instruct

16A2MCDS0.0750.05m0.05InternVL-3.5-30B

217A3SSDD208.764.05m/s4.05LLaVA-13B[][]

- Table 37. Selected examples of text inputs and answers (part 2). Question is the natural-language prompt presented to the VLM. GT Prior is the physical prior provided to the model. GT Depth Info is the depth annotation used for 3D reasoning tasks. GT Posterior is the numeric ground-truth answer to the kinematic inference question. See Table 3 for detailed explanation. Raw Response shows the corresponding VLM output for each model, and Parsed Value shows the parsed value extracted from the Raw Response.

encoded. We normalize all videos to 480p resolution. Our preliminary exploration indicated that while lower spatial resolution (480p) has negligible impact on physics reasoning, temporal subsampling (dropping frames) significantly degrades the tracking of velocity and acceleration. Therefore, we prioritize temporal fidelity by retaining all frames from the source video. This approach also distinguishes our methodology from most prior works, which often prioritize spatial resolution at the expense of frame rate.

- • [System Prompt]. This serves as the behavioral instruction, establishing the persona that “You are an expert video analyst...” We selected this formulation following a pilot study of five prompt variations on a subset of 15 videos, which revealed no significant difference and this persona yielded the relatively highest adherence to formatting constraints without altering reasoning accuracy.
- • [Ground Truth Prior and/or Depth Info]. This includes physical constants and contextual priors (e.g., “Given that acceleration a = 9.8m/s2”). For 3D scenes, this also includes depth information (e.g., “At t = 1.0s, the distance of the ball to the camera is 1.779m...”) for estimation.
- • [Question]. The specific physics query (e.g., “What is the total displacement in the y-axis?”).
- • [Post-prompt]. A final instruction reinforcing the output format (e.g., “Provide... ONLY the numerical answer with units”) to mitigate the tendency of models to generate verbose ”Chain-of-Thought” explanations in the final output.

#### G. Answer Retrieval and Parsing

For each question, we query the model once. To ensure robustness against API instability or transient errors, we implement an automated retry mechanism: if a query fails (e.g., timeout or server error) or yields a non-parsable response, the request is re-submitted up to a maximum of five times.

To extract quantitative data from potentially noisy model outputs, we employ a hierarchical parsing function (parse number) designed to handle both concise and verbose responses. The logic proceeds as follows.

- 1. Exact Match Validation: We first check if the raw response strictly matches a numerical format (with or without units). If the response is concise (containing only a value and a unit matching the requested physical quantity), the numerical value is extracted directly.
- 2. Delimiter Search: If the response is verbose, we scan for explicit answer markers, including ‘‘=", ‘‘Final Answer:", ‘‘Answer:", ‘‘=>", and ‘‘:". If a delimiter is identified, we discard the preceding text and retain only the substring immediately following it. If multiple cases are shown, we take the last occurrence.

- 3. Unit Sanitization: The retained text is cleaned of common physical units (e.g., “meters”, “m/s”, “kg”) to prevent string processing errors during numerical extraction.
- 4. Heuristic Extraction: Finally, a regular expression is applied to the cleaned text to identify floating-point numbers. We also take absolute value of the numerical answer. Crucially, if multiple numbers are found, we extract the last valid number in the sequence. This heuristic assumes that in verbose Chain-of-Thought responses, the final conclusion is located at the end of the text.
- 5. Failure Handling: If no valid number is identified after these steps, the response is recorded as a failure (parsed as None).

This strict parsing pipeline is essential because our evaluation metric relies on exact numerical regression. If a model provides only a qualitative description (e.g., “The ball is moving quickly”, “Cannot be determined”, or API error) or fails to reach a numerical conclusion within the decoding limit, it is then treated as a failure.

While most models adhered to the format constraints (i.e., [number] [unit]), we observed that Claude Sonnet 4.5, Fuyu-8B, and MiniCPM-V 4.5 frequently generated verbose, multi-sentence explanations despite instructions to the contrary. By targeting the post-delimiter text and the last numerical value, our parsing logic effectively retrieves the correct answer from these verbose outputs. Representative examples of raw responses and their parsed values are detailed in Figure 37.

#### H. Human Study Details.

To contextualize model performance and establish an empirical upper bound on human quantitative physical reasoning, we conducted a survey study using the Gorilla Experiment Builder platform. The platform supported video presentation, question randomization, and structured data collection. Below, we describe the participant cohort, task construction, interface design, evaluation methodology, and resulting performance trends.

###### H.1. Participants

Participants were recruited from a mix of undergraduate, graduate, and PhD researchers, including individuals with advanced training in various fields including engineering, physics, and mathematics. This allowed us to approximate both typical human performance and an expert-level upper bound. Participants were informed that they could freely choose their reasoning strategy—including intuitive estimation, visual approximation, or explicit calculation—in order to reflect natural human reasoning rather than enforce a prescribed computation protocol.

Our survey is divided into two versions: a 2D survey and

[Figure 167]

(a) 2D survey interface. Participants see the prior ground truth, question text, and a numeric answer box, and can replay or scrub the video.

[Figure 168]

(b) 3D survey interface. The layout mirrors the 2D condition, with additional depth prior ground truth.

Figure 38. Human study interface. Example screenshots of the 2D (top) and 3D (bottom) survey UIs. Both interfaces present the physical prior, quantitative question, and a numeric input field, while allowing participants to replay and scrub the video timeline.

a 3D survey. Both the 2D and 3D surveys were completed by multiple participants, including several with substantial technical backgrounds. A subset of participants completed both tasks, enabling direct cross-dimensional comparisons of individual consistency.

###### H.2. Task Construction and Experimental Design

Following the 36 fine-grained video categories defined in subsection B.2, we organized the full dataset into these category units for both model and human evaluation. Each category contains a small pool of representative videos, from which stimuli were sampled during the experiment.

Participants were randomly assigned to either the 2D survey or the 3D survey. Each participant viewed 18 videos, one sampled per category within their assigned dimensionality. Each video was followed by 1–3 quantitative kinematic questions, with the same priors and task formulations used in our VLM evaluation (e.g., estimating acceleration magnitude, inferring relative size changes, or recovering object velocity).

The UI interface was intentionally designed to be concise and unobtrusive. For each trial, participants were presented with:

- • A video with standard playback controls;
- • The physical prior ground truth (A/S/V) and/or corresponding depth information for 3D videos;
- • The quantitative question texts;
- • Numeric input boxes (numbers only).

Participants could freely replay, pause, and scrub the video timeline. Scrubbing resolution was restricted to whole seconds to match the temporal information available to VLMs, which process videos solely as pixel sequences without access to exact timestamps. The interface permitted unrestricted video replay to minimize memory effects and ensure that both humans and VLMs operated over comparable visual evidence.

- H.3. Evaluation Metric

Human responses were evaluated using the same MRA metric introduced in Section 4. For each individual question, we compute MRA by averaging accuracy over ten relative-error tolerance thresholds, yielding a smooth, threshold-agnostic measure of quantitative precision. After computing an MRA score for every answered question, we aggregate performance at the participant level by taking the mean MRA across all questions completed by that participant. Figure 39 presents participants performance for the 2D and 3D surveys, respectively.

- H.4. Results and Observations We observe a few findings from the results.

[Figure 169]

Figure 39. Distribution of human quantitative reasoning performance. Horizontal boxplots summarize participant-level mean MRA scores for the 2D (top) and 3D (bottom) survey conditions.

Strong cross-dimensional consistency. Across participants who completed both tasks, MRA scores for 2D and 3D surveys were highly correlated. High-performing participants in 2D almost universally remained high-performing in 3D, suggesting that human physical intuition is stable and transferrable across dimensional modalities.

Human upper bound substantially exceeds VLM performance. Top human participants achieve MRA = 0.721 in 2D and MRA = 0.724 in 3D, showing better performance than the evaluated VLMs. Although the average human performance is not higher than model performance, the human upper bound remains far above current VLM capabilities. This gap is particularly notable given the modest sample size of our study; even with limited data, humans exhibit strong physical reasoning competence that remains challenging for contemporary models.

Practical implications. These findings indicate that while VLMs may approximate average-level human intuition in some settings, they remain far from achieving human-like precision or matching expert-level reasoning. The human study therefore provides a meaningful benchmark for assessing the gap between current VLM capabilities and the aspirational goal of physically grounded, human-level visual reasoning.

#### I. Sketchfab Model Sources

In Table 7, we list the Model Name, Author, and Model ID for all Sketchfab models used in the Blender-generated videos in our dataset.

Table 7. Sketchfab 3D Models

|Model Name|Author<br><br>|Model ID|
|---|---|---|
|Supermarket trolley broken<br><br>|kreems|7f460877380349f8886280a596253034|
|supermarket 1|amogusstrikesback2<br><br>|ca1def2b7be544068def3cec5852c67e|
|Pool Table (Animation)|Yanez Designs<br><br>|0f2ae181a2dd4b00a6ec25073692037f|
|Bowling Pack (Bowling Pins & Ball)<br><br>|EverZax|52743c4714c14211ac71d2fe1e5c8da3|
|Bowling Club<br><br>|tiunov.se|0b8fae45fcda4fe78f93bb2a899401a6|
|Elephant<br><br>|planeta-elefante<br><br>|f8778fc3d161481abba7ec23a8ddd1e8|
|Walking Astronaut<br><br>|Unknown Animaker<br><br>|d9062a2003df422abdafdc02afdac085|
|Mr Man Walking|Instinto Ideal Studio<br><br>|98ccac2b0e2845789b6f789978ca06ed|
|Jupiter & satellites|Sakado<br><br>|379fd77b970c4821898c05c483913dec|
|Model 92A - Great White Shark|DigitalLife3D<br><br>|702e7b53637f4ded9ca479a8124e810d|
|Model 95A - Adult Leatherback Sea Turtle, ”Mac”|DigitalLife3D<br><br>|4974c93644a24da280fde68cba74a12d|
|Model 69A - Striped Bass<br><br>|DigitalLife3D|35be3af9a7c4441c98109e5562d36c09|
|Underground Parking Lot<br><br>|Janis Zeps|0ad5c221525b4bbba3a164c6235d28b8|
|Airport Car|Jungle Jim|29d1c6260b134a3baf5231f34de1b24f|
|Airport Catering Truck<br><br>|rwy00<br><br>|289f4e2cfa3f4722b0476b1fc37681d8|
|Towel|TheSpacePunk|6edc341457e44603ab351470ab800493|
|Plain Mug|LightSwitch|19c8fe5702b544d0a1409d3dac1cf90e|
|Flour ARIDLL|fwild<br><br>|d40db94d92ab4e7a82a1c199312f3985|
|Milk<br><br>|Multipainkiller Studio<br><br>|5e8d71045e2040ba8f6619d86d204cf5|
|Apple [Scan]<br><br>|hoxsvl.scan<br><br>|facb1aa6928c4f8f82d87a019b9f134e|
|PACKAGING EMPAQUE DE HUEVOS|willinando1w08|cd51c5d25bbd4370881aebd3648cfe8c|
|Kitchen Pack|GRIP420|a513f2e85bc94ae5b6d8cbd74909e3c4|
|Kitchen Tools|anybody<br><br>|417e3873fcb34f7ab9744506d7bcc838|
|Coffee Shop Cup<br><br>|David Zerba<br><br>|37e6805f2b7a4158a1d61fe75f8e2a33|
|Cup<br><br>|Dmytro Nikonov<br><br>|7d450bb714034fceaa7b59a0e564f46b|
|Bird Animations Alex<br><br>|ahitch3|081fa7f0cfd649b9b07babb4c619acc7|
|Dove Bird Rigged<br><br>|FourthGreen|6b91be2a28fb4404a2d57d5ca98bd4dc|
|POLICE CAR - Belgian VW Transporter<br><br>|Mickael Boitte|35eaf9505e13464385404402ad865508|
|Jo on Bike - Rigged & Animated<br><br>|NEEEU Spaces GmbH|36ee5344e81149858a664cde9f98e835|
|Simple Factory Scene|Pickeri<br><br>|804fbe0cd1fa44fa9ca86ae42c82d63d|
|Car|Paulius|a619dd25a6c04af0b2d8730aa1cb058b|
|Basketball Court|tiunov.se<br><br>|5faad7b528124907ab82732ed0c6b743|
|Basketball NBA Championship Official Trophy<br><br>|johnnokomis|04f6f1135ffb48749c43c9c20c75fc19|
|Bouncing Basketball|Maurice Svay<br><br>|b8731a2fda6849c9a164d1966dc16ff8|
|Zen Japanese Tea House with Go Board|winters810|a1e95e5efee349a693f30eda32401aef|
|Chinese Chess|chung the artist<br><br>|7ad0f4f0ebbc455d9e9f829c956dda80|
|paper airplane<br><br>|vesicalsnail|0967ab4a9c654a569a13ea1f8d9dca0c|
|3d soccer ball|BlenderMaster|0400146e0e3c4d8f8b57bfa06d7dfb4c|
|Cardboard Box<br><br>|Pricey1600|4e622ef1a09c43e28a49d9fa37f9eeee|
|Tennis Ball Low-poly PBR<br><br>|MaX3Dd<br><br>|e5c2b0e5860549acaa2dfe8b764d5f94|
|Old Camera Bag .::RAWscan::.|Andrea Spognetta (Spogna)<br><br>|788f8b75874f417ebde498ffd231410c|
|Animated ROBOT SDC|SDC PERFORMANCE|3d127f327a6c4033a32b810b5fb071ed|
|Gift Box|local.yany<br><br>|83296611584143a3afad6c0d0c0a4227|
|Sci-fi Box<br><br>|Igor K.<br><br>|ca415724f32043489fcab2ec74582619|
|Cat Walk<br><br>|LostBoyz2078|915680200c064815bba75e008ba9efb5|
|Deer walk|LostBoyz2078|229ba6ba0d1e4811ab89382f74601e16|

Continued on next page

###### Table 7 – Continued from previous page

|Model Name|Author<br><br>|Model ID|
|---|---|---|
|Horse|kenchoo<br><br>|86d47bdcd5ab41238ba44547e4d21f9c|
|Horse Walk|Amitesh Nandan<br><br>|93b53ddcec414592842753d1819f3133|
|Rabbit Rigged<br><br>|FourthGreen|e7213589744d436b9d96e2dbb31198a5|
|phoenix bird|NORBERTO-3D<br><br>|844ba0cf144a413ea92c779f18912042|
|First Aid DX2 - 300 Followers Celebration<br><br>|re1monsen|c5ddfa9e6309403083bbce60bdcc3d71|
|Nathan Animated 003 - Walking 3D Man<br><br>|Renderpeople<br><br>|143a2b1ea5eb4385ae90a73657aca3bc|
|Chernovan Nemesis|Swiss Fox<br><br>|c6c91c73e93444f4b72d6c24db778e73|
|Dragon Fly|LostBoyz2078<br><br>|1443a7efe5d5450b8db4c15d8ff5c343|
|Borboleta Azul - Butterfly|Lancaster Modelagem 3D<br><br>|ab9192b6bc8f49e3baed63e984c7073a|
|Blue Whale - Textured<br><br>|Bohdan Lvov<br><br>|d24d19021c724c3a9134eebcb76b0e0f|
|jellyfish<br><br>|yanix|d06a5a553fe641ab92f720527b2278f3|
|Koi Fish<br><br>|7PLUS|236859b809984f52b70c94fd040b9c59|
|Running Raccoon Animation|Santrez|bfce4d4815234c39bcf012352e52c27e|
|(FREE) Cyberpunk Hovercar<br><br>|Lionsharp Studios|3205b1075bb44ffc826bce0c2a04d74c|
|White Eagle Animation Fast Fly<br><br>|GremorySaiyan|30203bf39e5145f19c79e83c550139d3|

