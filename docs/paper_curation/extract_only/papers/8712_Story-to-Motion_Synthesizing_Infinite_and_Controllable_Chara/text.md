# arXiv:2311.07446v1[cs.CV]13Nov2023

## Story-to-Motion: Synthesizing Infinite and Controllable Character Animation from Long Text

Zhongfei Qing

SenseTime Research BeiJing, China qingzhongfei@sensetime.com

Zhitao Yang

SenseTime Research ShenZhen, China yangzhitao@sensetime.com

Zhongang Cai

SenseTime Research Singapore, Singapore caizhongang@sensetime.com

Lei Yang∗

SenseTime Research ShenZhen, China yanglei@sensetime.com

|The sun was shining brightly, casting a warm glow over the quiet residential quarter. The air was filled with the sounds of birds chirping and leaves<br><br>|rustling in the gentle breeze. We decided to start our day by dancing in front of the modest buildings, moving to the beat of our favorite songs. As we danced, we felt the energy of the neighborhood pulsing through us. After a while, we continued to dance around the trees, feeling the cool breeze on our faces as we moved. It was a refreshing break from the quietness of the residential quarter, and we felt invigorated by the fresh air and natural beauty|
|---|
<br><br>around us. As our stomachs began to growl, we spotted a food cart selling hot dogs......|
|---|

[Figure 1]

Dancing, around the trees

Eating hot dogs from a food cart in the vicinity

Dancing, in front of modest buildings

Figure 1: Story-to-Motion is a new task that takes a story (top green area) and generates motions and trajectories that align with the text description.

### ABSTRACT

specific motions based on a long text description. This task demands a fusion of low-level control (trajectories) and high-level control (motion semantics). Previous works in character control and text-to-motion have addressed related aspects, yet a comprehensive solution remains elusive: character control methods do not handle text description, whereas text-to-motion methods lack position constraints and often produce unstable motions. In light of these limitations, we propose a novel system that generates controllable, infinitely long motions and trajectories aligned with the input text. 1) We leverage contemporary Large Language Models to act as a text-driven motion scheduler to extract a series of (text, position, duration) pairs from long text. 2) We develop a text-driven motion retrieval scheme that incorporates motion matching with motion semantic and trajectory constraints. 3) We design a progressive mask transformer that addresses common artifacts in the transition motion such as unnatural pose and foot sliding. Beyond

Generating natural human motion from a story has the potential to transform the landscape of animation, gaming, and film industries. A new and challenging task, Story-to-Motion, arises when characters are required to move to various locations and perform

∗corresponding author

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

SA Technical Communications ’23, December 12–15, 2023, Sydney, NSW, Australia © 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0314-0/23/12...$15.00 https://doi.org/10.1145/3610543.3626176

its pioneering role as the first comprehensive solution for Storyto-Motion, our system undergoes evaluation across three distinct sub-tasks: trajectory following, temporal action composition, and motion blending, where it outperforms previous state-of-the-art (SOTA) motion synthesis methods across the board. Homepage: https://story2motion.github.io/

### CCS CONCEPTS

• Computing methodologies → Motion capture.

### KEYWORDS

motion in-betweening, motion generation, text-to-motion, motion matching

ACM Reference Format:

Zhongfei Qing, Zhongang Cai, Zhitao Yang, and Lei Yang. 2023. Story-toMotion: Synthesizing Infinite and Controllable Character Animation from Long Text. In SIGGRAPH Asia 2023 Technical Communications (SA Technical Communications ’23), December 12–15, 2023, Sydney, NSW, Australia. ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/3610543.3626176

### 1 INTRODUCTION

Imagine the potential of effortlessly translating a long textual narrative detailing a series of human activities traversing diverse locations into seamless, lifelike human motions. This transformation not only ushers in a new era of content generation but also has the power to reshape the animation, gaming, and film industries. Leveraging the capabilities of Large Language Models, this ambitious vision has now evolved from a distant goal into a tangible reality. Herein, we introduce a pioneering effort, Story-to-Motion, a task that takes a "story" (exemplified in Fig. 1) as input and faithfully generates a sequence of motions that simultaneously conforms to low-level kinematic and high-level semantic constraints.

Previous works in character control and text-to-motion have addressed related aspects. Some efforts [Holden et al. 2017; Zhang and Tang 2022] mainly focus on matching trajectories, supporting relatively few semantic motions, while the others [Zhang et al. 2022, 2023] only focus on generating short semantic motions and ignore long motions with trajectories. Neither approach fully integrates trajectory and semantic description, making it difficult to solve the problem effectively. Besides, directly generating motions with learning-based methods is limited by motion quality, particularly for long-tail training sets. Neglected due to data imbalance, less frequent motions like handstands yield unsatisfactory results.

To generate high-quality and long motion, motion matching is widely adopted in the industry. While practical, it cannot be directly applied to this problem due to its inability to utilize text input and the tendency for blending artifacts. To address these challenges, we incorporate text embedding to match candidate motions. Furthermore, kinematics features and learned features are leveraged for further retrieval. Additionally, dynamic target trajectory is proposed to improve trajectory matching. Besides, heuristic-based blending algorithms yield subpar results with complex motions. To address this, we design a progressive mask transformer for motion transitions.

Our contributions are three-fold: 1) We propose a new task, Storyto-Motion, which considers both trajectory and semantics when

generating motions. 2) We propose Text-based Motion Matching, a promising long text-driven, controllable system to address this task. 3) Through experimentation on standard datasets, our design outperforms the current SOTA methods in all three sub-tasks.

### 2 RELATED WORKS

- 2.1 Trajectory-based Motion Synthesis

To generate motion from a given trajectory, motion matching [Clavet 2016] retrieves segments of animation stored in a database, based on the current pose and the target trajectory. Learned motion matching [Holden et al. 2020] employs an auto-regressive neural network to predict the next motion state based on a given control signal. Moreover, Phase-Functioned Neural Networks [Holden et al. 2017] incorporates a cyclic Phase Function to generate the network weights. GAMMA [Zhang and Tang 2022] employs reinforcement learning to generative motion primitives via body surface markers.

- 2.2 Text-based Motion Synthesis

Earlier research in motion synthesis from text such as JL2P [Ahuja and Morency 2019] constructs a joint space to which both text descriptions and motion sequences are mapped. Variational mechanisms have also been introduced for higher diversity. For example, TEACH [Athanasiou et al. 2022] utilizes transformer-based VAEs to produce motion with text conditions and is able to achieve temporal action compositionality. Moreover, MDM [Tevet et al. 2022], MotionDiffuse [Zhang et al. 2022] adapt diffusion models to generate human motions from text input. ReMoDiffuse [Zhang et al. 2023] further integrates a retrieval mechanism to refine the denoising process and enhance the generalizability and diversity.

- 2.3 Motion blending

Motion blending can be considered as a specialized form of motion completion [Duan et al. 2021], where the resulting motion is constrained by given context frames. Traditional solutions involve interpolating keyframes using techniques such as Bezier curves or polynomials [Bollo 2016]. In the realm of deep learning methods, SOTA approaches primarily rely on transformer networks. Duan et al. [Duan et al. 2021] utilize a pretrained language model encoder and 1D convolution to generate transition motion. Qin et al. [Qin et al. 2022] propose a two-stage approach that demonstrates improved ability for longer in-betweenings by utilizing Keyframe Positional Encoding and Learned Relative Positional Encoding.

### 3 METHODOLOGY 3.1 Text-driven Motion Scheduler

Given a story, the Text-based Motion Scheduler module prompts ChatGPT [Brown et al. 2020] in natural language for a list of text descriptions of the character’s actions T = {𝑡𝑒𝑥𝑡𝑖}, location names L, and duration 𝑡 of those actions (Fig. 2). Assuming that the 3D scene is known, we can look up the corresponding coordinates P = {(𝑥𝑖,𝑦𝑖)} with L. Then, the trajectory generation module converts the discrete P into a continuous curve via a path-finding algorithm. For discrete textual descriptions, each motion is additionally given a designated duration by LLM, while the remaining idle time is filled in with “walking”. Thus, we transform the story into a continuous

###### Text-driven Motion Scheduler

|Walk......|
|---|

###### Neural Motion Blending

Target pos

10s

5s

###### Story Motion Schedule

|Dance ......|
|---|

pos

| |
|---|
|Eat......|
| |

[Figure 2]

We started our day by dancing in front of the tall buildings. We incorporated some hip-hop and contemporary dance moves, feeling the energy and

[Figure 3]

Trajectory Generation

⚪Dance - in front of the tall buildings - 10s

Obstacle

###### LLM

| |
|---|
|Walk......|
| |

⚪Walk - to the car selling hot dogs - 12s

[Figure 4]

pos

joy of movement. After that, we walked to the car selling hot dogs and ordered some delicious snacks to refuel our

| |
|---|
|Dance ......|
| |

⚪Eat the hot dogs - near the car selling hot dogs - 5s

Current pos

10s

energy. We ate the hot dogs nearby......

⚪Chat - near the car selling hot dogs - 9s

| |Progressive| |
|---|---|---|
| |Mask Transformer| |

Query Feature

| | |
|---|---|
| | |
|Motion|Retrieval|
| | |
| | |

###### Text-driven Motion Retrieval

Database Feature

walk dance Full Motion Sequence

flip Motion Segments

| | |
|---|---|
| | |
| | |
| | |

Pretrained Sentence Encoder

[Figure 5]

[Figure 6]

[Figure 7]

Joint Kinematics Extraction

Motion Database

Pretrained Motion Encoder

- Figure 2: The proposed Text-based Motion Matching framework comprises three modules: (1) The Text-Driven Motion Scheduler extracts semantic information from the input story using a pretrained Large Language Model and obtains the trajectory based on known scene position information. (2) The Text-based Motion Retrieval module retrieves motions from the database that conform to both semantic and trajectory constraints. (3) Finally, the Neural Motion Blending module generates transition motion and concatenates motion segments into a natural-looking sequence.

[Figure 8]

[Figure 9]

t

t+1

Dynamic Mask

Q

| | |
|---|---|
|K| |

V

| | |
|---|---|
|softmax| |

Rel-attention Layer

|Step Embedding|
|---|

input

|output|
|---|

|FFN|
|---|

|RAL|
|---|

|FFN|
|---|

M

M

MM

|RAL|
|---|

- Figure 3: The Progressive Mask Transformer generates motion over multiple iterations (e.g., 3), with the number of masked missing frames decreasing linearly. FFN: feedforward network. RAL: Rel-attention Layer.

while also maintaining a similar body pose and motion style as the previous motion for coherence. We achieve this goal in two steps.

In the first step, to incorporate semantic information, a pretrained sentence encoder [Liu et al. 2019] extracts text embedding 𝑓𝑡𝑒𝑥𝑡 from the given text. Top-𝐾1 results, selected via cosine similarity, serve as candidates for subsequent matching. However, two challenges arise: (1) Imperfect text matching results in irrelevant motions. (2) Some datasets contain a small amount of low-quality clips. Consequently, the presence of noisy matched results will adversely affect the quality of the generated motion. Thus outlier removal is employed to reject noisy motion clips that are far from the distribution center.

In the second step, trajectory and coherence constraints are incorporated through motion-matching. The crucial aspect is determining the similarity measure. The original Motion Matching [Holden et al. 2020] method mainly adopts trajectory and joint position similarity. The features they used include lower body part 𝑓𝑙𝑜𝑤𝑒𝑟 = {𝑓𝑜𝑜𝑡, 𝑓𝑜𝑜𝑡, ℎ𝑖𝑝} and trajectory part 𝑓𝑡𝑟𝑎𝑗 = {𝑝𝑜𝑠, 𝑑𝑖𝑟𝑒𝑐𝑡𝑖𝑜𝑛}, where𝑝𝑜𝑠 denote the 2D future trajectory positions projected on the ground, 𝑑𝑖𝑟𝑒𝑐𝑡𝑖𝑜𝑛 are the future trajectory facing directions, 𝑓𝑜𝑜𝑡 are the two foot joint positions, 𝑓𝑜𝑜𝑡 are the two foot joint velocities, and ℎ𝑖𝑝 is the hip joint velocity. However, this strategy overlooks the consistency of the upper-body, resulting in potential swaying of the upper-body. Moreover, sudden changes in motion style can negatively impact the visual quality. Therefore, we include the upper-body feature 𝑓𝑢𝑝𝑝𝑒𝑟 = {𝑢𝑝𝑝𝑒𝑟 −𝑏𝑜𝑑𝑦, 𝑢𝑝𝑝𝑒𝑟 − 𝑏𝑜𝑑𝑦} which is the upper-body joints positions and velocities. Moreover, we train an auto-encoder model F to extract motion features 𝑓𝑙𝑒𝑎𝑟𝑛𝑒𝑑 that encompass the entire body information, together with temporal cues and motion style, enhancing the matching capability of hand-crafted features. With all designed features, we compute the Euclidean similarity for each of them, including lower-body, upper-body, trajectory, and learned features. Z-Score normalizing the features is crucial due to their potential significant differences in magnitudes. To summarize, we first use text embeddings to select top-𝐾1 candidate motions, then the top-𝐾2 desired motions are selected by the weighted sum 𝑆 of the above similarity, with

function over time, named Scheduler S(𝑡), where each time point 𝑡 corresponds to (𝑥𝑖,𝑦𝑖,𝑡𝑒𝑥𝑡𝑖), containing both low-level locations (𝑥𝑖,𝑦𝑖) and high-level textual descriptions 𝑡𝑒𝑥𝑡𝑖 about motion.

### 3.2 Text-based Motion Retrieval

Given a motion database D = {(𝑡𝑒𝑥𝑡𝑖,𝑚𝑖)}, the Text-based Motion Retrieval module matches motions in an auto-regressive manner. Here,𝑡𝑒𝑥𝑡𝑖 is the text label (e.g., "sitting") while𝑚𝑖 ∈ R𝐿×𝐷 denotes a motion sequence with 𝐿 frames. Each frame is a 𝐷-dimensional vector representing body joint rotation and overall translation. It is notable that the database consists of short clips. The duration of the target action is included in ChatGPT-generated instruction, which affects the number of clips. Hence, the system is scalable to arbitrary lengths of motion, as the main idea is to retrieve short clips and blend them into long motions.

At 𝑡 = 0, a motion 𝑚0 is randomly selected based on 𝑡𝑒𝑥𝑡0 and placed at (𝑥0,𝑦0). At the time point𝑡, given (𝑥𝑡,𝑦𝑡,𝑡𝑒𝑥𝑡𝑡) from S(𝑡) and the previous motion 𝑚𝑡−1, metrics are used to find the next best matching motion from the database. The objective is to find a motion clip that aligns with both the query text and trajectory,

adjustable weights for low-level and high-level scenarios. To ensure diversity, we choose clips (e.g., 10) randomly from the most suitable candidates.

Iterating the aforementioned matching process can generate arbitrarily long motions. However, its auto-regressive nature can lead to cumulative trajectory errors. To address this, the target trajectory is dynamically adjusted based on the current position and target position: when a position error occurs, the subsequent step will correct this error by retrieving clips that minimize the error.

### 3.3 Neural Motion Blending

Motion matching generates a sequence of motion clips, which can be quite numerous when dealing with long text. Thus realistic transition motions is the key to high-quality result. The current SOTA method two-stage transformer [Qin et al. 2022] has two limitations: (1) It employs a full zero attention mask in the first stage, which results in unreliable information propagation when the length of the mask is large. (2) Although the two-stage design shows promising results, their parameters are not shared, which not only causes inefficiency but also prevents the second stage from fully utilizing the information learned in the first stage. We design a Progressive Mask Transformer (Fig.3) to tackle these problems. Inspired by the progressive strategy used in other generation tasks [Ghazvininejad et al. 2019], we propose a coarse-to-fine approach that generates motion progressively, sharing parameters among iterative steps. This approach is both parameter-efficient and can leverage knowledge from a previously well-learned model. Particularly, to ensure reliable information propagation, we design a dynamic attention mask that gradually introduces more information in each iteration. Refer to Appendix A for details.

### 4 EXPERIMENTS

To our best knowledge, no prior work can generate infinitely long motions and trajectories aligned with the given long text. Hence we compare our method with SOTA techniques in three sub-tasks: trajectory following, temporal action composition, and motion blending, to gauge different aspects of our system. Supplementary materials include an overall visualization. The experiments are conducted on the database AMASS [Mahmood et al. 2019] that unifies different datasets. More details are in Appendix B.

### 4.1 Trajectory Following

We compare our system with GAMMA [Zhang and Tang 2022], the current SOTA method for infinite long motion generation, and closely follow their experiment settings. As indicated in Table 2, it shows significant advantages in trajectory following (columns 4-6). GAMMA often takes a long time to reach a nearby goal with sudden speed changes or stops [Zhang and Tang 2022]. In contrast, thanks to the retrieval strategy that takes into account both speed and position, our method can faithfully follow the trajectory and maintain control over speed, which is crucial for generating multicharacter motion and avoiding collisions. Moreover, our retrievalbased system excels in addressing penetration and floating errors (columns 1-3).

200

Real Ours (5.09) Real TEACH (2.56) Con Ours (4.96) Con TEACH (3.07)

| |
|---|

| |
|---|

150

| |
|---|

counts

100

50

0

0 1 2 3 4 5 scores

Figure 4: A user study on temporal action composition (number in parentheses indicates average score). We selected 25 subjects (6 women and 19 men), with ages ranging from 20 to 35, including animators, AI researchers, and gaming enthusiasts. Our method is deemed more realistic and textconsistent.

### 4.2 Temporal Action Composition

Temporal action composition is to generate motions aligned with a series of text descriptions, following the prescribed temporal order. We conduct a user study to compare our method with the current SOTA method TEACH [Athanasiou et al. 2022], considering text consistency is hard to evaluate [Holden et al. 2020]. As shown in Figure 4, the proposed system shows superior performance in both realism and text-consistency scores, with much fewer artifacts like floating and penetration. Besides, deep-learning methods face significant challenges in generating rare motions. Our system can produce great quality in this situation while TEACH fails.

### 4.3 Motion Blending

In the motion blending task, we extensively evaluate models on transitions spanning from 5 to 70 frames. As shown in Table 3, our system shows superior performance compared with previous works. It exhibits great improvement (an average 37% relative improvement) compared with MC-Trans [Duan et al. 2021] (rows 2 and 6) which performs well in LAFAN1. Besides, the proposed method brings a considerable improvement (an average 15% relative improvement) compared with the strong baseline [Qin et al. 2022] (rows 4 and 6). Notably, our single-stage method surpasses the previous SOTA two-stage transformers [Qin et al. 2022] (rows 5 and 6) using nearly half of the parameters, which implies that the masked attention it used in the first stage may not be suitable for long-term motion generation since it cuts off the connection between missing frames. Using the two-stage approach [Qin et al. 2022], our method demonstrates additional advancement (row 7).

### 5 CONCLUSION

In this work, we propose a new task, Story-to-Motion, with the goal of generating human motion and trajectory from a long text. Moreover, we present a pioneering effort, Text-based Motion Matching, that leverages a large language model, motion matching, and neural blending for controllable and realistic motion generation. It surpasses previous SOTA methods in three sub-tasks: trajectory following, temporal action composition, and motion blending.

Since our system is retrieval-based, its diversity is ultimately limited by the size of the motion database. Thus it is promising to combine our method with learning-based methods.

#### Table 1: Motion Blending Benchmark in AMASS, all models are trained with random transition lengths from 5 to 60. The numbers in red and blue indicate the best and the second-best results. Our progressive mask transformer surpasses previous state-of-the-art motion completion methods.

Param Pos / m Rotation Mean Frames 5 15 30 45 60 70 5 15 30 45 60 70 FCN 15.89M 0.68 0.812 1.113 1.524 2.157 2.71 0.585 0.648 0.795 0.932 1.098 1.292 1.1955 MC-Trans [Duan et al. 2021] 13.03M 0.362 0.667 0.993 1.308 1.584 1.872 0.198 0.375 0.578 0.720 0.829 0.921 0.867 Transformer [Vaswani et al. 2017] 10.12M 0.258 0.522 0.865 1.198 1.47 1.703 0.212 0.352 0.542 0.684 0.794 0.885 0.7904 Context Trans [Qin et al. 2022] 10.39M 0.243 0.488 0.829 1.145 1.416 1.637 0.195 0.321 0.509 0.662 0.778 0.863 0.757 Detail Trans [Qin et al. 2022] 20.52M 0.142 0.398 0.762 1.082 1.358 1.576 0.123 0.265 0.465 0.624 0.744 0.828 0.697 Ours (one stage) 10.39M 0.167 0.402 0.728 1.030 1.296 1.506 0.161 0.290 0.474 0.636 0.735 0.819 0.687 Ours (two stages) 20.77M 0.122 0.379 0.714 1.022 1.291 1.508 0.113 0.262 0.455 0.621 0.726 0.817 0.669

#### Table 2: Long motion generation with trajectory and speed constraints. Our method shows stronger trajectory following ability with less physics error.

Physics Error / m Trajectory Error / m Trajectory Wave Circle Square Wave Circle Square

GAMMA 0.050 0.047 0.058 2.306 1.951 3.242 Ours 0.025 0.030 0.020 0.156 0.249 0.151

### REFERENCES

Advanced Computing Center for the Arts and Design. [n.d.]. ACCAD MoCap Dataset. https://accad.osu.edu/research/motion-lab/mocap-system-and-data Chaitanya Ahuja and Louis-Philippe Morency. 2019. Language2pose: Natural language grounded pose forecasting. In 3DV. IEEE, 719–728.

Andreas Aristidou, Ariel Shamir, and Yiorgos Chrysanthou. 2019. Digital Dance Ethnography: Organizing Large Dance Collections. J. Comput. Cult. Herit. 12, 4, Article 29 (Nov. 2019), 27 pages. https://doi.org/10.1145/3344383

Nikos Athanasiou, Mathis Petrovich, Michael J Black, and Gül Varol. 2022. TEACH: Temporal Action Composition for 3D Humans. arXiv:2209.04066 (2022). David Bollo. 2016. Inertialization: High-performance animation transitions in’gears of war’. Proc. of GDC 2018 (2016).

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.

2020. Language models are few-shot learners. NeurIPS 33 (2020), 1877–1901.

Carnegie Mellon University. [n.d.]. CMU MoCap Dataset. http://mocap.cs.cmu.edu Simon Clavet. 2016. Motion matching and the road to next-gen animation. In Proc. of

GDC, Vol. 2016.

Yinglin Duan, Tianyang Shi, Zhengxia Zou, Yenan Lin, Zhehui Qian, Bohan Zhang, and Yi Yuan. 2021. Single-shot motion completion with transformer. arXiv:2103.00776

(2021).

Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. 2019. Maskpredict: Parallel decoding of conditional masked language models. arXiv:1904.09324

(2019).

Saeed Ghorbani, Kimia Mahdaviani, Anne Thaler, Konrad Kording, Douglas James Cook, Gunnar Blohm, and Nikolaus F Troje. 2021. MoVi: A large multi-purpose human motion and video dataset. Plos one 16, 6 (2021), e0253157.

Félix G Harvey, Mike Yurick, Derek Nowrouzezahrai, and Christopher Pal. 2020. Robust motion in-betweening. ACM Transactions on Graphics (TOG) 39, 4 (2020), 60–1. Daniel Holden, Oussama Kanoun, Maksym Perepichka, and Tiberiu Popa. 2020. Learned

motion matching. ACM TOG 39, 4 (2020), 53–1. Daniel Holden, Taku Komura, and Jun Saito. 2017. Phase-functioned neural networks for character control. ACM TOG 36, 4 (2017), 1–13.

Franziska Krebs, Andre Meixner, Isabel Patzer, and Tamim Asfour. 2021. The KIT Bimanual Manipulation Dataset. In IEEE/RAS International Conference on Humanoid Robots (Humanoids). 499–506.

Wenbo Li, Zhe Lin, Kun Zhou, Lu Qi, Yi Wang, and Jiaya Jia. 2022. Mat: Maskaware transformer for large hole image inpainting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10758–10768.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv:1907.11692 (2019).

Eyes JAPAN Co. Ltd. [n.d.]. Eyes Japan MoCap Dataset. http://mocapdata.com

Naureen Mahmood, Nima Ghorbani, Nikolaus F Troje, Gerard Pons-Moll, and Michael J Black. 2019. AMASS: Archive of motion capture as surface shapes. In ICCV. 5442– 5451.

Christian Mandery, Ömer Terlemez, Martin Do, Nikolaus Vahrenkamp, and Tamim Asfour. 2016. Unifying representations and large-scale whole-body motion databases for studying human motion. IEEE Transactions on Robotics 32, 4 (2016), 796–809.

Christian Mandery, Ömer Terlemez, Martin Do, Nikolaus Vahrenkamp, and Tamim Asfour. 2015. The KIT whole-body human motion database. In 2015 International Conference on Advanced Robotics (ICAR). 329–336. https://doi.org/10.1109/ICAR. 2015.7251476

M. Müller, T. Röder, M. Clausen, B. Eberhardt, B. Krüger, and A. Weber. 2007. Documentation Mocap Database HDM05. Technical Report CG-2007-2. Universität Bonn.

Dario Pavllo, Christoph Feichtenhofer, Michael Auli, and David Grangier. 2020. Modeling human motion with quaternion-based neural networks. International Journal of Computer Vision 128 (2020), 855–872.

Abhinanda R Punnakkal, Arjun Chandrasekaran, Nikos Athanasiou, Alejandra QuirosRamirez, and Michael J Black. 2021. BABEL: Bodies, action and behavior with english labels. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 722–731.

Jia Qin, Youyi Zheng, and Kun Zhou. 2022. Motion In-betweening via Two-stage Transformers. ACM TOG 41, 6 (2022), 1–16.

L. Sigal, A. Balan, and M. J. Black. 2010. HumanEva: Synchronized video and motion capture dataset and baseline algorithm for evaluation of articulated human motion. International Journal of Computer Vision 87, 4 (March 2010), 4–27. https://doi.org/ 10.1007/s11263-009-0273-6

Omid Taheri, Nima Ghorbani, Michael J Black, and Dimitrios Tzionas. 2020. GRAB: A dataset of whole-body human grasping of objects. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16. Springer, 581–600.

Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit H Bermano. 2022. Human motion diffusion model. arXiv:2209.14916 (2022).

Nikolaus F. Troje. 2002. Decomposing Biological Motion: A Framework for Analysis and Synthesis of Human Gait Patterns. Journal of Vision 2, 5 (Sept. 2002), 2–2. https://doi.org/10.1167/2.5.2

Matt Trumble, Andrew Gilbert, Charles Malleson, Adrian Hilton, and John Collomosse.

2017. Total Capture: 3D Human Pose Estimation Fusing Video and Inertial Sensors. In 2017 British Machine Vision Conference (BMVC).

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).

Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. 2022. Motiondiffuse: Text-driven human motion generation with diffusion model. arXiv:2208.15001 (2022).

Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, and Ziwei Liu. 2023. ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model. arXiv preprint arXiv:2304.01116 (2023).

Yan Zhang and Siyu Tang. 2022. The wanderings of odysseus in 3D scenes. In CVPR. 20481–20491.

Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. 2019. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5745–5753.

### A DETAILS OF PROGRESSIVE MASK TRANSFORMER

In this section, we introduce the details of the proposed progressive mask transformer. We adopt the similar motion representation, position encoding, and loss functions introduced in two-stage transformers [Qin et al. 2022] to evaluate the proposed attention mechanism and progressive generation strategy.

Motion Representation. We follow the same representation used in [Qin et al. 2022] and adopt the continuous 6D rotation representation introduced by [Zhou et al. 2019]. Each frame is represented as 𝑥𝑡 = {𝑟𝑡,𝑝𝑡,𝑐𝑡,𝑚𝑡}, where 𝑟𝑡 ∈ R𝐽 ×6 is the first two rows or columns of 3×3 rotation matrix and 𝑝𝑡 ∈ R3 is the world position of the hip joint. 𝑐𝑡 ∈ {0, 1}4 represents whether the feet have contact with the ground and 𝑚𝑡 ∈ {0, 1} represents whether this frame is masked. Finally the motion clip 𝑥 ∈ R𝑇×𝐷, where 𝐷 = 𝐽 × 6 + 8 and 𝑇 is the number of frames.

Progressive Generation. Rather than employing a two-stage approach with separate models for coarse and fine generation [Qin et al. 2022], inspired by the progressive strategy in other generation tasks [Ghazvininejad et al. 2019], we utilize a single network that refines its output iteratively. The proposed progressive generation strategy runs for a predetermined number 𝑟 (e.g., 3) of iterations to gradually recover the masked motion. It takes a masked motion 𝑚𝑥 ∈ R𝑇×𝐷 as input and generates a motion 𝑚𝑥𝑖 ∈ R𝑇×𝐷, where 𝑖 is the iteration number. The output of each iteration will be the input of the next iteration, and the final motion is obtained after 𝑟 iterations. Using the progressive generation strategy, we find that the network first generates coarse results in the first iteration, which typically contain artifacts like foot sliding and jittering, and then refine it in the following iterations. This approach is both parameter-efficient and can leverage knowledge from a previously well-learned model.

Dynamic Attention Mask. A notable observation of previous works [Qin et al. 2022] [Li et al. 2022] is that for motions dominated by mission frames, the vanilla transformer’s default attention strategy may undermine the valid information, which causes limited generalization ability. And thus the masked multi-head self-attention is used to mask out missing frames. However, this strategy may not be optimal for motion blending since it cuts off the connection between most of the tokens. Inspired by the technique in image in-painting [Li et al. 2022], we use the updated attention mask instead. We proposed a simple yet efficient masking strategy that first masks all missing frames similar to [Qin et al. 2022] but later gradually reduces the number of masked frames. This strategy is naturally combined with the aforementioned progressive generation method. Different attention masks will be used in different iterations of the generation process. At the first iteration, all missing frames are masked. In the following iteration, the number of masked frames will linearly decrease.

Position Encoding. The position encoding contains two parts: the keyframe position encoding and the step embedding. The keyframe positional encoding [Qin et al. 2022] uses an MLP with a single hidden layer of 512 units to encode the relative position from the current frame to keyframes, which is added to the motion embedding and fed to the transformer backbone. Incorporating the progressive generation strategy, we introduce the concept of learned

step embedding, which assigns a distinct vector to each iteration step, allowing the model to differentiate between different iteration time points. The step embedding is similar to the relative positional encoding proposed in the two-stage transformers [Qin et al. 2022], which is a learnable lookup table containing 2 ×𝑇 − 1 embeddings, serving as shared key tokens for calculating attention in all transformer layers.

Loss Functions. In order to fairly compare different methods, for the experiments on the AMASS dataset we train all models using the same loss functions as follows.

State Loss. The state of motion contains local rotation in 6D rotation space, root position, and foot contact. The state loss is a weighted sum of the reconstruction loss of these three components:

𝐿𝑠𝑡𝑎𝑡𝑒 = 𝜆𝑐𝐿1(𝑐,𝑐ˆ) + 𝜆𝑟𝐿ˆ1(𝑟,𝑟ˆ) + 𝜆𝑝𝐿ˆ1(𝑝,𝑝ˆ), (1)

where 𝐿1 is the mean absolute error, and 𝐿ˆ1 stands for smooth L1 loss which uses a squared term if the absolute element-wise error falls below a threshold and an L1 term otherwise.

Joint Position Loss. Considering equally distributed joint orientation errors will lead to growing joint position errors along the kinematic chains, the reconstruction of joint positions is commonly used in motion generation tasks [Pavllo et al.2020]. Besides, the smoothness term of joint positions is also included:

𝐿𝑝𝑜𝑠 = 𝐿ˆ1(𝑔,𝑔ˆ) + 𝜆𝑠||𝑔ˆ′||1, (2)

where 𝑔 stands for the global joint position, which is calculated by root position and local joint rotations through forward kinematics, and 𝑔ˆ′ is the speed of joint positions.

Foot Sliding Loss. This term is calculated based on the predicted foot contact and the generated joint position speed. When foot contact happens, the speed of foot joints should be zero:

𝐿𝑓 𝑜𝑜𝑡 = ||𝑐ˆ𝑔ˆ′𝑓 𝑜𝑜𝑡 ||1, (3)

### B DETAILS OF EXPERIMENTS

Our experiments are conducted on the AMASS database, known for its popularity in motion generation. We use different scales of data in different sub-tasks for fair comparisons with previous methods. The metrics include L2P, L2Q, physics error, and trajectory error. The L2P and L2Q measure the average L2 distance of the global joint position and rotation (in quaternions) per joint per frame. The physics error is the sum of the foot floating distance and the foot-ground penetration distance. The trajectory error measures the average error between the desired path and the hip position of the character per frame.

Trajectory Following. For the trajectory following task, we carefully follow the experiment settings of the GAMMA [Zhang and Tang 2022] and use the same randomly chosen input seed poses and data in our experiments. The training data contains CMU [Carnegie Mellon University [n.d.]], MPI HDM05 [Müller et al. 2007], BMLmovi [Ghorbani et al. 2021], KIT [Mandery et al. 2016], Eyes Japan [Ltd. [n.d.]]. The evaluation data includes HumanEva [Sigal et al. 2010], and ACCAD [Advanced Computing Center for the Arts and Design [n.d.]]. In practical applications, it’s not only necessary to achieve high trajectory matching but also to reach the endpoint within a specified time. Therefore, we propose a metric that includes velocity: we defined trajectory as position-time pairs and

#### Table 3: Motion Blending Ablation, all models are trained with random transition lengths from 5 to 60. The proposed Dynamic Attention Mask (DAM) and Progressive Generation (PG) offer promising enhancements to the strong baseline method [Qin et al. 2022].

Param Pos / m Rotation Mean Frames 5 15 30 45 60 70 5 15 30 45 60 70

Detail Trans [Qin et al. 2022] 20.52M 0.142 0.398 0.762 1.082 1.358 1.576 0.123 0.265 0.465 0.624 0.744 0.828 0.697 ours w/o PG and DAM [Qin et al. 2022] 10.39M 0.243 0.488 0.829 1.145 1.416 1.637 0.195 0.321 0.509 0.662 0.778 0.863 0.757 ours w/o PG 10.39M 0.224 0.459 0.806 1.129 1.396 1.630 0.177 0.305 0.499 0.654 0.762 0.846 0.741 ours 10.39M 0.167 0.402 0.728 1.030 1.296 1.506 0.161 0.290 0.474 0.636 0.735 0.819 0.687 ours (two stages) 20.77M 0.122 0.379 0.714 1.022 1.291 1.508 0.113 0.262 0.455 0.621 0.726 0.817 0.669

used the point-wise L2 distance error. We mainly focus on this metric although FID or foot sliding will also be helpful, considering our method is retrieval-based and the motion quality is rather high. We follow previous work [Holden et al. 2017] and use different trajectories for evaluation. For the wave trajectory, we define it as a sine function about time 𝑥(𝑡) = 2𝑠𝑖𝑛(𝑡). For the square trajectory, we set the side length as 5. For the circle trajectory, the diameter is set as 5. We randomly select 50 seed poses from HumanEva and ACCAD respectively. Both methods generate a 20-22.5-second motion based on each seed pose.

Temporal Action Composition. For temporal action composition, the label we use is coarse and noisy but simple to collect. Note that TEACH uses the BABEL dataset [Punnakkal et al. 2021] which is a subset of AMASS that contains detailed text descriptions, in total about 8000 motion names and 43 hours of motion data. We select all motions in AMASS that have meaningful file names and are present in BABEL for a fair comparison with TEACH. In total, we collate 60 motion names such as “walk fast”, “turn left”, “wave” and “air guitar”. We ask the language model (ChatGPT [Brown et al. 2020]) to choose 5 motions from these motion names and create a short story about a human doing some activities. The evaluated methods are then asked to generate a 12.5-second motion based on these 5 motion names.

Motion Blending. Motion blending can be viewed as motion completion [Duan et al. 2021], namely, a number of the motion frames in the center temporal position will be masked, and the goal is to recover the original motion. We extensively evaluate models on transitions spanning from 5 to 70 frames, which is challenging because AMASS contains motions consisting of sporadic, random short movements that are extremely difficult to predict beyond short time horizons [Harvey et al. 2020]. In addition to the data employed for the trajectory following task, GRAB [Taheri et al.

- 2020], DanceDB [Aristidou et al. 2019], BMLrub [Troje 2002], and WEIZMANN [Mandery et al. 2015] are added to training data, in total 402 subjects, 15818 motions, and 56.3 hours motion data. For evaluation, we utilize 107 minutes of motion data performed by 32 objects, encompassing 666 motions, which includes contributions from various sources: HumanEva, ACCAD, EKUT [Krebs et al.
- 2021], and TotalCapture [Trumble et al. 2017]. Ablation Study. In Table 3, we also evaluate the impact of different

components on the final performance. The Context Transformer (row 2) [Qin et al. 2022] is our baseline. Row 3 is our method without the progressive generation strategy, which simply replaces the Full Attention Mask used in the Context Transformer as the

proposed Dynamic Attention Mask. The comparison of row 2 and row 3 shows the efficiency of the Dynamic Attention Mask. The full method (row 4) utilizes the proposed progressive generation strategy, which brings a further improvement compared with using Dynamic Attention Mask only (row 3). In a manner akin to the two-stage approach [Qin et al. 2022], our method demonstrates additional advancements, leading to a notable enhancement (rows 4 and 5). Our method also surpasses the state-of-the-art (row 1).

Qualitative Comparison. In this section, we show a qualitative comparison of the evaluation dataset. We compare our method (shown in blue) with the Context Transformer (shown in grey and upper) and Vanilla Transformer (shown in grey) [Vaswani et al. 2017] together with the ground truth (shown in green). As shown in Figure 5, the Vanilla Transformer may generate motions with inconsistent speed. Notably, our method can generate better highfrequency details while maintaining a consistent speed. When generating long motion, foot sliding is a longstanding problem. The Context Transformer and Vanilla Transformer cause noticeable foot sliding (see the leg foot in the red box). The proposed method generates motion with high-frequency details (more footsteps which is similar to the ground truth) while alleviating the foot sliding.

[Figure 10]

#### Figure 5: Smoothness and detail comparison. For complex motion, the proposed method generates smooth motion while maintaining high-quality details. Note that in this image the position of the actor is changed (using the same way) for better visibility.

[Figure 11]

#### Figure 6: Foot sliding comparison. Our method generates motions with more high-frequency details while alleviating foot sliding artifacts.

