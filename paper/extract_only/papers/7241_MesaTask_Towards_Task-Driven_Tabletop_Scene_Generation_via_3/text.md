[Figure 1]

[Figure 2]

2025-9-29

### MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning

Jinkun Hao1,*, Naifu Liang2,*, Zhen Luo3,4,*, Xudong Xu2,‡, Weipeng Zhong1, Ran Yi1, Yichen Jin5, Zhaoyang Lyu2, Feng Zheng4, Lizhuang Ma1,† and Jiangmiao Pang2 1Shanghai Jiao Tong University, 2Shanghai AI Laboratory, 3SII, 4Southern University of Science and Technology, 5Peking University, *Equal contributions

## arXiv:2509.22281v1[cs.CV]26Sep2025

The ability of robots to interpret human instructions and execute manipulation tasks necessitates the availability of task-relevant tabletop scenes for training. However, traditional methods for creating these scenes rely on timeconsuming manual layout design or purely randomized layouts, which are limited in terms of plausibility or alignment with the tasks. In this paper, we formulate a novel task, namely task-oriented tabletop scene generation, which poses significant challenges due to the substantial gap between high-level task instructions and the tabletop scenes. To support research on such a challenging task, we introduce MesaTask-10K, a large-scale dataset comprising approximately 10,700 synthetic tabletop scenes with manually crafted layouts that ensure realistic layouts and intricate inter-object relations. To bridge the gap between tasks and scenes, we propose a Spatial Reasoning Chain that decomposes the generation process into object inference, spatial interrelation reasoning, and scene graph construction for the final 3D layout. We present MesaTask, an LLM-based framework that utilizes this reasoning chain and is further enhanced with DPO algorithms to generate physically plausible tabletop scenes that align well with given task descriptions. Exhaustive experiments demonstrate the superior performance of MesaTask compared to baselines in generating task-conforming tabletop scenes with realistic layouts.

Code | Model & Data | Homepage

#### 1. Introduction

A fundamental challenge in robotic manipulation is enabling robots to accurately interpret human instructions and successfully execute complex tasks accordingly. The conventional pipeline for achieving this involves task definition, simulatable tabletop scene construction, and policy training. However, traditional scene construction methods, which rely on manual design or purely randomized layouts, are limited by their labor-intensive nature and the resulting constraints on diversity and plausibility, ultimately hindering the generalization of learned policies. Therefore, automatic taskoriented tabletop scene generation emerges as a promising approach for effectively bridging the gap between task descriptions and scenes. Crucially, tabletop scene generation must satisfy three key requirements: covering task variables, enabling scene interactivity, and ensuring realistic layouts, thereby facilitating the learning of robust policies.

Existing scene generation methods Dai et al. (2024); Huang et al. (2024); Wang et al. (2024) often start from a single scene image and attempt to recover the corresponding tabletop scene through object retrieval and layout optimization. Unfortunately, their ability to understand under-specified task instructions still requires empirical corroboration. Other approaches Çelen et al. (2024); Yang et al. (2024e,f) leverage powerful language models (LLMs) to interpret task prompts and then synthesize tabletop scenes in a zero-shot manner. Nevertheless, these methods are hindered by inherent limitations, no matter the inevitable occlusion in scene images or the lack of fine-tuning on a scene dataset, which significantly impede the modeling of realistic table layouts and complex inter-object relations, such as stacking and containment, within the scene. As a result, task-oriented

Corresponding author: Lizhuang Ma, Project lead: Xudong Xu © 2025 Shanghai Artificial Intelligence Laboratory. All rights reserved.

[Figure 3]

###### Task Instructions

[Figure 4]

Bathroom Vanity

Articulated Objects

Put all the food on the table into one bowl， stack the bowl.

Kitchen Counter

[Figure 5]

d

[Figure 6]

c

Kitchen Counter

Mesa Task

d

b

b

Prepare dishes use two pots

c

User

[Figure 7]

[Figure 8]

a e

[Figure 9]

Complex Inter-Relation

Dinning Table

Pick fruits out of bowl and cut them.

[Figure 10]

Task Alignment

“Stacking” “Containment”

[Figure 11]

[Figure 12]

Use two towels to clean the vanity

e a

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

11, 708 Tabletop Scenes with Manual-Craft Layout

# MesaTask-10K

[Figure 17]

Over 12, 000 3D Assets 6 Common Indoor Table Types

- Figure 1. We present MesaTask, a novel LLM-based framework for generating task-oriented 3D tabletop scenes directly from high-level human instructions, featuring realistic layouts, articulated objects, and complex inter-object relations like stacking and containment. To support this task, we introduce a large-scale dataset of tabletop scenes, MesaTask-10K, comprising over 12, 000 3D assets, 11, 708 tabletop scenes with manually crafted layouts covering 6 common indoor table types.

tabletop scene generation remains a challenging problem due to the scarcity of datasets and the substantial gap between task instructions and scene layouts.

To tackle these challenges, we collect a first-of-its-kind dataset of synthetic tabletop scenes with manually crafted layouts, dubbed MesaTask-10K. As shown in Figure 1, our dataset comprises approximately 10, 700 diverse tabletop scenes, spanning six common indoor table categories, including office tables, dining tables, kitchen counters, and more. The 3D objects in MesaTask-10K originate from a large asset library containing over 12, 000 rigid and articulated 3D assets, each with detailed semantic information, such as object category, description, and materials, and featuring a comprehensive taxonomy of over 200 object classes on the tables. As claimed in Wang et al. (2024), pretrained

- 2D image generative models better capture scene and object configurations both at the scene level and in fine-grained inter-object relations. Inspired by this, our dataset is built upon diverse tabletop scene images with diversity and realistic layouts, generated by a large text-to-image model Labs

(2024) pretrained on massive internet data. To obtain a coarse layout from the scene image, we estimate the depth Yang et al. (2024c) of each image, extract the instance point cloud, and acquire the 3D bounding box of objects. We then leverage the object descriptions labeled by VLM Achiam

- et al. (2023) to retrieve suitable 3D assets from the library and construct an initial replica of the tabletop scenes. Subsequently, human annotators meticulously refine these 3D layouts, adjusting the object size and positions as per the image prompt, addressing inaccuracies from occlusion and ensuring complex inter-object relations. Ultimately, all the scenes are put into a physical simulator, IsaacSim NVIDIA (2024), to prevent object collisions.

Confronted with the significant gap between tasks and scenes, we propose a novel paradigm referred to as Spatial Reasoning Chain, decomposing task-to-tabletop scene generation into a structured chain of thought (CoT). Given a high-level task description, this chain of thought begins with the inference of requisite objects, accompanied by their semantic attributes and spatial interrelations,

based on which a complete scene graph is formed, and finally leads to a concrete 3D layout of objects on the table. To establish trainable spatial reasoning chains with our dataset, we design a set of delicate rules to extract the object attributes and inter-object relations, thus forming a scene graph for each tabletop scene. Subsequently, we leverage a multimodal large language model, taking scene graphs and rendered scene images as input, to generate corresponding task information and detailed spatial reasoning descriptions for training.

Thanks to our structured reasoning chains, it’s convenient to empower LLM with 3D spatial reasoning and scene generation capability. In this paper, we propose MesaTask, a novel LLM-based framework for task-oriented tabletop scene generation. Specifically, we initially employ the supervised fine-tuning (SFT) strategy on our constructed reasoning data to inject the LLM with 3D spatial reasoning capabilities. However, MesaTask occasionally generates unsatisfactory tabletop scenes with minor object collisions and misalignment with the given task. To circumvent this hurdle, we devise paired training data and leverage a conventional RL algorithm, namely Direct Preference Optimization (DPO), to boost our MesaTask model, thereby ensuring that the generated scenes are devoid of object collisions and exhibit improved conformity with the provided task descriptions.

For a more comprehensive performance assessment, we leverage powerful VLMs to evaluate the rendered scene images from multiple perspectives, including task-scene alignment, physical viability, scene layout plausibility, etc. Through extensive experiments, our MesaTask framework is capable of generating physically plausible tabletop scenes with realistic layouts, outperforming baseline methods in terms of FID, VLM-based metrics, and the user study. In particular, our generated tabletop scenes strictly conform to given task instructions and exhibit rich inter-object relations, such as stacking or containing. In summary, our contributions are threefold:

- • We pioneer the formulation of Task-to-Scene generation task, which aims to generate physically plausible tabletop scenes directly from high-level task descriptions.
- • We introduce MesaTask-10k, a large-scale tabletop scene dataset with human-crafted realistic layouts, characterized by rich inter-object relations and a tremendous amount of synthetic 3D object assets.
- • Along with the delicate design of spatial reasoning chains, we propose MesaTask, an LLM-based framework endowed with the capability of 3D spatial reasoning and tabletop scene generation, achieving superior performance across various evaluation criteria.

#### 2. Related Work

Tabletop Scenes Dataset. Recent works have explored various approaches to constructing tabletop scene datasets. LVDiffusor Zeng et al. (2024) uses large VLMs to generate semantically plausible tabletop scene images, which are constrained in the 2D domain and lack 3D spatial information. StructFormer Liu et al. (2022c) and StructDiffusion Liu et al. (2022b) collect 3D object rearrangement data under language guidance using abstract geometric relations, which allows testing structural reasoning but lacks semantic richness and realism for real-world deployment. SetItUp Xu et al. (2024) presents manually designed functional scenes reflecting real-world usage like dining or working, but the object sets are fixed and lack diversity. TO-Scene Xu et al. (2022) provides a large-scale and richly annotated 3D tabletop dataset built by professional designers. However, its top-down, click-to-place annotation paradigm restricts the inclusion of intricate spatial relationships such as nesting and stacking. Despite these efforts, existing datasets frequently exhibit limitations in terms of data scale, layout, or realism. Accordingly, we introduce a large-scale tabletop scene dataset with diverse real-world 3D objects, realistic layouts, and rich 3D spatial relationships.

###### Scene Reconstruction from A Single Image. It’s a long-standing problem to reconstruct 3D

scenes from a single image. A line of previous methods Chen et al. (2024a); Liu et al. (2022a); Nie et al. (2020); Zhang et al. (2021) attempt to reconstruct the scenes by compressing the input images with an encoder and mapping the image features back to the 3D space via a decoder. Based on advancements in 3D object generation Li et al. (2024); Zhang et al. (2024), MIDI Huang et al. (2024) is capable of generating scenes with diverse 3D objects but struggles to generate complex inter-object relationships. Some other methods Chen et al. (2024b); Dai et al. (2024); Han et al. (2024); Ling et al. (2025); Wang

- et al. (2024); Yao et al. (2025) typically entail a multi-stage process, comprising object segmentation, occlusion completion, image-to-3D generation, and layout optimization. This protracted workflow inevitably gives rise to error accumulation, particularly in regions with severe occlusions. Moreover, these methods fall short of generating scenes from underspecified task descriptions. In contrast, our LLM-based framework is inherently designed to fit the task-oriented tabletop scene generation.

LLM-Based Scene Generation. Inspired by the prosperity of Large Language Models (LLMs), many researchers have exploited the capabilities of powerful LLMs to perform 3D scene generation. For instance, LayoutGPT Feng et al. (2023) explores direct 3D layout generation through in-context learning. Furthermore, some methods Çelen et al. (2024); Fu et al. (2024); Yang et al. (2024f) are built upon commercial LLMs and use multi-stage prompting to achieve open-vocabulary and datasetfree generation in a zero-shot manner. However, these methods encounter substantial challenges in modeling complex inter-object relations. LLPlace Yang et al. (2024e) attempts to fine-tune the LLM via supervised fine-tuning (SFT) on meticulously crafted 3D scene datasets, albeit without a specific focus on tabletop scenes. In contrast, we also leverage an LLM-based framework for tabletop scene generation, but our approach involves training on a large-scale scene dataset with manually crafted layouts, thereby empowering our model with superior capabilities for generating realistic layouts and intricate inter-object relationships.

#### 3. MesaTask-10K Dataset

Inspired by Architect Wang et al. (2024), we build the MesaTask-10K dataset upon diverse tabletop scene images generated by a pretrained text-to-image model Labs (2024), ensuring realistic scene layouts and complex inter-object relationships.

Tabletop Scene Image Generation. To better facilitate manipulation tasks, we intend to synthesize diverse tabletop scene images of six common indoor table types in our daily life: office table, dining table, kitchen counter, coffee table, bathroom vanity, and dressing table. As illustrated in Figure 2, the pretrained LLM Achiam et al. (2023) is guided to output an object list on the table and their spatial relations, respectively, which are subsequently combined to form final scene descriptions. Conditioned on these scene descriptions, FLUX Labs (2024), a cutting-edge text-to-image model, is utilized to produce a diverse range of reference scene images w.r.t. six distinct table categories.

Coarse Tabletop Scene Construction. To create 3D replicas of scene images, we first collect a highquality 3D asset library through meticulous asset curation from two datasets, namely Objaverse Deitke et al. (2022), and PartNet-Mobility Xiang et al. (2020). It’s noteworthy that our library consists of over 12, 000 rigid and interactive objects along with rich object semantic information, including object category, text descriptions, and materials. As shown in Figure 2, Grounded-SAM Achiam et al.

- (2023) is employed to identify all the object instances in the scene image, and a multimodal LLM like GPT-4o is responsible for providing the corresponding semantic information for subsequent

- 3D object retrieval. During the object retrieval process, we specifically rely on textual descriptions of objects rather than their visual appearance, considering severe occlusions in the tabletop scene images. Furthermore, we utilize Depth Anything v2 Yang et al. (2024b) to construct the point cloud of tabletop scenes, and leverage instance masks to obtain 3D bounding boxes for each object within

Tabletop Scene Image Generation

Coarse Tabletop Scene Construction

|[Figure 18]<br><br>Reference Image|
|---|

|[Figure 19]<br><br>Point Cloud|
|---|

Depth Anything V2

[Figure 20]

3D Asset Library

[Figure 21]

“Generate some diverse {Kitchen Counter} descriptions.

[Figure 22]

[Figure 23]

“a white bowl”

Grounded SAM

[Figure 24]

[Figure 25]

- 1. Infer common object
- 2. Imagine spatial relation

Vision Language Model

“a half lemon”

Object retrieval

…

Generate Object relation: Cutting boards, knife, and spoon are on the left. Nearby is a white bowl and a glass jar. In the middle, there’s a wooden plate with chopsticks. On the right, there are a white pot, a bowl of yellow lemons.

Infer Object List： Lemon Knife Bowl Spoon Cookingpot Cutting board

[Figure 26]

[Figure 27]

|[Figure 28]<br><br>Final Scene|
|---|

|[Figure 29]<br><br>IsometricPlacement|
|---|

[Figure 30]

Physical Simulation

Manual Craft

Description = “{InferObjectList},{Objectrelation}”

[Figure 31]

[Figure 32]

[Figure 33]

2000+ Hours

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Wrong Size Collision

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Text-to-Image Model

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Redundant Objects

Generated Images

Human-Assisted Layout Refinement

Dataset Statistics

- Figure 2. The dataset construction pipeline. First, an LLM is used to generate diverse tabletop scene descriptions, including relevant object lists and spatial relations. Conditioned on the scene description, a text-to-image model synthesizes reference images, from which coarse 3D layouts are built using depth estimation, object detection, and 3D asset retrieval. These layouts are refined through human annotations and physical simulation to ensure spatial plausibility.

the scene, thereby yielding a coarse 3D layout of the tabletop scene.

Human-Assisted Layout Refinement. Owing to the intricate inter-object relationships and severe occlusions in the reference images, the obtained coarse scene layouts inevitably contain various flaws, including inaccuracies in object scale, redundant object instances, and object collisions or floating, as shown in Figure 2. To the best of our knowledge, these awkward issues can only be effectively addressed with human assistance. Therefore, 20 expertize annotators undertake a manual layout refinement in Blender, wherein they adjust the object size and positions, as well as eliminate redundant instances, following the reference images.

During annotation, annotators are provided with the coarse 3D scene in GLB format, which includes a Unitree H1 robot model with an absolute height of 1.7m to facilitate the construction of metric-scale 3D scenes, along with reference images and all object snapshots from the images. They will adjust each object’s relative size and position with reference to the given tabletop scene images, calibrate the overall scene scale using the H1 model, and rotate objects to match their orientations in the images. On average, annotators spend 10 to 20 minutes on each tabletop scene. Subsequently, we put all tabletop scenes into the physical simulator to prevent object collisions, manually exclude unsatisfactory scenes, and finally create our dataset.

Dataset Statistics. MesaTask-10K is a large-scale dataset with approximately 10, 700 diverse tabletop scenes spanning six common indoor table categories. Meanwhile, our curated 3D asset library contains a vast collection of over 12, 000 diverse objects, covering a broad spectrum of more than 200 object classes that are typically encountered on tables. In particular, the 100 object categories that occur the most frequently within this library are visually represented in Figure 2. Moreover, there are roughly 15 objects per tabletop scene on average, and the distribution of the object number is also visualized in Figure 2. We believe that our MesaTask-10K dataset possesses substantial potential to drive research advancements in the realm of task-oriented tabletop scene generation.

Task instructions: “Organize fruit from bowl

###### Spatial Reasoning Chain

###### DPO Data Construction

into empty bowl and

###### place tray near lamp.” Object Completion

###### Interrelation inference

###### Scene Graph

User

[Figure 48]

Edge Damage

[Figure 49]

Desk lamp … left side … empty bowl … right side… open space … middle …

2 kettles, 1 desk lamp, 1

(tray_0, is at, left center)

###### Task info:

tray, 1 bowl with fruit, 1 empty bowl, 1 potted plant

(fruit_0, in, bowl_1) (tray_0, face to, front_right)

[Figure 50]

[Figure 51]

Environment: A tabletop with a desk lamp,

right of

[Figure 52]

right of

above

5 fruits

…

[Figure 53]

left of

two kettle, a tray holding a

potted plant and a bowl of fruit, and an empty bowl.

[Figure 54]

###### Object retrieval

Final Result Layout Perturbation

[Figure 55]

Sub-Goal: "Pick all fruit out of the bowl on the tray",

"Place all fruit into the empty

[Figure 56]

[Figure 57]

bowl on the right", "Pick up the tray with plant

left of right of

[Figure 58]

right of

[Figure 59]

[Figure 60]

and empty bowl",

above left of

[Figure 61]

Task Object Removal

left of

"Place the tray near the desk lamp on the left side” Task related objects: ["Fruit", "Bowl", "Tray", "Lamp”]

[Figure 62]

[Figure 63]

in

[Figure 64]

[Figure 65]

left of

[Figure 66]

Scene Layout

Scene Graph Task

Task Info

[ {object: tray_0 size: [10.4, 14.6, 3.2]

Instruction

[Figure 67]

position: [36.8, 13.0, 1.5]

Scene

Rendered

rotation: 30.0 description: "wooden tray…”},

Complete Object Lists

Complete

Description

Image

MesaTask-10K

Interrelations

- Figure 3. Overview of our MesaTask Framework. 1) Task-to-Scene Generation (upper-left). Given a task instruction, we extract detailed task information including environment, sub-goals, and task-relevant objects. A structured spatial reasoning chain performs object list completion, interrelation inference, and scene graph construction, which guides the generation of 3D layouts. Final scenes are obtained via 3D asset retrieval. 2) Reasoning Data Construction (bottom). Based on scene graphs and descriptions of our MesaTask-10K dataset, A multimodal LLM is leveraged to produce task instructions, detailed task information, and complete object lists and interrelations. 3) DPO Data Construction (upper right). To enable DPO training, we generate negative examples by randomly perturbing object positions or relations and removing key objects from normal layouts.
- 4. Method

In this section, we present our novel LLM-based framework MesaTask for generating realistic 3D tabletop scenes from manipulation task descriptions. The crux of our approach lies in endowing an LLM with the capability of 3D spatial reasoning, enabling it to infer the complex spatial arrangements necessary to fulfill the requirements of a given task. We formalize the problem in Section 4.1 and describe the system outlined in Figure 3. To empower the LLM with 3D spatial reasoning, we propose a novel spatial reasoning chain in Section 4.2 and introduce our model’s training via SFT and DPO algorithms in Section 4.3.

###### 4.1. Problem Formulation

Task-oriented tabletop scene generation aims at generating suitable tabletop scenes S from high-level manipulation task instructions T. Following prior works Tang et al. (2024); Yang et al. (2024d,e), a tabletop scene S is a composition of 𝑁 3D objects arranged in a specific 3D layout L = {l1, l2, ..., l𝑁}. The layout of each 3D object l𝑖 = [p𝑖, s𝑖, 𝜽𝑖, t𝑖] is defined by its location p ∈ R3, axis-aligned 3D bounding box size s ∈ R3, rotation angle around the vertical axis 𝜽 ∈ R, and its textual description t detailing the category, shape, and appearance. Given a task instruction T, we will leverage a pretrained LLM model to generate more detailed task information, including the table environment description E, a sequence of decomposed goals G for this given task, and a set of task-relevant objects O. Based on

- them, the scene generation model M is responsible for generating a corresponding 3D scene layout L: L = M (E, G, O) , [E, G, O] = LLM(T). (1)

Based on the object descriptions in the layout L, appropriate 3D assets will be retrieved from the 3D asset database to form a complete tabletop scene S. It’s noteworthy that the generated tabletop scenes will contain all the 3D objects recommended in O and typically include many other objects to ensure realistic layouts.

###### 4.2. Spatial Reasoning Chain

While task instructions are typically conveyed through natural language expressions, 3D scene layouts are inherently represented with structured spatial configurations. Considering the large gap between tasks and tabletop scenes, we propose the spatial reasoning chain to decompose the challenging task-to-scene generation problem into a structured chain of thought (CoT), significantly easing the training and inference of LLM-based models like ours.

Task-to-Scene Generation via Spatial Reasoning Chain. The spatial reasoning chain encompasses three pivotal steps, namely object list completion, interrelationship inference, and scene graph construction, which function as an effective bridge between input tasks and desirable 3D layouts. In the object list completion stage, the generation model M is motivated to infer a complete list of 3D objects V given the aforementioned task-relevant objects O. Then, the model M will generate inter-object relations E expressed with text descriptions, conditioned on the given task instructions and typical object co-occurrence patterns. With graph nodes V and graph edges E, the scene graph G(V, E) can be represented as:

###### [V, E] = M (E, G, O) , L = M G(V, E) (2)

Notably, to better guide the generation of 3D layouts, we further enrich the graph nodes by incorporating objects’ coarse positions and orientations expressed in natural language. Specifically, the orientation is discretized into eight categories, namely front, back, left, right, left-front, left-back, right-front, and right-back, with a 45-degree quantization. The coarse positions correspond to a 3 × 3 grid of the table, including center, front, back, left-center, right-center, left-front, right-front, left-back, right-back. Therefore, given a task description, our spatial reasoning chain will prompt the model to sequentially reason about the scene composition, the spatial interrelationship, the scene graph, and ultimately, the 3D scene layout.

Reasoning Data Construction for Model Training. To guide the model’s reasoning process along our designed spatial reasoning chain, we construct massive reasoning data for training by using our collected tabletop scene dataset, MesaTask-10K. For a certain scene S inside, we first assign coarse positions and orientations for 3D objects on the table following the quantization rules above, infer the inter-object relations based on the 3D layouts, and finally obtain a complete scene graph GS. To compensate for the spatial relations missing in the scene graph, we also utilize a multimodal LLM (MLLM) like GPT-4o Achiam et al. (2023) to output a detailed scene description D based on a high-quality rendering image I of the tabletop scene and the scene graph. Given the scene graph GS and scene descriptions D, the multimodal LLM is prompted to generate a complete object list V and inter-object relations E, as well as the corresponding task instructions T, in particular including aforementioned detailed task informations z, i.e., [E, G, O] following:

###### D = MLLM(I, GS), [T, E, G, O, V, E] = MLLM D, GS . (3)

###### 4.3. LLM-based Framework for Tabletop Scene Generation

Our proposed MesaTask framework is a novel paradigm for tabletop scene generation, comprising an LLM-based model M for 3D layout generation and a post-processing module responsible for 3D asset retrieval. Given the constructed spatial reasoning data, we perform supervised fine-tuning (SFT) on the MesaTask model, thereby empowering it with the capability to reason about spatial relationships and generate structured 3D layouts from high-level task instructions. Despite the SFT on our high-quality data, the MesaTask model still generates suboptimal 3D layouts, including minor object collisions, unreasonable inter-object relationships misaligned with the task, and the omission of crucial task-relevant objects. Accordingly, we employ the Direct Preference Optimization (DPO) algorithmRafailov et al. (2023) to tackle such issues.

DPO Data Construction and Training. To facilitate the DPO training, we construct massive training pairs with positive and negative 3D layouts. Here, positive data stands for the high-quality 3D layouts sourced from our dataset, MesaTask-10K, while the negative data is generated by intentionally corrupting the positive layouts in three distinct ways, each corresponding to a specific shortcoming of our MesaTask model after the SFT. For a 3D layout L from our dataset L+, we randomly select a subset of objects and perturb their positions, rotations, and sizes to deliberately create object collisions,

resulting in the negative layout Lcol− reflecting the collisions. Then, some normal inter-object relations are damaged by altering their relation types, leading to a negative sample Lrel− contradicting the task instruction. Finally, we manually remove one or more critical objects to create a negative layout Lobj− that neglects task-relevant objects. Therefore, we can obtain a negative dataset L− with three distinct layout corruptions. Along with the corresponding task instructions T, we represent the whole paired dataset D for the DPO training with:

obj)} for L ∈ L (4) With the constructed dataset D, we optimize our MesaTask model via the DPO objective following

###### D = (L+, L−, T), L− = {(L−

col, L−

rel, L−

𝜋𝜃 (L− | T) 𝜋ref (L− | T)

𝜋𝜃 (L+ | T) 𝜋ref (L+ | T)

, (5)

− 𝛽 log

max

𝔼(L+,L−,T)∈D log𝜎 𝛽 log

𝜋𝜃

where 𝜋𝜃 is the policy of the fine-tuned LLM, 𝜎(·) represents the sigmoid function for preference scoring, and 𝛽 is a temperature parameter that controls the sharpness of the preference margin between positive and negative layouts. With the DPO training, our MesaTask model seeks to acquire a policy 𝜋0 that favors normal 3D layouts L+, thereby alleviating three limitations observed in the model after the SFT and consequently enhancing the overall quality of generated tabletop scenes.

#### 5. Experiment

###### 5.1. Experiment setup

Dataset. We build our training data based on the training split of our MesaTask-10k dataset, which contains 10, 000 tabletop scenes. For each scene, we generate five task instructions following the reasoning data creation process above, resulting in a total of 50, 000 task-scene pairs for the supervised fine-tuning. During the stage of DPO training, we construct the paired dataset using 5, 000 previously unseen scenes, where each normal layout sample corresponds to two disrupted layouts on average, thereby yielding a total of 10, 000 positive-negative layout pairs for the DPO training.

Implementation details. We adopt Qwen3-8bYang et al. (2024a) as the base LLM for both supervised fine-tuning (SFT) and direct preference optimization (DPO). We perform full-parameter

- Table 1. Quantitative comparison with baseline methods. Our method MesaTask achieves the best generation performance on all evaluation metrics, consistently outperforming other baselines. Meanwhile, we can also observe the performance boost brought by our proposed spatial reasoning chain and the employed DPO training.

GPT Score User

Success Rate(%) FID↓

Model

Study CwT OSR PPI LCR OV Avg.

GPT-4o w/o reason. 91.6 84.3 5.02 8.04 8.90 5.74 7.05 6.95 3.11 GPT-4o 91.4 74.4 5.30 8.06 8.99 5.96 7.12 7.09 4.21

Holodeck-table 99.3 91.3 2.62 7.20 8.58 3.82 4.89 5.42 2.29 I-Design-table 56.5 96.0 4.99 8.00 8.86 5.88 6.62 6.87 1.73

Our w/o reason. 100.0 40.8 7.14 8.59 9.13 7.48 8.66 8.20 5.43 Ours w/o DPO 98.4 41.4 7.18 8.59 9.15 7.52 8.71 8.23 5.75 Ours 99.1 40.3 7.22 8.64 9.17 7.53 8.71 8.25 6.12

fine-tuning in both stages. In the SFT stage, the model is trained for one epoch using the learning rate of 1 × 10−5. In the DPO stage, we train for one epoch, with the learning rate of 1 × 10−6. All experiments are conducted on a cluster of eight A800 GPUs.

Baselines. We evaluate our model against two categories of benchmark methods. The first is closed-source large language models, specifically GPT-4o, where we perform our task in a zero-shot manner. The second category comprises modular scene generation methods, like Holodeck Yang et al.

- (2024f) and I-DesignÇelen et al. (2024). These approaches are originally targeted for indoor scene generation, and are now adapted to fit our task without changing their core frameworks, which are noted as Holodeck-table and I-Design-table here.

Metrics We first employ Fréchet Inception Distance (FID) to measure the realism of the generated scenes and the success rate to reflect the syntactic correctness of the LLM-generated output format. A 100% success rate indicates that all the model’s outputs are interpretable and can be directly parsed for downstream object retrieval, ultimately enabling the construction of tabletop scenes. Moreover, to conduct a more comprehensive evaluation, we propose the GPT-score, a metric designed to assess the multi-dimensional performance of generated scenes, including Consistency with Task (CwT), Object Size Reasonableness (OSR), Placement Plausibility & Intersections (PPI), Layout Coherence & Realism (LCR), and Object Visibility (OV).

###### 5.2. Comparison to baselines

For a fair comparison, each method generates 500 tabletop scenes according to the corresponding task instructions, which will serve as the evaluation corpus for the aforementioned metrics.

Quantative evaluation. As shown in Table 1, our method MesaTask demonstrates superior overall performance across all evaluation protocols. In comparison to multiple baseline methods, MesaTask achieves significantly better performance in terms of FID, thereby indicating its capability to generate more realistic tabletop scenes. With respect to the GPT-based multi-dimensional metrics, MesaTask consistently outperforms alternative baseline methods, with particularly notable advantages in CwT and LCR. This superior performance reflects enhanced task-scene alignment and more plausible

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

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

GPT-4o I-Design-table Holodeck-table MesaTask

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

###### Objects cluster:

Task: Group all markers and pens

Environment: A tabletop withvariousstationary items including markers,

stacking

Marker, Pen, Laptop

pens, books, and pot plants mostly centered and to the back middle.

on the right side of the laptop.

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Objects cluster: Candle, Tray, Plant

Task: Remove the candle from the tray

Environment: A tabletop with a candle on a tray at center, cosmetic products on the left, and a potted plant with brushes on the right.

above

and place it in front of the potted plant.

[Figure 94]

###### Objects cluster:

Task: Arrange fruits from bowl onto

Environment: Tabletop with a large succulent plant at center, smaller

Fruit, Bowl, Plant

the table near the large plant.

plant to left, bowl of fruit in front, and various candles and a pitcher.

containment

- Figure 4. Qualitative comparison under the same input task descriptions. Our proposed method, MesaTask, outperforms all baseline approaches across multiple perspectives, specifically exhibiting enhanced realism, superior task-scene alignment, more plausible tabletop layouts, and improved modeling of complex inter-object relationships.

tabletop layouts, which can be attributed to the high-quality MesaTask-10K dataset we constructed. To assess the perceptual quality of these generated scenes, a total of 127 participants are invited to conduct a comprehensive user study from three distinct assessment dimensions. The scores presented in Table 1 further confirm that our method achieves the most favorable outcomes in terms of human preference. More evaluation details are put in the supplementary materials.

Qualitative results To further substantiate the superior performance of MesaTask, Figure 4 presents qualitative comparisons between our method and three representative baselines. MesaTask consistently produces more realistic and diverse scenes, with a greater number of objects arranged with semantically meaningful and spatially coherent layouts. Moreover, its outputs align more closely with task instructions, capturing nuanced spatial relations such as stacking, containment, and precise object relocation. In contrast, baseline methods often generate overly simplistic or symmetrical layouts, miss key objects, or struggle to interpret complex spatial commands. These qualitative results highlight MesaTask’s impressive ability to model task-driven tabletop scenes with plausible layouts.

###### 5.3. Ablation study

To better understand the impact of each component in our framework, we conduct a comprehensive ablation study analyzing the contribution of spatial reasoning and preference tuning via Direct Preference Optimization (DPO). Table 1 presents the results of a quantitative ablation study conducted on MesaTask. The removal of either the spatial reasoning module or the DPO training component results in a measurable degradation of MesaTask’s overall performance. The qualitative comparisons shown in Figure 5 illustrate the advantages brought by the supplementary DPO training. As observed, DPO effectively alleviates common failure modes exhibited by the SFT-only model, including severe object collisions, the absence of task-relevant objects, and erroneous inter-object relationships. These improvements validate the effectiveness of our adopted DPO training in ensuring coherent table layouts and correct interrelations, leading to realistic tabletop scenes that exhibit greater visual plausibility and enhanced functional fidelity to the given task instructions.

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

SFT

[Figure 100]

[Figure 101]

[Figure 102]

DPO

Task: Put the metallic coffee pot to the left side of the plate.

Task: Organize fruit from bowl into empty bowl and place tray near lamp.

Task: Create a neat corner by grouping the lamp, potted plants, and pen holder.

- Figure 5. Ablation study on DPO training. Compared to the model fine-tuned solely via SFT, the model additionally trained with DPO maintains a lower collision rate (left), higher fidelity to task-related objects (middle), and superior alignment with input task instructions (right).

[Figure 103]

[Figure 104]

[Figure 105]

Task: Prepare cash and receipts for end-ofday closing. (Cashier Counter)

Task: Insert the DVD into the TV and play it. (TV Stand)

Task: Organize the side table surface. (Side Table)

- Figure 6. MesaTask is capable of generating realistic tabletop scenes belonging to novel categories that are not present in the training dataset.

###### 5.4. Generalization capability

To validate the generalization capability of our proposed method, we select tabletop categories not present in MesaTask-10K, including nightstands, TV stands, and side tables from household scenes, as well as cashier counters from shop scenes. For these four tabletop categories, we employ GPT-4o to generate plausible tasks, with 16 tasks assigned to each category.

As shown in Table 2, MesaTask exhibits robust generalization capability when tested on the four unseen table categories. The performance of MesaTask across all metrics is comparable to its performance on the test set of six seen categories from MesaTask-10K, as listed in Table 1. Notably, in the case of cashier counters, even though cash registers are not included in MesaTask-10K, our method can accurately generate their descriptions and sizes while placing them correctly. Notably, we can not compute FID since these new scenes are not included in MesaTask-10K. Figure 6 additionally presents several generated tabletop scenes, which belong to these four novel tabletop categories. MesaTask is capable of generating realistic tabletop scenes belonging to novel categories and strictly aligns with the given task instructions.

- Table 2. Generalization capability of MesaTask on four unseen tabletop categories. MesaTask exhibits comparable performance across six distinct evaluation protocols relative to its performance on the test set of the MesaTask-10K dataset.

GPT Score CwT OSR PPI LCR OV Avg.

Category Success Rate(%)

Nightstand 100.0 7.44 8.12 9.00 7.69 8.75 8.20 TV stand 100.0 6.56 8.06 9.06 6.94 7.44 7.61 Side table 100.0 7.62 8.62 9.25 7.69 8.50 8.34 Cashier counter 100.0 6.25 8.38 9.06 6.62 7.50 7.56

#### 6. Conclusion

In this paper, we introduce a novel task, namely task-oriented tabletop scene generation, which presents significant challenges owing to the substantial disparity between high-level task instructions and scene layouts. To support this demanding task, we propose a large-scale dataset, MesaTask-10K, consisting of roughly 10, 700 tabletop scenes that span six distinct indoor table categories. Thanks to our proposed spatial reasoning chain, our LLM-based framework MesaTask will sequentially reason about the scene composition, the spatial interrelationship, the scene graph, and ultimately, the 3D scene layout, based on which 3D assets are retrieved to form a complete tabletop scene. In evaluations, MesaTask demonstrates superior performance over existing baselines in accurately conforming to task instructions and modeling complex inter-object relations. We believe our dataset and framework will inspire a promising research direction and unveil new challenges in this field.

#### References

J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt,

- S. Altman, S. Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

- A. Çelen, G. Han, K. Schindler, L. Van Gool, I. Armeni, A. Obukhov, and X. Wang. I-design: Personalized llm interior designer. arXiv preprint arXiv:2404.02838, 2024.

Y. Chen, J. Ni, N. Jiang, Y. Zhang, Y. Zhu, and S. Huang. Single-view 3d scene reconstruction with high-fidelity shape and texture. In 2024 International Conference on 3D Vision (3DV), pages 1456–1467. IEEE, 2024a.

- Y. Chen, T. Wang, T. Wu, X. Pan, K. Jia, and Z. Liu. Comboverse: Compositional 3d assets creation using spatially-aware diffusion guidance. In European Conference on Computer Vision, pages 128–146. Springer, 2024b.

L. Dai, H. Wang, W. Wan, and H. Su. Manitaskgen: A comprehensive task generator for benchmarking and improving vision-language agents on embodied decision-making. arXiv preprint arXiv:2505.20726, 2025.

- T. Dai, J. Wong, Y. Jiang, C. Wang, C. Gokmen, R. Zhang, J. Wu, and L. Fei-Fei. Automated creation of digital cousins for robust policy learning. arXiv preprint arXiv:2410.07408, 2024.

M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. VanderBilt, L. Schmidt, K. Ehsani,

- A. Kembhavi, and A. Farhadi. Objaverse: A universe of annotated 3d objects. arXiv preprint arXiv:2212.08051, 2022.

W. Feng, W. Zhu, T.-j. Fu, V. Jampani, A. Akula, X. He, S. Basu, X. E. Wang, and W. Y. Wang. Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems, 36:18225–18250, 2023.

R. Fu, Z. Wen, Z. Liu, and S. Sridhar. Anyhome: Open-vocabulary generation of structured and textured 3d homes. In European Conference on Computer Vision, pages 52–70. Springer, 2024.

H. Han, R. Yang, H. Liao, J. Xing, Z. Xu, X. Yu, J. Zha, X. Li, and W. Li. Reparo: Compositional 3d assets generation with differentiable 3d layout alignment. arXiv preprint arXiv:2405.18525, 2024.

- Z. Huang, Y. Guo, X. An, Y. Yang, Y. Li, Z. Zou, D. Liang, X. Liu, Y. Cao, and L. Sheng. Midi: Multi-instance diffusion for single image to 3d scene generation. arXiv preprint arXiv:2412.03558, 2024.

##### B. F. Labs. Flux. https://github.com/black-forest-labs/flux, 2024.

W. Li, J. Liu, R. Chen, Y. Liang, X. Chen, P. Tan, and X. Long. Craftsman: High-fidelity mesh generation with 3d native generation and interactive geometry refiner. arXiv preprint arXiv:2405.14979, 2024.

L. Ling, C.-H. Lin, T.-Y. Lin, Y. Ding, Y. Zeng, Y. Sheng, Y. Ge, M.-Y. Liu, A. Bera, and Z. Li. Scenethesis: A language and vision agentic framework for 3d scene generation. arXiv preprint arXiv:2505.02836, 2025.

H. Liu, Y. Zheng, G. Chen, S. Cui, and X. Han. Towards high-fidelity single-view holistic reconstruction of indoor scenes. In European Conference on Computer Vision, pages 429–446. Springer, 2022a.

W. Liu, T. Hermans, S. Chernova, and C. Paxton. Structdiffusion: Object-centric diffusion for semantic rearrangement of novel objects. In Workshop on Language and Robotics at CoRL 2022, 2022b.

W. Liu, C. Paxton, T. Hermans, and D. Fox. Structformer: Learning spatial structure for languageguided semantic rearrangement of novel objects. In 2022 International Conference on Robotics and Automation (ICRA), pages 6322–6329. IEEE, 2022c.

Y. Nie, X. Han, S. Guo, Y. Zheng, J. Chang, and J. J. Zhang. Total3dunderstanding: Joint layout, object pose and mesh reconstruction for indoor scenes from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 55–64, 2020.

NVIDIA. Isaac sim 4.0 - robotics simulation and synthetic data generation. https://developer.nvidia.com/isaac-sim, 2024.

D. Paschalidou, A. Kar, M. Shugrina, K. Kreis, A. Geiger, and S. Fidler. Atiss: Autoregressive transformers for indoor scene synthesis. Advances in Neural Information Processing Systems, 34:12013–12026, 2021.

R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan, et al. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024.

###### C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818–2826, 2016.

J. Tang, Y. Nie, L. Markhasin, A. Dai, J. Thies, and M. Nießner. Diffuscene: Denoising diffusion models for generative indoor scene synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20507–20518, 2024.

R. Tedrake et al. Drake: Model-based design and verification for robotics, 2019. Y. Wang, X. Qiu, J. Liu, Z. Chen, J. Cai, Y. Wang, T.-H. J. Wang, Z. Xian, and C. Gan. Architect:

Generating vivid and interactive 3d scenes with hierarchical 2d inpainting. Advances in Neural Information Processing Systems, 37:67575–67603, 2024.

F. Xiang, Y. Qin, K. Mo, Y. Xia, H. Zhu, F. Liu, M. Liu, H. Jiang, Y. Yuan, H. Wang, et al. Sapien: A simulated part-based interactive environment. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11097–11107, 2020.

M. Xu, P. Chen, H. Liu, and X. Han. To-scene: A large-scale dataset for understanding 3d tabletop scenes. In European conference on computer vision, pages 340–356. Springer, 2022.

Y. Xu, J. Mao, Y. Du, T. Lozáno-Pérez, L. P. Kaebling, and D. Hsu. " set it up!": Functional object arrangement with compositional generative models. arXiv preprint arXiv:2405.11928, 2024.

- A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang,

- J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024a.

L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao. Depth anything v2. arXiv:2406.09414, 2024b.

L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911, 2024c.

Y. Yang, B. Jia, P. Zhi, and S. Huang. Physcene: Physically interactable 3d scene synthesis for embodied ai. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16262–16272, 2024d.

Y. Yang, J. Lu, Z. Zhao, Z. Luo, J. J. Yu, V. Sanchez, and F. Zheng. Llplace: The 3d indoor scene layout generation and editing via large language model. arXiv preprint arXiv:2406.03866, 2024e.

Y. Yang, F.-Y. Sun, L. Weihs, E. VanderBilt, A. Herrasti, W. Han, J. Wu, N. Haber, R. Krishna, L. Liu, et al. Holodeck: Language guided generation of 3d embodied ai environments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16227–16237, 2024f.

- K. Yao, L. Zhang, X. Yan, Y. Zeng, Q. Zhang, L. Xu, W. Yang, J. Gu, and J. Yu. Cast: Component-aligned 3d scene reconstruction from an rgb image. arXiv preprint arXiv:2502.12894, 2025.

Y. Zeng, M. Wu, L. Yang, J. Zhang, H. Ding, H. Cheng, and H. Dong. Lvdiffusor: Distilling functional rearrangement priors from large models into diffusor. IEEE Robotics and Automation Letters, 2024.

C. Zhang, Z. Cui, Y. Zhang, B. Zeng, M. Pollefeys, and S. Liu. Holistic 3d scene understanding from a single image with implicit representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8833–8842, 2021.

- L. Zhang, Z. Wang, Q. Zhang, Q. Qiu, A. Pang, H. Jiang, W. Yang, L. Xu, and J. Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG), 43(4):1–20, 2024.

In this supplementary, we demonstrate the detailed implementation of our main paper. The content of each section is as follows:

- • A covers 3D asset annotation, image generation, coarse scene construction, dataset statistics, and benchmark evaluation for MesaTask-10K.
- • B explains inference processes, reasoning data construction (including scene graph rules), and DPO data construction for the MesaTask framework.
- • C outlines baseline model implementations, evaluation metrics (Success Rate, FID, GPT-Score), and user study details.
- • D presents additional qualitative comparisons and further examples of scenes generated by MesaTask.
- • E discusses current limitations and future work.
- • F contains all detailed prompts used throughout the methodologies.

#### A. Details of MesaTask-10K

###### A.1. 3D asset annotation via GPT4o

We employ OpenAI’s gpt-4o-mini to annotate the 3D asset database for MesaTask, aiming to enhance the accuracy of its retrieval and placement results. Inspired by HOLODECK Yang et al. (2024f), we render images of an object from orthogonal rotations (0°, 90°, 180°, and 270°) as input, prompting GPT-4o-mini to output the following object attributes:

- • Category: Specifies the precise class to which the object belongs.
- • Description: Includes details such as the object’s name, shape, color, and material.
- • OnTable: A boolean value indicating whether the object is suitable for placement on a table. For instance, items like chairs or sofas would typically be marked as false.
- • Mass: Represents the mass of the object, preparing this data for potential future physics simulation applications.
- • Front View: An integer value defining the standard frontal orientation of the object. Objects of the same category should share a consistent front view, which is usually the more symmetrical or informative perspective.
- • IsContainer: A boolean value determining if the object can hold other items. If this attribute is true, the object will possess an “in” relationship when constructing the scene graph, indicating containment.
- • Material: The material of the 3D objects, such as "wooden", "glass".

###### A.2. Tabletop scene image generation stage

For the table type specified by the user, we first use OpenAI’s GPT-4o-mini to generate five descriptions of the table scene, and then feed prompt text F.1 to GPT.

###### A.3. Coarse tabletop scene construction stage

Our process for reconstructing a coarse 3D scene from a single image begins with comprehensive visual understanding. We first employ a Vision Language Model (VLM), specifically GPT-4o-mini, to identify the categories of objects in the input image. For each identified object category, GroundedSAM Ren et al. (2024) is utilized to generate precise instance-level segmentation masks. Concurrently, we use DepthAnything V2 Yang et al. (2024b) to estimate the image’s depth map. We subsequently convert this depth map into a 3D point cloud. By combining these instance-level segmentation

masks with the overall scene point cloud, we get the point cloud for each object instance. Finally, an axis-aligned bounding box (AABB) is computed for each instance’s point cloud, providing an initial layout.

Given the initial layout, we retrieve the 3D asset for each instance. Leveraging the textual descriptions for each object instance obtained from the VLM, we choose the 3D asset that has the highest textual similarity within assets in our database.

Our 3D asset database includes objects curated from Holodeck, the PartNet Mobility dataset, and assets generated by the image-to-3D model (Hunyuan3D), which comprises 11, 000 rigid 3D assets and 1, 034 articulated objects. Objects in PartNet-Mobility are already orientation-aligned. For the other 3D assets, we first centered the camera around the object and uniformly rendered eight views around the z-axis (up). We manually exclude assets that cannot be standardized via z-axis rotation.

- A VLM model then identifies which of the eight images shows the front view, and the 3D asset is rotated accordingly. After rotating the assets, we manually inspect each object’s orientation. Assets not properly rotated to face the front are re-oriented manually. This annotation pipeline ensures all objects in our database are orientation-aligned.

In the coarse placement stage, we aim to resize retrieved 3D assets 𝑜 to align with the target Axis-Aligned Bounding Box (AABB) d′ = (𝑤′, 𝑑′, ℎ′). To this end, we employ the Isometric Placement strategy. This strategy places the retrieved 3D assets by uniformly scaling them to preserve their intrinsic XYZ aspect ratio. Specifically, let the dimensions of a retrieved asset 𝑜 be d = (𝑤, 𝑑, ℎ). The scaling factor, 𝑠, is calculated by resizing the shortest dimension of the asset (along its X, Y, or Z principal axes) to match the length of the corresponding dimension of the target instance’s AABB d′. For example, if 𝑤 is the shortest dimension of d, then 𝑠 = 𝑤′/𝑤. All other asset dimensions are

- then scaled by the same factor 𝑠 to maintain their original aspect ratio, resulting in scaled dimensions 𝑠 · d = (𝑠 · 𝑤, 𝑠 · 𝑑, 𝑠 · ℎ). This particular approach to scaling is predicated on the inherent limitations of single-view depth estimation; challenges such as partial occlusions and errors in predicted depth values frequently compromise the reliability of these image-derived AABB dimensions d′, making a cautious, proportion-preserving scaling method essential. Preserving the asset’s relative XYZ proportions in this manner is also crucial for facilitating subsequent manual annotation and refinement. Furthermore, it is important to note that the aim of this coarse placement stage is not to achieve perfect 3D reconstruction but rather to generate approximate bounding box data 𝑠 · d sufficiently aligned with the scene to serve as effective training data for later stages.

###### A.4. The effort of human annotators

The coarse construction stage is not sufficiently plausible due to challenges such as occlusion, inaccurate depth estimation, and retrieval errors. Thus, human annotators played a critical and extensive role in refining the layout. Specifically, for each sample, annotators were provided with: the GLB file of the tabletop scene, the rendered scene image, as well as individual object snapshots and their indices. Using Blender, annotators manually adjusted the scene layout to match the reference image, which involves several steps: 1) Translating objects to correct positions, 2) Scaling them to appropriate sizes, 3) Rotating them to align orientations correctly, 4) Ensuring spatial plausibility of inter-object relations. The annotation complexity varied substantially depending on the number of objects and the complexity of their spatial configurations. For example, scenes with many small objects and dense relations were significantly more complex and time-consuming. On average, annotators spend 10 to 20 minutes per scene.

After receiving each annotated 3D file from annotators, we render all annotated 3D scenes from four directions: front, back, left, and right. These renderings are compared with reference images. If

Distribution of Top 100 Object Categories

- 102

- 103

Frequency

box

pillow

food

tool

ball

bowl

laptop

die

apple

book

flowerpot

pot

spoon

toy

bag

bottle

radio

flower

lamp

dice

plate

coin

vegetable

flashlight

medal

pencil

pliers

block

tablelamp

map

jar

plant

weapon

teapot

pen

tablet

vase

phone

figurine

dagger

knife

ring

telescope

monitor

televisionset

mug

lighter

candle

telephone

gun

clock

cup

fruit

pottery

handbag

cube

fruitvegetable

tray

can

guitar

decorativeobject

binoculars

candleholder

banana

first-aidkit

headphones

pitcher

toaster

watch

cellulartelephone

carton

statue

sunglasses

container

camera

chesspiece

hammer

card

kitchen-appliance

computermonitor

lantern

alarmclock

sweetsdessert

tradingcard

kettle

cake

basket

bread

speaker

musicalinstrument

controller

microphone

flowerarrangement

radioreceiver

sculpture

wrench

firearm

screwdriver

computerkeyboard

remotecontrol

Object Category

Figure 7. Distribution of the top 100 object categories

quality fails to meet requirements, such as frequent issues like unreasonable object sizes or objects floating in the air, we request annotators to revise the errors until they meet the acceptance criteria.

###### A.5. Dataset statistics

Our 3D asset database has an extensive collection of over 200 common tabletop object categories, which has numerous high-fidelity 3D models as shown in Figure 7.

###### A.6. Tabletop scene generation benchmark

In Table 3, we introduce Mesatask-10k as a new benchmark dataset to evaluate scene generation methods across five representative tabletop environments: Coffee Table, Dining Table, Dressing Table, Kitchen Counter, and Office Table. There is no comparison with the bathroom vanity because there is a pool in the bathroom vanity, and these methods cannot deal with this problem. The methods assessed include ATISS Paschalidou et al. (2021), DiffuScene Tang et al. (2024), and PhyScene Yang et al. (2024d). Evaluation metrics comprise FID (Fréchet Inception Distance), KID (Kernel Inception Distance), and CKL (Category KL Divergence), with lower values indicating better performance. The visualization results of these methods are shown in Figure 8.

ATISS, while slightly lagging in FID and KID, shows consistent results across all scenes, indicating stability in its generative quality. PhyScene exhibits relatively higher FID and KID in simpler layouts but achieves competitive scores in more complex scenes such as Kitchen Counter and Dressing Table, likely due to its integration of physical constraints that enhance realism in interaction-heavy settings.

The CKL metric reveals complementary insights. Lower CKL values indicate that a model captures the categorical distribution of objects more faithfully. ATISS and PhyScene generally achieve lower CKL scores in scenes with moderate complexity, suggesting a more accurate modeling of object co-occurrence and diversity. DiffuScene, despite strong performance in FID and KID, often yields higher CKL values, particularly in scenes such as Kitchen Counter and Dining Table, where a large number of distinct object categories are present. This discrepancy suggests that DiffuScene may

###### Table 3. Comparison of ATISS, DiffuScene, and PhyScene trained on MesaTask-10k.

Scene ATISS Paschalidou et al. (2021) DiffuScene Tang et al. (2024) PhyScene Yang et al. (2024d)

FID ↓ KID ↓ CKL ↓ FID ↓ KID ↓ CKL ↓ FID ↓ KID ↓ CKL ↓

Coffee Table 41.85 0.0101 0.217 40.36 0.0091 0.224 43.41 0.0137 0.247 Dining Table 59.89 0.0291 0.806 53.61 0.0185 1.051 50.13 0.0131 0.790 Dressing Table 42.07 0.0103 0.685 45.95 0.0117 0.772 39.54 0.0098 0.464 Kitchen Counter 51.81 0.0232 1.229 51.49 0.0194 1.231 50.07 0.0182 0.756 Office Table 59.22 0.0313 0.217 42.47 0.0118 0.224 35.09 0.0047 0.357

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

ATISS DiffuScene

PhyScene

[Figure 111]

[Figure 112]

Coffee Table

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Dining Table

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Dressing Table

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Kitchen Counter

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

Office Table

Figure 8. Qualitative results of ATISS, DiffuScene, and PhyScene Trained on MesaTask-10k

prioritize visual plausibility over accurate semantic diversity, whereas ATISS and PhyScene better balance both.

Overall, the results highlight a trade-off between visual quality and semantic alignment. While DiffuScene excels in producing visually coherent layouts, ATISS and PhyScene demonstrate stronger alignment with real-world category distributions. The inclusion of CKL as an evaluation metric in Mesatask-10k proves critical for revealing these nuances, emphasizing the importance of considering object diversity and distribution fidelity alongside traditional image-level metrics.

It’s noteworthy that our method, MesaTask, is quite different from these three representative scene

generation methods. Our approach focuses on generating scenes from task instructions, whereas prior methods such as ATISS, DiffuScene, and PhyScene generate scenes from simple scene descriptions. This fundamental difference in settings precludes direct comparisons. Moreover, MesaTask is capable of generating open-vocabulary tabletop scenes given task instructions. Leveraging a fine-tuned LLM

- to generate scene layouts, it can produce layouts and textual descriptions for objects not present in the training set. In contrast, ATISS, DiffuScene, and PhyScene generate predefined object features and cannot generate objects out of the training set at inference time.

- B. Details of MesaTask

- B.1. Inference details

During the inference phase, the user input consists of three parts: 1. Task Instruction, which is typically a description of the task (e.g., "Organize fruit from bowl into empty bowl and place tray near lamp"); 2. Size of the tabletop surface. The tabletop surface is specified as a list [𝑥𝑚𝑖𝑛, 𝑦𝑚𝑖𝑛, 𝑥𝑚𝑎𝑥, 𝑦𝑚𝑎𝑥] in centimeters (cm), defining a rectangular area, which is the object’s place. The user input is then processed via the OpenAI GPT-4o API to generate the detailed task information with prompt F.2. 3. Size of the No Placement Area (Optional): a list [𝑥𝑚𝑖𝑛, 𝑦𝑚𝑖𝑛, 𝑥𝑚𝑎𝑥, 𝑦𝑚𝑎𝑥] in centimeters (cm), defining a rectangular area that objects can not be placed, in the table type like bathroom vanity, the this area refer to the sink area.

Our MesaTask model subsequently generates tabletop scene layouts from the detailed task information. Its primary outputs encompass two key components: a Spatial Reasoning Chain—which includes stages such as Object Completion, Interrelation Inference, and Scene Graph generation—and ultimately, the final tabletop scene layout. This layout specifies the position and size of the objects.

Given the generated scene layout, we retrieve the objects from the 3D assets database and place

- them on the tabletop. Our 3D asset database comprises objects 𝑜, each defined by a tuple 𝑜 = (𝑡, d), where 𝑡 is a textual description (e.g., "red coffee mug") and d = (𝑤, 𝑑, ℎ) represents the dimensions (width, depth, height) of its 3D bounding box. Correspondingly, each object 𝑜′ in the generated scene layout is characterized by its target 3D bounding box size d′ = (𝑤′, 𝑑′, ℎ′), its desired position

p′ = (𝑥′, 𝑦′, 𝑧′), and an associated textual description 𝑡′.

Inspired by HOLODECK Yang et al. (2024f), the selection of the most appropriate 3D asset from the library for each target object 𝑜′ is guided by two metrics: Text Similarity and Size Similarity. Text similarity, 𝑇(𝑡, 𝑡′), between the asset description 𝑡 and the target description 𝑡′ is computed using a Sentence Transformer (SBERT), specifically the all-mpnet-base-v2 checkpoint: 𝑇(𝑡, 𝑡′) = SBERT(𝑡, 𝑡′).

Complementing textual matching, Size Similarity addresses the practical consideration that a single textual description can correspond to objects of various sizes. For this, we represent the size of the asset 𝑜 as a vector d and that of the target object 𝑜′ as d′. The size similarity, 𝑆(𝑜, 𝑜′), is then computed using the cosine similarity as: 𝑆(𝑜, 𝑜′) = cos(d, d′).

The final retrieval score 𝑅(𝑜, 𝑜′) for a library asset 𝑜 with respect to a target object 𝑜′ is formulated as a weighted sum of these two similarities:

𝑅(𝑜, 𝑜′) = 𝛼 · 𝑇(𝑡, 𝑡′) + 𝛽 · 𝑆(𝑜, 𝑜′)

The weights are empirically set to 𝛼 = 0.9 and 𝛽 = 0.1, prioritizing textual relevance while still accounting for dimensional congruence. The 3D asset that yields the highest retrieval score 𝑅(𝑜, 𝑜′) is

- then selected for subsequent placement into the scene according to p′ and size d′.

###### B.2. Reasoning data construction

For a scene S in MesaTask-10K, we first extract the scene graph by following the rule-based method below. To compensate for the spatial relations missing in the scene graph, we utilize GPT-4o Achiam et al. (2023) with Table Description prompt F.6 to output a detailed scene description D based on the rendered tabletop scene image. Given the scene graph GS and scene descriptions D, the multimodal LLM is prompted to generate a complete object list V and inter-object relations E, as well as the corresponding task instructions T, in particular including aforementioned detailed task information with Task from Scene Graph prompt and Reasoning Context Generation prompt F.6.

Scene graph extraction. We define a set of interpretable geometric rules grounded in relative positions, distances, and orientations, as shown in Table4. The Left of, Right of, In front of, and Behind relationships are determined based on the relative x and y position differences between object centroids, constrained by a distance threshold proportional to the table size. Vertical relationships, including above and below, are defined using the z-coordinate range of objects and require sufficient horizontal overlap to ensure meaningful interaction. The In relationship captures containment, requiring both high horizontal and vertical overlap ratios.

Object orientation is characterized by the Face to relations, which map the z-axis rotation angle of an object to eight discrete directional bins, such as Front, Back, Left, and the diagonals. Positional context is described by the Is at relation, which locates an object within a 3×3 spatial grid overlaid on the table surface. Finally, the Are equally spaced along rule detects linear arrangements of three or more objects with approximately uniform spacing along either the x- or y-axis, within a 10% tolerance range.

Table 4. Rules to determine the spatial relationships between objects.

###### Relationship Rule

Left of |𝑑𝑥| > |𝑑𝑦| and 𝑑𝑥 < 0 and 𝑑(𝑠, 𝑜) ≤ 0.4 · max(𝑡𝑎𝑏𝑙𝑒_𝑠𝑖𝑧𝑒) Right of |𝑑𝑥| > |𝑑𝑦| and 𝑑𝑥 > 0 and 𝑑(𝑠, 𝑜) ≤ 0.4 · max(𝑡𝑎𝑏𝑙𝑒_𝑠𝑖𝑧𝑒)

In front of |𝑑𝑦| > |𝑑𝑥| and 𝑑𝑦 < 0 and 𝑑(𝑠, 𝑜) ≤ 0.4 · max(𝑡𝑎𝑏𝑙𝑒_𝑠𝑖𝑧𝑒)

Behind |𝑑𝑦| > |𝑑𝑥| and 𝑑𝑦 > 0 and 𝑑(𝑠, 𝑜) ≤ 0.4 · max(𝑡𝑎𝑏𝑙𝑒_𝑠𝑖𝑧𝑒) Above 𝑧_𝑚𝑖𝑛1 > 𝑧_𝑚𝑎𝑥2 − 𝑡ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑 and 𝑜𝑣𝑒𝑟𝑙𝑎𝑝ℎ𝑜𝑟𝑖𝑧𝑜𝑛𝑡𝑎𝑙 ≥ 0.5 Below 𝑧_𝑚𝑎𝑥1 < 𝑧_𝑚𝑖𝑛2 + 𝑡ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑 and 𝑜𝑣𝑒𝑟𝑙𝑎𝑝ℎ𝑜𝑟𝑖𝑧𝑜𝑛𝑡𝑎𝑙 ≥ 0.5

In 𝑜𝑣𝑒𝑟𝑙𝑎𝑝ℎ𝑜𝑟𝑖𝑧𝑜𝑛𝑡𝑎𝑙 ≥ 0.9 and 𝑜𝑣𝑒𝑟𝑙𝑎𝑝𝑣𝑒𝑟𝑡𝑖𝑐𝑎𝑙 ≥ 0.5 Face to front −𝜋8 ≤ 𝜃 < 𝜋8

Face to front_right 𝜋8 ≤ 𝜃 < 38𝜋

Face to right 38𝜋 ≤ 𝜃 < 58𝜋 Face to back_right 58𝜋 ≤ 𝜃 < 78𝜋

Face to back 78𝜋 ≤ 𝜃 or 𝜃 < −78𝜋 Face to back_left −78𝜋 ≤ 𝜃 < −58𝜋

Face to left −58𝜋 ≤ 𝜃 < −38𝜋 Face to front_left −38𝜋 ≤ 𝜃 < −𝜋8

Is at Relative position in 9-grid division of table Are equally spaced along Three or more objects with equal spacing in X or Y direction

- B.3. DPO data construction

To facilitate training via Direct Preference Optimization (DPO), we construct a dataset of preferencelabeled 3D scene layout samples. Each sample consists of a natural language prompt along with a pair of completions: a positive layout that is preferred, and a negative layout that is dispreferred. These pairs are used to model the implicit preferences between layout candidates under the same instruction, allowing the DPO algorithm to learn alignment signals from relative quality judgments.

The positive samples are drawn from our curated dataset, MesaTask-10K, which contains highquality 3D layouts that successfully fulfill the spatial requirements of the associated instructions. Each entry includes a reasoning process (referred to as “thinking process”) and an output layout represented in structured JSON format, including geometric attributes such as object positions, rotations, and sizes, as well as an explicit symbolic scene graph describing inter-object spatial relationships (e.g., “(Cup, left of, Bowl)”).

To obtain corresponding negative samples, we apply one of the following three corruption strategies

- to the original layout, each designed to reflect a typical failure mode observed in model outputs after supervised fine-tuning (SFT):

- • Geometric Perturbation (Collision Induction): A subset of objects in the layout is selected at random, and their spatial attributes (position, rotation, or size) are perturbed. The perturbations are bounded to remain within the layout’s item placement zone but are large enough (e.g., up to 20% of the region’s width or height) to induce spatial conflicts or object collisions. The type of perturbation is chosen probabilistically, favoring position changes (e.g., with 80% probability). The resulting layout Lcol− often violates basic physical plausibility.
- • Scene Graph Corruption (Semantic Misalignment): The reasoning trace, particularly the scene graph, is altered by either removing a subset of the spatial relations or replacing them with incorrect ones. For example, “(Book, on, Shelf)” might be replaced with “(Book, under,

Shelf).” This yields a logically inconsistent reasoning trace, denoted as Lrel− , which conflicts with the spatial semantics of the original instruction.

- • Object Removal (Functional Deficiency): One or more task-relevant objects are randomly deleted from the layout. This operation simulates incomplete or underspecified layouts, pro-

ducing samples denoted by Lmiss− that are geometrically valid but semantically deficient with respect to the task requirements.

For each instruction, a prompt is constructed by concatenating the instruction and input fields. The chosen completion consists of the original reasoning trace followed by the correct layout output. The rejected completion is created by applying one of the aforementioned corruption strategies. To increase supervision signal diversity, we sample two independent rejected completions per prompt using different corruption methods or random seeds.

- C. Details of experiment

- C.1. Implementation of baseline model

Holodeck-Table HOLODECKYang et al. (2024f) is an excellent method for generating scene layouts. It leverages a commercial, closed-source large language model (GPT) to first generate the number, types, and spatial relationships of objects in a scene. Then, it retrieves appropriate 3D assets and uses an optimization-based search algorithm to place them plausibly within the environment.

We believe this layout generation approach can be adapted to desktop environments, so we implemented a modified version tailored for desktop object placement, which we call Holodeck-Table.

A key aspect of leveraging large language models lies in prompt engineering. Building upon the original HOLODECK prompts, we designed prompts more suited to desktop scenarios and removed modules irrelevant to desktop layouts, such as the Wall Module and Window Module. The modified prompt is shown in F.4.

However, prompt modification alone is insufficient to generate reasonable desktop layouts. This is due to fundamental differences between room-scale and desktop-scale environments. For example, furniture tends to be placed near walls, while desktop objects are often positioned away from the table edges. Therefore, we also modified HOLODECK’s optimization phase to better accommodate desktop scene constraints and ensure the generated layouts are as realistic as possible.

I-Design-Table I-DesignÇelen et al. (2024) enables users to easily express their interior design preferences through natural language interaction, and transforms those preferences into visualized 3D layouts. The system employs a set of large language model agents that communicate and reason with each other to convert user input into a feasible scene graph, establishing the relative spatial relationships between objects. After generating the scene graph, I-Design uses a backtracking algorithm to determine the optimal placement of each object in the scene. Owing to the relatively simple design of I-Design’s optimization stage, no substantial modifications to the underlying algorithm were required. By appropriately adapting the original prompt, we developed I-Design-Table, which is capable of generating semantically and spatially coherent desktop object arrangements.The modified prompt is shown in the F.5.

###### C.2. Details of metrics

Success Rate For LLM-based methods, we define a response as successful if it adheres to the expected output format and includes at least one valid object. The success rate is computed as the ratio of successful responses to the total number of test cases.

Fréchet Inception Distance (FID) We adopt Fréchet Inception Distance (FID) to evaluate the visual realism of rendered tabletop scenes. FID compares the distribution of deep features extracted from generated images against those from ground-truth images, reflecting perceptual similarity at a high semantic level. Specifically, we render both real and generated 3D scenes from a front-facing camera view to obtain consistent 2D images. Each image is resized to 299 × 299, normalized to the [−1, 1] range, and converted to RGB (with alpha-composited black background if needed). We extract 2048-dimensional feature vectors using a pretrained Inception-V3 Szegedy et al. (2016) network, where the classification head is replaced with an identity mapping to preserve penultimate-layer activations.

Let X𝑟 and X𝑔 be the feature sets extracted from real and generated images, with empirical means (𝜇𝑟, 𝜇𝑔) and covariances (Σ𝑟, Σ𝑔), respectively. FID is computed as:

FID = ∥𝜇𝑟 − 𝜇𝑔∥22 + Tr(Σ𝑟 + Σ𝑔 − 2(Σ𝑟Σ𝑔)1/2). (6)

To ensure numerical stability, we follow standard practices: the matrix square root is computed via the SciPy sqrtm function, and only the real part is retained. When necessary, a small regularization term is added to the diagonal of covariance matrices to ensure positive semi-definiteness. FID in our setting serves as a quantitative proxy for how photorealistic and coherent the generated 3D scenes appear when rendered to 2D.

[Figure 129]

###### Figure 9. User study interface

GPT-Score To evaluate the semantic alignment and perceptual quality of generated 3D tabletop scenes, we propose a multi-dimensional assessment protocol based on GPT-based scoring. A pretrained large language model is prompted to analyze both the rendered front-view and perspective-view images of each scene, along with the corresponding task description, including environment and task, and to rate the layout across five distinct criteria:

- • Consistency with Task: Whether the scene’s object composition and spatial arrangement are coherent with the task description.
- • Object Size Reasonableness: Whether object sizes are proportionate and physically realistic.
- • Placement Plausibility & Intersections: Whether objects are grounded naturally without unrealistic interpenetrations.
- • Layout Coherence & Realism: Whether the overall layout is visually functional, realistic, and context-appropriate.
- • Object Visibility: Whether key task-relevant objects are clearly visible and identifiable in at least one view.

Each criterion is scored on a scale from 1 (poor) to 10 (excellent), and includes a short explanation to justify the score. The evaluation is carried out via a structured prompt specifically designed to elicit detailed, consistent assessments across scenes. For full prompt details, please refer to Section F.3.

###### C.3. User study

To further assess the perceptual quality and human preference of generated tabletop scenes, we conducted a user study based on rendered scene images. Participants were asked to rate scenes across three key dimensions:

- • Realism & Visual Quality: The overall visual realism and aesthetic fidelity of the image. (1 = Poor, 7 = Excellent)
- • Alignment with Text & Task: The degree to which the scene matches the given task description and aligns with the intended semantic content. (1 = Poor, 7 = Excellent)
- • Spatial Coherence: Whether the spatial arrangement and inter-object relationships in the scene are logical and physically plausible. (1 = Poor, 7 = Excellent)

Each rendered scene is rated on a 7-point Likert scale for the above criteria, and the final score for a scene is computed as the average of the three dimension scores.

Our custom user study interface (see Figure 9) randomly samples 5 scenes per participant from the test set. For each scene, the interface displays images generated by different methods in a randomly shuffled order to mitigate position bias. Each image is evaluated independently without revealing the identity of the generating method. A total of 127 participants completed the study, resulting in a diverse and robust set of human evaluations for comparative analysis across methods.

#### D. More result

Inspired by ManiTaskGen Dai et al. (2025), we further investigate our model’s performance across tasks with varying complexity levels. Accordingly, we categorized tasks of varying types and complexity levels into four task difficulty levels, specifically:

- • Level 1: Single-step pick-and-place tasks with a unique target object and no perceptual ambiguity (e.g., "Move the red dictionary on the bookshelf to the table");
- • Level 2: Single-step pick-and-place tasks with non-unique target objects, requiring additional

- Table 5. Quantitative performance results of MesaTask are presented across tasks with varying types and complexity levels.

Level Success Rate(%) FID↓

GPT Score CwT OSR PPI LCR OV Avg.

level1 99.8 59.3 7.20 8.88 9.40 7.22 7.80 8.10 level2 99.4 55.5 7.44 8.40 9.36 7.44 8.16 8.16 level3 99.2 50.9 7.04 8.36 9.46 7.28 8.10 8.05 level4 98.4 43.9 7.46 8.78 9.70 7.68 8.88 8.50

- Table 6. MesaTask generates collision-free scenes after physics-based post-processing, satisfying physical plausibility.

MesaTask w/o simulation

Method GPT-4o I-Design-table Holodeck-table

MesaTask Col. Rate(%) 21.13 9.92 0 11.21 0

descriptions for distinction (e.g., "Move the blue cup on the table to the coffee table" where multiple cups exist in the scene);

- • Level 3: Multi-step tasks formed by two Level 1 or Level 2 tasks connected by "THEN" (e.g., "First move the book from the bookshelf to the left of the table, then move it to the right of the table");
- • Level 4: Outcome-based abstract tasks describing the target scene state rather than specific steps (e.g., "Tidy up the messy desk", "Make the living room cleaner").

We provided the definitions of these four task levels to GPT-4o to generate diverse tasks, yielding 500 tasks per level. Each level’s tasks cover six common indoor tabletops (bathroom vanity, dining table, kitchen counter, coffee table, dressing table, office table). We evaluated the generated scenes using the same metrics as in the main paper. As shown in Table 5, tabletop scenes generated under all task levels achieved high scores in the multi-dimensional assessment, confirming that our method can effectively handle tasks of varying types and complexity levels.

However, we observed variations in FID across different task levels. This discrepancy arises because task instructions in the training set are predominantly at Level 4 difficulty (when training set tasks were classified using GPT-4o according to the above criteria, 83.8% fell into Level 4, 11.4% into Level 3, 3.8% into Level 2, and 1% into Level 1).

To calculate physical plausibility metrics, we thus evaluate occupancy overlaps between objects rather than bounding box intersections. Specifically, the layout of each scene, along with corresponding object assets, is transformed into Drake’s Tedrake et al. (2019) internal scene tensor representation. We then use Drake’s geometry engine to compute signed distances between all pairs of collision geometries. A negative signed distance indicates interpenetration, which is counted as a collision event. We obtained physical plausibility results on the test set in the main paper, with the collision rate metric defined as the number of collision object pairs 𝑁collision divided by the total number of potentially collision object pairs 𝑁total. The quantitative comparison is listed in Table 6.

We provide more qualitative comparisons between our method MesaTask and existing baselines, including GPT-4o Achiam et al. (2023), I-Design-table Yang et al. (2024f), and Holodeck-table. As shown in Figure 10, each row shows a task-conditioned tabletop scene generated by different methods

under the same instructions, illustrating their differences in object selection and arrangement. We also include additional qualitative results generated by our MesaTask model in Figure 11, Figure 12, and Figure 13. These examples show how MesaTask handles various task types such as cleaning, organizing, and preparing tabletop scenes. In the figures, the pink flat squares represent the sink area of a bathroom vanity, which is treated as a no-placement zone.

#### E. Limitation and future work

MesaTask mainly focuses on 6 common indoor table types. It means our work doesn’t yet cover the full table types found in everyday life, such as cashier counters or conference tables. Additionally, our MesaTask approach relies on 3D object retrieval, which naturally limits the object diversity to what’s available in our 3D object database.

In the future, we will explore integrating 3D object generation methods based on bounding box conditions into our tabletop scene generation pipeline. This should allow us to create various objects and more realistic tabletop scenes.

#### F. Prompt

###### F.1. Table Image Generation Prompt

Table Image Generation Prompt: Help me to generate a realistic table type: 𝑡𝑎𝑏𝑙𝑒𝑡𝑦𝑝𝑒 placement description, this description will become my input to the picture to generate the model of the prompt, the requirements of a variety of items, a variety of relationships between the placement of a variety of reasonable, list the items that appear in the image, and then generate the brief location relationship.

Attention:

- 1. Do not generate items on walls or hang them on walls like a mirror or a picture.
- 2. Do not generate rare small items.
- 3. Do not generate chairs, only desktop placements.
- 4. Do not generate people, only items.
- 5. Do not generate tablecloths or table mats.
- 6. Generate images that show the entire tabletop, with the items displayed in their entirety.
- 7. Do not generate shelves or cabinets.
- 8. Only generate items can be placed on the table.
- 9. Give simple item names (1 or 2 words).
- 10. The number of Items should be 𝑛𝑢𝑚𝑖𝑡𝑒𝑚𝑠.
- 11. "COFFEE TABLE" does not necessarily have to have coffee. It is mainly a low table that is placed near the sofa in the living room. Generate five prompts, each of which should be 80 words or fewer. You should only output the prompt; do not output any other content, list number, or true. The output template should be: 𝑡𝑎𝑏𝑙𝑒𝑡𝑦𝑝𝑒 parallel to the picture. blank wall, frontal top view, photograph. Items: .

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

GPT-4o I-Design-table Holodeck-table MesaTask

[Figure 146]

Environment: Vanity tabletop with various soap dispensers and lotion pumps on the left and center, sink in the middle, and towels on the right.

Task: Clean the sink area by moving soap dispensers and lotion pumps.

Objects cluster: Pump, Dispenser

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

Environment: Workspace with a monitor, keyboard, mouse, clock, glass, and a notebook on the right side, and a drawer and plant.

Objects cluster: Glass, Clock, Computer

[Figure 152]

Task: Arrange glass, clock, and notebook neatly on right side.

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Task: Clear the center workspace by moving the bottle, plant, and stapler.

Objects cluster: Bottle, Plant, Stapler

Environment: A desk with a closed Dell laptop at center, surrounded by a notepad, stapler, plant, holder, smartphone, mouse.

[Figure 157]

[Figure 158]

Objects cluster: Bottle, Plant, Stapler

Task: Clear the center workspace by moving the bottle, plant, and stapler.

Environment: A desk with a closed Dell laptop at center, surrounded by a notepad, stapler, plant, holder, smartphone, mouse.

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Objects cluster: Container, vase

Task: Arrange decorative vase and small round container to the front right corner.

Environment: Minimalist tabletop with a sink, soap dispenser, a round container, a container on its side and a vase in the upper right corner.

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Task: Prepare workspace by placing notebook and pen in front of laptop.

Environment: Desk with laptop center, pink notebook in front, desk lamp and globe on right, sticky note and pens on left.

###### Objects cluster:

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Notebook, Pen, Laptop, Lamp

Task: Put the knife on the cutting board.

Objects cluster: Cutting Board, Knife, Bowl

Environment: A tabletop with two bowls, a cutting board, a knife and a small plant in a pot arranged with some space.

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Objects cluster: Bowl, Bowl, Palettes

Task: Clear the center by moving the palettes and bowl to the left

Environment: A tabletop with jars, a salt lamp, a bowl, a bottle, cosmetic palettes, and a tray arranged mostly on the back and center.

Figure 10. More qualitative comparisons of task-conditioned scene generation results across GPT-4o, I-Design-table, Holodeck-table, and our proposed MesaTask.

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

Task: Clear the right side of the table by grouping mugs and stapler.

Task: Organize the three mugs to the front left corner near the container after removing its lid.

[Figure 179]

[Figure 180]

Task: Organize notebooks and sticky notes to the left corner.

Task: Clear the area to the right of the sink area by moving all toiletries and hair straightener to the left side near the vases.

[Figure 181]

[Figure 182]

Task: Turn on the desk lamp and place the smartphone next to it.

Task: Organize jars and cup near the potted plant.

Task: Prepare items for a morning routine by grouping the hair dryer, mirror, and bottles.

Task: Clear the center by moving the cup, candle and pitcher to the left.

Figure 11. Additional qualitative results generated by our proposed MesaTask.

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Task: Organize the notebooks and place it to the right side. Task: Clear the desk by moving small sticky notes objects to

the right side.

[Figure 187]

[Figure 188]

Task: Clear brushes from tabletop and place bowl to left side.

Task: Clear the tray by removing all glasses and placing them near the flower vase.

[Figure 189]

[Figure 190]

Task: Water the plants using the soap dispenser bottle. Task: Stack the books neatly behind the tray and place the

pot_plant_1 to the left of the stack.

Task: Create a plant corner by grouping potted cacti and soap dispenser.

Task: Move the white mug on the right to the top-left corner.

Figure 12. Additional qualitative results generated by our proposed MesaTask (continued).

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Task: Clean and organize the workspace by removing unnecessary items.

Task: Organize soap dispensers and bowls symmetrically around the sink.

[Figure 195]

[Figure 196]

Task: Move the largest black container with colorful labels from the center to the back-left corner of the table.

Task: Group the three spray or pump bottles on the left side into a straight line.

[Figure 197]

[Figure 198]

Task: Organize wooden utensils on the board. Task: Prepare the pot and place the plate near the chopping

board.

Task: Cut the pumpkin and place slices into the bowl on the right.

Task: Clear wooden spoons and place them inside the bowl.

Figure 13. Additional qualitative results generated by our proposed MesaTask (continued).

###### F.2. Inference Prompt

Task Instruction to Task Info Prompt: You are an AI assistant for robotic task planning. Given a high-level abstract task, expand it into a structured JSON format. Input: A single string: "Task" (e.g., "High-level abstract task < 20 words"). Output Format (Strict JSON): { "Environment": "Brief scene description for the task", "Task": "The original input task string", "Goal": ["Ordered sub-objectives to achieve the task (aim for >=3 objects if task allows)"], "Action Sequence": ["Primitive robotic actions with parameters (e.g., Pick(Object))"], "Objects cluster": ["List of unique object types involved"] } Instructions for Fields:

- 1. Task: Copy the input task string directly.
- 2. Environment: Briefly describe a plausible environment for the task.
- 3. Goal: Break down the task into ordered sub-objectives. Involve at least 3 object types if the task context permits; otherwise, use only necessary objects.
- 4. Action Sequence: List primitive actions for the goals. Use object *types*. * Available actions: ‘Pick(obj)‘, ‘PlaceOn(obj)‘, ‘PlaceAt(pos)‘, ‘Push(obj, dir, dist)‘, ‘RevoluteJointOpen(obj)‘, ‘RevoluteJointClose(obj)‘, ‘PrismaticJointOpen(obj)‘, ‘PrismaticJointClose(obj)‘, ‘Press(obj)‘.
- 5. Objects cluster: List all unique object types from the task, goals, and actions. Ensure the output is only the JSON object.

###### F.3. GPT-score Evaluation Prompt

System Prompt You are an expert evaluator for 3D desktop scene layouts. Your task is to analyze desktop scenes and provide a detailed assessment based on specific criteria. Please carefully examine the provided front and perspective views of the desktop scene, along with the task description. Analyze how well the scene layout aligns with the intended task. Consider both the visible objects and their arrangement in relation to the task requirements. You must provide numerical scores (1-10) for each criterion along with brief explanations to justify your ratings. Be objective and consistent in your evaluation across different scenes.

User Prompt You are an expert at evaluating desktop scene layouts. Given both a front and a perspective view of a desktop scene, and a description of the tasks that can be performed on the desktop, please rate the scene’s quality on a scale from 1 (poor) to 10 (excellent) according to the following criteria. For each criterion, consider both views:

- 1. **Consistency with Task:** Does the scene layout (objects present, their arrangement and relevance) align well with the provided task description (environment, objects, goals)?

- - High score (7-10): All key objects are present, their arrangement is entirely logical and directly contributes to the task. The scene perfectly reflects the task requirements.
- - Mid score (4-6): Most key objects are present and generally align with the task, but there might be minor inconsistencies or some less relevant elements.
- - Low score (1-3): Significant deviations from the task description. Important objects are missing, or the arrangement is largely unrelated or illogical for the task.

- 2. **Object Size Reasonableness:** Are the sizes of the objects in the scene highly realistic, both relative to each other and to the overall desktop environment? Are they consistent across all objects?

- - High score (7-10): All objects have highly realistic and perfectly proportionate sizes. No inconsistencies are noticeable.
- - Mid score (4-6): Most objects have reasonable sizes, but there might be slight or occasional inaccuracies in proportion for some items.
- - Low score (1-3): Multiple or glaring inconsistencies in object sizes. Some objects are obviously and significantly too large or too small, severely impacting realism.

- 3. **Placement Plausibility & Intersections:** Are objects placed stably and naturally on surfaces (e.g., not floating)? Are there any signs of unnatural intersection or penetration between objects? Objects should appear physically grounded and interact realistically.

- - High score (7-10): All objects rest naturally and stably on surfaces. No aphysical interactions or intersections are visible. Objects are well-supported.
- - Mid score (4-6): Most objects are placed plausibly, but there might be minor, subtle issues like slight floating or minimal, non-critical intersections.
- - Low score (1-3): Obvious or frequent issues with object placement. Objects float, unnaturally overlap, or clearly penetrate each other, indicating a lack of physical realism.

- 4. **Layout Coherence & Realism:** Does the overall arrangement look highly functional, convincingly realistic, and typical for the task context? Does it avoid being overly staged, unnaturally sparse, or chaotically cluttered?

- - High score (7-10): Layout is highly functional, convincingly realistic, and well-suited for the described task. The scene feels authentic and natural.
- - Mid score (4-6): The layout is generally coherent and functional, but might lack some fine-tuning for optimal realism or could appear somewhat staged.
- - Low score (1-3): Layout is chaotic, illogical, too empty, overly cluttered, or looks clearly artificial and unrealistic for the task.

- 5. **Object Visibility:** Are important objects mentioned in the task easily and unambiguously identifiable in at least one of the views? Are they sufficiently well-lit and resolved?

- - High score (7-10): All key objects are clearly and unambiguously visible and identifiable. Their details are well-resolved.
- - Mid score (4-6): Most important objects are visible, but some might require closer inspection to identify.
- - Low score (1-3): Critical objects are very difficult to identify, or completely missing from view, hindering task understanding.

**Task Description:**

{task_description}

**Please provide a single score (1-10) for each criterion.**

**Strictly output in the following JSON format with no additional text:**

{

"Evaluation": [ {

"criterion": "Consistency with Task", "explanation": "Your detailed explanation here", "score": X

}, {

"criterion": "Object Size Reasonableness", "explanation": "Your detailed explanation here", "score": X

}, {

"criterion": "Placement Plausibility & Intersections", "explanation": "Your detailed explanation here", "score": X

}, {

"criterion": "Layout Coherence & Realism", "explanation": "Your detailed explanation here", "score": X

}, {

"criterion": "Object Visibility", "explanation": "Your detailed explanation here", "score": X

} ]

}

**Where X is an integer score from 1 to 10. Do not output anything else.**

###### F.4. Holodeck-Table Prompt

Table Prompt: You are an experienced indoor designer, focusing on optimizing space and arranging objects on the tabletop appropriately. Please assist me in crafting a tabletop. Each table is a rectangle. You need to define the four coordinates. Note: the units for the coordinates are meters. For example:

- • Computer Desk: [(0, 0), (0, 0.8), (0.5, 0.8), (0.5, 0)]
- • Writing Desk: [(0, 0), (0, 0.6), (0.4, 0.6), (0.4, 0)]

Here are some guidelines:

- 1. A table’s size ranges from 0.5m to 2m in length or width. The maximum area is 4 m2. Provide a tabletop within this range.
- 2. Desktop types include, but are not limited to, computer desk, writing desk, dining table, coffee table, study desk, and bar table.

Now, I need a design for {input}. Additional requirements: {additional_requirements}. Your response should be direct and without additional text at the beginning or end.

Object Selection Prompt: You are an experienced indoor designer, focusing on optimizing space and arranging objects on the tabletop appropriately. Please assist me in selecting objects to decorate the table. Provide a description and desired size for each object in JSON format:

{

"object_name": { "description": "A short sentence describing the object.", "size": [length, width, height], "quantity": number, "variance_type": "same" or "varied"

},

... }

For example:

{

"Flower Vase": { "description": "A clear glass vase filled with fresh flowers.", "size": [10, 10, 20], "quantity": 1, "variance_type": "same"

},

... }

Currently, we are working on the {TABLE_TYPE} with a size of {TABLE_SIZE}. Please also consider the following additional requirements: {additional_requirements}.

Object Constraints Prompt: You are an experienced indoor designer, focusing on optimizing space and arranging objects on the tabletop appropriately. Help me arrange objects on the tabletop by assigning constraints to each object:

- 1. Global constraint:

- • edge: at the edge of the table.
- • middle: not close to the edge of the table.

- 2. Distance constraint:

- • near, object: near to another object, distance < 15cm.
- • far, object: far away from another object, distance >= 15cm.

- 3. Position constraint:

- • in front of, object: in front of another object.
- • around, object: around another object.
- • side of, object: on the side (left or right) of another object.
- • left of, object: to the left of another object.
- • right of, object: to the right of another object.

- 4. Alignment constraint:

• center aligned, object: align the center of the object with the center of another object.

- 5. Rotation constraint:

• face to, object: face towards the center of another object.

For each object, provide one global constraint and select from the other constraint types to ensure an optimal arrangement. Format each constraint as: object | global constraint | constraint 1 | constraint 2 | ...

Here are some guidelines:

- • Start with an anchor object that does not depend on other objects.
- • Place larger objects first and ensure objects of the same type are aligned.

Now, design {table_type} with a table size of {table_size}. Here are the objects I want to place on the {table_type}: {objects} Please explain your high-level design strategy first, then strictly follow the desired format for providing constraints (do not add any additional text at the beginning or end).

- F.5. I-Design-Table Prompt

Initiate Prompt: The table has the size [table length]m x [table width]m x [table height]m User Preference (in triple backquotes): [user input] Table layout elements on the table (in triple backquotes): [’north_edge’, ’south_edge’, ’west_edge’, ’east_edge’, ’middle_of_table’]

Refiner Prompt: Context: We are refining the placement of a cluster of objects relative to each other. These objects are all children of a parent object. Parent Object ID: [parent_id] Children Object IDs in this cluster: [obj_names] The children objects are all positioned ’[prep]’ the parent object ’[parent_id]’. [Insert possibilities string] Task: Define the spatial relationships (e.g., “left of”, “right of”, “in front of”, “behind”) BETWEEN the children’s objects listed above. The goal is to arrange them neatly and logically within their shared space relative to the parent.

###### Output Format:

Your response MUST be a JSON object (presented plainly, without delimiters). The JSON object must have a single key "children_objects", which is a list. Each item in the "children_objects" list must be a dictionary representing one of the children.

Each child’s dictionary must contain:

- • "name_id": the ID of the child object itself (from the list).
- • "placement": a dictionary describing placement relative to other children in the same cluster:

– "children_objects": a list of dictionaries, each defining a relationship: ∗ "name_id": the ID of the other child. ∗ "preposition": the spatial preposition (e.g., "left of", "right of"). ∗ "is_adjacent": true or false, whether they are adjacent.

Example:

{

"children_objects": [ {

"name_id": "apple", "placement": {

"children_objects": [ {

"name_id": "banana", "preposition": "left of", "is_adjacent": true

} ]

} }

] }

Layout Refiner: Every time the Admin speaks, look at the parent object (e.g., a table or a book) and its children objects (e.g., cup, pen, phone). Identify the first preposition connecting them, and then suggest a second relative positioning among the children. Use the JSON schema below:

{

"children_objects": { "type": "array", "items": {

"type": "object", "properties": {

"name_id": { "type": "string"

}, "placement": {

"type": "object", "properties": {

"children_objects": { "type": "array", "items": {

"type": "object", "properties": {

"name_id": { "type": "string", "description": "The name_id of the other child object "

}, "preposition": {

"type": "string", "description": "e.g. left of the cup, behind the phone", "enum": ["on", "left of", "right of", "in front", "behind", "under", "above"]

}, "is_adjacent": {

"type": "boolean", "description": "Whether the objects are adjacent"

}

}, "required": ["name_id", "preposition", "is_adjacent"]

} }

}, "required": ["children_objects"]

}

}, "required": ["name_id", "placement"]

} }

}

###### F.6. Reasoning Data Construction Prompt

Task from Scene Graph: As an embodied task planner, analyze the given scene graph to generate diverse robotic manipulation tasks. Follow these guidelines: Task Requirements:

- • Propose one high-level tasks combining primitive actions
- • Assume you are a human, you can command the assistant to do the task
- • The proposed tasks should be as diverse as possible
- • Each task must involve at least 3 objects with spatial reasoning. (The number of objects should be as many as possible, but if objects in scene graph are less than 3, only use the objects in scene graph)
- • Use object types (e.g., "Dinner Plates") not instance IDs (e.g., "Dinner Plates-0")
- • Consider object states (open/closed, filled/empty, etc.) and spatial relationships (near/far, left/right)
- • Consider object affordance (e.g., container, cut, etc.)
- • If there is a "in" or "above" relation, consider design the sub-goal to take the object out of the container or from the top of the object.
- • Consider the complicated action, like put the object on the top of the object, or put the object in the container.

Action Constraints: You should consider the primitive actions the robot arm could take:

- • Pick(obj_name)
- • PlaceOn(obj_name)
- • PlaceAt(position)
- • Push(obj_name)
- • RevoluteJointOpen(obj_name)
- • RevoluteJointClose(obj_name)
- • PrismaticJointOpen(obj_name)
- • PrismaticJointClose(obj_name)
- • Press(obj_name)

Output Structure: For each task, provide results:

{

"Environment": "Brief scene description", "Task": "High-level abstract task (less than 20 words)", "Goal": [

"Ordered sequence of sub-objectives",

], "Action Sequence": [

"Primitive actions with parameters (e.g., Pick(Dinner Plate))"

], "Objects cluster": ["List of involved object types"]

}

The two given images are the front view and perspective view of the tabletop (corresponding to the scene graph), you can use the images to help you generate reasonable tasks. The given tabletop description can be used to help you understand the scene and generate more reasonable and diverse tasks.

Table Description: The given images are the front view (in which the table is symmetric) and the perspective view of a tabletop. Only describe the layout on the tabletop; do not include the description of the table. Please describe the scene in detail, including the objects (**do not describe the color/material/color of the object**), their positions, and the overall layout of the desktop, including empty and densely packed areas Output the description in a paragraph, no more than 200 words.

Reasoning Context Generation: Objective: Generate a reasoning process explaining the initial object arrangement for the task. Start by identifying objects, inferring context, and then providing placement reasoning as a paragraph, concluding with a transition to the scene graph. Ensure consistency with the reference scene_graph without explicitly mentioning the comparison. Input:

- 1. task_goal_object: Describes the task, goals, actions, and core objects.
- 2. scene_graph: Provides the ground truth layout for internal consistency reference only.

Output: Structure the output as follows:

- 1. Introductory Sentence: Briefly state the goal is to determine the setup based on the task.
- 2. Scene Context Inference: Briefly infer the environment type based on the objects and task described in task_goal_object (e.g., "The task suggests a typical office desk setting.").
- 3. Core Task Objects: Use the exact header Core Task Objects: followed by a list of object types and counts (no instance names, derived from task_goal_object["Objects"]).
- 4. Environment Objects: Use the exact header Environment Objects: followed by a list of other object types and counts present in the context (no instance names, derived from scene_graph). Their presence should align with the inferred scene context. Avoid explicitly mentioning verification against the scene graph.
- 5. Placement Reasoning Paragraph: A coherent paragraph of detailed scene description, including the objects, their positions, and the overall layout of the tabletop inferred from the given environment/Action Sequence/Goal in task info. Consider accessibility, non-interference, and the overall plausible layout. Use the input reference scene_graph, input scene images, scene description internally to ensure resulting locations mentioned (e.g., ’middle_left’) are correct, but do NOT state this comparison explicitly, and do NOT simply list relationships from scene_graph. Reference to the input scene description and scene images to make the paragraph more detailed and accurate.
- 6. Concluding Transition Sentence: End the paragraph with a transition sentence leading into the scene graph description (e.g. "Based on this task analysis, the scene graph is arranged as follows:"). Avoid generic evaluative summaries.

— Input Data Task/Goal/Object Description:

##### {task_goal_object}

