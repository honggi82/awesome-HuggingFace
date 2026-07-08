# arXiv:2409.02877v1[cs.AI]4Sep2024

## Configurable Foundation Models: Building LLMs from a Modular Perspective

Chaojun Xiao1, Zhengyan Zhang1, Chenyang Song1, Dazhi Jiang1, Feng Yao2, Xu Han1∗ Xiaozhi Wang1, Shuo Wang1, Yufei Huang1, Guanyu Lin3, Yingfa Chen1, Weilin Zhao1 Yuge Tu4, Zexuan Zhong6, Ao Zhang7, Chenglei Si8, Khai Hao Moo4, Chenyang Zhao9 Huimin Chen1, Yankai Lin5, Zhiyuan Liu1,4∗, Jingbo Shang2, Maosong Sun1∗

1Tsinghua University, 2University of California San Diego, 3Carnegie Mellon University 4ModelBest Inc., 5Renmin University of China, 6Princeton University 7National University of Singapore, 8Stanford University, 9University of California, Los Angeles xiaocj20@mails.tsinghua.edu.cn {hanxu2022,liuzy,sms}@tsinghua.edu.cn

#### Abstract

Advancements in large language models (LLMs) have recently unveiled challenges tied to computational efficiency and continual scalability due to their requirements of huge parameters, making the applications and evolution of these models on devices with limited computation resources and scenarios requiring various abilities increasingly cumbersome. Inspired by modularity within the human brain, there is a growing tendency to decompose LLMs into numerous functional modules, allowing for inference with part of modules and dynamic assembly of modules to tackle complex tasks, such as mixture-of-experts. To highlight the inherent efficiency and composability of the modular approach, we coin the term brick to represent each functional module, designating the modularized structure as configurable foundation models. In this paper, we offer a comprehensive overview and investigation of the construction, utilization, and limitation of configurable foundation models. We first formalize modules into emergent bricks - functional neuron partitions that emerge during the pre-training phase, and customized bricks - bricks constructed via additional posttraining to improve the capabilities and knowledge of LLMs. Based on diverse functional bricks, we further present four brick-oriented operations: retrieval and routing, merging, updating, and growing. These operations allow for dynamic configuration of LLMs based on the instruction to handle complex tasks. To verify our perspective, we conduct an empirical analysis on widely-used LLMs, Llama-3-8B-Instruct and Mistral-7B-Instruct-v0.3. We find that the FFN layers follow modular patterns with functional specialization of neurons and functional neuron partitions. Finally, as the domain of configurable LLMs remains nascent and evolving, we highlight several open issues and directions for future research, including the correlation between emergent and customized bricks, general brick development protocols, evaluation of configurable LLMs, efficient brick computing frameworks, and systems consisting of multiple model-level bricks. Overall, this paper aims to offer a fresh modular perspective on existing LLM research and inspire the future creation of more efficient and scalable foundational models.

“Rome was not built in a day, but they were laying bricks every hour.”

— John Heywood

∗Corresponding authors.

#### Contents

- 1 Introduction 4
- 2 Configurable Foundation Models 9

- 2.1 Emergent Bricks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 2.1.1 Observations on Parameter Differentiation . . . . . . . . . . . . . . . . . . . . . . . 9
- 2.1.2 Human-Defined Emergent Bricks . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 2.1.3 Self-Organized Emergent Bricks . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- 2.2 Customized Bricks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

- 2.2.1 Observations on Intrinsic Dimension of LLMs . . . . . . . . . . . . . . . . . . . . 12
- 2.2.2 Typical Customized Bricks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 2.2.3 Task Bricks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 2.2.4 Knowledge Bricks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 2.2.5 Modality Bricks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

- 2.3 Brick Granularity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 2.3.1 Solitary Neuron Granularity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 2.3.2 Neuron Group Granularity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 2.3.3 Layer Granularity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 2.3.4 Full Model Granularity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 2.3.5 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- 2.4 Benefits of Configurable Bricks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- 3 Operations for Configurable Bricks 19

- 3.1 Routing and Retrieval . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 3.1.1 Emergent Brick Routing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 3.1.2 Customized Brick Retrieval . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 3.1.3 Routing and Retrieval Granularity . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- 3.1.4 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- 3.2 Combination . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- 3.2.1 Parameter Weighted Averaging . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- 3.2.2 Brick Stitching . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- 3.2.3 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

- 3.3 Updating . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

- 3.3.1 Locating and Updating Emergent Bricks . . . . . . . . . . . . . . . . . . . . . . . . 24
- 3.3.2 Injecting New Customized Bricks . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- 3.3.3 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

- 3.4 Growing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- 3.4.1 Growing for Pre-Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- 3.4.2 Growing for Post-Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- 3.4.3 Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

##### 4 Empirical Analysis 27

- 4.1 Functionality Localization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- 4.2 Experimental Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- 4.3 Sparse Activation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- 4.4 Functionality Specialization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- 4.5 Functionality Partition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

##### 5 Open Problems and Future Directions 33

- 5.1 Correlation between Emergent and Customized Bricks . . . . . . . . . . . . . . . . . . . . 33
- 5.2 Brick Construction Protocols . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- 5.3 Evaluation of Configurable Foundation Models . . . . . . . . . . . . . . . . . . . . . . . . 34
- 5.4 Efficient Brick Computing Frameworks . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- 5.5 Multi-Model Cooperation System . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35

##### 6 Conclusion 36

#### 1 Introduction

Large pre-trained models, especially large pre-trained language models (LLMs), have achieved remarkable success in a variety of tasks (Devlin et al., 2019; Brown et al., 2020; Han et al., 2021a; OpenAI, 2023; Touvron et al., 2023b; Zhao et al., 2023b). LLMs have become the foundation models of artificial intelligence applications by providing amounts of world knowledge (Petroni et al., 2019; Shin et al., 2020) and powerful reasoning capabilities (Bommasani et al., 2021). Current advanced LLMs, such as GPT-4 (OpenAI, 2023), are deployed on large-scale central servers with high-bandwidth memory and GPUs to address various user instructions. With the development of LLMs, the future applications of LLMs will inevitably face the following trends, which in turn present challenges for LLMs:

(1) Deployment on end devices. With the capabilities of LLMs continuing to improve, the trend of deploying these models on devices with limited computing power, such as smartphones and personal computers, is attracting increasing attention, allowing LLMs to serve as personal assistants for millions of users (Apple, 2024; Xue et al., 2024; Hu et al., 2024). The use of monolithic LLMs that require substantial computational resources is gradually becoming infeasible, and improving the computational efficiency of LLMs is a significant challenge. (2) Widespread application across multiple domains. LLMs are widely applied in various fields and applications to enhance people’s work efficiency (Kaddour et al., 2023). However, the knowledge and capabilities required by different domains, users, and even different instructions vary greatly. Storing all world knowledge in monolithic LLMs and serving all scenarios with full parameters often leads to redundant computations, and conflicts between different domain knowledge may even result in sub-optimal performance. (3) Rapid evolution in new scenarios. As application scenarios increase and time progresses, we usually need LLMs to efficiently adapt to new tasks and learn from environment feedbacks (Li et al., 2024; Tao et al., 2024). Meanwhile, the world knowledge stored in LLMs is constantly updating and expanding. This demands that LLMs are able to evolve efficiently and continuously, learning new knowledge and skills while avoiding forgetting existing knowledge.

To address these issues, studying and analyzing LLMs from a modular perspective has gradually become an important focus for current researchers (Pfeiffer et al., 2023; Fedus et al., 2022a; Ding et al., 2023; Zhang et al., 2023c). These works decompose LLMs into functional modules. In this way, for each computation step, we can only involve part of modules to save computation costs and achieve efficient continual training by constructing or updating modules. Modularity has long been an endogenous mechanism or a central principle in diverse fields, ranging from biomedical sciences to engineering fields (Simon, 1962). A module is conceived as an independent component with specialized functionality that can coordinate and interface with other modules, thereby giving rise to complex systemic behavior. Owing to modularity, many complex systems become more understandable and scalable. For instance, in cognitive neuroscience, the modularity of mind

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Figure 1: The illustration of a configurable foundation model consisting of emergent and customized bricks. For a given instruction, we select and combine tiny bricks to build an efficient instruction-specific model with minimal performance loss.

hypothesis posits that the human brain comprises functionally specialized modules, such as the visual cortex for processing visual inputs (Grill-Spector & Malach, 2004) and Broca’s area for speech production (Fodor, 1983). These modules operate on distinct types of information while collaborating to generate integrated cognition and behavior. In software engineering, decomposing programs into logical modules can significantly enhance development efficiency, reduce project complexity, and encourage code reuse (Stefik & Bobrow, 1986). In industrial manufacturing, items like electrical appliances and automobiles are also produced by assembling modular components (Bejan & Lorente, 2006).

The core characteristics of modularity are independence, specificity, and composability. Modules are decoupled from each other and can thus be specified to focus on certain functionality. By constructing and composing modules, complex systems can be built and maintained more easily. Owing to modularity, it becomes possible to simplify the systems and enhance their efficiency. It also allows for a streamlined process in developing and updating systems, ensuring they remain adaptable and scalable over time. Inspired by existing observations in other disciplines, modularity is becoming a promising conceptual perspective for designing and analyzing the next generation of foundation models (Zador et al., 2022). Some preliminary efforts indicate that LLMs have the potential to be decomposed into various specialized functional modules. For example, we can find that LLMs adopt single neurons as memory modules to store a specific structured knowledge triplet like (China, Capital, Beijing) (i.e., Beijing is the capital of China) (Dai et al., 2022; Sajjad et al., 2022; Meng et al.,

- 2022). Recent efforts in parameter-efficient tuning have demonstrated that constructing modules with several dozen neurons can equip LLMs with specific task abilities (Ding et al., 2023; Hu et al., 2022; Houlsby et al., 2019; Li & Liang, 2021). To augment LLMs with multi-modal processing capabilities, some researchers attempt to adopt visual models as the modules of LLMs to analyze visual semantics (Alayrac et al., 2022; Xu et al., 2023a). Therefore, building LLMs with modules from the neuron level to the model level can significantly enhance the abilities of LLMs without costly re-training from scratch. Generally, a module comprises a collection of function-specific neurons, the size of which can vary according to functional requirements. In some cases, we can even regard entire pre-trained models as a functional module, beyond the definition of a module in the general sense. To avoid ambiguity and more intuitively show the benefits of modularity for LLMs, we term the functional components to build LLMs as bricks instead of modules.

Training and deploying LLMs from the perspective of brick combination enable configurable model usage. As shown in Figure 1, for a given instruction, unlike computing the monolithic LLMs, we can select part of bricks with specific functionalities for computation according to the instruction functionality requirements (Shen et al., 2023b; Zhang et al., 2023c), which significantly benefits the computational efficiency. Besides, the adaptation and augmentation of LLMs can be further formalized as the problem of constructing new bricks or updating existing bricks (Gururangan et al., 2022; Cao et al., 2021; Xiao et al., 2023b; Zhang et al., 2023b), which are more cost-efficient and scalable than continual training of full models. Owing to the high scalability brought by brick combination, we term the LLMs built on bricks as configurable foundation models. These beneficial endeavors can well propel the utilization of LLMs in daily applications, thereby offering fresh insights and approaches for the development of next-generation architectures for foundation models. Therefore, to facilitate the progress of configurable foundation models, this paper places its emphasis on a comprehensive analysis of existing efforts, future directions, and potential challenges.

To take advantage of configurable foundation models, this paper focuses on addressing two problems:

##### Problem 1: how can we formalize and construct bricks for configurable foundation models? (§ 2)

From the micro parameter level rather than the macro model level, both pre-training and post-training essentially involve constructing bricks for LLMs. In recent years, the prevailing paradigm for building LLMs involves two steps: pre-training and post-training. Here, post-training includes fine-tuning and preference learning. During the pre-training phase, LLMs acquire versatile knowledge and learn general language reasoning through self-supervised learning on massive unsupervised data. During the subsequent post-training phase, LLMs are adapted to obtain additional capabilities, including downstream task ability, and domain knowledge (Bommasani et al., 2021). Despite the pre-training and post-training processes being conducted on monolithic LLMs, recent works indicate that the impact of these processes on the internal parameters of LLMs tends to be modular (Zhang et al., 2023c; Aghajanyan et al., 2021).

Pre-training is to construct emergent bricks, the functional specialization of which emerges during pre-training. (§ 2.1) It includes both dense and sparse pre-training processes. (1) Prior analyses on the typical dense pre-training of LLMs reveal that model parameters undergo differentiation throughout the process (Dai et al., 2022; Foroutan et al., 2022; Dobs et al., 2022; Zhang et al., 2023c). This dynamic and implicit process leads to the emergence of functional partitions, giving rise to an implicit brick structure. Notably, Zhang et al. (2023c) identify distinct groups of parameters within LLMs dedicated to semantic, knowledge, and task

Table 1: The definition of concepts coined in this paper.

Concept Definition

Configurable Foundation Model Refers to a foundation model composed of multiple functional components, derived from decomposing LLMs. Upon receiving specific instructions, a configurable foundation model can dynamically select and combine these components as required for compositional abilities, thereby offering configurable functionality.

Brick Refers to a functional unit composed of a group of function-specific neurons. Its size can vary from a single neuron to even an entire model. Compared to the traditional definition of “module”, the brick demonstrates a richer granularity and functionality.

Emergent Brick Refers to a functional brick formed within an LLM during the pre-training process, where randomly initialized parameters gradually evolve and differentiate into specific functions. Parameters with similar functions constitute these emergent bricks.

Customized Brick (Plugin) Refers to a functional brick specifically designed and trained to meet the needs of downstream applications. These bricks or plugins can be tailored for specific tasks, knowledge graphs, or various modalities.

functions. Foroutan et al. (2022) uncover the presence of language-neutral sub-networks for multilingual foundation models. Moreover, using the pruning technique, previous works have explored to discover taskspecific subnetworks from LLMs (Frankle & Carbin, 2019; Wang et al., 2020; Xia et al., 2022). (2) Some recent efforts have attempted to build LLMs with the sparse mixture-of-expert (MoE) structure (Shazeer et al., 2017; Lepikhin et al., 2021; Fedus et al., 2022b,a). In the MoE structure, an LLM is composed of multiple experts, each of which has the same architecture as the feed-forward network or the attention network in the original model. Only a subset of experts is activated at a time by a gating network to process input data. While the primary intent of MoE methods is to enhance model capacity without escalating computational cost, these methods invisibly formalize LLMs into a pre-defined structure akin to combining bricks, where each expert functions as a distinct and specialized brick. Further studies show that the MoE structure can yield comparable results to conventional dense structures (MistralAI, 2023) and even prove more advantages for understanding task-specific instructions due to the functional specialization (Shen et al., 2023a). Based on such evidence, whether we intentionally or unintentionally design the structure of LLMs, bricks are spontaneously formed to target specific functions during the pre-training process.

Post-training essentially is to construct additional customized bricks for the whole model, the functional specialization of which is manually defined to meet specific human-defined requirements. (§ 2.2) To enhance models with additional abilities, such as domain knowledge and task-specific capabilities, traditional methods involve full-parameter fine-tuning. Recent research shows that parameter changes are intrinsically lowrank (Li et al., 2018; Aghajanyan et al., 2021; Qin et al., 2021), which implies that only a small proportion of the parameters necessitate tuning for further capabilities adaptation. Inspired by these findings, parameter-efficient fine-tuning (PEFT) freezes LLMs and introduces extra parameters to achieve efficient task adaptation (He et al.,

- 2022b; Ding et al., 2023; Liu et al., 2023b). Beyond PEFT, many studies find that the additional parameters can not only endow LLMs with task-specific capabilities, but also supplement them with much extra knowledge and functionalities, such as knowledge bricks for world knowledge injection (Zhang et al., 2023b; Xiao et al.,
- 2023b), modality bricks for multi-modal composition (Alayrac et al., 2022; Li et al., 2023a), memory bricks for long-text processing (Xiao et al., 2024), and compression bricks for inference acceleration (Xiao et al.,

- 2023a). Therefore, the essence of fine-tuning LLMs is to customize bricks, which can fully supplement and stimulate knowledge and capabilities for LLMs to meet specific requirements. Furthermore, each LLM itself can also become a customized brick in a multi-model system. For example, in a multi-agent system, each model is responsible for a specific sub-task (Wang et al., 2023a; Chen et al., 2023b); in a combination of multi-modal models, each model is tasked with processing data from a specific modality (Alayrac et al., 2022).

After formalizing pre-training and fine-tuning into constructing bricks, we delve into a discussion on selecting brick granularity for building configurable foundation models. This entails a thoughtful evaluation of both brick capabilities and brick management, highlighting the relationship between the granularity and the capability complexity and the inclusion among different bricks (§ 2.3). Subsequently, we further summarize five major advantages of configurable foundation models, including efficiency, reusability, traceability, sustainability, and distributed computation (§ 2.4).

Table 2: Existing representative efforts for configurable foundation models.

Topic Representative Reference

Activation Sparsity: Activation Sparsity of Neurons (Zhang et al., 2022c; Li et al., 2023c; Zhang et al., 2024b), Improving the Sparsity (Song et al., 2024a,b), Inference Infrastructure with Sparsely Activated Neurons (Song et al., 2023b; Xue et al., 2024; Alizadeh et al., 2023)

Function Localization: Knowledge (Dai et al., 2022; Meng et al., 2022), Skill (Wang

Emergent Bricks

- et al., 2022a; Ackermann & Ohmer, 2023; Panigrahi et al., 2023), Linguistic (Gurnee
- et al., 2023; Voita et al., 2023; Tang et al., 2024; Zhao et al., 2023a)

Human-Defined Emergent Bricks: Human-Defined Layers (Rogers et al., 2020; Geva et al., 2021), Mixture-of-Expert (Shazeer et al., 2017; Fedus et al., 2022b)

Self-Organized Emergent Bricks: Neuron Grouping in FFNs (Zhang et al., 2022c; Piórczynski et al., 2023; Zhang et al., 2023c; Dong et al., 2024)

Intrinsic Dimensionality: Intrinsic Dimension of LLM Fine-tuning (Aghajanyan et al., 2021; Qin et al., 2021; Zhang et al., 2023d)

Architecture

Task Bricks: Adapter (Houlsby et al., 2019), Prompt (Liu et al., 2023b; Lester et al., 2021), Prefix Tuning (Li & Liang, 2021), LoRA (Hu et al., 2022), BitFit (Zaken et al., 2022), Steering Vector (Zou et al., 2023; Liu et al., 2023c)

Knowledge Bricks: Knowledge Graph Brick (Zhang et al., 2023b; Pörner et al., 2020; Zhang et al., 2024a), Document Brick (Xiao et al., 2023b; Gim et al., 2024)

Customized Bricks

Modality Bricks: Modality Bricks with Textual Interface (Li et al., 2023b; Wu et al., 2023a; Shen et al., 2023c; Yang et al., 2023), Modality Bricks with Continuous Interface (Alayrac et al., 2022; Li et al., 2023a; Liu et al., 2023a; Zhu et al., 2023; Luo et al., 2023; Wu et al., 2023b)

Other Customized Bricks: Bricks for Tool Using (Shi et al., 2023; Yu et al., 2023; Hao et al., 2023b), Bricks for Debiasing (Dathathri et al., 2020), Bricks for Acceleration (Xiao et al., 2023a), Bricks for Style Transfer (Pascual et al., 2021; Dathathri et al., 2020)

Routing for Emergent Brick: Trainable Routing (Fedus et al., 2022b; Lewis et al., 2021; Zhou et al., 2022a; Puigcerver et al., 2023; Qiu et al., 2024), Fixed Router (Roller et al., 2021; Zuo et al., 2022a; Gururangan et al., 2022)

Routing and Retrieval

Retrieval for Customized Brick: Knowledge Brick Retrieval (Friedman et al., 2021; Pfeiffer et al., 2021; Huang et al., 2023a), Task Brick Retrieval (Zhao et al., 2024b)

Parameter Weighted Averaging: Ensemble of Bricks with the Same Ability (Garipov et al., 2018; Wortsman et al., 2022; Qin et al., 2022a; Chronopoulou et al., 2023; Ruan et al., 2023; Arpit et al., 2022; Ramé et al., 2022; Li et al., 2022), Composition of Bricks with Different Abilities (Matena & Raffel, 2022; Jin et al., 2023; Ilharco et al., 2023; Zhang et al., 2023a; Daheim et al., 2023)

Combination

Brick Stitching: Heuristic Stitching (Alayrac et al., 2022; Li et al., 2023a; Wang et al., 2023a; Pan et al., 2023; Akiba et al., 2024), Planner-based Stitching (Andreas et al., 2016; Hu et al., 2017; Fashandi, 2023; Yao et al., 2023a; Gao et al., 2023)

Operation

Locating and Updating Emergeng Bricks: Knowledge Locating (Geva et al., 2021; Dai et al., 2022; Sundararajan et al., 2017; Meng et al., 2022; Hase et al., 2023), Knowledge Updating (Zhu et al., 2020; Meng et al., 2022, 2023; Onoe et al., 2023; Padmanabhan et al., 2023; Cao et al., 2021; Mitchell et al., 2022a)

Updating

Injecting New Customized Bricks: Plug-and-Play Knowledge Injection (dos Santos et al., 2022; Huang et al., 2023c; Mitchell et al., 2022b; Hernandez et al., 2023; Turner et al., 2023; Zou et al., 2023)

Growing for Pre-training: Progressive Dense Parameter Growing (Gong et al., 2019; Gu et al., 2021; Qin et al., 2022b; Wang et al., 2023c; Wu et al., 2024), Progressive Sparse Expert Growing (Li et al., 2022; Komatsuzaki et al., 2023; Shen et al., 2023b; Wang et al., 2024)

Growing

Growing for Post-training: Multitask Continual Learning (Mahabadi et al., 2021; Wang et al., 2022c,b; Razdaibiedina et al., 2023; Madotto et al., 2021; Song et al., 2023a)

- Problem 2: how can we leverage existing bricks to build configurable foundation models for the everincreasing complex requirements of real-world tasks? (§ 3)

Bricks consist of a collection of parameters with specialized functionalities. In real-world tasks, singular knowledge and capabilities frequently fall short of meeting task requirements, implying that we need to combine multiple bricks to understand instructions and accomplish specific tasks effectively. In this paper, we summarize four primitive operations on bricks and argue that through the composition of these primitive operations, configurable foundation models can be built based on bricks to fulfill complex task instructions efficiently. The operations are summarized as follows:

- • Routing and Retrieving. The routing and retrieving operation involves the dynamic selection of specific bricks from a brick repository based on instructions. This operation acts as a dynamic brick gatekeeper and allows the configurable foundation model to adapt its brick composition to the instruction at hand. (§ 3.1)
- • Combining. The combining operation involves the synergistic integration of multiple bricks, facilitating collaborative and comprehensive processing. This can be achieved by directly merging isomorphic bricks to enhance abilities, or by simultaneously inserting multiple bricks to create composite skills. This operation empowers the model to harness the collective capabilities of various bricks, facilitating the creation of informative responses. (§ 3.2)
- • Updating. The updating operation involves the refinement and adaptation of bricks over time, based on new knowledge and feedback. This operation enables bricks to be fine-tuned or adjusted to improve their performance continually. It also empowers foundation models to adapt and remain pertinent in dynamic real-world scenarios. (§ 3.3)
- • Growing. The growing operation pertains to the expansion of the brick repository itself. New bricks with specialized functionalities can be added to the repository to address emerging requirements. By incorporating new modules, configurable models can keep up with the increasing complexity of real-world applications and offer effective solutions to a broader range of challenges. (§ 3.4)

Besides the above two important problems, in this paper, we conduct empirical analysis on widely-used LLMs to investigate whether these existing well-trained LLMs exhibit functional partitioning similar to the human brain. The experimental results from two models, Llama-3-8B-Instruct and Mistral-7B-Instruct-v0.3, demonstrate that: (1) Neuron activation is sparse, meaning that processing each instruction requires only a small subset of neurons. (2) Neurons are specialized for specific functionalities, with the removal of these neurons having minimal impact on other capabilities. (3) There is evidence of neuronal partitioning, indicating that different capabilities require distinct sets of neurons.

In the end, we discuss the future research directions for the application of configurable foundation models (§ 5), including:

- • Analyzing the correlation between emergent and customized bricks: Here, we focus on delineating roles between emergent and customized bricks, as well as identifying and handling knowledge conflicts and redundancies arising from their interaction. (§ 5.1)
- • Unifying the protocol to construct bricks: We engage in a discourse on a novel paradigm for developing foundation models, wherein the shift moves from training a whole model to training individual bricks. This envisioned paradigm entails a shared core foundation model for open-source communities, enabling individuals to develop their bricks based on a unified protocol and openly share bricks for collective utilization. (§ 5.2)
- • Evaluating configurable models: This facet centers on evaluating the foundation models from the perspective of bricks and discussing the evaluation metrics for configurable bricks. (§ 5.3)
- • Implementing the framework for efficient computing: Our deliberation encompasses the foundational computational operators of configurable models, characterized by sparsity and computational decoupling. Additionally, we delve into the prospects of distributed center-edge computing frameworks for configurable foundation models. (§ 5.4)
- • Combining multiple model-level bricks for composite capability: In the rapidly evolving AI community, a vast array of large pre-trained models has been open-sourced, which can serve as model-level bricks for completing complex instructions. We discuss the potential and challenges for scalable multi-model cooperation systems. (§5.5)

In summary, we present the concepts coined in this paper in Table 1 and present existing representative efforts for configurable foundation models in Table 2. We aspire for our paper to serve as an inspiration for future researchers, driving forward the progress of efficient and scalable foundation models.

#### 2 Configurable Foundation Models

In this section, we elaborate on the general framework for configurable foundation models, consisting of various bricks. These bricks encompass both the emergent bricks from the pre-training process and customized bricks from post-processing to enhance LLMs. Specifically, we first present that the pre-trained LLMs naturally possess the property of modularity and can be split into bricks with pre-defined structures or self-organized functional neuron clusters (§ 2.1). Then, in the pursuit of advancing LLMs, it is promising to parameterize the external knowledge and capacities into neural bricks, which can be inserted into LLMs in a plug-and-play manner (§ 2.2). Subsequently, we discuss how to select the granularity of bricks to trade off efficiency and effectiveness (§ 2.3). Lastly, we present five benefits to constructing LLMs with configurable bricks, including high efficiency, reusability, traceability, sustainability, and distributed computation (§ 2.4).

##### 2.1 Emergent Bricks

The emergent property of modularity has been observed in the pre-training process of language models (Wang et al., 2022a; Zhang et al., 2023c), which indicates that a subset of the parameters can function properly as the entire model for specific instructions. Such property makes it possible to break down the gigantic LLMs, including both dense models (Brown et al., 2020; Touvron et al., 2023a) and sparse models (Lepikhin et al., 2021; Fedus et al., 2022b), into tiny modules. With the breakdown, the aforementioned issues of efficiency and scalability can be tackled via module dropping (Fan et al., 2020; Zeng et al., 2023; Liu et al., 2023d), subnetwork extraction (Frankle & Carbin, 2019; Wang et al., 2020; Xia et al., 2022), and recombination (Gururangan et al., 2022; Zhang et al., 2022c). We term these modules directly broken down from the pre-trained models as emergent bricks, which acquire certain capabilities of the entire model from the pre-training process. In this subsection, we summarize the potential inspirations for emergent bricks and introduce two different categories of emergent bricks, including bricks with human-defined and self-organized structures. The discussion may boost our understanding of the working mechanism inside the LLMs and help us better configure the LLMs with various internal modules.

##### 2.1.1 Observations on Parameter Differentiation

LLMs tend to be over-parameterized when performing some specific tasks (Houlsby et al., 2019; Ding et al., 2023), which indicates that there exists a sub-module functioning nearly the same as the entire model with the rest parameters being redundant. This over-parameterization phenomenon leads to two general questions: (1)

“Which part of the model is actually functioning?” (2) “What kind of ability does it have?”. In this subsection, we discuss existing observations about the functional specialization of internal parameters in LLMs, that is, each parameter is only responsible for a specific function.

Activation Sparsity Inspired by the sparsity in human brains (Olshausen & Field, 1996; Kerr et al., 2005; Poo & Isaacson, 2009; Barth & Poulet, 2012) that only a small portion of the neurons activate at each time, special architectures such as sparsely-activated Mixture-of-Experts (MoE) are introduced into Transformers to enforce activation sparsity and thus improve model efficiency (Du et al., 2022; Rajbhandari et al., 2022; Jaszczur et al.,

- 2021; Fedus et al., 2022b,a). Different from the sparsity of expert activation in MoE, researchers also explore the activation sparsity of model neurons, which is in finer granularity. Specifically, the neuron “activation” refers to the intermediate output of the fully connected layer after the non-linear activation function, and “sparsity” indicates that only a few entries of the activation values are nonzero for each given input. Zhang

- et al. (2022c) inspect the computational pattern of pre-trained Transformers and find that the activation sparsity naturally exists in pre-trained dense Transformers. Specifically, they delve into the feed-forward networks (FFNs), which constitute two-thirds of the Transformer model parameters, and find the emergence of sparse activation (e.g., only around 5% of the neurons are with nonzero activation values for 90% of the input for a fine-tuned T5-Large model (Raffel et al., 2020)). Li et al. (2023c) comprehensively investigate sparse activation in Transformers and conclude that it is a ubiquitous phenomenon that emerges for both natural language and vision models, on both training and evaluation data, on datasets of varying scale, on Transformers of varying configurations, and across all layers of a Transformer. Although the above works focus on the sparsity within ReLU-based models, an increasing number of modern LLMs have been trained with non-ReLU activations, and it is more tricky to explore activation sparsity for them since there are typically many near-zero but nonzero

| | | | |
|---|---|---|---|
| | | | |

| | | | |
|---|---|---|---|
| | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |

Random Initialization

Functional Differentiation

Emergent Bricks

- Figure 2: The illustration for emergent bricks. In the randomly initialized, functional differentiation emerges following the pre-training phase. Neurons with similar functionalities can be aggregated to form small functional bricks. Here, the model is divided into two human-defined layer bricks, of which each is further subdivided into several self-organized expert bricks.

small activation values. However, recent research shows that these non-ReLU models may be converted to ReLU versions without major performance degradation by fine-tuning with ReLU activations (Zhang et al.,

- 2022c; Mirzadeh et al., 2023), making activation sparsity pragmatic. Moreover, Zhang et al. (2024b) shows that for non-ReLU models, there are still some neurons whose outputs are close to zero and can be discarded without performance degradation, which also indicates the existence of activation sparsity in non-ReLU models. In summary, activation sparsity refers to the phenomenon that only a small portion of weights play a role for each input, including both expert activation sparsity in MoE and neuron activation sparsity in dense models, and it is different from the sparsity in the weight matrix leading to pruning (Frankle & Carbin, 2019; Wang et al., 2020; Xia et al., 2022). Besides, activation sparsity can greatly accelerate the inference process by involving only parts of the parameters for computation (Song et al., 2023b; Xue et al., 2024).

Function Localization In addition to activation sparsity, neurons in the human brain exhibit a modular characteristic: neurons with similar functions tend to cluster together to form specific functional partition (Bullmore & Sporns, 2009; Meunier et al., 2010). Similarly, it is widely reported that substantial functions are specifically localized in a small number of parameters within pre-trained models, i.e., “neurons” or “circuits”: (1) Knowledge Neuron. Dai et al. (2022) and Meng et al. (2022) find that factual knowledge tuplets are stored in neurons of FFNs, and manipulating the activations or weights of these “knowledge neurons” can effectively edit the knowledge-related predictions of LLMs. (2) Skill Neuron. Some researchers dive into finding skill neurons, of which the activations are highly predictive of the task labels. These skill neurons are task-specific and perturbations to skill neurons can drastically impair the performance of corresponding tasks, implying that the task skills are surely localized in these neurons (Wang et al., 2022a; Ackermann & Ohmer, 2023; Panigrahi et al., 2023). (3) Linguistic Neuron. Gurnee et al. (2023) and Voita et al. (2023) study the linguistic features encoded in neurons and observe that neuron activations have correlations with a wide range of features like n-grams and positions. Zhao et al. (2023a) and Tang et al. (2024) discover language regions of LLMs that are specialized in multilingual text processing. Inspired by these observations, Zhang et al. (2023c) expand the neuron analysis to include MoE (Mixture of Experts) experts, which represent clusters of neurons. It demonstrates that these sparsely-activated experts are specialized in different functions including knowledge storing, task skills, and semantic understanding.

##### 2.1.2 Human-Defined Emergent Bricks

Neural networks are usually constructed by stacking multiple human-designed network modules at different granularity levels. For example, the original Transformer (Vaswani et al., 2017) model consists of multiple identical blocks, within which there are multi-head attention (MHA) layers and FFNs. Simultaneously, each of the MHA layers is a combination of multiple attention heads and each FFN can be viewed as a collection of single artificial neurons that can be further regrouped based on their activations. Here, the structures and connections of these stacked modules, including neurons, attention heads, and whole layers, are explicitly designed by humans. Thus, we term these bricks as human-defined emergent bricks.

After defining the structures of neural models, the next pivotal step to empower each brick with problem-solving capabilities is training, to which there are generally two main approaches:

End-to-end Training End-to-end training of the entire network, which is the most common practice in the deep learning community. In this case, the skills or abilities of each module are not explicitly defined. Existing intriguing observations find that these human-defined bricks can gradually become functionally specialized

during the end-to-end training process of the full model (Zhang et al., 2023c). For a specific task or input, only a subset of these emergent bricks are functional or informative with others being redundant, and there can be multiple granularity levels of the potential emergent bricks. Here we give three examples from top to bottom. Fan et al. (2020) show that it is possible to reduce the model computational costs by selecting only some of the layers from a pre-trained model. Michel et al. (2019) find that simply using one attention head can achieve comparable performance to the full model on certain tasks. Zuo et al. (2022b) identify the important neurons in the FFNs for a specific task and then reorganize the sub-network based on the importance of different neurons to achieve better efficiency. The intrinsic modular characteristic of these human-defined bricks ensures the effectiveness of the parameter pruning methods.

Modular Training Modular training conducts training for different modules separately, with their functionalities predefined in advance. Andreas et al. (2016) first propose the neural module networks for visual question answering, where multiple question-specific modular networks are dynamically initiated according to different reusable functional components (e.g., for recognizing dogs, and classifying colors). In this case, each of the reusable components represents a human-defined emergent brick that possesses predefined functionality. Such training paradigm also appeals to Transformer-based models, characterized by the MoE models (Gururangan

- et al., 2022; Zhang et al., 2022a), where each expert brick is responsible for one domain or one task. Based on these human-defined emergent bricks with predefined functionalities, it is cost-efficient to construct a new task-specific model by intentionally combining part of the bricks within a general-purpose pre-trained model.

Generally, human-defined emergent bricks are commonly observed and hierarchically structured, which naturally expedites the research on various brick configurations. However, there still exist inescapable obstacles to configuring foundation models with human-defined bricks: (1) The functionalities of humandefined bricks acquired by end-to-end training are hard to interpret and localize, preventing them from being clearly and effectively utilized; (2) Modular training of human-defined bricks requires delicate design of each single brick, such as the structure and scale of the modules, to ensure that each brick can effectively attain the functionality predefined by humans in advance.

##### 2.1.3 Self-Organized Emergent Bricks

Language models acquire various capabilities from the pre-training process, which are further stored as parametric knowledge within the model parameters. Observations from the sparse activation and function localization phenomenon imply that the internal parametric knowledge or capabilities are not universally distributed among the entire model, but instead are stored in a centralized manner. However, such centralized distribution of the parametric knowledge does not strictly follow the human-defined module structure, as most modern LLMs adopt the end-to-end training paradigm which intrinsically does not constrain the learning objective of each module explicitly. In this case, there must exist an implicit structure of parametric knowledge distribution that differs from the human-defined module structure and the universal distribution over the entire model. Such an implicit structure is naturally formed during the training process, where different parts of the model interact and collaborate without explicit instructions by humans. Therefore, though language models are built upon human-defined bricks, dependencies and connections between bricks also emerge during the training process, resulting in self-organized emergent bricks.

Distinct from any individual human-defined brick, the concept of self-organized bricks emerges from the interaction between multiple human-defined bricks. There have already been preliminary explorations of self-organized bricks in the literature shedding light on their characteristics and implications. For example, Zhang et al. (2022c) find that some small proportion of the neurons within the Transformer feed-forward networks tend to activate together while the rest of the neurons are inactivated, indicating that such subset of neurons are self-organized to function properly after pre-training and thus give rise to a new form of emergent bricks that is different from those pre-defined by humans. Following this finding, Piórczynski

- et al. (2023) enforce the activation sparsity and only adopt the self-organized bricks for improvement in inference efficiency. Furthermore, Zhang et al. (2023c) demonstrate that these self-organized groups of neurons possess high productivity for specific functions and any perturbations to them can lead to drastic degradation of the performance in the corresponding function, implying that such self-organized neuron clusters can serve as emergent bricks and are functionally specialized. Liu et al. (2023d) dive deeper into the concept of self-organized bricks by exploring both Transformer feed-forward networks and muti-head attention layers. Specifically, based on the discovery that the residual connections (He et al., 2016) in LLMs make token embeddings barely change across different layers, they envision input-dependent subsets of neurons in feed-forward networks and attention heads to yield performance comparable to employing the entire model. Moreover, Mirzadeh et al. (2023) introduce ReLU non-linear activation functions into the layer normalization of Transformers, which further enhances the collaboration among human-defined bricks and leads to more

compact self-organized bricks. These works suggest that the functionality of the monolithic LLMs relies on the interaction and collaboration between multiple human-defined bricks at different granularities, and these newly clustered parameters form the self-organized bricks that are specialized in certain functions.

Introducing self-organized bricks is beneficial to improving both the efficiency and interpretability of language models. First, it is possible to decompose a task-specific sub-model with minimal cost from an existing LLM based on its self-organized bricks, which are more compact compared with its human-defined bricks. For example, we can replace the conventional layer-wise pruning with a more concise selection of functional neuron subsets in the feed-forward networks to improve efficiency. Second, we can align the functionality to the self-organized bricks more flexibly than the human-defined bricks without the explicit structure constraint. Hence, the inner working mechanism of the model can be better interpreted by analyzing the status of the various self-organized bricks.

Though preliminary progress has been made in self-organized emergent bricks, there are still several unexplored aspects that demand further attention from the community: (1) Inspecting cross-layer organization: Currently investigated self-organized emergent bricks are relatively flat and are mostly constructed in parallel within homogeneous layers, whereas they can be organized across different layers. For example, the sparse activation of Transformer feed-forward layers and multi-head attention layers, as observed in Liu et al. (2023d), could correlate to each other with chained dependencies. (2) Enhancing training strategy: Though the existence of self-organized emergent bricks can be revealed in the models trained in an end-to-end manner, models with explicit modular structure still struggle to learn the modular data distribution with the conventional end-toend training algorithms (Mittal et al., 2022). Enhanced training strategies should be proposed to explicitly encourage the emergence of self-organized emergent bricks. (3) Guiding network design: Scrutinizing self-organized emergent bricks can provide valuable guidance for designing networks with improved efficiency and interpretability in the future. For example, it is possible to identify the minimum combination of bricks that are necessary for a specific task, which can be further employed as a reference for exploring the scaling law and emergent abilities of existing LLMs.

##### 2.2 Customized Bricks

A monolithic LLM can be split into several emergent bricks, even when LLM layers are trained end-to-end with fully connected neurons. As LLM parameters continue to scale, the number of emergent bricks acquired during pre-training also increases, which enables satisfactory downstream performance. However, as the world continually changes, the capacities and knowledge that models need to master are constantly being updated and expanded. For example, world knowledge contained in widely-used Wikipedia is edited and updated daily (Cao et al., 2021; Meng et al., 2022); new academic papers published on scholarly websites continuously advance domain knowledge (Gururangan et al., 2022; Jin et al., 2022); and novel tasks emerging in different application scenarios also demand ever-growing task capacities from LLMs (Wang et al., 2023b). For ease of introduction, we use the term, knowledge, to refer to both knowledge and capacities, which can be regarded as a type of abstract knowledge.

Training the whole model to incorporate new knowledge is computation-intensive and requires massive storage space. To address this issue, many efforts have been devoted into parameterizing the external knowledge as customized bricks, which can be injected into LLMs for performance promotion (Dathathri et al., 2020; Lauscher et al., 2021; Zhang et al., 2023b; Xiao et al., 2023b; Ding et al., 2023). Specifically, customized bricks are usually constructed after pre-training with the original parameters frozen. Customized bricks can serve as an external knowledge bank for LLMs. Given an instruction, we first retrieve bricks with relevant knowledge and then insert them into LLMs for better responses. Different from training LLMs to store knowledge in emergent bricks, customized bricks possess the plug-and-play characteristic for dynamic and reusable knowledge injection. Therefore, customized bricks are usually named “plugins”. In this subsection, we will first summarize the underlying reasons for the feasibility of customized bricks and introduce several typical customized bricks.

##### 2.2.1 Observations on Intrinsic Dimension of LLMs

Customized bricks aim to inject external knowledge or new task adaptations into foundation models with tiny neural modules with a few parameters. This raises a natural question: “Can the external knowledge be represented in limited parameters?”

Intrinsic Dimensionality It has been widely recognized that LLMs are highly over-parameterized (Frankle & Carbin, 2019; Liang et al., 2021; Prasanna et al., 2020), which is also the case of almost all the modern

LoRA

|Linear|
|---|

[Figure 8]

Adapter

Prefix

Attention Q K V

Anne Hathaway

It is a tragic play by Shakespeare, set in the city of Verona, Italy …

spouse

William Shakespeare

Image Decoder

Audio Decoder

Video Decoder

Romeo author_of and Juliet

Knowledge Graph

### LLM

### LLM

[Figure 9]

Image Encoder

Audio Encoder

Video Encoder

Generate a summary for Romeo and Juliet

Document

Task Bricks Knowledge Bricks Modality Bricks

- Figure 3: The illustration for three typical customized bricks, including task bricks, knowledge bricks, and modality bricks.

neural network models. This brings a natural question of what is the minimal number of parameters needed to describe the training of these models. Li et al. (2018) introduce the concept of “Intrinsic Dimension” of an objective landscape. For a given training objective landscape, its intrinsic dimension is defined as the minimal number of free variables required to well define the optimization problem, i.e., the minimal possible dimension to reparameterize the objective into. Li et al. (2018) propose to estimate the intrinsic dimension by randomly projecting the parameters of the original model into a low-dimensional subspace and observing if the random subspace contains a good enough solution for the training objective. If there is a solution, the dimension of the current subspace serves as an upper bound of the intrinsic dimension. With this method, Aghajanyan

- et al. (2021) examine the intrinsic dimension of pre-trained language models and find that the fine-tuning of PLMs has a very low intrinsic dimension (∼ 200 for RoBERTa) and pre-training implicitly minimizes the intrinsic dimension. It has also been observed that larger models tend to have lower intrinsic dimensions. Qin et al. (2021) further find that the PLM adaptations to many different tasks not only commonly have low intrinsic dimension but the many tasks can also be reparameterized into a shared universal low-dimensional subspace, which partially explains the prevalent effectiveness of foundation models and the transferability of parameter-efficient tuning (Su et al., 2022). Zhang et al. (2023d) also find that fine-tuning can be performed within a low-dimensional subspace and some outlier dimensions play an important role. The low intrinsic dimensionality and the universal existence of intrinsic subspace make us believe that adding new abilities and knowledge into foundation models with relatively small-scale customized bricks is possible.

##### 2.2.2 Typical Customized Bricks

Customized bricks have emerged as a significant avenue for enhancing LLMs during their post-processing phase, which involves the insertion of tiny bricks after the pre-training or fine-tuning procedures. This approach aims at efficiently enhancing the LLM’s customized capabilities. As shown in Figure 3, we categorize widely used bricks into three types based on their capabilities: task bricks, knowledge bricks, and modality bricks.

##### 2.2.3 Task Bricks

Task bricks, also known as parameter-efficient tuning or delta tuning (He et al., 2022b; Ding et al., 2023), are widely explored as a substitute for full-parameter fine-tuning, which usually requires substantial computational and storage costs. Task bricks adopt task adaptation by tuning only a small portion of parameters. As the number of model parameters grows, the performance gap between task bricks and full-parameter fine-tuning narrows. Consequently, in the realm of LLMs, employing task bricks for adaptation has become a widely accepted paradigm. Following Ding et al. (2023), we divide task bricks into three types according to the operations on the tunable parameters. Besides, recent efforts demonstrate that the task bricks can also be obtained by extracting task vectors from the internal representations without tuning, and we term these efforts as training-free task bricks.

Addition-based Bricks Addition-based bricks introduce extra parameters into the LLM for fine-tuning. Within this category, the most extensively studied methods are adapter tuning (Houlsby et al., 2019) and prompt tuning (Liu et al., 2023b). The fundamental structure of adapter tuning consists of two linear layers with

a notably low intermediate dimension, which enables efficient computation and storage. This layer can be inserted into the standard transformer architecture, for instance, following the self-attention layers or the feed-forward network layers, to facilitate task adaptation. Different from adapter layers to modify the model architectures, prompt tuning conducts task adaptation by inserting token embeddings in the input layers (Liu

- et al., 2023b; Ding et al., 2022). The early researches prepend hard discrete tokens to the inputs, which aims to bridge the gap between pre-training and fine-tuning via formalizing all NLP tasks into sequence generation problem (Brown et al., 2020; Gao et al., 2021; Schick & Schütze, 2021). Then to make prompt tunable, soft prompt is proposed to prepend randomly initialized continuous embeddings to inputs and optimize task objectives via gradient descent (Lester et al., 2021). Prompt tuning is a human interaction-friendly algorithm as the users can drive LLMs to accomplish various tasks by utilizing different prompts eliminating the need to modify the model’s architecture.

Specification-based Bricks Specification-based bricks specify some existing parameters in LLMs to be tunable and do not introduce additional parameters. BitFit (Zaken et al., 2022) presents that only optimizing the bias vectors inside the linear projections can achieve satisfactory performance. Cui et al. (2023a) attempt to only tune the output layer to protect data privacy and enable an efficient API-based tuning framework. Besides manually specifying which parameters are adjustable, Guo et al. (2021) and Zhao et al. (2020) learn a binary mask for the parameters, determining which ones should be optimized for a given task. When the model scales to billions of parameters, the performance gap introduced by the design differences becomes negligible, and even arbitrarily specifying modules to be tunable can also lead to comparable results with full-parameter fine-tuning (Su et al., 2023).

Reparameterization-based Bricks Reparameterization-based bricks rewrite the computational formula of existing layers into a parameter-efficient form and specify part of the parameters tunable. The most widely adopted approaches within this category assume that the variations in some parameters during training are low-rank, subsequently optimizing these low-rank variations with tiny parameters. For instance, the intrinsic dimension, as mentioned earlier, maps a low-dimensional vector into the parameter space, allowing the model training process to solely optimize this low-dimensional vector (Aghajanyan et al., 2021; Zhang et al., 2023d). Similarly, LoRA models the variation of a particular parameter matrix as the product of two matrices with significantly low intermediate dimensions (Hu et al., 2022).

Training-free Task Bricks In addition to the aforementioned methods that require additional training, many researchers attempt to activate the intrinsic task capabilities of foundation models without any training. Recent progress shows that the representation space of intermediate layers in LLMs possesses semantically meaningful structures (Zou et al., 2023). It indicates that we can directly control the behaviors of LLMs by operating the intermediate representations. Inspired by these findings, many efforts reveal that the demonstrations in in-context learning can be transformed into a function vector with simple representation arithmetic, and inserting the function vector into intermediate representations of inputs can trigger the foundation model to generate the task predictions (Liu et al., 2023c; Todd et al., 2023; Hendel et al., 2023).

##### 2.2.4 Knowledge Bricks

Knowledge bricks aim to supplement LLMs with external knowledge. While it is well-documented that LLMs internalize amounts of world knowledge to facilitate robust language comprehension (Petroni et al.,

- 2019; Roberts et al., 2020; Jiang et al., 2020), their finite parameter space inevitably limits the capacity to encapsulate the nearly infinite spectrum of external knowledge. This limitation often manifests itself in the form of “hallucinations”, where the model generates erroneous information in responses due to a lack of relevant knowledge (Maynez et al., 2020; Lin et al., 2022; Honovich et al., 2022; Ji et al., 2023). Moreover, the computational overhead of LLMs make them less agile in adapting to the ever-changing landscape of world knowledge (Wu et al., 2022; Wang et al., 2023b). In the following paragraphs, we will present methods for representing knowledge as compact neural bricks and discuss the advantages of knowledge bricks compared to traditional knowledge injection methods. Based on the knowledge type, we can divide the knowledge bricks into structured knowledge graph (KG) bricks and unstructured text bricks.

Structured KG Bricks Leveraging structured KGs to enhance pre-trained language models has long been a pivotal direction in NLP (Zhang et al., 2019; Han et al., 2021b). Compared to traditional models that incorporate knowledge during the pre-training phase, the construction of tiny structured KG bricks is costefficient, with stronger scalability to build bricks across diverse knowledge types. The core of structured KG bricks lies in pluggable knowledge representations, which can be injected into LLMs to provide external knowledge. One main line of research attempts to concatenate informative entity representations with original input embedding sequences for knowledge fusion. To this end, Ye et al. (2022) average the output vectors

of masked entity’s occurrences as pluggable representations. Pörner et al. (2020) and Zhang et al. (2023b) learn a neural projection to bridge the gap between KG embeddings (Bordes et al., 2013; Yamada et al., 2016) and token embeddings in pre-trained models. Besides, different from incorporating knowledge features, K-Adapter adopts knowledge-specific objectives, such as entity relation extraction for KG and dependency prediction for linguistic trees, to optimize adapters for knowledge-enriched representations from originally hidden vectors (Wang et al., 2021).

Unstructured Text Bricks Unstructured text is the primary medium for humans to record and store world knowledge. An increasing number of researchers are exploring ways to augment LLMs with an external retrievier, where the relevant textual knowledge is concatenated with input instructions (Guu et al., 2020; Lewis et al., 2020; Nakano et al., 2021; Mialon et al., 2023). However, these approaches often suffer from poor reusability of encoded knowledge, requiring redundant knowledge re-encoding for different instructions. Therefore, constructing cross-task reusable, plug-and-play textual knowledge bricks presents an efficient method. Saad-Falcon et al. (2023) adopt the activations of long documents as reusable representations. During downstream inference, the activations are directly fed into the top layers of pre-trained models to reduce computational overhead. Xiao et al. (2023b) represent documents as prefix tokens and conduct self-supervised training to enable document bricks suitable for both inference and fine-tuning.

##### 2.2.5 Modality Bricks

Multimodal large language models (MLLM), which utilize LLMs as the brain for reasoning and are capable of processing various perceptual signals such as images and speech, have become pivotal in the pursuit of artificial general intelligence. With the continuous growth in LLM training data and parameter scale, LLMs have exhibited numerous surprising emergent abilities, including instruction-following, in-context learning, and chain-of-thoughts (Wei et al., 2022a,b). To leverage these remarkable abilities in multimodal tasks and scenarios, many researchers have shifted their focus towards the training of MLLM (Yin et al., 2023). However, building an MLLM from scratch necessitates substantial computation and multimodally aligned data pairs. As a result, much of the current work treats pre-trained models from other modalities as bricks for LLMs, effectively leveraging them to transform multi-modal signals into features that the LLMs can readily process and understand. Based on the types of interface features communicated between models, these models can be categorized as bricks with textual interface and continuous interface.

Bricks with Textual Interface This line of work initially converts multi-modal data into text, which is then combined with textual instructions and fed into the LLMs. For example, Li et al. (2023b) adopts open-source video caption and detection models to convert videos into textual descriptions, which are then fed into LLM to generate responses. Recent popular tool-augmented LLMs treat models for other modalities as APIs. When given the functional descriptions and input-output formats of models, LLMs decompose the input instruction into multiple sub-tasks, where various models are involved to solve one by one (Wu et al., 2023a; Shen et al.,

- 2023c; Yang et al., 2023). These approaches require no additional training and do not necessitate access to the model’s parameters, making them particularly suitable for API-based models such as ChatGPT and GPT-4 (OpenAI, 2023). However, converting other modalities into text often results in inevitable information loss, which can considerably impact the model’s performance.

Bricks with Continuous Interface To eliminate the information loss, many researchers attempt to construct a learnable continuous interface between LLMs and other pre-trained models. As the models are trained separately on single-modality data, the representation spaces between these models are quite different, which poses challenges for the learnable interface to translate visual and audio inputs as LLM continuous prompts. As for the model architecture, a widely-used method is adopting attention mechanism to extract important information with several learnable query vectors (Alayrac et al., 2022; Li et al., 2023a; Chen et al., 2023a). Further investigation indicates that a simple multi-layer perceptron is powerful for modalities bridge (Liu et al., 2023a; Zhu et al., 2023; Luo et al., 2023; Wu et al., 2023b). Different from bricks with textual interface, learnable continuous interface relies heavily on multi-modality aligned data (Hu et al., 2023; Zhao et al., 2023c). Besides, due to the limited representation capacities of continuous prompts, the learnable interface sometimes suffers from fine-grained information loss.

Besides the three types of customized bricks mentioned above, many researchers devote efforts to developing plugins with diverse functionalities. These include plugins that enable models to manipulate external tools (Shi et al., 2023; Yu et al., 2023; Hao et al., 2023b), debias the model response (Dathathri et al., 2020), reduce the computational costs (Xiao et al., 2023a), and transform the style of generated text (Pascual et al., 2021; Dathathri et al., 2020). The practice of constructing tiny customized bricks - plugins for LLMs to supplement their functionalities and knowledge has become a widely accepted paradigm. Despite this success, the plugin

learning for LLMs is still faced with the following challenges: (1) Combining multiple plugins: In real-world scenarios, it often becomes necessary to combine multiple plugins to execute complex commands. However, since different types of plugins are trained independently, combining them during the inference stage can lead to out-of-distribution (OOD) problems. Exploring the combination of multiple plugins is therefore crucial to unlocking the full potential of large model plugins. (2) Unified training strategy: Currently, different training methods, datasets, and insertion points are required for plugins with different capabilities. Discussing the construction of different types of plugins from a unified perspective could greatly benefit the future development of numerous plugins. A standardized approach to training would streamline the process, ensuring consistency and efficiency across different plugin types, which would also benefit compatibility issues.

##### 2.3 Brick Granularity

As stated previously, the granularity of a brick is highly customizable, ranging from a solitary neuron to a whole pre-trained model. As the size of bricks increases, their capacity expands correspondingly and the computational resources required also increase. This presents a challenge in selecting the optimal brick size, necessitating a careful balance between efficiency and effectiveness. In this section, we will first review existing observations on the capability of four different granularity of bricks: the solitary neuron (§ 2.3.1), the neuron group (§ 2.3.2), the layer (§ 2.3.3), and the full model (§ 2.3.4). Furthermore, we discuss how to choose the brick granularity properly (§ 2.3.5).

##### 2.3.1 Solitary Neuron Granularity

The neuron, defined as a row or a column in the weight matrix of the linear layer, is often considered the finest functional unit in Transformer-based foundation models (Suau et al., 2020; Zhang et al., 2023c). After being trained properly, they can carry certain skills or knowledge, laying a solid foundation for the complex behavior of the entire deep learning system.

From the perspective of skills, neurons in well-trained neural networks are demonstrated to possess the ability to capture specific input patterns and predictivity for some basic NLP tasks. Some early works find that neurons can learn the position of words (Karpathy et al., 2015) or parts of speech (Dalvi et al., 2019) such

- as nouns, verb forms, articles, numbers, etc. Others prove the specialization of certain neurons in capturing groups of words with similar meanings (e.g., electronic items or legislative terms) (Li et al., 2016; Kádár et al., 2017; Na et al., 2019). Further, recent studies demonstrate the potential of visual model neurons in learning meaningful perceptual concepts such as the tall structures in images (Mu & Andreas, 2020). The sensitivity of neurons to various input patterns constitutes their high predictivity for some fundamental NLP tasks, including sentiment analysis, natural language inference, topic classification, etc (Wang et al., 2022a).

Moreover, factual knowledge is also an important aspect of information that can be obtained by neurons. An important work in this field is done by Dai et al. (2022), who demonstrates the potential storage of factual knowledge in specific neurons (i.e., knowledge neurons). The activation of these knowledge neurons is positively correlated to the expression of the corresponding knowledge triplet, which sheds light on a promising approach to training-free knowledge editing and model manipulation (Sajjad et al., 2022).

Another interesting fact found in previous work is that the skills or knowledge contained in a solitary neuron can even be non-singular. Polysemous neurons capturing multiple concepts or word senses widely exist in deep neural networks (Xin et al., 2019; Suau et al., 2020). The knowledge neurons were responsible for different factual knowledge are also proven to have intersections (Dai et al., 2022). This observation underscores the potential of exploring smaller units within LLMs for understanding the storage of knowledge and skills.

##### 2.3.2 Neuron Group Granularity

Neuron groups, namely tiny sublayers involving a group of neurons, can often display more complex behaviors than solitary neurons.As demonstrated in Zhang et al. (2023c), neurons can be emergently clustered into different function groups during the pre-training. Besides, the customized bricks usually consist of tiny sublayers to store certain knowledge and abilities.

One of the most popular organization forms of neuron groups is Mixture-of-Expert (MoE), where each expert is a specialized neuron group and the MoE output is aggregated from the expert outputs through a routing function. As for Transformers, pre-defined MoE is usually implemented by replacing a single linear layer in the attention module (Zhang et al., 2022b) or the feedforward network (Fedus et al., 2022b) with multiple linear experts. In some previous works (Zhang et al., 2022b; Zoph et al., 2022; Zhang et al., 2023c), these

experts are demonstrated to possess specialized functions at different levels, ranging from simple semantic functions (e.g., word sense classification), knowledge functions (e.g., factual knowledge recognition), to more complex language understanding tasks such as GLUE (Wang et al., 2019a). Other works also provide clues for expert specialization through the analyses on expert activations, expert usage, or ablation studies (Chen

- et al., 2022; Mustafa et al., 2022; Zhao et al., 2024a; Shen et al., 2023a,b). They also demonstrate that the effectiveness of MoE and expert specialization are consistent in textual, visual, and multimodal models. In addition to the pre-defined MoE structure, we can also probably explore MoE structures inside an already pre-trained model without additional parameters. For instance, Zhang et al. (2022c) constructs experts by splitting the FFN parameters into functional partitions, which reduces the computation significantly without harming the overall performance.

Another line of research focusing on a group of neurons is parameter-efficient tuning. To reduce the huge computation costs of tuning large language modules, PET only updates a small number of neurons (inherently inside the model or additionally introduced) while freezing the remaining parts of the model (Ding et al., 2023). The tunable neuron groups in representative PET methods (e.g., Adapter (Houlsby et al., 2019), Prefix-Tuning (Li & Liang, 2021), BitFit (Zaken et al., 2022), and LoRA (Hu et al., 2022)) are demonstrated to have high versatility and satisfactory performance on over 100 NLP tasks, from simple text classification to complex conditional generation (Ding et al., 2023). PET neuron groups can also carry external knowledge to empower the frozen language model in a plug-and-play manner (Zhang et al., 2023b; Xu et al., 2023b).

##### 2.3.3 Layer Granularity

The utilization of stacked layers within deep models has consistently showcased its superior performance across numerous scenarios (Wang et al., 2019c; Devlin et al., 2019). LLMs typically comprise multiple layers with the same architecture, each possessing unique parameters (Touvron et al., 2023a,b). Understanding and manipulating different layers are both crucial aspects to maximizing the potential of LLMs.

Numerous studies have delved into the examination of the functions of distinct model layers. For instance, Lin et al. (2019) present that the word order information is mostly contained in lower layers, while Hewitt & Manning (2019) propose the structural probing framework and find that syntactic knowledge is most prominent in middle layers. Liu et al. (2019) also find that final layers are usually more related to specific tasks. Rogers et al. (2020) provide a relatively thorough survey about the function of each layer in BERT (Devlin et al., 2019). In addition to linguistic functions, Geva et al. (2021) demonstrate that the feed-forward layers in Transformer models (Vaswani et al., 2017) can be construed as key-value memories, which inspired a new way to addressing and editing the knowledge stored in language models. As models progress from lower to higher layers, the functional scope transitions from local, lexical aspects to global, semantic dimensions.

Based on the observations, there exist several approaches for the manipulation of inner model layers to improve efficiency, among which a straightforward one is to swap out some layers to accelerate model training or inference. Zhang & He (2020) present a Switchable-Transformer block, which introduces a gate to determine whether the corresponding layer is disabled or not in the training stage, based on which they further proposed a progressive-layer-dropping approach that can effectively reduce the training cost. Regarding inference, Xin et al. (2020) introduces DeeBERT, which reduces inference time by bypassing certain upper layers rather than passing through the entire model. Another branch of layer manipulation is knowledge editing, which aims to change the existing knowledge within pre-trained models by modifying some specific layers. For example, Huang et al. (2023b) proposed to inject multilingual knowledge into the feed-forward layers. Recently, Wang et al. (2023d) present EasyEdit, which supports various cutting-edge knowledge editing methods and applies to representative large language models, such as Llama 2 (Touvron et al., 2023b).

##### 2.3.4 Full Model Granularity

Various types of models frequently exhibit distinct advantages and drawbacks. For instance, large models excel in performance, while small models offer higher speed and demand fewer computational resources. Combining existing models is an efficient strategy for harnessing the strengths of individual models. Li et al. (2021) propose to dynamically select models of different sizes for input samples with different difficulties. Inspired by the dual-process theory of human cognition, Lin et al. (2023) propose to employ a small model that performs fast and intuitive thinking and a large model for complex and slow thinking. In addition to amalgamating models of varying sizes, as discussed in §2.2 the integration of models from different modalities offers a practical approach to constructing multimodal models (Li et al., 2023a).

The studies mentioned above primarily concentrate on the integration of independent models. However, there are also notable works that involve the extraction of sub-networks from larger models. Zhang et al. (2021) explore the out-of-distribution generalization capabilities of sub-networks and find that even in biased models there still exist unbiased sub-networks. Xu et al. (2021) identify a sub-network, which they called the child network, in a pre-trained model and only update the weights of the child network for downstream tasks. Xu

- et al. (2022) also propose S4-Tuning, a technique that partitions the entire model into sub-networks dedicated to each target language. It exclusively updates the relevant sub-network for tasks in a specific language, thereby enhancing language-specific task performance.

##### 2.3.5 Discussion

Based on the above statements, we turn back to the issue of selecting the appropriate granularity according to the required ability. Intuitively, coarser-grained bricks with more parameters are better suited for addressing complex tasks. For instance, solitary neurons can discern specific patterns of low-level linguistic units (Karpathy et al., 2015; Dalvi et al., 2019; Li et al., 2016; Kádár et al., 2017; Na et al., 2019). By contrast, full model bricks are typically associated with higher-level capabilities, encompassing the general understanding of specific modalities (Li et al., 2023a), language, or textual corpus (e.g., codes). However, there exists evidence supporting the presence of certain abilities shared by multiple granularities. Both solitary neurons and neuron groups have been established as having predictive functions in some language understanding tasks, including sentiment analysis and natural language inference (Wang et al., 2022a; Zhang et al., 2023c). Consequently, the alignment of diverse granularity levels with specific abilities remains an open question. Scaling laws, the empirical studies on the relationships between task performance, model scale, dataset size, and computation, may provide insights into the granularity-ability association (Kaplan et al., 2020).

Another point worth attention is that different levels of brick granularities and abilities have potential inclusive relationships. Specifically, high-level abilities, such as general language understanding, can be decomposed into low-level NLP tasks. Similarly, bricks of coarser granularities can be viewed as a combination of multiple finer-grained ones. Therefore, a possible approach involves structuring foundational models with hierarchical bricks. Consequently, when retrieving, constructing, or updating bricks, operations can be efficiently executed

- at the appropriate granularity, guided by the hierarchical functional partitions. Nevertheless, some works also demonstrate that the functional bricks in LLMs can appear emergently during the training process. Given the yet-to-be-fully-explored association between granularity and abilities, the manual design of well-suited brick hierarchies or abilities presents a challenge. Thus, it becomes necessary to conduct comparative analyses of the performance and costs for different brick granularities, which we leave for future work.

##### 2.4 Benefits of Configurable Bricks

As elucidated earlier in this section, we can decompose pre-trained models into emergent bricks and enhance their capabilities by constructing custom bricks, thereby realizing a configurable LLM architecture. In this subsection, we will highlight five advantages of configurable LLMs compared to traditional monolithic LLMs.

Efficiency The vast number of parameters in language models, encapsulating knowledge and capabilities, are essential for executing a wide range of tasks. However, for a specific task or instruction, we often only need to utilize a subset of these parameters for language comprehension and task inference. This means that, given a particular command, we can dynamically select the relevant bricks associated with the current instruction for computation. For instance, early-exiting models treat each layer of the model as a brick. For each instruction, they decide whether to engage subsequent layers for computation based on the predicted confidence (Zhao

- et al., 2023c; Han et al., 2021c; Laskaridis et al., 2021). Models utilizing a mixture-of-experts (MoE) approach regard each expert network (typically an FFN layer) as a brick. For every token, they select a few experts that match the token’s characteristics from a set of expert networks to participate in the computation (Zoph et al., 2022; Lepikhin et al., 2021). Consequently, even as the number of model parameters grows in response to expanding knowledge and capabilities, the computational demand for a given instruction or task remains relatively low.

Reusability Configurable LLMs decompose the model into several distinct functional bricks, facilitating the performance for real-world complex requirements through combinations of these bricks. In various configurations, knowledge and capability transfer can be achieved by reusing different bricks. For instance, Pfeiffer et al. (2020) decomposes multi-lingual task fine-tuning into language bricks and task bricks. Training task bricks on a high-resource language and then combining it with a low-resource language brick enables the task knowledge transfer across different languages. Both Zhang et al. (2023b) and Xiao et al. (2023b) model

KG and textual knowledge as task-agnostic bricks, allowing for efficient knowledge injection across different tasks without the need for further task-specific adaptations.

Traceability Decomposing the black-box LLMs into bricks with interpretable functions allows us to trace the underlying mechanisms behind their superior performance by monitoring the activation of bricks. As aforementioned, efforts have been made to identify the knowledge (Dai et al., 2022), concept (Suau et al.,

- 2020; Mu & Andreas, 2020), and task-specific skills Wang et al. (2022a) encoded in specialized neurons or expert units. When a particular brick is activated, it indicates that the knowledge contained within the brick has been utilized to generate a response to the given instruction. Such observations provide a fresh perspective on understanding the behavior of LLMs. After the source tracking, we can predictively control model behavior by manipulating relevant bricks without affecting other parts of the architecture (Sajjad et al., 2022). For example, Dai et al. (2022) update or erase learned relational facts by directly modifying the parameters in the corresponding knowledge neuron. This provides a novel viewpoint for the oversight and alignment of LLMs: instead of focusing on a holistic ethical and safety review of the entire LLMs, the focus can shift to a brick-by-brick examination, allowing for the repair or replacement of bricks that may induce the model to generate unethical responses (Lauscher et al., 2021; Geva et al., 2022).

Sustainability Enhancing LLMs with continuous capabilities and knowledge to adapt to the ever-evolving global environment remains a focal point in research. Unlike monolithic LLM, which necessitates the updating of whole parameters and can result in catastrophic forgetting of existing knowledge, configurable LLMs can achieve continual learning through the growth and updating of specific bricks without undermining previously acquired knowledge. For instance, in the realm of multi-domain pre-training, many scholars focus on the MoE model, leveraging the expansion of experts to assimilate knowledge across an increasing array of domains (Li et al., 2022; Shen et al., 2023b; Komatsuzaki et al., 2023). Furthermore, when knowledge embedded within the LLM requires updating or overwriting, strategically modifying specific emergent bricks (Geva et al., 2022; Meng et al., 2022), or constructing a supplementary customized brick (Dong et al., 2022; Lauscher et al.,

- 2021), stands as an efficient solution.

Distributed Computation Configurable LLMs decompose the monolithic computation into modularized operations. The distributed computation trait makes configurable LLMs more practical to deploy on computational clusters: each machine can be tasked with the computation of a specific brick, exchanging information with others through hidden vectors. This distributed computing trait can harness the full computational capacity of each device, thereby reducing deployment costs. For instance, many researchers propose to distribute different bricks across distinct machines, training them with domain-specific data, and eventually merging all the trained bricks to produce a larger language model with enhanced capabilities (Wortsman et al., 2022; Alayrac et al.,

- 2022; Li et al., 2022; Huang et al., 2023a). Moreover, the nature of distributed computing can serve as a safeguard for model and data privacy. For example, Cui et al. (2023a) and Zhou et al. (2022b) place the main LLM on a central server endowed with substantial computational resources, while the task-specific bricks and output layers reside on the user’s machine. This setup allows users to reap the benefits of the LLM’s superior performance without compromising data confidentiality.

#### 3 Operations for Configurable Bricks

In the previous section, we introduce the construction of emergent and custom bricks for LLMs. As the number and variety of bricks increases, it becomes crucial to configure LLMs for intricate requirements in real-world applications. This involves utilizing multiple different bricks to execute complex instructions. In this section, we mainly describe several operations associated with the LLMs configurable bricks: routing and retrieving from a vast array of bricks based on instructions (§ 3.1); combining multiple single-purpose bricks to endow the system with composite capabilities (§ 3.2); updating or refining bricks to align with shifts in world knowledge and requirements (§ 3.3); growing the bricks to accommodate new capabilities acquired from continuously emerging data (§ 3.4).

##### 3.1 Routing and Retrieval

Given the abundance of emergent and customized bricks, as shown in Figure 4, it is essential to establish suitable routing and retrieval methods to utilize these bricks in various situations effectively. In this subsection, we first provide an overview of existing routing and retrieval methods, examining them from two perspectives: the categories of bricks and their granularity. Then, we engage in a discussion concerning the improvement of current retrieval methods.

- Inst. 1

- Inst. 2

- Inst. 3

- Output 1

- Output 2

- Output 3

- Figure 4: The illustration for brick router and retrieval. It is only necessary to retrieve a subset of bricks to participate in the computation for each instruction.

##### 3.1.1 Emergent Brick Routing

Regarding emergent bricks, whether they are defined by humans or self-organized, the main objective of retrieving these bricks is usually to enhance the efficiency of the current model, where only limited parameters are selected for computation. Due to emergent bricks being generated during the pre-training, the number of emergent bricks for an LLM is typically limited, often amounting to only a few dozen. Consequently, emergent bricks are selected through a routing function, which assigns a score to each brick based on given instructions or tokens. The bricks with the highest scores are then engaged in the computation process. By selectively activating the retrieved bricks, it becomes possible to significantly reduce both the training and inference FLOPs, thereby improving computational efficiency. Current routing methods for emergent bricks mainly focus on the pre-defined brick architecture, MoE, and can be categorized into two main categories: trainable route function and fixed route function.

Trainable Route Function In many MoE models, a brick refers to an expert, and a trainable routing function is employed to determine the assignment of each token to its corresponding brick. Typically, the routing function in SwitchTransformer (Fedus et al., 2022b) and GShard (Lepikhin et al., 2021) consists of a trainable projection layer, which takes the token representation as input to calculate the gate values for different bricks. The token is then routed to the corresponding expert based on the top-k gate values. However, this method can lead to multiple tokens being assigned to the same brick, resulting in an imbalance of FLOPs among different bricks. To address this issue, alternative approaches have been proposed. One such approach, suggested by Lewis et al. (2021), involves solving a linear assignment problem to route tokens, rather than simply selecting the top-k bricks. Qiu et al. (2024) adopt a recurrent router to establish the associations between routing choices of different layers. Zhou et al. (2022a) propose that bricks select tokens instead of tokens selecting bricks. This method achieves a more balanced distribution of FLOPs among bricks by controlling the number of tokens each brick selects. Additionally, Puigcerver et al. (2023) propose the use of soft slots to gather information from all tokens, which are then further processed by different expert bricks. Although these trainable routing functions exhibit meaningful patterns after training (Zoph et al., 2022), fully explaining the routing behavior remains a challenging task.

Fixed Route Function Instead of training an unexplainable routing function through the training process, some researchers explore alternative fixed route functions that do not introduce any training parameters. In their work, Roller et al. (2021) utilize pre-computed hash functions to route tokens to different bricks in a perfectly balanced manner. Similarly, Zuo et al. (2022a) find that the behavior of route functions in SwitchTransformer (Fedus et al., 2022b) is akin to random routing. As a result, they suggest employing random routing without any additional parameters. Aside from these random routing methods, Gururangan et al. (2022) propose a token routing approach based on the domains of the current instance, which offers better explainability and encourages specialization among different bricks. However, it should be noted that this method requires domains of nearly equal size to maintain a balance across the different bricks.

##### 3.1.2 Customized Brick Retrieval

For customized bricks, the retrieval objective is to enhance LLMs with specific external capabilities that are relevant to the current situation. While retrieval methods are predominantly used in knowledge bricks, as

they can be quite extensive, it is crucial to retrieve the most pertinent knowledge from vast sources such as Wikidata.

Several studies (Févry et al., 2020; Ye et al., 2022; Zhang et al., 2023b) have focused on augmenting LLMs with entity knowledge brick from structured knowledge graphs and have investigated entity linking as a retrieval method to incorporate specific entity brick into the current model. In addition, Cheng et al. (2023) explore encoding Wikipedia into an external memory as knowledge bricks, and they utilize Maximum Inner Product Search (MIPS) to retrieve the most suitable brick for different instances.

In the case of other types of customized bricks, such as task-specific modules, their scale is generally not as vast. Therefore, previous works (Friedman et al., 2021; Pfeiffer et al., 2021; Huang et al., 2023a) have focused on combining and merging all bricks rather than specifically retrieving the most relevant ones. Considering the growing number of task bricks, Zhao et al. (2024b) propose to retrieve and then compose multiple LoRA modules relevant to the input instructions.

Overall, retrieval methods play a crucial role in incorporating customized bricks, especially knowledge bricks, into LLMs, and further advancements in retrieval techniques can significantly contribute to the effective utilization of these customized bricks.

##### 3.1.3 Routing and Retrieval Granularity

Different routing and retrieval methods can be applied at different levels of granularity, including token-level, sentence-level, and task-level. Each level of granularity offers distinct advantages and considerations.

Token-level routing and retrieval provide greater flexibility and enable precise control over the specific information required for a given task and instance. Many MoE architectures (Fedus et al., 2022b; Lepikhin

- et al., 2021; Zhou et al., 2022a) employ token-level routing to ensure that different experts acquire more generalized and fundamental capabilities. Additionally, token-level retrieval can be utilized within the context of knowledge brick retrieval to enhance token-level knowledge, such as entity bricks (Févry et al., 2020; Ye
- et al., 2022; Zhang et al., 2023b). However, it is important to note that token-level retrieval and routing can be time-consuming, requiring significant computation and communication costs when applied to every token and every layer.

Sentence-level routing and retrieval provide a higher-level view of the information within a sentence. Gururangan et al. (2022) utilize sentence-level information about the domain of the current instance to route each token to its corresponding domain bricks and promote expertise specialization in specific domains. Zhao et al. (2024b) adopt a retrieve-then-compose framework to utilize massive LoRA task bricks, where the model first retrieves several related task bricks based on the input instructions and averages these bricks for final computation. Furthermore, sentence-level information can also be valuable for text-based knowledge bricks, as the semantic coherence makes the related knowledge of tokens within a sentence highly likely to be the same. Cheng et al. (2023) attempt to construct a sentence-level representation using an average representation of all tokens, which serves as the query to retrieve knowledge brick for the whole sequence understanding.

Task-level routing and retrieval are specifically designed for downstream tasks. Unlike token-level and sentence-level retrieval, which are used during training or inference, task-level retrieval occurs before training and inference. In this approach, retrieval methods aim to identify the most relevant task bricks given all the data associated with a specific task. The retrieved task bricks can then be utilized to perform the task without the need for additional tuning or can serve as a better starting point for subsequent training (Pfeiffer et al., 2021; Huang et al., 2023a).

##### 3.1.4 Discussion

Based on the development of configurable bricks, more advanced routing and retrieval methods need to be proposed.

Efficient Routing and Retrieval Bricks routing and retrieval during training and inference can be timeconsuming due to the increased calculation and communication. He et al. (2021) show that the training speed of an MoE model can be 3 times slower than a compute-match dense model due to the additional computation and communication on the brick route. Even for customized bricks, some compromise solutions, like retrieval only on specific tokens (Févry et al., 2020; Ye et al., 2022; Zhang et al., 2023b) or use higher level representation (Cheng et al., 2023), are proposed to reduce the frequency of routing and retrieval. Better retrieval methods need to be designed to balance accuracy and efficiency.

Multi-Level Routing and Retrieval Most current retrieval methods are based on single-level retrieval, which may not capture all the information required for accurate retrieval. Additionally, multi-level retrieval is crucial for the collaboration between emergent bricks and customized bricks, as token-level information is empirically used in emergent bricks, while sentence-level and task-level information is more relevant to customized bricks. Chen et al. (2022) have made preliminary attempts to incorporate task-level information and token-level information into routing various emergent bricks. However, there is still much to explore in combining multi-level information for retrieval purposes.

Active Routing and Retrieval Current routing and retrieval methods typically decide in advance where to conduct routing and retrieval based on fixed rules, such as the position of entity mentions or just every token and sentence. We refer to these methods as passive routing and retrieval methods. On the contrary, more proactive methods can allow the LLM to decide where to conduct routing and retrieval during the generation process, which we call active routing and retrieval methods. Active methods can also address the efficiency problem by significantly reducing the frequency of retrieval. Zhang et al. (2022d) have attempted to dynamically augment entity memory when the model generates a specific special token. Furthermore, it is important to investigate whether current LLMs can determine when to retrieve bricks and how to enhance this ability for more advanced configurable foundation models.

##### 3.2 Combination

Single-function bricks often fall short in fulfilling complex instruction demands. Brick combination aims to fuse multiple bricks to possess combined abilities. For example, it enables cross-lingual transfer for the named entity recognition (NER) task to combine a brick trained on English NER datasets with a brick proficient in Chinese capabilities. In this way, the brick combination can obviate the need to build high-quality annotated datasets for a specific requirement and train models from scratch, thus significantly reducing both human effort and computational costs.

In this section, we divide brick combination methods into two categories based on the operations: parameter weighted average and brick stitching. Besides, we also provide a discussion about the future directions for brick combination.

##### 3.2.1 Parameter Weighted Averaging

Parameter weighted averaging obtains a merged brick by directly performing an element-wise weighted average of multiple bricks with the same structure (homogenous bricks combination). In early research, parameter averaging is applied to the ensemble of models trained from the same task, aiming to boost the model’s robustness and performance. Due to the inherent randomness in the training of deep neural networks and the non-convex nature of loss functions, linearly weighting parameters from two distinct training processes usually fail to yield satisfactory results. As a result, many researchers explore the “mode connectivity” of deep neural networks (Garipov et al., 2018; Draxler et al., 2018). These explorations seek to uncover the interconnected paths between the parameters of two models, ensuring that the parameters along this path achieve commendable accuracy. Further, Frankle et al. (2020) discover that if two models are initialized from the same well-trained parameters, a straightforward linear weighting can enable the merged model to exhibit superior performance, which inspires subsequent research on parameter average based on pre-trained models. Models fine-tuned from the same pre-trained model, though with different training configurations, including various hyperparameters and data sampling, can be linearly weighted together (Wortsman et al., 2022; Qin et al., 2022a; Chronopoulou et al., 2023; Viswanathan et al., 2023; Ruan et al., 2023). The resulting merged birck achieves performance comparable or superior to a multi-brick ensemble. Besides, averaging bricks from the same task but different domains has been demonstrated to effectively enhance the domain generalization capacity (Cha et al., 2021; Arpit et al., 2022; Ramé et al., 2022; Li et al., 2022). Thus, parameter averaging is also employed in efficient federated learning, allowing for a generalized merged brick while simultaneously protecting private data (McMahan et al., 2017; Li et al., 2020; Karimireddy et al., 2020).

Furthermore, parameter weighted averaging has been introduced to combine bricks from different tasks to facilitate knowledge transfer. In such settings, the contribution of the source task bricks to the target task often varies, which implies that careful design is required to determine the weighting coefficients for the different bricks. Matena & Raffel (2022) employ Fisher-weighted averaging to transfer capabilities from intermediate tasks to target tasks. Huang et al. (2023a) leverage combinatorial optimization algorithm to optimize weighting coefficients, aiming to reduce the number of training instances required for the target task. Jin et al. (2023) determine the coefficients by minimizing the L2 distance between the parameters of merged bricks and source bricks. Beyond parameter addition, some researchers discern that subtracting parameters can enable a model to

5 minutes

LLM

w1 w2 w3

If it takes 5 minutes to boil an egg, how long will it take to boil three eggs simultaneously?

Math Brick

Commonsense Brick

QA Brick

Brick 1 Image Understanding

Brick 2 Knowledge Retrieval / Memory

Brick 3

Shakespeare Hamlet

Summarization

[Figure 10]

Generate a summary about the most famous work of the writer shown in the picture.

[Figure 11]

Planner

Parameter Weighted Average Brick Stitching

- Figure 5: Two widely-used operations for brick combination. (a) Parameter weighted average performs an element-wise average of multiple bricks with the same structures. (b) Brick stitching sequentially concatenates bricks together for complex reasoning.

unlearn an undesired capability (Ilharco et al., 2023). For instance, Zhang et al. (2023a) perform detoxification by subtracting a brick trained from toxicity instructions. Similarly, Daheim et al. (2023) mitigate hallucinations by subtracting a brick trained on hallucinated examples.

##### 3.2.2 Brick Stitching

Brick stitching involves concatenating several bricks together in sequence based on functional requirements, such that the output of one brick serves as the input for the next. Birck stitching can be applied to combine bricks with different structures (heterogeneous bricks combination). Given the substantial discrepancies between features processed by different bricks, a crucial aspect of brick stitching is the training of interaction interfaces between them. This ensures that the outputs from preceding bricks can be effectively interpreted by subsequent ones. While the types of interfaces have been discussed in prior sections (Modality Bricks in § 2.2), in this section, our primary focus will be on the way to determine the order and structure of stitched bricks.

Heuristic Stitching The manual definition of stitching order based on inference sequence is the most common method of brick stitching. This approach often involves explicitly decomposing a task into several inference steps. For instance, in a visual question-answering task, there’s a need to first comprehend the objects and scenes within the given image and then answer questions based on the image content. Correspondingly, concatenating an image encoding model before a language model becomes a widely adopted structure for multi-modal understanding (Alayrac et al., 2022; Li et al., 2023a; Yin et al., 2023). Besides, recent popular LLM-based multi-agent collaboration systems are also based on heuristic brick stitching, where each brick is a whole LLM-based agent and required to solve a subtask, such as front-end design in game development (Qian et al., 2023; Wang et al., 2023a; Hao et al., 2023a). Heuristic stitching is generally adopted for concatenating model-level bricks, which can independently accomplish specific inference steps. In contrast, fine-grained bricks tend to have abstract functions and are usually dependent on surrounding bricks. Arbitrarily concatenating any two bricks typically results in suboptimal collaboration. Recent studies observed that the hidden spaces of two pre-trained models from the same structures and tasks but differing sizes can be linearly transferred (Bansal et al., 2021; Csiszárik et al., 2021). Based on this insight, Pan et al. (2023) propose an approach that concatenates layer-level bricks from a family of pre-trained models with different sizes. This allows for optimal utilization of computational resources, ensuring maximum performance under given computational constraints. To further improve the flexibility of stitching different layers, Akiba et al. (2024) adopt evolutionary optimization algorithms to select optimal composition architectures from predefined stitching search spaces.

Planner-based Stitching While heuristic stitching is suitable for tasks with fixed inference steps, real-world instructions often demand varied inference sequences. This implies that we usually need to determine the execution order of different bricks based on the specific given instruction. Inspired by early neural module network (Andreas et al., 2016; Hu et al., 2017; Fashandi, 2023), many brick stitching models are composed of three components: a task planner, which is responsible for decomposing an instruction into several sub-tasks; a controller, which is tasked with generating and receiving the signals for each sub-task and ultimately produces the final answer; multiple bricks, where each brick handles a specific type of sub-task. In such a framework, the stitching order is determined by the task planner.

An intuitive approach is to have the planner generate the execution sequences based on the functional descriptions and usage demonstrations of each brick before performing instruction reasoning (Hsieh et al.,

- 2023; Shen et al., 2023c). Furthermore, many scholars propose to dynamically decide which brick to call after each reasoning step, allowing the planner to leverage intermediate reasoning results for a more precise determination of the execution sequence (Yao et al., 2023a; Gao et al., 2023). However, a linear execution sequence can be problematic: any error in the selection of bricks can directly impact the final prediction. To address this issue, recent research attempts to incorporate searching strategies for inference, which require the planner to provide multiple options at each step and sequentially test them until a satisfying response is generated (Ye et al., 2023; Qin et al., 2023).

##### 3.2.3 Discussion

Brick combination aims to fuse multiple single-function bricks to fulfill complex instructions. In this section, we delve into parameter-weighted averaging applied to homogenous brick combinations and brick stitching suitable for heterogeneous brick combinations. The essence of parameter-weighted averaging lies in determining the weights for each brick, whereas the key to brick stitching is the alignment of feature spaces across different bricks. While current efforts have initiated preliminary exploration into the brick combination, several challenges remain to be addressed.

Combination of Fine-grained Heterogeneous Bricks Most efforts in combination with heterogeneous bricks focus on the integration at the model level. However, parameter redundancies also exist between different models, for instance, both language and image models internally possess neural bricks responsible for understanding real-world concepts (Bau et al., 2020; Geva et al., 2022). Combining fine-grained heterogeneous bricks holds the potential to further reduce parameter redundancies and enhance the reusability of merged bricks across diverse scenarios.

A Universal Brick Interaction Interface Within the context of brick stitching, there are two primary interaction interfaces between different bricks: one utilizes discrete, human-readable signals, and the other engages through continuous hidden vectors. The former offers a low training cost but can suffer from information loss; the latter typically results in superior data representation but demands extensive training data and lacks generalizability across different scenarios. Hence, devising a more universally applicable and efficient module interaction method is crucial to enhancing the practicality of brick stitching algorithms. Besides, a universal interaction interface could boost the scalability of the multi-brick system, allowing bricks capable of the interface to seamlessly stitch with others.

##### 3.3 Updating

The continuous growth and evolution of world knowledge over time presents a unique challenge for LLMs. These models, once trained, may contain outdated information, leading to the phenomenon of hallucination. Therefore, LLMS needs to adapt to shifts in world knowledge. As LLMs become gradually larger, the cost of retraining the model for every new knowledge update request is prohibitively expensive. To this end, methods for quickly updating the knowledge encoded in LLMs have been developed in recent years. One significant advantage of configurable LLMs is that they allow updating bricks in an isolated manner, which is more efficient (in terms of computation or data) than full parameter fine-tuning. Keeping other parameters frozen may also help minimize unwanted detrimental impacts on other capabilities of the models (Wang et al., 2023e). Existing works related to updating bricks have largely focused on editing the knowledge encoded in neural models (Yao et al., 2023b). Therefore, the discussion in this section primarily revolves around knowledge bricks updating. However, many concepts for updating knowledge bricks can also be applied to updating other kinds of bricks as well. From the perspective of configurable bricks, we can categorize knowledge editing methods into (1) methods that locate and update naturally emergent knowledge bricks, and (2) methods that inject new customized bricks.

##### 3.3.1 Locating and Updating Emergent Bricks

Because of the above challenges, some recent works have explored methods to edit knowledge encoded in the emergent bricks of LLMs. To this end, we need to first locate the emergent bricks storing the target knowledge and then update the parameters for knowledge editing.

Locating Knowledge Bricks Inspired by the hypothesis that FFNs can be regarded as key-value memories (Geva et al., 2021), existing researches mainly focus on exploring emergent knowledge bricks at the neuron level in the FFN layers. Dai et al. (2022) use integrated gradients (Sundararajan et al., 2017) to

discover that some factual associations are positively correlated with the activation of a “knowledge neuron” in FFNs. This enables the deletion of knowledge by zeroing the activation of knowledge neurons or amplifying knowledge by scaling up the activation. Meng et al. (2022) strengthens this hypothesis by causal intervention on activation values and finds that middle-layer FFNs are most responsible for fact recalling. Hase et al. (2023) show that editing the layers identified by causal intervention does not result in better editing performance. They find that this is because causal intervention identifies which brick carries the target knowledge, but manipulating the parameters of these bricks does not lead to superior performance. It indicates that the knowledge storage of even one entity or tuple involves multiple neurons and defining knowledge bricks at the neuron-group level and layer level may be more suitable. These works shed light on the challenges of knowledge bricks localization methods.

Updating Knowledge Bricks After locating the target knowledge neurons, we need to edit the parameters for injecting the updated knowledge. Specifically, the whole FFN layer is regarded as a key-value memory network, and a knowledge neuron is treated as a key-value pair (Geva et al., 2021). The editing operation is usually performed by replacing the value vector with the target knowledge-enriched representation. Notably, one emergent knowledge brick usually is responsible for more than one target knowledge, which means knowledge editing usually results in undesirable changes for unrelated knowledge. To alleviate this issue, extra efforts are made to minimize the impact on unrelated knowledge and capabilities. Zhu et al. (2020) apply L2 normalization loss on parameter updates to reduce the drift in parameter space. Meng et al. (2022) utilize the layer statistics on a large corpus to directly infer an update that induces a given key-value mapping while minimizing the norm of the weight updates. Meng et al. (2023) improves upon this by applying multiple key-value mapping at the same time. However, a small change in the parameter space might not translate to a small change in the function space. Hence, many works use a KL divergence loss with adversarial examples to minimize the change of predictions on unrelated inputs after the update (Onoe et al., 2023; Padmanabhan et al.,

- 2023; Mitchell et al., 2022a). Some works also investigate the possibility of using a hyper-network to produce a better parameter update a given gradient information on the representations of a piece of knowledge (Cao et al., 2021; Mitchell et al., 2022a).

##### 3.3.2 Injecting New Customized Bricks

An alternative to the locate-and-update paradigm is to inject new bricks that override the existing knowledge. This generally does not require knowledge attribution of neurons or the knowledge update requests are known in advance, since the bricks are created on demand. The main challenges lie in the efficiency of the injection process and its effectiveness in replacing existing knowledge.

Many knowledge-injection methods use full-parameter updates (Lewis et al., 2020; de Jong et al., 2022; Zhang et al., 2019; Sun et al., 2020; Agarwal et al., 2021) and they are typically regarded too compute-heavy for updating knowledge. One alternative is plug-and-play methods that inject knowledge into a frozen LLM by adding new bricks. dos Santos et al. (2022) propose to train entity embeddings that can be prepended to the hidden states after the input layer. These can be trained on demand, but the update can only happen at the entity level. Huang et al. (2023c) build on the work of knowledge neurons and propose to insert and train a new FFN neuron for every knowledge update. Mitchell et al. (2022b) re-route examples concerning updated knowledge to a small model conditioned on a memory of updated knowledge. Additionally, a line of works has explored knowledge bricks in the representations instead of the parameters. Hernandez et al. (2023) add an external brick that produces an updated hidden representation that induces the target knowledge when the subject representation is replaced with it. Turner et al. (2023) and Zou et al. (2023) investigate the possibility of using the difference between the hidden representations of two prompts as steering vectors to induce the target knowledge.

##### 3.3.3 Discussion

Currently, updating methods have been largely limited to knowledge bricks. This is because the capabilities of other widely used bricks (see § 2.2.2) such as task and modality rarely require frequent updates. Moreover, these bricks are often trained in isolation without updating the base model. Thus, directly retraining the bricks for the capabilities of interest is typically effective enough. Of course, it is not hard to imagine that as we scale up LLMs, even efficient adaptation methods may be too expensive for many applications. Therefore, we believe that exploring more efficient methods for updating tasks, modalities, and other kinds of bricks is a promising future research direction. Besides, during the deployment process, we often encounter some undesired behaviors of LLMs, such as generating offensive responses when subjected to jailbreaking attacks.

Therefore, quickly locating the bricks that lead to the undesired behaviors and correcting them is a promising research direction for efficient alignment.

##### 3.4 Growing

In light of continually growing new knowledge and tasks, the demand for LLM growth thus becomes imperative, necessitating enhancements in the knowledge and capabilities of LLMs while avoiding catastrophic forgetting of existing capabilities. A straightforward strategy to address this challenge involves repetitive training from scratch on both old and new data, which demands prohibitively high costs. To this end, many efforts have been devoted to continual learning strategies aiming at enabling LLMs to acquire information from new data effectively and efficiently. In this section, we mainly focus on the strategies for continual learning by increasing the number of bricks.

##### 3.4.1 Growing for Pre-Training

Continually learning new knowledge and capabilities from the new pre-training corpus is also important. Moreover, the relationship between the performance of LLMs and model scale has been well-established through scaling laws (Kaplan et al., 2020). Based on associated observations, increasing the number of emergent bricks presents an effective approach to improving the model performance.

Early works attempt to achieve model continual pre-training by expanding the existing dense parameters, especially, expanding the width, i.e., hidden dimension (neuron-level bricks), and the depth, i.e., the number of layers (layer-level bricks). Then the new bricks and original parameters are trained with the new pre-trained corpus. An straightforward way to expand the number of parameters is to initialize the expanded parameters with the original parameters and conduct continual pre-training with a mixed corpus (Gong et al., 2019; Gu

- et al., 2021). Then many efforts are devoted into further avoiding forgetting old knowledge learned in original parameters and improve the training efficiency. ELLE (Qin et al., 2022b) enlarges the pre-trained model in width and depth and carefully recovers its capabilities on old tasks through a recovering warmup process. Next, the expanded model undergoes training on a mixture of new data and replayed old data to acquire new information. Similarly, LiGO (Wang et al., 2023c) employs a linear mapping of parameters from an existing model to initialize a model with increased width and depth. Wu et al. (2024) frozen the orignal parameters and only train the expanded parameters with new training corpus to avoid forgeting. In this way, the enlarged LLM can inherit the knowledge of the original small models and thus reduce the costs for continual pre-training.

An other promising direction is to employ the sparse-activated modular architecture, where only related bricks are selected for computation. Therefore, constructing new bricks and preserving the original bricks frozen will not introduce knowledge forgetting and avoid additional computation costs for inference. Among these works, the sparse MoE architecture is widely used. The growing operation is performed by increasing the number of experts and keeping the parameters of other layers and experts fixed. During inference, each token will only select the most related experts for computation (Li et al., 2022; Komatsuzaki et al., 2023; Shen et al., 2023b; Wang et al., 2024).

##### 3.4.2 Growing for Post-Training

Continual learning has been studied for decades for multi-task learning, where the model is required to acquire new knowledge for new task instances and preserve the abilities for existing old tasks (Kirkpatrick et al., 2016; Lee et al., 2017; Chaudhry et al., 2018; Li & Hoiem, 2016; Zhang et al., 2020). Among them, a popular line of research attempts to build an episodic memory, which can be regarded as a memory brick, to store a few representative and informative instances of old tasks (Isele & Cosgun, 2018; Rolnick et al., 2019; Wang et al., 2019b; Han et al., 2020; Zhao et al., 2022). In this way, the model can be continually trained only on instances of new tasks and instances saved in the memory brick, which can save the computational costs.

Nowadays, benefiting from the plug-and-play characteristic of task bricks, we can continually train LLMs for multiple tasks by constructing a new brick for each new task (Mahabadi et al., 2021; Wang et al., 2022c,b; Razdaibiedina et al., 2023). For example, Madotto et al. (2021) and Song et al. (2023a) increase the model capacity with pluggable Adapter and LoRA modules respectively. To differentiate between the forward propagation routes of new and old data, both works adopt a router to select the appropriate plugin. Besides, constructing new plugins can also introduce more world knowledge, domain knowledge, and complex capabilities for LLMs as discussed in § 2.2.

##### 3.4.3 Discussion

Based on the insights provided by the aforementioned studies, we suggest choosing the means of brick growing with consideration to the following factors:

Task Complexity The task complexity serves as a pivotal determinant in the selection of an appropriate model growth strategy. For less complex tasks such as acquiring modest amounts of knowledge, recognizing a new entity category, or manipulating an unseen tool, a viable approach involves growing the model at finer granularities (e.g., introducing a plugin). However, scenarios may arise where a large volume of information should be injected into the model or its knowledge capacity and general performance should be substantially boosted, necessitating growth on a larger scale.

Computation Budget The computation budget is a rigid constraint on the model scale. While model performance generally improves with growth, the training and deployment costs associated with an expanded model must not exceed the computational budget. Strategies such as plugins, and sparse MoE architectures are representative approaches to model growth within acceptable expenses.

Application Targets Finally, the growth of the model is always linked to the application targets. For instance, lightweight plugins prove highly advantageous in scenarios emphasizing user-oriented customization. Conversely, when the target is to scale up a model to achieve heightened general AI capabilities, the introduction of a more extensive parameter set through a direct increase in width and depth or other sophisticated architectures (e.g., MoE or progressive networks) becomes imperative.

#### 4 Empirical Analysis

In previous sections, we discuss that LLMs can be decomposed into emergent bricks and custom bricks from a modular perspective. Similar to the human brain, neurons in LLMs exhibit the characteristics of sparse activation and function differentiation, meaning each neuron is responsible for specific functionality and is activated when an input instruction requires those functionalities. Previous works have explored the sparsity (Zhang et al., 2022c; Li et al., 2023c), functionality specialization on specific classification tasks (Wang

- et al., 2022a), and modular grouping (Zhang et al., 2023c) on encoder model, BERT (Devlin et al., 2019), or encoder-decoder model, T5 (Raffel et al., 2020). In this paper, we focus on the analysis of widely-used decoder-only models with instruction-following chat data.

Specifically, we conduct a detailed analysis of the following questions: (1) Are LLMs sparsely activated, meaning that only a few neurons influence the final output when processing each token? (2) Do neurons exhibit functional specialization, with their activation values highly correlated to the capabilities required by the instruction? (3) Do LLMs have the potential to be modularly split, which means that different capabilities activate different partitions of neurons? In this section, we present our empirical analysis, beginning with an introduction to the formal definition of neurons and their activation values, the functionality localization of neurons, and finally, we present the experimental results.

##### 4.1 Functionality Localization

In this section, we attempt to the analysis of neurons in the feedforward layers. Previous works indicate that feedforward layers in Transformer can be regarded as key-value memory networks (Geva et al., 2021) and provide world knowledge for sequence understanding. Therefore, we mainly focus on the feedforward layers for analysis.

Neurons and Activations The feedforward layers (FFNs) employ two-layer projections or gated projections for each token in the sequences. The calculation can be written as FFN(x) = FFNO(FFNI(x)) = WO(FFNI(x)) + bO. Here, WO ∈ Rd×d

ff and bO ∈ Rd are the weight matrix and bias vector for the output linear layer FFNO(·). As for FFNI(·), there are two variants:

Vallina FFN: FFNI(x) = σ WIx + bI ,

##### Gated FFN: FFNI(x) = σ (WGx + bG) ⊙ WIx + bI . Here, WG,WI ∈ Rd

ff are the weight matrices and bias vectors for the input linear layer FFNI(·) and gate linear layer FFNG(·). Following previous works (Zhang et al., 2023c; Wang et al.,

ff×d and bI,bG ∈ Rd

- 2022a), we can split an FFN layer as dff neurons, each consisting of a row in the input and gate layer as well as

Table 3: The functionalities and their corresponding data labels used in this paper. The data labels in InfinityInstruct are not mutually exclusive, meaning that a single instance can belong to multiple different data labels.

Functionality Data Labels Coding Python Programming, SQL Programming, Java Programming, C++ Programming,

Javascript Programming, C# Programming, Object-oriented Programming, Code Comments, Code Writing

Math Mathematical Reasoning, Mathematical Modeling, Basic Mathematics, Mathematical Analysis, Mathematical Applications, Mathematical Proof, Mathematical Explanation, Mathematical Concept Explanation, Solving Complex Mathematical Problems, Basic Mathematics Calculations

Linguistic Sentence Structure Analysis, Syntactic Understanding, Linguistic Knowledge,

Syntactic Generation, Syntactic Analysis

Knowledge Health Knowledge, Geographic Knowledge, General Knowledge about Science, Legal Knowledge, Physics Knowledge, Chemistry Knowledge, Literary Knowledge, Sociology Knowledge, Popular Science Knowledge, Biology Knowledge, Astronomy Knowledge, Psychological Knowledge, Economic Knowledge, Clinical Medical Knowledge, Environmental Knowledge, Religious Studies Knowledge, Geometry Knowledge

Translation Multilingual Translation, Translation Ability, Chinese English Translation, Machine

Translation, French Translation

Ethics and Moral Ethical Judgment, Ethical Reasoning, Ethical Analysis, Ethical and Moral Reasoning, Ethical Thinking, Ethical Guidance, Unethical Behavior Simulation, Unethical Behavior, Ethics and Morality, Moral Standards

Writing Scriptwriting, Creative Writing, Narrative Writing, Technical Writing, Writing Guidance, News Writing, Script Writing, Creativity Writing, Product Description Writing, Screenwriting Ability

a column in the output layer. The outputs of FFN layers can be rewritten as the sum of all neuron outputs: FFN(x) = diff FFNI(x)iW:O,i + bOi . We define the intermediate output, FFNI(x)i, as the activation value of i-th neuron. Intuitively, if the magnitudes of activation values are small, then the corresponding neuron will have a limited impact on the final outputs and vice versa. Therefore, the activation values are widely used as indicators for the functionality of neurons.

Functionality Score In the following paragraphs, we will introduce how to locate neurons with specific functionalities. As mentioned before, the activation values can reflect the contributions of each neuron to the FFN layer output and thus are usually used as the indicator for the functionality. Then we will present the process to calculate the functionality scores of neurons on given functionalities.

Specifically, we denote the functionality, such as coding ability, as f, a collection of chat instances as C = {(p0,r0),...,(pn,rn)}, where pi and ri are user input prompt and model-generated response. We define the functionality label for each chat instance (pi,ri) as yif, where yif = 1 when pi requires M to have the capability f to generate the correct answer; otherwise, yif = 0. For example, given the prompt p = “How can we select unique elements from a list in Python?′′, its functionality label for code ability is 1 and its functionality label for translation ability is 0.

We denote the LLM requiring to be analyzed as M, which has L Transformer layers and L × dff neurons. Given a neuron n, as the FFN layers are computed in a token-wise manner, we can collect the activation values

of the neuron n on the collection C. For the instance, (pi,ri), there are li activation values, and we define the collection of the absolute value of activation values as: Ai = {|ai0|,...,|ail

|} from the li tokens in pi. We

i

define the activation value of neuron n on the instance (pi,ri) as the average value of Ai. Then, following Zhang et al. (2023c), we use the average precision score as the functionality score of n on the functionality f:

###### FuncScore(n,f) = AvgPrecision({A0,...,Ai},{y0f,...,ynf}). (1)

32

32

1.0

1.0

2.75

Ethical Coding Writing Math

Ethical Coding Writing Math

| |
|---|

12

CumulativeDistributionFunction

CumulativeDistributionFunction

0.9

2.50

0.8

| |
|---|

10

2.25

0.8

Translation Knowledge Linguistic

Translation Knowledge Linguistic

2.00

8

0.7

0.6

Layer

Layer

Loss

Loss

| |
|---|

1.75

0.6

6

0.4

1.50

0.5

4

1.25

0.4

0.2

1.00

2

0.3

| |
|---|

| |
|---|

0.75

0.0

1

1

0 20 40 60 80 Masked Ratio - Activation (%)

0.0 0.2 0.4 0.6 0.8 1.0 Normalized Output Magnitude

0 20 40 60 80 Masked Ratio - Magnitude (%)

0.0 0.2 0.4 0.6 0.8 1.0 Normalized Activation

(a)

(b)

(c)

(d)

- Figure 6: Sparsity activation for neurons in Llama-3-8B-Instruct. (a)(c) The cumulative distribution function of normalized activation values and output magnitudes. (b)(d) Impact of neurons with low activation values and output magnitudes.

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.0 0.2 0.4 0.6 0.8 1.0 Normalized Activation

0.0

0.2

0.4

0.6

0.8

1.0

CumulativeDistributionFunction

1

32

Layer

(a)

0 20 40 60 80 Masked Ratio - Activation (%)

2

4

6

8

10

Loss

| |
|---|

| |
|---|

| |
|---|

Ethical Coding Writing Math

Translation Knowledge Linguistic

(b)

0.0 0.2 0.4 0.6 0.8 1.0 Normalized Output Magnitude

0.3

0.4

0.5

0.6

0.7

0.8

0.9

1.0

CumulativeDistributionFunction

1

32

Layer

(c)

0 20 40 60 80 Masked Ratio - Magnitude (%)

1.0

1.5

2.0

2.5

3.0

Loss

| |
|---|

| |
|---|

| |
|---|

Ethical Coding Writing Math

Translation Knowledge Linguistic

(d)

- Figure 7: Sparsity activation for neurons in Mistral-7B-Instruct-v0.3. (a)(c) The cumulative distribution function of normalized activation values and output magnitudes. (b)(d) Impact of neurons with low activation values and output magnitudes.

Intuitively, a higher functionality score suggests a stronger correlation between neuron n and capability f. That is to say, if the FuncScore(n,f) is high, neuron n exhibits higher activation levels when the input prompt necessitates capability f and lower activation when it does not.

##### 4.2 Experimental Settings

To analyze the functionality specialization and partition, we need a dataset annotated with abilities required by each instance. Therefore, in the experiments, we adopt the Infinity-Instruct dataset 2. Each instance in Infinity-Instruct consists of user prompts, model-generated responses, and several abilities required by the given user prompts. There are thousands of different abilities across the entire dataset and we summarize 7 typical and widely-used functionalities and its corresponding data labels for our analysis. The detailed functionalities and data labels are listed in Table 3. (1) Coding: Specializes in a variety of programming languages including Python, Java, and C++, with expertise in object-oriented programming and effective code documentation. (2) Math: Encompasses a wide range of mathematical skills, from basic calculations to complex problem-solving and theoretical proofs. (3) Linguistic: Focuses on the analysis and generation of syntactic structures, enhancing understanding and applying linguistic knowledge effectively. (4) Knowledge: Covers an extensive array of subjects such as science, literature, and religion, reflecting the deep understanding and application of specific disciplinary knowledge. (5) Translation: Showcases multilingual capabilities, specializing in Chinese-English translations among other languages, with proficiency in machine translation systems. (6) Ethics and Moral: Concentrates on ethical reasoning and judgment, exploring concepts from ethical analysis to the implications of unethical behaviors and moral standards. (7) Writing: Spans a variety of genres and formats including scriptwriting, creative writing, and technical documentation, emphasizing creativity and clear communication.

2https://huggingface.co/datasets/BAAI/Infinity-Instruct

Coding

Ethical

Knowledge

Linguistic

1.0

1.0

1.0

1.0

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.8

0.8

0.8

0.8

FunctionalityScore

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

0 5 10 15 20 25 30

0 5 10 15 20 25 30

0 5 10 15 20 25 30

0 5 10 15 20 25 30

Layer

Layer

Layer

Layer

Math

Translation

Writing

1.0

1.0

1.0

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.8

0.8

0.8

FunctionalityScore

Best Functionality Score

Best 5 Functionality Score

0.6

0.6

0.6

Average Functionality Score

Functionality Score for Random Activation

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

0 5 10 15 20 25 30

0 5 10 15 20 25 30

0 5 10 15 20 25 30

Layer

Layer

Layer

(a) Llama-3-8B-Instruct

Coding

Ethical

Knowledge

Linguistic

1.0

1.0

1.0

1.0

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.8

0.8

0.8

0.8

FunctionalityScore

0.6

0.6

0.6

0.6

0.4

0.4

0.4

0.4

0.2

0.2

0.2

0.2

0.0

0.0

0.0

0.0

0 5 10 15 20 25 30

0 5 10 15 20 25 30

0 5 10 15 20 25 30

0 5 10 15 20 25 30

Layer

Layer

Layer

Layer

Math

Translation

Writing

1.0

1.0

1.0

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.8

0.8

0.8

FunctionalityScore

Best Functionality Score

Best 5 Functionality Score

0.6

0.6

0.6

Average Functionality Score

Functionality Score for Random Activation

0.4

0.4

0.4

0.2

0.2

0.2

0.0

0.0

0.0

0 5 10 15 20 25 30

0 5 10 15 20 25 30

0 5 10 15 20 25 30

Layer

Layer

Layer

(b) Mistral-7B-Instruct-v0.3

- Figure 8: Distribution of functionality scores across different layers. We also present the average functionality scores and functionality scores for random activation.

We manually select data labels that meet our specific functionality requirements. Since each instance may demand various abilities, our goal is to analyze the specialization and distribution of functionality across different abilities. Therefore, we retain only those instances that belong exclusively to one of the aforementioned types. We randomly sample 1,000 instances from each data type for further analysis.

As for the backbone model, we adopt two widely used models, Llama-3-8B-Instruct and Mistral-7B-Instructv0.3, for analysis. Both two models are trained with large-scale corpus and chat data.

##### 4.3 Sparse Activation

In this subsection, we focus on the first question: are LLMs sparsely activated, just similar to human brains? If LLMs are sparsely activated, we can select part of them for the computation of each input to reduce the computational costs. To evaluate the sparsity of LLMs, we attempt to calculate the distribution of neuron activations and the impact of the neurons with low activation values. Besides, as the activation values are intermediate results of FFN, following Zhang et al. (2024b), we directly observe the impact of each neuron on the output, i.e., output magnitude, for sparsity evaluation. Specifically, for i-th neuron in FFN, the output magnitude is defined as ||FFN(x)i||2. For the convenience of introduction, we term the activation values and

300

[Figure 12]

Coding

112% 6% 8% 7% 8% 4% 8%

250

Ethical

10% 210% 21% 14% 10% 15% 24%

PrunedFunctionality

200

Knowledge

67% 296% 122% 81% 79% 184% 86%

Linguistic

16% 8% 7% 59% 7% 26% 10%

150

Math

15% 3% 9% 10% 147% 9% 7%

100

Translation

13% 10% 12% 26% 5% 444% 11%

50

Writing

5% 8% 11% 7% 5% 9% 32%

CodingEthicalKnowledgeLinguisticMathTranslationWriting

Target Functionality

(a) Llama-3-8B-Instruct

300

[Figure 13]

Coding

118% 0% 6% 10% 7% 9% 7%

250

Ethical

5% 1774% 14% 13% 4% 10% 14%

PrunedFunctionality

200

Knowledge

11% 105% 375% 19% 9% 23% 17%

Linguistic

150

11% 3% 8% 115% 8% 31% 9%

Math

19% 4% 12% 17% 200% 14% 6%

100

Translation

377% 391% 477% 381% 241% 9243% 440%

50

Writing

3% 16% 10% 4% 3% 9% 41%

0

CodingEthicalKnowledgeLinguisticMathTranslationWriting

Target Functionality

(b) Mistral-7B-Instruct-v0.3

Figure 9: Increase in perplexity after pruning neurons for specific functionality: PPLprunedPPL−PPLorigin

(%).

origin

output magnitudes as indicators. Then, to further evaluate the impact of these neurons with low indicators on the model performance, we further assess the loss variation with these neurons masked. Specifically, given an input sequence, we first compute the intermediate indicators for all tokens and in all FFN layers. Then we can select and mask k% neurons with the lowest indicators for each token and each layer.

The results are shown in Figure 6 and Figure 7. The results indicate that: (1) The normalized indicators for 80% neurons are lower than 0.2. It indicates that the impact of neurons is present in a long-tail distribution, and only a few neurons have a significant impact on the output of FFN layers. (2) The distributions are similar across different layers and two decoder-only models with different training data. It proves that the sparsity issues widely exist in LLMs. (3) Across all data from the seven functionalities, when the proportion of masked neurons is low (70% when using activation values as indicators, 80% when using output magnitudes as indicators), there is almost no decline in the model performance. This further demonstrates that similar to human brains, LLM exhibits sparsity in its parameter usage. This suggests the potential to significantly reduce computational costs without compromising performance by using part of the parameters for each input. (4) Compared to activation values, using output magnitudes as indicators results in a greater number of neurons with normalized indicators below 0.2, allowing more neurons to be masked without a performance drop. This indicates that output magnitudes can achieve a higher degree of sparsity. This finding encourages further research to identify more effective indicators for assessing neuron usefulness.

##### 4.4 Functionality Specialization

In this subsection, we discuss the functionality specialization of neurons in LLMs. First, we attempt to locate neurons corresponding to different functionalities within LLMs. We calculate the functionality scores of all neurons across each layer for seven functionalities. As shown in Figure 8, we present the best functionality scores for neurons in each layer. We also provide the functionality scores of randomly activated neurons and the average functionality scores of all neurons in a single FFN layer as baselines. From the results, we can observe that: (1) The best functionality scores across these seven functionalities are significantly higher than those of randomly activated neurons and the mean functionality scores. This demonstrates that each FFN layer contains neurons highly associated with these seven functionalities, indicating that these neurons have differentiated into distinct functionalities during pre-training and alignment processes. (2) The functionalities of Coding, Math, and Translation can achieve functionality scores higher than 0.8 in most layers. This suggests that neurons associated with these three functionalities are more specialized compared to the other four functionalities. Instructions for these functionalities require LLMs to understand or generate sequences distinctly different from the English natural language, hence the neurons activated show high specificity. (3) We also present the

###### 5‰ highest functionality scores across different layers. We can observe that there are large gaps between the best functionality scores and the best 5‰ scores. As mentioned before, the neurons are sparsely activated and there are only a few neurons are highly associated with each specific functionality. Thus, the functionality scores drop quickly with the increase of the number of neurons. (4) The average functionality scores are almost equivalent to the functionality scores of randomly activated neurons, indicating that the functionality

Distribution Similarity

Distribution Similarity

[Figure 14]

[Figure 15]

0.14

0.14

Coding

Coding

1.00 0.00 0.01 0.01 0.03 0.01 0.01

1.00 0.00 0.00 0.01 0.04 0.01 0.00

0.12

0.12

Ethical

Ethical

- 0.00 1.00 0.02 0.01 0.00 0.01 0.04
- 0.01 0.02 1.00 0.01 0.00 0.01 0.02

- 0.00 1.00 0.02 0.02 0.00 0.01 0.04

- 0.00 0.02 1.00 0.01 0.00 0.01 0.01
- 0.01 0.02 0.01 1.00 0.02 0.09 0.00

0.04 0.00 0.00 0.02 1.00 0.00 0.00

- 0.01 0.01 0.01 0.09 0.00 1.00 0.01

0.10

0.10

Knowledge

Knowledge

0.08

0.08

Linguistic

Linguistic

0.01 0.01 0.01 1.00 0.01 0.08 0.00

0.06

0.06

Math

Math

0.03 0.00 0.00 0.01 1.00 0.00 0.00

0.04

0.04

Translation

Translation

0.01 0.01 0.01 0.08 0.00 1.00 0.01

0.02

0.02

Writing

Writing

0.01 0.04 0.02 0.00 0.00 0.01 1.00

0.00 0.04 0.01 0.00 0.00 0.01 1.00

0.00

0.00

CodingEthicalKnowledgeLinguisticMathTranslationWriting

CodingEthicalKnowledgeLinguisticMathTranslationWriting

(a) Llama-3-8B-Instruct

(b) Mistral-7B-Instruct-v0.3

Figure 10: Distribution similarity between specialized neurons of different functionalities, where the similarity between two randomly selected neuron groups are 0.05.

scores of most neurons are close to the random baseline. This further suggests that for each capability, only a small subset of neurons is highly associated with it.

Besides, inspired by Zhang et al. (2023c), we further conduct a perturbation study to verify the functionality specialization of neurons with high functionality scores. In this study, given the specific functionalities, we prune 5% neurons with high functionality scores and evaluate the pruned models on all data from 7 functionalities. For instance, given the coding functionality, we manually set the activation value of the neurons with high functionality scores as zero and evaluated the impact of the pruned neurons on all functionalities. Theoretically, if the pruned neurons are highly specialized to specific functionality, they are supposed to only have an impact on the corresponding functionality and have minor impacts on other functionalities. As all data used in our experiments are generation tasks without a clear task format, we adopt perplexity as our evaluation metric. The results of the perturbation study are shown in Figure 9. From the results, we can observe that:

- (1) Values on the diagonal are generally higher than those of the diagonal. This indicates that after pruning neurons for specific functionality, the model’s performance significantly deteriorates in the corresponding functionality while having less impact on other functionalities. (2) Pruning neurons for knowledge in Llama and neurons for translation in Mistral significantly affect all other functionalities. This may be due to the presence of resident neurons in the FFN (Song et al., 2023b), which are frequently activated for most inputs. Including these neurons when selecting for functional specificity results in a substantial impact on model performance. In the future, we will explore more effective methods to locate function-specific neurons.

- 4.5 Functionality Partition

From experimental results in previous subsections, we can observe that neurons are sparsely activated, and each neuron exhibits specific functionalities. Based on these observations, in this sub-section, we further explore the potential for modular partitioning in LLMs. Similar to the human brain, neurons can be divided into several regions, each region containing neurons specialized for specific capabilities, collaborating yet not interfering with each other. Therefore, in this subsection, we attempt to visualize whether there are distinct partitions within the LLMs across the aforementioned seven capabilities.

Specifically, we compute the distribution similarity between top-5% neurons of different functionalities. The results are shown in Figure 10. From the results, we can observe that: (1) Values on the diagonal significantly outperform those off the diagonal, indicating that neurons for different functionalities are distinctly different.

- (2) The similarity between neurons for translation and linguistic functionalities is greater than the random value, which is due to the need to ensure grammatical correctness in the output language during the translation process. In the future, an important research direction is to explore how to accurately cluster different neurons into distinct groups. This approach could avoid the need to select parameters at a neuron level.

#### 5 Open Problems and Future Directions

##### 5.1 Correlation between Emergent and Customized Bricks

The emergent and customized bricks are the essence of configurable foundation models that make the training and updating of LLMs more flexible and scalable. Configuring LLMs with both emergent and customized bricks can promote decomposing and recombining functionalities for existing LLMs. However, as these two types of bricks acquire capabilities through different stages, there exist subtle discrepancies between their properties. For instance, emergent bricks can learn some outdated factual knowledge from the pretraining corpus, while customized bricks post-processed with updated documents may have the latest but also overlapped knowledge. This could lead to unexpected collisions and redundancy in their functionalities, resulting in potential performance degradation and extra computation costs. We advocate further efforts to better manage the integration and cooperation of emergent and customized bricks for ensuring optimal performance and efficiency in configurable LLMs.

Construction The potential collision and redundancy between the functionalities of emergent and customized bricks can be traced back to their construction process. Though emergent bricks can be human-defined or self-organized, their capabilities are attained through the large-scale pre-training procedure, which is typically conducted in an end-to-end manner, making them relatively hard to interpret and localize. For adaptations to new tasks and knowledge that the existing model does not have, customized bricks are constructed after the pre-training stage with delicately designed structures and learning objectives. However, since it is impossible to enumerate the capabilities and knowledge of existing models, incorporating multiple customized bricks for new capabilities and knowledge can also introduce redundancy and collision.

In addition, the granularities of both emergent and customized bricks have several variations and each of them may possess distinct abilities at different levels. The diverse combinations of emergent and customized bricks with different granularities may lead to varying extents of redundancy and collision of capabilities and knowledge. Therefore, detecting the underlying collision and redundancy between bricks is necessary for constructing customized bricks that effectively align with emergent bricks, which makes it possible to achieve optimal performance at minimal cost.

Utilization The other perspective of avoiding collision and reducing redundancy lies in the joint operations of emergent and customized bricks. As mentioned in § 3.1, emergent bricks tend to be selected by routing due to their relatively limited number. In contrast, customized bricks are retrieved to augment the current model with various external capabilities. Currently, the routing and retrieval processes of emergent and customized bricks are typically conducted independently, ignoring the potential collaboration. Jointly routing and retrieving emergent and customized bricks can benefit mutually, optimizing collision detection and selection efficiency. In addition, compared with integrating bricks at the model level, stitching emergent and customized bricks with varying granularities may improve the efficiency and reusability of configurable LLMs.

##### 5.2 Brick Construction Protocols

Configurable LLMs transform the paradigm of LLM alignment and adaptation from a full-parameter training approach to one focusing on the construction and updating of a limited number of bricks. However, most existing algorithms for brick construction and updating, while not requiring the entire LLMs to be updated, still necessitate involving all LLM parameters in the error backpropagation to compute the gradients of bricks. This means that the brick training process demands substantial computational resources. This leads to a contradiction where the bricks offer computational benefits for inference while still being constrained by traditional, resource-intensive training methods.

Efficient brick construction has emerged as a critical challenge. In configurable LLMs, different bricks exchange information through continuous hidden vectors. The primary objective in constructing a brick is to enable it to comprehend the input hidden vectors and generate output hidden vectors that are information-rich and understandable by subsequent modules. It implies that if one can effectively define the input and output hidden vector spaces of a brick, its construction can be independent of the massive parameters of the original LLM. To this end, Xiao et al. (2023c) and Ni et al. (2022) make preliminary exploration by introducing a small auxiliary model that serves as an emulator for the original LLM. The emulator shares the same brick structure as the original LLM and the hidden vector spaces for inter-brick communication are also pre-aligned with LLM. Each brick in the emulator has significantly fewer parameters compared to its counterpart in the original LLM. Therefore, we can utilize the emulator to construct functional bricks efficiently, which can be directly

applied to the original LLM. Here, the emulator can be regarded as the brick construction protocol designed for LLMs, and bricks built following the protocol can be seamlessly integrated into the LLM.

A unified and efficient brick construction protocol holds immense potential for the collaborative construction of future LLMs, enabling a paradigm akin to open-source code repository development. The protocol allows multiple developers to engage in collaborative LLM training, and brings two key benefits: (1) Protection of Data Privacy: Developers can utilize their private data to construct high-quality bricks without exposing privacy to a central training process. (2) Distributed Model Training: Each developer can develop and share bricks based on a unified protocol, without the need for gradient or data communication between different computational nodes.

Despite these advantages, developing more effective and efficient protocols still requires considerable future efforts to fully realize the potential of this collaborative approach:

(1) Universal Protocols: The emulator-based approach is usually limited to the inherent structure of the emulator, restricting its applicability to the construction of specific types of bricks. For instance, existing studies develop emulators that preserve only the layer-wise structure of origin LLMs, tailored for bricks that are inserted between transformer layers. However, due to the loss of intra-layer vector spaces, the emulator falls short when it comes to constructing bricks within a layer, such as prefixes inserted in attention mechanisms (Li & Liang, 2021). Therefore, how to design universal protocols suitable for multiple types of bricks remains a great challenge. (2) Effective Protocols: Existing emulators created via pruning or distillation, despite their smaller parameter scale, struggle to accurately represent the vector space of LLMs, thus leading to a performance loss. Therefore, a key focus of future research lies in enhancing the ability of small emulators to better approximate the vector space of LLMs.

##### 5.3 Evaluation of Configurable Foundation Models

Configurable foundation models consist of various functional bricks. It introduces a fresh methodology to evaluate models from the perspective of bricks for existing metrics. Besides, the modular structure and further operations for bricks require evaluating the brick decomposition performance, i.e., whether the bricks can effectively support complex brick operations.

Evaluation from the Perspective of Bricks Traditional evaluation methods and metrics usually treat the given LLMs as black-box systems, assessing the ability to generate responses that meet predefined requirements given specific instructions. However, such evaluations typically employ coarse-grained metrics that fail to capture the fine-grained performance. For instance, many efforts use quality scores of model responses or the winning rate against reference responses as metrics for LLM alignment. Such methods can provide coarsegrained performance evaluation but cannot measure the performance in intention understanding, multi-step reasoning, and other fine-grained capabilities. Configurable foundation models provide a new perspective to model evaluation, allowing us to shift from end-to-end black-box evaluation to brick-by-brick capability evaluations. This approach enables more precise identification of model shortages and directly updating bricks in urgent need of improvement. In this regard, some researchers have begun to explore the brick evaluation. Such as, Geva et al. (2022) examine the functionality of neurons in LLMs, and find that some neurons are responsible for generating undesired toxic language, and deactivating them can achieve effective detoxicity.

Evaluation for the Brick Decomposition The configurable foundation model also introduces new requirements for model evaluation, particularly regarding whether the bricks within the model can effectively support the diverse brick operations. In this context, we propose the following evaluation metrics for configurable foundation models: (1) Sparsity: A configurable foundation model, during inference, only needs to select a small subset of bricks with relevant functionalities for computation, thereby enhancing inference efficiency. Thus, the goal is to achieve high performance with the least possible number of bricks engaged for given instructions. The fewer bricks required, the more efficient and sparse the model is considered to be. Some existing efforts focus on actively enhancing model sparsity to minimize the parameters involved for each instruction, thereby improving the computational efficiency of LLMs (Song et al., 2024a; Szatkowski et al., 2024). (2) Coupling: As a core concept in software development, decoupling aims to isolate the code that performs a specific task from the code that performs another task (Raghavan et al., 2012). Indeed, decoupling makes the code more maintainable, reusable, and easier to test (Mo et al., 2016), which is also important for LLMs. In a configurable foundation model, different bricks are required to be combined to achieve complex capabilities. Besides, the updating and growing operation also needs to update the brick parameters for continual learning. These operations require low-dependency relationships between different bricks, allowing each brick to cooperate with others and be reused across various scenarios multiple times. Additionally, low

coupling ensures that changes in one module do not adversely affect the performance of others. In this regard, some efforts have been made to construct task-decoupled knowledge plugins, enabling the reuse of knowledge encoding across various tasks (Zhang et al., 2023b; Xiao et al., 2023b).

##### 5.4 Efficient Brick Computing Frameworks

Decomposing LLMs into bricks allows for computation with only a fraction of the parameters, thereby reducing computational load. However, this approach also introduces additional time for brick selection and memory scheduling. Moreover, decoupling the computations of different bricks shows potential for distributed training. Consequently, to enhance the practicality of configurable foundation models, it is crucial to develop corresponding sparse and heterogeneous computing operators. These research directions are vital for optimizing the efficiency and effectiveness of configurable LLMs, making them more feasible and scalable in diverse computational environments.

Sparse Operator We have introduced numerous bricks. However, to handle specific inputs, we do not need to use all bricks. If only the bricks that are most effective for specific inputs are used and other bricks are ignored, the computational cost can be significantly reduced. However, if the size of the brick is small, sequentially calling multiple bricks will result in a lower utilization rate of CUDA computing units. Therefore, Zhang et al. (2022c) and Cui et al. (2023b) cluster bricks that are frequently used simultaneously and fuse them into one kernel for parallel execution. Liu et al. (2023d) implements sparse operators dynamically based on actual input to aggregate bricks. Given an input, whether a brick is suitable or not generally needs to be judged based on the actual activation value after the calculation of the brick. To avoid these calculations, the above solutions need to perform statistical analysis on a large amount of data beforehand to quickly predict the brick selection plan based on the input. However, during the training process, the parameters of the brick continue to change, and the applicability for input also changes dynamically. The above solutions that require prior statistical analysis can only be applied to the inference stage after the model is fixed, how to apply them to the training stage remains a challenging problem.

Heterogeneous Operator Due to the independence between bricks, bricks can be distributed across different machines for collaborative training and inference. Gshard (Lepikhin et al., 2021) and Switch Transformers (Fedus et al., 2022b) leverage the MoE architecture Shazeer et al. (2017) to distribute multiple bricks across different GPUs for parallel pre-training, efficiently scaling up the model size. In particular, the parameter count of the Switch Transformer has reached the trillion level, which is far beyond the size of single-module models. Recent work (He et al., 2021, 2022a; Zhai et al., 2023; Hwang et al., 2022; Gale et al., 2022) has attempted to solve the problem of load imbalance across different GPUs during MoE training by optimizing brick allocation strategies and scheduling schemes.

During inference, we can place core bricks on servers and custom bricks on user machines (Zhou et al., 2022b; Cui et al., 2023a). In this way, users can conveniently adjust the sub-functions of the model according to their personalized needs, while leaving the core, general, and computation-intensive modules to be computed by the model service provider. On the other hand, when the personalized modules and core modules are placed on different machines, more personalized problems such as how to avoid privacy concerns when transfering private data over the network and how to reduce the inference latency caused by cross-machine communication remain to be solved.

##### 5.5 Multi-Model Cooperation System

In configurable LLMs, individual bricks collaborate to complete complex instructions. However, building bricks from scratch requires the collection of massive data and consumes significant computational resources. In the rapidly evolving AI community, numerous researchers have open-sourced various pre-trained models with unique capabilities, such as image generation, speech recognition, etc. Reusing and combining these models as model-level bricks can cost-effectively construct a multimodal system capable of handling complex instructions (Zolna et al., 2024).

As discussed in § 2.2 and § 3.2, there have already been many attempts to combine multiple models to achieve composite capabilities. For example, different modality models are concatenated to achieve multimodal understanding and generation, or different models act as agents that interact with each other through humanreadable signals. However, implementing a multi-model cooperation system still faces the following challenges:

Scalable Cooperation Most current works focus on the cooperation of a limited number of models and adapt each model to the entire system, often requiring training of the whole system, which incurs significant

overhead. Therefore, designing a highly scalable multi-model framework is an important future direction, which enables the system to efficiently integrate an independently trained model into this multi-model system.

Effective Scheduling and Communication A complex instruction requires different models to perform their duties and coordinate with each other, necessitating that the multi-model system effectively schedules different models and ensures efficient information communication between them. Using human-readable signals for information exchange among different models often leads to information loss. However, the representational spaces of different models vary significantly, and direct interaction using intermediate representations makes it difficult for models to understand each other. To effectively address this issue, one possible approach is to introduce an intermediary model that acts as a bridge and information relay between the different models. Another possible approach is to design a unified intermediate representation form for interactions between different models. Overall, achieving efficient collaboration in complex multi-model systems is an important topic that warrants further research.

#### 6 Conclusion

In this paper, we explore configurable foundation models that consist of emergent bricks generated during pre-training and customized bricks created during post-training. We first describe how the bricks constituting a foundational model are trained and further discuss the capabilities of bricks at different granularities. We summarize the advantages of decomposing the foundation models from a modular perspective, including computational efficiency, parameter reusability, traceable results, sustainable capability growth, and optimization for distributed computing. Furthermore, we define four fundamental operations for configurable bricks, including routing and retrieval, brick combination, brick growing, and brick updating. These four operations enable the completion of complex instructions even when each brick is responsible for a single capability. Finally, we discuss the open problems and challenges that remain unresolved for configurable foundation models. We hope this paper will stimulate further research to construct more efficient and scalable foundation models from a modular perspective.

#### Contributions

The contributions of the authors are listed as follows: Chaojun Xiao, Zhengyan Zhang, Xu Han, Zhiyuan Liu, and Maosong Sun initiated and organized the research. Chaojun Xiao drafted the abstract. Chaojun Xiao, Zhengyan Zhang, and Xu Han drafted the introduction. Zhengyan Zhang, Feng Yao, and Xiaozhi Wang drafted § 2.1. Chaojun Xiao and Xiaozhi Wang drafted § 2.2. Chenyang Song and Shuo Wang drafted § 2.3. Chaojun Xiao and Yuge Tu drafted § 2.4. Yufei Huang drafted § 3.1. Chaojun Xiao drafted § 3.2. Yingfa Chen drafted § 3.3. Chenyang Song drafted § 3.4. Chaojun Xiao, Dazhi Jiang, and Chenyang Song conducted experiments for § 4. Feng Yao drafted § 5.1. Chaojun Xiao drafted § 5.2. Guanyu Lin and Chaojun Xiao drafted § 5.3. Weilin Zhao drafted § 5.4. Chaojun Xiao drafted § 5.5. Chaojun Xiao drafted the conclusion. Jingbo Shang, Huimin Chen, Yankai Lin, Zexuan Zhong, Ao Zhang, and Chenglei Si proofread the paper and provided valuable feedback on the paper structure. Khai Hao Moo and Chenyang Zhao proofread the paper and provided valuable suggestions for grammar correction.

#### References

Leon Ackermann and Xenia Ohmer. On the relationship between skill neurons and robustness in prompt tuning. CoRR, abs/2309.12263, 2023.

Oshin Agarwal, Heming Ge, Siamak Shakeri, and Rami Al-Rfou. Knowledge graph based synthetic corpus generation for knowledge-enhanced language model pre-training. In Proceedings of NAACL-HLT, pp. 3554–3565, 2021.

Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. In Proceedings of ACL-IJCNLP, pp. 7319–7328, 2021.

Takuya Akiba, Makoto Shing, Yujin Tang, Qi Sun, and David Ha. Evolutionary optimization of model merging recipes. arXiv preprint arXiv:2403.13187, 2024.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In Proceedings of NeurIPS, 2022.

Keivan Alizadeh, Iman Mirzadeh, Dmitry Belenko, Karen Khatamifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, and Mehrdad Farajtabar. Llm in a flash: Efficient large language model inference with limited memory. arXiv preprint arXiv:2312.11514, 2023.

Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Neural module networks. In Proceedings of

CVPR, pp. 39–48, 2016. Apple. Introducing Apple’s on-device and server foundation models. 2024. Devansh Arpit, Huan Wang, Yingbo Zhou, and Caiming Xiong. Ensemble of averages: Improving model

selection and boosting performance in domain generalization. In Proceedings of NeurIPS, 2022. Yamini Bansal, Preetum Nakkiran, and Boaz Barak. Revisiting model stitching to compare neural representations. In Proceedings of NeurIPS, pp. 225–236, 2021. Alison L Barth and James FA Poulet. Experimental evidence for sparse firing in the neocortex. Trends in neurosciences, 35(6):345–355, 2012.

David Bau, Jun-Yan Zhu, Hendrik Strobelt, Àgata Lapedriza, Bolei Zhou, and Antonio Torralba. Understanding the role of individual units in a deep neural network. Proc. Natl. Acad. Sci. USA, 117(48):30071–30078, 2020.

Adrian Bejan and Sylvie Lorente. Constructal theory of generation of configuration in nature and engineering. Journal of applied physics, 100(4), 2006.

Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ B. Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. CoRR, abs/2108.07258, 2021.

Antoine Bordes, Nicolas Usunier, Alberto García-Durán, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. In Proceedings of NeurIPS, pp. 2787–2795, 2013.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In Proceedings of NeurIPS 2020, 2020.

Ed Bullmore and Olaf Sporns. Complex brain networks: graph theoretical analysis of structural and functional systems. Nature reviews neuroscience, 10(3):186–198, 2009.

Nicola De Cao, Wilker Aziz, and Ivan Titov. Editing factual knowledge in language models. In Proceedings of EMNLP, pp. 6491–6506, 2021.

Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol Cho, Seunghyun Park, Yunsung Lee, and Sungrae Park. SWAD: domain generalization by seeking flat minima. In Proceedings of NeurIPS, pp. 22405–22418, 2021.

Arslan Chaudhry, Puneet Kumar Dokania, Thalaiyasingam Ajanthan, and Philip H. S. Torr. Riemannian walk for incremental learning: Understanding forgetting and intransigence. In Proceedings of ECCV, volume 11215, pp. 556–572, 2018.

Feilong Chen, Minglun Han, Haozhi Zhao, Qingyang Zhang, Jing Shi, Shuang Xu, and Bo Xu. X-LLM: bootstrapping advanced large language models by treating multi-modalities as foreign languages. CoRR, abs/2305.04160, 2023a.

Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chen Qian, Chi-Min Chan, Yujia Qin, Yaxi Lu, Ruobing Xie, Zhiyuan Liu, Maosong Sun, and Jie Zhou. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors in agents. CoRR, abs/2308.10848, 2023b.

Zitian Chen, Yikang Shen, Mingyu Ding, Zhenfang Chen, Hengshuang Zhao, Erik G. Learned-Miller, and Chuang Gan. Mod-squad: Designing mixture of experts as modular multi-task learners. CoRR, abs/2212.08066, 2022.

Xin Cheng, Yankai Lin, Xiuying Chen, Dongyan Zhao, and Rui Yan. Decouple knowledge from paramters for plug-and-play language modeling. In Proceedings of ACL: Findings, pp. 14288–14308, 2023.

Alexandra Chronopoulou, Matthew E. Peters, Alexander Fraser, and Jesse Dodge. Adaptersoup: Weight averaging to improve generalization of pretrained language models. In Proceedings of EACL: Findings, pp. 2009–2018, 2023.

Adrián Csiszárik, Péter Korösi-Szabó, Ákos K. Matszangosz, Gergely Papp, and Dániel Varga. Similarity and matching of neural network representations. In Proceedings of NeurIPS, pp. 5656–5668, 2021.

Ganqu Cui, Wentao Li, Ning Ding, Longtao Huang, Zhiyuan Liu, and Maosong Sun. Decoder tuning: Efficient language understanding as decoding. In Proceedings of ACL, pp. 15072–15087, 2023a.

Weihao Cui, Zhenhua Han, Lingji Ouyang, Yichuan Wang, Ningxin Zheng, Lingxiao Ma, Yuqing Yang, Fan Yang, Jilong Xue, Lili Qiu, Lidong Zhou, Quan Chen, Haisheng Tan, and Minyi Guo. Optimizing dynamic neural networks with brainstorm. In Proceedings of OSDI, pp. 797–815, 2023b.

Nico Daheim, Nouha Dziri, Mrinmaya Sachan, Iryna Gurevych, and Edoardo M. Ponti. Elastic weight removal for faithful and abstractive dialogue generation. CoRR, abs/2303.17574, 2023.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. In Proceedings of ACL, pp. 8493–8502, 2022.

Fahim Dalvi, Nadir Durrani, Hassan Sajjad, Yonatan Belinkov, Anthony Bau, and James R. Glass. What is one grain of sand in the desert? analyzing individual neurons in deep NLP models. In Proceedings of AAAI, pp. 6309–6317, 2019.

Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero Molino, Jason Yosinski, and Rosanne Liu. Plug and play language models: A simple approach to controlled text generation. In Proceedings of ICLR, 2020.

Michiel de Jong, Yury Zemlyanskiy, Nicholas FitzGerald, Fei Sha, and William W. Cohen. Mention memory: incorporating textual knowledge into transformers through entity mention attention. In Proceedings of ICLR, 2022.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT, pp. 4171–4186, 2019.

Ning Ding, Shengding Hu, Weilin Zhao, Yulin Chen, Zhiyuan Liu, Haitao Zheng, and Maosong Sun. Openprompt: An open-source framework for prompt-learning. In Proceedings of ACL, pp. 105–113, 2022.

Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, et al. Parameter-efficient fine-tuning of large-scale pre-trained language models. Nat. Mac. Intell., 5(3):220–235, 2023.

Katharina Dobs, Julio Martinez, Alexander JE Kell, and Nancy Kanwisher. Brain-like functional specialization emerges spontaneously in deep neural networks. Science advances, 8(11):eabl8913, 2022.

Harry Dong, Beidi Chen, and Yuejie Chi. Prompt-prompted mixture of experts for efficient llm generation. arXiv preprint arXiv:2404.01365, 2024.

Qingxiu Dong, Damai Dai, Yifan Song, Jingjing Xu, Zhifang Sui, and Lei Li. Calibrating factual knowledge in pretrained language models. In Proceedings of EMNLP: Findings, pp. 5937–5947, 2022.

Cícero Nogueira dos Santos, Zhe Dong, Daniel Cer, John Nham, Siamak Shakeri, Jianmo Ni, and Yun-Hsuan Sung. Knowledge prompts: Injecting world knowledge into language models through soft prompts. CoRR, abs/2210.04726, 2022.

Felix Draxler, Kambis Veschgini, Manfred Salmhofer, and Fred A. Hamprecht. Essentially no barriers in neural network energy landscape. In Proceedings of ICML, volume 80, pp. 1308–1317, 2018.

Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. Glam: Efficient scaling of language models with mixture-ofexperts. In Proceedings of ICML, volume 162, pp. 5547–5569, 2022.

Angela Fan, Edouard Grave, and Armand Joulin. Reducing transformer depth on demand with structured dropout. In Proceedings of ICLR, 2020.

Homa Fashandi. Neural module networks: A review. Neurocomputing, 552:126518, 2023. William Fedus, Jeff Dean, and Barret Zoph. A review of sparse expert models in deep learning. CoRR,

abs/2209.01667, 2022a. William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. J. Mach. Learn. Res., 23:120:1–120:39, 2022b. Thibault Févry, Livio Baldini Soares, Nicholas FitzGerald, Eunsol Choi, and Tom Kwiatkowski. Entities as

experts: Sparse memory access with entity supervision. In Proceedings of EMNLP, pp. 4937–4951, 2020. Jerry A Fodor. The modularity of mind. 1983. Negar Foroutan, Mohammadreza Banaei, Rémi Lebret, Antoine Bosselut, and Karl Aberer. Discovering

language-neutral sub-networks in multilingual language models. In Proceedings of EMNLP, pp. 7560–7575, 2022.

Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In Proceedings of ICLR, 2019.

Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis. In Proceedings of ICML, volume 119, pp. 3259–3269, 2020.

Dan Friedman, Ben Dodge, and Danqi Chen. Single-dataset experts for multi-dataset question answering. In Proceedings of EMNLP, pp. 6128–6137, 2021.

Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia. Megablocks: Efficient sparse training with mixture-of-experts. CoRR, abs/2211.15841, 2022.

Difei Gao, Lei Ji, Luowei Zhou, Kevin Qinghong Lin, Joya Chen, Zihan Fan, and Mike Zheng Shou. Assistgpt: A general multi-modal assistant that can plan, execute, inspect, and learn. CoRR, abs/2306.08640, 2023.

Tianyu Gao, Adam Fisch, and Danqi Chen. Making pre-trained language models better few-shot learners. In Proceedings of ACL-IJCNLP, pp. 3816–3830, 2021.

Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry P. Vetrov, and Andrew Gordon Wilson. Loss

surfaces, mode connectivity, and fast ensembling of dnns. In Proceedings of NeurIPS, pp. 8803–8812, 2018. Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value

memories. In Proceedings of EMNLP, pp. 5484–5495, 2021. Mor Geva, Avi Caciularu, Kevin Ro Wang, and Yoav Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Proceedings of EMNLP, pp. 30–45, 2022.

In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, and Lin Zhong. Prompt cache: Modular attention reuse for low-latency inference. Proceedings of Machine Learning and Systems, 6: 325–338, 2024.

Linyuan Gong, Di He, Zhuohan Li, Tao Qin, Liwei Wang, and Tieyan Liu. Efficient training of bert by progressively stacking. In Proceedings of ICML, pp. 2337–2346. PMLR, 2019.

Kalanit Grill-Spector and Rafael Malach. The human visual cortex. Annu. Rev. Neurosci., 27:649–677, 2004. Xiaotao Gu, Liyuan Liu, Hongkun Yu, Jing Li, Chen Chen, and Jiawei Han. On the transformer growth for

progressive bert training. In Proceedings of ACL, pp. 5174–5180, 2021. Demi Guo, Alexander M. Rush, and Yoon Kim. Parameter-efficient transfer learning with diff pruning. In Proceedings of ACL/IJCNLP, pp. 4884–4896, 2021. Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, and Dimitris Bertsimas. Finding neurons in a haystack: Case studies with sparse probing. CoRR, abs/2305.01610, 2023.

Suchin Gururangan, Mike Lewis, Ari Holtzman, Noah A. Smith, and Luke Zettlemoyer. Demix layers: Disentangling domains for modular language modeling. In Proceedings of NAACL, pp. 5557–5576, 2022.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Retrieval augmented language model pre-training. In Proceedings of ICML, volume 119, pp. 3929–3938, 2020.

Xu Han, Yi Dai, Tianyu Gao, Yankai Lin, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. Continual relation learning via episodic memory activation and reconsolidation. In Proceedings of ACL, pp. 6429–6440, 2020.

Xu Han, Zhengyan Zhang, Ning Ding, Yuxian Gu, Xiao Liu, Yuqi Huo, Jiezhong Qiu, Yuan Yao, Ao Zhang, et al. Pre-trained models: Past, present and future. AI Open, 2:225–250, 2021a.

Xu Han, Zhengyan Zhang, and Zhiyuan Liu. Knowledgeable machine learning for natural language processing. Commun. ACM, 64(11):50–51, 2021b.

Yizeng Han, Gao Huang, Shiji Song, Le Yang, Honghui Wang, and Yulin Wang. Dynamic neural networks: A survey. CoRR, abs/2102.04906, 2021c.

Rui Hao, Linmei Hu, Weijian Qi, Qingliu Wu, Yirui Zhang, and Liqiang Nie. Chatllm network: More brains, more intelligence. CoRR, abs/2304.12998, 2023a.

Shibo Hao, Tianyang Liu, Zhen Wang, and Zhiting Hu. Toolkengpt: Augmenting frozen language models with massive tools via tool embeddings. CoRR, abs/2305.11554, 2023b.

Peter Hase, Mohit Bansal, Been Kim, and Asma Ghandeharioun. Does localization inform editing? surprising differences in causality-based localization vs. knowledge editing in language models. CoRR, abs/2301.04213, 2023.

Jiaao He, Jiezhong Qiu, Aohan Zeng, Zhilin Yang, Jidong Zhai, and Jie Tang. Fastmoe: A fast mixture-ofexpert training system. CoRR, abs/2103.13262, 2021.

Jiaao He, Jidong Zhai, Tiago Antunes, Haojie Wang, Fuwen Luo, Shangfeng Shi, and Qin Li. Fastermoe: modeling and optimizing training of large-scale dynamic pre-trained models. In Proceedings of SIGPLAN, pp. 120–134, 2022a.

Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, and Graham Neubig. Towards a unified view of parameter-efficient transfer learning. In Proceedings of ICLR, 2022b.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of CVPR, pp. 770–778, 2016.

Roee Hendel, Mor Geva, and Amir Globerson. In-context learning creates task vectors. In Proceedings of EMNLP: Findings, pp. 9318–9333, 2023.

Evan Hernandez, Belinda Z. Li, and Jacob Andreas. Inspecting and editing knowledge representations in language models, 2023.

John Hewitt and Christopher D. Manning. A structural probe for finding syntax in word representations. In Proceedings of NAACL, pp. 4129–4138, Minneapolis, Minnesota, June 2019.

Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai Taitelbaum, Doron Kukliansy, Vered Cohen, Thomas Scialom, Idan Szpektor, Avinatan Hassidim, and Yossi Matias. TRUE: re-evaluating factual consistency evaluation. In Proceedings of NAACL, pp. 3905–3920, 2022.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for NLP. In Proceedings of ICML, volume 97, pp. 2790–2799, 2019.

Cheng-Yu Hsieh, Si-An Chen, Chun-Liang Li, Yasuhisa Fujii, Alexander Ratner, Chen-Yu Lee, Ranjay Krishna, and Tomas Pfister. Tool documentation enables zero-shot tool-usage with large language models. CoRR, abs/2308.00675, 2023.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In Proceedings of ICLR, 2022.

Jinyi Hu, Xu Han, Xiaoyuan Yi, Yutong Chen, Wenhao Li, Zhiyuan Liu, and Maosong Sun. Efficient cross-lingual transfer for chinese stable diffusion with images as pivots. CoRR, abs/2305.11540, 2023.

Ronghang Hu, Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Kate Saenko. Learning to reason: End-to-end module networks for visual question answering. In Proceedings of ICCV, pp. 804–813, 2017.

Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. Lorahub: Efficient cross-task generalization via dynamic lora composition. CoRR, abs/2307.13269, 2023a.

Kaiyu Huang, Peng Li, Jin Ma, Ting Yao, and Yang Liu. Knowledge transfer in incremental learning for multilingual neural machine translation. In Proceedings of ACL, pp. 15286–15304, Toronto, Canada, July 2023b.

Zeyu Huang, Yikang Shen, Xiaofeng Zhang, Jie Zhou, Wenge Rong, and Zhang Xiong. Transformer-patcher: One mistake worth one neuron. In Proceedings of ICLR, 2023c.

Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu, Zilong Wang, Rafael Salas, Jithin Jose, Prabhat Ram, Joe Chau, Peng Cheng, Fan Yang, Mao Yang, and Yongqiang Xiong. Tutel: Adaptive mixture-of-experts at scale. CoRR, abs/2206.03382, 2022.

Gabriel Ilharco, Marco Túlio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In Proceedings of ICLR, 2023.

David Isele and Akansel Cosgun. Selective experience replay for lifelong learning. In Proceedings of AAAI-18, pp. 3302–3309, 2018.

Sebastian Jaszczur, Aakanksha Chowdhery, Afroz Mohiuddin, Lukasz Kaiser, Wojciech Gajewski, Henryk Michalewski, and Jonni Kanerva. Sparse is enough in scaling transformers. In Proceedings of NeurIPS, pp. 9895–9907, 2021.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Comput. Surv., 55(12): 248:1–248:38, 2023.

Zhengbao Jiang, Frank F. Xu, Jun Araki, and Graham Neubig. How can we know what language models know. Trans. Assoc. Comput. Linguistics, 8:423–438, 2020.

Xisen Jin, Dejiao Zhang, Henghui Zhu, Wei Xiao, Shang-Wen Li, Xiaokai Wei, Andrew O. Arnold, and Xiang Ren. Lifelong pretraining: Continually adapting language models to emerging corpora. In Proceedings of NAACL, pp. 4764–4780, 2022.

Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang Cheng. Dataless knowledge fusion by merging weights of language models. In Proceedings of ICLR, 2023.

Ákos Kádár, Grzegorz Chrupala, and Afra Alishahi. Representation of linguistic form and function in recurrent neural networks. Comput. Linguistics, 43(4), 2017.

Jean Kaddour, Joshua Harris, Maximilian Mozes, Herbie Bradley, Roberta Raileanu, and Robert McHardy. Challenges and applications of large language models. arXiv preprint arXiv:2307.10169, 2023.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. CoRR, abs/2001.08361, 2020.

Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J. Reddi, Sebastian U. Stich, and Ananda Theertha Suresh. SCAFFOLD: stochastic controlled averaging for federated learning. In Proceedings of ICML, volume 119, pp. 5132–5143, 2020.

Andrej Karpathy, Justin Johnson, and Li Fei-Fei. Visualizing and understanding recurrent networks. CoRR, abs/1506.02078, 2015.

Jason ND Kerr, David Greenberg, and Fritjof Helmchen. Imaging input and output of neocortical networks in vivo. Proceedings of the National Academy of Sciences, 102(39):14063–14068, 2005.

James Kirkpatrick, Razvan Pascanu, Neil C. Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. CoRR, abs/1612.00796, 2016.

Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, and Neil Houlsby. Sparse upcycling: Training mixture-of-experts from dense checkpoints. In Proceedings of ICLR, 2023.

Stefanos Laskaridis, Alexandros Kouris, and Nicholas D. Lane. Adaptive inference through early-exit networks: Design, challenges and directions. In Proceedings of EMDL@MobiSys, pp. 1–6, 2021.

Anne Lauscher, Tobias Lüken, and Goran Glavas. Sustainable modular debiasing of language models. In Proceedings of EMNLP: Findings, pp. 4782–4797, 2021.

Sang-Woo Lee, Jin-Hwa Kim, Jaehyun Jun, Jung-Woo Ha, and Byoung-Tak Zhang. Overcoming catastrophic forgetting by incremental moment matching. In Proceedings of NeurIPS, pp. 4652–4662, 2017.

Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. In Proceedings of ICLR, 2021.

Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In Proceedings of EMNLP, pp. 3045–3059, 2021.

Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, and Luke Zettlemoyer. BASE layers: Simplifying training of large, sparse models. In Proceedings of ICML, volume 139, pp. 6265–6274, 2021.

Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrievalaugmented generation for knowledge-intensive NLP tasks. In Proceedings of NeurIPS, 2020.

Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. Measuring the intrinsic dimension of objective landscapes. In Proceedings of ICLR, 2018.

Jialong Li, Mingyue Zhang, Nianyu Li, Danny Weyns, Zhi Jin, and Kenji Tei. Generative ai for self-adaptive systems: State of the art and research roadmap. ACM Transactions on Autonomous and Adaptive Systems, 2024.

Jiwei Li, Xinlei Chen, Eduard H. Hovy, and Dan Jurafsky. Visualizing and understanding neural models in NLP. In Proceedings of NAACL-HLT, pp. 681–691, 2016.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In Proceedings of ICML, volume 202, pp. 19730–19742, 2023a.

Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. CoRR, abs/2305.06355, 2023b.

Lei Li, Yankai Lin, Deli Chen, Shuhuai Ren, Peng Li, Jie Zhou, and Xu Sun. CascadeBERT: Accelerating inference of pre-trained language models via calibrated complete models cascade. In Proceedings of EMNLP: Findings, pp. 475–486, Punta Cana, Dominican Republic, November 2021.

Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A. Smith, and Luke Zettlemoyer. Branch-train-merge: Embarrassingly parallel training of expert language models. CoRR, abs/2208.03306, 2022.

Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua Zhang. On the convergence of fedavg on non-iid data. In Proceedings of ICLR, 2020.

Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of ACL-IJCNLP, pp. 4582–4597, 2021.

Zhizhong Li and Derek Hoiem. Learning without forgetting. In Proceedings of ECCV, volume 9908, pp. 614–629, 2016.

Zonglin Li, Chong You, Srinadh Bhojanapalli, Daliang Li, Ankit Singh Rawat, Sashank J. Reddi, Ke Ye, Felix Chern, Felix X. Yu, Ruiqi Guo, and Sanjiv Kumar. The lazy neuron phenomenon: On emergence of activation sparsity in transformers. In Proceedings of ICLR, 2023c.

Chen Liang, Simiao Zuo, Minshuo Chen, Haoming Jiang, Xiaodong Liu, Pengcheng He, Tuo Zhao, and Weizhu Chen. Super tickets in pre-trained language models: From model compression to improving generalization. In Proceedings of ACL-IJCNLP, pp. 6524–6538, 2021.

Bill Yuchen Lin, Yicheng Fu, Karina Yang, Prithviraj Ammanabrolu, Faeze Brahman, Shiyu Huang, Chandra Bhagavatula, Yejin Choi, and Xiang Ren. Swiftsage: A generative agent with fast and slow thinking for complex interactive tasks, 2023.

Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings of ACL, pp. 3214–3252, 2022.

Yongjie Lin, Yi Chern Tan, and Robert Frank. Open sesame: Getting inside BERT’s linguistic knowledge. In Proceedings of ACL: Workshop BlackboxNLP, pp. 241–253, Florence, Italy, August 2019.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. CoRR, abs/2304.08485, 2023a.

Nelson F. Liu, Matt Gardner, Yonatan Belinkov, Matthew E. Peters, and Noah A. Smith. Linguistic knowledge and transferability of contextual representations. In Proceedings of NAACL, pp. 1073–1094, Minneapolis, Minnesota, June 2019.

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Comput. Surv., 55(9):195:1–195:35, 2023b.

Sheng Liu, Lei Xing, and James Zou. In-context vectors: Making in context learning more effective and controllable through latent space steering. CoRR, abs/2311.06668, 2023c.

Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Ré, and Beidi Chen. Deja vu: Contextual sparsity for efficient llms at inference time. In Proceedings of ICML, volume 202, pp. 22137–22176, 2023d.

Gen Luo, Yiyi Zhou, Tianhe Ren, Shengxin Chen, Xiaoshuai Sun, and Rongrong Ji. Cheap and quick: Efficient vision-language instruction tuning for large language models. CoRR, abs/2305.15023, 2023.

Andrea Madotto, Zhaojiang Lin, Zhenpeng Zhou, Seungwhan Moon, Paul A. Crook, Bing Liu, Zhou Yu, Eunjoon Cho, Pascale Fung, and Zhiguang Wang. Continual learning in task-oriented dialogue systems. In Proceedings of EMNLP, pp. 7452–7467, 2021.

Rabeeh Karimi Mahabadi, Sebastian Ruder, Mostafa Dehghani, and James Henderson. Parameter-efficient multi-task fine-tuning for transformers via shared hypernetworks. In Proceedings of ACL-IJCNLP, pp. 565–576, 2021.

Michael Matena and Colin Raffel. Merging models with fisher-weighted averaging. In Proceedings of NeurIPS, 2022.

Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan T. McDonald. On faithfulness and factuality in abstractive summarization. In Proceedings of ACL, pp. 1906–1919, 2020.

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas. Communicationefficient learning of deep networks from decentralized data. In Proceedings of AISTATS, volume 54, pp. 1273–1282, 2017.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Proceedings of NeurIPS, 2022.

Kevin Meng, Arnab Sen Sharma, Alex J. Andonian, Yonatan Belinkov, and David Bau. Mass-editing memory in a transformer. In Proceedings of ICLR, 2023.

David Meunier, Renaud Lambiotte, and Edward T Bullmore. Modular and hierarchically modular organization of brain networks. Frontiers in neuroscience, 4:200, 2010.

Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos Nalmpantis, Ramakanth Pasunuru, Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, Edouard Grave, Yann LeCun, and Thomas Scialom. Augmented language models: a survey. CoRR, abs/2302.07842, 2023.

Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? In Proceedings of NeurIPS, pp. 14014–14024, 2019.

Iman Mirzadeh, Keivan Alizadeh, Sachin Mehta, Carlo C Del Mundo, Oncel Tuzel, Golnoosh Samei, Mohammad Rastegari, and Mehrdad Farajtabar. Relu strikes back: Exploiting activation sparsity in large language models, 2023.

MistralAI. Mixtral of experts: A high quality sparse mixture-of-experts. Mistral Report, 2023. Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D. Manning. Fast model editing

at scale. In Proceedings of ICLR, 2022a. Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning, and Chelsea Finn. Memory-based model editing at scale. In Proceedings of ICML, volume 162, pp. 15817–15831, 2022b. Sarthak Mittal, Yoshua Bengio, and Guillaume Lajoie. Is a modular architecture enough? In Proceedings of NeurIPS, 2022. Ran Mo, Yuanfang Cai, Rick Kazman, Lu Xiao, and Qiong Feng. Decoupling level: A new metric for

architectural maintenance complexity. In Proceedings of ICSE, pp. 499–510, 2016. Jesse Mu and Jacob Andreas. Compositional explanations of neurons. In Proceedings of NeurIPS, 2020. Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, and Neil Houlsby. Multimodal

contrastive learning with limoe: the language-image mixture of experts. In Proceedings of NeurIPS, 2022.

Seil Na, Yo Joong Choe, Dong-Hyun Lee, and Gunhee Kim. Discovery of natural language concepts in individual units of cnns. In Proceedings of ICLR, 2019.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. CoRR, abs/2112.09332, 2021.

Zanlin Ni, Yulin Wang, Jiangwei Yu, Haojun Jiang, Yue Cao, and Gao Huang. Deep model assembling. CoRR, abs/2212.04129, 2022.

Bruno A Olshausen and David J Field. Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature, 381(6583):607–609, 1996.

Yasumasa Onoe, Michael J. Q. Zhang, Shankar Padmanabhan, Greg Durrett, and Eunsol Choi. Can lms learn new entities from descriptions? challenges in propagating injected knowledge. In Proceedings of ACL, pp. 5469–5485, 2023.

OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023. Shankar Padmanabhan, Yasumasa Onoe, Michael J. Q. Zhang, Greg Durrett, and Eunsol Choi. Propagating

knowledge updates to lms through distillation. CoRR, abs/2306.09306, 2023. Zizheng Pan, Jianfei Cai, and Bohan Zhuang. Stitchable neural networks. In Proceedings of CVPR, pp. 16102–16112, 2023. Abhishek Panigrahi, Nikunj Saunshi, Haoyu Zhao, and Sanjeev Arora. Task-specific skill localization in fine-tuned language models. In Proceedings of ICML, volume 202, pp. 27011–27033, 2023. Damian Pascual, Beni Egressy, Clara Meister, Ryan Cotterell, and Roger Wattenhofer. A plug-and-play method for controlled text generation. In Proceedings of EMNLP: Findings, pp. 3973–3997, 2021.

Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick S. H. Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander H. Miller. Language models as knowledge bases? In Proceedings of EMNLP-IJCNLP, pp. 2463–2473, 2019.

Jonas Pfeiffer, Ivan Vulic, Iryna Gurevych, and Sebastian Ruder. MAD-X: an adapter-based framework for multi-task cross-lingual transfer. In Proceedings of EMNLP, pp. 7654–7673, 2020.

Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych. Adapterfusion: Non-destructive task composition for transfer learning. In Proceedings of EACL, pp. 487–503, 2021.

Jonas Pfeiffer, Sebastian Ruder, Ivan Vulic, and Edoardo Maria Ponti. Modular deep learning. CoRR, abs/2302.11529, 2023.

Mikolaj Piórczynski, Filip Szatkowski, Klaudia Balazy, and Bartosz Wójcik. Exploiting transformer activation sparsity with dynamic inference. CoRR, abs/2310.04361, 2023.

Cindy Poo and Jeffry S Isaacson. Odor representations in olfactory cortex:“sparse” coding, global inhibition, and oscillations. Neuron, 62(6):850–861, 2009.

Nina Pörner, Ulli Waltinger, and Hinrich Schütze. E-BERT: efficient-yet-effective entity embeddings for BERT. In Proceedings of EMNLP: Findings, volume EMNLP 2020, pp. 803–818, 2020.

Sai Prasanna, Anna Rogers, and Anna Rumshisky. When BERT plays the lottery, all tickets are winning. In Proceedings of EMNLP, pp. 3208–3229, 2020.

Joan Puigcerver, Carlos Riquelme, Basil Mustafa, and Neil Houlsby. From sparse to soft mixtures of experts. CoRR, abs/2308.00951, 2023.

Chen Qian, Xin Cong, Cheng Yang, Weize Chen, Yusheng Su, Juyuan Xu, Zhiyuan Liu, and Maosong Sun. Communicative agents for software development. CoRR, abs/2307.07924, 2023.

Yujia Qin, Xiaozhi Wang, YuSheng Su, Yankai Lin, Ning Ding, Zhiyuan Liu, Juanzi Li, Lei Hou, Peng Li, Maosong Sun, and Jie Zhou. Exploring low-dimensional intrinsic task subspace via prompt tuning. CoRR, abs/2110.07867, 2021.

Yujia Qin, Cheng Qian, Jing Yi, Weize Chen, Yankai Lin, Xu Han, Zhiyuan Liu, Maosong Sun, and Jie Zhou. Exploring mode connectivity for pre-trained language models. In Proceedings of EMNLP, pp. 6726–6746, 2022a.

Yujia Qin, Jiajie Zhang, Yankai Lin, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. ELLE: efficient lifelong pre-training for emerging data. In Proceedings of ACL: Findings, pp. 2789–2810, 2022b.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. CoRR, abs/2307.16789, 2023.

Zihan Qiu, Zeyu Huang, Shuang Cheng, Yizhi Zhou, Zili Wang, Ivan Titov, and Jie Fu. Layerwise recurrent router for mixture-of-experts. arXiv preprint arXiv:2408.06793, 2024.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21:140:1–140:67, 2020.

Barath Raghavan, Martín Casado, Teemu Koponen, Sylvia Ratnasamy, Ali Ghodsi, and Scott Shenker. Software-defined internet architecture: decoupling architecture from infrastructure. In Proceedings of HotNets, pp. 43–48, 2012.

Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation AI scale. In Proceedings of ICML, volume 162, pp. 18332–18346, 2022.

Alexandre Ramé, Matthieu Kirchmeyer, Thibaud Rahier, Alain Rakotomamonjy, Patrick Gallinari, and Matthieu Cord. Diverse weight averaging for out-of-distribution generalization. In Proceedings of NeurIPS, 2022.

Anastasia Razdaibiedina, Yuning Mao, Rui Hou, Madian Khabsa, Mike Lewis, and Amjad Almahairi. Progressive prompts: Continual learning for language models. In Proceedings of ICLR, 2023.

Adam Roberts, Colin Raffel, and Noam Shazeer. How much knowledge can you pack into the parameters of a language model? In Proceedings of EMNLP, pp. 5418–5426, 2020.

Anna Rogers, Olga Kovaleva, and Anna Rumshisky. A primer in BERTology: What we know about how BERT works. Transactions of the Association for Computational Linguistics, 8:842–866, 2020.

Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, and Jason Weston. Hash layers for large sparse models. In Proceedings of NeurIPS, pp. 17555–17566, 2021.

David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy P. Lillicrap, and Gregory Wayne. Experience replay for continual learning. In Proceedings of NeurIPS, pp. 348–358, 2019.

Yangjun Ruan, Saurabh Singh, Warren Richard Morningstar, Alexander A. Alemi, Sergey Ioffe, Ian Fischer, and Joshua V. Dillon. Weighted ensemble self-supervised learning. In Proceedings of ICLR, 2023.

Jon Saad-Falcon, Amanpreet Singh, Luca Soldaini, Mike D’Arcy, Arman Cohan, and Doug Downey. Embedding recycling for language models. In Proceedings of EACL: Findings, pp. 1888–1908, 2023.

Hassan Sajjad, Nadir Durrani, and Fahim Dalvi. Neuron-level interpretation of deep NLP models: A survey. Trans. Assoc. Comput. Linguistics, 10:1285–1303, 2022.

Timo Schick and Hinrich Schütze. Exploiting cloze-questions for few-shot text classification and natural language inference. In Proceedings of EACL, pp. 255–269, 2021.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In Proceedings of ICLR, 2017.

Sheng Shen, Le Hou, Yanqi Zhou, Nan Du, Shayne Longpre, Jason Wei, Hyung Won Chung, Barret Zoph, William Fedus, Xinyun Chen, et al. Mixture-of-experts meets instruction tuning:a winning combination for large language models, 2023a.

Yikang Shen, Zheyu Zhang, Tianyou Cao, Shawn Tan, Zhenfang Chen, and Chuang Gan. Moduleformer: Learning modular large language models from uncurated data. CoRR, abs/2306.04640, 2023b.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving AI tasks with chatgpt and its friends in huggingface. CoRR, abs/2303.17580, 2023c.

Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. REPLUG: retrieval-augmented black-box language models. CoRR, abs/2301.12652, 2023.

Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, and Sameer Singh. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. In Proceedings of EMNLP, pp. 4222–4235, 2020.

Herbert A Simon. The architecture of complexity. Proceedings of the American philosophical society, 106(6): 467–482, 1962.

Chenyang Song, Xu Han, Zheni Zeng, Kuai Li, Chen Chen, Zhiyuan Liu, Maosong Sun, and Tao Yang. Conpet: Continual parameter-efficient tuning for large language models. CoRR, abs/2309.14763, 2023a.

Chenyang Song, Xu Han, Zhengyan Zhang, Shengding Hu, Xiyu Shi, Kuai Li, Chen Chen, Zhiyuan Liu, Guangli Li, Tao Yang, et al. Prosparse: Introducing and enhancing intrinsic activation sparsity within large language models. arXiv preprint arXiv:2402.13516, 2024a.

Yixin Song, Zeyu Mi, Haotong Xie, and Haibo Chen. Powerinfer: Fast large language model serving with a consumer-grade gpu. arXiv preprint arXiv:2312.12456, 2023b.

Yixin Song, Haotong Xie, Zhengyan Zhang, Bo Wen, Li Ma, Zeyu Mi, and Haibo Chen. Turbo sparse: Achieving llm sota performance with minimal activated parameters. arXiv preprint arXiv:2406.05955, 2024b.

Mark Stefik and Daniel G. Bobrow. Object-oriented programming: Themes and variations. AI Mag., 6(4): 40–62, 1986.

Yusheng Su, Xiaozhi Wang, Yujia Qin, Chi-Min Chan, Yankai Lin, Huadong Wang, Kaiyue Wen, Zhiyuan Liu, Peng Li, Juanzi Li, Lei Hou, Maosong Sun, and Jie Zhou. On transferability of prompt tuning for natural language processing. In Proceedings of NAACL, pp. 3949–3969, 2022.

Yusheng Su, Chi-Min Chan, Jiali Cheng, Yujia Qin, Yankai Lin, Shengding Hu, Zonghan Yang, Ning Ding, Zhiyuan Liu, and Maosong Sun. Arbitrary few parameters are good enough for adapting large-scale pre-trained language models. CoRR, abs/2306.02320, 2023.

Xavier Suau, Luca Zappella, and Nicholas Apostoloff. Finding experts in transformer models. CoRR, abs/2005.07647, 2020.

Tianxiang Sun, Yunfan Shao, Xipeng Qiu, Qipeng Guo, Yaru Hu, Xuanjing Huang, and Zheng Zhang. Colake: Contextualized language and knowledge embedding. In Proceedings of COLING, pp. 3660–3670, 2020.

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proceedings of ICML, volume 70, pp. 3319–3328, 2017.

Filip Szatkowski, Bartosz Wójcik, Mikołaj Piórczy´nski, and Kamil Adamczewski. Sadmoe: Exploiting activation sparsity with dynamic-k gating. arXiv preprint arXiv:2402.13516, 2024.

Tianyi Tang, Wenyang Luo, Haoyang Huang, Dongdong Zhang, Xiaolei Wang, Xin Zhao, Furu Wei, and Ji-Rong Wen. Language-specific neurons: The key to multilingual capabilities in large language models. arXiv preprint arXiv:2402.16438, 2024.

Zhengwei Tao, Ting-En Lin, Xiancai Chen, Hangyu Li, Yuchuan Wu, Yongbin Li, Zhi Jin, Fei Huang, Dacheng Tao, and Jingren Zhou. A survey on self-evolution of large language models. arXiv preprint arXiv:2404.14387, 2024.

Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, and David Bau. Function vectors in large language models. CoRR, abs/2310.15213, 2023.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288, 2023b.

Alexander Matt Turner, Lisa Thiergart, David Udell, Gavin Leech, Ulisse Mini, and Monte MacDiarmid. Activation addition: Steering language models without optimization. CoRR, abs/2308.10248, 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Proceedings of NeurIPS, pp. 5998–6008, 2017.

Vijay Viswanathan, Chenyang Zhao, Amanda Bertsch, Tongshuang Wu, and Graham Neubig. Prompt2model:

Generating deployable models from natural language instructions. arXiv preprint arXiv:2308.12261, 2023. Elena Voita, Javier Ferrando, and Christoforos Nalmpantis. Neurons in large language models: Dead, n-gram,

positional. CoRR, abs/2309.04827, 2023.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of ICLR, 2019a.

Hong Wang, Wenhan Xiong, Mo Yu, Xiaoxiao Guo, Shiyu Chang, and William Yang Wang. Sentence

embedding alignment for lifelong relation extraction. In Proceedings of NAACL-HLT, pp. 796–806, 2019b. Huiyi Wang, Haodong Lu, Lina Yao, and Dong Gong. Self-expansion of pre-trained models with mixture of

adapters for continual learning. arXiv preprint arXiv:2403.18886, 2024.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Ji-Rong Wen. A survey on large language model based autonomous agents. CoRR, abs/2308.11432, 2023a.

Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. A comprehensive survey of continual learning: Theory, method and application. CoRR, abs/2302.00487, 2023b.

Peihao Wang, Rameswar Panda, Lucas Torroba Hennigen, Philip Greengard, Leonid Karlinsky, Rogério Feris, David Daniel Cox, Zhangyang Wang, and Yoon Kim. Learning to grow pretrained models for efficient transformer training. In Proceedings of ICLR, 2023c.

Peng Wang, Ningyu Zhang, Xin Xie, Yunzhi Yao, Bozhong Tian, Mengru Wang, Zekun Xi, Siyuan Cheng, Kangwei Liu, Guozhou Zheng, and Huajun Chen. Easyedit: An easy-to-use knowledge editing framework for large language models, 2023d.

Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, and Lidia S. Chao. Learning deep transformer models for machine translation. In Proceedings of ACL, pp. 1810–1822, Florence, Italy, July 2019c.

Ruize Wang, Duyu Tang, Nan Duan, Zhongyu Wei, Xuanjing Huang, Jianshu Ji, Guihong Cao, Daxin Jiang, and Ming Zhou. K-adapter: Infusing knowledge into pre-trained models with adapters. In Proceedings of ACL: Findings, volume ACL/IJCNLP 2021, pp. 1405–1418, 2021.

Song Wang, Yaochen Zhu, Haochen Liu, Zaiyi Zheng, Chen Chen, and Jundong Li. Knowledge editing for large language models: A survey. CoRR, abs/2310.16218, 2023e.

Xiaozhi Wang, Kaiyue Wen, Zhengyan Zhang, Lei Hou, Zhiyuan Liu, and Juanzi Li. Finding skill neurons in pre-trained transformer-based language models. In Proceedings of EMNLP, pp. 11132–11152, 2022a.

Zifeng Wang, Zizhao Zhang, Sayna Ebrahimi, Ruoxi Sun, Han Zhang, Chen-Yu Lee, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer G. Dy, and Tomas Pfister. Dualprompt: Complementary prompting for rehearsal-free continual learning. In Proceedings of ECCV, volume 13686, pp. 631–648, 2022b. doi: 10.1007/978-3-031-19809-0\_36.

Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer G. Dy, and Tomas Pfister. Learning to prompt for continual learning. In Proceedings of CVPR, pp. 139–149, 2022c.

Ziheng Wang, Jeremy Wohlwend, and Tao Lei. Structured pruning of large language models. In Proceedings of EMNLP, pp. 6151–6162, Online, November 2020. doi: 10.18653/v1/2020.emnlp-main.496.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. Trans. Mach. Learn. Res., 2022, 2022a.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Proceedings of NeurIPS, 2022b.

Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In Proceedings of ICML, volume 162, pp. 23965–23998, 2022.

Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. CoRR, abs/2303.04671, 2023a.

Chengyue Wu, Yukang Gan, Yixiao Ge, Zeyu Lu, Jiahao Wang, Ye Feng, Ping Luo, and Ying Shan. Llama pro: Progressive llama with block expansion. arXiv preprint arXiv:2401.02415, 2024.

Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal LLM. CoRR, abs/2309.05519, 2023b.

Tongtong Wu, Massimo Caccia, Zhuang Li, Yuan-Fang Li, Guilin Qi, and Gholamreza Haffari. Pretrained language model in continual learning: A comparative study. In Proceedings of ICLR, 2022.

Mengzhou Xia, Zexuan Zhong, and Danqi Chen. Structured pruning learns compact and accurate models. In Proceedings of ACL, pp. 1513–1528, Dublin, Ireland, May 2022. doi: 10.18653/v1/2022.acl-long.107.

Chaojun Xiao, Yuqi Luo, Wenbin Zhang, Pengle Zhang, Xu Han, Yankai Lin, Zhengyan Zhang, Ruobing Xie, Zhiyuan Liu, Maosong Sun, and Jie Zhou. Variator: Accelerating pre-trained models with plug-and-play compression modules. In Proceedings of EMNLP: Findings, pp. 9947–9959, 2023a.

Chaojun Xiao, Zhengyan Zhang, Xu Han, Chi-Min Chan, Yankai Lin, Zhiyuan Liu, Xiangyang Li, Zhonghua Li, Zhao Cao, and Maosong Sun. Plug-and-play document modules for pre-trained models. In Proceedings of ACL, pp. 15713–15729, 2023b.

Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, Song Han, and Maosong Sun. Infllm: Unveiling the intrinsic capacity of llms for understanding extremely long sequences with training-free memory. arXiv preprint arXiv:2402.04617, 2024.

Guangxuan Xiao, Ji Lin, and Song Han. Offsite-tuning: Transfer learning without full model. CoRR, abs/2302.04870, 2023c.

Ji Xin, Jimmy Lin, and Yaoliang Yu. What part of the neural network does this? understanding lstms by measuring and dissecting neurons. In Proceedings of EMNLP-IJCNLP, pp. 5822–5829, 2019.

Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and Jimmy Lin. Deebert: Dynamic early exiting for accelerating BERT inference. In Proceedings of ACL, pp. 2246–2251, 2020.

Canwen Xu, Yichong Xu, Shuohang Wang, Yang Liu, Chenguang Zhu, and Julian J. McAuley. Small models are valuable plug-ins for large language models. CoRR, abs/2305.08848, 2023a.

Runxin Xu, Fuli Luo, Zhiyuan Zhang, Chuanqi Tan, Baobao Chang, Songfang Huang, and Fei Huang. Raise a child in large language model: Towards effective and generalizable fine-tuning. In Proceedings of EMNLP, pp. 9514–9528, Online and Punta Cana, Dominican Republic, November 2021.

Runxin Xu, Fuli Luo, Baobao Chang, Songfang Huang, and Fei Huang. S4-tuning: A simple cross-lingual sub-network tuning method. In Proceedings of ACL, pp. 530–537, Dublin, Ireland, May 2022.

Yuzhuang Xu, Shuo Wang, Peng Li, Xuebo Liu, Xiaolong Wang, Weidong Liu, and Yang Liu. Pluggable neural machine translation models via memory-augmented adapters. CoRR, abs/2307.06029, 2023b.

Zhenliang Xue, Yixin Song, Zeyu Mi, Le Chen, Yubin Xia, and Haibo Chen. Powerinfer-2: Fast large language model inference on a smartphone. arXiv preprint arXiv:2406.06282, 2024.

Ikuya Yamada, Hiroyuki Shindo, Hideaki Takeda, and Yoshiyasu Takefuji. Joint learning of the embedding of words and entities for named entity disambiguation. In Proceedings of CoNLL, pp. 250–259, 2016.

Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. MM-REACT: prompting chatgpt for multimodal reasoning and action. CoRR, abs/2303.11381, 2023.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In Proceedings of ICLR, 2023a.

Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, and Ningyu Zhang. Editing large language models: Problems, methods, and opportunities. CoRR, abs/2305.13172, 2023b.

Deming Ye, Yankai Lin, Peng Li, Maosong Sun, and Zhiyuan Liu. A simple but effective pluggable entity lookup table for pre-trained language models. In Proceedings of ACL, pp. 523–529, 2022.

Yining Ye, Xin Cong, Yujia Qin, Yankai Lin, Zhiyuan Liu, and Maosong Sun. Large language model as autonomous decision maker. CoRR, abs/2308.12519, 2023.

Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. CoRR, abs/2306.13549, 2023.

Zichun Yu, Chenyan Xiong, Shi Yu, and Zhiyuan Liu. Augmentation-adapted retriever improves generalization of language models as generic plug-in. In Proceedings of ACL, pp. 2421–2436, 2023.

Anthony Zador, Sean Escola, Blake Richards, Bence Ölveczky, Yoshua Bengio, Kwabena Boahen, Matthew Botvinick, Dmitri Chklovskii, Anne Churchland, Claudia Clopath, et al. Toward next-generation artificial intelligence: Catalyzing the neuroai revolution. arXiv preprint arXiv:2210.08340, 2022.

Elad Ben Zaken, Yoav Goldberg, and Shauli Ravfogel. Bitfit: Simple parameter-efficient fine-tuning for transformer-based masked language-models. In Proceedings of ACL, pp. 1–9, 2022.

Yujie Zeng, Wenlong He, Ihor Vasyltsov, Jiali Pang, and Lin Chen. Acceleration of large transformer model training by sensitivity-based layer dropping. In Proceedings of AAAI, pp. 11156–11163, 2023.

Mingshu Zhai, Jiaao He, Zixuan Ma, Zan Zong, Runqing Zhang, and Jidong Zhai. Smartmoe: Efficiently training sparsely-activated models through combining offline and online parallelization. In Proceedings of ATC, pp. 961–975, 2023.

Dinghuai Zhang, Kartik Ahuja, Yilun Xu, Yisen Wang, and Aaron Courville. Can subnetwork structure be the key to out-of-distribution generalization? In Proceedings of ICML, volume 139, pp. 12356–12367, 18–24 Jul 2021.

Fan Zhang, Duyu Tang, Yong Dai, Cong Zhou, Shuangzhi Wu, and Shuming Shi. Skillnet-nlu: A sparsely activated model for general-purpose natural language understanding. arXiv preprint arXiv:2203.03312, 2022a.

Jiajie Zhang, Shulin Cao, Linmei Hu, Ling Feng, Lei Hou, and Juanzi Li. Kb-plugin: A plug-and-play framework for large language models to induce programs over low-resourced knowledge bases. arXiv preprint arXiv:2402.01619, 2024a.

Jinghan Zhang, Shiqi Chen, Junteng Liu, and Junxian He. Composing parameter-efficient modules with arithmetic operations. CoRR, abs/2306.14870, 2023a.

Junting Zhang, Jie Zhang, Shalini Ghosh, Dawei Li, Serafettin Tasci, Larry P. Heck, Heming Zhang, and C.-C. Jay Kuo. Class-incremental learning via deep model consolidation. In Proceedings ofWACV, pp. 1120–1129, 2020.

Minjia Zhang and Yuxiong He. Accelerating training of transformer-based language models with progressive layer dropping. In Proceedings of NeurIPS, volume 33, pp. 14011–14023, 2020.

Xiaofeng Zhang, Yikang Shen, Zeyu Huang, Jie Zhou, Wenge Rong, and Zhang Xiong. Mixture of attention heads: Selecting attention heads per token. In Proceedings of EMNLP, pp. 4150–4162, 2022b.

Zhengyan Zhang, Xu Han, Zhiyuan Liu, Xin Jiang, Maosong Sun, and Qun Liu. ERNIE: enhanced language representation with informative entities. In Proceedings of ACL, pp. 1441–1451, 2019.

Zhengyan Zhang, Yankai Lin, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. Moefication: Transformer feed-forward layers are mixtures of experts. In Proceedings of ACL: Findings, pp. 877–890, 2022c.

Zhengyan Zhang, Zhiyuan Zeng, Yankai Lin, Huadong Wang, Deming Ye, Chaojun Xiao, Xu Han, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. Plug-and-play knowledge injection for pre-trained language models. In Proceedings of ACL, pp. 10641–10658, 2023b.

Zhengyan Zhang, Zhiyuan Zeng, Yankai Lin, Chaojun Xiao, Xiaozhi Wang, Xu Han, Zhiyuan Liu, Ruobing Xie, Maosong Sun, and Jie Zhou. Emergent modularity in pre-trained transformers. In Proceedings of ACL: Findings, pp. 4066–4083, 2023c.

Zhengyan Zhang, Yixin Song, Guanghui Yu, Xu Han, Yankai Lin, Chaojun Xiao, Chenyang Song, Zhiyuan Liu, Zeyu Mi, and Maosong Sun. Relu2 wins: Discovering efficient activation functions for sparse llms. arXiv preprint arXiv:2402.03804, 2024b.

Zhihan Zhang, Wenhao Yu, Chenguang Zhu, and Meng Jiang. A unified encoder-decoder framework with entity memory. In Proceedings of EMNLP, pp. 689–705, 2022d.

Zhong Zhang, Bang Liu, and Junming Shao. Fine-tuning happens in tiny subspaces: Exploring intrinsic task-specific subspaces of pre-trained language models. In Proceedings of ACL, pp. 1701–1713, 2023d.

Chenyang Zhao, Xueying Jia, Vijay Viswanathan, Tongshuang Wu, and Graham Neubig. Self-guide: Better task-specific instruction following via self-synthetic finetuning. arXiv preprint arXiv:2407.12874, 2024a.

Jun Zhao, Zhihao Zhang, Yide Ma, Qi Zhang, Tao Gui, Luhui Gao, and Xuanjing Huang. Unveiling a core linguistic region in large language models, 2023a.

Kang Zhao, Hua Xu, Jiangong Yang, and Kai Gao. Consistent representation learning for continual relation extraction. In Proceedings of ACL: Findings, pp. 3402–3411, 2022.

Mengjie Zhao, Tao Lin, Fei Mi, Martin Jaggi, and Hinrich Schütze. Masking as an efficient alternative to finetuning for pretrained language models. In Proceedings of EMNLP, pp. 2226–2241, 2020.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. CoRR, abs/2303.18223, 2023b.

Zijia Zhao, Longteng Guo, Tongtian Yue, Sihan Chen, Shuai Shao, Xinxin Zhu, Zehuan Yuan, and Jing Liu. Chatbridge: Bridging modalities with large language model as a language catalyst. CoRR, abs/2305.16103, 2023c.

Ziyu Zhao, Leilei Gan, Guoyin Wang, Wangchunshu Zhou, Hongxia Yang, Kun Kuang, and Fei Wu. Loraretriever: Input-aware lora retrieval and composition for mixed tasks in the wild. CoRR, abs/2402.09997, 2024b.

Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M. Dai, Zhifeng Chen, Quoc V. Le, and James Laudon. Mixture-of-experts with expert choice routing. In Proceedings of NeurIPS, 2022a.

Zhe Zhou, Xuechao Wei, Jiejing Zhang, and Guangyu Sun. Pets: A unified framework for parameter-efficient transformers serving. In Proceedings of ATC, pp. 489–504, 2022b.

Chen Zhu, Ankit Singh Rawat, Manzil Zaheer, Srinadh Bhojanapalli, Daliang Li, Felix X. Yu, and Sanjiv Kumar. Modifying memories in transformer models. CoRR, abs/2012.00363, 2020.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. CoRR, abs/2304.10592, 2023.

Konrad Zolna, Serkan Cabi, Yutian Chen, Eric Lau, Claudio Fantacci, Jurgis Pasukonis, Jost Tobias Springenberg, and Sergio Gomez Colmenarejo. GATS: gather-attend-scatter. CoRR, abs/2401.08525, 2024.

Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. Designing effective sparse expert models. CoRR, abs/2202.08906, 2022.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to AI transparency. CoRR, abs/2310.01405, 2023.

Simiao Zuo, Xiaodong Liu, Jian Jiao, Young Jin Kim, Hany Hassan, Ruofei Zhang, Jianfeng Gao, and Tuo Zhao. Taming sparsely activated transformer with stochastic experts. In Proceedings of ICLR, 2022a.

Simiao Zuo, Qingru Zhang, Chen Liang, Pengcheng He, Tuo Zhao, and Weizhu Chen. Moebert: from BERT to mixture-of-experts via importance-guided adaptation. In Proceedings of NAACL, pp. 1610–1623, 2022b.

