Is Diversity All You Need for Scalable Robotic Manipulation?

Modi Shi, Graduate Student Member, IEEE, Li Chen, Graduate Student Member, IEEE, Jin Chen, Yuxiang Lu, Graduate Student Member, IEEE, Chiming Liu, Guanghui Ren, Ping Luo, Senior Member, IEEE, Di Huang, Senior Member, IEEE, Maoqing Yao and Hongyang Li, Senior Member, IEEE

Task Diversity: Beneficial Embodiment Diversity: Non-essential Expert Diversity: Confounding

[Figure 1]

Task Diversity Accelerates Scaling

One-to-Many

Embodiment Transfer ≈ 2.5 × training data

[Figure 2]

[Figure 3]

15%

Predictable Scaling Law

OptimalityGap

Distribution Debiasing

r = 0.99

[Figure 4]

## arXiv:2507.06219v2[cs.RO]4Jun2026

GO-1 GO-1-Pro

(ours)

Pretraining Data

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Comprehensive Evaluation

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

AgiBot G1 Maniskill Robotwin Piper

Fig. 1: We investigate critical aspects of data diversity for robotic manipulation systematically, i.e., task, embodiment, and expert diversity. Through comprehensive evaluation in simulation and the real world, we reveal key insights that challenge conventional assumptions on data scaling. (a) Task diversity benefits policy learning with predictable power-law scaling. (b) Multi-embodiment pre-training data is non-essential for cross-embodiment transfer capabilities-models pre-trained on high-quality single-embodiment data can efficiently adapt to different embodiments and show desirable scaling properties during finetuning, which can serve as a competitive alternative to large-scale multi-embodiment pre-training. (c) Expert diversity confuses robot learning, towards which we devise a distribution debiasing method based on GO-1 [1]; the yielding GO-1-Pro attains superior data efficiency during both pre-training and finetuning, where it achieves substantial performance gains of 15%, equivalent to using 2.5 times the pre-training data. Project page: https://github.com/OpenDriveLab/AgiBot-World.

Abstract—Data scaling has driven remarkable success in foundation models for Natural Language Processing (NLP) and Computer Vision (CV), yet the principles of effective data scaling in robotic manipulation remain insufficiently understood. In this work, we investigate the nuanced role of data diversity in robot learning by examining three critical dimensions—task (what to do), embodiment (which robot to use), and expert (who demonstrates)—challenging the conventional intuition of “more diverse is better”. Throughout extensive experiments on various robot platforms, we reveal that (1) task diversity proves more critical than per-task demonstration quantity, with scene diversity playing a more important role than skill diversity for robustness and generalization under distribution shifts; (2) multi-embodiment

pre-training data is non-essential for cross-embodiment transfer—models trained on high-quality single-embodiment data can efficiently transfer to different platforms, showing desirable scaling property during fine-tuning and its potential of replacing largescale multi-embodiment pre-training; and (3) expert diversity, arising from individual operational preferences and stochastic variations in human demonstrations, can be confounding to policy learning, with action rate multimodality emerging as a key contributing factor. Based on this insight, we propose a distribution debiasing method to mitigate action rate ambiguity, the yielding GO-1-Pro achieves substantial performance gains of 15%, equivalent to using 2.5× pre-training data. Collectively, these findings provide new perspectives and offer practical guidance on how to scale robotic manipulation datasets effectively. The code will be released.

This work is in part supported by the JC STEM Lab of Autonomous Intelligent Systems funded by The Hong Kong Jockey Club Charities Trust. (Correspondence author: Hongyang Li.)

Index Terms—Robotic Manipulation, Data Diversity, Scaling Law, Cross-Embodiment, Distribution Debias.

Modi Shi and Jin Chen are with Shanghai Innovation Institute, Shanghai, China (email: modishi@buaa.edu.cn).

Li Chen, Ping Luo, and Hongyang Li are with The University of Hong Kong, Hong Kong SAR, China (email: hongyang@hku.hk).

I. INTRODUCTION

# R

Yuxiang Lu, Chiming Liu, Guanghui Ren, and Maoqing Yao are with

ECENT advances in foundation models across NLP and CV have demonstrated remarkable generalization capabil-

AgiBot, Shanghai, China. Modi Shi and Di Huang are with Beihang University, Beijing, China. Modi Shi, Li Chen, and Jin Chen contribute equally to this project.

ities, such as GPT-4 [2], Gemini [3], and SAM2 [4]. A critical

factor underlying these breakthroughs is systematic data scaling, where training on massive, diverse, while carefully curated datasets yields superior performance and broader applicability. Given that data scaling principles have revolutionized multiple domains, a natural question emerges: can similar data scaling approaches pave the way toward robotic foundation models?

Building on state-of-the-art Vision Language Models (VLMs) [5]–[10] and visual foundation models [4], [11]– [13], the robotics community has developed several large-scale robotic models including RT-2 [14], OpenVLA [15], Pi-0 [16], RDT [17], GO-1 [1], UniVLA [18], and GR00T [19]. Despite representing significant progress, these models are still far from genuine robotic foundation models. Their generalization capabilities remain constrained, struggling with novel objects, unfamiliar environments, new tasks, and different robot embodiments. Even minor variations in object positioning or lighting conditions can significantly compromise performance [20]. This persistent gap between robotic manipulation and advances in NLP and CV domains can be attributed to multiple factors, with the limited quantity and quality of robot datasets being a critical bottleneck. To address this data scarcity, recent efforts have focused on large-scale data collection initiatives such as Bridge Data [21], [22], DROID [23], Open X-Embodiment (OXE) [24], and AgiBot World [1]. However, these approaches primarily advocate a “more is better” philosophy, relying on brute-force collection or simple aggregation, without carefully considering what constitutes effective data for robot learning. This limitation is exemplified by OpenVLA’s finding that removing DROID data actually improves model performance [15]. Such results raise fundamental questions about what makes a good manipulation dataset and how we should strategically scale up the datasets to maximize learning outcomes.

Some preliminary studies have attempted to address these questions by scaling data for direct Imitation Learning (IL). For instance, Lin et al. [25] find a power-law relationship between policy performance and object diversity and environment diversity, while ManiBox [26] highlights the benefits of spatial diversity for spatial generalization. These findings naturally lead to the intuition that data diversity is universally beneficial for robotic manipulation. In this study, we systematically explore three underexplored dimensions of data diversity, e.g., task diversity, embodiment diversity, and expert diversity, and investigate data diversity in both policy pre-training and finetuning stages, to provide comprehensive insights into effective data scaling strategies. In the end, our investigation suggests a more complex picture. We find that the impact of diversity varies significantly across different dimensions: while some aspects of diversity are indeed critical and beneficial, others may be less important, or even confounding.

First, we investigate how task diversity in large-scale pretraining affects downstream performance. Given that current robotic foundation models struggle to adapt to new tasks or skills, a fundamental challenge emerges regarding how models should acquire transferable knowledge. Two potential approaches exist: broad exposure to diverse tasks for general knowledge acquisition, or intensive training on focused skill sets and scenarios for specialized expertise development. To address this challenge, we construct three pre-training datasets

with identical sample sizes but different task compositions: one with high task diversity (episode-based sampling), one focused on target-relevant skills (task-based sampling), and one with controlled visual scene diversity (scenario-based sampling). Our results demonstrate that task diversity outweighs the number of demonstrations per task. More importantly, we find that scene diversity plays a more critical role than skill diversity in enhancing robustness and generalization under distribution shifts. Building on this insight, we further investigate whether model performance continues to improve with increasing training samples when task diversity is sufficiently maintained, examining the power-law scaling relationship under this condition.

Second, we explore embodiment diversity and its implications for cross-embodiment generalization. A truly foundational robotic model should be capable of adapting to different robot embodiments. While the robotics community conventionally considers that achieving this capability requires diverse embodiments in the training data, cross-embodiment training is inherently complex due to morphological and state space heterogeneity across robots [27]–[30]. However, intuitively, the end-effector action spaces of robots with different configurations are fundamentally similar—robots with different morphologies can produce comparable behaviors when their end-effectors follow the same trajectory in world coordinates, suggesting that action space transformation across embodiments may be feasible. This observation leads to a critical hypothesis: models pre-trained on a single embodiment may easily transfer their learned knowledge to new robot configurations, thereby circumventing the difficulties of cross-embodiment training. To verify this hypothesis, we evaluate models trained solely on AgiBot G1 across diverse simulated and real-world platforms. Remarkably, we find that the model adapts well, even showing a more desirable scaling property during fine-tuning than models pre-trained on the OXE dataset [24], which includes the test embodiment and thus has a smaller embodiment gap. Importantly, we are not attempting to prove that singleembodiment pre-training is universally superior. Rather, our findings provide existence proof: high-quality and consistent single-embodiment pre-training alone can achieve crossembodiment adaptation capabilities comparable to, or even better than, models pre-trained on the massive OXE dataset. This challenges the conventional wisdom that pre-training must include the target embodiment to ensure generalization, and leads to our core conclusion: multi-embodiment pre-training is non-essential or not strictly required, prioritizing high-quality, consistent data, even from a single robot, is a practical pathway that circumvents the inherent challenges of cross-embodiment training.

Third, we study expert diversity, an often overlooked aspect in robot learning. Expert diversity refers to the distributional variations in collected demonstrations arising from different teleoperators’ habits, skill levels, and inherent randomness. Unlike standardized NLP and CV datasets collected from the internet, robotic datasets are composed of continuous robot motion that is highly sensitive to teleoperator behaviors [31]. This sensitivity results in demonstrations that, while achieving the same task, exhibit distinct distributional characteristics.

[Figure 21]

- As shown in Figure 2, expert diversity manifests as both spatial multimodality [32] in trajectory paths and action rate multimodality in execution speeds. Crucially, these two types of multimodality have fundamentally different implications for learning: spatial variations represent meaningful task strategies that should be retained, whereas action rate variations introduce undesirable noise that complicates training. To address this, we introduce an action rate model that performs distribution debiasing, specifically eliminating action rate multimodality while preserving spatial multimodality. Our experiments show that this design significantly improves model performance. These results shed light on the fundamental distinction between robotic action data and image/text data, revealing that current imitation learning approaches may be limited by overlooked data characteristics rather than insufficient model capacity or dataset scale.

Spatial Multimodality Action Rate Multimodality

[Figure 22]

[Figure 23]

Time

Fig. 2: Illustration of the multimodal expert behavior in task Push-T [32]. The robot (blue circle) needs to move the gray T to the green target area. Expert demonstrations exhibit multimodality in both spatial and action rate dimensions: (a) Spatial multimodality arises from different trajectory choices, where the robot can approach T from either left or right sides, resulting in distinct spatial paths; (b) Action rate multimodality occurs when robots execute similar trajectory at different speeds, generating completely different demonstration profiles over time. Both spatial and action rate multimodal characteristics require models to learn these distributional properties in current action chunk-based imitation learning.

We hope this consolidation report will shed new light on scalable robotic manipulation and offer practical guidelines for the research community. In summary, the contributions of this article are as follows:

- 1) We demonstrate task diversity’s benefits for robot learning and validate power-law scaling relationships. Crucially, scene diversity outweighs skill diversity for robustness and generalization.
- 2) We show that multi-embodiment data is non-essential for cross-embodiment transfer, as models pre-trained on single-embodiment data with high quality can efficiently adapt to different embodiments with desirable scaling property during fine-tuning, serving as effective alternatives to multi-embodiment pre-trained models.
- 3) We discover that expert diversity could be confounding to the imitation learning process and demonstrate that targeted elimination of action rate multimodality can significantly improve model performance.

for modeling multimodal distributions. Diffusion Policy [32] incorporates a diffusion module to adapt to trajectory multimodality. RDT [17] employs DiT [53] for improved pretraining on heterogeneous multi-robot datasets [24] and dualarm trajectories. In our work, we conduct experiments based on GO-1 [1] and RDT [17] to ensure generalization potential across tasks and embodiments.

- B. Large-scale Manipulation Dataset

Robotic manipulation is experiencing a significant transformation toward scaling up data [21]–[23], [41], [54]–[57], seeking to enable models to develop general manipulation capabilities. The Open X-Embodiment dataset [24] exemplifies this effort by consolidating multiple datasets across 22 different embodiments and various camera configurations, reaching a significant scale of 2.4 million trajectories. DROID [23] emphasizes increasing data diversity by covering various tasks, objects, scenes, camera viewpoints, and interaction locations, collecting data across 564 scenes in 52 real-world buildings. AgiBot World [1] has compiled over 1 million high-quality bimanual trajectories through a unified embodiment and camera configuration, skilled teleoperators and rigorous verification protocols. Additionally, other works explore learning robotic manipulation knowledge from human manipulation videos [43], [44], leveraging the abundance of such videos to address the scarcity of robotic data. In this work, we conduct a more comprehensive analysis of manipulation data diversity, aiming to provide deeper insights into constructing large-scale manipulation datasets of high quality.

- C. Crucial Dimensions Regarding Data Distribution

II. RELATED WORK A. Scalable Manipulation Policy

Recently, the success of foundation models in CV [5], [11]– [13] and NLP [33]–[36] has motivated research into establishing manipulation foundation models [15], [16], [19], [37]–[40] through IL, aiming to leverage large-scale pre-training data to enable models to absorb rich prior knowledge. Early studies like RT-1 [41] and Octo [42] employ transformer-based policies to learn generalizable manipulation knowledge from diverse heterogeneous data. Works like RT-2 [14] and OpenVLA [15] utilize advanced VLMs to process camera inputs and human instructions, which have been extensively trained on web-scale image-text pairs and possess substantial world knowledge. However, these methods directly integrate VLMs to generate lowlevel robotic actions, which may not be optimal for achieving generalization across different embodiments and skills. Some works utilize large-scale human manipulation videos [43], [44] to pre-train models that learn inverse dynamics by predicting latent action representations [45]–[47], or forward dynamics by predicting future changes [48]–[52], thus enhancing crossembodiment and cross-skill generalization. Other works focus on designing policy architectures with stronger capabilities

Scaling robot datasets requires strategic approaches beyond simply expanding data volume. Recent research has demonstrated how data diversity enhances model generalization [25],

[58]–[60]. Lin et al. [25] show that incorporating datasets with diverse environments and objects significantly improves model generalization, establishing a power-law relationship between policy performance and data diversity. Manibox [26] highlights spatial diversity benefits, demonstrating that varied spatial layouts improve spatial generalization in manipulation tasks. Saxena et al. [61] identify camera viewpoints and spatial arrangements as crucial dimensions for collection diversity and retrieval alignment. Hejna et al. [62] propose automatic curation of large-scale robotics datasets using group distributionally robust optimization, enabling efficient utilization of heterogeneous data for imitation learning. Similar findings exist in autonomous driving. Zheng et al. [63] collect data from various driving scenarios and behaviors, discovering that driving model performance exhibits a power-law relationship with scenario and behavior diversity. Our research challenges this conventional view, suggesting that not all diversity forms are equally beneficial. While diversity in environments and objects proves crucial, other diversity types may be less important or even confounding.

In the following sections, we address three critical aspects of diversity in detail: task diversity (Section III), embodiment diversity (Section IV), and expert diversity (Section V).

III. TASK DIVERSITY

Recent studies have explored scaling properties, such as [25] on environment diversity, i.e., variations in lighting conditions, distractor objects, background changes, and RT-1 [41] on skill diversity, which corresponds to atomic skills or verbs in task instructions, excluding the targeted objects. Meanwhile, these analyses were limited to single-task scenarios or significantly smaller scales (e.g., RT-1 utilized only six skills and ∼10% of our data volume). In contrast, we investigate how different types of task diversity in large-scale pre-training datasets affect downstream task performance, specifically examining the tradeoff between skill diversity (task coverage) and scenario diversity (environmental variations). We decompose task diversity into these two fundamental components and analyze their relative importance for in-domain performance versus generalization under distribution shifts.

- A. Experiment Design

Our experiments employ GO-1 [1] as the policy architecture, which excels at extracting task-agnostic latent actions and generalizing across diverse tasks. We leverage AgiBot World [1], a large-scale robotic learning dataset containing over 1 million trajectories across more than 100 real-world scenarios. This dataset offers unique advantages, as all data is collected using a single robot platform, AgiBot G1, which eliminates cross-embodiment variables while ensuring high data quality through standardized collection protocols. Our training follows a two-phase process: first pre-training on the largescale manipulation dataset, followed by fine-tuning on target evaluation tasks. To isolate the effects of skill diversity and scene diversity, we construct three pre-training datasets with identical data quantity (10%) but different sampling strategies:

scenario-based sampling (limited scenario diversity), task-based sampling (limited skill diversity), and episode-based sampling (full spectrum of task variety). We then fine-tune all models on identical evaluation task data to compare how different diversity compositions affect downstream performance.

Our evaluation encompasses four challenging tasks: Wipe Table (contact-rich cleaning), Fold Shorts (deformable object manipulation), Pour Water (fine-grained pouring), and Make Sandwich (long-horizon assembly). Each task is evaluated across three scenarios: an in-domain scenario, an objectenvironment generalization scenario, and a visual distraction scenario. We conduct ten trials per scenario with position perturbations under consistent indoor lighting conditions, ensuring identical evaluation settings across all models. For evaluation, we use normalized scores to record the performance of each trial. We establish specific scoring criteria for each action within every task, with action scores categorized into three levels: 1, 0.5, and 0. The evaluation score for each trial corresponds to the average of all action scores, where a score of 1 indicates perfect completion of all actions, and fractional scores represent partial success. Detailed scoring criteria can be found in Appendix.

B. Task Diversity for Robotic Manipulation Pre-training

When applying a model to a specific downstream task out of the pre-training domain, a fundamental consideration emerges regarding whether to construct a pre-training dataset with the richest possible diversity, or to utilize fewer tasks but with potentially higher relevance to the target downstream task. To explore this trade-off, we design three pre-training datasets with distinct diversity characteristics while controlling for other factors, allowing us to decompose task diversity into two key components: skill diversity and scene diversity.

We leverage the Agibot-World Beta dataset, one of the most comprehensive robotic datasets, as our source for constructing these contrasting pre-training datasets. This allows us to maintain consistency in other aspects of the data, such as the included robot embodiments, while varying only the diversity dimensions. We employ three sampling strategies to create datasets with different diversity but identical sizes:

- • Scenario-based sampling (10% Scenario): Randomly samples 10% of scenarios from the dataset, where each scenario represents a specific environmental setup (e.g., kitchen counter, dining table). This strategy provides limited diversity in scenes.
- • Task-based sampling (10% Task): Manually selects 10% of tasks that are most relevant to our target downstream tasks. This results in a dataset with high scenario coverage but limited skill diversity.
- • Episode-based sampling (10% Episode): Randomly samples 10% of episodes from each task in the original dataset, preserving the full spectrum of task variety. This strategy captures both skill diversity across different tasks and scene diversity within each task, as episodes naturally include varied object configurations, lighting conditions, and spatial arrangements.

Task-Based Sampling (10% tasks)

Episode-Based Sampling (10% episodes)

Others

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Pick

Others

Hang

Pick

[Figure 28]

Release

[Figure 29]

Grasp

[Figure 30]

[Figure 31]

Handover

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Fold

Grasp

Move

Push

Fold

Pour

Pour

Place Handover

Push

Carry

Place

- Fig. 3: Distribution of atomic skills in two pre-training datasets. Task-based sampling (10% tasks) shows lower skill diversity but concentrates on the most commonly used skills, while episode-based sampling (10% episodes) demonstrates a more balanced distribution.

Our criterion for selecting relevant tasks is based on the inclusion of atomic skills required for task completion. Our evaluation tasks encompass five common atomic skills: pick, place, grasp, pour, and fold. In Figure 3, we present the distribution of atomic skills in both pre-training datasets, the episode-sampled dataset exhibits significantly greater task and skill diversity, yet consequently contains fewer episodes (59.2% vs. 71.1%) corresponding to the specific atomic skills needed for the target evaluation tasks.

The experimental results in Figure 4 demonstrate that data diversity significantly impacts downstream performance.

- At 10% data scale, scenario-based sampling achieves 0.33 average performance, task-based sampling improves to 0.36, while episode-based sampling reaches 0.46. The episodebased sampling approach achieves an average performance improvement of 0.1 compared to task-based sampling, with the most significant gains observed in tasks requiring higher semantic and spatial understanding, such as Make Sandwich and Pour Water. This finding corroborates the conclusions in [25] that, given fixed data quantity, enhancing diversity provides greater benefits than simply increasing data volume.

To understand how skill diversity and scene diversity affect

model capabilities under distribution shifts, we evaluate all pretraining strategies across three conditions: in-domain evaluation, visual distraction robustness, and object-environment generalization. As shown in Figure 5, in-domain performance shows episode-based sampling outperforming task-based and scenariobased by comparable margins. However, under distribution shifts, scene diversity demonstrates greater impact. For visual distraction, episode-based exceeds scenario-based by 0.13 compared to only 0.09 over task-based. In object-environment generalization, this pattern intensifies: episode-based surpasses scenario-based by 0.13 versus 0.08 over task-based.

These results reveal that scene diversity is more critical than skill diversity for robustness and generalization. While both diversity types provide similar in-domain benefits, scene diversity advantages consistently exceed skill diversity advantages under distribution shifts with statistically significant differences (p < 0.05). The continued gains from data scaling highlight its capacity to enhance robustness and generalization in real-world deployment.

C. Pre-training Data Scaling Law

The Agibot World [1] encompasses 217 common daily tasks and 87 frequently used skills. Building upon the conclusion that task diversity benefits robotic learning, we further investigate the relationship between data quantity and performance when scaling up pre-training data while maintaining dataset diversity. The left panel in Figure 6 demonstrates consistent performance improvements across different pre-training data scales, with GO-1 average scores increasing from 0.28 (No pre-training) to 0.47 (100K demonstrations), 0.53 (250K demonstrations), and reaching 0.58 (1M demonstrations).

To further explore this relationship, we fit the data using a power law curve Y = β · Xα, where Y represents the optimality gap, defined as the deviation from the maximum score (i.e., 1 − Normalized Score), and X represents the number of demonstrations [25]. Since the no pre-train case corresponds to zero pre-training data, which cannot be fitted in the scaling law curve, we use the fine-tuning data quantity to replace the number of demonstrations for fitting purposes. The experimental results in Figure 6 reveal a clear power-law relationship between model performance and pre-training data, with a Pearson correlation coefficient reaching −0.99. This finding suggests that, under the condition of adequate task diversity, robotic learning can achieve systematic performance gains through increased data scale, providing a clear path for developing more capable robotic systems through data scaling.

IV. EMBODIMENT DIVERSITY

Cross-embodiment learning faces significant challenges due to morphological and state space heterogeneity across robot platforms. However, it remains unclear whether pre-training datasets must include multi-embodiment data to achieve effective cross-embodiment transfer. In this section, we investigate whether single-embodiment pre-training—thereby avoiding cross-embodiment training complexities—can still yield models with cross-embodiment capabilities.

A. Experiment Design

To address the challenges of cross-embodiment training, we explore whether single-embodiment pre-training can still enable effective cross-embodiment transfer by utilizing the large-scale single-embodiment dataset Agibot World (1M trajectories from AgiBot G1) for pre-training and systematically evaluating the resulting model’s cross-embodiment generalization. As a reference point, we include RDT [17] pre-trained on OXE [24], a widely recognized architecture specifically designed for multiembodiment learning with high-quality, community-validated open-source checkpoints. Note that OXE exhibits extensive atomic skills compared to Agibot World, rich in task diversity mentioned in Section III. This ensures the multi-embodiment baseline represents a strong, state-of-the-art standard, allowing performance differences to be strictly attributed to data diversity rather than implementation quality.

We evaluate cross-embodiment transfer across 3 distinct benchmarks, ManiSkill (Franka arm), RoboTwin (Arx arm), and real-world Agilex (Cobot Magic), which all exhibit substantial differences from the AgiBot G1 used in pre-training. Regarding

[Figure 44]

Pretrain Data

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Fine-grained Deformable Long-horizon Contact-rich

- Fig. 4: Real-robot evaluation of GO-1 [1] on four challenging tasks subsequent to pre-training on different datasets. Error bars show standard errors. The tasks assess fine-grained manipulation, deformable object handling, long-horizon planning, and contact-rich interactions respectively. Results show that episode-based sampling (10% Episode) outperforms task-based sampling (10% Task) by 0.1 in average score with the same data amount, and performance improves consistently with increased pre-training data while ensuring sufficient task diversity.

In-Domain Visual Distraction Object-Environment Generalization

0.0

0.2

0.4

0.6

0.8

No Pretrain

| |
|---|

10% Scenario

| |
|---|

10% Task

| |
|---|

10% Episode

| |
|---|

25% Episode

| |
|---|

Full Data

0.36

0.23 0.26

0.39

0.31 0.29

0.40

0.35 0.34

0.51

0.44 0.42

0.60

0.51

0.47

0.62

0.59

0.55

- Fig. 5: Generalization performance across different pretraining data. Error bars show standard errors. The model is evaluated under three generalization scenarios: In-Domain, Visual Distraction, and Object-Environment Generalization. Episode-based sampling (10% Episode) consistently outperforms task-based sampling (10% Task) and scenario-based sampling (10% Scenario). Performance improves with increased pretraining data, demonstrating the importance of both data scale and diversity for robust generalization.

[Figure 50]

Fig. 6: Performance scales with pre-training data size while maintaining adequate task diversity, following a predictable power-law relationship. Left: GO-1 performance scales with pre-training data size. Right: Power-law relationship between pre-training data size and model performance. The dashed line represents a power-law fit with equation y = 1.24x−0.08 and correlation coefficient r = −0.99, indicating a strong adherence to power-law scaling with pre-training data volume.

compared to real-world Agibot World data.

action spaces: Agibot G1 uses dual-arm 12-DOF delta endeffector + 2-DOF grippers; ManiSkill’s Franka uses single-arm 7-DOF absolute joint + 1-DOF gripper; RoboTwin’s Arx and Agilex’s Cobot Magic both use dual-arm 12-DOF absolute joint + 2-DOF grippers, with Cobot Magic adding 2-DOF mobile base control. For camera configurations: Agibot G1 employs 1 head-view + 2 wrist-view cameras; ManiSkill uses only 1 third-person view; RoboTwin and Agilex use 2 wrist-view + 1 head-view cameras. While RoboTwin and Agilex share the same camera configuration type as Agibot G1, camera mounting positions, viewing angles, and fields of view differ significantly due to different robot morphologies and workspace setups. Moreover, simulated visual rendering in ManiSkill [64] and RoboTwin [65] introduces a substantial sim-to-real gap

Importantly, all evaluation settings (absolute joint control, camera configurations) are already present in OXE’s training data (Franka robots, dual-arm systems, diverse cameras), giving RDT-OXE prior exposure to these exact embodiments. In contrast, RDT-AWB must transfer from a completely different embodiment. This means the comparison is not entirely fair, where, in fact, RDT-OXE has an inherent advantage through embodiment alignment with evaluation benchmarks.

Performance is measured by the average success rates for simulation tasks and the average scores for real-world tasks. ManiSkill includes 5 tasks (PegInsertionSide, PickCube, StackCube, PlugCharger, PushCube), while RoboTwin includes 4 tasks (BlockHammerBeat, BlocksStack, ContainerPlace,

[Figure 51]

Fig. 7: Cross-embodiment adaptation to Franka arm in ManiSkill with varying training data sizes. Left: Performance vs. number of demonstrations per task in fine-tuning data. Right: Power-law relationship between downstream performance and fine-tuning data size.

[Figure 52]

Fig. 8: Cross-embodiment adaptation to Franka arm in ManiSkill with varying training steps. Left: Performance vs. training steps with 1000 fine-tuning episodes per task. Right: Performance vs. training steps with 500 fine-tuning episodes per task.

DualBottlesPick). Each simulation task involves 25 rollouts across 10 random seeds. Real-world evaluation covers 5 tasks: Package Product, Fold Shorts, Clean Trash, Industrial Sorting, and Push Chairs.

- B. One-to-Many Embodiment Transfer Evaluation

We fine-tune two pre-trained models on ManiSkill: RDTOXE (pre-trained on OXE) and RDT-AWB (pre-trained on Agibot World beta). OXE includes the Franka robot embodiment and ManiSkill data, while Agibot World uses a completely different robot embodiment. Conventional wisdom suggests that RDT-AWB should underperform RDT-OXE or require substantially more fine-tuning due to the cross-embodiment gap. However, our results in Figure 8 demonstrate that singleembodiment pre-training can also achieve effective crossembodiment capabilities. While RDT-OXE converges faster initially and slightly outperforms RDT-AWB in early stages, RDT-AWB achieves effective cross-embodiment adaptation and surpasses RDT-OXE without requiring extensive fine-tuning data or training steps.

As shown in Figure 7, with 125 samples per task, RDTOXE performs slightly better. At 250 samples, RDT-AWB matches RDT-OXE. With more data, RDT-AWB surpasses RDT-OXE1, with the gap increasing proportionally, exhibiting a power-law relationship. Similarly, Figure 8 shows RDT-OXE performing better with fewer steps (around 10,000), but RDTAWB surpasses RDT-OXE as training steps increase. These results provide compelling evidence that single-embodiment pre-training can develop robust cross-embodiment transfer capabilities while circumventing the complexities inherent in multi-embodiment training.

To more comprehensively validate our conclusions, we further compare RDT-OXE and RDT-AWB in both the RoboTwin [65] simulation environment (using Arx) and the realworld Agilex environment (using Cobot Magic), testing crossembodiment adaptation across multiple platforms. Following the same experimental protocol as in Figure 7, we conduct

1Liu et al. [17] report RDT-OXE’s fine-tuning performance on ManiSkill as 53.6%. Our evaluation results differ, possibly due to varying training configurations or inconsistent inference random seeds. Importantly, our RDTOXE and RDT-AWB evaluations use identical training and inference setups.

experiments in RoboTwin to compare model performance under varying fine-tuning data sizes and analyze the power-law relationship between performance and data size, as illustrated in Figure 9. The results demonstrate that RDT-AWB achieves performance comparable to that of RDT-OXE with minimal fine-tuning data, successfully adapting to Arx. Additionally, we compare the fine-tuning performance of RDT-OXE and RDTAWB in the real-world Agilex environment using identical data sizes (100 demonstrations per task), as presented in Table I. RDT-AWB achieves superior performance compared to RDTOXE on 4 of 5 tasks, indicating effective adaptation to the real-world Cobot Magic with limited fine-tuning requirements.

Collectively, these results across simulated and real-world environments substantiate our core argument: multi-embodiment pre-training is non-essential, not mandatory, for effective cross-embodiment transfer. While RDT-OXE benefits from prior exposure to the evaluation embodiments, RDT-AWB’s superior performance demonstrates that the target embodiment need not be present during pre-training to achieve strong generalization. We do not assert that single-embodiment pretraining is universally superior; rather, these findings highlight that prioritizing high-quality data, even from a single source, offers a viable and efficient alternative to the complexities of scaling multi-embodiment datasets. This challenges the prevailing assumption that embodiment diversity is a prerequisite for generalizable policy learning.

V. EXPERT DIVERSITY

During the data collection process, different human demonstrators exhibit distinct collection habits and inherent randomness in their execution, leading to diverse and complex data distributions with varying trajectory patterns. While some variations in the data distribution represent meaningful multimodal spatial distributions that capture legitimate alternative approaches [32], others constitute distribution bias that merely increases training difficulty without providing valuable information. In this section, we propose a distribution debiasing method to eliminate bias in the action rate dimension (i.e., speed of the movement), thus enhancing learning efficiency and overall model performance. The experiment setting is the same as Section III-A.

- TABLE I: Performance in the real world Agilex environment. The performance of RDT-OXE and RDT-AWB is compared to assess RDT-AWB’s ability to transfer to the downstream Cobot Magic robot.

Model Package Product Fold Shorts Clean Trash Industrial Sorting Push Chairs Average Score

RDT-OXE 0.40±0.05 0.65±0.04 0.33±0.02 0.23±0.03 0.27±0.03 0.38±0.03 RDT-AWB 0.57±0.06 0.48±0.05 0.47±0.05 0.27±0.02 0.35±0.03 0.43±0.04

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

##### Stage1 Stage2

Action Chunk Biased Action

Action Chunk

Debias

[Figure 57]

[Figure 58]

[Figure 59]

Action Rate

Unbiased Action Rate

Unbiased Action Predict

MSE Loss

Action Loss

ARM Policy

🔥 ARM ❄

[Figure 60]

[Figure 61]

[Figure 62]

🔥

Fig. 11: Two-stage distribution debiasing framework using ARM. Stage 1: ARM is trained to predict action rate from action chunks using MSE loss, learning the expected action rate for each input from action rate-biased training data. Stage 2: During policy training, ARM first predicts the unbiased action rate for each training sample, which is then used to transform the original actions into unbiased actions. The policy is subsequently trained using these unbiased actions as supervision targets, effectively simplifying the distribution complexity.

- Fig. 9: How the model crosses the embodiment gap to Arx in RoboTwin as data sizes increase. Left: Performance vs. number of demonstrations per task in fine-tuning data. Right: Power-law relationship between downstream performance and fine-tuning data size.

|| | |[Figure 63]|
|---|---|---|
| | | |
| | | |
| | | |
| | | |
<br><br>Action Chunk<br><br>A<br><br>C<br><br>𝑡 𝑡<br><br>Debias Demo<br><br>B<br><br><br>D|
|---|

[Figure 64]

[Figure 65]

|| | |[Figure 66]<br><br>|
|---|---|---|
| | | |
| | | |
| | | |
<br><br>A<br>B<br><br><br>D<br><br>𝑡 𝑡<br><br>Demo 1 Demo 2 Action Chunk|
|---|

Distribution Debiasing

- Fig. 10: Illustration of distribution debiasing. Top: Two demonstrations (Demo 1 & 2) follow the same trajectory (A→D) but have different action rates, resulting in distinct action chunks within the same time window (A→B vs A→D). Bottom: After action rate-based distribution debiasing, both demonstrations are normalized to similar action chunks (A→C), reducing action rate ambiguity and facilitating model learning.

spatial trajectories from point A to point D, their varying execution speeds result in different action chunk representations: [A→B] versus [A→B→C→D], causing spatially equivalent motions to be treated as distinct training samples.

To address action rate bias, we initially consider two straightforward approaches. The first method normalizes all demonstrations to uniform action rate, ensuring consistent spatial distance per action chunk. However, this approach fails to capture task-specific action rate requirements: finegrained tasks like plug insertion require precise alignment phases, while pouring tasks necessitate deliberate pauses during execution. The second approach rescales all demonstrations of a task to identical temporal duration [66], but this episode-level temporal normalization cannot eliminate action rate distribution bias due to heterogeneous speed patterns across trajectory segments within episodes and the inherent requirement for varying episode lengths under different initial conditions.

Action rate bias destabilizes training: identical inputs are mapped to action chunks with inconsistent action rate distributions across demonstrations. To address this issue, we propose an Action Rate Model (ARM) that predicts the expected robot action rate conditioned on observations ot. We train the ARM using MSE loss:

- A. Distribution Debiasing Formally, we model a demonstration trajectory as a temporal

sequence comprising states {st}, observations {ot}, and actions {αt}. At any given timestep t, the policy π receives the current observation ot and predicts a sequence of future actions αt:t+T−1. We refer to this fixed-length sequence of size T as an action chunk.

t,at:t+T)∼D ∥ARM(ot) − v(at:t+T)∥22 , (1)

LARM = E(o

As shown in Figure 10, we demonstrate the action rate multimodal distribution through an example of a sandwich making task. While two expert demonstrations follow identical

where v(at:t+T) denotes the action rate metric extracted from action sequence at:t+T, and D represents the demonstration dataset. This formulation enables the ARM to learn the expected

action rate profile for demonstrations with similar observations. Specifically, we define v(at:t+T) based on the end-effector

relative displacement representation. Let aeeft:t+T ∈ RT×D denote the end-effector actions, where T represents the action

chunk size and D corresponds to the degrees of freedom. We initially normalize each dimension of aeef to the range [−1,1], subsequently defining the action rate metric as:

v(at:t+T) = ∥aeeft:t+T∥1, (2) where ∥ · ∥1 denotes the L1 norm.

Figure 11 illustrates the complete process of employing our ARM for distribution debiasing. During policy training via imitation learning, for each training sample (ot,at:t+T), we determine the optimal chunk length L by:

|ARM(ot) − v(at:t+L)|. (3)

L = arg min

L

To ensure training stability and mitigate potential interference from ARM prediction errors, we constrain the search range of L to lie within 0.5T and 1.5T. Subsequently, we employ interpolation to transform at:t+L into a˜t:t+T with the desired chunk size T. This temporal rescaling ensures that all training samples with similar observations o exhibit consistent action rates, thereby achieving action rate distribution debiasing.

Our ARM employs a simple yet effective architecture, consisting of a SigLIP [67] visual encoder followed by an MLP head. The ARM processes three input images through SigLIP to extract visual features, which are subsequently mapped to a scalar action rate value via the MLP. We normalize the output action rate to [0,1] using min-max scaling to enhance training stability. Critically, we freeze the SigLIP encoder during training to prevent the ARM from overfitting to finegrained visual details in training samples, thereby ensuring it learn to predict the average action rate for similar observations rather than memorizing specific visual patterns.

Importantly, implementing action rate distribution debiasing imposes specific requirements on the underlying robot controller. Direct interpolation of action rates necessitates a high-precision controller to ensure that the modified dynamics are tracked accurately, as highlighted in concurrent works focusing on demonstration acceleration [68], [69]. To mitigate this in scenarios with lower-precision hardware, we can adopt the strategy proposed by Arachchige et al. [69]: training the policy to predict reached poses (actual robot states) rather than commanded poses, effectively decoupling policy learning from the imperfections of the data collection controller, allowing a high-fidelity tracking controller to execute the predicted trajectory during deployment.

- B. Distribution Debiasing in the Pre-training Phase

We first investigate how distribution debiasing influences model performance across different training stages using 10% episode-based sampling from the AgiBot World Beta dataset.

As presented in Table II, applying distribution debiasing exclusively during pre-training yields a 6.5% average improvement (from 0.46 to 0.49), with particularly notable gains in Pour Water (+35%). However, this configuration introduces a distribution mismatch: the debiased pre-trained representations

[Figure 67]

- Fig. 12: GO-1-Pro consistently outperforms GO-1 on both the Wipe Table and Make Sandwich tasks. GO-1-Pro achieves comparable results using only 50% of the training data that GO-1 uses, demonstrating superior data efficiency.

[Figure 68]

- Fig. 13: Performance scaling with fine-tuning data usage for GO-1 and GO-1-Pro on Wipe Table and Make Sandwich tasks. Both models follow power law relationships (fitted equations shown), with GO-1-Pro demonstrating faster convergence rates

must adapt to biased fine-tuning data, which constrains potential performance gains and may introduce training instability.

More substantial improvements are observed when distribution debiasing is consistently applied across both pretraining and fine-tuning stages. This unified debiasing strategy achieves a 15% overall improvement (from 0.46 to 0.53), with Pour Water showing the most significant enhancement (+60% from 0.20 to 0.32). Notably, this performance gain equals that achieved by scaling the pre-training dataset by 2.5× (as demonstrated in Figure 4), underscoring the data efficiency benefits of our debiasing method.

The task-specific improvements reveal distinct patterns: manipulation-heavy tasks such as Pour Water and Fold Shorts benefit more from consistent debiasing (+60% and +23% respectively), while navigation-oriented tasks show more modest but meaningful gains. This suggests that distribution debiasing is particularly effective for tasks requiring precise action sequences and complex manipulation strategies, where biased demonstrations can impede learning efficiency.

C. Distribution Debiasing in the Fine-tuning Phase

Due to the substantial computational cost of pre-training, in this section we explore the effectiveness of applying distribution debiasing exclusively during the fine-tuning stage. For the Wipe Table and Make Sandwich tasks, we evaluate both tasks using

- TABLE II: Performance evaluation using 10% episode-based sampling from AgiBot World Beta dataset for pre-training. Distribution debiasing applied during the pre-training phase demonstrates consistent performance improvements. Additional debiasing during fine-tuning further enhances model capabilities across all evaluated tasks. The improvements from distribution debiasing show statistically significant differences (p < 0.05).

Training Setting Task Completion Score Average Pre-training Data Fine-tuning Data Pour Water Fold Shorts Make Sandwich Wipe Table Average

Biased Biased 0.20±0.02 0.30±0.03 0.67±0.04 0.66±0.04 0.46±0.04 Debiased Biased 0.27±0.02 0.30±0.02 0.71±0.03 0.68±0.04 0.49±0.03

Biased Debiased 0.26±0.02 0.33±0.03 0.73±0.02 0.69±0.04 0.50±0.03 Debiased Debiased 0.32±0.03 0.37±0.04 0.73±0.04 0.70±0.03 0.53±0.03

the GO-1 model pre-trained on the full AgiBot World Beta dataset under different fine-tuning data scales. We refer to the model fine-tuned on distribution-debiased data as GO-1-Pro. The experimental results presented in Figure 12 demonstrate that GO-1-Pro consistently outperforms GO-1 across both tasks and all data scales, achieving an average score of 0.93 on Wipe Table compared to GO-1’s 0.83, and reaching 0.79 on Make Sandwich while GO-1 plateaus at 0.7.

Notably, GO-1-Pro exhibits exceptional data efficiency—it achieves comparable or superior performance using only half the training data required by GO-1. Specifically, GO-1-Pro with 60 demonstrations outperforms GO-1 with 120 demonstrations on both tasks, effectively doubling data utilization efficiency. These results underscore the critical importance of addressing data distribution biases in robotic learning.

The benefits of our distribution debiasing approach become particularly pronounced in low-data regimes, where GO-1-Pro improves performance from 0.35 to 0.52 for Make Sandwich and from 0.38 to 0.53 for Wipe Table with only 15 demonstrations. Under data-scarce conditions, the multimodal distribution across action rate and spatial dimensions creates substantial interference in the model’s learning process, impeding its ability to effectively capture essential spatial distribution patterns. By disentangling these confounding factors, our distribution debiasing method enables the model to focus on learning core spatial relationships despite limited data availability, thereby facilitating more efficient and robust policy learning.

To investigate how distribution debiasing methods affect model performance across different fine-tuning data scales, we fit power-law curves to analyze the impact of fine-tuning data scale on final model performance, with results presented in Figure 13. Both GO-1 and GO-1-Pro exhibit power-law scaling behavior across the two tasks, but with notably different characteristics. For the Wipe Table task, GO-1-Pro demonstrates significantly faster convergence with an exponent of -1.01 compared to GO-1’s -0.67, indicating that GO-1Pro achieves near-optimal performance more rapidly as data volume increases. The steeper negative exponent suggests that our distribution debiasing method more effectively leverages additional training data to reduce the optimality gap. For the Make Sandwich task, while both models exhibit similar exponents (-0.38 for GO-1-Pro vs -0.36 for GO-1), GO-1Pro maintains consistently lower optimality gaps across all data scales. This parallel scaling with a constant performance

offset indicates that the benefits of distribution debiasing persist regardless of data volume.

VI. CONCLUSION AND FUTURE WORK

This work systematically investigates data scaling principles for robotic manipulation, revealing three key insights that challenge conventional wisdom. We find that (1) task diversity proves more critical than per-task demonstration quantity for effective transfer, (2) embodiment diversity is not mandatory for achieving cross-embodiment transfer capabilities, but data quality and consistency is essential, and (3) expert diversity can be confounding due to action rate multimodality, leading us to propose a distribution debiasing method that yields substantial performance gains. These findings challenge the “more diverse is better” paradigm and provide practical guidance for strategically scaling robotic manipulation datasets.

Limitations and future work. While our distribution debiasing method successfully eliminates the action rate multimodality, it cannot be applied to dynamic tasks such as ping-pong where the varying action rates are crucial for robot-environment interaction. Additionally, there remain other aspects of expert diversity that harm policy learning, such as meaningless pauses during data collection and suboptimal behavioral patterns that could cause robots to enter infinite loops. Future work could further explore these areas by developing methods to identify and mitigate confounding expert diversity while preserving beneficial variations.

REFERENCES

- [1] Q. Bu, J. Cai, L. Chen, X. Cui, Y. Ding, S. Feng, S. Gao, X. He, X. Huang, S. Jiang et al., “AgiBot World Colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems,” arXiv preprint arXiv:2503.06669, 2025. 1, 2, 3, 4, 5, 6
- [2] OpenAI, “GPT-4 Technical Report,” arXiv preprint arXiv:2303.08774,

2023. 1

- [3] R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican et al., “Gemini: a family of highly capable multimodal models,” arXiv preprint arXiv:2312.11805, 2023. 1
- [4] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. R¨adle, C. Rolland, L. Gustafson et al., “SAM 2: Segment anything in images and videos,” arXiv preprint arXiv:2408.00714, 2024. 1, 2
- [5] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in ICML, 2021. 2, 3
- [6] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” in NeurIPS, 2023. 2
- [7] J. Li, D. Li, S. Savarese, and S. Hoi, “BLIP-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in ICML, 2023. 2

- [8] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu et al., “InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks,” in CVPR, 2024. 2
- [9] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang et al., “Qwen2.5-VL technical report,” arXiv preprint arXiv:2502.13923, 2025. 2
- [10] S. Karamcheti, S. Nair, A. Balakrishna, P. Liang, T. Kollar, and D. Sadigh, “Prismatic VLMs: Investigating the design space of visually-conditioned language models,” in ICML, 2024. 2
- [11] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in ICLR, 2021. 2, 3
- [12] K. He, X. Chen, S. Xie, Y. Li, P. Doll´ar, and R. Girshick, “Masked autoencoders are scalable vision learners,” in CVPR, 2022. 2, 3
- [13] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “DINOv2: Learning robust visual features without supervision,” TMLR, 2024. 2, 3
- [14] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn et al., “RT-2: Vision-languageaction models transfer web knowledge to robotic control,” in CoRL, 2023. 2, 3
- [15] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi et al., “OpenVLA: An opensource vision-language-action model,” in CoRL, 2024. 2, 3
- [16] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai,

L. Groom, K. Hausman, B. Ichter et al., “π0: A vision-language-action flow model for general robot control,” in RSS, 2025. 2, 3

- [17] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu, “RDT-1B: a diffusion foundation model for bimanual manipulation,” in ICLR, 2025. 2, 3, 5, 7, 13
- [18] Q. Bu, Y. Yang, J. Cai, S. Gao, G. Ren, M. Yao, P. Luo, and H. Li, “UniVLA: Learning to act anywhere with task-centric latent actions,” in RSS, 2025. 2
- [19] J. Bjorck, F. Casta˜neda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang et al., “GR00T N1: an open foundation model for generalist humanoid robots,” arXiv preprint arXiv:2503.14734, 2025. 2, 3
- [20] L. Chen, C. Sima, K. Chitta, A. Loquercio, P. Luo, Y. Ma, and H. Li, “Intelligent Robot Manipulation Requires Self-Directed Learning,” Authorea Preprints, 2025. 2
- [21] F. Ebert, Y. Yang, K. Schmeckpeper, B. Bucher, G. Georgakis, K. Daniilidis, C. Finn, and S. Levine, “Bridge Data: Boosting generalization of robotic skills with cross-domain datasets,” in RSS, 2022. 2, 3
- [22] H. R. Walke, K. Black, T. Z. Zhao, Q. Vuong, C. Zheng, P. HansenEstruch, A. W. He, V. Myers, M. J. Kim, M. Du et al., “BridgeData v2: A dataset for robot learning at scale,” in CoRL, 2023. 2, 3
- [23] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis et al., “DROID: A large-scale in-the-wild robot manipulation dataset,” in RSS, 2024. 2, 3
- [24] A. Padalkar, A. Pooley, A. Jain, A. Bewley, A. Herzog, A. Irpan, A. Khazatsky, A. Rai, A. Singh, A. Brohan et al., “Open X-Embodiment: Robotic learning datasets and RT-X models,” in ICRA, 2024. 2, 3, 5
- [25] F. Lin, Y. Hu, P. Sheng, C. Wen, J. You, and Y. Gao, “Data scaling laws in imitation learning for robotic manipulation,” in ICLR, 2025. 2, 3, 4, 5
- [26] H. Tan, X. Xu, C. Ying, X. Mao, S. Liu, X. Zhang, H. Su, and J. Zhu, “ManiBox: Enhancing spatial grasping generalization via scalable simulation data generation,” arXiv preprint arXiv:2411.01850, 2024. 2, 4
- [27] L. Wang, X. Chen, J. Zhao, and K. He, “Scaling proprioceptive-visual learning with heterogeneous pre-trained transformers,” in NeurIPS, 2024. 2
- [28] J. Zheng, J. Li, D. Liu, Y. Zheng, Z. Wang, Z. Ou, Y. Liu, J. Liu, Y.-Q. Zhang, and X. Zhan, “Universal actions for enhanced embodied foundation models,” in CVPR, 2025. 2
- [29] J. Yang, C. Glossop, A. Bhorkar, D. Shah, Q. Vuong, C. Finn, D. Sadigh, and S. Levine, “Pushing the limits of cross-embodiment learning for manipulation and navigation,” in RSS, 2024. 2
- [30] R. Doshi, H. Walke, O. Mees, S. Dasari, and S. Levine, “Scaling CrossEmbodied Learning: One policy for manipulation, navigation, locomotion and aviation,” in CoRL, 2024. 2
- [31] H. Li, Y. Cui, and D. Sadigh, “How to train your robots? the impact of demonstration modality on imitation learning,” in ICRA, 2025. 2
- [32] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion Policy: Visuomotor policy learning via action diffusion,” in RSS, 2023. 3, 7

- [33] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in NAACL,

2019. 3

- [34] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozi`ere, N. Goyal, E. Hambro, F. Azhar et al., “LLaMA: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971,

2023. 3

- [35] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., “Language models are few-shot learners,” in NeurIPS, 2020. 3
- [36] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “MiniGPT-4: Enhancing vision-language understanding with advanced large language models,” in ICLR, 2024. 3
- [37] X. Li, M. Liu, H. Zhang, C. Yu, J. Xu, H. Wu, C. Cheang, Y. Jing, W. Zhang, H. Liu, H. Li, and T. Kong, “Vision-language foundation models as effective robot imitators,” in ICLR, 2024. 3
- [38] D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu et al., “PaLM-E: An embodied multimodal language model,” in ICML, 2023. 3
- [39] J. Wen, Y. Zhu, J. Li, M. Zhu, Z. Tang, K. Wu, Z. Xu, N. Liu, R. Cheng, C. Shen et al., “TinyVLA: Towards fast, data-efficient vision-languageaction models for robotic manipulation,” RA-L, 2025. 3
- [40] J. Wen, Y. Zhu, J. Li, Z. Tang, C. Shen, and F. Feng, “DexVLA: Visionlanguage model with plug-in diffusion expert for general robot control,” in CoRL, 2025. 3
- [41] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu et al., “RT-1: Robotics transformer for real-world control at scale,” in RSS, 2023. 3, 4
- [42] D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu et al., “Octo: An open-source generalist robot policy,” in RSS, 2024. 3
- [43] R. Goyal, S. Ebrahimi Kahou, V. Michalski, J. Materzynska, S. Westphal, H. Kim, V. Haenel, I. Fruend, P. Yianilos, M. Mueller-Freitag et al., “The” something something” video database for learning and evaluating visual common sense,” in ICCV, 2017. 3
- [44] K. Grauman, A. Westbury, E. Byrne, Z. Chavis, A. Furnari, R. Girdhar, J. Hamburger, H. Jiang, M. Liu, X. Liu et al., “Ego4D: Around the world in 3,000 hours of egocentric video,” in CVPR, 2022. 3
- [45] J. Bruce, M. D. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps et al., “Genie: Generative interactive environments,” in ICML, 2024. 3
- [46] S. Ye, J. Jang, B. Jeon, S. Joo, J. Yang, B. Peng, A. Mandlekar, R. Tan, Y.-W. Chao, B. Y. Lin et al., “Latent action pretraining from videos,” in ICLR, 2025. 3
- [47] X. Chen, J. Guo, T. He, C. Zhang, P. Zhang, D. C. Yang, L. Zhao, and J. Bian, “IGOR: Image-goal representations are the atomic control units for foundation models in embodied ai,” arXiv preprint arXiv:2411.00785,

2024. 3

- [48] H. Wu, Y. Jing, C. Cheang, G. Chen, J. Xu, X. Li, M. Liu, H. Li, and T. Kong, “Unleashing large-scale video generative pre-training for visual robot manipulation,” in ICLR, 2024. 3
- [49] C.-L. Cheang, G. Chen, Y. Jing, T. Kong, H. Li, Y. Li, Y. Liu, H. Wu, J. Xu, Y. Yang et al., “GR-2: A generative video-language-action model with web-scale knowledge for robot manipulation,” arXiv preprint arXiv:2410.06158, 2024. 3
- [50] Y. Du, S. Yang, B. Dai, H. Dai, O. Nachum, J. Tenenbaum, D. Schuurmans, and P. Abbeel, “Learning universal policies via text-guided video generation,” in NeurIPS, 2023. 3
- [51] J. Zeng, Q. Bu, B. Wang, W. Xia, L. Chen, H. Dong, H. Song, D. Wang, D. Hu, P. Luo et al., “Learning manipulation by predicting interaction,” in RSS, 2024. 3
- [52] Q. Bu, J. Zeng, L. Chen, Y. Yang, G. Zhou, J. Yan, P. Luo, H. Cui, Y. Ma, and H. Li, “Closed-loop visuomotor control with generative expectation for robotic manipulation,” in NeurIPS, 2024. 3
- [53] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in ICCV, 2023. 3
- [54] H.-S. Fang, H. Fang, Z. Tang, J. Liu, C. Wang, J. Wang, H. Zhu, and C. Lu, “RH20T: A comprehensive robotic dataset for learning diverse skills in one-shot,” in ICRA, 2024. 3
- [55] Z. Wang, H. Zheng, Y. Nie, W. Xu, Q. Wang, H. Ye, Z. Li, K. Zhang, X. Cheng, W. Dong et al., “All Robots in One: A new standard and unified dataset for versatile, general-purpose embodied agents,” arXiv preprint arXiv:2408.10899, 2024. 3
- [56] K. Wu, C. Hou, J. Liu, Z. Che, X. Ju, Z. Yang, M. Li, Y. Zhao, Z. Xu, G. Yang et al., “RoboMIND: Benchmark on multi-embodiment intelligence normative data for robot manipulation,” in RSS, 2025. 3

- [57] L. Wu, C. Yu, J. Ren, L. Chen, R. Huang, G. Gu, and H. Li, “FreeTacMan: Robot-free visuo-tactile data collection system for contactrich manipulation,” in ICRA, 2026. 3
- [58] Y. Pan, R. Qiao, L. Chen, K. Chitta, L. Pan, H. Mai, Q. Bu, C. Zheng, H. Zhao, P. Luo, and H. Li, “Agility Meets Stability: Versatile humanoid control with heterogeneous data,” in ICRA, 2026. 3
- [59] G. R. Team, A. Abdolmaleki, S. Abeyruwan, J. Ainslie, J.-B. Alayrac, M. G. Arenas, A. Balakrishna, N. Batchelor, A. Bewley, J. Bingham et al., “Gemini Robotics 1.5: Pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer,” arXiv preprint arXiv:2510.03342, 2025. 3
- [60] S. Kareer, K. Pertsch, J. Darpinian, J. Hoffman, D. Xu, S. Levine, C. Finn, and S. Nair, “Emergence of human to robot transfer in vision-languageaction models,” arXiv preprint arXiv:2512.22414, 2025. 3
- [61] V. Saxena, M. Bronars, N. R. Arachchige, K. Wang, W. C. Shin, S. Nasiriany, A. Mandlekar, and D. Xu, “What matters in learning from large-scale datasets for robot manipulation,” in ICLR, 2025. 4
- [62] J. Hejna, C. A. Bhateja, Y. Jiang, K. Pertsch, and D. Sadigh, “ReMix: Optimizing data mixtures for large scale imitation learning,” in CoRL,

2024. 4

- [63] Y. Zheng, Z. Xia, Q. Zhang, T. Zhang, B. Lu, X. Huo, C. Han, Y. Li, M. Yu, B. Jin et al., “Preliminary investigation into data scaling laws for imitation learning-based end-to-end autonomous driving,” arXiv preprint arXiv:2412.02689, 2024. 4
- [64] T. Mu, Z. Ling, F. Xiang, D. Yang, X. Li, S. Tao, Z. Huang, Z. Jia, and H. Su, “ManiSkill: Generalizable manipulation skill benchmark with large-scale demonstrations,” in NeurIPS Datasets and Benchmarks, 2021. 6
- [65] Y. Mu, T. Chen, S. Peng, Z. Chen, Z. Gao, Y. Zou, L. Lin, Z. Xie, and P. Luo, “Robotwin: Dual-arm robot benchmark with generative digital twins (early version),” in ECCV, 2025. 6, 7
- [66] N. Masuya, S. Sakaino, and T. Tsuji, “Variable-frequency imitation learning for variable-speed motion,” in ICM, 2025. 8
- [67] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in ICCV, 2023. 9
- [68] L. Guo, Z. Xue, Z. Xu, and H. Xu, “DemoSpeedup: Accelerating visuomotor policies via entropy-guided demonstration acceleration,” in CoRL, 2025. 9
- [69] N. R. Arachchige, Z. Chen, W. Jung, W. C. Shin, R. Bansal, P. Barroso, Y. H. He, Y. C. Lin, B. Joffe, S. Kousik et al., “SAIL: Faster-thandemonstration execution of imitation learning policies,” in CoRL, 2025. 9

[Figure 69]

Modi Shi received the B.S. degree from the School of Computer Science and Engineering, Beihang University, Beijing, China. He is currently pursuing the Ph.D. degree at the School of Computer Science and Engineering, Beihang University. His research interests include humanoids and robotic manipulation.

[Figure 70]

Yuxiang Lu is currently a Ph.D. student at the School of Computing and Data Science, the University of Hong Kong. He received the M.E. and B.E. from Shanghai Jiao Tong University. His research interests include embodied AI and computer vision.

[Figure 71]

Chiming Liu is a Large Model Architect at AgiBot. He was previously a Large Language Model (LLM) Architect at Biomap, and prior to that, an Expert Researcher at Tencent. He holds degrees from Nanjing University, China, and the University of Southampton, UK. His research interests lie in the design, pre-training, and post-training of large models, and in the development of large-scale embodied AI systems.

[Figure 72]

Guanghui Ren is currently the Director of AI Algorithms at the Embodied Intelligence Business Unit of AgiBot. He received his master’s degree from the Chinese Academy of Sciences. His research focuses on multimodal large language models, embodied manipulation models, and embodied world models, aiming to develop general-purpose robots.

[Figure 73]

Ping Luo received the PhD degree in information engineering, from the Chinese University of Hong Kong (CUHK), in 2014. He is an associate professor with the Department of Computer Science, University of Hong Kong (HKU). He was a postdoctoral fellow with CUHK from 2014 to 2016. He joined SenseTime Research as a Principal Research Scientist from 2017 to 2018. His research interests are machine learning and computer vision.

[Figure 74]

Di Huang received the B.S. and M.S. degrees in computer science from Beihang University, Beijing, China, in 2005 and 2008, respectively, and the Ph.D. degree in computer science from the Ecole´ Centrale de Lyon, Lyon, France, in 2011. He joined School of Computer Science and Engineering, Beihang University, where he is currently a Professor. His research interests include biometrics, 2D/3D face analysis, image/video processing, and pattern recognition.

[Figure 75]

Li Chen is currently a Ph.D. student at Department of Computer Science, the University of Hong Kong. He received the B.E. in mechanical engineering from Shanghai Jiao Tong University, and the M.S. in Robotics from the University of Michigan, Ann Arbor, USA. His research interests lie in physical AI including robotics and autonomous driving.

[Figure 76]

Jin Chen is currently a Ph.D. student at Shanghai Innovation Institute and in Computer Engineering at Fudan University. He received the M.S. in Mathematics from Xi’an Jiaotong University, and the B.S. in Mathematics from China University of Mining and Technology. His research interests lie in embodied AI, specifically whole-body loco-manipulation.

[Figure 77]

Maoqing Yao currently serves as Partner, Senior Vice President, and President of the Embodied Intelligence Business Unit at AgiBot. He received his bachelor’s degree from Tsinghua University and his Ph.D. from the University of Southern California. Previously, he held technical leadership positions at NIO, Waymo, Google, and Oracle. His research focuses on embodied intelligence, with expertise in AI and robotics research, development, and deployment.

[Figure 78]

Hongyang Li is an Assistant Professor at School of Computing and Data Science, University of Hong Kong. His research focuses on autonomous driving and embodied AI. He led the UniAD project, winning the IEEE CVPR 2023 Best Paper Award, and created AgiBot World for investigating scaling laws in robotic manipulation. He proposed BEVFormer, selected as Top 100 AI Papers in 2022. He served as Area Chair for CVPR, NeurIPS, ICLR, ICCV, ICML, and RSS, and is a Senior Member of IEEE. He received the China AI Wu Wen Jun Early Career Award 2024.

APPENDIX

- A. Hardware Setup

Our real-world experiments are conducted on two robotic platforms: AgiBot G1 and AgileX Cobot Magic with Piper, as shown in Figure 14 and Figure 15 respectively. Both platforms are equipped with a multi-camera setup consisting of one frontfacing camera mounted on the robot’s head and two wrist cameras attached to each arm.

[Figure 79]

Fig. 14: Deployment on AgiBot G1.

[Figure 80]

- Fig. 15: Deployment on AgileX Cobot Magic with Piper.

- B. Implementation Details

- 1) Task Diversity Our experimental configurations vary according to the scale

of pre-training data. For pre-training with 10% of the AgiBot World dataset, we employ 32 H100 GPUs with a batch size of 64 per GPU, training for 100K steps. When scaling to 25% of the data, we maintain the same hardware setup (32 GPUs, batch size 64 per GPU) but extend training to 250K steps. For full-scale pre-training on the complete dataset, we scale up to 96 H100 GPUs with batch size 64 per GPU, training for 500K steps. During the pre-training phase, we train the Vision-Language Model (VLM) component alongside other model parameters. For fine-tuning, we use 8 GPUs with a batch size of 128 per GPU, training for 20K steps while keeping the VLM parameters frozen.

2) Embodiment Diversity

Embodiment Diversity Experiments: We fine-tuned RDTOXE and RDT-AWB on three platforms: Maniskill, RoboTwin, and real-world Agilex, with results reported in Figure 7, Figure 9, and Table I respectively.

For Maniskill, we evaluated five tasks (PegInsertionSide, PickCube, StackCube, PlugCharger, PushCube) using only the third-person observations. We fine-tuned with 125/250/500/1000 demonstrations per task for 37.5K/75K/150K/300K steps respectively, saving checkpoints every 10K steps and reporting peak success rates.

For RoboTwin, we tested four tasks (BlockHammerBeat, BlocksStack, ContainerPlace, DualBottlesPick) using four camera views (front, head, left/right wrist). Fine-tuning used 25/50/100/200 demonstrations per task for 10K/30K/50K/60K steps respectively.

For real-world Agilex, we evaluated four tasks (PackageProduct, FoldShorts, CleanTrash, IndustrialSorting) using three camera views (front, left/right wrist). We used 100 demonstrations per task (200 for FoldShorts due to complexity) and trained for 100K steps.

Pre-training Setup: RDT-AWB was pre-trained on AgiBot World using 96 H100 GPUs (batch size 32 per GPU, 200K steps). RDT-OXE used the official checkpoint (48 H100 GPUs, batch size 32 per GPU, 1M steps). All other hyperparameters remained consistent between models.

3) Expert Diversity

Our action rate model employs a SigLIP vision encoder coupled with a three-layer MLP featuring LayerNorm, GELU activations, and 0.1 dropout. For fine-tuning experiments, we train the action rate model using 8 H100 GPUs with a batch size of 128 per GPU for 20K steps, while pre-training experiments employ the same hardware configuration but extend training to 100K steps. The GO-1-Pro training setup remains identical to GO-1.

C. Additional Experiment Results

1) Task Diversity

To provide a more granular analysis of how different pretraining data compositions affect policy performance across task execution, we present per-step scores in Table III. Each task is decomposed into sequential manipulation steps (e.g., grasp, place, wipe), allowing evaluation of performance distribution throughout the entire task sequence rather than relying solely on average scores.

Furthermore, to verify that our findings on task diversity generalize beyond the GO-1 architecture, we conduct additional experiments using RDT [17] as an alternative policy model. We employ identical experimental settings as described in Section III: the same three pre-training datasets (scenariobased, task-based, and episode-based sampling at 10% scale), the same evaluation tasks (Wipe Table, Fold Shorts, Pour Water, Make Sandwich), and the same evaluation protocols across three scenarios (in-domain, visual distraction, and objectenvironment generalization).

As shown in Figure 16, RDT exhibits consistent patterns with GO-1 findings. At 10% data scale, scenario-based sampling achieves 0.20 average performance, task-based sampling

TABLE III: Performance comparison across different pretraining data.

Make Sandwich Wipe Table

Pretrain Data

Grasp Place Grasp Place Grasp Place Grasp Place Grasp Wipe

No Pretrain 0.53 0.54 0.43 0.45 0.36 0.30 0.17 0.12 0.75 0.41 10% Scenario 0.63 0.60 0.37 0.37 0.31 0.27 0.23 0.18 0.84 0.48 10% Task 0.70 0.60 0.43 0.40 0.35 0.33 0.27 0.20 0.82 0.48 10% Episode 0.80 0.83 0.76 0.73 0.63 0.67 0.46 0.48 0.82 0.50 25% Episode 0.83 0.80 0.76 0.70 0.67 0.63 0.57 0.48 0.85 0.67 Full Data 0.86 0.80 0.83 0.73 0.63 0.60 0.57 0.57 0.92 0.73

### Fold Shorts Pour Water

Avg. Grasp Grasp Fold Grasp Grasp Fold Grasp Pour Place

No Pretrain 0.55 0.46 0.07 0.03 0.00 0.00 0.00 0.00 0.00 0.28 10% Scenario 0.53 0.47 0.23 0.09 0.06 0.00 0.10 0.03 0.00 0.33 10% Task 0.80 0.60 0.27 0.13 0.10 0.07 0.13 0.06 0.00 0.36 10% Episode 0.73 0.60 0.23 0.13 0.07 0.03 0.30 0.17 0.13 0.46 25% Episode 0.77 0.74 0.13 0.09 0.07 0.00 0.53 0.37 0.18 0.52 Full Data 0.80 0.77 0.43 0.33 0.27 0.22 0.50 0.33 0.19 0.58

[Figure 81]

- Fig. 16: Real-robot evaluation of RDT on four challenging tasks with different pre-training datasets. Similar to GO-1 results, episode-based sampling consistently outperforms both task-based and scenario-based sampling, demonstrating that the importance of scene diversity generalizes across different model architectures.

In-Domain Visual Distraction Object-Environment Generalization

0.0

0.2

0.4

0.6

0.8

No Pretrain

| |
|---|

10% Scenario

| |
|---|

10% Task

| |
|---|

10% Episode

| |
|---|

25% Episode

| |
|---|

Full Data

0.21

0.13 0.12

0.26

0.18 0.15

0.25 0.23 0.22

0.35

0.30 0.28

0.40

0.34 0.33

0.43

0.39

0.37

- Fig. 17: RDT generalization performance across different pretraining strategies. The model exhibits similar patterns to GO-1: episode-based sampling demonstrates superior performance under distribution shifts, with scene diversity providing greater robustness benefits than skill diversity alone.

improves to 0.23, while episode-based sampling reaches 0.31. Figure 17 further validates our conclusions about scene diversity versus skill diversity. For in-domain evaluation, episodebased sampling (0.35) outperforms task-based (0.25, +0.10) and scenario-based (0.26, +0.09) by similar margins. Under distribution shifts, scene diversity advantages become more pronounced: for visual distraction, episode-based (0.30) exceeds scenario-based (0.18) by 0.12 compared to 0.07 over task-based (0.23). For object-environment generalization, episode-based (0.28) surpasses scenario-based (0.15) by 0.13 versus 0.06 over task-based (0.22).

These results with RDT corroborate our main findings from GO-1 experiments: (1) scene diversity is more critical than skill diversity for robustness and generalization; (2) while both diversity types provide comparable in-domain benefits, scene diversity consistently provides greater advantages under

- TABLE IV: Performance on each task in ManiSkill. The performance of RDT-OXE and RDT-AWB fine-tuned with various numbers of demonstrations for each task. Higher scores are bolded for emphasis.

Model Demonstrations PegInsertionSide PickCube StackCube PlugCharger PushCube Average Score

RDT-OXE

125 0.00 0.06 0.09 0.00 0.80 0.22 250 0.00 0.25 0.34 0.00 0.97 0.31 500 0.04 0.40 0.42 0.03 0.98 0.38

1000 0.04 0.76 0.66 0.04 1.00 0.50

RDT-AWB

125 0.02 0.18 0.07 0.00 0.78 0.21 250 0.02 0.19 0.40 0.00 0.96 0.32 500 0.00 0.54 0.65 0.01 0.96 0.43

1000 0.12 0.86 0.90 0.03 1.00 0.58

- TABLE V: Performance on each task in RoboTwin. The performance of RDT-OXE and RDT-AWB fine-tuned with various numbers of demonstrations for each task. Higher scores are bolded for emphasis.

Model Demonstrations BlockHammerBeat BlocksStack ContainerPlace DualBottlesPick Average Score

25 0.55 0.02 0.52 0.19 0.32 50 0.61 0.13 0.49 0.47 0.42

RDT-OXE

100 0.87 0.32 0.57 0.58 0.59 200 0.60 0.53 0.49 0.68 0.58

25 0.63 0.04 0.43 0.42 0.38 50 0.67 0.20 0.40 0.42 0.42

RDT-AWB

100 0.90 0.51 0.50 0.60 0.63 200 0.94 0.55 0.60 0.67 0.69

distribution shifts; (3) these patterns generalize across different policy architectures, suggesting that the importance of diversity composition is a fundamental property of robotic manipulation learning rather than architecture-specific behavior.

2) Embodiment Diversity

In this section, we provide additional detailed results for the simulation experiments in Section IV, showcasing the performance of RDT-OXE and RDT-AWB on each task in ManiSkill (Table IV) and RoboTwin (Table V).

3) Expert Diversity

To further validate the effectiveness of distribution debiasing at scale, we conduct additional experiments using the full AgiBot World Beta dataset for pre-training. As pre-training data scale increases, the action rate distribution becomes more complex and heterogeneous, potentially amplifying the benefits of distribution debiasing. As shown in Table VI, distribution debiasing applied across both pre-training and fine-tuning stages yields the best performance (0.64 average score), consistent with our findings on 10% data. Notably, the absolute performance improvements remain substantial even at this larger scale, demonstrating that debiasing benefits persist and potentially amplify as dataset complexity increases. This validates that our distribution debiasing approach is effective across different data scales and that addressing action rate multimodality becomes increasingly important as pre-training datasets grow in size and diversity.

D. Evaluation Details

In all the simulation experiments, we use the simple success rate as the evaluation metric. For real-world experiments, we define more fine-grained evaluation metrics to more precisely compare the capabilities of the models within a limited number of rollouts. To ensure unbiased and consistent evaluation, we implemented rigorous protocols: dedicated personnel responsible for full-time model evaluation were not informed about which policy was being evaluated during trials to prevent potential bias. We have three independent scorers in total, with all trials from one specific task scored by a single scorer to ensure consistency within each task and eliminate inter-scorer variability. Importantly, these scorers are also the developers of the scoring rubric, ensuring deep understanding and consistent application of the evaluation criteria. In Section III, Section IV and Section V, we cover seven tasks: Wipe Table, Fold Shorts, Pour Water, Make Sandwich, Package Product, Clean Trash, and Industrial Sorting. We have defined different evaluation metrics for each of them:

1) Wipe Table

Task description. The robot is tasked with a contact-rich manipulation: Wipe Table, which demands the use of a sponge to clean beverage stains from the table surface. The task consists of 2 steps: first, the right arm grasps the sponge; and second, the right arm wipes the stain clean. The sponge, being a deformable object, introduces complexities in grasping and manipulation due to its softness and flexibility. The stain, being a liquid, presents additional challenges due to its irregular shape and the need for precise contact control.

- TABLE VI: Performance evaluation using full AgiBot World Beta dataset for pre-training. Distribution debiasing applied during the pre-training phase demonstrates consistent performance improvements. Additional debiasing during fine-tuning further enhances model capabilities across all evaluated tasks.

Training Setting Task Completion Score Average Pre-training Data Fine-tuning Data Pour Water Fold Shorts Make Sandwich Wipe Table Average

Biased Biased 0.34±0.03 0.47±0.06 0.70±0.04 0.83±0.03 0.58±0.04 Debiased Biased 0.38±0.05 0.52±0.05 0.70±0.03 0.85±0.03 0.61±0.04

Biased Debiased 0.41±0.04 0.49±0.04 0.72±0.03 0.85±0.02 0.62±0.03 Debiased Debiased 0.44±0.03 0.53±0.05 0.73±0.04 0.86±0.02 0.64±0.03

Scoring criteria.

- • Step 1: Grasping the sponge

- – scoring 0: The gripper does not grasp the sponge.
- – scoring 0.5: The gripper attempts to grasp the sponge multiple times, struggles to maintain a stable grasp, but eventually succeeds.
- – scoring 1: The gripper successfully grasps the sponge without any slippage on the first attempt.

- • Step 2: Wiping the stain clean

- – scoring 0: The sponge does not make contact with the stain, or the sponge drops during the wiping process.
- – scoring 0.5: The sponge makes contact with the stain but does not effectively wipe it clean, or the stain is only partially cleaned.
- – scoring 1: The sponge successfully wipes the stain completely clean on the first attempt.

2) Fold Shorts

Task description. Fold Shorts is a complex, bimanual deformable object manipulation task that involves multiple folding operations to neatly organize shorts. This task requires the robot to perform a series of coordinated actions using both arms to fold the shorts accurately. The task consists of six steps: first, the left arm grasps the left side of the shorts; second, the right arm grasps the right side of the shorts; third, both arms fold the shorts forward; fourth, the left arm grasps the left side of the shorts again; fifth, the right arm grasps the right side of the shorts again; and sixth, both arms fold the shorts again, with the left arm holding the shorts in place and the right arm folding them over to the left arm’s position. The complexity of this task lies in the need for precise bimanual coordination and the challenges posed by the deformable nature of the shorts. The softness and flexibility of the shorts make grasping and folding operations difficult, requiring careful control of force and motion. Additionally, the thin material of the shorts increases the risk of collisions with the table, adding another layer of complexity.

Scoring criteria.

• Step 1: Left arm grasping the left side

- – scoring 0: The left arm does not grasp the shorts.
- – scoring 0.5: The left arm attempts to grasp the shorts multiple times, struggles to maintain a stable grasp, or takes too long to succeed.
- – scoring 1: The left arm successfully grasps the shorts on the first attempt without any slippage.

#### • Step 2: Right arm grasping the right side

- – scoring 0: The right arm does not grasp the shorts.
- – scoring 0.5: The right arm attempts to grasp the shorts multiple times, struggles to maintain a stable grasp, or takes too long to succeed.
- – scoring 1: The right arm successfully grasps the shorts on the first attempt without any slippage.

#### • Step 3: Bimanual forward folding of the shorts

- – scoring 0: The shorts are not folded forward or the folding is incomplete.
- – scoring 0.5: The shorts are folded forward but not perfectly aligned or the folding is not symmetrical.
- – scoring 1: The shorts are folded forward perfectly, with symmetrical alignment.

#### • Step 4: Left arm grasping the left side again

- – scoring 0: The left arm does not grasp the shorts.
- – scoring 0.5: The left arm attempts to grasp the shorts multiple times, struggles to maintain a stable grasp, or takes too long to succeed.
- – scoring 1: The left arm successfully grasps the shorts on the first attempt without any slippage.

#### • Step 5: Right arm grasping the right side again

- – scoring 0: The right arm does not grasp the shorts.
- – scoring 0.5: The right arm attempts to grasp the shorts multiple times, struggles to maintain a stable grasp, or takes too long to succeed.
- – scoring 1: The right arm successfully grasps the shorts on the first attempt without any slippage.

#### • Step 6: Bimanual folding of the shorts with the left arm holding and the right arm folding

- – scoring 0: The shorts are not folded or the folding is incomplete.
- – scoring 0.5: The shorts are folded but not perfectly aligned or the folding is not symmetrical.
- – scoring 1: The shorts are folded perfectly, with symmetrical alignment.

3) Pour Water

Task description. Pour Water is a fine-grained manipulation task requiring the robot to grasp a kettle handle and pour water into a cup. The task consists of two steps: first, the robot must grasp the kettle handle; and second, the robot must pour water from the kettle into the cup. The task requires precise control over the kettle’s position and pouring angle to ensure

accurate pouring. The complexity of this task lies in the need for precise spatial and temporal control during the pouring process, including accurate positioning, controlled pouring, and managing the flow rate of the water.

Scoring criteria.

- • Step 1: Grasping the kettle handle

- – scoring 0: The gripper does not grasp the kettle handle.
- – scoring 0.5: The gripper attempts to grasp the kettle handle multiple times, struggles to maintain a stable grasp, or takes too long to succeed.
- – scoring 1: The gripper successfully grasps the kettle handle on the first attempt without any slippage.

- • Step 2: Pouring water from the kettle into the cup

- – scoring 0: The robot does not pour the water, or the water is poured outside the cup.
- – scoring 0.5: The robot pours the water but spills some water outside the cup, or the pouring process takes too long or requires multiple attempts to succeed.
- – scoring 1: The robot successfully pours the water into the cup without any spillage on the first attempt.

4) Make Sandwich

Task description. Make Sandwich is a long-horizon task that sequentially involves picking up bread, ham, and lettuce to assemble a sandwich in proper order. The task consists of eight steps: first, the robot must grasp the first slice of bread; second, place the bread on the plate; third, grasp a slice of ham; fourth, place the ham on the bread; fifth, grasp a piece of lettuce; sixth, place the lettuce on the ham; seventh, grasp the second slice of bread; and eighth, place the bread on the lettuce. The complexity of this task lies in the need for precise sequential manipulation and the long-horizon nature of the task, requiring the robot to maintain accuracy and stability over multiple steps.

Scoring criteria.

- • Step 1: Grasping the first slice of bread

- – scoring 0: The gripper does not grasp the bread.
- – scoring 0.5: The gripper attempts to grasp the bread multiple times, struggles to maintain a stable grasp, or takes too long to succeed.
- – scoring 1: The gripper successfully grasps the bread on the first attempt without any slippage.

- • Step 2: Placing the bread on the plate

- – scoring 0: The bread is not placed on the plate or is dropped.
- – scoring 0.5: The bread is placed on the plate but not accurately or takes multiple attempts.
- – scoring 1: The bread is accurately placed on the plate on the first attempt.

- • Step 3: Grasping a slice of ham

- – scoring 0: The gripper does not grasp the ham.
- – scoring 0.5: The gripper attempts to grasp the ham multiple times, struggles to maintain a stable grasp, or takes too long to succeed.
- – scoring 1: The gripper successfully grasps the ham on the first attempt without any slippage.

- • Step 4: Placing the ham on the bread

- – scoring 0: The ham is not placed on the bread or is dropped.
- – scoring 0.5: The ham is placed on the bread but not accurately or takes multiple attempts.
- – scoring 1: The ham is accurately placed on the bread on the first attempt.

#### • Step 5: Grasping a piece of lettuce

- – scoring 0: The gripper does not grasp the lettuce.
- – scoring 0.5: The gripper attempts to grasp the lettuce multiple times, struggles to maintain a stable grasp, or takes too long to succeed.
- – scoring 1: The gripper successfully grasps the lettuce

on the first attempt without any slippage.

#### • Step 6: Placing the lettuce on the ham

- – scoring 0: The lettuce is not placed on the ham or is dropped.
- – scoring 0.5: The lettuce is placed on the ham but not accurately or takes multiple attempts.
- – scoring 1: The lettuce is accurately placed on the ham on the first attempt.

#### • Step 7: Grasping the second slice of bread

- – scoring 0: The gripper does not grasp the bread.
- – scoring 0.5: The gripper attempts to grasp the bread multiple times, struggles to maintain a stable grasp, or takes too long to succeed.
- – scoring 1: The gripper successfully grasps the bread on the first attempt without any slippage.

#### • Step 8: Placing the bread on the lettuce

- – scoring 0: The bread is not placed on the lettuce or is dropped.
- – scoring 0.5: The bread is placed on the lettuce but not accurately or takes multiple attempts.
- – scoring 1: The bread is accurately placed on the lettuce on the first attempt.

5) Package Product

Task description. Package Product is a precision manipulation task that requires the robot to grasp a product and place it into a bag. The task consists of two steps: first, the robot must grasp the product; and second, the robot must place the product into the bag. The complexity of this task lies in the need for precise control over the grasping and placement actions, ensuring that the product is handled carefully and placed accurately into the bag.

Scoring criteria.

- • Step 1: Grasping the product

- – scoring 0: The gripper does not grasp the product.
- – scoring 0.5: The gripper attempts to grasp the product multiple times, struggles to maintain a stable grasp, or takes too long to succeed.
- – scoring 1: The gripper successfully grasps the product on the first attempt without any slippage.

- • Step 2: Placing the product into the bag

- – scoring 0: The product is not placed into the bag or is dropped.
- – scoring 0.5: The product is placed into the bag but not accurately or takes multiple attempts.

– scoring 1: The product is accurately placed into the

bag on the first attempt.

6) Clean Trash

Task description. Clean Trash is a precision manipulation task that requires the robot to grasp trash and place it into a trash bin. The task consists of two steps: first, the robot must grasp the trash; and second, the robot must place the trash into the trash bin. The complexity of this task lies in the need for precise control over the grasping and placement actions, ensuring that the trash is handled carefully and placed accurately into the trash bin.

Scoring criteria.

- • Step 1: Grasping the trash

- – scoring 0: The gripper does not grasp the trash.
- – scoring 0.5: The gripper attempts to grasp the trash multiple times, struggles to maintain a stable grasp, or takes too long to succeed.
- – scoring 1: The gripper successfully grasps the trash on the first attempt without any slippage.

- • Step 2: Placing the trash into the trash bin

- – scoring 0: The trash is not placed into the trash bin or is dropped.
- – scoring 0.5: The trash is placed into the trash bin but not accurately or takes multiple attempts.
- – scoring 1: The trash is accurately placed into the trash bin on the first attempt.

7) Industrial Sorting

Task description. Industrial Sorting is a precision manipulation task that requires the robot to identify two different items and place them into their corresponding designated areas using the left and right arms. The task consists of four steps: first, the right arm must grasp item 1; second, the right arm must place item 1 into its designated area; third, the left arm must grasp item 2; and fourth, the left arm must place item

- 2 into its designated area. The complexity of this task lies in the need for precise control over the grasping and placement actions, ensuring that each item is handled carefully and placed accurately into the correct area.

Scoring criteria.

- • Step 1: Right arm grasping item 1

- – scoring 0: The right gripper does not grasp item 1.
- – scoring 0.5: The right gripper attempts to grasp item 1 multiple times, struggles to maintain a stable grasp, or takes too long to succeed.
- – scoring 1: The right gripper successfully grasps item 1 on the first attempt without any slippage.

- • Step 2: Right arm placing item 1 into its designated area

- – scoring 0: Item 1 is not placed into its designated area or is dropped.
- – scoring 0.5: Item 1 is placed into its designated area but not accurately or takes multiple attempts.
- – scoring 1: Item 1 is accurately placed into its designated area on the first attempt.

- • Step 3: Left arm grasping item 2

– scoring 0: The left gripper does not grasp item 2.

- – scoring 0.5: The left gripper attempts to grasp item 2 multiple times, struggles to maintain a stable grasp, or takes too long to succeed.
- – scoring 1: The left gripper successfully grasps item 2 on the first attempt without any slippage.

• Step 4: Left arm placing item 2 into its designated

area

- – scoring 0: Item 2 is not placed into its designated area or is dropped.
- – scoring 0.5: Item 2 is placed into its designated area but not accurately or takes multiple attempts.
- – scoring 1: Item 2 is accurately placed into its designated area on the first attempt.

8) Push Chairs

Task description. Push Chairs is a mobile manipulation task that requires the robot to coordinate base mobility with manipulation to tidy up a room by returning two chairs to their designated positions (e.g., under a table). The task consists of four steps: first, the robot must navigate to a position behind the first chair; second, the robot must push the first chair to its target location; third, the robot must navigate to a position behind the second chair; and fourth, the robot must push the second chair to its target location. The complexity of this task lies in the precise coupling of navigation and manipulation, requiring accurate alignment relative to the large objects to ensure effective pushing forces and correct final poses.

Scoring criteria.

- • Step 1: Moving behind the first chair

- – scoring 0: The robot fails to move or navigates to an incorrect location.
- – scoring 0.5: The robot moves towards the chair, but its orientation is significantly skewed or the distance is too large, rendering the subsequent pushing action impossible or highly unstable.
- – scoring 1: The robot successfully navigates to the correct position behind the first chair with proper alignment for pushing.

- • Step 2: Pushing the first chair to the target position

- – scoring 0: The robot fails to push the chair, knocks it over, or pushes it to a completely incorrect area.
- – scoring 0.5: The robot pushes the chair, but the final orientation is significantly misaligned or the chair remains far from the specific target position.
- – scoring 1: The robot successfully pushes the first chair to the designated position with correct orientation on the first attempt.

- • Step 3: Moving behind the second chair

- – scoring 0: The robot fails to move or navigates to an incorrect location.
- – scoring 0.5: The robot moves towards the chair, but its orientation is significantly skewed or the distance is too large, rendering the subsequent pushing action impossible or highly unstable.
- – scoring 1: The robot successfully navigates to the correct position behind the second chair with proper alignment for pushing.

#### • Step 4: Pushing the second chair to the target position

- – scoring 0: The robot fails to push the chair, knocks it over, or pushes it to a completely incorrect area.
- – scoring 0.5: The robot pushes the chair, but the final orientation is significantly misaligned or the chair remains far from the specific target position.
- – scoring 1: The robot successfully pushes the second chair to the designated position with correct orientation on the first attempt.

