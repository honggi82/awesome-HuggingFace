## arXiv:2311.06430v1[cs.RO]10Nov2023

# GOAT: GO to Any Thing

Matthew Chang,∗1 Theophile Gervet,∗2 Mukul Khanna,∗3 Sriram Yenamandra,∗3 Dhruv Shah,4, So Yeon Min,2, Kavit Shah,5, Chris Paxton,5 Saurabh Gupta,1 Dhruv Batra,5 Roozbeh Mottaghi,5 Jitendra Malik,∗4,5 Devendra Singh Chaplot∗6

∗Equal Contribution, 1University of Illinois Urbana-Champaign, 2Carnegie Mellon University, 3Georgia Institute of Technology, 4University of California, Berkeley, 5Meta AI Research, 6Mistral AI Project Website

In deployment scenarios such as homes and warehouses, mobile robots are expected to autonomously navigate for extended periods, seamlessly executing tasks articulated in terms that are intuitively understandable by human operators. We present GO To Any Thing (GOAT), a universal navigation system capable of tackling these requirements with three key features: a) Multimodal: it can tackle goals specified via category labels, target images, and language descriptions, b) Lifelong: it benefits from its past experience in the same environment, and c) Platform Agnostic: it can be quickly deployed on robots with different embodiments. GOAT is made possible through a modular system design and a continually augmented instance-aware semantic memory that keeps track of the appearance of objects from different viewpoints in addition to category-level semantics. This enables GOAT to distinguish between different instances of the same category to enable navigation to targets specified by images and language descriptions. In experimental comparisons spanning over 90 hours in 9 different homes consisting of 675 goals selected across 200+ different object instances, we find GOAT achieves an overall success rate of 83%, surpassing previous methods and ablations by 32% (absolute improvement). GOAT improves with experience in the environment, from a 60% success rate at the first goal to a 90% success after exploration. In addition, we demonstrate that GOAT can readily be applied to downstream tasks such as pick and place and social navigation.

### 1 Introduction

Ever since there were animals that could move, navigation to desired locations – food, mates, nests - has been a fundamental aspect of animal and human behavior. The scientific study of navigation is a very interdisciplinary field to which contributions have been made by, among others, researchers in ethology, zoology, psychology, neuroscience, and robotics. In this paper, we present a mobile robot system that was inspired by some of the most salient findings from animal and human navigation.

- • Cognitive maps. Many animals maintain internal spatial representations of their environment. There has been a vigorous debate on the nature of this map – is it metric in a Euclidean sense or just topological- and in Nobel prize-winning work, neural correlates of cognitive maps have been found in the hippocampus. This suggests that a purely reactive, memoryless navigation system is inadequate for robotics. [58]
- • How is this internal spatial representation acquired? From human studies, it has been argued that these are built up through “route-based” knowledge. In the process of daily or other episodic activities, we learn the structure of a route – origin, destination, waypoints etc. Over time features from different experiences are integrated together into a single layout representation, the “map”. For mobile robots, this motivates a version of “lifelong learning” - continual improvement of the internal spatial representation as the mobile robot performs active search and exploration. [26]
- • Is navigation driven exclusively by the geometric configuration of locations? No, as the visual appearance of landmarks plays a major role in animal and human navigation. This suggests maintaining rich multimodal representations of the spatial environment of the mobile robot. [12]

Let us get concrete. Consider a robot starting in an unseen environment as shown in Figure 1, and suppose it is asked to find a dining table image (goal 1). Navigating to this goal requires recognizing that the picture shows a dining table and having the semantic understanding of indoor spaces to efficiently explore the home (e.g. dining tables are not found in the bathroom). Suppose the robot is then asked to Go to the potted plant next to the couch (goal 2). This requires visual grounding of the text instruction in the physical space. The next instruction is to Go to a SINK (goal 3), the capitalization emphasizing that any object of the category SINK is a valid goal. In this example, the robot has already seen a sink in the house during the first task, so it should remember its location and be able to plan a path to reach it efficiently. This requires the robot to build, maintain and update a lifelong memory of the objects in the environment, their visual and linguistic properties and their latest location. Given any new multimodal goal, the robot should also be able to query the memory to determine whether the goal object already exists in the memory or requires further exploration. In addition to these capabilities for multimodal perception, exploration, lifelong memory, and goal localization, the robot also needs effective planning and control to reach the goal while avoiding obstacles.

[Figure 1]

- Figure 1: GOAT (GO to Any Thing) task. The GOAT task requires lifelong learning, meaning taking advantage of past experience in the same environment, for multimodal navigation. The robot must be able to reach any object specified in any way and remember object locations to come back to them.

In this paper, we present GO to Any Thing (GOAT), a universal navigation system with three key features: a) Multimodal: it can tackle goals specified via category labels, target images, and language descriptions, b) Lifelong: it benefits from its past experience in the same environment in the form of a map of objects instances (as opposed to stored implicitly within the parameters of a machine learning model) updated over time, and c) Platform Agnostic: it can be seamlessly deployed on robots with different embodiments — we deploy GOAT on a quadruped and a wheeled robot. GOAT is made possible through the design of an instanceaware semantic memory that keeps track of the appearance of objects from different viewpoints in addition to category-level semantics. This enables GOAT to distinguish between different

instances of the same category to enable navigation to targets specified by images and finegrained language descriptions. This memory is continually augmented as the agent spends more time in the environment, leading to improved efficiency in reaching goals over time.

In experimental comparisons spanning over 90 hours in 9 different homes consisting of 675 goals selected across 200+ different object instances, we find GOAT achieves an overall success rate of 83%, surpassing previous methods and ablations by 32% (absolute improvement). GOAT performance improves with experience in the environment from a 60% success rate at the first goal to 90% success rate once the environment is fully explored. In addition, we demonstrate that GOAT, as a general navigation primitive, can readily be applied to downstream tasks like pick and place and social navigation. GOAT’s performance can in part be attributed to the modular nature of the system: it leverages learning in the components in which it is required (i.e. object detection, image/language matching) while still leveraging strong classical methods (i.e. mapping and planning). Modularity is also responsible for the ease of deployment across different robot embodiments and downstream applications, as individual components can be easily adapted or new components introduced.

While there is a large body of work on navigation [57], most only evaluate in simulation or develop specialized solutions to tackle a subset of these tasks. Classical robotics works [56] employed geometric reasoning to solve navigation to geometric goals. With advances in semantic understanding of images, researchers started using semantic reasoning to improve exploration efficiency in novel environments [9] and tackling semantic goals specified via categories [46, 21, 3, 8, 34, 53, 6], images [63, 10, 22, 33, 32] and language instructions [41, 55, 18]. Most of these methods are a) specialized to a single task (i.e. they are uni-modal), b) only tackle a single goal in each episode (i.e. are not lifelong), and c) evaluated only in simulation (or rudimentary real-world environments). GOAT advances upon these works on all three fronts and tackles multiple goal specifications in a lifelong manner in the real world. This supersedes past works that only innovate along one axis, e.g. past works [59, 10] tackle a sequence of goal but goals are limited to either be object goals [59] or image goals [10] in simulation, [1] tackle flexible goal specifications but only show simulated results for one goal per episode, and [19] show real world results but only for reaching one object goal per episode.

Inspired by animal and human navigation, GOAT maintains a map of the environment as well as visual landmarks - egocentric views of object instances - which are stored in our novel instance-aware object memory. This memory should be queryable with both images and natural language to satisfy GOAT’s multimodality requirement. We enable this by storing raw images for visual landmarks, as opposed to features, allowing us to leverage recent advances in image-image matching and image-language matching independently. We use Contrastive Language-Image Pretraining (CLIP) [45] for image-language matching and SuperGlue [49] for image-image matching. CLIP follows a long history of associating text with images or regions in images [25, 16, 17, 15, 31, 35, 43] and has led to the development of languageconditioned open-vocabulary object detectors [62, 37, 42]. CLIP itself, or object detectors derived from CLIP have recently been used for robotic tasks, e.g. object search [18], mobile manipulation [61], and table-top manipulation [52]. Similarly, SuperGlue follows a long his-

tory of geometric image matching [27, 38] with recent learning-based methods [49] leading to better performance in certain situations. Recent work has started evaluating these in embodied settings where a robot must navigate either to an image in the world [33, 10] or to an image corresponding to a particular object instance [32].

GOAT’s memory representation follows a long history of scene representation in robotics over the last 40 years: occupancy maps (with geometry [14], explicit semantics [48, 7], or implicit semantics [21]), topological representations [10, 11, 36, 50], and neural feature fields [54, 51, 40, 5]. Many of these works have started using pre-trained vision-language features like CLIP [45] and either projecting them into 3D directly [29] or capturing them in an implicit neural field [51, 5]. Parametric representations summarize the environment into low-dimensional abstract features, while non-parametric representations view the collection of images itself as a representation. Our work leverages aspects of both. We build a semantic map for navigating to objects but also store raw images associated with discovered objects (landmarks).

### 2 Results

Video 1 summarizes our results. We deployed GOAT on and conducted qualitative experiments with Boston Dynamics Spot and Hello Robot Stretch robots. We conducted large-scale quantitative experiments with GOAT on Spot (due to its higher reliability) against 3 baselines in 9 real-world homes to reach a total of 200+ different object instances (see Figure 2).

#### 2.1 Go To Any Thing: Lifelong Learning for Multimodal Navigation

We formalize the Go to Any Thing task T as follows. We construct navigation episodes consisting of a sequence of unseen goal objects to be reached in unseen environments. The robot is spawned at a random location. At every timestep t, the robot receives observations consisting of an RGB image It, depth image Dt, and pose reading xt from onboard sensors, as well as the current object goal gk, k ∈ {1,2,..,5 − 10}, which consists in an object category (SINK, CHAIR), an image or language description (the potted plant next to the couch, the black and white striped bed) uniquely identifying an object instance in the environment. The robot must reach the goal object gk as efficiently as possible within a limited time budget. As soon as it reaches the current goal or when the time budget is exhausted, the robot receives the next goal to navigate to, gk+1. In searching for this sequence of goals the agent is allowed to maintain a memory computed using incoming observations. In this way, if gk+1 has been observed during the process of reaching gk the agent can often more efficiently navigate to gk+1.

#### 2.2 Navigation Performance in Unseen Natural Home Environments

In this section, we evaluate the ability of the GOAT agent to tackle the GOAT task, i.e., reach a sequence of unseen multimodal object instances in unseen environments.

[Figure 2]

##### Figure 2: “In-the-wild” evaluation. We deploy the GOAT navigation policy in 9 visually diverse homes and evaluate in on reaching 200+ different object instances as category, image, or language goals. GOAT is platform-agnostic: we deploy it on both Boston Dynamics Spot and Hello Robot Stretch.

GOAT Agent Figure 7 (A) shows an overview of the GOAT system. As the agent moves through the scene, the perception system processes RGB-D camera inputs to detect object instances and localize them into a top-down semantic map of the scene. In addition to the semantic map, GOAT maintains an Object Instance Memory that localizes individual instances of object categories in the map and stores images in which each instance has been viewed. This Object Instance Memory gives GOAT the ability to perform lifelong learning for multimodal navigation. When a new goal is specified to the agent, a global policy first searches the Object Instance Memory to see if the goal has already been observed. After an instance is selected, its stored location in the map is used as a long-term point navigation goal. If no instance is localized, the global policy outputs an exploration goal. A local policy finally computes actions towards the long-term goal. We will dive into more details in the Materials and Methods section.

Instance Matching Strategy The matching module of the global policy has to identify the goal object instance among previously seen object instances in the Object Instance Memory. We evaluated different design choices and settled on the following: match language goal descriptions with object views in memory using the cosine similarity score between their CLIP [45] features, match image goals with object views in memory using keypoint-based matching with SuperGLUE [49], represent object views in memory as bounding boxes with some padding to include additional context, match the goal only against instances of the same object category, match the goal with the instance with the maximum matching score across all views. Further details are in Section 5.1 in the Supplementary Materials.

Experimental Setting We evaluate the GOAT agent as well as three baselines in nine visually diverse homes (see Figure 2) with 10 episodes per home consisting of 5-10 object instances randomly selected out of objects available in the home, representing 200+ different object instances in total (see Figures 9 and 10). We selected goals across 15 different object categories (‘chair’, ‘couch’, ‘potted plant’, ‘bed’, ‘toilet’, ‘tv’, ‘dining table’, ‘oven’, ‘sink’, ‘refrigerator’, ‘book’, ‘vase’, ‘cup’, ‘bottle’, ‘teddy bear’), took a picture for image goals following the protocol in Krantz et al. [32], and annotated 3 different language descriptions uniquely identifying the object. To generate an episode within a home, we sampled a random sequence of 5-10 goals split equally among language, image, and category goals among all object instances available. We evaluate approaches in terms of success rate to reach the goal and SPL [2], which measures path efficiency as the ratio of the agent’s path length over the optimal path length. We report evaluation metrics per goal within an episode with two standard deviation error bars.

Baselines We compare GOAT to three baselines: 1. CLIP on Wheels [18] - the existing work that comes closest to being able to address the GOAT problem setting - which keeps track of all images the robot has ever seen and, when given a new goal object, decides whether the robot has already seen it by matching CLIP [45] features of the goal image or language description with CLIP features of all images in memory, 2. GOAT w/o Instances, an ablation that treats

- Table 1: Navigation Performance in Unseen Natural Home Environments. We compare GOAT to three baselines in 9 unseen homes with 10 episodes per home consisting of 5-10 image, language, or category goal object instances in terms of success rate and SPL [2], a measure of path efficiency, per goal instance.

SR per Goal SPL Per Goal

Image Language Category Average Image Language Category Average

GOAT 86.4 ± 1.1 68.2 ± 1.5 94.3 ± 0.8 83.0 ± 0.7 0.679 ± 0.013 0.511 ± 0.014 0.737 ± 0.010 0.642 ± 0.007 CLIP on Wheels 46.1 ± 1.8 40.8 ± 1.9 65.3 ± 1.5 50.7 ± 1.0 0.368 ± 0.014 0.317 ± 0.013 0.569 ± 0.015 0.418 ± 0.008

GOAT w/o Instances 28.6 ± 1.7 27.6 ± 1.6 94.1 ± 0.8 49.4 ± 0.8 0.219 ± 0.013 0.222 ± 0.012 0.739 ± 0.011 0.398 ± 0.007 GOAT w/o Memory 59.4 ± 1.5 45.3 ± 1.6 76.4 ± 1.3 60.3 ± 0.8 0.193 ± 0.020 0.134 ± 0.022 0.239 ± 0.021 0.188 ± 0.012

GOAT COW GOAT w/o Memory

GOAT COW GOAT w/o Memory

- 1.0

1.0

0.9

0.8

0.8

SuccessRate

0.7

0.6

SPL

0.6

0.4

0.5

0.4

0.2

0.3

0.0

1 2 3 4 5-10

1 2 3 4 5-10

Number of sequential goals

Number of sequential goals

- Figure 3: Navigation performance based on sequential goal count. GOAT performance improves with experience in the environment: from a 60% success rate (0.2 SPL) at the first goal to 90% (0.8 SPL) for goals 5-10 after thorough exploration. Conversely, GOAT without memory shows no improvement from experience, while COW benefits but plateaus at much lower performance.

all goals as object categories, i.e., always navigating to the closest object of the correct category instead of distinguishing between different instances of the same category as in [19], allowing us to quantify the benefits of GOAT’s instance awareness, and 3. GOAT w/o Memory, an ablation that resets the semantic map and Object Instance Memory after every goal, allowing us to quantify the benefits of GOAT’s lifelong memory.

Quantitative Results Table 1 reports metrics for each method aggregated over the 90 episodes.

GOAT achieves 83% average success rate (94% for object categories, 86% for image goals, and 68% for language goals). We observed that localizing language goals is harder than image goals (detailed in the Discussions section). CLIP on Wheels [18] attains a 51% success rate, showing that using GOAT’s Object Instance Memory for goal matching is more effective than CLIP feature matching against all previously viewed images. GOAT w/o Instances achieves 49% success rate, with 29% and 28% success rates for image and language goals respectively. This demonstrates the need to keep track of enough information in memory to be able to dis-

[Figure 3]

- Figure 4: Online evaluation qualitative trajectories. We compare methods on the same sequence of
- 5 goals (top) in the same environment. GOAT localizes all goals and navigates efficiently (with an SPL of 0.78). CLIP on Wheels localizes only 1 out of 5 goals, illustrating the superiority of GOAT’s Object Instance Memory for matching. GOAT without memory is able to localize 4 our of 5 goals, but with an SPL of only 0.40 as it has to re-explore the environment with every goal. See Section 2.1 for details.

tinguish between different object instances, which [19] wasn’t able to do. GOAT w/o memory achieves 61% success rate with an SPL of only 0.19 compared to the 0.64 of GOAT. It has to re-explore the environment with every goal, explaining the low SPL and low success rate due to many time-outs. This demonstrates the need to keep track of a lifelong memory. Figure 3 further emphasizes this point: GOAT performance improves with experience in the environment from a 60% success rate (0.20 SPL) at the first goal to 90% (0.80 SPL) for goals 5-10 after thorough exploration. Conversely, GOAT without memory shows no improvement from experience, while COW benefits but plateaus at much lower performance.

Qualitative Results We visualize representative trajectories in Figure 4. Here we show the performance of GOAT, CLIP on wheels, and GOAT without Memory on the same sequence of 5 goals, from the same initialization point. When matching image or language goals CLIP on Wheels computes features of the entire observed frame. This makes the matching threshold hard to tune, leading to more false positives (matches the wrong bed for task 1) and false negatives (misses the correct plant for task 2, eventually matching the incorrect plant). Without memory, the GOAT agent will continue to re-explore previously seen regions (tasks 3 and 4 reexplore previously explored already). Additionally, matching performance is worse because the agent forgets previously observed instances. Matching performance improves as more of the environment is explored and mapped because the effect of the matching threshold is diminished (see Section 5.1 for details). The full GOAT system can handle these issues. GOAT is able to match all instances and efficiently navigate to them correctly.

#### 2.3 Applications

As a general navigation primitive, the GOAT policy can readily be applied to downstream tasks such as pick and place and social navigation.

Open Vocabulary Mobile Manipulation The ability to perform rearrangement tasks is essential in any deployment scenarios for mobile robots (homes, warehouses, factories) [4, 61, 13, 28, 20]. These are commands such as “pick up my coffee mug from the coffee table and bring it to the sink,” requiring the agent to search for and navigate to an object, pick it up, search for and navigate to a receptacle, and place the object on the receptacle. The GOAT navigation policy can easily be combined with pick and place skills (we use built-in skills from Boston Dynamics) to fulfill such requests. We evaluate this ability on 30 such queries with image/language/category objects and receptacles across 3 different homes. GOAT can find objects and receptacles with 79% and 87% success rates, respectively.

We visualize one such trajectory in Figure 5 (A). The agent is tasked with first finding a bed, finding a specific toy, and then moving that toy to the bed. We see that while exploring for the bed, the agent observes the toy and keeps it in the instance memory. Consequently, after finding the bed, the agent is able to directly navigate back to the toy (column 2), then efficiently pick it up, and move it back to the bed (column 5).

[Figure 4]

##### Figure 5: A - Application: Rearrangement. The GOAT policy searches for then picks up a toy and places it on the bed. B - Application: Social Navigation. The GOAT policy finds a refrigerator while avoiding a person, then follows a person.

Social Navigation To operate in human environments, mobile robots need the ability to treat people as dynamic obstacles, plan around them, and search for and follow people [39, 44]. To give the GOAT policy such skills, we treat people as image object instances with the PERSON category. This enables GOAT to deal with multiple people, just like it can deal with multiple instances of any object category. GOAT can then remove someone’s previous location from the map after they have moved. To evaluate the ability to treat people as dynamic obstacles, we introduce moving people in 5 trajectories, otherwise following the same experimental setting as our main experiments. GOAT preserves an 81% success rate. We further evaluate the ability of GOAT to search for and follow people by introducing such goals in 5 additional trajectories. GOAT can localize and follow people with 83% success, close to the 86% success rate for static image instance goals.

We visualize a qualitative example of a trajectory in Figure 5 (B). Here the agent must navigate to the refrigerator and then follow the human. We see that the agent recognizes the refrigerator (column 1), but the route there is blocked by the human so the agent must plan around (column 2). After reaching the refrigerator, the agent begins following the human while constantly updating the map based on new sensor observations. This allows the agent to move through space that had been previously marked as occupied by the person (column 4). The navigation target continues to track the person as they move around the apartment (column 5).

### 3 Discussion

Modularity allows GOAT to Achieve Robust General-Purpose Navigation in the Real World The GOAT system as a whole is a robust navigation platform, achieving a success rate of 83% across image, language, and category goals in the wild (up to 90% once the environment is fully explored). This is possible in-part due to the modular nature of the system. A modular system allows learning to be applied in the components in which it is required (i.e. object detection, image/language matching), while still leveraging strong classical methods (i.e. mapping and planning). Furthermore, for learning-based components, we can use models trained on large datasets (i.e. CLIP, MaskRCNN), or specialized tasks (monocular depth estimation) to full effect, where a task-specific end-to-end learned approach would be limited by the available data for this specific task. GOAT is able to tie all of these components together using our Object Instance memory to achieve state-of-the-art performance for lifelong real-world navigation.

Furthermore, the modular design of GOAT allows it to be easily adapted to different robot embodiments and a variety of downstream applications. GOAT can be deployed on any robot with an RGB-D camera, a pose sensor (onboard SLAM), and the ability to execute low-level locomotion commands (move forward, turn left, turn right). GOAT’s modularity eliminates the need for new data collection or training when deployed on a new robot platform. This stands in contrast to end-to-end methods, which would require new data collection and retraining for every different embodiment.

[Figure 5]

- Figure 6: Qualitative examples of trends observed in matching. (A) Matching with a threshold during exploration can result in false negatives, which would be correct matches postexploration. (B) Image-image SuperGLUE is more reliable than image-language CLIP matching. (C) Matching within a category performs better than matching across categories.

Matching Performance During Exploration Lags Behind Performance After Exploration Using a predefined threshold for a successful goal to object matching score during exploration (on the fly) as goals is tricky because an inflexible threshold can cause true positives to be ignored (Figure 6-A) and false positives to be counted in. On the other hand, once the scene has been explored, the agent has the privilege of selecting the instance with the best matching score as the goal. This is reflected in improved performance of the agent post-exploration (6% higher success rate). Refer to Section 5.1 Table 2 in the Supplementary Materials for details.

Image Goal Matching is More Reliable than Language Goal Matching We observe that image-to-image goal matching is more successful at identifying goal instances as compared to matching instance views with semantic features of language descriptions of the goal. This is expected because SuperGLUE-based image keypoint matching can leverage correspondences in geometric properties between predicted instances and goal objects. However, the semantic feature encodings from CLIP can be incapable of capturing fine-grained instance properties – that can often be crucial for goal matching (see examples in Figure 6-B). As a result, matching instance views with image goals is 23% more successful than matching with language description features.

Goal Matching Improves by Subsampling Instances by Category and Adding Context When sifting through seen instances to find a match with the goal, the agent can either compare against all instances seen so far, or do this comparison only against instances that belong to the goal category. We observe that filtering out non-goal categories improves matching accuracy by 23% – preventing false positives from getting matched (Figure 6-C). Moreover, this is computationally also better – as comparing only against a subset of instances is also faster and more efficient. Additionally, regardless of whether we use SuperGLUE or CLIP for matching instances to goals, we observe that providing more context about the instance’s background – using wider, enlarged bounding boxes – results in improved matching accuracy (up to 22% more successful than matching bounding boxes alone).

Real-World Open-Vocabulary Detection: Limitations and Opportunities An interesting and noteworthy observation is that despite the rapid advances in open (or large) vocabulary vision-and-language models (VLMs) [37, 42], we find their performance to be significantly worse than a Mask RCNN model from 2017. We attribute this observation to two possible hypotheses: (i) open-vocabulary models trade-off robustness for being more versatile, and supporting more queries, and (ii) the internet-scale weakly labeled data sources used to train modern VLMs under-represent the kind of embodied interaction data that would benefit robots occupying real-world environments with humans. The latter represents a challenging opportunity to develop such large-scale models that are simultaneously versatile and robust for embodied applications in real-world environments.

### 4 Materials and Methods

#### 4.1 Go To Any Thing System Architecture

System Overview Figure ?? (A) shows an overview of the GOAT system. The perception system detects object instances, localizes them in a top-down semantic map of the scene, and stores images in which each instance has been viewed in an Object Instance Memory. When a new goal is specified, a global policy first tries to localize it within the Object Instance Memory. If no instance is localized, the global policy outputs an exploration goal. A local policy finally computes actions towards the long-term goal.

Perception: Figure 7 (B) shows perception system. It takes as input the current depth image Dt, RGB image It, and pose reading xt from onboard sensors. It uses an off-the-shelf model to segment instances in the RGB image. We use MaskRCNN [23] with a ResNet50 [24] backbone pretrained on MS-COCO for object detection and instance segmentation. We chose MaskRCNN as current open-set models, such as Detic [62], were less reliable for common categories in early experiments. We also estimate depth to fill in holes for reflective objects in raw sensor readings.

To fill holes in the depth image we first use a monocular depth estimation model to obtain a dense depth estimation from It (we used the MiDaS [47] model, although any depth estimation model would be applicable). Since depth estimation models typically predict relative distances instead of absolute distances, we ground the predicted depth using the known true depth values from Dt. Specifically, we solve for the scale factor that minimizes the mean squared error between estimated depth and sensed depth across all pixels where there is a depth reading.

arg min

A,b i

∥Dt,i − AXt,i − b∥2

Where Dt,i is the ith depth reading from the depth reading Dt and Xt,i is the depth estimate at the same point. This optimization can be easily solved as a system of equations yielding a scale factor and offset to project estimated depth points into absolute distances. We use these depth estimates for pixels in the depth image for which no reading was registered.

Using the dense depth computed above, we project the first-person semantic segmentation into a point cloud, bin the point cloud into a 3D semantic voxel map, and finally sum over the height to compute a 2D instance map mt. For each detected object instance, we also store the image in which the object was detected as part of the object instance memory.

Semantic Map Representation: The semantic map (mt at timestep t) is a spatial representation of the environment that keeps track of object instance locations, obstacles, and explored areas. Concretely, it is a K ×M ×M matrix of integers where M ×M is the map size, and K is the number of map channels. Each cell of this spatial map corresponds to 25cm2 (5cm × 5cm)

[Figure 6]

- Figure 7: (A) GOAT system overview. The perception system detects and localizes object instances, the global policy outputs high-level navigation commands depending on whether the robot should explore or reach a goal already in memory, and the local policy executes these commands. (B) Perception and memory update. The perception system processes RGB-D input to infill depth, segment object instances, project them into a top-down semantic map, and store views in the Object Instance Memory.

[Figure 7]

- Figure 8: (A) Object Instance Memory. We cluster object detections, along with image views in which they were observed, into instances using their location in the semantic map and their category. (B) Global Policy. When a new goal is specified, the global policy first tries to localize it within the Object Instance Memory. If no instance is localized, it outputs an exploration goal.

in the physical world. Map channels K = C + 4 where C is the number of semantic object categories, and the remaining 4 channels represent the obstacles, the explored area, and the agent’s current and past locations. An entry in the map is non-zero if the cell contains an object of a particular semantic category, an obstacle, or is explored, depending on the channel, and zero otherwise. In this semantic map representation, the first C channels store the unique instance ids of the projected objects. The map is initialized with all zeros at the beginning of an episode, and the agent starts at the center of the map facing east.

Object Instance Memory Figure 8 (A) shows the Object Instance Memory (ot at timestep t). Our object instance memory clusters object detections across time into instances using their location in the map and their category.

Each object instance i is stored as a set of cells in the map Ci, a set of object views represented as bounding boxes with context Mi, and an integer indicating the semantic category Si. For each incoming RGB image I, we detect objects. For each detection d we use the bounding box around the detection Id, the semantic class Sd, and the corresponding points in the map Cd based on projected depth. We dilate each instance on the map Cd by p units to obtain a dilated set of points Dd per instance, which is used for matching to instances that were previously added in the memory and map. We check for matches pairwise between each detection and each existing object instance. A detection d and instance i are considered to match if the semantic category is the same, and any projected locations in the map overlap, Sd = Si and Dd ∩ Ci ̸= ∅. If there is a match, we update the existing instance with the new points and new image

Ci ← Ci ∪ Cd,Mi ← Mi ∪ {Id} Otherwise, a new instance is added using Cd and Id.

This procedure aggregates unique object instances over time, allowing new goals to be matched against all images of specific instances or categories easily.

Global Policy Figure 8 (B) shows the global policy. When a new goal is specified to the agent, the global policy πG first searches the Object Instance Memory to see if the goal has already been observed. The method for computing matches is tailored to the modality of the goal specification. For category goals, we simply check whether any object of the category is in the semantic map. For language goals, we first extract an object category from the language description (by prompting with Mistral 7B [30] in our experiments), then match CLIP features of the language description with CLIP features of each object instance of the inferred category in our Object Instance Memory. Similarly, for image goals, we first extract an object category from the image with MaskRCNN, then match keypoints of the goal image with keypoints of each object instance of the inferred category with an off-the-shelf SuperGlue model. While the environment is being explored, we consider the object instance matches the goal if the matching score is above some threshold (0.28 for CLIP, 6.0 for Superglue), while when the environment is fully explored, we select the object instance with the highest similarity score. After an instance

is selected, its stored location in the top-down map is used as a long-term point navigation goal. If no instance is localized, the global policy outputs an exploration goal. We use frontier-based exploration [60], which selects the closest unexplored region as the goal.

Local Policy Given a long-term goal output by the global policy πG, the local policy πL uses the Fast Marching Method to plan a path towards it. On the Spot robot, we use the built-in point navigation controller to reach waypoints along this path. On the Stretch robot with no such built-in controller, we plan the first low-level action along this path deterministically as in [19].

#### 4.2 Experimental Methodology

Hardware Platforms The GOAT navigation policy is platform agnostic: no component of our system is tied to any particular robot hardware. We deployed GOAT on and conducted qualitative experiments with Boston Dynamics Spot and Hello Robot Stretch robots. We conducted large-scale quantitative experiments with GOAT on Spot (due to its higher reliability) against 3 baselines in 9 real-world homes to reach a total of 200+ different object instances (see Figure 2).

Navigation Performance in Unseen Natural Home Environments We evaluate GOAT “in the wild” in 9 unseen rented homes without pre-computed maps or locations of objects. We evaluate each method for 10 trajectories per home with 5-10 goals per trajectory for a total of 90 hours of experiments. We selected goals across 15 different object categories (”chair”, ”couch”, ”potted plant”, ”bed”, ”toilet”, ”tv”, ”dining table”, ”oven”, ”sink”, ”refrigerator”, ”book”, ”vase”, ”cup”, ”bottle”, ”teddy bear”), took a picture for image goals following the protocol in [32], and annotated 3 different language descriptions uniquely identifying the object. To generate an episode within a home, we sampled a random sequence of 5-10 goals split equally among language, image, and category goals among all object instances available.

Metrics We report metrics per goal within an episode, as compound metrics over the entire trajectory are not meaningful. We consider navigation to a goal within an episode successful if the robot called the stop action close enough (less than 1 meter) to the correct instance of the goal category within a reasonable time budget (200 robot steps). To compute the Success weighted by Path Length (SPL) [2] per goal, we measure the geodesic distance to the goal instance closest to the previous goal instance.

### 5 Supplementary

#### 5.1 Offline Comparison of Instance Matching Strategies

In this section, we compare design choices for the matching module of the global policy, whose role is to identify the goal object instance among previously seen object instances. This module is particularly important as it determines the form of the Object Instance Memory and allows the GOAT agent to perform lifelong learning for multimodal navigation. Recall that our matching module uses CLIP for matching language goals and SuperGLUE for matching image goals. We first filter instances by target category and use a cropped version of each instance view by including some context around the object. We then aggregate the scores across views via a ”max” operation. During exploration, we use a threshold of 3.0 for image-image SuperGLUE matching and a threshold of 0.75 for language-image CLIP matching. Post-exploration, we pick the best matching instance without using any threshold.

We manually annotated 3 trajectories per home with ground-truth object instances corresponding to each goal, for a total of 27 trajectories. This enables us to evaluate the effect of different design choices on the matching success rate: the percentage of goals that get correctly matched. Table 2 presents ablations for the following design choices:

- • Matching method: Storing raw image views in our Object Instance Memory lets us use different matching methods per goal modality. We match language goal descriptions with object views in memory using the cosine similarity score between their CLIP [45] features. On the other hand, to match image goals with object views in memory, we evaluate both CLIP feature matching and keypoint-based matching with SuperGLUE [49].
- • Matching threshold: The threshold for a successful matching score. We show results for a fixed non-zero threshold (the best hyper-parameter) and a zero threshold. We use the former when the agent is still exploring the scene because it has to decide whether the instance in the current observation matches the goal or to keep exploring, and the latter when the agent has already explored the entire scene and always navigates to the best match. Note that we assume that a match always exists. Hence when the agent has fully explored the environment, we expect the best match to be correct assuming the agent detects the object.
- • Instance sub-sampling: Whether to compare the goal with views of all instances captured so far or only against instances of the goal’s category. Intuitively, the latter is faster with higher precision but potentially lower recall, as it relies on accurate object detection.
- • Context: The instance view context to use when matching: (i) only the bounding box crop of the detected instance (‘bbox’), (ii) add some of the surrounding context (‘bbox+pad’), or (iii) the full image in which the instance is seen (‘full image’).

| | |Threshold<br><br>|Sub-sampling<br><br>|Context|Max Median Avg Avg Top-2|
|---|---|---|---|---|---|
|Image-to-image|SuperGLUE<br><br>|Exploration threshold = 3.0<br><br>|All images|Bbox|48.4 25.8 41.9 46.8 72.6 56.5 62.9 67.7 58.1 40.3 40.3 53.2 51.6 25.8 43.5 48.4 92.1 83.0 85.1 89.4 91.9 80.6 85.5 90.3<br><br>|
| | | | |Bbox + pad| |
| | | | |Full image| |
| | | |By category|Bbox<br><br>| |
| | | | |Bbox + pad| |
| | | | |Full image| |
| | |No threshold<br><br>|All images|Bbox<br><br>|50.0 38.7 50.0 51.6 72.6 58.1 64.5 67.7 58.1 40.3 41.9 53.2 64.5 43.5 64.5 62.9 95.2 85.1 89.4 93.6 91.9 85.5 88.7 91.9<br><br>|
| | | | |Bbox + pad<br><br>| |
| | | | |Full image<br><br>| |
| | | |By category<br><br>|Bbox<br><br>| |
| | | | |Bbox + pad| |
| | | | |Full image<br><br>| |
| |CLIP<br><br>|Exploration threshold = 0.75|All images|Bbox<br><br>|40.3 17.7 25.8 38.7 48.4 33.9 29.0 45.2 46.8 30.6 32.3 50.0 53.2 22.6 25.8 46.8 79.0 64.5 72.6 74.2 79.0 56.5 59.7 74.2<br><br>|
| | | | |Bbox + pad<br><br>| |
| | | | |Full image<br><br>| |
| | | |By category|Bbox<br><br>| |
| | | | |Bbox + pad| |
| | | | |Full image| |
| | |No threshold<br><br>|All images<br><br>|Bbox|40.3 22.6 29.0 38.7 48.4 35.5 30.6 46.8 46.8 32.3 33.9 50.0 66.1 56.5 61.3 66.1 82.5 67.7 75.8 75.8 82.3 71.0 72.6 79.0<br><br>|
| | | | |Bbox + pad| |
| | | | |Full image| |
| | | |By category|Bbox<br><br>| |
| | | | |Bbox + pad| |
| | | | |Full image| |
|Language-to-image<br><br>|CLIP|Exploration threshold = 0.26|All images<br><br>|Bbox|54.8 33.9 25.8 53.2 48.4 24.2 22.6 45.2 38.7 17.7 22.6 38.7<br><br>58.1 35.5 27.4 59.7 69.4 48.4 51.6 62.9<br><br>59.7 48.4 45.2 53.2<br>|
| | | | |Bbox + pad<br><br>| |
| | | | |Full image<br><br>| |
| | | |By category<br><br>|Bbox<br><br>| |
| | | | |Bbox + pad| |
| | | | |Full image<br><br>| |
| | |No threshold<br><br>|All images<br><br>|Bbox|58.1 41.9 35.5 54.8 51.6 30.6 29.0 50.0 43.5 24.2 30.6 48.4<br><br>71.0 62.9 58.1 72.6<br><br>72.6 66.1 69.4 71.0<br><br><br>69.4 67.7 67.7 66.1|
| | | | |Bbox + pad| |
| | | | |Full image| |
| | | |By category<br><br>|Bbox| |
| | | | |Bbox + pad| |
| | | | |Full image| |

SuperGLUE

- Table 2: Offline comparison of instance matching strategies. We compare different design choices for the goal-to-instance matching module of the global policy on 27 trajectories manually annotated with ground-truth object instances corresponding to each goal. We report the matching success rate (higher the better), which is the percentage of goals that get correctly matched. The best entries for each matching method and matching threshold are highlighted in

lime , and the parameters used in online experiments are highlighted in green .

• Best match selection criteria: When comparing multiple views of multiple instances with one goal, we can select the best match through: (i) max: choosing the instance with the maximum matching score (from any one view), (ii) median: the highest median matching score (across all views), (iii) the highest average matching score (across all views), and (iv) the highest average score across the top-k views.

Looking at the image-to-image matching section of Table 2, we can see that:

- 1. SuperGLUE-based image keypoint matching is much more reliable than CLIP feature matching — on average 13% more successful. This helps explain the superior performance of GOAT over COW [18], which uses CLIP feature matching.
- 2. Introducing a matching threshold to ignore low confidence has a cost — on average 6% worse than with no threshold. As we’ll see in the Discussion, this means matching is more challenging during exploration than once the environment is fully explored.
- 3. Sub-sampling instances based on goal category works better than sifting through all instances — on average 23% more successful. This helps explain the superior performance of GOAT over COW, which doesn’t subsample instances by category.
- 4. Matching padded (enlarged) bounding box of instance views works best — on average 4.6% better than using the full image (the second best approach) and 22% better than using the object’s bounding box across all settings.
- 5. Matching the maximum matching score across all views of all instances works better than median, average, and top-2 average — on average 2% to 16% better across all settings.

Similar trends can be observed in language-to-image matching. However, image-to-image matching (using SuperGLUE-based keypoint matching) is much more reliable than (CLIPbased) language-to-image matching — on average 23% better across all settings.

[Figure 8]

##### Figure 9: Example goal object instances. Image and language descriptions used to uniquely identify target object instances.

[Figure 9]

##### Figure 10: More example goal object instances. Image and language descriptions used to uniquely identify target object instances.

Changing one parameter at a time

Best

Agg-fn= Avg Top-2

Threshold= exploration

Img-width= bbox

Sampling= All images

Img-width= full

Goal setting

[Figure 10]

No matches

No matches

- Figure 11: Matching strategy ablation for selecting language goals. In the second column, we show the language goals identified by our best setting (Img-width=Bbox+pad, Agg-fn=max, Sampling=By Category, Threshold=0.0). In the subsequent columns, we show results after changing one parameter at a time with respect to the best setting. The object instance selected in each setting is highlighted with a border which is green for correct matches and red for incorrect matches.

Changing one parameter at a time

Best

CLIP matching

Threshold= exploration

Img-width= bbox

Sampling= All images

Img-width= full

Goal setting

[Figure 11]

[Figure 12]

###### No matches

- Figure 12: Matching strategy ablation for selecting image goals. In the second column, we show the image goals identified by our best setting (Img-width=Bbox+pad, Agg-fn=max, Sampling=By Category, Threshold=0.0, Method=SuperGLUE). In the subsequent columns, we show results after changing one parameter at a time with respect to the best setting. The object instance selected in each setting is highlighted with a border which is green for correct matches and red for incorrect matches.

### References

- [1] Ziad Al-Halah, Santhosh Kumar Ramakrishnan, and Kristen Grauman. Zero experience required: Plug & play modular transfer learning for semantic visual navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17031–17041, 2022.
- [2] Peter Anderson, Angel Chang, Devendra Singh Chaplot, Alexey Dosovitskiy, Saurabh Gupta, Vladlen Koltun, Jana Kosecka, Jitendra Malik, Roozbeh Mottaghi, Manolis Savva, et al. On evaluation of embodied navigation agents. arXiv preprint arXiv:1807.06757, 2018.
- [3] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko S¨underhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3674–3683, 2018.
- [4] Dhruv Batra, Angel X Chang, Sonia Chernova, Andrew J Davison, Jia Deng, Vladlen Koltun, Sergey Levine, Jitendra Malik, Igor Mordatch, Roozbeh Mottaghi, et al. Rearrangement: A challenge for embodied ai. arXiv preprint arXiv:2011.01975, 2020.
- [5] Benjamin Bolte, Austin Wang, Jimmy Yang, Mustafa Mukadam, Mrinal Kalakrishnan, and Chris Paxton. Usa-net: Unified semantic and affordance representations for robot memory. arXiv preprint arXiv:2304.12164, 2023.
- [6] Matthew Chang, Arjun Gupta, and Saurabh Gupta. Semantic visual navigation by watching youtube videos. In Advances in Neural Information Processing Systems, 2020.
- [7] Devendra Singh Chaplot, Murtaza Dalal, Saurabh Gupta, Jitendra Malik, and Russ R Salakhutdinov. Seal: Self-supervised embodied active learning using exploration and 3d consistency. Advances in neural information processing systems, 34:13086–13098, 2021.
- [8] Devendra Singh Chaplot, Dhiraj Gandhi, Abhinav Gupta, and Ruslan Salakhutdinov. Object goal navigation using goal-oriented semantic exploration. In In Neural Information Processing Systems (NeurIPS), 2020.
- [9] Devendra Singh Chaplot, Dhiraj Gandhi, Saurabh Gupta, Abhinav Gupta, and Ruslan Salakhutdinov. Learning to explore using active neural mapping. In International Conference on Learning Representations, 2020.
- [10] Devendra Singh Chaplot, Ruslan Salakhutdinov, Abhinav Gupta, and Saurabh Gupta. Neural topological slam for visual navigation. In Computer Vision and Pattern Recognition (CVPR), 2020.

- [11] Howie Choset and Keiji Nagatani. Topological simultaneous localization and mapping (slam): toward exact localization without explicit localization. IEEE Transactions on robotics and automation, 17(2):125–137, 2001.
- [12] Thomas S Collett and Matthew Collett. Memory use in insect visual navigation. Nature Reviews Neuroscience, 3(7):542–552, 2002.
- [13] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.
- [14] Alberto Elfes. Using occupancy grids for mobile robot perception and navigation. Computer, 22(6):46–57, 1989.
- [15] Hao Fang*, Saurabh Gupta*, Forrest Iandola*, Rupesh K Srivastava*, Li Deng, Piotr Doll´ar, Jianfeng Gao, Xiaodong He, Margaret Mitchell, John C Platt, C Lawrence Zitnick, and Geoffrey Zweig. From captions to visual concepts and back. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1473–1482, 2015.
- [16] Ali Farhadi, Mohsen Hejrati, Mohammad Amin Sadeghi, Peter Young, Cyrus Rashtchian, Julia Hockenmaier, and David Forsyth. Every picture tells a story: Generating sentences from images. In Computer Vision–ECCV 2010: 11th European Conference on Computer Vision, Heraklion, Crete, Greece, September 5-11, 2010, Proceedings, Part IV 11, pages 15–29. Springer, 2010.
- [17] Andrea Frome, Greg S Corrado, Jon Shlens, Samy Bengio, Jeff Dean, Marc’Aurelio Ranzato, and Tomas Mikolov. Devise: A deep visual-semantic embedding model. Advances in neural information processing systems, 26, 2013.
- [18] Samir Yitzhak Gadre, Mitchell Wortsman, Gabriel Ilharco, Ludwig Schmidt, and Shuran Song. Cows on pasture: Baselines and benchmarks for language-driven zero-shot object navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23171–23181, 2023.
- [19] Theophile Gervet, Soumith Chintala, Dhruv Batra, Jitendra Malik, and Devendra Singh Chaplot. Navigating to objects in the real world. Science Robotics, 8(79):eadf6991, 2023.
- [20] Jiayuan Gu, Devendra Singh Chaplot, Hao Su, and Jitendra Malik. Multi-skill mobile manipulation for object rearrangement. arXiv preprint arXiv:2209.02778, 2022.
- [21] Saurabh Gupta, James Davidson, Sergey Levine, Rahul Sukthankar, and Jitendra Malik. Cognitive mapping and planning for visual navigation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

- [22] Meera Hahn, Devendra Singh Chaplot, Shubham Tulsiani, Mustafa Mukadam, James M Rehg, and Abhinav Gupta. No rl, no simulation: Learning to navigate without navigating. Advances in Neural Information Processing Systems, 34:26661–26673, 2021.
- [23] Kaiming He, Georgia Gkioxari, Piotr Doll´ar, and Ross Girshick. Mask r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 2961–2969, 2017.
- [24] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.
- [25] Evan Hernandez, Sarah Schwettmann, David Bau, Teona Bagashvili, Antonio Torralba, and Jacob Andreas. Natural language descriptions of deep visual features. In International Conference on Learning Representations, 2021.
- [26] Christopher Hilton and Jan Wiener. Route sequence knowledge supports the formation of cognitive maps. Hippocampus, 2023.
- [27] Daniel P Huttenlocher and Shimon Ullman. Recognizing solid objects by alignment with an image. International journal of computer vision, 5(2):195–212, 1990.
- [28] brian ichter, Anthony Brohan, Yevgen Chebotar, Chelsea Finn, Karol Hausman, Alexander Herzog, Daniel Ho, Julian Ibarz, Alex Irpan, Eric Jang, Ryan Julian, Dmitry Kalashnikov, Sergey Levine, Yao Lu, Carolina Parada, Kanishka Rao, Pierre Sermanet, Alexander T Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Mengyuan Yan, Noah Brown, Michael Ahn, Omar Cortes, Nicolas Sievers, Clayton Tan, Sichun Xu, Diego Reyes, Jarek Rettinghouse, Jornell Quiambao, Peter Pastor, Linda Luu, Kuang-Huei Lee, Yuheng Kuang, Sally Jesmonth, Nikhil J. Joshi, Kyle Jeffrey, Rosario Jauregui Ruano, Jasmine Hsu, Keerthana Gopalakrishnan, Byron David, Andy Zeng, and Chuyuan Kelly Fu. Do as i can, not as i say: Grounding language in robotic affordances. In Karen Liu, Dana Kulic, and Jeff Ichnowski, editors, Proceedings of The 6th Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pages 287–318. PMLR, 14–18 Dec 2023.
- [29] Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd Omama, Tao Chen, Shuang Li, Ganesh Iyer, Soroush Saryazdi, Nikhil Keetha, Ayush Tewari, et al. Conceptfusion: Open-set multimodal 3d mapping. arXiv preprint arXiv:2302.07241, 2023.
- [30] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
- [31] Justin Johnson, Andrej Karpathy, and Li Fei-Fei. Densecap: Fully convolutional localization networks for dense captioning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4565–4574, 2016.

- [32] Jacob Krantz, Theophile Gervet, Karmesh Yadav, Austin Wang, Chris Paxton, Roozbeh Mottaghi, Dhruv Batra, Jitendra Malik, Stefan Lee, and Devendra Singh Chaplot. Navigating to objects specified by images. In ICCV, 2023.
- [33] Jacob Krantz, Stefan Lee, Jitendra Malik, Dhruv Batra, and Devendra Singh Chaplot. Instance-specific image goal navigation: Training embodied agents to find object instances. arXiv preprint arXiv:2211.15876, 2022.
- [34] Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Vision-and-language navigation in continuous environments. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVIII 16, pages 104–120. Springer, 2020.
- [35] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017.
- [36] Benjamin Kuipers and Yung-Tai Byun. A robot exploration and mapping strategy based on a semantic hierarchy of spatial representations. Robotics and autonomous systems, 8(1-2):47–63, 1991.
- [37] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Rene Ranftl. Language-driven semantic segmentation. In International Conference on Learning Representations, 2022.
- [38] David G Lowe. Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60:91–110, 2004.
- [39] Matthias Luber, Luciano Spinello, Jens Silva, and Kai O Arras. Socially-aware robot navigation: A learning approach. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 902–907. IEEE, 2012.
- [40] Pierre Marza, Laetitia Matignon, Olivier Simonin, Dhruv Batra, Christian Wolf, and Devendra Singh Chaplot. Autonerf: Training implicit scene representations with autonomous agents. arXiv preprint arXiv:2304.11241, 2023.
- [41] So Yeon Min, Devendra Singh Chaplot, Pradeep Ravikumar, Yonatan Bisk, and Ruslan Salakhutdinov. Film: Following instructions in language with modular methods. arXiv preprint arXiv:2110.07342, 2021.
- [42] Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, et al. Simple open-vocabulary object detection. In European Conference on Computer Vision, pages 728–755. Springer, 2022.

- [43] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE international conference on computer vision, pages 2641–2649, 2015.
- [44] Xavier Puig, Eric Undersander, Andrew Szot, Mikael Dallaire Cote, Tsung-Yen Yang, Ruslan Partsey, Ruta Desai, Alexander William Clegg, Michal Hlavac, So Yeon Min, et al. Habitat 3.0: A co-habitat for humans, avatars and robots. arXiv preprint arXiv:2310.13724, 2023.
- [45] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- [46] Santhosh Kumar Ramakrishnan, Devendra Singh Chaplot, Ziad Al-Halah, Jitendra Malik, and Kristen Grauman. Poni: Potential functions for objectgoal navigation with interactionfree learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18890–18900, 2022.
- [47] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(3), 2022.
- [48] Renato F Salas-Moreno, Richard A Newcombe, Hauke Strasdat, Paul HJ Kelly, and Andrew J Davison. Slam++: Simultaneous localisation and mapping at the level of objects. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1352–1359, 2013.
- [49] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4938–4947, 2020.
- [50] Nikolay Savinov, Alexey Dosovitskiy, and Vladlen Koltun. Semi-parametric topological memory for navigation. arXiv preprint arXiv:1803.00653, 2018.
- [51] Nur Muhammad Mahi Shafiullah, Chris Paxton, Lerrel Pinto, Soumith Chintala, and Arthur Szlam. Clip-fields: Weakly supervised semantic fields for robotic memory. arXiv preprint arXiv:2210.05663, 2022.
- [52] Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Cliport: What and where pathways for robotic manipulation. In Conference on Robot Learning, pages 894–906. PMLR, 2022.

- [53] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10740–10749, 2020.
- [54] Anthony Simeonov, Yilun Du, Andrea Tagliasacchi, Joshua B Tenenbaum, Alberto Rodriguez, Pulkit Agrawal, and Vincent Sitzmann. Neural descriptor fields: Se (3)equivariant object representations for manipulation. In 2022 International Conference on Robotics and Automation (ICRA), pages 6394–6400. IEEE, 2022.
- [55] Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2998–3009, 2023.
- [56] Sebastian Thrun, Maren Bennewitz, Wolfram Burgard, Armin B Cremers, Frank Dellaert, Dieter Fox, Dirk Hahnel, Charles Rosenberg, Nicholas Roy, Jamieson Schulte, et al. Minerva: A second-generation museum tour-guide robot. In Proceedings 1999 IEEE International Conference on Robotics and Automation (Cat. No. 99CH36288C), volume 3. IEEE, 1999.
- [57] Sebastian Thrun, Wolfram Burgard, and Dieter Fox. Probabilistic Robotics (Intelligent Robotics and Autonomous Agents). The MIT Press, 2005.
- [58] Edward C Tolman. Cognitive maps in rats and men. Psychological review, 55(4):189, 1948.
- [59] Saim Wani, Shivansh Patel, Unnat Jain, Angel Chang, and Manolis Savva. Multion: Benchmarking semantic map memory using multi-object navigation. Advances in Neural Information Processing Systems, 33:9700–9712, 2020.
- [60] Brian Yamauchi. Frontier-based exploration using multiple robots. In Proceedings of the second international conference on Autonomous agents, pages 47–53, 1998.
- [61] Sriram Yenamandra, Arun Ramachandran, Karmesh Yadav, Austin Wang, Mukul Khanna, Theophile Gervet, Tsung-Yen Yang, Vidhi Jain, Alexander William Clegg, John Turner, Zsolt Kira, Manolis Savva, Angel Chang, Devendra Singh Chaplot, Dhruv Batra, Roozbeh Mottaghi, Yonatan Bisk, and Chris Paxton. Homerobot: Open vocabulary mobile manipulation. 2023.
- [62] Xingyi Zhou, Rohit Girdhar, Armand Joulin, Philipp Kr¨ahenb¨uhl, and Ishan Misra. Detecting twenty-thousand classes using image-level supervision. In ECCV, 2022.

- [63] Yuke Zhu, Roozbeh Mottaghi, Eric Kolve, Joseph J Lim, Abhinav Gupta, Li Fei-Fei, and Ali Farhadi. Target-driven visual navigation in indoor scenes using deep reinforcement learning. In 2017 IEEE international conference on robotics and automation (ICRA), pages 3357–3364. IEEE, 2017.

