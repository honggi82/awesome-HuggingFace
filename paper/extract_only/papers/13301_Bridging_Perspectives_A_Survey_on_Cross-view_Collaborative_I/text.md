## Bridging Perspectives: A Survey on Cross-view Collaborative Intelligence with Egocentric-Exocentric Vision

Yuping He, Yifei Huang, Guo Chen, Lidong Lu, Baoqi Pei, Jilan Xu, Tong Lu, Yoichi Sato

arXiv:2506.06253v1[cs.CV]6Jun2025

Abstract—Perceiving the world from both egocentric (firstperson) and exocentric (third-person) perspectives is fundamental to human cognition, enabling rich and complementary understanding of dynamic environments. In recent years, allowing the machines to leverage the synergistic potential of these dual perspectives has emerged as a compelling research direction in video understanding. In this survey, we provide a comprehensive review of video understanding from both exocentric and egocentric viewpoints. We begin by highlighting the practical applications of integrating egocentric and exocentric techniques, envisioning their potential collaboration across domains. We then identify key research tasks to realize these applications. Next, we systematically organize and review recent advancements into three main research directions: (1) leveraging egocentric data to enhance exocentric understanding, (2) utilizing exocentric data to improve egocentric analysis, and (3) joint learning frameworks that unify both perspectives. For each direction, we analyze a diverse set of tasks and relevant works. Additionally, we discuss benchmark datasets that support research in both perspectives, evaluating their scope, diversity, and applicability. Finally, we discuss limitations in current works and propose promising future research directions. By synthesizing insights from both perspectives, our goal is to inspire advancements in video understanding and artificial intelligence, bringing machines closer to perceiving the world in a human-like manner. A GitHub repo of related works can be found at https://github.com/ayiyayi/ Awesome-Egocentric-and-Exocentric-Vision.

Index Terms—Video understanding, Egocentric video, Exocentric video, datasets and benchmarks.

I. INTRODUCTION

# P

ERCEIVING the world from both egocentric (firstperson) and exocentric (third-person) perspectives is a

fundamental ability in human intelligence. The mirror neuron theory [1] posits that the same neural mechanisms are activated when an individual performs an action and when they observe another performing the same action. This biological insight underscores the intrinsic connection between first- and thirdperson viewpoints, inspiring efforts to emulate this capability. By enabling machines to integrate and leverage information across these perspectives, we can advance video understanding and move closer to human-like perception.

Y. He, Y. Huang, and G. Chen have equal contributions. Y. He, G. Chen, L. Lu and T. Lu are with the State Key Laboratory for Novel Software Technology, Nanjing University, Nanjing 210023, China. Y. Huang and Y. Sato are with the University of Tokyo, Tokyo, Japan. B. Pei is with Zhejiang University, Zhejiang 310027, China. J. Xu is with Fudan University, Shanghai 200433, China.

[Figure 1]

1642

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

###### EgoInstructor Nymeria Ego-Exo4D OAKINK2 EgoExoLearn Exo2Ego

1600

1400

Numberofcitations

[Figure 8]

[Figure 9]

[Figure 10]

1200

LookCloser Assembly101

H2O

AIDE

CVMHT

Visual-GPS

PEN

Ego-Exo STA-GAN

LOCATE EgoFish3D

1000

Cross-View-AG

800

602

[Figure 11]

[Figure 12]

Actor-

Ego2Top

600

Observer

355

400

178 228

14 39 57 93 127

200

0

2015 2016 2017 2018 2019 2020 2021 2022 2023 2024

Year

Fig. 1. Number of citations to egocentric-exocentric related papers from 2015 to 2024. Citation data was collected from Google Scholar. The statistics are computed based on papers and datasets discussed in Sections IV and V, all of which utilize both egocentric and exocentric perspectives.

The exocentric (third-person) and egocentric (first-person) perspectives offer complementary views of human activity, akin to two sides of the same coin. The egocentric view provides an actor-centered perspective [2], capturing rich humanobject interactions and reflecting the wearer’s intentions and goals [3]–[5]. Unlike the exocentric view, egocentric videos are inherently more dynamic, featuring continuous motion and shifting backgrounds, which pose challenges such as partial visibility of the wearer [6], [7]. Still, the release of large-scale egocentric datasets [3], [4], [8], [9] has spurred substantial progress in egocentric video understanding [10]–[15].

In contrast, the exocentric view offers an observer-like perspective [2], providing a broader context of the scene and the subject’s actions. Different from egocentric videos, these videos are usually recorded from a stable, fixed position, covering a wide field of view and capturing detailed scene context. These videos can be easily captured using devices such as smartphones and surveillance cameras, and their widespread availability on the Internet has led to the creation of diverse large-scale datasets, for example, [16]–[21]. These datasets have driven significant advancements in third-person video understanding [19], [22]–[27].

While egocentric and exocentric perspectives have distinct characteristics, they are inherently complementary [8]. The ego-view provides details from the actor’s perspective, while the exo-view offers a broader contextual understanding of the scene. Researchers can unlock new opportunities to advance video understanding by integrating these perspectives. This synergy has led to a growing body of work exploring crossview learning, as demonstrated in Fig. 1.

Despite these advancements, there remains a lack of surveys that summarize progress in integrating both perspectives. In video understanding, most surveys [28]–[31] focus on specific tasks and primarily concentrate on exocentric videos. In egocentric vision, Plizzari et al. [32] review advancements across multiple tasks. However, to the best of our knowledge, no survey has yet addressed the integration of both perspectives.

Thus, our work fills this gap by systematically organizing and reviewing existing research into three primary directions: (1) leveraging egocentric data to enhance exocentric understanding, (2) utilizing exocentric data to improve egocentric analysis, and (3) joint learning frameworks for cross-view video understanding.

The overall structure of this survey is illustrated in Fig. 2. Inspired by [32], we also adopt a “future-to-present” approach. Specifically, we start by highlighting the transformative potential of integrating egocentric and exocentric perspectives [8], demonstrating how cross-view collaboration can benefit various domains (Section II). We then identify key research tasks to realize these applications (Section III). In addition to the systematic review of existing research works (Section IV), we also analyze benchmark datasets that support both perspectives (Section V), evaluate their diversity and applicability. Finally, we discuss the limitations of current approaches and propose promising research directions (Section VI).

II. APPLICATIONS

In this section, we highlight the practical value of egocentric and exocentric video understanding techniques. We select eight representative application scenarios that have a significant demand for ego-exo collaboration. For each scenario, we provide examples of how egocentric or exocentric techniques are applied in real-world systems. Notably, most current applications are limited to a single perspective. Therefore, we explore how ego-exo collaboration could drive future innovations, as demonstrated in Fig. 3.

- A. Cooking

Vision-based kitchen assistants have recently emerged, with systems like the Samsung Family Hub refrigerator [33] and the June Oven [34] using exocentric cameras for food recognition and task-specific automation. However, these systems are limited in scope and lack holistic cooking support.

In future kitchens, we imagine exocentric cameras will work with head-mounted AR glasses to assist cooking. The headmounted camera will identify ingredients and their freshness, recommend items, and display them in the AR glasses. During cooking, the AR glasses will recognize current steps and display the next step, while overhead cameras will monitor the workspace to prevent accidents. Thus, techniques like ego-exo action recognition and cross-view associations of key steps, can greatly help these kitchen applications.

- B. Sports

Exocentric vision systems currently dominate sports analysis, with applications such as sports tracking systems [35] and referee assistance system [36]. For broadcasting, Fox Sports’ “Be The Player” [37] generates egocentric replays

from exocentric views. However, as wearing cameras can hinder players’ movements, the use of egocentric perspectives and multi-view collaboration remains limited.

In the future, advancements in wearable technology will enable lightweight egocentric devices tailored for athletes. These devices can capture fine-grained details of athletes’ movements. For referees, integrating multi-view video footage can enhance decision-making. Realizing this multi-perspective approach requires techniques like cross-view person identification and tracking for seamless cross-view data alignment.

C. Healthcare

Currently, exocentric cameras are extensively deployed in hospitals, providing real-time observations of patients’ health conditions. Besides, egocentric cameras worn by onsite doctors enable remote assistance [38] and emergency services [39]. However, most current applications rely on a single view and lack multi-view collaboration.

For the future, integrating both views can enhance future medical practices. Remote experts can utilize both the surgeon’s egocentric view and the exocentric recording cameras to give effective guidance. Similarly, remote therapists can benefit from multi-view data to give personalized care plans. These applications necessitate techniques such as ego-exo action assessment and pose estimation.

- D. Education

Nowadays, cameras are widely installed on classroom ceilings to track student movements and enhance safety [40]. Additionally, class recording systems capture lectures, supporting both review sessions and online learning [41]. However, these systems currently operate as passive recording tools, lacking the ability to actively contribute to teaching activities.

Future intelligent classrooms will leverage egocentric and exocentric video collaboration for enhanced learning experiences. During laboratory sessions, egocentric cameras can complement exocentric demonstrations to teach unfamiliar instruments. Besides, transforming exocentric demonstrations into egocentric perspectives enhances intuitive learning. Thus, techniques like ego-exo affordance analysis and cross-view transformation will be key to personalized educational service.

- E. Traffic

Currently, onboard cameras are widely employed in driving assistance [42], [43], and autonomous driving [44] systems. Traffic management systems utilize surveillance cameras to monitor intersections to control traffic signals adaptively. However, data from onboard cameras and surveillance systems often lack coordination, limiting their combined potential.

Future traffic systems will enable information sharing between vehicles and road infrastructure. Onboard cameras will combine with the surveillance network to monitor the driver’s state and enhance scene awareness. This requires techniques such as ego-exo action recognition and cross-view semantic segmentation. Additionally, vehicle footage will be uploaded to the cloud and combined with surveillance footage to optimize traffic management, making cross-view object identification required to track vehicles across videos.

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

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

### Exocentric Egocentric

#### Applications

Sports Education Traffic Embodied

###### Section Ⅱ

###### Public Service

Cooking Healthcare Industry

Intelligence

###### Joint Learning

Egocentric Wearer Identification

Video Generation

Research Tasks

Video Captioning

Cross-View Human Identification

Action Understanding

- Section Ⅳ

Ego for Exo Exo for Ego

Video Generation

View Birdification

View Selection

Video Generation

Video Captioning

Action Understanding

Affordance Grounding

Remote Drone Teleoperation

From Applications to Research Tasks

Section Ⅲ

Datasets

- Section Ⅴ

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Cooking Bike Repair

Basketball Dance

Action Understanding Affordance Grounding

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Inside View

Front View

Left View Right View

Driving

Discussion

- Section Ⅵ Data Model Application

Cross-View Retrieval

Remote Drone Teleoperation

Robot Manipulation

3D Camera Localization

Action Understanding

View Selection

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

...

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

- Task1

- Task2

- Task3

Application

... General

Model

Task

...

- Fig. 2. The overall structure of the survey. We first highlight the application value of egocentric and exocentric collaboration (Section II). We then identify critical research tasks for each application (Section III). Next, we provide a comprehensive overview of the current research advancements (Section IV). This section is divided into: ego for exo, exo for exo, and joint learning, each covering various research tasks. Additionally, we examine datasets that encompass both perspectives (Section V). Finally, we discuss limitations and future directions (Section VI).

- F. Embodied Intelligence

Modern robots leverage both egocentric and exocentric vision for diverse applications, including space exploration, medical assistance, customer service, and security [45]. They can also learn from human demonstrations by mapping exocentric instructional videos onto their own egocentric views for skill acquisition. Looking ahead, multi-agent robotic systems will increasingly depend on cross-perspective collaboration. Estimating egocentric camera positions within a global exocentric frame will enhance coordination, while combining views across robots will enable accurate 3D scene reconstruction for improved situational awareness. These advancements require progress in ego-exo localization, multi-view reconstruction, and collaborative perception.

- G. Public Service

Egocentric and exocentric videos play an essential role in public services. Surveillance cameras aid in locating criminals

and missing persons, while body-worn cameras capture onsite scenes for law enforcement [46]. In search and rescue, aerial drone footage complements ground-level views for timely response [47]. However, these systems typically operate in isolation, limiting their effectiveness. Future urban systems will benefit from integrating egocentric footage with surveillance networks. For instance, in suspect tracking, the police system uses data from the egocentric cameras on the policemen and street surveillance to track suspects and dispatch forces accordingly. To achieve this, cross-view human identification and egocentric wearer identification are essential for associating individuals across multiple perspectives.

H. Industry

In modern manufacturing, ceiling-mounted cameras are widely employed for safety monitoring [48]. On automated assembly lines, cameras on robotic arms help precisely locate

[Figure 55]

[Figure 56]

|Affordance Analysis|
|---|

|Recommend Receipt: Salad|
|---|

|Cooperative Vehicle Infrastucure System|
|---|

[Figure 57]

###### Ego view & Demonstration videos

Steps:

√ Information sharing √ Danger warning √ Road optimization

...

√ √

Tear lettuce into pieces. Cut tomatoes in half. Put them in the bowl. Add salad dressing.

[Figure 58]

[Figure 59]

√ Traffic light planning

graduated cylinder conical flask

...

Next step

Cooking Education Traffic

[Figure 60]

|Referee Assistance|
|---|

|Multi-Agent Collaboration|
|---|

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Ego Exo

&

[Figure 66]

[Figure 67]

Exo View Ego View

###### Ego-Exo Collaboration in Future Applications

Sport

Embodied Intelligence

|Remote Assistance|
|---|

|Suspect Tracking|
|---|

|Remote Assistance|
|---|

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Place component on the base

Ego & Exo View

[Figure 72]

[Figure 73]

Ego & Exo View

Police Officer A: Suspect spotted. System: Dispatch B & C for support.

[Figure 74]

[Figure 75]

Healthcare Public Service Industry

- Fig. 3. Examples of the potential collaboration of egocentric and exocentric vision in diverse applications. We illustrate how integrating egocentric and exocentric video understanding techniques can enhance these applications.

and assemble parts [49]. During quality inspection, multiview scans accurately identify product defects [50]. However, current industrial vision systems operate largely in isolated viewpoints, limiting their ability to provide comprehensive process monitoring.

Future smart factories will integrate wearable and fixed cameras for real-time process optimization and worker support. Overhead cameras can capture the overall workflow, while egocentric cameras track individual worker actions to identify inefficiencies. When operators encounter issues, realtime video streaming from both perspectives can facilitate remote troubleshooting by experts. Enabling these applications will require advancements in cross-view action assessment and multi-view scene understanding.

III. FROM APPLICATIONS TO RESEARCH TASKS

The previous section outlines how egocentric and exocentric perspectives can collaborate to enable a wide range of applications. However, realizing these envisioned applications requires addressing several fundamental research challenges. In this section, we identify key research tasks that demand egocentric-exocentric collaboration and review existing efforts that contribute to their development.

We categorize these tasks from three directions: (1) Exocentric for Egocentric, leveraging exocentric knowledge to enhance egocentric video understanding; (2) Egocentric for Exocentric, utilizing egocentric cues to improve exocentric tasks; and (3) Joint Learning, which integrates both perspectives for cross-view understanding.

Cooking. Analyzing human actions is crucial for providing personalized cooking guidance. Recent research explores how exocentric knowledge can improve egocentric action recognition [51]–[55] and how joint learning of representations from both perspectives enhances overall action modeling [56]. Additionally, transforming exocentric cooking videos into egocentric perspectives has been shown to enhance the immersive

experience [57], [58]. Furthermore, exocentric data also proves beneficial for improving egocentric video captioning [59], [60], facilitating the summarization of cooking procedures.

Sports. Analyzing dynamic actions in sports is critical for skill assessment and injury prevention. Several studies [61], [62] enhance egocentric pose estimation with exocentric data, while others focus on cross-view action recognition [63], [64]. Moreover, transforming exocentric sports video into egocentric viewpoints can provide immersive training experiences, which has been investigated in basketball scenarios [57], [58].

Healthcare. Egocentric perspectives play a crucial role in medical training and remote assistance. Exocentric-toegocentric transformations have been explored for procedural skill acquisition, including COVID testing and CPR [57], [58]. In surgical environments, the doctor’s first-person views can be utilized to select best view for recording system [65]. Additionally, multi-view setups improve pose estimation of surgical instruments [66], facilitating precise tool manipulation.

Traffic. Monitoring driver behavior is a key component of driver monitoring systems to enhance safety. As discussed in [67], [68], both in-vehicle and out-vehicle view are essential to recognize the driver’s condition.

Embodied Intelligence. For robotic manipulation, multiview settings enables precise control [69]–[76]. Additionally, affordance grounding helps robots learn to use tools [77], [78]. Transforming exocentric demonstration videos into the robot’s view facilitates imitation learning [79], [80]. Moreover, the transformed exocentric view can address the limited egocentric view of submersible vehicles [81]. Robots can also act as valuable assistants in human-drone collaboration [82], exocentric camera registration [83] and lifelog video captioning [84].

Industry. Affordance grounding assists robots to use tools. In [78], this task is extended to predict tool-based grasping regions. In technical training, converting exocentric demonstrations to first-person perspectives helps workers visualize procedural steps from their own views. This task has been

Cooking EmbodiedIntelligence Industry

Healthcare

Application Research Task

raffic

Sports

###### ...

T

[57][58] [85]

[79][80]

Video Generation

[57][58] [57][58] [57][58]

- [91] [81]

- [92]

Video Captioning

- [82]

- [83]

- [84]

[59][60]

Action Understanding

[51]-[55]

[66]

[61][62] [63][64]

[67][68]

[80]

[56]

Camara Localization

Affordance Grounding

[77][78]

[78]

Robotic Manipulation

[69]-[76]

View Selection

[183][184] [183][184]

[65]

[183][184]

Remote Drone Teleoperation

###### ...

Ego for Exo

Exo for Ego

Joint Learning

| |
|---|

| |
|---|

- Fig. 4. Mapping relevant research works to applications and research tasks.

explored in bike repair [57], [58] and assembly [85] scenarios.

In summary, existing works demonstrate a promising foundation for exploring egocentric and exocentric collaboration in specific applications. However, despite this progress, current developments remain insufficient to meet the growing demands of real-world deployment. As illustrated in Fig. 4, many tasks critical to applications remain under-investigated. Consequently, the next section presents a detailed review of research tasks and their associated advancements, emphasizing the capabilities and limitations of existing research.

IV. RESEARCH TASKS

The previous section introduces research progress tailored to specific applications. Building on this foundation, this section provides a comprehensive review of advancements in cross-view collaboration with both egocentric and exocentric perspectives. We organize the research directions into three categories, which are defined as follows:

- 1) Exocentric for Egocentric: This direction focuses on leveraging knowledge from the exocentric domain to enhance egocentric video understanding.
- 2) Egocentric for Exocentric: Inversely, this direction emphasizes utilizing knowledge from the egocentric domain to improve exocentric video understanding.
- 3) Joint Learning: This direction aims to integrate egocentric and exocentric perspectives to address cross-view video understanding tasks.

For each direction, we cover various research tasks and review the existing work. An overview is illustrated in Fig. 5.

A. Egocentric for Exocentric

The unique viewpoints of egocentric videos provide rich details that are often missing from exocentric perspectives. This subsection reviews research efforts that leverage egocentric perspectives to enhance exocentric tasks.

Video Generation. Ego-to-exo video generation involves generating an exocentric video from an egocentric one, offering a different perspective of the same environment. It offers significant research value across various fields. For instance, in virtual touring, travelers can review their routes from the third-person perspective to plan their trips effectively.

Video generation has made significant progress in recent years [86]–[88]. However, ego-to-exo video generation poses unique challenges. Egocentric view often includes obscured regions, making it difficult to reconstruct the broader scene of the exocentric perspective. Additionally, maintaining consistency across views is challenging due to their significant disparity. Recent studies in video generation use depth maps [89], poses [86], and other conditional inputs [87], [90] to provide spatialtemporal constraints. However, acquiring such cues in both egocentric and exocentric settings remains difficult.

"Hold a cup" Text Encoder

[Figure 76]

[Figure 77]

[Figure 78]

LatentDiffusionModel

Ego Video

Trajectory Encoder

Cross-view Feature Perception

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Decoder

Initial Exo Frame

Exo Video

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Diffusion

train only

Exo Video Noise

Fig. 6. Illustration of a diffusion-based framework for ego-to-exo video generation, adapted from [91].

For ego-to-exo video generation, IDE [91] introduces a novel framework that leverages human intention to maintain consistency across perspectives. It proposes that human intention is view-independent and can be used to establish connections between views. Specifically, it represents human intention through human movement and action descriptions, which serve as conditional inputs for the diffusion model, as illustrated in Fig. 6. Different from IDE [91], another work [81] investigates this task for underwater vehicles. Although onboard cameras provide a first-person view, this limited perspective restricts the operator’s ability to maneuver in complex underwater environments. To address this, this approach uses past egocentric views and camera poses to create an eye-onthe-back view. This synthesis exocentric views provide broader scene context and enhance operational efficiency.

• Discussion: Despite prior efforts in video generation, ego-toexo video generation remains under-explored, particularly in applications such as robotics and autonomous driving. In these domains, first-person videos (e.g., from onboard cameras) are the primary data source, but their limited view restricts comprehensive scene understanding. In contrast, third-person videos provide broader context, enabling better analysis and decision-making. To realize ego-to-exo video generation in real-world systems, future research must address domainspecific challenges. For example, resource-constrained edge devices cannot support state-of-the-art video generation models, necessitating the development of lightweight architectures. Furthermore, delayed inference in ego-to-exo synthesis could disrupt robotic control or vehicle safety. These challenges highlight the need for real-time processing in future solutions.

Video Generation

[81] [91]

Action Understanding

[92]

View Birdification [93] [94] [95] [65]

Ego for Exo

View Selection

GAN-based Diffusion-based

[79] [101] [102] [103] [57] [58] [80] [85] [59] [60]

Video Generation

Video Captioning

Semantic-based [52] [53] [109] [110] [111] Adversarial-based [61] [112]

Action Understanding

Knowledge Distillation [51] [55] Self-supervised [113] [114]

Exo for Ego

[77] [78] [121][122] [123] [124]

Pair-based Pair-free

Research Task

Affordance Grounding

[127] [132] [131] [133] [134] [84] [137]

VR Technique Addtional Camera

Remote Drone Teleoperation

Video Captioning

Cross-View Retrieval

[2] [101] [142] [143][144][145] [146][147][83]

3D Camera Localization

[2] [56] [62] [63] [66] [67][68] [150] [151] [54] [64] [152] [153] [155] [156] [157] [7] [159] [160] [161] [162] [157] [167] [168] [169] [170] [69] [70] [71] [72] [73][74][75][76]

Pair-based Pair-free

Action Understanding

Graph-based

Egocentric Wearer Identification

Semantic-based

Joint Learning

Cross-View Human Identification

Robotic Manipulation

Remote Drone Teleoperation

[82]

View Selection

[183][184]

- Fig. 5. Overall structure of Section Research Tasks. We discuss the research task from three aspects: Egocentric for Exocentric, Exocentric for Egocentric, and Joint Learning. Each subsection reviews a variety of tasks and their existing works.

Action Understanding. Human action analysis is widely studied with third-person data [16]–[19]. The exocentric perspective captures the full body movements but often misses action details. In contrast, egocentric videos excel at capturing detailed human-object and human-human interactions, which offer a complementary viewpoint to enhance exocentric action understanding.

To leverage complementary egocentric perspectives, Reilly et al. [92] propose a distillation approach, as illustrated in

- Fig. 7. This approach employs projectors to align video features with large language models embeddings, followed by knowledge distillation to transfer egocentric cues into exocentric representations. It highlights the potential of egocentric cues in improving exocentric activity understanding for large vision-language models.

• Discussion: using egocentric perspectives to complement exocentric action analysis is under-explored in fields like industry and surgery. In these domains, performance evaluation is typically conducted via third-person cameras or in-person

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

###### LargeLanguageModel

| | |
|---|---|
| | |

Ego Projector

| | |
|---|---|
| | |

Ego Video

Visual Encoder

Online Distillation

[Figure 91]

The man is grating a carrot.

[Figure 92]

[Figure 93]

Exo Projector

Exo Video

What is the man doing?

Text Encoder

Fig. 7. Illustration of a typical method for ego-for-exo action understanding, adapted from [92]. This method distills egocentric cues into exocentric representations.

monitoring. However, the egocentric perspective can capture more fine-grained details from the actor’s viewpoint. To enable ego-for-exo action analysis, future research should develop lightweight wearable devices that don’t disrupt operations and address issues like motion blur and rapid viewpoint shifts in egocentric videos to improve alignment with exocentric views. View Birdification. This task aims to estimate the trajectories of a crowd from a bird-eye’s view from egocentric videos captured by an observer. It recovers the global movements of

people from the observations of the observer. This task has a wide range of applications such as crowd behavior analysis and surveillance.

Birdified Views

t-2 t-1

t

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

| | |
|---|---|

[Figure 99]

Object Trackor

| | |
|---|---|
| | |

Encoder Decoder

Observer trajectory

Ego Video

Pedestrian trajectory

Queries

Bbox & Velocity

- Fig. 8. Illustration of a typical method for view birdification, adapted from [93]. This task aims to estimate the trajectories of a crowd in a bird’s-eye view from an observer’s egocentric perspective.

In [94], a cascaded optimization based method is proposed to alternate between estimating the displacements of the egocentric camera and its surrounding pedestrians. However, this iterative approach incurs high computational cost. To address this issue, ViewBirdiformer [93] proposes a transformer-based architecture that performs view birdification in a single forward pass. As illustrated in Fig. 8, it first utilizes a multiobject tracking algorithm to extract pedestrian movements, including bounding box coordinates and velocity vectors. These features are then encoded via a transformer encoder to model pedestrian interactions. Subsequently, the transformer decoder leverages camera queries and pedestrian trajectory queries from the previous timestep to predict pedestrian trajectories for the next timestep. In subsequent work, InCrowdFormer [95] addresses uncertainties caused by unknown pedestrian heights and simultaneously predicts pedestrian trajectories along with their associated uncertainty probabilities.

• Discussion: view birdification has promising applications in crowd management and security monitoring. These scenarios mainly rely on fixed surveillance cameras, which are often hindered by limited coverage. In contrast, mobile egocentric cameras can effectively capture blind spots and dynamically track targets. To support on-site applications, future research must address the unique challenges inherent to egocentric videos. For instance, the mobile nature of egocentric cameras introduces issues such as rapid viewpoint changes and environmental transitions (e.g., indoor-to-outdoor shifts). These factors can degrade video quality and hinder trajectory estimation. Future work could integrate video enhancement techniques [96]–[98] to mitigate these challenges.

View Selection. Surgery recordings serve as an essential resource for medical education and surgical assessment. To minimize occlusion and fully capture the surgical field, recording systems often employ multiple cameras mounted in the surgical lump. Therefore, a crucial task is to automatically select the optimal camera view at every moment.

As discussed in [65], the doctor’s perspective is considered the most effective to capture surgical targets. Therefore, this method selects the exocentric camera view that best matches the doctor’s egocentric perspective, as demonstrated in Fig. 9. Future work can leverage sequential information from egocentric videos to reduce frequent camera switching and incorporate other learning algorithms [99], [100].

[Figure 100]

Surgeon's View

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Cam.1 Cam.2

Best View with Less Occlusion

[Figure 105]

Encoder-Decoder

Best Match

Cam.3

- Fig. 9. Illustration of a typical method for view selection in surgical recording, adapted from [65]. It aims to identify the exocentric view with minimal occlusion by using the surgeon’s egocentric perspective as a selection criterion.

B. Exocentric for Egocentric

Exocentric perspectives can complement egocentric analysis by providing a broader view of the environment. Additionally, large-scale exocentric video datasets [16]–[19] has driven significant progress in exocentric video understanding [19], [22]– [24]. Building on these advancements, recent studies have investigated leveraging data and models from the exocentric domain to enhance egocentric analysis. This subsection reviews key approaches that utilize exocentric video techniques to improve egocentric tasks.

Video Generation. Exo-to-ego generation aims to create a first-person view from third-person recordings. This task benefits various fields. For example, in VR and AR applications, exo-to-ego generation can help the users understand procedures by converting third-person videos into their own perspectives. Similarly, the embodied agents can leverage exo-to-ego generation to better understand their surrounding environment.

Current exo-to-ego generation approaches can be categorized into GAN-based [79], [101]–[103] and diffusion-based [57], [58], [80], [85], [104] methods. In [79], [101], exocentric images are used as conditional inputs to GAN for synthesizing egocentric images. Fig. 10 illustrates the general framework of GAN-based approaches. Liu et al. [103] proposes a twoparallel-GANs architecture to transform images from one viewpoint to another. However, these works [79], [101], [103] are limited to image generation. For video generation, STAGAN [102] proposes a bi-directional GAN to learn both spatial and temporal information. However, it relies on semantic maps for guidance to overcome generation ambiguities. More recent work [57], [58], [80], [85] leverages diffusion models. Exo2Ego [85] and Exo2Ego-V [57] focus on synthesizing videos of human activities, while [80] targets robot manipulation scenarios.

Exo Video

Exo Video

Real / Fake Generator

~[0,1] Ego Video Discriminator

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

- Fig. 10. Illustration of the general GAN-based framework for exo-toego video generation. The generator uses exocentric images to synthesize egocentric views, while the discriminator distinguishes between real and synthesized egocentric images.

• Discussion: despite ongoing research efforts, transforming instructor demonstration videos into egocentric views for educational purposes remains under-explored in applications such

- as industrial training, engineering, and surgery. In these fields, egocentric perspectives can provide trainees with immersive experiences and enhance their understanding of complex procedures. First, viewpoint transformation from exocentric to egocentric is inherently ill-posed, as it requires synthesizing visual content that is not directly observed in the source view. This demands robust geometry-aware models capable of inferring occluded or unobserved regions while maintaining spatial coherence. Second, accurately modeling head motion and gaze dynamics is critical for generating realistic egocentric views, yet remains difficult due to the lack of ground truth head-pose trajectories in most instructional videos. Third, current systems struggle with fine-grained temporal alignment, making it difficult to synchronize key actions across views, especially in long, unstructured demonstrations. Finally, achieving semantic consistency—ensuring that important taskrelevant elements (e.g., tools, hands, and object interactions) are preserved and emphasized in the transformed view—is an open challenge, particularly in cluttered or multi-agent scenes. Video Captioning. This task involves generating descriptive textual narratives for videos, aiming to produce coherent sentences that describe the actions, objects, and interactions in the video.

Traditionally, video captioning has been extensively studied in the context of third-person videos [105]–[107], supported by large-scale exocentric video datasets. In contrast, egocentric video captioning has received less attention due to the limited availability of large-scale, high-quality egocentric datasets.

Currently, a promising direction for egocentric video captioning is leveraging large-scale third-person data. To mitigate domain shift, Ohkawa et al. [59] introduce an intermediate ego-like view to gradually adapt from exocentric to egocentric views. On the other hand, EgoInstructor [60] is a retrievalaugmented captioning model that uses semantically relevant exocentric videos as references for egocentric video captioning, as shown in Fig. 11.

• Discussion: Egocentric video captioning has significant potential for assistive devices designed to enhance environmental awareness for visually impaired individuals. In such scenarios, wearable devices, such as smart glasses, can use egocentric video feeds to generate real-time descriptions of user’s surroundings [108], [109]. However, as discussed in [108], [109], the limited field of view of egocentric cameras primarily

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Cross-view Retrieval Model

[Figure 123]

...

###### Exo Reference Videos

Ego Video

Captioning Model

#C C drops the bowl on the counter.

Text Description of Ego Video

Fig. 11. Illustration of a typical method for exocentric for egocentric video captioning, adapted from [60]. This method retrieves relevant exocentric videos to serve as references for captioning egocentric videos.

captures salient foreground objects while often fails to capture broader scene layouts. This constraint impairs users’ ability to reconstruct spatial relationships. A promising approach to addressing this limitation is augmenting egocentric captioning with exocentric 3D spatial data. However, integrating exocentric data into assistive systems should address challenges like translating exocentric 3D layouts into user-centric spatial references (e.g., egocentric distance and orientation) to meet user-specific demands.

Action Understanding. Due to the availability of large-scale exocentric datasets [16]–[19], exocentric action understanding has been extensively studied. Consequently, a body of research explores leveraging knowledge from the exocentric domain to improve understanding of egocentric action.

Semantic-based methods focus on leveraging shared semantics between egocentric and exocentric videos to bridge the gap between the two domains. Existing studies have explored the use of activity sounds [110], geometric correlations [52], skeleton poses [111], and narrations [112] to establish relationships between egocentric and exocentric perspectives. In addition, EMBED [53] utilizes hand-object interactions to transform exocentric video-language datasets into egocentric style.

Adversarial-based methods employ adversarial strategies to minimize the discrepancy between the exocentric domain (source domain) and the egocentric domain (target domain). In [61], [113], a domain classifier is utilized to differentiate whether the feature originates from egocentric or exocentric videos. During training, the model is optimized to generate features to fool the domain classifier, thereby aligning egocentric features with exocentric features. Fig. 12 demonstrates the general adversarial-based framework.

Knowledge distillation methods seek to distill knowledge from exocentric models to improve egocentric action understanding. In [51], [55], the model is first trained on exocentric videos. Subsequently, knowledge distillation losses are applied to adapt the model for egocentric videos.

Self-supervised methods address the challenge of requiring large-scale labeled egocentric data. Egofish3D [114] utilizes 3D poses estimated by an exocentric pose estimator as supervision signals to train an egocentric pose estimator without 3D ground truth annotations. Ex2Eg-MAE [115] first learns to reconstruct exocentric frontal facial videos using synthesized multi-view data that emulate egocentric environments and then evaluates on egocentric social role understanding tasks.

Exo Feature

[Figure 124]

| | | | |
|---|---|---|---|

Classification Head

Action Label

Exo View

Domain Classifier

Shared Weight ~[0,1]

[Figure 125]

| | | | |
|---|---|---|---|

Classification Head

Action Label

Ego Feature

Ego View

Train only

Fig. 12. Illustration of a general adversarial-based approach for exo-forego action understanding. During training, the domain classifier differentiates between egocentric and exocentric features, while the model is optimized to deceive it. During inference, only egocentric videos are used.

• Discussion: existing exo-for-ego frameworks mainly focus on basic tasks such as action recognition [61], [111], [112],

- [116] and pose estimation [114]. However, with the growing demand for advanced applications like skill assessment
- [117] and automated commentary generation [118], [119], we propose expanding the use of exocentric data to tackle more complex challenges. For instance, exocentric expert demonstrations could serve as references to guide egocentric actions and deliver tailored feedback. To advance this in real-world systems, future research should establish skill-level evaluation criteria and improve cross-view action alignment. Affordance Grounding. This task aims to identify and localize the interaction regions of objects based on given instructions. In this task, the exocentric view captures the interactions between human and object while the egocentric view refers to the object only images. Affordance grounding plays a critical role in applications such as embodied intelligence [120], [121], where robots must not only recognize objects but also understand how to interact with them.

Exo-for-Ego affordance grounding methods can be categorized into two types based on training data: pair-based and pair-free. Fig. 13 presents a general framework for this task.

Pair-based method [77], [78], [122]–[125] learn from a group of exocentric images and the corresponding egocentric object image that share the same affordance label. During inference, only the egocentric object image is used. Luo et al. [122] introduce Cross-View-AG based on Class Activation Mapping (CAM) [126], which has served as a foundational paradigm for many subsequent studies. However, CAM is only used in post-processing during inference and lacks effective supervision for the generated affordance map. To address this, LOCATE [123] replaces the vanilla CAM with a learnable module to enable supervision of the CAM-generated map. Furthermore, GAAF-Dex [78] enhances [123] by applying concentration loss to make the affordance map more compact.

With advances in large language models (LLMs), a number of studies [77], [124], [125] integrate language signals into affordance grounding learning. WSMA uses CLIP [127] to encode affordance labels and fuses them with egocentric image embeddings. However, it does not address the issue of action ambiguity, where an object may support multiple actions. To address this limitation, Zhang et al. [77] enable the model to predict both affordance region and object-action descriptions. In contrast to [77], Rai et al. [125] utilize world knowledge from LLMs to generate more detailed captions that include

Exo Demonstration

| |
|---|
| |
| |
| |

Exo Feature

[Figure 126]

Train only

Classification Head

Label

Feature Extraction

Feature Align

| |
|---|
| |
| |
| |

[Figure 127]

[Figure 128]

Classification Head

Ego Feature

Affordance Heatmap

Object Image

CAM / Decoder

Fig. 13. Illustration of the exo-for-ego affordance grounding framework. During training, egocentric object images are aligned with exocentric demonstration images with the same affordance label. During inference, only the egocentric image is inputted to identify the affordance region.

information about object parts and attributes.

Unlike previous work, pair-free methods do not require paired inputs. Instead, they learn from a group of exocentric images and their affordance information. INTRA [128] uses contrastive learning as a weakly supervised objective to extract shared knowledge from different affordance labels. • Discussion: using human demonstration videos to learn robot-centric affordance has not been fully investigated in scenarios like industrial automation. In these domains, egocentric videos from onboard robot cameras are the primary data source. However, human operation videos can guide robots in mastering precise tasks, such as assembly and material handling. While promising, existing research has yet to address the domain-specific challenges. For example, operating precision instruments demands high affordance accuracy, as even minor deviations can lead to operational failures. Due to factors like cross-view object scale discrepancy [129], current methods struggle to effectively transfer affordance regions across views to achieve such precision requirement.

Remote Drone Teleoperation. Drones can navigate challenging environments or locations impassable for humans. It has a wide range of applications such as disaster investigations [130] and product delivery [131]. Typically, drone control systems offer an egocentric view through an on-board camera. However, this limited field of view fails to fully capture the surroundings.

To address the limitations of egocentric views, previous research has explored using VR technique [133] or additional cameras [132], [134], [135] to provide exocentric views. Fig. 14 illustrates using overhead camera to provide exocentric views for drone teleoperation. In [133], VR technique provides a 3D model of the environment, allowing pilots to perceive the drone’s surroundings. Another line of works [132], [134], [135] utilize additional cameras to capture the environment of the drone. StarHopper [132] uses a fixed overhead camera while Temma et al. [134] uses a secondary drone that semi-automatically flies around the primary drone. Inspired by [134], BirdViewAR [135] further uses AR overlays to highlight the primary drone’s spatial status and proposes an automatic framing method to ensure the secondary drone follows the primary drone in fast-moving scenarios.

C. Joint Learning

Joint learning aims to leverage both egocentric and exocentric perspectives to address cross-view video understanding tasks. It requires both egocentric and exocentric views as input during both training and inference. This contrasts with

[Figure 129]

Contr olCommand

Camera Drone Overview Camera

V ideoStr eam

Track in 3D

[Figure 130]

3D Tracking System

Control Software and User Interface

Fig. 14. Illustration of remote drone teleoperation with additional cameras, adapted from [132]. To overcome the limited view of egocentric cameras on drones, exocentric cameras are used to capture the surrounding environment.

unidirectional paradigms (e.g., exo-for-ego or ego-for-exo), where often one view serves as auxiliary information during training, but only a single view is utilized at test time. In joint learning, however, it emphasizes bidirectional collaboration to resolve cross-view tasks. Below, we systematically review advancements in cross-view tasks, highlighting diverse strategies for effectively integrating the complementary nature of egocentric and exocentric perspectives.

generate comprehensive activity logs, enabling assistants to memorize historical events and support downstream tasks like temporal grounding and visual question answering.

Cross-View Retrieval. This task focuses on identifying and retrieving corresponding visual elements, such as videos [146], frames [2], [101], [147], and moments [2], [147], from different viewpoints, as demonstrated in Fig. 16.

Video Captioning. In daily life, video captioning can document a wide range of human activities in natural language. This capability can enhance the development of smart assistants [136]–[138] to help humans memorize and retrieve items.

[Figure 131]

[Figure 132]

| | |
|---|---|
| | |

[Figure 133]

Ego Video

[Figure 134]

[Figure 135]

[Figure 136]

Feature Encoder Unified Feature Representation

Features

Best Match

Exo Videos 1...N

[Figure 137]

Fig. 16. Illustration of the general cross-view retrieval framework. Exocentric and egocentric videos are encoded into a shared representation space to retrieve the best match from the alternate view.

First-person View

[Figure 138]

a man sitting at a desk holding a book

Caption Decoder

Second-person View

Concatenation

Early work [146] explores linear and non-linear mappings to transform motion features between two views. More recent approaches [2], [101] first utilize separate branches to extract features from different views and then employ contrastive learning to align representations. Furthermore, T-JANet [147] leverages overlapping attention regions between views to guide representation learning. However, these works mainly address cross-view correspondence at the video level. Recently, EgoExo4D [8] introduces a cross-view object correspondence task, which aims to predict object masks in one view given queries from another view. PSALM [148] demonstrates zeroshot capability for this task. It first utilizes LLM to process visual and textual prompts, followed by a general segmentation model to generate object masks. Building on PSALM [148], ObjectRelator [149] generates descriptive language prompts for query objects to exploit the LLM’s reasoning ability. To address object appearance disparities across views, ObjectRelator

Text Description of Video

[Figure 139]

Image Encoder

Third-person View

Fig. 15. Illustration of video captioning using videos from first-person, second-person, and third-person perspectives.

Current research [84], [139] investigates captioning lifelog videos in multi-view settings. The logging system comprises a first-person view from an individual, a second-person view from a service robot, and a third-person view from a fixed camera, as demonstrated in Fig. 15. In [84], multi-view images are independently processed into image features, which are then concatenated and projected into a unified feature space. The unified features are subsequently input into a caption decoder to generate captions. In contrast, Nakashima et al. [139] employ attention mechanisms for feature fusion. This method first uses Faster R-CNN [140] to detect salient regions from each view. To address redundant cross-view information, the detected features are clustered into several groups and then fused via attention mechanisms.

- [149] further introduces a cross-view object alignment module to project masks from different views into unified space.

• Discussion: Current approaches primarily learn shared representations across views. However, inherent view disparities lead to significant differences in appearance and motion. These challenges are further exacerbated by occlusions and out-ofview scenarios. Such issues complicate representation alignment. Future work could explore disentangling features into view-invariant and view-specific components [54]. Beyond technical challenges, cross-view retrieval is under-explored in applications like surveillance systems. For instance, retrieving relevant surveillance clips based on egocentric videos from law enforcement agents could enhance event understanding, crime localization, and object tracking. However, retrieving from large-scale data is computationally intensive. Future work should optimize retrieval speed for practical deployment.

3D Camera Localization. This task aims to determine the position and orientation of a camera in the environment.

Han et al. [150] and Qian et al. [151] propose to localize egocentric cameras from a global top-down view. Han et al.

- [150] leverage shadow to relate egocentric and top views and propose a shadow detection model to predict shadow direction, as shown in Fig. 17. Furthermore, Qian et al. [151] utilize

• Discussion: For ego-exo video captioning, several challenges remain for future research. One key issue is balancing description granularity. Due to the different fields of view, egocentric and exocentric videos may emphasize different visual elements. This requires models to reconcile these disparities to generate consistent captions. Additionally, as discussed in [108], [109], users may prefer different levels of detail. Future research should enable model to adjust description granularity to align with user-specific needs. Another challenge is managing redundant and complementary information across views. While prior work [139] addresses this by clustering features

- at frame-level, it overlooks action-level correspondences. For example, an egocentric view might depict “hand pulls a lever”, while an exocentric view captures “doors open”. To generate coherent captions, models must integrate cross-view action dependencies. To achieve this, future work can integrate techniques like action segmentation [55], [141], [142] and action relation [143]–[145]. Beyond technical challenges, joint video captioning holds significant promise for smart assistants [136], [137]. By integrating multiple perspectives, such systems can

the spatial distribution of subjects in the 3D environment to estimate egocentric camera poses in a virtual top-down view. In contrast to [150], [151], YOWO [83] introduces a novel approach to localize ceiling-mounted cameras (CMCs). Previous methods [152], [153] typically use SLAM for scene reconstruction and subsequently employ visual localization to estimate camera poses. However, the perspective disparity between egocentric and exocentric views poses challenge for cross-view localization. Moreover, the static nature of CMCs prevents using motion information to correct localization errors. To address these limitations, YOWO jointly optimizes scene reconstruction and CMC registration. It employs a mobile agent to navigates the environment to generate both agent trajectories and scene layout. Meanwhile, CMCs capture the agent to provide pseudo trajectories. By correlating these trajectories, YOWO aligns CMC poses with the scene layout.

Ego View

[Figure 140]

Geometric Transformation

| | |
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 141]

Shadow Detector

[Figure 142]

Ego Camera Ego Camera View Direction

Top View

- Fig. 17. Illustration of a typical method for egocentric camera localization, adapted from [150]. This method uses shadows to relate egocentric and top views, and estimates the egocentric camera direction in the top view.

Action Understanding. As discussed in [10], [11], models predominantly trained on exocentric videos exhibit poor performance in egocentric data. Cross-view action understanding has emerged as a promising approach to enable a single model to achieve viewpoint-invariant action analysis. This field encompasses multiple key tasks, including action recognition, gaze estimation, and pose estimation, as illustrated in Fig. 18. Current research in this area can be broadly classified into two categories based on training data: pair-based and pair-free.

Feature Encoder Unified Feature Representation

Task Head

[Figure 143]

| | |
|---|---|
| | |

[Figure 144]

[Figure 145]

Ego View

Exo View

Action Label

Action Recognition

Pose Estimation

[Figure 146]

Gaze Estimation

- Fig. 18. Illustration of cross-view action understanding. This involves action recognition, gaze estimation, and pose estimation tasks.

Paired-based methods use synchronized egocentric and exocentric video pairs. For action recognition task, [2], [56], [63], [154] leverage paired videos to learn a unified feature across different views. Soran et al. [56] jointly predict action labels and assess each camera’s importance. In [2], [63], egocentric and exocentric videos are encoded by separate branches and subsequently aligned into a unified feature space. Yonetani et al. [154] use a pair of egocentric videos from two individuals to recognize micro-actions and reactions. In driving scenarios, LBW [67] utilizes both the driver’s face and the forward road scene for gaze estimation. Similarly, Yang et al. [68] integrate in-vehicle and out-vehicle views to recognize the driver’s state. In the field of pose estimation, Ameya et al. [62] map multi-

view RGB frames and optical flow into a joint embedding space, while Hein et al. [66] evaluate multi-view methods [155] for the pose estimation of surgical instruments.

While effective, paired-based approaches are limited by the expense of obtaining synchronized paired data. To address this limitation, recent research [54], [64], [156], [157] has shifted towards leveraging unpaired videos.

Paired-free approaches aim to learn shared action representations from unpaired egocentric and exocentric video data. During inference, pair-free models demonstrate flexibility by accepting either egocentric or exocentric video inputs for action analysis. This line of work can easily utilize existing large-scale third-person and first-person datasets. To align unpaired data, AE2 [64] introduces a temporal alignment strategy. Based on the assumption that aligning egocentric and exocentric videos is inherently easier than aligning them when one sequence is temporally reversed, this approach employs reversed frames as negative samples for contrastive learning. In contrast to AE2 [64], LaGTran [156] leverages language descriptions to mitigate the domain gap between egocentric and exocentric videos. The method is based on the premise that text descriptions exhibit a smaller domain discrepancy compared to the original videos. POV [157] incorporates learnable prompts to video tokens to learn view-agnostic representations. Unlike previous work, Huang et al. [54] highlight the importance of view-specific information and disentangle features into viewinvariant and view-specific components.

• Discussion: Current methods [2], [56], [62], [63], [66]– [68], [154] mainly use paired videos to learn view-invariant representations. However, paired videos still exhibit large discrepancies due to perspective differences. Egocentric videos often suffer from blurring, distortion, and partial visibility, while exocentric videos may depict performers occupying minimal screen space, limiting fine-grained detail capture. These issues hinder shared representation learning. To bridge the disparity, promising solutions include video deblurring [97], [98] for egocentric videos, cropping action performers in exocentric videos [53] , and integrating IMU data [158] to enhance motion information. Furthermore, current research is confined to fundamental tasks like action recognition and pose estimation. Advanced tasks such as action assessment and feedback generation remain unexplored despite their potential in domains like sports. In this domain, integrating both perspectives can offer a holistic understanding of action regularity and proficiency, enabling personalized guidance. To enable practical deployment, a key challenge is effectively integrating dynamic granularity action information across views. Future work should balance between fine-grained hand-object interactions and full-body kinematics to achieve holistic analysis.

Egocentric Wearer Identification. Given both third-person and first-person videos captured in the same environment, this task aims to identify the egocentric camera wearer in thirdperson videos. It is similar to person re-identification across different views, but is more challenging since the camera wearer seldom appears in the egocentric view.

Early researches [159]–[161] employ graph-based techniques to identify the camera holder of egocentric videos in top-view videos. [159] models each video view as a graph

and proposes a spectral graph matching technique. Building on this, [160] extends the work of [159] by considering time delays across videos. Furthermore, [161] employs visual, geometric, and spatiotemporal reasoning to generate candidates and then uses graph cuts [162] to evaluate candidates.

More recent approaches [7], [163]–[166] leverage shared semantic across views. Fan et al. [7] leverage spatial (RGB frames) and temporal (optical flow) similarities to relate two views, as shown in Fig. 19. It employs contrastive learning to predict the camera wearer, utilizing first-person videos paired with third-person videos (masking the correct wearer) as positive samples, and third-person videos (masking a random person) as negative samples. However, this approach primarily focuses on appearance similarity across views, overlooking the dynamic nature of the environment. To address this limitation, Visual-GPS [165] leverages motion and action information to improve robustness, as these features are less sensitive to environmental variations. Subsequent work [166] proposes a more challenging setting: predicting the camera wearer’s location and pose in a third-person scene frame, where the wearer is absent. Furthermore, [163] and [164] jointly address person identification and segmentation and prove that solving these two problems simultaneously is mutually beneficial.

• Discussion: Current appearance-based methods [7], [163], [164] may fail when the wearer is partially visible in the exocentric view. In such cases, even motion cues may struggle if critical body parts are occluded. Furthermore, in crowded scenarios, similar appearances (e.g., shared clothing) or similar actions (e.g., group sports) further hinder discriminative feature extraction. To address these limitations, future work could incorporate additional cues, such as object interactions [167]– [169] or person-person interactions [154], [170], to provide more distinctive information. Beyond technical challenges, egocentric wearer identification remains unexplored in applications like rescue and emergency. In these fields, when critical events are detected in egocentric videos, command centers can locate the wearer in third-person views to dispatch assistance. To enable real-world deployment, future research must address domain-specific challenges. For instance, in large-scale emergencies, systems must distinguish between multiple egocentric

|[Figure 147]|
|---|

| | |
|---|---|
| | |

Ego Frame

| | |
|---|---|
| | |

|[Figure 148]<br><br>|
|---|

|[Figure 149]|
|---|

Spatial Encoder

| | |
|---|---|
| | |

[Figure 150]

Exo Frame (masked)

Exo Frame (masked)

[Figure 151]

[Figure 152]

[Figure 153]

Joint Feature Space

| | |
|---|---|
| | |

Ego Camera Wearer

Ego Flows

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

Temporal Encoder

Exo Flows Exo Flows

- Fig. 19. Illustration of a typical method for egocentric wearer identification, adapted from [7]. This method uses spatial and temporal information to learn view-invariant features and identify the egocentric wearer in exocentric images.

Clip t Clip t+1

Ego View

Cluster 1 Cluster 2

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

| | |
|---|---|

| | |
|---|---|

Detector

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Detector

Cluster 3 Cluster 4

Top View

Clip t Clip t+1

- Fig. 20. Illustration of a typical method for cross-view human tracking and association, adapted from [172]. This method segments video pairs into clips and tracks individuals across clips and views.

[Figure 174]

[Figure 175]

Exo View

| | |
|---|---|
| | |

Feature Encoder

Cross-View Learning

Reinforcement Learning Policy

Robot Control Parameters

Ego View

- Fig. 21. Illustration of a typical framework of multi-view robotic manipulation, adapted from [70]. Multi-view data is integrated via cross-view learning module.

wearers in overlapping exocentric views, requiring multiagent identification algorithms. Additionally, the system must process high-volume, streaming data with minimal latency for quick response, necessitating online processing approaches.

Cross-View Human Identification. This task aims to detect and identify the same individuals across views. Current approaches [161], [171]–[174] study this task on top-view and side-view. The top view, captured by drones at high altitudes, covers large areas and displays human spatial distribution. In contrast, side views from mounted cameras provide more details. Ardeshir et al. [161] propose a graph-based technique while Han et al. [171] use a multi-view human association algorithm to match individuals across different views. However, these works [161], [171] are limited to human identification across views and do not address tracking. Han et al. [172] propose a joint optimization model for identifying and tracking. This approach first segments video pairs into clips and tracks individuals across clips and views, as demonstrated in

- Fig. 20. Additionally, [173] extends this work by incorporating spatial distribution for cross-view association and introducing a new approach for appearance reasoning. Previous approaches [171]–[173] rely on offline detection models [175] to detect human bounding boxes, which may hinder association performance. To address this, Han et al. [174] propose a joint method for cross-view multi-human detection and association. Robotic Manipulation. This task involves controlling robots to interact with objects and perform actions, such as grasping or moving, to achieve specific goals.

Multi-view robot manipulation has been widely studied. However, most approaches simply concatenate multi-view observations at the image level [176] or feature level [177]– [182], without fully exploiting their complementary characteristics. We focus on approaches that explore integrating the complementary strengths of different perspectives.

Lookcloser [70] utilizes cross-view attention mechanisms to integrate egocentric and exocentric perspectives, as shown in

- Fig. 21. In [69], a variational information bottleneck is applied to third-person representations to mitigate their impact on outof-distribution generalization. Acar et al. [72] utilize multi-

view data to train a teacher policy, which then guides a singleview student policy through knowledge distillation. Sharma et al. [73] first use third-person human demonstration videos to generate task goal in robot’s perspective, which are then combined with robot’s current observation to predict actions. Similarly, Shang et al. [71] leverage synchronized first-person and third-person demonstrations to learn viewpoint-agnostic representations and then use third-person demonstrations for policy learning. Both MV-MWM [75] and MFSC [76] introduce multi-view masked reconstruction strategies to learn representations from multi-view observations. Unlike previous approaches, MVD [74] introduces a robust method that supports varying numbers of cameras in inference.

Remote Drone Teleoperation. Traditional drone manipulation primarily focuses on unidirectional collaboration, where humans send commands to control drones. In contrast, joint learning emphasizes bidirectional information exchange, allowing drones to access the human’s perspective for decisionmaking. This enhanced interaction supports a wider range of collaborative tasks. For instance, in a rescue mission, if a human operator identifies a potential victim through a wearable camera, the drone can autonomously navigate to the location to provide assistance. Such bidirectional communication improves operational efficiency.

A notable work in this field is presented in [82]. In this study, point cloud data from the drone and the user’s wearable device are merged into a unified environmental representation, as demonstrated in Fig. 22. Then, this approach provides visualizations of the environment from both the user’s and the drone’s perspectives, ensuring mutual awareness of the surroundings between the user and the drone.

[Figure 176]

Ego Cam. Point Cloud

[Figure 177]

[Figure 178]

Human View of Space 1

Spatial Awareness Module

Transformation

[Figure 179]

Drone Cam. Point Cloud

Merged Maps of Two Space

Drone View of Space 2

Fig. 22. Illustration of a typical method for remote drone teleoperation with human-drone collaboration, adapted from [82]. This method combines user and drone perspectives into a unified environmental representation.

• Discussion: To facilitate bidirectional information exchange between humans and drones, future research should optimize real-time data processing and minimize communication latency. Additionally, enhancing autonomous decision-making in drones based on both human and drone perspectives could further advance collaboration.

View Selection. The task of selecting the optimal viewpoint from multi-view videos has been widely studied. Prior work explores determining the best camera angles and positions in panoramic 360° views [183], [184], and automating viewpoint selection in multi-view systems [185], [186]. However, these methods typically address egocentric or exocentric views separately, ignoring scenarios where both views are available.

Unlike previous work, recent work [187], [188] propose to address view selection in instructional videos, which incorpo-

rate both egocentric and exocentric perspectives. Majumder et al. [187] utilize language descriptions as weak supervision, as shown in Fig. 23. Specifically, the approach generates captions for each view via video captioning models. These captions are scored against ground-truth narration and ranked to produce best-view pseudo-labels, which are utilized to train the view selection model. Another work [188] proposes a pretext task to detect view switches in instructional videos with varying viewpoints. The model trained for this task is subsequently repurposed to train a view selection model.

[Figure 180]

[Figure 181]

- Caption 1
- Caption 2
- Caption 3

Ground Truth Narration

| | |
|---|---|
| | |

| | |
|---|---|
| | |

VideoCaptioner

- View 1
- View 2
- View 3

- View 1
- View 2
- View 3

|[Figure 182]|
|---|

|[Figure 183]|
|---|

[Figure 184]

[Figure 185]

View Selector

View Ranker

Best-view label

Best-view pseudo-label

[Figure 186]

[Figure 187]

1) Generate best-view pseudo-labeles 2) Train best-view selector

Fig. 23. Illustration of a typical ego-exo view selection method, adapted from [187]. This approach leverages video captions as weak supervision for selecting the best view.

• Discussion: In ego-exo settings, the field-of-view disparity between egocentric (close-up) and exocentric (wide-angle) perspectives poses unique challenges compared to traditional multi-view selection systems. This requires models to determine whether the current task phase demands a focused “zoomed-in” or a contextual “far-view” perspective. This challenge is especially important in instructional videos for educational purposes. Future research could integrate userspecific preferences into view selection criteria.

V. DATASETS

We introduce publicly available datasets offering both egocentric and exocentric perspectives. We categorize these datasets based on domain and describe their intended purposes, views, annotations, and unique features. This overview helps researchers select suitable datasets for their studies.

Table I provides a summary of the datasets. For datasets that provide synchronized videos, we list the number of first-person and third-person viewpoints. Most datasets cover multiple activity types, while others focus on activities in specialized scenarios. Additionally, datasets [6], [7], [66], [83], [151], [159], [163], [172], [189]–[191] include multi-agent settings, involving multiple participants in a video. This facilitates the analysis of human interactions and collaboration in complex activities. Furthermore, datasets [8], [66], [67], [191]–[194] provide egocentric eye gaze information, offering valuable insights into human intention and decision-making process. Below, we provide a detailed description of each dataset.

A. Action Understanding. Most ego-exo action understanding datasets focus on activities in specific scenarios or controlled environments. CMU-MMAC [195] records videos of individuals cooking recipes in a lab kitchen. H2O [196], Assembly101 [197], ARCTIC [198] and OAKINK2 [199] focus on hand-object manipulation on the tabletop. Homage [200] captures daily life activities in two houses. LEMMA

[189] features multi-agent goal-directed daily activities in living room and kitchen scenarios. FT-HID [190] focuses on multi-person interactions and includes 30 human interaction action classes. EgoExo-Fitness [201] focuses on full-body action understanding in natural fitness scenarios. CharadesEgo [2] leverages scripts from the Charades [202] and selfcollected data, recording multi-view videos of participants performing these scripts. CORE4D-Real [203] uniquely captures multi-person and object interactions in household object rearrangement.

More recent datasets involve diverse activities in multiple environments. Ego-Exo4D [8] is a large-scale multi-view dataset focused on skilled human activities. It offers multimodal annotations, including audio, eye gaze, 3D point clouds, and detailed language descriptions. Both EgoExoLearn [192] and EgoMe [204] include exocentric demonstration videos and corresponding egocentric recordings of individuals performing the tasks based on the demonstrations. These datasets offer valuable resources for studying how humans interpret and adapt actions from an external perspective to their own.

To analyze human motion, EgoPW [205], First2ThirdPose [62] and ECHP [114] are designed for egocentric human full-body pose estimation with support from third-person cameras. Specifically, egocentric videos in ECHP [114] are recorded using a head-mounted fisheye camera. AssemblyHands [206] and ThermoHands [207] focus on hand-object interaction and provide hand pose annotations. EgoHumans [6] features 3D pose estimation and tracking. Nymeria [208] is a large-scale motion dataset collected in the wild, featuring multimodal egocentric data and a third-person view by an observer. In the surgical domain, Hein et al. [66] propose a multiview dataset for the pose estimation of surgical instruments.

OVR [209] is the first multi-view dataset for temporal repetition counting. This task aims to identify repetitive events in a video. Videos in OVR [209] are sourced from exocentric dataset Kinetics [19] and egocentric dataset Ego4D [4]. Annotations include the start and end times of repetitions, the number of repetitions, and action descriptions. The openvocabulary semantics of OVR [209] support text-conditioned repetition counting.

- B. Driving. Integrating both in-vehicle and out-vehicle

views can provide a comprehensive understanding of the driver’s behavior. LBW [67] is a multi-view driving dataset for driver’s attention estimation. It includes gaze data from eye-tracking glasses and the forward road scene. AIDE [68] is designed for assistive driving perception, capturing naturalistic driving from four views: three external (front, left, right) and one internal (driver’s state). Annotations cover facial expressions, body postures, gestures, and vehicle conditions. WTS [191] provides not only vehicle and infrastructure perspectives, but also pedestrian perspectives. It can advance fine-grained video event detection.

- C. Affordance Grounding. AGD20K [122] is the earliest

image-level multi-view affordance grounding dataset. It classifies the collected data into seen and unseen sets to evaluate the model’s generalization ability. It has become a widely used benchmark for numerous methods. To advance dexterous manipulation research, FAH [78] identifies multi-finger grasp-

ing regions through detailed hand movement categorization. PAD [210] provides pixel-level annotations, enabling precise affordance grounding through semantic segmentation models.

##### D. Generation. ThirdtoFirst [51] is designed for exocentric

to egocentric image synthesis. It consists of 531 temporally aligned video pairs. Video collectors perform various actions in front of the exocentric camera (side or top-view), while a body-worn camera captures their motion from the first-person perspective.

##### E. Scene Understanding. 360+x [211] is a multi-view,

multi-modal panoptic scene understanding dataset. It includes third-person panoramic and front views, as well as first-person monocular and binocular views. The dataset also offers audio, location data, and textual scene descriptions. Benchmarks include video scene classification, temporal action localization, and cross-modality retrieval.

##### F. Video Question Answering. GazeVQA [193] is designed

for task-oriented video question answering. It features collaboration between an instructor and a novice in assembling or disassembling an industrial product. A key feature of GazeVQA [193] is the inclusion of egocentric eye gaze information, which aids in understanding human intention.

##### G. Egocentric Wearer Identification. Ego2Top [159],

IUShareView [7] , and TF2023 [164] utilize a fixed exocentric camera and multiple egocentric cameras mounted on different individuals in the environment. In IUShareView [7] and TF2023 [164], each person is annotated with a unique ID. Additionally, TF2023 [164] provides segmentation masks for individuals in third-person views.

##### H. Cross-View Human Identification. CVMHT [172] com-

prises over 23K frames of top-view and horizontal-view videos from five different locations. Annotations include bounding boxes and cross-view ID numbers for subjects. DMHA [174] is a synthetic dataset featuring top-view and side-view videos from common outdoor surveillance scenes. Compared to CVMHT [172], it also includes the side-view camera’s location and view direction in the top-view.

##### I. Camera Localization. CSRD-II [151] and CSRD-V

[151] are synthetic datasets for egocentric camera localization. Annotations include subject positions and camera poses in the bird’s-eye view. YOWO [83] is a synthetic dataset for exocentric camera localization. An agent with an egocentric camera traverses the scene, collaborating with ceiling-mounted cameras for scene reconstruction and camera localization.

VI. DISCUSSION

This section discusses the limitations of current research and offers insights into future directions from the perspectives of data, model, and application.

Insights from Data. Most existing datasets focus on daily life activities, resulting in a scarcity of data tailored to specific scenarios such as public service, healthcare, and education. This limitation hinders the development of approaches for specialized applications. Additionally, most datasets use sophisticated multi-camera setups to record synchronized egocentric and exocentric videos. This significantly increases costs and limits the scalability of data collection. Future research could

TABLE I OVERVIEW OF EGO-EXO DATASETS: ‘DATA STATISTICS’ SHOWS VIDEO/FRAME/HOUR STATS. ‘EGO/EXO VIEWS’ LISTS VIEWPOINTS FOR SYNCHRONIZED DATASETS. ‘MULTI-ACTIVITIES’ INDICATES VARIED ACTIVITIES. ‘MULTI-AGENTS’ DENOTES INTERACTIONS AMONG MULTIPLE PEOPLE.

|Dataset|Year<br><br>|Domain|Data Statistics|Exo Views|Ego Views|Multi-Activities|Multi-Agents|Gaze|
|---|---|---|---|---|---|---|---|---|
|CMU-MMAC [195] Charades-Ego [2] LEMMA [189] H2O [196] HOMAGE [200] Assembly101 [197] EgoPW [205] ARCTIC [198] FT-HID [190] EgoHumans [6] AssemblyHands [206] First2Third-Pose [62] ECHP [114] Hein et al. [66] OAKINK2 [199] EgoExo-Fitness [201] CORE4D-Real [203] Ego-Exo4D [8] EgoExoLearn [192] ThermoHands [207] Nymeria [208] OVR [209] EgoMe [204]|2008 2018<br><br>2020<br><br>2021<br><br>2021<br><br>2022<br><br>2022 2023<br><br>2023<br><br>2023<br><br>2023<br><br>2023<br><br>2023<br><br>2023<br><br>2024<br><br><br>2024<br><br><br>2024<br><br><br>2024<br><br><br>2024<br><br><br>2024<br><br><br>2024<br><br><br>2024<br><br>2025<br><br><br>|Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding Action Understanding<br><br>|1050 videos 7.4M frames 4.1M frames 571K frames 25.5 hours 110M frames 318K frames<br><br>2.1M frames 6.4M frames 571K frames<br>3.03M frames 190K frames<br><br>75K frames 1.7M frames<br><br>4.01M frames 1276 videos<br><br><br>1K videos 1286 hours 120 hours 96K frames 201M frames 72552 videos 15804 videos|3<br><br>1<br><br>2<br><br><br>4<br><br><br>1-4 8<br><br>1<br><br>8 3 8-15 8<br><br>2-3<br><br>2<br><br>5<br><br>3<br><br>3 4<br><br>4<br><br><br><br><br>1 1 1<br><br>|2 1 1 1 1 4 1<br><br>1<br><br>2 1 4 1<br><br><br>1<br><br>2 1<br><br>3 1 1<br><br><br>1 1 1<br><br>|✗ ✓ ✓ ✗ ✓ ✗ ✓ ✗ ✓ ✓ ✗ ✓ ✓ ✗ ✓ ✓ ✗ ✓ ✓ ✓ ✓ ✓ ✓|✗ ✗ ✓ ✗ ✗ ✗ ✗ ✗ ✓ ✓ ✗ ✗ ✗ ✓ ✗ ✗ ✓ ✗ ✗ ✗ ✗ ✗ ✗<br><br>|✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✓ ✗ ✗ ✗ ✓ ✓ ✗ ✓ ✗ ✓|
|LBW [67] AIDE [68] WTS [191]<br><br>|2022<br><br>2023<br><br>2024<br><br><br>|Driving Driving Driving<br><br>|123K frames 521.6K frames 52.8K frames<br><br>|1 1 18|2<br><br>3<br><br><br>2<br><br>|✗ ✗ ✗|✗ ✗ ✓<br><br>|✓ ✗ ✓|
|PAD [210] AGD20K [122] FAH [78]<br><br>|2021<br><br>2022 2024<br><br><br>|Affordance Grounding Affordance Grounding Affordance Grounding|4K frames 20K frames 6K frames<br><br>|1 1 1|1 1 1<br><br>|-|-<br><br>|-<br><br>|
|Thirdtofirst [51]|2021<br><br>|Generation|334.6K frames|1<br><br>|1|✓|✗<br><br>|✗|
|360+x [211]<br><br>|2024<br><br>|Scene Understanding<br><br>|8.5M frames<br><br>|2|2<br><br>|✓<br><br>|✗|✗|
|GazeVQA [193]|2023<br><br>|Video Question Answering<br><br>|125 hours<br><br>|2<br><br>|1|✗<br><br>|✗|✓|
|Ego2Top [159] IUShareView [7] TF2023 [164]|2016<br><br>2017<br><br><br>2024<br><br>|Egocentric Wearer Identification Egocentric Wearer Identification Egocentric Wearer Identification|225K frames 11.2K frames 49.8K frames<br><br>|1 1 1<br><br>|1-6<br>2 2<br>|✓ ✓ ✓<br><br>|✓ ✓ ✓|✗ ✗ ✗<br><br>|
|CVMHT [172] DMHA [174]<br><br>|2020 2022|Cross-View Human Identification Cross-View Human Identification<br><br>|23K frames 84.8K frames<br><br>|1 1|2-3 1<br><br>|✓ ✗|✓ ✓<br><br>|✗ ✗<br><br>|
|CSRD-II [151] CSRD-V [151] YOWO [83]<br><br>|2022 2022 2024|Camera Registration Camera Registration Camera Registration<br><br>|2K frames 5K frames -|1 1 5-17<br><br>|2 5 1|✓ ✓ -<br><br>|✓ ✓ -<br><br>|✗ ✗ -|

investigate transforming existing unpaired egocentric [3], [4] and exocentric [17]–[19] datasets to enable collaboration between these perspectives. Furthermore, integrating video data with other modalities, such as audio [116] and IMU sensors [158], could enrich the captured information, providing a more comprehensive understanding of complex scenarios.

Insights from Model. Most existing models are designed for specific tasks and lack generalizability. In contrast, recent advancements in vision-language models (VLMs) [212]–[215] highlight their effectiveness to handle diverse tasks. Future research could explore equipping VLMs with the capability to integrate egocentric and exocentric perspectives, facilitating unified cross-view tasks in a single framework. Moreover, current methods often rely on synchronized egocentric and exocentric data. However, the limited scale of such paired datasets hinders the effective training of large models. To overcome this limitation, promising directions include leveraging alignment strategies or retrieval-augmented methods [216] to better utilize unpaired data.

Insights from Application. Current research are primarily centered on daily life contexts, with limited attention to specialized application domains. For instance, while affordance grounding has been well-studied for everyday objects [77],

[122]–[125], predicting affordance regions for surgical tools or industrial components receives less attention. Extending egocentric and exocentric collaboration techniques to domains such as medicine and industry could unlock new opportunities in these fields.

VII. CONCLUSION

This survey presents a comprehensive review of crossview collaboration with egocentric and exocentric vision. We begin by discussing the practical value of egocentric and exocentric collaboration across various applications. We then link these applications to key research tasks required to realize them. Current research advancements are categorized into three directions: egocentric for exocentric, exocentric for egocentric, and joint learning, with a detailed overview of progress in each area. In addition, we review relevant datasets that support both perspectives. Finally, we provide a discussion on data, models, and applications, and outline future research directions. We hope this review inspires deeper exploration into egocentric-exocentric collaboration, paving the way for artificial intelligence to perceive the world with human-like vision.

REFERENCES

- [1] G. Rizzolatti and L. Craighero, “The mirror-neuron system,” Annual review of neuroscience, vol. 27, pp. 169–92, 02 2004.
- [2] G. A. Sigurdsson, A. Gupta, C. Schmid, A. Farhadi, and K. Alahari, “Actor and observer: Joint modeling of first and third-person videos,” in IEEE Conf. Comput. Vis. Pattern Recog., 2018, pp. 7396–7404.
- [3] D. Damen et al., “The epic-kitchens dataset: Collection, challenges and baselines,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 43, no. 11, pp. 4125–4141, 2021.
- [4] K. Grauman et al., “Ego4d: Around the world in 3,000 hours of egocentric video,” IEEE Trans. Pattern Anal. Mach. Intell., pp. 1–32, 2024.
- [5] Y. Song, E. Byrne, T. Nagarajan, H. Wang, M. Martin, and L. Torresani, “Ego4d goal-step: Toward hierarchical understanding of procedural activities,” Adv. Neural Inform. Process. Syst., vol. 36, 2024.
- [6] R. Khirodkar, A. Bansal, L. Ma, R. Newcombe, M. Vo, and K. Kitani, “Ego-humans: An ego-centric 3d multi-human benchmark,” in Int. Conf. Comput. Vis., 2023, pp. 19807–19819.
- [7] C. Fan et al., “Identifying first-person camera wearers in third-person videos,” in IEEE Conf. Comput. Vis. Pattern Recog., July 2017.
- [8] K. Grauman et al., “Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 19383–19400.
- [9] T. Perrett et al., “Hd-epic: A highly-detailed egocentric video dataset,” in IEEE Conf. Comput. Vis. Pattern Recog., June 2025.
- [10] K. Q. Lin et al., “Egocentric video-language pretraining,” in Adv. Neural Inform. Process. Syst., vol. 35, 2022, pp. 7575–7586.
- [11] S. Pramanick et al., “Egovlpv2: Egocentric video-language pre-training with fusion in the backbone,” in Int. Conf. Comput. Vis., 2023, pp. 5262–5274.
- [12] C. Zhang, A. Gupta, and A. Zisserman, “Helping hands: An objectaware ego-centric video recognition model,” in Int. Conf. Comput. Vis., 2023, pp. 13855–13866.
- [13] G. Chen et al., “Internvideo-ego4d: A pack of champion solutions to ego4d challenges,” 2022, arXiv: 2211.09529.
- [14] B. Pei et al., “Egovideo: Exploring egocentric foundation model and downstream adaptation,” 2024, arXiv:2406.18070.
- [15] ——, “Modeling fine-grained hand-object dynamics for egocentric video representation learning,” 2025, arXiv:2503.00986.
- [16] A. Miech, D. Zhukov, J.-B. Alayrac, M. Tapaswi, I. Laptev, and J. Sivic, “Howto100m: Learning a text-video embedding by watching hundred million narrated video clips,” Int. Conf. Comput. Vis., pp. 2630–2640, 2019.
- [17] C. Gu et al., “Ava: A video dataset of spatio-temporally localized atomic visual actions,” in IEEE Conf. Comput. Vis. Pattern Recog., 2018, pp. 6047–6056.
- [18] K. Soomro, A. R. Zamir, and M. Shah, “Ucf101: A dataset of 101 human actions classes from videos in the wild,” 2012, arXiv:1212.0402.
- [19] J. Carreira and A. Zisserman, “Quo vadis, action recognition? a new model and the kinetics dataset,” IEEE Conf. Comput. Vis. Pattern Recog., pp. 4724–4733, 2017.
- [20] Y. Wang et al., “Internvid: A large-scale video-text dataset for multimodal understanding and generation,” 2023, arXiv: 2307.06942.
- [21] G. Chen et al., “Cg-bench: Clue-grounded question answering benchmark for long video understanding,” 2024, arXiv: 2412.12075.
- [22] K. Simonyan and A. Zisserman, “Two-stream convolutional networks for action recognition in videos,” in Adv. Neural Inform. Process. Syst. Cambridge, MA, USA: MIT Press, 2014, p. 568–576.
- [23] A. Arnab, M. Dehghani, G. Heigold, C. Sun, M. Luˇci´c, and C. Schmid, “Vivit: A video vision transformer,” in Int. Conf. Comput. Vis., 2021, pp. 6816–6826.
- [24] G. Bertasius, H. Wang, and L. Torresani, “Is space-time attention all you need for video understanding?” in Int. Conf. Mach. Learn., July 2021.
- [25] G. Chen, Y.-D. Zheng, L. Wang, and T. Lu, “Dcan: improving temporal action detection via dual context aggregation,” in AAAI Conf. Artif. Intell., vol. 36, no. 1, 2022, pp. 248–257.
- [26] G. Chen et al., “Video mamba suite: State space model as a versatile alternative for video understanding,” 2024, arXiv: 2403.09626.
- [27] J. Wang, G. Chen, Y. Huang, L. Wang, and T. Lu, “Memory-andanticipation transformer for online action understanding,” in Int. Conf. Comput. Vis., 2023, pp. 13824–13835.
- [28] H. Liu et al., “Video super-resolution based on deep learning: a comprehensive survey,” Artif Intell Rev, vol. 55, pp. 5981–6035, 2022.
- [29] A. Stergiou and R. Poppe, “About time: Advances, challenges, and outlooks of action understanding,” 2024, arXiv:2411.15106.

- [30] L. Jiao et al., “New generation deep learning for video object detection: A survey,” IEEE Trans. Neural Netw. Learn. Syst., vol. 33, no. 8, pp. 3195–3215, 2022.
- [31] B. Ramachandra, M. J. Jones, and R. R. Vatsavai, “A survey of singlescene video anomaly detection,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 5, pp. 2293–2312, 2022.
- [32] C. Plizzari et al., “An outlook into the future of egocentric vision,” Int. J. Comput. Vis., pp. 1–57, 2024.
- [33] N. Lavars, “Samsung’s new smart fridge lets you check in on its contents through internal cameras,” Jan 2016. [Online]. Available: https://newatlas.com/samsung-family-hub-smart-fridge/41192/
- [34] J. Oven, “June oven,” Aug 2018. [Online]. Available: https: //firewireblog.com/2018/08/19/june-oven/
- [35] J. McDonald, “Sportvu stats can be helpful, overwhelming,” Nov

2013. [Online]. Available: https://www.expressnews.com/sports/spurs/ article/SportVU-stats-can-be-helpful-overwhelming-4993731.php

- [36] D. Winter, “Hawk-eye’s eagle eye on wimbledon tennis,” 2024. [Online]. Available: https://www.redsharknews.com/hawk-eyes-eyeon-wimbledon
- [37] J. Dachman, “Super bowl li preview: Inside fox sports’ “be the player” first-person pov replay tech,” Jan 2017. [Online]. Available: https://www.sportsvideo.org/2017/01/13/super-bowl-li-previewinside-fox-sports-be-the-player-360-pov-replay-technology/
- [38] H. M. Asia, “Smart glasses in hospitals: Viewing care delivery through a new lens,” HMA, 03 2022. [Online]. Available: https://www.hospitalmanagementasia.com/tech-innovation/ smart-glasses-in-hospitals-viewing-care-delivery-through-a-new-lens/
- [39] ——, “Here’s how asean’s first 5g smart hospital is using ai to usher in a new era of healthcare,” HMA, 02 2022. [Online]. Available: https://www.hospitalmanagementasia.com/tech-innovation/heres-howaseans-first-5g-smart-hospital-is-using-ai-to-usher-in-a-new-era-ofhealthcare/
- [40] K. Rahman, “Cameras could be installed in classrooms in these states,” Jan 2024. [Online]. Available: https://www.newsweek.com/camerasinstalled-classrooms-1859098
- [41] Reolink, “Classroom camera: Transform education,” Nov 2024. [Online]. Available: https://reolink.com/blog/classroom-camera
- [42] E. News, “3d surround view system,” Jun 2018. [Online]. Available: https://www.ien.eu/article/3d-surround-view-system/
- [43] FreightWaves, “Nauto launches real-time driver behavior learning platform for fleets,” Nov 2019. [Online]. Available: https://finance. yahoo.com/news/nauto-launches-real-time-driver-144742897.html
- [44] P. L. Liu, “Vision-centric semantic occupancy prediction for autonomous driving,” May 2023. [Online]. Available: https://towardsdatascience.com/vision-centric-semanticoccupancy-prediction-for-autonomous-driving-16a46dbd6f65
- [45] harkiran78, “Top 10 applications of robotics in 2024,” geeksforgeeks, Feb 2024. [Online]. Available: https://www.geeksforgeeks.org/ applications-of-robotics
- [46] N. I. of Justice, “Research on body-worn cameras and law enforcement,” National Institute of Justice, Jan 2022. [Online]. Available: https://nij.ojp.gov/topics/articles/research-bodyworn-cameras-and-law-enforcement
- [47] I. Singh, “Over 1,000 people saved with drone search and rescue: Dji,” DroneDJ, Jul 2023. [Online]. Available: https://dronedj.com/ 2023/07/12/dji-drone-search-rescue-map/
- [48] Fogsphere, “Empowering health & safety monitoring in manufacturing

- fogsphere,” Sep 2023. [Online]. Available: https://fogsphere.com/ industries-served/manufacturing/

- [49] R. Begg, “A vision-guided robotic system designed to grab any object,” Aug 2024. [Online]. Available: https://www.machinedesign.com/markets/robotics/video/55131589/ cynlr-a-vision-guided-robotic-system-designed-to-grab-any-object
- [50] Jobit, “How visual ai transforms assembly line operations in factories,”

2024. [Online]. Available: https://randomwalk.ai/blog/how-visual-aitransforms-assembly-line-operations-in-factories/

- [51] Y. Li, T. Nagarajan, B. Xiong, and K. Grauman, “Ego-exo: Transferring visual representations from third-person to first-person videos,” in IEEE Conf. Comput. Vis. Pattern Recog., 2021, pp. 6943–6953.
- [52] T.-D. Truong and K. Luu, “Cross-view action recognition understanding from exocentric to egocentric perspective,” 2023, arXiv:2305.15699.
- [53] Z.-Y. Dou et al., “Unlocking exocentric video-language data for egocentric video representation learning,” 2024, arXiv:2408.03567.
- [54] Y. Huang, X. Yang, J. Gao, and C. Xu, “Holographic feature learning of egocentric-exocentric videos for multi-domain action recognition,” IEEE Trans. Multimedia, vol. 24, pp. 2273–2286, 2022.

- [55] C. Quattrocchi, A. Furnari, D. Di Mauro, M. V. Giuffrida, and G. M. Farinella, “Synchronization is all you need: Exocentric-to-egocentric transfer for temporal action segmentation with unlabeled synchronized video pairs,” 2023, arXiv:2312.02638.
- [56] B. Soran, A. Farhadi, and L. G. Shapiro, “Action recognition in the presence of one egocentric and multiple static cameras,” in Lect. Notes Comput. Sci., 2014.
- [57] G. Liu, H. Tang, H. Latapie, J. J. Corso, and Y. Yan, “Cross-view exocentric to egocentric video synthesis,” ACM Int. Conf. Multimedia, 2021.
- [58] F. Cheng et al., “4diff: 3d-aware diffusion model for third-to-first viewpoint translation,” in Eur. Conf. Comput. Vis., 2024.
- [59] T. Ohkawa et al., “Exo2egodvc: Dense video captioning of egocentric procedural activities using web instructional videos,” 2023, arXiv:2311.16444.
- [60] J. Xu et al., “Retrieval-augmented egocentric video captioning,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 13525–13536.
- [61] J. Wang, L. Liu, W. Xu, K. Sarkar, D. Luvizon, and C. Theobalt, “Estimating egocentric 3d human pose in the wild with external weak supervision,” in IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 13157–13166.
- [62] A. Dhamanaskar, M. Dimiccoli, E. Corona, A. Pumarola, and F. Moreno-Noguer, “Enhancing egocentric 3d pose estimation with third person views,” Pattern Recognition, vol. 138, p. 109358, 2023.
- [63] S. Ardeshir and A. Borji, “An exocentric look at egocentric actions and vice versa,” Comput Vision Image Understanding, pp. 61–68, 2018.
- [64] Z. S. Xue and K. Grauman, “Learning fine-grained view-invariant representations from unpaired ego-exo videos via temporal alignment,” in Adv. Neural Inform. Process. Syst., vol. 36, 2023, pp. 53688–53710.
- [65] Y. Saito, R. Hachiuma, H. Saito, H. Kajita, Y. Takatsume, and T. Hayashida, “Camera selection for occlusion-less surgery recording via training with an egocentric camera,” IEEE Access, vol. 9, pp. 138307–138322, 2021.
- [66] J. Hein et al., “Next-generation surgical navigation: Marker-less multi-view 6dof pose estimation of surgical instruments,” 2023, arXiv:2305.03535.
- [67] I. Kasahara, S. Stent, and H. S. Park, “Look both ways: Self-supervising driver gaze estimation and road scene saliency,” in Eur. Conf. Comput. Vis. Springer, 2022, pp. 126–142.
- [68] D. Yang et al., “Aide: A vision-driven multi-view, multi-modal, multitasking dataset for assistive driving perception,” in Int. Conf. Comput. Vis., 2023, pp. 20402–20413.
- [69] K. Hsu, M. J. Kim, R. Rafailov, J. Wu, and C. Finn, “Vision-based manipulators need to also see from their hands,” 2022, arXiv:2203.12677.
- [70] R. Jangir, N. Hansen, S. Ghosal, M. Jain, and X. Wang, “Look closer: Bridging egocentric and third-person views with transformers for robotic manipulation,” IEEE Trans. Robot. Autom., vol. 7, no. 2, pp. 3046–3053, 2022.
- [71] J. Shang and M. S. Ryoo, “Self-supervised disentangled representation learning for third-person imitation learning,” in IEEE Int. Conf. Intell. Rob. Syst., 2021, pp. 214–221.
- [72] C. Acar, K. Binici, A. Tekirda˘g, and Y. Wu, “Visual-policy learning through multi-camera view to single-camera view knowledge distillation for robot manipulation tasks,” IEEE Trans. Robot. Autom., vol. 9, no. 1, pp. 691–698, 2024.
- [73] P. Sharma, D. Pathak, and A. K. Gupta, “Third-person visual imitation learning via decoupled hierarchical controller,” in Adv. Neural Inform. Process. Syst., 2019.
- [74] M. Dunion and S. V. Albrecht, “Multi-view disentanglement for reinforcement learning with multiple cameras,” 2024, arXiv:2404.14064.
- [75] Y. Seo, J. Kim, S. James, K. Lee, J. Shin, and P. Abbeel, “Multi-view masked world models for visual robotic manipulation,” in Int. Conf. Mach. Learn., 2023.
- [76] Z. Wang, Y.-H. Li, X. Li, H. Zang, R. Laroche, and R. Islam, “Learning fused state representations for control from multi-view observations,” 2025, arXiv: 2502.01316.
- [77] Z. Zhang, Z. Wei, G. Sun, P. Wang, and L. Van Gool, “Self-explainable affordance learning with embodied caption,” 2024, arXiv:2404.05603.
- [78] F. Yang et al., “Learning granularity-aware affordances from humanobject interaction for tool-based functional grasping in dexterous robotics,” 2024, arXiv:2407.00614.
- [79] L. Garello, F. Rea, N. Noceti, and A. Sciutti, “Towards third-person visual imitation learning using generative adversarial networks,” in IEEE Int. Conf. Dev. Learn. IEEE, 2022, pp. 121–126.
- [80] J. Spisak, M. Kerzel, and S. Wermter, “Diffusing in someone else’s shoes: Robotic perspective taking with diffusion,” 2024, arXiv:2404.07735.

- [81] A. Abdullah, R. Chen, I. Rekleitis, and M. J. Islam, “Ego-to-exo: Interfacing third person visuals from egocentric views in real-time for improved rov teleoperation,” 2024, arXiv:2407.00848.
- [82] L. Morando and G. Loianno, “Spatial assisted human-drone collaborative navigation and interaction through immersive mixed reality,” in Int. Conf. Robot. Autom. IEEE, 2024, pp. 8707–8713.
- [83] F. Yang, S. Yamao, I. Kusajima, A. Moteki, S. Masui, and S. Jiang, “Yowo: You only walk once to jointly map an indoor scene and register ceiling-mounted cameras,” IEEE Trans. Circuit Syst. Video Technol., pp. 1–1, 2024.
- [84] K. Nakashima, Y. Iwashita, A. Kawamura, and R. Kurazume, “Fourthperson captioning: Describing daily events by uni-supervised and triregularized training,” in IEEE Trans. Syst., Man, Cybern., 2018, pp. 2122–2127.
- [85] M. Luo, Z. Xue, A. Dimakis, and K. Grauman, “Put myself in your shoes: Lifting the egocentric perspective from exocentric videos,” in Eur. Conf. Comput. Vis. Springer, 2025, pp. 407–425.
- [86] L. Hu, X. Gao, P. Zhang, K. Sun, B. Zhang, and L. Bo, “Animate anyone: Consistent and controllable image-to-video synthesis for character animation,” IEEE Conf. Comput. Vis. Pattern Recog., pp. 8153–8163, 2023.
- [87] A. Blattmann et al., “Stable video diffusion: Scaling latent video diffusion models to large datasets,” 2023, arXiv:2311.15127.
- [88] Z. Yang et al., “Cogvideox: Text-to-video diffusion models with an expert transformer,” 2024, arXiv:2408.06072.
- [89] P. Esser, J. Chiu, P. Atighehchian, J. Granskog, and A. Germanidis, “Structure and content-guided video synthesis with diffusion models,” in Int. Conf. Comput. Vis., 2023, pp. 7312–7322.
- [90] S. Yin et al., “Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory,” 2023, arXiv:2308.08089.
- [91] H. Luo, K. Zhu, W. Zhai, and Y. Cao, “Intention-driven ego-to-exo video generation,” 2024, arXiv:2403.09194.
- [92] D. Reilly, M. K. Govind, and S. Das, “From my view to yours: Egoaugmented learning in large vision language models for understanding exocentric daily living activities,” 2025, arXiv:2501.05711.
- [93] M. Nishimura, S. Nobuhara, and K. Nishino, “Viewbirdiformer: Learning to recover ground-plane crowd trajectories and ego-motion from a single ego-centric view,” IEEE Trans. Robot. Autom., vol. 8, no. 1, pp. 368–375, 2023.
- [94] ——, “View birdification in the crowd: Ground-plane localization from perceived movements,” Int. J. Comput. Vis., vol. 131, no. 8, p. 2015–2031, May 2023.
- [95] ——, “Incrowdformer: On-ground pedestrian world model from egocentric views,” 2023, arXiv:2303.09534.
- [96] W. Li, G. Wu, W. Wang, P. Ren, and X. Liu, “Fastllve: Real-time lowlight video enhancement with intensity-aware look-up table,” in ACM Int. Conf. Multimedia, 2023, pp. 8134–8144.
- [97] J. Lin et al., “Flow-guided sparse transformer for video deblurring,” in Int. Conf. Mach. Learn., vol. 162, 17–23 Jul 2022, pp. 13334–13343.
- [98] Z. Zhong, M. Cao, X. Ji, Y. Zheng, and I. Sato, “Blur interpolation transformer for real-world motion from blur,” in Int. Conf. Comput. Vis., 2023, pp. 5713–5723.
- [99] T. Shimizu, K. Oishi, R. Hachiuma, H. Kajita, Y. Takatsume, and H. Saito, “Surgery recording without occlusions by multi-view surgical videos,” in 15th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications, 2020, pp. 837–844.
- [100] R. Hachiuma, T. Shimizu, H. Saito, H. Kajita, and Y. Takatsume, “Deep selection: A fully supervised camera selection network for surgery recordings,” in Medical Image Computing and Computer Assisted Intervention. Springer, 2020, pp. 419–428.
- [101] M. Elfeki, K. Regmi, S. Ardeshir, and A. Borji, “From third person to first person: Dataset and baselines for synthesis and retrieval,” 2018, arXiv:1812.00104.
- [102] G. Liu, H. Tang, H. M. Latapie, J. J. Corso, and Y. Yan, “Crossview exocentric to egocentric video synthesis,” in ACM Int. Conf. Multimedia, 2021, pp. 974–982.
- [103] G. Liu, H. Latapie, O. Kilic, and A. Lawrence, “Parallel generative adversarial network for third-person to first-person image generation,” in IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 1917–1923.
- [104] J. Xu et al., “Egoexo-gen: Ego-centric video prediction by watching exo-centric videos,” 2025, arXiv:2504.11732.
- [105] S. Liu, Z. Ren, and J. Yuan, “Sibnet: Sibling convolutional encoder for video captioning,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 43, pp. 3259–3272, 2018.

- [106] B. Pan et al., “Spatio-temporal graph for video captioning with knowledge distillation,” IEEE Conf. Comput. Vis. Pattern Recog., pp. 10867–10876, 2020.
- [107] H. Munusamy and C. C. Sekhar, “Domain-specific semantics guided approach to video captioning,” IEEE Winter Conf. Appl. Comput. Vis., pp. 1576–1585, 2020.
- [108] R.-C. Chang, Y. Liu, and A. Guo, “Worldscribe: Towards context-aware live visual descriptions,” in ACM Symp. User Interface Softw. Technol., 2024, pp. 1–18.
- [109] M. Kuribayashi, K. Uehara, A. Wang, S. Morishima, and C. Asakawa, “Wanderguide: Indoor map-less robotic guide for exploration by blind people,” 2025, arXiv:2502.08906.
- [110] Y. Zhang, H. Doughty, L. Shao, and C. G. Snoek, “Audio-adaptive activity recognition across video domains,” in IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 13791–13800.
- [111] B. Rocha, P. Moreno, and A. Bernardino, “Cross-view generalisation in action recognition: Feature design for transitioning from exocentric to egocentric views,” in Iberian Robotics Conference, 2023, pp. 155–166.
- [112] Q. Wang, L. Zhao, L. Yuan, T. Liu, and X. Peng, “Learning from semantic alignment between unpaired multiviews for egocentric video recognition,” in Int. Conf. Comput. Vis., 2023, pp. 3284–3294.
- [113] J. Choi, G. Sharma, M. Chandraker, and J.-B. Huang, “Unsupervised and semi-supervised domain adaptation for action recognition from drones,” in IEEE Winter Conf. Appl. Comput. Vis., 2020, pp. 1717– 1726.
- [114] Y. Liu, J. Yang, X. Gu, Y. Chen, Y. Guo, and G.-Z. Yang, “Egofish3d: Egocentric 3d pose estimation from a fisheye camera via selfsupervised learning,” IEEE Trans. Multimedia, pp. 8880–8891, 2023.
- [115] M. Tran, Y. Kim, C.-C. Su, C.-H. Kuo, M. Sun, and M. Soleymani, “Ex2eg-mae: A framework for adaptation of exocentric video masked autoencoders for egocentric social role understanding,” in Eur. Conf. Comput. Vis. Springer, 2025, pp. 1–19.
- [116] W. Jia et al., “The audio-visual conversational graph: From an egocentric-exocentric perspective,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 26396–26405.
- [117] C. Xu, Y. Fu, B. Zhang, Z. Chen, Y.-G. Jiang, and X. Xue, “Learning to score figure skating sport videos,” IEEE Trans. Circuit Syst. Video Technol., vol. 30, no. 12, pp. 4578–4590, 2019.
- [118] K. Ashutosh, T. Nagarajan, G. Pavlakos, K. Kitani, and K. Grauman, “Expertaf: Expert actionable feedback from video,” 2024, arXiv:2408.00672.
- [119] J. Rao, H. Wu, H. Jiang, Y. Zhang, Y. Wang, and W. Xie, “Towards universal soccer video understanding,” 2024, arXiv:2412.01820.
- [120] Y. Song et al., “Learning 6-dof fine-grained grasp detection based on part affordance grounding,” 2024, arXiv: 2301.11564.
- [121] Y. Ju, K. Hu, G. Zhang, G. Zhang, M. Jiang, and H. Xu, “Robo-abc: Affordance generalization beyond categories via semantic correspondence for robot manipulation,” in Eur. Conf. Comput. Vis., 2025, pp. 222–239.
- [122] H. Luo, W. Zhai, J. Zhang, Y. Cao, and D. Tao, “Learning affordance grounding from exocentric images,” in IEEE Conf. Comput. Vis. Pattern Recog., June 2022, pp. 2252–2261.
- [123] G. Li, V. Jampani, D. Sun, and L. Sevilla-Lara, “Locate: Localize and transfer object parts for weakly supervised affordance grounding,” in IEEE Conf. Comput. Vis. Pattern Recog., June 2023, pp. 10922–10931.
- [124] L. Xu, Y. Gao, W. Song, and A. Hao, “Weakly supervised multimodal affordance grounding for egocentric images,” in AAAI Conf. Artif. Intell., vol. 38, no. 6, 2024, pp. 6324–6332.
- [125] A. Rai, K. Buettner, and A. Kovashka, “Strategies to leverage foundational model knowledge in object affordance grounding,” in IEEE. Conf. Comput. Vis. Pattern Recog. Workshops., June 2024, pp. 1714– 1723.
- [126] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba, “Learning deep features for discriminative localization,” in IEEE Conf. Comput. Vis. Pattern Recog., 2016, pp. 2921–2929.
- [127] A. Radford et al., “Learning transferable visual models from natural language supervision,” in Int. Conf. Mach. Learn., 2021.
- [128] J. H. Jang, H. Seo, and S. Y. Chun, “Intra: Interaction relationship-aware weakly supervised affordance grounding,” 2024, arXiv:2409.06210.
- [129] M. Hassanin, S. Khan, and M. Tahtali, “Visual affordance and function understanding: A survey,” ACM Comput. Surv., vol. 54, no. 3, Apr. 2021.
- [130] C. Kyrkou and T. Theocharides, “Emergencynet: Efficient aerial image classification for drone-based emergency monitoring using atrous convolutional feature fusion,” IEEE J. Sel. Topics Appl. Earth Observations Remote Sens., vol. 13, pp. 1687–1699, 2020.

- [131] J. R. Cauchard, M. Khamis, J. Garcia, M. Kljun, and A. M. Brock, “Toward a roadmap for human-drone interaction,” Interactions, vol. 28, pp. 76–81, 03 2021.
- [132] J. Li, R. Balakrishnan, and T. Grossman, “Starhopper: A touch interface for remote object-centric drone navigation,” in Proc Graphics Interface, ser. GI 2020, 2020, pp. 317 – 326.
- [133] O. Erat, W. A. Isop, D. Kalkofen, and D. Schmalstieg, “Droneaugmented human vision: Exocentric control for drones exploring hidden areas,” IEEE Trans. Vis. Comput. Graphics, vol. 24, pp. 1437– 1446, 04 2018.
- [134] R. Temma, K. Takashima, K. Fujita, K. Sueda, and Y. Kitamura, “Third-person piloting: Increasing situational awareness using a spatially coupled second drone,” in ACM Symp. User Interface Softw. Technol., 2019, pp. 507–519.
- [135] M. Inoue, K. Takashima, K. Fujita, and Y. Kitamura, “Birdviewar: Surroundings-aware remote drone piloting using an augmented thirdperson perspective,” in Conf Hum Fact Comput Syst Proc, 2023.
- [136] Y. Huang et al., “Vinci: A real-time embodied smart assistant based on egocentric vision-language model,” 2024, arXiv: 2412.21080.
- [137] D.-A. Huang et al., “Lita: Language instructed temporal-localization assistant,” in Eur. Conf. Comput. Vis., 2024.
- [138] Y. Huang et al., “An egocentric vision-language model based portable real-time smart assistant,” 2025, arXiv:2503.04250.
- [139] K. Nakashima, Y. Iwashita, and R. Kurazume, “Lifelogging caption generation via fourth-person vision in a human–robot symbiotic environment,” Robomech J., vol. 7, 2020.
- [140] S. Ren, K. He, R. B. Girshick, and J. Sun, “Faster r-cnn: Towards real-time object detection with region proposal networks,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 39, pp. 1137–1149, 2015.
- [141] M. S. Sarfraz, N. Murray, V. Sharma, A. Diba, L. V. Gool, and R. Stiefelhagen, “Temporally-weighted hierarchical clustering for unsupervised action segmentation,” IEEE Conf. Comput. Vis. Pattern Recog., pp. 11220–11229, 2021.
- [142] Y. Huang, Y. Sugano, and Y. Sato, “Improving action segmentation via graph-based temporal reasoning,” in IEEE Conf. Comput. Vis. Pattern Recog., 2020, pp. 14024–14034.
- [143] T. He et al., “Collaborative weakly supervised video correlation learning for procedure-aware instructional video analysis,” AAAI Conf. Artif. Intell., vol. 38, no. 3, pp. 2112–2120, Mar. 2024.
- [144] Y. Qian, W. Luo, D. Lian, X. Tang, P. Zhao, and S. Gao, “Svip: Sequence verification for procedures in videos,” in IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 19858–19870.
- [145] Y. Huang, M. Cai, Z. Li, F. Lu, and Y. Sato, “Mutual context network for jointly estimating egocentric gaze and action,” IEEE Transactions on Image Processing, vol. 29, pp. 7795–7806, 2020.
- [146] S. Ardeshir, K. Regmi, and A. Borji, “Egotransfer: Transferring motion across egocentric and exocentric domains using deep neural networks,” 2016, arXiv:1612.05836.
- [147] H. Yu, M. Cai, Y. Liu, and F. Lu, “First- and third-person video co-analysis by learning spatial-temporal joint attention,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 6, pp. 6631–6646, 2023.
- [148] Z. Zhang, Y. Ma, E. Zhang, and X. Bai, “Psalm: Pixelwise segmentation with large multi-modal model,” in Eur. Conf. Comput. Vis., 2025, pp. 74–91.
- [149] Y. Fu, R. Wang, Y. Fu, D. P. Paudel, X. Huang, and L. V. Gool, “Objectrelator: Enabling cross-view object relation understanding in ego-centric and exo-centric videos,” 2024, arXiv:2411.19083.
- [150] R. Han, Y. Gan, L. Wang, N. Li, W. Feng, and S. Wang, “Relating view directions of complementary-view mobile cameras via the human shadow,” Int. J. Comput. Vis., vol. 131, no. 5, p. 1106–1121, Jan. 2023.
- [151] Z. Qian, R. Han, W. Feng, F. F. Wang, and S. Wang, “From a bird’s eye view to see: Joint camera and subject registration without the camera calibration,” IEEE Conf. Comput. Vis. Pattern Recog., pp. 863–873, 2022.
- [152] E. Ataer-Cansizoglu, Y. Taguchi, S. Ramalingam, and Y. Miki, “Calibration of non-overlapping cameras using an external slam system,” in 2014 2nd International Conference on 3D Vision, vol. 1, 2014, pp. 509–516.
- [153] X. Yi et al., “Egolocate: Real-time motion capture, localization, and mapping with sparse body-mounted sensors,” ACM Transactions on Graphics, vol. 42, no. 4, 2023.
- [154] R. Yonetani, K. M. Kitani, and Y. Sato, “Recognizing micro-actions and reactions from paired egocentric videos,” in IEEE Conf. Comput. Vis. Pattern Recog., 2016, pp. 2629–2638.
- [155] R. L. Haugaard and T. M. Iversen, “Multi-view object pose estimation from correspondence distributions and epipolar geometry,” in Int. Conf. Robot. Autom., 2023, pp. 1786–1792.

- [156] T. Kalluri, B. P. Majumder, and M. Chandraker, “Tell, don’t show: Language guidance eases transfer across domains in images and videos,” in Int. Conf. Mach. Learn., Jul 2024, pp. 22879–22894.
- [157] B. Xu, S. Zheng, and Q. Jin, “Pov: Prompt-oriented view-agnostic learning for egocentric hand-object interaction in the multi-view world,” in ACM Int. Conf. Multimedia, 2023, pp. 2807–2816.
- [158] M. Zhang, Y. Huang, R. Liu, and Y. Sato, “Masked video and bodyworn imu autoencoder for egocentric action recognition,” in Eur. Conf. Comput. Vis., 2025, pp. 312–330.
- [159] S. Ardeshir and A. Borji, “Ego2top: Matching viewers in egocentric and top-view videos,” in Eur. Conf. Comput. Vis., 2016, pp. 253–268.
- [160] ——, “Egocentric meets top-view,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 41, no. 6, pp. 1353–1366, 2019.
- [161] ——, “Integrating egocentric videos in top-view surveillance videos: Joint identification and temporal alignment,” in Eur. Conf. Comput. Vis., Sep 2018.
- [162] B. Fulkerson, A. Vedaldi, and S. Soatto, “Class segmentation and object localization with superpixel neighborhoods,” Int. Conf. Comput. Vis., pp. 670–677, 2009.
- [163] M. Xu, C. Fan, Y. Wang, M. S. Ryoo, and D. J. Crandall, “Joint person segmentation and identification in synchronized first- and third-person videos,” in Eur. Conf. Comput. Vis., 2018, pp. 656–672.
- [164] Z. Zhao, Y. Wang, and C. Wang, “Fusing personal and environmental cues for identification and segmentation of first-person camera wearers in third-person views,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 16477–16487.
- [165] L. Yang, H. Jiang, Z. Huo, and J. Xiao, “Visual-gps: Ego-downward and ambient video based person location association,” in IEEE Conf. Comput. Vis. Pattern Recog Workshops., 2019, pp. 371–380.
- [166] Y. Wen, K. K. Singh, M. Anderson, W.-P. Jan, and Y. J. Lee, “Seeing the unseen: Predicting the first-person camera wearer’s location and pose in third-person scenes,” in Int. Conf. Comput. Vis., 2021, pp. 3446–3455.
- [167] Y. Xu et al., “Egopca: A new framework for egocentric hand-object interaction understanding,” in Int. Conf. Comput. Vis., 2023, pp. 5273– 5284.
- [168] T. Shiota, M. Takagi, K. Kumagai, H. Seshimo, and Y. Aono, “Egocentric action recognition by capturing hand-object contact and object state,” in IEEE Winter Conf. Appl. Comput. Vis., 2024, pp. 6541–6551.
- [169] Y. Yang, W. Zhai, C. Wang, C. Yu, Y. Cao, and Z.-J. Zha, “Egochoir: Capturing 3d human-object interaction regions from egocentric views,” Adv. Neural Inform. Process. Syst., vol. 37, pp. 54529–54557, 2025.
- [170] W. Jia, M. Liu, H. Jiang, I. Ananthabhotla, J. M. Rehg, V. K. Ithapu, and R. Gao, “The audio-visual conversational graph: From an egocentric-exocentric perspective,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 26386–26395.
- [171] R. Han, J. Zhao, W. Feng, Y. Gan, L. Wan, and S. Wang, “Complementary-view co-interest person detection,” in ACM Int. Conf. Multimedia, 2020, p. 2746–2754.
- [172] R. Han, W. Feng, J. Zhao, Z. Niu, Y. Zhang, and L. Wan, “Complementary-view multiple human tracking,” in AAAI Conf. Artif. Intell., vol. 34, 02 2020.
- [173] R. Han, W. Feng, Y. Zhang, J. Zhao, and S. Wang, “Multiple human association and tracking from egocentric and complementary top views,” IEEE Trans. Pattern Anal. Mach. Intell., pp. 5225–5242, 2022.
- [174] R. Han, Y. Gan, J. Li, F. Wang, W. Feng, and S. Wang, “Connecting the complementary-view videos: Joint camera identification and subject association,” in IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 2406–2415.
- [175] J. Redmon, S. K. Divvala, R. B. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” IEEE Conf. Comput. Vis. Pattern Recog., pp. 779–788, 2015.
- [176] A. Zhan, R. Zhao, L. Pinto, P. Abbeel, and M. Laskin, “Learning visual robotic control efficiently with contrastive pre-training and data augmentation,” in IEEE Int. Conf. Intell. Rob. Syst., 2022, pp. 4040– 4047.
- [177] Y. Seo, J. Kim, S. James, K. Lee, J. Shin, and P. Abbeel, “Multi-view masked world models for visual robotic manipulation,” in Int. Conf. Mach. Learn., vol. 202, 23–29 Jul 2023, pp. 30613–30632.
- [178] A. Brohan et al., “Rt-1: Robotics transformer for real-world control at scale,” 2022, arXiv:2212.06817.
- [179] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware,” in Proc. Robot.: Sci. Syst., Daegu, Republic of Korea, July 2023.
- [180] H. Bharadhwaj, J. Vakil, M. Sharma, A. Gupta, S. Tulsiani, and V. Kumar, “Roboagent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking,” in Int. Conf. Robot. Autom., 2024, pp. 4788–4795.

- [181] C. Chi et al., “Diffusion policy: Visuomotor policy learning via action diffusion,” in Proc. Robot.: Sci. Syst., 2023.
- [182] T. Z. Zhao et al., “Aloha unleashed: A simple recipe for robot dexterity,” 2024, arXiv:2410.13126.
- [183] H.-N. Hu, Y.-C. Lin, M.-Y. Liu, H.-T. Cheng, Y.-J. Chang, and M. Sun, “Deep 360 pilot: Learning a deep agent for piloting through 360 sports videos,” in IEEE Conf. Comput. Vis. Pattern Recog., 2017, pp. 1396– 1405.
- [184] Y.-C. Su and K. Grauman, “Making 360 video watchable in 2d: Learning videography for click free viewing,” in IEEE Conf. Comput. Vis. Pattern Recog., 2017, pp. 1368–1376.
- [185] R. Yus, E. Mena, S. Ilarri, A. Illarramendi, and J. Bernad, “Multicamba: a system for selecting camera views in live broadcasting of sport events using a dynamic 3d model,” Multimedia Tools and Applications, vol. 74, pp. 4059–4090, 2015.
- [186] J. Chen, K. Lu, S. Tian, and J. Little, “Learning sports camera selection from internet videos,” in IEEE Winter Conf. Appl. Comput. Vis. IEEE, 2019, pp. 1682–1691.
- [187] S. Majumder, T. Nagarajan, Z. Al-Halah, R. Pradhan, and K. Grauman, “Which viewpoint shows it best? language for weakly supervising view selection in multi-view videos,” 2024, arXiv: 2411.08753.
- [188] S. Majumder, T. Nagarajan, Z. Al-Halah, and K. Grauman, “Switcha-view: Few-shot view selection learned from edited videos,” 2024, arXiv: 2412.18386.
- [189] B. Jia, Y. Chen, S. Huang, Y. Zhu, and S.-c. Zhu, “Lemma: A multiview dataset for le arning m ulti-agent m ulti-task a ctivities,” in Eur. Conf. Comput. Vis. Springer, 2020, pp. 767–786.
- [190] Z. Guo, Y. Hou, P. Wang, Z. Gao, M. Xu, and W. Li, “Ft-hid: a large-scale rgb-d dataset for first-and third-person human interaction analysis,” Neural Computing and Applications, pp. 2007–2024, 2023.
- [191] Q. Kong et al., “Wts: A pedestrian-centric traffic video dataset for fine-grained spatial-temporal understanding,” 2024, arXiv:2407.15350.
- [192] Y. Huang et al., “Egoexolearn: A dataset for bridging asynchronous ego-and exo-centric view of procedural activities in real world,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 22072–22086.
- [193] M. Ilaslan et al., “Gazevqa: A video question answering dataset for multiview eye-gaze task-oriented collaborations,” in Conf. Empir. Methods Nat. Lang. Process., Proc., 2023, pp. 10462–10479.
- [194] Y. Huang, M. Cai, Z. Li, and Y. Sato, “Predicting gaze in egocentric video by learning task-dependent attention transition,” in Eur. Conf. Comput. Vis. Springer, 2018, pp. 789–804.
- [195] F. D. la Torre Frade et al., “Guide to the carnegie mellon university multimodal activity (cmu-mmac) database,” Carnegie Mellon University, Pittsburgh, PA, Tech. Rep. CMU-RI-TR-08-22, April 2008.
- [196] T. Kwon, B. Tekin, J. St¨uhmer, F. Bogo, and M. Pollefeys, “H2o: Two hands manipulating objects for first person interaction recognition,” Int. Conf. Comput. Vis., pp. 10118–10128, 2021.
- [197] F. Sener et al., “Assembly101: A large-scale multi-view video dataset for understanding procedural activities,” in IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 21096–21106.
- [198] Z. Fan et al., “ARCTIC: A dataset for dexterous bimanual hand-object manipulation,” in IEEE Conf. Comput. Vis. Pattern Recog., 2023.
- [199] X. Zhan et al., “Oakink2: A dataset of bimanual hands-object manipulation in complex task completion,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 445–456.
- [200] N. Rai et al., “Home action genome: Cooperative compositional action understanding,” in IEEE Conf. Comput. Vis. Pattern Recog., 2021, pp. 11179–11188.
- [201] Y.-M. Li, W.-J. Huang, A.-L. Wang, L.-A. Zeng, J.-K. Meng, and W.-S. Zheng, “Egoexo-fitness: Towards egocentric and exocentric full-body action understanding,” 2024, arXiv:2406.08877.
- [202] G. A. Sigurdsson, G. Varol, X. Wang, A. Farhadi, I. Laptev, and A. K. Gupta, “Hollywood in homes: Crowdsourcing data collection for activity understanding,” in Eur. Conf. Comput. Vis., 2016.
- [203] C. Zhang, Y. Liu, R. Xing, B. Tang, and L. Yi, “Core4d: A 4d humanobject-human interaction dataset for collaborative object rearrangement,” 2024, arXiv:2406.19353.
- [204] H. Qiu, Z. Shi, L. Wang, H. Xiong, X. Li, and H. Li, “Egome: Follow me via egocentric view in real world,” 2025, arXiv: 2501.19061.
- [205] J. Wang, L. Liu, W. Xu, K. Sarkar, D. Luvizon, and C. Theobalt, “Estimating egocentric 3d human pose in the wild with external weak supervision,” in IEEE Conf. Comput. Vis. Pattern Recog., 2022, pp. 13157–13166.
- [206] T. Ohkawa, K. He, F. Sener, T. Hodan, L. Tran, and C. Keskin, “Assemblyhands: Towards egocentric activity understanding via 3d hand pose estimation,” in IEEE Conf. Comput. Vis. Pattern Recog., 2023, pp. 12999–13008.

- [207] F. Ding, Y. Zhu, X. Wen, and C. X. Lu, “Thermohands: A benchmark for 3d hand pose estimation from egocentric thermal image,” 2024, arXiv:2403.09871.
- [208] L. Ma et al., “Nymeria: A massive collection of multimodal egocentric daily motion in the wild,” in Eur. Conf. Comput. Vis., 2024.
- [209] D. Dwibedi, Y. Aytar, J. Tompson, and A. Zisserman, “Ovr: A dataset for open vocabulary temporal repetition counting in videos,” 2024, arXiv:2407.17085.
- [210] H. Luo, W. Zhai, J. Zhang, Y. Cao, and D. Tao, “One-shot affordance detection,” 2021, arXiv:2106.14747.
- [211] H. Chen, Y. Hou, C. Qu, I. Testini, X. Hong, and J. Jiao, “360 + x: A panoptic multi-modal scene understanding dataset,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 19373–19382.
- [212] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Adv. Neural Inform. Process. Syst., vol. 36, 2024.
- [213] Z. Chen et al., “Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks,” in IEEE Conf. Comput. Vis. Pattern Recog., 2024, pp. 24185–24198.
- [214] Z. Li et al., “Eagle 2: Building post-training data strategies from scratch for frontier vision-language models,” 2025, arXiv: 2501.14818.
- [215] G. Chen et al., “Videollm: Modeling video sequence with large language models,” 2023, arXiv: 2305.13292.
- [216] Y. Luo et al., “Video-rag: Visually-aligned retrieval-augmented long video comprehension,” 2024, arXiv:2411.13093.

