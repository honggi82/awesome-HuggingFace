## Robot Utility Models: General Policies for Zero-Shot Deployment in New Environments

Haritheja Etukuru*1, Norihito Naka1, Zijin Hu1, Seungjae Lee1, Julian Mehu2, Aaron Edsinger2, Chris Paxton2, Soumith Chintala3, Lerrel Pinto1 and Nur Muhammad “Mahi” Shafiullah*1,2 1New York University, 2Hello Robot Inc., 3Meta Inc. Project page: robotutilitymodels.com

Robot models, particularly those trained with large amounts of data, have recently shown a plethora of realworld manipulation and navigation capabilities. Several independent efforts have shown that given sufficient training data in an environment, robot policies can generalize to demonstrated variations in that environment. However, needing to finetune robot models to every new environment stands in stark contrast to models in language or vision that can be deployed zero-shot for open-world problems. In this work, we present Robot Utility Models (RUMs), a framework for training and deploying zero-shot robot policies that can directly generalize to new environments without any finetuning. To create RUMs efficiently, we develop new tools to quickly collect data for mobile manipulation tasks, integrate such data into a policy with multi-modal imitation learning, and deploy policies on-device on Hello Robot Stretch, a cheap commodity robot, with an external mLLM verifier for retrying. We train five such utility models for opening cabinet doors, opening drawers, picking up napkins, picking up paper bags, and reorienting fallen objects. Our system, on average, achieves 90% success rate in unseen, novel environments interacting with unseen objects. Moreover, the utility models can also succeed in different robot and camera set-ups with no further data, training, or fine-tuning. Primary among our lessons are the importance of training data over training algorithm and policy class, guidance about data scaling, necessity for diverse yet high-quality demonstrations, and a recipe for robot introspection and retrying to improve performance on individual environments.

### arXiv:2409.05865v1[cs.RO]9Sep2024

# Robot Utility Models

Bag pick up

Object reorientation

[Figure 1]

[Figure 2]

#### train once, deploy zero-shot

2. Train multi-modal behavior generation model

1. Collect large, diverse, task speciﬁc dataset.

â RUM policy

[Figure 3]

[Figure 4]

[Figure 5]

3. Deploy in new environments, zero shot, with mLLM feedback

|mLLM|
|---|

Failure?

<restart> â

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

RUM policy

E.g. your home

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Drawer opening Door opening Tissue pick up

Figure 1: Robot Utility Models are trained on a diverse set of environments and objects, and then can be deployed in novel environments with novel objects without any further data or training.

Corresponding author: mahi@cs.nyu.edu. (*) denotes equal contribution.

##### 1. Introduction

We have seen rapid progress in training manipulation skills recently (Brohan et al., 2023; Zhao et al., 2023b; Fu et al., 2024a,b; Haldar et al., 2024; Kim et al., 2024; Lin et al., 2024), largely brought about by fitting deep networks on data collected by teleoperating robots (Mandlekar et al., 2018; Arunachalam et al., 2023b; Cheng et al., 2024; Iyer et al., 2024; Khazatsky et al., 2024). The mechanism for deploying such skills in new environments mimics the pretrain-then-finetune strategy first developed by the vision community circa 2014 (Girshick et al., 2014). There, models were first pretrained on ImageNet and then finetuned on task-specific data such as detection, segmentation, and pose estimation (Girshick et al., 2014; Gkioxari et al., 2014). In the context of robotics, this strategy involves pretraining on large robot datasets (Padalkar et al., 2023; Shafiullah et al., 2023; Walke et al., 2023; Khazatsky et al., 2024) to produce a robot foundation model, which is then fine-tuned on data collected in new environments or tasks (Shafiullah et al., 2023; Kim et al., 2024; Team et al., 2024). This need to fine-tune the foundation model for each and every new environment is limiting as it requires humans to collect data in the very environment where the robot is expected to perform. So while vision and language models have moved on to zero-shot deployments, i.e. without any environment-specific finetuning data, such a capability eludes most robot manipulators. This is not to say that there have not been attempts to create zero-shot manipulation models – several foundational work in grasping and pick-and-place (Mahler et al., 2017; Sundermeyer et al., 2021; Fang et al., 2023b) have tackled this problem albeit with a task-specific solution.

So what makes creating a general policy for an arbitrary task that can work zero-shot hard? First is the concern about sufficient data – the necessary amount of data to train such a general model could be large. Since collecting robot data is hard, creating a large dataset is also hard and often expensive since humans are usually tasked to collect robot demonstrations. Second, when a large dataset is collected in the open-world it would necessarily have large diversity and multiple modes in demonstrator behavior. Fitting a robot models on this diverse data is a challenge. Third, unlike vision and language, where the native form of data, i.e. images and text are largely standard, robotics is far from having a standard camera and hardware setup along with physical challenges of running models in realtime on onboard compute. Creating zero-shot models that can run with even minor changes to hardware setup between training and deployment requires careful attention to details. Finally, any model deployed zero shot on a novel environment naturally has a higher failure rate than a model that has been fine-tuned on that environment. Thus, to deploy a model zero-shot, it is important to have a mechanism for error detection and recovery.

In this work, we introduce Robot Utility Models (RUMs), a new framework for training focused and functional utility models to complete helpful tasks that can be deployed zero-shot without further training or fine-tuning in novel environments. This is done by taking a systems-first approach. To scale up our datasets without compromising on data quality, we develop a new tool, building on prior work in untethered data collection (Shafiullah et al., 2023; Chi et al., 2024). We train policies on these diverse dataset with state-of-the-art multi-modal behavior learning algorithms (Chi et al., 2023; Lee et al., 2024) and show how they can absorb and scale with large-scale demonstration data. Finally, we deploy the policy in multiple different environments out of the box, with self-critique via mLLMs (Guo et al., 2023) and retrying, showing how the policy can be robustly executed on cheap, general-purpose hardware. A selection of our trained models are available on the Hello Robot Stretch without much modifications. Beyond the default Stretch deployment, we also enable deployment on other robot arms, cameras, and lighting conditions, showing the generalizability of our approach.

Creating and deploying RUMs led us to several interesting lessons. First, we find that the quantity and quality of data is crucial for training a utility model, with the choice of model architecture being less critical. Second, we see that the diversity of the data collected is crucial for the model to generalize

to new environments, and more important than the raw quantity of data. Third, we find that the model can be made more capable in single environments by performing self-critique on the model performance with an independent model and retrying when appropriate.

To validate RUMs, we run a total of 2,950 robot rollouts in real-world environments including homes in New York City (NY), Jersey City (NJ), and Pittsburgh (PA). These experiments reveal the following:

- • We show that it is possible to create general Robot Utility Models with a moderate amount of data in the order of 1,000 demonstrations (Section 2). These RUMs achieve a 90% average success rate on zero-shot deployment in 25 novel environments (Section 3.1).
- • The success of RUMs relies primarily on two key techniques. First, the use of multi-modal policies

- (Section 2.3) provides a zero-shot success rate of 74.4% (Section 3.2). Second, the mLLM based selfcritique and retrying system (Section 2.4) further improves the success rate by 15.6% (Section 3.6).

- • While the overall framework for RUMs is straightforward, the devil is in the details, where we find gains from unexpected sources, e.g. data diversity vs. data quantity (Section 3.4 and 3.5).

To encourage the development of RUMs for a wider variety of tasks, our code, data, models, hardware designs, as well as our experiment and deployment videos are open sourced and can be found on our website: robotutilitymodels.com.

##### 2. Robot Utility Models

We take a full-stack approach to create Robot Utility Models. At its core, our system follows the imitation learning framework. However, to effectively scale imitation learning to the point where our trained policies are deployable zero-shot, we create new tools and techniques to improve data collection, model training, inference, and deployment.

###### 2.1. Data collection tool

One of the primary requirements of our system is to be able to scale up diverse yet accurate demonstration data for cheap. To this end, we continue on the evolutionary path of hand-held, portable data collection tools (Song et al., 2020; Young et al., 2020; Pari et al., 2021; Shafiullah et al., 2023; Chi et al., 2024) that let us quickly collect precise demonstrations. Following our previous work (Shafiullah et al., 2023), we call this tool Stick-v2, which is a hand-held data collection tool built out of an iPhone Pro and a bill of materials that adds up to $25. We combine inspirations from the quick deployability of Stick-v1, and the compact, handheld form factor of UMI gripper. For a detailed build instruction and the bill of materials, we refer the reader to the supplementary materials (Appendix A.5).

Our design decisions are predicated on a few factors: portability, convenience, and set-up speed. We experimentally found these factors to be important to quickly scale up robot datasets and training RUMs. As we show with experiments in Section 3.3, one of the most crucial aspect of data collection for RUMs is data diversity, i.e. collecting data from a large number of diverse environmets. Thus, it is crucial to have a portable tool that is easy to mass-print, carry, and deploy in a new environment. Secondly, it is important for the collected data to be accurate across many environments with many variations. Finally, it is important to minimize the “per-environment set-up time”, whether that time is spent setting up the data collection system, calibrating the camera, or the tool’s SLAM system.

For the above reason, we design our data collection tool, Stick-v2, around the ARKit API from the widely available and used iPhone Pro (Figure 2). Given its technical capabilities, the only digital component in our Stick-v2 is this iPhone, which makes our tool particularly robust to shipping and

[Figure 15]

Robot Utility Models: General Policies for Zero-Shot Deployment in New Environments

[Figure 16]

###### 3D printed chassis with cable-driven trigger Wrist mounted iPhone Pro Flexible ﬁngers

[Figure 17]

- Figure 2: Stick-v2, our data collection tool (left: real photo, right: render), is built out of an iPhone Pro and a bill of materials that adds up to $25. The tool is portable, robust, and makes it easy to start collecting data in a new environment in seconds.

handling. The iPhone, and therefore Stick-v2, can collect RGB video and depth data at up to 60 Hz and high precision 6D pose and position information from the ARKit API at up to 100Hz. To capture the gripper opening information, we trained an RGB-based model that predicts the gripper aperture from images. Furthermore, this data is automatically synchronized and timestamped by the iPhone without the need for any calibration. This allows us to collect data from a wide variety of environments with no set-up time. This is in contrast to other data collection tools based on visual SLAM systems which has limited precision and are non-robust around “textureless” scenes such as close to flat walls, ceilings, or corners (Young et al., 2020; Chi et al., 2024). Finally, not needing camera calibration makes our system deployable out-of-the-box in any environment, especially in the real world where the environment is not controlled. This enables us to, for example, collect data from retail home goods stores with minimal interruptions to enrich our datasets, which would be hindered if we had to calibrate the camera and odometry system for each new environment.

- 2.2. Collected datasets We collect data for each of our five tasks, which are as defined below:

- • Door opening: Open doors with a long handle, on e.g. cabinets and microwaves. Due to hardware limitations, our robot cannot open doors with round knobs, so we exclude them from our dataset.
- • Drawer opening: Open a drawer with a handle. We exclude drawers with knobs from our dataset for similar reasons as above.
- • Reorientation: Pick up a cylindrical object (e.g. bottle) lying on a flat surface and place it upright on the same surface.
- • Tissue pickup: Pick up a soft, flexible tissue paper from any tissue paper box.
- • Bag pickup: Pick up a kraft paper bag or similar other bags from a flat surface.

For each of our five RUMs, we focused on gathering approximately 1,000 demonstrations on approximately 40 environments, with about 25 demonstrations per environment on average. The only exceptions are door opening with 1,200 and drawer opening with 525 demonstrations. A small collection of such environments are shown in Figure 3. For the door opening task, we seeded this dataset

[Figure 18]

Robot Utility Models: General Policies for Zero-Shot Deployment in New Environments

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Door opening Drawer opening Object reorientation Tissue pick up Bag pick up

- Figure 3: A small sample of environment and objects from our collected dataset. We collect data for each of our five tasks on a diverse set of environments and objects using Stick-v2.

with the Homes of New York dataset (Shafiullah et al., 2023) as well as demonstrations collected during the Dobb·E experiments. For the other tasks, our dataset consists of new demonstrations collected using the Stick-v2 tool on a novel set of environments and objects. For demonstrations collected from the previous dataset by inexperienced data collectors, we do a manual quality check and exclude any environment that has a high number of low-quality demonstrations, such as failed demonstrations. Note that, to keep our experiments unbiased, we hold out test environments and objects and never collect any data on them. To gain quick insight on different task data we use for training, we created an interactive data diversity visualization tool: robotutilitymodels.com/data_diversity/.

###### 2.3. Model training

Given that our data is collected by a large set of demonstration collectors, conceptually it is important for the model to handle any resultant multi-modality in the dataset. In this work, we train a large set of policy classes on our datasets for each task. Among the policy classes, the best performing ones are VQ-BeT (Lee et al., 2024) and Diffusion Policy (DP) (Chi et al., 2023). We also train ACT (Zhao et al., 2023b) and MLP-BC policies on a limited set of tasks. Each policy class shares some features, such as a ResNet34-based vision encoder initialized to the HPR encoder from Shafiullah et al. (2023), and a transformer-based policy trunk. We also train each model for the same 500 epochs. Beyond that, we sweep to find the best hyperparameters for learning rate, history length, and chunk size, and use the recommended hyperparameters from the original papers for each model. Our final VQ-BeT models are trained on data subsampled at 3.75Hz, and uses 6 most recent frames of history to predict the next action. All of our models predict the action in relative 6D space for the robot end-effector, and absolute value in the range [0,1] for the gripper opening. We discuss the impact of choosing different training algorithms in Section 3.2. Training all of our models took between 24 and 48 hours on 2 Nvidia A100 GPUs on our cluster, with proportional speed-ups by using more GPUs or using more recent GPUs like H100s.

###### Trial 1 Trial 2

[Figure 26]

[Figure 27]

As the timesteps progress, does the robotic arm open the door AND is the robot arm grasping the handle in the LAST timestep? Please respond with only 'Yes' or ‘No'

As the timesteps progress, does the robotic arm open the door AND is the robot arm grasping the handle in the LAST timestep? Please respond with only 'Yes' or ‘No'

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

No

Yes

[Figure 38]

[Figure 39]

<Reset and retry>

<Terminate>

Robot Utility Model Multimodal LLM (gpt-4o-2024-05-13)

- Figure 4: Automated retrying with feedback from multimodal LLM critic. We use a multimodal LLM (gpt-4o-2024-05-13 in our experiments) to verify the success of a task given a summary of robot observations. If the mLLM detects a failure, we automatically reset the robot and retry the task with a new initial robot state until success or timeout.

###### 2.4. Retrying with GPT-4o feedback

While a pre-trained model can solve the task in a new environment, to achieve the best possible performance, it is helpful to have additional runtime support for the model. For our deployment, we use an multimodal LLM (gpt-4o-2024-05-13) as an introspection module for our policies for a success detection and retrying mechanism. We define a single verification prompt for each task, and ask the mLLM to verify the success of the task given a summary of robot observations. As for the run summary, we give the mLLM every other frame from the robot camera, which is either from the head or the wrist camera depending on the task. If the mLLM detects a failure (Figure 4), RUM automatically resets the robot to a home position and retries the task with a new initial robot state.

###### 2.5. Deployment Details

Our primary hardware for Robot Utility Models deployment is the Hello Robot: Stretch robots with an iPhone on the wrist, but we support deploying our models on any robot arm with relative 6D pose and position control (Figure 5). We design and release an associated robot end-effector that can be mounted on standard robot arms, such as the xArm or Franka Panda. Similarly, while we primarily use the iPhone Pro as the deployment camera, we also show deployment on other wrist cameras, such as the Intel Realsense D405, which is the default wrist camera on Hello Stretch Edition

- 3 onwards. Overall, our deployment hardware system really relies on three things: our end-effector with a flexible two-fingered gripper and gripper tips, a wrist camera with a sufficient field of view, and an arm with six degrees of freedom to mount our wrist. We release default integration code for Hello Stretch 3 and an xArm wrist mount that we created, which should serve as illustrative examples for other arms.

Robot arm with 6D pose & position control

Wrist mounted camera

[Figure 40]

[Figure 41]

Flexible ﬁngers

###### Hello Robot: Stretch UFactory xArm 7

(Default gripper) (Custom gripper)

- Figure 5: Picture of the some robot setups where our Robot Utility Models can be deployed. We show the Hello Robot: Stretch, and the xArm 7 robot with iPhone Pros on the wrist. Beyond these, we also deploy on Stretch robots with default D405 wrist cameras.

##### 3. Capabilities of Robot Utility Models

To understand the capabilities of RUMs, we evaluate each of our models on a diverse set of environments. At the same time, we try to examine our recipe for training utility models and answer a set of questions about the trained models by running a set of ablation experiments. The primary questions that we try to answer are the following:

- • How well do Robot Utility Models solve a task in an unseen environment while operating on unseen objects?
- • What is the relative importance of different components of Robot Utility Models, such as training data, training algorithm, and self-verification?

- – What scale of data is needed to train capable RUMs?
- – What properties of data are most important for training RUMs?
- – How does mLLM-based self-critique affect RUMs, and where does it succeed or fail?

- • How well can we deploy RUMs on new robot embodiments?

Evaluation details: For each task, we set up 25 novel environments – five for each task – with objects and props not seen in the training dataset. To create these evaluation environments, we take the robot to previously unseen kitchens, purchase new furniture online (door and drawer opening), and source new objects manually verified to not be in the training set (reorientation, bag and tissue pick up). We show sample pictures of each of the environments and objects on our Appendix A.2. We evaluate each system and policy for 10 trials in each of these environments, starting from the same grid of starting positions facing the task space used by Shafiullah et al. (2023) as we show in Appendix Figure 15. For the retrying-based experiments, while RUMs take 1.31 tries in average to succeed (Section 3.6), we set a 10-try timeout to avoid getting stuck in infinite retry loops.

###### 3.1. Zero-shot evaluation of RUMs on unseen environments

The most important test of capability for a Robot Utility Model is whether such a model is capable of solving the target task in a new environment operating on new objects. We test for this capability by running our RUMs on our set of 25 eval environments and objects not seen during training.

Chart 1

100

X X X X X

X X X X X

X X X X

X X X

X X X

Average

X

90

80

X

Successrate(%)

X

X

60

X

86 94 94 92 84

40

20

0

Reorientation Drawer opening Door opening Tissue pick up Bag pick up

Figure 6: Success rate of Robot Utility Models on average over five novel scenes in five different tasks. The X’s on the figure denote success rates from individual environments.

On Figure 6, we see that on unseen and novel environments, RUMs perform well, achieving a 90% success rate overall, and ranging between 84% to 94% on individual tasks. We discuss some of the failure cases we observe in the Appendix Section A.7. Additionally, we show the performance of RUMs on each test environment on Table 1, showing that across all of our evaluation experiments, RUMs achieves some success in every environment. This success implies that our policies have a general idea of solving the target task; then such policies are further boosted with post-training methods (Section 3.6). factors separately: the raw perfo n and retrying on the performance

VQ-BeT

Table 1

|On all of our followi|ngVQ-BeTexperiments,|weVQ-BeTtrystddevto unders|VQ-BeTtand thesestderr two|
|---|---|---|---|
|performanceReorientationof the under|lying RUM polici86|es, and the effec9|t of introspection4|
|DrawerofopeningRUMs.|94|9|4|
|Door opening|94|13|6|
|ofTissuepolicypick uparchitecture|and training92m|ethod on RUM13|s 6|
|Bag pick up|84|15|7|

###### 3.2. Effect o

Once we have verified that RUMs can actually solve tasks in novel environments, we investigate the relative importance of different components within the training recipe. In particular, we compare the raw performance of different policy architectures on our dataset without the introspection component. We train a set of policy classes on our datasets for each task, including VQ-BeT (Lee et al., 2024), Diffusion Policy (DP) (Chi et al., 2023), and as baselines, ACT (Zhao et al., 2023b) and MLP-BC on two of the tasks. We show the relative comparison of the base success rates of different policy architectures, without retrying, in Figure 7 and 8.

As we see in Figure 7, VQ-BeT and DP are the top two algorithms in terms of performance, with comparable performance in most tasks and overlapping error bars. Moreover, we see from Figure 8 that while ACT and MLP-BC are not exactly on par, they are not far behind either. This observation implies that with training data of sufficient quality, the choice of algorithm may not be a make-or-break decision, and more energy should be spent on collecting diverse and accurate data. While we have similar performances on the test environment, we use VQ-BeT over DP for our final models due the higher performance and a lower latency on the robot CPU itself during deployment.

1

100

75

Successrate(%)

50

25

0

Reorientation Drawer opening Door opening Tissue pick up Bag pick up Diﬀusion Policy VQ-BeT

- Figure 7: Relative comparison of the success rate (with standard error) of different policy architectures on our dataset on all five tasks without automated error correction. We see that the performance of VQ-BeT and Diffusion Policy is generally close, with VQ-BeT narrowly outperforming Diffusion Policy.

Chart 1

Reorientation

Tissue pick up

Success rate (%)

0 20 40 60 80 100

MLP-BC ACT Diﬀusion Policy VQ-BeT

Table 1

|implies that the tra|iningDiﬀusiondataPolicyis sig|VQ-BeTnificantly more|MLP-BCimportant than|ACTtraining|
|---|---|---|---|---|
|Reorientation|62|68|44|48|
|Tissue pick up|72|80|64|66|

2

- Figure 8: Relative comparison of different policy architectures on our dataset on two tasks without automated error correction. We see that while the performance of VQ-BeT and Diffusion Policy is generally neck-to-neck, while the performance of other algorithms is not far behind. Our experiment

Table 1

###### Diﬀusion Policy VQ-BeT DP stddev VQ stddev DP stderr VQ stderr

Reorientation 62 68 31.144823 18.16590212 13.92838827503988.12403840258551 Drawer opening 66 76 28.80972058 33.61547263 12.884098725930915.0332963792927 Door opening 62 76 23.87467277 21.67948339 10.67707825085669.69535971542352 Tissue pick up 72 80 38.98717738 12.24744871 17.43559577450445.47722557330042 Bag pick up 82 84 20.49390153 11.40175425 9.165151389053395.09901951314943

algorithm.

###### 3.3. Effect of scaling datasets on RUMs

As our experiments show the importance of training data in creating RUMs, we investigate the properties of the dataset that a successful RUMs relies on. In particular, we dig into the scale of dataset at which reliable generalization emerges, and how RUMs’ performance vary with dataset size. We train our policies on a random subset of environments from the task-specific datasets, and evaluate them on our evaluation environments.

In Figure 9, we show the performance of VQ-BeT and Diffusion Policy without retrying trained on such data subsets on our evaluation environments as we scale up the dataset. We see that while Diffusion

9

1

Diﬀusion VQ-BeT

Diﬀusion VQ-BeT

Diﬀusion VQ-BeT

100

100

100

Successrate(%)

75

75

75

50

50

50

25

25

25

| | |
|---|---|
| | |

0

0

0

20% 40% 60% 80% 100%

25% 50% 75% 100%

20% 40% 60% 80% 100%

Data usage (% of full dataset)

Data usage (% of full dataset)

Data usage (% of full dataset)

###### Door opening

Tissue pick up

###### Reorientation

- Figure 9: Understanding the performance change of RUMs as the dataset scales up on three of our tasks, with standard error on error bars. We see better performance from Diffusion Policy (DP) on smaller datasets, but as we scale up, VQ-BeT outperforms DP in 900–1,200 demonstrations limit.

Successrate(%)

0

25

50

75

100

Door opening Object reorientation

18

64 68

76

Diverse data (25 demo/env)

Uniform data (200 demo/env)

Table 1

|performance, con|traryDiversedatato comm|onConcentratedknowledge.data|Co-training|Drop|
|---|---|---|---|---|
|Door opening|76|64|76|12|
|Object reorientation|68|18|34|50|

0

25

50

75

100

Door opening Drawer opening

34

76 76

52

32 22

Non-expert data Expert data Co-training

1

- Figure 10: Understanding the importance of different qualities of data in training RUMs. On the left, we see that diverse datasets are more valuable than more uniform datasets, with strong effects on the reorientation task with many unseen environments and object. On the right, we see that usually expert data is more valuable than non-expert or play data while learning behavior on a same sized dataset. Moreover, we see that co-training with expert data and play data may sometimes reduce the policy

Table 2

Door Opening

Reorientation

###### Diﬀusion VQ-BeT

25% 46 22 12.08304597 5.830951895 50% 64 24 15.03329638 6.782329983 75% 52 38 20.34698995 11.5758369

###### Diﬀusion VQ-BeT

###### Diﬀusion VQ-BeT

20% 42 8 24 14.35270009 40% 48 10.67707825 22 7.348469228 60% 58 11.13552873 38 9.695359715 80% 58 9.121403401 64 11.22497216

20% 20 12 7.071067812 2 40% 38 24 7.348469228 5.099019514 60% 52 38 9.695359715 5.830951895 80% 62 64 10.67707825 6.782329983

100% 76 84 7.483314774 10.09950494

100% 62 13.92838828 76 8.124038405

100% 60 68 10.29563014 9.695359715

Policy performs better on smaller datasets, it saturates on larger datasets where VQ-BeT outperforms it. This observation implies that while a smaller dataset may be sufficient for training a capable RUMs, a larger dataset is crucial for achieving the best performance. Even on our largest datasets, we see that the performance of VQ-BeT continues to improve as the dataset scales up, implying that more data may improve RUMs even further.

###### 3.4. Importance of data diversity in training RUMs

Beyond the scale of the dataset, we also investigate how the diversity of the training data impacts the performance of RUMs in Figure 10 (left). We create two alternate datasets of equal size for the door opening and the object reorientation tasks. The first datasets are composed of a large number of diverse environments with roughly 25 demonstrations in each environment. The second dataset is composed of fewer, between 5 and 6, distinct environments with roughly 200 demonstrations on each environment. We see that on the door opening task, where the scene diversity is narrower, both diverse and uniform environment trained policies performed well. However, in the reorientation task, with many different unseen environments and objects, only diverse-environment trained RUM policy performs well – the policy trained on more uniform environments experiences a 50% performance drop. This result implies that to train an effective RUM, collecting a diverse dataset is important.

10

1

###### 3.5. Impact of using expert demonstrations on training policies

While scaling up the dataset size and diversity is important for training RUMs, an important question to consider is the quality of the training dataset. Namely, while it may be easy to collect a large number of demonstrations by a large number of demonstrators, the quality of the demonstrations may vary. In this section, we investigate the value of using expert demonstrations in training RUMs.

- In Figure 10 (right) we compare the performance of RUMs trained on roughly 500 demonstrations, where the data is either sampled from expert or non-expert demonstration collectors. Here, “expertise” is defined as experience deploying Dobb·E policies on the robot. We see that in general, expert data is more valuable than non-expert data, with expert data outperforming non-expert data in all tasks. Moreover, we see that co-training with expert and non-expert data can sometimes, but not always, improve the performance of the policy. This observation implies depending on the task, data quality can have different levels of suboptimality, and in extreme cases may even hurt performance in co-training, which goes against a common practice in some earlier works (Zhao et al., 2023b; Khazatsky et al., 2024).

3.6. Effects of introspection and retrying with self-critique in RUMs

Table 1

|Task|Mean tries to success|False positive r|ateImprovement rate|
|---|---|---|---|
|Object reorientation<br><br>is predict|1.348837209<br><br>ing false pos|0%<br><br>itives: c|18<br><br>lassifying|
|average,Draweropening4.8%|of our1.617021277trajec|tories4%ex|hibit such20|
|Door opening<br><br>Figure 11|1.382978723<br><br>(right).|2.86%|26|
|Tissue pick up|1.173913043|7%|10|
|Bag pick up|1.047619048|10%|4|

0.5

1.0

1.5

2.0

Mean tries to success

2.5%

5%

7.5%

10%

False positive rate Object reorientation Drawer opening Door opening Tissue pick up Bag pick up

+7%

+14%

+21%

+28%

Improvement rate

+15.6%

4.8%

1.31

1

Figure 11: Understanding the details of introspection and retrying in RUMs. On the left, we see that retrying improves the performance of RUMs significantly, with an average 15.6% improvement. In the middle, we see that with retrying, most tasks get solved quite fast, on average with 1.31 tries. On the right, we see that while the mLLM is able to help, it can also have false positives (4.8% average over five tasks) which may let some errors slip past.

In RUMs, we are using a multimodal large language model (mLLM) as a self-critique method to identify failures. However, a pretrained mLLM in practice is just another layer of fail-safe for our robot deployment, and not a guarantee of success in itself. Thus, in this section we try to understand how it helps, and how such introspection method can fail.

- In Figure 11 (left), we can see the improvement rate of using self-critique over simply using the RUM policies without any retrying mechanism. On average over our 5 tasks, we see a 15.6% improvement over simply using RUM policies. While retrying is crucial to a higher success rate, a system that is stuck retrying for a long time is much less useful. Thankfully, on average, when RUMs succeeds, it does so within 1.31 tries on average, as we see from Figure 11 (middle). Finally, we analyze the primary failure mode of mLLMs, which a trajectory as a success when it’s actually a failure. On ave behavior, constituting of half of the total errors, as seen on

11

###### 3.7. Transferring RUMs to different embodiments Chart 1

Finally, we investigate the ability of RUMs to be transferred to different embodiments and cameras. We test the performance of two RUMs on the other robot setup shown in Figure 5: UFactory xArm 7, which is different from the Hello Robot Stretch setup we run other experiments on. We see that RUMs can be transferred to different embodiments and cameras with minimal loss in performance: roughly 10% drop in performance in both cases without corrective mLLM feedback, as shown in Figure 12. We expect combining RUMs with the mLLM self-critique would result in similar increase in performance in other embodiments as well; in fact, with an external third person camera, we expect to see a higher portion of the errors being experiment implies that RUMs can be easily deployed on different robots effort, making it a versatile tool for a wide range of robotic applications.

100

Successrate(%)

80 84

75

70 76

50

25

0

Tissue pick up Bag pick up

Hello Robot: Stretch UFactory xArm 7

Figure 12: Performance of RUMs without corrections on different embodiments as shown in Figure 5: RUMs can transfer to different embodiments with minimal loss in performance.

Table 1

|caught and co|Hello Robot: Stretch<br><br>rrected. Th|UFactory xArm 7<br><br>is| |
|---|---|---|---|
|Tissue pick up|80|70| |
|Bagpickandup cameras|with minim84|al 76| |

##### 4. Related works

Large Scale Data Collection: The data acquisition pipeline represents one of the most critical element of a data-driven robot learning framework. Previous works has employed a diverse array of data acquisition techniques, combining many open-sourced datasets across diverse simulation or real-world data including diverse robot embodiment from many institutions across the globe (Reed et al., 2022; Brohan et al., 2023; Padalkar et al., 2023; Khazatsky et al., 2024).

The most common approaches to robot demonstration collection involves pairing the robot or endeffector with remote controller devices or kinematically isomorphic equipment. The devices utilized have a range of complexity and forms: they encompass full robotic exoskeletons (Ishiguro et al., 2020; Fang et al., 2023c; Zhao et al., 2023a), as well as simpler data collection tools (Wu et al., 2023; Zhao et al., 2023b; Fu et al., 2024b), and also methods that don’t require physically moving a robot (Song et al., 2020; Young et al., 2020; Pari et al., 2021; Shafiullah et al., 2023; Chi et al., 2024). Additionally, various control methods have also been employed, including the use of video game controllers (Sian et al., 2004; Liu et al., 2024a), Virtual Reality (VR) devices (Arunachalam

1

- et al., 2022; Cui et al., 2022; Arunachalam et al., 2023a; Cheng et al., 2024; Fu et al., 2024a; Iyer et al., 2024; Park and Agrawal, 2024; Yang et al., 2024), and mobile phones (Mandlekar et al., 2018).

While the most intuitive method is to physically move a real robot, it is both difficult to do and hard to scale to a diverse set of environments. The hardware controller approach can be inefficient because it requires the demonstrator to mentally map robot behavior to controller inputs. The opposite, using a device without moving the robot is efficient in that the demonstrator’s movements can be mapped directly to the robot, but it is challenging to apply force feedback. Studies that provides perspective on the relative merits of these two direction are (Shafiullah et al., 2023; Chi et al., 2024), which combines the versatility of simple controller with the intuitiveness of moving a physical end-effector. In this work, we employ a device that inherits and improves the device proposed from (Shafiullah

- et al., 2023; Chi et al., 2024) for our data collection pipeline.

Pretrained Robot Models: Pre-trained foundation models have demonstrated a wide range of generalization performance across various domains, with the capability to learn from internet-scale

pre-training data (Devlin et al., 2018; Radford et al., 2021; Kirillov et al., 2023; Dubey et al., 2024). However, in comparison to these vision and language pre-trained models, learning a foundation model for robotics has been considered a relatively challenging area, due to the limited quantity of available datasets (Kappler et al., 2015; Levine et al., 2016; Depierre et al., 2018; Zhu et al., 2023), the significant discrepancy across the domains (Dasari et al., 2019; Kalashnikov et al., 2021; Padalkar et al., 2023), and the inherently challenging nature of the action datasets in terms of tokenization (Brohan et al., 2023; Lee et al., 2024; Zheng et al., 2024).

To address these issues, recent research is increasingly adopting techniques that introduce modular and hierarchical systems, incorporate pre-trained language and visual models (Nair et al., 2022b; Shafiullah et al., 2022; Karamcheti et al., 2023; Li et al., 2023; Gupta et al., 2024; Liu et al., 2024b), and collect large scale data with efficient data collection schemes (Ebert et al., 2022; Brohan et al., 2023; Fang et al., 2023a; Walke et al., 2023; Khazatsky et al., 2024). Consequently, they have enabled the pre-trained foundation robot models to exhibit enhanced generalization performance, thereby showcasing that the robotic agents are capable of operating in more than one robot embodiment and operating environment (Reed et al., 2022; Doshi et al., 2024; Kim et al., 2024; Team et al., 2024). In contrast with the aforementioned approaches, which follow a method of training on internet-scale data and fine-tuning on task-specific data, our approach does not expect that the model will have access to a dataset in the environments where the robot is expected to operate. Rather, this project demonstrates the capacity of generalizable performance without a necessity to fine-tune the model for each novel robot embodiment and environment.

Large Models Feedback and Improvement: Due to their capacity to comprehend intricate semantics and relations, Natural language and Large language models (LLM), have recently been applied to robotic agents powered by imitation learning (Fried et al., 2018; Jang et al., 2021; Shridhar et al., 2022; Kim et al., 2024) and reinforcement learning (Goyal et al., 2021; Du et al., 2023).

Among the wide capabilities afforded by language models, those commonly employed in the context of decision-making include providing feedback in the resolution of uncertain information (Huang et al., 2022b; Guo et al., 2023; Liu et al., 2023; Park et al., 2023; Ren et al., 2023; Gao et al., 2024; Mullen Jr and Manocha, 2024), suggesting affordance of what is possible in the environments by combining with Value functions (Ahn et al., 2022), and imagination of outcomes (Zhang et al., 2024) or planning and decompose complex tasks into mid-level plans (Sharma et al., 2021; Huang et al., 2022a; Zeng et al., 2022; Song et al., 2023). Language models could also be used to improve the overall performance of autonomous agent systems by improving reward signal (Goyal et al., 2021; Nair et al., 2022a; Ma et al., 2023), leveraging their long-horizon reasoning (Blukis et al., 2022; Zhou et al., 2023; Dalal et al., 2024), or designing environments (Ma et al., 2024). In this project, we employ the mLLM to provide feedback in the form of a reset signal in open-ended environments, a manner analogous to that of the studies above.

##### 5. Limitations and Conclusion

While in this work we create Robot Utility Models that can perform particular tasks zero-shot in novel environments, there are certain limitations that future versions can improve upon. The primary limitation that we see are of hardware: for example, two-fingered grippers like our Stick-v2 are unable to open doors with round doorknobs. Similarly, while flexible fingertips can be more lenient for the policy, it makes it hard to manipulate heavy objects. We encourage more research on better gripper and fingertip design to address these issues. Secondly, we assume navigation to be a separate component, and in this work assume that the robot is in the task space facing the task objective.

Combining with modular navigation work such as (Liu et al., 2024b) should address this issue. Finally, for mLLM introspection and retrying, we assume that the errors made by our model (a) leaves the task-space somewhat in-distribution, and (b) allows for an easy reset of the robot to the initial state. Increasing training data with failure recovery behavior in our dataset should let our robots recover more naturally from such failure cases.

##### Acknowledgements

We thank Shenglong Wang and the NYU HPC team for helping us with compute, Blaine Matulevic and Binit Shah for supporting our hardware needs, and Siddhant Haldar and Jeff Cui for providing feedback on the paper. NYU authors are supported by grants from Honda, Hyundai, NSF award 2339096 and ONR awards N00014-21-1-2758 and N00014-22-1-2773. MS is supported by the Apple Fellowship. LP is supported by the Packard Fellowship. SL is supported by the Daishin Songchon Foundation. Hello Robot authors are supported by NIH NIA R43AG072982.

##### References

Neo Ee Sian, Kazuhito Yokoi, Shuuji Kajita, Fumio Kanehiro, and Kazuo Tanie. Whole body teleoperation of a humanoid robot development of a simple master device using joysticks. Journal of the Robotics Society of Japan, 22(4):519–527, 2004.

Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 580–587, 2014.

Georgia Gkioxari, Bharath Hariharan, Ross B. Girshick, and Jitendra Malik. R-cnns for pose estimation and action detection. CoRR, abs/1406.5212, 2014. URL http://arxiv.org/abs/1406.5212.

Daniel Kappler, Jeannette Bohg, and Stefan Schaal. Leveraging big data for grasp planning. In ICRA, 2015.

Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. JMLR, 17(1):1334–1373, 2016.

Jeffrey Mahler, Jacky Liang, Sherdil Niyaz, Michael Laskey, Richard Doan, Xinyu Liu, Juan Aparicio Ojea, and Ken Goldberg. Dex-Net 2.0: Deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics. In Robotics: Science and Systems (RSS), 2017.

Amaury Depierre, Emmanuel Dellandréa, and Liming Chen. Jacquard: A large scale dataset for robotic grasp detection. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 3511–3516. IEEE, 2018.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, pages 4171–4186, 2018. doi: 10.18653/v1/N19-1423.

Daniel Fried, Ronghang Hu, Volkan Cirik, Anna Rohrbach, Jacob Andreas, Louis-Philippe Morency, Taylor Berg-Kirkpatrick, Kate Saenko, Dan Klein, and Trevor Darrell. Speaker-follower models for vision-and-language navigation. In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada, pages 3318–3329, 2018.

Ajay Mandlekar, Yuke Zhu, Animesh Garg, Jonathan Booher, Max Spero, Albert Tung, Julian Gao, John Emmons, Anchit Gupta, Emre Orbay, et al. Roboturk: A crowdsourcing platform for robotic skill learning through imitation. In Conference on Robot Learning, pages 879–893. PMLR, 2018.

Sudeep Dasari, Frederik Ebert, Stephen Tian, Suraj Nair, Bernadette Bucher, Karl Schmeckpeper, Siddharth Singh, Sergey Levine, and Chelsea Finn. RoboNet: Large-scale multi-robot learning. In Conference on Robot Learning (CoRL), volume 100, pages 885–897. PMLR, 2019.

Yasuhiro Ishiguro, Tasuku Makabe, Yuya Nagamatsu, Yuta Kojio, Kunio Kojima, Fumihito Sugai, Yohei Kakiuchi, Kei Okada, and Masayuki Inaba. Bilateral humanoid teleoperation system using whole-body exoskeleton cockpit tablis. IEEE Robotics and Automation Letters, 5(4):6419–6426, 2020.

Shuran Song, Andy Zeng, Johnny Lee, and Thomas Funkhouser. Grasping in the wild: Learning 6dof closed-loop grasping from low-cost demonstrations. IEEE Robotics and Automation Letters, 5(3): 4978–4985, 2020.

Sarah Young, Dhiraj Gandhi, Shubham Tulsiani, Abhinav Gupta, Pieter Abbeel, and Lerrel Pinto. Visual imitation made easy. arXiv e-prints, pages arXiv–2008, 2020.

Prasoon Goyal, Scott Niekum, and Raymond Mooney. Pixl2r: Guiding reinforcement learning using natural language by mapping pixels to rewards. In Conference on Robot Learning, pages 485–497. PMLR, 2021.

Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, and Chelsea Finn. BC-Z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning (CoRL), pages 991–1002, 2021.

Dmitry Kalashnikov, Jacob Varley, Yevgen Chebotar, Benjamin Swanson, Rico Jonschkowski, Chelsea Finn, Sergey Levine, and Karol Hausman. MT-Opt: Continuous multi-task robotic reinforcement learning at scale. arXiv preprint arXiv:2104.08212, 2021.

Jyothish Pari, Nur Muhammad Shafiullah, Sridhar Pandian Arunachalam, and Lerrel Pinto. The surprising effectiveness of representation learning for visual imitation, 2021.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), volume 139, pages 8748–8763, 2021.

Pratyusha Sharma, Antonio Torralba, and Jacob Andreas. Skill induction and planning with latent language. arXiv preprint arXiv:2110.01517, 2021.

Martin Sundermeyer, Arsalan Mousavian, Rudolph Triebel, and Dieter Fox. Contact-graspnet: Efficient 6-dof grasp generation in cluttered scenes. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 13438–13444. IEEE, 2021.

Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022.

Sridhar Pandian Arunachalam, Sneha Silwal, Ben Evans, and Lerrel Pinto. Dexterous imitation made easy: A learning-based framework for efficient dexterous manipulation. arXiv preprint arXiv:2203.13251, 2022.

Valts Blukis, Chris Paxton, Dieter Fox, Animesh Garg, and Yoav Artzi. A persistent spatial semantic representation for high-level natural language instruction execution. In Conference on Robot Learning, pages 706–717. PMLR, 2022.

Zichen Jeff Cui, Yibin Wang, Nur Muhammad Mahi Shafiullah, and Lerrel Pinto. From play to policy: Conditional behavior generation from uncurated robot data. arXiv preprint arXiv:2210.10047, 2022.

Frederik Ebert, Yanlai Yang, Karl Schmeckpeper, Bernadette Bucher, Georgios Georgakis, Kostas Daniilidis, Chelsea Finn, and Sergey Levine. Bridge data: Boosting generalization of robotic skills with cross-domain datasets. In Robotics: Science and Systems (RSS) XVIII, 2022.

Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In International conference on machine learning, pages 9118–9147. PMLR, 2022a.

Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, et al. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608, 2022b.

Suraj Nair, Eric Mitchell, Kevin Chen, Silvio Savarese, Chelsea Finn, et al. Learning languageconditioned robot behavior from offline data and crowd-sourced annotation. In Conference on Robot Learning, pages 1303–1315. PMLR, 2022a.

Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3m: A universal visual representation for robot manipulation. In CoRL, 2022b.

Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gómez Colmenarejo, Alexander Novikov, Gabriel Barth-maron, Mai Giménez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, Tom Eccles, Jake Bruce, Ali Razavi, Ashley Edwards, Nicolas Heess, Yutian Chen, Raia Hadsell, Oriol Vinyals, Mahyar Bordbar, and Nando de Freitas. A generalist agent. Transactions on Machine Learning Research, 2022. ISSN 2835-8856.

Nur Muhammad Mahi Shafiullah, Chris Paxton, Lerrel Pinto, Soumith Chintala, and Arthur Szlam. Clip-fields: Weakly supervised semantic fields for robotic memory. arXiv preprint arXiv:2210.05663, 2022.

Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Cliport: What and where pathways for robotic manipulation. In Conference on Robot Learning, pages 894–906. PMLR, 2022.

Andy Zeng, Maria Attarian, Brian Ichter, Krzysztof Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael Ryoo, Vikas Sindhwani, et al. Socratic models: Composing zeroshot multimodal reasoning with language. arXiv preprint arXiv:2204.00598, 2022.

Sridhar Pandian Arunachalam, Irmak Güzey, Soumith Chintala, and Lerrel Pinto. Holo-dex: Teaching dexterity with immersive mixed reality. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 5962–5969. IEEE, 2023a.

Sridhar Pandian Arunachalam, Sneha Silwal, Ben Evans, and Lerrel Pinto. Dexterous imitation made easy: A learning-based framework for efficient dexterous manipulation. In 2023 ieee international conference on robotics and automation (icra), pages 5954–5961. IEEE, 2023b.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. arXiv preprint arXiv:2303.04137, 2023.

Yuqing Du, Olivia Watkins, Zihan Wang, Cédric Colas, Trevor Darrell, Pieter Abbeel, Abhishek Gupta, and Jacob Andreas. Guiding pretraining in reinforcement learning with large language models. In International Conference on Machine Learning, pages 8657–8677. PMLR, 2023.

Hao-Shu Fang, Hongjie Fang, Zhenyu Tang, Jirong Liu, Junbo Wang, Haoyi Zhu, and Cewu Lu. RH20T: A robotic dataset for learning diverse skills in one-shot. In RSS 2023 Workshop on Learning for Task and Motion Planning, 2023a.

Hao-Shu Fang, Chenxi Wang, Hongjie Fang, Minghao Gou, Jirong Liu, Hengxu Yan, Wenhai Liu, Yichen Xie, and Cewu Lu. Anygrasp: Robust and efficient grasp perception in spatial and temporal domains. IEEE Transactions on Robotics, 2023b.

Hongjie Fang, Hao-Shu Fang, Yiming Wang, Jieji Ren, Jingjing Chen, Ruo Zhang, Weiming Wang, and Cewu Lu. Low-cost exoskeletons for learning whole-arm manipulation in the wild. arXiv preprint arXiv:2309.14975, 2023c.

Yanjiang Guo, Yen-Jen Wang, Lihan Zha, Zheyuan Jiang, and Jianyu Chen. Doremi: Grounding language model by detecting and recovering from plan-execution misalignment. arXiv preprint arXiv:2307.00329, 2023.

Siddharth Karamcheti, Suraj Nair, Annie S Chen, Thomas Kollar, Chelsea Finn, Dorsa Sadigh, and Percy Liang. Language-driven representation learning for robotics. Robotics: Science and Systems (RSS), 2023.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023.

Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, et al. Vision-language foundation models as effective robot imitators. arXiv preprint arXiv:2311.01378, 2023.

Zeyi Liu, Arpit Bahety, and Shuran Song. Reflect: Summarizing robot experiences for failure explanation and correction. arXiv preprint arXiv:2306.15724, 2023.

Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Eureka: Human-level reward design via coding large language models. arXiv preprint arXiv:2310.12931, 2023.

Abhishek Padalkar, Acorn Pooley, Ajinkya Jain, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anikait Singh, Anthony Brohan, et al. Open x-embodiment: Robotic learning datasets and rt-x models. arXiv preprint arXiv:2310.08864, 2023.

Jeongeun Park, Seungwon Lim, Joonhyung Lee, Sangbeom Park, Minsuk Chang, Youngjae Yu, and Sungjoon Choi. Clara: classifying and disambiguating user commands for reliable interactive robotic agents. IEEE Robotics and Automation Letters, 2023.

Allen Z Ren, Anushri Dixit, Alexandra Bodrova, Sumeet Singh, Stephen Tu, Noah Brown, Peng Xu, Leila Takayama, Fei Xia, Jake Varley, et al. Robots that ask for help: Uncertainty alignment for large language model planners. arXiv preprint arXiv:2307.01928, 2023.

Nur Muhammad Mahi Shafiullah, Anant Rai, Haritheja Etukuru, Yiqian Liu, Ishan Misra, Soumith Chintala, and Lerrel Pinto. On bringing robots home. arXiv preprint arXiv:2311.16098, 2023.

Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su. Llmplanner: Few-shot grounded planning for embodied agents with large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2998–3009, 2023.

Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao, Philippe Hansen-Estruch, Quan Vuong, Andre He, Vivek Myers, Kuan Fang, Chelsea Finn, and Sergey Levine. Bridgedata v2: A dataset for robot learning at scale, 2023.

Philipp Wu, Yide Shentu, Zhongke Yi, Xingyu Lin, and Pieter Abbeel. Gello: A general, low-cost, and

intuitive teleoperation framework for robot manipulators. arXiv preprint arXiv:2309.13037, 2023. Liang Zhao, Tie Yang, Yang Yang, and Peng Yu. A wearable upper limb exoskeleton for intuitive

teleoperation of anthropomorphic manipulators. Machines, 11(4):441, 2023a.

Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023b.

Haoyu Zhou, Mingyu Ding, Weikun Peng, Masayoshi Tomizuka, Lin Shao, and Chuang Gan. Generalizable long-horizon manipulations with large language models. arXiv preprint arXiv:2310.02264, 2023.

Xinghao Zhu, Ran Tian, Chenfeng Xu, Mingxiao Huo, Wei Zhan, Masayoshi Tomizuka, and Mingyu Ding. Fanuc manipulation: A dataset for learning-based manipulation with fanuc mate 200iD robot. https://sites.google.com/berkeley.edu/fanuc-manipulation, 2023.

Xuxin Cheng, Jialong Li, Shiqi Yang, Ge Yang, and Xiaolong Wang. Open-television: Teleoperation with immersive active visual feedback, 2024. URL https://arxiv.org/abs/2407.01512.

Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. arXiv preprint arXiv:2402.10329, 2024.

Murtaza Dalal, Tarun Chiruvolu, Devendra Chaplot, and Ruslan Salakhutdinov. Plan-seq-learn: Language model guided rl for solving long horizon robotics tasks. arXiv preprint arXiv:2405.01534, 2024.

Ria Doshi, Homer Walke, Oier Mees, Sudeep Dasari, and Sergey Levine. Scaling cross-embodied learning: One policy for manipulation, navigation, locomotion and aviation. arXiv preprint arXiv:2408.11812, 2024.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Zipeng Fu, Qingqing Zhao, Qi Wu, Gordon Wetzstein, and Chelsea Finn. Humanplus: Humanoid shadowing and imitation from humans. arXiv preprint arXiv:2406.10454, 2024a.

Zipeng Fu, Tony Z Zhao, and Chelsea Finn. Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation. arXiv preprint arXiv:2401.02117, 2024b.

Jensen Gao, Bidipta Sarkar, Fei Xia, Ted Xiao, Jiajun Wu, Brian Ichter, Anirudha Majumdar, and Dorsa Sadigh. Physically grounded vision-language models for robotic manipulation. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 12462–12469. IEEE, 2024.

Arjun Gupta, Michelle Zhang, Rishik Sathua, and Saurabh Gupta. Opening cabinets and drawers in the real world using a commodity mobile manipulator. arXiv preprint arXiv:2402.17767, 2024.

Siddhant Haldar, Zhuoran Peng, and Lerrel Pinto. Baku: An efficient transformer for multi-task policy learning. arXiv preprint arXiv:2406.07539, 2024.

Aadhithya Iyer, Zhuoran Peng, Yinlong Dai, Irmak Guzey, Siddhant Haldar, Soumith Chintala, and Lerrel Pinto. Open teach: A versatile teleoperation system for robotic manipulation. arXiv preprint arXiv:2403.07870, 2024.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246, 2024.

Seungjae Lee, Yibin Wang, Haritheja Etukuru, H Jin Kim, Nur Muhammad Mahi Shafiullah, and Lerrel Pinto. Behavior generation with latent actions. arXiv preprint arXiv:2403.03181, 2024.

Toru Lin, Yu Zhang, Qiyang Li, Haozhi Qi, Brent Yi, Sergey Levine, and Jitendra Malik. Learning visuotactile skills with two multifingered hands. arXiv preprint arXiv:2404.16823, 2024.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36, 2024a.

Peiqi Liu, Yaswanth Orru, Chris Paxton, Nur Muhammad Mahi Shafiullah, and Lerrel Pinto. Okrobot: What really matters in integrating open-knowledge models for robotics. arXiv preprint arXiv:2401.12202, 2024b.

Yecheng Jason Ma, William Liang, Hung-Ju Wang, Sam Wang, Yuke Zhu, Linxi Fan, Osbert Bastani, and Dinesh Jayaraman. Dreureka: Language model guided sim-to-real transfer. arXiv preprint arXiv:2406.01967, 2024.

James F Mullen Jr and Dinesh Manocha. Towards robots that know when they need help: Affordance-

based uncertainty for large language model planners. arXiv preprint arXiv:2403.13198, 2024. Younghyo Park and Pulkit Agrawal. Using apple vision pro to train and control robots, 2024. Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari,

Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

Shiqi Yang, Minghuan Liu, Yuzhe Qin, Runyu Ding, Jialong Li, Xuxin Cheng, Ruihan Yang, Sha Yi, and Xiaolong Wang. Ace: A cross-platform visual-exoskeletons system for low-cost dexterous teleoperation, 2024. URL https://arxiv.org/abs/2408.11805.

Ceng Zhang, Xin Meng, Dongchen Qi, and Gregory S Chirikjian. Rail: Robot affordance imagination with large language models. arXiv preprint arXiv:2403.19369, 2024.

Ruijie Zheng, Ching-An Cheng, Hal Daumé III, Furong Huang, and Andrey Kolobov. Prise: Llmstyle sequence compression for learning temporal action abstractions in control. In Forty-first International Conference on Machine Learning, 2024.

##### A. Appendix

- A.1. Detailed Results from Experiments with Self-critique and Retrying

Task Environment/Object Success ·/10 Door Opening Kitchen Trash Door 7

Kitchen Cabinet Door 10 Brown Cabinet Door 10 Metal Cabinet Door 10 White File Cabinet Door 10

Drawer Opening Kitchen Drawer 10 Cloth Drawer 9 White File Cabinet Drawer 10 Small File Cabinet Drawer 10 Dresser Drawer 8

Bag Pick Up Hollister Bag 9 American Eagle Bag 10 Qdoba Bag 8 Journey’s Bag 9 Yellow Bag 6

Tissue Pick Up White Tall Box 10 White Short Box 10 Black Square Box 9 Red Square Box 10 Kleenex Box 7

Object Reorientation Pink Bottle 9 White Board Cleaner 8 Spices Container 8 Coke Can 8 Compressed Air 10

Table 1: Detailed success statistics of RUMs on our evaluation environments.

###### A.2. Evaluation Environments

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Reorientation environments

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Drawer opening environments

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Door opening environments

- Figure 13: Picture of evaluation environments for the tasks Reorientation, Drawer opening, and Door opening.

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Tissue pick up environments

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Bag pick up environments

- Figure 14: Pictures of the evaluation environments for the task Tissue pick up and Bag pick up.

- A.3. Multimodal Large Language Model Prompts for Success Verification Here, we present the prompt that we use to verify RUMs success with mLLMs.

Door Opening

As the timesteps progress, does the robotic arm open the door AND is the robot arm grasping the handle in the LAST timestep? Please respond with only ’Yes’ or ’No’.

###### Drawer Opening

As the timesteps progress, does the robotic arm grasp the drawer handle and open it AND is the drawer open in the last timestep? Please respond with only ’Yes’ or ’No’.

###### Reorientation

As the timesteps progress, does the robotic arm/gripper reorient the object upright AND is the object upright in the LAST frame? Please respond with only ’Yes’ or ’No’.

###### Tissue Pick-Up

As the timesteps progress, does the robotic arm/gripper grasp the tissue AND is the gripper grasping the tissue in the LAST timestep? Please respond with only ’Yes’ or ’No’.

###### Bag Pick-Up

As the timesteps progress, does the robotic arm/gripper grasp the bag AND is the gripper grasping the bag in the LAST timestep? Please respond with only ’Yes’ or ’No’.

###### A.4. Evaluation Schedule

In Figure 15, we show the starting position of the robot for our 10-run evaluations to understand the positional generalization capabilities of Robot Utility Models.

[Figure 67]

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

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

- Figure 15: 10-run evaluation schedule used to evaluate Robot Utility Models, with robot starting positions denoted by the pale blue dots in the image. We assume that the robot is at the task space facing the object, but it can be at different offsets with respect to the target object. On our object centric tasks (reorientation, bag and tissue pickup) we also randomize the position of the object itself.

###### A.5. Bill of Materials

Here, we present the bill of materials for our hardware components, assuming that the interested researcher or user owns an iPhone Pro already. The total cost comes out to be slightly below $25 for the entire setup.

Item Price Unit Price Qty Reacher Grabber Tool 26.99 13.50 1 Brass Tapered Heat-Set Inserts 21.82 0.22 3 Thread-Forming Screws 7.75 0.31 3 Button Head Screw - M4 x 0.70 - 8mm 12.91 0.13 1 Button Head Screw - M4 x 0.70 - 5mm 8.64 0.09 2 Button Head Screw - M4 x 0.70 - 35mm 16.77 0.34 2 Nylon-Insert Locknut 5.57 0.06 2 Dowel Pin 16.09 0.32 3 Nylon Unthreaded Spacer 18.41 0.18 2 Kevlar Cord 20.99 20.99 1/100 Heat Shrink Tubing 10.79 10.79 1/30 Black 3D Printer Filament 25.99 25.99 3/20 Total 21.99

Table 2: Stick-v2 Main Body

Item Price Unit Price Qty Socket Head Screw - M3 x 0.5mm - 8mm 12.52 0.13 2 Steel Hex Nut - M3 x 0.5mm 2.62 0.03 2 M3 Steel Washer 2.19 0.02 2 Red 3D Printer Filament 25.99 25.99 3/1000 Oomoo 25 Silicone Rubber 33.99 33.99 1/200 Total 0.61

- Table 3: Gripper Tips

Item Price Unit Price Qty Socket Head Screw - M5 x 0.8mm - 20mm 17.10 0.17 1 Socket Head Screw - M5 x 0.8mm - 50mm 4.26 0.85 1 Steel Hex Nut - M5 x 0.8mm 5.24 0.05 2 Button Head Screw - M4 x 0.70 - 8mm 12.91 0.13 1 Black 3D Printer Filament 25.99 25.99 3/20 Total 2.03

- Table 4: Phone Holder

###### A.6. Deploying on Stretch’s Default D405 Camera

Deploying our Robot Utility Models on the standard Hello Robot Stretch SE3 requires normalizing the image coming out of the default Intel Realsense D405 wrist camera. We created an affine transformation that maps the D405 image to the same pixel coordinates as the iPhone camera.

###### Long range Short range

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Intel Realsense D405 (with aﬃne transform)

Intel Realsense D405 (with aﬃne transform)

iPhone Pro

iPhone Pro

- Figure 16: We can see the corresponding D405 camera image alongside the iPhone Pro image. While in the long range, the images look similar, in the short range iPhone images are out of focus because of the different focal lengths of the cameras.

As we can see from Figure 16, applying the affine transform to the D405 camera maps it to pretty similar viewpoint as the wrist mounted iPhone. While we can run RUMs directly with this camera transform, we see a performance drop which we hypothesize happens because of the especially apparent difference in close-range. This difference is caused by the different focal lengths of the two cameras, and may be solved in the future with image augmentations.

###### A.7. Failure Modes

[Figure 101]

[Figure 102]

[Figure 103]

Reorientation failure: dropped bottle o! the table, retry impossible

[Figure 104]

[Figure 105]

[Figure 106]

Tissue pick up failure: picked up tissue, pulled box o! the table

[Figure 107]

[Figure 108]

[Figure 109]

Tissue pick up failure: picked up tissue AND the box

- Figure 17: Examples of some failures in real world rollouts. Since RUMs retries on failure with mLLM feedback, the failure modes tend to be peculiar, some examples of which are shown here.

As we mention in the main paper, with mLLM guided retries, our failures tend to be more peculiar than simply “robot failed to complete task”. In this section, we try to shine some light on what kind of failures we experience in our system.

- • Reorientation: Primary failure modes for this task are when retry becomes impossible because of environmental issues, such as the target bottle rolling away on the table, being dropped off the surface (an example of which is shown on the Figure 17), pushing it too far into the table (to a position too far for our robot arm), or being rotated sideways by the gripper before grasping. In out-of-distribution surfaces, it can be hard to estimate how large the surface is visually and thus placing the object after reorientation may miss the surface or the robot may run into the surface.
- • Drawer opening: Beyond the most direct failure mode of missing the drawer handle, we experienced some failure modes where the model does not know when to stop pulling on cloth drawers and thus pulls out the entire drawer. Without force feedback, it can be hard to tell visually when the drawer starts sagging. Force feedback on the fingertips would help the robot correct for it.
- • Door opening: Here, the primary failure mode we experience are on unusual doors, such as the trash cabinet door with a hole in it. There, GPT sometimes classifies the door as “open” even when it is closed. In some rare cases, when door handles are close together, the robot may grasp around both handles and fail to reset as it gets stuck when retracting.
- • Tissue pick up: The tissue box itself being light and easy to move means that sometimes the box

- moves with the tissue as its being picked up. As a result, the box may get picked up with the tissue, or get pushed off from its table by the robot (Figure 17.)
- • Bag pick up: The case of bag picking up is interesting because it has one of the highest success rates from the raw RUM policy but also sees the smallest improvement (4%) from GPT feedback. This failure from mLLM feedback happens usually because from the robot wrist or head camera, it can be hard to tell whether the bag has been picked up. As a result, GPT tends to have a high number of false positives for this task. Having a better third-person view of the workspace should help address this issue.

