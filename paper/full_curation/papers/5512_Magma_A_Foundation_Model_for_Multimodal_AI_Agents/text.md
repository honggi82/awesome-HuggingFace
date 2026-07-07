# arXiv:2502.13130v1[cs.CV]18Feb2025

[Figure 1]

## Magma: A Foundation Model for Multimodal AI Agents

Jianwei Yang1*† Reuben Tan1† Qianhui Wu1† Ruijie Zheng2‡ Baolin Peng2‡ Yongyuan Liang2‡ Yu Gu1 Mu Cai3 Seonghyeon Ye4 Joel Jang5 Yuquan Deng5 Lars Liden1 Jianfeng Gao1▽ 1Microsoft Research, 2University of Maryland, 3University of Wisconsin-Madison 4KAIST, 5University of Washington https://microsoft.github.io/Magma

[Figure 2]

[Figure 3]

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

Physical Environment

User: What is they doing? Magma: They are camping in a forest and hiking on a rocky mountain.

User: On what date did the individual leave a place that Simon thought was very important to him?

User: Close the middle drawer

Magma: Trajectory Rollout on SimplerEnv Google Robot

Task: Push the chip bag to left Task: Put the can to left

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Magma: The date is June 9, 2021.

[Figure 20]

[Figure 21]

Temporal

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

User: The figure represents a 5x5 grid containing various animals, where each 1x1 square is considered a block and each block contains an animal from ['cat', 'dog', 'elephant', 'giraffe', 'rabbit']. What is the animal of the block located at the first row, second column of the grid?

Task: Peel the potato Task: Stir the food in the pot

User: Pick up the mushroom to the pot Magma: Trajectory Rollout on Widow-X Real Robot

Planning

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

M

[Figure 34]

Magma: The animal is a cat.

Spatial

User: Find baby shoes under 20 dollars and rated 5 stars.

Magma: Trajectory on Mind2Web

[Figure 35]

|[Figure 36]|
|---|

|[Figure 37]|
|---|

User: What is unusual thing about this image?

Task: Plan a visit to the museum

User: How to increase the sound?

[Figure 38]

[Figure 39]

Magma: Tap on the "Volume" icon. Then, slide the volume up to increase the volume.

Magma: The unusual aspect of this image is that a man is ironing clothes on the back of a yellow taxi. ….

Digital Environment

Task: Lock Screen Task: View my Account

Task: Install app "Instagram"

Magma: Trajectory on AITW

Multimodal Understanding

Multimodal Action Grounding

Multimodal Agentic Taks

Figure 1. We introduce Magma, the first foundation model that is capable of interpreting and grounding multimodal inputs within its environment. Given a described goal, Magma is able to formulate plans and execute actions to achieve it. By effectively transferring knowledge from freely available visual and language data, Magma bridges verbal and spatial intelligence to navigate complex tasks.

#### Abstract

We present Magma, a foundation model that serves multimodal AI agentic tasks in both the digital and physical worlds. Magma is a significant extension of vision-language (VL) models in that it not only retains the VL understanding ability (verbal intelligence) of the latter, but is also equipped with the ability to plan and act in the visualspatial world (spatial-temporal intelligence) and complete agentic tasks ranging from UI navigation to robot manipulation. To endow the agentic capabilities, Magma is pretrained on large amounts of heterogeneous datasets spanning from images, videos to robotics data, where the actionable visual objects (e.g., clickable buttons in GUI) in images are labeled by Set-of-Mark (SoM) for action grounding, and

† First Authors; ‡ Second Authors; ∗ Project Lead; ▽ Leadership

the object movements (e.g., the trace of human hands or robotic arms) in videos are labeled by Trace-of-Mark (ToM) for action planning. Extensive experiments show that SoM and ToM reach great synergy and facilitate the acquisition of spatial-temporal intelligence for our Magma model, which is fundamental to a wide range of tasks as shown in Fig. 1. In particular, Magma creates new state-of-the-art results on UI navigation and robotic manipulation tasks, outperforming previous models that are specifically tailored to these tasks. On image and video-related multimodal tasks, Magma also compares favorably to popular large multimodal models that are trained on much larger datasets. We make our model and code public for reproducibility1.

1https://microsoft.github.io/Magma

#### 1. Introduction

A long-standing research topic of AI is to develop autonomous agents that can perceive visual stimuli, language inputs, and other environmentally-grounded data and produce meaningful embodied actions in physical and digital environments to complete specific tasks.

Recently, there has been a growing interest in developing AI agents based on Vision-Language-Action (VLA) models [5, 6, 19, 29, 42, 54]. These models are typically pretrained on large amounts of vision-language datasets and then action trajectories to attain ability to take actions given VL inputs. However, due to the inherent difference between various environments (e.g., 2D digital world and 3D physical ones), VLA models are typically trained separately for simplicity and then used for different tasks. Exemplary models in the digital world include Pix2ACT [108], WebGUM [34], and Ferret-UI [131] for UI navitation. VLA models in the 3D physical world include RT-2 [5] and OpenVLA [54] for robotics manipulation. Although claimed as generalist, most of these models prioritize learning a taskspecific action policy at the cost of a significant decline in generic multimodal understanding capabilities, rendering limited genralizability across tasks and domains.

In this research, we strive to develop a foundation model for multimodal AI agents and argue that it requires simultaneously possessing the following capabilities:

- • Multimodal Understanding to understand multimodal input from various domains (both digital and physical) not only semantically, but also spatially and temporally.
- • Multimodal Action Prediction to break down the longhorizon task into an accurate action sequence, which can be effectively executed by AI agent systems.

Such an agent system should be driven by external goals specified by human commands as shown in Fig. 2.

To endow the broad capabilities, we effectively leverage large amounts of heterogeneous vision-language and action datasets, including UI datasets such as SeekClick [19], robotic manipulation dataset OXE [23], human instructional videos like Ego-4d [40] and image-text pairs used in LMMs [13, 71]. Instead of sequentially training on one domain and adapting to another, we train a single foundation model which can be applied in a zero-shot manner to different downstream tasks in various settings.

Simply combining those datasets, however, does not bring benefits to the foundation model, due to the significant gap between multimodal understanding which is mostly verbal (i.e., textual descriptions for images and videos) and the action-taking tasks which are mostly spatial (i.e., 2D coordinates for UI or 7-DoF for robot arm). To bridge the gap, we propose two surrogate tasks for model training, action grounding and action planning, by asking the model to predict the proximal action outputs given the visual-spatial observations, represented as images or video frames. Specifi-

[Figure 40]

Figure 2. A multimodal AI agent should be capable of mutimodal understanding and action-prediction towards a given goal.

cally, in each image, we label the visual objects that are actionable by Set-of-Mark (SoM) (e.g., clickable buttons in Fig. 1 bottom-middle) and labeled in each video the object movements, which are the results of actions, with Traceof-Mark (ToM) (e.g., the trace of human hand or robotic arm in Fig. 1 top-middle). In this way, the image and video datasets, which are not labeled with actions, are transformed into “vision-language-action” data to morph the gap among different types of tasks. We show through extensive empirical studies that SoM and ToM achieve are environmentagnostic and easy to generalize to new agentic tasks, offering an effective and efficient approach to scaling up our Magma model pretraining using large amounts of unlabeled videos, such as raw instructional videos.

To the best of our knowledge, Magma is the first foundation model for multimodal AI agents that can understand multimodal inputs (see Fig. 1 left), perform action grounding and planning for the future (see Fig. 1 middle), and finally adapt to downstream (unseen) agentic tasks in both the digital and physical environments(see Fig. 1 right). We evaluated Magma on three task categories: UI navigation (e.g., Mind2Web, AITW), where it has to reason and act in evolving digital environments; vision-language understanding (e.g., GQA, VideoMME), where it grounds language in visual objects and events; and finally robotic manipulation (e.g., Bridge, LIBERO), which tests its 3D spatial intelligence for physical interaction. Magma achieves new SOTA results on UI navigation and robotic manipulation tasks, outperforming even domain-specific models while maintaining strong performance on VL tasks which are comparable to SOTA LMMs.

In summary, the main contributions of this work are:

- • We propose Magma, the first foundation model that acquires not only multimodal understanding but also spatial-temporal reasoning abilities for agentic tasks in both digial and physical environments.
- • We propose the use of Set-of-Mark and Trace-of-Mark techniques to significantly enhance the spatial-temporal intelligence for action grounding and planning, and allow Magma to be pretrained effectively on large amounts of heterogeneous datasets.
- • We curate a large-scale pretraining dataset, which consists of not only open-source VL datasets, but also UI, robotics data and human instructional videos, autolabeled using SoM and ToM. In total, our training corpus

- contains approximately 39 million diverse samples.
- • We extensively evaluate the pretrained Magma model to demonstrate the superior model performance across a wide range of tasks. Magma with a single suite of parameters achieves new SOTA on both robotic manipulation and UI navigation over open-sourced counterparts.
- • We show that the proposed Magma pretraining method significantly improves model’s verbal and spatialtemporal intelligence abilities. For instance, Magma can achieve SOTA performance on the BLINK dataset without instruction fine-tuning, and SOTA performance on video question-answering benchmarks despite being pretrained on much fewer frames.

#### 2. Related Work

Large Multimodal Models (LMMs). Large Language Models (LLMs) like ChatGPT [97], GPT-4 [98], and Llama [115] have demonstrated impressive reasoning and generalization capabilities for text. The introduction of models that integrate visual data has brought about a significant shift in the landscape of LLMs, such as GPT4V(ision)[96]. Building upon open-source LLMs [21, 115], a wide range of multimodal models have achieved remarkable progress, led by pioneering models such as LLaVA [71, 73] and MiniGPT-4 [145], which combine LLMs’ capabilities with a CLIP [101] based image encoder. Recently, a growing number of LMMs have been developed to handle a wider range of tasks and modalities, such as region-level LMMs [7, 12, 100, 134, 137], and video LMMs [69, 112, 135, 139]. In parallel, more sophisticated benchmarks are proposed to assess these capabilities [8, 32, 33].

UI Agent in Digital World. Recently there has been a lot of work on designing autonomous GUI agents to perform tasks in place of human users. One line of work is to train an end-to-end model to directly predict the next action, representative works include Pixel2Act [107] and WebGUM[35] in web domain, Ferret [132], CogAgent [43], and Fuyu [4] in Mobile domain. Another line of work involves leveraging existing multimodal models such as GPT-4V to perform user tasks. Representative works include MindAct [27], SeeAct [141] in web domain and others [103, 117, 122] for mobile domain. These works often leverage the DOM information in web browsers, or the view hierarchies in mobile apps to get the ground truth position of interactable elements of the screen, and use Set-of-Mark [125] or more advanced localization model [83] to overlay the bounding boxes on top of the screenshot that feed into the visionlanguage models.

Vision-Language-Action for Robotics. Several studies have investigated the application of LMMs in robotics [6, 54, 66, 95, 130, 142, 146]. Among these, RT-2 [6] finetuned LMMs on robotic trajectory data, enabling the output of discretized robot action tokens. OpenVLA [54] is

the first open-source VLA foundation that is fine-tuned an open-source Prismatic VLM backbone [49]. LLARVA [95] generated 2D visual traces for robot arms along with textual representations of actions, using visual trace prediction as an auxiliary task, while TraceVLA [142] used visual trace prompting to improve spatial-temporal awareness of robot policy. Most recently, learning from videos by predicting the latent VQVAE tokens is explored in [11, 129]. In this work, we follow a similar approach as OpenVLA to represent the action but leverage rich multimodal data far beyond robotics datasets. Also, instead of asking model to predict latent tokens, we propose SoM and ToM techniques to significantly enhance the spatial-temporal intelligence, demonstrating significantly stronger performance and generalization capability for agentic tasks.

#### 3. Multimodal Agentic Modeling

##### 3.1. Problem Definition

A generalist multimodal AI agent should be performant for both multimodal understanding and action-taking. We define a multimodal AI agent π, which takes past visual observations I = {I1,...,Ik} and a task description task in text as input and outputs a set of T ≥ 1 tokens O as:

O = π(I,task,ctx) = {ol1,· · ·,olT} (1)

where ctx denotes the context, l ∈ {verbal,spatial} indicates if the i-th token oi is a verbal or spatial token. This formula generalizes across different tasks:

- • UI navigation in 2D screenshots. The task could be “book a hotel” and output should include both language tokens denoting the semantic type of action (e.g., “type”, “click”, etc.) and the location (x,y) or box (x,y,w,h) to which actions are applied.
- • Robotic manipulation in the 3D world. For a task like “close the drawer”, the output consists of 6-DoF displacements (x,y,z,yaw,pitch,roll) of the end effector and, in some cases, one additional dimension to indicate whether the gripper is open or not.
- • Multimodal understanding tasks. When the task is purely about I, e.g., a VQA task, the problem is reduced to a multimodal understanding task that generates a textual description and/or location of objects for input images/videos.

For these seemingly different output modalities, we follow a common practice to transform all output into textual tokens to facilitate model learning. Specifically, we convert 2D actions into a textual dictionary as in [19], and represent robot actions with the last 256 discrete language tokens that is barely used in LLMs, following [54]. Despite such unification into language space, we notice considerable conflicts among tasks, as we will show in our experiments. In what

follows, we will discuss how to mitigate such challenges to train agentic foundation on a wide range of datasets.

##### 3.2. Method

We approach two key challenges while building a highly capable foundation for the multimodal AI agent.

Pretraining objectives: How to build a unified pretraining interface to facilitate joint training? A straightforward way would be to predict the 2D coordinates for the navigation of the UI, 3D positions for the end effectors, and regular textual outputs for VL tasks. However, in our experiments, we observed that these tasks have inherent domain gaps in both input and output. The former results in a huge search space at the pixel level, and the latter directly predicts the output of proprioceptive action, which is not grounded on the observations of the image. Can we come up with a surrogate task that can bridge the gap among all tasks?

Data scaling-up: Existing vision-language-action data have limited amount and diversity, unlike language or image-text corpus for LLMs and LMMs, respectively. For example, the largest open source robotic dataset OXE [23] consists of around 1M trajectories taken from 22 environments. On the other hand, large-scale image-text datasets like LAION [104] barely contain useful supervisions for action pretraining as they are all static without the notion of action. Videos, however, depict numerous human actions and human-object interactions. Can we largely take advantage of these video data for our agentic pretraining?

In this work, we propose a simple yet effective method to address the aforementioned challenges. Inspired by the generality of Set-of-Mark (SoM) rompting [126], we employ it to enable the action grounding onto images for both UI and robotic tasks in that model faces much less difficulties to predict the numeric marks for both clickable buttons or robot arms in the image space. We further extend it along the temporal axis and ask the model to predict Trace-of-Mark (ToM), which forces the model to learn a longer horizon by predicting distant future “actions”, and more importantly provides an effective way to leverage unlabeled video data. The combination of SoM and ToM enables a seamless synergy across agentic tasks in digital and physical domains, as well as a scalable way to curate “action” supervisions from raw videos. We describe them in detail below in Sec. 3.2.1 and 3.2.2, respectively.

###### 3.2.1. Set-of-Mark for Action Grounding

SoM prompting [126] was first proposed to enhance the grounding capability of GPT-4V and has then been widely adopted for various agentic tasks [18, 44, 70, 94, 123]. Unlike previous works that exploited it for prompting off-theshelf LMMs to enhance visual-language grounding, here we propose to train an agentic model for action grounding, i.e., locating actionable points / regions for a specific task and further predict atomic actions if needed.

[Figure 41]

Figure 3. Set-of-Mark supervisions for action grounding on UI screenshot (left), robot manipulation (middle) and human video (right). All coordinates are normalized by image size (height, width) and then quantized into 256 bins. Images better viewed by zooming in.

Given an image observation It ∈ RH×W×3 at timestep t, a task task and context ctx, we first extract a set of K candidate regions or points that are actionable P = {p1,...,pK}, where pk could be a four-dimensional box coordinate or two-dimensional point coordinates. Subsequently, we overlay the marks and boxes (if any) to the corresponding location of the image with numerical labels, i.e., M = {1 : p1,2 : p2,...,K : pK} giving us a new marked image ItM.

Given the prompted image ItM in an atomic action step, the model needs to select the candidate marks along with the original coordinates, significantly easing the action grounding for the agentic model. In this way, Eq. (1) can be reformulated as:

omarkt = actiont : markt = π(ItM,task,ctx) (2) where omarkt is a subset of marks M.

In Fig. 3, we show a few instances to demonstrate the SoM-based action grounding in Fig 1. To obtain candidate regions to mark, we can leverage different proposal networks such as image segmentation models [56, 147], object detection models [64, 80], or domain-specific models [83]. Readers refer to Supp. for more details.

###### 3.2.2. Trace-of-Mark for Action Planning

Video data contains a lot of information about human actions and activities, which can essentially be leveraged to boost the capability of agentic models. However, due to the absence of action labels, previous methods rarely explore this direction, apart from a few works focused on world model learning [76, 91]. We extend the strategy of “overlaying marks” from static images to dynamic videos by proposing Trace-of-Mark (ToM) to allow the agentic model to effectively learn to plan and act from videos.

Given the sequence of visual observations from a video I = {I1,...,It}, we extend along the time axis to the future

[Figure 42]

Figure 4. Trace-of-Mark supervisions for robot manipulation (left) and human action (right). Same coordinate normalization and quantization is used as SoM. Images show the future traces to predict.

l frames, Ifuture = {It+1,...,It+l}. Given the K marks at t-th frame It, we extract the corresponding positions of the overlay marks in the next l frames, denoted traces T = {Mt+1,...,Mt+l}. Following the prediction of action type and valid marks as in Eq. (2), we further ask the model to predict the future trajectories for the valid marks:

omarkt = actiont : markt : tracet+1:t+l = π({I1,...,It−1,ItM},task,ctx)

(3)

where tracet+1:t+l is a subset of the trace sequences for valid marks in markt in T . Our proposed ToM predicting is a simple yet effective way of leveraging video data and brings two unique modeling benefits: (i) It forces the model to understand the temporal dynamics in the video observations and to “look ahead of time” before taking the next actions; (ii) Unlike predicting next frames as used in [77], predicting traces uses much fewer tokens to capture much longer temporal horizon and action-related object dynamics, while disregarding ambient contents.

To extract ToM, we employ point tracking models CoTracker [48], though any performant model can be used. In particular, given a sequence of frames {It,It+1,...,It+l} ∈ R(l+1)×H×W×3, we apply a dense tracking for s2 grid points to extract s2 traces of length (l + 1). Given these s2 traces, we drop those traces whose average motion magnitudes between two adjacent timesteps are smaller than a certain value ϵ (Please see more details in the supplementary material). The remaining ones are regarded as foreground motions driven by a given task.

##### 3.3. Modeling

To retain the multimodal understanding capability required for Magma, we adopt the common practice used in current VLMs (e.g., LLaVA [71] and Phi-3-Vision [1]). Given the visual observations I, we use a vision encoder V to encode each frame into a number of tokens and then concatenate all tokens into a sequence and feed them to a decoder-only LLM along with the language tokens that encode task descriptions. Due to the task diversity, a vision encoder that can seamlessly encode images and videos of various resolutions is needed. In light of this, we propose to use convolutional networks ConvNeXt [81] as the vision backbone,

Data Type Set-of-Mark Trace-of-Mark

UI Screenshots ✓ ✗ Robotics Images ✓ ✓ Instructional Videos ✓ ✓

Table 1. SoM and ToM applied to various data types. ToM is not applied to UI data as they are a sequence of discrete screenshots.

considering that it supports arbitrary image resolutions by default. To handle the high-resolution images (e.g., up to 2000 for UI screenshots), we simply perform global encoding without the bells and whistles used in previous work and find that it can encode the global context as well as combining global and local crops [1, 75]. To that end, we formulate the agentic modeling as an autoregressive decoding procedure:

ol,t+1∗ ∼ p(olt+1|{ol1,...,olt};V(I),task,ctx). (4)

#### 4. Multimodal Agentic Pretraining

##### 4.1. Datasets

To develop a foundation model with both verbal and spatial intelligence that is capable of handling diverse agentic tasks, we curated a comprehensive pretraining dataset from a wide range of images, videos, and robotics domains.

- • Robotics manipulation data. For robotics task, we follow OpenVLA [54] and use the robotics dataset of OpenX-Embodiment [22].
- • UI navigation Data. We exploit two pretraining datasets, SeeClick [19] and Vision2UI [41].
- • Instructional videos. We compile Epic-Kitchen [24, 25], Ego4d [40], Somethingv2 [37] and other related considering the coarse but rich goal-driven human actions.
- • Multimodal understanding. Lastly, we include ShareGPT4V [14], instruction tuning data in LLaVA1.5 [75], and a few other OCR-related datasets [86, 90] to attain image understanding capability.

We noticed that many more related datasets could be used for our model pretraining, such as large-scale instruction tuning data [60, 114], more diverse video data [16]. In

Algorithm 1 SoM generation for UI images Require: image I, bounding boxes B, image height and

width (ih,iw)

- 1: B∗ ← []
- 2: for (idx,b) ∈ enumerate(B) do
- 3: text ← str(idx + 1)
- 4: I ← DrawRectangle(I,b)
- 5: (cy,cx) ← FindOptimalCorner(b,B∗,(ih,iw))

▷ Find corner that is far away from all boxes in B∗

- 6: (mh,mw) ← GetMarkSize(text,H,W)
- 7: text box ← (cy,cx,cy ± mh,cx ± mw)

- 8: I ← DrawRectangle(I,text box)

- 9: I ← DrawText(I,(cx,cy),text,color = white)
- 10: B∗ ← B∗ + [b] ▷ Add current drawn box to B∗
- 11: end for
- 12: Return I

this study, we focus on the demonstration of our pretraining methodology and leave the further scaling up for future. In the next, we elaborate on how we extract the surrogate action supervisions through Set-of-Mark (SoM) and Traceof-Mark (ToM).

##### 4.2. SoM and ToM Generation

As shown in Table 1, we apply SoM and ToM for different data types, where SoM is applied to all to learn a uinified action grounding. ToM is not fit for the UI data as it consists of sequences of discrete screenshots.

###### 4.2.1. SoM for UI Navigation

For UI screenshots in our pretraining data, we mainly rely on the original annotations extracted based on DoM Tree. In addition to the bounding boxes extracted from HTML code [19, 41], we further annotate the mobile screenshots in SeeClick data with bounding boxes derived from Android view hierarchies [111]. Given the extracted candidate bounding boxes for an image, we apply Alg. 1 to assign a textual label (line 3) and draw the boxes around the objects. To minimize overlapping box placements, we determine the optimal position for a label using previously drawn boxes (line 5) before computing the textbox size and assigning its coordinates (line 7). During the evaluation, we follow the common practice by applying OmniParser [83] for the zeroshot evaluation on ScreenSpot [19], and using the candidate boxes provided by [27] for downstream training and evaluation on Mind2Web.

###### 4.2.2. SoM and ToM for Videos and Robotic Data

We use marks and traces as surrogate action supervisions to pretrain our Magma model for action grounding and planning. To extract reliable traces, we use the state-of-the-art point tracking model CoTracker [48] to track the keypoints in each video segment. Unlike object detection and tracking

Algorithm 2 SoM and ToM generation for instructional videos and robotic data Require: image sequence I = {It,...Il}; grid size s; global motion threshold η; foreground threshold ϵ

- 1: M = {Mt,...,Ml} ← CoTracker(I,s)
- 2: if HasGlobalMotion(M,η) then
- 3: M ← H(M) ▷ Apply homography transformation
- 4: end if
- 5: Mf,Mb = ClassifyTraces(M,ϵ) ▷ Classify traces into foreground and background ones
- 6: k ← Random(1,min(5,|Mf|))
- 7: Mf,Mb = KMeans(Mf,k),KMeans(Mb,2k) ▷ Cluster foreground and background traces separately
- 8: It ← SoM(It,{Mtf,Mtb}) ▷ Apply SoM on 1st frame
- 9: Return I,M∗f

[Figure 43]

Figure 5. An illustration of Alg. 2 to handle videos with camera motions for SoM/ToM generation.

systems used in previous works [68, 95, 102], point tracking provides the finest grained moving trajectories for both end effectors (robot arms or human hands) and objects, and more importantly can be feasibly applied to any videos as it does not require object recognition.

Reliability of CoTracker. To determine the generalizability of such traces, we examine the reliability of CoTracker before running the algorithm on all our pretraining data. We note that CoTracker was already well validated on multiple video datasets such as TAP-Vid [28] and PointOdyssey [143] in the original paper. In this work, we proposed comprehensive strategies to handle scene transition and camera motions in videos (Alg. 2), which effectively scale to datasets like Ego4D and other instructional videos (Fig 13). To further validate the reliability of ToM, we quantitatively evaluated the traces on a subset of YouCook2-BB [144] with box annotations by humans. We extract the traces from each annotated box and count the number of future traces still falling into the box 1 second

Epic-Kitchens chop sundriedtomato

Somethingv2 pourwater intoacup

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

ideos(9.4M)

Ego4d (10.2M)

InstructionalV

Ego 4D

the camera wearer cleans sink

Open-X-embodiment

take the lid off the pot, put the pot on the plate...

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

ideoData

V

Instructional …use the black colorsoIam goingto wipe my brush alittle bit..

SeeClick and Vision2UI

Magma-SFT (820K)

Multimodal Understanding

UI Navigation

Shar

eGPT4V(1.2M)

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

UI navigation tasks such as "book a flight" and "rent a car"

Data

SeeClick(850K)

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Somethingv2 (4.5M)

V

ision2UI(1.9M)

Robotics Manipulation Data

Epic-Kitchen(1M)

cut both ends and remove fruit seeds

Describe the image in a detailed manner

Visual-language tasks,

ShareGPT4V LlaVA 1.5

[Figure 74]

including QA and OCR

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Open-X-Embodiment(9.4M)

- Figure 6. Overview of Pretraining Data Sources. A diverse collection of datasets including instructional videos (orange), robotics manipulation (green), UI navigation (pink), and multimodal understanding (blue). Note that we count the size of each dataset by the number of image samples. For video and robotics data, we extract the images from the short clips and trajectories, respectively.

forward. On 1320 clips, we got a precision of 0.89, indicating that the traces reliably capture temporal motions.

for both ego-centric videos and exo-centric ones, as ilustrated in Fig. 5.

After extracting the traces and applying the homography transformation if needed (lines 2-4), we classify them into two categories, foreground and background traces based on the average motion magnitude between two adjacent time steps, where traces with average motion magnitude of at least ϵ (line 5) are counted as foreground. Finally, we select the number of clusters (line 6) and perform a K-Means clustering for the foreground and background traces separately (line 7) before randomly selecting one or more points from each cluster as the final traces. In practice, we set s, η and ϵ to be 15, 2 and 2, respectively.

Segment and CLIP-score filtering As the point tracking system works in a short time window, we begin by using the annotations provided, curated or otherwise, to split each video into segments, and then run PySceneDetect [10] to further break each segment into short video clips with consistent shots. However, the detected video clips may not always be relevant to their associated text annotations. Thus, we use the pretrained CLIP [101] visual and text encoders to compute the cosine similarity score between each clip and text pair, and filter out clips with < 0.25 scores.

Once we have the fine-grained video clips in hand, we apply Alg. 2 to generate SoM and ToM. Given a video clip with l frames {I1,I2,...,Il} ∈ R(l)×H×W×3, we start from the time step t and put a grid of equally spaced s2 points on It. Then, we use CoTracker to extract s2 future traces of length (l − t) each. The output also contains predicted occlusion labels for each trace, which indicate if any points on the trace are obstructed at some time steps.

##### 4.3. Pretraining

The above data and annotation curation results in a comprehensive pretraining suite which covers (i) different digital and physical environments; (ii) both verbal and spatial annotations and (iii) various multimodal understanding and agentic tasks. As seen in Fig. 6 (left), we include close to 2.7M UI navigation screenshots from SeeClick [19] and Vision2UI [41]. We follow OpenVLA [54] to incorporate 970K trajectories in Open-X-Embodiment [23], which consists of 9.4M image- language-action triplets. Another majority of the pretraining data are videos which comprise over 25M samples sourced from around 4M shot-consistent video clips. Finally, we include 1.2M image and text pairs from ShareGPT4V [13], and LLaVa-1.5 [74] and a few other OCR-related datasets [86, 90], which we denote by Magma-SFT (820K).

Removal of global motions. Many instructional videos, particularly the ego-centric ones [40], contain significant camera movements. Consequently, the extracted traces may reflect external movements instead of relevant actions to accomplish a given task. We mitigate this issue by performing the homography transformation [31]. Specifically, we compute the 3×3 transformation matrix hi with the future mark positions and current ones:

hi = H(Mt,Mt+i) ∈ R3×3 (5)

By default, we use LLaMA-3-8B [30] as the language backbone and ConvNext-XXlarge [81] as the vision backbone. We show the pretraining architecture in Fig. 7. Our proposed SoM and ToM play as the bridge to connect verbal and action supervisions for all four types of data, and significantly enhance model’s spatial intelligence as we observe

Given hi, we apply the homography transformation to Mt+i to obtain Mt∗+i which shares the same coordinate system as Mt. Valid traces of marks to predict in Eq. (3) are then extracted from {Mt,Mt∗+1,Mt∗+l}. It turns out that the proposed method is effective to remove global camera motions

[Figure 82]

- Figure 7. Magma pretraining pipeline. For all training data, texts are tokenized into tokens, while images and videos from different domains are encoded by a shared vision encoder. The resulted discrete and continuous tokens are then fed into a LLM to generate the outputs in verbal, spatial and action types. Our proposed method reconcile the multimodal understanding and action prediction tasks.

during our experiments.

For comparisons, we run a few variants for the ablation studies in our experiments:

- • Magma-8B (SFT) is the model trained with MagmaSFT (820K) for the instruction tuning following a conventional recipe used on LMM training.
- • Magma-8B (UI) and Magma-8B (OXE) are the models pretrained on UI screenshots and OXE robotics data, respectively.
- • Magma-8B (ACT) is pretrained jointly on UI screenshots and robotics data.
- • Magma-8B (Full) is the full model trained with the whole dataset with SoM and ToM annotations.

Unless noted otherwise, all pretrainng includes the Magma-SFT (820K). We pretrain our model using our curated data for maximally three epochs with a constant learning rate of 1e-5, and evaluate the pretrained model on different tasks under the zero-shot setting as well as finetune its weights on the downstream tasks. The entire model including the parameters of the language model and the vision encoder is tuned. See Appendix for more detailed settings.

#### 5. Experiment

##### 5.1. Evaluating Agentic Capability

We examine the effectiveness of Magma as the foundation model for multmodal agents on UI Navigation tasks in the digital world, the robotic manipulation in the physical world, as well as the generic multimodal understanding.

###### 5.1.1. Zero-Shot Evaluation

To evaluate the zero-shot transferability of Magma, we employ ScreenSpot [19] and VisualWebBench [79] for evaluating UI action grounding and navigation, and SimplerEnv [65] for robotic manipulation. In addition to these

evaluations, we also validate our model on generic [39] and text-rich [110] VQA tasks as well as hallucination benchmark POPE [67]. As shown in Table 2, Magma consistently outperforms all other general-domain LMMs (e.g., LLaVA, Qwen-VL) and domain-specific agentic models such as SeeClick [19] for UI navigation and OpenVLA [54] for robotic manipulation. Notably, the zero-shot performance of Magma on UI is much better than the state-ofthe-art vision-based method that uses GPT-4V and Omniparser [84]. We report the results on two commonly used simulator embodiments in SimplerEnv [65], Bridge and Google Robot including 8 tasks with 172 visual matching and variant aggregation scenarios. Since OpenVLA uses real robot trajectories for pre-training, the model is susceptible to the domain gap for real-to-sim adaptation. In contrast, our Magma model, trained for multimodal understanding and action prediction on a wide range of heterogeneous datasets, is significantly more resilient to the gap and achieves significantly better success rates.

Fig. 8 shows detailed comparisons between our pretrained Magma model and other representative models. Remarkably, Magma surpasses the second-place OpenVLA by 19.6%, nearly doubling the average success rate. On those challenging tasks such as “Put Object in Drawer” and “Put Carrot on Plate”, Magma achieves a remarkable success rate while most baselines fail entirely. Notably, Magma tuned on our pretrained model showcases substantially better results than the version trained solely on robotic datasets, highlighting the value of spatial intelligence learned from diverse datasets for physical robotic manipulation tasks.

Ablation Studies. We ablate our model pretraining techniques and data mixtures. The results are shown in Table 3. First, we observe from the top three rows that simply combining UI and robotics data does not bring gains, but instead

Multimodal Understanding UI Action Grounding and Navigation Robot Manipulation Model Size VQAv2 TextVQA POPE SS-Mobile SS-Desktop SS-Web VWB-Ele-G VWB-Act-G SE-Google Robot SE-Bridge GPT-4V [99] n/a 77.2 78.0 n/a 22.6/24.5 20.2/11.8 9.2/8.8 67.5 75.7 - GPT-4V-OmniParser [83] n/a n/a n/a n/a 92.7/49.4 64.9/26.3 77.3/39.7 - - - LLaVA-1.5 [71] 7.4B 78.5 58.2 85.9 - - - 12.1 13.6 - LLaVA-Next [75] 7.4B 81.8 64.9 86.5 - - - 15.0 8.7 - Qwen-VL [3] 9.6B 78.8 63.8 n/a 7.5/4.8 5.7/5.0 3.5/2.4 14.0 10.7 - Qwen-VL-Chat [3] 9.6B 78.2 61.5 n/a - - - - - - Fuyu [4] 8B 74.2 n/a n/a 41.0/1.3 33.0/3.6 33.9/4.4 19.4 15.5 - SeeClick [19] 9.6B - - - 78.0/52.0 72.2/30.0 55.7/32.5 9.9 1.9 - Octo [113] 93M - - - - - - - - 6.0 15.9 RT-1-X [23] 35M - - - - - - - - 34.2 1.1 OpenVLA [54] 8B - - - - - - - - 31.7 14.5 Magma-8B (Ours) 8.6B 80.0 66.5 87.4 60.4/58.5 75.3/52.9 69.1/52.0 96.3 71.8 52.3 35.4

- Table 2. Zero-shot evaluation on agentic intelligence. We report the results for pretrained Magma without any domain-specific finetuning. Magma is the only model that can conduct the full task spectrum. “SS” denotes the ScreenSpot benchmark proposed in SeeClick [19]; “VWB” denotes VisualWebBench [79]; “SE” denotes the SimplerEnv simulator [65]. ‘n/a’ means not available and ‘-’ means not supported. For all related evaluations, we use OmniParser to provide the detection results only, without local semantics.

[Figure 83]

Average Pick Coke Can Move Near Open/Close

Drawer

Put Carrot on Plate Stack Blocks Put Spoon on Towel

Put Eggplant in Basket

Put Object in Drawer

Google Robot BridgeData V2

Octo-Base

RT-1-X OpenVLA

Magma

100 80

60 40

20 0

Success Rate (%)

76.3

33.8

65.6

54.0

8.8

46.9

31.1

3.6

54.7

0.0

15.7 12.5

43.2

11.9

43.1

12.5

8.3

0.0

4.2

0.0 0.0

4.2

31.0

0.0

12.7

8.3

37.5

45.8

60.5

38.2

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

43.9

11.0

17.6

23.1

Magma (OXE)

1.0 0.0

35.1

45.2

0.0

5.7 8.3

44.9 39.2

53.0

28.9

Figure 8. SimplerEnv performance comparison on Google Robots and Bridge. Magma(OXE) represents our model trained solely on Open-X-Embodiment (OXE) [22], while Magma is our pretrained model. Results for each task are averaged across visual matching and variant aggregation scenarios.

Model SoM+ToM SS-Overal VWB-Ele-G VWB-Act-G SE-Bridge SE-Google

Magma-8B (UI) ✗ 57.7 68.5 58.3 - Magma-8B (OXE) ✗ - - - 22.2 35.7 Magma-8B (ACT) ✗ 56.2 89.1 21.4 17.5 31.5 Magma-8B (Full) ✗ 57.4 90.1 25.2 17.7 37.5 Magma-8B (Full) ✓ 61.4 96.3 71.8 35.4 52.3

- Table 3. Ablation study on the effect of data mixtures and pretraining techniques. w/o SoM+Tom means using original action supervisions (2D coordinates for UI and 7DoF for robots.)

###### 5.1.2. Efficient Finetuning

With moderate finetuning, the pretrained Magma model can be easily transferred to various downstream agentic tasks.

UI Navigation. Following the prior works [19, 43], we finetune Magma on Mind2Web and AITW, to examine the web and mobile UI navigation capabilities, respectively. For Mind2Web, we first apply the SoM prompting to the training samples according to the top candidates selected by [140], and then finetune Magma on the same samples as in SeeClick [19]. Table 4 shows the results in three subtasks, and clearly indicates Magma’s superiority to both general-domain and specific-domain LMMs. Similarly, on AITW Magma outperforms the state-of-the-art methods based on open-source or prosperity models. Considering that we use a similar size of LLM and a moderate amount of UI-related pretraining data, this decent performance is largely due to the proposed SoM and ToM modeling techniques, which significantly facilitate action grounding for UI navigation.

hurts the performance for both tasks. This is expected because the two agentic tasks have significantly different image domains as well as action spaces (2D coordinates v.s. 7-DoF). Adding video data to the pretraining slightly improves the performance across board but still can not fill the gap in between, as the additional video narrations can only enhance the verbal intelligence. However, once we apply SoM and ToM to all the pretraining data to put them into the unified interface, our model can learn effectively from the heterogeneous data for both verbal and spatial intelligence. This study highlights the effectiveness of our proposed method and indicates equally importance of verbal and spatial understanding for agentic tasks.

Robotics Manipulation. Table 2 shows that the Magma

Input Source Cross-Website Cross-Task Cross-Domain

Method Backbone

DoM Tree Image Ele. Acc Op. F1 Step SR Ele. Acc Op. F1 Step SR Ele. Acc Op. F1 Step SR

GPT-4-MindAct [27] GPT-4 [98] ✓ 35.8 51.1 30.1 41.6 60.6 36.2 37.1 46.5 26.4 GPT-4V-OmniParser [83] GPT-4V [99] ✓ ✓ 41.0 84.8 36.5 42.4 87.6 39.4 45.5 85.7 42.0

GPT-4V [99] ✓ 13.9 - - 20.3 - - 23.7 Gemini-Pro [36] ✓ ✓ 21.5 67.7 19.6 21.5 67.7 19.6 20.7 64.3 18.0 GPT-4V [99] ✓ ✓ 38.0 67.8 32.4 46.4 73.4 40.2 42.4 69.3 36.8

SeeAct [141]

Fuyu-8B‡ Fuyu-8B [4] ✓ 4.8 81.3 4.0 8.3 83.9 6.6 3.6 83.0 3.0 Fuyu-8B-GUI [17] Fuyu-8B [4] ✓ 13.9 80.7 12.2 19.1 86.1 15.6 14.2 83.1 11.7 MiniCPM-V‡ MiniCPM-V [128] ✓ 8.2 78.2 6.0 11.0 85.6 8.5 6.5 81.4 5.2 MiniCPM-V-GUI [17] MiniCPM-V [128] ✓ 20.3 81.7 17.3 23.8 86.8 20.8 17.9 74.5 17.6 Qwen-VL♮ Qwen-VL [3] ✓ 13.2 83.5 9.2 15.9 86.7 13.3 14.1 84.3 12.0 SeeClick [19] Qwen-VL [3] ✓ 21.4 80.6 16.4 28.3 87.0 25.5 23.2 84.8 20.8 CogAgent† [43] CogVLM [118] ✓ 27.3 - 23.4 30.2 - 26.9 33.1 - 28.5 Qwen2-UIX [78] Qwen2 [124] ✓ 39.2 - 31.0 43.4 - 38.2 40.4 - 34.9

Magma-8B (Ours) LLaMA3 [92] ✓ 57.2 76.9 45.4 54.8 79.7 43.4 55.7 80.6 47.3

- Table 4. Efficient finetuning on Mind2Web for web UI navigation. “Ele. Acc” denotes element selection accuracy. “Op. F1” denotes the token-wise F1 score between predicted ground-truth operation. “Step SR” denotes the step-wise success rate. ‡ Numbers reported in Chen et al. [17]. ♮ Numbers reported in Cheng et al. [19]. † Numbers reported in Liu et al. [78].

|Method<br><br>|Backbone DoM Tree Image General Install GoogleApps Single WebShopping Overall|
|---|---|
|GPT-4V-SeeAct† [141] GPT-4V-ReAct† [127] GPT-4V-OmniParser [83]<br><br>|GPT-4V [99] ✓ 34.1 39.4 40.0 46.2 38.2 39.6 GPT-4V [99] ✓ 36.2 42.5 46.6 49.1 39.2 42.7 GPT-4V [99] ✓ ✓ 48.3 57.8 51.6 77.4 52.9 57.7|

|Fuyu-8B‡ Fuyu-8B-GUI [17] MiniCPM-V‡ MiniCPM-V-GUI [17] Qwen-VL♮ SeeClick [19]|Fuyu-8B [4] ✓ - 45.9 40.0 47.2 40.8 Fuyu-8B [4] ✓ - 50.9 41.6 45.7 43.8 MiniCPM-V [128] ✓ - 50.2 45.1 56.2 44.0 MiniCPM-V [128] ✓ - 62.3 46.5 67.3 57.5 -<br><br>Qwen-VL [3] ✓ 49.5 59.9 46.9 64.7 50.7 54.3 Qwen-VL [3] ✓ 54.0 66.4 54.9 63.5 57.6 59.3<br><br>|
|---|---|
|Magma-8B (Ours)<br><br>|LLaMA3 [92] ✓ 61.5 73.2 62.7 77.5 61.7 67.3|

- Table 5. Efficient finetuning on AITW for mobile UI navigation. We compared models either using DoM tree or image screenshot. We finetune our Magma jointly and then report the results on individual tasks. † Numbers reported in Zhang et al. [138]. ‡ Numbers reported in Chen et al. [17]. ♮ Numbers reported in Cheng et al. [19].

model without domain-specific finetuning already outperforms the recently proposed OpenVLA model pretrained for 27 epochs on the same amount of OXE data. Below, we testify the effectiveness of the finetuned Magma model by comparing it with OpenVLA in three settings:

- • Finetune on real robot data to evaluate on out-ofdistribution manipulation tasks;
- • Finetune in simulated robot settings with a limited number of trajectories using the LIBERO benchmark to evaluate Magma’s capability of task adaptation; and
- • Evaluate on the physical WidoxW 250 Arm. We collect four manipulation tasks each of which has

roughly 50 trajectories (See details in our supplementary material), and finetune both OpenVLA and Magma jointly on these tasks. For evaluation, we perform 10 trials per task, ensuring the same initial states (positions and orientations of end-effector and objects) across models. As shown in Fig. 9, the results clearly demonstrate Magma’s superior performance. For those challenging tasks that involve everyday objects like “Pick Place Hotdog Sausage”, “Put Mushroom in Pot”, and “Push Cloth Right to Left”, OpenVLA can hardly accomplish the tasks, mainly because of the imprecise arm movement and object localization per our

###### WidowX Robot (Real)

100 80 60 40 20 0

###### SuccessRate(%)

7/10

7/10

7/10

67.5

6/10

5/10

25.0

2/10

2/10

1/10

Pick Place Hotdog Sausage

Push Cloth Right to Left

Average Put Mushroom

Push Cloth Left to Right

Place Pot

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

OpenVLA Magma

###### Unseen

Figure 9. Few-shot finetuning and generalization performance on real robot. On a WidowX robot, we evaluate Magma on 4 tasks including diverse everyday object manipulation.

observation. In contrast, Magma performs well on these sophisticated tasks, largely owing to its strong spatial understanding and grounding capability obtained from pertaining. Additionally, we evaluate models’ performance on an unseen task “Push Cloth Left to Right” which are not included in our finetuning dataset. Magma substantially outperforms the baseline, indicating a stronger ability to preserve pretrained knowledge and generalize to new tasks.

The efficient adaptation (via finetuning) capability of Magma is further validated through few-shot finetuning

[Figure 95]

- Figure 10. Few-shot finetuning results on the LIBERO simulation benchmark, using 10 trajectories per task for fine-tuning.

VSR BLINK-val

SpatialEval2 Model Spatial Map Maze Nav. Spatial Grid

GPT-4o 74.8 60.0 - - Gemini - 61.4 - - -

- LLaVA-1.5-7B 57.1* 37.1 28.4 28.8 41.6

- LLaVA-1.6-7B [75] 52.2* - 28.0 34.8 32.2 Qwen-VL-9.6B [3] - 40.3 28.7 31.8 25.7

Magma-8B (Actw/o) 62.8 30.1 36.9 44.8 37.5 Magma-8B (Fullw/o) 58.1 38.3 27.5 33.5 47.3 Magma-8B (Full) 65.1 41.0 43.4 36.5 64.5

Table 6. Spatial reasoning evaluations. We use * to denote results that are obtained by us evaluating the provided model weights. Superscript ‘w/o’ means models pretrained without SoM/ToM.

[Figure 96]

- Figure 11. Spatial evaluation predictions. Spatial reasoning questions are challenging even for GPT-4o but Magma can answer relatively well despite relying on much fewer pretraining data.

evaluations on the LIBERO benchmark. For each task suite in the benchmark, we sample only 10 trajectories for finetuning. During the evaluation, we perform 100 trials per task suite. The results, shown in Fig. 10, indicate that Magma achieves a significantly higher average success rate in all task suites. Additionally, removing SoM and ToM during pretraining has a negative impact on model performance, underscoring the effectiveness of our pretraining method.

##### 5.2. Evaluating Spatial Reasoning

We attribute the much improved performance of our Magma model on the tasks of UI navigation and robotic manipulation, as shown above, to its improved ability to perform spatial reasoning. To verify this hypothesis, we evaluate the effectiveness of the spatial intelligence that is learned in our pretrained model on the challenging Visual Spatial Reasoning (VSR) [72], BLINK [33] and SpatialEval [116] benchmarks under the zero-shot setting. The results are summarized in Table 6. We see that Magma outperforms existing

Model VQAv2 GQA MME POPE TextVQA ChartQA DocVQA

LLaVA-1.5-7B [61] 76.6 62.6 1510.8 85.9 46.1 18.2 28.1 LLaVA-Next-7B [75] 80.1 64.2 1519.3 86.4 64.9 54.8 74.4

Magma-8B (SFT) 79.5 61.5 1510.1 86.2 67.7 73.0 80.4 Magma-8B (Actw/o) 81.3 63.5 1559.5 86.1 69.8 71.0 84.1 Magma-8B (Fullw/o) 81.3 62.9 1576.0 86.3 69.6 71.7 83.8 Magma-8B (Full) 81.4 64.0 1588.7 86.3 70.2 76.2 84.8

Table 7. Finetuned performance on multimodal image understanding tasks. Pretraining on full set with SoM and ToM (last row) attains the overall best performance compared with our own baselines and counterparts of the same model class.

approaches by significant margins on VSR and SpatialEval, and that Magma performs on par with CogVLM, despite only using ∼29M images for pretraining as compared to ∼1.5B images in the latter. In addition, our ablation study demonstrates the effectiveness of the SoM and ToM pretraining tasks in helping Magma improve its spatial reasoning capabilities. Last but not least, we also note the benefits of using video data during pretraining by showing that removing vidoes from training data leads to ∼8% performance drop on BLINK. Finally, we also provide some example predictions of our Magma model in Figure 11. We observe that spatial reasoning questions are also challenging for SOTA proprietary models such as GPT-4o. Despite the lack of pretraining on data with mazes, we see that Magma is still able to answer spatial reasoning questions about them.

##### 5.3. Evaluating Multimodal Understanding

Image instruction tuning. To further assess Magma’s multimodal understanding capability, we conduct continuous finetuning on our Magma-SFT-820K data. Then, we compare the finetuned Magma model with existing VLMs on a suite of commonly used image reasoning benchmarks, e.g. MME and GQA. As shown in Table 7, Magma outperforms recently-proposed VLMs on most of the tasks, with notable gains of ∼5% and ∼22% on TextVQA and ChartQA, respectively. Similarly to our observations in Table 6, our ablation study highlights the effectiveness of using SoM and ToM for pre-training, which leads to ∼ 5% improvement in ChartQA.

Video Instruction Tuning In Table 8, we report the performance of our Magma model on multiple challenging video question answering (QA) benchmarks including IntentQA [62], NextQA [120], VideoMME [32] and MVBench [63]. We use the LMMs-Eval framework [59] for the latter three benchmarks to ensure reproducibility of our evaluation results.

The results demonstrate the effectiveness of our pretraining approach, where we outperform most state-of-the-art models with comparable number of parameters consistently across the different benchmarks. For instance, our Magma

2We evaluate our model using the standard option matching before the official evaluation pipeline was released and will update in the next version.

IntentQA Next-QA VideoMME (w/o subs) MVBench

Method Backbone

Overall Overall Short Medium Long Action Prediction Action Sequence Action localization Overall Gemini-1.5 [36] - - - 81.7 74.3 67.4 - - - 37.7 GPT-4V [2] GPT-4 - - 70.5 55.8 53.5 - - - 43.7 LLaVA-OV [60] Qwen2-7B - 79.4 68.1 54.9 47.8 46.0 74.5 48.0 56.7 Long-Llava 9B [119] Long-Llava 9B - - 52.4 42.2 36.4 - - - 49.1 LongVA [136] Qwen2-7B - 69.3 61.1 50.4 46.2 49.0 53.0 42.5 51.3 ShareGPT4Video [15] LLaMA3-8B - - 48.3 36.3 35.0 40.0 49.5 41.5 51.2 Video-Llama2 [20] Llama2-7B - - 55.9 45.4 42.1 - - - 34.1 Video-Chat2 [63] Mistral 7B - 43.3 48.3 37.0 33.2 47.5 75.0 50.5 60.4 Video-Llava [69] Vicuna-7B - 51.4 45.3 38.0 36.2 50.0 38.5 30.5 43.0 IG-VLM [55] Vicuna-7B 60.3 - - - - - - - SF-LLaVA [121] Vicuna-7B 60.1 - - - - - - - Magma-8B (Ours) LLaMA3-8B 88.6 80.9 72.9 55.8 44.3 65.0 79.0 55.5 59.4

Table 8. Zero-shot Video QA benchmarks. We compare our Magma model to other state-of-the-art approaches with comparable numbers of parameters. Our Magma model performs competitively and even outperforms some state-of-the-art approaches such as Video-Llama2 and ShareGPT4Video on most benchmarks, despite using much fewer video instruction tuning data.

model achieves a performance gain over the IG-VLM and SF-LLaVA models by approximately 28%. The IntentQA benchmark evaluates a model’s capability to discern the intentions behind observed actions in videos. Thus, the significant improvement on this dataset achieved by Magma can possibly be attributed to the effectiveness of our ToM pretraining task, where it encourages the model to reason about temporal dynamics in future video frames. This is also corroborated by the notable improvement on the subtask of action prediction in MVBench that Magma obtains over stateof-the-art models such as VideoChat2 and LLaVA-OV.

State-of-the-art video LMMs often rely on much large video and text datasets such as Webvid and ShareGPT4Video for pretraining and these datasets span over 4M samples with curated text. Moreover, the aforementioned models also use a higher number of frames during pretraining. In contrast, even when multi-frame pretraining is performed in our case, we only use a maximum of 4 frames due to computational constraints. Thus, it is especially significant that Magma outperforms approaches such as LLaVA-OV and ShareGPT4Video on VideoMME and MVBench, since these approaches often use larger instruction tuning datasets that include both image and video data. Additionally, as evidenced by the performance gain obtained by Magma over the proprietary GPT-4V model, we note that such improvements in results are not solely due to using a more recent and powerful language model like LLama-3. It is also notable that Magma achieves substantially better performance than LongVA, despite using only 32 frames instead of the 64 frames used by the latter.

#### 6. Conclusion

We present the Magma foundation model that can understand and act on multimodal inputs to complete agentic tasks in different environments. Our experiments show that the use of SoM and ToM prediction tasks in pretraining

helps the model learn to ground and plan actions, respectively. In our experiments, Magma shows strong spatialtemporal reasoning ability and significantly outperforms baselines on downstream UI navigation and robotic manipulation tasks.

Social Impacts and Limitations. To develop a foundation model with both verbal and spatial intelligence capable of handling diverse agentic tasks in digital and physical environments, we curated a comprehensive pretraining dataset from a wide range of image, video, and robotics domains:

- • UI navigation data. We leverage two pretraining datasets SeeClick and Vision2UI.
- • Instructional videos. As our goal was to learn an agentic model that can undertake daily tasks like humans, we compile the videos from Epic Kitchen, Ego4d, Something-Something v2 and other instructional videos.
- • Robotics manipulation data. For robotics task, we follow OpenVLA to leverage the robotics data in Open-XEmbodiment.
- • Multimodal understanding data. Lastly, we include a small set of multi modal pretraining data ShareGPT4V, and instruction tuning data LlaVA-1.5 plus a number of other domain-specific data to retain the generic multimodal understanding capability of the pre-trained model.

The data markup of the robotics and UI navigation data is fairly standardized focusing on generic manipulation tasks (“Place x object on y object”) and generic UI navigation tasks (“Click search button”). We, however, performed a detailed data reflection exercise on the video data of people performing certain tasks. The core inferences we took from these videos were the trajectory of objects over time when the tasks were performed.

We note that the distribution of identities and activities in the instructional videos are not representative of the global human population and the diversity in society. We are cognizant of the unintended societal, gender, racial and other

biases in training with these data, so we will ensure required disclaimers are in place when publishing the models. The training dataset, task list and descriptions focus on the next action to perform only – not describe, act on, or perform any analysis on the subject itself. While there can be unintended outputs from the model based on adverse task descriptions, we will ensure to highlight the use cases the model was trained for and it’s intended use.

Responsible AI. It is important to note that the model is specifically designed for UI navigation in a controlled Web UI and Android simulator, and robotic manipulation tasks and should not be broadly applied to other tasks. The recommended usage is within the settings they were trained on, namely, an enclosure equipped with a robotic arm and everyday objects for robotic manipulation and an android simulator running on a computer for UI manipulation. For UI navigation task, researchers should make sure that a human is in the loop and in control for every action the agentic system generates. Since the model cannot act by itself, the sub-module a researcher uses to actually perform the UI navigation action should ensure that no unintended consequences can occur as a result of performing the UI action proposed by the model.

The model by itself demonstrates good-enough capability in UI navigation and robotic manipulation, but is not usable as is for exploitation scenarios. A threat actor, can however use specific training data for a specific malicious task, to leverage the model as a base to perform automated UI navigation. This is a generic risk associated with the agentic models.

Acknowledgments. We would also like to thank Professor Yong Jae Lee for thoughtful discussions, Xiyang Dai for valuable discussions and data support, Mei Yang and Denny Sun for early data engineering effort, and Swadheen Shukla for internal RAI and data reviews. We would also like to thank Doug Burger and Desney Tan for the multifaceted leadership support.

#### References

[1] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, S´ebastien Bubeck, Martin Cai, Qin Cai, Vishrav Chaudhary, Dong Chen, Dongdong Chen, Weizhu Chen, Yen-Chun Chen, Yi-Ling Chen, Hao Cheng, Parul Chopra, Xiyang Dai, Matthew Dixon, Ronen Eldan, Victor Fragoso, Jianfeng Gao, Mei Gao, Min Gao, Amit Garg, Allie Del Giorno, Abhishek Goswami, Suriya Gunasekar, Emman Haider, Junheng Hao, Russell J. Hewett, Wenxiang Hu, Jamie Huynh, Dan Iter, Sam Ade Jacobs, Mojan Javaheripi,

Xin Jin, Nikos Karampatziakis, Piero Kauffmann, Mahoud Khademi, Dongwoo Kim, Young Jin Kim, Lev Kurilenko, James R. Lee, Yin Tat Lee, Yuanzhi Li, Yunsheng Li, Chen Liang, Lars Liden, Xihui Lin, Zeqi Lin, Ce Liu, Liyuan Liu, Mengchen Liu, Weishung Liu, Xiaodong Liu, Chong Luo, Piyush Madan, Ali Mahmoudzadeh, David Majercak, Matt Mazzola, Caio C´esar Teodoro Mendes, Arindam Mitra, Hardik Modi, Anh Nguyen, Brandon Norick, Barun Patra, Daniel Perez-Becker, Thomas Portet, Reid Pryzant, Heyang Qin, Marko Radmilac, Liliang Ren, Gustavo de Rosa, Corby Rosset, Sambudha Roy, Olatunji Ruwase, Olli Saarikivi, Amin Saied, Adil Salim, Michael Santacroce, Shital Shah, Ning Shang, Hiteshi Sharma, Yelong Shen, Swadheen Shukla, Xia Song, Masahiro Tanaka, Andrea Tupini, Praneetha Vaddamanu, Chunyu Wang, Guanhua Wang, Lijuan Wang, Shuohang Wang, Xin Wang, Yu Wang, Rachel Ward, Wen Wen, Philipp Witte, Haiping Wu, Xiaoxia Wu, Michael Wyatt, Bin Xiao, Can Xu, Jiahang Xu, Weijian Xu, Jilong Xue, Sonali Yadav, Fan Yang, Jianwei Yang, Yifan Yang, Ziyi Yang, Donghan Yu, Lu Yuan, Chenruidong Zhang, Cyril Zhang, Jianwen Zhang, Li Lyna Zhang, Yi Zhang, Yue Zhang, Yunan Zhang, and Xiren Zhou. Phi-3 technical report: A highly capable language model locally on your phone, 2024. 5

- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 12
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 9, 10, 11
- [4] Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, and Sa˘gnak Ta¸sırlar. Introducing our multimodal models, 2023. 3, 9, 10
- [5] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 2
- [6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023. 2, 3
- [7] Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P. Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. Making large multimodal models understand arbitrary visual prompts. In IEEE Conference on Computer Vision and Pattern Recognition, 2024. 3
- [8] Mu Cai, Reuben Tan, Jianrui Zhang, Bocheng Zou, Kai Zhang, Feng Yao, Fangrui Zhu, Jing Gu, Yiwu Zhong, Yuzhang Shang, Yao Dou, Jaden Park, Jianfeng Gao, Yong Jae Lee, and Jianwei Yang. Temporalbench: Bench-

- marking fine-grained temporal understanding for multimodal video models, 2024. 3
- [9] Joao Carreira, Eric Noland, Chloe Hillier, and Andrew Zisserman. A short note on the kinetics-700 human action dataset. arXiv preprint arXiv:1907.06987, 2019. 27
- [10] Brandon Castellano. Pyscenedetect: Automated scene detection in videos, 2014–2024. Version 0.6.4, BSD 3-Clause License. 7
- [11] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, Hanbo Zhang, and Minzhao Zhu. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation, 2024. 3
- [12] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023. 3
- [13] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023. 2, 7, 25, 27
- [14] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions,

2023. 5

- [15] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024. 12
- [16] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, and Sergey Tulyakov. Panda-70m: Captioning 70m videos with multiple cross-modality teachers, 2024. 5
- [17] Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, Yuan Yao, Yankai Lin, Zhiyuan Liu, and Maosong Sun. Guicourse: From general vision language models to versatile gui agents, 2024. 10
- [18] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision language model. arXiv preprint arXiv:2406.01584, 2024. 4
- [19] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents,

2024. 2, 3, 5, 6, 7, 8, 9, 10, 23, 25, 26, 27

- [20] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 12
- [21] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, 2023. 3

[22] Embodiment Collaboration, Abby O’Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, Albert Tung, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anchit Gupta, Andrew Wang, Andrey Kolobov, Anikait Singh, Animesh Garg, Aniruddha Kembhavi, Annie Xie, Anthony Brohan, Antonin Raffin, Archit Sharma, Arefeh Yavary, Arhan Jain, Ashwin Balakrishna, Ayzaan Wahid, Ben Burgess-Limerick, Beomjoon Kim, Bernhard Sch¨olkopf, Blake Wulfe, Brian Ichter, Cewu Lu, Charles Xu, Charlotte Le, Chelsea Finn, Chen Wang, Chenfeng Xu, Cheng Chi, Chenguang Huang, Christine Chan, Christopher Agia, Chuer Pan, Chuyuan Fu, Coline Devin, Danfei Xu, Daniel Morton, Danny Driess, Daphne Chen, Deepak Pathak, Dhruv Shah, Dieter B¨uchler, Dinesh Jayaraman, Dmitry Kalashnikov, Dorsa Sadigh, Edward Johns, Ethan Foster, Fangchen Liu, Federico Ceola, Fei Xia, Feiyu Zhao, Felipe Vieira Frujeri, Freek Stulp, Gaoyue Zhou, Gaurav S. Sukhatme, Gautam Salhotra, Ge Yan, Gilbert Feng, Giulio Schiavi, Glen Berseth, Gregory Kahn, Guangwen Yang, Guanzhi Wang, Hao Su, Hao-Shu Fang, Haochen Shi, Henghui Bao, Heni Ben Amor, Henrik I Christensen, Hiroki Furuta, Homanga Bharadhwaj, Homer Walke, Hongjie Fang, Huy Ha, Igor Mordatch, Ilija Radosavovic, Isabel Leal, Jacky Liang, Jad Abou-Chakra, Jaehyung Kim, Jaimyn Drake, Jan Peters, Jan Schneider, Jasmine Hsu, Jay Vakil, Jeannette Bohg, Jeffrey Bingham, Jeffrey Wu, Jensen Gao, Jiaheng Hu, Jiajun Wu, Jialin Wu, Jiankai Sun, Jianlan Luo, Jiayuan Gu, Jie Tan, Jihoon Oh, Jimmy Wu, Jingpei Lu, Jingyun Yang, Jitendra Malik, Jo˜ao Silv´erio, Joey Hejna, Jonathan Booher, Jonathan Tompson, Jonathan Yang, Jordi Salvador, Joseph J. Lim, Junhyek Han, Kaiyuan Wang, Kanishka Rao, Karl Pertsch, Karol Hausman, Keegan Go, Keerthana Gopalakrishnan, Ken Goldberg, Kendra Byrne, Kenneth Oslund, Kento Kawaharazuka, Kevin Black, Kevin Lin, Kevin Zhang, Kiana Ehsani, Kiran Lekkala, Kirsty Ellis, Krishan Rana, Krishnan Srinivasan, Kuan Fang, Kunal Pratap Singh, Kuo-Hao Zeng, Kyle Hatch, Kyle Hsu, Laurent Itti, Lawrence Yunliang Chen, Lerrel Pinto, Li Fei-Fei, Liam Tan, Linxi ”Jim” Fan, Lionel Ott, Lisa Lee, Luca Weihs, Magnum Chen, Marion Lepert, Marius Memmel, Masayoshi Tomizuka, Masha Itkina, Mateo Guaman Castro, Max Spero, Maximilian Du, Michael Ahn, Michael C. Yip, Mingtong Zhang, Mingyu Ding, Minho Heo, Mohan Kumar Srirama, Mohit Sharma, Moo Jin Kim, Naoaki Kanazawa, Nicklas Hansen, Nicolas Heess, Nikhil J Joshi, Niko Suenderhauf, Ning Liu, Norman Di Palo, Nur Muhammad Mahi Shafiullah, Oier Mees, Oliver Kroemer, Osbert Bastani, Pannag R Sanketi, Patrick ”Tree” Miller, Patrick Yin, Paul Wohlhart, Peng Xu, Peter David Fagan, Peter Mitrano, Pierre Sermanet, Pieter Abbeel, Priya Sundaresan, Qiuyu Chen, Quan Vuong, Rafael Rafailov, Ran Tian, Ria Doshi, Roberto Mart’in-Mart’in, Rohan Baijal, Rosario Scalise, Rose Hendrix, Roy Lin, Runjia Qian, Ruohan Zhang, Russell Mendonca, Rutav Shah, Ryan Hoque, Ryan Julian, Samuel Bustamante, Sean Kirmani, Sergey Levine, Shan Lin, Sherry

Moore, Shikhar Bahl, Shivin Dass, Shubham Sonawani, Shubham Tulsiani, Shuran Song, Sichun Xu, Siddhant Haldar, Siddharth Karamcheti, Simeon Adebola, Simon Guist, Soroush Nasiriany, Stefan Schaal, Stefan Welker, Stephen Tian, Subramanian Ramamoorthy, Sudeep Dasari, Suneel Belkhale, Sungjae Park, Suraj Nair, Suvir Mirchandani, Takayuki Osa, Tanmay Gupta, Tatsuya Harada, Tatsuya Matsushima, Ted Xiao, Thomas Kollar, Tianhe Yu, Tianli Ding, Todor Davchev, Tony Z. Zhao, Travis Armstrong, Trevor Darrell, Trinity Chung, Vidhi Jain, Vikash Kumar, Vincent Vanhoucke, Wei Zhan, Wenxuan Zhou, Wolfram Burgard, Xi Chen, Xiangyu Chen, Xiaolong Wang, Xinghao Zhu, Xinyang Geng, Xiyuan Liu, Xu Liangwei, Xuanlin Li, Yansong Pang, Yao Lu, Yecheng Jason Ma, Yejin Kim, Yevgen Chebotar, Yifan Zhou, Yifeng Zhu, Yilin Wu, Ying Xu, Yixuan Wang, Yonatan Bisk, Yongqiang Dou, Yoonyoung Cho, Youngwoon Lee, Yuchen Cui, Yue Cao, Yueh-Hua Wu, Yujin Tang, Yuke Zhu, Yunchu Zhang, Yunfan Jiang, Yunshuang Li, Yunzhu Li, Yusuke Iwasawa, Yutaka Matsuo, Zehan Ma, Zhuo Xu, Zichen Jeff Cui, Zichen Zhang, Zipeng Fu, and Zipeng Lin. Open x-embodiment: Robotic learning datasets and rt-x models, 2024. 5, 9

[23] Open X-Embodiment Collaboration, Abby O’Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, Albert Tung, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anchit Gupta, Andrew Wang, Andrey Kolobov, Anikait Singh, Animesh Garg, Aniruddha Kembhavi, Annie Xie, Anthony Brohan, Antonin Raffin, Archit Sharma, Arefeh Yavary, Arhan Jain, Ashwin Balakrishna, Ayzaan Wahid, Ben Burgess-Limerick, Beomjoon Kim, Bernhard Sch¨olkopf, Blake Wulfe, Brian Ichter, Cewu Lu, Charles Xu, Charlotte Le, Chelsea Finn, Chen Wang, Chenfeng Xu, Cheng Chi, Chenguang Huang, Christine Chan, Christopher Agia, Chuer Pan, Chuyuan Fu, Coline Devin, Danfei Xu, Daniel Morton, Danny Driess, Daphne Chen, Deepak Pathak, Dhruv Shah, Dieter B¨uchler, Dinesh Jayaraman, Dmitry Kalashnikov, Dorsa Sadigh, Edward Johns, Ethan Foster, Fangchen Liu, Federico Ceola, Fei Xia, Feiyu Zhao, Felipe Vieira Frujeri, Freek Stulp, Gaoyue Zhou, Gaurav S. Sukhatme, Gautam Salhotra, Ge Yan, Gilbert Feng, Giulio Schiavi, Glen Berseth, Gregory Kahn, Guangwen Yang, Guanzhi Wang, Hao Su, Hao-Shu Fang, Haochen Shi, Henghui Bao, Heni Ben Amor, Henrik I Christensen, Hiroki Furuta, Homanga Bharadhwaj, Homer Walke, Hongjie Fang, Huy Ha, Igor Mordatch, Ilija Radosavovic, Isabel Leal, Jacky Liang, Jad Abou-Chakra, Jaehyung Kim, Jaimyn Drake, Jan Peters, Jan Schneider, Jasmine Hsu, Jay Vakil, Jeannette Bohg, Jeffrey Bingham, Jeffrey Wu, Jensen Gao, Jiaheng Hu, Jiajun Wu, Jialin Wu, Jiankai Sun, Jianlan Luo, Jiayuan Gu, Jie Tan, Jihoon Oh, Jimmy Wu, Jingpei Lu, Jingyun Yang, Jitendra Malik, Jo˜ao Silv´erio, Joey Hejna, Jonathan Booher, Jonathan Tompson, Jonathan Yang, Jordi Salvador, Joseph J. Lim, Junhyek Han, Kaiyuan Wang, Kanishka Rao, Karl Pertsch, Karol Hausman, Keegan Go, Keerthana Gopalakrishnan, Ken Goldberg, Kendra Byrne, Kenneth Oslund, Kento Kawa-

harazuka, Kevin Black, Kevin Lin, Kevin Zhang, Kiana Ehsani, Kiran Lekkala, Kirsty Ellis, Krishan Rana, Krishnan Srinivasan, Kuan Fang, Kunal Pratap Singh, Kuo-Hao Zeng, Kyle Hatch, Kyle Hsu, Laurent Itti, Lawrence Yunliang Chen, Lerrel Pinto, Li Fei-Fei, Liam Tan, Linxi ”Jim” Fan, Lionel Ott, Lisa Lee, Luca Weihs, Magnum Chen, Marion Lepert, Marius Memmel, Masayoshi Tomizuka, Masha Itkina, Mateo Guaman Castro, Max Spero, Maximilian Du, Michael Ahn, Michael C. Yip, Mingtong Zhang, Mingyu Ding, Minho Heo, Mohan Kumar Srirama, Mohit Sharma, Moo Jin Kim, Naoaki Kanazawa, Nicklas Hansen, Nicolas Heess, Nikhil J Joshi, Niko Suenderhauf, Ning Liu, Norman Di Palo, Nur Muhammad Mahi Shafiullah, Oier Mees, Oliver Kroemer, Osbert Bastani, Pannag R Sanketi, Patrick ”Tree” Miller, Patrick Yin, Paul Wohlhart, Peng Xu, Peter David Fagan, Peter Mitrano, Pierre Sermanet, Pieter Abbeel, Priya Sundaresan, Qiuyu Chen, Quan Vuong, Rafael Rafailov, Ran Tian, Ria Doshi, Roberto Mart’in-Mart’in, Rohan Baijal, Rosario Scalise, Rose Hendrix, Roy Lin, Runjia Qian, Ruohan Zhang, Russell Mendonca, Rutav Shah, Ryan Hoque, Ryan Julian, Samuel Bustamante, Sean Kirmani, Sergey Levine, Shan Lin, Sherry Moore, Shikhar Bahl, Shivin Dass, Shubham Sonawani, Shubham Tulsiani, Shuran Song, Sichun Xu, Siddhant Haldar, Siddharth Karamcheti, Simeon Adebola, Simon Guist, Soroush Nasiriany, Stefan Schaal, Stefan Welker, Stephen Tian, Subramanian Ramamoorthy, Sudeep Dasari, Suneel Belkhale, Sungjae Park, Suraj Nair, Suvir Mirchandani, Takayuki Osa, Tanmay Gupta, Tatsuya Harada, Tatsuya Matsushima, Ted Xiao, Thomas Kollar, Tianhe Yu, Tianli Ding, Todor Davchev, Tony Z. Zhao, Travis Armstrong, Trevor Darrell, Trinity Chung, Vidhi Jain, Vikash Kumar, Vincent Vanhoucke, Wei Zhan, Wenxuan Zhou, Wolfram Burgard, Xi Chen, Xiangyu Chen, Xiaolong Wang, Xinghao Zhu, Xinyang Geng, Xiyuan Liu, Xu Liangwei, Xuanlin Li, Yansong Pang, Yao Lu, Yecheng Jason Ma, Yejin Kim, Yevgen Chebotar, Yifan Zhou, Yifeng Zhu, Yilin Wu, Ying Xu, Yixuan Wang, Yonatan Bisk, Yongqiang Dou, Yoonyoung Cho, Youngwoon Lee, Yuchen Cui, Yue Cao, Yueh-Hua Wu, Yujin Tang, Yuke Zhu, Yunchu Zhang, Yunfan Jiang, Yunshuang Li, Yunzhu Li, Yusuke Iwasawa, Yutaka Matsuo, Zehan Ma, Zhuo Xu, Zichen Jeff Cui, Zichen Zhang, Zipeng Fu, and Zipeng Lin. Open X-Embodiment: Robotic learning datasets and RT-X models. https: //arxiv.org/abs/2310.08864, 2023. 2, 4, 7, 9

- [24] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Scaling egocentric vision: The epickitchens dataset. In European Conference on Computer Vision (ECCV), 2018. 5, 24
- [25] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Jian Ma, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100. International Journal of Computer Vision (IJCV), 130:33–55,

2022. 5

- [26] Biplab Deka, Zifeng Huang, Chad Franzen, Joshua Hibschman, Daniel Afergan, Y. Li, Jeffrey Nichols, and Ranjitha Kumar. Rico: A mobile app dataset for building datadriven design applications. Proceedings of the 30th Annual ACM Symposium on User Interface Software and Technology, 2017. 23
- [27] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web, 2023. 3, 6, 10, 25, 26, 27
- [28] Carl Doersch, Ankush Gupta, Larisa Markeeva, Adria Recasens, Lucas Smaira, Yusuf Aytar, Joao Carreira, Andrew Zisserman, and Yi Yang. Tap-vid: A benchmark for tracking any point in a video. Advances in Neural Information Processing Systems, 35:13610–13626, 2022. 6
- [29] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. PaLME: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023. 2
- [30] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov,

Nikolay Bogoychev, Niladri Chatterji, Olivier Duchenne, Onur ¸Celebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaoqing Ellen Tan, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aaron Grattafiori, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alex Vaughan, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Franco, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, ChingHsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, Danny Wyatt, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Firat Ozgenel, Francesco Caggioni, Francisco Guzm´an, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Govind Thattai, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Irina-Elena Veliche, Itai Gat, Jake Weiss-

man, James Geboski, James Kohli, Japhet Asher, JeanBaptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Karthik Prasad, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kun Huang, Kunal Chawla, Kushal Lakhotia, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Maria Tsimpoukelli, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikolay Pavlovich Laptev, Ning Dong, Ning Zhang, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Rohan Maheswari, Russ Howes, Ruty Rinott, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Kohler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, V´ıtor Albiero, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaofang Wang, Xiaojian Wu, Xiaolan Wang, Xide Xia, Xilun Wu, Xinbo Gao, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yuchen Hao, Yundi Qian, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, and Zhiwei Zhao. The llama 3 herd of models, 2024. 7

- [31] Elan Dubrofsky. Homography estimation. Diplomov´a pr´ace. Vancouver: Univerzita Britsk´e Kolumbie, 5, 2009. 7
- [32] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai

- Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Rongrong Ji, and Xing Sun. Video-mme: The firstever comprehensive evaluation benchmark of multi-modal llms in video analysis, 2024. 3, 11
- [33] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390, 2024. 3, 11
- [34] Hiroki Furuta, Kuang-Huei Lee, Ofir Nachum, Yutaka Matsuo, Aleksandra Faust, Shixiang Shane Gu, and Izzeddin Gur. Multimodal web navigation with instruction-finetuned foundation models. arXiv preprint arXiv:2305.11854,

2023. 2

- [35] Hiroki Furuta, Kuang-Huei Lee, Ofir Nachum, Yutaka Matsuo, Aleksandra Faust, Shixiang Shane Gu, and Izzeddin Gur. Multimodal web navigation with instruction-finetuned foundation models, 2024. 3
- [36] Gemini Team. Gemini: A family of highly capable multimodal models, 2024. 10, 12
- [37] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The” something something” video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, pages 5842–5850, 2017. 5
- [38] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913,

2017. 27

- [39] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913,

2017. 8, 27

- [40] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, Miguel Martin, Tushar Nagarajan, Ilija Radosavovic, Santhosh Kumar Ramakrishnan, Fiona Ryan, Jayant Sharma, Michael Wray, Mengmeng Xu, Eric Zhongcong Xu, Chen Zhao, Siddhant Bansal, Dhruv Batra, Vincent Cartillier, Sean Crane, Tien Do, Morrie Doulaty, Akshay Erapalli, Christoph Feichtenhofer, Adriano Fragomeni, Qichen Fu, Abrham Gebreselasie, Cristina Gonzalez, James Hillis, Xuhua Huang, Yifei Huang, Wenqi Jia, Weslie Khoo, Jachym Kolar, Satwik Kottur, Anurag Kumar, Federico Landini, Chao Li, Yanghao Li, Zhenqiang Li, Karttikeya Mangalam, Raghava Modhugu, Jonathan Munro, Tullie Murrell, Takumi Nishiyasu, Will Price, Paola Ruiz Puentes, Merey Ramazanova, Leda Sari, Kiran Somasundaram, Audrey Southerland, Yusuke Sugano, Ruijie Tao, Minh Vo,

- Yuchen Wang, Xindi Wu, Takuma Yagi, Ziwei Zhao, Yunyi Zhu, Pablo Arbelaez, David Crandall, Dima Damen, Giovanni Maria Farinella, Christian Fuegen, Bernard Ghanem, Vamsi Krishna Ithapu, C. V. Jawahar, Hanbyul Joo, Kris Kitani, Haizhou Li, Richard Newcombe, Aude Oliva, Hyun Soo Park, James M. Rehg, Yoichi Sato, Jianbo Shi, Mike Zheng Shou, Antonio Torralba, Lorenzo Torresani, Mingfei Yan, and Jitendra Malik. Ego4d: Around the world in 3,000 hours of egocentric video, 2022. 2, 5, 7, 24
- [41] Yi Gui, Zhen Li, Yao Wan, Yemin Shi, Hongyu Zhang, Yi Su, Shaoling Dong, Xing Zhou, and Wenbin Jiang. Vision2ui: A real-world dataset with layout for code generation from ui designs, 2024. 5, 6, 7, 23, 27
- [42] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. arXiv preprint arXiv:2312.08914,

2023. 2

- [43] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Cogagent: A visual language model for gui agents, 2023. 3, 9, 10
- [44] Haoxu Huang, Fanqi Lin, Yingdong Hu, Shengjie Wang, and Yang Gao. Copa: General robotic manipulation through spatial constraints of parts with foundation models. arXiv preprint arXiv:2403.08248, 2024. 4
- [45] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 27
- [46] Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5648–5656,

2018. 27

- [47] Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Akos Kadar, Adam Trischler, and Yoshua Bengio. Figureqa: An annotated figure dataset for visual reasoning,

2018. 27

- [48] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. arXiv:2307.07635,

2023. 5, 6

- [49] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visuallyconditioned language models. In International Conference on Machine Learning (ICML), 2024. 3
- [50] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014. 27
- [51] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is

- worth a dozen images. In Computer Vision – ECCV 2016, pages 235–251, Cham, 2016. Springer International Publishing. 27
- [52] Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5376–5384, 2017. 27
- [53] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. Ocrfree document understanding transformer. In European Conference on Computer Vision, pages 498–517. Springer,

2022. 27

- [54] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. OpenVLA: An open-source vision-language-action model. In 8th Annual Conference on Robot Learning, 2024. 2, 3, 5, 7, 8, 9, 25
- [55] Wonkyun Kim, Changin Choi, Wonseok Lee, and Wonjong Rhee. An image grid can be worth a video: Zeroshot video question answering using a vlm. arXiv preprint arXiv:2403.18406, 2024. 12
- [56] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023. 4
- [57] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual Genome: Connecting language and vision using crowdsourced dense image annotations. International Journal of Computer Vision (IJCV), 123(1):32–73, 2017. 27
- [58] LAION-4V. Laion gpt4v-dataset, 2023. 27
- [59] Bo Li*, Peiyuan Zhang*, Kaichen Zhang*, Fanyi Pu*, Xinrun Du, Yuhao Dong, Haotian Liu, Yuanhan Zhang, Ge Zhang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Accelerating the development of large multimoal models, 2024. 11
- [60] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 5, 12
- [61] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. arXiv preprint arXiv:2306.00890, 2023. 11
- [62] Jiapeng Li, Ping Wei, Wenjuan Han, and Lifeng Fan. Intentqa: Context-aware video intent reasoning. In Int. Conf. Comput. Vis., pages 11963–11974, 2023. 11
- [63] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin

- Wang, and Yu Qiao. Mvbench: A comprehensive multimodal video understanding benchmark, 2024. 11, 12
- [64] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. arXiv preprint arXiv:2112.03857, 2021. 4
- [65] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024. 8, 9, 26
- [66] Xiang Li, Cristina Mata, Jongwoo Park, Kumara Kahatapitiya, Yoo Sung Jang, Jinghuan Shang, Kanchana Ranasinghe, Ryan Burgert, Mu Cai, Yong Jae Lee, et al. Llara: Supercharging robot learning data for vision-language policy. arXiv preprint arXiv:2406.20095, 2024. 3
- [67] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 292–305, Singapore, 2023. Association for Computational Linguistics. 8
- [68] Yi Li, Yuquan Deng, Jesse Zhang, Joel Jang, Marius Memme, Raymond Yu, Caelan Reed Garrett, Fabio Ramos, Dieter Fox, Anqi Li, Abhishek Gupta, and Ankit Goyal. Hamster: Hierarchical action models for open-world robot manipulation, 2025. 6
- [69] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023. 3, 12
- [70] Fangchen Liu, Kuan Fang, Pieter Abbeel, and Sergey Levine. Moka: Open-vocabulary robotic manipulation through mark-based visual prompting. arXiv preprint arXiv:2403.03174, 2024. 4
- [71] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 2023. 2, 3, 5, 9, 27
- [72] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023. 11
- [73] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2024. 3
- [74] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2024. 7
- [75] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 5, 9, 11, 25, 27
- [76] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. arXiv preprint arXiv:2402.08268, 2024. 4
- [77] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with blockwise ringattention, 2024. 5

- [78] Junpeng Liu, Tianyue Ou, Yifan Song, Yuxiao Qu, Wai Lam, Chenyan Xiong, Wenhu Chen, Graham Neubig, and Xiang Yue. Harnessing webpage uis for text-rich visual understanding, 2024. 10
- [79] Junpeng Liu, Yifan Song, Bill Yuchen Lin, Wai Lam, Graham Neubig, Yuanzhi Li, and Xiang Yue. Visualwebbench: How far have multimodal llms evolved in web page understanding and grounding?, 2024. 8, 9
- [80] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 4
- [81] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976– 11986, 2022. 5, 7
- [82] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 2022. 27
- [83] Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. Omniparser for pure vision based gui agent,

2024. 3, 4, 6, 9, 10, 26, 27

- [84] Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. Omniparser for pure vision based gui agent,

2024. 8

- [85] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019. 27
- [86] Ahmed Masry, Do Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279, Dublin, Ireland, 2022. Association for Computational Linguistics. 5, 7
- [87] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning,

2022. 27

- [88] Joanna Materzynska, Tete Xiao, Roei Herzig, Huijuan Xu, Xiaolong Wang, and Trevor Darrell. Something-else: Compositional action recognition with spatial-temporal interaction networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1049– 1059, 2020. 24, 27
- [89] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021. 27
- [90] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Ap-

- plications of Computer Vision, pages 1697–1706, 2022. 5, 7, 27
- [91] Russell Mendonca, Shikhar Bahl, and Deepak Pathak. Structured world models from human videos. arXiv preprint arXiv:2308.10901, 2023. 4
- [92] Meta. Llama-3. https://ai.meta.com/blog/ meta-llama-3/, 2024. 10
- [93] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR), pages 947–952. IEEE, 2019. 27
- [94] Soroush Nasiriany, Fei Xia, Wenhao Yu, Ted Xiao, Jacky Liang, Ishita Dasgupta, Annie Xie, Danny Driess, Ayzaan Wahid, Zhuo Xu, et al. Pivot: Iterative visual prompting elicits actionable knowledge for vlms. arXiv preprint arXiv:2402.07872, 2024. 4
- [95] Dantong Niu, Yuvan Sharma, Giscard Biamby, Jerome Quenum, Yutong Bai, Baifeng Shi, Trevor Darrell, and Roei Herzig. Llarva: Vision-action instruction tuning enhances robot learning. arXiv preprint arXiv:2406.11815, 2024. 3, 6
- [96] OpenAI. Gpt-4v(ision) system card. https://cdn. openai.com/papers/GPTV_System_Card.pdf,

2023. 3

- [97] OpenAI. ChatGPT. https://openai.com/blog/ chatgpt/, 2023. 3
- [98] OpenAI. Gpt-4 technical report, 2023. 3, 10
- [99] OpenAI. Gpt-4v(ision) system card. https://cdn. openai.com/papers/GPTV_System_Card.pdf,

2023. 9, 10

- [100] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 3
- [101] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021. 3, 7
- [102] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, ChaoYuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos, 2024. 6
- [103] Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. Android in the wild: A largescale dataset for android device control, 2023. 3, 26, 27
- [104] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 4

- [105] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In European conference on computer vision, pages 146–162. Springer, 2022. 27
- [106] ShareGPT. ShareGPT, 2023. 27
- [107] Peter Shaw, Mandar Joshi, James Cohan, Jonathan Berant, Panupong Pasupat, Hexiang Hu, Urvashi Khandelwal, Kenton Lee, and Kristina Toutanova. From pixels to ui actions: Learning to follow instructions via graphical user interfaces, 2023. 3
- [108] Peter Shaw, Mandar Joshi, James Cohan, Jonathan Berant, Panupong Pasupat, Hexiang Hu, Urvashi Khandelwal, Kenton Lee, and Kristina N Toutanova. From pixels to ui actions: Learning to follow instructions via graphical user interfaces. Advances in Neural Information Processing Systems, 36:34354–34370, 2023. 2
- [109] Gunnar A Sigurdsson, G¨ul Varol, Xiaolong Wang, Ali Farhadi, Ivan Laptev, and Abhinav Gupta. Hollywood in homes: Crowdsourcing data collection for activity understanding. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14, pages 510–526. Springer,

2016. 27

- [110] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 8
- [111] Srinivas Sunkara, Maria Wang, Lijuan Liu, Gilles Baechler, Yu-Chung Hsiao, Jindong Chen, Abhanshu Sharma, and James Stout. Towards better semantic understanding of mobile interfaces. CoRR, abs/2210.02663, 2022. 6, 27
- [112] Reuben Tan, Ximeng Sun, Ping Hu, Jui-hsien Wang, Hanieh Deilamsalehy, Bryan A Plummer, Bryan Russell, and Kate Saenko. Koala: Key frame-conditioned long video-llm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13581– 13591, 2024. 3
- [113] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024. 9
- [114] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 5
- [115] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3
- [116] Jiayu Wang, Yifei Ming, Zhenmei Shi, Vibhav Vineet, Xin Wang, Yixuan Li, and Neel Joshi. Is a picture worth a thousand words? delving into spatial reasoning for vision language models. arXiv preprint arXiv:2406.14852, 2024. 11

- [117] Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception, 2024. 3
- [118] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models, 2023. 10
- [119] Xidong Wang, Dingjie Song, Shunian Chen, Chen Zhang, and Benyou Wang. Longllava: Scaling multi-modal llms to 1000 images efficiently via hybrid architecture. arXiv preprint arXiv:2409.02889, 2024. 12
- [120] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021. 11
- [121] Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, and Afshin Dehghan. Slowfast-llava: A strong training-free baseline for video large language models. arXiv preprint arXiv:2407.15841, 2024. 12
- [122] An Yan, Zhengyuan Yang, Wanrong Zhu, Kevin Lin, Linjie Li, Jianfeng Wang, Jianwei Yang, Yiwu Zhong, Julian McAuley, Jianfeng Gao, Zicheng Liu, and Lijuan Wang. Gpt-4v in wonderland: Large multimodal models for zeroshot smartphone gui navigation, 2023. 3
- [123] An Yan, Zhengyuan Yang, Wanrong Zhu, Kevin Lin, Linjie Li, Jianfeng Wang, Jianwei Yang, Yiwu Zhong, Julian McAuley, Jianfeng Gao, et al. Gpt-4v in wonderland: Large multimodal models for zero-shot smartphone gui navigation. arXiv preprint arXiv:2311.07562, 2023. 4
- [124] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024. 10
- [125] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v, 2023. 3
- [126] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023. 4, 23
- [127] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Syn-

- ergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022. 10
- [128] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 10
- [129] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, and Minjoon Seo. Latent action pretraining from videos, 2024. 3
- [130] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al. Latent action pretraining from videos. arXiv preprint arXiv:2410.11758, 2024. 3
- [131] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. arXiv preprint arXiv:2310.07704, 2023. 2
- [132] Keen You, Haotian Zhang, Eldon Schoop, Floris Weers, Amanda Swearngin, Jeffrey Nichols, Yinfei Yang, and Zhe Gan. Ferret-ui: Grounded mobile ui understanding with multimodal llms, 2024. 3
- [133] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14, pages 69–85. Springer, 2016. 27
- [134] Hao Zhang, Hongyang Li, Feng Li, Tianhe Ren, Xueyan Zou, Shilong Liu, Shijia Huang, Jianfeng Gao, Lei Zhang, Chunyuan Li, and Jianwei Yang. Llava-grounding: Grounded visual chat with large multimodal models, 2023. 3
- [135] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023. 3
- [136] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 12
- [137] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-ofinterest. arXiv preprint arXiv:2307.03601, 2023. 3
- [138] Shaoqing Zhang, Zhuosheng Zhang, Kehai Chen, Xinbei Ma, Muyun Yang, Tiejun Zhao, and Min Zhang. Dynamic planning for LLM-based graphical user interface automation. In Findings of the Association for Computational Linguistics: EMNLP 2024, Miami, Florida, USA, 2024. Association for Computational Linguistics. 10
- [139] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model,

2024. 3, 27

- [140] Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. Gpt-4v(ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614, 2024. 9
- [141] Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. Gpt-4v(ision) is a generalist web agent, if grounded,

2024. 3, 10

- [142] Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daum´e III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. arXiv preprint arXiv:2412.10345, 2024. 3
- [143] Yang Zheng, Adam W Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J Guibas. Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19855–19865, 2023. 6
- [144] Luowei Zhou, Nathan Louis, and Jason J Corso. Weaklysupervised video object grounding from text by loss weighting and object interaction. In British Machine Vision Conference, 2018. 6
- [145] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 3
- [146] Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. Vision mamba: Efficient visual representation learning with bidirectional state space model. arXiv preprint arXiv:2401.09417, 2024. 3
- [147] Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Gao, and Yong Jae Lee. Segment everything everywhere all at once. NeurIPS, 2023. 4

[Figure 97]

## Magma: A Foundation Model for Multimodal AI Agents

### Supplementary Material

#### A. Pretraining and Finetuning

Setting Pretraining Finetuning

UI Image/Video Real Robot

batch size 1024 32 base learning rate 1e-5 1e-5 1e-5 1e-5 learning rate scheduler Constant Cosine Cosine Constant training epochs 3 3 1 20 optimizer adamw adamw adamw adamw

Image Resolution 512 768 768 256 Number of Crops 4 or 1 4 4 or 1 1

Table 9. Experimental settings pretraining and finetuning of Magma models. We maximally use either 32 Nvidia H100s or 64 AMD MI300 GPUs for all training jobs.

For all the model variants, we use the same training recipe as shown in Table 9. To handle different image resolutions from different datasets, we also use a multicrop strategy to enable batch forward for a given minibatch, though the ConvNext vision backbone can naturally support arbitrary resolutions. Specifically, for our pretraining, we use 512 as the base image size, and resize an input image maximally to 4 crops for UI and image pretraining data, while use 1 crop for video and robotics data.

For downstream finetuning, we following common practice to tune the pretrained magma model as shown in Table 9 right. As mentioned above, the vision encoder can be effortlessly adapted to different image resolutions required for different tasks.

#### B. Datasets

##### B.1. Pretraining Data

Due to space constraints, we briefly introduced the datasets for our pretraining in Sec 4.1 of our main submission. To ensure the reproducibility of our pretraining stage, we provide additional details of our pretraining data below.

###### B.1.1. UI Navigation

Our pretraining data related to UI agent are sourced from two datasets, SeeClick [19] and Vision2UI [41]. We further process these source data by adding marks on screenshots to provide grounded supervisions.

SeeClick. We generally follow the original procedure and make the following modifications to associate with the Set of Mark [126] strategy. For each webpage screenshot, multiple (text, bounding box) pairs are available. Therefore, we directly overlay all the bounding boxes with corresponding marks on the screenshot. For each mobile screenshot, only a

Source Task Size

text 2 point 271K text 2 bbox 54K point 2 text 54K bbox 2 text 54K

SeeClick-Web

text 2 point 274K text 2 bbox 56K UI summarization 48K widget captioning 42K

SeeClick-Mobile

input 2 point 980K input 2 bbox 982K

Visison2UI

text 2 point 794K text 2 bbox 774K point 2 text 199K bbox 2 text 193K

Magma-PT-UI (Ours) Mixed 2.8M

Table 10. Statistics of UI related pretraining data.

single (text, bounding box) pair is available in the SeeClick data. To enrich the pairs, we incorporate additional pairs from the RICO dataset [26], and employ an OCR tool to obtain text boxes. Finally, we display the enriched bounding boxes along with their corresponding marks on the mobile screenshot.

Vision2UI. We consider all bounding boxes whose “content” property is not null. To prevent the marks from overwhelming the main content of the webpage, we sample bounding boxes with varying probabilities based on their ”type” property. Specifically, we assign a sampling weight of 0.5 to boxes of type h1, h2, a, button, option, and nav with 0.5, while other types are weighted at 0.1. Given the high importance of input areas for interaction, we include boxes of type input directly without sampling for mark plotting. After obtaining the elements of high interest, we apply similar tasks as SeeClick [19] to produce the instruction data, including (a) grounding task, which involves two forms: predicting center point coordinates (text 2 point) and predicting bounding box (text 2 bbox); (b) generating text for elements, categorized into predicting text based on the coordinates of center points (point 2 text) or bounding boxes (bbox 2 text); and further introduce the task of (C) locating input fields, including predicting center point coordinates (input 2 point) and bounding box coordinates (input 2 bbox) of the input fields.

Given a webpage, since the first two categories of tasks are grounding or generating texts for the same

|[Figure 98]|
|---|
|{"from": "user",<br><br>"value": "<image>\nI'll provide bounding boxes on this webpage screenshot. Please describe the text content within each box, adding the mark index when relevant.\n (0.10, 0.29, 0.13, 0.30)"},<br><br>{"from": "assistant", "value": "Text: News. Mark: 1."},<br><br>...<br><br>{"from": "user",<br><br>"value": "Identify the presence of input fields in the provided screenshot and return their click point coordinates along with the mark index if applicable."},<br><br>{"from": "assistant", "value": "Coordinate: (0.17, 0.43). Mark: 3. Coordinate: (0.31, 0.43). Mark: 4."}<br><br>(a)|

|[Figure 99]|
|---|
|{"from": "user",<br><br>"value": "<image>\nFrom this webpage capture, I'll give descriptions. Predict each element's bounding box and mark index, if present.\nNovember 7, 2013"}, {"from": "assistant",<br><br>"value": "Coordinate: (0.27, 0.29, 0.37, 0.30). Mark: 2."}, {"from": "user",<br><br>"value": "\"You're Whole\" Summary"},<br><br>{"from": "assistant", "value": "Coordinate: (0.07, 0.31, 0.66, 0.33). Mark: 0."},<br><br>... {"from": "user",<br><br>"value": "Analyze the screenshot for input fields and provide their bounding box coordinates and mark index if they exist."},<br><br>{"from": "assistant",<br><br>(b)"value": "No input areas found.\n"}<br><br>|

|[Figure 100]|
|---|
|{"from": "user",<br><br>"value": "<image>\nOn this software's interface, to execute the step \"its a stopwatch that measures time\", where do I direct my attention? Please provide the coordinate and the bounding box's mark index if<br><br>applicable."},<br><br>{"from": "assistant", "value": "Coordinate: (0.04, 0.12). Mark: 12"}<br><br>(c)|

|[Figure 101]|
|---|
|{"from": "user",<br><br>"value": "<image>\nHow does the element at (0.93, 0.41) (Mark 0) contribute to the overall user experience?"},<br><br>{"from": "assistant", "value": "make selection"}<br><br>(d)|

|[Figure 102]|
|---|
|{"from": "user",<br><br>"value": "<image>\nCould you depict the content and layout of the screen image provided?"},<br><br>{"from": "assistant",<br><br>"value": "display page showing list of items in shopping app"}<br><br>(e)|

- Figure 12. Training samples in our Magma-PT-UI. It covers a wide range of action grounding and UI understanding tasks including: (a) Given the bounding box or point coordinates as the query, assistant should return the natural language description or the content. (b) Given the natural language or the exact content as the query, assistant should return the value of the bounding box coordinates.. (c) Given the natural language as the query, assistant should return the value of the point coordinate. (d) Widget captioning. (e) UI summarization.

group of web elements, we further weight the four subtasks, i.e., (text 2 point), (text 2 bbox), (point 2 text), and (bbox 2 text) with [0.4, 0.4, 0.1, 0.1], and sample only one of them to construct the pretraining data. Similarly, we sample one subtask from (input 2 point) and (input 2 bbox) with equal probabilities. We merge the sampled subtasks from the same webpage into one example to improve training efficiency. We denote the full pretraining data related to UI by Magma-PT-UI, and list the sizes of individual subsets in Table 10.

###### B.1.2. Instructional Videos

As mentioned in the main submission, we curate the supervisions from human instructional videos to learn the agentic capability for our model. To cover different scenarios, we considered both 3rd point view videos and egocentric videos. In particular, we start with Epic-Kitchen [24] video data sets considering that their text annotations are relatively high quality. Afterwards, we expand to SomethingSoomething v2 [88] to include more human-object interactions, and Ego4D [40] and other related instructional videos for scaling up.

Epic-Kitchen [24]. Epic-Kitchen contains 495 egocentric videos recorded by 32 participants in kitchen rooms. Each video contains a number of segments labeled with narrations, start and end frame ids. However, the original video narrations (e.g., “open door”) are too coarse to depict the human actions in a certain time frame.

For the videos in Epic-Kitchen, we apply the video preprocessing method as discussed in Sec 4.2 of our main sub-

mission. Concretely, for each of the original video segments in the dataset, we run PySceneDetect to detect the temporal boundaries and split them into sub-segments. During our model pretraining, the textual annotations are used in two ways. Our model is asked to predict the detailed description in the first frame. In addition, they are used as the task description as input to the model for predicting the traces of marks.

Sth-Sth-v2 [88], Ego4D [40]. The Sth-Sth v2 dataset is a comprehensive collection of labeled video clips featuring humans performing predefined actions with everyday objects. The list of action classes spans a wide variety of atomic actions, including but not limited to “pushing something from right to left”, “throwing something” and “covering something with something”. In total, the dataset contains 220,847 seconds-long video clips. To create our pretraining data, we only leverage the videos in the train and validation splits. This amounts to around 160K video clips. We note that we do not use PySceneDetect for Sth-Sth v2 since the original video clips have been highly curated.

The Ego4D dataset is a large-scale egocentric dataset that contains approximately 3,025 hours of videos. It comprises over 3,670 hours of video footage captured from wearable cameras across a diverse environments and activities. The dataset spans a wide range of real-world scenarios, including daily activities and social interactions. Given the duration of these videos can span over 30 minutes, we leverage the original dense caption annotations that are provided to split each videos into seconds-long segments with consistent views.

Most common verbs in UI navigation Most common verbs in robotic manipulation

[Figure 103]

[Figure 104]

(a) (b)

Most common verbs in instructional videos

[Figure 105]

(c)

- Figure 13. Action distributions in three types of action-oriented pretraining datasets. (a) UI Navigation; (b) Robotic Manipulation; (c) Instructional Videos.

###### B.1.3. Robotic Manipulation

We follow the training recipe in OpenVLA [54] to prepare our pretraining data for robotics manipulation. Specifically, we take the data mixture “siglip-224px+mx-oxemagic-soup” as in OpenVLA, which gives us 9.4M imagelanguage-action triplets, extracted from 326K trajectories, from 23 separate datasets.

###### B.1.4. Multimodal Image Understanding

We simply include the 1.2M synthetic image-text pairs in ShareGPT4V [13] and 665K image instruction tuning data collected by LLaVA-1.5 [75] as our multimodal image pretraining data. The former helps our pretrained model to have a global understanding of visual contents, while the latter helps to get the model familiar with various types of human instructions. We denote this dataset by Magma-PTImage.

###### B.1.5. Data Statistics

Given our goal of training a general vision-language-action foundation model, we analyze the distribution of verbs present in the text annotations of the UI and robotic manipulation as well as instructional video datasets in Figure 13.

We see that the text annotations in the UI navigation component contain many helpful verbs that help guide agents to achieve a specific task such as “locate” and “turn”. This is complemented by the more action-oriented words in the vocabulary of the robot manipulation component, including “pick”, “push” and “slide”. Such annotations are especially valuable in helping our Magma model to learn to reason about interactions with everyday objects. Finally, we also scale up the amount of training data and diversity of verbs by including data from instructional videos (Figure 13c). As evidenced by the relatively high frequency of words such as “lifting” and “throwing”, such annotations can be very beneficial for gaining a stronger understanding the of temporal dynamics involved in common activities. More importantly, the diversity of activities present in these datasets can be effective at helping the model generalize better to a larger variety of tasks.

##### B.2. Downstream Data

###### B.2.1. UI Agent Navigation

We evaluated the UI grounding and navigation capability mainly on three datasets, ScreenSpot [19], Mind2Web [27]

[Figure 106]

|[Figure 107]<br><br>Push Cloth Left to Right|
|---|

|[Figure 108]<br><br>Pick Place Hotdog Sausage|
|---|

|[Figure 109]<br><br>Put Mushroom Place Pot|
|---|

|[Figure 110]<br><br>Push Cloth Right to Left|
|---|

WidowX 250

Robot

Gripper

- Figure 14. Real robot setup. Magma is deployed on a WidowX 250 robot arm to perform a sequence of kitchen manipulation tasks including object pick-place and soft manipulation.

and AITW [103].

ScreenSpot is a benchmark used to evaluate the UI action grounding proposed in [19]. It consists of 600 screenshots images associated with 1.2K instructions spanning iOS, Android, macOS, Windows, and web pages. The evalaution covers both text based elements and a variety of widgets and icons. To evaluate the zero-shot action grounding performance for our model, we use OmniParser [83] to help parse the screenshot and propose actionable regions/icons/buttons. We used the sample code and default settings provided in the official repo. For these candidate regions, we overlay numeric marks and ask our model to pick one.

Mind2Web is first proposed in [27] for text-based web agent. For fair comparison among vision-based web agent, we follow the protocol proposed in SeeClick [19]. Given a webpage, we convert it into a screenshot associated with ground-truth bounding boxes to which the actions should be applied. As the original screenshot of the full website is usually out of the scope of display. We follow a similar way as in [19] to crop the region of interests centering around the ground truth boxes, which gives us a local screenshot as wide as original webpage but with maximal height 1344. To propose the candidate marks for our model, we directly exploit the candidate ranks provided in Mind2Web, and use the top 30 candidates for evaluation.

AITW is a dataset originally collected in [103] for navigation of the android UI. The original dataset contains up to 715K trajectories, resulting in 5.7M screenshots. In our experiments, to examine the efficient finetuning performance, we alteratively follow the same protocol in SeeClick [19] and include a much smaller number of training samples. Specifically, there are 545, 688, 306, 700, 700 instructions from General/Install/GoogleApps/Single/WebShopping, respectively. 80% of each split is used for training and the remainder is used for evaluation. Instead of finetuning our model for each category, we jointly finetune our pretrained Magma on the combined data and evaluate across all categories using a single model.

###### B.2.2. Robot Manipulation

Simulator. We employ SimplerEnv [65] as the main testbed for our learned robot policy. As we do not need to tune our model on the simulated trajectories, we simply report the numbers following the protocol proposed in the original work.

Real-world Setting. We design four tabletop manipulation tasks for our physical WidowX-250 robot setup as shown in 14. As with BridgeData-v2, the RGB image observations from the robot are captured using a stationary third-person camera, maintaining a resolution of 256 × 256. For finetuning our pretrained Magma model, we collect approximately 50 robot demonstration trajectories for each task as our finetuning dataset. Our experimental design includes classic soft object manipulation and pick-and-place operations tasks. Detailed language instructions for the designed tasks are presented below. For each trial, we randomize the initial location of the target object and include 2-3 random distracting objects (e.g., corn, eggplant) in the scene. For reproducibility, we release the collected robot trajectories.

Tasks included in the finetuning dataset:

- • Hot dog assembly: Pick up the hot dog sausage from the desk and place it into the bun. The trial is counted as success only when the robot successfully grasps the sausage and accurately places it within the hot dog bun.
- • Mushroom placement: Pick up the mushroom and place it into the pot. The trial is counted as success only when the robot correctly grasps the mushroom and places it into the cooking pot without dropping or misaligning it.
- • Cloth pushing: Push the cloth from right to left across the surface. The trial is counted as success only when the robot successfully manipulates the cloth in the specified direction without disturbing other objects on the surface.

Unseen task for evaluating generalization:

• Bidirectional cloth manipulation: Push the cloth in both directions while maintaining its shape. This task examines the model’s spatial understanding and reasoning capabilities, as it requires generalization from unidirectional pushing in the training data to bidirectional manipulation in novel scenarios.

Dataset Size Domain

|ShareGPT [106] ShareGPT4V [13] LLaVA-Instruct [71] LAION-GPT4V [58]|40K Text<br><br>39K General 158K General 11K General<br><br>|
|---|---|
|VQAv2 [39] GQA [45] OKVQA [105] OCRVQA [93] ChartQA [87] DVQA [46] DocVQA [89] AI2D [51] SynthDog-EN [53] A-OKVQA RefCOCO [133] VG [57]<br><br>|83K General VQA 72K General VQA<br><br>9K Knowledge VQA 80K OCR VQA<br><br>7K Chart VQA 16K Chart VQA<br><br>10K Document VQA 2K Infographic VQA<br><br><br>20K Document Understanding 66K Knowledge VQA 48K Grounding Desc. 86K Referring Exp.|

InfographicsVQA [90] 24k Infographic VQA ChartQA (Aug) [87] 20k Chart VQA FigureQA [47] 20k Chart/Figure VQA TQA [52] 1.5k Textbook VQA ScienceQA [82] 5k Textbook VQA Magma-SFT-Image (Ours) 820k Mixed

Table 11. A detailed breakdown of our 820k Magma image instruction tuning data used in our multimodal image understanding experiments shown in Table 5 in our main submission.

###### B.2.3. Image Instruction Tuning

We show a breakdown of our 820k Magma image instruction tuning data in Table 11. As the 760k image instruction tuning data used in LLaVA-1.6 [75] is not released, we follow their guidance to curate 748k public available data including ShareGPT [106], LLaVAInstruct [71], ShareGPT4V [13], LAION-GPT4V [58], VQAv2 [38], GQA [45], OKVQA [85], OCRVQA [93], ChartQA [87], DVQA [46], DocVQA [89], AI2D [51], SynthDog-EN [53], A-OKVQA [105], RefCOCO [50] and VG [57]. To complement the claimed “improved reasoning, OCR and world knowledge”, we resort to a few other opensourced datasets including InfoGraphicsVQA [90], augmented ChartQA [87], FigureQA [47], TQA [52] and ScienceQA [82]. We denote the full set by Magma-SFT-Image.

###### B.2.4. Video Instruction Tuning

For comparisons with state-of-the-art video LMMs, we adopt the LLava-Video-178K dataset [139] for instruction tuning. It consists of approximately 1.6M video and text instruction samples from 178K videos. The dataset is compiled from multiple video sources ranging from Charades [109], Sth-SthV2 [88] to Kinetics-700 [9]. We refer interested readers to the original papers for more details.

###### B.2.5. Details about SoM for training and evaluation

we exploit three ways to extract the candidate bounding boxes for the SoM prompt:

- • DoM Tree. In addition to the bounding boxes extracted from HTML code [19, 41], we further annotate the mobile screenshots in SeeClick data with bounding boxes derived from Android view hierarchies [111]. These annotations are used during our model pretraining.
- • Vision model. For zero-shot evaluation on Screenspot [19], we exploit the OmniParser model [83] to make a fair comparison with the state-of-the-art methods [19, 83]. Note that we only use the bounding boxes without local semantics. The original bounding boxes in AITW [103] are identified using an OCR model and IconNet [111].
- • Language model. For evaluation on As discussed earlier, we directly apply the predictions provided by Mind2Web [27] using a pretrained language model DeBERTa-v3-base. This model gives approximately 85% recall@50.

#### C. Qualitative Analysis

##### C.1. UI Navigation

Given the performant UI navigation performance across different tasks, we show some Mobile UI navigation samples in Fig. 15. We prompt the model to complete two daily tasks starting from the home page: “What’s the weather like in Tokyo” and “Install app ‘Instagram’”. Despite that our model is never trained with the full trajectory, it can handle the tasks in the wild pretty well.

##### C.2. Robotics Manipulation

We further show the real robot manipulation rollout for OpenVLA and Magma model. As discussed in our main paper, our model exhibits much better generalization ability to different real robot manipulation tasks. In Fig. 16, we qualitatively show how two models handle a complicated task of “Pick up the sausage and put it inside the hotdog”. Thanks to the proposed pretraining techniques, our Magma model can not only precisely pick up the sausage but also move smoothly to the top of the hotdog, demonstrating superior spatial understanding and reasoning capability compared with the counterpart.

###### Task #1: What’s the weather like in Tokyo

###### Instruction Template

|[Figure 111]<br><br>|
|---|

|[Figure 112]<br><br>[Figure 113]|
|---|

|[Figure 114]<br><br>[Figure 115]|
|---|

|[Figure 116]<br><br>[Figure 117]|
|---|

<image> Imagine you are interacting with an Android device step by step. You can see the screen and know the previous actions through recorded history. The screenshots are labeled with bounding boxes and numeric marks to guide your interactions. Your task is to determine the next action to perform. The possible actions include tapping, swiping, navigating, typing, or setting the task status.

Use the following format for your output, selecting the appropriate key-value pair: ```json {

"TAP": dict, // Use {"MARK": mark_index} where `mark_index` is an integer corresponding to the bounding box on the screen, or {"NORMALIZED_COORDINATE": [y, x]} where `y` and `x` are float numbers between 0 and 1 representing the normalized coordinates on the screen.

"SWIPE": str, // Choose "UP", "DOWN", "LEFT", or "RIGHT" "NAVIGATE": str, // Choose "BACK", "HOME", or "ENTER" "TYPE": str, // The text to be typed "SET_TASK_STATUS": str, // Choose "COMPLETE" or "IMPOSSIBLE"

} ```

Current task: <current_task>

Previous actions:

<action_history>

Determine your next action (Step <step_idx>) and provide the output in the specified dictionary format.

{"TAP": {“Coordinate": [0.54, 0.88]}} {"TYPE": "What's the weather like in Tokyo"} {“NAVIGATE": "ENTER"} {"SET_TASK_STATUS": "COMPLETE"}

###### Task #2: install app “Instagram”

|[Figure 118]<br><br>|
|---|

|[Figure 119]<br><br>|
|---|

|[Figure 120]<br><br>|
|---|

|[Figure 121]<br><br>[Figure 122]|
|---|

|[Figure 123]<br><br>|
|---|

|[Figure 124]<br><br>|
|---|

|[Figure 125]<br><br>[Figure 126]|
|---|

{"TAP": {"MARK": 0}} {"TAP": {"MARK": 1}} {"TAP": {"MARK": 0}} {"TYPE": "Instagram"} {"TAP": {"MARK": 11}} {"TAP": {"MARK": 11}} {"SET_TASK_STATUS": "COMPLETE"}

- Figure 15. Examples for mobile UI navigation sample. We prompt the model with two tasks: “What’s the weather like in Tokyo” and “Install app ‘Instagram’”. The model take actions sequentially given the new observation and history action information.

[Figure 127]

(a) Robot policy rollout for task “Put the sausage to hotdog” for OpenVLA model. (Failure)

[Figure 128]

(b) Robot policy rollout for task “Pick up the mushroom to the pot” for OpenVLA model. (Failure)

[Figure 129]

(c) Robot policy rollout for task “Put the sausage to hotdog” for Magma model. (Success)

[Figure 130]

(d) Robot policy rollout for task “Pick up the mushroom to the pot” for Magma model. (Success)

- Figure 16. Comparison between OpenVLA (top two rows) and Magma (bottom two rows) for real robot manipulation task. The two robot policies starts with the same initial stage and asked to perform exactly the same task. The whole task requires precise spatial understanding and planning for the model. For both tasks, OpenVLA failed to accomplish while our model successfully handle.

