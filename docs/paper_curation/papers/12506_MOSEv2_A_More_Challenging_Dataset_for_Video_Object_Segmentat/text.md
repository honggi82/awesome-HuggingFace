# MOSEv2: A More Challenging Dataset for Video Object Segmentation in Complex Scenes

Henghui Ding, Kaining Ying, Chang Liu, Shuting He, Xudong Jiang, Fellow, IEEE, Yu-Gang Jiang, Fellow, IEEE, Philip H.S. Torr, Song Bai

## arXiv:2508.05630v2[cs.CV]22Sep2025

Abstract—Video object segmentation (VOS) aims to segment specified target objects throughout a video. Although state-of-the-art methods have achieved impressive performance (e.g., 90+%  &) on benchmarks such as DAVIS and YouTube-VOS, these datasets primarily contain salient, dominant, and isolated objects, limiting their generalization to real-world scenarios. To bridge this gap, the coMplex video Object SEgmentation (MOSEv1) dataset was introduced to facilitate VOS research in complex scenes. Building on the foundations and insights of MOSEv1, we present MOSEv2, a significantly more challenging dataset designed to further advance VOS methods under real-world conditions. MOSEv2 consists of 5,024 videos and 701,976 high-quality masks for 10,074 objects across 200 categories. Compared to its predecessor, MOSEv2 introduces much greater scene complexity, including more frequent object disappearance and reappearance, severe occlusions and crowding, smaller objects, as well as a range of new challenges such as adverse weather (e.g., rain, snow, fog), low-light scenes (e.g., nighttime, underwater), multi-shot sequences, camouflaged objects, non-physical targets (e.g., shadows, reflections), and scenarios requiring external knowledge. We benchmark 20 representative VOS methods under 5 different settings and observe consistent performance drops on MOSEv2. For example, SAM2 drops from 76.4% on MOSEv1 to only 50.9% on MOSEv2. We further evaluate 9 video object tracking methods and observe similar declines, demonstrating that MOSEv2 poses challenges across tasks. These results highlight that despite strong performance on existing datasets, current VOS methods still fall short under real-world complexities. Based on our analysis of the observed challenges, we further propose several practical tricks that enhance model performance. MOSEv2 is publicly available at https://MOSE.video.

Index Terms—Video Object Segmentation, Complex Scenes, MOSE Dataset, MOSEv2.

✦

### 1 INTRODUCTION

VIDEOobjectsegmentation(VOS)[1],[2],[3],[4]aimsto

segment specified target objects throughout an entire video. It is one of the most fundamental and challenging computer vision tasks, playing a crucial role in various practical applications involving video analysis and understanding, such as autonomous vehicle, augmented reality, and video editing. There are different settings for VOS, for example, semi-supervised VOS [5], [6] that gives the first-frame mask, bounding box, or points of the target object, unsupervised VOS [7], [8] that automatically finds primary or salient objects, and interactive VOS [9], [10] that relies on user interactions with the target object. VOS has been extensively studied in the past using traditional techniques [11], [12], [13] and deep learning methods [14], [15]. Deep-learning-based approaches have greatly improved VOS performance and surpassed traditional techniques by a large margin.

Current state-of-the-art VOS methods [14], [15], [16] have achieved near-saturation performance on two of the commonlyused VOS datasets DAVIS [2], [3] and YouTube-VOS [4]. For example, XMem [16] achieves 92.0%  & on DAVIS 2016 [2], 87.7%  & on DAVIS 2017 [3], and 86.1%  & on YouTubeVOS [4]. With such a high performance, it seems that video

∙ Henghui Ding, Kaining Ying, and Yu-Gang Jiang are with Fudan University, Shanghai, China. (e-mail: henghui.ding@gmail.com)

∙ Chang Liu and Song Bai are with ByteDance Inc.

∙ Shuting He is with Shanghai University of Finance and Economics, China.

∙ Xudong Jiang is with Nanyang Technological University, Singapore.

∙ Philip H.S. Torr is with University of Oxford, United Kingdom.

∙ Henghui Ding and Kaining Ying are co-first authors.

object segmentation has been well resolved. However, do we really perceive objects in realistic scenarios?

To explore this question, we introduced the coMplex video Object SEgmentation (MOSEv1) dataset in [1], revisiting VOS under more realistic and complex scenes where traditional datasets fall short. In contrast to DAVIS [2], [3] and YouTube-VOS [4], where target objects are typically salient and isolated, MOSEv1 focuses on challenging cases such as object disappearance and reappearance, small or inconspicuous objects, heavy occlusions, and crowded scenes. These real-world conditions significantly affect segmentation performance, with XMem [16] only achieving 57.6%  & on MOSEv1. Since its release in 2023, MOSEv1 has attracted broad and growing attention from the research community. Several competitions have been organized based on this dataset, including PVUW [17], [18] and LSVOS [19], facilitating research in this area. Meanwhile, a series of strong VOS methods such as SAM2 [15] have subsequently pushed the performance from initial baselines to 76.4%  &, highlighting both the difficulty and the value of the dataset while demonstrating substantial progress in addressing complex video segmentation scenarios.

In this work, building on the foundations and insights of MOSEv1, we present MOSEv2, a more challenging dataset that further pushes the boundaries of VOS in real-world scenes. MOSEv2 significantly increases the complexity across multiple dimensions. Core challenges of MOSEv1, such as object disappearancereappearance, occlusions, small objects, and crowded scenes, are retained but appear more frequently, with greater severity, and under more realistic conditions. Beyond that, MOSEv2 introduces a range of new challenges rarely covered in previous datasets, including adverse weather (e.g., rain, snow, fog), low-

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

①

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

- ②

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

- ③

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

- ④

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

- ⑤

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

- ⑥

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

- ⑦

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

- ⑨

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

- ⑩

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

- ⑧

Fig. 1: Example videos from the proposed MOSEv2 dataset. Selected target objects are masked in orange. The target in case ① is enlarged for better visualization. The most notable features of MOSEv2 include both challenges inherited from MOSEv1 [1] such as object disappearance-reappearance (①-⑩), small/inconspicuous objects (①,③,⑥), heavy occlusions (except ⑤), and crowded scenes (①,②), as well as newly introduced complexities such as adverse weather (⑥), low-light environments (⑤-⑦), multi-shots (⑧), camouflaged objects (⑤), non-physical objects (④), and knowledge dependency (⑨,⑩). The goal of MOSEv2 dataset is to provide a platform that promotes the development of more comprehensive and robust video object segmentation algorithms.

701,976 high-quality segmentation masks. Representative examples are shown in Fig. 1, illustrating both intensified and newly introduced challenges. A common pattern is object disappearance and reappearance, as shown in the 3rd example where a vehicle repeatedly disappears and reappears under overpasses, requiring robust temporal association. Challenges like small or inconspicuous objects, crowded scenes, and severe occlusions are also more prominent. For example, in the 1st example, a tiny person moves through a dense crowd, frequently occluded by others. Examples 4-10 highlight some new challenges in MOSEv2. Adverse weather

light scenes (e.g., nighttime, underwater), multi-shot sequences, camouflaged objects, non-physical targets (e.g., shadows, reflections), knowledge-dependent scenarios, etc. These additions aim to bridge the gap between current VOS datasets and the diverse, unconstrained nature of real-world scenes. With these multifaceted complexities, MOSEv2 serves as a next-generation benchmark for evaluating and advancing complex video object segmentation under realistic, dynamic, and highly unconstrained environments.

MOSEv2 consists of 5,024 videos and 10,074 annotated object instances spanning 200 diverse categories, resulting in

and video object tracking.

(e.g., fog in the 6th), low-light conditions (e.g., underwater in the 5th, nighttime in the 7th), and multi-shot sequences (e.g.,

∙ We perform an in-depth analysis of model performance and failure cases on MOSEv2, highlighting the key challenges it poses. Building on these insights, we propose practical tricks that substantially enhance model performance in complex scenarios, and outline future directions for advancing robust video understanding in the wild.

- 8th) introduce appearance instability, motion ambiguity, and temporal discontinuities. These demand strong generalization and long-range association. Moreover, MOSEv2 includes novel object categories that are difficult for existing methods. For example, camouflaged objects (5th) blend into backgrounds, while nonphysical targets like shadows (4th) lack stable visual cues and change shape based on external factors. In addition, MOSEv2 further introduces knowledge-dependent scenarios (e.g., 9th and 10th examples) that require high-level reasoning. For example, the
- 9th example requires optical character recognition to differentiate similar-looking blocks, while the 10th involves physics-based causality, where the target must be inferred from surrounding motion despite being invisible. These diverse and fine-grained challenges make MOSEv2 a comprehensive dataset for studying the robustness and generalization capabilities of VOS in openworld complex scenes. We expect MOSEv2 to spur meaningful progress toward real-world video understanding and deployment.

2 RELATED WORK

2.1 Video Object Segmentation

Video object segmentation (VOS) aims to segment a specific object throughout a video. Based on how the target object is specified, VOS can be categorized into four main settings: 1) semisupervised VOS (also known as semi-automatic VOS [30] or oneshot VOS), 2) unsupervised VOS (also called automatic VOS or zero-shot VOS), 3) interactive VOS, and 4) referring VOS.

∙ Semi-supervised VOS. Semi-supervised VOS [5] aims to segment the target object throughout a video, given its mask in the first frame. Most existing works can be categorized into propagation-based methods [4], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44] and matching-based methods [14], [16], [37], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55]. Propagation-based methods leverage the predicted mask from the previous frame to guide the segmentation of the current frame, thereby propagating object cues in a frameby-frame manner. Matching-based methods, on the other hand, first encode the target object into an embedding space and then perform per-pixel classification by comparing the similarity between each pixel’s feature and the stored object embedding. Since obtaining pixel-level annotations is often expensive and timeconsuming, some methods employ bounding box as the first-frame reference [56], [57], [58]. For example, SiamMask [56] integrates a mask prediction branch into a fully convolutional Siamese object tracker to generate binary segmentation masks.

To thoroughly analyze the proposed MOSEv2 dataset, we retrain and benchmark 20 representative VOS methods under different settings. Experimental results demonstrate that the complexity of real-world videos in MOSEv2 significantly degrades the performance of current state-of-the-art VOS methods. For example, the  & score of SAM2 [15] reaches 90.7% on DAVIS 2017 [3] and 76.4% on MOSEv1 [1], but notably drops to 50.9% on MOSEv2. Similarly, Cutie [14] achieves 87.9% on DAVIS 2017 and 69.9% on MOSEv1, but markedly declines to 43.9% on MOSEv2. These consistent performance drops highlight the significant challenges posed by the more realistic and complex scenarios in MOSEv2.

Beyond VOS, MOSEv2 extends naturally to a wide range of video perception tasks requiring fine-grained understanding. In particular, we demonstrate its applicability to video object tracking (VOT) by benchmarking 9 state-of-the-art VOT methods [15], [20], [21], [22], [23], [24], [25], [26], [27] on MOSEv2. While these methods perform well on standard VOT benchmarks such as LaSOT [28] and GOT-10k [29], consistent and notable performance drops are observed on MOSEv2. For example, SAMURAI [25] achieves 74.2% AUC on LaSOT but only 36.1% on MOSEv2, revealing that MOSEv2 introduces new and significant challenges not only for VOS but also for VOT. This demonstrates the broader applicability of MOSEv2 as a strong foundation for video understanding research in realistic and complex scenes.

Recently, SAM2 [15] adopts promptable visual segmentation, which allows the model to accept prompts in the form of positive/negative clicks, bounding boxes, or masks on any frame of a video. This flexible interaction significantly improves the model’s adaptability and generalization across diverse scenarios. Following SAM2, several efficient extensions [25], [26], [27], [59] have been proposed to improve its performance. For example, SAM2Long [27] addresses error accumulation by exploring multiple segmentation pathways via constrained tree search. DAM4SAM [26] introduces a distractor-aware memory and an introspection-based update strategy to mitigate ambiguity from visual distractors. To better handle dynamic scenes, recent works [25], [59] incorporate motion modeling into promptable segmentation. SAMURAI [25] integrates Kalman filtering [60] for adaptive memory selection, while MoSAM [59] enhances robustness through motion-aware sparse and dense prompts combined with spatiotemporal memory mechanisms. These SAM2 variant methods achieve impressive performance on the previous VOS datasets [1], [2], [3], [4], [15], [61], [62].

In summary, our main contributions are as follows:

∙ We present MOSEv2 (coMplex video Object SEgmentation), a more challenging dataset for video object segmentation in complex scenes. Compared to MOSEv1 [1], MOSEv2 introduces more frequent object disappearance-reappearance, more severe occlusions, denser crowding, and smaller targets, and also new complexities such as adverse weather, low-light scenes, multi-shot videos, camouflaged objects, non-physical targets, and knowledge-dependent scenarios.

∙ We provide detailed comparative analysis between MOSEv2 and existing VOS and VOT datasets, highlighting its unique challenges and greater complexity that better represent realworld video understanding scenarios.

∙ Interactive VOS. This task aims at segmenting the target object in a video indicated by user’s interaction (e.g., clicks or scribbles) [9], [10], [37], [63], [64], [65], [66], [67], it is a special form of semi-supervised VOS. Existing methods mainly follow a paradigm of interaction-propagation way. Besides the feature encoder that extracts pixel features, there are other two modules placed on the feature encoder to achieve interactive video object segmentation,

∙ We conduct comprehensive benchmarks of state-of-the-art methods on MOSEv2 across various VOS and VOT settings, including semi-supervised VOS with mask, box, and point initialization, as well as unsupervised VOS, interactive VOS,

∙ Video Semantic Segmentation (VSS). Driven by the success in image semantic segmentation [112], [113], [114] and large-scale video semantic segmentation datasets [115], [116], [117], video semantic segmentation has drawn lots of attention and achieved significant achievements. Compared to image domain, temporal consistency and model efficiency are the new efforts in the video domain. For example, Sun et al. [118], [119] propose Coarse-toFine Feature Mining to capture both static context and motional context. Syed Hesham et al. [120] propose a state space modelbased [121] architecture for efficient temporal feature sharing.

i.e., interactive segmentation module that corrects prediction based on user’s interaction and mask propagation module that propagates user-corrected masks to other frames. SAM2 [15] has also demonstrated strong capabilities in this task, offering superior performance with flexible interaction mechanisms, significantly enhancing both segmentation quality and user experience.

∙ Referring VOS. This is an emerging setting that aims to segment the target object in a video according to a text expression [68], [69], [70], [71], [72], [73]. Early methods can be broadly classified as bottom-up methods and top-down methods. Bottom-up methods [70], [74], [75] perform first-frame segmentation followed by mask propagation or per-frame segmentation with post-hoc association. Top-down methods [76], [77] first generate candidate tracklets and then select the one best aligned with the expression. The introduction of motion-centric datasets in MeViS [68] and MeViSv2 [69] has drawn increased attention to the importance of temporal dynamics. Subsequent works [78] highlight that temporal modeling is essential for accurate grounding. Recent works [79], [80] also leverage multimodal large language models [81], [82] to handle expressions requiring complex reasoning, which enables human-like understanding and generalization ability across diverse language descriptions. With the latest datasets such as MeViSv2 [69] and OmniAVS [83] supporting expressions across multiple modalities, omnimodal referring VOS is expected to gain increasing attention in future research [84].

∙ Video Panoptic Segmentation (VPS). Kim et al. [122] introduce panoptic segmentation to the video domain to simultaneously segment and track both the foreground instance objects and background stuff. They also build Cityscapes-VPS dataset with 500 videos. Then, Miao et al. [123] build a larger VPS dataset called VIPSeg with 3,536 videos. Existing methods [124], [125] mainly add temporal refinement or cross-frame association modules upon image panoptic segmentation models [126] to enhance temporal conformity and instance tracking performance. Li et al. [127] propose OMG-Seg, a unified transformer-based model that supports video panoptic segmentation along with over ten other segmentation tasks via task-specific queries and outputs. ∙ Video Object Tracking (VOT). Different from VOS that focuses on segmentation, VOT [128] aims to locate a target object with bounding boxes in subsequent frames given its initial bounding box annotation. VOT has seen significant progress in recent years, with methods designed to handle challenging scenarios such as scale variations, occlusions, distractors, and complex backgrounds. The dominant approaches can be broadly categorized into Siamese-based methods [129], [130], [131] that learn discriminative feature embeddings through twin networks, and transformerbased methods [132], [133], [134], [135], [136] that leverage selfattention mechanisms to model long-range dependencies for robust tracking. These methods have achieved impressive performance on existing VOT benchmarks like VOT [137], LaSOT [28], and GOT10k [29]. The proposed MOSEv2 dataset also supports VOT task while introducing more complex real-world scenarios like dense crowds, occlusions, and frequent disappearance-reappearance that pose significant challenges to existing tracking methods.

∙ Unsupervised VOS. This setting requires no manual input and aims to automatically segment primary objects in a video [85], [86], [87], [88], [89], [90], [91], [92], [93], [94], typically focusing on objects from pre-defined categories. Early methods relied on post-processing [85], while later end-to-end methods became mainstream, broadly divided into local content encoding and contextual encoding. Local content encoding methods [7], [8], [91], [93], [95], [96], [97] often employ two-stream architectures to separately process optical flow and RGB information. Contextual content encoding methods [98], [99], [100] aim to capture long-range dependencies and global context. Recent methods have adapted propagation frameworks for this task, DEVA [101] proposes a decoupled framework combining image-level segmentation with class-agnostic temporal propagation, eliminating the need for taskspecific video training data. EntitySAM [102] extends SAM2 for zero-shot video entity segmentation by automatically discovering and tracking all entities without explicit prompts.

#### 2.3 Complex Scene Understanding

Complex scene understanding has become a research focus in the image understanding domain [138], [139], [140], [141], [142], [143], [144], [145], [146], [147], [148], [149]. For example, Ke et al. [150] propose Bilayer Convolutional Network (BCNet) to decouple overlapping objects into occluder and occludee layers. Zhang et al. [140] propose a self-supervised approach to conduct de-occlusion by ordering recovery, amodal completion, and content completion. On the video domain, however, occlusion understanding is still underexplored with only several multi-object tracking works [151], [152], [153], [154]. For example, Chu et al. [151] propose a spatial temporal attention mechanism (STAM) to capture the visible parts of targets and deal with the drift brought by occlusion. Zhu et al. [152] propose dual matching attention networks (DMAN) to deal with the noisy occlusions in multi-object tracking. Li et al. [155] propose to track every thing in the open world by performing class-agnostic association. In this work, we build a new complex video object segmentation dataset, MOSEv2, to facilitate future research on complex scene understanding in VOS and other related video understanding tasks.

#### 2.2 Related Video Segmentation and Tracking Tasks

There are other video segmentation and tracking tasks related to VOS, e.g., video instance segmentation, video semantic segmentation, video panoptic segmentation, and video object tracking.

∙ Video Instance Segmentation (VIS). Video instance segmentation is extended from image instance segmentation by Yang et al. [103], it simultaneously conducts detection, segmentation, and tracking of instances of predefined categories in videos. Thanks to the large-scale VIS dataset YouTube-VIS [103], a series of learning methods have been developed and greatly advanced the performance of VIS [104], [105], [106], [107], [108], [109], [110]. Then, occluded video instance segmentation is proposed by [111] to study the VIS under occluded scenes. Similar to [111], we study video segmentation under complex scenarios like occlusions, but different from [111], we focus on video object segmentation, and the proposed MOSEv2 dataset contains more videos and covers a broader range of real-world challenges beyond occlusion.

- MOSEv1 (36)

- MOSEv2 (200)

512

256

128

#Objects

64

32

16

8

4

- 1

- 2

Window

Giraffe

Dog

Egg

SUV

Duck

Donkey

Axe

Cow

Dish

Lion

Moon

Groundhog

Goose

Door

Longan

Dolphin

Apple

Glove

Hole

Book

Box

Cup

Hedgehog

Monkey

Drink

Lemon

Laptop

Bus

Swing

Fox

Spoon

Food

Grab

Oar

Shadow

Kite

Mongoose

Swan

Mouse

Coin

Horse

Truck

Koala

Dice

Cube

Can

Tissue

Cone

Bag

Mole

Candy

Jujube

Toy

Mahjong

Pig

Chips

Hula-hoop

Fan

Pen

Fish

Button

Keyboard

Grape

Orange

Kangaroo

Deer

Tiger

Others

Cat

Boat

Chess

Shoes

Cookie

Mushroom

Sloth

Zebra

Earphone

Elephant

Otter

Llama

Bean

Bee

Wolf

Snowboard

Blocks

Penguin

Puppet

Bird

Floor-tile

Stone

Flag

Vehicle-other

Air-conditioner

Sedan

Sheep

Turtle

Panda

Snake

Ostrich

Card

Ball

Belt

Rabbit

Floor

Balloon

Clam

Lemur-catta

Driver

Lizard

Chicken

Snacks

Roadblock

Flower

Taxi

Stick

Shell

Shield

Airplane

Handrail

Guinea-pig

Screw

Raccoon

Banana

Hard-disk

Peanut

Seal

Chair

Bottle

Chopsticks

Razor-clam

Hamster

Trash-bin

Insect

Bear

Camel

Tomato

Spider

Cyclist

Flamingo

Football

Umbrella

Cherry

Pear

Paper

Sausage

Tadpole

Backpack

Jellyfish

Scissor

Capybara

Peacock

RedPanda

Carrot

Bicycle

Juggling-ball

Bowling-ball

Drink-holder

Game-ball

Parrot

Frisbee

Pelican

Circle

Crocodile

Motorcycle

Rubik's-cube

Picture

Instrument

Sea-lion

Stapler

Skateboard

Passenger

Blueberry

Basketball

Squirrel

Billiards

Ice-hockey

Camouflage

Motorcyclist

Newtons-cradle

Test-tube

Strawberry

Watermelon

Pedestrian

Reflection

Horse-rider

Acrobatic-rod

Tennis-ball

Weight-plate

Screwdriver

Character

Person-other

Poultry-other

Sliding-puzzle

Table-tennis

Volleyball

Ferris-wheel

Remote-control

Billiards-ball

Table-tennis-bat

Fig. 2: Category distributions of MOSEv1 [1] and the proposed MOSEv2.

end of videos to reduce low-motion or simple frames. Next we annotate the first-frame masks of the selected targets. Then, the videos along with their first-frame masks are sent to the annotation team for annotation of the subsequent video frames.

### 3 MOSEV2 DATASET

In this section, we introduce the newly built MOSEv2 dataset. We first present the video collection and annotation process in Section 3.1, followed by dataset statistics and analysis in Section 3.2. Finally, we report the refined evaluation metrics in Section 3.3.

Using the given first-frame mask as a reference, annotators are required to identify the corresponding target and then track and annotate its segmentation masks across all subsequent frames. To facilitate this process, an interactive annotation tool is developed to automatically load videos and target objects. Annotators can preview the video and first-frame mask, annotate and visualize masks in later frames, and save the results. The annotation tool also has a built-in interactive video object segmentation model SAM2 [15] to assist annotations in producing high-quality masks. To ensure annotation quality in complex scenes, annotators are required to consistently track the target and provide precise segmentation. For frames where the target disappears or is fully occluded, the masks must remain empty. All videos are annotated at a minimum of 5 FPS, while a subset is annotated at full FPS to evaluate the framerate robustness of VOS models.

#### 3.1 Video Collection and Annotation

Video Collection. The videos in MOSEv2 are obtained from two sources. The first source is inherited from MOSEv1 [1] with 2,149 videos. The second source consists of 2,875 newly self-captured videos from real-world scenarios and copyright-free videos from the internet that have not appeared in any existing dataset. MOSE is specifically designed for video object segmentation in complex scenes. To ensure the complexity and diversity of the collected videos, we follow a set of strict selection rules:

- R1. Each video should contain multiple objects, except for challenging cases (e.g., camouflage). Specifically, videos with crowded objects of similar appearance are highly valued.
- R2. Occlusions are encouraged. Videos with occlusions, particularly those caused by other moving objects, are preferred.
- R3. Great emphasis should be placed on scenarios where objects disappear and then reappear due to occlusions or out-of-view.
- R4. The target objects should encompass a diverse range of scales (e.g., small-scale, large-scale) and visibility conditions (e.g., conspicuous, partially visible).
- R5. The video must exhibit clear motion, either from object movement or camera motion. Videos with static objects and a stationary camera should be discarded.

Besides the points mentioned above, we further emphasize the following rules in the design of MOSEv2:

- R6. Target object categories should be diversified, including novel classes not present in MOSEv1, such as camouflaged objects, shadows, and reflections.
- R7. Longer videos are preferred for containing more challenging patterns, such as long-term occlusions, complex motion dynamics, and repeated object disappearance-reappearance, rather than merely for their duration.
- R8. A wide range of challenging environments is prioritized during collection, such as low-light scenes, cluttered scenes, and varying weather conditions (e.g., rain, fog, snow).
- R9. Multi-shot videos are encouraged, where objects undergo significant spatial or appearance changes across shots.
- R10. Videos requiring specific knowledge are deliberately included, such as optical character recognition, spatial reasoning, physical principles, and multi-view understanding.

After annotation, all videos are carefully reviewed by our verification team to ensure high-quality masks.

#### 3.2 Dataset Statistics

In TABLE 1, we analyze the data statistics of MOSEv2 in comparison with existing VOS datasets, such as DAVIS [2], [3], YouTube-VOS [4], LVOS [61], [62], SA-V [15], MOSEv1 [1], as well as VOT datasets, including GOT-10k [29], LaSOT [28], VOT [137], and DiDi [26]. MOSEv2 expands MOSEv1 by adding 2,875 new videos, reaching a total of 5,024 videos and 701,976 mask annotations for 10,074 objects.

Categories. MOSEv2 contains 200 object categories, the largest among existing VOS datasets. Fig. 2 presents the detailed category distribution of MOSE. Building on the 36 categories of MOSEv1, MOSEv2 significantly expands the scope to 200, covering not only common categories such as squirrels, footballs, and otters, but also rare ones like Newton’s cradle and camouflaged objects, as well as non-physical targets like shadows. This extensive coverage enables more comprehensive and robust evaluation of VOS methods.

Disappearance-Reappearance. MOSEv2 significantly surpasses its predecessor MOSEv1 in terms of object disappearance and reappearance. The “Disapp. Rate” increases from 41.5% to 61.8%, while the “Reapp. Rate” more than doubles from 23.9% to 50.3%. MOSEv2 also exceeds SA-V (58.7%) among VOS datasets and DiDi (40.0%) among VOT datasets in “Disapp. Rate”, while its 50.3% “Reapp. Rate” outperforms LVOSv1 (46.7%) and DiDi (40.0%). These characteristics make MOSEv2 the most challenging dataset for studying disappearance-reappearance scenarios.

Video Annotation. After collecting videos for MOSEv2, our research team manually reviews them to select suitable targetsof-interest for each video. We slightly trim the beginning and

Crowding. To assess crowding complexity, we compute the “Distractors” metric, which quantifies the average number of

- TABLE 1: Statistical comparison between MOSEv2 and existing video object segmentation and tracking datasets. “Annotations” denotes the number of annotated masks or boxes. “Duration” denotes the total duration of annotated videos, in minutes by default unless noted. “Disapp. Rate” measures the frequency of objects disappearing in at least one frame, while “Reapp. Rate” measures the frequency of objects that previously disappeared and later reappear. “Distractors” quantifies scene crowding as the average number of visually similar objects per target in the first frame. * Unless otherwise specified, SA-V uses the combination of manual and auto annotations.

Dataset Year Videos Categories Objects Annotations Duration Frames Disapp. Rate Reapp. Rate Distractors

Video Object Tracking (VOT) Dataset

GOT-10k [29] 2019 9,695 563 10,200 1.5M 40.0 hr 1.5M 2.1% 2.1% 3.1 LaSOT [28] 2019 1,500 85 2,148 3.9M 35.8 hr 3.9M 17.1% 16.9% 3.4 VOT [137] 2022 62 - 62 19,826 11.10 19,903 19.4% 17.7% 5.2 DiDi [26] 2025 180 - 180 268,084 152.71 274,882 40.0% 40.0% 10.6

Video Object Segmentation (VOS) Dataset

|SegTrack-v2 [156] YouTube-Objects [157]<br><br>FBMS [158] JumpCut [13]<br><br>DAVIS16 [2]<br><br>DAVIS17 [3]<br><br><br>YouTube-VOS [4] VOTS [159] VOST [160]<br><br>LVOSv1 [61] LVOSv2 [62]<br><br>SA-V* [15]|2013<br><br>2014<br><br><br>2014<br><br>2015<br><br>2016<br><br>2017<br><br>2018 2023<br><br><br>2023<br><br>2023<br><br>2024<br><br><br>2024<br>|14<br><br>126 59 22 50 90<br><br>4,453 144 713 220 720<br><br>50,900|11 10 16 14 94 -<br><br>154 27 44 -<br><br>|24 124 139<br><br>22 50<br><br>205 7,755<br><br>341 1,726<br><br>282 1,132<br><br>642,600|1,475<br><br>2,092<br><br>1,465 6,331<br><br>3,455<br><br><br>13,543<br><br>197,272 173,758 156,432 407,945<br><br>35.5M|0.59 9.01 7.70 3.52 2.88 5.17<br><br>334.81 166.00 251.92 351.00 823.00<br><br>196.0 hr|947<br><br>2,127<br><br>13,860<br><br>5,315<br><br>3,440<br><br>6,208<br><br><br><br><br>120,532 298,640<br><br>75,547 126,280 296,401<br><br>4.2M|8.3% 6.5%<br><br>11.2%<br><br>0.0% 11.1% 16.1% 13.0% 46.5% 50.0% 36.1% 58.7%<br><br>|0.0%<br><br>1.6%<br><br><br>-<br><br>0.0% 4.9%<br><br>10.7% 8.0%<br><br>44.4% 46.7% 32.5% 27.7%<br><br>|5.4<br><br>2.6 3.7 3.0 5.3 3.7 4.6 6.2<br><br>|
|---|---|---|---|---|---|---|---|---|---|---|
|MOSEv1 [1] MOSEv2<br><br>|2023 2025<br><br>|2,149 5,024<br><br>|36 200<br><br>|5,200 10,074<br><br>|431,725 701,976<br><br>|443.62 1,570.63<br><br>|130,149 468,251<br><br>|41.5% 61.8%<br><br>|23.9% 50.3%<br><br>|6.5 13.6<br><br>|

- TABLE 2: Occlusion rate comparison among different datasets. Dataset Mean mBOR mAOR mMLLMOR

and amodal mask areas generated by the amodal segmentation model DiffVAS [162]. MLLMOR leverages a multimodal large language model (we use Qwen2.5-VL-32B [163]) to assess occlusion severity. We compute the final occlusion estimate as the average of all three metrics. As shown in TABLE 2, MOSEv2 achieves a mean occlusion rate of 47.0, substantially exceeding MOSEv1 (36.4) and SA-V (36.1), establishing MOSEv2 as the most challenging dataset for studying occlusion in videos.

|DAVIS17 [3] YouTube-VOS [4]<br><br>LVOSv2 [62] SA-V [15]<br><br>|20.6 23.2 25.4 36.1<br><br>|3.4 5.7 8.4<br><br>27.4|23.7 26.0 30.6 37.2<br><br>|34.6 38.0 37.2 43.6<br><br>|
|---|---|---|---|---|
|MOSEv1 [1] MOSEv2<br><br>|36.4 47.0<br><br>|23.7 28.3<br><br>|41.2 54.8<br><br>|44.2 57.8<br><br>|

Mask Size. Fig. 4 compares the distribution of mask sizes (normalized by video resolution) across datasets. MOSEv2 contains a substantially higher proportion of small masks (size < 0.01), reaching 50.2%, significantly above DAVIS (25.3%), YouTubeVOS (18.4%), LVOSv2 (34.8%), SA-V (40.7%), and MOSEv1 (39.5%). This high prevalence of small objects poses greater challenges for fine-grained perception and accurate segmentation. Video Length. Fig. 5 presents the video length distribution in MOSEv2. Compared to only 11 videos exceeding 300 frames (around 1 minute) in MOSEv1, MOSEv2 provides 183 such long videos, with the longest reaching 7,825 frames, around 26 minutes. The average video length increases from 60.6 to 93.2 frames, enabling more comprehensive evaluations of long-term temporal consistency and tracking robustness. While LVOSv2 [62] includes 362 videos over 300 frames with an average length of 590.9 frames, our 183 long videos average 598.4 frames and extend up to 7,825 frames, far beyond LVOSv2’s maximum of 2,280. Importantly, video length alone does not imply difficulty. In MOSEv2, long videos are not included merely for their duration, but intentionally designed to include richer dynamics and more complex scenarios, such as object disappearance, occlusion, scene transitions, and multi-shot clips. For example, LVOSv2’s reappearance rate is only 32.5%, substantially lower than our 50.3%, highlighting the increased complexity in MOSEv2.

[Figure 61]

[Figure 62]

[Figure 63]

MLLM (Qwen-VL)

| |
|---|

[Figure 64]

[Figure 65]

[Figure 66]

|Prompt: Rate the occlusion level of the masked target on a scale of 0 to 1.|
|---|

|Response: 0.8|
|---|

= 0.006

= 0.930

| |
|---|

+

(a) BOR (b) AOR (c) MLLMOR

Fig. 3: Occlusion evaluation protocol. (a) BOR: Bounding-box Occlusion Rate [111], (b) AOR: Amodal-mask Occlusion Rate, (c) MLLMOR: MLLM-assisted Occlusion Rate.

visually similar objects per target in the first frame, using TRex2 [161]. MOSEv2 reaches 13.6 distractors per object, more than twice that of MOSEv1 (6.5), and also higher than SA-V (6.2) and LVOSv2 (4.6). Remarkably, it even surpasses DiDi [26] (10.6), a dataset specifically designed to emphasize distractors in VOT, underscoring MOSEv2’s complexity and its significance for advancing robust perception in densely crowded scenes.

Occlusion. We compare occlusion levels of MOSEv2 with other datasets in TABLE 2. While MOSEv2 achieves the highest mBOR score [111] of 28.3, this metric offers only a coarse estimation [1]. As shown in Fig. 3 (a), an object may be heavily occluded yet still yield a BOR near zero. To address this limitation, we introduce two complementary metrics: Amodal-mask Occlusion Rate (AOR) and MLLM-assisted Occlusion Rate (MLLMOR), shown in Fig. 3 (b) and (c), respectively. AOR measures the ratio between visible

Complex Environments. Fig. 6 shows the distribution of challenging environmental conditions. Compared to MOSEv1, MO-

443

0.4

DAVIS YouTube-VOS LVOSv2 SA-V MOSEv1 MOSEv2 (ours)

|0 10 1 10 2<br><br>Mask Ratio (log scale)<br><br>4: Mask size distribution, by video resolution.<br><br>Fig. 5 includ<br><br>3212<br><br>4931<br><br>6226<br><br>4028<br><br>5067<br><br>3719<br><br>|0-1920-3940-5960-7980-99100-119120-139140-159160-179180-199200-219220-239240-226<br><br>: Video length es more long v<br><br>MOSEv1 MOSEv2<br><br>SV<br><br>SC<br><br>Attribute Rela|590-279280-299300-319320-339340-359360-379380-399400-419420-439440-459460-479480-499500-519520-539540-559560-579580-599600-619620-639640-659660-679680-699720-739740-759760-779780-8<br><br>#Frames<br><br>distributions. Compared to ideos, with the longest reac<br><br>FM<br><br>OV<br><br>DR<br><br>LD<br><br>tionship Chord Diagram<br><br>TABLE 3: Def part of the attr additional com|79900-819820-839860-879880-899900-919940-959960-9791000-10191020-10391040-10591080-10991180-11991500-15191560-15792080-20992400-24192440-24597820-7839<br><br>MOSEv1, MOSEv2 hing 7,825 frames.<br><br>CloudyRainyHeavyRainSnowyFoggyUnderwaterNighttimeDisaster<br><br>0<br><br>Fig. 6: Challenging environment distribution.<br><br>initions of object attributes in MOSEv2. We adopt ibutes from DAVIS [3] (top) and extend them with plex video attributes (bottom).|
|---|---|---|---|
|BC FM OCC OV SV SC AC DR DV CRO CE NC M<br><br>2574 2609<br><br>1828<br><br>896<br><br>1462<br><br>609<br><br>27<br><br>7: (Left) Distribution of objects attr correlations in MOSEv2.<br><br>significantly expands the coverage o<br><br>rainy videos increased from 20 from 29 to 280. MOSEv2 also present in MOSEv1, including 142 fog, and 50 disaster scenarios (e.g.,<br><br>MOSEv2 provides 443 cloudy, 159 snowy, 60 foggy, 280 underwater, 2<br><br>and 50 disaster videos. This<br><br>MOSEv2 as a more comprehensive robustness under diverse complex<br><br>Attribute Analysis. Following D<br><br>attributes for MOSEv2 in TABLE MOSEv2 greatly expands the cove compared to MOSEv1. For exampl<br><br>increase from 2,100 to 4,931, from 1,243 to 5,067, complex<br><br>1,462, and long duration (LD) from introduces new attributes such a<br><br>objects), multi-shot sequences (MS, 2 (KD, 256). As shown in Fig. 7 the co-occurrence patterns betw<br><br>into the interplay of real-world attribute set makes MOSEv2 a m<br><br>evaluating model under diverse and co<br><br>Evaluation Metrics<br><br>previous VOS works [1], [2 similarity  and the contour<br><br>metrics. Given a predicted mask ground-truth mask 𝑀 ∈ {0,1}𝐻×𝑊 ,<br><br>as the Intersection-over-Union (𝑀̂ ∩ 𝑀)∕(𝑀̂ ∪ 𝑀). To measure th<br><br>recall R𝑐 and precision P𝑐 are matching [164]. Then, the conto<br><br>mean of the contour recall R?<br><br>2P𝑐R𝑐∕(P𝑐 + R𝑐), which represents h predicted masks resemble the contours<br><br>the average region similarity 𝑚𝑒𝑎 over all objects are calculated as the<br><br>use  and  to represent 𝑚𝑒𝑎𝑛 and performance is measured by  &|S LD KD<br><br>7 224 256<br><br>DV<br><br>BC<br><br>OCC<br><br>ibutes. (Right)<br><br>f adverse scena to 159, and<br><br>introduces new heavy rain, earthquake, f rainy, 142 he<br><br>55 nighttime ( substantial expa<br><br>dataset for environments.<br><br>AVIS [3], we<br><br>3. As shown rage of challen e, objects with ppearance-reap<br><br>environments (CE) 33 to 224. In s novel catego 77), and know<br><br>(right), a chord een attributes,<br><br>challenges. This ore rigorous mplex conditio<br><br>], [3], we com accuracy  as 𝑀̂ ∈ {0,1}? region similar<br><br>of 𝑀̂ and e contour qual calculated via ur accuracy<br><br>? and precision ow closely the<br><br>of ground-truth<br><br>𝑛 and contour final results. Fo<br><br>𝑚𝑒𝑎𝑛, respecti = ( + )∕2|CE<br><br>AC<br><br>CRO<br><br>NC<br><br>MS<br><br>KD<br><br>Attribute<br><br>rios. For underwater<br><br>conditions 73 snow, lood). In avy rain, vs. 75 in nsion esexploring<br><br>define 15 in Fig. 7 ging sceocclusion<br><br>pearance from 330<br><br>addition, ries (NC, ledge de-<br><br>diagram offering compre-<br><br>benchmark ns.<br><br>pute the<br><br>evalua?×𝑊 and ity  is<br><br>𝑀, i.e., ity of 𝑀̂ ,<br><br>bipartite  is the<br><br>P𝑐, i.e., contours<br><br>masks. accuracy r brevity,<br><br>vely. The<br><br>.<br><br>Attr. Definition BC Backgrou<br><br>similar vi FM Fast Mot<br><br>centroids OCC Occlusion OV Out-of-vi SV Scale Var<br><br>of range<br><br>SC Shape Co AC Appearan<br><br>and illum DR Disappea appearing DV Diverse V<br><br>camoufla CRO Crowding CE Complex<br><br>as underw NC Novel Ca<br><br>objects an MS Multi-Sho LD Long Dur KD Knowledg<br><br>(e.g., OC<br><br>Revisiting the limitations for MOSEv2 that works [1], [2], for images of t While effective ignores object As shown in F 0.039% of the i not overlap, ye and yields an i propose an ada<br><br>where 𝐴 is the Based on bou 𝛼 = 0.1 to mai objects while b threshold-base evaluation acro in Fig. 8, the im|nd Clutter. The background and the target object exhibit<br><br>sual appearances. ion. The average, per-frame object motion, computed as Euclidean distance, is larger than 𝜏𝑓𝑚 = 20 pixels.<br><br>. The target object is partially or fully occluded in video. ew. The target object leaves the video frame completely.<br><br>iation. The ratio of any pair of bounding boxes is outside<br><br>[0.5,2.0].<br><br>mplexity. The target exhibits complex boundary structures. ce Change. Significant appearance change, due to rotations ination changes.<br><br>The target object reappears after dis-|
| | | |rance-Reappearance. in the video.<br><br>isibility. The target object is small, inconspicuous, or<br><br>ged in the scene.<br><br>. Multiple similar objects appear in close proximity.<br><br>Environment. Object under challenging conditions such<br><br>ater, nighttime, and adverse weather (rain, snow). tegories. Novel object categories, especially camouflaged<br><br>d non-physical objects.<br><br>ts. The object sequence contains multiple camera shots. ation. Object duration exceeds 1 minute (300 frames).<br><br>e Dependency. Objects requiring specific knowledge<br><br><br>R, spatial reasoning) for precise segmentation in videos.<br><br>|
| | | | Score. The widely used  score has clear small objects, which is particularly problematic for contains a large number of small targets. Previous [3] adopt a fixed boundary threshold 𝑤 = 0.008×𝐷 he same resolution, where 𝐷 is the image diagonal.<br><br>for images with uniform object sizes, this threshold scale and biases the evaluation of small objects.<br><br>ig. 8, for a chopstick with only 955 pixels, merely mage area, the predicted and ground-truth masks do t the fixed threshold excessively dilates boundaries nflated  score of 0.91. To address this issue, we ptive boundary threshold:<br><br>𝑤̇ = min(0.008 × 𝐷,𝛼 × √𝐴), (1)<br><br>object’s area in pixels and 𝛼 is a scaling factor. ndary statistics from DAVIS and MOSE, we set ntain reasonable boundary widths for average-sized<br><br>etter handling small ones. We denote this adaptived metric as ̇ , which provides a fairer boundary<br><br>ss different object scales. For the small chopstick proved ̇ correctly assigns a score of 0, while for|

MOSEv1 MOSEv2

512

MOSEv1 MOSEv2

400

256

128

0.3

Percentage

280

#Videos

300

#Videos

64

255

32

0.2

16

200

159 142

8

4

100

73 60

0.1

50

- 1
- 2

AUC: 50.2%

0.0

10

Fig. normalized

6000

5000

#Obejcts

4000

3000

2000

1000

0

Fig. corre

SEv2 example, scenes not p 60 fo total, 73 sn MOSEv1), tablishes model Object object (left), narios (OCC) (DR) to 1, MOSEv2 609 o pendency illustrates insights hensive for ev

#### 3.3

Following region tion a gro computed  = contour graph harmonic  =

of pr Next, 𝑚𝑒𝑎𝑛 we u overall

point initialization, as well as unsupervised and interactive VOS. We further evaluate video object tracking methods on MOSEv2, demonstrating its broad applicability beyond segmentation.

Image (Diagonal=2,308) S (A=955) L (A=522,976)

|ℱ = 0.91 ℱ = 0.74<br><br>[Figure 67]<br><br>[Figure 68]<br><br>Previous|
|---|

[Figure 69]

Ground Truth Prediction

| |
|---|

Implementation Details. The proposed MOSEv2 follows the same data format as MOSEv1 [1] and YouTube-VOS [4]. For methods developed before SAM2, we ensure fair comparisons by replacing the YouTube-VOS [4] training set with MOSEv2 while strictly following the original YouTube-VOS training configurations. These methods are trained with image-pretrained backbones without using any additional video datasets. For SAM2-based models, we adopt SAM2.1 as the default initialization and finetune exclusively on MOSEv2. We evaluate model performance using standard metrics ( , , and  &) on MOSEv2 validation set, following the DAVIS protocol [2], [3]. To better capture the complex challenges in MOSEv2, we additionally report ̇ ,  &̇ ,  &̇ 𝑑, and  &̇ 𝑟 as described in Section 3.3. Among them,  &̇ is selected as the primary evaluation metric.

|ℱ = 0 ℱ = 0.74<br><br>[Figure 70]<br><br>Ours<br><br>[Figure 71]|
|---|

̇ ̇

- Fig. 8: Comparison of the commonly used  and our adaptive ̇ on Small and Large objects. For small objects, such as a chopstick with 955 pixels (only 0.039% of the image area),  yields exaggerated scores due to fixed resolution-based thresholds, while ̇ offers a more reliable measure by accounting for object scale. For large objects, such as a person (522k pixels, 21% of the image area), ̇ remains consistent with , both yielding 0.74.

Groundturth 𝑀 Predicted 𝑀

𝒥&ℱ̇ per frame 1 1 1 1 0 1 1 0 1 1

Frame ID 0 1 2 3 4 5 6 7 8 9

𝒥&ℱ̇ = = 0.8 𝒥&ℱ̇ = = 1. 𝒥&ℱ̇ = = 0.

- Fig. 9: Illustration of  &̇ ,  &̇ 𝑑, and  &̇ 𝑟. The slashed boxes denote frames with empty masks. For simplicity, all predicted masks are assumed to match the ground truth perfectly.

Dataset Splits. These videos are split into 3,666 training, 433 validation, and 614 testing videos, for model training, daily evaluation, and competition period evaluation1, respectively. An additional 311 videos, originally used as the validation set in MOSEv1, are temporarily retained for compatibility and may later serve as a local validation set when MOSEv2 becomes the standard.

#### 4.1 Semi-supervised Video Object Segmentation

Semi-supervised (or semi-automatic, one-shot) VOS offers the target’s mask, bounding box, or points on the first frame as reference for segmenting the entire video.

∙ Mask-initialization. This is the most common and actively studied setting in VOS. In TABLE 4, we benchmark two groups of mask-initialization semi-supervised VOS methods on MOSEv2. The first group includes traditional VOS methods, typically built on ResNet-50. The second group comprises SAM2-based variants, covering both SAM2-B+ and SAM2-L scales. Existing methods perform substantially worse on MOSEv2 compared to previous benchmarks such as DAVIS17 [3] and YouTube-VOS19 [4]. For example, SAM2-B+ [15] achieves only 47.1%  & on MOSEv2, far below its 74.7%  & on MOSEv1, 83.1%  & on LVOSv2, and 90.2%  & on DAVIS17. Among traditional methods, CutieB [14] performs best with 42.8%  &̇ , just 3.2% behind SAM2B+. However, its strength mainly comes from high  &̇ 𝑑 scores (64.5%) in disappearance handling, while struggling with reappearance, reaching just 18.3%  &̇ 𝑟, 4.9% lower than SAM2-B+.

large object, such as a person with 522k pixels (21% of the image area), ̇ maintains consistency with the original metric .

Disappearance and Reappearance Metrics. Given the frequent object disappearance-reappearance in MOSEv2, we compute dedicated scores:  &̇ 𝑑 for disappearance clips where the target is absent, and  &̇ 𝑟 for reappearance clips where the target returns. As shown in Fig. 9, we first compute these metrics per disappearance or reappearance clip, average them to obtain sequence-level scores. These metrics address a key limitation of the widely used  &, which are averaged over all frames in a video and thus biased by the proportion of empty-target frames. For example, in videos with many disappearance frames, models that tend to predict empty masks may appear strong, while in videos with few disappearance frames, models that blindly predict masks may benefit since errors on empty frames have little impact.

Taking a close look at the detailed metrics, we observe that all methods face significant challenges in reappearance scenarios, with  &̇ 𝑟 scores ranging only from 7.8% to 34.9%. This underscores the difficulty of re-identifying objects after disappearance in complex scenes. The proposed adaptive boundary metric ̇ consistently yields lower values than  across all methods, showing that the adaptive threshold provides a stricter and more reliable assessment of boundary quality for objects of different sizes.

By isolating evaluation to disappearance and reappearance

clips,  &̇ 𝑑 and  &̇ 𝑟 provide clearer insights: models that fail to suppress masks during disappearance perform poorly on

 &̇ 𝑑, while those unable to recover the target after its return are penalized on  &̇ 𝑟. Only models handling both cases effectively achieve high scores on both metrics. For  &̇ 𝑟, we deliberately exclude the initial continuous presence of the target, where reference is strongest, focusing instead on true reappearance after disappearance, which better reflects recovery under ambiguity.

SAM2-based methods [15] achieve superior performance, with even zero-shot models outperforming most finetuned traditional methods, demonstrating the effectiveness of foundation models on challenging video segmentation. Beyond SAM2, several SAM2based variants [25], [26], [27] demonstrate enhanced performance on MOSEv2. These methods are specifically designed to address complex scenarios: SAMURAI [25] incorporates Kalman filtering

### 4 EXPERIMENTS

We conduct comprehensive experiments on the newly built MOSEv2 dataset, benchmarking multiple video object segmentation settings, including semi-supervised VOS with mask, box, and

1. The testing set is used for evaluation during the competition periods, such as https://pvuw.github.io/ and https://lsvos.github.io/.

- TABLE 4: Benchmark results of mask-initialization semi-supervised VOS methods on MOSEv2 validation set. “ZS” indicates that the model uses zero-shot evaluation. Inference speed (FPS) and GPU memory usage (GiB) are measured on a single A6000 GPU. For SAM2 and its variants, video frames are offloaded to CPU memory to balance inference speed and memory usage.

MOSEv2 MOSEv1 SA-V𝑡𝑒𝑠𝑡 LVOSv2 DAVIS17 YT-VOS19 Method Pub. FPS Mem.  &̇  ̇  &̇ 𝑑  &̇ 𝑟   &  &    &  &  &  &

|AOT-L [165] STCN [166] RDE [167] XMem [16] DeAOT-L [168] DEVA [101] XMem++ [55] Cutie-B [14] JointFormer [169]|[NeurIPS’21]<br><br>[NeurIPS’21]<br><br>[CVPR’24] [ECCV’22]<br><br>[NeurIPS’22]<br><br><br>[ICCV’23] [ICCV’23] [CVPR’24] [PAMI’25]<br><br>|19.7 3.8 45.1 6.2 32.7 1.4 49.8 1.6 21.2 3.7<br><br>43.0 1.0 30.1 1.4<br><br>44.1 0.9 7.2 3.6<br><br><br>|30.2 29.0 31.4 67.8 7.8 32.9 31.0 29.7 28.9 30.5 79.4 8.1 31.4 30.2 32.0 30.7 33.3 62.7 12.6 35.0 32.8<br><br>36.3 34.7 37.9 56.6 14.8 40.0 37.4 32.6 30.7 34.5 33.5 18.3 37.2 33.9 38.3 36.6 40.0 55.1 18.5 42.2 39.4 34.2 32.5 35.9 51.6 15.5 37.9 35.2 42.8 41.1 44.4 64.5 18.3 46.8 43.9<br>37.7 36.0 39.4 57.3 18.3 41.1 38.6<br>|57.2 53.1 61.3 50.8 46.6 55.0 48.8 44.6 52.9 57.6 53.3 62.0<br><br>59.4 55.1 63.8<br><br>60.0 55.8 64.3<br><br><br>56.0 51.5 60.6<br><br>69.9 65.9 74.1<br><br>70.2 66.3 74.0<br>|50.3 62.5 53.9 62.3 61.8 56.2 60.7 -<br><br>|63.9 60.6 62.2 64.5 63.9 -|84.9 85.4 84.2 86.2 85.2 87.0 87.9 90.6<br><br>|84.1 82.7 81.9 85.6 86.0 85.4 87.0 87.5|
|---|---|---|---|---|---|---|---|---|
|SAM2-B+ (ZS) [15] SAM2-B+ [15] SAMURAI-B+ [25] DAM4SAM-B+ [26] SAM2Long-B+ [27]<br><br>|[ICLR’25] [ICLR’25] [Preprint’24] [CVPR’25] [ICCV’25]|23.4 2.7 23.4 2.7 17.7 2.7 17.3 2.7<br><br>9.4 6.0|43.1 41.4 44.8 60.6 20.6 47.0 44.2<br><br>46.0 44.2 47.8 61.6 23.2 50.0 47.1<br><br>47.4 45.3 49.5 45.9 33.6 52.2 48.8<br><br><br>47.9 45.8 50.0 51.3 32.0 52.6 49.2<br><br>48.6 46.7 50.5 58.4 29.2 52.8 49.7<br><br><br>|73.6 69.5 77.6<br><br>74.7 70.6 78.8 73.3 69.0 77.5<br><br><br>73.8 69.5 78.0<br><br>74.7 70.6 78.8<br><br><br>|77.0 80.8|83.1 85.2<br><br>|90.2 -<br><br>|88.6 -|
|SAM2-L (ZS) [15] SAM2-L [15] SAMURAI-L [25] DAM4SAM-L [26] SAM2Long-L [27]<br><br>|[ICLR’25] [ICLR’25] [Preprint’24] [CVPR’25] [ICCV’25]<br><br>|14.4 3.6 14.4 3.6 12.1 3.5 12.3 3.5<br><br>7.1 6.8|49.4 47.6 51.3 63.4 27.8 53.8 50.7 49.7 47.9 51.5 64.5 27.1 53.8 50.9<br><br>51.1 49.0 53.2 52.4 34.9 55.8 52.4<br><br>51.2 49.2 53.2 57.2 34.2 55.6 52.4<br><br><br>51.5 49.6 53.4 62.5 30.6 55.8 52.7|74.5 70.5 78.4<br><br>76.4 72.3 80.5<br><br>75.6 71.4 79.8 75.6 71.5 79.8<br><br>77.1 73.0 81.2<br><br><br><br><br>|78.4 81.2|84.0 85.3<br><br>|90.7 88.8<br><br>|89.3 90.2|

- TABLE 5: Benchmarking box-initialization semi-supervised VOS methods on MOSEv2 validation set.

TABLE 6: Benchmarking point-initialization semi-supervised VOS methods on MOSEv2 validation set. We use  &̇ as the evaluation metric. “n-clk”: using n positive clicks for initialization.

MOSEv2 MOSEv1 DAVIS17 Method Pub.  &̇  &̇ 𝑑  &̇ 𝑟  &  &  &

MOSEv2 MOSEv1 DAVIS17

Method Pub.

1-clk 3-clk 5-clk 1-clk 3-clk 5-clk 5-clk

|UniVS [170] Cutie+SAM<br><br>|[CVPR’24] [CVPR’24]|16.4 22.3 8.6 17.3 42.3 64.4 18.0 43.5<br><br>|38.0 63.0|61.8 82.3<br><br>|
|---|---|---|---|---|
|SAM2-B+ SAMURAI-B+ DAM4SAM-B+ SAM2Long-B+<br><br>|[ICLR’25] [Preprint’24] [CVPR’25] [ICCV’25]|46.0 61.9 22.1 47.2 46.5 45.7 32.7 48.0<br><br>46.2 49.9 31.3 47.6<br><br>47.7 57.4 28.3 49.0<br>|73.7 71.8 70.1 72.9<br><br>|85.3 86.1 86.6 85.5|
|SAM2-L SAMURAI-L DAM4SAM-L SAM2Long-L|[ICLR’25] [Preprint’24] [CVPR’25] [ICCV’25]<br><br>|49.0 61.9 26.2 50.3<br><br>49.2 49.9 33.8 50.7 47.5 51.5 32.2 48.8<br><br>50.2 60.6 29.8 51.5<br><br><br>|75.4 74.9 73.3 75.9<br><br>|89.0 88.9 86.6 88.3|

|Cutie+SAM<br><br>|[CVPR’24]<br><br>|35.2 38.2 36.7|54.2 58.5 58.3<br><br>|62.7|
|---|---|---|---|---|
|SAM2-B+ SAMURAI-B+ DAM4SAM-B+ SAM2Long-B+<br><br>|[ICLR’25] [Preprint’24] [CVPR’25] [ICCV’25]<br><br>|43.6 44.1 44.4<br><br>44.7 45.8 45.9 43.8 45.6 45.8<br><br>45.3 45.3 45.1<br><br><br>|66.8 66.8 70.6<br><br>65.7 65.7 68.6<br><br>66.3 66.3 69.3 66.4 66.4 70.3<br><br><br>|80.4 78.9 80.3 80.5|
|SAM2-L SAMURAI-L DAM4SAM-L SAM2Long-L<br><br>|[ICLR’25] [Preprint’24] [CVPR’25] [ICCV’25]|47.6 48.0 47.2<br><br>47.9 48.2 48.6<br><br>47.7 47.7 48.2<br><br>48.5 48.3 48.7<br><br><br><br><br>|69.6 69.6 74.8<br><br>69.3 69.3 74.1<br><br>69.4 69.4 74.4<br><br><br>69.7 69.7 75.2<br><br><br>|86.0 84.8 85.5 86.1|

for motion modeling to handle occlusions, DAM4SAM [26] introduces robust memory mechanisms to reduce distractor effects in crowded scenes, and SAM2Long [27] employs a memory tree to mitigate error accumulation in long videos with object disappearance and reappearance. While these designs target specific challenges, they fall short on MOSEv2, where occlusions and long-term tracking are more severe. In addition, new challenges such as adverse environments, multi-shot transitions, and knowledge-dependent scenarios remain unaddressed. For example, SAM2Long-L [27] achieves the best overall performance at only

XMem [16] and STCN [166] run faster at 49.8 and 45.1 FPS, respectively, but with lower performance. In contrast, SAM2based methods achieve better results but at the cost of slower inference, with SAM2Long-L running at 7.1 FPS, and greater memory usage of 6.8 GiB compared to 0.9 GiB for Cutie-B [14].

∙ Box-initialization. We benchmark box-initialization semisupervised VOS methods on MOSEv2 in TABLE 5. Similar to the mask-initialization setting, we evaluate both traditional methods (UniVS [170] and Cutie+SAM) and SAM2-based variants. The results show that SAM2-based methods clearly outperform traditional ones, with SAM2Long-L [27] achieving the best performance of 50.2%  &̇ . However, all methods struggle with reappearance scenarios evaluated by  &̇ 𝑟, while performing relatively better on disappearance cases evaluated by  &̇ 𝑑. The performance gap between MOSEv2 and other benchmarks such as DAVIS17 [3] highlights the increased difficulty posed by the diverse and complex scenarios in MOSEv2.

- 51.5%  &̇ , highlighting substantial room for improvement in addressing complex real-world scenarios. Most SAM2 variants

also show improved  &̇ 𝑟 but decreased  &̇ 𝑑, reflecting a tendency toward aggressive re-identification. SAMURAI [25]

achieves the highest  &̇ 𝑟 but sacrifices the most in  &̇ 𝑑, while SAM2Long provides a better balance and thus the best overall performance. The frequent disappearance–reappearance patterns and diverse complex scenarios in MOSEv2 impose dual demands on both recall and precision. Future models must suppress false predictions when targets are absent yet reliably re-identify them upon reappearance. Effectively balancing these competing objectives remains a key challenge for future research.

∙ Point-initialization. We benchmark point-initialization semisupervised VOS methods on MOSEv2 in TABLE 6, including Cutie+SAM and SAM2-based variants. The results show that SAM2-based methods significantly outperform Cutie+SAM, with SAM2Long-L achieving the best performance of 48.5%  &̇ using only a single click. However, increasing the number of clicks

In terms of computational efficiency, there is a clear tradeoff between accuracy and speed. Traditional methods such as

- TABLE 7: Benchmark results of unsupervised VOS methods on MOSEv2 validation set. We limit the number of proposals to 20.

MOSEv2 MOSEv1 DAVIS17 Method Pub.  &̇  &̇ 𝑑  &̇ 𝑟  &  &  &

|DEVA [101] EntitySAM [102]<br><br>|[ICCV’23] [CVPR’25]|34.9 80.4 7.5 36.0 28.2 96.7 4.1 28.4<br><br>|57.0 42.2|73.4 72.6<br><br>|
|---|---|---|---|---|
|SAM2-B+ SAMURAI-B+ DAM4SAM-B+ SAM2Long-B+<br><br>|[ICLR’25] [Preprint’24] [CVPR’25] [ICCV’25]<br><br>|28.3 77.3 6.3 28.8<br><br>27.5 52.5 12.0 28.4 25.9 52.5 6.9 26.7<br>28.9 61.0 7.5 29.7<br>|47.2 46.9 46.4 47.6<br><br>|57.3 57.4 57.7 57.4|
|SAM2-L SAMURAI-L DAM4SAM-L SAM2Long-L<br><br>|[ICLR’25] [Preprint’24] [CVPR’25] [ICCV’25]<br><br>|28.2 73.5 6.3 28.6<br>29.2 53.9 12.6 30.1<br><br><br>28.7 52.6 10.6 29.6<br>29.1 52.6 8.8 29.8<br>|48.1 46.5 47.8 48.3<br><br>|57.9 57.7 58.0 58.0|

- TABLE 8: Benchmark results of interactive VOS methods on MOSEv2 validation set.

TABLE 9: Benchmark results of video object tracking (VOT) methods on MOSEv2 validation set. AUC: area under the success curve; P and Pnorm: precision metrics measuring center location error (raw and size-normalized); AO: average overlap.

MOSEv2 MOSEv1 LaSOT GOT-10k Method Pub.

P Pnorm AUC AUC AUC AO

|SeqTrack-B [20] AQATrack-B [21] ODTrack-B [22] LORAT-B [23] SUTrack-B [24] SAM2-B+ SAMURAI-B+ DAM4SAM-B+ SAM2Long-B+|[CVPR’23]<br><br>[CVPR’24]<br><br>[AAAI’24]<br><br>[ECCV’24]<br><br>[AAAI’25]<br><br><br>[ICLR’25] [Preprint’24]<br><br>[CVPR’25]<br><br><br>[ICCV’25]<br><br>|21.3 24.8 23.7<br><br>22.6 25.6 24.6 21.3 23.8 23.5 20.8 24.1 23.3 24.3 26.4 26.0 29.2 30.0 29.1 35.2 35.5 34.3 35.0 35.4 33.9 32.0 32.6 31.4<br><br><br>|42.9 44.7 47.2 43.8 46.9 58.3 59.5 59.5 58.3<br><br>|71.5 72.7 73.2 71.7 74.4 66.0 70.7 -<br><br>|74.5 76.0 77.0 72.1 79.3 79.6 -|
|---|---|---|---|---|---|
|SeqTrack-L [20] ODTrack-L [22] LORAT-L [23] SUTrack-L [24] SAM2-L SAMURAI-L DAM4SAM-L SAM2Long-L<br><br>|[CVPR’23]<br><br>[AAAI’24] [ECCV’24]<br>[AAAI’25] [ICLR’25]<br><br><br>[Preprint’24] [CVPR’25] [ICCV’25]|23.5 26.3 25.3<br><br>24.4 26.7 25.9<br><br><br>23.6 26.7 25.5 26.9 28.4 27.8<br><br>33.1 33.6 32.1<br><br>37.4 37.8 36.1 36.8 37.3 35.6<br><br>34.2 34.8 33.1<br>|45.7 49.1 46.0 48.6 59.6 60.9 60.9 60.2<br><br>|72.5 74.0 75.1 75.2 70.0 74.2 75.1 73.9|74.8 78.2 77.5 81.5 80.7 81.7 81.1<br><br>|

MOSEv2 MOSEv1 DAVIS17

AUC  &  &@60𝑠  &@60𝑠  &@60𝑠

Method Pub.

MANet [65] [CVPR’20] 28.9 41.2 46.0 79.5 CiVOS [171] [CVPR’21] 32.7 46.1 51.7 84.0 MiVOS [63] [CVPR’21] 36.7 48.9 53.9 88.5 STCN [166] [NeurIPS’21] 39.8 54.1 59.5 88.8

from 1 to 5 does not consistently improve results, and some methods even degrade. This sensitivity to point initialization suggests that ambiguity from point prompts, combined with the complex scenes in MOSEv2, makes it difficult for models to maintain stable segmentation despite additional user input. Compared to DAVIS17, where methods achieve much higher scores, e.g., 86.1%  &̇ for SAM2Long-L, the large performance gap further underscores the challenges of point-based setting in MOSEv2.

#### 4.2 Unsupervised Video Object Segmentation

Unsupervised (or automatic, zero-shot) VOS aims to automatically identify and segment primary objects in videos without manual guidance. Following DAVIS [2], [3], we limit the number of proposals to 20 for a fair comparison. In TABLE 7, we benchmark unsupervised VOS methods on MOSEv2. The results show that all methods perform poorly on MOSEv2, especially in reappearance cases, where  &̇ 𝑟 scores drop to as low as 4.1%–12.6%. Although DEVA [101] achieves the highest  & of 36.0%, this remains far below its 73.4% performance on DAVIS17. For SAM2based methods, we use grid prompts on the first frame to generate candidate masks, which are then propagated to subsequent frames. However, incomplete initial masks limit their effectiveness, with SAM2Long-L reaching only 29.1%  &̇ . The substantial performance gap between MOSEv2 and other benchmarks highlights the challenging nature of our dataset for unsupervised VOS methods, which must handle complex scenes without any manual guidance.

#### 4.3 Interactive Video Object Segmentation

Following the interactive track of the 2019 DAVIS Challenge on VOS [172], we provide initial scribbles for the target object as the first interaction. Interactive video object segmentation methods must predict the full-video segmentation based on this input. After comparing predictions with the ground truth, corrective scribbles are added on the worst-performing frame for refinement. This process can be repeated up to 8 times with a 30-second time limit per object. We report  &@60s to reflect the trade-off between accuracy and efficiency, measuring model performance within

60 seconds of interactive processing. As shown in TABLE 8, we evaluate four recent interactive VOS methods on MOSEv2. All methods show substantial performance drops compared to DAVIS17. STCN [166] achieves the best performance of 54.1%  &@60s, which is far below its 88.8% on DAVIS17. This significant performance gap highlights the increased difficulty of the complex scenarios in MOSEv2.

#### 4.4 Video Object Tracking

Video object tracking (VOT) aims to track a target object throughout a video given an initial bounding box. Unlike VOS, VOT focuses on localization rather than segmentation. To adapt MOSEv2 for VOT, we convert segmentation masks to bounding boxes by using the minimal enclosing rectangle. In TABLE 9, we benchmark nine state-of-the-art VOT methods on MOSEv2, including both traditional trackers and SAM2-based variants. Following LaSOT [28], we adopt P, Pnorm, and AUC as evaluation metrics. The results show that all methods undergo a significant performance drop on MOSEv2 compared to existing VOT benchmarks. Among traditional trackers, SUTrack-L [24] performs best with 27.8% AUC, while LORAT-B [23] performs worst with only 23.3% AUC on MOSEv2. Overall, traditional methods remain weak, with scores between 23.3% and 27.8% AUC. SAM2-based methods achieve higher performance, with SAMURAI-L [25] leading at 36.1%, followed by DAM4SAM-L (35.6%) and SAM2Long-L (33.1%). However, these scores are far below those on other datasets. For example, SAMURAI-L achieves 60.9% AUC on MOSEv1 [1], 74.2% on LaSOT [28], and 81.7% on GOT-10k [29], but drops to only 36.1% on MOSEv2, underscoring the much greater challenges posed by MOSEv2. In addition, larger models (L variants) consistently outperform their base counterparts (B+ variants), suggesting that increased model capacity helps handle the diverse and challenging tracking conditions in MOSEv2.

Although SAMURAI [25] underperforms other SAM2 variants such as SAM2Long [27] in VOS tasks, it shows superior tracking performance in VOT. This is mainly due to two factors. First, VOT metrics do not penalize false positives when the ground truth is empty, which aligns with SAMURAI’s higher  &̇ 𝑟

- TABLE 10: Attribute-based performance analysis on MOSEv2 validation set, with attribute definitions detailed in TABLE 3. The overall metric represents the average value across all attributes. The best score in each metric is highlighted in bold.

Overall OCC DR CRO DV CE NC LD MS KD Method

 &̇  &̇ 𝑑  &̇ 𝑟  &̇  &̇ 𝑑  &̇ 𝑟  &̇  &̇ 𝑑  &̇ 𝑟  &̇  &̇ 𝑑  &̇ 𝑟  &̇  &̇ 𝑑  &̇ 𝑟  &̇  &̇ 𝑑  &̇ 𝑟  &̇  &̇ 𝑑  &̇ 𝑟  &̇  &̇ 𝑑  &̇ 𝑟  &̇  &̇ 𝑑  &̇ 𝑟  &̇  &̇ 𝑑  &̇ 𝑟

XMem 31.7 55.5 12.6 36.8 56.9 14.9 30.8 57.8 13.6 30.8 52.9 9.2 24.3 54.5 8.3 34.0 52.2 12.7 34.6 50.4 16.2 30.7 77.1 10.7 33.2 57.2 16.2 30.2 40.3 11.3 Cutie-B 35.8 61.9 15.7 43.4 64.7 18.4 35.7 59.8 17.3 36.8 60.1 14.2 26.8 67.0 9.6 42.0 68.7 15.2 39.9 55.1 20.7 35.4 81.8 13.5 31.8 51.2 19.6 30.5 48.3 12.7 SAM2-B+ (ZS) 36.8 56.0 17.0 43.5 59.7 20.8 38.6 53.4 23.5 36.6 52.5 14.3 28.5 49.9 10.5 49.2 67.3 24.4 37.3 52.1 17.5 40.5 65.9 18.5 30.1 45.9 14.7 26.9 57.5 9.0 SAM2-B+ 40.7 57.0 21.4 47.1 61.6 23.7 41.5 53.8 26.4 42.5 56.2 21.5 35.1 48.8 18.5 52.6 66.9 28.5 43.1 55.2 22.4 42.5 72.4 22.7 34.0 46.9 18.5 27.8 51.3 9.9 SAMURAI-B+ 42.6 40.7 30.1 48.9 46.6 33.9 44.2 38.7 33.4 41.0 40.6 26.1 37.5 32.5 29.6 55.6 52.7 39.8 43.6 39.1 32.5 51.6 50.8 39.8 32.7 28.4 20.1 28.2 37.1 15.4 DAM4SAM-B+ 42.4 46.4 27.9 48.7 52.1 32.2 44.5 47.2 31.5 40.9 45.5 24.7 39.4 43.2 29.7 52.9 56.6 32.8 43.8 44.5 30.6 51.1 56.9 35.5 34.1 31.5 19.3 25.9 40.4 14.5 SAM2Long-B+ 42.9 52.8 26.2 49.4 59.7 29.9 42.9 50.1 28.9 44.6 53.0 24.7 37.5 50.1 23.2 56.7 65.4 35.5 43.9 53.4 28.2 52.9 65.9 35.4 32.0 33.4 20.3 25.7 44.1 9.9

- TABLE 11: Comparison on long videos (>300 frames) in MOSEv2 and LVOSv2. Δ: the difference between  &̇ 𝑑 and  &̇ 𝑟.

more challenging nature of MOSEv2 dataset. 5) In knowledgedependent (KD) scenarios, all methods demonstrate significantly degraded performance, with Cutie-B [14] achieving only 30.5%  &̇ , underscoring the complexity of KD challenges. Traditional methods such as Cutie-B and XMem outperform SAM2 variants in KD scenarios, likely because they incorporate instance-level memory mechanisms that offer stronger semantic representation. SAM2 [15], in contrast, is not pretrained on such scenarios and lacks heuristic design for knowledge-intensive tasks. Among SAM2-based methods, SAMURAI-B+ [25] performs best in KD scenarios (15.4%  &̇ 𝑟), possibly due to its Kalman filter-based motion modeling provides spatial reasoning that is beneficial in certain KD cases requiring spatial cues.

MOSEv2 (LD) LVOSv2

Method  &̇  &̇ 𝑑  &̇ 𝑟 Δ  &̇  &̇ 𝑑  &̇ 𝑟 Δ

SAM2-B+ 42.5 72.4 22.7 +49.7 82.3 69.4 62.6 +6.8 SAMURAI-B+ 51.6 50.8 39.8 +11.0 81.5 56.8 71.3 -14.5 DAM4SAM-B+ 51.1 56.9 35.5 +21.4 81.4 65.7 71.4 -5.7 SAM2Long-B+ 52.9 65.9 35.4 +30.5 84.3 66.8 68.5 -1.7

scores as shown in TABLE 4. Second, the integration of Kalman filtering effectively captures temporal motion, enhancing localization and trajectory prediction in complex tracking scenarios.

#### 4.5 Attribute-Based Performance Analysis

#### 4.6 Qualitative Analysis

To better understand how different methods perform under specific challenges, TABLE 10 presents an attribute-based analysis on MOSEv2. We evaluate mask-initialization semi-supervised VOS methods across nine representative attributes defined in TABLE 3, including occlusion (OCC), disappearance-reappearance (DR), crowding (CRO), diverse visibility (DV), complex environment (CE), novel categories (NC), long duration (LD), multi-shots (MS), and knowledge dependency (KD).

Fig. 10 presents eight challenging cases that reveal key limitations of existing VOS methods. 1) Models struggle with re-identifying objects after disappearance and occlusion. While SAM2Long [27], which maintains multiple segmentation paths, successfully tracks a car undergoing simple linear motion (case a), it fails in more complex motion patterns such as a person walking around a crowd before reappearing (case b), indicating limitations in modeling long-term and nonlinear trajectories. 2) Densely crowded scenes containing small and heavily occluded targets (case c) remain extremely challenging, none of existing models succeed under such complexity. 3) In cases involving camouflaged objects or non-physical targets like shadows (cases d and e), Cutie [14] outperforms SAM2 [15] and SAM2Long [27], especially in boundary quality. This advantage may stem from Cutie’s compact instancelevel memory, which explicitly models foreground objects and enables better separation from background distractions, while SAM2 relies on global image features lacking instance-specific cues. 4) Under adverse environmental conditions such as heavy snow (case f), the combination of low contrast and occlusion causes all models to fail, with Cutie producing inaccurate masks and SAM2 variants completely losing the target. 5) When faced with dramatic changes in viewpoint and object pose across multiple camera shots (case g), all models fail to maintain consistent tracking, as exemplified by the shifting appearance of a Coke bottle. 6) In scenarios that require understanding physical object relationships and transformation rules (case h), such as tracking a rotating Rubik’s cube, the models fail to re-identify the correct block after disappearance, often incorrectly assigning it to adjacent blocks.

The results reveal several key insights about model performance across different challenges. 1) SAM2Long-B+ [27] achieves the best overall performance with 42.9%  &̇ , consistent with its strong results in previous experiments, suggesting that robustness to MOSEv2’s challenges translates into better general effectiveness. 2) Fine-tuning significantly improves SAM2-B+’s performance, raising its  &̇ from 36.8% to 40.7%, which highlights the importance of adaptation to complex video scenarios. 3) Traditional methods like Cutie-B [14] and XMem [16] excel in frames where objects are disappearing ( &̇ 𝑑), with Cutie-B achieving the highest scores across most attributes (up to 81.8% on LD). However, they struggle significantly on reappearance scenarios ( &̇ 𝑟), often failing to re-identify targets. For example, Cutie-B scores 81.8%  &̇ 𝑑 but only 13.5%  &̇ 𝑟 on LD videos, indicating a tendency toward false negatives when objects reappear. 4) A comparison with LVOSv2 [62], which specifically focuses on long videos, highlights that the longduration sequences in MOSEv2 involve not only extended frame counts but also greater scene complexity. As shown in TABLE 11, LVOSv2 exhibits small Δ values, i.e., the gap between  &̇ 𝑑 and  &̇ 𝑟, indicating minimal difficulty in reappearance cases. In contrast, the LD subset of MOSEv2 shows much larger Δ values (+11.0 to +49.7), indicating severe reappearance difficulty. These challenges arise from frequent occlusions, camera shot transitions, background clutter, ambiguous reappearance cases, etc. For example, SAM2Long-B+ achieves 84.3%  &̇ on LVOSv2 but only

#### 4.7 Enhancing SAM2 for Complex Scenarios

Based on the above experimental results and the characteristics of MOSEv2, we introduce several practical improvements to SAM2. ∙ Revisiting Memory Control in SAM2. SAM2 employs two types of memory: conditioned and unconditioned. The conditioned

- 52.9% on MOSEv2’s LD subset, underscoring the substantially

|[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>(a)|[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>(b)|
|---|---|
|[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>(c)|[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]<br><br>(d)|
|[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>(e)|[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>(f)|
|[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>(g)|[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]<br><br>[Figure 189]<br><br>[Figure 190]<br><br>[Figure 191]<br><br>[Figure 192]<br><br>[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>(h)|

GTCutieSAM2SAM2LongGTCutieSAM2GTCutieSAM2SAM2LongGTCutieSAM2SAM2LongSAM2Long

- Fig. 10: Qualitative results on MOSEv2. We compare Cutie [14], SAM2 [15], and SAM2Long [27] on 8 challenging cases that assess model performance under various complex conditions. These include object disappearance and reappearance (a, b, e, g, h), small/inconspicuous objects (c), heavy occlusions (c, f), crowded scenes (c), adverse weather (f), low-light environments (a, d), multishot sequences (g), camouflaged targets (d), non-physical targets (e), and knowledge-dependent scenarios (h).

memory, typically derived from the initial frame, stores reliable object features that serve as strong references for object tracking and are particularly important for re-identification during disappearance–reappearance scenarios. The unconditioned memories,

collected from the nearest temporal frames (up to 6), capture shortterm appearance variations and motion dynamics. By default, however, the conditioned memory contains only a single frame from the initialization, which poses two major limitations. First, relying

Algorithm 1 Reliable Conditioned Memory Selection (RCMS).

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

- 46
- 47
- 48
- 49
- 50

- 46
- 47
- 48
- 49
- 50

- 1 def RCMS(sam2_model, mask_0, frame_list, theta, N, K):

- 2 # sam2_model: SAM2 model for video segmentation

- 3 # M_0: Initial mask from the first frame

- 4 # frame_list: List of video frames to process

- 5 # K: Number of unconditioned memories, default 6

- 6 # N: Maximum number of conditioned memories to select

- 7 # theta: Quality threshold for memory selection

- 8 # Process first frame to get initial memory

- 9 memory_0 = sam2_model.init_state(frame_list[0], mask_0)

- 10 cond_memory = [memory_0]

- 11 all_memory = [] # Other memories expect cond

- 12 masks = [mask_0]

- 13 # Process remaining frames

- 14 disappeared = False

- 15 for t, frame in enumerate(frame_list[1:]):

- 16 # Track object in current frame

- 17 uncond_memory = select_nearest(all_memory, K)

- 18 mask_t, memory_t =

- 19 sam2_model.step(frame, cond_memory, uncond_memory)

- 20 all_memory.append(memory_t)

- 21 masks.append(mask_t)

- 22 # Apply RCMS if object disappeared

- 23 if is_empty(mask_t) and not disappeared:

- 24 disappeared = True

- 25 for i in range(len(all_memory) - 2, -1, -1):

- 26 memory = all_memory[i]

- 27 Q = compute_quality_scores(memory)

- 28 if Q > theta and len(cond_memory) < N + 1:

- 29 cond_memory.append(memory)

- 30 all_memory.pop(i)

- 31 return masks

&Score

&Score

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

0 1 2 3 4 5 6 7 8 9 10

(b) Quality threshold

(a) Number of selected memories

Fig. 11: Ablation on RCMS and MQF. (a) number of selected conditioned memories. (b) quality threshold 𝜃 for memory selection.

(from 46.0% to 49.3%), with particularly strong gains in  &̇ 𝑟 (+6.5%), confirming its effectiveness for object re-identification after disappearance. Incorporating MQF yields an additional +0.9% improvement in  &̇ , highlighting the importance of filtering for high-quality memories. Fig. 11 presents ablation studies on RCMS parameters. Performance increases steadily from 𝑁 = 0 (46.0%  &̇ ) to 𝑁 = 4 (50.2%  &̇ ), then saturates, suggesting that 4 additional memories provide sufficient diversity. For the quality threshold 𝜃, performance peaks at 𝜃 = 0.6 (50.2%  &̇ ). Compared to existing SAM2 variants, RCMS demonstrates clear advantages by leveraging reliable pre-disappearance memories. DAM4SAM [26] also adds additional memories, but relies solely on threshold-based selection without considering timing. In contrast, RCMS leverages SAM2’s strength in tracking continuously visible objects to introduce more reliable memories at appropriate moments. SAM2Long [27] maintains multiple segmentation paths but suffers heavy computational overhead (9.4 FPS vs. our 22.6 FPS). Our method overcomes the limitation of depending solely on the initial frame memory while preserving efficiency and exploiting optimal timing for memory augmentation.

- TABLE 12: Ablation study of SAM2 improvements. * denotes SAM2-L with all improvements applied. Method  &̇  ̇  &̇ 𝑑  &̇ 𝑟   &  &̇ 𝑏𝑑 FPS

|SAMURAI-B+ DAM4SAM-B+ SAM2Long-B+|47.4 45.3 49.5 45.9 33.6 52.2 48.8 73.3<br><br>47.9 45.8 50.0 51.3 32.0 52.6 49.2 73.1<br><br>48.6 46.7 50.5 58.4 29.2 52.8 49.7 72.9<br><br><br>|17.7 17.3<br><br>9.4|
|---|---|---|
|SAM2-B+<br><br>+RCMS w/o MQF<br><br>+MQF<br><br>+MSS<br><br>+LVT|46.0 44.2 47.8 61.6 23.2 50.0 47.1 73.5<br><br>49.3 47.4 51.2 61.0 29.7 53.5 50.4 73.7<br><br>50.2 48.2 52.2 59.5 31.4 54.6 51.4 74.0<br><br><br>50.6 48.5 52.7 55.8 33.7 55.4 51.9 73.9<br><br>51.5 49.5 53.6 56.6 36.5 56.3 52.9 74.0<br><br><br>|23.4 22.4 22.6 22.6 22.6<br><br>|
|SAM2-L SAM2-L* (ours)<br><br>|49.7 47.9 51.5 64.5 27.1 53.8 50.9 74.6 54.4 52.4 56.3 66.8 33.2 58.9 55.6 75.6<br><br>|14.4 14.3<br><br>|

In addition, we incorporate the Mask Scaling Strategy (MSS) and Long-Video Finetuning (LVT) to further enhance SAM2. MSS adjusts mask output distributions with a scaling factor of 7.5 and an offset of –4.0 [17], improving robustness to small objects and occlusions. LVT adapts SAM2 to challenging long-duration scenarios, which are particularly prevalent in MOSEv2. After the default finetuning with 8 frames, we perform additional finetuning with 16 frames while freezing the image encoder, enabling the model to better capture long-term temporal dependencies. As shown in TABLE 12, MSS yields a +0.4% gain in  &̇ with a notable +2.3% improvement in  &̇ 𝑟, while LVT further boosts performance by +0.9% in  &̇ and +2.8% in  &̇ 𝑟.

on a single frame restricts the diversity of object representations and limits the model’s ability to capture appearance variations. Second, if the initial frame provides only partial visibility of the target object (e.g., case ② in Fig. 1), the resulting memory lacks sufficient information for reliable segmentation in subsequent frames. This raises a key question: how can we obtain more reliable conditioned memories without incurring additional cost? ∙ Reliable Conditioned Memory Selection. As shown in TABLE 12, SAM2-B+ achieves a  &̇ 𝑏𝑑 of 73.5%, a metric that evaluates performance exclusively during the initial continuous segment before the first disappearance of the target object. This indicates that SAM2 is highly robust in tracking continuously visible targets. Motivated by this observation, we propose Reliable Conditioned Memory Selection (RCMS), which preserves SAM2’s strong tracking ability in pre-disappearance frames while strategically augmenting the conditioned memory when disappearance occurs. As shown in Algorithm 1 (L22-30), RCMS selects the 𝑁 nearest high-quality memories from the pre-disappearance sequence and incorporates them into the conditioned memory bank. To ensure memory reliability, we adopt Memory Quality Filtering (MQF). Specifically, for each candidate memory, a quality score is computed as: 𝑄 = 𝑠𝑐𝑜𝑟𝑒𝑖𝑜𝑢 × 𝑠𝑐𝑜𝑟𝑒𝑜𝑐𝑐 × 𝑚𝑎𝑠𝑘𝑛𝑒𝑠𝑠, where 𝑠𝑐𝑜𝑟𝑒𝑖𝑜𝑢 and 𝑠𝑐𝑜𝑟𝑒𝑜𝑐𝑐 are outputs of SAM2’s mask decoder indicating predicted IoU and occlusion confidence, respectively, and 𝑚𝑎𝑠𝑘𝑛𝑒𝑠𝑠 [173] measures mask quality. Only frames with 𝑄 above a threshold 𝜃 are incorporated into the conditioned memory. In total, RCMS produces at most 𝑁 + 1 conditioned memories.

Together, these improvements raise SAM2-B+ from 46.0% to 51.5%  &̇ , a substantial +5.5% gain while retaining competitive inference speed. Applying all improvements to SAM2-L, denoted as SAM2-L* in TABLE 12, further boosts performance from 49.7% to 54.4%  &̇ , a +4.7% increase without sacrificing speed. It is worth noting that SAM2-L* achieves lower  &̇ 𝑟 than SAM2-B+. Since the improvements primarily target reidentification after disappearance, they raise  &̇ 𝑟 for both backbones. However, the weaker B+ backbone struggles to balance  &̇ 𝑑 and  &̇ 𝑟, often gaining higher  &̇ 𝑟 at the cost of lower  &̇ 𝑑. In contrast, the stronger L backbone maintains a better balance, keeping both metrics at desirable levels.

### 5 DISCUSSION AND FUTURE DIRECTIONS

Based on the comprehensive analysis of MOSEv2 and the results of existing methods, we identify several key challenges and future research directions for complex video object segmentation.

#### ∙ Robust Re-identification for Disappearance-Reappearance. The large drop in  &̇ 𝑟 scores reveals a key challenge in handling

As shown in TABLE 12, RCMS improves  &̇ by +3.3%

ject masks across 200 categories. The dataset not only intensifies challenges in MOSEv1, such as object disappearancereappearance, occlusions, and crowded scenes, but also introduces new challenges, such as adverse weather, low-light scenes, multishot sequences, camouflaged targets, non-physical targets, and knowledge-dependent cases. Evaluation across multiple VOS and VOT settings reveals that current state-of-the-art methods suffer significant performance drops on MOSEv2. For example, SAM2 drops from 90.7%  & on DAVIS 2017 to 50.9% on MOSEv2. These results highlight the gap between existing algorithms and the demands of real-world deployment. Based on the analysis of the observed challenges, several practical tricks are proposed, which substantially enhance model performance with a 5.5%  &̇ gain. We believe MOSEv2 will serve as a valuable resource for advancing robust and generalizable video object segmentation and tracking in diverse and unconstrained environments.

disappearance-reappearance, especially with complex motions, viewpoint changes, or knowledge-dependent cues (e.g., Fig. 1⑩). Overly aggressive matching can inflate false positives during disappearance, reducing  &̇ 𝑑. Future research should develop adaptive re-identification strategies that integrate appearance, motion, and semantic reasoning to better handle these scenarios.

∙ Occlusion Handling. MOSEv2 contains frequent and complex occlusions. Current methods often fail when objects are partly or fully hidden. Future work should design occlusion-aware models, such as attention for hidden regions, multi-scale feature fusion, and temporal models that keep object identity through occlusion.

∙ Tracking in Crowded and Small-Target Scenarios. Small objects and crowded scenes often co-occur in MOSEv2, posing great challenges. Limited input resolutions (e.g., 480p in Cutie, 1024p in SAM2) lose fine details, making small-object tracking difficult. Future work should develop efficient ways to process high-resolution inputs and strengthen feature learning for small targets, such as multi-scale architectures, small-object–focused attention, and contrastive learning to distinguish targets from similar distractors in crowded scenes.

### REFERENCES

- [1] H. Ding, C. Liu, S. He, X. Jiang, P. H. Torr, and S. Bai, “MOSE: A new dataset for video object segmentation in complex scenes,” in ICCV,

2023. 1, 2, 3, 5, 6, 7, 8, 10

- [2] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung, “A benchmark dataset and evaluation methodology for video object segmentation,” in CVPR, 2016. 1, 3, 5, 6, 7, 8, 10
- [3] J. Pont-Tuset, F. Perazzi, S. Caelles, P. Arbeláez, A. Sorkine-Hornung, and L. Van Gool, “The 2017 davis challenge on video object segmentation,” arXiv preprint arXiv:1704.00675, 2017. 1, 3, 5, 6, 7, 8, 9, 10
- [4] N. Xu, L. Yang, Y. Fan, J. Yang, D. Yue, Y. Liang, B. Price, S. Cohen, and T. Huang, “Youtube-vos: Sequence-to-sequence video object segmentation,” in ECCV, 2018. 1, 3, 5, 6, 8
- [5] S. Caelles, K. Maninis, J. Pont-Tuset, L. Leal-Taixé, D. Cremers, and L. V. Gool, “One-shot video object segmentation,” in CVPR, 2017. 1, 3
- [6] H. Park, J. Yoo, S. Jeong, G. Venkatesh, and N. Kwak, “Learning dynamic network using a reuse gate function in semi-supervised video object segmentation,” in CVPR, 2021. 1
- [7] S. D. Jain, B. Xiong, and K. Grauman, “Fusionseg: Learning to combine motion and appearance for fully automatic segmention of generic objects in videos,” in CVPR, 2017. 1, 4
- [8] J. Cheng, Y.-H. Tsai, S. Wang, and M.-H. Yang, “Segflow: Joint learning for video object segmentation and optical flow,” in ICCV, 2017. 1, 4
- [9] Y. Chen, J. Pont-Tuset, A. Montes, and L. Van Gool, “Blazingly fast video object segmentation with pixel-wise metric learning,” in CVPR,

- 2018. 1, 3

[10] S. W. Oh, J.-Y. Lee, N. Xu, and S. J. Kim, “Fast user-guided video object segmentation by interaction-and-propagation networks,” in CVPR,

- 2019. 1, 3

- [11] T. Brox and J. Malik, “Object segmentation by long term analysis of point trajectories,” in ECCV, 2010. 1
- [12] Y. J. Lee, J. Kim, and K. Grauman, “Key-segments for video object segmentation,” in ICCV, 2011. 1
- [13] Q. Fan, F. Zhong, D. Lischinski, D. Cohen-Or, and B. Chen, “JumpCut: non-successive mask transfer and interpolation for video cutout.” ACM Tran. Graphics, vol. 34, no. 6, 2015. 1, 6
- [14] H. K. Cheng, S. W. Oh, B. Price, J.-Y. Lee, and A. Schwing, “Putting the object back into video object segmentation,” in CVPR, 2024. 1, 3, 8, 9, 11, 12
- [15] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson et al., “SAM 2: Segment anything in images and videos,” in ICLR, 2025. 1, 3, 4, 5, 6, 8, 9, 11, 12
- [16] H. K. Cheng and A. G. Schwing, “XMem: long-term video object segmentation with an atkinson-shiffrin memory model,” in ECCV, 2022. 1, 3, 9, 11
- [17] H. Ding, C. Liu, N. Ravi, S. He, Y. Wei, S. Bai, and P. Torr, “PVUW 2025 challenge report: Advances in pixel-level understanding of complex videos in the wild,” in CVPR Workshop, 2025. 1, 13
- [18] H. Ding, C. Liu, Y. Wei, N. Ravi, S. He, S. Bai, P. Torr, D. Miao, X. Li, Z. He et al., “PVUW 2024 challenge on complex video understanding: Methods and results,” in ECCV Workshop, 2025. 1
- [19] H. Ding, L. Hong, C. Liu, N. Xu, L. Yang, Y. Fan, D. Miao, Y. Gu, X. Li, Z. He et al., “LSVOS challenge report: Large-scale complex and long video object segmentation,” in ECCV Workshop, 2025. 1

∙ Generalization to Rare Categories. Although VOS methods are designed to be class-agnostic, generalizing to rare or unseen categories remains difficult. MOSEv2 contains 200 categories with a clear long-tail distribution, including uncommon targets like shadows and camouflaged objects. Current methods often fail on these categories due to limited training data and domain gaps. Future work could explore test-time adaptation using first-frame cues, or design stronger instance-level representations that better generalize to rare and visually ambiguous objects.

∙ Environmental Robustness. MOSEv2 covers diverse adverse environments, e.g., rain, snow, fog, nighttime, and underwater scenes, which severely degrade the performance of current VOS methods. In such settings, low visibility makes object appearance unreliable, while illumination changes and environmental occlusions disrupt temporal consistency. Future research should explore adaptive techniques such as weather-invariant and illuminationrobust representations, as well as the integration of auxiliary signals or priors to improve robustness in real-world scenarios.

∙Multi-Shot Video Handling. Most methods rely on appearance matching and position estimation under temporal continuity, which fails in multi-shot videos where scene transitions cause abrupt changes in object appearance and position. Such structures are common in real-world content, especially narrative-driven videos. Future research should explore shot-aware strategies that handle discontinuities while preserving object identity across shots.

∙ Knowledge-Dependent Tracking. Although recent methods have made progress in many VOS scenarios, they still struggle in cases requiring external knowledge such as spatial reasoning or common sense. These limitations stem from the fact that most models mainly rely on appearance and positional cues with limited reasoning ability. Future work could explore integrating MLLMs [81], [82] to enhance semantic understanding and high-level reasoning. The main challenge is achieving this integration while preserving computational efficiency and real-time performance.

### 6 CONCLUSION

In this work, we introduce MOSEv2, a significantly more challenging dataset for video object segmentation in complex scenes. It extends MOSEv1 in both scale and complexity of scenarios, comprising 5,024 high-resolution videos and 701,976 ob-

- [20] X. Chen, H. Peng, D. Wang, H. Lu, and H. Hu, “Seqtrack: Sequence to sequence learning for visual object tracking,” in CVPR, 2023. 3, 10
- [21] J. Xie, B. Zhong, Z. Mo, S. Zhang, L. Shi, S. Song, and R. Ji, “Autoregressive queries for adaptive tracking with spatio-temporal transformers,” in CVPR, 2024. 3, 10
- [22] Y. Zheng, B. Zhong, Q. Liang, Z. Mo, S. Zhang, and X. Li, “Odtrack: Online dense temporal token learning for visual tracking,” in AAAI,

2024. 3, 10

- [23] L. Lin, H. Fan, Z. Zhang, Y. Wang, Y. Xu, and H. Ling, “Tracking meets lora: Faster training, larger model, stronger performance,” in ECCV,

- 2024. 3, 10

[24] X. Chen, B. Kang, W. Geng, J. Zhu, Y. Liu, D. Wang, and H. Lu, “Sutrack: Towards simple and unified single object tracking,” in AAAI,

- 2025. 3, 10

- [25] C.-Y. Yang, H.-W. Huang, W. Chai, Z. Jiang, and J.-N. Hwang, “SAMURAI: Adapting segment anything model for zero-shot visual tracking with motion-aware memory,” arXiv preprint arXiv:2411.11922, 2024. 3, 8, 9, 10, 11
- [26] J. Videnovic, A. Lukezic, and M. Kristan, “A distractor-aware memory for visual object tracking with SAM2,” in CVPR, 2025. 3, 5, 6, 8, 9, 13
- [27] S. Ding, R. Qian, X. Dong, P. Zhang, Y. Zang, Y. Cao, Y. Guo, D. Lin, and J. Wang, “SAM2Long: Enhancing sam 2 for long video segmentation with a training-free memory tree,” in ICCV, 2025. 3, 8, 9, 10, 11, 12, 13
- [28] H. Fan, L. Lin, F. Yang, P. Chu, G. Deng, S. Yu, H. Bai, Y. Xu, C. Liao, and H. Ling, “LaSOT: A high-quality benchmark for large-scale single object tracking,” in CVPR, 2019. 3, 4, 5, 6, 10
- [29] L. Huang, X. Zhao, and K. Huang, “GOT-10k: A large high-diversity benchmark for generic object tracking in the wild,” IEEE TPAMI, vol. 43, no. 5, 2019. 3, 4, 5, 6, 10
- [30] T. Zhou, F. Porikli, D. J. Crandall, L. Van Gool, and W. Wang, “A survey on deep learning technique for video segmentation,” IEEE TPAMI,

2023. 3

- [31] F. Perazzi, A. Khoreva, R. Benenson, B. Schiele, and A. SorkineHornung, “Learning video object segmentation from static images,” in CVPR, 2017. 3
- [32] W.-D. Jang and C.-S. Kim, “Online video object segmentation via convolutional trident network,” in CVPR, 2017. 3
- [33] V. Jampani, R. Gadde, and P. V. Gehler, “Video propagation networks,” in CVPR, 2017. 3
- [34] H. Xiao, J. Feng, G. Lin, Y. Liu, and M. Zhang, “Monet: Deep motion exploitation for video object segmentation,” in CVPR, 2018. 3
- [35] P. Hu, G. Wang, X. Kong, J. Kuen, and Y.-P. Tan, “Motion-guided cascaded refinement network for video object segmentation,” in CVPR,

2018. 3

- [36] J. Han, L. Yang, D. Zhang, X. Chang, and X. Liang, “Reinforcement cutting-agent learning for video object segmentation,” in CVPR, 2018. 3
- [37] J. Cheng, Y.-H. Tsai, W.-C. Hung, S. Wang, and M.-H. Yang, “Fast and accurate online video object segmentation via tracking parts,” in CVPR,

2018. 3

- [38] S. Xu, D. Liu, L. Bao, W. Liu, and P. Zhou, “Mhp-vos: Multiple hypotheses propagation for video object segmentation,” in CVPR, 2019. 3
- [39] X. Chen, Z. Li, Y. Yuan, G. Yu, J. Shen, and D. Qi, “State-aware tracker for real-time video object segmentation,” in CVPR, 2020. 3
- [40] X. Huang, J. Xu, Y.-W. Tai, and C.-K. Tang, “Fast video object segmentation with temporal aggregation network and dynamic template matching,” in CVPR, 2020. 3
- [41] S. Wug Oh, J.-Y. Lee, K. Sunkavalli, and S. Joo Kim, “Fast video object segmentation by reference-guided mask propagation,” in CVPR, 2018. 3
- [42] A. Jabri, A. Owens, and A. Efros, “Space-time correspondence as a contrastive random walk,” in NeurIPS, 2020. 3
- [43] H. Lin, X. Qi, and J. Jia, “Agss-vos: Attention guided single-shot video object segmentation,” in CVPR, 2019. 3
- [44] L. Zhang, Z. Lin, J. Zhang, H. Lu, and Y. He, “Fast video object segmentation via dynamic targeting network,” in ICCV, 2019. 3
- [45] J. S. Yoon, F. Rameau, J. Kim, S. Lee, S. Shin, and I. S. Kweon, “Pixellevel matching for video object segmentation using convolutional neural networks,” in ICCV, 2017. 3
- [46] P. Voigtlaender, Y. Chai, F. Schroff, H. Adam, B. Leibe, and L.-C. Chen, “Feelvos: Fast end-to-end embedding learning for video object segmentation,” in CVPR, 2019. 3
- [47] Z. Wang, J. Xu, L. Liu, F. Zhu, and L. Shao, “Ranet: Ranking attention network for fast video object segmentation,” in ICCV, 2019. 3

- [48] K. Duarte, Y. S. Rawat, and M. Shah, “Capsulevos: Semi-supervised video object segmentation using capsule routing,” in ICCV, 2019. 3
- [49] S. W. Oh, J.-Y. Lee, N. Xu, and S. J. Kim, “Video object segmentation using space-time memory networks,” in ICCV, 2019. 3
- [50] Y. Zhang, Z. Wu, H. Peng, and S. Lin, “A transductive approach for video object segmentation,” in CVPR, 2020. 3
- [51] Z. Lai, E. Lu, and W. Xie, “MAST: A memory-augmented selfsupervised tracker,” in CVPR, 2020. 3
- [52] Z. Yang, Y. Wei, and Y. Yang, “Collaborative video object segmentation by foreground-background integration,” in ECCV, 2020. 3
- [53] L. Hu, P. Zhang, B. Zhang, P. Pan, Y. Xu, and R. Jin, “Learning position and target consistency for memory-based video object segmentation,” in CVPR, 2021. 3
- [54] B. Duke, A. Ahmed, C. Wolf, P. Aarabi, and G. W. Taylor, “Sstvos: Sparse spatiotemporal transformers for video object segmentation,” in CVPR, 2021. 3
- [55] M. Bekuzarov, A. Bermudez, J.-Y. Lee, and H. Li, “Xmem++: Production-level video segmentation from few annotated frames,” in ICCV, 2023. 3, 9
- [56] Q. Wang, L. Zhang, L. Bertinetto, W. Hu, and P. H. Torr, “Fast online object tracking and segmentation: A unifying approach,” in CVPR,

2019. 3

- [57] M. Sun, J. Xiao, E. G. Lim, B. Zhang, and Y. Zhao, “Fast template matching and update for video object tracking and segmentation,” in CVPR, 2020. 3
- [58] F. Lin, H. Xie, Y. Li, and Y. Zhang, “Query-memory re-aggregation for weakly-supervised video object segmentation,” in AAAI, 2021. 3
- [59] Q. Yang, Y. Yao, M. Cui, and L. Bo, “Mosam: Motion-guided segment anything model with spatial-temporal memory selection,” arXiv preprint arXiv:2505.00739, 2025. 3
- [60] R. E. Kalman, “A new approach to linear filtering and prediction problems,” Journal of Basic Engineering, 1960. 3
- [61] L. Hong, W. Chen, Z. Liu, W. Zhang, P. Guo, Z. Chen, and W. Zhang, “LVOS: A benchmark for long-term video object segmentation,” in ICCV, 2023. 3, 5, 6
- [62] L. Hong, Z. Liu, W. Chen, C. Tan, Y. Feng, X. Zhou, P. Guo, J. Li, Z. Chen, S. Gao et al., “LVOS: A benchmark for large-scale long-term video object segmentation,” arXiv preprint arXiv:2404.19326, 2024. 3, 5, 6, 11
- [63] H. K. Cheng, Y.-W. Tai, and C.-K. Tang, “Modular interactive video object segmentation: Interaction-to-mask, propagation and differenceaware fusion,” in CVPR, 2021. 3, 10
- [64] Y. Heo, Y. J. Koh, and C.-S. Kim, “Guided interactive video object segmentation using reliability-based attention maps,” in CVPR, 2021. 3
- [65] J. Miao, Y. Wei, and Y. Yang, “Memory aggregation networks for efficient interactive video object segmentation,” in CVPR, 2020. 3, 10
- [66] B. Chen, H. Ling, X. Zeng, G. Jun, Z. Xu, and S. Fidler, “Scribblebox: Interactive annotation framework for video object segmentation,” in ECCV, 2020. 3
- [67] Z. Yin, J. Zheng, W. Luo, S. Qian, H. Zhang, and S. Gao, “Learning to recommend frame for interactive video object segmentation in the wild,” in CVPR, 2021. 3
- [68] H. Ding, C. Liu, S. He, X. Jiang, and C. C. Loy, “MeViS: A large-scale benchmark for video segmentation with motion expressions,” in ICCV,

2023. 4

- [69] H. Ding, C. Liu, S. He, K. Ying, X. Jiang, C. C. Loy, and Y.-G. Jiang, “MeViS: A multi-modal dataset for referring motion expression video segmentation,” IEEE TPAMI, 2025. 4
- [70] S. Seo, J.-Y. Lee, and B. Han, “Urvos: Unified referring video object segmentation network with a large-scale benchmark,” in ECCV, 2020. 4
- [71] H. Ding, C. Liu, S. Wang, and X. Jiang, “Vision-language transformer and query generation for referring segmentation,” in ICCV, 2021. 4
- [72] L. Ye, M. Rochan, Z. Liu, X. Zhang, and Y. Wang, “Referring segmentation in images and videos with cross-modal self-attention network,” IEEE TPAMI, 2021. 4
- [73] K. Ying, H. Hu, and H. Ding, “MOVE: Motion-guided few-shot video object segmentation,” in ICCV, 2025. 4
- [74] H. Ding, C. Liu, S. Wang, and X. Jiang, “VLT: Vision-language transformer and query generation for referring segmentation,” IEEE TPAMI, 2023. 4
- [75] S. Liu, T. Hui, S. Huang, Y. Wei, B. Li, and G. Li, “Cross-modal progressive comprehension for referring segmentation,” IEEE TPAMI,

- 2021. 4

[76] A. Botach, E. Zheltonozhskii, and C. Baskin, “End-to-end referring video object segmentation with multimodal transformers,” in CVPR,

- 2022. 4

- [77] D. Wu, X. Dong, L. Shao, and J. Shen, “Multi-level representation learning with semantic alignment for referring video object segmentation,” in CVPR, 2022. 4
- [78] S. He and H. Ding, “Decoupling static and hierarchical motion perception for referring video segmentation,” in CVPR, 2024. 4
- [79] C. Yan, H. Wang, S. Yan, X. Jiang, Y. Hu, G. Kang, W. Xie, and E. Gavves, “Visa: Reasoning video object segmentation via large language models,” in ECCV, 2024. 4
- [80] Z. Bai, T. He, H. Mei, P. Wang, Z. Gao, J. Chen, Z. Zhang, and M. Z. Shou, “One token to seg them all: Language instructed reasoning segmentation in videos,” in NeurIPS, 2024. 4
- [81] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” in NeurIPS, 2023. 4, 14
- [82] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu et al., “Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks,” in CVPR, 2024. 4, 14
- [83] K. Ying, H. Ding, G. Jie, and Y.-G. Jiang, “Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation,” in ICCV, 2025. 4
- [84] H. Ding, S. Tang, S. He, C. Liu, Z. Wu, and Y.-G. Jiang, “Multimodal referring segmentation: A survey,” arXiv, 2025. 4
- [85] K. Fragkiadaki, P. Arbelaez, P. Felsen, and J. Malik, “Learning to segment moving objects in videos,” in CVPR, 2015. 4
- [86] Y. Yang, B. Lai, and S. Soatto, “Dystab: Unsupervised object segmentation via dynamic-static bootstrapping,” in CVPR, 2021. 4
- [87] S. Ren, W. Liu, Y. Liu, H. Chen, G. Han, and S. He, “Reciprocal transformations for unsupervised video object segmentation,” in CVPR,

2021. 4

- [88] D. Liu, D. Yu, C. Wang, and P. Zhou, “F2net: Learning to focus on the foreground for unsupervised video object segmentation,” in AAAI,

2021. 4

- [89] X. Lu, W. Wang, M. Danelljan, T. Zhou, J. Shen, and L. Van Gool, “Video object segmentation with episodic graph memory networks,” in ECCV, 2020. 4
- [90] X. Lu, W. Wang, J. Shen, Y.-W. Tai, D. J. Crandall, and S. C. Hoi, “Learning video object segmentation from unlabeled videos,” in CVPR,

2020. 4

- [91] P. Tokmakov, C. Schmid, and K. Alahari, “Learning to segment moving objects,” IJCV, vol. 127, no. 3, 2019. 4
- [92] Z. Yang, Q. Wang, L. Bertinetto, W. Hu, S. Bai, and P. H. Torr, “Anchor diffusion for unsupervised video object segmentation,” in ICCV, 2019. 4
- [93] H. Li, G. Chen, G. Li, and Y. Yu, “Motion guided attention for video salient object detection,” in ICCV, 2019. 4
- [94] W. Wang, H. Song, S. Zhao, J. Shen, S. Zhao, S. C. Hoi, and H. Ling, “Learning unsupervised video object segmentation through visual attention,” in CVPR, 2019. 4
- [95] G. Li, Y. Xie, T. Wei, K. Wang, and L. Lin, “Flow guided recurrent neural encoder for video salient object detection,” in CVPR, 2018. 4
- [96] P. Tokmakov, K. Alahari, and C. Schmid, “Learning video object segmentation with visual memory,” in ICCV, 2017. 4
- [97] T. Zhou, S. Wang, Y. Zhou, Y. Yao, J. Li, and L. Shao, “Motion-attentive transition for zero-shot video object segmentation,” in AAAI, 2020. 4
- [98] X. Lu, W. Wang, C. Ma, J. Shen, L. Shao, and F. Porikli, “See more, know more: Unsupervised video object segmentation with co-attention siamese networks,” in CVPR, 2019. 4
- [99] W. Wang, X. Lu, J. Shen, D. J. Crandall, and L. Shao, “Zero-shot video object segmentation via attentive graph neural networks,” in ICCV,

2019. 4

- [100] X. Lu, W. Wang, J. Shen, D. Crandall, and L. Van Gool, “Segmenting objects from relational visual data,” IEEE TPAMI, 2021. 4
- [101] H. K. Cheng, S. W. Oh, B. Price, A. Schwing, and J.-Y. Lee, “Tracking anything with decoupled video segmentation,” in ICCV, 2023. 4, 9, 10
- [102] M. Ye, S. W. Oh, L. Ke, and J.-Y. Lee, “Entitysam: Segment everything in video,” in CVPR, 2025. 4, 10
- [103] L. Yang, Y. Fan, and N. Xu, “Video instance segmentation,” in ICCV,

2019. 4

- [104] X. Li, H. He, Y. Yang, H. Ding, K. Yang, G. Cheng, Y. Tong, and D. Tao, “Improving video instance segmentation via temporal pyramid routing,” IEEE TPAMI, 2022. 4
- [105] L. Ke, H. Ding, M. Danelljan, Y.-W. Tai, C.-K. Tang, and F. Yu, “Video mask transfiner for high-quality video instance segmentation,” in ECCV,

2022. 4

- [106] L. Ke, M. Danelljan, H. Ding, Y.-W. Tai, C.-K. Tang, and F. Yu, “Mask-

- [107] K. Ying, Q. Zhong, W. Mao, Z. Wang, H. Chen, L. Y. Wu, Y. Liu, C. Fan, Y. Zhuge, and C. Shen, “CTVIS: Consistent Training for Online Video Instance Segmentation,” in ICCV, 2023. 4
- [108] T. Zhang, X. Tian, Y. Wu, S. Ji, X. Wang, Y. Zhang, and P. Wan, “Dvis: Decoupled video instance segmentation framework,” in ICCV, 2023. 4
- [109] Y. Zhou, T. Zhang, S. Ji, S. Yan, and X. Li, “Improving video segmentation via dynamic anchor queries,” in ECCV, 2024. 4
- [110] T. Zhang, X. Tian, Y. Zhou, S. Ji, X. Wang, X. Tao, Y. Zhang, P. Wan, Z. Wang, and Y. Wu, “Dvis++: Improved decoupled framework for universal video segmentation,” IEEE TPAMI, 2025. 4
- [111] J. Qi, Y. Gao, Y. Hu, X. Wang, X. Liu, X. Bai, S. Belongie, A. Yuille, P. H. Torr, and S. Bai, “Occluded video instance segmentation: A benchmark,” IJCV, vol. 130, no. 8, 2022. 4, 6
- [112] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in CVPR, 2015. 4
- [113] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, “Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs,” IEEE TPAMI, 2017. 4
- [114] H. Ding, X. Jiang, B. Shuai, A. Q. Liu, and G. Wang, “Context contrasted feature and gated multi-scale aggregation for scene segmentation,” in CVPR, 2018. 4
- [115] G. J. Brostow, J. Fauqueur, and R. Cipolla, “Semantic object classes in video: A high-definition ground truth database,” Pattern Recognition Letters, vol. 30, no. 2, 2009. 4
- [116] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset for semantic urban scene understanding,” in CVPR, 2016. 4
- [117] J. Miao, Y. Wei, Y. Wu, C. Liang, G. Li, and Y. Yang, “Vspw: A largescale dataset for video scene parsing in the wild,” in CVPR, 2021. 4
- [118] G. Sun, Y. Liu, H. Ding, T. Probst, and L. Van Gool, “Coarse-to-fine feature mining for video semantic segmentation,” in CVPR, 2022. 4
- [119] G. Sun, Y. Liu, H. Ding, M. Wu, and L. Van Gool, “Learning local and global temporal contexts for video semantic segmentation,” IEEE TPAMI, 2024. 4
- [120] S. A. S. Hesham, Y. Liu, G. Sun, H. Ding, J. Yang, E. Konukoglu, X. Geng, and X. Jiang, “Exploiting temporal state space sharing for video semantic segmentation,” in CVPR, 2025. 4
- [121] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” in COLM, 2024. 4
- [122] D. Kim, S. Woo, J.-Y. Lee, and I. S. Kweon, “Video panoptic segmentation,” in CVPR, 2020. 4
- [123] J. Miao, X. Wang, Y. Wu, W. Li, X. Zhang, Y. Wei, and Y. Yang, “Large-scale video panoptic segmentation in the wild: A benchmark,” in CVPR, 2022. 4
- [124] S. Woo, D. Kim, J.-Y. Lee, and I. S. Kweon, “Learning to associate every segment for video panoptic segmentation,” in CVPR, 2021. 4
- [125] S. Qiao, Y. Zhu, H. Adam, A. Yuille, and L.-C. Chen, “Vip-deeplab: Learning visual perception with depth-aware video panoptic segmentation,” in CVPR, 2021. 4
- [126] Y. Xiong, R. Liao, H. Zhao, R. Hu, M. Bai, E. Yumer, and R. Urtasun, “Upsnet: A unified panoptic segmentation network,” in CVPR, 2019. 4
- [127] X. Li, H. Yuan, W. Li, H. Ding, S. Wu, W. Zhang, Y. Li, K. Chen, and C. C. Loy, “OMG-Seg: Is one model good enough for all segmentation?” in CVPR, 2024. 4
- [128] A. Yilmaz, O. Javed, and M. Shah, “Object tracking: A survey,” ACM Comput. Surv., 2006. 4
- [129] Z. Chen, B. Zhong, G. Li, S. Zhang, and R. Ji, “Siamese box adaptive network for visual tracking,” in CVPR, 2020. 4
- [130] Z. Zhang and H. Peng, “Deeper and wider siamese networks for realtime visual tracking,” in CVPR, 2019. 4
- [131] D. Guo, J. Wang, Y. Cui, Z. Wang, and S. Chen, “Siamcar: Siamese fully convolutional classification and regression for visual tracking,” in CVPR, 2020. 4
- [132] Y. Cui, C. Jiang, L. Wang, and G. Wu, “Mixformer: End-to-end tracking with iterative mixed attention,” in CVPR, 2022. 4
- [133] B. Yan, H. Peng, J. Fu, D. Wang, and H. Lu, “Learning spatio-temporal transformer for visual tracking,” in ICCV, 2021. 4
- [134] X. Chen, B. Yan, J. Zhu, D. Wang, X. Yang, and H. Lu, “Transformer tracking,” in CVPR, 2021. 4
- [135] L. Lin, H. Fan, Z. Zhang, Y. Xu, and H. Ling, “Swintrack: A simple and strong baseline for transformer tracking,” in NeurIPS, 2022. 4
- [136] D. Guo, Y. Shao, Y. Cui, Z. Wang, L. Zhang, and C. Shen, “Graph attention tracking,” in CVPR, 2021. 4
- [137] M. Kristan, A. Leonardis, J. Matas, M. Felsberg, R. Pflugfelder, J.-K. Kämäräinen, H. J. Chang, M. Danelljan, L. Č. Zajc, A. Lukežič et al., “The tenth visual object tracking vot2022 challenge results,” in ECCV,

2022. 4, 5, 6

free video instance segmentation,” in CVPR, 2023. 4

- [169] J. Zhang, Y. Cui, G. Wu, and L. Wang, “Jointformer: A unified framework with joint modeling for video object segmentation,” IEEE TPAMI, 2025. 9
- [170] M. Li, S. Li, X. Zhang, and L. Zhang, “Univs: Unified and universal video segmentation with prompts as queries,” in CVPR, 2024. 9
- [171] S. Vujasinović, S. Bullinger, S. Becker, N. Scherer-Negenborn, M. Arens, and R. Stiefelhagen, “Revisiting click-based interactive video object segmentation,” in ICIP, 2022. 10
- [172] S. Caelles, J. Pont-Tuset, F. Perazzi, A. Montes, K.-K. Maninis, and L. Van Gool, “The 2019 davis challenge on vos: Unsupervised multiobject segmentation,” arXiv:1905.00737, 2019. 10
- [173] X. Wang, R. Zhang, C. Shen, T. Kong, and L. Li, “Solo: A simple framework for instance segmentation,” IEEE TPAMI, 2021. 13

- [138] H. Ding, X. Jiang, B. Shuai, A. Q. Liu, and G. Wang, “Semantic segmentation with context encoding and multi-path decoding,” IEEE TIP, vol. 29, 2020. 4
- [139] J. Lazarow, K. Lee, K. Shi, and Z. Tu, “Learning instance occlusion for panoptic segmentation,” in CVPR, 2020. 4
- [140] X. Zhan, X. Pan, B. Dai, Z. Liu, D. Lin, and C. C. Loy, “Self-supervised scene de-occlusion,” in CVPR, 2020. 4
- [141] A. Kortylewski, Q. Liu, A. Wang, Y. Sun, and A. Yuille, “Compositional convolutional neural networks: A robust and interpretable model for object recognition under occlusion,” IJCV, vol. 129, no. 3, 2021. 4
- [142] H. Zhang and H. Ding, “Prototypical matching and open set rejection for zero-shot semantic segmentation,” in ICCV, 2021. 4
- [143] X. Wang, T. Xiao, Y. Jiang, S. Shao, J. Sun, and C. Shen, “Repulsion loss: Detecting pedestrians in a crowd,” in CVPR, 2018. 4
- [144] S. Zhang, L. Wen, X. Bian, Z. Lei, and S. Z. Li, “Occlusion-aware r-cnn: Detecting pedestrians in a crowd,” in ECCV, 2018. 4
- [145] M.-J. Chiou, H. Ding, H. Yan, C. Wang, R. Zimmermann, and J. Feng, “Recovering the unbiased scene graphs from the biased ones,” in ACM MM, 2021. 4
- [146] X. Li, H. Ding, W. Zhang, H. Yuan, J. Pang, G. Cheng, K. Chen, Z. Liu, and C. C. Loy, “Transformer-based visual segmentation: A survey,” IEEE TPAMI, 2024. 4
- [147] J. Wu, X. Li, S. Xu, H. Yuan, H. Ding, Y. Yang, X. Li, J. Zhang, Y. Tong, X. Jiang, B. Ghanem, and D. Tao, “Towards open vocabulary learning: A survey,” IEEE TPAMI, 2024. 4
- [148] B. Miao, M. Bennamoun, Y. Gao, and A. Mian, “Region aware video object segmentation with deep motion modeling,” arXiv preprint arXiv:2207.10258, 2022. 4
- [149] G. Zhan, W. Xie, and A. Zisserman, “A tri-layer plugin to improve occluded detection,” in BMVC, 2022. 4
- [150] L. Ke, Y.-W. Tai, and C.-K. Tang, “Deep occlusion-aware instance segmentation with overlapping bilayers,” in CVPR, 2021. 4
- [151] Q. Chu, W. Ouyang, H. Li, X. Wang, B. Liu, and N. Yu, “Online multi-object tracking using cnn-based single object tracker with spatialtemporal attention mechanism,” in ICCV, 2017. 4
- [152] J. Zhu, H. Yang, N. Liu, M. Kim, W. Zhang, and M.-H. Yang, “Online multi-object tracking with dual matching attention networks,” in ECCV,

2018. 4

- [153] J. Xu, Y. Cao, Z. Zhang, and H. Hu, “Spatial-temporal relation networks for multi-object tracking,” in ICCV, 2019. 4
- [154] Q. Liu, Q. Chu, B. Liu, and N. Yu, “Gsm: Graph similarity model for multi-object tracking.” in IJCAI, 2020. 4
- [155] S. Li, M. Danelljan, H. Ding, T. E. Huang, and F. Yu, “Tracking every thing in the wild,” in ECCV, 2022. 4
- [156] F. Li, T. Kim, A. Humayun, D. Tsai, and J. M. Rehg, “Video segmentation by tracking many figure-ground segments,” in ICCV, 2013. 6
- [157] S. D. Jain and K. Grauman, “Supervoxel-consistent foreground propagation in video,” in ECCV, 2014. 6
- [158] P. Ochs, J. Malik, and T. Brox, “Segmentation of moving objects by long term video analysis,” IEEE TPAMI, vol. 36, no. 6, 2014. 6
- [159] M. Kristan, J. Matas, M. Danelljan, M. Felsberg, H. J. Chang, L. Č. Zajc, A. Lukežič, O. Drbohlav, Z. Zhang, K.-T. Tran et al., “The first visual object tracking segmentation vots2023 challenge results,” in ICCV Workshop, 2023. 6
- [160] P. Tokmakov, J. Li, and A. Gaidon, “Breaking the “object” in video object segmentation,” in CVPR, 2023. 6
- [161] Q. Jiang, F. Li, Z. Zeng, T. Ren, S. Liu, and L. Zhang, “T-rex2: Towards generic object detection via text-visual prompt synergy,” in ECCV,

2024. 6

- [162] K. Chen, D. Ramanan, and T. Khurana, “Using diffusion priors for video amodal segmentation,” in CVPR, 2025. 6
- [163] S. Bai, K. Chen, X. Liu, J. Wang, and et al, “Qwen2.5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025. 6
- [164] D. R. Martin, C. C. Fowlkes, and J. Malik, “Learning to detect natural image boundaries using local brightness, color, and texture cues,” IEEE TPAMI, vol. 26, no. 5, 2004. 7
- [165] Z. Yang, Y. Wei, and Y. Yang, “Associating objects with transformers for video object segmentation,” in NeurIPS, 2021. 9
- [166] H. K. Cheng, Y.-W. Tai, and C.-K. Tang, “Rethinking space-time networks with improved memory coverage for efficient video object segmentation,” in NeurIPS, 2021. 9, 10
- [167] M. Li, L. Hu, Z. Xiong, B. Zhang, P. Pan, and D. Liu, “Recurrent dynamic embedding for video object segmentation,” in CVPR, 2022. 9
- [168] Z. Yang and Y. Yang, “Decoupling features in hierarchical propagation for video object segmentation,” in NeurIPS, 2022. 9

