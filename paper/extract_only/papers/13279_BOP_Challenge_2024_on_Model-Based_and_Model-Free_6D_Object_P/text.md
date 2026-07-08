## BOP Challenge 2024 on Model-Based and Model-Free 6D Object Pose Estimation

# arXiv:2504.02812v4[cs.CV]23Apr2025

Van Nguyen Nguyen1 Stephen Tyree2 Andrew Guo3 M´ed´eric Fourmy4 Anas Gouda5 Taeyeop Lee6 Sungphill Moon7 Hyeontae Son7 Lukas Ranftl8,9 Jonathan Tremblay2 Eric Brachmann10 Bertram Drost8 Vincent Lepetit1 Carsten Rother11 Stan Birchfield2 Jiri Matas4 Yann Labb´e13 Martin Sundermeyer12 Tomas Hodan13

1ENPC 2NVIDIA 3University of Toronto 4CTU Prague 5TU Dortmund 6KAIST 7NAVER LABS 8MVTec 9TU Munich 10Niantic 11Heidelberg University 12Google 13Meta

[Figure 1]

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

[Figure 14]

HOT3D [1] HOPEv2 [45] HANDAL [12] Static/dynamic onboarding

Fig. 1. New BOP-H3 datasets with object onboarding sequences for model-free tasks. The first three columns show sample images from the new datasets, with the contour of 3D object models in the ground-truth poses drawn in green. The fourth column shows a static (top) and dynamic (bottom) onboarding sequences, which are available in BOP-H3 and used for learning objects in the newly introduced model-free tasks.

### Abstract

We present the evaluation methodology, datasets and results of the BOP Challenge 2024, the sixth in a series of public competitions organized to capture the state of the art in 6D object pose estimation and related tasks. In 2024, our goal was to transition BOP from lab-like setups to real-world scenarios. First, we introduced new model-free tasks, where no 3D object models are available and methods need to onboard objects just from provided reference videos. Second, we defined a new, more practical 6D object detection task where identities of objects visible in a test image are not provided as input (unlike in the classical 6D localization). Third, we introduced new BOP-H3 datasets recorded with high-resolution sensors and AR/VR headsets, closely resembling real-world scenarios. BOP-H3 include 3D models and onboarding videos to support both model-based and model-free tasks. Participants competed on seven challenge tracks, each defined by a task (6D localization, 6D detection, 2D detection), object

onboarding setup (model-based, model-free), and dataset group (BOP-Classic-Core, BOP-H3). Notably, the best 2024 method for model-based 6D localization of unseen objects (FreeZeV2.1) achieves 22% higher accuracy on BOP-Classic-Core than the best 2023 method (GenFlow), and is only 4% behind the best 2023 method for seen objects (GPose2023) although being significantly slower (24.9 vs 2.7s per image). A more practical 2024 method for this task is Co-op which takes only 0.8s per image and is 13% more accurate than GenFlow. Methods have similar rankings on 6D detection as on 6D localization but (as expected) higher run time. On model-based 2D detection of unseen objects, the best 2024 method (MUSE) achieves 21–29% relative improvement compared to the best 2023 method (CNOS). However, the 2D detection accuracy for unseen objects is still -35% behind the accuracy for seen objects (GDet2023), and the 2D detection stage is consequently the main bottleneck of existing pipelines for 6D localization/detection of unseen objects. The online evaluation system stays open and is available at: bop.felk.cvut.cz.

### 1. Introduction

2017–2023 summary. To measure the progress in 6D object pose estimation and related tasks, we created BOP (Benchmark Object Pose) in 2017 and have been organizing challenges on the benchmark datasets since then. Results of challenges from 2017, 2019, 2020, 2022, and 2023 are published in [17,19–21,43]. The field has come a long way, with the accuracy in model-based 6D localization of seen objects (target objects are seen during training) improving by more than 50% (from 56.9 to 85.6 AR). In

- 2023, as the accuracy in this classical task had been saturating, we introduced a more practical yet more challenging task of modelbased 6D localization of unseen objects, where new objects need to be onboarded just from their 3D models in under 5 minutes on a single GPU (the limit was chosen to be sufficient for operations like template rendering, but preventing exhaustive training and therefore ensuring relevance for practical applications). In addition to model-based 6D object localization, we have been evaluating model-based 2D object detection and 2D object segmentation.

New model-free setup. While the model-based tasks are relevant for warehouse or factory settings where 3D models of target objects are typically available, their applicability is limited in openworld scenarios. In 2024, we bridged this gap by introducing new model-free tasks, where 3D models are not available and methods need to learn new objects on the fly from onboarding videos (Sec. 2.3 and 3.2). Methods that can operate in such a model-free setup will minimize the onboarding burden of new objects and unlock new types of applications, including augmented-reality systems capable of prompt object indexing and re-identification.

New BOP-H3 datasets. To enable the model-free tasks and their comparison with the model-based variants, we introduced three new datasets referred jointly as BOP-H3: HOT3D [1], HOPEv2 [45], and HANDAL [12]. These datasets include texture-mapped 3D models and onboarding videos for 93 objects. To simulate different real-world scenarios, the datasets include two types of onboarding videos: static onboarding where the object is static and the camera is moving around and capturing the object from different viewpoints, and dynamic onboarding where the object is manipulated by hands and the camera is either static (on a tripod) or dynamic (on a head-mounted device). While methods were allowed to use all frames of the onboarding videos in 2024, we are planning to gradually limit the number of used frames to increase the practicality of the problem setup. See

- Fig. 1 for sample images from BOP-H3 and Sec. 2.2 for details.

New 6D object detection task. In 2024, we also revisited the evaluation of object pose estimation. Since the beginning of BOP, we distinguish two object pose estimation tasks: 6D object localization, where identifiers of present object instances are provided for each test image, and 6D object detection, where no prior information is provided (see appendix A.1 in [20] for a detailed comparison of these tasks). Up until 2024, we had been evaluating methods for object pose estimation only on 6D

object localization because (1) pose accuracy on this simpler task had not been saturated, and (2) evaluating this task requires only calculating the recall rate which is noticeably less expensive than calculating the precision/recall curve required for evaluating 6D detection. While still supporting the 6D localization task, in 2024 we started evaluating also the 6D detection task. This was possible thanks to new GPUs that we secured for the BOP evaluation server, run-time improvements of the evaluation scripts, and a simpler evaluation methodology – only MSSD and MSPD pose-error functions are calculated when evaluating 6D detection, not VSD (Sec. 3.4). The VSD pose error function is more expensive to calculate and requires depth images which are not available in HOT3D and HANDAL. Besides speeding up the evaluation, omitting VSD therefore enables evaluating on RGB-only datasets.

Summary of 2024 results. Participants competed on seven challenge tracks, with each track defined by a task, object onboarding setup, and a group of datasets. Three of the tracks were on BOP-Classic-Core datasets and focused on model-based 6D localization, 6D detection, and 2D detection of unseen objects. The other four tracks were on BOP-H3 datasets and focused on model-based and model-free 6D detection and model-based and model-free 2D detection of unseen objects. In all tracks, methods had to onboard a new object within 5 minutes on a single GPU, using provided 3D models in the model-based tasks and provided onboarding video sequences in the model-free tasks.

As detailed in Sec. 4, the best 2024 method for model-based 6D localization of unseen objects (FreeZeV2.1 [3]) achieves 22% higher accuracy on BOP-Classic-Core than the best 2023 method (GenFlow [32]; 82.1 vs 67.4AR). FreeZeV2.1 is only 4% behind the best 2023 method for seen objects (GPose2023 [49]; 82.1 vs 85.6AR), although being significantly slower (24.9 vs 2.7s for estimating poses of all objects in an image on average). A more practical 2024 method for this task is Co-op [33], which takes only 0.8s per image and achieves a decent accuracy of 75.9AR (13% higher than GenFlow). Many methods for model-based 6D object pose estimation were evaluated on both the 6D object localization and the new 6D object detection task. Rankings on the two tasks are similar, with the main (expected) difference being a higher run time on the 6D object detection task.

On model-based 2D detection of unseen objects, the best 2024 method (MUSE) achieves 21% relative improvement compared to the best 2023 method for this task (CNOS [35]; 52.0 vs 42.8AP). However, the 2D detection accuracy for unseen objects is still noticeably behind the accuracy for seen objects (GDet2023 [49] achieves 79.8AP). The 2D detection stage is consequently the primary bottleneck of recent pipelines for 6D detection/localization of unseen objects (all first detect target objects in 2D and then estimate the 6D pose per detection).

Participation in challenge tracks on the new BOP-H3 datasets and model-free tasks has been limited, supposedly due to the limited time and the non-negligible effort required to adopt the new datasets and develop new methods. However, the evaluation system staysopenandwehopetosee moresubmissions in the future.

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

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33

HOT3D objects (33 objects)

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28

HOPEv2 objects (28 objects)

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

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

- 1 2 3 4 5 6 7 8 9 10 11 12 13 141516 17 18 19 20 21 22 23 24 25 2627 28 29 30 31 32 33 34 353637 38 39 40 HANDAL objects (40 objects)

Fig. 2. Objects from the BOP-H3 datasets introduced in 2024. Objects are shown at the same scale. HOT3D and HOPEv2 share eight objects.

- 2. Datasets

Train. im. Val im. Test im. Test inst. Dataset Obj. Real PBR Real All Used All Used BOP-H3:

In 2024, methods were evaluated on the BOP-Classic datasets (Sec. 2.1), which were used in all previous challenges, and the newly introduced BOP-H3 datasets (Sec. 2.2). Each dataset includes 3D object models and images of the objects annotated with amodal 2D bounding boxes, modal 2D segmentation masks, and

HOT3D [1] 33 420600 50K – 154200 5140 709715 23642 HOPEv2 [45] 28 – 50K 50 457 457 9276 9276 HANDAL [12] 40 – 50K 2208 13261 1684 74771 9492

BOP-Classic-Core:

LM-O [2] 8 – 50K – 1214 200 9038 1445 T-LESS [18] 30 37584 50K – 10080 1000 67308 6423 ITODD [11] 28 – 50K 54 721 721 3041 3041 HB [22] 33 – 50K 4420 13000 300 67542 1630 YCB-V [48] 21 113198 50K – 20738 900 98547 4123 TUD-L [19] 3 38288 50K – 23914 600 23914 600 IC-BIN [9] 2 – 50K – 177 150 2176 1786

- 6D object poses. The object models are provided in the form of 3D meshes (in most cases with a color texture) which were created manually or 3D reconstructed [34]. Depending on the dataset, the images are RGB-D, RGB, or monochrome. All datasets include real test images showing the objects in scenes with various complexity, often with clutter and occlusion. Most datasets include training images which may be real and/or synthetic, with all BOPClassic-Core and BOP-H3 datasets offering 50K photorealistic physically-based rendered (PBR) training images generated and automatically annotated with BlenderProc [6–8]. Some datasets also include real validation images. Ground-truth annotations are publicly available only for training and validation images, and also for test images from BOP-Classic datasets that do not have validation images. Private ground-truth annotations are only accessible by the BOP evaluation server. Tab. 1 shows the dataset statistics.

BOP-Classic-Extra:

LM [15] 15 – 50K – 18273 3000 18273 3000 RU-APC [41] 14 – – – 5964 1380 5964 1380 IC-MI [44] 6 – – – 2067 300 5318 800 TYO-L [19] 21 – – – 1670 1670 1670 1670 HOPEv1 [45] 28 – 50K 50 188 188 3472 2898

Tab. 1. Parameters of BOP datasets. Column Test inst./All shows the number of annotated object instances for which at least 10% of the projected surface area is visible in test images. Columns Used show the number of used test images and object instances. All datasets include 3D object models. Only BOP-H3 include videos for object onboarding.

#### 2.1. BOP-Classic datasets

BOP-Classic is a group of twelve traditional 6D object pose estimation datasets. As in previous years, authors were required to evaluate their methods on at least seven of the datasets, called BOP-Classic-Core, to be considered for the challenge awards. The seven datasets include test images showing 272K instances (19K used for the evaluation) of 132 objects (details in Sec. 7.2 of [16]).

#### 2.2. BOP-H3 datasets

BOP-H3 is a group of three new datasets (HOT3D, HOPEv2, HANDAL) that enable the evaluation of both model-based and model-free methods by providing 3D models (Fig. 2) and onboarding videos for all objects. These three datasets include test images showing 794K instances (42K used for the evaluation, which is

- 2 times more than in BOP-Classic-Core) of 93 unique objects.

HOT3D [1] is a dataset for egocentric hand and object tracking in 3D with multi-view RGB and monochrome image streams showing participants interacting with 33 diverse rigid objects. The dataset is recorded with two recent head-mounted devices from Meta: Aria, which is a research prototype of light-weight AI glasses, andQuest3, aproduction VR headset that has been sold in millions of units. HOT3D also includes PBR materials for the 3D object models, real training images, 3D hand pose and shape annotations, and eye-tracking signal in recordings from Aria. In BOP, we use HOT3D-Clips, which is a curated subset of HOT3D. Each clip has 150 frames (5 seconds) that are all annotated with groundtruth poses of all modeled objects and hands. There are 4117 clips in total (2969 training, 1148 test). HOT3D-Clips are also used in the Multiview Egocentric Hand Tracking Challenge [14].

HOPEv2 [45] is a dataset for robotic manipulation featuring 28 toy grocery objects available from online retailers for about 50 USD. The original HOPEv1 dataset includes images of 50 static object arrangements in 10 household/office environments, with each arrangement captured under up to 5 lighting variations from multiple viewpoints. For the 2024 challenge, we released an updated version with additional test images showing 7 static object arrangements from multiple viewpoints. HOPEv2 is the only BOP-H3 dataset that provides images with the depth channel.

HANDAL [12] is a dataset with graspable or manipulable objects (hammers, ladles, measuring cups, power drills, spatulas, strainers, whisks). The objects are arranged in indoor and outdoor scenes and captured from multiple viewpoints. The original dataset has 212 objects from 17 categories. For the 2024 challenge, we captured additional test images and only consider 40 objects from

- 7 categories, with high-quality 3D models created by 3D artists.

#### 2.3. Object onboarding videos

In the new model-free tasks, 3D object models are not available and methods need to learn new objects from onboarding (reference) videos which are available in the BOP-H3 datasets. As shown in Fig. 3, we define two types of video-based onboarding:

Static onboarding: The object is static and the camera is moving around the object, capturing all possible object views. Two videos are available, with the object standing upright in one and upsidedown in the other. Such videos are useful for 3D object reconstruction by methods such as NeRF [31] or Gaussian Splatting [23]. Object poses are available for all frames since these poses could be relatively easily obtained with tools like COLMAP [42].

Dynamic onboarding: The object is manipulated by hands and the camera is either static (on a tripod) or dynamic (on a head-mounted device). This type of onboarding videos is useful for 3D object reconstruction methods such as BundleSDF [46] or Hampali et al. [13]. Ground-truth object poses are available only for the first frame to simulate a real-world setup (at least one ground-truth pose needs to be provided to define the object coordinate system necessary for the evaluation of object pose estimates). The dynamic onboarding setup is more challenging but more natural for AR/VR applications than the static setup.

In HOT3D, all onboarding videos are RGB (Aria) or monochrome (Quest 3). In HOPEv2 and HANDAL, static onboarding videos are RGB and dynamic onboarding videos are RGB-D.

#### 2.4. Synthetic training dataset

As in 2023, we provided over 2M synthetic training images showing 50K+ diverse objects from the GSO [10] and ShapeNetCore [4] datasets. These objects are not included in BOP-Classic nor BOP-H3 and the dataset can be therefore used for training methods for tasks on unseen objects. The images were originally synthesized for MegaPose [25] using BlenderProc [6–8].

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Fig. 3. Sample onboarding videos from HOPEv2 [45]. First two rows show sample frames from static onboarding videos, one with the object standing upright and one with the object standing upside-down. The third row shows sample frames from a dynamic onboarding video where the object is manipulated by hands. Ground-truth poses (shown with green contour) are provided for all frames of static but only the first frame of dynamic onboarding videos (see Sec. 2.3 for details).

### 3. Challenge tracks

In 2024, participants competed on seven challenge tracks on unseen objects (target objects are not seen during training), with each track defined by a task (6D localization, 6D detection, 2D detection), type of object onboarding (model-based, model-free), and dataset group (BOP-Classic-Core, BOP-H3):

- • Track 1: Model-based 6D localization on BOP-Classic-Core
- • Track 2: Model-based 6D detection on BOP-Classic-Core
- • Track 3: Model-based 2D detection on BOP-Classic-Core
- • Track 4: Model-based 6D detection on BOP-H3
- • Track 5: Model-based 2D detection on BOP-H3
- • Track 6: Model-free 6D detection on BOP-H3
- • Track 7: Model-free 2D detection on BOP-H3

We primarily focused on the more practical detection tasks (Track 2-7) while keeping the classical model-based 6D localization on BOP-Classic-Core to enable direct comparison with previous years (Track 1).

On all tracks, methods go through three phases: (1) training, where methods can be trained on non-target objects, (2) object onboarding, where methods have to quickly onboard target objects, and (3) inference, where methods are expected to produce predictions about target objects seen in test images. The following sections describe these phases in more detail.

#### 3.1. Training phase

At training time, methods can be trained on the provided dataset of 2M synthetic images, originally synthesized using BlenderProc [6,7] for MegaPose [25]. The images show training (non-target) objects annotated with ground-truth 6D poses, modal 2D segmentation masks (covering the visible object parts), and amodal 2D bounding boxes (covering the whole object silhouette, including the occluded parts). The dataset also includes color

- 3D mesh models of the objects. Methods are also free to use other datasets as long as the images did not show target objects.

#### 3.2. Object onboarding phase

After training, methods can spend up to 5 minutes of the wall-clock time on a single computer with one GPU to onboard a target object.1 The generated representation of onboarded objects (object templates, neural radiance fields, etc.) need to be fixed after onboarding (cannot be updated based on test images).

Model-based onboarding: Methods are provided 3D models of target objects that are not seen at training. To onboard an object, methods can, e.g., fine-tune a neural network or render images of the 3D models, but cannot use any real images of the objects.

Model-free onboarding: Methods are provided video(s) of target objects that are not seen at training. 3D models of the objects are not available. Methods can use either static or dynamic onboarding videos (Fig. 3, Sec. 2.2), e.g., to reconstruct a 3D model, render novel views, or fine-tune a neural network.

#### 3.3. Inference phase

Input: Methods are given a real-world test image showing an arbitrary number of instances of an arbitrary number of target objects, with all objects being from one specified dataset (e.g., YCB-V). Depending on the dataset, the test image may be RGB-D, RGB, or monochrome. In case of 6D localization, methods are additionally provided identities of objects visible in the test image in the form of a list L=[(o1,n1), ..., (om,nm)], where ni is the number of instances of object oi for which at least 10% of the projected surface area is visible in the test image. No prior information about the visible object instances is provided in the detection tasks.

As most recent methods for 6D object localization and detection start with a 2D detection stage, participants in the 6D localization and detection tasks are also encouraged to evaluate their methods using default 2D detections/segmentations produced by CNOS [35]. Starting from the same 2D detections enables direct comparison of the object pose estimation stages.

Output: For each test image, methods produce predictions with confidence scores and run time.2 Predictions are in the form of amodal 2D bounding boxes for 2D detection and in the form of 6D object poses for 6D localization and detection. A 6D object pose is defined by a matrix P=[R|t], where R is a 3D rotation matrix, and t is a 3D translation vector from the 3D object model space to the 3D camera space. In all tasks, methods need to produce predictions only for object instances for which at least 10% of the projected surface area is visible in the given test image.

- 1The time is measured from the point right after the raw data (a 3D mesh model in the model-based setup and reference video(s) in the model-free setup) is loaded to the point when the object is onboarded.
- 2Although the run time is measured on user machines, which may be different for each method, it still provides a general sense of how fast each method is. The time is measured from the point right after the raw data (the image, 3D models, etc.) is loaded to the point when predictions for all objects in the image are available (including the time needed to generate default 2D detections if used).

#### 3.4. Evaluation methodology

Measuring error of 6D object poses. The error of an estimated pose w.r.t. the ground-truth pose can be calculated by the following pose-error functions (see Sec. 2.2 of [20] for details): (1) MSSD (Maximum Symmetry-Aware Surface Distance) which considers a set of pre-identified global object symmetries and measures the surface deviation in 3D, (2) MSPD (Maximum Symmetry-Aware Projection Distance) which considers the object symmetries and measures the perceivable deviation, (3) VSD (Visible Surface Discrepancy) which treats indistinguishable poses as equivalent by considering only the visible object part. An estimated pose is considered correct w.r.t. a pose-error function e, if e<θe, where e∈{MSSD,MSPD,VSD} and θe is a threshold of correctness. Evaluating 6D object localization. The fraction of object instances for which a correct pose is estimated is called Recall. The Average Recall w.r.t. a pose-error function e on a dataset d, denoted as ARe,d, is defined as the average of the Recall rates calculated for multiple settings of θe and, in the case of VSD, also for multiple settings of the misalignment tolerance τ. The accuracy on a dataset d is measured by: ARd=(ARVSD,d+ARMSSD,d+ ARMSPD,d)/3, and the overall accuracy, denoted as AR, is defined as the average of ARd scores over selected datasets.3

Evaluating 2D object detection. The detection accuracy is measured by the Average Precision (AP, also known as mAP), following the evaluation methodology from the COCO 2020 Object Detection Challenge [27]. Specifically, a per-object APo score is calculated by averaging the precision values at multiple thresholds on the Intersection over Union (IoU) of 2D bounding boxes: [0.5,0.55,0.6,...,0.95]. The accuracy of a method on a dataset d is measured by APd calculated by averaging per-object APo,d scores over objects from the dataset. The overall accuracy, denoted as AP, is defined as the average of the per-dataset APd scores.4

Evaluating 6D object detection. The detection accuracy is measured by the Average Precision (AP), similarly as for 2D object detection but using pose-error functions instead of the IoU of 2D bounding boxes. Specifically, for each pose-error function e ∈ {MSSD, MSPD}5, the per-object accuracy APe,o is defined by averaging the precision values calculated at multiple thresholds of correctness θe (defined in Sec. 2.4 of [20]). The accuracy APe,d on a dataset d w.r.t. a pose-error function e is calculated by averaging per-object scoresAPe,o over objects from the dataset. The accuracy on the dataset d is then defined as APd= (APMSSD,d+APMSPD,d)/2, and the overall accuracy, denoted as AP, is defined as the average ofAPd scores over selected datasets.4

- 3When calculating AR, scores are not averaged over objects before averaging over datasets, which is done when calculating AP (in 2D/6D detection) to comply with the original COCO evaluation methodology [27].
- 4Up to 100 most confident detections per image are considered. Correct detections for instances visible from less than 10% are not counted as false positives.
- 5The VSD pose-error function is not considered as it is more expensive to calculate (6D detection requires evaluating more pose estimates than 6D localization) and needs depth images which are not available in HOT3D and HANDAL.

### 4. Results and discussion

Participants were submitting results to the online evaluation system from May 29th until November 29th, 2024. We received submissions for all but Track 6, presumably due to the limited time (BOP-H3 datasets were released later) and the extra effort required to prepare model-free methods. In this section we aim to provide a high-level summary of the results and refer the reader to the online system for more per-submission details. Evaluation scripts used by the system are publicly available in the BOP toolkit.

#### 4.1. Experimental setup

A method had to use a fixed set of hyperparameters across all objects and datasets. In the model-based onboarding, a method could render images of the 3D models or use a subset of the BlenderProc images (originally provided for BOP 2020 [20]), assuming that rendering a single BlenderProc image takes 2 seconds. The onboarding phase had to be done within 5 minutes on a single GPU. Note that methods marked as RGB-only could only use RGB/monochrome channels for onboarding and inference. Not a single pixel of test images may have been used for training and onboarding, nor ground-truth annotations that are available for test images of some datasets. Ranges of the azimuth and elevation camera angles, and the range of camera-object distances determined by the ground-truth poses from test images are the only information about the test set that may have been used during training and onboarding. Only subsets of test images were used (see Tab. 1) to remove redundancies and speed up the evaluation.

#### 4.2. Model-based 6D localization of unseen objects

Among the 44 new entries from 2024 for model-based 6D object localization on BOP-Classic-Core (Tab. 2), 20 entries (19 RGB-D, 1 RGB) outperform the best method from 2023, an RGBD variant of GenFlow [32] (#21 in Tab. 2). The best method from

- 2024, FreeZeV2.1 [3], is 22% more accurate (82.1 vs. 67.4AR) and 39% faster than GenFlow. Notably, FreeZeV2.1 is only 3.5 AR behind GPose2023, currently the best method for seen objects.

FreeZeV2.1 is an RGB-D method which, in the winning variant, relies on 2D segmentation masks from three methods: SAM6D [26], NIDS [30], and CNOS [35]. FreeZeV2.1 estimates a 6D pose with a confidence score for each mask and keeps only the best estimates by applying non-maximum suppression. Given a segmentation mask, the pose is estimated in a RANSAC-fitting scheme from 3D-3D correspondences established by matching features (2D visual features from DINOv2 [37] concatenated with 3D geometry features from GeDi [39]) between the query image and the 3D model (the features are extracted from renderings of the model and aggregated on its surface). Pose estimates are further refined by depth-based ICP and symmetry-aware refinement.

Another notable method is FRTPose.v1, ranked 2nd, which quickly trains a ResNet34 network (4.3 min per object on RTX 4090) at onboarding. Poses are estimated from predicted dense 2D-3D correspondences and refined by FoundationPose [47].

A more practical method is Co-op [33] (#9 in Tab. 2), which is

[Figure 128]

[Figure 129]

[Figure 130]

TUD-LHANDALHOT3DLM-OT-LESSIC-BINHBHOPEv2ITODDYCB-V

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

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Fig. 4. Results of the top methods for model-based 6D detection: FreeZeV2.1 [3] on BOP-Classic-Core (first 7 rows), and GigaPose [36] with GenFlow refinement [32] on BOP-H3 (last 3 rows). The methods perform well across diverse datasets, despite not being trained on them. Contours of 3D models in the ground-truth and estimated poses are in green and red respectively. Shown are estimates with confidence ≥0.3.

# Method Awards Year Det./seg. Refinement Train im. ...type Test image LM-O T-LESS TUD-L IC-BIN ITODD HB YCB-V AR Time

- 1 FreeZeV2.1 [3] 2024 Custom ICP - - RGB-D 77.1 75.5 97.6 69.7 74.2 89.2 91.5 82.1 24.9

- 2 FRTPose.v1 (SAM6D-FastSAM) 2024 SAM6D-FastSAM FoundationPose RGB-D PBR RGB-D 77.8 76.6 94.0 70.2 73.7 89.6 91.0 81.8 40.1

- 3 FRTPose.v1 (Default Detections) 2024 CNOS-FastSAM FoundationPose RGB-D PBR RGB-D 77.7 76.3 94.0 70.5 73.5 89.6 91.0 81.8 46.5

- 4 FRTPose.v1 (MUSE) 2024 MUSE FoundationPose RGB-D PBR RGB-D 78.6 76.8 94.2 70.6 71.0 90.3 91.0 81.8 27.6

- 5 FreeZeV2 [3] 2024 Custom ICP - - RGB-D 76.4 70.8 97.2 65.4 67.9 85.9 90.6 79.2 17.2

- 6 FRTPose (SAM6D-FastSAM) 2024 SAM6D-FastSAM FoundationPose RGB-D PBR RGB-D 78.3 71.7 92.5 60.1 64.6 89.6 91.3 78.3 20.7

- 7 FRTPose (Default Detections) 2024 CNOS-FastSAM FoundationPose RGB-D PBR RGB-D 78.3 71.4 92.2 59.0 61.8 89.6 91.3 77.7 23.4

- 8 Co-op (F3DT2D, 5 Hypo) [33] 2024 F3DT2D Co-op RGB-D PBR RGB-D 73.8 69.5 92.9 63.5 62.9 87.8 89.8 77.1 6.9

- 9 Co-op (F3DT2D, 1 Hypo) [33] 2024 F3DT2D Co-op RGB-D PBR RGB-D 73.0 68.0 92.9 62.4 60.0 86.3 88.6 75.9 0.8

- 10 Co-op (CNOS, 5 Hypo) [33] 2024 CNOS-FastSAM Co-op RGB-D PBR RGB-D 73.0 66.4 90.5 59.7 61.3 87.1 88.7 75.2 7.2

- 11 Co-op (CNOS, 1 Hypo) [33] 2024 CNOS-FastSAM Co-op RGB-D PBR RGB-D 71.5 64.6 90.5 57.5 58.2 85.7 87.4 73.6 2.3

- 12 FoundationPose [47] 2024 SAM6D FondationPose RGB-D PBR RGB-D 75.6 64.6 92.3 50.8 58.0 83.5 88.9 73.4 29.3

- 13 FRTPose (SAM6D-FastSAM & top k) 2024 SAM6D-FastSAM FondationPose RGB-D PBR RGB-D 70.3 58.1 87.1 59.9 64.4 80.4 86.9 72.4 0.8

- 14 Co-op (CNOS, Coarse) [33] 2024 CNOS-FastSAM - RGB-D PBR RGB-D 70.0 64.2 87.9 56.4 56.6 84.2 85.3 72.1 1.0

- 15 GZS6D-BP(coarse+refine+teaser) 2024 - Teaserpp RGB-D - RGB-D 67.8 69.4 92.2 55.0 59.7 80.3 77.2 71.7 6.5

- 16 FreeZe (SAM6D) [3] 2024 SAM6D ICP - - RGB-D 71.6 53.1 94.9 54.5 58.6 79.6 84.0 70.9 11.5

- 17 SAM6D [26] 2024 SAM6D-SAM - RGB-D PBR RGB-D 69.9 51.5 90.4 58.8 60.2 77.6 84.5 70.4 4.4

- 18 FreeZe (CNOS) [3] 2024 CNOS-FastSAM ICP - - RGB-D 68.9 52.0 93.6 49.9 56.1 79.0 85.3 69.3 13.5

- 19 GigaPose+GenFlow+kabsch (5 hypoth) [32,36] 2024 CNOS-FastSAM GenFlow RGB-D PBR RGB-D 67.8 55.6 81.1 56.3 57.5 79.1 82.5 68.6 11.1

- 20 Co-op (F3DT2D, 5 Hypo) [33] 2024 F3DT2D Co-op RGB-D PBR RGB 67.5 68.2 76.7 58.9 50.6 85.6 69.7 68.2 3.9

- 21 GenFlow-MultiHypo16 [32] 2023 CNOS-FastSAM GenFlow RGB-D PBR RGB-D 63.5 52.1 86.2 53.4 55.4 77.9 83.3 67.4 34.6

- 22 GenFlow-MultiHypo [32] 2023 CNOS-FastSAM GenFlow RGB PBR RGB-D 62.2 50.9 84.9 52.4 54.4 77.0 81.8 66.2 21.5

- 23 SAM6D-FastSAM [26] 2024 SAM6D-FastSAM - RGB-D PBR RGB-D 66.7 48.5 82.9 51.0 57.2 73.6 83.4 66.2 1.4

- 24 Co-op (CNOS, 5 Hypo) [33] 2024 CNOS-FastSAM Co-op RGB-D PBR RGB 65.5 64.8 72.9 54.4 49.1 85.0 68.9 65.8 4.2

- 25 SAM6D-CNOSfastSAM [26] 2024 CNOS-FastSAM - RGB-D PBR RGB-D 65.1 47.9 82.5 49.7 56.2 73.8 81.5 65.3 1.3

- 26 Co-op (CNOS, 1 Hypo) [33] 2024 CNOS-FastSAM Co-op RGB-D PBR RGB 64.2 63.5 71.7 51.2 47.3 83.2 67.0 64.0 1.7

- 27 Megapose-CNOS+Multih Teaserpp-10 [25] 2023 CNOS-FastSAM Teaserpp RGB PBR RGB-D 62.6 48.7 85.1 46.7 46.8 73.0 76.4 62.8 142.0

- 28 Megapose-CNOS+Multih Teaserpp [25] 2023 CNOS-FastSAM Teaserpp RGB PBR RGB-D 62.0 48.5 84.6 46.2 46.0 72.5 76.4 62.3 116.6

- 29 SAM6D-ZeroPose [26] 2024 SAM6D - RGB-D PBR RGB-D 63.5 43.0 80.2 51.8 48.4 69.1 79.2 62.2 5.5

- 30 SAM6D-CNOSmask [26] 2023 CNOS-FastSAM - RGB-D PBR RGB-D 64.8 48.3 79.4 50.4 35.1 72.7 80.4 61.6 3.9

- 31 PoZe (CNOS) 2023 CNOS-FastSAM ICP RGB-D Custom RGB-D 64.4 49.4 92.4 40.9 51.6 71.2 61.1 61.6 159.4

- 32 GigaPose+GenFlow (5 hypo) [32,36] 2024 CNOS-FastSAM GenFlow RGB-D PBR RGB 63.1 58.2 66.4 49.8 45.3 75.6 65.2 60.5 10.6

- 33 FoundPose+FeatRef+Megapose-5hyp [25,38] 2024 CNOS-FastSAM MegaPose+FeatRef RGB PBR RGB 61.0 57.0 69.3 47.9 40.7 72.3 69.0 59.6 20.5

- 34 OPFormer-Megapose refinement (CNOS) 2024 CNOS-FastSAM MegaPose RGB PBR RGB 59.6 53.4 69.3 47.0 39.2 76.0 67.0 58.8 1.5

- 35 GigaPose (Add) + Megapose (5 hypo) [25,36] 2024 CNOS-FastSAM MegaPose RGB PBR RGB 60.4 57.6 64.8 48.2 39.8 72.4 66.6 58.5 10.8

- 36 Co-op (CNOS, Coarse) [33] 2024 CNOS-FastSAM - RGB-D PBR RGB 59.7 59.2 64.2 45.8 39.1 78.1 62.6 58.4 1.0

- 37 GigaPose+MegaPose (5 Hypo) [25,36] 2024 CNOS-FastSAM MegaPose RGB PBR RGB 59.8 56.5 63.1 47.3 39.7 72.2 66.1 57.8 7.7

- 38 GenFlow-MultiHypo16 [32] 2023 CNOS-FastSAM GenFlow RGB-D PBR RGB 57.2 52.8 68.8 45.8 39.8 74.6 64.2 57.6 40.5

- 39 TF6D (Default, CNOS) + Megapose [25] 2024 CNOS-FastSAM MegaPose RGB PBR RGB 56.0 59.0 66.9 45.7 37.5 70.1 66.5 57.4 4.1

- 40 ZeroPose-Multi-Hypo-Refinement [5,35] 2023 FastSAM+ImBind MegaPose RGB-D PBR RGB-D 53.8 40.0 83.5 39.2 52.1 65.3 65.3 57.0 16.2

- 41 GenFlow-MultiHypo-RGB [32] 2023 CNOS-FastSAM GenFlow RGB-D PBR RGB 56.3 52.3 68.4 45.3 39.5 73.9 63.3 57.0 20.9

- 42 GigaPose+GenFlow (1 hypo) [32,36] 2024 CNOS-FastSAM GenFlow RGB-D PBR RGB 59.5 55.0 60.7 47.8 41.3 72.2 60.8 56.8 2.2

- 43 GenFlow [32] 2024 CNOS-FastSAM GenFlow RGB-D PBR RGB 54.7 51.4 67.0 43.7 38.4 73.0 61.9 55.7 10.6

- 44 FoundPose+FeatRef+Megapose [25,38] 2024 CNOS-FastSAM MegaPose+FeatRef RGB PBR RGB 55.6 51.1 63.3 40.0 35.7 69.7 66.1 55.0 6.4

- 45 Megapose-CNOS fastSAM+Multih-10 [25] 2023 CNOS-FastSAM MegaPose RGB PBR RGB 56.0 50.8 68.7 41.9 34.6 70.6 62.0 54.9 53.9

- 46 FoundPose+MegaPose [25,38] 2024 CNOS-FastSAM MegaPose RGB PBR RGB 55.4 51.0 63.3 43.0 34.6 69.5 66.1 54.7 4.4

- 47 GigaPose+MegaPose [25,36] 2024 CNOS-FastSAM MegaPose RGB PBR RGB 55.7 54.1 58.0 45.0 37.6 69.3 63.2 54.7 2.3

- 48 Megapose-CNOS fastSAM+Multih [25] 2023 CNOS-FastSAM MegaPose RGB PBR RGB 56.0 50.7 68.4 41.4 33.8 70.4 62.1 54.7 47.4

- 49 ZeroPose-Multi-Hypo-Refinement [5] 2023 FastSAM+ImBind MegaPose RGB-D PBR+Real RGB-D 49.3 34.2 79.0 39.6 46.5 62.9 62.3 53.4 19.0

- 50 MegaPose-CNOS fastSAM [25] 2023 CNOS-FastSAM MegaPose RGB PBR RGB 49.9 47.7 65.3 36.7 31.5 65.4 60.1 50.9 31.7

- 51 OPFormer-Coarse (CNOS) 2024 CNOS-FastSAM - - - RGB 52.5 41.8 61.5 34.2 27.8 67.3 60.6 49.4 0.5

- 52 SMC-1.0s-CNOS 2023 CNOS-FastSAM - - - D 55.8 42.3 59.9 31.6 38.9 58.5 45.4 47.5 6.1

- 53 SMC-0.5s-CNOS 2023 CNOS-FastSAM - - - D 51.2 41.5 51.1 29.0 35.8 53.8 40.3 43.3 3.0

- 54 FoundPose+FeatRef [38] 2024 CNOS-FastSAM FeatRef - - RGB 39.5 39.6 56.7 28.3 26.2 58.5 49.7 42.6 2.6

- 55 TF6D (Default, CNOS) 2024 CNOS-FastSAM - - - RGB 32.3 35.0 47.3 33.2 25.1 53.7 54.1 40.1 1.6

- 56 FoundPose-Coarse [38] 2024 CNOS-FastSAM - - - RGB 39.7 33.8 46.9 23.9 20.4 50.8 45.2 37.3 1.7

- 57 ZeroPose-One-Hypo [5] 2023 FastSAM+ImBind - RGB-D PBR+Real RGB-D 27.2 15.6 53.6 30.7 36.2 46.2 34.1 34.8 9.8

- 58 GigaPose [36] 2024 CNOS-FastSAM - RGB PBR RGB 29.6 26.4 30.0 22.3 17.5 34.1 27.8 26.8 0.4

- 59 GenFlow-coarse [32] 2023 CNOS-FastSAM - RGB-D PBR RGB 25.0 21.5 30.0 16.8 15.4 28.3 27.7 23.5 3.8

- 60 MegaPose-CNOS fastSAM+CoarseBest [25] 2023 CNOS-FastSAM - RGB PBR RGB 22.9 17.7 25.8 15.2 10.8 25.1 28.1 20.8 15.5

###### Tab. 2. Track 1: Model-based 6D localization of unseen objects on BOP-Classic-Core. Methods are ranked by the AR score, which is the average

of per-dataset ARd scores (Sec. 3.4). The last column shows the average time to generate predictions for all objects in a single image, averaged over the datasets (measured on different computers by the participants). Column Year is the year of submission, Det./seg. is the object detection/segmentation method, Refinement is the pose refinement method, Train im. and Test im. show image channels used at training and test time respectively, and Train im. type is the domain of training images. All test images are real. See Sec. 5 for description of the awards.

25 times faster (0.8s per image) and 13% more accurate than GenFlow [32]. Co-op estimates poses with a template-based approach, followed by a flow-based refinement similar to GenFlow.

#### 4.3. Model-based 6D detection of unseen objects

BOP-Classic-Core (Tab. 3). Ranking of methods on Track 2 is consistent with Track 1, suggesting that the 6D detection and 6D localization tasks present similar difficulties and that learnings from previous BOP challenges, focused on the 6D localization task, are also relevant for the newly introduced and more practical 6D detection task. On both tracks, FreeZeV2.1 is the best method, although the improvement of FreeZeV2.1 over Co-op is more significant on Track 2 than on Track 1: 18.8% (#1 and #4 in Tab. 3) vs. only 6.5% (#1 and #8 in Tab. 2). Adding refinement to Co-op tends to decrease the 6D detection score (see coarse poses on #6 in Tab. 3 vs. #7 with single- and #8 with multi-hypotheses refinement). This is a surprising result that is worth investigating.

BOP-H3 (Tab. 5). The best entry is GigaPose [36], a coarse pose estimation method using 2D-2D correspondences established between the input image and pre-rendered templates, followed by the GenFlow [32] refinement. GigaPose and GigaPose+GenFlow achieve 12.3 and 50.4AP on BOP-Classic-Core (#17 and #15 in Tab. 3), while only 9.4 and 31.2AP on BOP-H3, suggesting that BOP-H3 is more difficult. Another method on this track is OPFormer, which follows a similar pipeline as FoundPose [38], but predicts correspondences using a trained transformer head instead of relying on a simple kNN matching of DINOv2 features [37].

#### 4.4. Model-based 2D detection of unseen objects

BOP-Classic-Core (Tab. 4). The best 2024 method, MUSE, follows a similar pipeline as the best 2023 method, CNOS [35]: detection proposals are generated using SAM2 [40] and Grounding DINO [28] and then matched to templates using DINOv2 [37]. The key difference is in the matching stage, where MUSE intro-

# Method Awards Year Det./seg. Refinement Train im. ...type Test image LM-O T-LESS TUD-L IC-BIN ITODD HB YCB-V AP Time

- 1 FreeZeV2.1 [3] 2024 Custom ICP - - RGB-D 79.7 75.1 99.1 69.6 76.9 85.3 90.5 82.3 37.3

- 2 FreeZeV2 (SAM6D) [3] 2024 SAM6D-FastSAM ICP - - RGB-D 77.5 61.0 97.5 62.0 61.7 78.2 86.9 75.0 55.4

- 3 FreeZeV2 (SAM6D, Coarse-to-Fine) [3] 2024 SAM6D-FastSAM ICP - - RGB-D 74.3 60.1 90.2 53.1 57.3 74.1 85.8 70.7 12.9

- 4 Co-op (F3DT2D, Coarse, RGBD) [33] 2024 F3DT2D - RGB-D PBR RGB-D 69.8 62.0 84.1 56.4 57.6 74.6 80.8 69.3 0.9

- 5 Co-op (F3DT2D, 5 Hypo, RGBD) [33] 2024 F3DT2D Co-op RGB-D PBR RGB-D 70.1 61.3 76.6 42.6 62.7 73.4 81.2 66.9 12.2

- 6 Co-op (CNOS, Coarse, RGBD) [33] 2024 CNOS-FastSAM - RGB-D PBR RGB-D 68.3 59.6 80.8 46.9 56.0 74.3 78.2 66.3 2.2

- 7 Co-op (CNOS, 1 Hypo, RGBD) [33] 2024 CNOS-FastSAM Co-op RGB-D PBR RGB-D 67.0 58.3 76.9 45.8 57.5 73.7 76.6 65.1 6.9

- 8 Co-op (CNOS, 5 Hypo, RGBD) [33] 2024 CNOS-FastSAM Co-op RGB-D PBR RGB-D 68.2 58.9 73.8 39.9 60.4 73.1 76.6 64.4 14.3

- 9 Co-op (F3DT2D, 5 Hypo) [33] 2024 F3DT2D Co-op RGB-D PBR RGB 63.7 61.6 65.8 46.7 50.4 73.2 62.6 60.6 8.7

- 10 Co-op (CNOS, 1 Hypo) [33] 2024 CNOS-FastSAM Co-op RGB-D PBR RGB 61.5 58.8 64.1 40.9 46.5 72.7 59.2 57.7 6.4

- 11 GigaPose+GenFlow (5 hypothesis) [32,36] 2024 CNOS-FastSAM GenFlow RGB-D PBR RGB 59.7 56.5 68.8 43.9 42.5 70.7 60.7 57.5 15.5

- 12 Co-op (CNOS, 5 Hypo) [33] 2024 CNOS-FastSAM Co-op RGB-D PBR RGB 61.2 59.0 61.9 39.1 48.3 72.3 59.1 57.3 11.5

- 13 GigaPose+GenFlow (RGBD) [32,36] 2024 CNOS-FastSAM GenFlow RGB-D PBR RGB-D 57.8 46.7 71.5 40.2 44.7 67.6 68.2 56.7 4.5

- 14 Co-op (CNOS, Coarse) [33] 2024 CNOS-FastSAM - RGB-D PBR RGB 58.9 55.8 64.0 40.3 38.8 71.1 57.6 55.2 2.2

- 15 GigaPose+GenFlow [32,36] 2024 CNOS-FastSAM GenFlow RGB-D PBR RGB 55.4 43.6 60.8 35.3 36.7 66.8 54.6 50.4 4.7

- 16 GigaPose+GenFlow+kabsch (5 hypothesis) [32,36] 2024 CNOS-FastSAM GenFlow RGB-D PBR RGB-D 23.7 52.6 24.6 29.2 52.5 54.7 52.4 41.4 16.6

- 17 GigaPose-CVPR24 [36] 2024 CNOS-FastSAM - RGB PBR RGB 6.2 25.5 3.5 5.7 14.9 18.2 12.5 12.3 0.7

- Tab. 3. Track 2: Model-based 6D detection of unseen objects on BOP-Classic-Core. Methods are ranked by AP (Sec. 3.4). Columns as in Tab. 2.

# Method Awards Year Onboarding im. ...type Test image LM-O T-LESS TUD-L IC-BIN ITODD HB YCB-V AP Time

- 1 MUSE 2024 - - RGB 51.2 46.7 59.5 29.8 50.2 58.9 67.4 52.0 0.56

- 2 F3DT2D 2024 - - RGB 50.4 48.2 57.3 28.4 48.0 57.7 66.6 50.9 0.43

- 3 SAM6D-FastSAM [26] 2023 RGB-D PBR RGB-D 46.3 45.8 57.3 24.5 41.9 55.1 58.9 47.1 0.45

- 4 NIDS-Net WA Sappe [30] 2024 RGB Custom RGB 45.7 49.3 48.6 25.7 37.9 58.7 62.1 46.9 0.49

- 5 NIDS-Net WA [30] 2024 RGB Custom RGB 44.9 48.9 46.0 24.5 36.0 59.4 62.4 46.0 0.49

- 6 SAM6D [26] 2024 RGB-D PBR RGB-D 46.5 43.7 53.7 26.1 39.4 53.0 51.8 44.9 2.80

- 7 SAM6D-FastSAM [26] 2024 RGB-D PBR RGB 43.8 41.7 54.6 23.4 37.4 52.3 57.3 44.4 0.25

- 8 ViewInvDet 2024 - - RGB 44.9 40.3 50.8 26.8 32.8 55.4 58.1 44.2 1.70

- 9 NIDS-Net basic [30] 2024 RGB-D PBR RGB 44.9 42.8 43.4 24.4 34.9 54.8 56.5 43.1 0.49

- 10 CNOS (FastSAM) [35] 2023 - - RGB 43.3 39.5 53.4 22.6 32.5 51.7 56.8 42.8 0.22

- 11 CNOS (SAM) [35] 2023 - - RGB 39.5 33.0 36.8 20.7 31.3 42.3 49.0 36.1 1.85

- 12 ZeroPose [5] 2023 - - RGB 36.7 30.0 43.1 22.8 25.0 39.8 41.6 34.1 3.82

- Tab. 4. Track 3: Model-based 2D detection of unseen objects on BOP-Classic-Core. The methods are ranked by AP (Sec. 3.4). Columns as in Tab. 2.

# Method Awards Year Det./seg. Refine. Train im. ...type

HOT3DHOPEv2HANDAL AP Time

- 1 GigaPose+GenFlow [32,36] 2024 CNOS-FastSAM GenFlow RGB-D PBR 26.8 41.1 25.6 31.2 5.3

- 2 GigaPose [36] 2024 CNOS-FastSAM – RGB PBR 7.2 16.7 4.1 9.4 0.9

- 3 OPFormer-MegaPose 2024 CNOS-FastSAM MegaPose - - - 39.2 26.2 - -

- 4 OPFormer-Coarse 2024 CNOS-FastSAM - - - - 35.1 19.2 - -

- Tab. 5. Track 4: Model-based 6D det. of unseen objects on BOP-H3.

# Method Awards Year Onboarding im. ...type

HOT3D HOPEv2 HANDAL

AP Time

- 1 MUSE 2024 N/A N/A 42.6 47.4 27.0 39.0 1.5

- 2 CNOS (FastSAM) [35] 2024 RGB PBR 35.0 31.3 24.6 30.3 0.3

- 3 CNOS (SAM) [35] 2024 RGB PBR 31.7 36.5 19.7 29.3 1.8

- Tab. 6. Track 5: Model-based 2D det. of unseen objects on BOP-H3.

# Method Awards Year Onboarding type HOT3D HOPEv2 HANDAL AP Time

- 1 GFreeDet (FastSAM) [29] 2024 Static 33.8 36.4 25.5 31.9 0.3

- 2 GFreeDet (SAM) [29] 2024 Static 30.9 38.4 26.4 31.9 2.1

- Tab. 7. Track 7: Model-free 2D det. of unseen objects on BOP-H3.

### 5. Awards

The 2024 challenge awards are based on the results analyzed in Sec. 4. Methods receiving awards are marker with icons: for the best overall method, for the best fast method (the most accurate method with the average running time per image below 1s), for the best open-source method, for the best RGBonly method, and for the best method using default detections.

Authors of awarded entries: FreeZeV2.1 [3] by Andrea Caraffa, Davide Boscaini, Amir Hamza, Fabio Poiesi; Co-op [33] and GenFlow [32] by Sungphill Moon, Hyeontae Son, Dongcheol Hur, Sangwook Kim; FoundationPose [47] by Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield; GigaPose [36] by Van Nguyen Nguyen, Thibault Groueix, Mathieu Salzmann, Vincent Lepetit; SAM6D [26] by Jiehong Lin, Lihua Liu, Dekun Lu, Kui Jia; CNOS [35] by Van Nguyen Nguyen, Thibault Groueix, Georgy Ponimatkin, Vincent Lepetit, Tomas Hodan; GFreeDet [29] by Xingyu Liu, Yingyue Li, Chengxi Li, Gu Wang, Chenyangguang Zhang, Ziqin Huang, Xiangyang Ji; FRTPose.v1 and MUSE are anonymous for now.

duces a novel similarity metric that leverages both class and patch tokens, whereas CNOS only uses class tokens. MUSE is 21% more accurate than CNOS (#1 vs. #10 in Tab. 4), however, MUSE is still -35% behind the best method for seen objects (GDet2023).

### 6. Conclusions

BOP-H3 (Tab. 6). MUSE achieves a noticeably lower score on BOP-H3 than on BOP-Classic-Core (39.0 vs. 52.0AP), while still outperforming CNOS by 29% (#1 vs. #2 in Tab. 6).

In 2024, methods for 6D localization of unseen objects have almost reached the accuracy of methods for seen objects. While some of these methods now take less than 1 second per image, further speed up is needed for real-time applications. Rankings on 6D localization and 6D detection are consistent, suggesting that learnings from past BOP challenges are relevant also for the newly introduced and more practical 6D detection task. 2D detection of unseen objects has improved significantly, but is still noticeably behind 2D detection of seen objects. The evaluation system stays open and awaits new submissions on the model-free tasks.

#### 4.5. Model-free 2D detection of unseen objects

The only entry, GFreeDet [29], reconstructs a 3DGS model [23] from onboarding videos and renders templates from equidistantly sampled viewpoints. At inference, GFreeDet uses SAM [24] to generate proposals and DINOv2 [37] for proposal-template matching. Notably, GFreeDet outperforms the model-based CNOS on HOPEv2 and HANDAL (#1 in Tab. 7 vs. #2 Tab. 6).

### References

- [1] Prithviraj Banerjee, Sindi Shkodrani, Pierre Moulon, Shreyas Hampali, Shangchen Han, Fan Zhang, Linguang Zhang, Jade Fountain, Edward Miller, Selen Basol, Richard Newcombe, Robert Wang, Jakob Julian Engel, and Tomas Hodan. HOT3D: Hand and object tracking in 3D from egocentric multi-view videos. CVPR,

2025. 1, 2, 3

- [2] Eric Brachmann, Alexander Krull, Frank Michel, Stefan Gumhold, Jamie Shotton, and Carsten Rother. Learning 6D object pose estimation using 3D object coordinates. In ECCV, 2014. 3
- [3] Andrea Caraffa, Davide Boscaini, Amir Hamza, and Fabio Poiesi. Freeze: Training-free zero-shot 6d pose estimation with geometric and vision foundation models. ECCV, 2024. 2, 6, 7, 8
- [4] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3D model repository. arXiv preprint arXiv:1512.03012, 2015. 4
- [5] Jianqiu Chen, Mingshan Sun, Tianpeng Bao, Rui Zhao, Liwei Wu, and Zhenyu He. 3d model-based zero-shot pose estimation pipeline. arXiv preprint arXiv:2305.17934, 2023. 7, 8
- [6] Maximilian Denninger, Martin Sundermeyer, Dominik Winkelbauer, Dmitry Olefir, Tom´aˇs Hodaˇn, Youssef Zidan, Mohamad Elbadrawy, Markus Knauer, Harinandan Katam, and Ahsan Lodhi. BlenderProc: Reducing the reality gap with photorealistic rendering. RSS Workshops, 2020. 3, 4
- [7] Maximilian Denninger, Martin Sundermeyer, Dominik Winkelbauer, Youssef Zidan, Dmitry Olefir, Mohamad Elbadrawy, Ahsan Lodhi, and Harinandan Katam. Blenderproc. arXiv preprint arXiv:1911.01911, 2019. 3, 4
- [8] Maximilian Denninger, Dominik Winkelbauer, Martin Sundermeyer, Wout Boerdijk, Markus Wendelin Knauer, Klaus H Strobl, Matthias Humt, and Rudolph Triebel. Blenderproc2: A procedural pipeline for photorealistic rendering. Journal of Open Source Software, 8(82):4901, 2023. 3, 4
- [9] Andreas Doumanoglou, Rigas Kouskouridas, Sotiris Malassiotis, and Tae-Kyun Kim. Recovering 6D object pose and predicting next-best-view in the crowd. In CVPR, 2016. 3
- [10] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality dataset of 3D scanned household items. ICRA, 2022. 4
- [11] Bertram Drost, Markus Ulrich, Paul Bergmann, Philipp Hartinger, and Carsten Steger. Introducing MVTec ITODD – A dataset for 3D object recognition in industry. In ICCVW, 2017. 3
- [12] Andrew Guo, Bowen Wen, Jianhe Yuan, Jonathan Tremblay, Stephen Tyree, Jeffrey Smith, and Stan Birchfield. Handal: A dataset of real-world manipulable object categories with pose annotations, affordances, and reconstructions. In IROS, 2023. 1, 2, 3, 4
- [13] Shreyas Hampali, Tom´aˇs Hodaˇn, Luan Tran, Lingni Ma, Cem Keskin, and Vincent Lepetit. In-hand 3D object scanning from an RGB sequence. Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 4
- [14] Shangchen Han, Po-chen Wu, Yubo Zhang, Beibei Liu, Linguang Zhang, Zheng Wang, Weiguang Si, Peizhao Zhang, Yujun Cai, Tomas Hodan, et al. Umetrack: Unified multi-view end-to-end hand tracking for vr. SIGGRAPH Asia, 2022. https:// github.com/facebookresearch/hand_tracking_ toolkit?tab=readme-ov-file#evaluation. 3

- [15] S. Hinterstoisser, V. Lepetit, S. Ilic, S. Holzer, G. Bradski, K. Konolige, and N. Navab. Model based training, detection and pose estimation of texture-less 3D objects in heavily cluttered scenes. ACCV, 2012. 3
- [16] Tom´aˇs Hodaˇn. Pose estimation of specific rigid objects. PhD Thesis, Czech Technical University in Prague, 2021. 3
- [17] Tom´aˇs Hodaˇn, Eric Brachmann, Bertram Drost, Frank Michel, Martin Sundermeyer, Jiˇr´ı Matas, and Carsten Rother. BOP Challenge 2019. https://bop.felk.cvut.cz/media/ bop_challenge_2019_results.pdf, 2019. 2
- [18] Tom´aˇs Hodaˇn, Pavel Haluza, Stˇˇ ep´an Obdrˇz´alek, Jiˇr´ı Matas, Manolis Lourakis, and Xenophon Zabulis. T-LESS: An RGB-D dataset for 6D pose estimation of texture-less objects. WACV, 2017. 3
- [19] Tom´aˇs Hodaˇn, Frank Michel, Eric Brachmann, Wadim Kehl, Anders Glent Buch, Dirk Kraft, Bertram Drost, Joel Vidal, Stephan Ihrke, Xenophon Zabulis, Caner Sahin, Fabian Manhardt, Federico Tombari, Tae-Kyun Kim, Jiˇr´ı Matas, and Carsten Rother. BOP: Benchmark for 6D object pose estimation. ECCV, 2018. 2, 3
- [20] Tom´aˇs Hodaˇn, Martin Sundermeyer, Bertram Drost, Yann Labb´e, Eric Brachmann, Frank Michel, Carsten Rother, and Jiˇr´ı Matas. BOP Challenge 2020 on 6D object localization. In ECCV, 2020. 2, 5, 6
- [21] Tom´aˇs Hodaˇn, Martin Sundermeyer, Yann Labb´e, Van Nguyen Nguyen, Gu Wang, Eric Brachmann, Bertram Drost, Vincent Lepetit, Carsten Rother, and Jiˇr´ı Matas. BOP challenge 2023 on detection, segmentation and pose estimation of seen and unseen rigid objects. CVPRW, 2024. 2
- [22] Roman Kaskman, Sergey Zakharov, Ivan Shugurov, and Slobodan Ilic. HomebrewedDB: RGB-D dataset for 6D pose estimation of 3D objects. ICCVW, 2019. 3
- [23] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3D gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 2023. 4, 8
- [24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. InICCV, 2023. 8
- [25] Yann Labb´e, Lucas Manuelli, Arsalan Mousavian, Stephen Tyree, Stan Birchfield, Jonathan Tremblay, Justin Carpentier, Mathieu Aubry, Dieter Fox, and Josef Sivic. MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare. In CoRL, 2022. 4, 7
- [26] Jiehong Lin, Lihua Liu, Dekun Lu, and Kui Jia. Sam-6d: Segment anything model meets zero-shot 6d object pose estimation. In CVPR, 2024. 6, 7, 8
- [27] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. ECCV, 2014. 5
- [28] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision, pages 38–55. Springer, 2024. 7
- [29] Xingyu Liu, Yingyue Li, Chengxi Li, Gu Wang, Chenyangguang Zhang, Ziqin Huang, and Xiangyang Ji. Gfreedet: Exploiting gaussian splatting and foundation models for model-free unseen object detection in the bop challenge 2024. arXiv preprint arXiv:2412.01552, 2024. 8

- [30] Yangxiao Lu, Yunhui Guo, Nicholas Ruozzi, Yu Xiang, et al. Adapting pre-trained vision models for novel instance detection and segmentation. arXiv preprint arXiv:2405.17859, 2024. 6, 8
- [31] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 2021. 4
- [32] Sungphill Moon, Hyeontae Son, Dongcheol Hur, and Sangwook Kim. GenFlow: Generalizable Recurrent Flow for 6D Pose Refinement of Novel Objects. In arXiv preprint arXiv:2403.11510,

2024. 2, 6, 7, 8

- [33] Sungphill Moon, Hyeontae Son, Dongcheol Hur, and Sangwook Kim. Co-op: Correspondence-based novel object pose estimation. arXiv preprint arXiv:2503.17731, 2025. 2, 6, 7, 8
- [34] Richard A Newcombe, Shahram Izadi, Otmar Hilliges, David Molyneaux, David Kim, Andrew J Davison, Pushmeet Kohi, Jamie Shotton, Steve Hodges, and Andrew Fitzgibbon. KinectFusion: Real-time dense surface mapping and tracking. ISMAR, 2011. 3
- [35] Van Nguyen Nguyen, Thibault Groueix, Georgy Ponimatkin, Vincent Lepetit, and Tomas Hodan. CNOS: A Strong Baseline for CAD-based Novel Object Segmentation. In ICCVW, 2023. 2, 5, 6, 7, 8
- [36] Van Nguyen Nguyen, Thibault Groueix, Mathieu Salzmann, and Vincent Lepetit. Gigapose: Fast and robust novel object pose estimation via one correspondence. In CVPR, 2024. 6, 7, 8
- [37] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 6, 7, 8
- [38] Evin Pınar Ornek,¨ Yann Labb´e, Bugra Tekin, Lingni Ma, Cem Keskin, Christian Forster, and Tomas Hodan. Foundpose: Unseen object pose estimation with foundation features. In European Conference on Computer Vision, 2024. 7
- [39] Fabio Poiesi and Davide Boscaini. Learning general and distinctive 3d local deep descriptors for point cloud registration. PAMI, 2022. 6
- [40] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 7
- [41] Colin Rennie, Rahul Shome, Kostas E Bekris, and Alberto F De Souza. A dataset for improved RGBD-based object detection and pose estimation for warehouse pick-and-place. RA-L, 2016. 3
- [42] Johannes L Schonberger and Jan-Michael Frahm. Structure-frommotion revisited. In CVPR, 2016. 4
- [43] Martin Sundermeyer, Tomas Hodan, Yann Labb´e, Gu Wang, Eric Brachmann, Bertram Drost, Carsten Rother, and Jiri Matas. BOP challenge 2022 on detection, segmentation and pose estimation of specific rigid objects. CVPRW, 2023. 2
- [44] Alykhan Tejani, Danhang Tang, Rigas Kouskouridas, and Tae-Kyun Kim. Latent-class hough forests for 3D object detection and pose estimation. ECCV, 2014. 3
- [45] Stephen Tyree, Jonathan Tremblay, Thang To, Jia Cheng, Terry Mosier, Jeffrey Smith, and Stan Birchfield. 6-DoF pose estimation of household objects for robotic manipulation: An accessible dataset and benchmark. IROS, 2022. 1, 2, 3, 4
- [46] Bowen Wen, Jonathan Tremblay, Valts Blukis, Stephen Tyree, Thomas M¨uller, Alex Evans, Dieter Fox, Jan Kautz, and

- Stan Birchfield. BundleSDF: Neural 6-DoF tracking and 3D reconstruction of unknown objects. In CVPR, 2023. 4
- [47] Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchfield. Foundationpose: Unified 6d pose estimation and tracking of novel objects. In CVPR, 2024. 6, 7, 8
- [48] Yu Xiang, Tanner Schmidt, Venkatraman Narayanan, and Dieter Fox. PoseCNN: A convolutional neural network for 6D object pose estimation in cluttered scenes. RSS, 2018. 3
- [49] Ruida Zhang, Ziqin Huang, Gu Wang, Xingyu Liu, Chenyangguang Zhang, and Xiangyang Ji. GPose2023, a submission to the BOP Challenge 2023. Unpublished, 2023. http://bop.felk.cvut.cz/method_info/410/. 2

