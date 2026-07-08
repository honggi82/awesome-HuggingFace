## ObjectDrop: Bootstrapping Counterfactuals for Photorealistic Object Removal and Insertion

###### Daniel Winter1,2, Matan Cohen1, Shlomi Fruchter1, Yael Pritch1, Alex Rav-Acha1, Yedid Hoshen1,2

# arXiv:2403.18818v1[cs.CV]27Mar2024

1Google Research, 2The Hebrew University of Jerusalem https://ObjectDrop.github.io

### ABSTRACT

Diffusion models have revolutionized image editing but often generate images that violate physical laws, particularly the effects of objects on the scene, e.g., occlusions, shadows, and reflections. By analyzing the limitations of self-supervised approaches, we propose a practical solution centered on a “counterfactual” dataset. Our method involves capturing a scene before and after removing a single object, while minimizing other changes. By fine-tuning a diffusion model on this dataset, we are able to not only remove objects but also their effects on the scene. However, we find that applying this approach for photorealistic object insertion requires an impractically large dataset. To tackle this challenge, we propose bootstrap supervision; leveraging our object removal model trained on a small counterfactual dataset, we synthetically expand this dataset considerably. Our approach significantly outperforms prior methods in photorealistic object removal and insertion, particularly at modeling the effects of objects on the scene.

[Figure 1]

[Figure 2]

[Figure 3]

Input Baselines

Ours

ObjectRemoval

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

ObjectInsertion

[Figure 8]

Figure 1: Object removal and insertion. Our method models the effect of an object on the scene including occlusions, reflections, and shadows, enabling photorealistic object removal and insertion. It significantly outperforms state-of-theart baselines.

{daniel.winter, yedid.hoshen}@mail.huji.ac.il

### 1 Introduction

Photorealistic image editing requires both visual appeal and physical plausibility. While diffusion-based editing models have significantly enhanced aesthetic quality, they often fail to generate physically realistic images. For instance, object removal methods must not only replace pixels occluded by the object but also model how the object affected the scene e.g., removing shadows and reflections. Current diffusion methods frequently struggle with this, highlighting the need for better modeling of the effects of objects on their scene.

Object removal and insertion is a long-standing but challenging task. Classical image editing methods were unable to tackle the full task and instead targeted specific aspects e.g., removing hard shadows. The advent of text-to-image diffusion models enabled a new class of image editing techniques that aim to perform more general edits.

We analyze the limitations of self-supervised editing approaches through the lens of counterfactual inference. A counterfactual statement [27] takes the form "if the object did not exist, this reflection would not occur". Accurately adding or removing the effect of an object on its scene requires understanding what the scene would look like with and without the object. Self-supervised approaches rely solely on observations of existing images, lacking access to counterfactual images. Disentanglement research [17, 22, 33] highlights that it is difficult to identify and learn the underlying physical processes from this type of data alone, leading to incorrect edits. This often manifests as either incomplete object removal or physically implausible changes to the scene.

Here, we propose a practical approach that trains a diffusion model on a meticulously curated "counterfactual" dataset. Each sample includes: i) a factual image depicting the scene, and ii) a counterfactual image depicting the scene after an object change (e.g., adding/removing it). We create this dataset by physically altering the scene; a photographer captures the factual image, alters the scene (e.g., removes an object), and then captures the counterfactual image. This approach ensures that each example reflects only the scene changes related to the presence of the object instead of other nuisance factors of variation.

We find this technique highly effective for object removal, surprisingly even for large or inaccessible objects that were not seen in training. However, given its limited size, the same dataset proved insufficient for training the reverse task of modeling how a newly inserted object affects the scene. We hypothesize that object insertion, which requires synthesizing shadows and reflections rather than merely removing them, is inherently more complex. We expect this to require a dataset too large for us to collect.

To address this, we propose a two-step approach. First, we train an object removal model using a smaller counterfactual dataset. Second, we apply the removal model on a large unlabeled image dataset to create a vast synthetic dataset. We finetune a diffusion model on this large dataset to add realistic shadows and reflections around newly inserted objects. We term this approach bootstrap supervision.

Our approach, ObjectDrop, achieves unprecedented results for both adding and removing the effects of objects. We show that it compares favorably to recent approaches such as Emu Edit, AnyDoor, and Paint-by-Example. Our contributions are:

- 1. An analysis of the limitations of self-supervised training for editing the effects of objects on scenes, such as shadows and reflections.
- 2. An effective counterfactual supervised training method for photorealistic object removal.
- 3. A bootstrap supervision approach to mitigate the labeling burden for object insertion.

Input Image and Mask

Our Object Removal

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

- Figure 2: Generalization. Our counterfactual dataset is relatively small and was captured in controlled settings, yet the model generalizes exceptionally well to out-of-distribution scenarios such as removing buildings and large objects.

Our Small Counterfactual Dataset

Synthesizing Large Dataset Training to Reconstruct Object Effects

Training Object Removal

[Figure 17]

|[Figure 18]|
|---|

|[Figure 19]|
|---|

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

|[Figure 24]|[Figure 25]|
|---|---|

Counterfactual

Ground Truth

|[Figure 26]|
|---|

|[Figure 27]|
|---|

Noised Ground Truth

Denoising Loss

Denoising Loss

Noised Counterfactual

[Figure 28]

|[Figure 29]|[Figure 30]|
|---|---|

[Figure 31]

[Figure 32]

[Figure 33]

UNET

UNET

…

Pasted Object On Predicted Background

|[Figure 34]|
|---|

|[Figure 35]|
|---|

Prediction

Factual (With Object)

|[Figure 36]|[Figure 37]|
|---|---|

Prediction

[Figure 38]

[Figure 39]

Real Image + Object Mask

Predicted Background

Factual Image (With Object)

Counterfactual (w/o Object)

UNET

Object Mask

Object Mask

- Figure 3: Overview of our method. We collect a counterfactual dataset consisting of photos of scenes before and after removing an object, while keeping everything else fixed. We used this dataset to fine-tune a diffusion model to remove an object and all its effects from the scene. For the task of object insertion, we bootstrap bigger dataset by removing selected objects from a large unsupervised image dataset, resulting in a vast, synthetic counterfactual dataset. Training on this synthetic dataset and then fine tuning on a smaller, original, supervised dataset yields a high quality object insertion model.

###### 1.1 Related Work

Image inpainting. The task of inpainting missing image pixels has been widely explored in the literature. For several years, deep learning methods used generative adversarial network [11] e.g. [16, 30, 38, 40, 43, 58]. Several works use end-to-end learning methods [18, 29, 50, 56]. More recently, the impressive advancements of diffusion models [42, 44, 47, 48], have helped spur significant progress in inpainting [1, 35, 37, 45, 54]. We show (Sec. 3) that despite the great progress in the field and using very powerful diffusion models, these methods are not sufficient for photorealistic object removal.

Shadow removal methods Another line of work focuses on the sub-task of shadow removal. In this task, the model aims to remove the shadow from an image given the shadow mask. Various methods [6–8, 15, 20, 25, 26, 32, 52, 53, 62, 63] have been proposed for shadow removal. More recent methods [12, 36] used latent diffusion models. Unlike these methods that remove only shadows, our method aims to remove all effects of the object on the scene including: occlusions and reflections. Also, these methods require a shadow segmentation map [6, 55], while our method only requires an object segmentation map, which is easy to extract automatically e.g., [23]. OmniMatte [34] aimed to recover both shadows and reflections, however it requires video whereas this paper deals with images.

General image editing model. An alternative approach for removing objects from photos is to use a general purpose text-based image editing model [2, 3, 9, 46, 61]. For example, Emu Edit [46] trains a diffusion model on a large synthetic dataset to perform different editing tasks given a task embedding. MGIE [9] utilizes a diffusion model coupled with Multimodal Large Language Model (MLLM) [31, 51] to enhance the model’s cross-modal understanding. While the breadth of the capabilities of these methods is impressive, our method outperformed them convincingly on object removal.

Object Insertion. Earlier methods for inserting an object into a new image used end-to-end Generative Adversarial Network (GAN) such as Pix2Pix [19], ShadowGAN [60], ARShadowGAN [28] and SGRNet [14]. Recent studies used diffusion models. Paint-by-Example [57] and ObjectStitch [49] insert a reference object into an image using the guidance of a image-text encoder [41], but only preserve semantic resemblance to the inserted object. AnyDoor [4] used a self-supervised representation [39] of the reference object alongside its high-frequency map as conditions to enhance object identity preservation. While the fidelity of generated images produced by AnyDoor improved over former methods, it sometimes changes object identity entirely while we keep it unchanged by design. Furthermore, previous methods often do not model object reflections and shadows accurately, leading to unrealistic outcomes.

###### Input Image Extended Mask SD-XL Inpainting Tight Mask SD-XL Inpainting Ours

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

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

- Figure 4: Object removal - comparison with inpainting. Our model successfully removes the masked object, while the baseline inpainting model replaces it with a different one. Using a mask that covers the reflections (extended mask) may obscure important details from the model.

### 2 Task Definition

We consider the input image X depicting a physical 3D scene S. We want our model to generate how the scene would have looked, had an object O been added to or removed from it. We denote this, the counterfactual outcome. In other words, the task is defined as re-rendering the counterfactual image Xcf, given the physical change. The challenge is to model the effects of the object change on the scene such as occlusions, shadows and reflections. Formally, assume the physical rendering mechanism is denoted as Gphysics, the input image X is given by:

X = Gphysics(O,S) (1) The desired output is the counterfactual image Xcf s.t.,

Xcf = Gphysics(Ocf,S) (2)

For object removal, an object o is originally present O = o and we wish to remove the object so that Ocf = ϕ (ϕ is the empty object). For object insertion, the object is initially absent O = ϕ and we wish to add it Ocf = o.

While the physical rendering mechanism Gphysics is relatively well understood, the formulation in Eq. 1 cannot be used directly for editing as it requires perfect knowledge of the physical object and scene, which is rarely available.

### 3 Self-Supervision is Not Enough

As physical simulations are not a feasible way for photorealistic object removal and insertion in existing images, recent approaches used diffusion models instead. Diffusion models provide high-quality generative models of images i.e., they provide an effective way to estimate and sample from the distribution P(X) where X is a natural image. Generative models are not a natural fit for adding or removing objects from an image as they do not provide a direct way to access or modify its hidden causal variables: the physical object O and the properties of the scene S. The task of inferring the hidden variables (e.g. O and S) and the generative mechanism (e.g., Gphysics) is called disentanglement. Several influential works [17, 22, 33] established that unsupervised disentanglement from observational data is generally impossible without strong priors. Self-supervised methods attempt to perform disentanglement using heuristic schemes.

One common heuristic for object removal is to use diffusion-based inpainting. Such methods rely on an earlier segmentation step that splits the image pixels into two non-overlapping subsets, a subset of pixels that contain the object

Instruction Input Image Text-based Mask Emu Edit MGIE Ours

Remove the tennis ball from the photo

Remove the shoes in front of the cat.

Remove the table from the middle of the room

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

- Figure 5: Object removal - comparison with general editing methods. We compare to general editing methods: Emu Edit and MGIE. These methods often replace the object with a new one and introduce unintended changes to the input image. For this comparison we used a text-based segmentation model to mask the object according to the instruction and passed the mask as input to our model.

Xo and a subset of those that do not Xs s.t. X = Xo ∪ Xs. Inpainting then uses the generative model to resample the values of the object pixels given the scene:

xo ∼ P(Xo|Xs = xs) (3)

The main limitation of this approach is its dependence on the segmentation mask. If the mask is chosen too tightly around the object, then Xs includes the shadows and reflections of the object and thus has information about the object. The most likely values of P(Xo|xs) will contain an object that renders similar shadows and reflections as the original, which is likely a similar object to the original. If the mask is chosen so conservatively as to remove all scene pixels that are affected by the object, it will not preserve the original scene. We show both failure modes in Fig. 4.

Attention-based methods, such as prompt-to-prompt [13], use a sophisticated heuristic based on cross-attention which sometimes overcomes the failure modes of inpainting. However, as they bias the generative model P(Xo|Xe), they can result in unrealistic edits, sometimes removing the object but not its shadows. Also, the attention masks often fail to capture all scene pixels affected by the object, resulting in similar failures as inpainting. Note that Emu Edit [46] uses synthetic data created by an attention-based method for object removal and can therefore suffer from similar failure modes.

The core limitation of the above self-supervised approaches is the inability to directly infer the true generative mechanism and the causal hidden variables, the object O and scene S. While heuristic methods for doing so made progress, the core limitations are hard to overcome. Class-guided disentanglement methods [5, 10] attempt to solve the disentanglement task from observational data by assuming perfect knowledge of one of the hidden variables (here, the physical object O), and assuming that the object and scene are independent. Both assumptions are not sound in this setting, as the properties of the physical object and scene are not known perfectly, and only some objects are likely in a particular scene. Note that the generative mechanism is not perfectly identifiable even when the assumptions are satisfied [21, 22]. This motivates our search for a more grounded approach as will be described in the following sections.

### 4 Object Removal

In this section we propose ObjectDrop, a new approach based on counterfactual supervision for object removal. As mentioned in Sec. 3, it is insufficient to merely model the observed images, but we must also take into account their

##### Input Paint-by-Example AnyDoor Ours

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

- Figure 6: Intra-image object insertion - baseline comparison. We preserve the object identity better and achieve more photorealistic shadows and reflections than baselines Paint-by-Example and AnyDoor.

causal hidden variables i.e., the object and the scene. As we cannot learn these purely from observing images, we propose to directly act in the physical world to estimate the counterfactual effect of removing objects.

- 4.1 Collecting a counterfactual dataset. The key for unlocking such models is by creating a counterfactual dataset. The procedure consists of three steps:

- 1. Capture an image X ("factual") containing the object O in scene S.
- 2. Physically remove the object O while avoiding camera movement, lighting changes or motion of other objects.
- 3. Capture another image Xcf ("counterfactual") of the same scene but without the object O.

We use an off-the-shelf segmentation model [23] to create a segmentation map Mo for the object O removed from the factual image X. The final dataset contain input pairs of factual image and binary object mask (X,Mo(X)), and the output counterfactual image Xcf.

In practice, we collected 2,500 such counterfactual pairs. This number is relatively small due to the high cost of data collection. The images were collected by professional photographers with a tripod mounted camera, to keep the camera pose as stable as possible. As the counterfactual pairs have (almost) exactly the same camera pose, lighting and background objects, the only difference between the factual and counterfactual images is the removal of the object.

- 4.2 Counterfactual distribution estimation.

Given our high-quality counterfactual dataset, our goal is to estimate the distribution of the counterfactual images P(Xcf|X = x,Mo(x)), given the factual image x and segmentation mask. We do it by fine-tuning a large-scale diffusion model on our counterfactual dataset. We investigate the impact of using different foundational diffusion models in Sec. 6. The estimation is done by minimizing:

###### Reference Object Target Image Paint-by-Example AnyDoor Ours

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

- Figure 7: Cross-image object insertion. Similarly to the results of intra-image object insertion, our method preserves object identity better and synthesizes more photorealistic shadows and reflections than the baselines.

L(θ) = E

t∼U([0,T]),ϵ∼N(0,I)

N

∥Dθ(αtxcfi + σtϵ,xi,Mo(xi),t,p) − ϵ∥2 (4)

i=1

where Dθ(x˜t,xcond,m,t,p) is a denoiser network with following inputs: noised latent representation of the counterfactual image x˜t, latent representation of the image containing the object we want to remove xcond, mask m indicating the object’s location, timestamp t and encoding of an empty string (text prompt) p. Here, xt is calculated based on the forward process equation:

x˜t = αt · x + σt · ϵ (5)

Where x represents the image without the object (the counterfactual), αt and σt are determined by the noising schedule, and ϵ ∼ N(0,I).

Importantly, unlike traditional inpainting methods, we avoid replacing the pixels of the object with uniform gray or black pixels. This approach allows our model to leverage information preserved within the mask, which is particularly beneficial in scenarios involving partially transparent objects or imperfect masks.

###### 4.3 Advantages over video supervision.

Our procedure requires making physical world changes, which both limits the possible size of the dataset and the scope of removed objects as they cannot be very big, very heavy or very far. It is tempting to replace this expensive supervision by using the cheaper supervision obtained from videos, as done by several previous methods including AnyDoor [4] and [24]. At training time, the objective of these methods is to reconstruct a masked object in one video frame, by observing the object in another frame. While this procedure is cheaper, it has serious limitations: i) In a counterfactual dataset, the

|Input Composition|w/o Bootstrapping|Ours| |Input Composition|w/o Bootstrapping|Ours|
|---|---|---|---|---|---|---|
|[Figure 105]|[Figure 106]|[Figure 107]| |[Figure 108]|[Figure 109]|[Figure 110]|
|[Figure 111]<br><br>|[Figure 112]|[Figure 113]| |[Figure 114]|[Figure 115]|[Figure 116]|

Figure 8: Bootstrapping ablation. Bootstrap supervision improves model quality.

- Table 1: Object insertion - reconstruction metrics. A comparison with baselines: Paint-by-Example and AnyDoor, on the held-out test set. Furthermore, we ablate the contribution of the bootstrap supervision.

Model PSNR ↑ DINO ↑ CLIP ↑ LPIPS ↓

Paint-by-Example 17.523 0.755 0.862 0.138 AnyDoor 19.500 0.889 0.890 0.095

Ours w/o Bootstrap 20.178 0.929 0.945 0.066 Ours 21.625 0.939 0.950 0.057

only change between the images within a pair should be the removal of the object. Conversely, in video many other attributes also change, such as camera view point. This leads to spurious correlations between object removal and other attributes. ii) This procedure only works for dynamic objects (cars, animals, etc.) and cannot collect samples for inanimate objects. We show in Sec. 6 that our method works exceptionally well, and particularly outperforms methods that use video supervision. Furthermore, our method generalizes surprisingly well to objects that were too challenging to move in our counterfactual dataset including very heavy, large or immobile objects.

### 5 Object Insertion

We extend ObjectDrop to object insertion. In this task, we are given an image of an object, a desired position, and a target image. The objective is to predict how the target image would look like, had it been photographed with the given object. While collecting a relatively small-scale (2,500 samples) counterfactual dataset was successful for object removal, we observed that this dataset is insufficient for training an object insertion model (see Fig. 8). We hypothesize that this requires more examples, as synthesizing the shadows and reflections of the object may be more challenging than removing them.

###### 5.1 Bootstrapping counterfactual dataset.

We propose to leverage our small counterfactual dataset towards creating a large-scale counterfactual object insertion dataset. We take a large external dataset of images, and first detect relevant objects using a foreground detector. Let x1,x2,...,xn denote the original images and Mo(x1),Mo(x2),...,Mo(xn) denote the corresponding object masks. We use our object removal model P(Xcf|X,Mo(X)) to remove the object and its effects on the scene, denoting the results as z1,z2,...,zn where,

zi ∼ P(Xcf|xi,Mo(xi)) (6) Finally, we paste each object into the object-less scenes zi, resulting in images without shadows and reflections,

yi = Mo(xi) ⊙ xi + (1 − Mo(xi)) ⊙ zi. (7)

The synthetic dataset consists of a set of input pairs (yi,Mo(xi)). The corresponding targets are the original images xi. To clarify, both the input and output images contain the object oi, but the input images do not contain the effects of the object on the scene, while the output images do. The task of the model is to generate the effects as illustrated in Fig. 3.

|Input Image|Input Masks|0 Samples|250 Samples|500 Samples|1000 Samples|2500 Samples|
|---|---|---|---|---|---|---|
|[Figure 117]|[Figure 118]|[Figure 119]|[Figure 120]|[Figure 121]|[Figure 122]|[Figure 123]|
|[Figure 124]|[Figure 125]|[Figure 126]|[Figure 127]|[Figure 128]|[Figure 129]|[Figure 130]|

- Figure 9: Counterfactual dataset size. Increasing the size of the training dataset improves object removal performance. The results are of high quality with 2500 examples, but may improve further with more images.

In practice, we start with a dataset consisting of 14M images, we select 700k images with suitable objects. We run object removal on each image, and further filtered approximately half of them that did not have significant object effects on the scene. The final bootstrapped dataset consisted of 350K image, around 140 times bigger than the manually labeled dataset. Please see more details about the filtering process in the supplementary.

###### 5.2 Diffusion model training.

We use the bootstrapped counterfactual dataset to train an object insertion model with the diffusion objective presented in Eq. 4.2. In contrast to the object removal process, we use a pre-trained text-to-image model Dθ(x,t,p) that did not undergo inpainting pre-training. As the input mask increases input dimension, we add new channels to the input of the pre-trained text-to-image model, initializing their weights with 0.

###### 5.3 Fine-tuning on the ground truth counterfactual dataset.

The synthetic dataset is not realistic enough for training the final model, and is only used for pre-training. In the last stage, we fine-tune the model on the original ground truth counterfactual dataset that was manually collected. While this dataset is not large, pre-training on the bootstrapped dataset is powerful enough to enable effective fine-tuning using this small ground truth dataset.

### 6 Experiments

###### 6.1 Implementation Details

Counterfactual dataset. We created a counterfactual dataset of 2,500 pairs of photos using the procedure detailed in Sec. 4. Each pair contains a "factual" image of a scene and a second "counterfactual" image of exactly the same scene except that it was photographed after removing one object. We also held out 100 counterfactual test examples, captured after the completion of the research, depicting new objects and scenes.

Model architecture. We train a latent diffusion model (LDM) for the object removal task. We initialize using a pre-trained inpainting model, which takes as input a factual image, an object mask, and a noisy counterfactual image. We perform inference using default settings. We use a internal model with a similar architecture to Stable-Diffusion-XL. Unlike other inpainting models, we do not replace the removed object pixels with gray pixels.

Quantitative metrics. We compared the results of our method and the baselines on the held-out counterfactual test set. As this dataset has ground truth (see supplementary), we used standard reconstruction metrics: both classical (PSNR) and deep perceptual similarity metrics using: DINO [39], CLIP [41], and LPIPS [59] (AlexNet) features.

###### 6.2 Object Removal

Qualitative results. We evaluated our result on the benchmark published by Emu Edit [46]. As seen in Fig. 5, our model removes objects and their effects in a photorealistic manner. The baselines failed to remove shadows and reflections and sometimes adversely affected the image in other ways.

- Table 2: Object removal - reconstruction metrics. A comparison with the inpainting baseline on the held-out test set. Model PSNR ↑ DINO ↑ CLIP ↑ LPIPS ↓

Inpainting 21.192 0.876 0.897 0.056 Ours 23.153 0.948 0.959 0.048

Table 3: Object removal - user study. A comparison to Emu Edit and MGIE on the Emu Edit dataset [46] Which model did better job at following the object removal editing instruction?

Preferred Emu Edit 35.9% Preferred MGIE 13.5% Preferred ours 64.1% Preferred ours 86.5%

Quantitative results. Tab. 2 compares our method to the inpainting pre-trained model on the held-out test set using quantitative reconstruction metrics. Our method outperformed the baseline substantially.

User study. We conducted a user study on the benchmark by Emu Edit [46] between our method and baselines: Emu Edit and MGIE. As the benchmark does not have ground truth, user study is the most viable comparison. We used the CloudResearch platform to gather user preferences from 50 randomly selected participants. Each participant was presented with 30 examples of an original image, removal text instructions, and results generated by our method and the baseline. Tab. 3 displays the results. Notably, our method surpassed both baseline methods in user preference.

###### 6.3 Object Insertion

Qualitative results. We compare our object insertion model with state-of-the-art image reference-based editing techniques, Paint-by-Example [57] and AnyDoor [4]. Fig. 6 shows intra-image insertions, i.e., when the objects are re-positioned within same image. For achieving intra-image insertions we first use our object removal model to remove the object from its original position, obtaining the background image. For equitable comparisons, we used the same background image, obtained by our model, when comparing to the baselines. Fig. 7 shows inter-image insertions, i.e., when the objects come from different images. In both cases our method synthesizes the shadows and reflections of the object better than the baselines. It also preserves the identity of the object, while other methods modify it freely and in many cases lose the original identity entirely.

Quantitative results. We compare to Paint-by-Example and AnyDoor on the held-out counterfactual test dataset. The results are presented in Tab. 1. Our method outperforms the baselines by a significant margin on all metrics.

User study. We also conducted a user-study on 2 intra-image insertion datasets. The first is the held-out test set. The second is a set of 50 out-of-distribution images depicting more general scenes, some are very different from those seen in training e.g., inserting boats and building. Tab. 4 shows that users overwhelmingly preferred our method over the baselines.

###### 6.4 Ablation Study

Bootstrapping. We ablated the contribution of our bootstrapping method for object insertion. Here, we bootstrapped 2,500 real images into 350K synthetic images. In the ablation, we compare our full method to finetuning the original backbone on the original counterfactual dataset without bootstrapping. Both models used the same pretraining backbone. Both the qualitative results in Fig. 8 and the quantitative results in Tab. 1 clearly support bootstrapping.

Dataset size. Collecting large counterfactual datasets is expensive. We evaluate the influence of dataset size on the performance of our object removal method. We finetune the base model on subsets of the full counterfactual dataset with varying sizes. Fig. 9 shows that using the pre-trained inpainting model without finetuning ("0 samples") merely replaces the target object by another similar one. Also, its effects on the scene remain. The results start looking attractive around 1000 counterfactual examples, more examples further improve performance.

Text-to-image vs. inpainting pretrained model. Fig. 10 demonstrates that using a text-to-image (T2I) instead of an inpainting model for pre-training obtains comparable quality for removing shadows and reflections. This shows that inpainting models do not have better inductive bias for modeling object effects on scenes than T2I models. Unsurprisingly, the inpainting model is better at inpainting the pixels occluded by the objects. Furthermore, we compared the pre-trained models (not shown) on object insertion. Consequently, we used the inpainting backbone for the object removal experiments and the T2I backbone for the object insertion experiments in the paper.

|Input Image and Mask|Text-to-Image Pre-train|Inpainting Pre-train| |Input Image and Mask|Text-to-Image Pre-train|Inpainting Pre-train|
|---|---|---|---|---|---|---|
|[Figure 131]|[Figure 132]|[Figure 133]| |[Figure 134]|[Figure 135]|[Figure 136]|

- Figure 10: Inpainting vs. Text-to-Image pre-training. The inpainting model has better results on pixels occluded by the objects but results in comparable quality for removing or adding photorealistic reflections and shadows.

Table 4: Object insertion - user study. A comparison on in-distribution (ID) and out-of-distribution (OOD) intra-image object insertion datasets.

###### Held-Out Set (ID)

###### In-the-Wild (OOD)

AnyDoor 11.1% Paint-by-Example 3.3% Ours 88.9% Ours 96.7%

AnyDoor 5.0% Paint-by-Example 2.8% Ours 95.0% Ours 97.2%

Table 5: Stable-Diffusion results. Our method works well on the public Stable-Diffusion-Inpainting v1 model. Model PSNR ↑ DINO ↑ CLIP ↑ LPIPS ↓

SD Inpainting 19.198 0.775 0.884 0.083 Ours SD 21.363 0.876 0.930 0.076

Public models. We verified that our method works on publicly available models. Here, we trained our model using Stable-Diffusion-Inpainting v1 as the pre-trained backbone. We then computed the quantitative metrics for object removal as in Sec. 6.2. Our results in Tab. 5 show that our method improves this pretrained model significantly.

### 7 Limitations

This work focuses on simulating the effect that an object has on the scene, but not the effect of the scene on the object. As a result, our method may yield unrealistic results in scenarios where the orientation and lighting of the object are incompatible with the scene. This can be solved independently using existing harmonization methods, but this was not explored in the context of this work. Additionally, as our model does not know the physical 3D scene and lighting perfectly, it may result in realistic-looking but incorrect shadow directions.

### 8 Conclusion

We introduced ObjectDrop, a supervised approach for object removal and insertion to overcome the limitations of previous self-supervised approaches. We collected a counterfactual dataset consisting of pairs of images before and after the physical manipulation of the object. Due to the high cost of obtaining such a dataset, we proposed a bootstrap supervision method. Finally, we showed through comprehensive evaluation that our approach outperforms the state-of-the-art.

### 9 Acknowledgement

We would like to thank to Gitartha Goswami, Soumyadip Ghosh, Reggie Ballesteros, Srimon Chatterjee, Michael Milne and James Adamson for providing the photographs that made this project possible. We thank Yaron Brodsky, Dana Berman, Amir Hertz, Moab Arar, and Oren Katzir for their invaluable feedback and discussions. We also appreciate the insights provided by Dani Lischinski and Daniel Cohen-Or, which helped improve this work.

### References

- [1] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18208–18218,

2022. 3

- [2] Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. Text2live: Text-driven layered image and video editing. In European conference on computer vision, pages 707–723. Springer, 2022. 3
- [3] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 3
- [4] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. arXiv preprint arXiv:2307.09481, 2023. 3, 7, 10
- [5] Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8188–8197, 2020. 5
- [6] Xiaodong Cun, Chi-Man Pun, and Cheng Shi. Towards ghost-free shadow removal via dual hierarchical aggregation network and shadow matting gan. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 10680–10687, 2020. 3
- [7] Bin Ding, Chengjiang Long, Ling Zhang, and Chunxia Xiao. Argan: Attentive recurrent generative adversarial network for shadow detection and removal. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10213–10222, 2019.
- [8] Lan Fu, Changqing Zhou, Qing Guo, Felix Juefei-Xu, Hongkai Yu, Wei Feng, Yang Liu, and Song Wang. Autoexposure fusion for single-image shadow removal. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10571–10580, 2021. 3
- [9] Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, and Zhe Gan. Guiding instruction-based image editing via multimodal large language models. arXiv preprint arXiv:2309.17102, 2023. 3
- [10] Aviv Gabbay, Niv Cohen, and Yedid Hoshen. An image is worth more than a thousand words: Towards disentanglement in the wild. Advances in Neural Information Processing Systems, 34:9216–9228, 2021. 5
- [11] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 3
- [12] Lanqing Guo, Chong Wang, Wenhan Yang, Siyu Huang, Yufei Wang, Hanspeter Pfister, and Bihan Wen. Shadowdiffusion: When degradation prior meets diffusion model for shadow removal. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14049–14058, 2023. 3
- [13] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 5
- [14] Yan Hong, Li Niu, and Jianfu Zhang. Shadow generation for composite image in real-world scenes. In Proceedings of the AAAI conference on artificial intelligence, pages 914–922, 2022. 3
- [15] Xiaowei Hu, Yitong Jiang, Chi-Wing Fu, and Pheng-Ann Heng. Mask-shadowgan: Learning to remove shadows from unpaired data. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2472–2481, 2019. 3
- [16] Zheng Hui, Jie Li, Xiumei Wang, and Xinbo Gao. Image fine-grained inpainting. arXiv preprint arXiv:2002.02609,

2020. 3

- [17] Aapo Hyvärinen and Petteri Pajunen. Nonlinear independent component analysis: Existence and uniqueness results. Neural networks, 12(3):429–439, 1999. 2, 4
- [18] Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa. Globally and locally consistent image completion. ACM Transactions on Graphics (ToG), 36(4):1–14, 2017. 3
- [19] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1125–1134, 2017. 3
- [20] Yeying Jin, Aashish Sharma, and Robby T Tan. Dc-shadownet: Single-image hard and soft shadow removal using unsupervised domain-classifier guided network. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5027–5036, 2021. 3

- [21] Jonathan Kahana and Yedid Hoshen. A contrastive objective for learning disentangled representations. In European Conference on Computer Vision, pages 579–595. Springer, 2022. 5
- [22] Ilyes Khemakhem, Diederik Kingma, Ricardo Monti, and Aapo Hyvarinen. Variational autoencoders and nonlinear ica: A unifying framework. In International Conference on Artificial Intelligence and Statistics, pages 2207–2217. PMLR, 2020. 2, 4, 5
- [23] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023. 3, 6, 16
- [24] Sumith Kulal, Tim Brooks, Alex Aiken, Jiajun Wu, Jimei Yang, Jingwan Lu, Alexei A Efros, and Krishna Kumar Singh. Putting people in their place: Affordance-aware human insertion into scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17089–17099, 2023. 7
- [25] Hieu Le and Dimitris Samaras. Shadow removal via shadow image decomposition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8578–8587, 2019. 3
- [26] Hieu Le and Dimitris Samaras. From shadow segmentation to shadow removal. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XI 16, pages 264–281. Springer,

2020. 3

- [27] David K. Lewis. Counterfactuals. Blackwell, Malden, Mass., 1973. 2
- [28] Daquan Liu, Chengjiang Long, Hongpan Zhang, Hanning Yu, Xinzhi Dong, and Chunxia Xiao. Arshadowgan: Shadow generative adversarial network for augmented reality in single light scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8139–8148, 2020. 3
- [29] Guilin Liu, Fitsum A Reda, Kevin J Shih, Ting-Chun Wang, Andrew Tao, and Bryan Catanzaro. Image inpainting for irregular holes using partial convolutions. In Proceedings of the European conference on computer vision (ECCV), pages 85–100, 2018. 3
- [30] Hongyu Liu, Bin Jiang, Yibing Song, Wei Huang, and Chao Yang. Rethinking image inpainting via a mutual encoder-decoder with feature equalizations. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 725–741. Springer, 2020. 3
- [31] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 3
- [32] Zhihao Liu, Hui Yin, Xinyi Wu, Zhenyao Wu, Yang Mi, and Song Wang. From shadow generation to shadow removal. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4927–4936, 2021. 3
- [33] Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Schölkopf, and Olivier Bachem. Challenging common assumptions in the unsupervised learning of disentangled representations. In international conference on machine learning, pages 4114–4124. PMLR, 2019. 2, 4
- [34] Erika Lu, Forrester Cole, Tali Dekel, Andrew Zisserman, William T Freeman, and Michael Rubinstein. Omnimatte: Associating objects and their effects in video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4507–4515, 2021. 3
- [35] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11461–11471, 2022. 3
- [36] Kangfu Mei, Luis Figueroa, Zhe Lin, Zhihong Ding, Scott Cohen, and Vishal M Patel. Latent feature-guided diffusion models for shadow removal. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4313–4322, 2024. 3
- [37] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 3
- [38] Evangelos Ntavelis, Andrés Romero, Siavash Bigdeli, Radu Timofte, Zheng Hui, Xiumei Wang, Xinbo Gao, Chajin Shin, Taeoh Kim, Hanbin Son, et al. Aim 2020 challenge on image extreme inpainting. In Computer Vision–ECCV 2020 Workshops: Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16, pages 716–741. Springer, 2020. 3
- [39] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 3, 9

- [40] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2536–2544, 2016. 3
- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3, 9
- [42] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022. 3
- [43] Yurui Ren, Xiaoming Yu, Ruonan Zhang, Thomas H Li, Shan Liu, and Ge Li. Structureflow: Image inpainting via structure-aware appearance flow. In Proceedings of the IEEE/CVF international conference on computer vision, pages 181–190, 2019. 3
- [44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 3
- [45] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 Conference Proceedings, pages 1–10, 2022. 3
- [46] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. arXiv preprint arXiv:2311.10089,

2023. 3, 5, 9, 10

- [47] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR,

2015. 3

- [48] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 3
- [49] Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, and Daniel Aliaga. Objectstitch: Object compositing with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18310–18319, 2023. 3
- [50] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2149–2159, 2022. 3
- [51] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3
- [52] Jin Wan, Hui Yin, Zhenyao Wu, Xinyi Wu, Yanting Liu, and Song Wang. Style-guided shadow removal. In European Conference on Computer Vision, pages 361–378. Springer, 2022. 3
- [53] Jifeng Wang, Xiang Li, and Jian Yang. Stacked conditional generative adversarial networks for jointly learning shadow detection and shadow removal. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1788–1797, 2018. 3
- [54] Su Wang, Chitwan Saharia, Ceslee Montgomery, Jordi Pont-Tuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J Fleet, Radu Soricut, et al. Imagen editor and editbench: Advancing and evaluating text-guided image inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18359–18369, 2023. 3
- [55] Tianyu Wang, Xiaowei Hu, Qiong Wang, Pheng-Ann Heng, and Chi-Wing Fu. Instance shadow detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1880–1889, 2020. 3
- [56] Chenfei Wu, Jian Liang, Xiaowei Hu, Zhe Gan, Jianfeng Wang, Lijuan Wang, Zicheng Liu, Yuejian Fang, and Nan Duan. Nuwa-infinity: Autoregressive over autoregressive generation for infinite visual synthesis. arXiv preprint arXiv:2207.09814, 2022. 3
- [57] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18381–18391, 2023. 3, 10

- [58] Yanhong Zeng, Jianlong Fu, Hongyang Chao, and Baining Guo. Learning pyramid-context encoder network for high-quality image inpainting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1486–1494, 2019. 3
- [59] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 9
- [60] Shuyang Zhang, Runze Liang, and Miao Wang. Shadowgan: Shadow synthesis for virtual objects with conditional adversarial networks. Computational Visual Media, 5:105–115, 2019. 3
- [61] Shu Zhang, Xinyi Yang, Yihao Feng, Can Qin, Chia-Chih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, et al. Hive: Harnessing human feedback for instructional visual editing. arXiv preprint arXiv:2303.09618, 2023. 3
- [62] Yurui Zhu, Jie Huang, Xueyang Fu, Feng Zhao, Qibin Sun, and Zheng-Jun Zha. Bijective mapping network for shadow removal. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5627–5636, 2022. 3
- [63] Yurui Zhu, Zeyu Xiao, Yanchi Fang, Xueyang Fu, Zhiwei Xiong, and Zheng-Jun Zha. Efficient model-driven network for shadow removal. In Proceedings of the AAAI conference on artificial intelligence, pages 3635–3643,

2022. 3

### A Training and Inference

- A.0.1 Object Removal

In our object removal training process, we utilize a pre-trained text-to-image latent diffusion model (LDM) that was further trained for inpainting. Given an image of an object ("factual") and its mask, we finetune the LDM to denoise an image of the same scene without the object (the counterfactual image). We performed 50,000 optimization steps with batch size of 128 images and learning rate of 1e–5.

- A.0.2 Object Insertion

To train our object insertion model, we first finetune the model using a synthetic dataset as described in Section 5. This initial training phase consists of 100,000 optimization steps, employing a batch size of 512 images and a learning rate of 5e–5. Subsequently, we fine-tune the model on our original counterfactual dataset for an additional 40,000 steps, with batch size of 128 and decaying learning rates.

The denoiser function Dθ(xt,xcond,m,t,p) receives the following inputs:

- • xt: Noised latent representation of the image containing the object.
- • xcond: Latent representation of the object pasted onto a background image as is, without its effects on the scene.
- • m: Mask indicating the object’s location.
- • t: Timestamp.
- • p: Encoding of an empty string (text prompt).

A.1 Inference All images in this paper were generated at resolution of 512 × 512, with 50 denoising steps.

### B Bootstrapping

The bootstrapping procedure for creating the object insertion training set, as outlined in Section 5, follows these steps: We begin with an external dataset of 14 million images and extract foreground segmentation for each. We filter out images where the foreground mask covers less than 5% or more than 50% of the total image area, aiming to exclude objects that are either too small or too large. Additionally, we eliminate images where the foreground object extends across more than 20% of the lower image boundary, as the shadow or reflection of these objects is often not visible within the image. This filtering process results in 700,000 images potentially containing suitable objects for removal.

Using our object removal model, we generate predicted background images. However, in many of the original images, the object does not have a significant shadow or reflection, so that the difference between the synthesized input and output pairs consists of noise. To address this, we further discard images where the area showing significant differences between the object image and the predicted background is too small. This yields our final bootstrapped dataset of 350,000 examples.

### C Evaluation Datasets

To assess our object insertion model, we employed two datasets. The first, referred to as the held-out dataset, comprises 51 triplets of photos taken after the completion of the project. Each triplet consists of: (1) a scene without the object, (2) the same scene with an added object, and (3) another image of the same scene and the same object placed elsewhere. We automatically segmented [23] the added object and relocated it within the image by naively pasting it on the background scene image. The model’s inputs consist of the image with the pasted object and its corresponding mask. This dataset, along with our results, are presented in Fig. 13. With ground truth images illustrating how object movement should appear, we conducted quantitative metric assessments and user studies. Additionally, we used this dataset for evaluating the object removal model. In this test, we removed the object and compared the generated image to the ground truth background image.

The second test set, utilized for object insertion, comprises 50 examples, including some out-of-distribution images intended for moving large objects, as shown in Fig. 12. As this dataset is lacking ground truth images, we used this dataset solely for user study.

### D User Study

To assess the effectiveness of our object removal model, we conducted a user study using the test set provided by Emu Edit of 264 examples, as shown in Fig. 11. We compared our results separately with those of Emu Edit and MGIE. Utilizing the CloudResearch platform, we collected user preferences from 50 randomly selected participants. Each participant reviewed 30 examples consisting of an original image, removal instructions, and the outcomes produced by both our method and the baseline. We randomized both the order of the examples shown and the order of each model in each example. To improve the reliability of the responses, we duplicated a few questions, and removed questionnaires that showed inconsistency for those repeated questions. A similar user study was carried out to compare our object insertion model with AnyDoor and Paint-by-Example, using the datasets described in Section C. Different participants were used for each dataset and comparison with baselines. The majority of participants were located in the United States. Participant were compensated above the minimum wage.

Instruction Input Image Text-based Mask Emu Edit MGIE Ours

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Remove the cat from the photo

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

Remove the bag from the grass.

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

Remove the laptop from the desk

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Remove the baseball from inside the glove.

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

Delete the vent hood on top of the oven.

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Get rid of the bulls at the bottom of the picture.

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

Remove the bowl from on top of the plate at the right side of the image.

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Remove the brown goat from the image.

- Figure 11: Additional examples for comparison with general editing methods, Emu Edit and MGIE. In this comparison, we utilized a text-based segmentation model to generate a mask for the object based on given instructions, which was then used as input for our model.

###### Input Paint-by-Example AnyDoor Ours

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Figure 12: Additional examples of inta-image object insertion.

Background Original Location

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Model Conditions

Model Output Ground Truth Mask New Location

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

Figure 13: Our held-out test set. The object insertion model uses two conditions: (1) An image where the object was pasted naively on the background and (2) a mask of that object.

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

