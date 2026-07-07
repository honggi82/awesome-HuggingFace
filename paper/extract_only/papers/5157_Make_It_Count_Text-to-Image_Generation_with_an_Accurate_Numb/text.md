# arXiv:2406.10210v1[cs.CV]14Jun2024

## Make It Count: Text-to-Image Generation with an Accurate Number of Objects

Lital Binyamin1 Yoad Tewel2,3 Hilit Segev1 Eran Hirsch1 Royi Rassin1 Gal Chechik1,2 1Bar-Ilan University 2NVIDIA 3Tel-Aviv University https://make-it-count-paper.github.io

### Abstract

Despite the unprecedented success of text-to-image diffusion models, controlling the number of depicted objects using text is surprisingly hard. This is important for various applications from technical documents, to children’s books to illustrating cooking recipes. Generating object-correct counts is fundamentally challenging because the generative model needs to keep a sense of separate identity for every instance of the object, even if several objects look identical or overlap, and then carry out a global computation implicitly during generation. It is still unknown if such representations exist. To address count-correct generation, we first identify features within the diffusion model that can carry the object identity information. We then use them to separate and count instances of objects during the denoising process and detect over-generation and under-generation. We fix the latter by training a model that predicts both the shape and location of a missing object, based on the layout of existing ones, and show how it can be used to guide denoising with correct object count. Our approach, CountGen, does not depend on external source to determine object layout, but rather uses the prior from the diffusion model itself, creating prompt-dependent and seed-dependent layouts. Evaluated on two benchmark datasets, we find that CountGen strongly outperforms the countaccuracy of existing baselines.

“A realistic photo of Goldilocks and three bears eating a porridge”

“A photo of six kittens sitting on a branch”

“A realistic photo of seven dwarves dancing in the forest”

“A photo of ﬁve eggs in a carton”

“an illustration of four ninja turtles”

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

CountGen (ours)

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

SDXL

Figure 1: CountGen generates the correct number of objects specified in the input prompt while maintaining a natural layout that aligns with the prompt.

Preprint. Under review.

### 1 Introduction

Text-to-image diffusion models provide an accessible way to control the generation of visual content. A major failure mode is their inability to count, that is, they often fail to generate the correct number of items in response to text prompts. For instance, when asked to generate an image of Goldilocks and the three bears, models may generate only two bears (Figure 1). Counting failures are particularly frustrating: The accuracy is surprisingly low, and mistakes are often obvious for people to detect.

To illustrate the difficulty of the problem, consider some naive attempts to work around it. First, one can manually design layouts per count, to determine the spatial organization and the number of objects, then provide it as a conditioning signal to a generative model [1]. This approach would fail to generate prompt-dependent layouts, which is highly desirable. One could also try asking large vision-language models to propose layouts (e.g., [2, 3]), but these approaches do not use the visual prior information that text-to-image models already collected, and, as we show below, their performance is rather poor for the counting task.

Why is it so hard for diffusion models to count while they generate? First, counting objects requires that models capture "objectness" – the high-level coherent representation of something being a separate entity, even if surrounded by other similar entities. Capturing objectness is by itself a hard task in image understanding [4, 5], and long studied in cognitive psychology [6]. It is currently not known to what extent diffusion models represent objectness of entities they generate. A second main challenge is that text-to-image diffusion models struggle with controlling spatial layout just from text. Producing a correct number of objects requires obeying a global and complex spatial relation between entities in an image [7, 8].

To address the problem of accurate count generation we describe several new contributions, which together form our method CountGen. First, we analyze the representations of the self-attention layers in SDXL [9], and identify features that capture objectness and instance identity. We then use these features to develop ways to detect instances of objects during the denoising process, find their spatial layout and count them. Specifically, we localize the features that correspond to objects using the cross-attention maps and cluster them to form object instance segmentation. Importantly, we do not have to wait for an image to be fully generated, and we can accurately count the number of objects already at an intermediate step of the denoising process.

Given this new capability to count the number of objects being generated during the denoising process, we further develop methods to correct generation when the count does not match the prompt. First, we train a layout-modification network we call ReLayout. It takes a spatial layout of k objects and generates a similar spatial layout with one more instance of an object added in a natural location for the input layout. For example, given a row of five kittens sitting on a branch, it learned to add a sixth kitten to the same row. This model is trained using image-pair samples generated by the diffusion model itself. Finally, we show how to use the new layouts in a new test-time-optimization procedure.

We evaluate CountGen on text prompts from the T2I-CompBench [10] which includes prompts with numbers. CountGen greatly improves accuracy, as evident by human evaluation experiments, from 29% accuracy for SDXL to 48% by our method. It also improves over all other baseline methods including large commercial models like the recent DALL-E 3 [11]. To support future work in this field, we design and release a dataset that can be evaluated automatically. Specifically, we release CoCoCount, a set of prompts based on COCO classes[12], which can easily be evaluated using COCO-trained object detectors, like YOLO [13]. CountGen also significantly improves over all baseline methods on CoCoCount, importantly from 26% accuracy for SDXL to 54% by our method.

In summary, this paper makes the following new contributions (1) We identify novel features that represent objectness and instance identity in SDXL [9]. (2) We design an inference-time optimization to guide SDXL to generate an accurate number of instances for an object. (3) We describe a learning approach to automatically modify layouts to add a new instance of an object while preserving the structure of the scene. (4) We achieve state-of-the-art results in count-accurate generation.

### 2 Related Work

Generating images with accurate object count. Numerous papers noted that text-to-image diffusion models often fail to produce images that accurately match text prompts, especially when these prompts

specify an exact number of objects [14, 15, 16, 17, 18, 3, 19, 20, 21, 22, 8, 23, 7]. Various efforts were made to improve the accuracy of these models. Most relevant to our work, Kang et al. [14] proposed a classifier-guidance approach to improve object count accuracy. The method “counts” instances at each diffusion step using a pretrained counting network and adjusts the denoising process using gradient guidance. However, it requires using an additional U-Net in every denoising step.

An important line of work suggests breaking the generation process into two steps: (1) Text-to-layout - setting a spatial location for every object instance; and (2) Layout-to-image - generating an image with the correct object count using the given layout. Text-to-layout: Several studies used large language models (LLMs) to propose spatial layouts [2, 24, 3, 25]. LayoutGPT [3] injects visual commonsense into the LLM prompt which enables it to generate desirable layouts. Gani et al. [25] suggest decomposing complex prompts into smaller prompts before injecting them into the LLM. Layout-to-image: Providing a predefined layout with the exact number of subjects helps ensure that the generated images reflect the intended count [2, 26]. Bounded Attention [8] addresses this challenge by channeling attention to bounding boxes corresponding to object instances. However, this approach requires users to manually provide the bounding boxes for all the instances of each object. In contrast to these separate-step approaches, CountGen, addresses the two steps of count-accurate generation. It first corrects the layout that emerges during generation so it contains the correct number of instances. It then uses a novel test-time optimization method to generate a count-accurate image.

Controlling text-to-image models through attention-based loss. To address the issue of object neglect—when objects mentioned in a text prompt fail to appear in the generated image—Chefer et al. [7] developed a novel loss function that ensures all objects in the prompt are reflected in the cross-attention maps used during image generation. Rassin et al. [23] tackled the challenge of incorrect attribute association by designing a loss function that binds the cross-attention maps of subjects and their attributes more effectively. Inspired by these advancements, CountGen includes a novel cross-attention maps loss function designed to ensure the generation adheres closely to the input layout.

### 3 Our Approach: CountGen

Our method, CountGen, aims to enhance text-conditioned image generators to accurately produce the intended number of objects for complex input prompts. Our methodology involves a two-step process: initially, we generate a natural layout that specifies where and how objects should appear in the image (Section 3.2). That layout is based on a layout that emerges naturally from the text-conditioned generation (Section 3.1). At the second step (Section 3.3), we use this layout as a blueprint to generate the final image.

#### 3.1 Discovering Object-Instance Layout during Early Generation

To count object instances during generation, one must first find an internal representation that captures the separate identity of different object instances. It is not known if this representation exists in diffusion models like SDXL. We now discuss this representation and then show how we can detect the layout of object instances during early generation.

An emerging instance-identity representation in SDXL. We begin by exploring the notion of ‘objectness’ in SDXL. While previous work [7, 27, 28] utilized the cross-attention mechanism to localize objects of a given class in generated images, little research has been conducted on whether the model encodes information about object instances and how to distinguish between different instances of an object. We tackle this problem by exploring a variety of features across different layers and timesteps of the diffusion process to determine if and where the model encodes instance-level information. Figure 3 illustrates this analysis using PCA visualization of self-attention features from various layers across SDXL at timestep t = 500, which shows the most robust instance representation (See appendix Figure 12). While most layers do not exhibit separability at the instance level, we notice that layer l52up tends to generate different features for different instances of the same object, with each instance having its distinct color. Based on this finding, we select the self-attention features from layer l52up at timestep t = 500 to serve as our instance-level features.

[Figure 11]

[Figure 12]

[Figure 13]

"A photo of six kittens sitting on a branch"

Layout Guided SDXL

SDXL

[Figure 14]

[Figure 15]

Instance Localization

ReLayout

ReLayout Layout Guided SDXL

Attention-Guided Instance-Localization

Iterative object addition

Cross-Attention Mask

[Figure 16]

[Figure 17]

[Figure 18]

Padding

[Figure 19]

Resize

Self-Attention Masking

"kittens"

U-Net

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

DBSCAN

Padding

Resize

Diffusion Step

U-Net

Self-Attention Features

[Figure 26]

[Figure 27]

[Figure 28]

"kittens"

Object removal

[Figure 29]

[Figure 30]

CrossAttention

- Figure 2: Architecture outline. Given a prompt that includes a quantity, we begin generating a corresponding image using pretrained SDXL until timestep t = 500. We then perform Instance Localization, where we combine cross-attention maps corresponding with the object, and selfattention features extracted at timestep t to generate object clusters for each generated object. Then we apply ReLayout, which generates an object layout with the correct number of instances, while preserving the composition of the extracted layout. Finally, we perform Layout Guided generation,

which applies an inference time optimization based on the layout through cross-attention loss Lcross and self-attention masking.

Identifying object instances. Building on the findings of Hertz et al. [27], which show that cross-attention maps can pinpoint a token’s position in a generated image, we create a foreground mask for each object described in the prompt. By contrasting these foreground masks, derived from the cross-attention, with the self-attention features, we effectively segregate pixels associated with objects from those belonging to the background. Subsequently, we cluster the object-associated pixels from the self-attention map into distinct masks for each object. This approach allows us to refine our object representations and enhance the accuracy of the generated layouts.

Formally, let Aselfl,t ,Acrossl,t represent the self-attention and cross-attention maps, respectively, for layer l at timestep t within our diffusion network. We aggregate cross-attention maps associated with the tokens corresponding to the objects specified in the input prompt. We then use these crossattention maps to extract a foreground mask M based on dynamic thresholding M = Otsu(Acrossl,t ), where Otsu applies the Otsu thresholding method [29, 30] to determine foreground (object) pixels. We define pk ⊆ Aselfl,t as the set of features from the self-attention map that are identified as foreground by mask M. We then cluster these patches: Clusters = DBSCAN(pk,ϵ), where DBSCAN(·,ϵ) is the DBSCAN [31] clustering algorithm with a dynamic parameter ϵ. Finally, the initial layout L is created by grouping the object clusters: L = C∈Clusters C. At the end of this process, we obtain a set of masks, one for each object being generated. This is illustrated in Figure 2, left gray box.

#### 3.2 ReLayout: Correcting the Number of Objects in the Mask

We now introduce our layout-correction component, ReLayout, which preserves the overall scene composition while correcting the number of objects. For example, Figure 2 depicts an image generated using the prompt “a photo of six cats”, but only four cats were generated. Our ReLayout generates a new layout with the correct number of instances while keeping the overall composition of the kittens sitting in a row. More examples are shown in Figure 4.

up l8down down

l52

l42

|[Figure 31]|
|---|

[Figure 32]

[Figure 33]

[Figure 34]

mid up up

l48 l78

l130

[Figure 35]

[Figure 36]

[Figure 37]

- Figure 3: PCA Visualization. to explore the notion of objectness inside SDXL latent space, we visualize dimension-reduced self-attention feature maps from various layers across the network at timestep t = 500. We notice that although most layers do not exhibit a clear separation

between object instances, layer l52up displays a robust separation indicated by different ones hav-

ing distinct colors. Visualization across different timesteps is shown in the appendix (Figure 12).

Corrected Layout

Corrected Image

Original Image

Original Layout

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

FiveMelonsFiveBallsFourCupcakes

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Figure 4: Correcting under-generation. we show examples for the ReLayout correction of cases where SDXL generates less objects than specified in the prompt. It is evident that the generated layouts are natural and obey the same composition of the original generation, with the correct number of objects.

The input to the ReLayout is an object-layout described in Section 3.1, from which we initially infer the number of generated instances. Next, our ReLayout component takes one of two corrective actions based on the discrepancy between the generated and expected counts. In cases of over-generation, where more instances were generated than requested, ReLayout deterministically removes the smallest instances to achieve the desired cluster count. We find that this simple strategy produces appealing results. In cases of under-generation, a more intricate challenge arises: the ReLayout must insert new instances to the scene in a way that preserves the original scene structure. This process involves a sophisticated understanding of different object layouts—like the stark contrast between linearly arrayed bottles and the clustered arrangement of elephants—to seamlessly augment the layout. In Section 3.2.1, we detail our approach for handling under-generation. In cases where the number of instances is correct, the ReLayout maintains the initial layout.

#### 3.2.1 Handling Under-generation

To address under-generation issues, we train a U-Net model to predict a new layout, represented as a multi-channel mask, from an existing layout. In practice, each forward pass of the U-Net generates a mask with an additional instance. This process is applied in iterations until the mask reflects the correct number of instances. In what follows we provide detailed information on the architecture and training of our U-Net model.

Creating a training dataset. To train our ReLayout U-Net, we need a dataset of layout pairs with k and k+1 objects, that maintain the same scene composition. We begin with the empirical observation that slight variations in the object count specified in the prompt—while keeping the starting noise and the rest of the prompt consistent—typically results in images with similar layouts, as shown in Figure 5. This consistency is crucial as it allows us to generate a training dataset of layout pairs where each pair has a similar object composition, differing by only one object, thereby preserving the overall scene structure.

Following this observation, we generate a set of ~10K pairs of images of Ik and Ik+1, where each pair consists of images that differ by only one in the number of objects depicted. Each pair is generated with random fixed seeds and prompts that fit the same template, such as "a photo of two cats" versus "a photo of three cats”. To confirm that each image pair accurately represents an k and k + 1 object scenario, we extract object masks Mk and Mk+1 as described in Section 3.1, and verify the object count in one image is exactly one more than in its paired image. Overall, the final dataset for training consists of pairs of binary masks (Mk,Mk+1), representing the U-Net task of learning to generate a mask with k + 1 objects from a mask with k objects.

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

FourOrangesThreeOranges

ThreehorsesTwohorses

FourBoatsFiveBoats

TwoTiesThreeTies

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Figure 5: A training set for a ReLayout. We created pairs of images using SDXL, using the same seed and prompts that only differ by object count. We filtered out images that did not conform to the prompt, using the techniques described in Section 3.1. The resulting image pairs preserve the scene and layout except adding one object.

Matching objects. To train the U-Net, we need to establish a correspondence between each object i in Mk to its new position in Mk+1. We aim to find a matching that minimizes the shift in objects positions. We use the Hungarian algorithm [32] to find the optimal matching. More details in Appendix C.1.

Training the U-Net module. We trained the U-Net architecture by adapting it to handle 9 input channels – corresponding to the source tensor Mk ∈ {0,1}W×H×k with k objects, and output 10 channels – for the target tensor with k + 1 objects, to support counts up to 10. We optimized the U-Net parameters using two loss functions: (1) A Dice loss [33] between a predicted masks Mˆk+1 and the target masks Mk+1 of that object; and (2) Mask-to-mask overlap loss, designed to reduce the overlap between output masks of different instances. Specifically, this was computed as 1 − LDice between all pairs of predicted masks Mki+1, Mkj+1.

L = LDICE + λLoverlap (1) with λ being a weighing hyperparameter. Detailed definitions are provided in Appendix C.2.

Inference. At inference time, as a pre-processing step, we first add padding to input masks. After each iteration, we gradually and consistently increase the padding size around the original masks. This operation is beneficial when we need to add a large number of objects, as it creates a “zoom-out” effect, making space for new objects.

#### 3.3 CountGen Image: Layout-based Image Generation

Provided with correct object mask layouts (Section 3.2), our goal is to guide the image generation process to adhere to the input layout. Given a mask for each object in the desired layout, we apply an inference time optimization to match the layout in the generated image. To optimize object layouts at inference time, we propose a dual approach: object layout loss to encourage object creation in the foreground, i.e. pixels within the object masks, and self-attention masking to prevent object generation in the background.

Object layout loss. Consider the optimization of object placement within layouts using a weighted binary cross-entropy loss. Given c, the aggregated cross-attention scores, and m, a binary mask denoting object presence (foreground), the weighted binary cross-entropy loss is computed pixel-wise and is defined as follows:

L(c,m) = −

i

wi (mi log ci + (1 − mi)log(1 − ci)),

where ci is the cross-attention score at pixel i, mi is the value of the binary mask at pixel i, and wi is the weight assigned to each pixel i where wi = 10 if mi = 1, otherwise wi = 1.

Self-attention masking. The object-layout loss encourages objects to be generated in the foreground, but when applied on itself, generated objects may appear outside the object masks (Figure 7). To address this, we mask the self-attention connections between pixels in the background to pixels in the foreground. By disrupting these links, we stop the information flow from the objects to the rest of the image and prevent the model from forming objects in the background. Formally, at layer l and

timestep t, the masked self-attention St∗(l) is defined as:

St∗(l)[i,j] =

0 if i ∈ B(l) and j ∈ F(l), St(l)[i,j] otherwise.

where i and j are pixels indices, B(l) and F(l) represents the set of pixels belonging to the background and the foreground respectively, and St(l) is the self-attention map at layer l and timestep t. We discuss implementation details and computational efficiency in Appendix A.

### 4 Experiments

Compared methods. We compare CountGen against six baseline methods: (1) SDXL [9]; (2) Repeated Object: SDXL, with a modified prompt, where an object is repeated in the prompt the number of times it is required to generate, as in replacing “three cats” with “a cat and a cat and a cat”. This is a naive approach that parallels prompts like “A cat and a dog”. (3) Reason Out Your Layout: [2] uses GPT-3.5 [34] to generate layouts then trained an adapter to integrate it to SD-1.4 [35]; (4) DALL-E 3 [11]; (5) Random masks + BoundedAttn : generate a layout with the correct amount of clusters placed randomly in the image and apply a layout-guidance generation method on top; (6) Counting Guidance [14] : boost generation of SD with a counting network; Full details on how we used these baselines are given in Appendix B.2. We also compared our layout-to-image phase, CountGen-Image, described in section Section 3.3 with Bounded Attention [8].

Datasets. We evaluate our method and the baselines using two datasets. (1) T2I-CompbenchCount. A subset of T2I-Compbench [10], which is a benchmark for open-world compositional text-to-image generation. This subset specifically includes 218 prompts that specify a single object and its number (between 2 to 10). (2) CoCoCount (ours). We generate a dataset with automatic evaluation in mind. Specifically, we sample classes from COCO, which are more favorable to accurate and automatic detection by methods, like YOLOv9 [13]. We design simple prompts around these classes, with a number between 2 and 10. In total, there are 200 prompts with various classes, numbers and scenes (See full details in Appendix C.3).

Count accuracy evaluation. We evaluate the results of CountGen and the baselines using human and automatic evaluation method, which is standardized and reproducible. In both settings, we seek to identify if the number of instances generated by the object matches the request in the prompt.

Human evaluation. We quantified the count-accuracy of our method and baselines using human raters. Raters were asked for every image: (1) Is the object in the image?; (2) Are its instances well-formed?; (3) How many instances of the object are in the image? If the answer to question (1) or (2) is “no”, then we do not ask question (3). We provide details on the platform, rater selection and pay, and screenshots of the task in Appendix D.1.

Automatic evaluation. For automatic evaluation, we use the YOLOv9 model [13] with its default settings, as it represents the current state-of-the-art in the YOLO object detection benchmarks. To extract the number of objects in the image, we simply count the number of detected bounding-boxes corresponding to the target object.

Image quality evaluation. Forcing the diffusion model to obey the count in the text prompt is inevitably expected to reduce the naturalness and visual appeal of generated images, simply because more constraints are added. This effect has been observed in other studies using test-time optimization[23, 7]. We evaluate the image quality of CountGen by presenting human raters with two images, by CountGen and SDXL, and asking them to select whether one image is more natural and well-formed than the other or to indicate that both images are equally good.

### 5 Results

Quantitative results. Table 1 compares CountGen with competing baselines, showing its significant improvement over baselines in both CoCoCount and T2I-compbench-Count. Figure 8, and Figure 10 show CountGen outperforms all baselines for all values, except for two and three instances, where

###### CoCoCount T2I-Compbench [10] YOLOv9 Human Human

Model Accuracy Accuracy Accuracy

|SDXL Repeated Object Reason Out Your Layout DALL-E 3 Random masks + BoundedAttn Counting Guidance<br><br>|28 26 17 18 21 26 25 38<br>29 30 21 22<br>|29<br><br>14<br><br>15<br><br><br>36 35 22<br><br>|
|---|---|---|
|CountGen (ours)<br><br>|50 54|48|

Table 1: Generated count accuracy. Values are the percent of generated images that have the correct number of objects, for CoCoCount and T2ICompbench-Count.

DALL-E 3 slightly outperforms. We hypothesize that DALL-E 3 is larger and was trained on higherquality data than SDXL (our base model). In terms of image quality, out of 200 comparisons, in only 23 cases the majority of the raters preferred SDXL over our model. This indicates there is no significant loss of quality. We also include the confusion matrix figure of CountGen based on human evaluation in Appendix D.1.

“A photo of ﬁve cows on the road” “A photo of ﬁve bowls on the ground”

Repeated Object

CountGen (ours)

Reason Out Your Layout

DALL-E 3 SDXL CountGen Layout + seven birds Bounded Attention

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

on the ground”

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

“A photo of ten backpacks on the grass”

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

- Figure 6: Qualitative comparisons. We evaluated CountGen against DALLE 3, Reason Out Your Layout, SDXL, Repeated Object SDXL and Counten Layout + Bounded Attention. Our method successfully generates the correct number of objects, while other methods struggle in some or all of the examples. Additional results are shown in the supplemental material.

Qualitative results. Figure 6 shows examples of prompts and the images generated by various methods. In contrast to other methods, CountGen consistently generates the correct number of object instances.

### 6 Ablation Study

Contributions of CountGen-Layout and CountGen-Image. Table 2 quantifies the contribution of each of these components to the overall accuracy, by replacing it with a baseline alternative. Compared with a baseline (Random Masks + Bounded Attention) our first phase CountGen-Layout improves accuracy measured by people by 14% (from 30 to 44), and our second phase CountGen-Image by 12%. Together, the two components add up to improve accuracy by 21 points.

Layout guided generation ablation study. The second phase of our method, CountGen-Image, consists of two components: self-attention masking and object layout loss, as described at Section 3.3. To evaluate the contribution of each component, we deactivate it and compare the results. In Figure 7, we qualitatively observe that removing the layout loss leads to the objects scattering in the image, not constrained by the required mask. When removing the self-attention masking the objects tend to obey the mask unwanted object instances occur in the background.

w/o SA Masking

Layout w/o Loss Ours

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

- Figure 7: Component ablation. We ablate over two components of the layout-guided generation model: the optimization loss and Self-Attention Masking. Disabling the loss causes the generated image to deviate from the required layout. Removing the Self-Attention masking typically causes objects to appear outside of the layout foreground.

100

| |SDXL<br><br>Reason Layout<br><br>DALL-E 3<br><br>Object Repetition<br><br>Counting Guidance<br><br>CountGen (Ours)| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

80

Accuracy(%)

60

40

20

0

2 3 4 5 7 10 Number of Objects

Figure 8: Accuracy, as a function of the number of generated objects. Accuracy evaluated by human raters, over the set of 200 evaluation images. CountGen (blue) outperforms all methods for n > 3, and is on par with DALL-E 3 for 2 and 3 objects.

We confirm these observations quantitatively in Table 3, where we evaluate the adherence of the generated image to the input mask. We use YOLOv9 to detect the bounding boxes of generated objects and compare them to the input mask using three metrics: Precision is the percentage of bounding boxes that highly overlap (IOU>0.6) the mask (union of all object masks), Recall is the percentage of mask pixels that are covered by bounding boxes, and IOU is measured between the boxes and the mask. Our findings align with the qualitative observation: removing the self-attention masking leads to a worse precision score, meaning objects are generated in the background. Removing the layout loss leads to low recall and IOU, meaning poor adherence to the mask. CountGen-Image, employing both components, achieves balanced results by generating objects in accordance with the mask. Overall, these results emphasize the critical roles that both components in ensuring accurate adherence to the input mask.

Table 2: Model components Accuracy(%)

CoCoCount Compbench [10]

Text → Layout → YOLOv9 Human Human Layout Image Acc. Acc. Acc.

CountGen CountGen 50 54 48 CountGen B-Attn 40 42 40 Random CountGen 37 44 42 Random B-Attn 29 30 35

Table 3: CountGen-Layout Components. Error bars correspond to standard error of the means over 200 images.

Method Precision Recall IOU CountGen 59 ±3.1 82 ±2.5 52 ±1.2

- - SA masking 48 ±3.1 81 ±2.7 51 ±1.5

- - Layout loss 49 ±2.9 64 ±2.5 36 ±1.4

### 7 Limitations

Occasionally, our optimization (Section 3.3) results in multiple instances of an object in an area intended for just one by the layout. In other cases CountGen generates plain backgrounds compared to SDXL (Figure 11). Finally, the scope of our experiments may seem narrow, since we focus on generating scenes with up to 10 instances and a single object per prompt. Nevertheless, we have shown in Section 5 that even this setup is highly challenging to contemporary models, especially as the number of instances required to generate grows, as evident by the massive drop in performance, even for DALL-E 3 (see Figure 8).

### 8 Conclusions

The task of generating images that depict the number of requested objects correctly is a hard task. It requires models to capture “objectness", and obey global spatial constraints, at the same time they generate a well-formed natural image. Current text-to-image diffusion models perform poorly in this task (Table 1), especially when asked to generate more than three objects (Figure 8).

Our CountGen approach took three steps to address this task. First, we identified a notion of objectness from the self-attention layers of the diffusion model. Then, we trained a U-Net model that learned to correct the number of instances of an object in a given layout, whether it is removing or adding instances of an object such that the structure of the layout is preserved. Third, we developed a layout-guidance optimization method method to generate images from the corrected layout. Together, this approach almost doubled the counting accuracy from 26% in standard SDXL to 54% using our method applied to SDXL. We expect the lessons learned from this method, specifically the features that represent objectness and the process of learning to automatically fix a layout, to become useful in other problems of structured generation like spatial constraints in text-to-image models or spatio-temporal constraints in video generation.

### Acknowledgments and Disclosure of Funding

We thank Yuval Atzmon for useful discussions and for providing feedback on an earlier version of this manuscript. This study was supported by an equipment grant from the Israel Science Foundation (ISF 2332/18), and by a grant from the National committee of budgeting (VATAT).

### References

- [1] Omer Dahary, Or Patashnik, Kfir Aberman, and Daniel Cohen-Or. Be yourself: Bounded attention for multi-subject text-to-image generation. arXiv preprint arXiv:2403.16990, 2024.
- [2] Xiaohui Chen, Yongfei Liu, Yingxiang Yang, Jianbo Yuan, Quanzeng You, Li-Ping Liu, and Hongxia Yang. Reason out your layout: Evoking the layout master from large language models for text-to-image synthesis, 2023.
- [3] Weixi Feng, Wanrong Zhu, Tsu-Jui Fu, Varun Jampani, Arjun Reddy Akula, Xuehai He, S Basu, Xin Eric Wang, and William Yang Wang. LayoutGPT: Compositional visual planning and generation with large language models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.
- [4] Bogdan Alexe, Thomas Deselaers, and Vittorio Ferrari. Measuring the objectness of image windows. IEEE transactions on pattern analysis and machine intelligence, 34(11):2189–2202, 2012.
- [5] Weicheng Kuo, Bharath Hariharan, and Jitendra Malik. Deepbox: Learning objectness with convolutional networks. In Proceedings of the IEEE international conference on computer vision, pages 2479–2487, 2015.
- [6] Elizabeth S Spelke. Principles of object perception. Cognitive science, 14(1):29–56, 1990.
- [7] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attentionbased semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–10, 2023.
- [8] Omer Dahary, Or Patashnik, Kfir Aberman, and Daniel Cohen-Or. Be yourself: Bounded attention for multi-subject text-to-image generation, 2024.
- [9] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023.
- [10] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation, 2023.
- [11] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.
- [12] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.
- [13] Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao. Yolov9: Learning what you want to learn using programmable gradient information. arXiv preprint arXiv:2402.13616, 2024.
- [14] Wonjun Kang, Kevin Galim, and Hyung Il Koo. Counting guidance for high fidelity text-to-image synthesis, 2023.
- [15] Ruisu Zhang, Yicong Chen, and Kangwook Lee. Zero-shot improvement of object counting with CLIP. In R0-FoMo:Robustness of Few-shot and Zero-shot Learning in Large Foundation Models, 2023.
- [16] Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. Teaching clip to count to ten. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3170–3180, 2023.
- [17] Song Wen, Guian Fang, Renrui Zhang, Peng Gao, Hao Dong, and Dimitris N. Metaxas. Improving compositional text-to-image generation with large vision-language models, 2024.
- [18] Barak Battash, Amit Rozner, Lior Wolf, and Ofir Lindenbaum. Obtaining favorable layouts for multiple object generation, 2024.
- [19] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning text-to-image models using human feedback, 2023.
- [20] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for fine-tuning text-to-image diffusion models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.
- [21] Jiao Sun, Deqing Fu, Yushi Hu, Su Wang, Royi Rassin, Da-Cheng Juan, Dana Alon, Charles Herrmann, Sjoerd van Steenkiste, Ranjay Krishna, and Cyrus Rashtchian. Dreamsync: Aligning text-to-image generation with image understanding feedback, 2023.
- [22] Royi Rassin, Shauli Ravfogel, and Yoav Goldberg. DALLE-2 is seeing double: Flaws in word-to-concept mapping in Text2Image models. In Jasmijn Bastings, Yonatan Belinkov, Yanai Elazar, Dieuwke Hupkes, Naomi Saphra, and Sarah Wiegreffe, editors, Proceedings of the Fifth BlackboxNLP Workshop on Analyzing

- and Interpreting Neural Networks for NLP, pages 335–345, Abu Dhabi, United Arab Emirates (Hybrid), December 2022. Association for Computational Linguistics.
- [23] Royi Rassin, Eran Hirsch, Daniel Glickman, Shauli Ravfogel, Yoav Goldberg, and Gal Chechik. Linguistic binding in diffusion models: Enhancing attribute correspondence through attention map alignment. Advances in Neural Information Processing Systems, 36, 2024.
- [24] Quynh Phung, Songwei Ge, and Jia-Bin Huang. Grounded text-to-image synthesis with attention refocusing, 2023.
- [25] Hanan Gani, Shariq Farooq Bhat, Muzammal Naseer, Salman Khan, and Peter Wonka. LLM blueprint: Enabling text-to-image generation with complex and detailed prompts. In The Twelfth International Conference on Learning Representations, 2024.
- [26] Zhengyuan Yang, Jianfeng Wang, Zhe Gan, Linjie Li, Kevin Lin, Chenfei Wu, Nan Duan, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Reco: Region-controlled text-to-image generation. In CVPR, 2023.
- [27] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Prompt-toprompt image editing with cross-attention control. In The Eleventh International Conference on Learning Representations, 2023.
- [28] Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-to-image personalization, 2023.
- [29] Nobuyuki Otsu. A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1):62–66, 1979.
- [30] Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon. Trainingfree consistent text-to-image generation. arXiv preprint arXiv:2402.03286, 2024.
- [31] Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. A density-based algorithm for discovering clusters in large spatial databases with noise. In Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, KDD’96, page 226–231. AAAI Press, 1996.
- [32] H. W. Kuhn. The hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2(1-2):83–97, 1955.
- [33] Carole H Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, and M Jorge Cardoso. Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support: Third International Workshop, DLMIA 2017, and 7th International Workshop, ML-CDS 2017, Held in Conjunction with MICCAI 2017, Québec City, QC, Canada, September 14, Proceedings 3, pages 240–248. Springer, 2017.
- [34] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020.
- [35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [36] Mateusz Buda, Ashirbani Saha, and Maciej A Mazurowski. Association of genomic subtypes of lowergrade gliomas with shape features automatically extracted by a deep learning algorithm. Computers in Biology and Medicine, 109, 2019.

### A Appendix / Supplemental Material

Efficiency. CountGen takes ~36 seconds on average to generate an image on a single A100 80GB. We arrive at this number by iterating over CoCoCount. To put in context, Bounded-Attention[8] takes ~55 seconds and requires bounding boxes as input, while our solution is not input-dependent. SDXL takes ~8 seconds.

Compute. All experiments were conducted over a period of a week on a single A100 80GB.

### B Implementation Details & Reproducibility

#### B.1 CountGen

Layout guided generation. In our implementation, the self-attention masking is applied at timesteps t ∈ [1000,900], in the decoder layers of the U-Net. The object layout loss is applied at timesteps t ∈ [1000,500], in all layers of the U-Net. Our pipeline used the Attend-and-Excite [7] code base as a starting point.

ReLayout The ReLayout U-Net was built upon the U-Net Implementation of [36]. We trained the U-Net with a learning-rate of 8e-6, a batch-size of size 32 and the Adam optimizer. The intersection penalty is set to 0.25 and the Dice penalty is set to 1. During training we apply a horizontal flip augmentation across all masks, and shuffle augmentation where we randomly re-arrange the input channels.

Instance identification. In the DBSCAN clustering algorithm, we used a dynamic epsilon value in the range of [0.1,0.2] and used cosine similarity as the distance metric.

#### B.2 Compared Methods

Each prompt in CoCoCount and T2I-CompBench-Count was assigned a unique random seed and was used by all baselines and CountGen.

We compared CountGen with the following baselines: SDXL [9]. We used the stable-diffusion-xl-base-1.0 model.

Repeated Object In this baseline, we used the same model and seeds as in SDXL but modified the prompts. We repeated the object in the prompt as many times as the target count. For example, “a photo of three cats” was changed to “a photo of a cat and a cat and a cat”.

Reason Out Your Layout [2]. This baseline has two main steps. First, it leverages GPT-3.5-turbo to generate spatially reasonable coordinates to be used as a bounding box for each instance of an object (i.e., “a photo of three cats” results in three bounding boxes, one for each cat). Second, it uses the generated layout to guide the generation process. We followed the prompt used by the authors, however, it seems that the responses by GPT-3.5-turbo and the author’s parser are not completely cohesive, which at times leads to zero bounding boxes. We count such cases as failures. For the CoCoCount experiment, it successfully generated 134/200 images, and for T2I-CompBench-Count, just 89/200. Failures were counted as errors in the reported results. We did not need to make changes to the code to run it.

DALL-E 3 [11]. We used the OpenAI API interface for the DALL-E 3 model with “standard” image quality. We did not use seeds in this baseline.

Random masks + BoundedAttn [8]. Given a prompt with a required number of object instances, we create a corresponding layout with the correct number of objects randomly placed in the image plane in a way they do not intersect one another. Then we used Bounded Attention to generate an image condinitioned on that layout.

Counting Guidance [14]. The authors provided us with their code. We did not need to change it to run our experiments.

100

| |SDXL<br><br>Reason Layout<br><br>DALL-E 3<br><br>Object Repetition<br><br>Counting Guidance<br><br>CountGen (Ours)| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

80

Accuracy(%)

60

40

20

0

2 3 4 5 7 10 Number of Objects

Figure 9: Accuracy, as a function of the number of generated objects. Accuracy evaluated by human raters, over the set of 200 evaluation images. CountGen (blue) outperforms all methods for n > 3, and is on par with DALL-E 3 for 2 and 3 objects.

100

| |SDXL<br><br>Reason Layout<br><br>DALL-E 3<br><br>Object Repetition<br><br>Counting Guidance<br><br>CountGen (Ours)|
|---|---|
| | |
| | |
| | |
| | |

80

Accuracy(%)

60

40

20

0

2 3 4 5 7 10 Number of Objects

Figure 10: Accuracy, as a function of the number of generated objects. Accuracy evaluated by YOLOv9, over the set of 200 evaluation images. Here, CountGen (blue) outperforms all methods.

###### Merged Instances Duplicated Instances Plain Background

“A photo of four cats”

“A photo of ﬁve apples”

“A photo of ﬁve backpacks”

[Figure 94]

[Figure 95]

[Figure 96]

CountGenSDXL

LayoutCountGen

CountGenLayout

[Figure 97]

[Figure 98]

[Figure 99]

Figure 11: Limitations. Failure modes of CountGen.

### C Extended Details on CountGen

#### C.1 ReLayout: Matching Objects

We aim to understand how Mki transitions to Mki+1. Specifically, for each object i ∈ 1,...,k in the original Mk layout, our ReLayout objective is designed to predict how the corresponding mask Mki changes in the new image Mki+1, and additionally where to insert the added object k + 1. This design encourages the model to slightly modify existing objects while preserving spatial and shape consistency across the images.

t900

t800 t900 t800 t700

t700

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

t500 t300 t100

t300 t100

t500

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Figure 12: PCA visualization across timestamps. to explore the notion of objectness inside SDXL latent space, we visualize a dimension-reduced self-attention feature maps across different timestamps range from t = 900 to t = 100. Initially, up to timestamp t = 500, clear separation is not observed in some objects (e.g., some eggs appear in similar colors). However, starting from t = 500, a distinct separation emerges, with each object clearly distinguished by different shades.

To this end, we first have to establish a correspondence between the object masks (Mk,Mk+1). We employ the Hungarian algorithm [32] to find the optimal one-to-one matching between masks in the two images based on the overlap and similarity of the masks. This algorithm effectively pairs each object in Mk with a corresponding object in Mk+1. The object in Mk+1 that remains unmatched represents the additional object introduced in the new image, providing a clear identifier for the increment in object count.

#### C.2 Losses for Training the ReLayout

We use two training losses: Dice Loss: measures the overlap between the predicted mask and target mask across all channels containing objects:

2 p∈P Mki+1(p) · Mk∗+1i (p) p∈P(Mki+1(p) + M∗i k+1(p))

LiDice = 1 −

(2)

Here, p iterates over all pixels P in the masks, and i ranges over all possible object channels. For all k + 1 channels, the total dice loss is:

k+1

LiDice (3)

LDice =

i=1

Intersection Loss: To ensure distinctiveness among the predicted masks and to minimize overlap between different object masks, the intersection loss for all possible pairs of different masks in the output mask containing objects is given by:

2 p∈P Mki+1(p) · Mkj+1(p) p∈P(Mki+1(p) + Mkj+1(p))

k+1

k+1

(4)

LOverlap =

i=1

j̸=i

#### C.3 Datasets

CoCoCount. To create this set, we first select at random 20 classes from MSCOCO [12]. We then sample from six counting categories: 2,3,4,5,7, and 10. The two and three categories contain 34 samples, while the rest contain 33. Our prompts consist of the pattern “a photo of {number} {object}” with an optional variation of scenes: “on the grass”, “on the road”, or “on the ground”, which we incorporate for 50% of the prompts, also randomly. In total, we have 200 prompts.

Below are the complete lists from which elements were chosen:

- • Objects: ’car’, ’airplane’, ’bird’, ’cat’, ’dog’, ’horse’, ’sheep’, ’cow’, ’elephant’, ’bear’, ’backpack’, ’tie’, ’sports ball’, ’baseball glove’, ’cup’, ’bowl’, ’apple’, ’donut’, ’cell phone’, ’clock’

- • Counting Categories: ’two’, ’three’, ’four’, ’five’, ’seven’, ’ten’
- • Scenes: ’on the grass’, ’on the road’, ’on the ground’

- D Evaluation Automatic evaluation. We use the implementation by Ultralytics YOLO of YOLOv9e (large).

#### D.1 Human Evaluation

We use the Amazon Mechanical Turk platform and ensure the evaluation is of high quality by hiring raters with a minimum of 5,000 approved HITs and an approval rate exceeding 98%. Each example was shown to three raters and the majority selection was taken. The compensation was $15 per hour. Screenshots of the count precision task can be viewed in Figure 13, Figure 14, Figure 15 and the image fidelity task in Figure 16.

[Figure 114]

- Figure 13: Instructions for the Image Evaluation Task - Part 1.

[Figure 115]

- Figure 14: Instructions for the Image Evaluation Task - Part 2.

[Figure 116]

##### Figure 15: Example task to count the number of objects and assess their well-formedness.

[Figure 117]

##### Figure 16: Example task to compare image fidelity based on prompt matching and naturalness.

[Figure 118]

##### Figure 17: Confusion matrix of human evaluation (Section 4) of the count accuracy experiment for CountGen.

