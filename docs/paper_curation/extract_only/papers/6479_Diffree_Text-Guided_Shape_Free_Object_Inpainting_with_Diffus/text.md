# arXiv:2407.16982v1[cs.CV]24Jul2024

## Diffree: Text-Guided Shape Free Object Inpainting with Diffusion Model

Lirui Zhao1,2†, Tianshuo Yang2,3†, Wenqi Shao2†‡, Yuxin Zhang1, Yu Qiao2, Ping Luo2,3, Kaipeng Zhang2‡⋆, and Rongrong Ji1⋆

1Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, Xiamen University

2OpenGVLab, Shanghai AI Laboratory 3The University of Hong Kong https://github.com/OpenGVLab/Diffree

[Figure 1]

[Figure 2]

[Figure 3]

(1) (2) …… (16)

(8) Pizza

(7) Table

(1) Sofa

(2) Curtain

(3) Armchair

(4) Cabinet

(5) Fireplace

(6) Rug

Add

(16) Teddy bear

(15) Chandelier

(9) A pot of flowers

(10) Cat

(11) Floor lamp

(12) Blanket

(13) Trash can

(14) Painting

[Figure 4]

[Figure 5]

[Figure 6]

(1) … (7) …(16)

(1) Fabric sofa

(2) Table

(3) Cute dog

(4) Photo frame

(5) Carpet

(6) Potted

(7) Poster

(8) Teddy bear

Add

(9) Wall cabinet

(10) Rug

(11) birthday cake

(12) Ice cream

(13) Toy car

(14) Floor lamp

(15) Curtain

(16) Dragon

Fig. 1: Our approach iteratively generates inpainting results. The objects from text guided is reasonably added in images while ensuring the background consistency.

Abstract. This paper addresses an important problem of object addition for images with only text guidance. It is challenging because the new object must be integrated seamlessly into the image with consistent visual context, such as lighting, texture, and spatial location. While existing text-guided image inpainting methods can add objects, they either fail to preserve the background consistency or involve cumbersome human intervention in specifying bounding boxes or user-scribbled masks. To tackle this challenge, we introduce Diffree, a Text-to-Image (T2I) model that facilitates text-guided object addition with only text control. To this end, we curate OABench, an exquisite synthetic dataset by removing objects with advanced image inpainting techniques. OABench comprises 74K real-world tuples of an original image, an inpainted image with the object removed, an object mask, and object descriptions. Trained on OABench using the Stable Diffusion model with an additional mask prediction module, Diffree uniquely predicts the position of the new object and achieves object addition with guidance from only text. Extensive experiments demonstrate that Diffree excels in adding new objects with a high success rate while maintaining background consistency, spatial appropriateness, and object relevance and quality.

Keywords: Image inpainting · Text-guided image editing †Equal contribution ‡Project lead ⋆Corresponding author

### 1 Introduction

With the recent remarkable success of Text-to-Image (T2I) models (e.g., Stable Diffusion [21], Midjourney [17], and DALL-E [2,23]), creators can quickly generate high-quality images with text guidance. The rapid development has driven various text-guided image editing techniques [3,6,27,35,37]. Among these techniques, text-guided object addition which inserts an object into the given image has attracted much attention due to its diverse applications, such as advertisement creation, visual try-on, and renovation visualization. While important, object addition is challenging because the object must be integrated seamlessly into the image with consistent visual context, such as lighting, texture, and spatial location.

Existing techniques for object addition in images can be broadly categorized into mask-guided and text-guided approaches (Fig. 2). Mask-guided algorithms typically require the specification of a region where the new object will be inserted. For example, traditional image inpainting methods [1,15,20,30,34] focus on seamlessly filling user-defined masks within an image to match the surrounding context. Recent advancements, such as PowerPaint [38], have effectively incorporated objects into images given their shape and textual descriptions while maintaining background consistency. However, manually delineating an ideal region for all objects, considering shape, size, and position, can be labor-intensive and typically requires drawing skills or professional knowledge. On the other hand, text-guided object addition methods, such as InstructPix2Pix [3], attempt to add new objects using only text-based instructions. Despite this, these methods have a low success rate and often result in background inconsistencies, as demonstrated in Fig. 2 and Fig. 9. Additionally, when employing InstructPix2Pix for iterative object addition, the quality of the inpainted image tends to degrade progressively with each step, as illustrated in Fig. 10.

To tackle the above challenges, we introduce Diffree, a diffusion model with an additional object mask predictor module that can predict an ideal mask for a candidate inpainting object and achieve shape-free object addition with only text guidance. Compared with previous works [3, 6, 33, 38], our Diffree has three appealing properties. First, Diffree can achieve impressive text-guided object addition results while keeping the background unchanged. In contrast, previous text-guided methods [3] struggle to guarantee this. Second, Diffree does not require additional mask input, which is necessary for traditional maskguided methods [33]. In real scenarios, high-quality masks are hard to obtain. Third, Diffree can generate the instance mask and thus can be further combined with various existing works [4,19] to develop exciting applications. For example, Diffree can achieve image-prompted object addition when combined with AnyDoor [4] and plan to add objects suggested by GPT4V [19], as shown in Fig. 11.

Towards high-quality text-guided object addition, we curate a synthetic dataset

named Object Addition Benchmark (OABench) which consists of 74K real-world tuples including an original image, an inpainted image, a mask image of the object, and an object description. The data curation process is illustrated in

###### Text-guided methods

Mask-guided methods

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

bounding-box mask

Input: brown blanket

Input: Add brown blanket

InstructPix2pix

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

scribble mask

Diffree (ours)

###### Input:

brown blanket PowerPaint SD Inpainting

Fig. 2: Qualitative comparisons of Diffree and different kinds of methods.

Fig. 5. Note that object addition can be deemed as the inverse process of object removal. We build OABench by removing objects in the image using advanced image inpainting algorithms such as PowerPaint [38]. In this way, we can obtain an original image containing the object, an inpainted image with the object removed, the object mask, and the object descriptions. We use instance segmentation dataset COCO [7, 14] as the source data, which has two benefits. First, the source image captures comprehensive natural scenes where the location and shape of one individual object often exhibit intrinsic alignment with the overall scene. It helps guarantee the reasonability of new objects’ location. For instance, a monitor is commonly situated behind computer peripherals. Second, the ground-truth mask of the object already exists in the instance segmentation dataset, which can be directly utilized in removing objects with background consistency preserved. By contrast, InstructPix2Pix [3] collects image pairs using proprietary T2I model [24] under prompt pair with subtle modifications. While this approach maintains new objects’ reasonability, it poses difficulties in preserving background consistency.

bounding-box mask， scribble mask

With OABench, Diffree is trained to predict masks and images containing the new object given the original image and object text description. Thanks to the extensive coverage of objects in natural scenes in OABench, Diffree can add various objects to the same image while matching the visual context well as shown in Fig. 3. Moreover, Diffree can iteratively insert objects into a single image while preserving the background consistency using the generated mask as shown in Fig. 1 and Fig. 4.

For evaluation, we propose a set of evaluation rules through existing metrics [8,9,19,33,36], including consistency of background, reasonableness of object location, quality, diversity and correlation of generated object, and success rate. Extensive experiments show that Diffree performs better in object addition than previous mask-guided and text-guided techniques. For instance, Diffree obtains a significantly higher success rate than InstructPix2Pix. For successful cases, Diffree still outperforms InstructPix2Pix in various quantitative metrics.

The contributions of this work are three-fold. 1) We proposed Diffree, a model that can achieve text-guided shape-free object addition to free users from drawing the appropriate mask of objects. The inpainted image from Diffree includes the new objects with reasonable shapes and consistent visual context.

- 2) We introduced OABench, an exquisite synthetic dataset for object addition.

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Beach umbrella Paragliding Boat Coconut palm Ship

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Tractor Hot air balloon Eagle Car Wooden house

- Fig. 3: Diffree adds objects to the same image, with different spatial relationships.

OABench comprises 74K real-world training data for the task of object addition.

- 3) We evaluate this task with a set of rules for comprehensive assessment. Extensive experiments demonstrate the effectiveness of Diffree. For example, Diffree achieves a high success rate (e.g., 98.5% in COCO) and superior unified score (e.g., 38.92 versus 4.48) compared with other methods.

### 2 Related Work

Text-to-Image Diffusion Models Recently, text-to-image (T2I) diffusion mod-

els [2,18,23], have shown exceptional capability in image generation quality and extraordinary proficiency in accurately following text prompts, under the dual support of large-scale text-image dataset [26] and model optimizations [5,10,24]. GLIDE [18] incorporated text conditions into the diffusion model and empirically showed that leveraging classifier guidance leads to visually appealing outcomes. DALLE-2 [23] enhances text-image alignment via CLIP [22] joint feature space, DALLE-3 [2] further improves the prompt following abilities by training on highly descriptive generated image captions. Stable Diffusion [24], which is well-established and widely adopted, garners significant attention and application within and beyond the research community. Given that T2I models generate comprehensive images from text prompts, even minor alterations in prompts can result in substantial changes to the resultant image [3]. Consequently, there has been an increased focus not only on T2I generation but also on image editing based on additional conditions such as text inputs, masks, et al.

Text-Guided Image Editing The effectiveness of the text-guided image editing methos [3,6,27,35] largely depends on the composition of its dataset and how it is collected. InstructPix2Pix [3] combines two large pretrained models, a large language model [16] and a T2I model [24], to generate a dataset for training a diffusion model to follow written image editing text prompts. Its innovative data collection method allows InstructPix2Pix to follow instructions and shows amazing effects, while makes its consistency is difficult to guarantee due to both input and output are generated by the T2I model. InstructDiffusion [6] treats all computer vision tasks as image generation with multiple output formats, and aligns these tasks with human instructions. Emu Edit [27] adapt its architecture for multi-task learning and train it an unprecedented range of tasks formulated

as generative tasks, and displays strong and diverse results, especially in the combination of multiple tasks. Emu Edit generate the output image and apply the mask-based attention to generate input iamge for object addtion task. MagicBrush [35] introduces a manually annotated dataset which both input and output are generated by the T2I model. The image editing performance of fine-tuneing InstructPix2Pix on MagicBrush shows better. Unlike the previous methods, we propose a novel and easily expandable collection method, thanks to the existing instance segmentation dataset, we use real images as output and synthetic images without a specific object as input. Our work closely relates to the concurrent work PIPE [32], which independently explores similar concepts and methodologies. Both studies involve removing objects to collect an object addition dataset and train a diffusion model for text-guided object addition. Our approach additionally trains an Object Mask Predictor (OMP) module to predict the mask of objects. We believe that the concurrent exploration of these ideas underscores the significance and timeliness of this research direction.

Mask-Guided Image Inpainting Mask-guided image inpainting methods [4, 31,33,38] alter the image in specific areas under addition conditions (e.g., text), while maintaining the background in its original state. SmartBrush [33] achieves precise object inpainting guided by text and mask through a novel training and sampling strategy . Imagen Editor [31], finetuned on Imagen [25], captures fine details in the input image by conditioning the cascaded pipeline to accomplish precise image inpainting through a user defined mask and text prompts. AnyDoor [4] employs a discriminative ID extractor and a frequency-aware detail extractor to characterize the target object, thereby facilitating effective object addition given an area and corresponding object image. Powerpaint [38] demonstrates superior performance on various inpainting benchmarks attributed to the introduction of learnable tokens to distinguish different tasks. Although these methods have achieved amazing image inpainting effects, their commonality is the need for a mask. For ordinary users, drawing an object mask with appropriate shape, size, and aspect ratio, corresponding accurately to the object and image, presents an unignorable challenge.

### 3 Methodology

Given an image and the object description, our goal is to add the object to the image while preserving the background consistency. Following this, we initially introduce OABench, an synthetic dataset for this task, comprising image-text pairs (as input) with corresponding object masks and images containing the object (as output). We provide an overview of our data collection pipeline in Sec. 3.1 and. We next present Diffree, an architecture amalgamating a Stable Diffusion model with an Object Mask Predictor (OMP) module in Sec. 3.2. The evaluation procedure is presented in Sec. 3.3.

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

+ Mouse + Water glasses + Keyboard + Notebook + Photo frame

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

+ Plate + Sandwich + Berry jam + Glass of wine + Man

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

+ Bed + Rug + Photo frame + Carpet + Bedside table

- Fig. 4: Diffree iteratively generates results. Objects added later can relate to the earlier.

##### 3.1 OABench

We combine existing instance segmentation dataset [7, 14] with powerful image inpainting method [38] to generate the OABench. Unlike other instructions follow methods [3,35], generating both data pairs using existing text-to-image (T2I) models [23, 24] with prompt pairs and filtering, we use the real image with object to synthesize the image without the object, as depicted in Fig. 5. This can greatly ensure the consistency of the background as in other image inpainting methods [38] that require masks. Furthermore, an object in the real image naturally aligns with its background, i.e., it is appropriate for generating the corresponding image without the same object. In the following sections, we describe in detail the three steps of this process.

Collection and Filtering We gather and refine instances suitable for image inpainting by applying a set of rules from the LVIS dataset [7], a large instance segmentation dataset annotated for COCO [14] dataset. As depicted in Fig. 5, in images containing multiple instances, we enforce size constraints to exclude instances that are too big or too small (typically related to object components or background elements like buttons on clothing or rivers). Subsequently, incomplete instances are filtered out using edge detection and integrity assessments. Instances that are partially obscured are identified through cavity inspection, iterative IOU algorithm application, and common part comparison among various instances. Additionally, objects with exceptionally high aspect ratios, which tend to yield subpar inpainting outcomes, are also eliminated.

Data Synthesis We next employ a powerful image inpainting method, PowerPaint [38], to eliminate specific instances obtained in the preceding stage. There-

###### LVIS COCO OABench

[Figure 47]

PostProcessing

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

…….

Text Encoder

……

Image Encoder

“Computer monitor”

[Figure 55]

Synthesis Filtering

[Figure 56]

Local Region

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Image Inpainting

[object name]

[Figure 61]

[Figure 62]

Aspect ratio Non-occlusion

Size limit Non-edge contact Integrity Non- hollow

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Non-background

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

……

……

Fig. 5: The data collection process of OABench.

fore, we can generate a synthetic image without specific objects with background consistency with the original image. simultaneously, the original image, object mask, and corresponding name can be extracted from the LVIS and COCO.

Post-Processing In the post-processing stage, we filter out the results with poor effects in image inpainting. For some special cases (e.g., one of many dense and adjacent small cakes), image inpainting cannot effectively remove objects due to the complexity of the background. Thus we calculate the clip score [8] using the object name and the region of the inpainted image, setting a threshold to remove images with higher scores which are deemed suboptimal. Finally, OABench includes 74,774 high-quality data pairs, each data pair includes a synthetic image and object caption as input, object masks and original images as output.

##### 3.2 Diffree

For an image x and a text prompt d, Diffree predicts a binary mask m that specifies the region in x and generates an image x˜. The masked region x˜⊙m aligns with the text prompt d. To this end, Diffree is instantiated with a pre-trained T2I diffusion model (e.g. Stable Diffusion [24]) with a object mask prediction (OMP) module as shown in Fig. 6.

Diffusion Model learns to generate data samples by iteratively applying denoising autoencoders that estimate the score function [29] of a given data distribution [28]. Stable Diffusion [24] apply them in the latent space of powerful pre-trained variational autoencoder [12], including encoder E and decoder D, to reduce computing resources while maintaining quality and flexibility. Stable Diffusion encompasses both forward and reverse processes. Given an image x˜, the forward process adds noise to the encoded latent z˜ = E(˜x):

z˜t = √α¯tz˜ + √1 − α¯tϵ,ϵ ∼ N(0,I) (1)

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Mask 𝑚

𝒎 𝒎 𝒎 𝒎

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

###### ℰ

[Figure 83]

𝑧̃𝒕 𝟏 z𝒕 𝑧̃

𝑧 ̃𝒕 𝟏

…

Text prompt 𝑑 Time Step 𝑡

Cross-Attn. & Time Embed. Cross-Attn. & Time Embed.

Image 𝑥

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

[Figure 84]

[Figure 85]

#### …

[Figure 86]

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

𝒟

𝑧̃𝒕 𝒛

[Figure 87]

ℰ

Object Mask Predictor (OMP) module

Object Mask Predictor (OMP) module

Input image 𝑥

Predicted mask 𝑚 and inpainted image 𝑥

[Figure 88]

[Figure 89]

[Figure 90]

Object Mask Predictor (OMP) module

Spatial Transformer

Conv2D

ResBlock ResBlock Conv2D

Fig. 6: Overview of Diffree framework.

where z˜t is the noisy latent at timestep t, α¯t denotes the associated noise level.

In the reverse process, we learn a network ϵθ that predicts the noise added to the noisy latent z˜t, conditioned on both the image x and text d. To fine-tune Stable Diffusion for inpainting, we extend the channel of the first convolution layer to concatenate latent z = E(x) of image x with z˜t. This allows Diffree to generate images by denoising step by step from Gaussian noise concatenated with the latent of the input image. At the same time, the denoising process is guided by the associated feature Enctxt(d) of text d encoded through the CLIP text encoder [22]. The network ϵθ is optimized by minimizing the following objective function:

LDM = EE(˜x),E(x),d,ϵ∼N(0,I),t ∥ϵ − ϵθ(˜zt,z,Enctxt(d),t)∥22 . (2)

OMP Module and diffusion model are trained simultaneously and used to predict the binary mask m. The OMP module comprises two convolutional layers, two ResBlocks, and an attention block, as illustrated in Fig. 6. First, we calculate the predicted noise-free latent o˜t using the output of the diffusion model:

√1 − α¯tϵθ(˜zt,z,Enctxt(d),t) √α¯t

z˜t −

. (3)

o˜t =

Here, the concatenation of z = E(x) with o˜t serves as inputs to the OMP module. The gradient of o˜t is detached to optimize the two models without affecting each other. We conduct bilinear interpolation downsampling on the mask m to obtain m′, preserving its size identical to the input latent. The OMP module’s network τθ is optimized according to the following objective function:

LOMP = EE(˜x),E(x),d,m ∥m′ − τθ(˜ot,z)∥22 . (4)

99 79 59

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

39 19 0

Add a pair of sunglasses Different steps results of OMP Module Final output

- Fig. 7: Visualization of masks from OMP at different steps of diffusion inference process (total 100 steps). The mask of added objects can be acquired in the very beginning.

It is noteworthy that the OMP module can predict the mask through the reverse process of diffusion rather than after it, as o˜t is available at each step, enabling mask prediction in the initial steps, as illustrated in Fig. 7.

We train both the diffusion model and the OMP module simultaneously. Combining Eqs. (2) and (4), our final training objective can be expressed as follows:

L = LDM + λLOPS, (5) where λ is a hyper-parameter which balances the two losses.

Classifier-free Guidance Classifier-free diffusion guidance [11] is a method that involves the joint training of a conditional diffusion model and an unconditional diffusion model. By combining the output score estimates from both models, this approach achieves a balance between sample quality and diversity. Training for the unconditional diffusion model is achieved by fixing the conditioning value to a null variable intermittently throughout the training process. We follow the approach of Brook et al. [3] by stochastically and independently defining our input conditions x and d as null variables with a probability of 5%.

##### 3.3 Evaluation Metric

Due to the absence of robust quantitative metrics for shape-free object inpainting except the success rate, we propose a set of evaluation rules leveraging exits metrics [8,9,19,33,36] to evaluate different methods in different aspects.

We first randomly select and manually inspect 1,000 evaluation data pairs from COCO [14] and OpenImages [13] independently to ensure the validity of the object in the image and generalizability of the evaluation dataset. Each data pair comprises an original image xori, a text prompt of an object d, and an inpainted image x. The resulting output image xoutput and the corresponding object mask moutput are outcomes derived from distinct methods.

[Figure 99]

[Figure 100]

As an Object Placement Evaluator, your

𝑻:

{ "score": 3, "reason": "The car placement in the scene appears somewhat suitable, as cars can be driven onto wide open spaces like the one depicted. However, it slightly lacks coherence with the setting due to the car's scale appearing somewhat off when compared to the surrounding environment, creating a slight sense of disproportion. ” }

primary function is to assess the rationality and integration of an object‘s placement within a scene, using a pair of images: an original version and a modified version with the object added. Users will provide these images alongside the description of the object added.Your task is to evaluate whether the object's placement conforms to the physical laws and aligns with the overall context, and integration within the scene, strictly providing your evaluation in a dict format. This evaluation should rate the object's placement suitability on a scale from 1 to 5, where 1 signifies poor integration and 5 excellent integration, e.g., {"score": 5, "reason": "The reason."}. It's crucial to focus solely on offering insightful feedback through the specified dictionary output, without including any extraneous content.

[Figure 101]

[Figure 102]

[Figure 103]

{ "score": 1,

"reason": "The car is floating in the air which defies the laws of physics as it does not show any support or means to stay afloat, making the placement highly unrealistic in the context of the physical world.” }

[Figure 104]

- Fig. 8: GPT4V shows good distinguish ability in the reasonableness between objects.

Background Consistency We adapt LPIPS [36], a widely adopted and robust metric for assessing the similarity between images, to evaluate this aspect:

scon (x,xoutput,moutput) = LPIPS(x,x ⊙ moutput + xoutput ⊙ (1 − moutput)).

(6)

Location Reasonableness Assessing the reasonableness of the object’s location is a challenging task due to its inherent subjectivity. Surprisingly, we note GPT4V [19] demonstrates strong discriminative abilities in assessing variations and evaluating different locations by providing x, d, xoutput and an instruction T as illustrated in Fig. 8. GPT4V rates the appropriateness of the object’s position on a scale from 1 to 5, while also providing justifications for these ratings:

srea (x,xoutput,d,T) = GPT4V (x,xoutput,d,T) (7)

Object Correlation To quantify this relationship, we utilize CLIP Score [8], a metric to assess the correlation between text and image, by calculating the cosine similarity of their embeddings from CLIP [22]. we measure CLIP Score between the object area of xoutput and d, which is referred to as “Local CLIP Score”:

scor(d,xoutput,moutput) = CLIPScore(d,Local(xoutput,moutput)). (8)

where Local(x,m) denotes obtaining a cropped region from x using m. To mitigate influences from background or mask shape, we compute an average of two Local CLIP Scores (one with background removal and another without).

Object Quality and Diversity Following [33], we employ Local FID, measuring Fréchet Inception Distance (FID) [9] on the local regions, to evaluate the quality and diversity of generated object:

output||2+ Tr(ΣLX

sqd(LXorg,LXoutput) =||µLX

org − µLX

- 1

- 2 )

output − 2 ∗ (ΣLX

org ∗ ΣLX

+ ΣLX

)

org

output

###### (9) where LXorg and LXoutput respectively denote the sets comprising all local regions of the original images and output images, µ and Σ represent the mean and variance of the feature vectors obtained through a particular network [9].

Background Error 30.5%

Successful Cases 17.4%

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Both Error 1.6%

[Figure 111]

[Figure 112]

Examples

Foreground Error 50.5%

(a) Add a street sign Add a cat

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Both Error 1.3%

Foreground Error 57.4% Successful Cases 18.9%

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Examples

Background Error Add a traffic light Add a earphone 22.4%

(b)

- Fig. 9: InstructionPix2Pix’s result statistics in object addition. (a): COCO, (b): OpenImages. Foreground Error denotes failing to add objects or transforming existing objects, Background Error denotes inconsistent background.

Unified Metric Drawing upon the evaluation metrics delineated above (Eqs. (6) to (9)), we compute a unified score to holistically assess text-guided shape-free object inpainting. We treat the derivative of inverse metric results (LPIPS and Local FID) as positive metrics and normalized the outcomes across different methods for each metric. Ultimately, we average these normalized scores and multiply them by the success rate as a unified score. The Unified metric not only considers success rate but also focuses on quantitative performances.

### 4 Experiment

We comprehensively evaluated our model, Diffree, by conducting experiments on two benchmark datasets: COCO [14], and OpenImages [13]. Given the distinct input-output characteristics of our method compared to previous approaches, a quantitative comparison proves challenging. We align previous methods by adding auxiliary conditions, as depicted in Sec. 4.1, and provide quantitative comparison results (Sec. 4.2) to prove the effectiveness of Diffree more intuitively. We then showcase visualizations of generated images and give corresponding analyses to offer an intuitive assessment of Diffree’s capabilities and comparisons in Sec. 4.3. finally, we demonstrate some applications to prove that Diffree is highly compatible with existing methods (Sec. 4.4).

##### 4.1 Experimental Settings

Training Setups we employ OABench to train Diffree, initializing the diffusion model with the Stable Diffusion 1.5 [24] weights. We set λ = 2 in Eq. (5) and set a batch size of 256. Our model was trained around 10K steps on 8 A100 GPUs.

Evaluation Datasets and Metrics As outlined in Sec. 3.3, we employ four metrics (LPIPS [36], GPT4V [19], Local CLIP Score and Local FID [33]) alongside the unified metric for evaluation on COCO [14] and OpenImages [13].

Table 1: Main results on COCO and OpenImages. *: only calculate the successful cases’ results. †: use the masks from our Diffree as PowerPaint’s input.

InstructPix2pix [3] PowerPaint [38] Diffree

Success rate 17.4 N/A 98.5 LPIPS ↓ 0.11* 0.06 0.07 GPT4V Score ↑ 3.13* N/A 3.47 Local CLIP Score ↑ 29.30* 28.74 28.96 Local FID ↓ 156.25* 58.08 57.43

COCO [14]

- Unified Metric ↑ 4.48 37.20† 35.92

OpenImages [13]

Success rate 18.9 N/A 98.0 LPIPS ↓ 0.11* 0.06 0.07 GPT4V Score↑ 3.36* N/A 3.50 Local CLIP Score ↑ 29.21* 28.57 28.81 Local FID ↓ 143.82* 62.40 60.07

- Unified Metric ↑ 5.04 36.41† 35.47

Baselines To facilitate comparison with prior methods [3, 38], we manually check and annotate the object masks for InstructPix2Pix [3], and utilize our generated mask for PowerPaint [38] for generation. It is important to note that neither of these methods can complete evaluations independently. thus, their quantitative metrics should be used as references only.

##### 4.2 Main results

Tab. 1 shows the main results of Diffree with different baselines. We report the results of four powerful metrics and Unified Metric. It is worth to highlight that only successful InstructPix2pix results are computed and PowerPaint is utilized for image inpainting under the masks provided by the results of our approach. We can deduce the following conclusions from the results in several aspects.

Success Rate We achieved a success rate of over 98% on different dataset, while InstructPix2pix shows a lower success rate in object addition (17.2% and 18.9%). As shown in Fig. 9, most of the results of InstructPix2pix involve replacing existing object, without adding or significant changes to the background. This demonstrates our excellent ability to complete this task. Meanwhile it is not applicable to PowerPaint as it necessitates a mask input.

Consistency of Background Diffree significantly outperforms InstructPix2pix in the LPIPS scores across all datasets (all decreased by 36% than InstructPix2pix). In particular, only scores from carefully chosen successful cases of InstructPix2pix were computed, potentially leading to an overestimation. Furthermore, Diffree, as a shape free inpainting method, yields LPIPS results comparable to PowerPaint, as a shape required inpainting method. As discussed in

[Figure 121]

[Figure 122]

###### Diffree 13

|[Figure 123]|
|---|

|[Figure 124]|
|---|

[Figure 125]

[Figure 126]

[Figure 127]

Without mix with previous image using mask from OMP Module

[Figure 128]

[Figure 129]

(1) …

(5) …(11)

|[Figure 130]|
|---|

|[Figure 131]|
|---|

|[Figure 132]|
|---|

[Figure 133]

[Figure 134]

[Figure 135]

(1) … (5) …(11)

(1) Fabric sofa

(2) Table (6) Potted

(3) Cute dog

(4) Photo frame

(5) Carpet

Add

(7) Poster

(8) Teddy bear

(9) Wall cabinet

(10) Rug

(11) birthday cake

- Fig. 10: Visualization of with/without mix with previous image using mask from OMP.

Sec. 3.1, we expect achieving consistency of background like the image inpainting methods that necessitate masks. These methods inherently excel on this aspect, given that their input and ground truth are the same image during the training process. Therefore, we believe that we have a strong capability in this aspect.

Reasonableness of object location The results of GPT4V’s assessment demonstrate that Diffree has a considerable advantage in the reasonableness of object location (e.g., 0.34 higher than successful results from InstructPix2pix). This is not avaliable for PowerPaint due to it requires a mask as input.

Correlation, Quality and Diversity of Generated Object We conduct an evaluation of the generated object across these three dimensions, utilizing both Local CLIP Score and Local FID. Although Diffree exhibits a slightly lower Local CLIP Score in comparison to InstructPix2pix (e.g., 28.96 versus 29.30 on the COCO), this discrepancy can be rationalized by the fact that its successful results are inherently highly correlated while ours encompass all outcomes without any specific selection. Intriguingly, we demonstrate superiority over PowerPaint in terms of correlation. Furthermore, our performance according to the Local FID metric indicates a distinct advantage relative to all other methods.

Unified Metric of Diffree We combine the success rate with diverse metrics across various aspects to calculate a unified metric, thereby facilitating a more comprehensive comparison with extant text-guided methods. It is discernible that Diffree exhibits a substantial superiority over InstructPix2pix, for instance, ours’ 35.92 as opposed to InstructPix2pix’s 4.48 on the COCO. PowerPaint achieves superior results (e.g., 37.20 on the COCO dataset), the image inpainting of powerpaint buit upon the masks from our Diffree. This further underscores the excellent scalability of Diffree when integrated with other methods.

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Wooden house

[Figure 141]

###### +

Diffree Anydoor

[Figure 142]

[Figure 143]

[Figure 144]

|[Figure 145]|[Figure 146]|
|---|---|

Diffree

Add a dog

(a) (b)

- Fig. 11: Applications combined with Diffree. (a): combined with anydoor to add a specific object. (b): using GPT4V to plan what should be added.

##### 4.3 Visualization

We provide different types’ visualizations to more intuitively evaluate Diffree’s capabilities Figs. 1 to 4, 7, 10 and 11, please refer to the respective image captions for detailed explanations. For more results, please refer to the appendix.

###### 4.4 Application Diffree can be well combined with other methods for more expansion.

With GPT4V GPT4V [19] has a good ability to perceive and understand images, therefore we can use GPT4V for planning a object suitable for the image scene, seeing Fig. 11. However, when task with adding corresponding object without altering the background, DALL-E-3 [2] in GPT4, falls short.

With Other Methods AnyDoor [4] can add a specific object to the designated area by providing a mask and object image. As depicted in Fig. 11, Diffree can combined with AnyDoor to further achieve adding a specific object to image. DIffree also can effectively leverage the continuous progress in the image inpainting, to generate superior images, as demonstrated in Tab. 1.

Iterative Operation In Figs. 1 and 10, we present results of iterative inpainting. Leveraging the predicted mask from OMP module, Diffree can preserve the image background from cumulative degradation during successive inpainting. This holds potential applications within architectural and interior design domains.

### 5 Conclusion

We propose a novel method, Diffree, that leverages a diffusion model with an object mask predictor for text-guided object addition. Beyond the method, we build a high-quality synthetic dataset, OABench, through a novel data collection method for this task. Diffree distinguishes itself by preserving background consistency without requiring additional masks, which solves shortcomings of previous

text-guided and mask-guided object addition methods. The quantitative and qualitative results demonstrate the superiority of our method.

### References

- 1. Bertalmio, M., Sapiro, G., Caselles, V., Ballester, C.: Image inpainting. In: SIGGRAPH (2000) 2
- 2. Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., Manassra, W., Dhariwal, P., Chu, C., Jiao, Y.: Improving image generation with better captions (2023) 2, 4, 14
- 3. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: CVPR (2023) 2, 3, 4, 6, 9, 12
- 4. Chen, X., Huang, L., Liu, Y., Shen, Y., Zhao, D., Zhao, H.: Anydoor: Zero-shot object-level image customization. arXiv preprint arXiv:2307.09481 (2023) 2, 5, 14
- 5. Dhariwal, P., Nichol, A.: Diffusion models beat gans on image synthesis. In: NeurIPS (2021) 4
- 6. Geng, Z., Yang, B., Hang, T., Li, C., Gu, S., Zhang, T., Bao, J., Zhang, Z., Hu, H., Chen, D., et al.: Instructdiffusion: A generalist modeling interface for vision tasks. arXiv preprint arXiv:2309.03895 (2023) 2, 4
- 7. Gupta, A., Dollar, P., Girshick, R.: Lvis: A dataset for large vocabulary instance segmentation. In: CVPR (2019) 3, 6
- 8. Hessel, J., Holtzman, A., Forbes, M., Bras, R.L., Choi, Y.: Clipscore: A referencefree evaluation metric for image captioning. In: EMNLP (2021) 3, 7, 9, 10
- 9. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. In: NeurIPS

(2017) 3, 9, 10

- 10. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. In: NeurIPS

(2020) 4

- 11. Ho, J., Salimans, T.: Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022) 9
- 12. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013) 7
- 13. Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Kamali, S., Popov, S., Malloci, M., Kolesnikov, A., et al.: The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. IJCV (2020) 9, 11, 12
- 14. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: ECCV (2014) 3, 6, 9, 11, 12
- 15. Lugmayr, A., Danelljan, M., Romero, A., Yu, F., Timofte, R., Van Gool, L.: Repaint: Inpainting using denoising diffusion probabilistic models. In: CVPR (2022) 2
- 16. Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., et al.: Language models are fewshot learners. arXiv preprint arXiv:2005.14165 (2020) 4
- 17. Midjourney: Midjourney. https://www.midjourney.com/ (2022) 2
- 18. Nichol, A.Q., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., Mcgrew, B., Sutskever, I., Chen, M.: Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In: ICML (2022) 4

- 19. OpenAI: Gpt-4v(ision) system card (2023) 2, 3, 9, 10, 11, 14
- 20. Pathak, D., Krahenbuhl, P., Donahue, J., Darrell, T., Efros, A.A.: Context encoders: Feature learning by inpainting. In: CVPR (2016) 2
- 21. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023) 2
- 22. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML (2021) 4, 8, 10
- 23. Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., Chen, M.: Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125

(2022) 2, 4, 6

- 24. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: CVPR (2022) 3, 4, 6, 7, 11
- 25. Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic textto-image diffusion models with deep language understanding. In: NeurIPS (2022) 5
- 26. Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al.: Laion-5b: An open largescale dataset for training next generation image-text models. In: NeurIPS (2022) 4
- 27. Sheynin, S., Polyak, A., Singer, U., Kirstain, Y., Zohar, A., Ashual, O., Parikh, D., Taigman, Y.: Emu edit: Precise image editing via recognition and generation tasks. arXiv preprint arXiv:2311.10089 (2023) 2, 4
- 28. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamics. In: ICML (2015) 7
- 29. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Scorebased generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 (2020) 7
- 30. Suvorov, R., Logacheva, E., Mashikhin, A., Remizova, A., Ashukha, A., Silvestrov, A., Kong, N., Goka, H., Park, K., Lempitsky, V.: Resolution-robust large mask inpainting with fourier convolutions. In: WACV (2022) 2
- 31. Wang, S., Saharia, C., Montgomery, C., Pont-Tuset, J., Noy, S., Pellegrini, S., Onoe, Y., Laszlo, S., Fleet, D.J., Soricut, R., et al.: Imagen editor and editbench: Advancing and evaluating text-guided image inpainting. In: CVPR (2023) 5
- 32. Wasserman, N., Rotstein, N., Ganz, R., Kimmel, R.: Paint by inpaint: Learning to add image objects by removing them first (2024) 5
- 33. Xie, S., Zhang, Z., Lin, Z., Hinz, T., Zhang, K.: Smartbrush: Text and shape guided object inpainting with diffusion model. In: CVPR (2023) 2, 3, 5, 9, 10, 11
- 34. Yu, J., Lin, Z., Yang, J., Shen, X., Lu, X., Huang, T.S.: Generative image inpainting with contextual attention. In: CVPR (2018) 2
- 35. Zhang, K., Mo, L., Chen, W., Sun, H., Su, Y.: Magicbrush: A manually annotated dataset for instruction-guided image editing. In: NeurIPS (2024) 2, 4, 5, 6
- 36. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric. In: CVPR (2018) 3, 9, 10, 11
- 37. Zhang, S., Yang, X., Feng, Y., Qin, C., Chen, C.C., Yu, N., Chen, Z., Wang, H., Savarese, S., Ermon, S., et al.: Hive: Harnessing human feedback for instructional visual editing. arXiv preprint arXiv:2303.09618 (2023) 2

- 38. Zhuang, J., Zeng, Y., Liu, W., Yuan, C., Chen, K.: A task is worth one word: Learning with task prompts for high-quality versatile image inpainting. In: ECCV

(2024) 2, 3, 5, 6, 12

