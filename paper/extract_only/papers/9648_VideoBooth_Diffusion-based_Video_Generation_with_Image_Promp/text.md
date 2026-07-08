# arXiv:2312.00777v1[cs.CV]1Dec2023

#### VideoBooth: Diffusion-based Video Generation with Image Prompts

Yuming Jiang1 Tianxing Wu1 Shuai Yang1 Chenyang Si1 Dahua Lin2 Yu Qiao2 Chen Change Loy1 Ziwei Liu1

1S-Lab, Nanyang Technological University 2Shanghai AI Laboratory

https://vchitect.github.io/VideoBooth-project/

Image Prompt A horse eating grass.

Horse grazes in snowy meadow

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

Image Prompt Portrait of a dog, looks out the car window.

Dog walking in the green farm 4k

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Image Prompt Cat is looking at a laptop.

Close up of cat on top of a vintage chair

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Elephant drinking water in masai mara reserve, kenya

Elephant walk in the yellow grass of savannah

Image Prompt

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Figure 1. Videos synthesized by image prompts. Our VideoBooth generates videos with the subjects specified in the image prompts.

##### Abstract

the attention injection module at fine level, multi-scale image prompts are fed into different cross-frame attention layers as additional keys and values. This extra spatial information refines the details in the first frame and then it is propagated to the remaining frames, which maintains temporal consistency. Extensive experiments demonstrate that VideoBooth achieves state-of-the-art performance in generating customized high-quality videos with subjects specified in image prompts. Notably, VideoBooth is a generalizable framework where a single model works for a wide range of image prompts with feed-forward pass.

Text-driven video generation witnesses rapid progress. However, merely using text prompts is not enough to depict the desired subject appearance that accurately aligns with users’ intents, especially for customized content creation. In this paper, we study the task of video generation with image prompts, which provide more accurate and direct content control beyond the text prompts. Specifically, we propose a feed-forward framework VideoBooth, with two dedicated designs: 1) We propose to embed image prompts in a coarse-to-fine manner. Coarse visual embeddings from image encoder provide high-level encodings of image prompts, while fine visual embeddings from the proposed attention injection module provide multi-scale and detailed encoding of image prompts. These two complementary embeddings can faithfully capture the desired appearance. 2) In

##### 1. Introduction

Text-to-image models [16, 17, 20, 26, 31, 37, 41, 57, 62– 64, 68, 70] have attracted substantial attention. With Stable Diffusion [64], we can now easily generate images using

texts. Recently, the focus has been shifted to text-to-video models [6, 18, 30, 42, 44, 55, 71, 74, 78, 91] to generate videos by taking text descriptions as inputs. However, in some user cases, texts alone are not expressive enough to define the specific appearance of subjects [21, 66]. For example, as shown in Fig. 2, if we want to generate a video clip containing the dog as the third row, we need to use several attributive adjuncts to define the appearance of the dog in text prompts. Even with these extensive attributive adjuncts, models still cannot generate the desired appearance. Defining the appearance of the desired object with texts alone has the following flaws: 1) It is hard to enumerate all desired attributes, and 2) The model cannot capture all attributes accurately with a long text. Compared to using texts, a more straightforward way to define the appearance is to provide reference images, termed image prompts. The image prompts are complementary to text prompts and enrich the details that are hard to be depicted by text prompts.

<Dog> eating snack inside big iron cage at home.

|A dog|
|---|

[Figure 37]

[Figure 38]

[Figure 39]

longtextimagepromptsimpletext

|A dog with a brown and white coat and a distinctive collie appearance|
|---|

[Figure 40]

[Figure 41]

[Figure 42]

|[Figure 43]|
|---|

[Figure 44]

[Figure 45]

[Figure 46]

Figure 2. The Use of Image Prompts. We generate three video clips using different types of prompts: simple text prompt, long text prompt, and image prompt. We use the LLaVa model [51] to generate a text prompt describing the appearance of the image prompt. Using text prompts alone cannot fully capture the visual characteristics of the image prompt.

There are several attempts to introduce image prompts into text-to-image models, which can be roughly divided into two groups. One is to fine-tune parts of parameters using few-shot reference images [15, 21, 28, 47, 66], which contain the same objects captured under different circumstances. However, the requirement for the number of reference images is demanding as sometimes it is not practical to obtain multiple images of the same object. The other category [11, 40, 48, 77, 80, 81], aiming to address this limitation, proposes to embed image prompts into text-to-image models and the inference is tuning-free. Both of these two types of attempts achieve plausible results in generating images containing objects specified in image prompts.

to replace parts of the original text embeddings. The welltrained encoder embeds the coarse appearance information of the given image prompts. However, coarse visual embedding is a universal embedding: 1) It only contains high-level semantic information, and 2) It is shared across all blocks with the same scale. As a result, some visual details are missing in the coarse visual embeddings.

To further refine the generated details as well as maintain temporal consistency, different from the highly compact coarse visual embeddings, multi-scale image prompts are injected into cross-frame attention modules in different layers. The image prompts provide spatial information as well as details with different granularities. On the one hand, keeping spatial information of the image prompts can retain more details. On the other hand, different cross-frame attention modules need detailed information at different scales. Specifically, the latent representations of image prompts are appended as additional keys and values to refine the details in the first generated frame. To propagate the refined first frame to the following frames to maintain temporal consistency, we then use the updated values of the first frame as values for the remaining frames.

In this paper, we study a more challenging task, i.e., text-to-video generation with image prompts. The task has two main challenges: 1) Similar to text-to-image generation, the attributes of image prompts should be accurately captured and then reflected in the generated videos; 2) Different from text-to-image generation, we aim for the dynamic movement of the object rather than a static one. Directly adapting these methods to video domain results in mismatched appearance or unnatural degraded movements. To address these challenges, we proposed VideoBooth with elaborately designed coarse-to-fine visual embedding components: 1) Coarse visual embeddings via image encoder: An image encoder is trained to inject the image prompts into text embeddings; 2) Fine visual embeddings via attention injection: The image prompts are mapped to multiscale latent representations to control the generation process through cross-frame attentions of text-to-video models.

We set up a dedicated VideoBooth dataset to support the study of the new task. With each video, we provide an image prompt and a text prompt. Extensive experiments demonstrate the effectiveness of our proposed VideoBooth to generate videos with subjects specified in image prompts. As shown in Fig. 1, videos generated by VideoBooth better keep the visual attributes of image prompts. Besides, our proposed VideoBooth is tuning-free at inference time and

Specifically, inspired by early attempts [40, 77] in textto-image models with image prompts, we extract the CLIP image features of the provided image prompts using the pretrained CLIP model [61]. Then the extracted features are mapped into the text embedding space, which are inserted

videos can be generated with feed-forward pass only. The contributions are summarised as follows:

- • To the best of our knowledge, we are the first to explore the task of video generation using image prompts without finetuning at inference time. We propose a dedicated dataset to support the task. Our proposed VideoBooth framework can generate consistent videos containing the subjects specified in image prompts.
- • We introduce a new coarse-to-fine visual embedding strategy by image encoder and attention injection, which better captures the characteristics of the image prompts.
- • We propose a novel attention injection method, using the multi-scale image prompts with spatial information to refine the generated details.

##### 2. Related Work

Text-to-Video Models take the text descriptions as inputs and generate clips of videos. Early explorations [34, 74] on text-to-video models are based on the idea of VQVAE. Make-A-Video [71] proposes to add temporal attention to the architecture of DALLE2 model [63]. Recently, the emergence of diffusion models [32, 65] boosts research on text-to-video models [1, 23, 30, 33, 55, 76, 91]. Video LDM [6] proposes to train the text-to-video models on Stable Diffusion with temporal attention and 3D convolution introduced to handle the temporal generation. Gen1 [18] introduces depth maps to handle the temporal consistency of text-to-video models. Some methods [27, 89] resort to training separate modules for synthesizing motions. All of the methods initialize their models with pre-trained text-to-image models. Another paradigm of using text-toimage models is to directly apply Stable Diffusion [64] to few-shot or zero-shot settings. Tune-A-Video [78] adapts the self-attention into cross-frame attention and then finetunes the stable diffusion model on a video clip. Models trained in this way have the capability to transfer motions from original videos. Text2Video-Zero [44] proposes to generate videos by using correlated noise maps to improve consistency. Apart from video generation, diffusion models have been applied to video-to-video generations [8, 9, 19, 24, 36, 39, 50, 53, 59, 60, 69, 75, 83, 84, 88]. Customized Content Creation aims at generating images and videos using reference images [15]. For customized text-to-image generation, optimization-based methods [28, 35, 47, 49] are proposed to optimize the weights of the diffusion model. For example, Textual Inversion [21] optimizes the word embeddings, while DreamBooth [66] proposes to finetune the weights of Stable Diffusion as well. Optimization-based methods require several reference images with the same subject to avoid the overfitting of the model, which is demanding in real-world applications. The cost of finetuning hampers the practical usage of these methods. To address these limitations, encoder-based meth-

ods [56, 81, 85, 92] are proposed to learn a mapping network to embed the reference images. ELITE [77] proposes to learn a global mapping network and local mapping network to encode the images into word embeddings. Jia et al. [40] propose to use an additional cross-attention to embed the image features. With the trained encoder, the personalized generation can be achieved in a feed-forward pass. Some recent works [2, 11, 22, 25] combine the encoder-based model and finetuning-based model to improve the performance. BLIP-Diffusion [48] proposes to pretrain a multimodal encoder in a large-scale dataset and then finetune the model on the specific subject for inference. Customized image generation is also applied to place the objects into the user-specified scenes [4, 13, 46, 72, 87]. Also, some efforts [12, 14, 38, 67, 73, 79, 80, 86] have been made to personalized face generation. He et al. [29] propose to improve the performance from the data perspective. Some works [3, 43, 54] focus on composing multiple subjects in one image. Apart from works on image generation, there are some early attempts at personalized video manipulation. Make-A-Protagonist [90] edits an existing video in a personalized way using Stable Diffusion 2.1 to embed the image prompts. The motion of the original video is learned from Tune-A-Video [78]. VideoDreamer [10] proposes to generate personalized videos by generating the first frames using a finetuning-based method and then generating the video clip using the Text2Video-Zero [44]. Different from existing works, our proposed VideoBooth does not need to finetune any weights at the inference time.

##### 3. VideoBooth

Our proposed VideoBooth aims at generating videos from an image prompt I and a text prompt T. The image prompt specifies the appearance of the subject. An overview of our proposed VideoBooth is illustrated in Fig. 3. The image prompt is fed into VideoBooth in two levels. At the coarse level, it is fed into a pretrained CLIP Image encoder to extract visual features. An encoder, composed of several MLP layers, is trained to map visual features into the space of text embeddings. The obtained embedding fI will be inserted into text embedding, which is extracted by feeding text prompt T into CLIP text encoder. To further refine the synthesized details, we propose to inject image prompt I into the cross-frame attention module in the pretrained video diffusion model. Specifically, we append latent representation xtI of image prompt I into the cross-frame attention. In this way, multi-scale visual details with spatial information are involved in the calculation of attention maps so that visual characteristics can be better preserved. Two ways of feeding image prompt corporate with each other in a coarse-to-fine manner. The encoder provides coarse visual embeddings of the image prompt, while the attention injection provides fine visual embeddings.

%!

[Figure 47]

learned embeddings

[Figure 48]

❄

[Figure 49]

!

celebrates birthday with gifts, balloons and soap bubbles stock footage video

CLIP Image Encoder

MLPs

$ coarse visual embeddings

image prompt

ﬁne visual embeddings

!!" !!"

TemporalAttention

[Figure 50]

[Figure 51]

concatedvalues

concatedkeys

|V img Projection<br><br>[Figure 52]<br><br>!|
|---|
|V Projection<br><br>[Figure 53]<br><br>!|

|K img Projection<br><br>[Figure 54]<br><br>!|
|---|
|K Projection<br><br>[Figure 55]<br><br>!|

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

|K Projection<br><br>[Figure 60]<br><br>!|
|---|

|V Projection<br><br>[Figure 61]<br><br>!|
|---|

[Figure 62]

[Figure 63]

|Q Projection<br><br>[Figure 64]<br><br>❄|
|---|

|Q Projection<br><br>[Figure 65]<br><br>❄|
|---|

Cross-Frame Attention

Cross Attention

!! !!"#

Blocks in U-Net

Figure 3. Overview of VideoBooth. VideoBooth generates videos by taking image prompts I and text prompts T as inputs. The image prompt is fed into the CLIP image encoder, followed by MLP layers. The obtained coarse visual embedding fI is then inserted into the text embeddings. The composed embeddings serve as the input for cross attention. The embedding extracted by the encoder provides a coarse encoding of the visual appearance of the image prompt. To further refine the details in the generated videos, at the fine level, we append the latent representation of the image prompt to the cross-frame attention as additional keys and values. Different cross-frame attention layers receive latent representations with different scales. The multi-scale features with spatial details refine the synthesized details.

###### 3.1. Preliminary: Pretrained Text-to-Video Model

code visual information of image prompts by an image encoder. The image prompt and text prompt complement each other. The image prompt provides visual characteristics of the desired subject in the video, and the text prompt provides other orthogonal information. The extracted visual embeddings are combined with text embeddings as the final embeddings for the cross-attention module. Specifically, the CLIP image encoder is employed to extract the visual features fV of image prompt I. Since the discrepancy exists between the CLIP image and text embeddings, fV is then fed into MLP layers F(·) to map fV to the spaces of text embeddings. The final embedding fI for the image prompt is obtained as follows:

Our proposed VideoBooth is developed based on the pretrained text-to-video model [30, 76]. In this section, we will briefly introduce the framework of text-to-video model.

Inflated 2D Conv. To handle video data and capture the temporal correlation, 2D conv in the Stable Diffusion model is inflated to 3D conv. In this way, the U-Net can encode 3D features containing the temporal dimension.

Cross-Frame Attention Module. Stable Diffusion has a self-attention module, where the features are enhanced by attending to themselves. To improve temporal consistency, the self-attention is modified into cross-frame attention. Specifically, in cross-frame attention, the feature of each frame is enhanced by attending and referencing to the first frame and the previous frame. The cross-frame attention operates on both the spatial domain and temporal domain, thus the temporal consistency of the synthesized frames is improved.

###### fV = CLIPI(I),fI = F(fV ). (1)

As for the text prompt T, we feed it into the CLIP text encoder to extract the text embedding fT:

fT = [fT0,fT1,...,fTk,...], (2) where fTk is the k-th word embedding in the text prompt.

Temporal Attention Module. Apart from cross-frame attention, a temporal attention module is introduced to further improve temporal consistency. Temporal attention operates on temporal domain and attends to all frames.

To make the diffusion model generate videos conditioning on both text prompts and image prompts, we need to integrate these two embeddings, i.e., fI and fT. The idea is to replace the word embedding of the target subject with fI. Mathematically, fI and ft are fused to obtain the final text condition ct as follows:

###### 3.2. Coarse Visual Embeddings via Image Encoder

Given an image prompt I and text prompt T, the generated video is supposed to be consistent with visual elements and textual elements. Inspired by previous attempts at imagebased customization methods [40, 77], we propose to en-

###### cT = [fT0,fT1,...,fTk−1,fI,fTk+n,...], (3)

image prompt frame #0

frame #(i-1)

## [ ; ]

[ ; ]

K

KI K0

K0

Ki-1

## [ ; ]

## [ ; ]

V

V0new

VI

V0

Vi-1

Vi

[Figure 70]

Qi

Q0

Since text-to-video diffusion models operate in the latent space, we first feed the image prompt into the VAE of Stable Diffusion and get its latent representation xI. Moreover, since the sampling of the videos starts from the noise map, the latent in the intermediate timesteps contains the noises. If we append the clean latent xI of the image prompt to the cross-frame attention, the domain discrepancy exists. Therefore, we follow the diffusion forward process to add corresponding noises to xI:

[Figure 71]

[Figure 72]

[Figure 73]

image prompt frame #0

frame #(i-1)

## [ ; ]

[ ; ]

K

KI K0

K0

Ki-1

## [ ; ]

## [ ; ]

V

V0new

VI

V0

Vi-1

Vi

[Figure 74]

[Figure 75]

Qi

xIt = √αtxI0 + √1 − αtϵ, (4)

Q0

Figure 4. Fine Visual Embedding Refinement. We propose to inject the latent representation of image prompt (here we use the image for illustration purpose) directly into the cross-frame attention module. We use the keys and values from the image prompt to update the values of the first frame firstly. Then, the updated values of the first frame are used to update the remaining frames. Injecting the image prompt in the cross-frame attention helps to transfer the detailed visual characteristics of the image prompts to the synthesized frames. We perform the refinement in different cross-attention layers with different scales.

where α is a hyperparameter determined by the denoising schedule and ϵ ∼ N(0,I).

The cross-frame attention is used to improve the temporal consistency of the generated frames. For each frame, the key and value are the concatenation of the features of the first frame and the previous frame. Here, we introduce the image prompts as the additional keys and values for the frames. As shown in Fig. 4, we propose to update the values of the first frame firstly using the keys and values of the image prompts and the frame itself. Mathematically, the operation can be expressed as follows:

where k is the token index of the target subject in the text embedding, and n is the length of the text tokens for the target subject. For example, an image of a papillon dog is provided as an image prompt. To fuse the information from the image prompt and the text prompt “Papillon dog celebrates birthday with gifts”, the word embeddings of the “papillon dog” will be replaced with fI before they are fed into the cross-attention module of the diffusion models.

KQT0

V0new = softmax(

√

) · V, K = [KI,K0],V = [VI,V0],

(5)

d

where KI and VI are the keys and values obtained from the image prompts. The query, key, and value of the first frame are denoted Q0, K0, and V0, respectively. It should be noted that we use a separately trained K and V projection for latent representations xIt of image prompts because the image prompts have clean backgrounds, which are different from other frames. The parameters of the newly added K and V projections are initialized by original K and V projections.

During the training of the coarse stage, we fix the parameters of the CLIP image encoder, and train the MLP layers. To make the diffusion model accommodate with the composed text embeddings cT, we also finetune K and V projections (linear layers to map the input feature to the corresponding keys and values) in the cross-attention module.

###### 3.3.FineVisualEmbeddingsviaAttentionInjection

Then the updated first frame is used to refine the remaining frames. When updating the remaining frames, the keys used for calculating the attention maps are the original keys, while the values are the updated ones. The update is expressed as follows:

The well-trained image encoder embeds the coarse visual embeddings for image prompts and thus the synthesized videos contain the subjects specified in image prompts. However, the image encoder projects the image prompt into a flattened high-level representation, resulting in the loss of its detailed visual cues. Thus, some detailed visual characteristics in the image prompts may not be well preserved. To address this problem, a more effective way to preserve these details is to provide the model with the image prompts with spatial resolutions.

KQTi

Vinew = softmax(

√

) · V, K = [K0,Ki−1],V = [V0new,Vi−1].

(6)

d

To sum up, in the attention injection, we update the values of the first frame using the image prompts first, and then use the updated first frame to update the other frames. In this way, the visual cues from the image prompts can be consistently propagated to all the frames.

To further refine the synthesized details, we propose to inject image prompts into the cross-frame attention of the text-to-video models. By injecting image prompts into the cross-frame attention, the image prompts are involved in the updates of the synthesized frames so that the model can directly borrow some visual cues from image prompts.

It should be noted that the diffusion model has multiple cross-frame attention layers with different scales. To inject

multi-scale visual cues for better detail refinement, in different cross-frame attention layers, we feed latent representations of the image prompts with corresponding resolutions, which are obtained from different stages of the U-Net.

###### 3.4. Coarse-to-Fine Training Strategy

The visual details of the image prompts are embedded into the final synthesized results in two stages: coarse visual embeddings using an image encoder and fine visual embedding by attention injection. We propose to train these two modules in a coarse-to-fine manner. In other words, we train the coarse image encoder and tune the parameters in the cross-attention first. After the model has the capability of generating videos containing the subjects specified in image prompts, we then train the attention injection module to embed image prompts into cross-frame attention layers. As we will show in the ablation study (Sec. 5.5), if these two modules are trained together, the fine attention injection module leaks the strong visual cues and the coarse encoder learns meaningless representations. As a result, in sampling phase, the image encoder for the coarse visual embedding cannot provide the coarse information and then the fine attention module cannot refine the details. Therefore, it is necessary to train VideoBooth in a coarse-to-fine manner.

##### 4. VideoBooth Dataset

We establish the VideoBooth dataset to support the task of video generation using image prompts. We start from the WebVid dataset [5], a well-known open-source dataset for text-to-video generation. In the WebVid dataset, there is a text prompt with each video. In this paper, we study the task of generating a video clip from one text prompt and one image prompt. Hence, in addition to the original text prompt, we need to provide an image prompt for each video. We propose to segment the subjects from the first frame of the video using the Grounded-SAM (Grounded Segment Anything) [45, 52], and the segmented subjects are image prompts. The Grounded-SAM receives word prompts as inputs and generates segmentation masks for the target subjects specified in word prompts. To obtain the word prompt for the input to Grounded-SAM, we use the spaCy library to parse the noun chunks from the original text prompts, which are used as the word prompts. After the segmentation, we perform data filtering to ensure the data quality. We filter out small objects and large objects (those are almost the same size as the original video) according to the ratio of the object to the whole video. Also, since we focus on generating video clips containing moving objects, we further filter the videos containing moving objects. The keywords we used for filtering are dog, cat, bear, car, panda, tiger, horse, elephant, and lion. In the current version, we have processed 2.5M subset of the WebVid dataset. After data filtering, we have 48,724 video data pairs for training. We

will process the full set of the WebVid dataset and include the filtered data in our VideoBooth dataset.

To evaluate the performance, we also set up a test benchmark. The test benchmark consists of 650 test pairs. For each pair, an image prompt and a text prompt are provided. The test pairs are selected from the rest of the WebVid-10M dataset, which does not overlap with the training set.

##### 5. Experiments

###### 5.1. Comparison Methods

Textual Inversion [21] is a method for customized text-toimage generation. The appearance of the target subjects is embedded into the text embeddings. Concretely, a text token S∗ is optimized to represent the subject. We adapt it to the task of text-to-video generation by replacing the image model with the video model.

DreamBooth [66] is also proposed for customized text-toimage generation. It injects the target subject into the text tokens as well as model weights. During the training, both the model weights and the special token S∗ are optimized.

ELITE [77] is an encoder-based method for personalized generation. An encoder is trained to embed the images into the text embeddings. Local mapping and global mapping are employed to transform the CLIP embedding of image prompts into the features, which are injected into the crossattention module. We adapt and retrain the method using the same pretrained video model we use.

###### 5.2. Evaluation Metrics

We use three metrics to evaluate the performance [77]. To measure the alignment of the generated videos and given text prompts, we use the CLIP-Text metric. The metric is calculated using the cosine similarity of the CLIP text embeddings of text prompts and CLIP image embeddings of the generated frames. For each video, the value is obtained by averaging values of the all frames. As for the evaluation of the similarity between the given image prompts and the generated videos, we adopt two metrics: CLIP-Image and DINO [7, 58]. The CLIP-Image metric is calculated by the cosine similarity between the CLIP image embedding of image prompts and generated frames. Since the CLIP model is trained to align image embeddings and text embeddings, we follow the practice in previous methods [13, 66] and use the DINO similarity as another indicator. DINO is trained to differentiate the differences between objects of the same classes. We use the ViT-S/16 model to extract the features of the image prompts and generated frames. The final score is obtained by averaging over all frames.

###### 5.3. Quantitative Comparisons

We report quantitative results in Table 1. As shown in Table 1, our proposed VideoBooth achieves state-of-the-art

[Figure 76]

[Figure 77]

|playing ball in the<br><br>[Figure 78]<br><br>[Figure 79]<br><br>Textual Inve|[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>Textual Inversion<br><br>car in the bush<br><br>rsion DreamBooth ELITE Ours<br><br>|
|---|---|
| | |
| |DreamBooth|
|[Figure 89]|[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]|
| | |
| | |
|[Figure 94]<br><br>Image Alignm|[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>ELITE<br><br>ent Text Alignment Overall Quality|

close-up of cat with a yellow b white interior

dog laying on ground

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

Textual Inversion

Textual Inversion

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

- 0.8
- 1

DreamBooth

DreamBooth

0.6

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

0.4

0.2

0

ELITE

ELITE

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

VideoBooth (Ours)

VideoBooth (Ours)

VideoBooth (Ours)

Figure 5. Qualitative Comparison. VideoBooth effectively preserves the fidelity of image prompts and achieves better visual quality.

Table 1. Quantitative Comparisons. VideoBooth achieves the best image alignment and comparable text alignment performance.

Textual Inversion DreamBooth ELITE Ours

Textual Inversion DreamBooth ELITE Ours

(%)

100

|Method|CLIP-Text ↑|CLIP-Image ↑<br><br>|DINO ↑|
|---|---|---|---|
|Textual Inversion [21] DreamBooth [66] ELITE [77] VideoBooth (Ours)|29.9749 30.6877 30.0881 30.0967<br><br>|69.7995 71.2078 73.7518 74.7971<br><br>|45.3143 52.9661 58.9522 65.0979|

80

60

40

20

0

Image Alignment Text Alignment Overall Quality

Image Alignment Text Alignment Overall Quality

image alignment performance compared to baseline methods. As for the alignment with the text prompts, our proposed VideoBooth has comparable performance with the baseline models. It should be noted that the CLIP-Text score of DreamBooth is significantly higher than the other methods. The reason lies in that the optimized token S∗ in DreamBooth is inserted into the text embeddings, rather than replacing the original word embeddings like other methods. This would result in the generated videos of DreamBooth being highly related to the text prompts but having no correlation to the image prompts in some cases. We also conducted a user study, in which 25 users participated. Each user is presented with twelve groups of videos, and each group contains four videos generated by four methods. For each group, users are asked to make three choices: 1) which one has the best image alignment? 2) which one has the best text alignment? 3) which one has the best overall quality. Figure 6 summarizes the results. Our results are preferred by most users in all three dimensions.

Figure 6. User Study. Our proposed VideoBooth achieves the highest user preference ratios on all three dimensions.

in Fig. 5. In the first example, the model is supposed to generate videos containing the dog specified in the image laying on the ground. Textual Inversion cannot correctly embed the appearance of the dog and results in generating another totally different dog. DreamBooth and ELITE can embed the coarse appearance of the dog but the generated details vary from the image prompt. Our proposed VideoBooth successfully embeds the details of the image prompts into the synthesized videos. In the second example, the condition is to generate a cat playing with the yellow ball. All the methods can generate a yellow ball and a cat but only our proposed VideoBooth can accurately generate the cat having the appearance from the image prompt. As for the last example, Textual Inversion fails to generate a car. DreamBooth generates a distorted car in the bush. ELITE model can generate a car in the bush, but the color differs from the image prompt. By contrast, our model can generate videos having the same car in the bush.

###### 5.4. Qualitative Comparisons

We show three visual comparisons on the generated video frames of our proposed VideoBooth and baseline methods

Table 2. Ablation Study. The full model has the best scores.

[Figure 127]

Dog as a chef at the table

|Variants|CLIP-Image ↑<br><br>|DINO ↑|
|---|---|---|
|(a) Coarse Embeddings only<br>(b) Fine Embeddings only<br>(c) Unified Training<br>(d) Full Model<br>|75.4366 75.5553 75.8254 76.1631<br><br>|64.9568 66.0378 67.4201 69.7374|

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

with fine embeddings only cannot refine the details. As shown in Fig. 7(b), the first frame contains the exact appearance as the image prompt, but the temporal consistency cannot be guaranteed. The generated dog is distorted in the following frames. The reason is that the model trained in this way overfits the image prompt. In the first frame, the model can copy the information from image prompts. In the following frames, without the coarse embeddings, the generation of the appearance only relies on the propagation of the appearance from the first frame.

- (a) Only Coarse Embeddings with Image Encoder
- (b) Only Fine Embeddings with Attention Injection

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

(c) Unified Training for Image Encoder and Attention Injection

The Necessity of Coarse-to-Fine Training. In VideoBooth, we propose the coarse-to-fine training strategy, i.e., train the coarse embeddings first and then train the attention injection module. In this ablation model, we train these two modules within one stage. The unified training makes the model rely heavily on the strong guidance provided in attention injection. In this way, the trained image encoder has limited capability. Thus, the model trained in this way has a similar behavior as the model with only fine embeddings. The model also overfits the image prompts from the attention injection. As shown in the first example in Fig. 7(c), the appearance in the first frame is correct, but the generated dog in the following frames is distorted. Due to the image encoder having limited capability to provide the correct coarse encoding of image prompts, the attention injection cannot refine the details. By contrast, the full model can generate consistent frames with all details well preserved.

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

(d) Full Model

Figure 7. Ablation Study. (a) With only coarse embeddings from image encoder, generated patterns in the body of the dog are different from the image prompt. (b) With only fine embeddings from attention injection, there lacks the coarse encodings of the dog for the attention injection module to refine and thus the generated dog is distorted at later synthesized frames. (c) The unified training degrades the capability of image encoder and thus the dog is also distorted. (d) The full model better keeps the all visual details of the image prompt.

###### 5.5. Ablation Study

To evaluate the effectiveness of the proposed components, we perform three ablation studies. Due to the computational resources, we train these models on the subset of our training set. The quantitative metrics are shown in Table 2.

##### 6. Discussion

In this paper, we propose a novel framework VideoBooth to generate videos using image prompts and text prompts. The image prompts specify the appearance of the subjects. We inject the image prompts into the model in two modules: Coarse Embeddings via Image Encoder and Fine Embeddings via Attention Injection. The Image Encoder provides the coarse embeddings of image prompts for the refinement of the Attention Injection module. These two modules cooperate with each other and they are trained in a coarse-tofine manner. Our proposed VideoBooth generates consistent videos containing the desired subjects.

Only Coarse Embeddings. This ablation model injects the image prompts with only coarse embeddings via Image Encoder. In the example shown in Fig. 7(a), the ablation model only encodes the coarse appearance of the image prompts. The pattern in the legs of the synthesized dog is different from that in the image prompt. By contrast, the results of our full model show that our proposed model can transfer all the details in the image prompts to the synthesized videos.

Only Fine Embeddings. In this ablation model, we only have fine embeddings of the image prompt in cross-frame attention layers. The main purpose of using fine embedding is to refine the coarse encoding of image prompts from the Image Encoder. Without coarse embeddings, the model

Potential Negative Societal Impacts. Our model can be used to synthesize videos. The model may be applied to generate fake videos, which can be potentially avoided by using more advanced fake video detection methods.

##### References

- [1] Jie An, Songyang Zhang, Harry Yang, Sonal Gupta, Jia-Bin Huang, Jiebo Luo, and Xi Yin. Latent-shift: Latent diffusion with temporal shift for efficient text-to-video generation. arXiv preprint arXiv:2304.08477, 2023. 3
- [2] Moab Arar, Rinon Gal, Yuval Atzmon, Gal Chechik, Daniel Cohen-Or, Ariel Shamir, and Amit H Bermano. Domainagnostic tuning-encoder for fast personalization of text-toimage models. arXiv preprint arXiv:2307.06925, 2023. 3
- [3] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel CohenOr, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. arXiv preprint arXiv:2305.16311, 2023. 3
- [4] Jinbin Bai, Zhen Dong, Aosong Feng, Xiao Zhang, Tian Ye, Kaicheng Zhou, and Mike Zheng Shou. Integrating view conditions for image synthesis. arXiv preprint arXiv:2310.16002, 2023. 3
- [5] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE International Conference on Computer Vision, 2021. 6, 13
- [6] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 2, 3
- [7] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 6
- [8] Duygu Ceylan, Chun-Hao Huang, and Niloy J. Mitra. Pix2video: Video editing using image diffusion. arXiv:2303.12688, 2023. 3
- [9] Wenhao Chai, Xun Guo, Gaoang Wang, and Yan Lu. Stablevideo: Text-driven consistency-aware diffusion video editing. arXiv preprint arXiv:2308.09592, 2023. 3
- [10] Hong Chen, Xin Wang, Guanning Zeng, Yipeng Zhang, Yuwei Zhou, Feilin Han, and Wenwu Zhu. Videodreamer: Customized multi-subject text-to-video generation with disen-mix finetuning. arXiv preprint arXiv:2311.00990,

2023. 3

- [11] Hong Chen, Yipeng Zhang, Xin Wang, Xuguang Duan, Yuwei Zhou, and Wenwu Zhu. Disenbooth: Disentangled parameter-efficient tuning for subject-driven text-to-image generation. arXiv preprint arXiv:2305.03374, 2023. 2, 3
- [12] Li Chen, Mengyi Zhao, Yiheng Liu, Mingxu Ding, Yangyang Song, Shizun Wang, Xu Wang, Hao Yang, Jing Liu, Kang Du, et al. Photoverse: Tuning-free image customization with text-to-image diffusion models. arXiv preprint arXiv:2309.05793, 2023. 3
- [13] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. arXiv preprint arXiv:2307.09481, 2023. 3, 6

- [14] Zhuowei Chen, Shancheng Fang, Wei Liu, Qian He, Mengqi Huang, Yongdong Zhang, and Zhendong Mao. Dreamidentity: Improved editability for efficient face-identity preserved image generation. arXiv preprint arXiv:2307.00300, 2023. 3
- [15] Jooyoung Choi, Yunjey Choi, Yunji Kim, Junho Kim, and Sungroh Yoon. Custom-edit: Text-guided image editing with customized diffusion models. arXiv preprint arXiv:2305.15779, 2023. 2, 3
- [16] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. Advances in Neural Information Processing Systems, 34:19822–19835, 2021. 1
- [17] Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generation via hierarchical transformers. arXiv preprint arXiv:2204.14217,

2022. 1

- [18] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. arXiv preprint arXiv:2302.03011, 2023. 2, 3
- [19] Ruoyu Feng, Wenming Weng, Yanhui Wang, Yuhui Yuan, Jianmin Bao, Chong Luo, Zhibo Chen, and Baining Guo. Ccedit: Creative and controllable video editing via diffusion models. arXiv preprint arXiv:2309.16496, 2023. 3
- [20] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scenebased text-to-image generation with human priors. arXiv preprint arXiv:2203.13131, 2022. 1
- [21] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion, 2022. 2, 3, 6, 7, 13
- [22] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Encoder-based domain tuning for fast personalization of text-to-image models. ACM Transactions on Graphics (TOG), 42(4):1–13, 2023. 3
- [23] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, MingYu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In ICCV, 2023. 3
- [24] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023. 3
- [25] Yuan Gong, Youxin Pang, Xiaodong Cun, Menghan Xia, Haoxin Chen, Longyue Wang, Yong Zhang, Xintao Wang, Ying Shan, and Yujiu Yang. Talecrafter: Interactive story visualization with multiple characters. arXiv preprint arXiv:2305.18247, 2023. 3
- [26] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10696–10706, 2022. 1
- [27] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 3

- [28] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. arXiv preprint arXiv:2303.11305, 2023. 2, 3
- [29] Xingzhe He, Zhiwen Cao, Nicholas Kolkin, Lantao Yu, Helge Rhodin, and Ratheesh Kalarot. A data perspective on enhanced identity preservation for diffusion personalization. arXiv preprint arXiv:2311.04315, 2023. 3
- [30] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221, 2022. 2, 3, 4
- [31] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 1
- [32] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 3
- [33] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 3
- [34] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 3
- [35] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. 3
- [36] Zhihao Hu and Dong Xu. Videocontrolnet: A motion-guided video-to-video translation framework by using diffusion model with controlnet. arXiv preprint arXiv:2307.14073,

2023. 3

- [37] Ziqi Huang, Tianxing Wu, Yuming Jiang, Kelvin CK Chan, and Ziwei Liu. Reversion: Diffusion-based relation inversion from images. arXiv preprint arXiv:2303.13495, 2023. 1
- [38] Junha Hyung, Jaeyo Shin, and Jaegul Choo. Magicapture: High-resolution multi-concept portrait customization. arXiv preprint arXiv:2309.06895, 2023. 3
- [39] Hyeonho Jeong and Jong Chul Ye. Ground-a-video: Zeroshot grounded video editing using text-to-image diffusion models. arXiv preprint arXiv:2310.01107, 2023. 3
- [40] Xuhui Jia, Yang Zhao, Kelvin CK Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, and Yu-Chuan Su. Taming encoder for zero fine-tuning image customization with text-to-image diffusion models. arXiv preprint arXiv:2304.02642, 2023. 2, 3, 4
- [41] Yuming Jiang, Shuai Yang, Haonan Qju, Wayne Wu, Chen Change Loy, and Ziwei Liu. Text2human: Text-driven controllable human image generation. ACM Transactions on Graphics (TOG), 41(4):1–11, 2022. 1
- [42] Yuming Jiang, Shuai Yang, Tong Liang Koh, Wayne Wu, Chen Change Loy, and Ziwei Liu. Text2performer: Textdriven human video generation. In Proceedings of the

IEEE/CVF International Conference on Computer Vision,

2023. 2

- [43] Chen Jin, Ryutaro Tanno, Amrutha Saseendran, Tom Diethe, and Philip Teare. An image is worth multiple words: Learning object level concepts using multi-concept prompt learning. arXiv preprint arXiv:2310.12274, 2023. 3
- [44] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-toimage diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023. 2, 3
- [45] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 6
- [46] Sumith Kulal, Tim Brooks, Alex Aiken, Jiajun Wu, Jimei Yang, Jingwan Lu, Alexei A Efros, and Krishna Kumar Singh. Putting people in their place: Affordance-aware human insertion into scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17089–17099, 2023. 3
- [47] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. 2023. 2, 3
- [48] Dongxu Li, Junnan Li, and Steven CH Hoi. Blipdiffusion: Pre-trained subject representation for controllable text-to-image generation and editing. arXiv preprint

- arXiv:2305.14720, 2023. 2, 3

[49] Yuheng Li, Haotian Liu, Yangming Wen, and Yong Jae Lee. Generate anything anywhere in any scene. arXiv preprint

- arXiv:2306.17154, 2023. 3

- [50] Jun Hao Liew, Hanshu Yan, Jianfeng Zhang, Zhongcong Xu, and Jiashi Feng. Magicedit: High-fidelity and temporally coherent video editing. In arXiv, 2023. 3
- [51] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 2
- [52] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 6
- [53] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-p2p: Video editing with cross-attention control. arXiv preprint arXiv:2303.04761, 2023. 3
- [54] Zhiheng Liu, Yifei Zhang, Yujun Shen, Kecheng Zheng, Kai Zhu, Ruili Feng, Yu Liu, Deli Zhao, Jingren Zhou, and Yang Cao. Cones 2: Customizable image synthesis with multiple subjects. arXiv preprint arXiv:2305.19327, 2023. 3
- [55] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10209–10218, 2023. 2, 3
- [56] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subject-diffusion: Open domain personalized text-to-image

- generation without test-time fine-tuning. arXiv preprint arXiv:2307.11410, 2023. 3
- [57] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. arXiv preprint arXiv:2211.09794, 2022. 1
- [58] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 6
- [59] Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. Codef: Content deformation fields for temporally consistent video processing. arXiv preprint arXiv:2308.07926, 2023. 3
- [60] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. arXiv preprint arXiv:2303.09535, 2023. 3
- [61] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 2
- [62] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021. 1
- [63] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125,

2022. 3

- [64] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2021. 1, 3
- [65] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 3
- [66] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. 2022. 2, 3, 6, 7, 13
- [67] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. arXiv preprint arXiv:2307.06949, 2023. 3
- [68] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022. 1

- [69] Chaehun Shin, Heeseung Kim, Che Hyun Lee, Sang-gil Lee, and Sungroh Yoon. Edit-a-video: Single video editing with object-aware consistency. arXiv preprint arXiv:2303.07945,

2023. 3

- [70] Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. Freeu: Free lunch in diffusion u-net. arXiv preprint arXiv:2309.11497, 2023. 1
- [71] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 2, 3

- [72] Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, and Daniel Aliaga. Objectstitch: Object compositing with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18310–18319, 2023. 3
- [73] Dani Valevski, Danny Wasserman, Yossi Matias, and Yaniv Leviathan. Face0: Instantaneously conditioning a text-toimage model on a face. arXiv preprint arXiv:2306.06638,

2023. 3

- [74] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual description. arXiv preprint arXiv:2210.02399, 2022. 2, 3
- [75] Wen Wang, kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. Zero-shot video editing using off-the-shelf image diffusion models. arXiv preprint arXiv:2303.17599, 2023. 3
- [76] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023. 3, 4
- [77] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. arXiv preprint arXiv:2302.13848, 2023. 2, 3, 4, 6, 7, 13
- [78] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. arXiv preprint arXiv:2212.11565, 2022. 2, 3
- [79] Zijie Wu, Chaohui Yu, Zhen Zhu, Fan Wang, and Xiang Bai. Singleinsert: Inserting new concepts from a single image into text-to-image models for flexible editing. arXiv preprint arXiv:2310.08094, 2023. 3
- [80] Guangxuan Xiao, Tianwei Yin, William T Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multisubject image generation with localized attention. arXiv preprint arXiv:2305.10431, 2023. 2, 3
- [81] Xingqian Xu, Jiayi Guo, Zhangyang Wang, Gao Huang, Irfan Essa, and Humphrey Shi. Prompt-free diffusion: Taking ”text” out of text-to-image diffusion models. arXiv preprint arXiv:2305.16223, 2023. 2, 3

- [82] Tianfan Xue, Baian Chen, Jiajun Wu, Donglai Wei, and William T Freeman. Video enhancement with task-oriented flow. International Journal of Computer Vision, 127:1106– 1125, 2019. 13, 14
- [83] Hanshu Yan, Jun Hao Liew, Long Mai, Shanchuan Lin, and Jiashi Feng. Magicprop: Diffusion-based video editing via motion-aware appearance propagation. arXiv preprint arXiv:2309.00908, 2023. 3
- [84] Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. Rerender a video: Zero-shot text-guided video-to-video translation. arXiv preprint arXiv:2306.07954, 2023. 3
- [85] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 3

- [86] Ge Yuan, Xiaodong Cun, Yong Zhang, Maomao Li, Chenyang Qi, Xintao Wang, Ying Shan, and Huicheng Zheng. Inserting anybody in diffusion models via celeb basis. arXiv preprint arXiv:2306.00926, 2023. 3
- [87] Ziyang Yuan, Mingdeng Cao, Xintao Wang, Zhongang Qi, Chun Yuan, and Ying Shan. Customnet: Zero-shot object customization with variable-viewpoints in text-to-image diffusion models. arXiv preprint arXiv:2310.19784, 2023. 3
- [88] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. Controlvideo: Training-free controllable text-to-video generation. arXiv preprint arXiv:2305.13077, 2023. 3
- [89] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. Motiondirector: Motion customization of text-tovideo diffusion models. arXiv preprint arXiv:2310.08465,

2023. 3

- [90] Yuyang Zhao, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. Make-a-protagonist: Generic video editing with an ensemble of experts. arXiv preprint arXiv:2305.08850, 2023. 3
- [91] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022. 2, 3
- [92] Yufan Zhou, Ruiyi Zhang, Tong Sun, and Jinhui Xu. Enhancing detail preservation for customized text-to-image generation: A regularization-free approach. arXiv preprint arXiv:2305.13579, 2023. 3

Supplementary

- A. Comparison methods

Texutal Inversion. In Textual Inversion [21], the appearance of target subjects is embedded into the text embeddings. A text token S∗ is optimized to represent one specific subject. When applied to text-to-image models, multiple images containing the same object are required to optimize the text token S∗. In the setting of video generation, we directly use multiple video clips split from the original long video to optimize the text token. Once optimized, the text token S∗ is used to replace the word embeddings of the target subject in the sentence to sample new videos.

DreamBooth. In DreamBooth [66], target subjects are injected into text tokens and model weights simultaneously. During the training, both model weights and a special token S∗ are optimized. Similar to Textual Inversion, we use the original video and text description to train the model. Multiple video clips sampled from the long video are employed to optimize weights and text token S∗. Once trained, the text token S∗ is inserted before the word embeddings of the target object to sample new videos.

ELITE. Different from Textual Inversion and DreamBooth, ELITE [77] is an encoder-based method for fast customized generation. An encoder is trained to transform the images into embeddings. Local mapping and global mapping are employed to transform the CLIP embedding of image prompts into the features, which are fed into the crossattention module. We adapt the ELITE to video generation. We train the model using the same data and the same base video model.

- B. More Discussions on Ablation Study

In this section, we show one more visual example of the ablation study. In Fig. 7 of the main paper, we show that the model with only coarse embeddings results in imprecise encoding of appearance. Both the model trained with only fine embeddings and the model trained using the unified training strategy overfit to image prompts. In the two examples shown in Fig. 7 of the main paper, the first frames can take the image prompt, but the generated appearance is distorted along the frames. In Fig. A9, we discuss another case, which exhibits a different behaviour.

Only Coarse Embeddings. This ablation model injects the image prompts with only coarse embeddings via Image Encoder. As shown in Fig. A9(a), the coarse embeddings provide coarse but not precise guidance. The face of the dog and the shape of the head are not accurately captured. Our full model shown in Fig. A9(d) can capture the visual details correctly.

Only Fine Embeddings. In this ablation model, we only have fine embeddings of image prompts in cross-frame at-

tention layers. Recall that the purpose of fine embeddings is to refine the encoding from coarse levels. In the example shown in Fig. A9(b), the first frame does not successfully embed the image prompt. Without coarse embeddings, the generation of the appearance of the dog relies purely on the propagation from the first frame. The failure in encoding the image prompt into the first frame results in the following frames having random appearances for the dog.

The Necessity of Coarse-to-Fine Training. In VideoBooth, we propose the coarse-to-fine training strategy, i.e., train the coarse embeddings first and then train the attention injection module. This ablation model is trained within one stage. In the example shown in Fig. A9(c), the first frame successfully takes the appearance of the image prompt. The model generates a consistent appearance in all frames, but the motion of this generated clip is small and not aligned with the text prompt. We found that in the case of generating small motions or static frames, the coarse-to-fine training strategy can work well. However, when it comes to generating large motions as shown in Fig. 7 of the main paper, the appearance will be distorted along frames.

##### C. WaterMark Removal Module

Since the videos in WebVid dataset [5] have a watermark, the model trained using this dataset generates videos with a watermark in nature. To generate videos without watermark for better visual quality, we finetune the model with an additional module using the Vimeo dataset [82]. We only use text prompts and original videos to finetune the model. As shown in Fig. A8, we add six blocks before the last conv out layer of the base video model. The added six blocks can be regarded as a small UNet. After the first block, we downsample features by two times. Then after the second block, features are downsampled by two times. Then features are enhanced by two blocks. Finally, features are upscaled with two consecutive blocks. Each block upsamples features by two times. Inside each block, there are two ResNet blocks. Skip connections are adopted between downsampling blocks and upsampling blocks. After all blocks, we feed the model to one conv layer, which is initialized with zero. The motivation for zero initialization is to avoid the newly added blocks affecting the model. We add the obtained features to the original features as residues. The added features are fed into the final layer (i.e. Conv Out layer) of the base video generation model. The newly added modules and the last layer are optimized during the finetuning. After finetuning, the watermark can be removed without influencing the generative capability of VideoBooth. It should be noted that we use the model without watermark removal module when comparing with baselines.

### …

[Figure 148]

[Figure 149]

❄ ❄ 🔥 ❄

[Figure 150]

[Figure 151]

❄ 🔥

[Figure 152]

[Figure 153]

UNet WaterMark Removal Module

Cross-Frame Attention Cross Attention Temporal Attention Residual Block Conv Layer Conv Out

Figure A8. Illustration of Watermark Removal Module. We add a WaterMark Removal Module before the conv out layer. The output of the watermark removal module is added as a residue to the original features. We finetune the newly added module and the conv out layer using the video data [82] without watermarks.

[Figure 154]

Dog is running

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

(a) Only Coarse Embeddings with Image Encoder

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

(b) Only Fine Embeddings with Attention Injection

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

(c) Unified Training for Image Encoder and Attention Injection

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

(d) Full Model

###### Figure A9. More Visual Analysis on Ablation Study.

