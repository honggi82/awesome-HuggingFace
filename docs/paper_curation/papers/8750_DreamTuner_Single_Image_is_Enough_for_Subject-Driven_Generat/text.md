## DreamTuner: Single Image is Enough for Subject-Driven Generation

# arXiv:2312.13691v1[cs.CV]21Dec2023

Miao Hua∗ Jiawei Liu∗ Fei Ding∗ Wei Liu Jie Wu Qian He ByteDance Inc. BeiJing, China {huamiao,liujiawei.cc22,dingfei.212,liuwei.jikun,wujie.10,heqian}@bytedance.com https://dreamtuner-diffusion.github.io/

* Equal Contribution

|[Figure 1]<br><br>[Figure 2]|[Figure 3]|[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]|[Figure 8]|
|---|---|---|---|

|[Figure 9]<br><br>[Figure 10]|
|---|

[Figure 11]

User Input Image, [S*] (Single Image is enough)

[S*], swimming in the pool [S*], wearing sunglasses, sitting on the park bench

[S*], lying on the grass, eating watermelon

[S*], driving a convertible sports car

|[Figure 12]<br><br>[Figure 13]|
|---|

|[Figure 14]<br><br>[Figure 15]|[Figure 16]|[Figure 17]<br><br>[Figure 18]<br><br>[Figure 19]<br><br>[Figure 20]|[Figure 21]|
|---|---|---|---|

[Figure 22]

[S*], sitting on the sofa, hand on head

User Input Image, [S*] (Single Image is enough)

[S*], sitting on the grass, under the tree

[S*], in the forest, with butterflies

[S*], carrying a handbag, walking on the street

[Figure 23]

|[Figure 24]<br><br>[Figure 25]<br><br>[Figure 26]|[Figure 27]| |[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]| |
|---|---|---|---|---|

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

User Input Image, [S*] [S*] [S*], with white wings User Input Pose Condition

[S*], on the road, with buildings in the background

[S*], closed eyes

Figure 1. Subject-driven image generation results of our method. Our proposed DreamTuner could generate high-fidelity images of user input subject, guided by complex texts (the first two rows) or other conditions as pose (the last row), while maintaining the identity appearance of the specific subject. We found that a single image is enough for surprising subject-driven image generation.

### Abstract

Diffusion-based models have demonstrated impressive capabilities for text-to-image generation and are expected

for personalized applications of subject-driven generation, which require the generation of customized concepts with one or a few reference images. However, existing methods based on fine-tuning fail to balance the trade-off between

subject learning and the maintenance of the generation capabilities of pretrained models. Moreover, other methods that utilize additional image encoders tend to lose important details of the subject due to encoding compression. To address these challenges, we propose DreamTurner1, a novel method that injects reference information from coarse to fine to achieve subject-driven image generation more effectively. DreamTurner introduces a subject-encoder for coarse subject identity preservation, where the compressed general subject features are introduced through an attention layer before visual-text cross-attention. We then modify the self-attention layers within pretrained text-to-image models to self-subject-attention layers to refine the details of the target subject. The generated image queries detailed features from both the reference image and itself in selfsubject-attention. It is worth emphasizing that self-subjectattention is an effective, elegant, and training-free method for maintaining the detailed features of customized subjects and can serve as a plug-and-play solution during inference. Finally, with additional subject-driven fine-tuning, DreamTurner achieves remarkable performance in subject-driven image generation, which can be controlled by a text or other conditions such as pose.

### 1. Introduction

Latest developments in diffusion models have shown astonishing achievements in both text-to-image (T2I) generation [24, 30, 32, 34] and editing [3, 11, 45]. Further researches are expected for personalized applications of subject-driven image generation [9, 17, 23, 33, 41]. The goal of subjectdriven image generation is to synthesis various high fidelity images of customized concepts, as shown in Fig. 1. This is a practical task with a wide range of multimedia applications. For example, merchants can generate advertising images for specific products, and designers can create storybooks by simply drawing an initial character image. When generating images with only a single reference image, it becomes more practical as it reduces user operations and allows the initial reference image to be a generated one.

However, achieving single-reference image subjectdriven generation remains a challenging task due to the difficulty in balancing detailed identity preservation and model controllability. Current subject-driven image generation methods are primarily based on fine-tuning or additional image encoders. Fine-tuning-based methods [9, 17, 33] train pretrained T2I models on specific reference images to learn the identities of the target subjects. Though more training steps may improve identity preservation, they may also undermine the generation capacity of the pretrained models. Image-encoder-based methods [16, 23, 41] train

1This work was done in May 2023.

[Figure 36]

[Figure 37]

Generated images w/ reference features

[Figure 38]

[Figure 39]

[Figure 40]

Generated images w/o reference features

Reference image

“A small brown and white dog sitting on an orange background.”

“A photo of dog”

Figure 2. Exploration experiment of self-attention. It makes the generated images more similar to the reference one to use the reference image features for self-attention. Detailed text can better serve its purpose.

additional image encoders to inject features of the reference image into the pretrained T2I models. However, these methods require a trade-off of encoding compression. Strong compression may result in the loss of details of the target subject, while weak compression can easily result in the generated image collapsing to a simple reconstruction of the reference image.

In fact, as shown in some video generation methods [18, 42], self-attention layers in T2I models are detailed spatial contextual association modules that could be extended to multiple images for temporal consistency. We performed a simple experiment to demonstrate that selfattention layers could also preserve the detailed identity features of the subject, as shown in Fig. 2. At each diffusion time step, the reference image is noised through the diffusion process and the reference features could be extracted from the noised image. Then the reference features and those of the generated image are concatenated and used as the key and value of the self-attention layers. This approach resembles image outpainting, but it does not modify the convolutional layers and cross-attention layers. Consequently, we found that the generated images with reference features were significantly more similar to the reference image in terms of detailed subject identity preservation, indicating that self-attention plays a role in this aspect. However, as shown in the comparison of generated images guided by detailed and concise text, it is still necessary to introduce coarse identity preservation methods so that the model could generate a rough appearance to better utilize the self-attention layers.

Based on the above motivations, we propose DreamTuner, a novel subject-driven image generation method that injects the reference information from coarse to fine. Firstly, a subject-encoder is proposed for coarse identity preservation. The main object of the reference im-

###### Stage 1: Subject Encoder Pre-training

Stage 2: Subject-Driven Fine-tuning

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

ℒ

×T

[Figure 45]

ControlNet

[Figure 46]

[Figure 47]

[Figure 48]

diffusion

SE

layout

“A dog [S*]”

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

diffusion

| | |
|---|---|
| | |

[Figure 55]

Text Image

SE

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

SE

[Figure 61]

[Figure 62]

content

[Figure 63]

ℒ

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Subject Encoder

“A dog”

ℒ

Regular Data Partial Frozen

Self-Subject-Attention Transformer Block

[Figure 68]

Frozen

[Figure 69]

Stage 3: Subject-Driven Inference

###### Subject-Driven Inference with Text Condition. Subject-Driven Inference with Text and Layout Condition.

Edit Layout

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

diffusion

[Figure 80]

[Figure 81]

diffusion

ControlNet

| |SE|
|---|---|
| | |

| | |
|---|---|
| | |

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Reference Text

Reference Text

Text [S*]

| | |
|---|---|
| | |

SE

Text [S*]

×T

×T

Figure 3. Overview of the proposed DreamTuner framework. Firstly, a subject-encoder (SE) is trained for coarse identity preservation, where a frozen ControlNet is utilized to maintain the layout. Then an additional fine-tuning stage like existing methods is conducted with proposed subject-encoder and self-subject-attention for fine identity preservation. Finally a refined subject driven image generation model is obtained which could synthesis high-fidelity images of the specific subject controlled by text or other layout conditions. It is worth noting that both of the subject-driven fine-tuning stage and inference stage require only a single reference image.

age is compressed by a pretrained CLIP [27] image encoder. And the compressed features are injected to the T2I model through an additional attention layer before visualtext cross-attention. Furthermore, we introduce a training technique that decouples content and layout for the subjectencoder. To achieve this, we add a pretrained depth-based ControlNet [45] to avoid the subject-encoder focusing on the layout of reference image. During training, ControlNet is frozen to control the layout of generated images so that the subject-encoder could only need to learn the content of the reference image. Next, we alter the self-attention layers to self-subject-attention to preserve fine identity. Based on the above-mentioned method to inject reference features into self-attention layers, self-subject-attention further introduces weight and mask strategies for improved control. Notably, self-subject-attention is a graceful, efficient, and training-free technique for preserving identity that can be used plug-and-play during inference. Lastly, an extra phase of fine-tuning can enhance identity preservation. With the aid of subject-encoder and self-subject-attention, the finetuning procedure only needs a few training steps.

erate high-fidelity images of a specific subject with only a single reference image.

- • A subject-encoder is proposed for coarse identity preservation, which could also reduce the fine-tuning time cost, along with a content and layout decoupled training method.
- • A novel self-subject-attention is proposed as an elegant, effective, and plug-and-play method for refined subject identity preservation.

### 2. Related Works

#### 2.1. Text-to-Image Generation

The development of text-to-image generation has evolved from Generative Adversarial Networks (GANs) [10] based methods [1, 26, 40, 43] in early years to Vector Quantization Variational AutoEncoders (VQVAEs) [31, 38] and Transformers [39] based methods [7, 8, 29, 44], and more recently to diffusion based methods [13, 15, 30, 32, 34, 37].

Diffusion models are probabilistic generative models that learn to restore the data distribution perturbed by the forward diffusion process. At the training stage, the data distribution is perturbed by a certain scale of Gaussian noise in different time steps, and the diffusion models are trained

The main contributions are summarized as follows:

• We propose a novel image encoder and fine-tuning based subject-driven image generation method that could gen-

to predict the denoised data. The data distribution with the largest scale of noise tends to standard normal distribution. Thus at the inference stage, diffusion models could sample high-fidelity data from Gaussian noise step by step.

Text-to-Image U-Net Model

Conv Block

… …

Transformer Block

Large-scale diffusion models [30, 32, 34] have achieved state-of-the-art performance on text-to-image generation. DALLE-2 [30] builds a prior model to transform CLIP [27] text embeddings to CLIP image embeddings and further generates high-resolution images using large-scale hierarchical U-Nets which consists of a image generation decoder and some image super-resolution models. IMAGEN [34] adopt a larger pretrained text encoder [28] for better imagetext consistency. LDM [32] compresses the images through a pretrained auto-encoder and perform diffusion in the latent space, which leads to efficient high-resolution image generation. With the help of classifier-free guidance [12], high-fidelity images can be generated by these large-scale diffusion based text-to-image models. However, these models are all designed for general scenarios that are controlled only by text. For subject driven generation that requires subject identity preservation, text is far from enough.

Attention

Attention

Attention

Frozen Trainable

Cross

FFN

Self

S-E

𝛽

[Figure 86]

ResBlock

[Figure 87]

|Paired Text|
|---|

Image

CLIP

CLIP

Text

Subject Encoder

Figure 4. Illustration of the text-to-image generation U-Net model with proposed subject-encoder.

age synthesis and editing without fine-tuning. Our proposed method combines the strengths of those works, it first coarsely injects the subject information using a subjectencoder, then fine-tunes the model with the novel selfsubject-attention for better identity preservation.

#### 2.2. Subject-Driven Generation

Subject-driven image generation requires the model to understand the visual subject contained in the initial reference images and synthesize totally unseen scene of the demonstrated subjects. Most existing works use a pretrained synthesis network and perform test-time training for each subject. For instance, MyStyle [25] adopts a pretrained StyleGAN for personalized face generation. Later on, some works fine-tune the pretrained T2I models to adapt image generation to a specific unseen subject. Dreambooth [33] fine-tunes all the parameters, Texture Inversion [9] introduces and optimizes a word vector for the new concept, Custom-Diffusion [21] only fine-tunes a subset of crossattention layer parameters. For those fine-tuning based methods, more training steps leads to better identity preservation but undermines the generation capacity of the pretrained model. Another line of works train additional image encoders to inject the reference image information for subject generation. ELITE [41] and Instantbooth [36] use two mapping networks to convert the reference images to a textual embedding and additional cross-attention layers for subject generation. Taming Encoder[16] adopts an encoder to capture high-level identifiable semantics of subjects. SuTI [6] achieves personalized image generation by learning from massive amount of paired images generated by subject-driven expert models. However, these methods are trained with weakly-supervised data and performs without any test-time fine-tuning, thus leading to much worse faithfulness than DreamBooth [33]. Recently, MasaCtrl [4] proposes a mask-guided mutual self-attention strategy similar as our self-subject-attention for text-based non-rigid im-

### 3. Method

In this paper, we propose DreamTuner as a novel framework for subject driven image generation based on both fine-tuning and image encoder, which maintains the subject identity from coarse to fine. As shown in Fig. 3, DreamTuner consists of three stages: subject-encoder pretraining, subject-driven fine-tuning and subject-driven inference. Firstly, a subject-encoder is trained for coarse identity preservation. Subject-encoder is a kind of image encoder that provides compressed image features to the generation model. A frozen ControlNet [45] is utilized for decoupling of content and layout. Then we fine-tune the whole model on the reference image and some generated regular images as in DreamBooth [33]. Note that subjectencoder and self-subject-attention are used for regular images generation to refine the regular data. At the inference stage, the subject-encoder, self-subject-attention, and subject word [S*] obtained through fine-tuning, are used for subject identity preservation from coarse to fine. Pretrained ControlNet [45] could also used for layout controlled generation.

#### 3.1. Subject-Encoder

Image encoder based subject driven image generation methods [36, 41] have proved that the compressed reference image features can guide the pretrained text-to-image model to generate a general appearance of the subject. Thus we

propose subject-encoder as a kind of image encoder that provides a coarse reference for subject driven generation. As shown in Fig. 4, a frozen CLIP [27] image encoder is used to extract the compressed features of reference image. Salient Object Detection (SOD) model or segmentation model are used to remove the background of the input image and emphasize the subject. Then some residual blocks (ResBlock) are introduced for domain shift. Multilayer features extracted by CLIP are concatenated in the channel dimension and then adjust to the same dimension

|[Figure 88]<br><br>[Figure 89]|
|---|

S-A Value

| | |
|---|---|

S-A Key

| | |
|---|---|

Reference Image

Softmax

|[Figure 90]<br><br>[Figure 91]|
|---|

[Figure 92]

[Figure 93]

log

S-A Query

∗ 𝜔

Generated Image

Self-Subject-Attention

Figure 5. Illustration of the proposed self-subject-attention. S-A indicates self-attention.

- as the generated features through the residual blocks. The encoded reference features of subject-encoder are

noised reference image, which share the same data distribution with the generated features at time step t. The original self-attention layers are modified to self-subject-attention layers by utilizing reference features. As demonstrated in Fig. 5, the features of generated image are taken as the query and the concatenation of generated features and reference features is taken as the key and value. To eliminate the influence of reference image background, SOD model or segmentation model are used to create a foreground mask, which uses 0 and 1 to indicate the background and foreground. Besides, the mask can also be used to adjust the scale of the impact of reference image through a weight strategy, i.e., multiply the mask by an adjustment coefficient ωref. The mask works as an attention bias, thus a log function is used as a preprocessing. It is worth noting that the extraction of reference features is unrelated to the generated image, so additional reference text can be used to enhance consistency with the reference image.

injected to the text-to-image model using additional subjectencoder-attention (S-E-A) layers. The subject-encoderattention layers are added before the visual-text crossattention, because the cross-attention layers are the modules that control the general appearance of generated images. We build the subject-encoder attention according to the same settings as cross-attention and zero initial the output layers. An additional coefficient β is introduced to adjust the influence of subject-encoder, as in Eq. 1.

S-E-A(z,SE(r))′ = β ∗ S-E-A(z,SE(r)) (1)

where z is the generated image features and r is the reference image. It should be noted that only the additional residual blocks and subject-encoder-attention layers are trainable during training to maintain the generation capacity of the text-to-image model.

Besides, we found that the subject-encoder will provide both the content and the layout of the reference image for text-to-image generation. However, in most cases, layout is not required in subject driven generation. Thus we further introduce ControlNet [45] to help decouple the content and layout. Specifically, we train the subject-encoder along with a frozen depth ControlNet, as in Fig. 3. As the ControlNet has provided the layout of reference image, the subject-encoder can focus more on the subject content.

The original Self-Attention (S-A) is formulated as Eq. 2.

q · kT √

S-A(z) = Softmax(

) · v w.r.t. q = Query(z),k = Key(z),v = V alue(z)

(2)

d

where Query,Key,V alue are linear layers in selfattention, d is the dimension of q,k,v. The proposed SelfSubject-Attention (S-S-A) is formulated as Eq. 3.

#### 3.2. Self-Subject-Attention

q · [k,kr]T √

S-S-A(z,zr) = Softmax(

+ log m) · [v,vr] w.r.t. kr = Key(zr),vr = V alue(zr)

As discussed in some text-to-video works [18, 42], selfattention layers in pretrained text-to-image models could be used for content consistency across-frames. Since the subject-encoder has provided general appearance of the specific subject, we further propose self-subject-attention based on the self-attention layers for fine subject identity preservation. The features of reference image extracted by the text-to-image U-Net model are injected to the selfattention layers, which can provide refined and detailed reference because they share the same resolution with the generated features. Specifically, as shown in Fig. 3 Stage-2 and Stage-3, the reference image are noised through diffusion forward process at time step t. Then the reference features before self-attention layers are extracted from the

d

m = [J,ωref ∗ mr]

(3) where zr is the reference image features, [] indicates concatenation, mr is the foreground mask of reference image, J is a all-ones matrix. As in Re-Imagen [5], there are more than one conditions in DreamTuner, thus we modify the original classifier free guidence to a interleaved version:

ˆϵrt = ωr ∗ ϵθ(xt,c,xrt−∆t,t) − (ωr − 1) ∗ ϵθ(xt,c,∅,t)

- (4)

ˆϵct = ωc ∗ ϵθ(xt,c,xrt−∆t,t) − (ωc − 1) ∗ ϵθ(xt,uc,xrt+∆t′,t)

- (5)

ϵ ˆrt, λt ≤ pr ϵˆct, λt > pr

λt ∼ U(0,1) (6)

ϵˆt =

where xt is the generated image at time step t, c is the condition, uc is the undesired condition, ωr and ωc are the guidance scales of reference image and text condition. xrt−∆t and xrt+∆t′ are the diffusion noised reference images

- at time step t − ∆t and t + ∆t′, ∆t and ∆t′ are small time step biases which are used for the adjustment of noise intensity of reference image. ϵˆθ is the diffusion model and θ represents the parameters. ϵˆt is the final output at time step t. Eq. 4 emphasizes the guidance of reference image

and Eq. 5 emphasizes the guidance of condition, where pr controls the possibility of selecting Eq. 4.

- 3.3. Subject Driven Fine-tuning

While excellent performance can be achieved through SE and S-S-A, a few additional fine-tuning steps leads to better subject identity preservation. As shown in Fig. 3 Stage2, We conduct the fine-tuning stage similar to DreamBooth [33], while other fine-tuning methods, such as LoRa [14], also works. Firstly, some regular images are generated to help the model learn the specific identity of the subject and maintain the generation capacity. The paired data are built as {subject reference image, ”A class word [S*]”} and {regular image, ”A class word”}. Then the paired data is used to fine-tune the pretrained text-to-image model. All of the parameters including CLIP text encoder are trainable in this stage as in DreamBooth. In order to achieve more refined subject learning, we make four improvements:

- 1. The background of reference image is replaced by a simple white background to focus on the subject itself.
- 2. The word [S*] is an additional trainable embedding as in Textual Inversion [9] rather than a rare word [33].
- 3. The subject-encoder is trained with the text-to-image generation model for better subject identity preservation.
- 4. To learn the detailed identity of the subject, we generate regular images that are more similar to the reference image by using subject-encoder, self-subject-attention and detailed caption of the reference image. The model are required to learn the subject details to distinguish ”A class word” and ”A class word [S*]”

- 4. Experiments

- 4.1. Implementation Details

We build our DreamTuner based on the Stable Diffusion model [32]. For the subject-encoder, CLIP features from layers with layer indexes {25, 4, 8, 12, 16} are selected as in ELITE [41]. Elastic transform and some other data augmentation methods are used to enhance the difference between input reference images and the generated images. InSPyReNet [20] is adopted as the SOD model for natural images and a pretrained anime segmentation model is

Table 1. Subject fidelity (CLIP-I) and prompt fidelity (CLIP-T) quantitative metric comparison. The top-2 results of each metric have been emphasized.

Method CLIP-T CLIP-I Textual Inversion 0.251 0.722

Masactrl 0.296 0.735

ELITE 0.271 0.726 DreamBooth 0.267 0.788 DreamTuner 0.281 0.767

used to separate the foreground and background for anime character images. For self-subject-attention, pr is set to 0.9, ωref is set to 3.0 for natural images and 2.5 for anime images. The training steps of subject-driven fine-tuning is set to 800-1200. BLIP-2 [22] and deepdanbooru [19] are used as the caption models for natural images and anime character images. More details could be found in Appendix-B.

#### 4.2. Subject Driven Generation Results

Subject driven generation results could be found in Fig. 6, consisting of static objects, animals, and anime characters. It could be found that the subject identities have been preserved well and the generated images are consistent with the text condition. The subject appearance is maintained even with some complex texts. It is worth noting that DreamTuner could achieve excellent identity preservation on details, e.g., the words on the can in the first row, the white stripes of the dog in the second raw, the eyes and clothes of the anime characters, etc.

#### 4.3. Comparison with Other Methods

The comparison results between our proposed DreamTuner and other subject driven image generation methods are shwon in Fig. 7. The corresponding quantitative metric comparison could be found in Table. 1. We use CLIPT, the average cosine similarity between CLIP image embeddings and CLIP text embeddings, as the metric to evaluate the consistency between generated images and input text prompts. CLIP-I, the average cosine similarity between CLIP embeddings of generated images and reference images, is adopted as a metric that indicates the subject fidelity. Only a single image are used as the reference image for each method. It could be found that our proposed DreamTuner outperforms the fine-tuning based methods (DreamBooth [33] and Textual Inversion [9]), image encoder based method (ELITE [41]) and improved selfattention based method (MasaCtrl [4]). DreamBooth [33] performs best in CLIP-I but not well on CLIP-T, as it is easy to overfit on such a single reference image with only 1200 steps, and less steps leads to worse subject fidelity. Besides, it could be found in Fig. 7 that DreamTuner can generate

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

With a mountain in the background

Input Image In the snow

Input Image In the snow Floating on top of water

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

Input Image On the beach In the snow Input Image In the jungle On top of a wooden floor

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Sitting on a cobblestone street Input Image

Running on the grass with autumn leaves

Jumping into the air and biting a frisbee

Sleeping in a cardboard box

Input Image

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Sitting at the table with a cup of tea in hands, sunlight streaming through the window

Input Image

Sitting on the sofa Input Image Sitting on the park bench Sleep in bed, closed eyes

Figure 6. Subject driven generation results of DreamTuner.

better subject details benefiting from subject-encoder and self-subject-attention. Similar to DreamBooth, Textual Inversion [9] trained with only a single image also has difficulties in the trade-off between avoid overfitting and identity preservation. Compared with ELITE [41], DreamTuner use additional self-subject-attention and subject-driven finetuning that leads to better identity preservation. MasaC-

trl [4] shows the best CLIP-T because it does not train the text-to-image model and inherits the creative power of the pretrained model. However, it only use a improved self-attention for identity preservation while our proposed DreamTuner introduces subject-encoder and subject-driven fine-tuning, thus achieves better subject fidelity.

A can [S*] on the beach

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

A vase [S*] with a wheat field in the background

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

A dog [S*] with a mountain in the background, red hat and sunglasses

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

A dog [S*] swimming in the pool

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Input Ours DreamBooth Textual Inversion ELITE MasaCtrl

Figure 7. Comparison between our proposed DreamTuner and state-of-the-art subject driven image generation methods.

#### 4.4. Ablation Study

We conduct ablation studies to evaluate the effects of various components in our method, including the subjectencoder and the self-subject-attention. Fig. 10 shows the input reference image, corresponding prompt, and results of different settings.

Effect of Subject-Encoder. For subject-encoder, β is introduced to control how much subject information is injected into the model. To evaluate its effect, we vary its value from 0.0 to 0.5, and the generated results are shown in the first row of Fig. 10. From the figure, we see that with the increasing of β, the consistency between the synthesized image and generated image is improved. However, when the value of β is too large(i.e., 0.5), it may lead to degenerated editing results. As a result, for a tradeoff between inversion and editability, we set β = 0.2. We found that this

setting works well for most cases. We also conduct studies of training subject-encoder without ControlNet [45]. In this case, the subject-encoder cannot decouple the content and layout properly, therefore the generated images may have similar layout as the reference images, rather than matching the input prompt.

Effect of Self-Subject-Attention. We further analyze the effect of the proposed self-subject-attention. From the images in the last row of Fig. 10, we observe a similar phenomenon when increasing the ωref from 0.0 to 10.0. When ωref is too small, such as close to 0.0, the generated image is not similar to the input image, and a large ωref value may degrade the quality of the generated image. When using SS-A without mask, i.e., mr is set to an all-ones matrix, the white background of reference image will also play a significant role and do harm to the generated image. More ablations are provided in Appendix-C.

[Figure 148]

Input

[Figure 149]

Input

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

β=0.0

β=0.1

β=0.2

β=0.5

SE without ControlNet

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

S-S-A without mask

ωref=0.0

ωref=1.0

ωref=2.5

ωref=10.0

Figure 8. Ablation study of subject-encoder and self-subject-attention. The input text is ’a photo of girl [S*], sitting at a desk, typing on a computer’

### 5. Conclusion

In this paper, we focus on the task of subject driven image generation with only a single reference image. Our method, DreamTuner, combines the strengths of existing fine-tuning and additional image encoder-based methods. Firstly, we use a subject-encoder for coarse subject identity preservation. Secondly, we fine-tune the model with self-subjectattention layers to refine the details of the target subject. This allows us to generate high-fidelity images based on only one reference image, with fewer training steps. Furthermore, our method is highly flexible in editing learned concepts into new scenes, guided by complex texts or additional conditions such as pose. This makes it an invaluable tool for personalized text-to-image generation.

### References

- [1] Rameen Abdal, Peihao Zhu, John Femiani, Niloy J. Mitra, and Peter Wonka. Clip2stylegan: Unsupervised extraction of stylegan edit directions. In Special Interest Group on Computer Graphics and Interactive Techniques Conference, pages 48:1–48:9, 2022. 3
- [2] Anonymous, Danbooru community, and Gwern Branwen. Danbooru2021: A large-scale crowdsourced and tagged anime illustration dataset. https://gwern.net/ danbooru2021, January 2022. Accessed: DATE. 11
- [3] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions.

2022. 2

- [4] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. arXiv preprint arXiv:2304.08465, 2023. 4, 6, 7
- [5] Wenhu Chen, Hexiang Hu, Chitwan Saharia, and William W.

- Cohen. Re-imagen: Retrieval-augmented text-to-image generator. arXiv preprint arXiv:2209.14491, 2022. 5
- [6] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and William W. Cohen. Subject-driven text-to-image generation via apprenticeship learning. 2023. 4
- [7] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, and Jie Tang. Cogview: Mastering text-toimage generation via transformers. In Advances in Neural Information Processing Systems, pages 19822–19835, 2021. 3
- [8] Patrick Esser, Robin Rombach, and Bj¨orn Ommer. Taming transformers for high-resolution image synthesis. In Conference on Computer Vision and Pattern Recognition, pages 12873–12883, 2021. 3
- [9] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. 2022. 2, 4, 6, 7
- [10] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 3
- [11] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. 2022. 2
- [12] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 4
- [13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 3
- [14] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 6

- [15] Nisha Huang, Fan Tang, Weiming Dong, and Changsheng Xu. Draw your art dream: Diverse digital art synthesis with multimodal guided diffusion. In Proceedings of the 30th ACM International Conference on Multimedia, pages 1085– 1094, 2022. 3
- [16] Xuhui Jia, Yang Zhao, Kelvin C.K. Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, and YuChuan Su. Taming encoder for zero fine-tuning image customization with text-to-image diffusion models. 2023. 2, 4
- [17] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. 2022. 2
- [18] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-toimage diffusion models are zero-shot video generators. 2023. 2, 5
- [19] Kichang Kim, Rachmadani Haryono, and Arthur Guo. Deepdanbooru. 2019. 6
- [20] Taehun Kim, Kunhee Kim, Joonyeong Lee, Dongmin Cha, Jiho Lee, and Daijin Kim. Revisiting image pyramid structure for high resolution salient object detection. In Asian Conference on Computer Vision, pages 257–273, 2022. 6
- [21] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. 2023. 4
- [22] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models.

2023. 6

- [23] Yiyang Ma, Huan Yang, Wenjing Wang, Jianlong Fu, and Jiaying Liu. Unified multi-modal latent diffusion for joint subject and text conditional image generation. 2023. 2
- [24] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2
- [25] Yotam Nitzan, Kfir Aberman, Qiurui He, Orly Liba, Michal Yarom, Yossi Gandelsman, Inbar Mosseri, Yael Pritch, and Daniel Cohen-Or. Mystyle: A personalized generative prior. arXiv preprint arXiv:2203.17272, 2022. 4
- [26] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Text-driven manipulation of stylegan imagery. In International Conference on Computer Vision, pages 2065–2074, 2021. 3
- [27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763, 2021. 3, 4, 5, 11
- [28] Colin Raffel, Minh-Thang Luong, Peter J Liu, Ron J Weiss, and Douglas Eck. Online and linear-time attention by en-

- forcing monotonic alignments. In International Conference on Machine Learning, pages 2837–2846, 2017. 4
- [29] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In Proceedings of the International Conference on Machine Learning, pages 8821– 8831, 2021. 3
- [30] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125,

2022. 2, 3, 4

- [31] Ali Razavi, A¨aron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with VQ-VAE-2. In Advances in Neural Information Processing Systems, pages 14837–14847, 2019. 3
- [32] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Conference on Computer Vision and Pattern Recognition, pages 10684–10695,

2022. 2, 3, 4, 6

- [33] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. 2022. 2, 4, 6, 11
- [34] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022. 2, 3, 4
- [35] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models. 2022. 11
- [36] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without testtime finetuning. 2023. 4
- [37] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 3
- [38] A¨aron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In Advances in Neural Information Processing Systems, pages 6306–6315, 2017. 3
- [39] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998–6008, 2017. 3
- [40] Hao Wang, Guosheng Lin, Steven C. H. Hoi, and Chunyan Miao. Cycle-consistent inverse gan for text-to-image synthesis. In ACM International Conference on Multimedia, page 630–638, 2021. 3
- [41] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual con-

- cepts into textual embeddings for customized text-to-image generation. 2023. 2, 4, 6, 7
- [42] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. 2022. 2, 5
- [43] Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and Xiaodong He. Attngan: Finegrained text to image generation with attentional generative adversarial networks. In Conference on Computer Vision and Pattern Recognition, pages 1316–1324, 2018. 3
- [44] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation. 2022. 3
- [45] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. 2023. 2, 3, 4, 5, 8, 11

### A. More Implementation Details

In this section, we will provide more implementation details.

For the pretraining of subject-encoder, we use the pretrained OpenCLIP 2 model to extract multilayer image features. We selected the features from indices {25, 4, 8, 12, 16}. We then train our encoder on two datasets: the natural images dataset, which is a sub-set of LAION-5B [35], and the anime character images dataset, which is selected from Danbooru [2]. For natural images, we use the V1-

- 5 version of Stable Diffusion, and for anime images, we use the Anything V3 model 3. We add a pretrained depthbased ControlNet [45] to make the subject-encoder focus on content since layout is controlled by the ControlNet. To enhance the robustness of the subject-encoder, we use data augmentation techniques such as elastic transform, random flip, blur, and random scale and rotation. We set β = 1.0 during the training process and use a batch size of 16 with a learning rate of 1e − 5. We conduct all experiments on 4 A100 GPUs, and the training time for each experiment is approximately one week.

At the subject-driven fine-tuning stage, we generate 32 regular images for each subject. Subject-encoder and selfsubject-attention are used to generate regular images that are similar to the reference image, where β is set to 0.2, ωref is set to 3.0 for natural images and 2.5 for anime images. Mask strategy is not used in self-subject-attention when generating regular images, as the background is also expected to be maintained. The U-Net and CLIP [27] text

- 2https://github.com/mlfoundations/open_clip
- 3https://huggingface.co/Linaqruf/anything-v3.0

encoder in the Stable Diffusion model , and the ResBlocks in the subject-encoder are trainable during fine-tuning. The CLIP image encoder is still frozen. To use a specific word to represent the subject, we train an additional word embedding [S*], which is initialized by the class word. The learning rate of the [S*] word embedding is set to 5e-3 and the learning rate of other parameters is set to 1e-6. The batch size is set to 2, consisting of a reference image and a regular image. Flip with a probability of 0.5 is adopted as a data augmentation method. We fine-tune the model for 800-1200 steps. It is worth noting that 800 steps are enough for most subjects.

### B. More Generated Results

In the paper, we have shown the generated results guided by text descriptions. And the pretrained ControlNets [45] could also be used for DreamTuner through a simple model transfer process [45]. As shown in Fig. ??, with an additional pose version ControlNet, DreamTuner can generate high fidelity images of the reference anime character, which share the same pose as the input conditions. Besides, the text prompts can still play its role. For example, the background could be controlled by the text descriptions, such as simple white background, road, etc. The expression of the generated characters could also be controlled by the text, such as closed eyes. Additional objects like flowers and white wings could also be added through text prompts. Besides, DreamTuner could also be used for pose controlled subject-driven video generation, as shwon in Fig. 9. We also use the first frame and the previous frame as the reference images in self-subject-attention for cross-frame coherence. The generated video could be found in the project page4.

### C. Ablation Study of Fine-tuning

We conduct ablation studies on fine-tuning stage to evaluate the effects of subject-encoder and self-subject-attention, as depicted in Fig. 10. The figure displays the input reference image, corresponding prompt, and results of various settings. The first row presents the results of original DreamBooth [33] achieved at different training steps. The second row exhibits results of DreamTuner without subjectencoder at the fine-tuning stage, while the last row shows the results of integral DreamTuner. The figure elucidates that our method generates images that resemble the input reference image and comply well with the target prompt. Additionally, our method converges faster than the original DreamBooth, which benefits from the subject-encoder. At approximately 300 fine-tuning steps our method produces satisfactory results.

4https://dreamtuner-diffusion.github.io/

|[Figure 160]|
|---|

|[Figure 161]|
|---|

|[Figure 162]|
|---|

|[Figure 163]|
|---|

|[Figure 164]|
|---|

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

|[Figure 170]|
|---|

[Figure 171]

|[Figure 172]|
|---|

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

|[Figure 176]|
|---|

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Reference Image

Figure 9. Text and pose controlled subject-driven video generation results.

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

DreamBooth

a photo of girl [S*], sleep in bed, closed eyes.

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

DreamTuner w/o subject-encoder

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

Input

DreamTuner

0 step 300 steps 600 steps 900 steps 1200 steps

Figure 10. Ablation study at the subject-driven fine-tuning stage.

