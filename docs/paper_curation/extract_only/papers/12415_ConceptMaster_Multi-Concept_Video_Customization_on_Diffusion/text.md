## ConceptMaster: Multi-Concept Video Customization on Diffusion Transformer Models Without Test-Time Tuning

# arXiv:2501.04698v2[cs.CV]13May2025

Yuzhou Huang1,2,3∗ Ziyang Yuan2,4∗ Quande Liu2† Qiulin Wang2 Xintao Wang2 Ruimao Zhang1† Pengfei Wan2 Di Zhang2 Kun Gai2

1Sun Yat-sen University 2Kuaishou Technology 3The Chinese University of Hong Kong, Shenzhen 4Tsinghua University https://yuzhou914.github.io/ConceptMaster/

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

“A man standing in front of a store.”

“A man and a woman kissing each other in front of a bridge.”

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

“A man rides a horse running in a grassy field.”

“A woman taking a bouquet of flowers, snowy winter.”

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

|[Figure 24]|
|---|

|[Figure 25]|
|---|

|[Figure 26]|
|---|

|[Figure 27]|
|---|

“A cat is playing with a Christmas tree at home.”

“A dog is playing with a bird on the road.”

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

|[Figure 36]|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

|[Figure 40]|
|---|

|[Figure 41]|
|---|

“A child feeding a dog in front of the small house in the park.”

“A man is happily playing the guitar on sofa on the beach.”

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Figure 1. We propose ConceptMaster, a Multi-Concept Video Customization (MCVC) method that can create high-quality conceptconsistent videos based on given multiple reference images without test-time tuning. Representatively, we demonstrate ConceptMaster’s video customization capacity on six scenarios, including 1) multiple persons, 2) persons with livings, 3) persons with stuffs, 4) multiple livings, 5) livings with stuffs and 6) persons with both livings and stuffs.

#### Abstract

multi-concept embeddings and inject them into diffusion models in a standalone manner, which effectively guarantees the quality of customized videos with multiple identities, even for highly similar visual concepts. To overcome the scarcity of high-quality MCVC data, we establish a data construction pipeline, which enables collection of high-quality multi-concept video-entity data pairs across diverse scenarios. A multi-concept video evaluation set is further devised to comprehensively validate our method from three dimensions, including concept fidelity, identity decoupling ability, and video generation quality, across six different concept composition scenarios. Extensive experiments demonstrate that ConceptMaster significantly outperforms previous methods for video customization tasks, showing great potential to generate personalized and semantically accurate content for video diffusion models.

Text-to-video generation has made remarkable advancements through diffusion models. However, Multi-Concept Video Customization (MCVC) remains a significant challenge. We identify two key challenges for this task: 1) the identity decoupling issue, where directly adopting existing customization methods inevitably mix identity attributes when handling multiple concepts simultaneously, and 2) the scarcity of high-quality video-entity pairs, which is crucial for training a model that can well represent and decouple various customized concepts in video generation. To address these challenges, we introduce ConceptMaster, a novel framework that effectively addresses the identity decoupling issues while maintaining concept fidelity in video customization. Specifically, we propose to learn decoupled

∗ Work done during an internship at KwaiVGI, Kuaishou Technology † Corresponding author

#### 1. Introduction

Diffusion-based text-to-video generation models, trained on extensive text-video data pairs, have demonstrated remarkable success in generating high-quality videos from textual inputs [2, 3, 6, 11, 15, 19, 27, 42, 52, 56, 58, 64]. These advancements have sparked increasing interest in personalizing video generation through user-defined concepts. Recently, some methods have been proposed to produce customized videos using additional image guidance, with proven effectiveness for the customization on object [26], human [18], style [38], etc.

Existing approaches for concept customization primarily fall into two methodological categories: tuning-based solutions and pre-training-based methods. The tuning-based solutions [9, 13, 16, 31, 50] typically first optimize model parameters (e.g., variants of LoRAs [23] or fully training latent diffusion models [50]) each time for customized concepts and then incorporate them for inference. However, these methods are computationally time-consuming and often require the manual collection of multiple reference samples, rendering them impractical in most time-sensitive and user-friendly scenarios. Conversely, pre-training-based methods [8, 14, 18, 26, 34, 36, 60–62, 67] aim to integrate visual embeddings into diffusion models at training time in a data-driven manner, enabling personalization without additional test-time tuning. Despite their progress, how to adopt these approaches to simultaneously process multiple concepts in videos to keep both concept fidelity and factorization in a feed-forward approach remains challenging.

In this paper, we study the unsolved challenging problem of Multi-Concept Video Customization (MCVC) without test-time tuning, which introduces two critical difficulties: 1) The identity decoupling problem, unlike single-concept processing, the MCVC task demands not only representing every concept individually based on the given multiple references, but also precisely differentiating the attributes across them in generated videos. Simply adopting existing pretrain-based approaches often leads to the conflation of visual concepts, inadvertently blending attributes from distinct individuals. This issue becomes more pronounced when dealing with concepts that contain similar attributes. A naive composite method is to firstly apply multi-concept image customization based on multiple references, and then input the generated image into image-to-video (I2V) models [68] for animation. However, this approach will be simultaneously subject to the representation and decoupling capability of two models, easily resulting in inferior generation quality and concept consistency. As illustrated in Fig. 2, both these two solutions can neither represent each concept well, nor clearly decouple them across visual appearances, resulting in unacceptable customized videos. 2) The scarcity of suitable high-quality MCVC datasets. Ideally, training such a customization model requires extensive

|[Figure 50]|[Figure 51]|
|---|---|

“A man and a woman walking down a city street.”

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

(a) ID-Animator

|[Figure 56]|[Figure 57]|
|---|---|

“A dog is playing with a bird on the road.”

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

(b) MS-Diffusion & I2VGen-XL

Figure 2. Directly applying single-concept method cannot handle the MCVC task, while the naive solution by combining multiconcept image generation and image-to-video generation models can also hardly create satisfactory customized results.

videos featuring diverse concepts, accompanied by precise textual descriptions and reference images of each entity. Current data sources significantly fall short of these requirements, while how to collect large-scale, pairwise videoentity data remains extremely challenging due to the accurate extraction of multiple concepts contained in videos across vast diversity of both visual and textual concepts.

To overcome these challenges, we propose ConceptMaster (see Fig. 1), an effective MCVC method that can effectively maintain the fidelity of multiple concepts and address the identity decoupling problem, even for highly similar concepts. Unlike previous approaches to integrate visual embeddings with textual counterparts [26, 34, 36], or directly aggregate visual embeddings as a whole into diffusion models [18, 61, 67], our key insight is to learn the decoupled multi-concept embeddings and inject into diffusion transformer models in a standalone manner. Specifically, the process includes: 1) Extracting comprehensive visual embeddings from given reference images, where we initially extract dense visual tokens by the CLIP image encoder [45], and integrate a learnable query transformer (Q-Former) network [35] to better represent comprehensive visual embeddings and align with the diffusion model space. 2) Incorporating visual representation with corresponding text description of every concept, where we propose the Decouple Attention Module (DAM) to conduct intra-pair attention to separately bind the extracted visual embedding with corresponding textual embedding for each concept, the process could effectively capture semantic differences across multiple concepts while maintain conceptspecific uniqueness. 3) Introducing a novel multi-concept embeddings injection strategy, where we firstly composite the multi-concept embeddings and then inject them using an individual Multi-Concept Injector (MC-Injector), which

is a standalone cross-attention layer, into the diffusion transformer models without affecting the original textual crossattention. This strategy separates the functionality of the original textual cross-attention from the learning process of the newly injected composite multi-concept embeddings, effectively enhancing the representation of multiple identities. The designed ConceptMaster could efficiently create high-fidelity customized videos during inference without additional parameter tuning, which significantly provides the potential for the practicality of real-world applications.

Furthermore, to address the scarcity of suitable MCVC data, we carefully establish a data collection pipeline, which could collect high-quality MCVC data that precisely extract entity images and corresponding text descriptions of diverse concepts in videos. By utilizing this pipeline, we collect over 1.3 million video-entity pairs spanning diverse conceptual domains, including humans, livings, and various object categories. To further facilitate the evaluation, we introduce a multi-concept evaluation set to comprehensively validate this task from 1) concept fidelity, 2) effectiveness of identity decoupling, and 3) video generation quality across six distinct multi-concept composition scenes. Overall, our key contributions can be summarized:

- • We propose ConceptMaster, a novel multi-concept video customization framework to personalize video generation based on user-defined concepts. It effectively addresses the identity decoupling problem while ensuring every concept fidelity, even for highly similar concepts.
- • We present a novel strategy of learning decoupled multiconcept embeddings and injecting them into diffusion models without influencing the original attention operations, which effectively guarantees the fidelity of various concepts in customized videos.
- • We introduce a dedicated data construction pipeline that enables the collection of high-quality multi-concept video-entity pairs across diverse concepts, which effectively address the scarcity of high-quality MCVC data.
- • We collect a multi-concept evaluation set that could comprehensively validate video customization performance from six distinct concept composition scenarios and various dimensions including concept fidelity, identity decoupling and video quality. Extensive experiments demonstrate the superiority of ConceptMaster in video customization.

#### 2. Related Work

##### 2.1. Foundation Text-to-Video Diffusion Models

The rapid development of text-to-video (T2V) models has been phenomenal. Early works in T2V diffusion models such as AnimateDiff [15], VideoCrafter [6] and ModelScope [58] are mainly based on latent diffusion models [48] with UNet backbones [49]. By using transform-

ers [55] as the backbone of diffusion models, such as Diffusion Transformers (DiT) [44], SORA [3], and other transformer-based variants [21, 32, 70], advanced T2V models have scaled parameters and demonstrated impressive capabilities in generating realistic, long-range, and physically consistent videos. This advancement significantly expands the possibilities for content generation.

##### 2.2. Image-based Concept Customization

Customization in diffusion models enables users to provide reference images to generate results retain the given identities. These customization methods are primarily categorized into tuning-based and pretrain-based approaches. Early represented tuning-based methods [13, 50] are designed to online-optimize word embeddings or weights of diffusion models when new reference images are provided by users, which are constrained by consuming time and manually collecting training samples. Pretrain-based methods [8, 14, 34, 36, 60, 62, 65, 67] usually train an encoder on certain concept datasets to learn visual representation for conditional diffusion generation process. Some works primarily focus on general-domain concept customization [8, 14, 34, 62, 67], while others mainly aim at human face identity scenarios [36, 60, 65]. While aforementioned methods mainly customized single provided concept, the problem also extends to process multiple references. For example, CustomDiffusion [31] optimizes additional multiple key-value pairs in cross-attention. SSR-Encoder [69] aligns query inputs with image patches and preserves fine features of the subjects. MS-Diffusion [61] pretrains a grounding resampler and generates images with bounding box layout guidance. These approaches remarkably promote the development of image customization.

##### 2.3. Video-based Concept Customization

Pretrained-based multi-concept customized video generation raises little attention. Preliminary methods [18, 24, 26, 63] predominantly focus on single-concept scenarios. DreamVideo [63] employs a tuning-based approach to simultaneously customize identities and motion. Videobooth [26] simply utilizes Grounded-SAM [30, 40, 47] to extract foreground information and tags from the first frame of each video from WebVid dataset [1] including nine categories as training data, and further trains a coarse-to-fine visual embedding on that data. In contrast, ID-Animator [18] leverages the CelebV dataset [71] to construct a face identity dataset, and integrates pre-trained IP-Adapter [67] with AnimateDiff [15] for joint optimization. However, neither the data collection methods nor the proposed models targeting for single-concept customization are sufficient to directly transfer into multi-concept scenarios. ConceptMaster, on the other hand, could solve the challenging MCVC task well in a feed-forward manner, we believe that Con-

ceptMaster has substantially promoted the development of video customization and paved the way for its future.

#### 3. Preliminary: Diffusion Transformer Models for Text-to-Video Generation

Transformer-based text-to-video diffusion models demonstrate huge potential on video content generation. Our ConceptMaster is built upon a transformer-based latent diffusion model, which employs a 3D Variational Autoencoder (VAE) [28] to transform videos from the pixel level to a latent space. Each basic transformer block consists of 2D spatial self-attention, 3D spatial-temporal self-attention, text cross-attention, and feed-forward network (FFN). The text prompt embedding ctext for cross-attention is obtained by T5 encoder ET5 [46]. We use Rectified Flow [12, 41] to define a probability flow ordinary differential equation (ODE), which transfers the clean data z0 to a noised data zt with straight path zt = (1 − t)z0 + tϵ at timestep t, where ϵ is a normal gaussian noise. The diffusion transformer output directly parameterizes the vΘ(zt,t,ctext) to regress velocity (z1 − z0) with the Flow Matching objective [37]:

0,ϵ||vΘ(zt,t,ctext) − (z1 − z0)||22. (1)

LLCM = Et,z

#### 4. ConceptMaster

##### 4.1. Multi-Concept Video Customization

Problem: Given a caption T describing a video, along with a set of concept images {Xi|i = 1...N} and their corresponding labels {Yi|i = 1...N} (e.g., a man and a woman with their respective images), where N represents the number of distinct concepts, the task of MultiConcept Video Customization (MCVC) aims to generate high-quality videos that incorporate all image-defined visual concepts while aligning them with the given descriptive caption T. Each concept should maintain its identity as the provided images while precisely expressing its semantic behavior as described in the caption. We define the paire images and label for each concept as the intra-pair customized concept for convenience of expression.

Overview: To achieve this goal, we first meticulously design a data collection pipeline, resulting in the creation of a dataset comprising over 1.3 million high-quality MCVC samples. These training videos provide precise information about each entity’s image and corresponding text description. Additionally, we incorporate several existing singleconcept image and video datasets to further enhance the concept representation. Afterwards, in order to generate videos that could effectively maintain the fidelity of each concept and decouple multiple visual representation, we firstly extract thoughtful visual embeddings of given reference images, and then design the Decouple Attention Module (DAM) to perform intra-pair attention across paired

image-label features, achieving multi-modal representation for each identity. Subsequently, we combine every multimodal concept embedding into the composite ones, and further introduce a Multi-Concept Injector (MC-Injector) in a cross-attention manner to embed the multi-modal composite representation into the diffusion transformer models, where the composite features serve as keys and values. In Fig. 3, we demonstrate the overview framework of our proposed ConceptMaster.

##### 4.2. Decoupling and Injecting Concept Embeddings

Visual Concept Representation Extraction. To enable the model to process multiple concepts with high fidelity, we need to obtain reasonable visual representation from the concept images {Xi|i = 1...N}. We opt to use the CLIP image encoder Eimg [45] to extract the last layer output as dense visual tokens with shapes 16 × 16 × 768, i.e., {fi|fi = Eimg(Xi),i = 1...N}. These tokens have demonstrated more complete visual representation of image conditions [51, 62, 67]. However, directly applying these dense visual tokens in diffusion generation often achieves inadequate alignment with representation space of diffusion models, results in unsatisfactory visual fidelity. To prevent such trivial visual conditions injection and achieve better alignment with the diffusion transformer context, we integrate a learnable Q-Former architecture Q, which comprises stacked cross-attention layers and FFN [34, 35, 66]. We utilize the dense visual tokens as a key-value corpus and employ the Q-Former to query these tokens {xi|xi = Q(fi),i = 1...N}, thereby extracting comprehensive visual semantic representation.

Decoupling Intra-Pair Embeddings. After obtaining the appropriate visual representation, we integrate the corresponding text labels to create visual-text aligned concept representation. While previous works [36, 65] directly combine the visual representation with the corresponding word from the caption embedding ctext = Etext(T), we hope to fully leverage the textual label information associated with related image to enhance the representation specific to each concept. Therefore, unlike these approaches, we employ T5-encoder ET5 to encode each concept label individually to obtain the text representation {yi|yi = ET5(Yi),i = 1...N}. Subsequently, we introduce the Decouple Attention Module (DAM) to fuse each pair of the visual and text label embedding {(xi,yi)|i = 1...N}. The DAM operation can be formulated as:



Qi = WQ · xi;Ki = WK · yi;Vi = WV · yi, Attention(Qi,Ki,Vi) = Softmax(

QiKiT



√

) · Vi,

(2)

d

bi = Qi + Attention(Qi,Ki,Vi) ci = bi + FFN(bi)



where WQ, WK, and WV are projection matrices, d is

[Figure 62]

[Figure 63]

Decouple Attention Module Feature Extraction

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Entity textual embeddings

Entity visual embeddings

|[Figure 69]|
|---|

[Figure 70]

[Figure 71]

[Figure 72]

- Entity-1: a woman
- Entity-2: a man

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

###### KV Q

Cross-Attention

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Text Encoder Visual Encoder

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

Attention

FFN

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

Prompt: A woman with brown hair and a leopard print shirt is talking to a man with glasses and a blue shirt.

× 𝑵 Linear Projection

|[Figure 106]|
|---|

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

FFN

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Caption embeddings

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

Composite embeddings

[Figure 128]

[Figure 129]

Self-Attention3D

Self-Attention2D

Cross-Attention

[Figure 130]

MC-Injector

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

3D VAE Encoder +noise 𝑧𝑡 𝑣

FFN

Trainable Frozen

× 𝑵

- Figure 3. Overview of ConceptMaster framework. Given a caption along with a set of concept images and their semantic labels, we firstly extract comprehensive visual concept representations with the CLIP image encoder and a learnable Q-Former, then bind the visual representations with corresponding text embeddings of each concept through the Decouple Attention Module (DAM). Finally, the multiconcept visual-text embeddings are injected into the diffusion transformer models with the Multi-Concept Injector (MC-Injector).

the embedding dimension, and FFN is a two-layer multilayer perceptron (MLP) with GLUE [57] as the middle activation function. The residual connection [17] is existed in both attention and MLP layers. With the designed DAM, every visual representation could integrate its corresponding textual label to serve as the visual-text aligned representation for the diffusion transformer models.

Composite Multi-Concept Representation Injection. After obtaining the multi-modal representation of each pair {ci|i = 1...N}, we firstly concatenate all concept embeddings into a composite one, where D is the dimension of concept embedding:

c∗IDs = Concat(c1,...,cN), c∗IDs ∈ RN×D (3)

Additionally, we design a Multi-Concept Injector (MCInjector) to encode the composite multi-concept embeddings into the diffusion transformer models. Specifically, the MC-Injector is an additional specialized cross-attention layer integrated within each transformer block, positioned after the original text cross-attention layer. The additional standalone cross-attention layer can effectively learn the concepts without interference of the original text crossattention. Comparing with merging the composite embeddings into the original text cross-attention layer, our experiments in Sec. 5.4 indicate that by interleaving the MCInjector with the original one could achieve both better decoupling ability and visual fidelity on generated videos. Finally, the specific diffusion process assisted by the composite embeddings c∗IDs can be formulated as:

0,ϵ||vΘ(zt,t,ctext,c∗IDs)−(z1 −z0)||22. (4)

LLCM = Et,z

##### 4.3. MC-Oriented Video Data Construction

Training a good MCVC model requires high-quality MCoriented video data. Previous studies [26, 43, 61] heav-

ily relied on the state-of-the-art open-set object detection methods, such as Grounding-DINO [40], to obtain bounding boxes for each concept based on its text label. They then employ the segmentation model SAM [30] to extract masks using the bounding boxes as input. However, the simplistic method is far insufficient for our objectives, as Grounding-DINO, equipped with the CLIP text encoder, often perform poorly in distinguishing similar concepts, especially those that have high visual appearance or textual semantic similarity. Additionally, incorporating lowquality videos or data not suitable for customization task in training can adversely affect the quality of generated videos. Consequently, in order to collect high-quality and large-scale MCVC data, we carefully design the data collection pipeline into two levels: 1) Fast elimination of unsuitable videos, we filtering out low-quality videos that are not unsuitable for the task with time efficiency and low resources. 2) Fine-grained identity information extraction, we guarantee the accuracy of extracted identity reference images and corresponding text labels. We finally collect more than 1.3 million MCVC data for our ConceptMaster. Fig. 4(a) demonstrates the overview of our dataset collection pipeline. Additionally, we randomly sample 2000 samples from Panda-2M [7], and we count the success rate of collect videos. Fig. 4(b) demonstrates our designed data pipeline is significantly better than simply using GroundedSAM. More discussions could be found in appendix.

###### Fast elimination of unsuitable videos.

• Scene transition detection and low-quality video elimination. We initially collect more than 6.4 million videos from Internet as sources. To ensure the basic attributes of our video data are maintained at a high standard, we initially use PySceneDetect [5] to filter out videos that contain scene transitions to maintain the temporal coher-

|52%<br><br>90%<br><br>0%<br><br>10%<br><br>20%<br><br>30%<br><br>40%<br><br>50%<br><br>60%<br><br>70%<br><br>80%<br><br>90%<br><br>100%<br><br>Grounded-SAM Ours<br><br>Success Rate|
|---|

MCVC data labels

Fast elimination of unsuitable videos

|[Figure 139]<br><br>(“person”, “a man”)<br><br>[Figure 140]<br><br>(“person”, “another man”)|
|---|

|Scene transition detection and low-<br><br>quality video elimination<br><br>Scene Transition<br><br>[Figure 141]<br><br>Low Dynamics<br><br>[Figure 142]<br><br>Low Light Contrast<br><br>[Figure 143]<br><br>1|
|---|

Grounded-SAM

[Figure 144]

|[Figure 145]<br><br>“a man” “another “a man” man”<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>“another man”<br><br>[Figure 149]<br><br>“a man”|
|---|

|Video captioning and concept labels<br><br>extraction<br><br>Captioning<br><br>Noun Extraction<br><br>[Figure 150]<br><br>[Figure 151]<br><br>A man is crying and another man is giving comfort. (“person”, “a man”), (“person”, “another man”)<br><br>2|
|---|

|[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>Video Pools|
|---|

[Figure 156]

[Figure 157]

Fine-grained identity information extraction

|Segmentation<br><br>Wrong Classification<br><br>Human<br><br>Without Face Too Small/Large<br><br>[Figure 158]<br><br>High Fragmentation<br><br>[Figure 159]<br><br>[Figure 160]<br><br>4<br><br>Accurate visual and textual labels extraction<br><br>[Figure 161]<br><br>[Figure 162]|
|---|

|Elimination of unsuitable videos<br><br>Overlap<br><br>[Figure 163]<br><br>Too Small/Large<br><br>[Figure 164]<br><br>Wrong Classification<br><br>[Figure 165]<br><br>3|
|---|

(a) (b)

- Figure 4. (a) The overview of multi-concept data collection pipeline. When dealing with complex scenarios that contain concepts with high visual appearance or textual semantic similarity, our data pipeline could still extract precise entity images and corresponding labels, while simply exploit previous methods like Grounded-SAM would introduce a large number of errors and it is difficult to remove these errors through subsequent processing. (b) The success rate of testing videos comparison between Grounded-SAM and our data pipeline.

ence in videos. We also remove videos with low optical flow scores [54] to guarantee the dynamic integrity. Additionally, videos with low light contrast are excluded.

ing capabilities, to extract entity masks. LISA provides highly accurate segmentation results, even for similar visual appearance and textual semantics. Those masks are either too large or too small, or with a high degree of fragmentation are removed. We then derive box regions from these masks and remove any misclassified ones through CLIP classification. Additionally, we use FaceAnalysis1 to detect all regions belonging to the person class, retaining only those that contain face regions (i.e., removing humans where only the body parts are visible).

- • Video captioning and concept labels extraction. We employ Qwen2-VL [59] to produce accurate and concise captions for videos. To extract potential concept entity textual description from the caption, we define a taxonomy of 120 classes, with each class encompassing several sub-words (for instance, the class dog includes sub-words such as dog, puppy and beagle). we utilize SpaCy [22] to extract nouns from the captions, ensuring that these nouns fall within the predefined set of sub-words. The extracted nouns serve as the textual input for text-guided detection and segmentation algorithms.
- • Elimination of unsuitable videos for MCVC task. Since most videos are unsuitable for video customization, we hope to quickly exclude those clearly cannot meet our requirements with minimal resource expenditure and time consuming. For each video, we uniformly sample 10% of the frames and use the extracted nouns to identify entity boxes through text-guided Grounding-DINO. Simultaneously, we apply Non-Maximum Suppression (NMS) to filter out duplicate boxes and remove boxes that are either too large or too small (e.g., areas smaller than 10% or larger than 90% of the video frame size). Subsequently, we classify each box using CLIP, eliminating any box if the label classified by CLIP is inconsistent with the original one. If all the boxes are eliminated through this process, the corresponding video will be excluded.

##### 4.4. Joint Training with Auxiliary Datasets

In addition to the MCVC data we have constructed, we also utilize auxiliary datasets to enhance concept representation. We reproduce the single-concept image dataset from BLIPDiffusion [34] (around 300k) for high-specificity concept enhancement. Furthermore, we incorporate the singleconcept video dataset CelebV [71] (about 60k) to improve human representation. The data sampling ratio of our built data, BLIP-Diffusion and CelebV is 8:1:1.

#### 5. Experiments 5.1. Experimental Setup

Implementation Details. The implementation details of ConceptMaster can be found in the supplementary material. Evaluation Metrics. In order to comprehensively evaluate MCVC methods, we consider three different dimensions: 1) Concept fidelity, where we exploit the commonly used CLIP-cap [45] to globally evaluate if the generated videos match the semantics of given video captions. 2) Decoupling ability, where we utilize LISA [33] to segment the mask area of each concept in generated videos, and then compute CLIP-I and DINO-I [4] scores between the original concept images and the mask areas of each concept in generated videos. Additionally, we compute the semantic

###### Fine-grained identity information extraction.

• Accurate visual and textual labels extraction. To accurately extract the region and text label of each identity, we employ the same frame sampling strategy and use LISA [33], an MLLM-based [39] segmentor, input by both text prompts and images with strong visual reason-

1https://github.com/deepinsight/insightface

|[Figure 166]|[Figure 167]|
|---|---|

|[Figure 168]|[Figure 169]|
|---|---|

“An elephant is playing with a dog on the road.”

“A dog is drinking water from a bowl on the road.”

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

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

(a) CustomDiffusion (b) SSR-Encoder (c) IP-Adapter (d) MS-Diffusion (e) ConceptMaster

(a) CustomDiffusion (b) SSR-Encoder (c) IP-Adapter (d) MS-Diffusion (e) ConceptMaster

- Figure 5. Qualitative comparison on multi-concept customization. When compared to several different methods to conduct the MCVC task, our approach clearly demonstrates superior capabilities on concept fidelity, identity decoupling and caption semantic consistency.

|Methods<br><br>|Concept Fidelity and Decoupling Ability|Video Quality|
|---|---|---|
| |CLIP-cap↑ CLIP-tag↑ CLIP-I↑ DINO-I↑ CLIP-tagdis↓<br><br>|Motion Smoothness↑ Dynamic Degree↑ Aesthetic Quality↑ Imaging Quality↑|
|CustomDiffusion SSR-Encoder IP-Adapter MS-Diffusion|26.096 21.101 0.737 0.467 16.321<br><br>23.766 21.153 0.739 0.466 16.651<br><br>26.097 21.980 0.749 0.492 16.425<br><br><br>28.031 21.901 0.736 0.487 16.500<br><br>|0.962 15.283 0.560 0.672 0.946 21.804 0.535 0.657 0.969 17.800 0.496 0.693 0.978 15.849 0.571 0.624<br><br>|
|ConceptMaster (Ours)<br><br>|28.246 22.165 0.781 0.584 16.169|0.967 26.115 0.572 0.657|

- Table 1. Quantitative comparison with different methods on the introduced multi-concept evaluation set, where bold represents the best result and underline represents the second best result. matching scores between each concept label and its corre-

are provided in supplementary material.

sponding mask area. We refer CLIP-tag and CLIP-tagdis as the similarity and dissimilarity between label and its corresponding area. These four metrics are used to validate the model capacity for decoupling various concepts. 3) Video generation quality, where we adopt the motion smoothness, dynamic degree, aesthetic quality and imaging quality collected as suggested by VBench [25] to evaluate the generation quality.

##### 5.3. Comparing with other methods

We compare several open-sourced multi-concept image customization methods [31, 61, 67, 69], combining with the image-to-video (I2V) generation model I2VGen-XL [68], as a naive solution for the MCVC task with our ConceptMaster. According to the qualitative results in Fig. 5, we can see that our ConceptMaster has clear advantages on customizing multiple concepts in videos. The naive solution will be subject to the decoupling and representation ability of both two models, the instruction-following capability of I2V models could further influence the quality of generated videos. In contrast, the end-to-end video customization models could achieve better results. The quantitative results in Tab. 1 also demonstrates that our method could not only maintain the representation of multiple concepts, but also generate high-quality text-aligned videos.

##### 5.2. Multi-Concept Evaluation Set

In order to comprehensively evaluate the performance of MCVC methods, we establish a multi-concept evaluation set, including diverse concept composition scenarios of 1) multiple persons, 2) persons with livings, 3) persons with stuffs, 4) multiple livings, 5) livings with stuffs and 6) persons with both livings and stuffs. The sample number for each scenario is 40, 40, 40, 30, 30, and 30 respectively (Total 210). It should be noticed that we manually collect reference images and provide suitable captions for these scenarios, since we hope to eliminate information leakage for evaluation when extracting concepts from videos via the same MCVC data collection pipeline mentioned in Section 4.3 (e.g., the caption is A man and a woman are talking to each other, and the extracted man and woman are doing the same thing). All the quantitative experiments are evaluated by this introduced multi-concept evaluation set. More details

##### 5.4. Multi-Concept Embeddings Injection Manner

In section 4.2, our ConceptMaster introduces a standalone MC-Injector to integrate the multi-concept visualtext aligned representation into the diffusion models. Some previous methods, represented as BLIP-Diffusion [34] and IP-Adapter [67], the former combines visual embeddings with textual caption embeddings as the whole condition representation, while the latter encodes the whole image as vi-

|Methods<br><br>|Concept Fidelity and Decoupling Ability|Video Quality|
|---|---|---|
| |CLIP-cap↑ CLIP-tag↑ CLIP-I↑ DINO-I↑ CLIP-tagdis↓<br><br>|Motion Smoothness↑ Dynamic Degree↑ Aesthetic Quality↑ Imaging Quality↑|
|Merge Textual and Visual Embeddings IP-Adapter-like ConceptMaster (Ours)|27.221 21.885 0.760 0.548 16.186<br><br>28.195 22.051 0.736 0.473 16.330<br><br><br>28.246 22.165 0.781 0.584 16.169<br><br>|0.969 17.025 0.538 0.675 0.969 21.957 0.530 0.652 0.967 26.115 0.572 0.657<br><br>|

- Table 2. Quantitative comparison of different multi-concept embeddings injection manner, bold represents the best result and underline represents the second best result. All the methods we compared have been trained on the same data as that used by ConceptMaster.

|Methods<br><br>|Concept Fidelity and Decoupling Ability<br><br>|Video Quality|
|---|---|---|
| |CLIP-cap↑ CLIP-tag↑ CLIP-I↑ DINO-I↑ CLIP-tagdis↓<br><br>|Motion Smoothness↑ Dynamic Degree↑ Aesthetic Quality↑ Imaging Quality↑|
|Without Q-Former Without DAM Concate-MLP Self-Attn DAM (Ours)<br><br>|29.312 21.876 0.726 0.416 16.379 28.916 21.981 0.730 0.439 16.365<br><br>27.746 22.063 0.775 0.577 16.251<br><br>28.146 22.045 0.778 0.576 16.211 28.246 22.165 0.781 0.584 16.169<br><br><br>|0.963 24.984 0.518 0.666 0.968 25.252 0.531 0.649<br><br>0.966 23.770 0.551 0.648 0.968 22.864 0.550 0.655<br><br>0.967 26.115 0.572 0.657<br><br><br>|

- Table 3. Quantitative comparison of the design choice of Q-Former and DAM modules, bold represents the best result and underline represents the second best result. All the methods we compared have been trained on the same data as that used by ConceptMaster.

|[Figure 215]|[Figure 216]|
|---|---|

“A man and a girl smiling and shaking hands in the office.”

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

(a) Merge Textual and Visual Embeddings

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

(b) IP-Adapter-like

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

(b) ConceptMaster

- Figure 6. Different injection methods of multi-concept references.

sual embeddings and aggregates into models by a decoupled cross-attention layer. However, in multi-concept scenarios, merging multi-modal features as the whole conditions can make it challenging to distinguish the semantic meanings among different identities. In addition, integrating all visual concepts on one image is not the optimal choice for multi-concept representation, especially when they contain similar visual appearances. We conduct the above two integration approaches of the multi-concept embeddings on our text-to-video generation models with the same training data for ConceptMaster. In Tab. 2 and Fig. 6, we can see that when customizing multiple concepts, both these two methods can hardly maintain the concept fidelity and deal with the identity decoupling problem. Additionally, when merging the textual and visual embeddings, the dynamic degree is significantly reduced, as the original text crossattention layer is influenced by the additional visual embeddings. Therefore, our designed ConceptMaster adopts the optimal solution to inject the decoupled multi-concept embeddings into the diffusion models.

##### 5.5. Ablation Study

In section 4.2, our ConceptMaster proposes to firstly utilizes a Q-Former network to integrate the dense visual tokens extracted by CLIP image encoder into the comprehensive visual embeddings. While after simply replacing the Q-Former by an MLP layer, the generated videos cannot

|[Figure 229]|[Figure 230]|
|---|---|

“A man and a woman walking down a city street.”

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

(a) Without Q-Former

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

(b) Without DAM

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

(c) Concate-MLP

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

(d) Self-Attn

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

(e) DAM

Figure 7. Demonstration of the effectiveness of the Q-Former and DAM modules.

capture the appearances of given images, as in Fig. 7, and the quantitative metrics also largely drop in Tab. 3.

In addition, in order to demonstrate the effectiveness of the designed DAM module, which is the intra-pair attention module based on paired visual embeddings and textual descriptions representation, we conduct several variants include: 1) Replacing the Q-Former, where we only use an MLP layer instead. 2) Removing the DAM module, where only the extracted visual embeddings are further injected into the diffusion models, and the textual descriptions are unused. 3) Replacing the intra-pair cross-attention by firstly concatenating the visual and textual embeddings along channel dimension (double the channel dimension), and integrating to the original channel dimension by an MLP layer. 4) Conducting the intra-pair self-attention instead of cross-attention, the features conduct self-attention are obtained by directly adding the visual and textual embeddings. In Tab. 3 and Fig. 7, we can see that the proposed DAM module is the optimal design. Cooperating the textual descriptions is significant, which not only en-

hances the uniqueness of each concept representation, but also assists the alignment of the multi-concept embeddings and the original diffusion model space. Furthermore, fusing visual and textual embeddings by the MLP layer is less comprehensive as it does not involve sufficient token-level interaction between them, lowering the concept representation and instruction-following in generated results. Additionally, the cross-attention operation is better than selfattention, which maintains consistent visual appearances in videos, while more artifacts would be created by the selfattention operation.

- 6. Conclusion In this paper, we introduce ConceptMaster, an innovative framework that effectively addresses the critical issues of identity decoupling while maintaining concept fidelity when customizing multiple identities in videos. ConceptMaster introduces a novel strategy for learning decoupled multi-concept embeddings and injecting them into diffusion models in a standalone manner. This strategy ensures the quality of customized videos with multiple identities, even for highly similar visual concepts. To further address the scarcity of high-quality multi-concept video-entity data, we have established a meticulous data construction pipeline. This pipeline enables the systematic collection of precise multi-concept video-entity data across diverse concepts. Additionally, we have designed a comprehensive testing set to validate the effectiveness of our model from three critical dimensions across six different concept composition scenarios. Extensive experiments demonstrate that ConceptMaster significantly outperforms previous approaches, paving the way for generating personalized and semantically accurate videos across multiple concepts. References

- [1] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1728–1738,

2021. 3

- [2] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 2
- [3] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators.

2024. 2, 3

- [4] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Pro-

- ceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 6
- [5] Brandon Castellano. PySceneDetect. 5
- [6] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7310– 7320, 2024. 2, 3
- [7] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13320–13331, 2024. 5, 2
- [8] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6593–6602, 2024. 2, 3
- [9] Jooyoung Choi, Yunjey Choi, Yunji Kim, Junho Kim, and Sungroh Yoon. Custom-edit: Text-guided image editing with customized diffusion models. arXiv preprint arXiv:2305.15779, 2023. 2
- [10] Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36, 2024. 1
- [11] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. arXiv preprint arXiv:2302.03011, 2023. 2
- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 4
- [13] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 2, 3
- [14] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Encoder-based domain tuning for fast personalization of text-to-image models. ACM Transactions on Graphics (TOG), 42(4):1–13, 2023. 2, 3
- [15] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2, 3
- [16] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact pa-

- rameter space for diffusion fine-tuning. arXiv preprint arXiv:2303.11305, 2023. 2
- [17] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 5
- [18] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, Man Zhou, and Jie Zhang. Id-animator: Zero-shot identity-preserving human video generation. arXiv preprint arXiv:2404.15275, 2024. 2, 3
- [19] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221, 2022. 2
- [20] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 1
- [21] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 3
- [22] Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. spaCy: Industrial-strength Natural Language Processing in Python. 2020. 6
- [23] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 2
- [24] Yuzhou Huang, Yiran Qin, Shunlin Lu, Xintao Wang, Rui Huang, Ying Shan, and Ruimao Zhang. Story3d-agent: Exploring 3d storytelling visualization with large language models. arXiv preprint arXiv:2408.11801, 2024. 3
- [25] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 7
- [26] Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si, Dahua Lin, Yu Qiao, Chen Change Loy, and Ziwei Liu. Videobooth: Diffusion-based video generation with image prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6689– 6700, 2024. 2, 3, 5
- [27] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Texttoimage diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023. 2
- [28] Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 4
- [29] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 1
- [30] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 3, 5,

1

- [31] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023. 2, 3, 7
- [32] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan, 2024. 3
- [33] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9579–9589, 2024. 6
- [34] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pretrained subject representation for controllable text-to-image generation and editing. Advances in Neural Information Processing Systems, 36, 2024. 2, 3, 4, 6, 7
- [35] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–

19742. PMLR, 2023. 2, 4

- [36] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, MingMing Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8640–8650, 2024. 2, 3, 4
- [37] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 4
- [38] Gongye Liu, Menghan Xia, Yong Zhang, Haoxin Chen, Jinbo Xing, Yibo Wang, Xintao Wang, Yujiu Yang, and Ying Shan. Stylecrafter: Enhancing stylized text-to-video generation with style adapter. arXiv preprint arXiv:2312.00330,

2023. 2

- [39] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485,

2023. 6

- [40] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 3, 5, 1
- [41] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 4
- [42] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10209–10218, 2023. 2
- [43] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subjectdiffusion: Open domain personalized text-to-image generation without test-time fine-tuning. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 5
- [44] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 3

- [45] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2, 4, 6, 1
- [46] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 4
- [47] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, Zhaoyang Zeng, Hao Zhang, Feng Li, Jie Yang, Hongyang Li, Qing Jiang, and Lei Zhang. Grounded sam: Assembling open-world models for diverse visual tasks,

2024. 3, 1

- [48] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3
- [49] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015. 3
- [50] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 2, 3
- [51] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without test-time finetuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8543–8552, 2024. 4
- [52] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 2

- [53] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 1
- [54] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part II 16, pages 402–419. Springer,

2020. 6

- [55] A Vaswani. Attention is all you need. Advances in Neural Information Processing Systems, 2017. 3
- [56] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan.

- Phenaki: Variable length video generation from open domain textual description. arXiv preprint arXiv:2210.02399, 2022. 2
- [57] Alex Wang. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018. 5
- [58] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 2, 3
- [59] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 6
- [60] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024. 2, 3
- [61] X Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multi-subject zero-shot image personalization with layout guidance. arXiv preprint arXiv:2406.07209, 2024. 2, 3, 5, 7
- [62] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15943–15953, 2023. 2, 3, 4
- [63] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6537–6549, 2024. 3
- [64] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. arXiv preprint arXiv:2212.11565, 2022. 2
- [65] Guangxuan Xiao, Tianwei Yin, William T Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multisubject image generation with localized attention. International Journal of Computer Vision, pages 1–20, 2024. 3, 4
- [66] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision, pages 399–417. Springer, 2025. 4
- [67] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2, 3, 4, 7

- [68] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video

- synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023. 2, 7
- [69] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8069–8078, 2024. 3, 7, 2
- [70] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024. 3
- [71] Hao Zhu, Wayne Wu, Wentao Zhu, Liming Jiang, Siwei Tang, Li Zhang, Ziwei Liu, and Chen Change Loy. Celebvhq: A large-scale video facial attributes dataset. In European conference on computer vision, pages 650–667. Springer,

2022. 3, 6

## ConceptMaster: Multi-Concept Video Customization on Diffusion Transformer Models Without Test-Time Tuning

### Supplementary Material

We provide the following contents in supplementary materials:

- 1. Introduction of our text-to-video diffusion transformer models.
- 2. Implementation Details of ConceptMaster.
- 3. Discussions on Comparison between Our Data Collection Pipeline and Grounded-SAM.
- 4. More details of Multi-Concept Evaluation Set.
- 5. Comparison Methods Implementation.
- 6. More Discussions on Multi-Concept Embeddings Injection.
- 7. More Discussions on Ablation Study.
- 8. More Qualitative Results Demonstration.

#### 1. Introduction of our text-to-video diffusion transformer models

We utilize a transformer-based latent diffusion model as the foundational text-to-video (T2V) generation model, as depicted in Fig. 8. Initially, we employ a 3D Variational Autoencoder (3D-VAE) to transform videos from the pixel space into a latent space, upon which we build a transformer-based video diffusion model. Unlike previous models that rely on UNets or transformers with an additional 1D temporal attention module for video generation, our approach addresses the limitations of spatiallytemporally separated designs, which often do not yield optimal results. We replace the 1D temporal attention with 3D self-attention, allowing the model to more effectively perceive and process spatiotemporal tokens. This results in a high-quality and physically coherent video generation model. Specifically, before each attention or feed-forward network (FFN) module, we map the timestep to a scale and apply RMSNorm to the spatiotemporal tokens.

#### 2. Implementation Details of ConceptMaster

Implementation Details. We train ConceptMaster using our proprietary transformer-based text-to-video diffusion models. Initially, we employ the CLIP image encoder [45] as the external vision encoder to extract visual features from reference images. Subsequently it is followed by the stacked cross-attention and FFN layers, collectively referred to as the Q-Former, and an additional cross-attention layer for the DAM module. During training, we drop the video captions and reference conditions (both paired images and text descriptions) with probabilities of 50% and 33% for classifier-free guidance [20], respectively. We freeze the 3D

[Figure 251]

[Figure 252]

[Figure 253]

Skip Addition Trainable Frozen

Transformer Block

3D VAE

| |RMSNorm&Scale<br><br>Self-Attention3D|
|---|---|
| | |

| |RMSNorm&Scale<br><br>Feed-ForwardNN<br><br>|
|---|---|
| | |

| |RMSNorm&Scale<br><br>Cross-Attention<br><br>|
|---|---|
| | |

Encoder

Self-Attention2D

Self-Attention3D

Feed-ForwardNN

Cross-Attention

RMSNorm&Scale

RMSNorm&Scale

RMSNorm&Scale

RMSNorm&Scale

[Figure 254]

[Figure 255]

[Figure 256]

𝑧𝑡

𝑧0 Noise

3D VAE Decoder

Caption: A woman with brown hair and a leopard print shirt is talking to a man with glasses and a

T5 Encoder

[Figure 257]

[Figure 258]

[Figure 259]

TimeStep

blue shirt.

Figure 8. Overview framwork of our base text-to-video generation models.

spatiotemporal self-attention layer and fine-tune the other parameters of the transformer to enhance video dynamics. Consequently, the entire transformer backbone, except for the 3D spatiotemporal layer, the Q-Former, and the DAM module, are jointly optimized. Additionally, inspired by NaViT [10], we adopt the similar strategy of padding the videos to the same height and width with effective attention masks within each batch during training, and the training video segments consist of 77 frames, corresponding to a duration of 5 seconds at 15 frames per second (fps). The training process employs the Adam optimizer [29] and is conducted on 64 NVIDIA H800 GPUs, with a learning rate set to 5 × 10−6 and a global batch size of 256. During inference, we utilize 100 DDIM steps [53] and set the CFG scale to 7.5, and the inference videos are uniformly resized to a resolution of 384 × 672 pixels.

#### 3. Discussions on Comparison between Our Data Collection Pipeline and GroundedSAM

Previous studies typically exploit open-set object detection and segmentation methods, represented by GroundedSAM [30, 40, 47], to extract concepts information in source images or videos. However, we claim that the simplistic method is far insufficient for our objectives, since Grounding-DINO is built on top of the CLIP text encoder, which often perform poorly in distinguishing similar concepts, especially those that have high visual appearance or textual semantic similarity (see Fig 4(a)). Additionally, we need to guarantee the quality in a high standard of the source data as well as the extracted concept information (e.g., appropriate size and completeness) to train the model for our task. We carefully design the data collection pipeline

into two levels including: 1) Fast elimination of unsuitable videos and 2) Fine-grained identity information extraction. In order to evaluate the effectiveness of our dataset pipeline, we first randomly sample 2000 samples from Panda-2M [7], and we count the success rate of collect videos. The standard for success rate statistics is based on whether there are errors in the extracted concepts information from a video (i.e., wrong classification label for the concepts). A success is defined as the absence of any errors, while the presence of any error is considered a failure. We additionally hire 20 experienced workers in data construction to manually evaluate the results of the constructed evaluation samples from Panda-2M. We present the success rate statistics in Fig. 4(b). The results indicate that the success rate of our designed data pipeline is significantly higher than that achieved by simply using Grounded-SAM. Grounded-SAM exhibits a success rate of only 52%, which poses challenges in training a robust model capable of representing and decoupling multiple concepts in generated videos, especially when dealing with MCVC data that contains numerous errors. Consequently, our constructed data pipeline is essential for high-quality MCVC data collection. We also hope that our data collection process could provide inspiration for future MCVC works.

#### 4. More details of Multi-Concept Evaluation Set

As mentioned in section 4.3, we manually collect reference images and give out suitable captions for these scenarios, in order to eliminate information leakage when extracting concepts from videos. We demonstrate the video caption templates for the six different scenarios in Tab. 4. The <ID> will be replaced with concept label.

#### 5. Comparison Methods Implementation

We supplement the implementation details of the compared methods for the MCVC task. We compare several open-sourced multi-concept image customization methods, including CustomDiffusion [31], SSR-Encoder [69], IPAdapter [67] and MS-Diffusion [61], combining with the image-to-video (I2V) generation model I2VGen-XL [68], as a naive solution for the MCVC task with our ConceptMaster. While SSR-Encoder, IP-Adapter and MS-Diffusion do not need additional training and can be viewed as feedforward customization models, CustomDiffusion requires users to manually few-shot examples for additional parameter training. Therefore, we use the collected reference images for each concept as training samples. Additionally, we follow its usage requirements, where we label specific character for each concept (i.e., s1* elephant is playing with s2* dog on the road). In addition to Tab. 1 and Fig. 5 in the main paper, we demonstrate more quantitative results in Fig. 9.

Comparing with aforementioned methods, our end-to-end video customization models ConceptMaster could achieve better generation results more practicality than two-stage solutions.

#### 6. More Discussions on Multi-Concept Embeddings Injection

We demonstrate more quantitative results between these three different multi-concept embeddings injection methods in Fig. 10. Our key insight is to inject the represented multi-concept embeddings into the diffusion models in a standalone cross-attention layer. While previous methods that the most representative ones include 1) BLIPDiffusion [34], which combines visual and textual caption embeddings as the whole condition representation. 2) IPAdapter [67], which encodes the whole image as visual embeddings and aggregates into models by a decoupled crossattention layer. These two technical thoughts are widely adopted in previous image and video concept customization, which are different from our core insight. In addition to Tab. 1 and Fig. 5 in the main paper, we demonstrate more quantitative results in Fig. 10, where both these two methods can hardly deal with the identity decoupling problem when there are similar concepts (e.g., a man and a girl). Even when concepts have huge semantic differences (i.e., a man and a red jacket), both two methods cannot maintain the concept fidelity of each concept. Additionally, when merging the visual and textual embeddings together as the conditions, the instruction following ability becomes unsatisfactory and the dynamic degree also drops. Therefore, our ConceptMaster adopts the most suitable manner of the injection of the multi-concept embeddings, which could represent and decouple multiple identities well.

#### 7. More Discussions on Ablation Study

We demonstrate more quantitative results of the effectiveness of the Q-Former and DAM modules in Fig. 11. Initially, our ConceptMaster proposes to firstly utilizes a QFormer network to integrate the dense visual tokens extracted by CLIP image encoder into the comprehensive visual embeddings. Additionally, we introduce the DAM module, which conducts the intra-pair attention module based on paired visual embeddings and textual descriptions representation. These designs demonstrate effective capability of decoupling and representing of multiple given references. While we also conduct several ablation architectures: 1) Replacing the Q-Former by an MLP layer, in Fig. 11 we can see that the visual appearances of provided woman and dog could no longer be captured by the generation results. Therefore, the Q-Former is significant to assist the representation of comprehensive visual embeddings. 2) Removing the DAM module, where only the extracted vi-

sual embeddings are further injected into the diffusion models, and the textual descriptions are unused. Similar phenomenon could be observed in Fig. 11 as the concepts could not keep their fidelity. This is because the absence of adequate text label representation fails to effectively represent and differentiate the uniqueness of multiple concepts, and it also results in poor alignment with the original diffusion space. 3) Replacing DAM by firstly concatenating the visual and textual embeddings along channel dimension, and downsampling the dimension to the original one by an MLP layer. While the insufficient representation would lead to the inharmonious combination of multiple concepts. Since when concatenating along the channel dimension and then downsamling by MLP, the tokens could not conduct interaction among them. As in Fig. 11, the way that the woman who walks dog on the beach is unreasonable. 4) Replacing the intra-pair cross-attention by self-attention, where we firstly add the visual and textual embeddings and then conduct the self-attention operation. According to the qualitative and quantitative results in Tab. 3 and Fig. 7 in main paper and Fig. 11, the cross-attention operation is better than self-attention, as the latter easily leads to more artifacts and inharmonious movements. Therefore, our proposed QFormer and DAM modules would be the best designated architectures to simultaneously represent and decouple multiple references, and could create high-quality customized videos.

#### 8. More Qualitative Results Demonstration

Our ConceptMaster could create high-quality and conceptconsistent customized videos based on given multiple reference images in diverse scenarios, including but not limited to 1) multiple persons, 2) persons with livings, 3) persons with stuffs, 4) multiple livings, 5) livings with stuffs and 6) persons with both livings and stuffs. We demonstrate more qualitative results including these scenes in Fig. 12 and Fig. 13.

|Diverse Scenarios<br><br>|Caption Templates|
|---|---|
|1) Multiple Persons|<ID1> and <ID2> hugging each other in front of a bridge. <ID1> and <ID2> kissing each other in front of a bridge. <ID1> and <ID2> walking down a city street. <ID1> and <ID2> dancing on a city street. <ID1> and <ID2> smiling and shaking hands in the office. <ID1> and <ID2> walking on the beach.<br><br>|
|2) Persons with Livings|<ID1> walking <ID2> on the beach. <ID1> walking <ID2> in the woods. <ID1> petting <ID2> in the park. <ID1> feeding <ID2> in the garden. <ID1> and <ID2> running on the grass. <ID1> rides <ID2> running on the farm. <ID1> petting <ID2> in the stable. <ID1> raising <ID2> in the garden. <ID1> holding <ID2> and walking on the street. <ID1> feeding <ID2> on the street. <ID1> and <ID2> walking in the desert.<br><br>|
|3) Persons with Stuffs|<ID1> wearing <ID2> walking in the shopping mall. <ID1> wearing <ID2> walking along the river. <ID1> wearing <ID2> running in the stadium. <ID1> wearing <ID2> dancing on the floor. <ID1> rides <ID2> in the desert. <ID1> rides <ID2> on the road. <ID1> is happily playing <ID2> on the bench. <ID1> is happily playing <ID2> in the desert. <ID1> taking <ID2>, snowy winter. <ID1> holding <ID2> and walking in the park. <ID1> raising <ID2> in the garden. <ID1> holding <ID2> and walking on the street. <ID1> walking around <ID2> on the street. <ID1> dancing in front of <ID2>, snowy day. <ID1> walking in front of <ID2> on the street. <ID1> dancing in front of <ID2> on the street.<br><br>|
|4) Multiple Livings<br><br>|<ID1> is playing with <ID2> on the road. <ID1> and <ID2> walking on the grass. <ID1> walking around <ID2> in the desert. <ID1> is playing with <ID2> on the street.|
|5) Livings with Stuffs<br><br>|<ID1> walking around <ID2> in a grassy field. <ID1> playing with <ID2> in a grassy field. <ID1> walking around <ID2> at home. <ID1> is playing with <ID2> at home. <ID1> is walking around <ID2> on the road. <ID1> is staying besides <ID2> in the snow. <ID1> is drinking water from <ID2> in the garden. <ID1> is drinking water from <ID2> on the road.|
|6) Persons with both Livings and Stuffs|<ID1> walking <ID2> in front of <ID3> on the street. <ID1> and <ID2> walking around <ID3> in the snow. <ID1> and <ID2> walking around <ID3> on the road. <ID1> is happily playing <ID2> and <ID3> surrounds in the garden. <ID1> taking <ID2> and walking with <ID3> on the grass.<br><br>|

###### Table 4. Caption templates for Multi-Concept Evaluation Set.

|[Figure 260]|
|---|

|[Figure 261]|
|---|

|[Figure 262]|
|---|

|[Figure 263]|
|---|

“A dog walking around the bouquet of flowers in a grassy field.”

“A man rides a motorcycle in the desert.”

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

[Figure 321]

[Figure 322]

[Figure 323]

(a) CustomDiffusion (b) SSR-Encoder (c) IP-Adapter (d) MS-Diffusion (e) ConceptMaster

(a) CustomDiffusion (b) SSR-Encoder (c) IP-Adapter (d) MS-Diffusion (e) ConceptMaster

Figure 9. More Qualitative comparison on multi-concept customization between ConceptMaster and naively combining the multi-concept image customization with image-to-video generation models.

|[Figure 324]|
|---|

|[Figure 325]|
|---|

“A man wearing a red floral cotton jacket walking in the shopping mall.”

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

(a) Merge Textual and Visual Embeddings

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

- (b) IP-Adapter-like

[Figure 338]

[Figure 339]

- (c) ConceptMaster

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

Figure 10. More Qualitative comparison on different injection methods of multi-concept references.

|[Figure 344]|
|---|

|[Figure 345]|
|---|

“A woman wearing cowboy hat is walking a dog on the beach.”

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

- (a) Without Q-Former
- (b) Without DAM
- (c) Concate-MLP
- (d) Self-Attn

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

(e) DAM

Figure 11. More Qualitative comparison the effectiveness of the Q-Former and DAM modules.

|[Figure 376]|[Figure 377]|
|---|---|

“A man and a woman kissing each other in front of a bridge.”

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

|[Figure 384]|
|---|

|[Figure 385]|
|---|

“A child rides a horse running on the farm.”

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

|[Figure 392]|
|---|

|[Figure 393]|
|---|

“A man wearing a red floral cotton jacket running in the stadium.”

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

|[Figure 400]|
|---|

|[Figure 401]|
|---|

“A dog staying besides a teddy bear in the desert.”

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

|[Figure 408]|
|---|

|[Figure 409]|
|---|

“A cat lying besides a car, snowy day.”

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

|[Figure 416]|[Figure 417]|
|---|---|

|[Figure 418]|
|---|

“A man is happily playing the guitar and a dog surrounds him in the garden.”

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

Figure 12. More qualitative results of ConceptMaster on diverse scenarios (1/2).

|[Figure 425]|
|---|

|[Figure 426]|
|---|

“A man and a woman dancing on a city street.”

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

|[Figure 433]|[Figure 434]|
|---|---|

“A man raising a teddy bear with hands in the garden.”

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

|[Figure 441]|
|---|

|[Figure 442]|
|---|

“A man rides a motorcycle in the desert.”

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

|[Figure 449]|
|---|

|[Figure 450]|
|---|

“A cat is playing with another cat on the road.”

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

|[Figure 457]|
|---|

|[Figure 458]|
|---|

“A dog laying on the floor next to a bed.”

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

|[Figure 465]|
|---|

|[Figure 466]|
|---|

|[Figure 467]|
|---|

“A man wearing a suit rides a horse running in a grassy field.”

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

|[Figure 474]|[Figure 475]|[Figure 476]|
|---|---|---|

“A girl petting a dog in front of the sofa at home.”

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

Figure 13. More qualitative results of ConceptMaster on diverse scenarios (2/2).

