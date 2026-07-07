[Figure 1]

## Customizing Realistic Human Photos via Stacked ID Embedding

Zhen Li1,2* Mingdeng Cao2,3* Xintao Wang2† Zhongang Qi2 Ming-Ming Cheng1† Ying Shan2 1VCIP, CS, Nankai University 2ARC Lab, Tencent PCG 3The University of Tokyo https://photo-maker.github.io/

A man wearing headphones with red hair

A woman wearing sunglasses and necklace

A man wearing a Christmas hat

A boy wearing a doctoral cap

# arXiv:2312.04461v1[cs.CV]7Dec2023

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

User inputs

User inputs

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

(a) Attributes change

A man coding in front of a computer

A man in a helmet and vest riding a motorcycle

A man wearing headphones

A man wearing a spacesuit

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

User inputs User inputs

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

###### (b) Artwork / old-photo to reality

User inputs

User inputs

A woman holding a bottle of red wine

A photo of a woman A woman with red hair

A woman in the snow

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

(c) Identity mixing

Figure 1. Given a few images of input ID(s), the proposed PhotoMaker can generate diverse personalized ID images based on the text prompt in a single forward pass. Our method can well preserve the ID information from the input image pool while generating realistic human photos. PhotoMaker also empowers many interesting applications such as (a) changing attributes, (b) bringing persons from artworks or old photos into reality, or (c) performing identity mixing. (Zoom-in for the best view)

### Abstract

(ID) fidelity, and flexible text controllability. In this work, we introduce PhotoMaker, an efficient personalized textto-image generation method, which mainly encodes an arbitrary number of input ID images into a stack ID embedding for preserving ID information. Such an embedding, serving as a unified ID representation, can not only encapsulate the characteristics of the same input ID comprehensively, but also accommodate the characteristics of different IDs for subsequent integration. This paves the way for

Recent advances in text-to-image generation have made remarkable progress in synthesizing realistic human photos conditioned on given text prompts. However, existing personalized generation methods cannot simultaneously satisfy the requirements of high efficiency, promising identity

∗ Interns in ARC Lab, Tencent PCG † Corresponding authors

more intriguing and practically valuable applications. Besides, to drive the training of our PhotoMaker, we propose an ID-oriented data construction pipeline to assemble the training data. Under the nourishment of the dataset constructed through the proposed pipeline, our PhotoMaker demonstrates better ID preservation ability than test-time fine-tuning based methods, yet provides significant speed improvements, high-quality generation results, strong generalization capabilities, and a wide range of applications.

### 1. Introduction

Customized image generation related to humans [28, 35, 51] has received considerable attention, giving rise to numerous applications, such as personalized portrait photos [1], image animation [67], and virtual try-on [59]. Early methods [39, 41], limited by the capabilities of generative models (i.e., GANs [19, 29]), could only customize the generation of the facial area, resulting in low diversity, scene richness, and controllability. Thanks to larger-scale textimage pair training datasets [55], larger generation models [44, 52], and text/visual encoders [45, 46] that can provide stronger semantic embeddings, diffusion-based textto-image generation models have been continuously evolving recently. This evolution enables them to generate increasingly realistic facial details and rich scenes. The controllability has also greatly improved due to the existence of text prompts and structural guidance [40, 65]

Meanwhile, under the nurturing of powerful diffusion text-to-image models, many diffusion-based customized generation algorithms [17, 50] have emerged to meet users’ demand for high-quality customized results. The most widely used in both commercial and community applications are DreamBooth-based methods [2, 50]. Such applications require dozens of images of the same identity (ID) to fine-tune the model parameters. Although the results generated have high ID fidelity, there are two obvious drawbacks: one is that customized data used for fine-tuning each time requires manual collection and thus is very timeconsuming and laborious; the other is that customizing each ID requires 10-30 minutes, consuming a large amount of computing resources, especially when the generation model becomes larger. Therefore, to simplify and accelerate the customized generation process, recent works, driven by existing human-centric datasets [29, 36], have trained visual encoders [11, 62] or hypernetworks [5, 51] to represent the input ID images as embeddings or LoRA [25] weights of the model. After training, users only need to provide an image of the ID to be customized, and personalized generation can be achieved through a few dozen steps of fine-tuning or even without any tuning process. However, the results customized by these methods cannot simultaneously possess ID fidelity and generation diversity like DreamBooth (see

Fig. 3). This is because that: 1) during the training process, both the target image and the input ID image sample from the same image. The trained model easily remembers characteristics unrelated to the ID in the image, such as expressions and viewpoints, which leads to poor editability, and 2) relying solely on a single ID image to be customized makes it difficult for the model to discern the characteristics of the ID to be generated from its internal knowledge, resulting in unsatisfactory ID fidelity.

Based on the above two points, and inspired by the success of DreamBooth, in this paper, we aim to: 1) ensure that the ID image condition and the target image exhibit variations in viewpoints, facial expressions, and accessories, so that the model does not memorize information that is irrelevant to the ID; 2) provide the model with multiple different images of the same ID during the training process to more comprehensively and accurately represent the characteristics of the customized ID.

Therefore, we propose a simple yet effective feedforward customized human generation framework that can receive multiple input ID images, termed as PhotoMaker. To better represent the ID information of each input image, we stack the encodings of multiple input ID images at the semantic level, constructing a stacked ID embedding. This embedding can be regarded as a unified representation of the ID to be generated, and each of its subparts corresponds to an input ID image. To better integrate this ID representation and the text embedding into the network, we replace the class word (e.g., man and woman) of the text embedding with the stacked ID embedding. The result embedding simultaneously represents the ID to be customized and the contextual information to be generated. Through this design, without adding extra modules in the network, the cross-attention layer of the generation model itself can adaptively integrate the ID information contained in the stacked ID embedding.

At the same time, the stacked ID embedding allows us to accept any number of ID images as input during inference while maintaining the efficiency of the generation like other tuning-free methods [56, 62]. Specifically, our method requires about 10 seconds to generate a customized human photo when receiving four ID images, which is about 130× faster than DreamBooth1. Moreover, since our stacked ID embedding can represent the customized ID more comprehensively and accurately, our method can provide better ID fidelity and generation diversity compared to state-of-theart tuning-free methods. Compared to previous methods, our framework has also greatly improved in terms of controllability. It can not only perform common recontextualization but also change the attributes of the input human image (e.g., accessories and expressions), generate a human photo with completely different viewpoints from the

1Test on one NVIDIA Tesla V100

input ID, and even modify the input ID’s gender and age (see Fig. 1).

It is worth noticing that our PhotoMaker also unleashes a lot of possibilities for users to generate customized human photos. Specifically, although the images that build the stacked ID embedding come from the same ID during training, we can use different ID images to form the stacked ID embedding during inference to merge and create a new customized ID. The merged new ID can retain the characteristics of different input IDs. For example, we can generate Scarlett Johansson that looks like Elun Musk or a customized ID that mixes a person with a well-known IP character (see Fig. 1(c)). At the same time, the merging ratio can be simply adjusted by prompt weighting [3, 21] or by changing the proportion of different ID images in the input image pool, demonstrating the flexibility of our framework.

Our PhotoMaker necessitates the simultaneous input of multiple images with the same ID during the training process, thereby requiring the support of an ID-oriented human dataset. However, existing datasets either do not classify by IDs [29, 35, 55, 68] or only focus on faces without including other contextual information [36, 41, 60]. We therefore design an automated pipeline to construct an IDrelated dataset to facilitate the training of our PhotoMaker. Through this pipeline, we can build a dataset that includes a large number of IDs, each with multiple images featuring diverse viewpoints, attributes, and scenarios. Meanwhile, in this pipeline, we can automatically generate a caption for each image, marking out the corresponding class word [50], to better adapt to the training needs of our framework.

### 2. Related work

Text-to-Image Diffusion Models. Diffusion models [23, 58] have made remarkable progress in text-conditioned image generation [30, 47, 49, 52], attracting widespread attention in recent years. The remarkable performance of these models can be attributable to high-quality largescale text-image datasets [9, 54, 55], the continuous upgrades of foundational models [10, 43], conditioning encoders [26, 45, 46], and the improvement of controllability [34, 40, 63, 65]. Due to these advancements, Podell et al. [44] developed the currently most powerful open-source generative model, SDXL. Given its impressive capabilities in generating human portraits, we build our PhotoMaker based on this model. However, our method can also be extended to other text-to-image synthesis models.

Personalization in Diffusion Models. Owing to the powerful generative capabilities of the diffusion models, more researchers try to explore personalized generation based on

- them. Currently, mainstream personalized synthesis methods can be mainly divided into two categories. One relies on additional optimization during the test phase, such

as DreamBooth [50] and Textual Inversion [17]. Given that both pioneer works require substantial time for finetuning, some studies have attempted to expedite the process of personalized customization by reducing the number of parameters needed for tuning [2, 20, 32, 64] or by pre-training with large datasets [18, 51]. Despite these advances, they still require extensive fine-tuning of the pretrained model for each new concept, making the process time-consuming and restricting its applications. Recently, some studies [12, 13, 27, 37, 38, 56, 61] attempt to perform personalized generation using a single image with a single forward pass, significantly accelerating the personalization process. These methods either utilize personalization datasets [12, 57] for training or encode the images to be customized in the semantic space [11, 27, 38, 56, 61, 62]. Our method focuses on the generation of human portraits based on both of the aforementioned technical approaches. Specifically, it not only relies on the construction of an IDoriented personalization dataset, but also on obtaining the embedding that represents the person’s ID in the semantic space. Unlike previous embedding-based methods, our PhotoMaker extracts a stacked ID embedding from multiple ID images. While providing better ID representation, the proposed method can maintain the same high efficiency as previous embedding-based methods.

### 3. Method

#### 3.1. Overview

Given a few ID images to be customized, the goal of our PhotoMaker is to generate a new photo-realistic human image that retains the characteristics of the input IDs and changes the content or the attributes of the generated ID under the control of the text prompt. Although we input multiple ID images for customization like DreamBooth, we still enjoy the same efficiency as other tuning-free methods, accomplishing customization with a single forward pass, while maintaining promising ID fidelity and text edibility. In addition, we can also mix multiple input IDs, and the generated image can well retain the characteristics of different IDs, which releases possibilities for more applications. The above capabilities are mainly brought by our proposed simple yet effective stacked ID embedding, which can provide a unified representation of the input IDs. Furthermore, to facilitate training our PhotoMaker, we design a data construction pipeline to build a human-centric dataset classified by IDs. Fig. 2(a) shows the overview of the proposed PhotoMaker. Fig. 2(b) shows our data construction pipeline.

#### 3.2. Stacked ID Embedding

Encoders. Following recent works [27, 56, 61], we use the CLIP [45] image encoder Eimg to extract image embeddings for its alignment with the original text representation space

Training ID Images

[Figure 39]

1. Image Downloading

Diffusion Model

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Image Encoder

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

2. Face Detection & Filtering

[Figure 50]

[Figure 51]

| |
|---|

[Figure 52]

| |
|---|

[Figure 53]

| |
|---|

[Figure 54]

Image Embeddings

[Figure 55]

[Figure 56]

3. ID Verification

[Figure 57]

[Figure 58]

###### Stacked ID Embedding

[Figure 59]

Grouped by IDs

MLPs

Text Embedding

###### Inference ID Images

- 4. Cropping & Segmentation

- 5. Captioning & Marking

[Figure 60]

[Figure 61]

Grouped by IDs

[Figure 62]

[Figure 63]

Updated Text Embedding

Text Encoder(s)

Training Inference Shared

“a woman with long hair and a white shirt is looking down” “a man in a black suit with a sword in her hand” …

“A man in a suit and tie with a badge on his lapel”

“A man wearing a blue cap”

(a) (b)

- Figure 2. Overviews of the proposed (a) PhotoMaker and (b) ID-oriented data construction pipeline. For the proposed PhotoMaker, we first obtain the text embedding and image embeddings from text encoder(s) and image encoder, respectively. Then, we extract the fused embedding by merging the corresponding class embedding (e.g., man and woman) and each image embedding. Next, we concatenate all fused embeddings along the length dimension to form the stacked ID embedding. Finally, we feed the stacked ID embedding to all cross-attention layers for adaptively merging the ID content in the diffusion model. Note that although we use images of the same ID with the masked background during training, we can directly input images of different IDs without background distortion to create a new ID during inference.

in diffusion models. Before feeding each input image into the image encoder, we filled the image areas other than the body part of a specific ID with random noises to eliminate the influence of other IDs and the background. Since the data used to train the original CLIP image encoder mostly consists of natural images, to better enable the model to extract ID-related embeddings from the masked images, we finetune part of the transformer layers in the image encoder when training our PhotoMaker. We also introduce additional learnable projection layers to inject the embedding obtained from the image encoder into the same dimension

- as the text embedding. Let {Xi | i = 1...N} denote N input ID images acquired from a user, we thus obtain the extracted embeddings {ei ∈ RD | i = 1...N}, where D denotes the projected dimension. Each embedding corresponds to the ID information of an input image. For a given text prompt T, we extract text embeddings t ∈ RL×D using

the pre-trained CLIP text encoder Etext, where L denotes the length of the embedding.

Stacking. Recent works [17, 50, 62] have shown that, in the text-to-image models, personalized character ID information can be represented by some unique tokens. Our method also has a similar design to better represent the ID information of the input human images. Specifically, we mark the corresponding class word (e.g., man and woman) in the input caption (see Sec. 3.3). We then extract the feature vector

- at the corresponding position of the class word in the text embedding. This feature vector will be fused with each im-

age embedding ei. We use two MLP layers to perform such a fusion operation. The fused embeddings can be denoted as {eˆi ∈ RD | i = 1...N}. By combining the feature vector of the class word, this embedding can represent the current input ID image more comprehensively. In addition, during the inference stage, this fusion operation also provides stronger semantic controllability for the customized generation process. For example, we can customize the age and gender of the human ID by simply replacing the class word (see Sec. 4.2).

After obtaining the fused embeddings, we concatenate them along the length dimension to form the stacked id embedding:

s∗ = Concat([ˆe1,...,eˆN]) s∗ ∈ RN×D. (1)

This stacked ID embedding can serve as a unified representation of multiple ID images while it retains the original representation of each input ID image. It can accept any number of ID image encoded embeddings, therefore, its length N is variable. Compared to DreamBooth-based methods [2, 50], which inputs multiple images to finetune the model for personalized customization, our method essentially sends multiple embeddings to the model simultaneously. After packaging the multiple images of the same ID into a batch as the input of the image encoder, a stacked ID embedding can be obtained through a single forward pass, significantly enhancing efficiency compared to tuning-based methods. Meanwhile, compared to other

embedding-based methods [61, 62], this unified representation can maintain both promising ID fidelity and text controllability, as it contains more comprehensive ID information. In addition, it is worth noting that, although we only used multiple images of the same ID to form this stacked ID embedding during training, we can use images that come from different IDs to construct it during the inference stage. Such flexibility opens up possibilities for many interesting applications. For example, we can mix two persons that exist in reality or mix a person and a well-known character IP (see Sec. 4.2).

Merging. We use the inherent cross-attention mechanism in diffusion models to adaptively merge the ID information contained in stacked ID embedding. We first replace the feature vector at the position corresponding to the class word in the original text embedding t with the stacked id embedding s∗, resulting in an update text embedding t∗ ∈ R(L+N−1)×D. Then, the cross-attention operation can be formulated as:

Q = WQ · ϕ(zt); K = WK · t∗; V = WV · t∗ Attention(Q,K,V) = softmax(QK

T

d ) · V,

√

(2) where ϕ(·) is an embedding that can be encoded from the input latent by the UNet denoiser. WQ, WK, and WV are projection matrices. Besides, we can adjust the degree of participation of one input ID image in generating the new customized ID through prompt weighting [3, 21], demonstrating the flexibility of our PhotoMaker. Recent works [2, 32] found that good ID customization performance can be achieved by simply tuning the weights of the attention layers. To make the original diffusion models better perceive the ID information contained in stacked ID embedding, we additionally train the LoRA [2, 25] residuals of the matrices in the attention layers.

#### 3.3. ID-Oriented Human Data Construction

Since our PhotoMaker needs to sample multiple images of the same ID for constructing the stacked ID embedding during the training process, we need to use a dataset classified by IDs to drive the training process of our PhotoMaker. However, existing human datasets either do not annotate ID information [29, 35, 55, 68], or the richness of the scenes they contain is very limited [36, 41, 60] (i.e., they only focus on the face area). Thus, in this section, we will introduce a pipeline for constructing a human-centric text-image dataset, which is classified by different IDs. Fig. 2(b) illustrates the proposed pipeline. Through this pipeline, we can collect an ID-oriented dataset, which contains a large number of IDs, and each ID has multiple images that include different expressions, attributes, scenes, etc. This dataset not only facilitates the training process of our PhotoMaker but

also may inspire potential future ID-driven research. The statistics of the dataset are shown in the appendix.

Image downloading. We first list a roster of celebrities, which can be obtained from VoxCeleb1 and VGGFace2 [7]. We search for names in the search engine according to the list and crawled the data. About 100 images were downloaded for each name. To generate higher quality portrait images [44], we filtered out images with the shortest side of the resolution less than 512 during the download process.

Face detection and filtering. We first use RetinaNet [16] to detect face bounding boxes and filter out the detections with small sizes (less than 256 × 256). If an image does not contain any bounding boxes that meet the requirements, the image will be filtered out. We then perform ID verification for the remaining images.

ID verification. Since an image may contain multiple faces, we need first to identify which face belongs to the current identity group. Specifically, we send all the face regions in the detection boxes of the current identity group into ArcFace [15] to extract identity embeddings and calculate the L2 similarity of each pair of faces. We sum the similarity calculated by each identity embedding with all other embeddings to get the score for each bounding box. We select the bounding box with the highest sum score for each image with multiple faces. After bounding box selection, we recompute the sum score for each remaining box. We calculate the standard deviation δ of the sum score by ID group. We empirically use 8δ as a threshold to filter out images with inconsistent IDs.

Cropping and segmentation. We first crop the image with a larger square box based on the detected face area while ensuring that the facial region can occupy more than 10% of the image after cropping. Since we need to remove the irrelevant background and IDs from the input ID image before sending it into the image encoder, we need to generate the mask for the specified ID. Specifically, we employ the Mask2Former [14] to perform panoptic segmentation for the ‘person’ class. We leave the mask with the highest overlap with the facial bounding box corresponding to the ID. Besides, we choose to discard images where the mask is not detected, as well as images where no overlap is found between the bounding box and the mask area.

Captioning and marking We use BLIP2 [33] to generate a caption for each cropped image. Since we need to mark the class word (e.g., man, woman, and boy) to facilitate the fusion of text and image embeddings, we regenerate captions that do not contain any class word using the random mode of BLIP2 until a class word appears. After obtaining the caption, we singularize the class word in the caption to focus on a single ID. Next, we need to mark the position of the class word that corresponds to the current ID. Captions that contain only one class word can be directly

annotated. For captions that contain multiple class words, we count the class words contained in the captions for each identity group. The class word with the most occurrences will be the class word for the current identity group. We

- then use the class word of each identity group to match and mark each caption in that identity group. For a caption that does not include the class word that matches that of the corresponding identity group, we employ a dependence parsing model [24] to segment the caption according
- to different class words. We calculate the CLIP score [45] between the sub-caption after segmentation and the specific ID region in the image. Besides, we calculate the label similarity between the class word of the current segment and the class word of the current identity group through SentenceFormer [48]. We choose to mark the class word corresponding to the maximum product of the CLIP score and the label similarity.

### 4. Experiments 4.1. Setup

Implementation details. To generate more photorealistic human portraits, we employ SDXL model [44] stable-diffusion-xl-base-1.0 as our text-toimage synthesis model. Correspondingly, the resolution of training data is resized to 1024 × 1024. We employ CLIP ViT-L/14 [45] and an additional projection layer to obtain the initial image embeddings ei. For text embeddings, we keep the original two text encoders in SDXL for extraction. The overall framework is optimized with Adam [31] on 8 NVIDIA A100 GPUs for two weeks with a batch size of 48. We set the learning rate as 1e − 4 for LoRA weights, and 1e−5 for other trainable modules. During training, we randomly sample 1-4 images with the same ID as the current target ID image to form a stacked ID embedding. Besides, to improve the generation performance by using classifierfree guidance, we have a 10% chance of using null-text embedding to replace the original updated text embedding t∗. We also use masked diffusion loss [6] with a probability of 50% to encourage the model to generate more faithful IDrelated areas. During the inference stage, we use delayed subject conditioning [62] to solve the conflicts between text and ID conditions. We use 50 steps of DDIM sampler [58]. The scale of classifier-free guidance is set to 5.

Evaluation metrics. Following DreamBooth [50], we use DINO [8] and CLIP-I [17] metrics to measure the ID fidelity and use CLIP-T [45] metric to measure the prompt fidelity. For a more comprehensive evaluation, we also compute the face similarity by detecting and cropping the facial regions between the generated image and the real image with the same ID. We use RetinaFace [16] as the detection model. Face embedding is extracted by FaceNet [53]. To evaluate the quality of the generation, we employ the FID met-

ric [22, 42]. Importantly, as most embedding-based methods tend to incorporate facial pose and expression into the representation, the generated images often lack variation in the facial region. Thus, we propose a metric, named Face Diversity, to measure the diversity of the generated facial regions. Specifically, we first detect and crop the face region in each generated image. Next, we calculate the LPIPS [66] scores between each pair of facial areas for all generated images and take the average. The larger this value, the higher the diversity of the generated facial area.

Evaluation dataset. Our evaluation dataset includes 25 IDs, which consist of 9 IDs from Mystyle [41] and an additional 16 IDs that we collected by ourselves. Note that these IDs do not appear in the training set, serving to evaluate the generalization ability of the model. To conduct a more comprehensive evaluation, we also prepare 40 prompts, which cover a variety of expressions, attributes, decorations, actions, and backgrounds. For each prompt of each ID, we generate 4 images for evaluation. More details are listed in the appendix.

#### 4.2. Applications

In this section, we will elaborate on the applications that our PhotoMaker can empower. For each application, we choose the comparison methods which may be most suitable for the corresponding setting. The comparison method will be chosen from DreamBooth [50], Textual Inversion [17], FastComposer [62], and IPAdapter [63]. We prioritize using the official model provided by each method. For DreamBooth and IPAdapter, we use their SDXL versions for a fair comparison. For all applications, we have chosen four input ID images to form the stacked ID embedding in our PhotoMaker. We also fairly use four images to train the methods that need test-time optimization. We provide more samples in the appendix for each application.

Recontextualization We first show results with simple context changes such as modified hair color and clothing, or generate backgrounds based on basic prompt control. Since all methods can adapt to this application, we conduct quantitative and qualitative comparisons of the generated results (see Tab. 1 and Fig. 3). The results show that our method can well satisfy the ability to generate high-quality images, while ensuring high ID fidelity (with the largest CLIP-T and DINO scores, and the second best Face Similarity). Compared to most methods, our method generates images of higher quality, and the generated facial regions exhibit greater diversity. At the same time, our method can maintain a high efficiency consistent with embedding-based methods. For a more comprehensive comparison, we show the user study results in Sec. B in the appendix.

Bringing person in artwork/old photo into reality. By taking artistic paintings, sculptures, or old photos of a person as input, our PhotoMaker can bring a person from the

References DreamBooth Textual Inversion FastComposer IPAdapter PhotoMaker (Ours)

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

adoctoralcap Awearingaman

spacesuit Awearingaman

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Christmashat

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Asittingwoman

atthebeach,

withpurple

sunset

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Awithredwoman

hair Awearingwoman

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

- Figure 3. Qualitative comparison on universal recontextualization samples. We compare our method with DreamBooth [50], Textual Inversion [17], FastComposer [62], and IPAdapter [63] for five different identities and corresponding prompts. We observe that our method generally achieves high-quality generation, promising editability, and strong identity fidelity. (Zoom-in for the best view)

CLIP-T↑ (%) CLIP-I↑ (%) DINO↑ (%) Face Sim.↑ (%) Face Div.↑ (%) FID↓ Speed↓ (s)

DreamBooth [50] 29.8 62.8 39.8 49.8 49.1 374.5 1284 Textual Inversion [17] 24.0 70.9 39.3 54.3 59.3 363.5 2400 FastComposer [62] 28.7 66.8 40.2 61.0 45.4 375.1 8 IPAdapter [63] 25.1 71.2 46.2 67.1 52.4 375.2 12 PhotoMaker (Ours) 26.1 73.6 51.5 61.8 57.7 370.3 10

Table 1. Quantitative comparison on the universal recontextualization setting. The metrics used for benchmarking cover the ability to preserve ID information (i.e., CLIP-I, DINO, and Face Similarity), text consistency (i.e., CLIP-T), diversity of generated faces (i.e., Face Diversity), and generation quality (i.e., FID). Besides, we define personalized speed as the time it takes to obtain the final personalized image after feeding the ID condition(s). We measure personalized time on a single NVIDIA Tesla V100 GPU. The best result is shown in bold, and the second best is underlined.

last century or even ancient times to the present century to “take” photos for them. Fig. 4(a) illustrate the results. Compared to our method, both Dreambooth and SDXL have difficulty generating realistic human images that have not appeared in real photos. In addition, due to the excessive re-

liance of DreamBooth on the quality and resolution of customized images, it is difficult for DreamBooth to generate high-quality results when using old photos for customized generation.

Changing age or gender. By simply replacing class words

References DreamBooth SDXL PhotoMaker (Ours)

DreamBooth SDXL PhotoMaker (Ours)

References

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

A woman happily smiling, looking at the camera

A man coding in front of a computer

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

A man piloting a spaceship

A girl wearing a Christmas hat

(a) (b)

- Figure 4. Applications on (a) artwork and old photo, and (b) changing age or gender. We are able to bring the past people back to real life or change the age and gender of the input ID. For the first application, we prepare a prompt template A photo of <original prompt>, photo-realistic for DreamBooth and SDXL. Correspondingly, we change the class word to the celebrity name in the original prompt. For the second one, we replace the class word to <class word> <name>, (at the age of 12) for them.

References

DreamBooth SDXL PhotoMaker (Ours)

(e.g. man and woman), our method can achieve changes in gender and age. Fig. 4(b) shows the results. Although SDXL and DreamBooth can also achieve the corresponding effects after prompt engineering, our method can more easily capture the characteristic information of the characters due to the role of the stacked ID embedding. Therefore, our results show a higher ID fidelity.

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

A man holding a bottle of red wine

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Identity mixing. If the users provide images of different IDs as input, our PhotoMaker can well integrate the characteristics of different IDs to form a new ID. From Fig. 5, we can see that neither DreamBooth nor SDXL can achieve identity mixing. In contrast, our method can retain the characteristics of different IDs well on the generated new ID, regardless of whether the input is an anime IP or a real person, and regardless of gender. Besides, we can control the proportion of this ID in the new generated ID by controlling the corresponding ID input quantity or prompt weighting. We show this ability in Fig. 10-11 in the appendix.

A woman frowning at the camera

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

A man wearing a spacesuit

- Figure 5. Identity mixing. We are able to generate the image with a new ID while preserving input identity characteristics. We prepare a prompt template <original prompt>, with a face blended with <name:A> and <name:B> for SDXL. (Zoom-in for the best view)

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

A Ukiyo-e painting of a <class>

A painting of a <class>, in Van Gogh style

A <class> in a comic References A sketch of a <class> book

- Figure 6. The stylization results of our PhotoMaker. The symbol <class> denotes it will be replaced by man or woman accordingly. (Zoom-in for the best view)

Stylization. In Fig. 6, we demonstrate the stylization capabilities of our method. We can see that, in the generated images, our PhotoMaker not only maintains good ID fidelity but also effectively exhibits the style information of the input prompt. This reveals the potential for our method to drive more applications. Additional results are shown in the Fig. 12 within the appendix.

#### 4.3. Ablation study

We shortened the total number of training iterations by eight times to conduct ablation studies for each variant.

The influence about the number of input ID images. We explore the impact that forming the proposed stacked ID embedding through feeding different numbers of ID images. In Fig. 7, we visualize this impact across different metrics. We conclude that using more images to form a stacked ID embedding can improve the metrics related to ID fidelity. This improvement is particularly noticeable when the number of input images is increased from one to two.

Upon the input of an increasing number of ID images, the growth rate of the values in the ID-related metrics significantly decelerates. Additionally, we observe a linear de-

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

(a) (b) (c) (d)

- Figure 7. The impact of the number of input ID images on (a) CLIP-I, (b) DINO, (c) CLIP-T, and (d) Face Similarity, respectively.

CLIP-T↑ DINO↑ Face Sim.↑ Face Div.↑

Average 28.7 47.0 48.8 56.3 Linear 28.6 47.3 48.1 54.6 Stacked 28.0 49.5 53.6 55.0

(a) Embedding composing choices.

CLIP-T↑ DINO↑ Face Sim.↑ Face Div.↑

Single embed 27.9 50.3 50.5 56.1 Single image 27.3 50.3 60.4 51.7 Ours 28.0 49.5 53.6 55.0

(b) Training data sampling strategy.

Table 2. Ablation studies for the proposed PhotoMaker. The best results are marked in bold.

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

A photo of a man

A woman wearing a red sweater

[Figure 139]

[Figure 140]

References 1 image 2 images 6 images 10 images

A photo of a woman

- Figure 8. The impact of varying the quantity of input images on the generation results. It can be observed that the fidelity of the ID increases with the quantity of input images.

its effectiveness. Besides, such a way offers greater flexibility than others, including accepting any number of images and better controlling the mixing process of different IDs.

The benefits from multiple embeddings during training. We explore two other training data sampling strategies to demonstrate that it is necessary to input multiple images with variations during training. The first is to choose only one image, which can be different from the target image, to form the ID embedding (see “single embed” in Tab. 2b). Our multiple embedding way has advantages in ID fidelity. The second sampling strategy is to regard the target image as the input ID image (to simulate the training way of most embedding-based methods). We generate multiple images based on this image with different data augmentation methods and extract corresponding multiple embeddings. In Tab. 2b, as the model can easily remember other irrelevant characteristics of the input image, the generated facial area lacks sufficient changes (low diversity).

cline on the CLIP-T metric. This indicates there may exist a trade-off between text controllability and ID fidelity. From Fig. 8, we see that increasing the number of input images enhances the similarity of the ID. Therefore, the more ID images to form the stacked ID embedding can help the model perceive more comprehensive ID information, and then more accurately represent the ID to generate images. Besides, as shown by the Dwayne Johnson example, the gender editing capability decreases, and the model is more prone to generate images of the original ID’s gender.

### 5. Conclusion

We have presented PhotoMaker, an efficient personalized text-to-image generation method that focuses on generating realistic human photos. Our method leverages a simple yet effective representation, stacked ID embedding, for better preserving ID information. Experimental results have demonstrated that our PhotoMaker, compared to other methods, can simultaneously satisfy high-quality and diverse generation capabilities, promising editability, high inference efficiency, and strong ID fidelity. Besides, we also have found that our method can empower many interesting applications that previous methods are hard to achieve, such as changing age or gender, bringing persons from old photos or artworks back to reality, and identity mixing.

The choices of composing multiple embeddings. We explore three ways to compose the ID embedding, including averaging the image embeddings, adaptively projecting embeddings through a linear layer, and our stacking way. From Tab. 2a, we see the stacking way has the highest ID fidelity while ensuring a diversity of generated faces, demonstrating

### References

- [1] Photo ai. https://photoai.com/. Accessed: 202312-08. 2
- [2] Low-rank adaptation for fast text-to-image diffusion finetuning. https://github.com/cloneofsimo/ lora, 2022. 2, 3, 4, 5
- [3] Prompt weighting. https://huggingface.co/ docs/diffusers/using-diffusers/weighted_ prompts, 2023. 3, 5, 2
- [4] Yuval Alaluf, Or Patashnik, and Daniel Cohen-Or. Only a matter of style: Age transformation using a style-based regression model. TOG, 2021. 5
- [5] Moab Arar, Rinon Gal, Yuval Atzmon, Gal Chechik, Daniel Cohen-Or, Ariel Shamir, and Amit H Bermano. Domainagnostic tuning-encoder for fast personalization of text-toimage models. TOG, 2023. 2
- [6] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel CohenOr, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. In SIGGRAPH Asia, 2023. 6
- [7] Qiong Cao, Li Shen, Weidi Xie, Omkar M Parkhi, and Andrew Zisserman. Vggface2: A dataset for recognising faces across pose and age. In FG, 2018. 5
- [8] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 6, 1
- [9] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pretraining to recognize long-tail visual concepts. In CVPR,

2021. 3

- [10] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 3
- [11] Li Chen, Mengyi Zhao, Yiheng Liu, Mingxu Ding, Yangyang Song, Shizun Wang, Xu Wang, Hao Yang, Jing Liu, Kang Du, et al. Photoverse: Tuning-free image customization with text-to-image diffusion models. arXiv preprint arXiv:2309.05793, 2023. 2, 3
- [12] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Rui, Xuhui Jia, Ming-Wei Chang, and William W Cohen. Subject-driven text-to-image generation via apprenticeship learning. arXiv preprint arXiv:2304.00186, 2023. 3
- [13] Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zero-shot object-level image customization. arXiv preprint arXiv:2307.09481, 2023. 3
- [14] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In CVPR,

2022. 5

- [15] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In CVPR, 2019. 5, 1

- [16] Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou. Retinaface: Single-shot multi-level face localisation in the wild. In CVPR, 2020. 5, 6
- [17] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In ICLR, 2023. 2, 3, 4, 6, 7, 1
- [18] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Designing an encoder for fast personalization of text-to-image models. arXiv preprint arXiv:2302.12228, 2023. 3
- [19] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. ACM Communications, 2020. 2
- [20] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. In ICCV, 2023. 3
- [21] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. In ICLR, 2023. 3, 5, 2
- [22] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 6
- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 3
- [24] Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. spacy: Industrial-strength natural language processing in python, 2020. 6
- [25] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In ICLR, 2022. 2, 5
- [26] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. 3
- [27] Xuhui Jia, Yang Zhao, Kelvin CK Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, and Yu-Chuan Su. Taming encoder for zero fine-tuning image customization with text-to-image diffusion models. arXiv preprint arXiv:2304.02642, 2023. 3
- [28] Xuan Ju, Ailing Zeng, Chenchen Zhao, Jianan Wang, Lei Zhang, and Qiang Xu. HumanSD: A native skeleton-guided diffusion model for human image generation. In ICCV, 2023. 2
- [29] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, 2019. 2, 3, 5
- [30] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, 2023. 3
- [31] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015. 6

- [32] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In CVPR, 2023. 3, 5
- [33] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML,

2023. 5

- [34] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In CVPR, 2023. 3
- [35] Xian Liu, Jian Ren, Aliaksandr Siarohin, Ivan Skorokhodov, Yanyu Li, Dahua Lin, Xihui Liu, Ziwei Liu, and Sergey Tulyakov. Hyperhuman: Hyper-realistic human generation with latent structural diffusion. arXiv preprint arXiv:2310.08579, 2023. 2, 3, 5
- [36] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In ICCV, 2015. 2, 3, 5
- [37] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subject-diffusion: Open domain personalized text-to-image generation without test-time fine-tuning. arXiv preprint arXiv:2307.11410, 2023. 3
- [38] Yiyang Ma, Huan Yang, Wenjing Wang, Jianlong Fu, and Jiaying Liu. Unified multi-modal latent diffusion for joint subject and text conditional image generation. arXiv preprint arXiv:2303.09319, 2023. 3
- [39] Andrew Melnik, Maksim Miasayedzenkau, Dzianis Makarovets, Dzianis Pirshtuk, Eren Akbulut, Dennis Holzmann, Tarek Renusch, Gustav Reichert, and Helge Ritter. Face generation and editing with stylegan: A survey. arXiv preprint arXiv:2212.09102, 2022. 2
- [40] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023. 2, 3
- [41] Yotam Nitzan, Kfir Aberman, Qiurui He, Orly Liba, Michal Yarom, Yossi Gandelsman, Inbar Mosseri, Yael Pritch, and Daniel Cohen-Or. Mystyle: A personalized generative prior. TOG, 2022. 2, 3, 5, 6, 1
- [42] Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On aliased resizing and surprising subtleties in gan evaluation. In CVPR, 2022. 6
- [43] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 3
- [44] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2, 3, 5, 6, 1
- [45] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 3, 6
- [46] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and

- Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020. 2, 3
- [47] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125,

2022. 3

- [48] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In EMNLP, 2019. 6
- [49] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 3
- [50] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 2, 3, 4, 6, 7, 1
- [51] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. arXiv preprint arXiv:2307.06949, 2023. 2, 3
- [52] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022. 2, 3
- [53] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In CVPR, 2015. 6
- [54] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 3
- [55] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:2210.08402, 2022. 2, 3, 5
- [56] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without testtime finetuning. arXiv preprint arXiv:2304.03411, 2023. 2, 3
- [57] Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. Styledrop: Text-to-image generation in any style. In NeurIPS, 2023. 3
- [58] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 3, 6
- [59] Bochao Wang, Huabin Zheng, Xiaodan Liang, Yimin Chen, Liang Lin, and Meng Yang. Toward characteristicpreserving image-based virtual try-on network. In ECCV,

2018. 2

- [60] Kaisiyuan Wang, Qianyi Wu, Linsen Song, Zhuoqian Yang, Wayne Wu, Chen Qian, Ran He, Yu Qiao, and Chen Change

- Loy. Mead: A large-scale audio-visual dataset for emotional talking-face generation. In ECCV, 2020. 3, 5
- [61] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In ICCV, 2023. 3, 5
- [62] Guangxuan Xiao, Tianwei Yin, William T Freeman, Fr´edo Durand, and Song Han. Fastcomposer: Tuning-free multisubject image generation with localized attention. arXiv preprint arXiv:2305.10431, 2023. 2, 3, 4, 5, 6, 7, 1
- [63] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 3, 6, 7, 1, 2

- [64] Ge Yuan, Xiaodong Cun, Yong Zhang, Maomao Li, Chenyang Qi, Xintao Wang, Ying Shan, and Huicheng Zheng. Inserting anybody in diffusion models via celeb basis. In NeurIPS, 2023. 3
- [65] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2, 3
- [66] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6
- [67] Wenxuan Zhang, Xiaodong Cun, Xuan Wang, Yong Zhang, Xi Shen, Yu Guo, Ying Shan, and Fei Wang. Sadtalker: Learning realistic 3d motion coefficients for stylized audiodriven single image talking face animation. In CVPR, 2023. 2
- [68] Yinglin Zheng, Hao Yang, Ting Zhang, Jianmin Bao, Dongdong Chen, Yangyu Huang, Lu Yuan, Dong Chen, Ming Zeng, and Fang Wen. General facial representation learning in a visual-linguistic manner. In CVPR, 2022. 3, 5

## Appendix

Evaluation IDs

- 1 Alan Turing 14 Kamala Harris

- 2 Albert Einstein 15 Marilyn Monroe

- 3 Anne Hathaway 16 Mark Zuckerberg

- 4 Audrey Hepburn 17 Michelle Obama

- 5 Barack Obama 18 Oprah Winfrey

- 6 Bill Gates 19 Ren´ee Zellweger

- 7 Donald Trump 20 Scarlett Johansson

- 8 Dwayne Johnson 21 Taylor Swift

- 9 Elon Musk 22 Thomas Edison

- 10 Fei-Fei Li 23 Vladimir Putin

- 11 Geoffrey Hinton 24 Woody Allen

- 12 Jeff Bezos 25 Yann LeCun

- 13 Joe Biden

- Table 3. ID names used for evaluation. For each name, we collect four images totally.

### A. Dataset Details

Training dataset. Based on Sec. 3.3 in the main paper, following a sequence of filtering steps, the number of images in our constructed dataset is about 112K. They are classified by about 13,000 ID names. Each image is accompanied by a mask for the corresponding ID and an annotated caption.

Evaluation dataset. The image dataset used for evaluation comprises manually selected additional IDs and a portion of MyStyle [41] data. For each ID name, we have four images that serve as input data for comparative methods and for the final metric evaluation (i.e., DINO [8], CLIP-I [17], and Face Sim. [15]). For single-embedding methods (i.e., FastComposer [62] and IPAdapter [63]), we randomly select one image from each ID group as input. Note that the ID names exist in the training image set, utilized for the training of our method, and the test image set, do not exhibit any overlap. We list ID names for evaluation in Tab. 3. For text prompts used for evaluation, we consider six factors: clothing, accessories, actions, expressions, views, and background, which make up 40 prompts that are listed in the Tab. 4.

### B. User Study

In this section, we conduct a user study to make a more comprehensive comparison. The comparative methods we have selected include DreamBooth [50], FastComposer [62], and IPAdapter [63]. We use SDXL [44] as the base model for both DreamBooth and IPAdapter because of their open-sourced implementations. We display 20 text-

| | |
|---|---|
| | |
| | |
| | |
| | |

Figure 9. User preferences on ID fidelity, generation quality, face diversity, and text fidelity for different methods. For ease of illustration, we visualize the proportion of total votes that each method has received. Our PhotoMaker occupies the most significant proportion in these four dimensions.

image pairs for each user. Each of them includes a reference image of the input ID and corresponding text prompt. We have four randomly generated images of each method for each text-image pair. Each user is requested to answer four questions for these 20 sets of results: 1) Which method is most similar to the input person’s identity? 2) Which method produces the highest quality generated images? 3) Which method generates the most diverse facial area in the images? 4) Which method generates images that best match the input text prompt? We have anonymized the names of all methods and randomized the order of methods in each set of responses. We had a total of 40 candidates participating in our user study, and we received 3,200 valid votes. The results are shown in Fig. 9.

We find that our PhotoMaker has advantages in terms of ID fidelity, generation quality, diversity, and text fidelity, especially the latter three. In addition, we found that DreamBooth is the second-best algorithm in balancing these four evaluation dimensions, which may explain why it was more prevalent than the embedding-based methods in the past. At the same time, IPAdapter shows a significant disadvantage in terms of generated image quality and text consistency, as it focuses more on image embedding during the training phase. FastComposer has a clear shortcoming in the diversity of the facial region for their single-embedding training pipeline. The above results are generally consistent with Tab. 1 in the main paper, except for the discrepancy in the CLIP-T metric. This could be due to a preference for selecting images that harmonize with the objects appearing in the text when manually choosing the most text-compatible images. In contrast, the CLIP-T tends to focus on whether the object appears. This may demonstrate the limitations of the CLIP-T. We also provide more visual samples in Fig. 14-17

Category Prompt

|General|a photo of a <class word>|
|---|---|
|Clothing|a <class word> wearing a Superman outfit<br><br>a <class word> wearing a spacesuit<br><br>a <class word> wearing a red sweater<br><br>a <class word> wearing a purple wizard outfit<br><br>a <class word> wearing a blue hoodie|
|Accessory<br><br>|a <class word> wearing headphones a <class word> with red hair a <class word> wearing headphones with red hair a <class word> wearing a Christmas hat a <class word> wearing sunglasses a <class word> wearing sunglasses and necklace a <class word> wearing a blue cap a <class word> wearing a doctoral cap a <class word> with white hair, wearing glasses|
|Action|a <class word> in a helmet and vest riding a motorcycle<br><br>a <class word> holding a bottle of red wine<br><br>a <class word> driving a bus in the desert<br><br>a <class word> playing basketball<br><br>a <class word> playing the violin<br><br>a <class word> piloting a spaceship<br><br>a <class word> riding a horse<br><br>a <class word> coding in front of a computer<br><br>a <class word> playing the guitar|

(a)

Category Prompt

|Expression<br><br>|a <class word> laughing on the lawn a <class word> frowning at the camera a <class word> happily smiling, looking at the camera a <class word> crying disappointedly, with tears flowing a <class word> wearing sunglasses|
|---|---|
|View|a <class word> playing the guitar in the view of left side<br><br>a <class word> holding a bottle of red wine, upper body<br><br>a <class word> wearing sunglasses and necklace, close-up, in the view of right side<br><br>a <class word> riding a horse, in the view of the top<br><br>a <class word> wearing a doctoral cap, upper body, with the left side of the face facing the camera a <class word> crying disappointedly, with tears flowing, with left side of the face facing the camera<br><br>|
|Background<br><br>|a <class word> sitting in front of the camera, with a beautiful purple sunset at the beach in the background<br><br>a <class word> swimming in the pool a <class word> climbing a mountain a <class word> skiing on the snowy mountain a <class word> in the snow a <class word> in space wearing a spacesuit|

(b)

- Table 4. Evaluation text prompts categorized by (a) general setting, clothing, accessory, action, (b) expression, view, and background. The class word will be replaced with man, woman, boy, etc. For each ID and each prompt, we randomly generated four images for evaluation.

for reference.

### C. More Ablations

Adjusting the ratio during identity mixing. For identity mixing, our method can adjust the merge ratio by either controlling the percentage of identity images within the input image pool or through the method of prompt weighting [3, 21]. In this way, we can control that the person generated with a new ID is either more closely with or far away from a specific input ID. Fig. 10 shows how our method customizes a new ID by controlling the proportion of different IDs in the input image pool. For a better description, we use a total of 10 images as input in this experiment. We can observe a smooth transition of images with the two IDs. This smooth transition encompasses changes in skin color and age. Next, we use four images per generated ID to conduct prompt weighting. The results are shown in Fig. 11. We multiply the embedding corresponding to the images related to a specific ID by a coefficient to control its proportion of integration into the new ID. Compared to the way

to control the number of input images, prompt weighting requires fewer photos to adjust the merge ratio of different IDs, demonstrating its superior usability. Besides, the two ways of adjusting the mixing ratio of different IDs both demonstrate the flexibility of our method.

### D. Stylization Results

Our method not only possesses the capability to generate realistic human photos, but it also allows for stylization while preserving ID attributes. This demonstrates the robust generalizability of the proposed method. We provide the stylization results in Fig. 12.

### E. More Visual Results

Recontextualization. We first provide a more intuitive comparison in Fig. 14. We compare our PhotoMaker with DreamBooth [50], FastComposer [62], and IPAdapater [63], for universal recontextualization cases. Compared to other methods, the results generated by our method can simultaneously satisfy high-quality, strong text controllabil-

[Figure 141]

[Figure 142]

80% Obama 20% Biden

20% Obama 80% Biden

50%

[Figure 143]

A man wearing a red sweater

[Figure 144]

[Figure 145]

80% Obama 20% Scarlett

20% Obama 80% Scarlett

50%

[Figure 146]

A woman happily smiling, looking at the camera

- Figure 10. The impact of the proportion of images with different IDs in the input sample pool on the generation of new IDs. The first row illustrates the transition from Barack Obama to Joe Biden. The second row depicts the shift from Michelle Obama to Scarlett Johansson. To provide a clearer illustration, percentages are used in the figure to denote the proportion of each ID in the input image pool. The total number of images contained in the input pool is 10. (Zoom-in for the best view)

[Figure 147]

[Figure 148]

A man in the snow

[Figure 149]

0.1 0.5 0.8 0.9 1.0 1.1 1.2 1.5 2.0

: 1.0

[Figure 150]

[Figure 151]

A woman wearing sunglasses

[Figure 152]

0.1 0.5 0.8 0.9 1.0 1.1 1.2 1.5 2.0

: 1.0

- Figure 11. The impact of prompt weighting on the generation of new IDs. The first row illustrates a blend of Barack Obama and Joe Biden. The first row from left to right represents the progressive increase in the weight of the ID image embedding corresponding to Barack Obama in the image. The second row illustrates a blend of Elsa (Disney) and Anne Hathaway. The weight for Elsa is gradually increased. (Zoom-in for the best view)

A sketch of a References <class>

A <class> in a comic book

A <class> in Ghibli animation style

A painting of a <class>, in Van Gogh style

A Ukiyo-e painting of a <class>

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

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

- Figure 12. The stylization results of our PhotoMaker with different input IDs and different style prompts. Our method can be seamlessly transferred to a variety of styles, concurrently preventing the generation of realistic results. The symbol <class> denotes it will be replaced by man or woman accordingly. (Zoom-in for the best view)

A photo of Mira Murati A photo of OpenAI CTO

### F. Limitations

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

First, our method only focuses on maintaining the ID information of a single generated person in the image, and cannot control multiple IDs of generated persons in one image simultaneously. Second, our method excels at generating half-length portraits, but is relatively not good at generating full-length portraits. Third, the age transformation ability of our method is not as precise as some GAN-based methods [4]. If the users need more precise control, modifications to the captions of the training dataset may be required. Finally, our method is based on the SDXL and the dataset we constructed, so it will also inherit their biases.

[Figure 199]

[Figure 200]

A photo of chief scientist in OpenAI

A photo of Ilya Sutskever

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

### G. Broader Impact

In this paper, we introduce a novel method capable of generating high-quality human images while maintaining a high degree of similarity to the input identity. At the same time, our method can also satisfy high efficiency, decent facial generation diversity, and good controllability.

- Figure 13. Two examples that are unrecognizable by the SDXL. We replaced two types of text prompts (e.g., name and position) but were unable to prompt the SDXL to generate Mira Murati and Ilya Sutskever.

For the academic community, our method provides a strong baseline for personalized generation. Our data creation pipeline enables more diverse datasets with varied poses, actions, and backgrounds, which can be instrumental in developing more robust and generalizable computer vision models.

ity, and high ID fidelity. We then focus on the IDs that SDXL can not generate itself. We refer to this scenario as the “non-celebrity” case. Compared Fig. 15 with Fig. 13, our method can successfully generate the corresponding input IDs for this setting.

In the realm of practical applications, our technique has the potential to revolutionize industries such as entertainment, where it can be used to create realistic characters for movies or video games without the need for extensive CGI work. It can also be beneficial in virtual reality, providing more immersive and personalized experiences by allowing users to see themselves in different scenarios. It is worth noticing that everyone can rely on our PhotoMaker to quickly customize their own digital portraits.

Bringing person in artwork/old photo into reality. Fig. 16-17 demonstrate the ability of our method to bring past celebrities back to reality. It is worth noticing that our method can generate photo-realistic images from IDs in statues and oil paintings. Achieving this is quite challenging for the other methods we have compared.

However, we acknowledge the ethical considerations that arise with the ability to generate human images with high fidelity. The proliferation of such technology may lead to a surge in the inappropriate use of generated portraits, malicious image tampering, and the spreading of false information. Therefore, we stress the importance of developing and adhering to ethical guidelines and using this technology responsibly. We hope that our contribution will spur further discussion and research into the safe and ethical use of human generation in computer vision.

Changing age or gender. We provide more visual results for changing age or gender in Fig. 18. As mentioned in the main paper, we only need to change the class word when we conduct such an application. In the generated ID images changed in terms of age or gender, our method can well preserve the characteristics in the original ID.

Identity mixing. We provide more visual results for identity mixing application in Fig. 19. Benefiting from our stacked ID embedding, our method can effectively blend the characteristics of different IDs to form a new ID. Subsequently, we can generate text controlled based on this new ID. Additionally, our method provides great flexibility during the identity mixing, as can be seen in Fig. 10-11. More importantly, we have explored in the main paper that existing methods struggle to achieve this application. Conversely, our PhotoMaker opens up a multitude of possibilities.

References

Dreambooth

FastComposer IPAdapter PhotoMaker (Ours)

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

[Figure 217]

[Figure 218]

[Figure 219]

A man in the snow

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

A man crying disappointedly, with tears flowing

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

A photo of a woman

[Figure 246]

[Figure 247]

[Figure 248]

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

A man wearing a doctoral cap

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

A woman climbing a mountain

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

A man in space wearing a spacesuit

- Figure 14. More visual examples for recontextualization setting. Our method not only provides high ID fidelity but also retains text editing capabilities. We randomly sample three images for each prompt. (Zoom-in for the best view)

References

Dreambooth

FastComposer IPAdapter PhotoMaker (Ours)

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

A woman happily smiling, looking at the camera

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

A photo of a woman

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

A man in the snow

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

A man wearing sunglasses

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

A man in a helmet and vest riding a motorcycle

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

A man sitting in front of the camera, with a beautiful purple sunset at the beach in the background

- Figure 15. More visual examples for recontextualization setting. Our method not only provides high ID fidelity but also retains text editing capabilities. We randomly sample three images for each prompt. (Zoom-in for the best view)

References

Dreambooth FastComposer IPAdapter PhotoMaker (Ours)

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

A man wearing a Christmas hat

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

A man riding a horse

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

A man frowning at the camera

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

A man wearing a red sweater

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

A man wearing a blue cap

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

A man wearing sunglasses

- Figure 16. More visual examples for bringing person in old-photo back to life. Our method can generate high-quality images. We randomly sample three images for each prompt. (Zoom-in for the best view)

References

Dreambooth FastComposer IPAdapter PhotoMaker (Ours)

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

A woman wearing a spacesuit

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

A woman wearing sunglasses

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

A man holding a bottle of red wine

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

A man in the snow

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

A photo of a man

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

A man in space wearing a spacesuit

- Figure 17. More visual examples for bringing person in artworks back to life. Our PhotoMaker can generate photo-realistic images while other methods are hard to achieve. We randomly sample three images for each prompt. (Zoom-in for the best view)

References A boy happily smiling A boy wearing a blue hoodie A boy wearing sunglasses

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

A woman with white hair, wearing the sunglasses

A woman wearing a spacesuit

A woman with red hair

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

A boy wearing a red sweater A boy swimming in the pool A boy frowning at the camera

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

A woman wearing a doctoral cap

A woman in the snow A woman happily smiling

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

- Figure 18. More visual examples for changing age or gender for each ID. Our PhotoMaker, when modifying the gender and age of the input ID, effectively retains the characteristics of the face ID and allows for textual manipulation. We randomly sample three images for each prompt. (Zoom-in for the best view)

A man wearing headphones References A man wearing a doctoral cap with red hair

A man wearing a Christmas cap

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

A man wearing headphones with red hair

A man in space wearing a spacesuit

A man in the snow

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

A man wearing sunglasses and necklace

A man happily smiling A man wearing a red sweater

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

A woman wearing a red sweater

A woman wearing a Christmas cap

A woman wearing a blue hoodie

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

A woman wearing headphones with red hair

A woman in wearing a spacesuit

A woman swimming in the pool

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

A woman in wearing a spacesuit

A woman wearing a red sweater

A woman piloting a spaceship

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

- Figure 19. More visual results for identity mixing applications. Our PhotoMaker can maintain the characteristics of both input IDs in the new generated ID image, while providing high-quality and text-compatible generation results. We randomly sample three images for each prompt. (Zoom-in for the best view)

