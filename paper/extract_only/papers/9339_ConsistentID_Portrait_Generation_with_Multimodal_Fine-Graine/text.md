[Figure 1]

## ConsistentID : Portrait Generation with Multimodal Fine-Grained Identity Preserving

Jiehui Huang, Xiao Dong, Wenhui Song, Zheng Chong, Zhenchao Tang, Jun Zhou, Yuhao Cheng, Long Chen, Hanhui Li, Yiqiang Yan, Shengcai Liao, and Xiaodan Liang

### arXiv:2404.16771v2[cs.CV]28Dec2024

Abstract—Diffusion-based technologies have made significant strides, particularly in personalized and customized facial generation. However, existing methods struggle to achieve high-fidelity and detailed identity (ID) consistency. This is mainly due to two challenges: insufficient fine-grained control over specific facial areas and the absence of a comprehensive strategy for ID preservation that accounts for both intricate facial details and the overall facial structure. To address these limitations, we introduce ConsistentID, an innovative method crafted for diverse identity-preserving portrait generation under fine-grained multimodal facial prompts, utilizing only a single reference image. ConsistentID comprises two core components: a multimodal facial prompt generator and an ID-preservation network. The facial prompt generator combines localized facial features, facial feature descriptions, and overall facial descriptions to enhance the precision of facial detail reconstruction. The ID-preservation network, optimized with a facial attention localization strategy, ensures consistent identity preservation across facial regions. Together, these components leverage fine-grained multimodal identity information to improve identity preservation accuracy significantly. To drive ConsistentID’s training, we propose a fine-grained portrait dataset, FGID, with over 500,000 facial images, offering greater diversity and comprehensiveness than existing public facial datasets. Experimental results substantiate that our ConsistentID achieves exceptional precision and diversity in personalized facial generation, surpassing existing methods in the MyStyle dataset. In addition, although ConsistentID introduces more multimodal ID information, it still maintains rapid inference speed during the generation process. Our codes and pre-trained checkpoints are available at https://github.com/JackAILab/ConsistentID.

Index Terms—Portrait generation, fine-grained control, identity preservation.

✦

1 INTRODUCTION

# I

-mage-generation technology [1–5] has undergone significant evolution, driven by the emergence and advancement of

diffusion-based [6, 7] text-to-image large models like GLIDE [8], DALL-E 2 [9], Imagen [10], Stable Diffusion (SD) [11], eDiffI [12] and RAPHAEL [13]. This progress has given rise to a multitude of application approaches across diverse scenarios. Among these, personalized portrait generation has garnered significant attention in both academia and industry due to its wide-ranging applications, including e-commerce advertising, personalized gift customization, and virtual try-ons.

The primary challenge in customized facial generation lies in maintaining facial image consistency across different attributes based on one or multiple reference images, leading to two key issues: ensuring accurate identity (ID) consistency and achieving high-fidelity, diverse facial details. Current text-to-image models [3, 14–17], despite incorporating structural and content

• J.H. Huang, W.H. Song, Z. Chong, Z.C. Tang, J. Zhou, H.H. Li, X.D. Liang are with the School of Artificial Intelligence, Shenzhen Campus, Sun YatSen University, Shenzhen, P.R. China, 518107. X. Dong with the School of Intelligent Systems Engineering, Zhuhai Campus, Sun Yat-Sen University, Zhuhai, P.R. China, 519082. Y.H. Cheng, L. Chen, Y.Q. Yan are with the Lenovo Research Group, Shenzhen, P.R. China, 518038. S.C. Liao is with College of Information Technology, United Arab Emirates University, Al Ain, UAE. E-mail: (jhhuang117@gmail.com, huangjh336@mail2.sysu.edu.cn); (dx.icandoit@gmail.com, dongx55@mails2.sysu.edu.cn); songwh6@mail2.sysu.edu.cn; chongzh@mail2.sysu.edu.cn; tangzhch7@mail2.sysu.edu.cn; zhouj235@mail2.sysu.edu.cn; chengyh5@Lenovo.com; chenlong12@lenovo.com; lihanhui@mail3.sysu.edu.cn; yanyq@lenovo.com; scliao@ieee.org; xdliang328@gmail.com

Xiaodan Liang is the Corresponding Author.

guidance, face limitations in accurately controlling personalized and customized generation, particularly concerning the fidelity of generated images to reference images.

To improve the precision and diversity of personalized portrait generation with reference images, numerous customized methodologies have emerged, meeting users’ demands for highquality customized images. These personalized approaches are categorized based on whether fine-tuning occurs during inference, resulting in two distinct types: test-time fine-tuning and direct inference. Test-time fine-tuning: This category includes methods such as Textual Inversion [18], HyperDreambooth[3], and CustomDiffusion[19]. Users can achieve personalized generation by providing a set of target ID images for post-training. Despite achieving commendable high-fidelity results, the quality of the generated output depends on the quality of manually collected data. Additionally, the manual collection of customized data for fine-tuning introduces a labor-intensive and time-consuming aspect, limiting its practicality. Direct inference: Another category of models, including IP-Adapter [15], Fastcomposer[20], Photomaker [21], and InstantID [22], adopts a single-stage inference approach. These models enhance global ID consistency by either utilizing the image as a conditional input or manipulating image-trigger words. However, most methods frequently overlook fine-grained information, such as landmarks and facial features. Although InstantID [22] enhances ID consistency to some extent by incorporating landmarks, the use of visual prompt landmarks limits the diversity and flexibility of key facial regions, resulting in rigidly generated facial features. Building on these observations, it becomes evident that two pivotal challenges persist in personalized portrait generation, requiring meticulous

consideration: 1) neglect of fine-grained facial information and 2) identity inconsistency between facial areas and the whole face, as illustrated in Figure 9.

To tackle identity consistency and detail preservation challenges in personalized image generation, we propose ConsistentID, a novel method that excels in maintaining identity fidelity while capturing diverse facial details. With just a single facial image as input, our approach leverages multimodal fine-grained ID features to deliver high-quality results. As illustrated in Figure 1, ConsistentID enables personalized transformations such as revitalizing old photos, altering gender, or changing outfits. By seamlessly integrating facial inputs with text prompts, our model generates diverse yet consistent identity representations, offering unmatched adaptability in personalized generation tasks.

Figure 2 provides an overview of our ConsistentID. ConsistentID comprises two key modules: 1) a multimodal facial prompt generator and 2) an ID-preservation network. The former component includes a fine-grained multimodal feature extractor and a facial ID feature extractor, enabling the generation of more detailed facial ID features using multi-conditions, incorporating facial images, facial regions, and corresponding textual descriptions extracted from the multimodal large language model LLaVA1.5 [23]. The facial ID features generated by the first module are then passed to the latter module, which enhances ID consistency across facial regions through a facial attention localization strategy. Additionally, we recognize the limitations of existing portrait datasets [24–28], particularly in capturing diverse and fine-grained facial details that preserve identity, crucial to the effectiveness of ConsistentID. To address this, we introduce the inaugural Fine-Grained ID Preservation (FGID) dataset, along with a fine-grained identity consistency metric, providing a unique and comprehensive evaluation approach to enhance our training and provide a comprehensive evaluation of its performance in capturing facial details.

In summary, our contributions are as follows.

- • We introduce ConsistentID to improve fine-grained customized facial generation by incorporating detailed descriptions of facial regions and local facial features. The experimental results demonstrate the superiority of ConsistentID in terms of ID consistency and high fidelity, even with only one reference image. Additionally, although ConsistentID introduced more detailed multimodal finegrained ID information during training, it achieves this with a single fixed prompt 1 during inference. It does not depend on facial descriptions generated by LLaVA1.5, ensuring a streamlined and efficient approach, as shown in Table 3.
- • We devise an ID-preservation network optimized by facial attention localization strategy, enabling more accurate ID preservation and more vivid facial generation. This mechanism ensures the preservation of ID consistency within each facial region by preventing the blending of ID information from different facial regions.
- • We introduce the inaugural fine-grained facial generation dataset, FGID, addressing limitations in existing datasets for capturing diverse identity-preserving facial details. This dataset includes facial features and descriptions of both facial regions and the entire face, complemented by a novel fine-grained identity consistency metric, establishing

1. This person has one nose, two eyes, two ears, and a mouth.

a comprehensive evaluation framework for fine-grained facial generation performance.

#### 2 RELATED WORK

Text-to-image Diffusion Models. Diffusion models have made notable advancements, garnering significant attention from both industry and academia, primarily due to their exceptional semantic precision and high fidelity. The success of these models can be attributed to the utilization of high-quality image-text datasets, continual refinement of foundation modality encoders, and the iterative enhancement of controlled modules. In the domain of text-to-image generation, the encoding of text prompts involves utilizing a pretrained language encoder, such as CLIP [29], to transform it into a latent representation, subsequently inserted into the diffusion model through the cross-attention mechanism. Pioneering models in this domain include GLIDE [8], SD [11], DiT [30], among others, and further developments and innovations are continuing to emerge [13, 31]. A notable advancement in this lineage is SDXL [32], which stands out as the most powerful text-to-image generation model. It incorporates a larger Unet [33] model and employs two text encoders for enhanced semantic control and refinement. As a follow-up, we use the SD [11] model as our base model to achieve personalized portrait generation.

Personalization in Diffusion Models. Due to the potent generative capability of the text-to-image diffusion model, many personalized generation models are constructed based on it. The mainstream personalized image synthesis methods are categorized into two groups based on whether fine-tuning occurs during test time. One group relies on optimization during test-time, with typical methods including Dreambooth [34], Textual Inversion [18], IP-Adapter [22], ControlNet [14], Custom Diffusion [19], and LoRA [35]. Dreambooth and Textual Inversion fine-tune a special token S* to learn the concept during the fine-tuning stage. In contrast, IP-Adapter, ControlNet, and LoRA insert image semantics using an additional learned module, such as cross-attention, to imbue a pre-trained model with visual reasoning understanding ability.

Despite their advancements, these methods necessitate resource-intensive backpropagation during each iteration, making the learning process time-consuming and limiting their practicality. Recently, researchers have focused more on methods bypassing additional fine-tuning or inversion processes, mainly including IP-Adapter [15], FastComposer [20], PhotoMaker [21], and InstantID [22]. This type of method performs personalized generation using only an image with a single forward process, which is more advantageous in calculation efficiency compared to the former type. However, we observe that fine-grained facial features are not fully considered in the training process, easily leading to ID inconsistency or lower image quality, as shown in Figure 7. To address these limitations, we introduce ConsistentID, aiming to mitigate the ID-preserving issue and enhance finegrained control capabilities while reducing data dependency. Our approach incorporates a specially designed facial encoder using detailed descriptions of facial features and local image conditions as inputs. Additionally, we contribute to a new landmark in the facial generation field by proposing: 1) the introduction of the first fine-grained facial generation datasets and 2) the presentation of a new metric that redefines the performance evaluation of facial generation.

A woman in a wedding dress

[Figure 2]

A man cooking a meal

A woman wearing a cap

[Figure 3]

Mars Style

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

A street art stencil of a person

A person wearing a spacesuit

A boy wearing sunglasses

A man firefighter with helmet

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

(a) old photos of people to life -changing age & gender

(b) attributes change

- Fig. 1: Given some images of input IDs, our ConsistentID can generate diverse personalized ID images based on text prompts using only a single image.

#### 3 METHOD

In this section, we will introduce our multimodal facial prompt generator in Subsection 3.1 and propose a meticulously designed ID-preservation network in Subsection 3.2. Next, we detail the training process of ConsistentID and its inference procedure in Subsection 3.3. Finally, we provide a comprehensive description of the FGID dataset in Subsection 3.4.

##### 3.1 Multimodal Facial Prompt Generator

Fine-grained Multimodal Feature Extractor. In this module, we independently learn fine-grained facial visual and textual embeddings and feed them into the designed lightweight facial encoder to generate fine-grained multi-modal facial features. Three key components are used in the module, including text embedding, facial embedding and facial encoder.

- 1) Text Embedding. Motivated by recent works [2, 36–38] in personalized facial generation, our goal is to introduce more detailed and accurate facial descriptions. To achieve this, the entire facial image is processed by the Multimodal Large Language Model (MLLM) LLaVA1.5 [23] using the prompt: ‘Describe this person’s facial features, including the face, ears, eyes, nose, and mouth’. The mentioned process will generate feature-level descriptions of the facial areas. Next, the terms ‘face’, ‘ears’, ‘eyes’, ‘nose’, and ‘mouth’ in these descriptions are replaced with the delimiter ‘<facial>’, and the modified text is concatenated with captions describing the entire facial image. Finally, the concatenated descriptions are input into a pre-trained text encoder to learn the fine-grained text embedding of the facial features. With the abundant descriptions from facial regions, the text embeddings are enriched with more precise identity information, effectively resolving identity inconsistency.
- 2) Facial Embedding. In contrast to existing methods [39, 40], [18, 20–22, 39–43] that rely on brief textual descriptions or coarsegrained visual prompts, our goal is to integrate more fine-grained multimodal control information at the facial region level, aiming to achieve greater accuracy in capturing detailed facial features. To enrich the ID-preservation information, we delve into more finegrained facial features, including eye gaze, earlobe characteristics,

nose shape, and others. Following the previous method [44–49], we employ the pre-trained face model BiSeNet [50] to extract segmentation masks of facial areas, encompassing eyes, nose, ears, mouth, and other regions, from the entire face. Subsequently, the facial regions obtained from these masks are fed into the pretrained image encoder to learn fine-grained facial embeddings. The inclusion of facial regions’ features results in fine-grained facial embeddings containing more abundant ID-preservation information compared to features learned from the entire face.

3) Facial Encoder. Previous studies [18, 20, 29, 34, 50] have demonstrated that relying solely on visual or textual prompts cannot comprehensively maintain ID consistency both in appearance and semantic details. While IP-Adapter [15] makes the initial attempt to simultaneously inject multimodal information through two distinct decoupled cross-attention mechanisms, it overlooks ID information from crucial facial regions, rendering it susceptible to ID inconsistency in facial details.

To cultivate the potential of image and text prompts, inspired by the token fusion approach of multimodal large language models, we design a facial encoder to seamlessly integrate visual prompts with text prompts along the dimension of the text sequence, as depicted in Figure 3. Specifically, given a facial embedding and a caption embedding, the facial encoder initially employs a self-attention mechanism to align the entire facial features with facial areas’ features, resulting in aligned features denoted as fi ∈ RN×D, where N = 5 represents the number of facial feature areas, including eyes, mouth, ears, nose, and other facial regions, and D represents the dimension of text embeddings. In cases where face images lack a complete set of N facial features, the missing features are padded using an allzero matrix. Subsequently, we replace the text features at the position of the delimiter ‘<facial>’ with fi using the visual token replacement operation, as illustrated in Figure 3 (right). Finally, the text features, now enriched with visual identity information, are fed into two multi-layer perceptions (MLP) to learn the text conditional embeddings.

Facial ID Feature Extractor. Except for the input condition of fine-grained facial features, we also inject the character’s

|Fine-grainedMultimodal<br><br>FacialFeature<br><br>Facial Encoder<br><br>[Figure 22]<br><br>|[Figure 23]|
|---|
<br><br>Facial Embedding<br><br>[Figure 24]<br><br>| |A young woman wearing a white sweater and a scarf|
|---|---|
| | |
<br><br>concatenate<br><br>A. Fine-grained Multimodal Feature Extractor<br><br>❄ BiSeNet<br><br>[Figure 25]<br><br>❄ LLaVA1.5<br><br>[Figure 26]<br><br>Text Embedding<br><br>Detailed Facial Descriptions<br><br>Facial Caption<br><br>Facial Image<br><br>Text Encoder<br><br>[Figure 27]<br><br>❄<br><br>Image Encoder<br><br>[Figure 28]<br><br>❄<br><br>|
|---|

||Overall Facial ID Feature Extractor<br><br>Fine-grained Multimodal Feature Extractor<br><br>Multimodal Facial Prompt Generator<br><br>A . B .|
|---|
<br><br>[Figure 29]<br><br>Cross Attention<br><br>Cross Attention<br><br>Enhanced Text Image<br><br>[Figure 30]<br><br>Trainable Modules Frozen Modules<br><br>Denoising U-Net<br><br>[Figure 31]<br><br>❄ ID-Preservation network<br><br>[Figure 32]<br><br>[Figure 33]<br><br>❄<br><br>ℒ = ℒ + ℒloc<br><br>[Figure 34]<br><br>[Figure 35]|
|---|

Fine-grainedMultimodal

FacialFeature

|InsightFace Model<br><br>Projection<br><br>|[Figure 36]|
|---|
<br><br>FacialID<br><br>Feature<br><br>B. Overall Facial ID<br><br>Feature Extractor Training Inference Shared<br><br>[Figure 37]<br><br>Image Encoder<br><br>Facial Image<br><br>[Figure 38]<br><br>❄<br><br>[Figure 39]<br><br>❄|
|---|

Projection

FacialID

Feature

- Fig. 2: The overall framework of our proposed ConsistentID. The framework comprises two key modules: a multimodal facial ID generator and a purposefully crafted ID-preservation network. The multimodal facial prompt generator consists of two essential components: a fine-grained multimodal feature extractor, which focuses on capturing detailed facial information, and a facial ID feature extractor dedicated to learning facial ID features. On the other hand, the ID-preservation network utilizes both facial textual and visual prompts, preventing the blending of ID information from different facial regions through the facial attention localization strategy. This approach ensures the preservation of ID consistency in the facial regions.

overall ID information into our ConsistentID as a visual prompt. This process relies on the pre-trained CLIP image encoder and the pre-trained face model from the specialized version of the IP-Adapter [15] model, IPA-FaceID-Plus [15]. Specifically, the complete facial images are simultaneously fed into both encoders for visual feature extraction. Following these two encoders, a lightweight projection module, with parameters initialized by IPAFaceID-Plus, is used to generate the face embedding of the whole image.

each layer, where P[i,j,k] signifies the attention map from the kth conditional token to the (i, j) latent pixel. Ideally, the attention maps of facial region tokens should focus exclusively on facial feature areas, preventing the blending of identities between facial features and averting propagation to the entire facial image. To achieve this goal, we propose localizing the focus of the crossattention map by using segmentation masks that correspond to the reference facial regional features.

Let M = {m1,m2,m3,...,mN} represent the segmentation masks of the reference facial regions, I = {i1,i2,i3,...,iN} as the index list indicating which facial feature corresponds to visual and textual tokens in the multimodal prompt, and Pi = P[:,:,i] ∈ [0,1]h×w denote the cross-attention map of the i-th facial region’s token, where I is generated using the special token ‘<facial>’.

##### 3.2 ID-Preservation Network

The effectiveness of visual prompts in pre-trained text-to-image diffusion models [14, 20, 21, 51–53] significantly enhances textual prompts, especially for content that is challenging to describe textually. However, visual prompts alone often provide only coarsegrained control due to the semantic fuzziness of visual tokens. To solve this, we integrate fine-grained multimodal ID prompts and overall ID prompts into the UNet model through the crossattention module to achieve precise ID preservation.

j, it should closely correspond to the facial region identified by the j-th multimodal token, segmented by mj. To achieve this, we introduce mj and apply it to Pi

Given the cross-attention map Pi

j to obtain its corresponding activation region, aligning with the segmentation mask mj of the j-th facial feature token. For achieving this correspondence, a balanced L1 loss is employed to minimize the distance between the cross-attention map and the segmentation mask:

Specifically, we first introduced an ID consistency network, which maintains the consistency of local ID features by guiding the attention of facial features to align with the corresponding facial regions [20].

This optimization strategy is derived from the observation that traditional cross-attention maps tend to simultaneously focus on the entire image rather than specific facial regions, making it challenging to maintain ID features during each facial region generation. To address this issue, we introduce facial segmentation masks during training to obtain attention scores learned from the enhanced text cross-attention module for facial regions.

N

1 N

[1 − mj] − mean Pi

Lfacial =

mean Pi

[mj] ,

j

j

j=1

(1) where N denotes the number of the segmentation masks. This loss formulation aims to ensure that each facial feature token’s attention map aligns closely with its corresponding segmentation mask, promoting precise and localized attention during the generation process.

Let P ∈ [0,1]h×w×n represent the cross-attention map that connects latent pixels to multimodal conditional embeddings at

|[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>K<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>Attention<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>Fuse<br><br>[Figure 57]<br><br>Q<br><br>V<br><br>Attention<br><br>[Figure 58]<br><br>FFN<br><br>FFN<br><br>Updated Facial Embedding<br><br>Text Embedding<br><br>|…| |
|---|---|
| | |
<br><br>|…| |
|---|---|
| | |
<br><br>Learnable Queries<br><br>Facial Embedding<br><br>(L, 1024)<br><br>(N, 1024)<br><br>|…|
|---|
<br><br>|…|
|---|
<br><br>(N, 768)<br><br>(N, 768)<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]|
|---|

||[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>: Visual Token Replace|
|---|
<br><br>eye ear mouth nose others<br><br>[Facial] [Facial] … [Facial]<br><br>Updated Facial Embedding<br><br>Text Embedding<br><br>Replace:|
|---|

Attention

Attention

Fig. 3: The framework of our facial encoder for generating fine-grained multimodal facial features.

Dataset Pipeline: As for the selected images, each image is first processed through BiSeNet [50] to generate a fine-grained binary mask (Imask) that identifies predefined facial components. Using Imask, the corresponding facial regions are segmented from the original image, denoted as Iface. Meanwhile, the InsightFace [56] model is used to extract comprehensive facial identity features, ensuring both global and regional identity information is captured effectively.

##### 3.3 Training and Inference Details

The training data for ConsistentID consists of facial image-text pairs. In ConsistentID, to enhance text controllability, we prioritize the caption as the primary prompt and concatenate it with more detailed descriptions of facial regions extracted from LLaVA1.5, forming the ultimate textual input. During training, only the parameters of the facial encoder and projection module within the facial ID feature extractor are optimized, while the pre-trained diffusion model remains frozen. Regarding training loss functions, they align with those used in the original stable diffusion models and are expressed as:

For textual data, the LLaVA1.5 model generates detailed descriptions using the embedded prompt: ‘Please describe the people in the image, including their gender, age, clothing, facial expressions, and any other distinguishing features.’. This ensures fine-grained textual annotations are paired with the visual data.

Lnoise = Ezt,t,Cf,Cl,ϵ∼N(0,1) ∥ϵ − ϵθ (zt,t,Cf,Ci)∥22 , (2)

where Cl denotes the facial ID feature, and Cf is the fine-grained multimodal facial feature.

- 3.4.2 Dataset Characteristics The FGID dataset is designed to provide comprehensive textual and visual information for training models like ConsistentID. Below, we summarize its key features and characteristics: Dataset Composition: The FGID dataset includes 15 distinct identity types, as detailed in Table 1. To ensure diversity and inclusiveness, the distributions of gender and age across all identities are presented in Figure 4, demonstrating a relatively balanced representation of these properties. Dataset Scenarios: The dataset spans 45 scenarios, categorized into four application areas: Clothing & Accessory, Action, Background, and Style. Table 2 lists the prompts used to define these scenarios, ensuring a wide range of contextual diversity. Rich Data Representation: The FGID dataset is tailored to capture both whole-face and specific facial feature information, providing fine-grained textual and visual details essential for model training. This design enhances its utility for generating accurate and diverse facial representations. Future Directions: To further augment diversity and information content, we plan to expand the dataset by incorporating additional resources, such as higher-level datasets like LAION-Face [24] and self-collected multi-ID data.

4 EXPERIMENTS

- 4.1 Implementation Details

The total loss function is Ltotal = Lnoise + λLfacial. During the inference process, we employ a straightforward delayed primacy condition as similar to Fastcomposer. This allows the use of a separate text representation initially, followed by enhanced text representation after a specific step, effectively balancing identity preservation and editability.

##### 3.4 FGID Dataset

Our ConsistentID necessitates detailed facial features and corresponding textual prompts to address issues like deformation, distortion, and blurring prevalent in current facial generation methods. However, existing datasets [24, 40, 54, 55] predominantly focus on local facial areas and lack fine-grained ID annotations for specific features such as the nose, mouth, eyes, and ears.

To address this limitation, we introduce the FGID dataset, which provides comprehensive fine-grained ID information and detailed facial descriptions essential for training the ConsistentID model. In the following, we will describe the dataset curation process and key characteristics of this dataset.

- 3.4.1 Dataset Curation Data source: Our facial images in the FGID dataset are from three public datasets, including FFHQ [40], CelebA [26], and SFHQ [54]. We separately select 70,000, 30,000, and 424,258 images from these datasets. Finally, a total of 524,258 images are selected, where 107,048 images have recognizable IDs. Figure 6 shows some examples of these images and their corresponding captions.

Experimental Implements. In ConsistentID, we employ the Stable Diffusion V1.5 (SD1.5) as the foundational text-to-image training model. For the fine-grained multimodal feature extractor,

Evaluation IDs

- 1. Andrew Ng 6. Scarlett Johansson 11. Joe Biden
- 2. Barack Obama 7. Taylor Swift 12. Kamala Harris
- 3. Dwayne Johnson 8 Albert Einstein 13. Kaming He
- 4. Fei-Fei Li 9. Elon Mask 14. Lecun Yann
- 5. Michelle Obama 10. Geoffrey Hinton 15. Sam Altman TABLE 1: ID names used for evaluation.

|Category<br><br>|Prompt|
|---|---|
|Clothing& Accessory|a <class word> wearing a red hat<br><br>a <class word> wearing a santa hat<br><br>a <class word> wearing a rainbow scarf<br><br>a <class word> wearing a black top hat and a monocle<br><br>a <class word> in a chef outfit<br><br>a <class word> in a firefighter outfit<br><br>a <class word> in a police outfit<br><br>a <class word> wearing pink glasses<br><br>a <class word> wearing a yellow shirt<br><br>a <class word> in a purple wizard outfit<br><br>|
|Background<br><br>|a <class word> in the jungle a <class word> in the snow a <class word> on the beach a <class word> on a cobblestone street a <class word> on top of pink fabric a <class word> on top of a wooden floor a <class word> with a city in the background a <class word> with a mountain in the background a <class word> with a blue house in the background a <class word> on top of a purple rug in a forest|
|Action|a <class word> holding a glass of wine<br><br>a <class word> riding a horse<br><br>a <class word> holding a piece of cake<br><br>a <class word> giving a lecture<br><br>a <class word> reading a book<br><br>a <class word> gardening in the backyard<br><br>a <class word> cooking a meal<br><br>a <class word> working out at the gym<br><br>a <class word> walking the dog<br><br>a <class word> baking cookies<br><br>a <class word> wearing a doctoral cap<br><br>a <class word> wearing a spacesuit<br><br>a <class word> wearing sunglasses and necklace<br><br>a <class word> coding in front of a computer<br><br>a <class word> in a helmet and vest riding a motorcycle<br><br>|
|Style<br><br>|a painting of a <class word> in the style of Banksy a painting of a <class word> in the style of Vincent Van Gogh a colorful graffiti painting of a <class word> a watercolor painting of a <class word> a Greek marble sculpture of a <class word> a street art mural of a <class word> a black and white photograph of a <class word> a pointillism painting of a <class word> a Japanese woodblock print of a <class word> a street art stencil of a <class word>|

- TABLE 2: Evaluation text prompts are categorized by Clothing&Accessories, Background, Action, and Style. During inference, the term ‘class’ will be substituted with ‘man’, ‘woman‘, ‘girl‘, etc. For each identity, we use these prompts to generate 45 images for evaluation.

we initialize the parameters of all text encoders and image encoders with CLIP-ViT-H [57]. Additionally, we use the image projection layers from CLIP-ViT-H to initialize the learnable projection module within the overall facial ID feature extractor. The entire framework is optimized using Adam [58] on 8 NVIDIA 3090 GPUs, with a batch size of 16. We set the learning rate for all trainable modules to 1 × 10−4. During training, we probabilistically remove 50% of the background information from the characters with a 50% probability to mitigate interference. The hyperparameter coefficient (λ) of facial features loss Lfacial is set to 0.01. Additionally, to enhance generation performance through

Gender

55.5% 44.5% 0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%

Female male

Age

22.6% 37.6% 20.3% 10.7% 5.8%2%

0% 10% 20% 30% 40% 50% 60% 70% 80% 90% 100%

0-19 20-29 30-39 40-49 50-59 60+

- Fig. 4: The statistical characteristics of age and gender distribution in the FGID training dataset.

[Figure 108]

|12.6%<br>13.1% 20.8%<br><br><br>8.7%<br><br>50.0%<br><br>8.3%<br><br>21.7%<br><br>16.7%<br><br>58.3%<br><br>56.5%<br><br>33.3%<br><br>0% 20% 40% 60% 80% 100%<br><br>ID Fidelity<br><br>Facial Fidelity<br><br>Realistic level<br><br>FastComposer Photomaker InstantID ConsistentID<br><br>|
|---|

- Fig. 5: User preferences across image fidelity, fine-grained ID fidelity, overall ID fidelity for different methods.

Original Segmentation Caption

|The woman in the image has a beautiful and well-defined facial structure. She has a prominent nose, which is slightly pointed, and her eyes are large and wide-set. Her ears are small and delicate, and her mouth is slightly open, giving her a smile. The combination of these features creates a captivating and attractive appearance.|
|---|

[Figure 113]

[Figure 114]

|The woman in the image has a beautiful and smiling face. She has a prominent nose, which is located in the center of her face. Her eyes are wide open, and her mouth is slightly open, showing her teeth. Her skin appears to be smooth , but with a few frekles on it. Her hair is long and dark, covering both of her ears, making them invisible in this image.|
|---|

[Figure 115]

[Figure 116]

|The man in the image has a smiling face with a prominent smile. The man has a prominent nose, which is located in the center of his face. His eyes are blue, and they are positioned close together. He has a small mouth, which is slightly open, and his teeth are visible as he smiles. The man's ears are relatively small, and his cheeks are square.|
|---|

[Figure 117]

[Figure 118]

|The man in the image has a prominent nose, which is a noticeable feature of his face. He also has a thin mustache, adding to his facial hair. His eyes are open, and he is smiling, which gives him a friendly and approachable appearance. The man has a black shirt on, and his ears are visible, completing his overall facial features.|
|---|

[Figure 119]

[Figure 120]

Fig. 6: Several training data demos from our FGID dataset.

classifier guidance, there is a 10% chance of replacing the original updated text embedding with a zero text embedding. During inference, we use delayed topic conditioning [20, 40] to resolve text and ID condition conflicts. We generate portrait images using a 50-step DDIM sampler [6] with a classifier guidance scale set to 5.

Experimental Metrics. To evaluate the effectiveness and efficiency of ConsistentID, we employ six widely used metrics [34]: CLIP-I [18], CLIP-T [29], DINO [59], FaceSim [60], FID [61], and inference speed. CLIP-T measures the average cosine similarity between prompt and image CLIP embeddings, evaluating ID fidelity. CLIP-I calculates the average pairwise cosine similarity between CLIP embeddings of generated and real images, assessing prompt fidelity. DINO represents the average cosine similarity between ViT-S/16 [62] embeddings of generated and real images, indicating fine-grained image-level ID quality. FaceSim determines facial similarity between generated and real images using FaceNet [60]. FID gauges the quality of the generated images [61].

Reference FastComposer IP-Adapter Photomaker InstantID ConsistentID

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

awomanwearing

ayellowshirt

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

purplewizardoutfit amanina

awomanina

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

cobblestonestreet amaninthesnow

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

firefighteroutfit awomanona

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

- Fig. 7: Qualitative comparison of universal recontextualization samples is conducted, comparing our approach with other methods using five distinct identities and their corresponding prompts. Our ConsistentID exhibits a more powerful capability in high-quality generation, flexible editability, and strong identity fidelity.

Inference speed denotes the calculation time under the same running environment. To fully evaluate and quantify quality at the facial region level, we propose a novel metric, FGIS (FineGrained Identity Similarity). FGIS is computed as the average cosine similarity between DINO embeddings of the generated facial regions in reference and generated images. A higher FGIS value indicates greater ID fidelity in the generated facial regions.

##### 4.2 Comparation Results

To demonstrate the effectiveness of ConsistentID, we conduct a comparative analysis against state-of-the-art methods, including Fastcomposer [20], IP-Adapter[15], Photomaker [21], and InstantID [22]. During testing, we utilized the officially provided models and generate portrait images using only a single reference image. Consistent with the evaluation protocol of Photomaker, we utilize the Mystyle [27] dataset for quantitative assessment and incorporate over ten identity datasets for qualitative visualization.

Quantitative results: Following Photomaker [21], we used the test dataset from Mystyle [27]. Notably, during the inference process, we did not use LLaVA to generate the descriptions of each facial region but obtained facial feature information using a fixed phrase: ‘This person has one nose, two eyes, two ears, and a mouth.’ as the text prompt. The quantitative comparison is conducted under the universal recontextualization setting, utilizing a set of metrics to benchmark various aspects.

The results are showcased in Table 3. A thorough analysis of the table demonstrates that ConsistentID consistently outperforms other methods across most evaluated metrics, and surpasses other IP-Adapter-based methods in terms of generation efficiency. This is attributed to ConsistentID’s fine-grained ID preservation capability and the efficiency of the lightweight multimodal facial prompt generator. Regarding the FID metric, the relatively lower performance may be influenced by the inherent generative limitations of the base model SD1.5.

|CLIP-T ↑ CLIP-I ↑ DINO ↑ FaceSim ↑ FGIS ↑ FID ↓|Speed (s)<br><br>|
|---|---|
|Fastcomposer [20] 27.8 67.0 68.4 75.2 77.7 372.8 IP-Adapter [15] 27.6 75.0 74.5 75.6 73.4 320.0 Photomaker [21] 30.7 71.7 72.6 69.3 73.2 336.5<br><br>InstantID [22] 30.3 68.2 77.6 76.5 78.3 271.9 ConsistentID 31.1 76.7 78.5 77.2 81.4 312.4<br><br>|10 13 17 19 16|

- TABLE 3: Quantitative comparison of the universal recontextualization setting on the MyStyle test dataset. The benchmark metrics assessed text consistency (CLIP-T), the preservation of coarse- and fine-grained ID information (CLIP-I, DINO, FaceSIM, and FGIS), generation quality (FID), and inference efficiency (speed in seconds).

FastComposer IPAdapter Photomaker InstantID ConsistentID

ID

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

cinematicrealityqueen

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

smile

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

- Fig. 8: Qualitative comparison of our model with other models on the task of decoupling ID and text instructions. Notably, our method can control both the expression and style of the ID, and also demonstrates excellent results in maintaining the identity of anime characters.

Visualized comparisons under different scenarios: To demonstrate the advantages of ConsistentID visually, we present the text-edited generation results of all methods, using reference images from five distinct identities, in Figure 7. This visualization highlights ConsistentID’s capability to produce vibrant and realistic images, with a particular emphasis on facial features. To further elucidate this observation, we selectively magnify and compare specific facial details across all methods in four identities, as depicted in Figure 9. Our model showcases exceptional ID preservation capabilities in facial details, especially in the eyes and nose, attributed to fine-grained multimodal prompts and facial regions’ ID information.

To validate the capability of accurate text understanding, we additionally present style-based and action-based text-edited results in Figure 8. It can be observed that the images generated by InstantID exhibit limited flexibility in facial poses. This limitation is likely attributed to Controlnet-based prompt insertion methods, which may easily overlook textual prompts. Simultaneously, we notice that, while Photomaker can accurately comprehend textual and visual prompts, it lacks the ID consistency of facial regions. In contrast, our ConsistentID achieves optimal generation results due to its precise understanding of textual and visual prompts. This further emphasizes the importances of multimodal fine-grained ID information. To fully show the advantages of our ConsistentID, more visualized comparisons are provided in following section, including comparative experiments with fine-tuning-based models (Facechain [63], and Easyphoto [64]), and the MLLM model GPT4o.

##### 4.3 Compare with more models

To further illustrate the advantages of our model, we compare ConsistentID with two LoRA-based fine-tuning methods, Facechain [63] and Easyphoto [64], as well as the closed-source model GPT4o 2. To ensure a fair comparison, all models are provided with a single reference image as input. For the LoRAbased methods, the input comprises image + prompt, where image represents the reference image and prompt specifies the editing instruction for generating personalized outputs. For the GPT4o model, the template used is: ‘Edit this image using the following instruction: person_name + prompt’, where person_name specifies the individual’s name. In Figure 10, we show the visualized results of the personalization comparison using a single image, demonstrating the superior capability of our ConsistentID in keeping ID consistency.

In addition, we compare ConsistentID with models specifically designed using IP-Adapter as the base model, as shown in Figure 12. The figure clearly demonstrates that these models struggle to achieve detailed ID preservation in facial regions, primarily due to the lack of fine-grained textual and visual prompts. In contrast, ConsistentID exhibits a robust capability to preserve the integrity of facial ID and seamlessly blend it into various styles, utilizing multimodal fine-grained prompts. This highlights ConsistentID’s advantage in retaining identity while providing enhanced flexibility and control over style customization.

##### 4.4 Human Study

To accurately capture user preferences, we conducted surveys to evaluate perceptions of image fidelity, fine-grained ID fidelity, and overall ID fidelity. Figure 5 visualizes the proportion of total votes received by each method. Across all three metric dimensions, ConsistentID achieves the highest user preference share. This result highlights the effectiveness of ConsistentID in meeting user expectations for both identity preservation and image quality.

##### 4.5 Ablation Study

Facial ID Types: We conducted an ablation study on facial IDs, considering three variations: using only the holistic facial ID, using our designed fine-grained IDs, and using both the holistic facial ID and our designed fine-grained IDs. The results in Table 4 indicate that using fine-grained ID features alone effectively maintains ID consistency in the facial region, particularly evidenced by an increase in DINO values. In contrast, using only overall facial features (FaceID) results in decreased values across all metrics, due to the coarse-grained characteristic of the overall facial feature. Finally, the combination of holistic facial ID and fine-grained ID features achieves the best overall ID fidelity, demonstrating the complementary strengths of both approaches and their contribution to enhanced model performance.

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Origin FastComposer ConsistentID Origin InstantID ConsistentID

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

Origin IP-Adapter ConsistentID Origin Photomaker ConsistentID

- Fig. 9: Comparison of facial feature details between our method and existing approaches. Notably, the characters generated by our method exhibit superior ID consistency in facial features such as eyes, nose, and mouth.

|Lnoise Lfacial CLIP-I ↑ DINO ↑<br><br>|FGIS ↑<br><br>|LLaVA1.5 CLIP-I ↑ DINO ↑<br><br>|FGIS ↑|
|---|---|---|---|
|✓ − 66.4 77.8 ✓ ✓ 75.5 86.1|82.9 85.6<br><br>|− 67.4 84.3 ✓ 75.3 85.4|84.9 85.8<br><br>|

|Facial Feature CLIP−I ↑ DINO ↑<br><br>|FGIS ↑|ImageProjection CLIP−I ↑ DINO ↑<br><br>|FGIS ↑|
|---|---|---|---|
|Overall Facial Feature 72.9 80.7 Fine-grained Feature 75.2 86.6 Overall Facial & Fine-grained Feature 75.5 86.1|84.2 85.4 85.6<br><br>|− 61.0 75.6 ✓ 75.5 86.1<br><br>|82.9 85.6|

TABLE 4: Ablation study on ID features, loss functions, ImageProjection module, and the usage of LLaVA1.5 in training.

[Figure 188]

[Figure 189]

[Figure 190]

Index

CLIP-I CLIP-T FGIS

Fig. 16: Performance variations of CLIP-I, CLIP-T, and FGIS metrics with increasing ‘merge step’.

Facial attention localization strategy: We investigated the effectiveness of facial attention localization strategies during training. The first strategy involves Lnoise, while the second strategy adds attention loss Lfacial. From Table 4, we observe that ConsistentID experiences a clear improvement in metrics related to facial feature consistency and fine-grained ID preservation when Lfacial is considered. This confirms the effectiveness of maintaining ID consistency between facial regions and the entire face during the training process.

Image projection module: Additionally, we compared two training strategies. The first involves frozen weights of the image projection model and only training our designed FacialEncoder. The

2. https://chatgpt.com/

second strategy involves training both simultaneously. The results from Table 4 indicate that concurrently training ImageProjection brings the maximum benefits to the model. This is attributed to our model being an ID preservation method of cooperative training with multimodal text and image information.

The LLaVA1.5 usage: In Table 4 (bottom), we analyze the impact of using LLaVA 1.5 to generate textual descriptions during training. Our observations indicate that incorporating LLaVA 1.5 leads to noticeable improvements across various metrics, with particularly significant gains in the CLIP-I metric. Notably, during inference, our model achieves satisfactory performance without relying on LLaVA 1.5, using only the simplified facial description, ‘This person has one nose, two eyes, two ears, and a mouth,’ instead of generating detailed facial region descriptions. This demonstrates that our model effectively learns fine-grained features during the training phase with LLaVA 1.5, enabling it to maintain strong image understanding and identity preservation during inference.

Different Facial Areas’ Number: To explore the influence of different numbers of facial areas, we adhere to the sequence ‘face, nose, eyes, ears, and mouth’ and incrementally introduce the selected facial areas, as depicted in Figure 13. We note a progressive enhancement in image quality as the number of facial

Easyphoto (fine-tuning based)

Facechain (fine-tuning based)

GPT4o (direct inference)

ConsistentID (direct inference)

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

Awomanwearing

aSantahat Awoman

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

thebackground Awoman

injungle Aman

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

bakingcookies

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

colorfulgraffitiin

inthebackground Amanwitha

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

withamountain

- Fig. 10: The comparisons with more fine-tuning-based and direct inference models. For direct inference method GPT4o, we use stars’ names in the text prompt.

regions increases, attributed to the richer multi-modal prompts. However, with regard to the CLIP-T metric, detailed textual descriptions encompass a greater variety of objects, potentially leading to oversight by the CLIP model.

Attention loss Lfacial: To further validate the effectiveness of our Lfacial, we present several comparisons using two different identities in Figure 14. From the figure, we draw the following conclusions: 1) The details of key facial areas, such as Taylor’s eye shape, are well preserved. 2) The appearances of facial regions, such as eyes, ears, and mouths, effectively reflect identity consistency with the reference images.

Additionally, as shown in Figure 17, the attention scores in the model’s facial feature regions gradually increase as training

progresses. This indicates improved semantic alignment between the facial regions and their corresponding textual descriptions, further supporting the efficacy of Lfacial.

Delay control: In Figure 15, we provide visualized results to evaluate the impact of delay control during inference. The term ‘merge step’ denotes the first time step in which we incorporate fine-grained facial image features. It helps to control the balance between text prompts and face images. In general, as the ‘merge step’ increases, the influence of fine-grained facial image features gradually diminishes.

For example, if the ‘merge step’ is set to 0, it indicates that fine-grained facial image features are dominant in the generation process and might result in semantic inconsistencies. On the

[Figure 216]

Photomaker IPAdapter InstantID

Reference ConsistentID-SD1.5

ConsistentID-SDXL

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

A manwearinga

rainbowscarf

[Figure 222]

santahat amaninafirefighter

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

outfit

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

lecture A womanwearinga

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

A womangivinga

[Figure 239]

- Fig. 11: Results of our proposed ConsistentID model extended on the SDXL base model, compared with other SDXL-based models.

MarsNeonpunk

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

References IPAdapter IPA-FaceID

IPA-FaceIDPlus

IPA-FaceIDPortrait ConsistentID

- Fig. 12: Comparison of ConsistentID with IP-Adapter and its face version variants conditioned on different styles.

[Figure 251]

Fig. 13: Ablation study of the number of facial regions.

per image. This quantitative analysis highlights the competitive performance of the system in terms of inference efficiency 3.

contrary, setting the ‘merge step’ to 0 will maximize the guidance of text prompts, yet might harm the identity consistency.

Extensibility of methods for base models: We further replace the base model in the ConsistentID framework with the SDXL model, as illustrated in Figure 11. From the figure, ConsistentID effectively enhances facial feature consistency and outperforms other models in terms of ID similarity. Furthermore, we conducted

To visually illustrate the impact of the ‘merge step’, we display the variation curves for the CLIP-I, CLIP-T, and FGIS metrics as the ‘merge step’ increases in Figure 16. From the figure, we observe a consistent trend where the textual control gradually strengthens with each increment of the ‘merge step’.

3. Notably, during the inference phase, the usage of the LLaVa or ChatGPT4v module is not mandatory. Satisfactory results can be obtained by relying exclusively on predefined facial features prompts, such as ‘face, nose, eyes, ears, and mouth.’

Inference time for each module: In Table 5, we show the processing time for each module involved in handling a single image during inference, with a total duration of 16 seconds

With Attention Loss

No Attenion Loss

With Attention Loss

No Attention Loss

Reference

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

a person in a firefighter outfit a person in a chef outfit

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

a person in the jungle

a person on the beach

- Fig. 14: Visualized results with or without using attention loss.

Reference

Merge Step = 0

Merge Step = 10

Merge Step = 20

Merge Step = 30

Merge Step = 40

Merge Step = 50

a man with a mountain in the background

a man in a firefighter outfit

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

a man wearing a spacesuit

[Figure 271]

a woman with a blue house in the background

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

- Fig. 15: Visualized results under different ‘merge steps’. ‘Merge Step’ indicates when to start adding facial image features to the text prompt.

additional ablation experiments using the SDXL model as the base for the ConsistentID framework, as shown in Table 6. Compared to the SD1.5-based ConsistentID model, the SDXL-based version demonstrates notable improvements across key metrics, including facial similarity, CLIP-I, CLIP-T, and FID. These results further validate the robust generalization capabilities of the ConsistentID model and demonstrate its flexibility in utilizing more advanced base models for improved performance.

|BiSeNet InsightFace FacialEncoder UNet|Inference<br><br>|
|---|---|
|Time (s) 1 3 3 5|4|

TABLE 5: Inference time of each module.

#### 5 CONCLUSION

In this work, we introduce ConsistentID, an innovative method designed to maintain identity consistency and capture diverse facial details. We have developed two novel modules: a multimodal facial prompt generator and an identity preservation network. The

|CLIP-T ↑ FaceSim ↑ FID ↓<br><br>|Speed (s) ↓|
|---|---|
|ConsistentID-SD1.5 19.8 62.9 307.9<br><br>|16|
|ConsistentID-SDXL 21.9 73.1 304.5<br><br>|18|

TABLE 6: ConsistentID extension ablation experiment, evaluating the performance using different base models. The average performance is tested on 180 images under 45 prompt conditions, using the character ID shown in Figure 11.

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

Origin IMG A man in a forest adventuring

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

Result person face nose eye ear mouth

Fig. 17: Experiments to visualize facial changes in attention map during training results.

former is dedicated to generating multimodal facial prompts by incorporating both visual and textual descriptions at the facial region level. The latter aims to ensure ID consistency in each facial area through a facial attention localization strategy, preventing the blending of ID information from different facial regions. By leveraging multimodal fine-grained prompts, our approach achieves remarkable identity consistency and facial realism using only a single facial image. Additionally, we present the FGID dataset, a comprehensive dataset containing fine-grained identity information and detailed facial descriptions essential for training the ConsistentID model. Experimental results demonstrate outstanding accuracy and diversity in personalized facial generation, surpassing existing methods on the MyStyle dataset.

#### REFERENCES

- [1] X. Ju, A. Zeng, C. Zhao, J. Wang, L. Zhang, and Q. Xu, “Humansd: A native skeleton-guided diffusion model for human image generation,” arXiv preprint arXiv:2304.04269, 2023.
- [2] X. Liu, J. Ren, A. Siarohin, I. Skorokhodov, Y. Li, D. Lin, X. Liu, Z. Liu, and S. Tulyakov, “Hyperhuman: Hyperrealistic human generation with latent structural diffusion,” arXiv preprint arXiv:2310.08579, 2023.
- [3] N. Ruiz, Y. Li, V. Jampani, W. Wei, T. Hou, Y. Pritch, N. Wadhwa, M. Rubinstein, and K. Aberman, “Hyperdreambooth: Hypernetworks for fast personalization of text-toimage models,” arXiv preprint arXiv:2307.06949, 2023.
- [4] Z. Huang, K. C. Chan, Y. Jiang, and Z. Liu, “Collaborative diffusion for multi-modal face generation and editing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 6080–6090.
- [5] M. Stypułkowski, K. Vougioukas, S. He, M. Zie˛ba, S. Petridis, and M. Pantic, “Diffused heads: Diffusion models beat gans on talking-face generation,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 5091–5100.
- [6] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020.

- [7] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.
- [8] A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen, “Glide: Towards photorealistic image generation and editing with text-guided diffusion models,” arXiv preprint arXiv:2112.10741, 2021.
- [9] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchical text-conditional image generation with clip latents,” arXiv preprint arXiv:2204.06125, vol. 1, no. 2, p. 3, 2022.
- [10] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans et al., “Photorealistic text-to-image diffusion models with deep language understanding,” Advances in Neural Information Processing Systems, vol. 35, pp. 36479– 36494, 2022.
- [11] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10684– 10695.
- [12] Y. Balaji, S. Nah, X. Huang, A. Vahdat, J. Song, K. Kreis, M. Aittala, T. Aila, S. Laine, B. Catanzaro et al., “ediffi: Text-to-image diffusion models with an ensemble of expert denoisers,” arXiv preprint arXiv:2211.01324, 2022.
- [13] Z. Xue, G. Song, Q. Guo, B. Liu, Z. Zong, Y. Liu, and P. Luo, “Raphael: Text-to-image generation via large mixture of diffusion paths,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [14] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3836–3847.
- [15] H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang, “Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models,” arXiv preprint arXiv:2308.06721, 2023.
- [16] B. Wang, H. Zheng, X. Liang, Y. Chen, L. Lin, and M. Yang, “Toward characteristic-preserving image-based virtual tryon network,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 589–604.
- [17] D. Valevski, D. Lumen, Y. Matias, and Y. Leviathan, “Face0: Instantaneously conditioning a text-to-image model on a face,” in SIGGRAPH Asia 2023 Conference Papers, 2023, pp. 1–10.
- [18] R. Gal, Y. Alaluf, Y. Atzmon, O. Patashnik, A. H. Bermano, G. Chechik, and D. Cohen-Or, “An image is worth one word: Personalizing text-to-image generation using textual inversion,” arXiv preprint arXiv:2208.01618, 2022.
- [19] N. Kumari, B. Zhang, R. Zhang, E. Shechtman, and J.Y. Zhu, “Multi-concept customization of text-to-image diffusion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1931– 1941.
- [20] G. Xiao, T. Yin, W. T. Freeman, F. Durand, and S. Han, “Fastcomposer: Tuning-free multi-subject image generation with localized attention,” arXiv preprint arXiv:2305.10431, 2023.
- [21] Z. Li, M. Cao, X. Wang, Z. Qi, M.-M. Cheng, and Y. Shan, “Photomaker: Customizing realistic human photos via stacked id embedding,” arXiv preprint arXiv:2312.04461,

- 2023.
- [22] Q. Wang, X. Bai, H. Wang, Z. Qin, and A. Chen, “Instantid: Zero-shot identity-preserving generation in seconds,” arXiv preprint arXiv:2401.07519, 2024.
- [23] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” arXiv preprint arXiv:2310.03744, 2023.
- [24] Y. Zheng, H. Yang, T. Zhang, J. Bao, D. Chen, Y. Huang, L. Yuan, D. Chen, M. Zeng, and F. Wen, “General facial representation learning in a visual-linguistic manner,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 18697–18709.
- [25] Q. Cao, L. Shen, W. Xie, O. M. Parkhi, and A. Zisserman, “Vggface2: A dataset for recognising faces across pose and age,” in 2018 13th IEEE international conference on automatic face & gesture recognition (FG 2018). IEEE, 2018, pp. 67–74.
- [26] Z. Liu, P. Luo, X. Wang, and X. Tang, “Deep learning face attributes in the wild,” in Proceedings of the IEEE international conference on computer vision, 2015, pp. 3730–3738.
- [27] Y. Nitzan, K. Aberman, Q. He, O. Liba, M. Yarom, Y. Gandelsman, I. Mosseri, Y. Pritch, and D. Cohen-Or, “Mystyle: A personalized generative prior,” ACM Transactions on Graphics (TOG), vol. 41, no. 6, pp. 1–10, 2022.
- [28] K. Wang, Q. Wu, L. Song, Z. Yang, W. Wu, C. Qian, R. He, Y. Qiao, and C. C. Loy, “Mead: A large-scale audio-visual dataset for emotional talking-face generation,” in European Conference on Computer Vision. Springer, 2020, pp. 700– 717.
- [29] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PMLR, 2021, pp. 8748–8763.
- [30] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4195–4205.
- [31] L. Huang, D. Chen, Y. Liu, Y. Shen, D. Zhao, and J. Zhou, “Composer: Creative and controllable image synthesis with composable conditions,” arXiv preprint arXiv:2302.09778, 2023.
- [32] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach, “Sdxl: Improving latent diffusion models for high-resolution image synthesis,” arXiv preprint arXiv:2307.01952, 2023.
- [33] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention– MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer, 2015, pp. 234–241.
- [34] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman, “Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 22500–22510.
- [35] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” arXiv preprint arXiv:2106.09685, 2021.
- [36] Z. Liu, M. Li, Y. Zhang, C. Wang, Q. Zhang, J. Wang, and Y. Nie, “Fine-grained face swapping via regional gan

- inversion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 8578– 8587.
- [37] O. R. Nasir, S. K. Jha, M. S. Grover, Y. Yu, A. Kumar, and R. R. Shah, “Text2facegan: Face generation from fine grained textual descriptions,” in 2019 IEEE Fifth International Conference on Multimedia Big Data (BigMM). IEEE, 2019, pp. 58–67.
- [38] L. Wan, J. Wan, Y. Jin, Z. Tan, and S. Z. Li, “Fine-grained multi-attribute adversarial learning for face generation of age, gender and ethnicity,” in 2018 International Conference on Biometrics (ICB). IEEE, 2018, pp. 98–103.
- [39] C. Meng, Y. Song, J. Song, J. Wu, J.-Y. Zhu, and S. Ermon, “Sdedit: Image synthesis and editing with stochastic differential equations,” arXiv preprint arXiv:2108.01073, 2021.
- [40] T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for generative adversarial networks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 4401–4410.
- [41] J. Gu, Y. Wang, N. Zhao, T.-J. Fu, W. Xiong, Q. Liu, Z. Zhang, H. Zhang, J. Zhang, H. Jung et al., “Photoswap: Personalized subject swapping in images,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [42] R. Liu, B. Ma, W. Zhang, Z. Hu, C. Fan, T. Lv, Y. Ding, and X. Cheng, “Towards a simultaneous and granular identityexpression control in personalized face generation,” arXiv preprint arXiv:2401.01207, 2024.
- [43] F. Rosberg, E. E. Aksoy, F. Alonso-Fernandez, and C. Englund, “Facedancer: pose-and occlusion-aware high fidelity face swapping,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2023, pp. 3454–3463.
- [44] Z. Huang and Y. Li, “Interpretable and accurate fine-grained recognition via region grouping,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 8662–8672.
- [45] S. Umirzakova and T. K. Whangbo, “Detailed feature extraction network-based fine-grained face segmentation,” Knowledge-Based Systems, vol. 250, p. 109036, 2022.
- [46] C. Yu, G. Lu, Y. Zeng, J. Sun, X. Liang, H. Li, Z. Xu, S. Xu, W. Zhang, and H. Xu, “Towards high-fidelity text-guided 3d face generation and manipulation using only images,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 15326–15337.
- [47] L. Li, T. Zhang, Z. Kang, and X. Jiang, “Mask-fpan: Semisupervised face parsing in the wild with de-occlusion and uv gan,” Computers & Graphics, vol. 116, pp. 185–193, 2023.
- [48] J. Yu, Y. Wang, C. Zhao, B. Ghanem, and J. Zhang, “Freedom: Training-free energy-guided conditional diffusion model,” arXiv preprint arXiv:2303.09833, 2023.
- [49] K. Kim, Y. Kim, S. Cho, J. Seo, J. Nam, K. Lee, S. Kim, and K. Lee, “Diffface: Diffusion-based face swapping with facial guidance,” arXiv preprint arXiv:2212.13344, 2022.
- [50] C. Yu, J. Wang, C. Peng, C. Gao, G. Yu, and N. Sang, “Bisenet: Bilateral segmentation network for real-time semantic segmentation,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 325–341.
- [51] H. Chefer, Y. Alaluf, Y. Vinker, L. Wolf, and D. CohenOr, “Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models,” ACM Transactions on Graphics (TOG), vol. 42, no. 4, pp. 1–10, 2023.

- [52] Z. Qiu, W. Liu, H. Feng, Y. Xue, Y. Feng, Z. Liu, D. Zhang, A. Weller, and B. Schölkopf, “Controlling text-to-image diffusion by orthogonal finetuning,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [53] N. Tumanyan, M. Geyer, S. Bagon, and T. Dekel, “Plugand-play diffusion features for text-driven image-to-image translation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1921– 1930.
- [54] D. Beniaguev, “Synthetic faces high quality (sfhq) dataset,”

2022. [Online]. Available: https://github.com/SelfishGene/ SFHQ-dataset

- [55] W. Chen, H. Hu, Y. Li, N. Ruiz, X. Jia, M.-W. Chang, and W. W. Cohen, “Subject-driven text-to-image generation via apprenticeship learning,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [56] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, “Arcface: Additive angular margin loss for deep face recognition,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 4690–4699.
- [57] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman et al., “Laion-5b: An open large-scale dataset for training next generation image-text models,” Advances in Neural Information Processing Systems, vol. 35, pp. 25278– 25294, 2022.
- [58] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.
- [59] W. Cong, J. Zhang, L. Niu, L. Liu, Z. Ling, W. Li, and L. Zhang, “Dovenet: Deep image harmonization via domain verification,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 8394– 8403.
- [60] F. Schroff, D. Kalenichenko, and J. Philbin, “Facenet: A unified embedding for face recognition and clustering,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 815–823.
- [61] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” Advances in neural information processing systems, vol. 30, 2017.
- [62] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.
- [63] Y. Liu, C. Yu, L. Shang, Z. Wu, X. Wang, Y. Zhao, L. Zhu, C. Cheng, W. Chen, C. Xu, H. Xie, Y. Yao, W. Zhou, C. Yingda, X. Xie, and B. Sun, “Facechain: A playground for identity-preserving portrait generation,” arXiv preprint arXiv:2308.14256, 2023.
- [64] Z. Wu, J. Xu, X. Zou, K. Huang, X. Shi, and J. Huang, “Easyphoto: Your smart ai photo generator,” arXiv preprint arXiv:2310.04672, 2023.

