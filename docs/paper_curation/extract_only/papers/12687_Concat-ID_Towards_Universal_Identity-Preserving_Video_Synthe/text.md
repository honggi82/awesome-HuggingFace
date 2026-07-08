# Concat-ID: Towards Universal Identity-Preserving Video Synthesis

Yong Zhong1* Zhuoyi Yang2 Jiayan Teng2 Xiaotao Gu3 Chongxuan Li1† 1 Gaoling School of AI, Renmin University of China, Beijing, China 2 Tsinghua University 3 Zhipu AI

yongzhong@ruc.edu.cn, {yangzy22,tengjy24}@mails.tsinghua.edu.cn, xiaotao.gu@zhipuai.cn, chongxuanli@ruc.edu.cn

Project page and code: https://ml-gsai.github.io/Concat-ID-demo/

arXiv:2503.14151v3[cs.CV]2Jul2025

[Figure 1]

[Figure 2]

##### (a) Single identity

Identity

A man standing next to an airplane, engaged in a conversation on his cell phone. He is wearing sunglasses and a black top, and he appears to be talking seriously. The airplane has a green stripe ...

[Figure 3]

[Figure 4]

Identity A woman dressed in elegant court attire, a flowing gown, a pearl necklace, and a delicate ivory fan in her gloved hands. She fanned herself slowly, her movements graceful and deliberate. The ...

[Figure 5]

[Figure 6]

[Figure 7]

##### (b) Multiple identities

Multi-Identities

Two people sit on a park bench, each holding a cup of hot coffee. One leans back, head tilted up, eyes closed, enjoying the warmth. The other turns slightly, smiling as they speak, ...

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Multi-Identities ... The person in the middle wears a light-colored sweater, speaking with enthusiasm ... The person on the left, dressed in a dark jacket, leans forward slightly... a thoughtful smile on ... lips. The person on the right wears a blue T-shirt, head tilted slightly, expression calm with a smile. ...

[Figure 12]

[Figure 13]

[Figure 14]

##### (c) Multiple subjects

Identity Clothing

A person wearing a shirt takes a casual walk through a tree-lined avenue. The scent of freshly cut grass fills the air, ... The head tilts slightly downward, a content expression settling on the face.

[Figure 15]

[Figure 16]

[Figure 17]

Identity Background A person wears a casual shirt, reading a book. The person's head is slightly tilted, eyes focused on the pages, lips curved into a faint smile. Sunlight ... casting a warm glow on the face, ...

- Figure 1. Concat-ID produces natural videos for identity-preserving video generation. We select samples for (a) single-identity, (b) multi-identity, and (c) multi-subject scenarios, respectively.

## Abstract

We present Concat-ID, a unified framework for identitypreserving video generation. Concat-ID employs variational autoencoders to extract image features, which are then concatenated with video latents along the sequence dimension. It relies exclusively on inherent 3D self-attention mechanisms to incorporate them, eliminating the need for additional parameters or modules. A novel cross-video pairing strategy and a multi-stage training regimen are introduced to balance identity consistency and facial editability while enhancing video naturalness. Extensive experiments demonstrate Concat-ID’s superiority over existing methods in both single and multi-identity generation, as well as its seamless scalability to multi-subject scenarios, including virtual try-on and background-controllable generation. Concat-ID establishes a new benchmark for identity-preserving video synthesis, providing a versatile and scalable solution for a wide range of applications.

## 1. Introduction

Identity-preserving video generation, which seeks to create human-centric videos of a specific identity accurately matching a user-provided face image, has recently gained significant attention, as evidenced by the success of commercial tools such as Vidu [23] and Pika [19].

A primary challenge in this field is achieving a balance between maintaining identity consistency and enabling facial editability. Prior work [9, 13, 28, 30] fails to effectively preserve identity despite utilizing special face encoders and incorporating extra adapters to mitigate crossmodal disparities. To mitigate this limitation, some approaches [4, 29] substitute the spatially aligned reference image in pre-trained image-to-video models [2, 27] with facial images, leading to a significant improvement in identity consistency. However, they still face challenges in preventing the replication of facial expressions from the reference image. Moreover, the supplementary modules and parameters introduced by these methods contribute to increased complexity in both model training and inference.

In this work, we introduce Concat-ID, a concise, effective, and versatile framework for identity-preserving video generation. By unifying the model architecture, data processing, and training procedure, Concat-ID not only achieves single-identity video generation but also seamlessly integrates multiple identities and accommodates diverse subjects. Specifically, Concat-ID employs Variational Autoencoders (VAEs) to extract image features, which are then concatenated with video latents along the sequence

*Work done during the internship at Zhipu. †Correspondence to Chongxuan Li.

dimension. This approach relies exclusively on 3D selfattention mechanisms, which are inherently present in stateof-the-art video generation models, to incorporate image features, thereby eliminating the need for extra modules or parameters. Furthermore, to effectively balance identity consistency and facial editability while enhancing video naturalness, we develop a novel cross-video pairing strategy and a multi-stage training regimen.

The quantitative and qualitative results, along with the user study (see Sec. 5.2), demonstrate that Concat-ID produces videos with the most consistent identity and superior facial editability across all baselines, for both singleidentity and multi-identity video generation. Moreover, we illustrate that Concat-ID can seamlessly extend to multisubject scenarios, including virtual try-on and backgroundcontrollable generation, while effectively preserving identity (see Sec. 5.3). These findings underscore Concat-ID’s capability to scale effectively to diverse subjects, ensuring consistent high performance across various applications.

The principal contributions of this work are as follows:

- • We propose Concat-ID, an effective framework for unified identity-preserving video generation across singleidentity, multi-identity, and multi-subject scenarios.
- • Concat-ID utilizes VAEs to extract image features and integrates them via inherent 3D self-attention mechanisms, without introducing additional parameters or modules.
- • We develop a cross-video pairing strategy and a multistage training regimen to balance identity consistency and facial editability, while enhancing video naturalness.
- • Concat-ID demonstrates superior identity consistency and facial editability in single and multi-identity scenarios, and seamlessly scales to multi-subject scenarios.

## 2. Related works

The rapid advancement of text-to-video and image-to-video diffusion models [8, 17, 20, 27, 31] has spurred significant interest in fine-tuning these models for downstream tasks, particularly identity-preserving video generation. Tuningbased methods [11, 18] adapt pre-trained video models for each new identity through test-time fine-tuning. Alternatively, tuning-free methods [9, 13, 28, 30] typically leverage face encoders [3, 21] to extract facial features and incorporate additional adapters to mitigate cross-modal discrepancies. Some approaches [4, 25, 29] further enhance identity consistency by integrating face features extracted from a Variational Autoencoder (VAE). For instance, ConsisID [29] and Ingredients [4] replace spatially aligned reference images in pre-trained image-to-video models for single-identity and multi-identity generation, respectively. Placing greater emphasis on enhancing video naturalness, Movie-Gen [20] refines the balance between identity consistency and facial editability for single-identity generation through cross-paired data construction. In this work, we

## 4. Concat-ID

[Figure 18]

VAE

Given a reference image containing a human face, our goal is to generate identity-preserving videos based on userprovided text prompts, while also enabling the integration of additional identities or subjects. To address this challenge, we propose Concat-ID, a concise, effective, and versatile framework. As illustrated in Fig. 2, we introduce a unified architecture for extracting and injecting features from any number of identities and subjects without requiring extra modules or parameters (see Sec. 4.1). To balance identity consistency and facial editability while enhancing video naturalness, we further construct cross-video pairs as training data (see Sec. 4.2) and propose a novel multi-stage training strategy (see Sec. 4.3).

decoder

[Figure 19]

Text/Image-to-Video diffusion transformer 3D self-attention

Text encoder

“Two people .”

VAE encoder

VAE encoder

[Figure 20]

Video latent Image latent

Trainable

[Figure 21]

[Figure 22]

### 4.1. A unified architecture

Image latent of more subjects

We focus on designing a unified model architecture capable of extracting and fusing the identity feature and readily extendable to multi-identity and multi-subject scenarios. Revisiting the role of VAEs, we recognize their ability to compress conditioning images into the same latent space as the video latent Z. Consequently, our denoising transformer ϵθ can inherently interpret these features. Based on this insight, we adopt the VAE as our feature extractor.

- Figure 2. The architecture of Concat-ID. We utilize VAEs to extract image latents from reference images and concatenate them at the end of the video latents along the sequence dimension. ConcatID relies solely on 3D self-attention mechanisms, which are commonly present in state-of-the-art video generation models, to integrate image features without adding extra modules or parameters.

explore a unified framework capable of handling singleidentity, multi-identity, and multi-subject generation while maintaining a crucial balance between consistency and editability, without requiring test-time fine-tuning.

- 3. Preliminary

Specifically, for M reference images {Ii}Mi=1, we encode each Ii to obtain the image feature ci = E(Ii) ∈ R1×HW×C, and then concatenate these features with Z in sequence. Thus, the input to ϵθ is given by:

Z′ = Concat(Z,c1,c2,··· ,cM), (1) where Concat(·,·,···) denotes concatenation along the sequence dimension and Z′ ∈ R(N+M)×HW×C. As shown in Fig. 2, this feature injection through concatenation is compatible with any video generation model that utilizes 3D self-attention, which are generally present in state-of-theart video generation models. Since Z and ci are in the same latent space, ϵθ can seamlessly integrate identity-preserving features without the need for additional modules or parameters to address cross-modal disparities.

Existing state-of-the-art text-to-video and image-to-video models [8, 15, 17, 20, 27] generally consist of three main components: a 3D variational autoencoder (VAE) E, text encoders T , and a denoising transformer ϵθ. Given a video X = {xi}Ni=1 with N frames, E initially compresses the video into a latent representation Z ∈ RT×HW×C along the spatiotemporal dimensions, where HW denotes the spatial dimension, C represents the channel dimension, and T is the temporal dimension. To simplify, we refer to T × HW as the sequence dimension. The ϵθ then takes the noisecorrupted latent representation Z as its input, and applies a 3D (spatiotemporal) self-attention mechanism [8, 27] to model the distribution of video content. Additionally, a 3D relative positional encoding (i.e., 3D-ROPE) is incorporated within the 3D attention module to enhance the model’s ability to capture both temporal and spatial dependencies in videos. Meanwhile, the text encoder T processes the text prompt and encodes it into a text representation ctxt. ϵθ typically integrates ctxt either through cross-attention layers [20] or by concatenating it with Z [27]. A mean squared error loss [5, 32] is commonly used to optimize ϵθ.

Concatenating Z and ci along the channel dimension is another direct method for feature injection, as employed in ConsisID [29] and Ingredients [4]. However, this strategy introduces artifacts (see Fig. 4 and Fig. 5) due to spatial misalignment between face images and video latents. In contrast, by leveraging a 3D self-attention mechanism, our sequence concatenation promotes spatial interactions without compromising the quality of any generated frame. Furthermore, it scales efficiently to handle multi-identity and multi-subject scenarios (see Fig. 1).

### 4.2. Data construction

The task of identity-preserving video generation relies on image-video pairs as training data, where an image must de-

detect faces using SCRFD [6], and remove videos if more than 30% of the frames have inconsistent numbers of individuals.1 Finally, for frames with the same face count, we compute the ArcFace cosine similarity [3] between consecutive frames and discard videos if more than 30% of the frames have a similarity score below 0.5.

Caption-video pairs

Retrieving humanrelated terms

Face detection Video filtering

Pre-training image-video pairs

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

The above processes yield 1.3 million videos featuring a single identity, and we uniformly select 5 face images per video, defining pre-training pairs Spre = {(Iki ,Xk)}i,k where k denotes the video index and Iki represents the i-th reference image of Xk. The self-supervised nature of this paired data, where images from the same video serve as labels, inherently limits facial editability. Specifically, models trained on such data may produce frames in which facial expressions unintentionally mirror those of the reference images (see Fig. 7), leading to unnatural content. This issue becomes particularly pronounced when a semantic gap exists between the reference images and the text prompts. To enhance facial editability and naturalness, we propose a cross-video image-video pairing strategy.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

0.7 0.75 0.91

Cosine similarity

0.66 0.74

0.62

[Figure 35]

[Figure 36]

Minimum pairing Random pairing

High aesthetic & dynamic quality

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Cross-video pairs Trade-off pairs

(a) The procedure of data processing.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Cross-video pairs. The standard process for constructing video clips involves segmenting raw long videos into multiple shorter segments using various algorithms that detect scene transitions, such as motion variations and shot changes. Theoretically, many existing video clips in training sets feature varied facial expressions and head poses of the same person. To construct cross-video pairs where the reference image originates from a different video, we calculate the cosine similarity among images {Iv1}v. For the k-th video, we randomly select an image Ij1 from {Iv1}v as its paired reference image, ensuring that 0.7 ≤ cos(Ij1,Ik1) < 0.9, where the function cos(·,·) computes the cosine similarity. The final cross-video pairs Scross include 0.8 million image-video pairs with 0.5 million reference images, indicating a reference image can correspond to multiple videos.

(b) Some samples of paired cross-video reference images.

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

(c) Some samples of trade-off pairs.

Personalized image generation can also synthesize reference images with the same identity as given videos but varied identity-irrelevant factors, as demonstrated in [10, 20]. However, this approach incurs high computational costs, particularly for large-scale image-video pairs. Additionally, existing personalized generation methods [7, 28, 33] often struggle to preserve detailed facial features, which limits their effectiveness. In contrast, as shown in Fig. 3b, our retrieval-based method efficiently gathers a large-scale set of real reference images that accurately match the identity of corresponding videos while exhibiting diversity across multiple dimensions, such as facial expressions, hairstyles, lighting conditions, and other identity-irrelevant factors.

- Figure 3. Constructing three types of image-video pairs for a single identity: pre-training, cross-video and trade-off pairs.

pict a human face that matches the identity of corresponding videos. To progressively balance identity consistency and facial editability, as illustrated in Fig. 3, we construct three types of image-video pairs for a single identity: pre-training pairs Spre, cross-video pairs Scross, and trade-off pairs Strade. Pre-training pairs. To ensure data quality, we filter out videos that are unrelated to humans, contain inconsistent numbers of individuals, or exhibit inconsistencies in identity. Specifically, to retrieve human-related videos from the caption-video pairs, we design a human term table that includes various categories such as basic human descriptors, gender, and occupation. We then exclude videos whose captions do not contain any human-related terms. Next, we uniformly sample two frames per second from each video,

Trade-off pairs. Similar to the construction of cross-video pairs, for the k-th video, we identify its reference image Ij1 with the smallest cos(Ij1,Ik1), ensuring that 0.9 ≤

1The common person count across frames is considered the video’s person count.

cos(Ij1,Ik1) < 0.99. This forms our trade-off dataset Strade with 160 thousand videos, improving consistency between reference images and videos compared to cross-video pairs. Additionally, we filter out videos where the facial region occupies less than 4% or more than 90% of the frame area and rank Strade based on the weighted sum of aesthetics scores, optical flow scores, and motion scores [27]. Finally, we retain the top 50,000 videos for training.

In this section, we detail the data construction process for a single identity. However, this procedure can be seamlessly scaled to multi-identity by independently processing each identity within a video. Similarly, it can be extended to general subjects by replacing face detectors with openset detectors, such as Grounding DINO [22], and substituting ArcFace cosine similarity with general feature similarity metrics, such as CLIP cosine similarity [21]. Please refer to

- Appendix B for further details on the training data construction for our multi-identity and multi-subject scenarios.

### 4.3. Training strategy

Building on our innovative data construction, we introduce a multi-stage training process: pre-training stage, crossvideo fine-tuning, and trade-off fine-tuning. In the pretraining stage, we optimize a text-to-video model on Spre to map facial details into generated videos. This selfsupervised training method may constrain certain generated video frames to adhere strictly to the given condition images, potentially degrading the editability of facial expressions and the overall naturalness. The cross-video finetuning on Scross, using image-video pairs derived from different videos, can alleviate this issue. However, we observe that this fine-tuning enhances facial editability at the expense of identity fidelity (see Sec. 5.4).

A simple strategy to further balance fidelity and editability is to mix pre-trained pairs and cross-video pairs in a 1:1 ratio, a similar method adopted by Movie-Gen [20]. However, our initial experiments suggest that this approach results in unstable training due to varying identity consistency between pre-trained pairs and cross-video pairs. To address this issue while ensuring high-degree motion and high artistic quality, we ultimately fine-tune the model on Strade.

Throughout all training stages, we proportionally scale, pad, and center-crop images to match the video resolution. To ensure the model focuses on facial regions during training and prevents background leakage during inference, we segment and drop the background of reference images [26]. Additionally, to improve robustness and generalization, we introduce random noise to reference images during training, while omitting this noise during inference. To further differentiate the image latent ci from the video latent Z and distinguish between different ci, we extend 3D-RoPE to incorporate multiple reference images along the sequence dimension. Specifically, we introduce a temporal bias N to

define the 3D position of a token th,w in ci: 3D-Pos(th,w) = (i + N,h,w), (2)

where 3D-Pos(·) denotes the 3D position and (h,w) are the spatial coordinates of the token.

Owing to the simplicity and efficiency of Concat-ID in both data construction and model architecture, our training strategy can seamlessly scale to multi-identity and multisubject scenarios. Moreover, we establish that singleidentity pre-training facilitates enhanced identity preservation in these downstream tasks (see Tab. 2).

## 5. Experiments

### 5.1. Experimental settings

Datasets. We evaluate all methods on the ConsistIDBenchmark [29], which consists of 172 reference images and 90 text prompts spanning nine categories. To ensure a fair comparison, we exclude reference images present in our training data using a combination of automated and manual filtering techniques. Consequently, our evaluation dataset comprises 873 prompt-image pairs, derived from 97 reference images, with one prompt randomly selected from each category for each image. For multi-identity evaluation, we additionally construct 14 distinct pairs of reference images and design 20 textual prompts using ChatGPT [1]. Please refer to Appendix A.1 for further details.

Metrics. We evaluate all methods on identity consistency, text alignment, and facial editability. (1) Identity consistency: Following [29], we use FaceSim-Arc (ArcSim) and FaceSim-Cur (CurSim) to assess the average cosine similarity between reference images and generated videos based on ArcFace [3] and CurricularFace [12], respectively. These face recognition models are specifically designed to disentangle identity-related features from identity-unrelated ones. (2) Text alignment: We adopt ViCLIP [24] to compute the similarity between text prompts and generated videos, following [14, 20]. (3) Facial editability: We calculate the cosine distance of CLIP image embeddings [21] (CLIPDist) between reference images and video frames. CLIP effectively captures comprehensive facial features, and thus a larger CLIPDist indicates improved facial editability.

Implementation details. We use the text-to-video model CogVideoX-5B [27] as our base model. The learning rates are set to 1.0×10−5, 5.0×10−6, and 5.0×10−6 for the first, second, and third training stages, respectively. We finetune all model parameters with a linear learning rate decay across all stages. The training data resolution is maintained at 480 × 720 pixels with 49 frames per video. Text and image prompts are independently dropped with a probability of 0.1. Further details are provided in Appendix A.2.

Baselines. For a comprehensive comparison, we use three representative open-source approaches as baselines. (1)

Identity consistency Text alignment Facial editability ArcSim ↑ CurSim ↑ ViCLIP ↑ CLIPDist ↑

Method

Single identity

ID-Animator [9] ‡ 0.289 0.304 0.204 0.297 ConsisID [29]†‡ 0.432 0.451 0.237 0.303 Concat-ID (Ours) † 0.442 0.466 0.242 0.325

Multiple identities

Ingredients [4] †‡ 0.293 0.316 0.199 0.407 Concat-ID (Ours) † 0.492 0.514 0.190 0.410

Table 1. Quantitative results for single-identity and multiidentity generation. † denotes that these methods share the same video model. ‡ indicates corresponding methods introduce additional adapters and auxiliary loss. Concat-ID achieves superior identity consistency and facial editability while maintaining better or comparable text alignment relative to the baselines.

Single-identity personalization methods: ID-Animator [9] and ConsisID [29]. (2) Multi-identity personalization methods: Ingredients [4]. ID-Animator, ConsisID, and Ingredients all incorporate additional adapters and auxiliary loss functions to enhance identity consistency. Notably, ConcatID, ConsisID, and Ingredients are all built upon the same video model, CogVideoX-5B.

### 5.2. Main results

We demonstrate the effectiveness of Concat-ID through quantitative metrics, qualitative assessments, and the user study for single-identity and multi-identity generation.

Quantitative comparisons. Table 1 presents the quantitative results for single-identity and multi-identity generation. For single-identity generation, ID-Animator performs the worst, exhibiting the lowest ArcSim, CurSim, and CLIPDist scores. This suggests that it achieves the least effective balance between identity preservation and facial editability. Moreover, ID-Animator, ConsisID, and Ingredients incorporate additional adapters and auxiliary loss functions to enhance identity consistency, increasing the complexity of both training and generation processes.

In contrast, for both single-identity and multi-identity generation, Concat-ID achieves superior identity consistency simply by concatenating image latents after video latents, highlighting the effectiveness of our architecture. Furthermore, by constructing cross-video pairs, Concat-ID attains a higher CLIPDist score than ID-Animator, ConsisID, and Ingredients, demonstrating an optimal balance between identity preservation and facial editability.

Qualitative comparisons. Fig. 4 presents qualitative comparisons for single-identity generation. ID-Animator fails to maintain facial characteristics. ConsisID achieves better identity consistency, but some frames replicate facial expressions of reference images. In contrast, Concat-ID mitigates this issue while preserving identity by leveraging advantages of cross-video pairs. For multi-identity generation,

as shown in Fig. 5, Concat-ID produces videos that more accurately match identities in given images compared to Ingredients, demonstrating its effectiveness and scalability.

To maximize the potential of image-to-video models, ConsisID and Ingredients concatenate the reference image with the first latent frame along the channel dimension. However, this feature injection approach can introduce artifacts in the first generated frame due to spatial misalignment between faces images and generated videos, as evident in the initial frames of all videos. As a comparison, ConcatID excels in identity preservation without compromising the quality of any generated frames, highlighting the validity of our concatenation along the sequence dimension.

User study. According to both quantitative and qualitative results, we compare Concat-ID with the strongest baseline, ConsisID, through human evaluation. Specifically, we generate 100 videos using 10 reference images and 10 prompts designed by ChatGPT [1] to focus on expression and head pose variation. For each video group, voters answer three questions, selecting the video that: (1) best matches the reference image in facial similarity (identity consistency), (2) best aligns with the facial expressions and head poses described in the prompt (facial motion alignment), and (3) exhibits the most natural and smooth facial motion (facial motion naturalness). With 100 video groups, three types of questions, and three voters participating, we collect a total of 900 video comparison results. As shown in Fig. 6, Concat-ID surpasses ConsisID by a significant margin in identity consistency and motion alignment and naturalness, demonstrating the effectiveness of our architecture and the advantages of cross-video pair construction.

### 5.3. Multiple identities and subjects

We demonstrate that the architecture, data construction, and training strategy of Concat-ID make it seamlessly extendable to multi-identity and multi-subject scenarios.

Multi-identity scenarios. As illustrated in Fig. 1b, when provided with face images of different individuals, ConcatID can generate multi-person videos while preserving their identities, without requiring any additional parameters or modules compared to single-identity generation. Notably, despite being trained on only 40,000 videos, Concat-ID can generate three-identity videos while maintaining distinct identities, leveraging the prior knowledge from twoidentity pre-training and a powerful 3D self-attention mechanism that effectively captures both temporal and spatial dependencies. Moreover, Concat-ID determines the spatial position of each identity in the generated videos based on the concatenation sequence of the reference images.

Multi-subject scenarios. As illustrated in Fig. 1c, by sequentially concatenating a face image with a clothing image, Concat-ID enables virtual try-on while preserving both the given identity and intricate clothing details, such as lo-

[Figure 53]

[Figure 54]

ID-AnimatorConsisIDOursID-AnimatorConsisIDOursID-AnimatorConsisIDOurs

Reference Image

A person wearing a Superman outfit stands tall, arms crossed over the chest. The person's chin is lifted, eyes locked on the horizon with a determined gaze. A confident smirk appears as the cape flutters slightly in the breeze.

[Figure 55]

[Figure 56]

Reference Image

A street artist in a worn-out denim jacket and a colorful bandana stands before a weathered wall, holding a can of spray paint. The head tilts slightly as the person examines the work—a vibrant bird taking shape. A focused expression crosses the face, lips pressed together in concentration.

[Figure 57]

[Figure 58]

Reference Image

A person sits alone on a park bench, head bowed slightly, eyes glistening as a single tear slides down the cheek. The lips quiver, pressed into a thin line, while the shoulders tremble with the weight of an unspoken sorrow. The cool breeze carries away the muffled sigh.

- Figure 4. Qualitative comparisons for single-identity generation. ID-Animator fails to preserve facial details, while ConsisID replicates the expressions of the reference images, particularly in the third case, where the semantic gap between texts and reference is significant. Concat-ID effectively preserves identity, while simultaneously preventing the direct replication of facial expressions from reference images.

### 5.4. Ablation study

gos and textures. This capability also highlights ConcatID’s potential in simulating interactions between people and objects. Furthermore, the background-controllable identitypreserving generation achieved by Concat-ID demonstrates its ability to manipulate spatial layouts in generated videos by integrating spatially aligned conditions.

Fig. 7 present the qualitative ablation of Concat-ID. The pre-training stage achieves the best identity consistency but results in low facial editability. For example, facial expressions of some frames in the pre-training stage closely resemble those in reference images. However, the crossvideo stage enhances editability at the expense of identity consistency, aligning with the findings in [20]. In the third stage, Concat-ID further refines the matching threshold of cross-video pairs to better balance identity preservation and facial editability. Leveraging prior knowledge from both pre-training and cross-video fine-tuning, the trade-off stage achieves an optimal balance using only 50,000 videos. These results underscore the effectiveness of each stage in

In this section, we introduce two-identity and threeidentity generation, along with two additional subjects (i.e. clothing and background). Further details on training and data are provided in Appendix B. We posit that ConcatID’s architecture, characterized by its simplicity and effectiveness, coupled with the generalizability of its data construction and training strategy, enables effective scalability to more identities and diverse subjects, ensuring consistent high performance across a wider range of applications.

[Figure 59]

[Figure 60]

[Figure 61]

Two people sit on a park bench, each holding a cup of hot coffee. One leans back, head tilted up, eyes closed... The other turns slightly, smiling as they speak, ...

[Figure 62]

OursIngredients

[Figure 63]

[Figure 64]

Two friends sit at a train station, . One stretches, arms raised above the head, leting out a relaxed sigh, while the other leans forward, elbows on knees, .

- Figure 6. Human evaluation. Concat-ID produces more precise and natural videos while effectively preserving identity.

[Figure 65]

A person sits in a spotless room, playing a guitar. The person's head is slightly tilted, lips parted as if humming along. Fingers glide efortlessly over the strings, completely immersed in the melody. the cape fluters slightly in the breeze.

Stage

- I

[Figure 66]

Stage

- II

Stage

- III

- Figure 7. Qualitative ablation. Stage I, Stage II, and Stage III indicate the pre-training stage, cross-video stage, and trade-off stage.

[Figure 67]

OursIngredients

[Figure 68]

[Figure 69]

Two students . One leans over a laptop, typing furiously, eyes locked on the screen. The other tilts the head slightly, holding a pen to the lips, deep in thought.

[Figure 70]

OursIngredients

- Figure 5. Qualitative comparisons for multi-identity generation. Concat-ID better maintains different identities.

Identity-1 Identity-2 ArcSim ↑ CurSim ↑ ArcSim ↑ CurSim ↑

Method

No single-identity pre-training 0.514 0.535 0.526 0.550 Concat-ID (Pre-training) 0.629 0.650 0.651 0.674

our training strategy. Moreover, the quantitative analysis in

Table 2. The effect of single-identity pre-training on multiidentity pre-training. The single-identity pre-training enhances identity consistency in downstream tasks.

- Appendix C consistently supports our findings.

We also investigate the influence of single-identity pretraining on multi-identity and multi-subject pre-training. Specifically, we conduct a comparative analysis of ConcatID with and without single-identity pre-training. Although the two-identity generation is pre-trained on approximately 0.3 million videos, as presented in Tab. 2, single-identity pre-training still results in improved ArcSim and CurSim scores across all identities. This enhancement indicates that single-identity pre-training effectively strengthens identity preservation in downstream tasks. These findings provide empirical support for the scalability of our architecture, data construction methodology, and training strategy.

## 6. Conclusions

In this paper, we introduce Concat-ID, a unified framework for identity-preserving video generation. Concat-ID relies solely on 3D self-attention mechanisms, which are commonly used in state-of-the-art video generation models, without introducing additional modules or parameters. We also present a novel cross-video pairing strategy and a multi-stage training regimen to balance identity consistency and facial editability while enhancing video naturalness.

Thanks to its architecture, data construction, and training strategy, Concat-ID can scale seamlessly to multi-identity and multi-subject scenarios.

Limitations. Similar to common video generation models, our approach faces challenges in preserving the integrity of human body structures, such as the number of fingers, when handling particularly complex motions. In this paper, we focus on the single-identity scenario, and further improvement and evaluation of Concat-ID’s performance in multiple-identity and multi-subject scenarios is left for future work.

## References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 5, 6

- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2
- [3] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690–4699, 2019. 2, 4, 5
- [4] Zhengcong Fei, Debang Li, Di Qiu, Changqian Yu, and Mingyuan Fan. Ingredients: Blending custom photos with video diffusion transformers. arXiv preprint arXiv:2501.01790, 2025. 2, 3, 6, 12
- [5] Ruiqi Gao, Emiel Hoogeboom, Jonathan Heek, Valentin De Bortoli, Kevin P. Murphy, and Tim Salimans. Diffusion meets flow matching: Two sides of the same coin. 2024.

- 3

[6] Jia Guo, Jiankang Deng, Alexandros Lattas, and Stefanos Zafeiriou. Sample and computation redistribution for efficient face detection. arXiv preprint arXiv:2105.04714, 2021.

- 4

- [7] Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, Peng Zhang, and Qian He. Pulid: Pure and lightning id customization via contrastive alignment. arXiv preprint arXiv:2404.16022, 2024. 4
- [8] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103,

2024. 2, 3

- [9] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, and Jie Zhang. Id-animator: Zero-shot identity-preserving human video generation. arXiv preprint arXiv:2404.15275, 2024. 2, 6, 12
- [10] Zecheng He, Bo Sun, Felix Juefei-Xu, Haoyu Ma, Ankit Ramchandani, Vincent Cheung, Siddharth Shah, Anmol Kalia, Harihar Subramanyam, Alireza Zareian, et al. Imagine yourself: Tuning-free personalized image generation. arXiv preprint arXiv:2409.13346, 2024. 4, 13
- [11] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 2
- [12] Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, and Feiyue Huang. Curricularface: adaptive curriculum learning loss for deep face recognition. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5901–5910, 2020. 5

- [13] Yuzhou Huang, Ziyang Yuan, Quande Liu, Qiulin Wang, Xintao Wang, Ruimao Zhang, Pengfei Wan, Di Zhang, and Kun Gai. Conceptmaster: Multi-concept video customization on diffusion transformer models without test-time tuning. arXiv preprint arXiv:2501.04698, 2025. 2
- [14] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 5
- [15] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954,

2024. 3

- [16] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. Advances in neural information processing systems, 33:12104–12114, 2020. 12
- [17] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2, 3
- [18] Ze Ma, Daquan Zhou, Chun-Hsiao Yeh, Xue-She Wang, Xiuyu Li, Huanrui Yang, Zhen Dong, Kurt Keutzer, and Jiashi Feng. Magic-me: Identity-specific video customized diffusion. arXiv preprint arXiv:2402.09368, 2024. 2
- [19] Pika. Pikascenes. https : / / pika . art / ingredients, 2024. 2
- [20] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

2024. 2, 3, 4, 5, 7

- [21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2, 5
- [22] Tianhe Ren, Qing Jiang, Shilong Liu, Zhaoyang Zeng, Wenlong Liu, Han Gao, Hongjie Huang, Zhengyu Ma, Xiaoke Jiang, Yihao Chen, et al. Grounding dino 1.5: Advance the” edge” of open-set object detection. arXiv preprint arXiv:2405.10300, 2024. 5
- [23] Vidu. Reference to video. https://www.vidu.com/ create/character2video, 2024. 2
- [24] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023. 5
- [25] Tao Wu, Yong Zhang, Xiaodong Cun, Zhongang Qi, Junfu Pu, Huanzhang Dou, Guangcong Zheng, Ying Shan, and Xi Li. Videomaker: Zero-shot customized video genera-

- tion with the inherent force of video diffusion models. arXiv preprint arXiv:2412.19645, 2024. 2
- [26] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in neural information processing systems, 34: 12077–12090, 2021. 5
- [27] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2, 3, 5
- [28] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 2, 4

- [29] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyuan Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identitypreserving text-to-video generation by frequency decomposition. arXiv preprint arXiv:2411.17440, 2024. 2, 3, 5, 6, 12
- [30] Yuechen Zhang, Yaoyang Liu, Bin Xia, Bohao Peng, Zexin Yan, Eric Lo, and Jiaya Jia. Magic mirror: Id-preserved video generation in video diffusion transformers. arXiv preprint arXiv:2501.03931, 2025. 2
- [31] Min Zhao, Guande He, Yixiao Chen, Hongzhou Zhu, Chongxuan Li, and Jun Zhu. Riflex: A free lunch for length extrapolation in video diffusion transformers. arXiv preprint arXiv:2502.15894, 2025. 2
- [32] Yong Zhong, Min Zhao, Zebin You, Xiaofeng Yu, Changwang Zhang, and Chongxuan Li. Posecrafter: One-shot personalized video synthesis following flexible pose control. In European Conference on Computer Vision, pages 243–260. Springer, 2024. 3
- [33] Yufan Zhou, Ruiyi Zhang, Kaizhi Zheng, Nanxuan Zhao, Jiuxiang Gu, Zichao Wang, Xin Eric Wang, and Tong Sun. Toffee: Efficient million-scale dataset construction for subject-driven text-to-image generation. arXiv preprint arXiv:2406.09305, 2024. 4

# Supplementary material

## A. Experimental settings

- A.1. Datasets

We remove reference images from the ConsistID-Benchmark that may appear in our training data using both manual and automated filtering methods. (1) Manual filtering: For each reference image in the ConsistID-Benchmark, we compute its cosine similarity with all training images and identify the most similar one. Human evaluators then determine whether the two images depict the same person. If so, all reference images of the corresponding identity are excluded. (2) Automated filtering: All reference images of an identity are discarded if any training image has a cosine similarity greater than 0.45 with one of its reference images.

- A.2. Implementation details

In the first stage, we randomly select one reference image from a set of five for each video. Traditional data augmentation techniques, such as flipping, are not used for face images, as they can cause data augmentation leakage [16], leading the model to learn the augmented data distribution rather than the original distribution. For instance, horizontal flipping may result in incorrectly mirrored faces in generated videos.

- A.3. Baselines

We try our best not to change original settings of baselines to maintain their original capabilities. IDAnimator [9] and ConsisID [29] can produce 16-frame and 49-frame videos at a resolution of 480 × 720, respectively. The multi-identity baseline Ingredients [4] generates 49-frame videos at a resolution of 480 × 720, integrating two distinct identities.

- A.4. Training cost

The first, second, and third stages of training in the single-identity scenario required 3,260, 2,104, and 135 NVIDIA H800 GPU hours, respectively, with the cost of the third stage being negligible.

## B. Multiple identities and subjects

### B.1. Multi-identity scenarios

Through the data construction process of pre-training pairs, we obtain approximately 300,000 videos featuring two identities. For each identity, we determine the sequence order by computing the mean horizontal position of face boxes across all reference images. We discard reference images where the face position does not align with the determined sequence order. Next, we construct cross-video pairs by independently processing each identity within a video. Finally, we collect around 8,000 videos, each of which contains identities that have corresponding cross-video reference images.

A similar strategy is used to construct three-identity training data, resulting in a final dataset of approximately 40,000 pre-training videos. For cross-video pairs, we retain videos in which at least two identities have corresponding cross-video reference images, resulting in about 2,000 videos.

For multiple identities, the pairing cosine similarity ranges between 0.87 and 0.97. We initialize the model using singleidentity pre-training weights and train it only on the first two stages (i.e., the pre-training stage and cross-pair fine-tuning stage). Our findings indicate that single-identity pre-training facilitates multi-identity convergence and enhances identity consistency.

### B.2. Multi-subject scenarios

We select a subset from the cross-video pairs of single-identity, comprising approximately 200,000 videos, where the pairing cosine similarity ranges between 0.87 and 0.97. To achieve virtual try-on, we use Grounded-SAM-22 to detect and segment the clothing of identities. For background-controllable generation, we extract the first frame and use Grounded-SAM-2 to obtain human masks. We then apply SDXL3 to inpaint the masked areas to get bacground images, using a randomly selected classification label from YOLO4 as input prompts.

- 2https://github.com/IDEA-Research/Grounded-SAM-2
- 3https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1
- 4https://github.com/ultralytics/ultralytics

Identity consistency Text alignment Facial editability ArcSim ↑ CurSim ↑ ViCLIP ↑ CLIPDist ↑

Method

- Concat-ID (Stage I) 0.560 0.581 0.237 0.274
- Concat-ID (Stage II) 0.185 0.200 0.248 0.434
- Concat-ID (Stage III) 0.442 0.466 0.242 0.325

Table 3. Quantitative ablation. Stage I, Stage II, and Stage III indicate the pre-training stage, cross-video stage, and trade-off stage of Concat-ID, respectively. The second-best result is underlined. Concat-ID in the third stage demonstrates the optimal balance.

We use weights from single-identity pre-training as initialization and apply only random horizontal flip augmentation to clothing images. Additionally, we introduce random noise to both the background and clothing images during training. In multi-subject scenarios, we only train models on the cross-pair fine-tuning stage.

In this paper, we focus on the single-identity scenario, and improving the performance of Concat-ID in multiple-identity and multi-subject settings is left for future work. To maximize model performance, we independently train different specialized models for specific tasks. The development of a comprehensive model capable of addressing multiple tasks simultaneously remains a direction for future research.

## C. Ablation study

Tab. 3 presents the quantitative ablation study of Concat-ID. The pre-training stage achieves the best identity consistency (i.e., ArcSim and CurSim) but has the worst facial editability (i.e., CLIPDist ). However, the cross-video stage significantly improves CLIPDist but degrades ArcSim and CurSim. In the third stage, Concat-ID obtains the second-best results across all metrics, demonstrating that it achieves an optimal balance. These results highlight the superiority of our multi-stage training strategy, which balances the knowledge learned in different stages to achieve optimal performance in the final stage.

Trade-off pairs can naturally enhance the identity consistency of Concat-ID, as they maintain better alignment between reference images and videos compared to cross-video pairs. An interleaved training strategy—alternating between Stage I for improving identity and Stage II for enhancing editability—can also achieve a favorable trade-off, a method similarly adopted in Imagine-yourself [10]. However, our multi-stage training approach achieves an optimal balance just by adding a third stage where we carefully control identity consistency and sample quantity, showing that a simple design can be highly effective.

