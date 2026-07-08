## DreamRelation: Relation-Centric Video Customization

Yujie Wei1, Shiwei Zhang2∗, Hangjie Yuan2, Biao Gong3, Longxiang Tang2, Xiang Wang2, Haonan Qiu4, Hengjia Li5, Shuai Tan3, Yingya Zhang2, Hongming Shan1†

1Fudan University 2Alibaba Group 3Ant Group 4Nanyang Technological University 5Zhejiang University yjwei22@m.fudan.edu.cn, zhangjin.zsw@alibaba-inc.com, hmshan@fudan.edu.cn

# arXiv:2503.07602v1[cs.CV]10Mar2025

Project page: https://dreamrelation.github.io

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

hugging A bear is hugging with a panda.

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

punching A polar bear is punching a penguin on an icy plain.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

cheering A panda is raising a toast with a red panda in a bamboo grove.

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

shaking hands A dog is shaking hands with a cat in a cyberpunk city.

Figure 1. Relational video customization results of DreamRelation. Given a few exemplar videos, our method can customize specific relations and generalize them to novel domains, where animals mimic human interactions.

alize subject appearances and motions, they still struggle with complex relational video customization, where precise relational modeling and high generalization across subject categories are essential. The primary challenge arises from the intricate spatial arrangements, layout variations, and nuanced temporal dynamics inherent in relations; consequently, current models tend to overemphasize irrelevant visual details rather than capturing meaningful interactions. To address these challenges, we propose DreamRelation, a

### Abstract

Relational video customization refers to the creation of personalized videos that depict user-specified relations between two subjects, a crucial task for comprehending realworld visual content. While existing methods can person-

* Project Leader † Corresponding Author

novel approach that personalizes relations through a small set of exemplar videos, leveraging two key components: Relational Decoupling Learning and Relational Dynamics Enhancement. First, in Relational Decoupling Learning, we disentangle relations from subject appearances using relation LoRA triplet and hybrid mask training strategy, ensuring better generalization across diverse relationships. Furthermore, we determine the optimal design of relation LoRA triplet by analyzing the distinct roles of the query, key, and value features within MM-DiT’s attention mechanism, making DreamRelation the first relational video generation framework with explainable components. Second, in Relational Dynamics Enhancement, we introduce spacetime relational contrastive loss, which prioritizes relational dynamics while minimizing the reliance on detailed subject appearances. Extensive experiments demonstrate that DreamRelation outperforms state-of-the-art methods in relational video customization. Code and models will be made publicly available.

[Figure 21]

[Figure 22]

[Figure 23]

”A bear is hugging with a tiger.”

- (a) Results of the base model Mochi.

[Figure 24]

[Figure 25]

[Figure 26]

“A bear and a tiger stand on their hind legs, each extending their front legs wide open towards each other in a warm embrace. Both animals lean ......”

“A bear is hugging with a tiger.”

- (b) Results of our DreamRelation.

[Figure 27]

[Figure 28]

[Figure 29]

- Figure 2. (a) General Video DiT models like Mochi [69] often struggle to generate unconventional or counter-intuitive interactions, even with detailed descriptions. (b) Our method can customize a specific relation to generate videos on new subjects.

[Figure 30]

[Figure 31]

Generated Video Value Feature

- Figure 3. Averaged value feature across all layers and frames in Mochi. We identify that the relations encompass intricate spatial arrangements, layout variations, and nuanced temporal dynamics, presenting challenges in relational video customization.

### 1. Introduction

Recent advancements in text-to-video (T2V) generation, particularly through powerful video diffusion transformers (DiT) [5, 55, 92], have significantly propelled customized video generation [35, 82, 96]. While existing methods succeed in customizing subject appearances and single-object motions [74, 86, 100], the challenging task of customizing higher-order interactions between subjects (i.e., Relational Video Customization) remains under-explored due to its intrinsic complexity. Enhancing video generation through customized relations is crucial for real-world applications such as filmmaking, enabling a more profound comprehension and production of complex relational visual content.

may hinder high-level relation learning due to severe appearance leakage. Similarly, motion customization methods such as MotionInversion [74] excel in transferring singleobject motions but struggle to precisely capture relational dynamics between two subjects. We identify that the key challenge stems from the complexity inherent in the relations, which involve intricate spatial arrangements, layout variations, and nuanced temporal dynamics. To illustrate this, we visualize the Value features in Fig. 3 and provide detailed analysis in Sec. 3.3. This tangled nature may prevent accurate modeling of relations and cause models to focus on irrelevant subject appearances. This raises a critical research question: How can we decouple relations and subject appearances while accurately modeling relational dynamics to enhance generalizability?

We formulate the task of Relational Video Customization as follows: given exemplar videos representing a relational pattern <subject, relation, subject>, the model aims to generate videos that exhibit the specified relation within the pattern, as shown in Fig. 1. While general text-to-video DiTs like Mochi [69] can generate videos depicting certain relational concepts, they often fail to: (1) produce unconventional or counter-intuitive interactions, such as animals engaging in human-like relationships as illustrated in Figs. 2, even when provided with detailed prompts; (2) generate videos that adhere to precise relational dynamics, such

- as “two people approaching each other from predefined positions.” These issues highlight the need for a novel video generation method to precisely customize desired relations.

To that end, we propose DreamRelation, a relational video customization method that personalizes userspecified relations from exemplar videos through two concurrent processes: relational decoupling learning and relational dynamics enhancement. In relational decoupling learning, we decompose the relational pattern from input

A straightforward approach involves adapting existing video subject or motion customization methods to customize relations between subjects. However, while subject customization techniques like Dreamix [52] capture detailed appearances using low-level reconstruction loss, they

videos into relational and appearance information using devised relation LoRA triplet, a composite LoRA [30] set comprising relation LoRA sets and subject LoRA sets. To facilitate this decoupling, we introduce hybrid mask training strategy that guides the two types of LoRAs to focus on designated regions with corresponding masks, achieved by a LoRA selection strategy and an enhanced diffusion loss based on masks to amplify the learning in target areas.

Furthermore, building on the MM-DiT [17] architecture, we analyze the query, key, and value features within the full attention, and empirically identify that the query, key, and value matrices serve distinct roles in the relation customization task. This insight motivates our design of relation LoRA triplet, particularly in determining the optimal placement of LoRA components within the model architecture to maximize relational customization effectiveness.

To explicitly enhance relational dynamics learning, we propose a novel space-time relational contrastive loss, which emphasizes relational dynamics while reducing the focus on detailed appearances during training. Concretely, we pull relational dynamics representations closer through frame differences in model outputs of videos depicting the same relation, while distancing them from appearance representations derived from single-frame outputs.

We curate a dataset comprising 26 human interactions from publicly available action recognition datasets [44, 61] to comprehensively evaluate relational video customization. Each video is annotated with a textual prompt, and approximately 20 videos per relation type are randomly selected for training. The evaluation is conducted on diverse subjects using 40 designed textual prompts. Extensive experimental results demonstrate that our DreamRelation outperforms state-of-the-art methods in this task.

Our contributions are summarized as follows:

- • We make the first attempt at the Relational Video Customization task by presenting DreamRelation, a method that generates videos depicting customized relations based on the MM-DiT architecture.
- • We devise relation LoRA triplet with hybrid mask training strategy to explicitly decouple relation and subject appearances. To determine the optimal model design of our method, we further analyze the roles of query, key, and value features in MM-DiT full attention.
- • We propose a novel space-time relational contrastive loss to enhance relation learning by emphasizing relational dynamics while reducing focus on appearances.
- • Extensive experimental results demonstrate that DreamRelation achieves state-of-the-art performance on relational video customization.

### 2. Related Work

Text-to-video diffusion models. Text-to-video generative models have achieved breakthroughs in generating

high-quality and diverse videos using textual prompts [1, 3, 4, 16, 22, 27, 38, 43, 57, 64–67, 75–79, 91, 95, 97]. VDM [28] introduces diffusion models into video generation by modeling video distribution in pixel space. ModelScopeT2V [73] and VideoCrafter [10, 12] integrate spatiotemporal blocks for text-to-video generation. With the success of DiT [55] that introduces Transformers [72] as the backbone of diffusion models, the generated video quality has improved with increased parameters [5, 18, 40, 49, 101]. CogVideoX [92] incorporates 3D VAE and expert transformers, enhancing video coherence. Mochi [69] proposes an Asymmetric Diffusion Transformer architecture to scale parameters. HunyuanVideo [39] enhances architecture design and model training, achieving leading performance. These advancements pave the way for relational video customization.

Customized video generation. Building upon achievements in image generation and personalization [7, 14, 15, 19, 26, 56, 59, 60, 81, 87, 98, 103], customized video generation has garnered growing attention [8, 24, 50, 52]. Many studies focus on generating personalized videos using a few subject or facial images [41, 62, 82–86, 96, 99, 102], while others tackle the challenging multi-subject video customization [9, 11, 13, 32, 80]. Besides subject customization, motion customization or motion transfer have also gained significant interest [34, 35, 58, 70, 71, 88, 93, 100]. For example, MotionInversion [74] integrates motion embeddings into the temporal attention of video diffusion models to learn motion dynamics. While these methods effectively capture the subject appearances or single-object motions, the challenging task of customizing interactions between two subjects remains underexplored due to its inherent complexity. In this work, we pioneer this relational video customization task by presenting DreamRelation, which can personalize specific relations and generate diverse videos aligned with text prompts.

Relation generation. Early works on relational image generation focus on human-object interactions using additional conditions like bounding boxes [20, 29, 31]. Recently, inspired by image customization methods, several works have explored relational image customization to personalize user-specific interactions from a few relational images [21, 33, 63]. For instance, ReVersion [33] utilizes inversion techniques to capture relational information in the text embedding space. Despite these advancements, existing methods are confined to the relatively simple relations depicted in images. Direct adaptation of these image-based methods for relational video customization often leads to inaccurate relation modeling since dynamic and sequential interactions cannot be fully represented in a single image. In contrast, we design our method based on Video DiT architecture and precisely model relations through relational decoupling learning and relational dynamics enhancement.

###### MM-DiT Block × 𝒏

[Figure 32]

[Figure 33]

Relational Dynamics Enhancement

[Figure 34]

A person is shaking hands with a person.

[Figure 35]

[Figure 36]

Attention FFN

Anchor Features

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Frame

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Noise 𝜀 Relation LoRA triplet

FFN LoRA

Difference

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Model Output

Pairwise Differences

Contrastive Loss

Relational Decoupling Learning

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Supervision

Training

[Figure 57]

Relation LoRA

Q K

SingleFrame Output

[Figure 58]

…

Frozen

[Figure 59]

- Subject1 Mask Loss

Relation Mask Loss

- Subject2 Mask Loss

- Subject1 LoRA
- Subject2 LoRA

Negative Features

Positive Features

Sample

Loss Supervision

| |
|---|
| |
|[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]|
| |
| |

Other Pairwise Differences

[Figure 63]

V

Only in training

Memory Bank

Hybrid Mask Training Strategy

1D feature

| |
|---|

Relation LoRA Triplet

- Figure 4. Overall framework of DreamRelation. Our method decomposes relational video customization into two concurrent processes.

(1) In Relational Decoupling Learning, Relation LoRAs in relation LoRA triplet capture relational information, while Subject LoRAs focus on subject appearances. This decoupling process is guided by hybrid mask training strategy based on their corresponding masks. (2) In Relational Dynamics Enhancement, the proposed space-time relational contrastive loss pulls relational dynamics features (anchor and positive features) from pairwise differences closer, while pushing them away from appearance features (negative features) of single-frame outputs. During inference, subject LoRAs are excluded to prevent introducing undesired appearances and enhance generalization.

### 3. DreamRelation

their relational patterns as a triplet <subject, relation, subject>, denoted as <S1, R, S2> for brevity, where S1 and S2 are two subjects and R is the relation [94].

Our DreamRelation aims to generate videos depicting a specified relation expressed in a few exemplar videos while aligning with textual prompts, as illustrated in Fig. 4. We begin by introducing preliminaries in Sec. 3.1. We then detail relational decoupling learning and relational dynamics enhancement in Secs. 3.2 and 3.4, respectively, along with an analysis of the query, key, and value features in Sec. 3.3.

To differentiate relations and subject appearances in the relational pattern, we introduce relation LoRA triplet, a composite LoRA set comprising Relation LoRAs to model relational information and two Subject LoRAs to capture appearance information, as depicted in Fig. 4. Specifically, we inject Relation LoRAs into the query and key matrices of the MM-DiT full attention. Concurrently, we design two Subject LoRAs corresponding to the two subjects involved in the relation and inject them into the value matrix. This design is motivated by our empirical findings that the query, key, and value matrices serve distinct roles within the MMDiT full attention. More details on the analysis are provided in Sec. 3.3. Additionally, we devise an FFN LoRA to refine the outputs of the Relation and Subject LoRAs and inject it into the linear layers of full attention. Note that the two branches of text and vision tokens in MM-DiT are processed by different LoRA sets.

#### 3.1. Preliminaries of Video DiT

Text-to-video diffusion transformers (DiTs) show growing attention due to their capacity to generate high-fidelity, diverse, and long-duration video. Current Video DiTs [69, 92] predominantly adopt MM-DiT [17] architecture with full attention and employ diffusion processes [26] in latent space with a 3D VAE [36]. Given latent code z0 ∈ Rf×h×w×c from video data x0 ∈ RF×H×W×3 with its textual prompt c, the optimization process is defined as:

L(θ) = Ez,ϵ,c,t ∥ϵ − ϵθ(zt,c,t)∥22 , (1)

Hybrid mask training strategy. To achieve the decoupling of relational and appearance information in the introduced relation LoRA triplet, we propose hybrid mask training strategy (HMT) to guide Relation and Subject LoRAs to focus on designated regions using corresponding masks. We first employ Grounding DINO [45] and SAM [37] to derive masks for the two individuals in a video, indicated as Subject Masks MS

where ϵ ∈ N(0,1) is random noise from a Gaussian distribution, and zt is a noisy latent code at timestep t based on z0 with the predefined noise schedule. In this work, we choose Mochi [69] as our base Video DiT model.

#### 3.2. Relational Decoupling Learning

. Inspired by representative relation detection approaches [68, 89, 90] that utilize minimum enclosing rectangles to delineate subject-object interaction zones, we define the Relation Mask MR as the union of the two Subject Masks to indicate the relation area. Since the 3D VAE in Video DiT compresses the video’s temporal

and MS

1

2

Relation LoRA triplet. To customize complex relations between subjects, we decompose the relational pattern from exemplar videos into distinct components emphasizing subject appearances and relations. Formally, given a few videos depicting interactions between two subjects, we represent

Value Query Key

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Query-Key Matrices Subspace Similarity

[Figure 70]

A cat wearing blue glasses and a birthday hat is lying on the sofa.

Value Query Key

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Query-Value Matrices Subspace Similarity

[Figure 75]

A boy is shaking hands with a girl in the park.

Key-Value Matrices Subspace Similarity

(a) Visualization of Query, Key, and Value Features in MM-DiT Full Attention.

(b) Subspace similarity between Query, Key, and Value Matrices.

- Figure 5. Features and subspace similarity analysis of MM-DiT. (a) Value features across different videos encapsulate rich appearance information, and relational information often intertwines with these appearance cues. Meanwhile, query and key features exhibit similar patterns that differ from those of value features. (b) We perform singular value decomposition on the query, key, and value matrices of each MM-DiT block and compute the similarity of the subspaces spanned by their top-k left singular vectors, indicating query and key matrices share more common information while remaining independent of the value matrix.

Visualization analysis. We start with two types of videos: a single-subject video with multiple attributes, and a twosubject interaction video, as illustrated in Fig. 5(a). We compute the averaged query, key, and value features across all layers and attention heads at timestep 60, focusing solely on those associated with vision tokens. These features are then reshaped into an f ×h×w format, and we visualize the averaged features across all frames with shape h×w. From the observations in Fig. 5(a), we draw two conclusions:

dimensions by a factor of Tc, we average the masks over every Tc frame to represent the latent masks.

We then devise a LoRA selection strategy and an enhanced diffusion loss for better disentanglement during training. Specifically, we randomly select either the Relation LoRAs or one type of Subject LoRAs in relation LoRA triplet to update for each training iteration. When the Relation LoRAs are chosen, the two Subject LoRAs are trained simultaneously to provide appearance cues, assisting the Relation LoRAs in concentrating on relational information. This process facilitates the decoupling of relational and appearance information. The FFN LoRAs are consistently engaged throughout training to refine outputs from the selected Relation or Subject LoRAs.

- 1) Value features across different videos encapsulate rich

appearance information, and relational information often intertwines with these appearance cues. For instance, in the single-subject video, high-value feature responses occur at locations like “blue glasses” and “birthday hat.” In the two-subject video, high values are observed both in regions of relations (e.g., handshakes) and appearances (e.g., human face and clothing), indicating the entanglement of relational and appearance information within the features.

- 2) Query and key features exhibit highly abstract yet sim-

Following LoRA selection, we apply the corresponding masks to amplify the loss weight within the focused area, which can be defined as:

Lrec = Ez,ϵ,c,t λmMl + 1 · ϵ − ϵθ(zt,c,t) 22, (2)

ilar patterns, distinctly diverging from the value features. Unlike the obvious appearance information in value features, query, and key features exhibit homogeneity across different videos, clearly differing from value features. To further validate this point, we analyze query, key, and value matrices from a quantitative perspective.

where l ∈ {S1,S2,R} indicates the selected mask type, and λm is the mask weight. By employing the LoRA selection strategy and the enhanced diffusion loss, Relation and Subject LoRAs are encouraged to concentrate on their designated area, facilitating effective relation customization and improving the generalization capacity.

Subspace similarity analysis. We further analyze the similarity of the subspace spanned by the singular vectors of the query, key, and value matrix weights from the base Video DiT model Mochi. This similarity reflects the degree of overlap in contained information between two matrices. For the query and key matrices, we apply singular value decomposition to obtain left-singular unitary matrices UQ and UK. Following [30, 47], we select the top r singular vectors from UQ and UK, and measure their normalized subspace similarity based on the Grassmann distance [23]

Inference. During inference, we exclude Subject LoRAs to prevent undesired appearances and inject only Relation LoRAs and FFN LoRAs into the base Video DiT to maintain learned relations and enhance generalization.

#### 3.3. Analysis on Query, Key, and Value Features

To determine the optimal model design of our method, we analyze the roles of query, key, and value features or matrices in MM-DiT’s full attention through visualization and singular value decomposition, revealing their impacts on relational video customization.

using 1r UQr⊤UKr 2F. The other similarities are calculated in a similar way. The results in Fig. 5(b) demonstrate that

the subspaces of the query and key matrices are highly similar, whereas their similarity to the value matrix is minimal. This suggests that the query and key matrices in MM-DiT share more common information while remaining largely independent of the value matrix. In other words, the query and key matrices exhibit a strongly non-overlapping relationship with the value matrix, which facilitates the design of our decoupling learning. This finding is consistent with the visualization results in Fig. 5(a).

Building on these observations, we empirically argue that the query, key, and value matrices serve distinct roles in relational video customization, motivating our design of relation LoRA triplet. Specifically, given that value features are rich in appearance information, we inject Subject LoRAs into the value matrix to focus on learning appearances. In contrast, due to the homogeneity observed in the query and key features and their non-overlapping nature with the value matrix, which facilitates decoupling learning, we inject Relation LoRAs into both the query and key matrices to better disentangle relations from appearances. The experimental results in Tab. 3 confirm our analysis and verify that this design achieves the best performance. We believe our findings can advance research in video customization based on MM-DiT architecture.

#### 3.4. Relational Dynamics Enhancement

To explicitly enhance relational dynamics learning, we propose a novel space-time relational contrastive loss (RCL), which emphasizes relational dynamics while reducing the focus on detailed appearance during training. Specifically,

- at each timestep t, we compute the pairwise differences of the model output along the frame dimension, denoted as ϵ¯ ∈ R(f−1)×h×w×c. We then reduce dependency on pixellevel information by averaging these differences across the spatial dimensions, resulting in 1D relational dynamics features A ∈ R(f−1)×c, which serve as anchor features.

Subsequently, we sample npos 1D relational dynamics features from other relation videos as positive samples P ∈ R(f−1)×n

pos×c. For each frame in A, we sample nneg 1D features from single-frame model outputs ϵi ∈ R1×h×w×c as negative samples N ∈ R(f−1)×n

neg×c, which capture appearance information while excluding relational dynamics.

Our objective is to learn representations with relational dynamics by pulling together the pairwise differences from different videos depicting the same relation, while distancing them from spatial features of single-frame outputs to mitigate appearance and background leakage. Following InfoNCE [51, 54] loss, we formulate the proposed loss as:

f−1

LRCL = log

i=1

npos

⊤ i Pij

exp(A

−

τ ) npos

j=1

, (3)

nneg

⊤ i Pij

⊤ i Nik

exp(A

exp(A

τ ) +

τ )

j=1

k=1

where τ is the temperature hyper-parameter.

Additionally, we maintain a memory bank M to store and update the positive and negative samples. Both positive and negative samples are randomly selected from the 1D features of current batch videos and previously seen videos. This online dynamic update strategy can enlarge the number of positive and negative samples, enhancing the contrastive learning effect and training stability. At each iteration, we store all current anchor features A and the 1D features of ϵi into M. The memory bank is implemented as a First In, First Out (FIFO) queue.

Overall, the training loss Ltotal consists of both reconstruction and contrastive learning loss, defined as:

Ltotal = Lrec + λ1LRCL, (4) where λ1 is the loss balancing weight.

### 4. Experiment

#### 4.1. Experimental Setup

Datasets. We conduct experiments on the NTU RGB+D Action Recognition Dataset [44, 61]. We select 26 types of human relations, such as handshakes and hugs, each labeled with a text prompt like “A person is shaking hands with a person.” For evaluation, we design 10×26 prompts with uncommon subject interactions, such as “A dog is shaking hands with a cat”, to assess generalization to novel domains. More details are provided in Appendix A.1.

Baselines. Given the absence of existing methods for relational video customization, we define four baseline categories: 1) The base model Mochi. 2) Direct LoRA finetuning. 3) Adapted relational image customization methods. We reproduce ReVersion [33] on Mochi for relational video customization. 4) Motion customization methods, which mostly rely on Temporal Attention Layers that are absent in MM-DiT, face challenges in direct adaptation. Thus, we choose the recent and adaptable MotionInversion [74] as a baseline, reproducing it on Mochi for comparison.

Evaluation metrics. We evaluate our method by focusing on four aspects: 1) Relation Accuracy. Instead of using biased classifiers trained on test sets with limited diversity like previous methods [21, 33], which hinders test accuracy and generalizability, we propose the Relation Accuracy metric to assess relations using advanced Vision-Language Models (VLMs). Specifically, we input generated videos to QwenVL-Max [2], a leading VQA model, asking if the video matches the specified relation, and convert the yes/no responses into a relation accuracy percentage. We repeat this process 10 times to calculate the average accuracy. 2) Text Alignment. We employ CLIP image-text similarity (CLIPT) to measure alignment with text prompts. 3) Temporal Consistency, which computes the average cosine similarity across consecutive frames [16]. 4) Video Quality. We use

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Relational Video

###### …

…

A person is shaking hands with a person.

A person is high-fiving with a person.

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Base Model

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Direct LoRA FT

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

ReVersion

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Motion Inversion

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Ours

A dog is shaking hands with a cat.

A bear is high-fiving with a raccoon in a serene forest clearing.

- Figure 6. Qualitative comparison results. Our method outperforms all baselines in precisely capturing the intended relation and mitigating appearance and background leakage.

Relation Accuracy

Temporal Consistency

Mochi

[Figure 114]

[Figure 115]

Direct LoRA FT

ReVersion MotionInversion

Method

CLIP-T

FVD↓

DreamRelation

80 60 40 20 0

Mochi (base model) [69] 0.2623±0.04 0.3237 0.9888 2047.37 Direct LoRA finetuning 0.3258±0.05 0.2966 0.9945 2229.08 ReVersion [33] 0.2690±0.01 0.3013 0.9921 2682.69 MotionInversion [74] 0.3151±0.03 0.3217 0.9855 2084.51 DreamRelation 0.4452±0.01 0.3248 0.9954 2079.87

shaking hands

[Figure 116]

Ours

[Figure 117]

shaking hands

Relation Alignment

Text Alignment

Overall Quality

“A dog is shaking hands with a cat.”

Table 1. Quantitative comparison results.

- (a) Visualization of Attention Maps. (b) Human evaluation results. Figure 7. (a) Our method focuses on the desired relational region.
- (b) Our method is most preferred by users across all aspects.

FVD to evaluate the video quality. The reference videos are 800 videos from the AnimalKingdom test dataset [53].

Implementation details. We adopt Mochi [69] as our base model. During training, we use AdamW [48] optimizer with a learning rate of 2e-4. The weight decay is set to 0.01, and the training iteration is 2400. We set LoRA rank to 16, λm to 50, and λ1 to 0.01. The resolution of generated videos is 61×480×848, and the batch size is 1. We set npos to 4 and nneg to 10. The memory bank size is set to 64, and τ is 0.07. During inference, we generate 30-fps videos using Mochi’s default Euler Discrete method [42, 46] with 64 steps. The classifier-free guidance [25] scale is 6.0.

while other methods like MotionInversion cannot capture desired relational dynamics due to the complexity inherent in relations. In contrast, our DreamRelation precisely generates videos with intended relations and diverse subjects, effectively preventing appearance and background leakage. Quantitative results. Tab. 1 presents the quantitative comparison results. Direct LoRA finetuning improves the base model’s Relation Accuracy but suffers from reduced CLIP-T and FVD due to appearance leakage. Inversionbased methods like ReVersion and MotionInversion achieve better CLIP-T than finetuning but fail to model desired relations accurately. In contrast, while comparable to the base model in FVD, our DreamRelation consistently surpasses baselines across other metrics, verifying its effectiveness.

#### 4.2. Main Results

Qualitative results. Qualitative comparisons in Fig. 6 reveal that all baseline methods, including the base model Mochi, fail to generate videos that match the relations defined in exemplar videos. For example, Direct LoRA finetuning struggles with appearance and background leakage,

Attention map analysis. To verify the effectiveness of our method, we compute averaged attention maps from all

[Figure 118]

[Figure 119]

Relation LoRA

Subject LoRA

Relation Accuracy

Temporal Consistency

[Figure 120]

[Figure 121]

CLIP-T

FVD↓

Relational Video

…

V Q, K 0.3444±0.02 0.3225 0.9953 2233.48 Q K, V 0.3921±0.03 0.3301 0.9951 2284.65

K, V Q 0.3937±0.04 0.3196 0.9954 2180.27 Q, K V 0.4452±0.01 0.3248 0.9954 2079.87

A person is patting a person’s shoulder.

[Figure 122]

[Figure 123]

[Figure 124]

Table 3. Ablation study of Relation LoRA position.

w/o HMT

Relation Accuracy

Temporal Consistency

Method

CLIP-T

FVD↓

[Figure 125]

[Figure 126]

[Figure 127]

MotionInversion [74] 0.3151±0.03 0.3217 0.9855 2084.51 MotionInversion + RCL 0.3633±0.05 0.3181 0.9862 2063.30

w/o RCL

Table 4. Effects of space-time relational contrastive loss on motion customization method (MotionInversion).

[Figure 128]

[Figure 129]

[Figure 130]

Ours

contrastive loss reduces background leakage but results in videos exhibiting inaccurate relations.

A bear is patting a tiger’s shoulder on the grassland.

Quantitative results in Tab. 2 show that removing hybrid mask training strategy or space-time relational contrastive loss degrades performance across all metrics, confirming that each component is crucial to overall performance; see Appendix A.3 for more ablation studies.

###### Figure 8. Qualitative ablation study on each component.

Relation Accuracy

Temporal Consistency

Method

CLIP-T

FVD↓

w/o HMT 0.3574±0.02 0.3244 0.9938 2248.52 w/o RCL 0.3416±0.03 0.3185 0.9953 2136.95 w/o Relation LoRAs 0.3626±0.02 0.3035 0.9950 2318.49 w/o Subject LoRAs 0.3769±0.04 0.3147 0.9949 2408.59 w/o FFN LoRAs 0.4021±0.03 0.3241 0.9914 2369.98 ours 0.4452±0.01 0.3248 0.9954 2079.87

Ablation on each LoRA in relation LoRA triplet. We conduct ablation studies to verify each LoRA’s effects. The results in Tab. 2 indicate that removing Relation LoRAs or Subject LoRAs significantly reduces Relation Accuracy and CLIP-T due to insufficient decoupling of appearance and relational information. Excluding FFN LoRAs also lowers accuracy, highlighting the need for refinement.

Table 2. Ablation studies on effects of hybrid mask training strategy (HMT), space-time relational contrastive loss (RCL), and each type of LoRA. Removing any of the above components significantly reduces the overall performance.

Ablation on Relation LoRAs position. To determine the optimal position of Relation LoRAs, we experiment with different settings in the query (Q), key (K), and value (V) matrices, as shown in Tab. 3. Inserting Relation LoRAs to the V matrix results in the lowest Relation Accuracy, likely because V features predominantly exhibit appearance information, making it challenging to accurately capture the desired relations. In contrast, placing Relation LoRAs in the Q matrix or KV matrices is suboptimal since the overlapping nature of the QK matrices hinders their ability to process different information separately, which is not conducive to decoupling relations from appearances. In contrast, inserting Relation LoRAs to the QK matrices achieves the best Relation Accuracy, consistent with our analysis of full attention in Fig. 5.

layers and heads, extracting values for text tokens of relations like “shaking hands” and all vision tokens [6]. These attention maps are reshaped and visualized in Fig. 7(a). We observe that the base model’s attention map for “shaking hands” is messy, leading to poor generation. In contrast, our method’s attention map effectively focuses on the relational area, producing more natural results and demonstrating its capability to capture relational information.

User study. We conduct user studies to evaluate our DreamRelation, involving 15 annotators who rate 180 video groups generated by four methods. Each group contains four generated videos, a reference video, and a textual prompt. Evaluations are based on majority votes in three aspects: Relation Alignment, Text Alignment, and Overall Quality. Results in Fig. 7(b) indicate that our method is most preferred by users across all aspects. More details about the user study are provided in Appendix A.2.

Ablation on space-time relational contrastive loss (RCL). To verify the effectiveness of RCL among different methods, we integrate it with MotionInversion [74]. Results in Tab. 4 show that incorporating RCL enhances Relation Accuracy and Temporal Consistency while maintaining comparable CLIP-T, demonstrating its potential for generalization across different methods.

#### 4.3. Ablation Studies

Ablation on each component. We perform an ablation study on the effects of each component, as shown in Fig. 8. Without hybrid mask training strategy, the model generates the desired relations but experiences some background leakage due to incomplete decoupling of relational and appearance information. Omitting space-time relational

### 5. Conclusion

In this paper, we present DreamRelation, a novel relational video customization method that accurately models com-

plex relations defined in exemplar videos through relational decoupling learning and relational dynamics enhancement. We introduce relation LoRA triplet to decompose relations into appearance and relational information and further enhance this decoupling with hybrid mask training strategy. Our analysis of query, key, and value features in MMDiT’s full attention motivates and offers interpretability for our model design. To further enhance relation dynamics learning, we propose space-time relational contrastive loss, which prioritizes relational dynamics over detailed appearances. Extensive experimental results demonstrate the superior customization capabilities of DreamRelation.

Limitations. Existing metrics for relation accuracy may not fully capture the customization capabilities of models. While the use of VLMs simplifies evaluation and reduces bias, the metric relies on VLM’s capabilities; future work should develop metrics that align better with human perception.

### References

- [1] Jie An, Songyang Zhang, Harry Yang, Sonal Gupta, JiaBin Huang, Jiebo Luo, and Xi Yin. Latent-shift: Latent diffusion with temporal shift for efficient text-to-video generation. arXiv preprint arXiv:2304.08477, 2023. 3
- [2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023. 6, 14
- [3] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Yuanzhen Li, Tomer Michaeli, et al. Lumiere: A spacetime diffusion model for video generation. arXiv preprint arXiv:2401.12945, 2024. 3
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3
- [5] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators.

2024. 2, 3

- [6] Minghong Cai, Xiaodong Cun, Xiaoyu Li, Wenze Liu, Zhaoyang Zhang, Yong Zhang, Ying Shan, and Xiangyu Yue. Ditctrl: Exploring attention control in multi-modal diffusion transformer for tuning-free multi-prompt longer video generation. arXiv preprint arXiv:2412.18597, 2024. 8
- [7] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22560–22570, 2023. 3

- [8] Hila Chefer, Shiran Zada, Roni Paiss, Ariel Ephrat, Omer Tov, Michael Rubinstein, Lior Wolf, Tali Dekel, Tomer Michaeli, and Inbar Mosseri. Still-moving: Customized video generation without customized video data. arXiv preprint arXiv:2407.08674, 2024. 3
- [9] Hong Chen, Xin Wang, Guanning Zeng, Yipeng Zhang, Yuwei Zhou, Feilin Han, and Wenwu Zhu. Videodreamer: Customized multi-subject text-to-video generation with disen-mix finetuning. arXiv preprint arXiv:2311.00990,

2023. 3

- [10] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 3
- [11] Hong Chen, Xin Wang, Yipeng Zhang, Yuwei Zhou, Zeyang Zhang, Siao Tang, and Wenwu Zhu. Disenstudio: Customized multi-subject text-to-video generation with disentangled spatial control. arXiv preprint arXiv:2405.12796, 2024. 3
- [12] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7310– 7320, 2024. 3
- [13] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, and Sergey Tulyakov. Multi-subject open-set personalization in video generation. arXiv preprint arXiv:2501.06187, 2025. 3
- [14] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and William W Cohen. Subject-driven text-to-image generation via apprenticeship learning. Advances in Neural Information Processing Systems, 36, 2024. 3
- [15] Yusuf Dalva and Pinar Yanardag. Noiseclr: A contrastive learning approach for unsupervised discovery of interpretable directions in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24209–24218, 2024. 3
- [16] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023. 3, 6
- [17] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning,

2024. 3, 4

- [18] Weichen Fan, Chenyang Si, Junhao Song, Zhenyu Yang, Yinan He, Long Zhuo, Ziqi Huang, Ziyue Dong, Jingwen He, Dongwei Pan, et al. Vchitect-2.0: Parallel transformer for scaling up video diffusion models. arXiv preprint arXiv:2501.08453, 2025. 3

- [19] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3
- [20] Chen Gao, Si Liu, Defa Zhu, Quan Liu, Jie Cao, Haoqian He, Ran He, and Shuicheng Yan. Interactgan: Learning to generate human-object interaction. In Proceedings of the 28th ACM International Conference on Multimedia, pages 165–173, 2020. 3
- [21] Mengmeng Ge, Xu Jia, Takashi Isobe, Xiaomin Li, Qinghe Wang, Jing Mu, Dong Zhou, Li Wang, Huchuan Lu, Lu Tian, et al. Customizing text-to-image generation with inverted interaction. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 10901–10909,

2024. 3, 6

- [22] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 3
- [23] Jihun Hamm and Daniel D Lee. Grassmann discriminant analysis: a unifying view on subspace-based learning. In Proceedings of the 25th international conference on Machine learning, pages 376–383, 2008. 5
- [24] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, Man Zhou, and Jie Zhang. Idanimator: Zero-shot identity-preserving human video generation. arXiv preprint arXiv:2404.15275, 2024. 3
- [25] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 7
- [26] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3, 4
- [27] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 3
- [28] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models. arXiv preprint arXiv:2204.03458, 2022. 3
- [29] Jiun Tian Hoe, Xudong Jiang, Chee Seng Chan, Yap-Peng Tan, and Weipeng Hu. Interactdiffusion: Interaction control in text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6180–6189, 2024. 3
- [30] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 3, 5
- [31] Tianyu Hua, Hongdong Zheng, Yalong Bai, Wei Zhang, Xiao-Ping Zhang, and Tao Mei. Exploiting relationship for complex-scene image generation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1584– 1592, 2021. 3
- [32] Yuzhou Huang, Ziyang Yuan, Quande Liu, Qiulin Wang, Xintao Wang, Ruimao Zhang, Pengfei Wan, Di Zhang, and

- Kun Gai. Conceptmaster: Multi-concept video customization on diffusion transformer models without test-time tuning. arXiv preprint arXiv:2501.04698, 2025. 3
- [33] Ziqi Huang, Tianxing Wu, Yuming Jiang, Kelvin CK Chan, and Ziwei Liu. Reversion: Diffusion-based relation inversion from images. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024. 3, 6, 7, 14
- [34] Hyeonho Jeong, Jinho Chang, Geon Yeong Park, and Jong Chul Ye. Dreammotion: Space-time self-similar score distillation for zero-shot video editing. In European Conference on Computer Vision, pages 358–376. Springer, 2024. 3
- [35] Hyeonho Jeong, Geon Yeong Park, and Jong Chul Ye. Vmc: Video motion customization using temporal attention adaption for text-to-video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9212–9221, 2024. 2, 3
- [36] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 4
- [37] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 4
- [38] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jos´e Lezama, Jonathan Huang, Rachel Hornung, Hartwig Adam, Hassan Akbari, Yair Alon, Vighnesh Birodkar, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023. 3
- [39] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 3
- [40] PKU-Yuan Lab and Tuzhan AI etc. Open-sora: Democratizing efficient video production for all, 2024. https://doi.org/10. 5281/zenodo.10948109. 3
- [41] Hengjia Li, Haonan Qiu, Shiwei Zhang, Xiang Wang, Yujie Wei, Zekun Li, Yingya Zhang, Boxi Wu, and Deng Cai. Personalvideo: High id-fidelity video customization without dynamic and semantic degradation. arXiv preprint arXiv:2411.17048, 2024. 3
- [42] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 7
- [43] Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model. arXiv preprint arXiv:2411.19108,

2024. 3

- [44] Jun Liu, Amir Shahroudy, Mauricio Perez, Gang Wang, Ling-Yu Duan, and Alex C Kot. Ntu rgb+ d 120: A large-scale benchmark for 3d human activity understanding. IEEE transactions on pattern analysis and machine intelligence, 42(10):2684–2701, 2019. 3, 6, 14
- [45] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun

- Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 4
- [46] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 7
- [47] Zhihang Liu, Jun Li, Hongtao Xie, Pandeng Li, Jiannan Ge, Sun-Ao Liu, and Guoqing Jin. Towards balanced alignment: Modal-enhanced semantic modeling for video moment retrieval. In Proceedings of the AAAI conference on artificial intelligence, pages 3855–3863, 2024. 5
- [48] I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 7
- [49] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 3
- [50] Ze Ma, Daquan Zhou, Chun-Hsiao Yeh, Xue-She Wang, Xiuyu Li, Huanrui Yang, Zhen Dong, Kurt Keutzer, and Jiashi Feng. Magic-me: Identity-specific video customized diffusion. arXiv preprint arXiv:2402.09368, 2024. 3
- [51] Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. End-to-end learning of visual representations from uncurated instructional videos. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9879– 9889, 2020. 6
- [52] Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex Rav Acha, Yossi Matias, Yael Pritch, Yaniv Leviathan, and Yedid Hoshen. Dreamix: Video diffusion models are general video editors. arXiv preprint arXiv:2302.01329, 2023. 2, 3
- [53] Xun Long Ng, Kian Eng Ong, Qichen Zheng, Yun Ni, Si Yong Yeo, and Jun Liu. Animal kingdom: A large and diverse dataset for animal behavior understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19023–19034,

2022. 7

- [54] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 6
- [55] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195– 4205, 2023. 2, 3
- [56] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 3
- [57] Zhiwu Qing, Shiwei Zhang, Jiayu Wang, Xiang Wang, Yujie Wei, Yingya Zhang, Changxin Gao, and Nong Sang. Hierarchical spatio-temporal decoupling for text-to-video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6635– 6645, 2024. 3

- [58] Yixuan Ren, Yang Zhou, Jimei Yang, Jing Shi, Difan Liu, Feng Liu, Mingi Kwon, and Abhinav Shrivastava. Customize-a-video: One-shot motion customization of text-to-video diffusion models. arXiv preprint arXiv:2402.14780, 2024. 3
- [59] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 3
- [60] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 3
- [61] Amir Shahroudy, Jun Liu, Tian-Tsong Ng, and Gang Wang. Ntu rgb+ d: A large scale dataset for 3d human activity analysis. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1010–1019,

2016. 3, 6, 14

- [62] D She, Mushui Liu, Jingxuan Pang, Jin Wang, Zhen Yang, Wanggui He, Guanghao Zhang, Yi Wang, Qihan Huang, Haobin Tang, et al. Customvideox: 3d reference attention driven dynamic adaptation for zero-shot customized video diffusion transformers. arXiv preprint arXiv:2502.06527,

2025. 3

- [63] Qingyu Shi, Lu Qi, Jianzong Wu, Jinbin Bai, Jingbo Wang, Yunhai Tong, Xiangtai Li, and Ming-Husan Yang. Relationbooth: Towards relation-aware customized object generation. arXiv preprint arXiv:2410.23280, 2024. 3
- [64] Shuai Tan, Biao Gong, Yutong Feng, Kecheng Zheng, Dandan Zheng, Shuwei Shi, Yujun Shen, Jingdong Chen, and Ming Yang. Mimir: Improving video diffusion models for precise text understanding. arXiv preprint arXiv:2412.03085, 2024. 3
- [65] Shuai Tan, Biao Gong, Xiang Wang, Shiwei Zhang, Dandan Zheng, Ruobing Zheng, Kecheng Zheng, Jingdong Chen, and Ming Yang. Animate-x: Universal character image animation with enhanced motion representation. arXiv preprint arXiv:2410.10306, 2024.
- [66] Shuai Tan, Bin Ji, Mengxiao Bi, and Ye Pan. Edtalk: Efficient disentanglement for emotional talking head synthesis. In European Conference on Computer Vision, pages 398–

416. Springer, 2024.

- [67] Shuai Tan, Bin Ji, and Ye Pan. Flowvqtalker: High-quality emotional talking face generation through normalizing flow and quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26317–26327, 2024. 3
- [68] Kaihua Tang, Yulei Niu, Jianqiang Huang, Jiaxin Shi, and Hanwang Zhang. Unbiased scene graph generation from biased training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3716– 3725, 2020. 4
- [69] Genmo Team. Mochi 1. https://github.com/ genmoai/models, 2024. 2, 3, 4, 7, 14

- [70] Shuyuan Tu, Qi Dai, Zhi-Qi Cheng, Han Hu, Xintong Han, Zuxuan Wu, and Yu-Gang Jiang. Motioneditor: Editing video motion via content-aware diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7882–7891, 2024. 3
- [71] Shuyuan Tu, Qi Dai, Zihao Zhang, Sicheng Xie, ZhiQi Cheng, Chong Luo, Xintong Han, Zuxuan Wu, and Yu-Gang Jiang. Motionfollower: Editing video motion via lightweight score-guided diffusion. arXiv preprint arXiv:2405.20325, 2024. 3
- [72] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 3
- [73] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 3
- [74] Luozhou Wang, Ziyang Mai, Guibao Shen, Yixun Liang, Xin Tao, Pengfei Wan, Di Zhang, Yijun Li, and Yingcong Chen. Motion inversion for video customization. arXiv preprint arXiv:2403.20193, 2024. 2, 3, 6, 7, 8, 14
- [75] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. Advances in Neural Information Processing Systems, 36:7594–7611, 2023. 3
- [76] Xiang Wang, Shiwei Zhang, Han Zhang, Yu Liu, Yingya Zhang, Changxin Gao, and Nong Sang. Videolcm: Video latent consistency model. arXiv preprint arXiv:2312.09109, 2023.
- [77] Xiang Wang, Shiwei Zhang, Changxin Gao, Jiayu Wang, Xiaoqiang Zhou, Yingya Zhang, Luxin Yan, and Nong Sang. Unianimate: Taming unified video diffusion models for consistent human image animation. arXiv preprint arXiv:2406.01188, 2024.
- [78] Xiang Wang, Shiwei Zhang, Hangjie Yuan, Zhiwu Qing, Biao Gong, Yingya Zhang, Yujun Shen, Changxin Gao, and Nong Sang. A recipe for scaling up text-to-video generation with text-free videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6572–6582, 2024.
- [79] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023. 3
- [80] Zhao Wang, Aoxue Li, Enze Xie, Lingting Zhu, Yong Guo, Qi Dou, and Zhenguo Li. Customvideo: Customizing textto-video generation with multiple subjects. arXiv preprint arXiv:2401.09962, 2024. 3
- [81] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-toimage generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15943– 15953, 2023. 3
- [82] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hong-

- ming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6537–6549, 2024. 2, 3
- [83] Yujie Wei, Shiwei Zhang, Hangjie Yuan, Xiang Wang, Haonan Qiu, Rui Zhao, Yutong Feng, Feng Liu, Zhizhong Huang, Jiaxin Ye, et al. Dreamvideo-2: Zero-shot subjectdriven video customization with precise motion control. arXiv preprint arXiv:2410.13830, 2024.
- [84] Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, and Kai Chen. Motionbooth: Motion-aware customized text-tovideo generation. arXiv preprint arXiv:2406.17758, 2024.
- [85] Tao Wu, Yong Zhang, Xiaodong Cun, Zhongang Qi, Junfu Pu, Huanzhang Dou, Guangcong Zheng, Ying Shan, and Xi Li. Videomaker: Zero-shot customized video generation with the inherent force of video diffusion models. arXiv preprint arXiv:2412.19645, 2024.
- [86] Tao Wu, Yong Zhang, Xintao Wang, Xianpan Zhou, Guangcong Zheng, Zhongang Qi, Ying Shan, and Xi Li. Customcrafter: Customized video generation with preserving motion and concept composition abilities. arXiv preprint arXiv:2408.13239, 2024. 2, 3
- [87] Chao Xu, Yang Liu, Jiazheng Xing, Weida Wang, Mingze Sun, Jun Dan, Tianxin Huang, Siyuan Li, Zhi-Qi Cheng, Ying Tai, et al. Facechain-imagineid: Freely crafting highfidelity diverse talking faces from disentangled audio. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1292–1302, 2024. 3
- [88] Chao Xu, Mingze Sun, Zhi-Qi Cheng, Fei Wang, Yang Liu, Baigui Sun, Ruqi Huang, and Alexander Hauptmann. Combo: Co-speech holistic 3d human motion generation and efficient customizable adaptation in harmony. arXiv preprint arXiv:2408.09397, 2024. 3
- [89] Danfei Xu, Yuke Zhu, Christopher B Choy, and Li FeiFei. Scene graph generation by iterative message passing. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5410–5419, 2017. 4
- [90] Jianwei Yang, Jiasen Lu, Stefan Lee, Dhruv Batra, and Devi Parikh. Graph r-cnn for scene graph generation. In Proceedings of the European conference on computer vision (ECCV), pages 670–685, 2018. 4
- [91] Nianzu Yang, Pandeng Li, Liming Zhao, Yang Li, ChenWei Xie, Yehui Tang, Xudong Lu, Zhihang Liu, Yun Zheng, Yu Liu, and Junchi Yan. Rethinking video tokenization: A conditioned diffusion-based approach. arXiv preprint arXiv:2503.03708, 2025. 3
- [92] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-tovideo diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2, 3, 4
- [93] Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, and Tali Dekel. Space-time diffusion features for zero-shot text-driven motion transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8466–8476, 2024. 3

- [94] Hangjie Yuan, Jianwen Jiang, Samuel Albanie, Tao Feng, Ziyuan Huang, Dong Ni, and Mingqian Tang. Rlip: Relational language-image pre-training for human-object interaction detection. Advances in Neural Information Processing Systems, 35:37416–37431, 2022. 4
- [95] Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, and Dong Ni. Instructvideo: Instructing video diffusion models with human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6463–6474, 2024. 3
- [96] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyuan Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identitypreserving text-to-video generation by frequency decomposition. arXiv preprint arXiv:2411.17440, 2024. 2, 3
- [97] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818, 2023. 3
- [98] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 3
- [99] Yunpeng Zhang, Qiang Wang, Fan Jiang, Yaqi Fan, Mu Xu, and Yonggang Qi. Fantasyid: Face knowledge enhanced id-preserving video generation. arXiv preprint arXiv:2502.13995, 2025. 3
- [100] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. Motiondirector: Motion customization of text-to-video diffusion models. arXiv preprint arXiv:2310.08465, 2023. 2, 3
- [101] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024. https://github.com/hpcaitech/Open-Sora. 3
- [102] Yufan Zhou, Ruiyi Zhang, Jiuxiang Gu, Nanxuan Zhao, Jing Shi, and Tong Sun. Sugar: Subject-driven video customization in a zero-shot manner. arXiv preprint arXiv:2412.10533, 2024. 3
- [103] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent selfattention for long-range image and video generation. arXiv preprint arXiv:2405.01434, 2024. 3

## DreamRelation: Relation-Centric Video Customization Supplementary Material

### A. Appendix

#### A.1. Experimental Setup

Datasets. We select 26 types of human interaction videos from the NTU RGB+D Action Recognition Dataset [44, 61] for training. The names of these interactions and their annotated textual descriptions are provided in Tab. 8.

Baselines. Due to the current lack of relational video customization methods, we consider four baselines and detail the implementation of each method below: 1) Base Model Mochi [69]. We input the test text prompts into the original Mochi for inference and evaluate the results. 2) Direct LoRA Fine-tuning. We insert LoRAs into all the Query, Key, Value matrices, and FFNs in Mochi for training and inference. The training iterations are set to 1,000. Other training settings, such as the optimizer and LoRA rank, are the same as those in our DreamRelation. 3) ReVersion [33]. As ReVersion is designed for relational image customization and cannot be directly applied for video generation, we adapt ReVersion to the base model Mochi based on their official code1. The training settings follow the default settings provided in the official ReVersion paper. 4) MotionInversion [74]. Given that MotionInversion is designed based on the Temporal Attention layers within the UNet architecture, and such layers are absent in the MM-DiT architecture, we adapt MotionInversion to Mochi using their official code2. Specifically, we integrate the two embeddings from MotionInversion into the query, key, and value matrices of full attention, adhering to their official paper. The learning rate is set to 2e-4, and the weight decay is set to 0.01. The training iterations are 3,000, with other settings consistent with our method. During inference, we utilize the differencing operation from their official paper to mitigate the appearance biases in motion embeddings.

Evaluation metrics. We detail the proposed Relation Accuracy metric utilizing Vision-Language Models (VLMs). Specifically, we input all generated videos into Qwen-VLMax [2], the state-of-the-art Visual Question Answering (VQA) model, to determine if the generated video conforms to the specified relation, prompting it to return either “yes” or “no.” Directly inputting an entire 61-frame video into the VLM would require significant resources and slow response times. To address this, we evenly extract five key frames from each video, including the first and last two frames, and input them into the VLM. The text input template for the VLM is: “Based on the keyframes of the video, analyze

- 1 https://github.com/ziqihuangg/ReVersion
- 2 https : / / github . com / EnVision - Research / MotionInversion

whether the two subjects are performing human-like {} interactions. The answer should be ’yes’ or ’no’.” The “{}” is replaced with a specific relation name, such as “handshaking”, for evaluation. We test all videos ten times, count the responses for all videos, convert these into percentages of relation accuracy, and compute the average accuracy as the Relation Accuracy score.

#### A.2. More Results

Details about the user study. We conduct a user study involving 180 groups of videos with 15 randomly selected relations. Participants are presented with three sets of questions for each of the four anonymous methods, paired with a reference video and a textual prompt. For each group of four generated videos, participants are asked the following questions: (1) Relation Alignment: “Which interaction exhibited in videos is more consistent with the reference video?”; (2) Text Alignment: “Which video better matches the text description?”; and (3) Overall Quality: “Which video exhibits better quality and minimal flicker?”. The results of the user study are illustrated in Fig. 7(b).

More qualitative results. To further demonstrate the effectiveness of our DreamRelation, we present additional visual results in Figs. 9 and 10. These examples illustrate the capability of our method to generate videos that align with the specified relations and textual descriptions.

#### A.3. More Ablation Studies

Effects of Loss Lam λ1. To determine the optimal value for the loss weight λ1, we vary its value and measure its impact. As shown in Tab. 5, increasing the loss weight of space-time relational contrastive loss results in degradation of Relation Accuracy. We argue that over-emphasizing contrastive learning may ignore detailed information from training videos, leading to degraded performance. Therefore, we set λ1 to 0.01 for the best performance.

Relation Accuracy

Temporal Consistency

CLIP-T

FVD↓

λ1

0.01 0.4452±0.01 0.3248 0.9954 2079.87

- 0.10 0.3964±0.03 0.3241 0.9954 2088.71

- 1.00 0.2998±0.01 0.3254 0.9954 1971.29 Table 5. Ablation study of the loss weight λ1.

Effects of Mask Lam. To identify the optimal mask weight λm, we explore various values and assess their impact. As shown in Tab. 6, both excessively high and low mask weights can result in poor performance. We argue that low mask weights fail to direct the model’s focus on

the area of interest, while high weights lead to excessive emphasis, causing the neglect of other visual cues. Based on the results, we set λm to 50.

Relation Accuracy

Temporal Consistency

CLIP-T

FVD↓

λm

1 0.3469±0.07 0.2826 0.9942 2294.98 25 0.3899±0.04 0.3185 0.9953 2117.49 50 0.4452±0.01 0.3248 0.9954 2079.87

100 0.4018±0.04 0.3246 0.9952 2050.10

Table 6. Ablation study of the mask weight λm.

Effects of positive and negative numbers. We conduct ablation studies to investigate the effects of varying the number of positive and negative samples in space-time relational contrastive loss. A higher number of positive samples emphasizes the alignment of relational information during training, while an increased number of negative samples focuses more on distinguishing appearance information. We observe that different combinations have varying effects, and based on the experimental results, we chose to set npos to 4 and nneg to 10.

Relation Accuracy

Temporal Consistency

CLIP-T

FVD↓

npos nneg

1 10 0.3151±0.04 0.3259 0.9954 2089.79

- 1 30 0.2817±0.03 0.3125 0.9957 2067.35

- 1 60 0.3338±0.06 0.3154 0.9954 2113.27

- 2 10 0.3321±0.02 0.3227 0.9950 2254.62 4 10 0.4452±0.01 0.3248 0.9954 2079.87

- 2 30 0.4378±0.02 0.3168 0.9953 2009.92 4 60 0.3793±0.03 0.3237 0.9952 2156.28

###### Table 7. Ablation study of the number of positive and negative samples.

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

cheering A gorilla is raising a toast with a parrot at the top of Machu Picchu.

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

high-five A walrus is high-fiving with an arctic fox on Saturn's rings.

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

walking towards A bear and a moose are walking towards each other in a serene mountain valley.

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

hugging A penguin is hugging with a polar bear on an icy plain.

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

pushing A raccoon is pushing a bear in a serene forest clearing.

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

shaking hands A boy is shaking hands with a dog on the beach.

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

point finger A kangaroo is pointing at an emu with its finger in the Australian bush.

###### Figure 9. More qualitative results of DreamRelation (1/2). Please zoom in for a better view.

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

hugging A otter is hugging with a pangolin on a forest path.

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

walking towards A dolphin and a shark are swimming towards each other in a mysterious underwater cave.

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

knock over A cat is knocking over a dog in a snowy landscape.

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

patting on shoulder A raccoon is patting a bear's shoulder in a mountain meadow.

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

kicking A dog is kicking a cat in a cyberpunk-themed setting.

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

carry object A bear is carrying a box with a raccoon in a serene forest clearing.

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

walking apart A bear and a moose are walking apart from each other in the Alaskan wilderness.

###### Figure 10. More qualitative results of DreamRelation (2/2). Please zoom in for a better view.

Table 8. The list of 26 human interactions with their textual prompts.

- 1. walking apart: “A person and a person are walking apart from each other.”
- 2. walking towards: “A person and a person are walking towards each other.”
- 3. shaking hands: “A person is shaking hands with a person.”
- 4. hugging: “A person is hugging with a person.”
- 5. point finger: “A person is pointing at a person with his finger.”
- 6. pat on back: “A person is patting a person’s shoulder.”
- 7. pushing: “A person is pushing a person.”
- 8. kicking: “A person is kicking a person.”
- 9. punch or slap: “A person is punching a person.”
- 10. rock-paper-scissors: “A person is playing rock-paper-scissors with a person.”
- 11. support somebody: “A person is supporting a person while walking.”
- 12. whisper: “A person is whispering to a person.”
- 13. follow: “A person is following a person.”
- 14. take a photo: “A person is taking a photo of a person.”
- 15. carry object: “A person is carrying a box with a person.”
- 16. cheers and drink: “A person is raising a toast with a person.”
- 17. high-five: “A person is high-fiving with a person.”
- 18. step on foot: “A person is stepping on a person’s foot.”
- 19. shoot with gun: “A person is shooting a person with a water gun.”
- 20. knock over: “A person is knocking over a person.”
- 21. giving object: “A person is giving an object to a person.”
- 22. touch pocket: “A person is touching a person’s pocket.”
- 23. hit with object: “A person is hitting a person with an object.”
- 24. wield knife: “A person is wielding a toy knife towards a person.”
- 25. grab stuff: “A person is grabbing an item from a person.”
- 26. exchange things: “A person and a person are exchanging items with each other.”

