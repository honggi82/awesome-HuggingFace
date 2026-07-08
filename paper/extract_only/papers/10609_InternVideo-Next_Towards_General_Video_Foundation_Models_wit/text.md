# arXiv:2512.01342v2[cs.CV]30Mar2026

## InternVideo-Next: Towards World Understanding Video Models

Chenting Wang1,2 Yuhan Zhu2,5 Yicheng Xu2 Jiange Yang2,5 Lang Lin2 Ziang Yan2 Yali Wang2,4 Yi Wang2,3 Limin Wang1,2,5,♠ 1Shanghai Jiao Tong University 2Shanghai AI Laboratory 3Shanghai Innovation Institute 4Shenzhen Institutes of Advanced Technology, China 5Nanjing University

### Abstract

Large-scale video-text pretraining achieves strong performance but depends on noisy, synthetic captions with limited semantic coverage, often overlooking implicit world knowledge such as object motion, 3D geometry, and physical cues. In contrast, masked video modeling (MVM) directly exploits spatiotemporal structures but trails textsupervised methods on general tasks. We find this gap arises from overlooked architectural issues: pixel-level reconstruction struggles with convergence and its low-level requirement often conflicts with semantics, while latent prediction often encourages shortcut learning. To address these, we disentangle the traditional encoder-decoder design into an Encoder-Predictor-Decoder (EPD) framework, where the predictor acts as a latent world model, and propose InternVideo-Next, a two-stage pretraining scheme that builds a semantically consistent yet detail-preserving latent space for this world model. First, conventional linear decoder in pixel MVM enforces the predictor’s output latent to be linearly projected to, thus separable in pixel space, causing the conflict with semantic abstraction. Our Stage 1 proposes a conditional diffusion decoder and injects reliable image-level semantic priors to enhance semantics and convergence, thus bridging pixellevel fidelity with high-level semantic abstraction. Stage 2 further learns world knowledge by predicting frozen Stage 1 targets within this space, mitigating shortcut learning. Trained on public, unlabeled videos, InternVideoNext achieves state-of-the-art results across benchmarks and provides a scalable path toward general video representation learning. Code and models will be released at https://github.com/OpenGVLab/InternVideo.

### 1. Introduction

Video, as one of the most dynamic and information-rich

♠ Corresponding author (lmwang@nju.edu.cn).

[Figure 1]

Figure 1. Comparison with previous SOTA methods of size ViT-Large. With only public sources, our model excels at general video benchmarks within a probing setting where ViTs are frozen to directly show representation’s quality. The benchmarks involve scene-related, motion-related, complex video-language related and implicit world knowledge (3D geometric prior, causal relations and fine-grained object motion) related tasks.

modalities, offers a window into the physical world. It embeds not only spatiotemporal dynamics [20, 36, 58, 64] but also causal relationships, 3D geometrical priors and physical cues [2, 10]. These are all essential components for building genuine video understanding models. Such models are crucial for advancing embodied AI [23, 71], procedural reasoning [83], and the next generation of multimodal large language models [31, 32, 35, 38, 39]. However, learning such knowledge from raw data, especially in a scalable and unbiased manner, remains a challenge.

Recent progress in large-scale video representation learning often falls into two categories. 1) Text-supervised methods, leveraging large-scale video-text alignment [33, 42, 70, 75, 82], have achieved impressive performance on semantic, human-centric tasks like action recognition. However, due to expensive annotation costs, currently their reliance on noisy, synthetic annotations with limited semantic coverage introduces external biases, making it difficult to

[Figure 2]

[Figure 3]

High-Semantic Latent token Image-level semantic token

Clean image patches Masked out tokens

Condition-Based Diffusion Decoder

𝑃𝑖𝑥𝑒𝑙

[Figure 4]

[Figure 5]

+ noise 𝜖

Reconstruction

###### MLP

noise 𝜖

[Mask] special token Guidance loss

condition z

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

𝑀𝑎𝑠𝑘𝑒𝑑 𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛

Predictor - Latent World Model

Predictor - Latent World Model

Decoder

𝑈𝑛𝑚𝑎𝑠𝑘𝑒𝑑 𝐴𝑙𝑖𝑔𝑛𝑚𝑒𝑛𝑡

[Figure 21]

[Figure 22]

| |
|---|

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

| |
|---|

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

| |
|---|

[Figure 36]

[Figure 37]

[Figure 38]

Semantic Encoder

Stage1 Encoder

Stage1 Encoder

Encoder

Encoder

[Figure 39]

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

| |
|---|

| |
|---|

[Figure 53]

[Figure 54]

| |
|---|

𝑺𝒕𝒂𝒈𝒆 𝟏

𝑺𝒕𝒂𝒈𝒆 𝟐

(a) Previous MVM (b) InternVideo-Next (ours)

- Figure 2. Our two-stage self-supervised video pretraining framework. For general video understanding and building real worldunderstanding video foundation models, we propose InternVideo-Next, which is simple, scalable, efficient and reproducible.

tic abstraction with implicit world knowledge like geometry and motion in the Encoder’s representation.

capture non-semantic vision features. They often fall short in tasks that require detailed object motion, causal structure or 3D geometrical priors. 2) Self-supervised video pretraining [2, 5, 57], in contrast, holds the promise of learning directly from the spatiotemporal structure of videos, offering a path towards truly unbiased and scalable world modeling. However, existing methods still lag behind textsupervised counterparts on benchmarks requiring strong semantics about main subject like Kinetics-400 [28].

In light of this, we propose InternVideo-Next, a novel two-stage video pretraining framework, as Figure 2:

In Stage 1, we construct such a semantically grounded, detail-preserving and structurally consistent latent space by: 1) leveraging a pixel reconstruction framework as foundation to learn fine-grained pixel details and spatiotemporal priors. 2) using a condition-based diffusion generative decoder in place of the popular Linear Decoder head in turning Predictor output latent to pixels. Such a diffusion decoder is helpful to maintain high semantics in the Predictor output space as common linear decoder may require the output to be linearly separable to pixels, which is harmful to balance semantic information with fine-grained details. 3) integrating semantic priors from a pretrained Image semantic model. Unlike video–text pretraining, which is limited by the sparsity and noisiness of video captions, image–text pretraining benefits from massive, high-quality corpora whose captions more comprehensively describe visual content. Such priors are cleaner and can accelerate model’s convergence by focusing on more temporal-centric information, meanwhile boosting multi-modal friendliness.

We contend that this performance gap of self-supervised video pretraining is not an intrinsic limitation, but rather stems from fundamental architectural challenges unsolved in current methods: 1) Bridge pixel-level fidelity with high-level semantic abstraction 2) Learn robust spatiotemporal dynamics, causal relations and 3D geometric priors from prediction without “shortcut solutions”.

Popular MVM methods, like MAE [25], take a conventional Encoder-Decoder paradigm, where ViT-Decoder [19] directly takes Encoder output and generates reconstructed pixels. We explicitly decouple it into Predictor and Decoder, and consider the whole MVM paradigm as a new unified Encoder–Predictor–Decoder (EPD) overview. Such disentanglement enables a clearer inspection of the Predictor’s output latent space that is often neglected, which we find is essential in dealing with the unsolved challenges. It also reveals a key insight overlooked by previous works: the Encoder and Predictor should share a semantically rich yet detail-faithful output latent space, making the predictor a possible Latent World Model. Enforcing this compels the Predictor to complete missing content using genuine spatiotemporal relationships and implicit world knowledge rather than trivial correlations, further boosting seman-

Building upon this, Stage 2 then focuses on learning spatiotemporal dynamics and causal relations within this already coherent latent space. We employ a latent-space prediction objective, where a student model learns to predict the representations generated by a frozen teacher, where the two models are both initialized with weights learned in Stage 1. Crucially, the semantically rich and detailpreserving latent space established in Stage 1 prevents the “shortcut solutions” that plague traditional latent-prediction

methods, pushing the model to learn genuine predictive world knowledge rather than superficial temporal statistics.

As shown in Figure 1, InternVideo-Next achieves stateof-the-art results across a wide range of video understanding tasks. Remarkably, it is the first video model trained without explicit video–text supervision that surpasses video–text pretrained counterparts on both Kinetics-400 (action recognition) and SSv2 (fine-grained motion recognition). The learned representations also exhibit strong generalization to tasks requiring physical and 3D spatial intelligence, such as depth estimation and object-centric tracking, which are implicit world knowledge inherent in videos. In addition, with lightweight LiT-style [78] fine-tuning, InternVideo-Next delivers competitive zero-shot videotext retrieval performance, highlighting our model representation’s multimodal friendliness. Also, with preliminary exploration in chat-centric tasks in video, it holds the potential to be the foundation Encoder of next-generation multimodal video chat models. Our framework provides a unified and scalable pathway toward general-purpose worldunderstanding video models. All the code and model will be released upon publication.

### 2. Related Works

Text Supervised Pre-training. Clip-style pretraining has dominated current mainstream image pretraining frameworks [50, 59, 79], benefiting from large-scale web-crawled image-text pair data. Video–text pretraining methods [13, 15, 34, 45, 75] also achieve strong performance on benchmarks relying heavily on the main subject semantics like Kinetics-400. However, video captions are more difficult to be directly crawled from internet and are often generated by summarizing title and ASR information. Such synthetic annotations are typically noisier than images’ and lack sufficient motion or spatial diversity, considering that videos are harder to fully describe than images. As can be seen with Figure1, although video-text models excel at semanticheavy tasks, they often fall short in tasks related to nonsemantic information like depth, fine-grained motion and causal relation. Our framework focuses more on unbiased learning from video itself, retaining high semantics, lowlevel details and world knowledge priors.

Masked Video Modeling. Masked video modeling (MVM) is the mainstream self-supervised video pretraining method, inspired by masked image modeling [4, 25, 48, 67]. VideoMAE [57] and MaskFeat [73] reconstruct masked video patches in the pixel domain, achieving strong performance but primarily capturing low-level appearance cues. Subsequent works such as VideoMAE-V2 [65] improve masking strategies and decoder designs but still struggle to preserve high-level semantics. Latent-space prediction approaches, such as V-JEPA [5], move toward semantic ab-

straction by predicting feature representations instead of pixels, yet symmetric teacher–student structures often lead to shortcut learning or semantic drift. As seen in Figure 1, it falls short in appearance-intensive tasks, depth estimation tasks that require low-level details, and high-semantic tasks, meaning that such shortcut learning leads to missing details and prevents higher-level information. Our approach unifies these paradigms under an Encoder–Predictor–Decoder (EPD) formulation and emphasizes maintaining semantic coherence between encoder and predictor latent spaces, combining the strengths of both.

The InternVideo Series. The InternVideo series [62, 68, 70] has explored multiple ways in integrating unsupervised video learning with text alignment to bridge pixel-level fidelity with high-level semantic abstraction. InternVideo merges weights from pretrained VideoMAE and CLIP models using model ensemble, while InternVideo2 introduces a video-only stage that aligns unmasked representations from VideoMAE and CLIP encoders. These methods can be viewed as weight- or embedding-level fusion of video priors and language knowledge. However, they still can not fully solve the conflict between fine-grained details and highlevel semantics. InternVideo-Next revisits video-only pretraining from a task-level perspective by integrating CLIPlevel semantic priors into an augmented video reconstruction framework, resolving such conflict.

### 3. Method

#### 3.1. The EPD Disentanglement

We formulate our unified Encoder–Predictor–Decoder (EPD) overview as follows: E: A Vision Transformer that extracts spatiotemporal representations from an input video. P: A lightweight transformer that predicts latent representations for masked regions based on visible tokens. D: A reconstruction module that maps the predictor’s output latent to target space, either pixels or target latent.

#### 3.2. S1: Semantic-Guided Pixel Reconstruction

Semantic Alignment Loss. Our approach leverages semantic knowledge from the image domain via a frozen SigLIP encoder. This allows us to inject a strong imagelevel semantic prior into our video-only pre-training, offloading the need for noisy video-text annotations while focusing the model’s learning on spatiotemporal dynamics:

Lsem = −cos E(Xvis),vis(SigLIP(X)) , (1)

We use a cosine similarity loss between the student’s embedding of a masked video and the corresponding visible region of the teacher’s embedding of the full video. Stage 1 jointly optimizes pixel reconstruction and semantic alignment with the same loss weight.

(a) Stage-1 module effect Variant K400 SSv2 Pixel Recon. Baseline 47.2 28.1 SigLIP Align only 70.7 32.1 Pixel Rec. + Align 69.8 31.8 + Diffusion Decoder 74.2 35.4 + Text-Decoder Init 69.4 31.3 + Keep Both 75.8 36.9

(b) Predictor size & init Predictor K400 SSv2 ModernBert-L last2 73.2 34.8 ModernBert-L last5 75.8 36.9 last5 w/o init 74.2 35.4 ModernBert-L last8 75.6 37.0 Depth-5 ViT 74.1 34.9 Depth-12 ViT 73.2 34.4

(c) Different Decoder settings Decoder K400 SSv2 Linear Head 69.4 31.3 3-Layer MLP Head 69.7 31.2 DiffMLP D3 W1024 73.4 33.2 DiffMLP D6 W1536 75.8 36.9 DiffMLP D9 W2048 75.5 36.4

- Table 1. Ablations of Stage 1 design. Linear probing results are shown. (a) Adding semantic alignment boosts overall performance, while directly merging it into pixel reconstruction framework causes performance drop. Combining our diffusion decoder and text decoder augmentation make them fuse perfectly and results in accuracy gain. (b) Moderate predictor depth with initialization performs best in our scheme, outperforming popular results with Depth-12 ViT. (c) D for number of MLP layers. and W for network width.

Semantic-Aware Masking. Semantic mask prioritizes temporally informative regions using attention scores derived from the semantic teacher with a top-k selection.

Diffusion Decoder. Unlike the popular linear decoder [25], we adopt a lightweight conditional diffusion [27] decoder to model the distribution of each patch independently. The diffusion process and loss function follow [46], where noise schedule has a cosine shape, with 1000 steps at training time. We use a small MLP consisting of a few residual blocks [24] for denoising. The denoising MLP is conditioned on a vector z produced by the Predictor and outputs corresponding pixels. Since it only models a patch’s latent distribution, a small MLP works perfectly and does not involve much overhead.

Text-Decoder Initialization. Conventional pixel reconstruction framework uses zero-init ViT. Our Predictor P is initialized with weights from a pretrained text decoder [42, 72], which provides better semantic priors and the smooth translation of high semantics between the two latent spaces, and proved to demand fewer layers than common methods.

#### 3.3. S2: Semantically Coherent Latent Prediction

Stage 2 builds upon the semantically aligned latent spaces obtained in Stage 1 to further enhance temporal abstraction and representation generalization.

Initialization. Both the student and teacher networks are initialized with the weights learned in Stage 1. And the Stage 1 predictor is also kept in Stage 2.

Multi-Block Masking. To strengthen temporal reasoning, we apply a multi-block masking [5], where large contiguous spatiotemporal blocks in a video are masked. This strategy increases prediction difficulty and mitigates information leakage, which is better for Stage 2’s target of implicit world knowledge by prediction.

Latent Prediction Objective. The student predicts the teacher’s latent representations for masked regions. This objective enforces consistency in the latent space without direct pixel reconstruction, enabling the model to focus on abstract semantic and temporal patterns.

Frozen Teacher Target. Unlike V-JEPA, the teacher network is frozen with Stage 1 initialization to prevent shortcut learning, as the Stage 1 initialization is already detailpreserving and of high semantics.

### 4. Experiments

Implementation. For our pretraining, we utilize the last five layers of ModernBert-Large [72] as our Predictor. We use SigLIP2-1B [59] as our semantic teacher in our final version, and use SigLIP2-Large in the Ablation study. For Stage 1 pre-training, we use a mask ratio of 80% and a learning rate of 1e-3. For ablation study, we use 32 A100 with a batch size of 1024 for 30 epochs in both Stage 1 and Stage 2. And for final training, we use 64 A100 with a batch size of 2048 for 50 epochs in Stage 1 and 100 epochs in Stage 2. For more, please refer to the Supplement.

Datasets. Unless otherwise stated, we use K-Mash [70] for training, including 1.1M video data from K400 [28], K600 [8], K700 [9], SSv2 [22], ANet [26], HACS [81], and MiT [43] with duplicates removed. For ablation, we use a subset K710 containing only the Kinetics datasets.

#### 4.1. Ablation Study

We conduct comprehensive ablations to analyze performance of our training framework on appearance-based, motion-centric and depth-estimation benchmarks, including K400 [28], SSv2 [22], and ScanNet [17]. We leverage Linear Probing results on action recognition tasks for clear presentation of representation quality improvement. Results on ScanNet are based on probing results using learnable queries and a single-layer trainable cross-attention head.

##### Stage1-Variant K400 SSv2 Scannet

Acc1↑ Acc1↑ δ1 ↑ DinoV2 Align only 68.4 29.5 43.3 Clip-ViT Align only 69.1 31.2 40.6 SigLIP Align only 70.7 32.1 42.1 InternVideo2 Stage1 Strategy

SigLIP+VideoMAE align 70.4 32.5 50.1 Ours w/ DinoV2 align 69.3 31.5 58.3 Ours w/ Clip-ViT align 73.2 34.6 57.8 Ours w/ SigLIP align 75.8 36.9 59.4

- Table 2. Comparison with different supervision signals. Also, we compare results with InternVideo2-S1, which directly aligns with two teachers as a first attempt to balance pixel-level detail and semantic abstraction, in fair training recipe.

Stage-2 Variant K400 SSv2 Stage 1 75.8 36.9 Our Stage 2 setting 76.9 56.9 replace w/ Zero-Init V-JEPA Predictor 74.8 53.8 replace w/ Momentum 0.9998 Target 74.1 54.3 replace w/ Frozen SigLIP2 Target 75.4 45.7 replace w/ Frozen InternVideo2 Target 74.3 47.4 if w/ Unmasked Token Alignment Loss 75.7 51.1 if w/ Pixel Reconstruction Loss 76.8 57.0

- Table 3. Stage-2 ablation. Our full configuration yields the strongest temporal abstraction and overall accuracy.

Effects of Stage 1 Settings. Table 1 presents detailed ablations on the Stage 1 design. As shown in (a), the pixel reconstruction baseline achieves 47.2% on K400, confirming its limited semantic abstraction ability. Using SigLIP alignment alone without reconstruction boosts recognition accuracy. However, naively combining pixel reconstruction and alignment hurts overall performance due to optimization conflicts. Introducing the diffusion decoder reverses such degradation into a +4.4% performance gain, demonstrating strong complementarity between the two supervision signals, if not hurt by the original Linear Decoder. It highlights the effect of our EPD disentanglement, which provides a novel reconsideration and inspires us to re-use the now seldom-utilized pixel reconstruction framework. (b) studies the impact of predictor architecture and initialization. The vanilla ViT predictor without initialization, as used in the original setting, underperforms our solution.

Panel (c) compares different decoder designs. A lightweight linear head underfits complex spatial relations, while our diffusion-based MLP decoders progressively improve as capacity grows. The medium-sized Depth 6,

##### Stage Mask Type #F K400 SSv2 Scannet

Acc1↑ Acc1↑ δ1 ↑ Effect of Masking Strategy (ViT-B)

##### Semantic 8 75.8 36.9 59.4

- Stage-1 Multi-block 8 74.4 36.3 58.1

Semantic 8 75.2 52.3 61.1

- Stage-2 Multi-block 8 76.9 56.9 66.1

Effect of Temporal Frame Count (ViT-B) 8 75.8 36.9 59.4

- Stage-1 Semantic 16 76.4 38.4 59.8 32 76.6 38.6 59.3

8 77.0 57.5 67.1

- Stage-2 Multi-block 16 77.4 58.4 68.7 32 78.1 59.4 70.1

Table 4. Impact of masking and frame settings. Each Stage-2 model is based on the highlighted Stage-1 model in the same table.

Width 1536 variant achieves the best trade-off between representation richness and computational cost.

Different supervision for Stage 1’s semantic prior. Table 2 reveals that our framework shows generalizable performance gain with different semantic alignment targets, whether on common recognition tasks or 3D understanding tasks with low-level details. Supervision from semantic image models performs best. It also compares our unified Stage 1 strategy with the multi-teacher alignment scheme in InternVideo2, which aligns with both CLIP and VideoMAE to balance pixel-level detail and semantic abstraction.

Effects of Stage 2 Settings. Stage 2 further boosts temporal abstraction and causal relations via masked latent prediction. As shown in Table 3, our configuration achieves the strongest results. Replacing our predictor with a zeroinitialized V-JEPA variant, unfreezing the target encoder or using other initialization of the target encoder all lead to degradation, confirming the importance of semantically coherent initialization and a stable target that encodes both high semantics and pixel-level details. Adding unmasked token alignment as in Stage 1 introduces noise to latent targets and harms motion modeling. Pixel-space reconstruction during this stage provides marginal benefit to SSv2, as the targets, produced by our Stage 1 encoder, already contain enough pixel-level details.

Masking Strategy and Temporal Window. Table 4 analyzes masking and temporal length configurations. Semantic masking in Stage 1 helps the model attend to spa-

Model Name ViT Size Data GPU-hrs K400 ↑ SSv2 ↑ Coin ↑ Image Models

SigLIP2 [59] Giant 10B - 85.6 47.9 91.8 DinoV2 [7] Giant 142M - 83.1 50.0 87.8

Models Trained with Video-Text Pairs InternVideo2s2 [70] Base 25.5M - 84.9 64.7 88.7 InternVideo2s2 [70] Large 25.5M - 86.0 65.9 90.1 InternVideo2s2 [70] 1B 25.5M 30K 87.9 67.3 91.7 InternVideo2s2 [70] 6B 400M 200K 88.8 67.7 92.6 VideoPrism [82] Base 618M - 84.2 63.6 VideoPrism [82] 1B 618M 250K∗ 87.2 68.5 -

Models Trained with Video Data Only

VideoMAEv2 [65] Large 1.35M - 80.9 54.9 83.2 VideoMAEv2 [65] 1B 1.35M 15K 82.8 56.1 84.6 V-JEPAv1 [5] Large 2M 8K∗ 80.8 69.5 83.0 V-JEPAv2 [2] Large 22M 10K∗ 83.3 72.0 85.9 InternVideo2s1 [70] 1B 1.1M 15K 85.6 57.4 89.6 InternVideo2s1 [70] 6B 2.1M 110K 86.0 59.0 90.3

- InternVideo-Nexts1 Base 1.1M 1.5K 84.8 58.6 89.9

- InternVideo-Nexts2 Base 1.1M 3.4K 85.9 70.1 91.4

- InternVideo-Nexts1 Large 1.1M 3.6K 87.1 65.5 92.0

- InternVideo-Nexts2 Large 1.1M 9.7K 88.4 73.0 93.6

- Table 5. Attentive Probing Results on Video Action Recognition Datasets K400, SSv2 and COIN. We report top@1 results with a single probing head trainable. The training cost is estimated with equivalent A100 GPU Hours. ∗ means estimated result, as not included in the paper. Models are tested with 16 frames, and with 1 clips ×3 crops. Our models achieve the best performance across model sizes and with the smallest, public and annotation-free training data, highlighting our method’s efficiency.

tially meaningful regions and bridge high-level semantic abstraction with pixel-level fidelity. Multi-block masking in Stage 2 makes the prediction task harder and forces robust spatiotemporal dynamics and causal relations learned in the Predictor. Moreover, increasing the number of input frames consistently enhances performance, demonstrating that richer temporal diversity strengthens temporal abstraction. We finally chose F16 for Stage 1 and F32 for Stage 2 for balance between accuracy and efficiency.

#### 4.2. Single-modality Tasks

We evaluate our method on several single-modality tasks: including the Kinetics-400 [28], Something-Something V2 [22] for action recognition, COIN [56] for long video classification, ScanNet [17] and KITTI [6] for monocular depth estimation, WayMo[54] for object tracking and EK100[18] for action prediction.

Depth Estimation. As discussed above, 3D geometric prior is also a part of the world knowledge that can be acquired from videos. Table 6 tests the Encoder’s probing performance on 3D-related Monocular Video Depth Estimation tasks Scannet [17] and KITTI[6] in two settings. One uses a single-layer cross-attention based Probing Head with learnable queries, as to directly show model’s spatiotemporal embedding quality regarding Video Depth. Our InternVideo-Next achieves far better results than image models as it learns more direct 3D spatial information in the backbone. And another uses a complex VDA head designed specifically for Image models with additional temporal fusion modules [14]. Our InternVideo-Next excels at both settings, even achieving nearly SoTA performance in such a ‘Probing Setting’ compared with the specifically designed and trained Video Depth Anything [14] model. Detailed test settings are left in the Supplement.

Video Classification. Table 5 reports probing results with different frozen backbones on general video classification tasks. We test the model in an ‘Attentive Probing’ setting where the encoders are frozen and a single-layer attention pooling head is trained. Such Frozen Encoder settings can test representation’s quality in an unbiased way. Our methods achieve the best results with only public data and less computation cost on these foundation tasks.

Object Tracking. We follow the setting in Scaling 4D representations [10] to test model embedding’s quality in capturing object-level motion with Object Tracking task on Waymo Open [54]. The task requires the model to predict Bounding Box given the coordinates of an object in the first frame. We use learnable queries and a single-layer crossattention pooling head with the frozen ViT backbone for testing. We will leave more details in the Supp. Our model

Frame

Ours-L

InternVideo2-L

V-JEPA2-L

DinoV3-L

SigLip2-L

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

|[Figure 58]|
|---|

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

|[Figure 65]|
|---|

|[Figure 66]|
|---|

- Figure 3. Video Depth estimation visualization. All models are trained with a frozen setting with only a probing head (VDA head from Video Depth Anything [14]) module trainable. Graphs are from the first video of the ScanNet Dataset. Our model shows potential for building next-generation 3D-spatial intelligence models from video sources. Videos are resized with a spatial resolution of 224.

Scannet KITTI

Waymo Open mean IOU↑ Image Models

Model Name

Model Name

ARel↓ δ1↑ ARel↓ δ1↑ SOTA solution with VDA head and specific training VideoDepthAnything [14] 8.7 92.6 8.3 94.6 Image Models w/ a simple probing head [10]

SigLip2-L [59] 52.3 DinoV3-L [53] 59.7

Video Models

SigLip2-L [59] 26.4 50.4 16.8 74.9 DinoV3-L [53] 24.0 59.1 13.5 81.8 Video Models w/ a simple probing head [10]

InternVideo2-L [70] 63.0 V-JEPA2-L [2] 68.9 InternVideo-Next-L 72.4

VideoMAEv2-L[65] 19.3 66.9 11.9 84.6 InternVideo2-L [70] 23.4 57.0 12.9 83.0 V-JEPA2-L [2] 20.9 63.1 10.9 86.8 InternVideo-Next-L 13.7 81.5 9.9 88.6 Image Models w/ VDA head with temporal fusion [14]

Table 7. Probing Results on Video Object Tracking Task Waymo Open. We follow the task setting in Scaling4D [10] to test representation’s capability in capturing object-level motion.

SigLip2-L [59] 19.9 65.6 12.2 84.2 DinoV3-L [53] 9.6 91.2 7.0 93.0 Video Models w/ VDA head with temporal fusion [14]

action given a contextual video clip. The outputs of the predictor and encoder are concatenated along the token dimension and fed to an attentive probe layer and three classification heads. We follow most of the settings in V-JEPA2 [2] and report the mean-class recall-at-5 result. Our model performs better at predicting future actions.

|VideoMAEv2-B [65] InternVideo2-B [70] InternVideo-Next-B<br><br>|21.8 60.6 15.7 76.2 11.3 87.3<br><br>|9.6 89.7 8.8 91.3 7.5 93.2<br><br>|
|---|---|---|
|VideoMAEv2-L [65] InternVideo2-L [70] V-JEPA2-L [2] InternVideo-Next-L<br><br>|17.9 70.4 15.1 77.8 14.4 79.6<br><br>9.2 92.2<br><br>|9.5 89.7 8.1 92.3 8.7 91.2 6.7 94.6<br><br>|

#### 4.3. Multi-modality Tasks

In this section, we evaluate the potential of our model in multi-modal learning scenarios through two representative experimental setups. First, following the LiT [78] methodology, we freeze the ViT backbone and train only a text encoder, initialized with MobileCLIP-B [60], on 25.5M [3, 11, 30, 37, 44, 47, 52] publicly available videotext pairs from UMT [33] for 5 epochs. This setup evaluates the model’s zero-shot video-text retrieval performance, serving as a direct probe into the completeness and correctness of the visual embeddings in the semantic space. We choose InternVideo2s2 and V-JEPA2 as our baseline model. Second, to assess the model’s applicability as a video foun-

- Table 6. Probing Results on Video Depth-Estimation Datasets Scannet and KITTI. Our InternVideo-Next with such a simple probing setting can achieve comparable results with the carefully designed and trained Video Depth Anything model. ARel for ‘absolute relative error’ and δ1 for ‘relative accuracy’ [14]. achieves the best result in capturing object-level motion.

Video Prediction. Table 8 shows video prediction results on Epic-Kitchens-100 [18]. The model predicts the future

Method Verb Noun Action MLLM MCQ result

VideoLLaMa-7B [80] 52.9 52.0 26.0 Frozen Backbone Evaluation

V-JEPA2-L [2] 57.8 53.8 32.7 InternVideo-Next-L 58.9 56.4 34.0

- Table 8. Results of Video Action Prediction task EK100. We report mean-class recall-at-5 for verb, noun and action on the validation set of EK100, following the test pipeline in V-JEPA2.

|Method|K400<br><br>|K700<br><br>|SSv2-MC|MiTv1|
|---|---|---|---|---|
|InternVideo2s2-B [70] InternVideo-Next-B<br><br>|67.7 68.9<br><br>|57.9 59.0<br><br>|55.9 61.2<br><br>|27.9 29.0<br><br>|
|V-JEPA2-L [2] InternVideo2s2-L [70] InternVideo-Next-L<br><br>|60.2 70.7 72.1<br><br>|51.4 61.9 63.0<br><br>|53.0 59.6 64.2<br><br>|25.4 30.6 32.0<br><br>|

- Table 9. Results of Zero-shot Action Recognition on K400, K700, SSv2-MC and MiT. Both models use a MobileCLIP-B text

- encoder and train with ViT frozen to test their representations’ separability, alignment and completeness in text space.

dation model for high-level dialogue tasks, we connect the frozen ViT with a frozen large language model, Qwen27B [66], via a trainable MLP connector. Only the MLP is fine-tuned. This is consistent with the common Stage 1 training protocol for video-based chat models to validate the compatibility and transferability of our visual representations in a generative video-LLM framework.

Zero-shot Action Recognition. Table 9 shows results on zero-shot action recognition tasks including K400 [28], K700 [9], SSv2 [22] and MiTv1 [43]. Our model outperforms InternVideo2s2, which is the previous SoTA encoder for multimodal retrieval tasks, especially on motionintensive task SSv2-Multiple Choice.

Zero-shot Retrieval. Table 10 shows results on zeroshot text-to-video retrieval tasks, including MSRVTT [76], DiDeMo [1], ActivityNet [29], LSMDC [51] and MSVD [12]. Our model shows comparable performance. This highlights our model’s embedding of the video is highly aligned with the semantic text space implicitly.

Chat-centric Tasks. We further evaluate our model’s potential as foundation Encoder for generative video-LLM frameworks. We follow the recipe of VideoChat-Flash [35] Stage 1 and utilize a large-scaled train-set including LLava558K[39], S-MiT [44], 700k filtered subset of WebVid10M [3], VidLN[61], and SSv2-open-ended [22] to fully train the connector. Such ‘Linear Probing’ setting is widely used in the Stage-1 training of multi-modal chat models.

Method MSR DDM ANet LSMDC MSVD Popular methods with full video-text training VINDLU-L [15] 32.0 36.9 30.9 - InternVideo-L [68] 40.7 31.5 30.7 17.6 43.4 UMT-L [33] 40.7 48.6 41.9 24.9 49.0 ViClip-L [69] 42.4 18.4 15.1 20.1 49.1 LanguageBind-L [84] 42.8 39.7 38.4 - 54.1 Frozen ViT with LiT Training using MobileCLIP-B

V-JEPA2-L [2] 34.4 36.3 35.7 19.2 40.1 InternVideo2s2-L [70] 42.1 42.8 43.6 21.4 44.5 InternVideo-Next-L 43.2 43.7 43.4 20.8 46.1

- Table 10. Results of zero-shot text-to-video retrieval on MSRVTT, DiDeMo, ActivityNet, LSMDC and MSVD. We only report the T2V R@1 accuracy here.

|Encoder|MVBench|Percept Test<br><br>|Dream1k|
|---|---|---|---|
|Clip-L [50] SigLIP336-L [79] SigLIP2336-L [59]|45.6 46.7 46.9<br><br>|45.1 44.1 45.8<br><br>|28.4 29.2 29.6|
|UMT-L [33] V-JEPA2-L [33] InternVideo2-L [70] InternVideo-Next-L<br><br>|45.0 44.3 47.0 50.6<br><br>|44.5 44.2 46.7 49.2<br><br>|24.6 24.3 28.7 29.8<br><br>|

- Table 11. Results on Chat-Centric benchmarks MVbench [74], Perception Test [49] for General Perception and Dream1k [63] for Detailed Caption. Models are trained in a multi-modal ‘linear prob’ setting where both the Encoder and the LLM are frozen, which is a common setting in the Stage 1 training for multi-modal chat models. For Dream1k we report the F1-score.

Our model achieves better results on general spatiotemporal perception Video QA tasks MVBench [74] and Perception Test [49], and caption task Dream1K [63].

### 5. Conclusion and Future Work

In this paper, we present InternVideo-Next, a two-stage video pretraining framework without video-text supervision that unifies pixel-level reconstruction and latent prediction under a coherent semantic space and learns by prediction with such a space. By enforcing semantic alignment between encoder and predictor, our method learns structured spatiotemporal priors from raw videos, surpassing previous video-only and video–text methods on general video tasks and proving InternVideo-Next’s capability as a strong general video foundation model. For future works, we plan to investigate the scalability of this method, and extend this framework toward multi-modal and interactive world modeling, exploring efficient scaling and open-world generalization for more versatile video foundation models.

Class

Depth

Bounding Boxes

Depth

Object Queries

Linear Head

Regression Head

Regression Head

VDA Head (Temporal Layer + Fusion Layer)

[Figure 67]

[Figure 68]

[Figure 69]

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

Query Encoder

Attention Pooler

Attention Pooler

Attention Pooler

[Figure 80]

[Figure 81]

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

[Figure 105]

[Figure 106]

Encoder

Encoder

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Encoder

Encoder

Intermediate Features

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

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

(a) Action Recognition

(b) Depth Estimation w/ Cross-Attention Head

(c) Depth Estimation w/ VideoDepthAnything Head

(b) Object Tracking w/ Cross-Attention Head

###### Figure 4. Explanations of different probing settings in our paper.

|config<br><br>|SthSth V2 Others|
|---|---|
|optimizer optimizer momentum weight decay learning rate schedule learning rate batch size warmup epochs [21] total epochs input frame spatial resolution flip augmentation augmentation<br><br>|AdamW [40] β1,β2=0.9,0.98 0.05 cosine decay [41] 1e-3 2048 5 50 16 224<br><br>no yes MultiScaleCrop [0.66, 0.75, 0.875, 1]|

|Stage|ViT-B<br><br>|Output Size|
|---|---|---|
|Data<br><br>|sparse sampling|3×8×224×224|
|Patch Embedding|1×14×14, 768 stride 1×14×14<br><br>|768×8×256|
|Position Embedding<br><br>|sine-cosine 768×2048|768×2048<br><br>|
|Mask|semantic mask mask ratio = ρ<br><br>|768×2048·(1-ρ)|
|Encoder|MHSA(768) MLP(3072) ×12<br><br>|768×2048·(1-ρ)|
|Projection|LN(768) MLP(1408) ×K<br><br>|K×1408×2048·(1-ρ)|

Table 12. Architecture of video encoder. We take ViT-B with 8-frame input as an example. “MHSA”, “MLP” and “LN” refer to spatiotemporal multi-head self-attention, multi-layer perceptron and layer normalization. K means the layer number for unmasked token alignment. We mark the channel number, frame number, spatial size and token number by different colors.

- Table 13. InternVideo-Next Stage 1 pre-training settings.

|config<br><br>|SthSth V2 Others|
|---|---|
|optimizer optimizer momentum weight decay learning rate schedule learning rate batch size warmup epochs [21] total epochs input frame spatial resolution flip augmentation augmentation<br><br>|AdamW [40] β1,β2=0.9,0.98 0.05 cosine decay [41] 1e-4 2048(Base) 1024(Large) 0 100 32 224<br><br>no yes MultiScaleCrop [0.66, 0.75, 0.875, 1]|

- Table 14. InternVideo-Next Stage 2 pre-training settings.

### A. More implementation details

In this section, we introduce the model architectures, training hyperparameters, and test settings in our experiments.

#### A.1. Model architecture and training details

We use a patch of size 14 for our InternVideo-Next models. We report the detailed architecture of a Base-scaled model in Table 12, and the pre-training details in Tables 13, 14.

#### A.2. Action Recognition

#### A.3. Depth Estimation

We show the detailed hyperparameters for action recognition probing in Table 15. We use an attention pooling head that averages the input tokens into a single token. And the single token is utilized as the common [CLS] token for classification, as shown in Figure 4 (a).

Experiment Setup. To assess the spatiotemporal, lowlevel, and 3D-geometry related capabilities of InternVideoNext, we evaluate it on two widely used monocular video depth-estimation benchmarks: ScanNet and KITTI. Scan-

class DiffusionLoss(nn.Module)

def __init__(depth = 6, width = 1536): # SimpleMLP takes in x_t, timestep, and condition, and outputs predicted noise. self.net = SimpleMLP(depth, width)

# GaussianDiffusion offers forward and backward functions q_sample and p_sample. self.diffusion = GaussianDiffusion()

# Given condition z and ground truth token x, compute loss def loss(self, z, x):

# sample random noise and timestep noise = torch.randn(x.shape) timestep = torch.randint(0, self.diffusion.num_timesteps, x.size(0))

# sample x_t from x x_t = self.diffusion.q_sample(x, timestep, noise)

# predict noise from x_t noise_pred = self.net(x_t, timestep, z)

# L2 loss loss = ((noise_pred - noise) ** 2).mean()

return loss

Algorithm 1 Pseudo-code illustrating the concept of Diffusion Loss. Here, the conditioning vector z is the output from the Predictor. The gradient is backpropagated to z. The GT x is generated by patching the video input with a pixel shuffle function and a patch size of 14. And then we input the masked part of x and the predicted part z into the model.

|config<br><br>|SthSth v2 Kinetics Coin|
|---|---|
|optimizer optimizer momentum learning rate schedule learning rate batch size repeated augmentation warmup epochs [21] total epochs layer-wise lr decay [4] flip augmentation label smoothing [55] cutmix [77] augmentation<br><br>|AdamW [40] β1,β2=0.9,0.999 cosine decay [41]<br><br>1e-3 3e-4 2e-4<br><br>1024 2 5<br><br>30 20 20<br><br>0.75 (B), 0.85 (L) no yes yes<br><br>0.1 1.0 RandAug(9, 0.5) [16]|

Table 15. Action recognition fine-tuning settings.

a Temporally Aware head [14] based on VideoDepthAnything, consisting of four stacked temporal modules specifically designed to adapt image models such as DINO to video depth-estimation tasks (see Figure 4(c)). As it contains spatial modules tailored for depth prediction, we also apply it to video models. We use a learning rate of 3×10−3 with batch sizes of 128 (KITTI) and 256 (ScanNet). Models are trained for 30 epochs, which we find sufficient for convergence. A sliding-window strategy is applied to segment input videos, using window sizes of 32 (stride 10) for KITTI and 16 (stride 5) for ScanNet. Following standard practice, the valid depth range is set to [0.1,80.0] for KITTI and [0.001,10.0] for ScanNet, with values outside these intervals masked or clamped.

Net is a large-scale RGB-D video dataset of indoor scenes, containing over 2.5 million views across more than 1500 scans with rich annotations. KITTI is an outdoor dataset collected using an autonomous-driving platform, where stereo videos are recorded by high-resolution color cameras and depth ground truth is captured via a Velodyne LiDAR sensor. Throughout experiments, we adopt a head-probing setup in which the backbone network remains frozen.

Training Details. To provide a comprehensive evaluation, we employ two representative prediction heads: (1) a Simple probing head, following Scaling4d[10], implemented as a lightweight cross-attention pooling layer with an embedding dimension of 1536 (see Figure 4(b)); (2)

Competitive Methods. We compare InternVideo-Next with state-of-the-art image and video representation models, re-implementing all baselines under a unified training and evaluation protocol for fairness. For image-based models, we include SigLip2 and DINOv3, known for strong semantic alignment and high-quality visual representations. For video-based models, we evaluate VideoMAEv2, InternVideo2, and V-JEPA, representing the current state of the art in video representation learning.

Evaluation Metrics. Following standard monocular depth-estimation protocols, we report two metrics that jointly assess geometric accuracy.

##### 1. Absolute Relative Error (AbsRel):

N

1 N

AbsRel =

i=1

|Di − Dˆi| Di

,

where Di and Dˆi denote the ground-truth and predicted depths, respectively. This metric reflects the average relative deviation of predictions from the ground truth.

2. δ1: The proportion of predictions within a factor of 1.25 of the ground truth, providing a measure of prediction robustness and can be viewed as the accuracy rate.

#### A.4. Object Tracking

Setup. To evaluate the model’s capability to capture object-level motion, we conduct experiments on the Waymo Open dataset, following the protocol of Scaling4D [10]. The videos include 2D and 3D bounding-box annotations. We use only the RGB frames as model input and the 2D bounding boxes for computing losses and evaluation metrics. Because the original Scaling4D benchmark construction code is not publicly available, we reconstruct the training and test sets based on descriptions provided in their paper. Given the target objects’ bounding boxes in the first frame, the models are required to output their bounding boxes in the subsequent frames.

Preprocessing. Raw 1280 × 1920 RGB videos are spatially downsampled to 224×336 and then centrally cropped to 224 × 224. All bounding boxes are remapped to the cropped coordinate space, and boxes occupying less than 0.5% of the cropped frame area are filtered out. Temporally, the original 20-second sequences recorded at 10 fps are further downsampled to 5 fps.

Dataset Construction. Training samples are generated via a sliding window of 16 frames (length = 16, stride = 1) applied to each downsampled video. To avoid excessive redundancy, the starting indices for extracted windows must be at least three frames apart. We impose several object-consistency constraints: each window must contain 1–25 objects in the first frame; each first-frame object must appear in at least 70% of frames within the window; all first-frame objects must co-occur in at least 10 frames; and each object may have at most a continuous 5-frame missing gap. No further area-based filtering beyond the initial 0.5% threshold is applied.

Training and Evaluation. During training, the backbone is frozen and only the probing head is fine-tuned. The head

- encodes the initial bounding boxes—specifying the objects to track—using a query encoder. The resulting query tokens are fed into a one-layer cross-attention module to obtain

pooled tokens, which are then decoded linearly to regress bounding boxes for all frames except the first. The training objective is a weighted sum of Smooth L1 loss (1.0) and GIoU loss (0.5). For evaluation, the Intersection-overUnion (IoU) is averaged across all objects and frames.

### Acknowledgement

This work is supported by the National Key R&D Program of China (No. 2022ZD0160900), and the Basic Research Program of Jiangsu (No. BK20250009).

### References

- [1] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In ICCV, 2017. 8
- [2] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. ArXiv, 2506.09985,

2025. 1, 2, 6, 7, 8

- [3] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In ICCV, 2021. 7, 8
- [4] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. In ICLR, 2021. 3, 10
- [5] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. ArXiv, 2404.08471, 2024. 2, 3, 4, 6
- [6] Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel, Sven Behnke, Cyrill Stachniss, and Juergen Gall. Semantickitti: A dataset for semantic scene understanding of lidar sequences. ArXiv, 2019. 6
- [7] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv’e J’egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 6
- [8] Jo˜ao Carreira, Eric Noland, Andras Banki-Horvath, Chloe Hillier, and Andrew Zisserman. A short note about kinetics-

600. ArXiv, abs/1808.01340, 2018. 4

- [9] Jo˜ao Carreira, Eric Noland, Chloe Hillier, and Andrew Zisserman. A short note on the kinetics-700 human action dataset. ArXiv, abs/1907.06987, 2019. 4, 8
- [10] Jo˜ao Carreira, Dilara Gokay, Michael King, Chuhan Zhang, Ignacio Rocco, Aravindh Mahendran, Thomas Albert Keck, Joseph Heyward, Skanda Koppula, Etienne Pot, Goker Erdogan, Yana Hasson, Yi Yang, Klaus Greff, Guillaume Le Mo-

- ing, Sjoerd van Steenkiste, Daniel Zoran, Drew A. Hudson, Pedro V´elez, Luisa Polan´ıa, Luke Friedman, Chris Duvarney, Ross Goroshin, Kelsey Allen, Jacob Walker, Rishabh Kabra, Eric Aboussouan, Jennifer Sun, Thomas Kipf, Carl Doersch, Viorica P˘atr˘aucean, Dima Damen, Pauline Luc, Mehdi S. M. Sajjadi, and Andrew Zisserman. Scaling 4d representations. arXiv, 2025. 1, 6, 7, 10, 11
- [11] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pretraining to recognize long-tail visual concepts. In CVPR,

2021. 7

- [12] David L. Chen and William B. Dolan. Collecting highly parallel data for paraphrase evaluation. In ACL, 2011. 8
- [13] Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, and Jing Liu. Vast: A vision-audio-subtitle-text omni-modality foundation model and dataset. In NIPS, 2023. 3
- [14] Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang. Video depth anything: Consistent depth estimation for super-long videos. In CVPR, 2025. 6, 7, 10
- [15] Feng Cheng, Xizi Wang, Jie Lei, David J. Crandall, Mohit Bansal, and Gedas Bertasius. Vindlu: A recipe for effective video-and-language pretraining. ArXiv, abs/2212.05051,

2022. 3, 8

- [16] Ekin Dogus Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V. Le. Randaugment: Practical automated data augmentation with a reduced search space. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2020. 10
- [17] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. ArXiv,

2017. 4, 6

- [18] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. The epic-kitchens dataset: Collection, challenges and baselines. TPAMI, 43, 2020. 6, 7
- [19] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 2
- [20] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In ICCV, 2019. 1
- [21] Priya Goyal, Piotr Doll´ar, Ross B. Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. ArXiv, abs/1706.02677,

2017. 9, 10

- [22] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The” something something” video

- database for learning and evaluating visual common sense. In ICCV, 2017. 4, 6, 8
- [23] Jing Gu, Eliana Stefani, Qi Wu, Jesse Thomason, and Xin Wang. Vision-and-language navigation: A survey of tasks, methods, and future directions. In ACL, 2022. 1
- [24] Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CVPR, 2016. 4
- [25] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll’ar, and Ross B. Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 2, 3, 4
- [26] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In CVPR,

2015. 4

- [27] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NIPS, 2020. 4
- [28] Will Kay, Jo˜ao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Apostol Natsev, Mustafa Suleyman, and Andrew Zisserman. The kinetics human action video dataset. ArXiv, abs/1705.06950, 2017. 2, 4, 6, 8
- [29] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In ICCV, 2017. 8
- [30] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. IJCV, 2017. 7
- [31] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2022. 1
- [32] Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. ArXiv, abs/2305.06355,

2023. 1

- [33] Kunchang Li, Yali Wang, Yizhuo Li, Yi Wang, Yinan He, Limin Wang, and Yu Qiao. Unmasked teacher: Towards training-efficient video foundation models. In ICCV, 2023. 1, 7, 8
- [34] Tianhao Li and Limin Wang. Learning spatiotemporal features via video and text pair discrimination. arXiv, 2021. 3
- [35] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, Yu Qiao, Yali Wang, and Limin Wang. Videochatflash: Hierarchical compression for long-context video modeling. arXiv, 2025. 1, 8
- [36] Ji Lin, Chuang Gan, and Song Han. Tsm: Temporal shift module for efficient video understanding. In ICCV, 2019. 1
- [37] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 7
- [38] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. ArXiv, abs/2310.03744, 2023. 1

- [39] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 1, 8
- [40] I. Loshchilov and F. Hutter. Fixing weight decay regularization in adam. ArXiv, abs/1711.05101, 2017. 9, 10
- [41] Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In ICLR, 2017. 9, 10
- [42] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. NeurIPS, 2019. 1, 4
- [43] Mathew Monfort, Bolei Zhou, Sarah Adel Bargal, Alex Andonian, Tom Yan, Kandan Ramakrishnan, Lisa M. Brown, Quanfu Fan, Dan Gutfreund, Carl Vondrick, and Aude Oliva. Moments in time dataset: One million videos for event understanding. TPAMI, 2020. 4, 8
- [44] Mathew Monfort, SouYoung Jin, Alexander Liu, David Harwath, Rogerio Feris, James Glass, and Aude Oliva. Spoken moments: Learning joint audio-visual representations from video descriptions. In CVPR, 2021. 7, 8
- [45] Bolin Ni, Houwen Peng, Minghao Chen, Songyang Zhang, Gaofeng Meng, Jianlong Fu, Shiming Xiang, and Haibin Ling. Expanding language-image pretrained models for general video recognition. ArXiv, abs/2208.02816, 2022. 3
- [46] Alex Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. arXiv, 2021. 4
- [47] Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. In NeurIPS, 2011. 7
- [48] Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, and Furu Wei. Beit v2: Masked image modeling with vector-quantized visual tokenizers. ArXiv, abs/2208.06366, 2022. 3
- [49] Viorica P˘atr˘aucean, Lucas Smaira, Ankush Gupta, Adri`a Recasens Continente, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alex Frechette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, and Jo˜ao Carreira. Perception test : A diagnostic benchmark for multimodal models. In NeurIPS, 2023. 8
- [50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 3, 8
- [51] Anna Rohrbach, Atousa Torabi, Marcus Rohrbach, Niket Tandon, Christopher Joseph Pal, H. Larochelle, Aaron C. Courville, and Bernt Schiele. Movie description. International Journal of Computer Vision, 2016. 8
- [52] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In ACL,

2018. 7

- [53] Oriane Sim’eoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan

- Wang, Timoth´ee Darcet, Th´eo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herv’e J’egou, Patrick Labatut, and Piotr Bojanowski. Dinov3. arXiv, 2025. 7
- [54] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In CVPR,

2020. 6

- [55] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016. 10
- [56] Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. Coin: A large-scale dataset for comprehensive instructional video analysis. In CVPR, 2019. 6
- [57] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training. In NeurIPS, 2022. 2, 3
- [58] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. A closer look at spatiotemporal convolutions for action recognition. In CVPR, 2018. 1
- [59] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier H´enaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. Siglip 2: Multilingual visionlanguage encoders with improved semantic understanding, localization, and dense features. ArXiv, 2025. 3, 4, 6, 7, 8
- [60] Pavan Kumar Anasosalu Vasu, Hadi Pouransari, Fartash Faghri, Raviteja Vemulapalli, and Oncel Tuzel. Mobileclip: Fast image-text models through multi-modal reinforced training. In CVPR, 2024. 7
- [61] Paul Voigtlaender, Soravit Changpinyo, Jordi Pont-Tuset, Radu Soricut, and Vittorio Ferrari. Connecting vision and language with video localized narratives. CVPR, 2023. 8
- [62] Chenting Wang, Kunchang Li, Tianxiang Jiang, Xiangyu Zeng, Yi Wang, and Limin Wang. Make your training flexible: Towards deployment-efficient video models. arXiv,

2025. 3

- [63] Jiawei Wang, Liping Yuan, Yuchen Zhang, and Haomiao Sun. Tarsier: Recipes for training and evaluating large video description models. arXiv, abs/2407.00634, 2024. 8
- [64] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. Temporal segment networks: Towards good practices for deep action recognition. In ECCV, 2016. 1
- [65] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2: Scaling video masked autoencoders with dual masking. In CVPR, 2023. 3, 6, 7
- [66] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. ArXiv, abs/2409.12191, 2024. 8

- [67] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, and Furu Wei. Image as a foreign language: Beit pretraining for all vision and vision-language tasks. ArXiv, abs/2208.10442, 2022. 3
- [68] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, Sen Xing, Guo Chen, Junting Pan, Jiashuo Yu, Yali Wang, Limin Wang, and Yu Qiao. Internvideo: General video foundation models via generative and discriminative learning. ArXiv, abs/2212.03191, 2022. 3, 8
- [69] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Jian Ma, Xinyuan Chen, Yaohui Wang, Ping Luo, Ziwei Liu, Yali Wang, Limin Wang, and Y. Qiao. Internvid: A large-scale video-text dataset for multimodal understanding and generation. ArXiv, 2023. 8
- [70] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Chenting Wang, Guo Chen, Baoqi Pei, Ziang Yan, Rongkun Zheng, Jilan Xu, Zun Wang, Yansong Shi, Tianxiang Jiang, Songze Li, Hongjie Zhang, Yifei Huang, Yu Qiao, Yali Wang, and Limin Wang. Internvideo2: Scaling foundation models for multimodal video understanding. In ECCV, 2024. 1, 3, 4, 6, 7, 8
- [71] Zun Wang, Jialu Li, Yicong Hong, Yi Wang, Qi Wu, Mohit Bansal, Stephen Gould, Hao Tan, and Yu Qiao. Scaling data generation in vision-and-language navigation. In ICCV,

2023. 1

- [72] Benjamin Warner, Antoine Chaffin, Benjamin Clavi´e, Orion Weller, Oskar Hallstr¨om, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Nathan Cooper, Griffin Adams, Jeremy Howard, and Iacopo Poli. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. ArXiv, 2412.13663, 2024. 4
- [73] Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan Yuille, and Christoph Feichtenhofer. Masked feature prediction for self-supervised visual pre-training. In CVPR, 2022. 3
- [74] Longhui Wei, Lingxi Xie, Wen gang Zhou, Houqiang Li, and Qi Tian. Mvp: Multimodality-guided visual pre-training. In ECCV, 2022. 8
- [75] Hu Xu, Gargi Ghosh, Po-Yao (Bernie) Huang, Dmytro Okhonko, Armen Aghajanyan, and Florian Metze Luke Zettlemoyer Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. ArXiv, abs/2109.14084, 2021. 1, 3
- [76] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In CVPR, 2016. 8
- [77] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Young Joon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In ICCV, 2019. 10
- [78] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In CVPR, 2022. 3, 7

- [79] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. CVPR, 2023. 3, 8
- [80] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. ArXiv, abs/2306.02858, 2023. 8
- [81] Hang Zhao, Antonio Torralba, Lorenzo Torresani, and Zhicheng Yan. Hacs: Human action clips and segments dataset for recognition and temporal localization. In ICCV,

2019. 4

- [82] Long Zhao, Nitesh B. Gundavarapu, Liangzhe Yuan, Hao Zhou, Shen Yan, et al. Videoprism: A foundational visual encoder for video understanding. In PMLR, 2024. 1, 6
- [83] Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. Dino-wm: World models on pre-trained visual features enable zero-shot planning. arXiv, 2025. 1
- [84] Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, Hongfa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zong-Rui Li, Wancai Zhang, Zhifeng Li, Wei Liu, and Liejie Yuan. Languagebind: Extending video-language pretraining to nmodality by language-based semantic alignment. In ICLR,

2024. 8

