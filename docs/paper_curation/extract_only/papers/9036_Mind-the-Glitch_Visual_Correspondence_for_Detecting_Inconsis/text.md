## arXiv:2509.21989v1[cs.CV]26Sep2025

### Mind-the-Glitch: Visual Correspondence for Detecting Inconsistencies in Subject-Driven Generation

###### Abdelrahman Eldesokey Aleksandar Cvejic Bernard Ghanem Peter Wonka KAUST, Saudi Arabia first.last@kaust.edu.sa

Visual Correspondence

###### Semantic

Visual

Semantic Features

|[Figure 1]|
|---|

Correspondence

Features

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

| |
|---|

Reference

Subject Subject-Driven

Visually Inconsistent Region

###### Mind-the-Glitch

|[Figure 7]|
|---|

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

| |
|---|

Generation

Figure 1: Mind-the-Glitch is the first pipeline that enables computing visual correspondences based on the backbone features of pre-trained diffusion models. The pipeline separates backbone features into semantic and visual components, allowing for visually matching keypoints across images, analogous to the well-established semantic correspondence task. This provides the first empirical framework for evaluating and localizing visual inconsistencies in subject-driven image generation.

##### Abstract

We propose a novel approach for disentangling visual and semantic features from the backbones of pre-trained diffusion models, enabling visual correspondence in a manner analogous to the well-established semantic correspondence. While diffusion model backbones are known to encode semantically rich features, they must also contain visual features to support their image synthesis capabilities. However, isolating these visual features is challenging due to the absence of annotated datasets. To address this, we introduce an automated pipeline that constructs image pairs with annotated semantic and visual correspondences based on existing subject-driven image generation datasets, and design a contrastive architecture to separate the two feature types. Leveraging the disentangled representations, we propose a new metric, Visual Semantic Matching (VSM), that quantifies visual inconsistencies in subject-driven image generation. Empirical results show that our approach outperforms global feature-based metrics such as CLIP, DINO, and vision–language models in quantifying visual inconsistencies while also enabling spatial localization of inconsistent regions. To our knowledge, this is the first method that supports both quantification and localization of inconsistencies in subject-driven generation, offering a valuable tool for advancing this task. Project Page: https://abdo-eldesokey.github.io/mind-the-glitch/

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

##### 1 Introduction

Recent advances in diffusion models have transformed the landscape of image generation, delivering exceptional quality, fidelity, and diversity [8, 5, 32, 36, 27, 30]. These capabilities have revolutionized several domains, including artistic creation, advertising, content production for video games, and storyboards for movies. In many of these applications, maintaining visual consistency of a subject, whether a character or an object, across different generations is crucial. For example, in cinematic and advertising workflows, preserving the identity and appearance of a subject across multiple scenes is essential for narrative coherence and branding. However, most diffusion models operate in a latent space where semantic and visual concepts are entangled, making it challenging to control or preserve specific content using text prompts alone.

To address these limitations, a growing research direction has focused on subject-driven generation, which aims to guide diffusion models to produce consistent images of a given subject in different scenes while preserving fine-grained visual details [13, 33, 28, 3, 49]. Nonetheless, a major bottleneck in this line of research has been the lack of reliable evaluation metrics for subject consistency. Since subjects may appear in varying poses and spatial configurations, traditional image similarity metrics such as LPIPS [53], or structural similarity (SSIM) are not well-suited for this task. As a result, many works have relied on feature-based similarities using models like CLIP [29] or DINO [4] as approximations of overall appearance. However, these metrics capture global semantics and tend to overlook subtle visual inconsistencies in object details [26].

Recent works [41, 26] have explored the use of Vision-Language Models (VLMs), such as ChatGPT, to assess consistency between a reference and a generated image. While these approaches are promising, they remain limited to global assessments, and it is often unclear which visual cues or criteria the models rely on to judge consistency. Furthermore, VLM-based metrics lack the ability to localize inconsistent regions within the subject. To enable robust evaluation of subject-driven image generation, it is essential to develop methods that can reliably match the visual appearance of a subject across images, irrespective of variations in pose, scale, or context, and accurately localize inconsistencies for potential correction through post-processing.

This challenge bears similarities to the task of semantic correspondence, where corresponding points are matched across image pairs of objects with varying poses, scales, and contexts. Diffusion models have been shown to encode semantically-rich features that have demonstrated great success in computing semantic correspondences across images [39, 42, 52, 22, 51]. However, as illustrated in Figure 1, these semantic features are typically insensitive to appearance-level variations, as they focus primarily on structural or categorical semantics rather than fine-grained visual details. Given that diffusion models are trained for image synthesis, it is reasonable to assume that their internal representations should encode both semantic and visual information, yet only the semantic component has been extensively leveraged so far.

Building on this hypothesis, we propose a novel framework for disentangling semantic and visual features from the backbone of pre-trained diffusion models. Due to the absence of datasets with annotated visually similar or dissimilar regions of a given subject, we introduce an automated dataset generation pipeline that constructs image pairs with annotated semantic and visual correspondences, derived from existing subject-driven generation datasets. Using this dataset, we propose an architecture that disentangles semantic and visual features in a contrastive manner. We then leverage the disentangled representations to derive a metric, the Visual Semantic Matching (VSM) metric, that quantifies the degree of visual inconsistency in subject-driven generation. Empirical results show that our approach outperforms existing feature-based metrics such as CLIP and DINO, as well as the Vision-Language Model (VLM) metric, in quantifying visual inconsistencies, while allowing for localizing the inconsistent regions as well. We believe our framework offers a valuable step forward in the evaluation and development of subject-driven image generation.

Our contributions can be summarized as follows:

- • We propose a novel framework comprising an automated dataset generation pipeline and an architecture for disentangling semantic and visual features in diffusion models.
- • Based on the disentangled features, we present a new metric for evaluating subject-driven generation methods that both quantifies and localizes visual inconsistencies.
- • We empirically demonstrate that the proposed metric outperforms existing feature-based metrics commonly used in subject-driven generation.

##### 2 Related Work

In this section, we first provide a brief overview of the semantic correspondence task, which serves as a key inspiration for our work. Then we review existing subject-driven image generation approaches, and how consistency is currently evaluated in these methods.

###### 2.1 Semantic Correspondence using Diffusion Models

Diffusion models have been shown to encode semantically rich features within their backbones, enabling computing semantic correspondence across instances of the same object class and even among semantically similar categories. Some approaches leveraged intermediate features from the decoder of diffusion models [42], and others combine them with features from DINO [4] to enhance semantic representation [52]. Several works [22, 51] proposed learnable aggregation networks that automatically select and fuse features from different layers, trained in a supervised manner using semantic correspondence datasets such as SPair-71k [24]. CleanDIFT [39] further introduces a distillation framework to compress diffusion features, allowing for efficient inference. These semantically rich features have been successfully applied to a range of downstream tasks, including segmentation [45, 23], controllable editing [25, 2, 7], and object manipulation [20]. Inspired by the task of semantic correspondence, we aim to learn feature representations that capture visual appearance rather than semantics. These visual features enable matching regions based on appearance and are well-suited for detecting visually inconsistent regions in subject-driven image generation.

###### 2.2 Subject-Driven Image Generation

Early personalized generation techniques using UNet-based diffusion models [32, 27] primarily focused on encoding subject identity through full-model fine-tuning [34, 17] or low-rank adaptation methods [12, 10, 37, 19], and even training-free approaches [43] Other approaches [35, 9, 15, 1] learned subject-specific embeddings, without modifying model weights, by associating subjects with new tokens in an image or text embedding space. To improve the trade-off between efficiency and fidelity, adapter-based methods [48, 44, 13, 28, 50] introduced zero-shot personalization by conditioning on reference image features through lightweight network modules. These methods showed that injecting image-driven signals or selectively updating parameters could effectively preserve subject identity and visual details, even with limited data.

The emergence of diffusion Transformers (DiTs) [8, 5] introduced a new class of architectures that replace UNets with Vision Transformers as the denoising backbone, offering greater scalability and enhanced contextual understanding. These models support more flexible conditioning mechanisms and exhibit strong in-context learning capabilities, enabling them to generate diverse images of a subject without explicit fine-tuning [38, 18]. To further improve fidelity and identity preservation, several works have applied LoRA-based fine-tuning to DiTs architectures [41, 54, 14, 3], achieving more precise control over subject appearance and consistency. More recently, visual autoregressive models (VARs) have been adopted for subject-driven generation due to their inherently strong conditional modeling capabilities [40, 47, 6]. For an exhaustive overview of subject-driven image generation approaches, we refer the readers to [46].

###### 2.3 Evaluating Subject-Driven Image Generation

Visual similarity is often measured using traditional metrics such as Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), and LPIPS [53], which assume that the object is spatially aligned across images. However, in subject-driven generation, this assumption does not hold, as the subject may appear in different poses, positions, and contexts. As a result, it has become common to use global feature-based metrics, such as CLIP-Image [29] and DINO [4], which compare the similarity of image-level embeddings extracted from different models. While these methods are robust to spatial changes, they are inherently global and often fail to capture fine-grained appearance details of the subject. A recent trend involves leveraging Vision-Language Models (VLMs) to evaluate subject-driven generation by prompting them to score specific criteria of the generated images [41, 26]. However, it remains unclear how VLMs form their judgments, and they lack the ability to localize the source of inconsistency within the image. In this work, we aim to bridge this gap by leveraging visual features extracted from the backbones of diffusion models to assess

|𝐼1′|
|---|

[Figure 13]

|𝐼1|
|---|

|[Figure 14]|
|---|

[Figure 15]

|𝑅1|
|---|

|(𝑥11,|𝑦11)|
|---|---|
| | |

[Figure 16]

[Figure 17]

| |
|---|

Controlled Inconsistency

Segment

“motorcycle” Semantic

Sample a

Point

Correspondence SAM Inpaint

Object

Match

Prompts

[Figure 18]

|𝑅2|
|---|

[Figure 19]

|[Figure 20]|
|---|

[Figure 21]

[Figure 22]

| | |
|---|---|
|(𝑥21,|𝑦21)|
| | |

| |
|---|

|𝐼2|
|---|

|𝐼2′|
|---|

- Figure 2: Automated Dataset generation pipeline for producing controlled visual inconsistencies. Access to such pairs of images enables separating visual features from pre-trained backbones in a contrastive manner.

the appearance similarity across images in subject-driven generation. Our approach enables both quantification and spatial localization of visual inconsistencies. By providing a reliable metric that offers fine-grained insight into subject consistency, we take a step toward more robust evaluation and improvement of subject-driven generation.

##### 3 Method

In subject-driven image generation, assessing whether different parts of a subject are visually consistent across two images requires matching visual representations that are robust to changes in pose, scale, and environment. This challenge is similar to the semantic correspondence task, where several approaches [39, 22, 51, 52, 42] were trained on SPair-71k [24], which includes annotated semantic points between image pairs of given subjects. As demonstrated in Figure 1, semantic features are insensitive to appearance changes, making them unsuitable for matching visual appearance. To the best of our knowledge, there exists no dataset with annotated visual correspondences between visually similar or dissimilar regions, similar to SPair-71k.

To bridge this gap, we first introduce a novel automated pipeline for constructing a dataset with visual correspondences, leveraging existing datasets of subject-driven image generation such as Subjects200k [41] (Section 3.1). Using the data generated from this pipeline, we then propose an architecture for disentangling semantic and visual representations from the internal features of diffusion models (Section 3.2). Lastly, we introduce a metric that leverages the disentangled features for empirically evaluating visual consistency between image pairs, which quantifies the degree of consistency and localizes inconsistent regions (Section 3.3).

###### 3.1 Data Generation Pipeline with Visual Correspondence

We aim to generate a dataset of image pairs with visual correspondences between visually similar and dissimilar regions inspired by the SPair-71k dataset. Given a consistent image pair I1 and I2 from a subject-driven dataset, we begin by segmenting the subject in each image using Grounded-SAM [31]. This isolates the subject from the background and makes the computation of correspondences more reliable. Next, we compute semantic correspondences between the two images within the segmented subject regions using the semantic correspondence method CleanDIFT [39]. Specifically, we extract features from the sixth decoder layer of the diffusion model UNet Φ, which has been shown to contain semantically rich information [42, 39], resulting in F16 = Φ(I1) and F26 = Φ(I2). We then compute the pairwise similarity between the two feature maps to form a similarity matrix D = F16 F26T. Corresponding points are obtained by selecting locations with the highest similarity scores using arg maxD, to obtain the following correspondence mapping:

C1 = ⟨(x11,y11),...,(xN1 ,y1N)⟩ , C2 = ⟨(x12,y21),...,(xN2 ,y2N)⟩ , C1,C2 ∈ RN×2 (1) where N denotes the number of correspondences.

Based on these correspondences, we aim to match semantically similar regions across both images and then alter them visually to generate a pair of images with known, localized visual inconsistencies. To

|𝒟[𝑘]|
|---|

|𝐼1|
|---|

|𝐼2|
|---|

|𝐼1|
|---|

|𝐼2|
|---|

|𝒟[𝑘]|
|---|

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Skewness=-0.5

Skewness=1.6

- Figure 3: Examples illustrating how the skewness of the matching scores distribution correlates with matching ambiguity. High skewness implies a distinct match, while low skewness indicates diffuse or ambiguous correspondences.

achieve this, we sample a point k from C1 that has a high similarity score and use the corresponding point pair (xk1,y1k) and (xk2,y2k) as prompts to the SAM model [16], which segments a localized region in each image. This produces region masks defined as Ri = SAM(Ii,(xki ,yik)). We configure SAM to return multiple candidate masks and select the one with the smallest area, as it is more likely to correspond to an isolated semantic part rather than the entire object.

Handling Ambiguous Matches: When segmenting regions, a common challenge arises with flattextured subjects, such as a whiteboard, where the selected point k tends to match broadly across the object (see Figure 3), leading SAM to segment the entire subject rather than a localized part. To address this, we propose using the skewness of the similarity distribution D[k] to identify ambiguous matches. The skewness is computed as:

n

n (n − 1)(n − 2)

Skewness(D[k]) =

i=1

D[k] − µ σ

3

(2)

where n = |D[k]|, µ = mean(D[k]), and σ = STD(D[k]). We observe that high skewness corresponds to matches in textured regions, characterized by a long-tailed distribution and low ambiguity. In contrast, low skewness indicates more uniform similarity scores, suggesting ambiguous matches typically associated with flat surfaces as illustrated in Figure 3.

Validating Matched Regions: We apply additional heuristics to alleviate segmentation failures and to ensure that the selected regions in both images correspond to the same semantic part, including constraints on aspect ratio and relative size with respect to the full object. For regions that pass all checks, we perform inpainting using a diffusion-based inpainting model (we use SDXL [27] for efficiency), to obtain two inconsistent images I1′ and I2′. Specifically, we crop a patch around the region with padding and feed it to the inpainting model to ensure that the model does not observe the rest of the object, promoting the generation of visually distinct content. To confirm that the inpainted region differs from the original, we compute the LPIPS score [53] between the original and inpainted regions in Ii and Ii′. We discard samples with low LPIPS scores to ensure meaningful variation. Further details on the filtering strategy are provided in the supplementary material. We provide an illustration for the data generation pipeline in Figure 2. Eventually, each dataset sample comprises a consistent image pair (I1,I2), an inconsistent pair (I1′,I2′) generated with our data generation pipeline, subject masks (O1,O2), inpainted region masks (R1,R2), subject prompt s and target prompt p.

- 3.2 Learning Disentangled Semantic and Visual Representations Given an inconsistent image pair I1′ and I2′, which were generated with our automated pipeline in

- Section 3.1, where the inconsistent regions are defined by the masks R1 and R2, we aim to disentangle semantic and visual features of the diffusion model backbone. We start by extracting multi-layer

features for both images using the diffusion model backbone Φ, yielding feature maps F1 = Φ(I1′) and F2 = Φ(I2′) that include features from multiple decoder layers l ∈ L. To aggregate features from the different layers, we follow the approach of [22], but instead of using a single aggregation network, we employ two separate networks, Ψls and Ψlv, to aggregate semantic and visual representations separately. Each aggregation network encompasses a ResNet [11] block per decoder layer that are

Disentangled Features

Inconsistent

Image Pair

Semantic Similarity

| |ψs| |
|---|---|---|
| | | |

|𝑃1| |
|---|---|
| | |

|𝑃2| |
|---|---|
| | |

|[Figure 31]<br><br>[Figure 32]|
|---|

Semantic

Aggregation

𝑆1

|ℒ𝑠|
|---|

|∙| |
|---|---|
| | |

𝐷12𝑠

CE

|𝛟|
|---|

|𝑃2𝑜𝑢𝑡| |
|---|---|
| | |

𝑆2

|𝑃1𝑜𝑢𝑡| |
|---|---|
| | |

|∙| |
|---|---|
| | |

|ℒ𝑣𝑜𝑢𝑡|
|---|

CE

|[Figure 33]<br><br>[Figure 34]|
|---|

- 𝑉1

- 𝑉2

𝐷12𝑣

Visual

Visual

Aggregation

|ℒ𝑣𝑖𝑛|
|---|

|∙| |
|---|---|
| | |

CE

Similarity

|ψv|
|---|

| | |
|---|---|
|𝑃2𝑖𝑛| |

| | |
|---|---|
|𝑃1𝑖𝑛| |

|∙|
|---|

Residual blocks Dot blocks

Indexing Negation CE Cross-Entropy

- Figure 4: Overview of the proposed architecture for disentangling semantic and visual features from a frozen diffusion backbone Φ. Inconsistent regions between the input image pairs are indicated by binary masks (left). The semantic branch (blue) encourages the features of corresponding semantic points in both images, P1 and P2, to align. In the visual branch (yellow), we bring visually consistent points outside the inpainted regions, P1out and P2out, closer together, while pushing apart features at inconsistent points inside the inpainted regions, P1in and P2in.

combined using trainable scalar weights wl:

Si =

L

l

wsl Ψls(Fil) , Vi =

L

l

wvl Ψlv(Fil) , Si,Vi ∈ Rd×q , (3)

where Si, Vi are the semantic and visual features respectively, i ∈ {1,2}, d = w × h denotes the flattened spatial dimensions, and q is the feature dimensionality. This architecture allows for flexible selection of features that capture either semantic content or visual appearance. An overview of the architecture is provided in Figure 4.

Next, we compute cyclic similarity matrices between the aggregated semantic and visual features of the two images using dot products:

###### D12s = S1S2T , D21s = S2S1T , D12s ,D21s ∈ Rd×d , (4)

D12v = V1V2T , D21v = V2V1T , D12v ,D21v ∈ Rd×d , (5) Our objective is to encourage semantic features to be similar across all point correspondences, while visual features should be similar only outside the inconsistent regions Ri and dissimilar within them. To achieve this, we adopt a contrastive learning framework that brings similar features closer and pushes dissimilar ones apart. As a first step, we partition the set of pre-computed correspondences Ci from Section 3.1 into two subsets: points that fall inside and outside the inconsistent regions:

Piin = (xji,yij) (xji,yij) ∈ Ci, Ri(xji,yij) = 1 , (6) Piout = (xji,yij) (xji,yij) ∈ Ci, Ri(xji,yij) = 0 . (7)

Then, inspired by the contrastive objective used in CLIP [29], we define a semantic correspondence loss as follows:

###### Ls = CrossEntropy D12s (P1), P2 , Pi = Piin ∪ Piout . (8)

This semantic loss encourages matched points, whether inside or outside the inconsistent region, to exhibit similar semantic feature representations.

To disentangle appearance-specific features, we define a visual loss that explicitly separates consistent from inconsistent regions. For inconsistent regions, we use a negative similarity objective:

###### Linv = CrossEntropy − D12v (P1in), P2in . (9)

This term penalizes similarity in appearance features for points known to be visually inconsistent. For consistent regions, we retain the standard contrastive loss to encourage matching:

Loutv = CrossEntropy D12v (P1out), P2out . (10) Additionally, visually corresponding points inside the inpainted regions of the original consistent pairs I1 and I2 can be incorporated into Equation (10) to provide additional supervision; however, we omit the formal definition of this part for simplicity. All loss terms are also computed in the reverse direction as well using D21s and D21v , and each loss term is averaged over both directions. The final training objective combines semantic and visual losses as follows:

L = Ls + α(Linv + Loutv ) (11) where α is a scaling factor used to prioritize the visual branch, since the semantic branch can already be extracted reliably even without any aggregation, as shown in [42, 39].

###### 3.3 A Metric for Evaluating Subject-Driven Image Generation

Having disentangled visual and semantic features from the backbone of pre-trained diffusion models in the previous section, we now aim to leverage these features to estimate and localize visual consistency in subject-driven image generation. Given two test images I1 and I2 generated by a subject-driven generation method, our goal is to evaluate the visual consistency of the subject across the two images.

We begin by passing both images through our architecture to extract semantic and visual feature maps, denoted as Fis and Fiv, respectively, following the aggregation described in Equation (3). As in the previous section, we compute pairwise similarities between features to obtain semantic and visual similarity matrices, denoted by Ds and Dv, respectively. We then take the maximum similarity score for each point to obtain Dˆs = max(Ds) and Dˆv = max(Dv), representing the best per-point match in the semantic and visual domains.

Semantic correspondences are identified by selecting points whose semantic similarity exceeds a predefined threshold Ts, i.e., Dˆs > Ts, resulting in a set of confident semantic matches indexed by Js. This filtering ensures that we only consider regions that are both visible and semantically coherent. We set Ts = 0.7 in all experiments.

To assess visual consistency at these semantically matched locations, we use the same index set Js to retrieve the corresponding visual similarity scores from Dˆv. Given a visual similarity threshold Tv, we define the Visual Semantic Match (VSM) metric as:

1 |Js| j∈J

δ D ˆjv > Tv (12)

VSM(Tv) =

s

where δ[·] is the indicator function, equal to 1 if the condition holds and 0 otherwise.

This metric captures the proportion of semantically aligned regions that are also visually consistent, providing a quantitative measure of consistency between the two generated images. Inconsistent regions can be identified by subtracting visually matched locations from the semantically matched ones, or by examining regions in Dˆv with low visual similarity scores.

##### 4 Experiments

In this section, we evaluate the effectiveness of our approach and compare it against commonly used feature-based metrics such as CLIP [29] and DINO [4], as well as VLM-based approaches (e.g. ChatGPT-4o). Then, we provide an ablation study to analyze the impact of key architectural design choices and provide further insights into the behavior of our model.

###### 4.1 Implementation Details

Dataset: We use the data generation pipeline described in Section 3.1 to construct the dataset used to train our architecture. The pipeline takes consistent image pairs from the Subjects200k dataset [41] as input, although any subject-driven generation dataset can be used.

Visual Features

Visual Matching

Inpainting Mask

Semantic Features

Semantic Correspondence

Visual Correspondence

Scores

|[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]|[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]|[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]|[Figure 46]<br><br>[Figure 47]|
|---|---|---|---|
|[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]|[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]|[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]|[Figure 59]<br><br>[Figure 60]|

Inconsistent GenerationReference SubjectInconsistent GenerationReference Subject

- Figure 5: Qualitative examples of semantic and visual features, along with their correspondences, produced by our architecture. Regions that fall within the inpainting mask exhibit visually dissimilar features, enabling the detection of visual inconsistencies based on feature similarity. Dark Red is most consistent and Yellow is least consistent.

During generation, we apply several filtering steps to ensure high-quality training data. We discard samples with matching skewness below 1.3 (see Figure 3), and we enforce that the inconsistent region occupies between 5% and 60% of the subject area. Additionally, we filter out samples where the inpainted region has an LPIPS score below 0.15 to ensure sufficient visual inconsistency. The final dataset consists of 5,000 image pairs for training and 500 image pairs for validation.

Testset: We generate a testset of 100 samples using our dataset, and we manually revise it to ensure the reliability of evaluation.

Hyperparameters: We use the backbone of Stable Diffusion 2.1 [32] for all experiments following CleanDIFT [39]. We set the spatial resolution of the aggregated features in Equation (3) to h = w = 48, resulting in d = 48×48 = 2304, following CleanDIFT [39]. The projected feature dimensionality is set to q = 384, as in [22]. We empirically found that setting α = 10 in Equation (11) yields the best performance. We train the model for 30 epochs using the AdamW optimizer [21] with a learning rate of 1e-3 that is divided by 10 every 10 epochs. The training is done on 1 A100 GPU (40GB) and takes 12 hours. The source code for the dataset generation pipeline and the proposed architecture are publicly available. 1

###### 4.2 Evaluating the VSM Metric

To validate our proposed Visual-Semantic Match (VSM) metric, described in Section 3.3, we first perform a controlled evaluation where we define an oracle based on the inpainted regions for each image pair in the test set. This oracle is defined based on the ratio of the inconsistent region R1 to the total object mask O1 (see Figure 2), computed as:

Oracle = 1 −

###### R1 O1

(13)

This oracle reflects the ground-truth degree of visual consistency between the image pair. An effective metric should produce estimates that strongly correlate with the oracle. We evaluate against commonly

1https://github.com/abdo-eldesokey/mind-the-glitch

| |Controlled Inconsistency CLIP DINO VLM* VSM (Ours)<br><br>|Subject-Driven Generation CLIP DINO VLM* VSM (Ours)<br><br>|
|---|---|---|
|Pearson Spearman|-0.053 0.087 0.072 0.448<br><br>-0.005 0.120 0.091 0.582<br>|0.156 0.164 0.079 0.405 0.112 0.146 0.073 0.369<br><br>|

Table 1: Average correlation scores across methods. Our VSM achieves significantly higher correlation with the oracle than other metrics both on controlled and realistic settings. *(ChatGPT-4o)

Reference Subject Diptych [CVPR2025] DSD [CVPR2025] EasyControl [ICCV 2025]

[Figure 61]

[Figure 62]

[Figure 63]

- Figure 6: Qualitative examples of evaluating subject-driven image generation approaches using our proposed VSM metric and other existing approaches. Our VSM metric can accurately quantify and localize inconsistency and is more consistent with the oracle. Dark Red is most consistent and Yellow is least consistent.

used metrics in the literature for comparing image pairs: CLIP and DINO image-to-image similarity, as well as ChatGPT-4o as a representative Vision-Language Model (VLM). For VLM, we adopt the prompt used in [26], but modify it to produce a numerical score in the range of 0 to 100.

We report Pearson and Spearman correlations between each evaluated metric and the oracle in Table 1 (left). Our VSM metric shows significantly higher correlation compared to both feature-based similarity metrics and VLM judgments, indicating its greater reliability for measuring visual consistency. Qualitative results in Figure 5 illustrate semantic and visual features with their correspondences. While semantic features align with semantically similar regions regardless of appearance, visual features differ notably within inpainted areas and do not match across image pairs. This is evident in the score heatmaps shown in the rightmost column.

###### 4.3 Evaluating Subject-Driven Image Generation

To evaluate the effectiveness of our VSM metric in a realistic subject-driven image generation setting, we evaluate three recent methods: Diptych [38], DSD-Diffusion [3], and EasyControl [54]. Using the reference image I1 and target prompt p from our test set, we generate 100 images per method and evaluate them using VSM and other metrics as before. To compute the oracle, we manually annotate the generated images to mark visually inconsistent regions and calculate the oracle score as defined in Equation (13).

- Table 1 shows that our VSM metric consistently outperforms all other metrics, demonstrating strong generalization to real-world subject-driven generation. We also present qualitative examples in Figure 6, illustrating that VSM aligns most closely with the oracle. This is further supported by the KDE plots in Figure 7, where VSM shows the highest agreement with the oracle, while VLM tends to score most image pairs between 75–95 regardless of visual discrepancies.

Comparison of Score Distributions Compared to Oracle Oracle

10

- VSM (Tv=0.5)

- VSM (Tv=0.6) DINO

| |
|---|

8

| |
|---|

Density

6

CLIP VLM

| |
|---|

4

2

0

0.2 0.4 0.6 0.8 1.0

Score

- Figure 7: Score Distribution of different metrics compared to the Oracle.

layer6 layer7 layer8 layer9 layer10

0.0

0.2

0.4

Semantic weights

Visual weights

Figure 8: Aggregation weights.

Tv = 0.5 Tv = 0.7 α = 1 Skewness> 1.0 Skewness> 1.5 VSM (Ours)

Pearson 0.465 0.352 0.118 0.232 0.224 0.448 Spearman 0.454 0.496 0.104 0.250 0.225 0.582

Table 2: Ablation analysis of different hyperparameters.

- 4.4 Ablation Analysis

We visualize the learned aggregation weights for the semantic and visual branches in Figure 8. Visual features are primarily derived from decoder layers 8 and 9, while semantic features draw from layers 8 and 10, indicating that certain layers (e.g., layer 8) contribute to both representations. Notably, our learned semantic weights differ from those in [22], likely due to differences in supervision: their model is trained on sparse keypoints, whereas ours uses dense correspondences, favoring later layers with higher spatial resolution.

- Table 2 presents an ablation study of key hyperparameters. For the VSM similarity threshold Tv, both low (0.5) and high (0.7) values reduce correlation with the oracle, with Tv = 0.6 yielding the best performance. Setting α = 1 in the training loss significantly degrades performance by reducing emphasis on visual features in favor of semantic features, which are easier to learn and already present in diffusion backbones without additional training [42, 39]. In the dataset pipeline, high skewness thresholds overly restrict sample diversity, while low thresholds allow ambiguous matches—both leading to performance drops.

- 5 Limitations and Future Work

Our feature disentanglement is inherently partial, as visual features may still carry semantic information. Achieving full disentanglement, enabling visual matching across semantically different objects, remains a promising future direction. The quality of the learned features also depends on the reference dataset used in the automated pipeline. The Subjects200k dataset [41], while large-scale, was validated automatically and may include noisy pairs. Manual curation or improved filtering could enhance supervision quality. Additionally, detecting fine-grained inconsistencies is further limited by the spatial resolution of the diffusion features. We plan to address this via multi-scale aggregation and higher-resolution extraction. Lastly, our current framework targets structural and appearance-level inconsistencies. Extending it to handle broader variations, such as artistic style or color, may require further decomposing. We include failure cases in the supplementary to guide future work.

- 6 Conclusion

We proposed a framework for disentangling diffusion model features into semantic and visual components. Using an automated dataset pipeline with controlled visual inconsistencies, we trained a contrastive architecture to separate the two. This enables visual correspondence across images, complementing semantic matching, and supports fine-grained analysis. Leveraging this, we introduced a metric that quantifies and localizes inconsistencies in subject-driven generation. Our approach outperforms existing metrics, offering a more reliable and interpretable evaluation framework.

##### Acknowledgment

The research reported in this publication was supported by funding from King Abdullah University of Science and Technology (KAUST) - Center of Excellence for Generative AI, under award number 5940.

##### References

- [1] O. Avrahami, A. Hertz, Y. Vinker, M. Arar, S. Fruchter, O. Fried, D. Cohen-Or, and D. Lischinski. The chosen one: Consistent characters in text-to-image diffusion models. In ACM SIGGRAPH 2024 conference papers, pages 1–12, 2024.
- [2] Q. Bai, H. Ouyang, Y. Xu, Q. Wang, C. Yang, K. L. Cheng, Y. Shen, and Q. Chen. Edicho: Consistent image editing in the wild. In arXiv preprint arXiv:2412.21079, 2024.
- [3] S. Cai, E. Chan, Y. Zhang, L. Guibas, J. Wu, and G. Wetzstein. Diffusion self-distillation for zero-shot customized image generation. In CVPR, 2025.
- [4] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.
- [5] J. Chen, J. YU, C. GE, L. Yao, E. Xie, Z. Wang, J. Kwok, P. Luo, H. Lu, and Z. Li. Pixart-$\alpha$: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In The Twelfth International Conference on Learning Representations, 2024.
- [6] J. Chung, S. Hyun, H. Kim, E. Koh, M. Lee, and J.-P. Heo. Fine-tuning visual autoregressive models for subject-driven generation. arXiv preprint arXiv:2504.02612, 2025.
- [7] A. Cvejic, A. Eldesokey, and P. Wonka. Partedit: Fine-grained image editing using pre-trained diffusion models. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–11, 2025.
- [8] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Müller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [9] R. Gal, Y. Alaluf, Y. Atzmon, O. Patashnik, A. H. Bermano, G. Chechik, and D. Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.
- [10] L. Han, Y. Li, H. Zhang, P. Milanfar, D. Metaxas, and F. Yang. Svdiff: Compact parameter space for diffusion fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7323–7334, 2023.
- [11] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.
- [12] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.
- [13] J. Huang, X. Dong, W. Song, Z. Chong, Z. Tang, J. Zhou, Y. Cheng, L. Chen, H. Li, Y. Yan, et al. Consistentid: Portrait generation with multimodal fine-grained identity preserving. arXiv preprint arXiv:2404.16771, 2024.
- [14] L. Huang, W. Wang, Z.-F. Wu, Y. Shi, H. Dou, C. Liang, Y. Feng, Y. Liu, and J. Zhou. In-context lora for diffusion transformers. arXiv preprint arxiv:2410.23775, 2024.
- [15] J. Jin, Z. Yu, Y. Shen, Z. Fu, and J. Yang. Latexblend: Scaling multi-concept customized generation with latent textual blending. arXiv preprint arXiv:2503.06956, 2025.
- [16] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023.
- [17] N. Kumari, B. Zhang, R. Zhang, E. Shechtman, and J.-Y. Zhu. Multi-concept customization of text-to-image diffusion, 2023.

- [18] Z.-Y. Li, R. Du, J. Yan, L. Zhuo, Z. Li, P. Gao, Z. Ma, and M.-M. Cheng. Visualcloze: A universal image generation framework via visual in-context learning. arXiv preprint arXiv:2504.07960, 2025.
- [19] C. Liu, V. Shah, A. Cui, and S. Lazebnik. Unziplora: Separating content and style from a single image. arXiv preprint arXiv:2412.04465, 2024.
- [20] W. Liu, J. Mao, J. Hsu, T. Hermans, A. Garg, and J. Wu. Composable part-based manipulation. In 7th Annual Conference on Robot Learning, 2023.
- [21] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [22] G. Luo, L. Dunlap, D. H. Park, A. Holynski, and T. Darrell. Diffusion hyperfeatures: Searching through time and space for semantic correspondence. Advances in Neural Information Processing Systems, 36:47500–47510, 2023.
- [23] L. Meng, S. Lan, H. Li, J. M. Alvarez, Z. Wu, and Y.-G. Jiang. Segic: Unleashing the emergent correspondence for in-context segmentation. In European Conference on Computer Vision, pages 203–220. Springer, 2024.
- [24] J. Min, J. Lee, J. Ponce, and M. Cho. Spair-71k: A large-scale benchmark for semantic correspondence. arXiv preprint arXiv:1908.10543, 2019.
- [25] C. Mou, X. Wang, J. Song, Y. Shan, and J. Zhang. Dragondiffusion: Enabling drag-style manipulation on diffusion models. arXiv preprint arXiv:2307.02421, 2023.
- [26] Y. Peng, Y. Cui, H. Tang, Z. Qi, R. Dong, J. Bai, C. Han, Z. Ge, X. Zhang, and S.-T. Xia. Dreambench++: A human-aligned benchmark for personalized image generation. In The Thirteenth International Conference on Learning Representations, 2025.
- [27] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024.
- [28] G. Qian, K.-C. Wang, O. Patashnik, N. Heravi, D. Ostashev, S. Tulyakov, D. Cohen-Or, and K. Aberman. Omni-id: Holistic identity representation designed for generative tasks. arXiv preprint arXiv:2412.09694, 2024.
- [29] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [30] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.
- [31] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan, et al. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024.
- [32] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [33] L. Rout, Y. Chen, N. Ruiz, A. Kumar, C. Caramanis, S. Shakkottai, and W.-S. Chu. RB-modulation: Training-free stylization using reference-based modulation. In The Thirteenth International Conference on Learning Representations, 2025.
- [34] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman. Dreambooth: Fine tuning text-toimage diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500–22510, 2023.
- [35] M. Safaee, A. Mikaeili, O. Patashnik, D. Cohen-Or, and A. Mahdavi-Amiri. Clic: Concept learning in context. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6924–6933, 2024.
- [36] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.

- [37] V. Shah, N. Ruiz, F. Cole, E. Lu, S. Lazebnik, Y. Li, and V. Jampani. Ziplora: Any subject in any style by effectively merging loras. In European Conference on Computer Vision, pages 422–438. Springer, 2024.
- [38] C. Shin, J. Choi, H. Kim, and S. Yoon. Large-scale text-to-image model with inpainting is a zero-shot subject-driven image generator. arXiv preprint arXiv:2411.15466, 2024.
- [39] N. Stracke, S. A. Baumann, K. Bauer, F. Fundel, and B. Ommer. Cleandift: Diffusion features without noise. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.
- [40] K. Sun, X. Liu, Y. Teng, and X. Liu. Personalized text-to-image generation with auto-regressive models. arXiv preprint arXiv:2504.13162, 2025.
- [41] Z. Tan, S. Liu, X. Yang, Q. Xue, and X. Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 3, 2024.
- [42] L. Tang, M. Jia, Q. Wang, C. P. Phoo, and B. Hariharan. Emergent correspondence from image diffusion. Advances in Neural Information Processing Systems, 36:1363–1389, 2023.
- [43] Y. Tewel, O. Kaduri, R. Gal, Y. Kasten, L. Wolf, G. Chechik, and Y. Atzmon. Training-free consistent text-to-image generation. ACM Transactions on Graphics (TOG), 43(4):1–18, 2024.
- [44] Q. Wang, X. Bai, H. Wang, Z. Qin, A. Chen, H. Li, X. Tang, and Y. Hu. Instantid: Zero-shot identitypreserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024.
- [45] Q. Wang, A. Eldesokey, M. Mendiratta, F. Zhan, A. Kortylewski, C. Theobalt, and P. Wonka. Zero-shot video semantic segmentation based on pre-trained diffusion modelss. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.
- [46] Y. Wei, Y. Zheng, Y. Zhang, M. Liu, Z. Ji, L. Zhang, and W. Zuo. Personalized image generation with deep generative models: A decade survey. arXiv preprint arXiv:2502.13081, 2025.
- [47] Y. Wu, L. Zhu, L. Liu, W. Qiao, Z. Li, L. Yu, and B. Li. Proxy-tuning: Tailoring multimodal autoregressive models for subject-driven image generation. arXiv preprint arXiv:2503.10125, 2025.
- [48] H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.
- [49] Y. Zeng, V. M. Patel, H. Wang, X. Huang, T.-C. Wang, M.-Y. Liu, and Y. Balaji. Jedi: Joint-image diffusion models for finetuning-free personalized text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6786–6795, June 2024.
- [50] Y. Zeng, V. M. Patel, H. Wang, X. Huang, T.-C. Wang, M.-Y. Liu, and Y. Balaji. Jedi: Joint-image diffusion models for finetuning-free personalized text-to-image generation. In CVPR, 2024.
- [51] J. Zhang, C. Herrmann, J. Hur, E. Chen, V. Jampani, D. Sun, and M.-H. Yang. Telling left from right: Identifying geometry-aware semantic correspondence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3076–3085, 2024.
- [52] J. Zhang, C. Herrmann, J. Hur, L. Polania Cabrera, V. Jampani, D. Sun, and M.-H. Yang. A tale of two features: Stable diffusion complements dino for zero-shot semantic correspondence. Advances in Neural Information Processing Systems, 36:45533–45547, 2023.
- [53] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.
- [54] Y. Zhang, Y. Yuan, Y. Song, H. Wang, and J. Liu. Easycontrol: Adding efficient and flexible control for diffusion transformer. arXiv preprint arXiv:2503.07027, 2025.

Supplementary Material for:

# Mind-the-Glitch: Visual Correspondence for Detecting Inconsistencies in Subject-Driven Generation

#### A Additional Qualitative Examples

Figures 9 and 10 presents additional qualitative examples of visual and semantic correspondences computed from the disentangled features produced by our architecture under controlled settings (i.e. inpainting). As shown in the figure, semantic features tend to match across semantically similar regions regardless of visual appearance, whereas visual features match only in regions with similar appearance. To enable the localization of inconsistencies, we provide heatmap visualizations of the visual feature matching scores, which highlight areas identified as visually inconsistent. This level of spatial interpretability is not offered by existing metrics, including CLIP, DINO, and VLM-based approaches.

We also provide more qualitative examples for the real settings when evaluating subject-driven image generation approaches in Figures 11 and 12.

#### B Additional Ablation Analysis

We provide additional ablation experiments that are summarized in Table 3. We evaluate on the test set and we report the correlations between the predicted VSM metric and the oracle as explained in

- Section 4.2 of the main paper.

Why do we need a semantic aggregation branch?: A natural question is why we train a semantic branch at all, given that we could use pre-computed diffusion features, such as those from layer 6 of the decoder, as semantic features, as done in [5, 6]. However, Table 3 shows that using these zero-shot features without a dedicated semantic aggregation network significantly degrades performance. This performance drop arises because, when the semantic and visual branches are trained jointly using a contrastive objective, the two representations become effectively disentangled and spatially aligned. This alignment is essential for accurately computing the VSM metric, which compares semantic and visual features across different regions of the subject to identify inconsistencies. Without spatial alignment, the metric cannot reliably quantify and localize inconsistent regions.

Using 2 Residual Blocks per Layer: Increasing the number of residual blocks per decoder layer in our aggregation network from 1 to 2 results in a slight improvement in Spearman correlation. However, this gain comes at a significant computational cost, increasing the model size by approximately 25%, from 4.5M to 6M parameters.

Varying the dimensionality of aggregated features q: Increasing the dimensionality of the aggregated features from 384 to 512 results in a significant drop in performance. This may be due to the introduction of redundant channels, which can introduce noisy features and degrade the representation quality, a phenomenon also observed in [2]. Reducing the dimensionality from 384 to 256 leads to a graceful decline in performance, suggesting that q = 384 is an optimal choice.

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

0-shot Sem. 2× Res. q = 256 q = 512 α = 20 M = 2500 VSM (Ours)

Pearson 0.267 0.472 0.413 0.275 0.412 0.437 0.448 Spearman 0.310 0.453 0.416 0.247 0.369 0.411 0.582

- Table 3: Additional ablation analysis of different hyperparameters and design choices. We report the correlation between the VSM metric under different hyperparameters and the oracle on the test set. 0-shot Sem. refers to using a zero-shot semantic features from a diffusion backbone based on CleanDIFT [5]. 2× Res. refers to using 2 residual blocks instead of 1 in our aggregation network. M = 2500 refers to training on half the size of the training set.

Higher α for the Visual Branch: As demonstrated in the main paper, setting a lower value of α = 1 degrades performance by placing insufficient emphasis on learning visual features. Improved results were observed with α = 10, indicating better balance between the semantic and visual branches. However, increasing α further to 20 leads to a performance drop, suggesting that α = 10 is the optimal trade-off point.

Fewer Training Samples: To examine whether the full training set of 5,000 samples is sufficient for effective feature disentanglement, we train the model using only half of the data, i.e., 2,500 samples, and we train for the same number of epochs. We observe a performance drop of approximately ∼ 10%, suggesting that the model already performs well even with half the training data. This indicates that performance gains diminish as more data is added, implying that a training set of 5,000 samples is sufficient to learn this task effectively.

#### C Additional Details on Benchmarking Subject-Driven Generation Approaches

To demonstrate the effectiveness of our proposed VSM metric in capturing and localizing inconsistencies in subject-driven image generation approaches, we evaluated several recent methods using existing metrics, including CLIP and DINO image-to-image similarity, as well as the VLM-based evaluation from DreamBench++ [3]. For evaluation, we used our test set, where each method is given only the subject image and prompted to generate a new image of the subject in a different environment.

The following methods are evaluated:

- • Diptych Prompting [4]: We use the official GitHub implementation1, which is based on the FLUX-dev backbone. Diptychs are generated using the official template, with original image captions and corresponding target caption pairs. Specifically, we use the prompt: "A diptych with two side-by-side images of the same subject. On the left, a photo of subject. On the right, replicate this subject exactly but as target prompt." All hyperparameters, including step size, scheduler, and guidance scale, are kept at their default values. Inference is performed at a resolution of 768×1536, producing a final output of 768×768. Each sample takes an average of 92 seconds to generate, with a maximum VRAM usage of 55 GB.
- • DSD Diffusion [1]: We use the official DSD Diffusion repository2 with the FLUX model. Inference is performed with 28 steps, a guidance scale of 3.5 (default), and an input resolution of 512×1024, producing a final output of 512×512. The model is conditioned on the subject image along with the description and target prompt. Inference is performed using bfloat16, taking an average of 8 seconds per sample and a maximum VRAM usage of 34 GB. DSD’s prompt enhancement with Gemini is not used in order to ensure a fair comparison with other methods.
- • EasyControl [7]: We ran EasyControl3 on the FLUX model using the subject control variant, conditioned on the description, target prompt, and subject image. All adapter

- 1https://github.com/chaehunshin/DiptychPrompting
- 2https://github.com/cai-lab/DSD-Diffusion
- 3https://github.com/Xiaojiu-z/EasyControl

weights and step parameters were kept as in the original paper: 25 inference steps and a guidance scale of 3.5, with the exception of the resolution, which was reduced from the default 1024×768 to 768×768. This method exhibited numerical instabilities when using float16 or bfloat16, resulting in completely black outputs for approximately 16% of samples. This issue persisted regardless of resolution or the presence of safety checkers. To avoid this, we ran EasyControl using full precision. On average, inference takes 94 seconds per sample, with a maximum VRAM usage of 70 GB.

#### D Additional Dataset Filtration Strategy

In our dataset generation pipeline, we apply additional heuristics beyond the matching skewness and LPIPS thresholds described in Section 3.1 of the main paper. These heuristics aim to mitigate failures in segmenting valid subject parts and ensure reliable training data.

Relative Size Check: We compute the relative size of each part mask Ri with respect to its corresponding object mask Oi using the ratio:

Ri Oi

ri =

.

We ensure that ri falls within the range [0.05,0.6], meaning the segmented region must occupy between 5% and 60% of the object. Additionally, we compare r1 and r2 between the two images in a pair and enforce that their relative size difference is less than 0.1.

Relative Aspect Ratio Check: For each part mask Ri, we compute its horizontal and vertical aspect ratios relative to the object mask Oi, defined as the ratios between the width and height of the part and those of the full object. We then compare these ratios between the two images to ensure consistency, requiring that the differences in both horizontal and vertical ratios are within 20%.

#### E VLM Evaluation Prompts

We follow the same setup as DreamBench++ [3] and we use the GPT-4o model with 0 temperature for deterministic output. We modify the system and user prompts slightly to provide a score from 0 to 100 to align with other metrics and the user-study. We provide the modified prompts in Figures 13 and 14.

#### F Failure Cases

We include representative failure cases in Figure 15 to highlight current limitations and guide future research. A major challenge lies in detecting incomplete or corrupted generations, as shown in (a) and (b). In the case of missing object parts, it is often unclear to tell whether a part was not generated or simply occluded. For corrupted outputs, identifying whether the subject itself is degraded remains difficult. Another limitation arises with significant lighting changes, as in (c), where visual features fail to match due to the absence of such variations in training data. Lastly, under- or over-segmentation leads to inaccurate VSM scores, since the metric evaluates less than or more than the intended subject region.

Visual Features

Predicted Inconsistent Regions

Inpainting

Semantic Features

Semantic Correspondence

Visual Correspondence

Mask

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Reference Subject

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Inconsistent GenerationInconsistent GenerationReference SubjectReference SubjectInconsistent GenerationInconsistent GenerationReference Subject

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

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

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Figure 9: Additional qualitative examples (page 1) of semantic and visual correspondences computed from the disentangled features produced by our proposed architecture. Heatmaps of visual feature matching scores are also shown, highlighting the visually inconsistent regions. Dark Red is most consistent and Yellow is least consistent.

Visual Features

Predicted Inconsistent Regions

Inpainting

Semantic Features

Semantic Correspondence

Visual Correspondence

Mask

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

Reference Subject

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Inconsistent GenerationInconsistent GenerationReference SubjectReference SubjectInconsistent GenerationInconsistent GenerationReference Subject

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

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

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

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

Figure 10: Figure 9 continued.

Reference Subject Diptych [CVPR2025] DSD [CVPR2025] EasyControl [ICCV 2025]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Figure 11: Additional qualitative examples of evaluating subject-driven image generation approaches using our proposed VSM metric and other existing approaches. Our VSM metric can accurately quantify and localize inconsistency and is more consistent with the oracle. Dark Red is most consistent and Yellow is least consistent.

Reference Subject Diptych [CVPR2025] DSD [CVPR2025] EasyControl [ICCV 2025]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Figure 12: Figure 11 continued.

### Task Definition You will be provided with an image generated based on reference image. As an experienced evaluator, your task is to evaluate the semantic consistency between the subject of the generated image and the reference image, according to the scoring criteria.

### Scoring Criteria It is often compared whether two subjects are consistent based on four basic visual features:

- 1. Shape: Evaluate whether the main body outline, structure, and proportions of the generated image match those of the reference image. This includes the geometric shape of the main body, clarity of edges, relative sizes, and spatial relationships between various parts composing the main body.
- 2. Color: Comparing the accuracy and consistency of the main colors generated in the image with those of the reference image. This includes saturation, hue, brightness, and whether the distribution of colors is similar to that of the subject in the reference image.
- 3. Texture: Focus on the local parts of the RGB image, whether the generated image effectively captures fine details without appearing blurry, and whether it possesses the required realism, clarity, and aesthetic appeal. Please note that unless specifically mentioned in the text prompt, excessive abstraction and formalization of texture are not necessary.
- 4. Facial Features: If the evaluation is of a person or animal, facial features will greatly affect the judgment of image consistency, and you also need to focus on judging whether the facial area looks very similar visually.

### Scoring Range You need to give a specific integer score based on the comprehensive performance of the visual features above, ranging from 0 to 100: Here are few examples of how to compute the score.

- - Very Poor (0): No resemblance. The generated image's subject has no relation to the reference.
- - Poor (25): Minimal resemblance. The subject falls within the same broad category but differs significantly.
- - Fair (50): Moderate resemblance. The subject shows likeness to the reference with notable variances.
- - Good (75): Strong resemblance. The subject closely matches the reference with only minor discrepancies.
- - Excellent (100): Near-identical. The subject of the generated image is virtually indistinguishable from the reference.

### Input format Every time you will receive two images, the first image is a reference image, and the second image is the generated image.

Please carefully review each image of the subject.

### Output Format Score: [Your Score]

You must adhere to the specified output format, which means that only the scores need to be output, excluding your analysis process.

- Figure 13: User prompt used for the VLM evaluation using GPT-4o model.

Yes, I understand the task. It involves evaluating the semantic consistency between a reference image and a generated image based on specific criteria. The evaluation focuses on four main aspects: shape, color, texture, and facial features (if applicable). The goal is to determine how closely the generated image matches the reference image in terms of these aspects. The evaluation should result in a specific score ranging from 0 (no resemblance) to 100 (near-identical resemblance).

To evaluate the images, I plan to follow these steps:

- 1. **Shape**: Assess if the main body outline, structure, and proportions of the generated image are consistent with the reference image. This includes looking at the geometric shape, clarity of edges, relative sizes, and spatial relationships between various parts.
- 2. **Color**: Compare the main colors in terms of accuracy and consistency, including saturation, hue, brightness, and the distribution of colors.
- 3. **Texture**: Examine the details in the local parts of the image to see if the generated image captures fine details without appearing blurry and maintains realism, clarity, and aesthetic appeal.
- 4. **Facial Features**: If the subject includes a person or animal, closely compare facial features to judge visual similarity.

After analyzing these aspects, I will assign a score based on the overall performance of the generated image in relation to the reference image. The score will reflect how similar the generated image is to the reference, strictly adhering to the evaluation criteria provided.

My output format should be Score: [0-100], and I don't need to write out the specific analysis process.

Please provide me with the samples I need to evaluate.

- Figure 14: System prompt used for the VLM evaluation using GPT-4o model.

(a) Incomplete generations (b) Corrupted generations

[Figure 178]

[Figure 179]

- (c) Excessive Light Changes

[Figure 180]

[Figure 181]

- (d) Segmentation Failures

[Figure 182]

[Figure 183]

Figure 15: Examples of failure cases.

#### References

- [1] S. Cai, E. Chan, Y. Zhang, L. Guibas, J. Wu, and G. Wetzstein. Diffusion self-distillation for zero-shot customized image generation. In CVPR, 2025.
- [2] T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski. Vision transformers need registers. In The Twelfth International Conference on Learning Representations, 2024.
- [3] Y. Peng, Y. Cui, H. Tang, Z. Qi, R. Dong, J. Bai, C. Han, Z. Ge, X. Zhang, and S.-T. Xia. Dreambench++: A human-aligned benchmark for personalized image generation. In The Thirteenth International Conference on Learning Representations, 2025.
- [4] C. Shin, J. Choi, H. Kim, and S. Yoon. Large-scale text-to-image model with inpainting is a zero-shot subject-driven image generator. arXiv preprint arXiv:2411.15466, 2024.
- [5] N. Stracke, S. A. Baumann, K. Bauer, F. Fundel, and B. Ommer. Cleandift: Diffusion features without noise. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.
- [6] L. Tang, M. Jia, Q. Wang, C. P. Phoo, and B. Hariharan. Emergent correspondence from image diffusion. Advances in Neural Information Processing Systems, 36:1363–1389, 2023.
- [7] Y. Zhang, Y. Yuan, Y. Song, H. Wang, and J. Liu. Easycontrol: Adding efficient and flexible control for diffusion transformer. arXiv preprint arXiv:2503.07027, 2025.

