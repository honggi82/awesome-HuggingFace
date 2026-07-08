## RestoreFormer++: Towards Real-World Blind Face Restoration from Undegraded Key-Value Pairs

Zhouxia Wang, Jiawei Zhang, Tianshui Chen, Wenping Wang, Fellow, IEEE, and Ping Luo, Member, IEEE

### arXiv:2308.07228v1[cs.CV]14Aug2023

Abstract—Blind face restoration aims at recovering high-quality face images from those with unknown degradations. Current algorithms mainly introduce priors to complement high-quality details and achieve impressive progress. However, most of these algorithms ignore abundant contextual information in the face and its interplay with the priors, leading to sub-optimal performance. Moreover, they pay less attention to the gap between the synthetic and real-world scenarios, limiting the robustness and generalization to real-world applications. In this work, we propose RestoreFormer++, which on the one hand introduces fully-spatial attention mechanisms to model the contextual information and the interplay with the priors, and on the other hand, explores an extending degrading model to help generate more realistic degraded face images to alleviate the synthetic-to-real-world gap. Compared with current algorithms, RestoreFormer++ has several crucial benefits. First, instead of using a multi-head self-attention mechanism like the traditional visual transformer, we introduce multi-head cross-attention over multi-scale features to fully explore spatial interactions between corrupted information and high-quality priors. In this way, it can facilitate RestoreFormer++ to restore face images with higher realness and fidelity. Second, in contrast to the recognition-oriented dictionary, we learn a reconstruction-oriented dictionary as priors, which contains more diverse high-quality facial details and better accords with the restoration target. Third, we introduce an extending degrading model that contains more realistic degraded scenarios for training data synthesizing, and thus helps to enhance the robustness and generalization of our RestoreFormer++ model. Extensive experiments show that RestoreFormer++ outperforms state-of-the-art algorithms on both synthetic and real-world datasets. Code will be available at https://github.com/wzhouxiff/RestoreFormerPlusPlus.

Index Terms—Blind Face Restoration, Transformer, Cross-Attention Mechanism, Dictionary, Computer Vision, Real-World.

✦

1 INTRODUCTION

# B

LIND face restoration aims at recovering high-quality faces from a series of unknown degradations, such as

blur, noise, downsampling, compression artifacts, etc. These degradations are complex and diverse in real-world scenarios, leaving limited information in the degraded face image. Therefore, the face restored directly from its degraded one is not good enough, even with powerful DNN structures [8], [9], [10], [11], [12]. Introducing priors to complement additional high-quality details can effectively solve this issue [1], [2], [4], [5], [6], [13], [14], [15], [16], [17], [18], [19].

Despite the acknowledgment of progress, current priorbased algorithms mainly depend on geometric priors [4], [13], [14], [15], [16], [17], [18] or recognition-oriented references [1], which are not accordant to the restoration task and thus lead to sub-optimal performance. The geometric priors are landmarks [13], [14], facial parsing maps [4], [15], or facial component heatmaps [16] that mainly provide shape informa-

- • Zhouxia Wang and Wenping Wang are with The University of Hong Kong, Hong Kong SAR, China. E-mail: {wzhoux@.connect, wenping@cs}.hku.hk
- • Jiawei Zhang is with SenseTime Research, China. E-mail: zhjw1988@gmail.com
- • Tianshui Chen is with The Guangdong University of Technology, Guangzhou, China. E-mail: tianshuichen@gmail.com
- • Ping Luo is with The University of Hong Kong and Shanghai AI Laboratory, China. E-mail: pluo@cs.hku.hk

tion to aid face restoration. Recognition-oriented references like the facial component dictionaries in DFDnet [1] are extracted with a recognition model and only cover limited facial components, such as eyes, mouth, and nose. Therefore, the restored faces of these algorithms tend to lack details. For example, in Fig. 1, the results of PSFRGAN [4], whose priors are facial parsing maps, and DFDnet [1] fail to recover facial details, especially in hair areas. Although the generative priors encapsulated in a face generation network aim at face reconstruction and achieve superior performance compared to the previous two kinds of priors, their restored results still fail to yield fine-grained facial details or exist obvious artifacts. Examples are the restored results of Wan et al. [2] and GPEN [5] in Fig. 1.

On the other hand, effectively integrating the identity information in the degraded face and high-quality details in the priors is a critical step to attaining face images in both realness and fidelity. However, current methods either take the degraded faces as supervision, e.g., PULSE [3], or locally combine these two kinds of information by pixelwise concatenation [20], [21], [22], spatial feature transform (SFT) [1], [4], [19], [23], or deformable operation [6]. They ignore the useful contextual information in the face image and its interplay with priors, and thus most of them cannot trade off the fidelity and realness of their restored results well. A typical example is PULSE [3]. As shown in Fig. 1, their restored results perform well in realness, but their identities cannot be preserved.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

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

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Input DFDNet [1] Wan et al. [2] PULSE [3] PSFRGAN [4] GPEN [5] VQFR [6] RestoreFormer [7] RestoreFormer++ real-world ECCV 20 CVPR 20 CVPR 20 CVPR 21 CVPR 21 ECCV 22 CVPR 22 Ours

- Fig. 1. Comparisons with state-of-the-art face restoration methods on some degraded real-world images. Our conference version, RestoreFormer [7], produces restored results with rich details and complete structures, making them more natural and authentic than the results of other methods. Our current version, RestoreFormer++, extends the multi-scale mechanism and EDM to remove haze from the degraded face images and process uneven degradation (highlighted with a red box in the third sample), resulting in a clearer and more pleasant look.

In this work, we propose RestoreFormer++, which introduces fully-spatial attention mechanisms to model the contextual information in the face image and its interplay with priors matched from a reconstruction-oriented dictionary. Unlike the existing ViT methods [24], [25], [26], [27] that achieve fully-spatial attentions with multi-head self-attention mechanism (MHSA) (Fig. 2 (a)), our RestoreFormer++ is equipped with multi-head cross-attention mechanism (MHCA) (Fig. 2 (b)) whose queries are the features of degraded face image while key-value pairs are highquality facial priors. In addition, MHCAs are applied to multi-scale features, enabling RestoreFormer++ to model the contextual information based on both semantic and structural information and effectively improve the restored performance in both realness and fidelity. It is also worth mentioning that the priors adopted in our work have better quality since they are from a reconstruction-oriented high-quality dictionary (ROHQD). Its elements are learned from plenty of uncorrupted faces by a high-quality face generation neural network implemented with the idea of vector quantization [28]. They are rich in high-quality facial details specifically aimed at face reconstruction (see Fig. 3 for a more intuitive comparison with the recognition-oriented dictionary).

In addition, RestoreFomer++ contains an extending degrading model (EDM) to generate more realistic degraded face images for alleviating the synthetic-to-real-world gap and further improving its robustness and generalization toward real-world scenarios. Observations show that in the real world, besides blur, noise, downsampling, and compression artifacts, haze and uneven degradation are also common. Relevant examples are shown in Fig. 1. However, existing methods cannot handle these degradations well. Therefore, we introduce haze and uneven degradation into our EDM, which enables RestoreFormer++ to effectively remove the haze covered in the degraded face images and avoid the artifacts raised by uneven degradation. Besides,

EDM applies a spatial shift operation on the high-quality face before synthesizing the degraded face to reduce the effect introduced by inaccurate face alignment. Due to the specificity of face structure, aligning the degraded face to a reference face (in this work, the reference face is from FFHQ [29], and its landmarks are shown as green points in Fig. 6 and Fig. 8) is helpful for the restoration of face images [1], [4], [19]. However, misalignment caused by severe degradation will lead to errors while restoring with existing methods. For example, as shown in the second sample in Fig. 8, its left eyebrow is aligned with the left eye of the reference image, and the existing methods, such as PSFGAN [4], GFP-GAN [19], and our conference version [7], tend to restore the left eye near the eyebrow area instead of its original area in the degraded face image. The small spatial shift adopted in EDM can improve the tolerance of RestoreFormer++ for face alignment error, thus improving its restoration performance as in Fig. 8 (g).

This work is an extension of our conference version [7]. In this version, we strengthen the work from three aspects. First, we extend our multi-head attention mechanisms used for fusing the degraded facial features and their corresponding high-quality facial priors from single-scale to multi-scale. This enables RestoreFormer++ to model contextual information based on both semantic and structural information, effectively improving the restored performance in both realness and fidelity. Second, we proposed an extending degrading model (EDM) to alleviate the synthetic-to-real-world gap and further improve the robustness and generalization of our RestoreFormer++ toward real-world scenarios. Finally, we conduct more experiments and analyses to verify the superiority of RestoreFormer++ against existing methods and the contributions of each component in RestoreFormer++.

In conclusion, our main contributions are as follows:

• We propose RestoreFormer++, which on the one hand introduces multi-head cross-attention mechanisms to model the fully-spatial interaction between the

degraded face and its corresponding high-quality priors and on the other hand, explores an extending degrading model to synthesize more realistic degraded face images for model training. It can restore face images with higher realness and fidelity for both synthetic and real-world scenarios.

- • We introduce a reconstruction-oriented high-quality dictionary learning algorithm to generate priors that are more accordant to the face restoration task and thus provide suitable priors to RestoreFormer++ to restore faces with better realness.
- • The extending degrading model contains more kinds of realistic degradations and simulates the face misaligned situation to further alleviate the syntheticto-real-world gap. It improves the robustness and generalization of RestoreFormer++.
- • Extensive experiments show that RestoreFormer++ outperforms current leading competitors on both synthetic and real-world datasets. We also conduct detailed ablation studies to analyze the contribution of each component to give a better understanding of RestoreFormer++.

The remaining of this work is organized as follows. We review the most related works in Sec. 2 and detailedly introduce the RestoreFormer++ in Sec. 3. We then present experiments with comparison and analysis in Sec. 4. Finally, conclusions are in Sec. 5.

2 RELATED WORKS

- 2.1 Blind Face Restoration

Blind face restoration aims to recover high-quality faces from face images that have undergone unknown and complex degradations. Owing to the effectiveness of Deep Neural Networks (DNN) [8], [30], [31], researchers [9], [10], [11], [12] have attempted to restore high-quality faces directly from degraded ones using DNN-based approaches. However, since the information contained in degraded faces is limited, researchers have sought assistance from additional priors, such as geometric priors [4], [13], [14], [15], [16], [17], [18], [32], [33], reference priors [1], [20], [21], [22], and generative priors [2], [3], [19], [34]. Most geometric priors are predicted from the degraded faces, and the quality of these priors is significantly constrained by the degree of degradation in the face images, which further impacts the final restoration results. Reference priors, which are high-quality faces distinct from degraded ones, alleviate the limitations of geometric priors. However, exemplars [20], [21], [22] with the same identity as the degraded face are not always available, and facial component dictionaries extracted from high-quality face images are partial and recognition-oriented, restricting the performance of reference-based methods. Recent studies [2], [3], [5], [19] have suggested that generative priors encapsulated in well-trained high-quality face generators possess considerable potential for blind face restoration, and works [6], [35], [36], published concurrently or after our conference version, propose obtaining high-quality priors from a codebook similar to our ROHQD. However, most of these previous studies employ pixel-wise concatenation [20], [21], [22], spatial feature transform (SFT) [1], [4], [19], [23],

or deformable operation [6] to fuse the degraded feature and priors. Both SFT [23] and deformable networks [37] are implemented with convolutional layers, and their receptive fields limit the attentive areas, leading to the neglect of useful contextual information when fusing degraded information and its corresponding priors.

In contrast, our RestoreFormer++ is a unified framework for globally modeling the contextual information in the face with fully-spatial attention while fusing the features of the degraded face and their corresponding priors matched from a reconstruction-oriented dictionary. Due to the rich contextual information and high-quality priors, RestoreFormer++ performs better than previous related methods in both realness and fidelity.

###### 2.2 Vision Transformer

These years, transformer [38] designed with attention mechanism performs pretty well on natural language processing areas [39], [40] and researchers turn to explore the potential possibility of transformer on computer vision. The first attempt is ViT [26], a pure transformer that takes sequences of image patches as input. It achieves high performance on image classification tasks. Then more works extend the transformer to object detection [24], [27], segmentation [41], and even low-level vision [25], [42], [43], [44], [45], [46], which may suffer from more difficulties on efficiency. In the low-level vision, Chen et al. [25] take the advantages of transformer on a large scale pre-training to build a model that covers many image processing tasks. Esser et al. [42] apply the transformer on codebook-indices directly to make the generation of a very high-resolution image possible. Zhu et al. [46] exploit the global structure of the face extracted by the transformer to help the synthesis of photo-sketch. Most of these works tend to search the global information in the patches of an image with a self-attention mechanism. To model the interplay between the degraded face and its corresponding priors cooperating with contextual information, RestoreFormer++ adopts multi-scale multi-head cross-attention mechanisms whose queries are the features of the corrupted face and key-value pairs are the priors.

###### 2.3 Face Degrading Model

Since there is no real training pair in blind face restoration, most previous works synthesize the training pairs with a degrading model. The degrading model proposed in [12] mainly consists of blur kernels, downsampling, and Gaussian noise. In this version, Gaussian noise is added before downsampling. Li et al. [22] find that adding Gaussian noise after downsampling can better simulate the long-distance image acquisition. They further upgrade the degrading model with JPEG compression. Most of the later methods follow this degrading model for degraded face synthesis except the work proposed by Wan et al. [2] that mainly focuses on the old photos that suffer from scratch texture. To further diminish the gap between the synthetic and real-world datasets, our EDM extends the degrading model proposed in [22] with additional commonly existing degradations: haze and uneven degradation. It also applies a spatial shift to highquality face images while synthesizing the degraded face to alleviate the inherent bias introduced by face alignment.

𝒁𝑑0

𝒁𝑑0

𝒁𝑝0

[Figure 28]

|Q| |
|---|---|
| | |

|K| |
|---|---|
| | |

|V| |
|---|---|
| | |

|Q| |
|---|---|
| | |

|K| |
|---|---|
| | |

|V| |
|---|---|
| | |

MHCAs

###### …

𝒁𝒁𝑑𝑑𝑆𝑆−1 𝒁𝒁1𝑑𝑑

###### …

|𝒁𝒁𝑑𝑑0|
|---|

Multi-Headed Attention

Multi-Headed Attention

|𝐄𝐄𝑑𝑑|
|---|

|𝑰𝑰𝑑𝑑|
|---|

[Figure 29]

MHCAs

Add & Normalize

Add & Normalize

…

Feed Forward

Feed Forward

…

MHCAs

|𝒁𝒁1𝑝𝑝|
|---|

|𝒁𝒁𝑝𝑝2|
|---|

|𝒁𝒁𝑝𝑝S|
|---|

|𝐃𝐃𝑑𝑑|
|---|

ROHQD

|𝑰𝑰𝑑𝑑|
|---|

𝔻𝔻

|𝒁𝒁𝑝𝑝0|
|---|

Fusion Block

𝒁𝑓

𝒁𝑎

###### (a) MHSA (b) MHCA (c) RestoreFormer++

- Fig. 2. Framework of RestoreFormer++. (a) MHSA is a transformer with multi-head self-attention used in most of the previous ViTs [24], [25],

- [26], [27]. Its queries, keys, and values are from the degraded information Zd0. (b) MHCA is a transformer with a multi-head cross-attention used in the proposed RestoreFormer++. It globally fuses the degraded information Zd0 and the corresponding high-quality priors Zp0 by taking Zd0 as queries while Zp0 as key-value pairs. (c) The whole pipeline of RestoreFormer++. First, a degraded face image Id is sent to Ed for multi-scale feature extraction (Zds, s ∈ {0, 1, . . . , S − 1}, S is the number of scales used for fusion). Then, the degraded feature Zds interacts with its corresponding priors Zp0 matched from ROHQD D or previous fused output Zps with MHCAs. Finally, a high-quality face Iˆd is restored from the final fused result ZpS by the decoder Dd.

3 RESTOREFORMER++

In this section, we will introduce the proposed RestoreFormer++ with the whole restored pipeline shown in Fig. 2 (c). The pipeline consists of four components: an encoder Ed, a reconstruction-oriented high-quality dictionary D (ROHQD), a fusion block consisting of several MultiHead Cross-Attention blocks (MHCAs), and a decoder Dd. First, a degraded face image Id is sent to Ed for feature extraction (Zds,s ∈ {0,1,...,S − 1}, S is the number of scales used for fusing). Then, the degraded feature Zds fuses with its corresponding priors Zp0 matched from ROHQD D or previous fused output Zps with MHCAs. Finally, a highquality face Iˆd is restored from the final fused result ZpS by the decoder Dd.

We will introduce the details of the restoration process in Sec. 3.1 and describe the learning of the reconstructionoriented high-quality dictionary (ROHQD) in Sec. 3.2. Besides, we will explain our extending degraded model (EDM) used for synthesizing degraded face images in Sec. 3.3.

3.1 Restoration

RestoreFormer++ aims at globally modeling the contextual information in a face and the interplay with priors for restoring a high-quality face image with both realness and fidelity. ViT (Vision Transformer) [38] is such an effective method for modeling contextual information in computer vision. However, most of the previous ViT-based methods [24], [25], [26],

- [27] model the contextual information with multi-head selfattention (MHSA) whose queries, keys and values are from different patches in the same image. In this work, we propose to simultaneously model the contextual information and the interplay between the degraded face and its corresponding priors. Therefore, our RestoreFormer++ adopts multi-head cross-attention (MHCA) mechanisms whose queries are from the features of degraded faces, while key-value pairs are from the corresponding priors. To clarify the delicate design of our MHCA for blind face restoration, we will first describe MHCA by comparing it with MHSA before going deep into the restoration process.

MHSA. As Fig. 2 (a) shown, MHSA aims at searching the contextual information in one source (for convenience, we set it as our degraded feature Zd0 ∈ RH′×W′×C, where H′,W′ and C are the height, width and the number of channels of the feature map, respectively). Its queries Q, keys K, and values V can be formulated as:

Q = Zd0Wq + bq , K = Zd0Wk + bk , V = Zd0Wv + bv, (1)

where Wq/k/v ∈ RC×C and bq/k/v ∈ RC are learnable weights and bias.

Multi-head attention is a mechanism for attaining powerful representations. It is implemented by separating the Q, K, and V into Nh blocks along the channel dimension and gets {Q1,Q2,...,QN

h}, and {V1,V2,...,VN

h}, {K1,K2,...,KN

h}, where Qi/Ki/Vi ∈ RH′×W′×C

h,

Ch = NC

, and i ∈ [0,Nh − 1]. Then the attention map is represented as:

h

QiKi⊺ √Ch

)Vi,i = 0,1,...,Nh − 1. (2)

Zi = softmax(

By concatenating all Zi, we get the final output of multi-head attention:

Zi. (3)

Zmh = concat

i=0,...,Nh−1

In the conventional transformer, the attention output is added back to the input before sequentially processed by a normalization layer and a feed-forward network, which can be formulated as:

Za = FFN(LN(Zmh + Zd0)), (4) where LN is a layer normalization, FFN is a feed-forward network implemented with two convolution layers, and Za is the final output of MHSA.

MHCA. As shown in Fig. 2 (b), since the MHCA adopted in our Restoreformer++ aims at modeling the contextual information in the face images and simultaneously attaining identity information in the degraded face and high-quality facial details in the priors, it takes both the degraded feature Zd0 and the corresponding priors Zp0 as inputs. In MHCA,

[Figure 30]

the queries Q are from the degraded feature Zd0 while the keys K and values V are from the priors Zp0:

[Figure 31]

Q = Zd0Wq + bq , K = Zp0Wk + bk , V = Zp0Wv + bv, (5)

𝑰ℎ

VGG

Its following operations for attaining the multi-head attention output Zmh are the same as Eq. 2 and Eq. 3. Since high-quality priors play a more important role in blind face restoration, Zmh is added with Zp0 instead of Zd0 in RestoreFormer++. The rest operations are:

###### Eh Dh

[Figure 32]

[Figure 33]

…

𝒁ℎ 𝒁𝑝

K-means

Vector Quantization

Zf = MHCA(Zd0,Zp0) = FFN(LN(Zmh + Zp0)). (6) Restoration. As described before, the restored pipeline consists of four components. The first component Ed is used for extracting multi-scale features Zds (s = {0,1,...,S − 1}, S means the number of scales) from the degraded face image Id. Then, we can get the priors Zp0 of Zd0 from ROHQD D = {dm}Mm=0−1 (dm ∈ RC, M is the number of elements in D) with minimum Euclidean distance:

[Figure 34]

[Figure 35]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

…

…

𝔻

Component Dictionaries ROHQD

(a) Component Dictionaries (b) ROHQD

Fig. 3. Recognition-Oriented Dictionary v.s. Reconstruction-Oriented Dictionary. (a) Component Dictionaries, proposed in DFDNet [1], are recognition-oriented dictionaries since they are extracted with an offline image recognition model (VGG [49]). (b) ROHQD, proposed in this paper, is a reconstruction-oriented dictionary since it is learned with a high-quality face generation network incorporating the idea of vector quantization [28]. Priors from ROHQD contain more facial details specifically aimed at face restoration.

Zp0(i,j) = arg min

∥Zd0(i,j) − dm∥22, (7)

dm∈D

where (i,j) is the spatial position of map Zp0 and Zd0 and || · ||2 means the L2-norm. After attaining the degraded features Zds (s = {0,1,...,S − 1}) and Zp0, these two kinds of information are fused in the Fusion Block. In this block, for each scale, the degraded features and priors or previous fused results (for convenience, we denote the fused results of each scale as Zps (s = {1,...,S})) are fused with MHCAs, which consists of K MHCA. We formula this procedure as follows:

these key facial components independently. These losses are expressed as:

Ladv = [log D(Ih) + log(1 − D(Iˆd))], Lcomp =

[log Dr(Rr(Ih)) + log(1 − Dr(Rr(Iˆd)))], (11)

Zps+1 = MHCAs(Zds,Zps)

r

= MHCA(Zds,...,MHCA(Zds,MHCA(Zds,Zps))), s = {0,1,...,S − 1}.

where D and Dr are the discriminators for the whole face image and a certain region r (r ∈{left eye, right eye, mouth}), respectively. The region r is attained with Rr implemented with ROI align [50].

(8) Finally, ZpS is fed into the rest layers of the decoder Dd for recovering the high-quality face image Iˆd.

Identity learning. In this work, except extracting the identity information from the degraded face by fusing it with the high-quality priors, we also adopt an identity loss [19] to attain the identity supervision from the ground truth:

Learning. For attaining high-quality faces with both realness and fidelity, we design the objective functions from three aspects: content learning, realness learning, and identity learning.

##### Lid = ∥η(Ih) − η(Iˆd)∥22, (12)

Content learning. We adopt L1 loss and perceptual loss [47], [48] for ensuring the content consistence between the restored face image Iˆd and its ground truth Ih:

where η denotes the identity feature extracted from ArcFace [51] which is a well-trained face recognition model.

Therefore, the overall objective function is:

##### Ll1 = |Ih − Iˆd|1 ; Lper = ∥ϕ(Ih) − ϕ(Iˆd)∥22, (9)

LRF = Ll1 + λperLper + λpLp + λadvLadv

where Ih is the ground truth high-quality image, ϕ is the pretrained VGG-19 [49], and the feature maps are extracted from {conv1,...,conv5}. Besides, for improving the accuracy of the matched priors, we tend to guide the extracted features Zd0 to approach their selected priors Zp0 and the corresponding objective function is:

(13)

+ λcompLcomp + λidLid,

where λ... are the weighting factors for different losses.

###### 3.2 Reconstruction-Oriented High-Quality Dictionary

In this subsection, we introduce the learning of the Reconstruction-Oriented High-Quality Dictionary (ROHQD) D = {dm}Mm=0−1(dm ∈ RC, M is the number of elements) used in RestoreFormer++.

##### Lp = ∥Zp0 − Zd0∥22. (10)

Realness learning. We adopt adversarial losses for the learning of realness in this work. Since some crucial facial components, such as the eyes and mouth, play an important role in face presenting [19], our adversarial losses are not only applied to the whole face image but also applied to

Different from the facial component dictionaries [1] (Fig. 3 (a)) whose elements are high-quality facial details of specific facial components extracted with an off-line

|[Figure 36]|
|---|

|[Figure 37]| |
|---|---|
| | |

| |[Figure 38]|
|---|---|
| | |

|[Figure 39]| |
|---|---|
| | |

[Figure 40]

|[Figure 41]| |
|---|---|
| | |

|[Figure 42]|
|---|

|[Figure 43]| |
|---|---|
| | |

|[Figure 44]|
|---|

|𝑰𝒉(𝑰𝒂)|
|---|

|𝜶𝟏|
|---|

|𝒌𝝈𝟏|
|---|

|↓𝒓𝟏|
|---|

|𝒏𝜹𝟏|
|---|

|𝐽𝑃𝐸𝐺𝒒𝟏|
|---|

|↑𝒓𝟏|
|---|

|𝑴|
|---|

|[Figure 45]| | | |
|---|---|---|---|
| | | | |
| |𝑺𝒉𝒊𝒇𝒕| | |

|[Figure 46]| | | |
|---|---|---|---|
| | | | |
| |𝒌𝝈𝟐| | |

|[Figure 47]| |
|---|---|
| | |

|[Figure 48]| |
|---|---|
| | |

| |[Figure 49]|
|---|---|
| | |

|[Figure 50]| |
|---|---|
| | |

|[Figure 51]| |
|---|---|
| | |

|[Figure 52]| |
|---|---|
| | |

|𝑰𝒅|
|---|

|𝜶𝟐|
|---|

|↓𝒓𝟐|
|---|

|𝒏𝜹𝟐|
|---|

|𝐽𝑃𝐸𝐺𝒒𝟐|
|---|

|↑𝒓𝟐|
|---|

|𝟏 − 𝑴|
|---|

- Fig. 4. The whole pipeline of the extending degrading model (EDM). The degradations represented in GREEN are the operations contained in the traditional degrading model (Eq. 19) while the degradations in RED are the additional operations extended by EDM (Eq. 20 to 22). Specifically, a high-quality face image Ia is first shifted with an operator Shift. Then, it is sequentially degraded with blur, haze, downsampling, noise, and JPEG

compression. The degraded face images will be upsampled back to the size of the original image. The degraded faces attained after ↑r1 and ↑r2 are degraded from the same high-quality face image but with two different degraded parameters: α1 and α2, σ1 and σ2, r1 and r2, δ1 and δ2, and q1 and q2. They are independently and randomly sampled from their own uniform distributions. Combining these two degraded faces with a mask M, the final unevenly degraded face image Id is attained.

recognition-oriented model (VGG [49]), our ROHQD provides richer high-quality facial details specifically aimed at face reconstruction. We achieve this goal by deploying a high-quality face encoder-decoder network with the idea of vector quantization [28]. As shown in Fig. 3 (b), this encoder-decoder network takes a high-quality face image Ih ∈ RH×W×3 as input and encodes it to feature Zh ∈ RH′×W′×C with encoder Eh. Then, instead of decoding Zh directly back to the high-quality face with decoder Dh, it quantizes feature vectors in Zh with the index of the nearest vectors in D and attains Zp ∈ RH′×W′×C: Zp(i,j) = arg min

∥Zh(i,j) − dm∥22, (14)

dm∈D

where (i,j) is the spatial position of map Zp and Zh. Finally, a high-quality face image Iˆh is restored from Zp by the decoder Dh.

Learning. The whole pipeline shown in Fig. 3 (b) is essentially a high-quality face generation network. Therefore, we apply an L1 loss, a perceptual loss, and an adversarial loss to the final result Iˆh with the supervision from its high-quality input Ih:

##### L′l1 = ∥Ih − Iˆh∥1, L′per = ∥ϕ(Ih) − ϕ(Iˆh)∥22, L′adv = [log D(Ih) + log(1 − D(Iˆh))].

(15)

The definitions of ϕ and D are same as Eq. 9 and Eq. 11. It is worth noting that since Eq 14 is non-differentiable, the gradients back-propagated from Iˆh reach Zh by copying the gradients of Zp to Zh directly [28].

The ultimate goal of ROHQD in this work is to optimize D to attain high-quality facial details used for face restoration.

Therefore, we update the elements dm constructed Zp (Eq. 14) by forcing them to be close to their corresponding high-quality features Zh with L2 loss:

L′d = ∥sg[Zh] − Zp∥22, (16) where sg[·] denotes the stop-gradient operation. Besides, as described in [28], to avoid collapse, a commitment loss is needed to adjust the learning pace of the encoder Eh and dictionary D. The commitment loss is represented as:

L′c = ∥Zh − sg[Zp]∥22. (17)

Finally, the objective function for learning ROHQD is:

LROHQD = L′l1 + λperL′per + λadvL′adv + λdL′d + λcL′c,

(18) where λ... are the weighting factors.

###### 3.3 Extending Degrading Model

To diminish the distance between the synthetic training data and the real-world data and further improve the robustness and generalization of RestoreFormer++, EDM extends the degrading model [19], [21], [22] whose original expression is:

q} ↑r, (19)

Id = {[(Ih ⊗ kσ) ↓r +nδ]JPEG

where Ih is a high-quality face image and Id is the final synthetic degraded face image. Ih is first blurred by a Gaussian blur kernel kσ with sigma σ. Then, it is downsampled by r with bilinear interpolation and added with a white Gaussian noise nδ whose sigma is δ. Next, the intermediate degraded result is further compressed with JPEG compression, whose quality is q. After that, it is upsampled back to the size of Ih with scale r. Then we get the final synthetic degraded face image Id. These operations are sequentially described in Fig. 4 with GREEN color.

Excepting the common degradations described in Eq. 19, EDM adds haze and uneven degradation with a certain probability since they also obviously exist in the real-world degraded faces (examples are in Fig. 8). In addition, EDM also attempts to ease the error introduced by face alignment in real-world data (the third sample in Fig. 6 and the second sample in Fig. 8) by disturbing the perfect alignment in the synthetic training set with a spatial shift operation. The EDM is expressed as:

Ih = Shift(Ia,sh,sw), (20) Ide = {[(α(Ih⊗kσ)+(1−α)Ihaze) ↓r +nδ]JPEG

q} ↑r, (21) Id =M ⊙ Ide(α1,σ1,r1,δ1,q1)+

(22)

(1 − M) ⊙ Ide(α2,σ2,r2,δ2,q2).

Ia is the well aligned high-quality face image (Ia is equal to Ih in Eq. 19) and Shift(·) means spatially shifting Ia with sh

and sw pixels in height and width dimensions, respectively. Then the degraded face image is synthesized from the shifted

high-quality face image Ih. We synthesize haze in Eq. 21. Before downsampled, the blurry face image will be combined with Ihaze with ratio α : (1−α),α ∈ [0,1]. Ihaze is a globally white image. In Eq. 21, the degraded result Ide is a globally evenly degraded face image. To attain an unevenly degraded face image Id, we first synthesize two evenly degraded faces, Ide(α1,σ1,r1,δ1,q1) and Ide(α2,σ2,r2,δ2,q2), whose parameters: α1 and α2, σ1 and σ2, r1 and r2, δ1 and δ2, and q1 and q2, are independently and randomly sampled from uniform distributions (the experimental setting of the uniform distribution of each parameter in this paper is described in Subsec. 4.2). Then we combine these two unevenly degraded face images with a mask map M whose size is the same as Ide. The whole map of M is set to 0 except that a random L × L patch of it is set to 1 (L is smaller than both the height and width of Ide). ⊙ is an elementwise multiplication operation. The whole pipeline of EDM is described in Fig. 4, and the operations in RED are the additional degradations extended by EDM.

- 4 EXPERIMENTS AND ANALYSIS

- 4.1 Datasets

Training Datasets. ROHQD is trained on FFHQ [29], which contains 70000 high-quality face images resized to 512 × 512. RestoreFormer++ is also trained on synthesized data attained by applying EDM to the high-quality face images in FFHQ. Testing Datasets. We evaluate RestoreFormer++ on one synthetic dataset and three real-world datasets. The synthetic dataset, CelebA-Test [52], contains 3000 samples and is attained by applying EDM on the testing set of CelebAHQ [52]. The three real-world datasets include LFW-Test [53], CelebChild-Test [19], and WebPhoto-Test [19]. Specifically, LFW-Test contains 1711 images and is built with the first image of each identity in the validation set of LFW [53]. Both CelebChild-Test and WebPhoto-Test are collected from the Internet by Wang et al. [19]. They respectively own 180 and 407 degraded face images.

###### 4.2 Experimental Settings and Metrics

Settings. The encoder and decoder in the RestoreFormer++ and ROHQD are constructed with 12 residual blocks and 5 nearest downsampling/upsampling operations. Each MHCAs contains K = 3 MHCA. The input size of the model is 512×512×3. After encoding, the size of Zd is 16×16×256. ROHQD contains M = 1024 elements whose length is 256. As for EDM, sh, sw, α, σ, r, δ, q, and L are randomly sampled from{0 : 32}, {0 : 32}, {0.7 : 1.0}, {0.2 : 10}, {1 : 8}, {0 : 20}, {60 : 100}, and {128 : 256}, respectively. While training, the batch size is set to 16 and the weighting factors of the loss function are λper = 1.0, λp = 0.25, λadv = 0.8, λcomp = 1.0, λid = 1.0, λd = 1.0, and λc = 0.25. Both RestoreFormer++ and ROHQD are optimized by Adam [54] with learning rate 0.0001. Noted that we do not update the elements of the ROHQD while training RestoreFormer++.

Metrics. In this paper, we evaluate the state-of-the-art methods and our RestoreFormer++ objectively and subjectively. From the objective aspect, we adopt the widely-used nonreference metric FID [55] to evaluate the realness of the restored face images and introduce an identity distance

(denoted as IDD) to judge the fidelity of the restored face images. IDD is the angular distance between the features of the restored face image and its corresponding ground truth. Features are extracted with a well-trained face recognition model ArcFace [51]. Besides, we adopt PSNR, SSIM, and LPIPS [56] to build a more comprehensive comparison. From the subjective aspect, we deploy a user study to evaluate the quality of the restored results from the perspective of humans.

###### 4.3 Comparison with State-of-the-art Methods

In this subsection, we compare our RestoreFormer++ with state-of-the-art prior-based methods, including DFDNet [1] based on component dictionaries, PSFRGAN [4] implemented with facial parsing maps, Wan et al. [2], PULSE [3], GPEN [5], and GFP-GAN [19] restored with generative priors, and VQFR [6] utilized codebook. We also compare RestoreFormer++ with our conference version, RestoreFormer. Compared to RestoreFormer++, RestoreFormer is trained with synthetic data attained with the traditional degrading model rather than EDM and its fusion between the degraded face and priors only involves one scale. Comparisons between these methods and our proposed method are conducted on synthetic and real-world datasets.

4.3.1 Performance on Synthetic Dataset

The quantitative results of the aforementioned state-of-theart methods and our RestoreFormer++ on the synthetic dataset CelebA-Test [52] are in TABLE 1. We can see that RestoreFormer++ performs better than other methods on FID and IDD, which means that the restored faces of RestoreFormer++ are more real and their identities are closer to the degraded faces. Our RestoreFormer++ also achieves comparable performance in terms of PSNR, SSIM, and LIPIS, which are pixel-wise and perceptual metrics. These metrics have been proved not that consistent with the subjective judgment of human beings [48], [58]. We also find that the visualized results of GPEN [5] which performs better on PSNR, SSIM, and LIPIS are over-smooth and lack details. Visualized results are shown in Fig. 5. Compared to other methods, the restored results of our RestoreFormer++ have a more natural look and contain more details, especially in the eyes, mouth, and glasses. Besides, our method can restore a more complete face, such as the left eye in the first sample and the glasses in the second sample. Due to severe degradations, most existing methods fail to restore the left eye and glasses, although they can properly restore the right eye and part of the glasses. On the contrary, since our RestoreFormer++ can model the contextual information in the face, its restored left eye and glasses are more natural and complete by utilizing the related information in the right eye area and the clear part of the glasses. The quantitative results in TABLE 1 show that RestoreFormer++ attains an obvious improvement compared to the conference version, RestoreFormer, due to the participation of EDM and multiscale mechanism. More detailed analyses of the contributions of these components are discussed in Subsec. 4.4.3 and Subsec. 4.4.5, and more visualized results are in the supplementary materials.

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Input DFDNet [1] PSFRGAN [4] GFP-GAN [19] GPEN [5] VQFR [6] Ours GT

- Fig. 5. Qualitative comparison on the CelebA-Test [52]. The results of our RestoreFormer++ have a more natural and complete overview and contain more details in the areas of eyes, mouth, and glasses. Note that DFDNet [1] relies on dlib [57] for facial detection while matching priors from its facial component dictionaries, and failure in detection results in no restoration, as seen in the second result. Zoom in for a better view.

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

Input DFDNet [1] Wan et al. [2] PSFRGAN [4] GFP-GAN [19] GPEN [5] VQFR [6] Ours

- Fig. 6. Qualitative comparison on three real-world datasets: LFW-Test [53], CelebChild-Test [19], and WebPhoto-Test [19] (from top to down, respectively). The results of our RestoreFormer++ have a more natural and complete overview and contain more details in the areas of eyes, glasses, hair, and mouth. The green points on the third degraded face image are the reference landmarks used for face alignment. In this sample, its mouth is aligned to the landmark of the nose, and the existing methods, e.g., [2], [4], restore the mouth with a nose-like shape. Benefiting from the spatial shift adopted in EDM, our restored result looks more natural. Zoom in for a better view.

- 4.3.2 Performance on Real-world Datasets

The quantitative and qualitative results of our Restoreformer++ and the compared methods on three real-world datasets are in TABLE 2 and Fig. 6, respectively. According to TABLE 2, RestoreFormer++ performs superiorly on FID compared to other methods. The qualitative results

- in Fig. 6 also reveal that although most of the current methods can attain clear faces from the corrupted face images with slight degradations (the first two samples), RestoreFormer++ attains more details on the crucial areas, such as the eyes with glasses, hair, and ear. That mainly benefits from the contextual information in the face and our learned reconstruction-oriented high-quality dictionary.

Besides, since our RestoreFormer++ is further enhanced with EDM, it can remove the haze covered on the face image and avoid restoration artifacts caused by misalignment, thus attaining more natural and pleasant results. For example, after face alignment, the mouth of the last sample in Fig. 6 is aligned to the reference landmark of the nose, which leads to the restored mouth of Wan et al. [2] and PSFRGAN [4] is nose-like. Although the restored results of other existing methods look better, they still look weird. With EDM, the restored result of RestoreFormer++ looks more natural.

In addition, as shown in TABLE 2, in the real-world datasets, the performance of RestoreFormer++ is better or comparable to our conference version, RestoreFormer.

TABLE 1 Quantitative comparisons on CelebA-Test [52]. Our RestoreFormer++ performs better in terms of FID and IDD, which indicates the realness and fidelity of the restored results of our method. It also gets comparable results on PSNR, SSIM, and LPIPS.

TABLE 3 User study results on WebPhoto-Test [19]. For “a/b”, a is the percentage where our RestoreFormer or RestoreFomer++ is better than the compared method, and b is the percentage where the compared method is considered better than our RestoreFormer or RestoreFomer++.

Methods FID↓ PSNR↑ SSIM↑ LPIPS↓ IDD↓

|Input<br><br>|132.05|24.91<br><br>|0.6637|0.4986<br><br>|0.9306|
|---|---|---|---|---|---|
|DFDNet [1] Wan et al. [2] PSFRGAN [4] PULSE [3] GPEN [5] GFP-GAN [19] VQFR [6]|50.88 67.13 40.69 84.03 48.97 40.87 38.51<br><br>|24.09 23.01 24.30 20.73 25.44 24.39 23.82<br><br>|0.6107 0.6174 0.6273 0.6151 0.6965 0.6671 0.6379<br><br>|0.4516 0.4789 0.4220 0.4745 0.3562 0.3575 0.3544|0.7700 0.8058 0.7284 1.2267 0.6434 0.6127 0.6354<br><br>|
|RestoreFormer RestoreFormer++<br><br>|39.90 38.41|24.19 24.40<br><br>|0.6232 0.6339|0.3716 0.3619<br><br>|0.5677 0.5375|
|GT<br><br>|41.66<br><br>|∞|1<br><br>|0<br><br>|0|

TABLE 2 Quantitative comparisons on three real-world dataset in terms of FID. RestoreFormer++ performs better.

Methods LFW-Test CelebChild-Test WebPhoto-Test

|Input<br><br>|126.12|144.36<br><br>|170.46|
|---|---|---|---|
|DFDNet [1] PSFRGAN [4] Wan et al. [2] PULSE [3] GPEN [5] GFP-GAN [19] VQFR [6]|72.87 53.17 71.24 66.08 55.52 50.30 50.22<br><br>|110.85 105.65 115.15 104.06 107.57 111.78 103.96<br><br>|100.45 83.50 99.91 86.39 86.07 87.82 74.22|
|RestoreFormer RestoreFormer++<br><br>|48.11 48.48|104.01 102.66<br><br>|75.49 74.21|

RestoreFormer is slightly superior to RestoreFormer++ on LFW-Test [53] since the degree of the degradation in this dataset is generally slight, and the delicate design in RestoreFormer is enough for attaining high-quality restored results. However, since the degradation in CelebChild-Test [19] and WebPhoto-Test [19] are more severe, RestoreFormer++, with additional EDM and multi-scale mechanism, can handle these two datasets better compared to RestoreFormer. More visualizations are in the supplementary materials.

Besides, a user study is adopted to collect the subjective judgment of human beings on the real-world dataset WebPhto-Test [19]. Specifically, we randomly select 100 samples from the real-world dataset and conduct pair comparisons between our conference version RestoreFormer and three other methods: DFDNet [1], PSFRGAN [4], and GFP-GAN [19]. Subjective comparisons between RestoreFormer++, RestoreFormer, and VQFR [6] are also conducted. We invite 100 volunteers to make their subjective selection on these pair comparisons. The statistic results are in Tab 3. It shows that a high percentage of volunteers vote for the results of our RestoreFormer and RestoreFormer++ as the more natural and pleasant restored results compared to other methods, and the restored results of RestoreFormer++ are better than those of RestoreFormer.

|Methods|DFDNet [1]<br><br>|PSFRGAN [4]|GFP-GAN [19]|
|---|---|---|---|
|RestoreFormer|89.60%/10.40%<br><br>|68.81%/31.19%|79.21%/20.79%|

|Methods<br><br>|VQFR [6]|RestoreFormer<br><br>|
|---|---|---|
|RestoreFormer++<br><br>|67.82%/32.18%|66.91%/33.19%|

###### 4.4 Ablation Study

Our proposed RestoreFormer++ consists of several components, including MHCA, EDM, ROHQD, multi-scale mechanism, and several losses. It also contains two kinds of inputs: the degraded face and high-quality priors. Each component plays an important role in the whole restoration pipeline. The followings are the detailed analyses of the effectiveness of these components. A discussion about the efficiency of our proposed method is also included.

- 4.4.1 Analysis of Spatial Attention Mechanism

In RestoreFormer++, global spatial attention mechanism is used to model the rich facial contextual information in the face image and its interplay with priors for aiding the face restoration. To validate the effectiveness of the spatial attention mechanism, we compare our single-scale RestoreFormer++ with and without attention mechanisms. As shown in TABLE 4, both exp1 and exp2 only get information from the degraded face image. By adopting self-attention (MHSA) to model contextual information, exp2 performs better than exp1 which is without MHSA in terms of FID and IDD. This conclusion is also valid when comparing exp4 to exp6, whose inputs include both degraded information and additional high-quality priors. In exp4, we replace MHCA in RestoreFormer++ with SFT [23] for locally fusing these two kinds of information. Since it ignores the facial contextual information in the face image, its result in Fig. 7 (d) fails to restore natural eyes. Exp6 is a version of RestoreFormer++ implemented with a single-scale fusion mechanism. It uses MHCA for globally fusing degraded information and priors. (1)-(4) in Fig. 7 are its multi-head (4 heads) attention maps of the left eye region in scale 16 × 16. It shows that the highlighted areas not only occur in the left eye area but also in other regions of the face image, especially the more related right eye region. It means that apart from the information in the left areas, our RestoreFormer++ with MHCA can also utilize the related information in other areas to restore the left eye with more natural appearance (Fig. 7 (e)).

- 4.4.2 Analysis of Degraded Information and Priors.

In this subsection, we analyze the roles of the degraded information extracted from the degraded face image and its corresponding high-quality priors matched from ROHQD. In exp2 and exp3 (TABLE 4), we replace the MHCA in our single-scale RestoreFormer++ with MHSA, whose queries, keys, and values are all from either the degraded information or the high-quality priors. We can see that exp2 attains a better average IDD score which means it performs better in fidelity. In contrast, exp3 has a better FID score, meaning

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

(a) Input (b) exp2 (c) exp3 (d) exp4 (e) exp6 (f) exp7 (g) RestoreFormer++ (h) GT PSNR: 26.21 PSNR: 25.29 PSNR: 23.83 PSNR: 23.98 PSNR: 25.40 PSNR: 25.97 PSNR: 26.56 PSNR: ∞ IDD: 1.0689 IDD: 0.5322 IDD: 0.7981 IDD: 0.6239 IDD: 0.5220 IDD: 0.4560 IDD: 0.4689 IDD: 0

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

(1) s16h0 (2) s16h1 (3) s32h0 (4) s32h1 (5) s64h0 (6) s64h1

- Fig. 7. The qualitative visualizations from (b) to (g) are results of the experiments whose settings correspond to those in TABLE 4. The result of (e) exp6, which takes the degraded face and priors as inputs, is better than the results of (b) exp2 and (c) exp3 in both realness and fidelity since exp2 and exp3 only take one of these two kinds of resources as input. By globally fusing the features of degraded faces and priors with MHCAs, (e) exp6 also performs better than (d) exp4 implemented with the local fusion algorithm SFT [19]. In (g), RestoreFormer++ with a two-scale setting can avoid the weird eye shape restored in (e) exp6 implemented with a single-scale setting, but while extending to a three-scale setting, the result of (f) exp7 cannot see an obvious improvement compared to (g). Images from (1) to (6) are the heatmaps of the left eye area attained on different scales. ‘sxhy’ means the y-th (y ∈ {0, 1}) head attention map in x × x (x ∈ {16, 32, 64}) resolution. In the low resolution, heatmaps (1)-(2) mainly focus on the most related eye areas while in the middle resolution, heatmaps (3)-(4) expand to salient edges that help the restoration of the shape of the left eye. In high resolution, heatmaps (5)-(6) focus on more detailed edges. They yield less further improvement on the final restoration, and thus our RestoreFormer++ adopts a two-scale setting.

TABLE 4 Quantitative results of ablation studies on CelebA-Test [52]. ‘degraded’ and ‘prior’ mean fusion information from degraded input and ROHQD, respectively. ‘none’ and ‘MHSA’ respectively mean the network uses either ‘degraded’ or ‘prior’ information without or with a self-attention mechanism. ‘SFT’, ‘MHCA-D’ and ‘MHCA-P’ use both ‘degraded’ and ‘prior’ information. ‘SFT’ uses SFT [19] to fuse the information, while ‘MHCA-D’ and

‘MHCA-P’ use multi-head cross attention. The difference between ‘MHCA-D’ and ‘MHCA-P’ is that ‘MHCA-D’ fuses Zmh with Zds while ‘MHCA-P’

fuses Zmh with Zps. ‘S’ is the number of feature scales used for fusion. S = 1 means the fusion only exists in 16 × 16 resolution while S = 2 means the fusion are involved in both 16 × 16 and 32 × 32 resolutions. S = 3 means it is further extended to 64 × 64 resolution. The proposed RestoreFormer++ integrated with ‘MHCA-P’ and set with more than one scale performs the best relative to other variants.

sources methods metrics

|No. of exp.|degraded<br><br>|prior|none|MHSA<br><br>|SFT|MHCA-D<br><br>|MHCA-P<br><br>|S|FID↓<br><br>|IDD↓|
|---|---|---|---|---|---|---|---|---|---|---|
|exp1|✓| |✓| | | | |1<br><br>|48.33<br><br>|0.6520|
|exp2|✓| | |✓| | | |1<br><br>|47.96|0.6461|
|exp3| |✓| |✓| | | |1|42.53<br><br>|0.7467|
|exp4|✓<br><br>|✓| | |✓| | |1|44.67<br><br>|0.6373|
|exp5<br><br>|✓<br><br>|✓| | | |✓| |1<br><br>|42.25<br><br>|0.6038|
|exp6<br><br>|✓|✓| | | | |✓<br><br>|1|39.31|0.5677|
|exp7|✓<br><br>|✓| | | | |✓<br><br>|3|39.11<br><br>|0.5355|
|RestoreFormer++|✓<br><br>|✓| | | | |✓<br><br>|2<br><br>|38.41<br><br>|0.5375|

its results contain more realness. By globally fusing the degraded information and priors with MHCA in our singlescale RestoreFormer++ (exp6 in TABLE 4), it performs better than exp2 and exp3 in both IDD and FID, which means that our RestoreFormer++ can restore faces with both realness and fidelity. The visualized results in Fig. 7 show that the result of exp2 (Fig. 7 (b)) is more similar to GT but contains fewer details compared to (c) and (e), which are the results of exp3 and exp6, respectively. Although the details in (c) are richer, it looks less similar to the GT, especially in the eyes. On the contrary, Our result shown in (e) is similar to GT and meanwhile contains rich details, and thus presents pleasantly. Besides, according to Fig. 2 (b) and Eq. 6, we tend to add the attended feature Zmh to Zp0 rather than Zd0 (corresponding to exp5 in TABLE 4), since we experimentally find that it can

attain better performance.

4.4.3 Analysis of Multi-scale Mechanism

Our multi-scale mechanism aims to facilitate RestoreFormer++ by modeling contextual information based on both semantic and structural information, thereby improving the restoration performance in both realness and fidelity. First, we apply MHCAs to fuse the degraded features and priors at a resolution of 16 × 16, which is the smallest resolution in our model (this setting corresponds to exp6 in TABLE 4). The features of a face at this scale are semantic information of facial components, such as eyes, mouth, nose, etc. The highlighted areas in the attention maps of the left eye in Fig. 7 (1)-(2) are eyes areas, which reveal that the restoration of the left eye in Fig. 7 (e) is achieved by leveraging contextual

TABLE 5 Quantitative results of methods with or without EDM measured on FID↓. Methods with EDM perform better than those without EDM on CelebChild-Test [19] and WebPhoto-Test [19] datasets whose degradations are more diverse and severe and perform comparably on LFW-Test [53] dataset with more common degradations. RestoreFormer++ is better than the other methods in both settings.

Methods LFW-Test [53] CelebChild-Test [19] WebPhoto-Test [19]

|PSFRGAN [60]<br><br>|53.17<br><br>|105.65<br><br>|83.50|
|---|---|---|---|
|PSFRGAN w/ EDM|53.20<br><br>|104.22|82.28|
|GFP-GAN [19]|50.30|111.78|87.82<br><br>|
|GFP-GAN w/ EDM<br><br>|50.72<br><br>|109.08<br><br>|86.17|
|Ours w/o EDM<br><br>|48.10<br><br>|103.86|75.42|
|Ours|48.48<br><br>|102.66<br><br>|74.21|

information from its semantic-related areas. Compared with the results in (d) attained with SFT [19], a spatial-based fusion approach, the restored left eye of (e) is more complete and real. However, its edge shape is not smooth enough, leading to a weird look. Therefore, we extend MHCAs to features with a larger scale, 32×32 (corresponding to Restoreformer++ in TABLE 4), and attain a restored result with a more natural look as shown in Fig. 7 (g). Its corresponding attention maps

- in Fig. 7 (3)-(4) show that apart from related eye areas, its highlighted areas diffuse to some salient edges that help reconstruct the smooth and natural shape of the left eye . FID and IDD scores on CelebA-Test [52] in TABLE 4 indicate that increasing the number of scales from one to two can improve restoration performance in both realness and fidelity. To make further exploration, we extend MHCAs to the features at a resolution of 64×64 (corresponding to exp7 in TABLE 4). Its attention maps (Fig. 7 (5)-(6)) focus on more detailed structures such as hairs. However, its restored result in

- Fig. 7 (f) does not show an obvious improvement compared to (g) attained with a two-scale setting. Its quantitative results in TABLE 4 show that it attains a better IDD score but worse FID score than RestoreFormer++ implemented with a twoscale setting. Comprehensively considering efficiency, where the running time of the three-scale setting increases by about 17% compared to the two-scale setting (TABLE 7), we adopt a two-scale setting in RestoreFormer++.

- 4.4.4 Analysis of ROHQD. Comparisons between our RestoreFormer++ and DFDNet [1], whose priors are recognition-oriented, have validated the effectiveness of ROHQD. To further evaluate the contribution of ROHQD in RestoreFormer++, we replace ROHQD with a recognition-oriented dictionary with the same learning process as ROHQD. We implement it by replacing the encoders Ed and Eh with a VGG [49]. Similar to [1], we initialize these encoders with weights attained with ImageNet [59] and freeze them while training. We conduct experiments on CelebA-Test [52]. Its scores in terms of FID and IDD are 50.39 and 0.7572, which is worse than RestoreFormer++ implemented with ROHQD. It indicates that the facial details in ROHQD that are accordant to reconstruction tasks are helpful for face restoration.
- 4.4.5 Analysis of EDM. In this subsection, we analyze the effect of EDM on blind face restoration. To make a more comprehensive comparison,

except conducting experiments on our RestoreFormer++ with and without EDM, we also retrained PSFRGAN [60] and GFP-GAN [19] with EDM. The quantitative results in TABLE 5 show that methods with EDM perform better or comparable compared to their counterpart without EDM on three real-world datasets, especially on CelebChildTest [19] and WebPhoto-Test [19], whose degradations are more diverse and severe. Since the degradations of LFWTest [53] are relatively slight and regular, EDM has little effect on this dataset. Besides, qualitative results show the notable effectiveness of EDM in real-world face image restoration, particularly towards face images with uneven degradation, haze, and aligned bias. In the first sample in Fig. 8, the degradation in the red box is uneven (degradation in the center is more severe than the neighbouring regions), leading to a weird look of the restored right eye (blue area). Compared to methods without EDM, their counterpart with EDM attains a clearer and more natural right eye. The second sample in Fig. 8 is covered with haze, and its left eyebrow is aligned to the left eye of the reference face. The restored results of methods without EDM are unclear, and their left eyes are not in the right position. After fine-tuning with EDM, their results become clear, and their left eyes are restored to the right position with a more natural look. More restored results of our proposed method with or without EDM are shown in Fig. 9.

Discussion about color changes. Blind face restoration aims to remove the degradations in a face image and recover its high-quality facial structures. Its colors will be restored if the degraded face image contains colors. For example, pink color can be observed on the cheek and forehead of the first sample in Fig. 1. Methods including Wan et al. [2], PULSE [3], GPEN [5], VQFR [6], and our RestoreFormer [7] detect and recover this color. Since our priors matched from ROHQD are more accordant to blind face restoration and contextual information is considered while fusing the degraded face information and high-quality priors, the color restored with our RestoreFormer looks more harmonized and natural. Besides, since haze is common in real-world degraded face images, our EDM is proposed to endow RestoreFormer++ with the capability of haze removal. As shown in Fig. 10, the capacity of haze removal of RestoreFormer++ aims to recover the original colors of the degraded face image. The colors of the restored results are close to their Ground Truth. The restored result of a gray degraded image is almost gray. With the help of EDM, the restored results of RestorFormer++ contain more facial details and look clearer.

TABLE 6 Quantitative results on CelebA-Test [53] of different loss settings.

|Ll1<br><br>|Lper|Lp<br><br>|Ladv|Lcomp<br><br>|Lid<br><br>|FID↓|IDD↓|
|---|---|---|---|---|---|---|---|
|✓|✓<br><br>|✓|✓|✓| |45.14<br><br>|0.6052|
|✓<br><br>|✓<br><br>|✓| | |✓|54.14<br><br>|0.5378|
|✓<br><br>|✓|✓|✓| |✓<br><br>|48.19|0.5384|
|✓|✓| |✓|✓<br><br>|✓|38.69<br><br>|0.5873|
|✓| |✓|✓<br><br>|✓<br><br>|✓<br><br>|48.66|0.5619|
|✓<br><br>|✓|✓|✓<br><br>|✓|✓<br><br>|38.41|0.5375|

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

(a) Input (b) PSFGAN [60] (c) PSFGAN w/ EDM (d) GFP-GAN [19] (e) GFP-GAN w/ EDM (f) Ours w/o EDM (g) Restoreformer++

- Fig. 8. Qualitative results of methods with or without EDM. Methods with EDM can rectify the distortion (area in the blue box) introduced by uneven degradation (area in the red box in the first sample) or misalignment (areas in the red box in the second sample) and remove the haze covered on the face image (the second sample). In the input of the second example, the green points are the reference landmarks defined in the training dataset. The reference landmark of the left eye is aligned to the left eyebrow of this sample, leading to the unnatural restored results of the methods without EDM. However, the methods with EDM can alleviate this issue.

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

[Figure 132]

Input Ours w/o EDM RestoreFormer++

- Fig. 9. More qualitative results of our methods with or without EDM. With the design of EDM, RestoreFormer++ can remove the haze covered on the faces, mitigate the artifacts raised by uneven degradation, alleviate the influence of bias introduced by face misalignment, and restore faces with a more natural appearance.

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Input RestoreFormer++ GT

Fig. 10. Restored results of RestoreFormer++ on two synthetic degraded face images from CelebA-Test [53]. The first sample shows that RestoreFormer++ aims to recover the original colors of the degraded face image. It does not change the colors of the restored results compared to the GT. The second sample indicates that our RestoreFormer++ tends to keep the gray degraded face image gray after restoration.

removal loss (Lp) for reducing the prior searching error, adversarial loss on the whole image (Ladv) and key components (Lcomp) for realness learning, and identity loss (Lid) for identity learning. To analyze the effect of each loss function, we remove one or two of them from the original RestoreFormer++. Since the training process without Ll1 tends to collapse, we keep it in all the experiments. As shown in TABLE 6, no matter removing which loss function, both scores in terms of FID and IDD increase, which means the realness and fidelity of their results become worse. Specifically, Ladv and Lcomp affect the realness of the final restored results more compared to their fidelity. This phenomenon conforms to the characteristics of adversarial loss which aims at generating real images. On the contrary, Lid and Lp affect the fidelity of the restored faces more than their realness. Lid specifically aim at identity reserving. Lp can reduce the bias introduced by noisy information by

- 4.4.6 Analysis of Losses. During the training process, RestoreFormer++ involves several loss functions, including L1 loss (Ll1) and perceptual loss ( Lper) for content consistence learning, degradation

removing degradations in the encoding time. Lper has similar effects on the realness and fidelity of the restored results for the reason that it can keep the semantic information while generating high-quality images [47].

TABLE 7 Analysis of efficiency in terms of running time, flops, and model size.

Methods Time/s Flops/G Size/M S

|DFDNet [1]|2.218|602.732|240.117|-|
|---|---|---|---|---|
|PSFRGAN [4]|0.2076|337.870|67.026|-|
|GPEN [5]|0.1193|168.279|71.005|-|
|GFP-GAN [19]|0.0083|54.734|76.564|-|
|VQFR [6]|0.7309|1071.492|71.829|-|
|RestoreFormer++|0.2229|343.071|72.680|1|
|RestoreFormer++|0.2260|345.490|73.473|2|
|RestoreFormer++|0.2643|374.493|74.265|3|

###### 4.5 Analysis of Efficiency

To analyze the efficiency, we test the running time, Flops, and model size of several current methods and our RestoreFormer++ with different scale settings on a GeForce GTX 1060. Comprehensively considering the efficiency in TABLE 7 and the effectiveness in TABLE 1 and TABLE 2, we can see that RestoreFormer++ attains high performance with more modest resource consumption. Although GPEN [5] and GFP-GAN [19] can run faster, their performance is slightly inferior to the performance of RestoreFormer++. Besides, although the performance of VQFR [6] approaches to our RestoreFormer++ and their codebook shares the same idea as our ROHQD, it is time-consuming and needs more computations because its codebook is built at 32 × 32 resolution and the fusions between the degraded feature and priors involve resolutions from 32 × 32 to 512 × 512, On the contrary, our RestoreFormer++ can attain better performance with fusions only in several smaller scales (16 × 16 and 32 × 32). Since the performance of the threescale setting is comparable to two scales setting but has a large increase in resource consumption, we adopt the two-scale setting in our RestoreFormer++. It also validates that by modeling the contextual information in the face with MHCAs, feature interactions in smaller resolutions are enough for RestoreFormer++ to attain high-quality face images with both realness and fidelity.

###### 4.6 Limitations

RestoreFormer++ cannot handle faces with obstacles or large poses well, which are also two issues for other methods. As shown in Fig. 11, the face in the first sample is covered with a tennis racket. PSFRGAN [4] and GFP-GAN [19] tend to remove the obstacle. Although our method can keep most of the tennis racket, it leads to artifacts. In the second sample, all the approaches cannot restore complete glasses due to the large facial pose. These limitations mainly result from the bias in the training data – most of the high-quality face images in FFHQ [29] are near-frontal and without obstacles. In the future, we will try to make an effort to mitigate these

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

Input PSFRGAN [4] GFP-GAN [19] RestoreFormer++

Fig. 11. RestoreFormer++ cannot handle faces with obstacles or large poses well. Other methods have similar issues.

limitations from two aspects: 1) extending the diversity of FFHQ [29] with more facial poses and obstacles. 2) explicitly modeling the information of facial poses and obstacles, and then merging it into the face restoration model.

#### 5 CONCLUSION

In this work, we propose a RestoreFormer++, which on the one hand introduces multi-head cross-attention mechanisms to model the fully-spatial interaction between the degraded face and its corresponding high-quality priors, and on the other hand, explores an extending degrading model to synthetic more realistic degraded face images for training. The goal of RestoreFormer++ is to restore face images with higher realness and fidelity for both synthetic and real-world scenarios. Specifically, the multi-head cross-attention mechanisms take the features of degraded faces as queries and the corresponding high-quality priors as the key-value pairs. It models the contextual information based on both semantic and structural information in features at different resolutions. Besides, the priors provided by our reconstruction-oriented high-quality dictionary are learned from plenty of highquality face images with a face generation network cooperating with vector quantization. They are more accordant to the face restoration task, enabling RestoreFormer++ to attain faces with better realness. What is more, the extending degrading model contains more realistic degradations and considers face misalignment situations, allowing our RestoreFormer++ to further alleviate the synthetic-to-real-world gap and improve its robustness and generalization towards realworld scenarios. Finally, we conduct extensive experiments on both synthetic and real-world datasets to demonstrate the superiority of the proposed method. We also discuss the limitations of RestoreFormer++ and our future works for further improvement.

#### ACKNOWLEDGMENTS

This paper is partially supported by the National Key R&D Program of China No.2022ZD0161000 and the General Research Fund of Hong Kong No.17200622.

#### REFERENCES

[1] X. Li, C. Chen, S. Zhou, X. Lin, W. Zuo, and L. Zhang, “Blind face restoration via deep multi-scale component dictionaries,” in ECCV, 2020.

- [2] Z. Wan, B. Zhang, D. Chen, P. Zhang, D. Chen, J. Liao, and F. Wen, “Bringing old photos back to life,” in CVPR, 2020.
- [3] S. Menon, A. Damian, S. Hu, N. Ravi, and C. Rudin, “Pulse: Self-supervised photo upsampling via latent space exploration of generative models,” in CVPR, 2020.
- [4] C. Chen, X. Li, L. Yang, X. Lin, L. Zhang, and K.-Y. K. Wong, “Progressive semantic-aware style transformation for blind face restoration,” in CVPR, 2021.
- [5] T. Yang, P. Ren, X. Xie, and L. Zhang, “Gan prior embedded network for blind face restoration in the wild,” in CVPR, 2021.
- [6] Y. Gu, X. Wang, L. Xie, C. Dong, G. Li, Y. Shan, and M.-M. Cheng, “Vqfr: Blind face restoration with vector-quantized dictionary and parallel decoder,” arXiv preprint arXiv:2205.06803, 2022.
- [7] Z. Wang, J. Zhang, R. Chen, W. Wang, and P. Luo, “Restoreformer: High-quality blind face restoration from undegraded key-value pairs,” in CVPR, 2022.
- [8] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” NIPS, 2012.
- [9] Q. Cao, L. Lin, Y. Shi, X. Liang, and G. Li, “Attention-aware face hallucination via deep reinforcement learning,” in CVPR, 2017.
- [10] H. Huang, R. He, Z. Sun, and T. Tan, “Wavelet-srnet: A waveletbased cnn for multi-scale face super resolution,” in ICCV, 2017.
- [11] Y. Shi, G. Li, Q. Cao, K. Wang, and L. Lin, “Face hallucination by attentive sequence optimization with reinforcement learning,” IEEE transactions on pattern analysis and machine intelligence, vol. 42, no. 11, pp. 2809–2824, 2019.
- [12] X. Xu, D. Sun, J. Pan, Y. Zhang, H. Pfister, and M.-H. Yang, “Learning to super-resolve blurry face and text images,” in ICCV, 2017.
- [13] Y. Chen, Y. Tai, X. Liu, C. Shen, and J. Yang, “Fsrnet: End-to-end learning face super-resolution with facial priors,” in CVPR, 2018.
- [14] D. Kim, M. Kim, G. Kwon, and D.-S. Kim, “Progressive face super-resolution via attention to facial landmark,” arXiv preprint arXiv:1908.08239, 2019.
- [15] Z. Shen, W.-S. Lai, T. Xu, J. Kautz, and M.-H. Yang, “Deep semantic face deblurring,” in CVPR, 2018.
- [16] X. Yu, B. Fernando, B. Ghanem, F. Porikli, and R. Hartley, “Face super-resolution guided by facial component heatmaps,” in ECCV, 2018.
- [17] X. Yu, B. Fernando, R. Hartley, and F. Porikli, “Super-resolving very low-resolution face images with supplementary attributes,” in CVPR, 2018.
- [18] S. Zhu, S. Liu, C. C. Loy, and X. Tang, “Deep cascaded bi-network for face hallucination,” in ECCV, 2016.
- [19] X. Wang, Y. Li, H. Zhang, and Y. Shan, “Towards real-world blind face restoration with generative facial prior,” in CVPR, 2021.
- [20] B. Dogan, S. Gu, and R. Timofte, “Exemplar guided face image super-resolution without facial landmarks,” in CVPRW, 2019.
- [21] X. Li, W. Li, D. Ren, H. Zhang, M. Wang, and W. Zuo, “Enhanced blind face restoration with multi-exemplar images and adaptive spatial feature fusion,” in CVPR, 2020.
- [22] X. Li, M. Liu, Y. Ye, W. Zuo, L. Lin, and R. Yang, “Learning warped guidance for blind face restoration,” in ECCV, 2018.
- [23] X. Wang, K. Yu, C. Dong, and C. C. Loy, “Recovering realistic texture in image super-resolution by deep spatial feature transform,” in CVPR, 2018.
- [24] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in ECCV, 2020.
- [25] H. Chen, Y. Wang, T. Guo, C. Xu, Y. Deng, Z. Liu, S. Ma, C. Xu, C. Xu, and W. Gao, “Pre-trained image processing transformer,” in CVPR, 2021.
- [26] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.
- [27] X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai, “Deformable detr: Deformable transformers for end-to-end object detection,” arXiv preprint arXiv:2010.04159, 2020.
- [28] A. v. d. Oord, O. Vinyals, and K. Kavukcuoglu, “Neural discrete representation learning,” arXiv preprint arXiv:1711.00937, 2017.
- [29] T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for generative adversarial networks,” in CVPR, 2019.
- [30] T. Chen, L. Lin, R. Chen, X. Hui, and H. Wu, “Knowledge-guided multi-label few-shot learning for general image recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 3, pp. 1371–1384, 2022.

- [31] T. Chen, T. Pu, Y. Xie, H. Wu, L. Liu, and L. Lin, “Cross-domain facial expression recognition: A unified evaluation benchmark and adversarial graph learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.
- [32] X. Li, G. Duan, Z. Wang, J. Ren, Y. Zhang, J. Zhang, and K. Song, “Recovering extremely degraded faces by joint super-resolution and facial composite,” in ICTAI, 2019.
- [33] X. Hu, W. Ren, J. Yang, X. Cao, D. P. Wipf, B. Menze, X. Tong, and H. Zha, “Face restoration via plug-and-play 3d facial priors,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.
- [34] J. Gu, Y. Shen, and B. Zhou, “Image processing using multi-code gan prior,” in CVPR, 2020.
- [35] S. Zhou, K. C. Chan, C. Li, and C. C. Loy, “Towards robust blind face restoration with codebook lookup transformer,” arXiv preprint arXiv:2206.11253, 2022.
- [36] Y. Zhao, Y.-C. Su, C.-T. Chu, Y. Li, M. Renn, Y. Zhu, C. Chen, and X. Jia, “Rethinking deep face restoration,” in CVPR, 2022.
- [37] J. Dai, H. Qi, Y. Xiong, Y. Li, G. Zhang, H. Hu, and Y. Wei, “Deformable convolutional networks,” in ICCV, 2017.
- [38] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in NIPS, 2017.
- [39] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., “Language models are few-shot learners,” arXiv preprint arXiv:2005.14165, 2020.
- [40] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pretraining of deep bidirectional transformers for language understanding,” arXiv preprint arXiv:1810.04805, 2018.
- [41] H. Wang, Y. Zhu, H. Adam, A. Yuille, and L.-C. Chen, “Maxdeeplab: End-to-end panoptic segmentation with mask transformers,” in CVPR, 2021.
- [42] P. Esser, R. Rombach, and B. Ommer, “Taming transformers for high-resolution image synthesis,” in CVPR, 2021.
- [43] N. Parmar, A. Vaswani, J. Uszkoreit, L. Kaiser, N. Shazeer, A. Ku, and D. Tran, “Image transformer,” in ICML, 2018.
- [44] F. Yang, H. Yang, J. Fu, H. Lu, and B. Guo, “Learning texture transformer network for image super-resolution,” in CVPR, 2020.
- [45] L. Zhao, Z. Zhang, T. Chen, D. N. Metaxas, and H. Zhang, “Improved transformer for high-resolution gans,” arXiv preprint arXiv:2106.07631, 2021.
- [46] M. Zhu, C. Liang, N. Wang, X. Wang, Z. Li, and X. Gao, “A sketchtransformer network for face photo-sketch synthesis,” IJCAI, 2021.
- [47] J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for real-time style transfer and super-resolution,” in ECCV, 2016.
- [48] C. Ledig, L. Theis, F. Husz´ar, J. Caballero, A. Cunningham, A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang et al., “Photorealistic single image super-resolution using a generative adversarial network,” in CVPR, 2017.
- [49] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” arXiv preprint arXiv:1409.1556, 2014.
- [50] K. He, G. Gkioxari, P. Doll´ar, and R. Girshick, “Mask r-cnn,” in ICCV, 2017.
- [51] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, “Arcface: Additive angular margin loss for deep face recognition,” in CVPR, 2019.
- [52] Z. Liu, P. Luo, X. Wang, and X. Tang, “Deep learning face attributes in the wild,” in ICCV, 2015.
- [53] G. B. Huang, M. Mattar, T. Berg, and E. Learned-Miller, “Labeled faces in the wild: A database forstudying face recognition in unconstrained environments,” in Workshop on faces in’Real-Life’Images: detection, alignment, and recognition, 2008.
- [54] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv, 2014.
- [55] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” NIPS, 2017.
- [56] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in CVPR, 2018.
- [57] D. E. King, “Dlib-ml: A machine learning toolkit,” The Journal of Machine Learning Research, 2009.
- [58] Y. Blau, R. Mechrez, R. Timofte, T. Michaeli, and L. Zelnik-Manor, “The 2018 pirm challenge on perceptual image super-resolution,” in ECCVW, 2018.
- [59] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in CVPR, 2009.

[60] M. Chen, A. Radford, R. Child, J. Wu, H. Jun, D. Luan, and I. Sutskever, “Generative pretraining from pixels,” in ICML, 2020.

[Figure 147]

Zhouxia Wang received the B.Eng. and M.Eng. degrees from the School of Data and Computer Science at Sun Yat-sen University, Guangzhou, China, in 2015 and 2018. She is currently a Ph.D. student at the University of Hong Kong (HKU), Hong Kong SAR, China. Her research interests include deep learning and low-level vision.

Ping Luo is an Assistant Professor in the department of computer science, The University of Hong Kong (HKU). He received his PhD degree in 2014 from Information Engineering, the Chinese University of Hong Kong (CUHK), supervised by Prof. Xiaoou Tang and Prof. Xiaogang Wang. He was a Postdoctoral Fellow in CUHK from 2014 to 2016. He joined SenseTime Research as a Principal Research Scientist from 2017 to 2018. His research interests are machine learning and computer vision. He has published 100+ peer-

[Figure 148]

reviewed articles in top-tier conferences and journals such as TPAMI, IJCV, ICML, ICLR, CVPR, and NIPS. His work has high impact with 18000+ citations according to Google Scholar. He has won a number of competitions and awards such as the first runner up in 2014 ImageNet ILSVRC Challenge, the first place in 2017 DAVIS Challenge on Video Object Segmentation, Gold medal in 2017 Youtube 8M Video Classification Challenge, the first place in 2018 Drivable Area Segmentation Challenge for Autonomous Driving, 2011 HK PhD Fellow Award, and 2013 Microsoft Research Fellow Award (ten PhDs in Asia).

[Figure 149]

Jiawei Zhang is a research scientist at SenseTime Research. He received a PhD degree from City University of Hong Kong in 2018, a master degree from Institute of Acoustics, Chinese Academy of Sciences in 2014 and a bachelor degree from University of Science and Technology of China in 2011 receptively. His research interests include image deblurring, image superresolution and related computer vision problems.

Tianshui Chen received the Ph.D. degree in computer science from the School of Data and Computer Science at Sun Yat-sen University, Guangzhou, China, in 2018. From 2016 to 2017, he was a research assistant at Hong Kong Polytechnic University. He is currently an associated professor at Guangdong University of Technology. His current research interests include computer vision and machine learning. He has authored and coauthored more than 30 papers published in top-tier academic journals and conferences,

[Figure 150]

including T-PAMI, T-NNLS, T-IP, T-MM, CVPR, ICCV, AAAI, IJCAI, ACM MM, etc. He has served as a reviewer for numerous academic journals and conferences. He was the recipient of the Best Paper Diamond Award at IEEE ICME 2017.

Wenping Wang is now Professor of the Department of Computer Science & Engineering at Texas A&M University. He has been Chair Professor and Head (2012-2017) of the Department of Computer Science at the University of Hong Kong. His research interests are computer graphics and computer vision. He has over 300 technical publications in related fields. He serves or has served as journal associate editor of CAGD, CAG, TVCG, CGF, IEEE Computer Graphics and Applications, and IEEE Transactions on Computers. He has

[Figure 151]

chaired a number of international conferences, including SPM 2006, SMI 2009, Pacific Graphics 2012, GD/SPM’13, SIGGRAPH Asia 2013, Geometry Summit 2019 and Geometry Summit 2023. He has been Founding Chairman of Asian Graphics Association (2016-2020).

