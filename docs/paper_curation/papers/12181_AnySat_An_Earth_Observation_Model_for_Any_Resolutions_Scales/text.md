# arXiv:2412.14123v3[cs.CV]9May2025

## AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities

Guillaume Astruc 1,3,4 Nicolas Gonthier 1,2 Cl´ement Mallet 1 Loic Landrieu1,4 1 LASTIG, Univ Gustave Eiffel, IGN, ENSG, France 2 IGN, France 3 CNES, France 4 LIGM, Ecole Nationale des Ponts et Chauss´ees, IP Paris, Univ Gustave Eiffel, CNRS, France

#### Abstract

Geospatial models must adapt to the diversity of Earth observation data in terms of resolutions, scales, and modalities. However, existing approaches expect fixed input configurations, which limits their practical applicability. We propose AnySat, a multimodal model based on joint embedding predictive architecture (JEPA) and scale-adaptive spatial encoders, allowing us to train a single model on highly heterogeneous data in a self-supervised manner. To demonstrate the advantages of this unified approach, we compile GeoPlex, a collection of 5 multimodal datasets with varying characteristics and 11 distinct sensors. We then train a single powerful model on these diverse datasets simultaneously. Once finetuned or probed, we reach state-of-the-art results on the test sets of GeoPlex and for 6 external datasets across various environment monitoring tasks: land cover mapping, tree species identification, crop type classification, change detection, climate type classification, and segmentation of flood, burn scar, and deforestation. Our code and models are available at https://github.com/gastruc/AnySat.

#### 1. Introduction

From a remote sensing perspective, the natural images of computer vision are remarkably uniform: they are captured by nearly identical sensors (standard cameras) with the same RGB channels and are often taken from similar perspectives. This consistency allows the creation of large composite image datasets from various sources [24, 53, 61], which are key for image foundation models to learn powerful, generalpurpose features [8].

In contrast, Earth observation (EO) data displays significant variability in modalities, scales, and spatial, temporal, and spectral resolutions. Existing EO foundation models are generally trained on a single dataset with a specific format [11, 31, 45, 74], and cannot be applied to datasets with different input types without retraining from scratch—defeating

PASTIS-HD TSAI-TS S2NAIP FLAIR PLANTED

1280m 60m 640m 102.4m 120m

Aerial, SPOT6, NAIP

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|Ima|[Figure 5]<br><br>ge|
|---|---|
| | |

|Ima|ge|
|---|---|
|[Figure 6]| |

|Ima|ge|
|---|---|
| |[Figure 7]|

|Ima|ge|
|---|---|
| |[Figure 8]|

|Ima|ge|
|---|---|
|[Figure 9]| |

- Sentinel1, ALOS-2
- Sentinel2

|Ima|[Figure 10]<br><br>ge|
|---|---|
| | |

|Ima|ge|
|---|---|
|[Figure 11]| |

|Ima|ge|
|---|---|
|[Figure 12]| |

|[Figure 13]<br><br>Ima|ge||Ima|ge|
|---|---|
| |[Figure 15]|
<br><br>|
|---|---|---|
| | | |
| | | |

|Ima|ge|
|---|---|
| |[Figure 15]|

|Ima|ge|
|---|---|
| |[Figure 16]|

|Ima|ge|
|---|---|
| |[Figure 17]|

LandSat 7/8/9

↑ SSL pre-training ↓ fine-tuning / linear probing

AnySat

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Picea Alnus Rutaceae

land cover mapping

tree species identification

crop type classification

deforestation detection

flood segmentation

Figure 1. Multi-Dataset Training. For the first time, a single model can be pretrained simultaneously on a collection of Earth Observation datasets with heterogeneous resolutions, scales, and modalities. The resulting model can be fine-tuned to achieve stateof-the-art results for a wide variety of data types and tasks.

the purpose of foundation models. EO foundation models should be able to seamlessly integrate new datasets for training and prediction, regardless of their resolution, scale, and modalities. As recent efforts provide more flexibility in terms of modalities [7, 38], scale [56], or spectral resolutions [77], none fully leverage the diversity of EO sensors.

We introduce AnySat, a novel EO model using the spatial alignment of multiple modalities as a source of selfsupervision. Indeed, while multiple observations of the same area from distinct sensors capture different information, they share the same underlying semantics. Therefore, we can expect the learned representations to be consistent across

modalities. Moreover, we should be able to reconstruct missing modalities from available ones, encouraging the use of cross-modal masked auto-encoding techniques [7, 34]. However, EO data are subject to complex disruptors such as weather conditions, acquisition angles, and variations in time of day or year. To overcome this issue, we design a new multimodal Joint Embedding Predictive Architecture (JEPA) [6] to learn representations that are consistent in feature space.

A key advantage of our JEPA model is that it eliminates the need for modality-specific decoders, allowing us to handle a wide variety of sensors seamlessly. Combined with our scale-adaptive patch encoder architecture, this approach enables us to train a single model on highly heterogeneous collections of multimodal EO datasets. Notably, over 75% of the learnable parameters in our model are shared across all modalities and resolutions, and thus fully benefit from large and varied training data for self-supervision.

To evaluate our approach, we compile GeoPlex, a collection of 5 multimodal datasets including 11 distinct modalities, with aerial images and satellite time series, radar and optical sensors. GeoPlex spans various spatial resolutions (from 0.2 to 250 m per pixel), revisit times (from single images to weekly time series), channel counts (3 to 11), and spatial extent (samples ranging from 0.4 to 160K hectares). To showcase the versatility of AnySat, we also consider 6 external evaluation datasets with diverse characteristics. After fine-tuning, AnySat achieves state-of-the-art performance on 9 downstream tasks, including classification, segmentation, and change segmentation across domains such as land cover mapping, crop type classification, tree species identification, and deforestation detection. Our contributions are as follows:

- • We present AnySat, a versatile architecture capable of learning from multiple EO sources with heterogeneous resolutions, scales, and modalities.
- • We introduce the first application of JEPA for multimodal EO data, enabling large-scale and efficient selfsupervised learning.
- • We demonstrate that, when pretrained on a curated collection of EO datasets, AnySat can be fine-tuned or linearlyprobed to achieve state-of-the-art performance across a diverse array of tasks and datasets.

Thanks to its flexible design, our pretrained model can be applied to scales ranging from a single forest plot to tiles covering hundreds of square kilometers, and adapt to diverse sensor setups—from unimodal data to any combination of the 11 sensors featured in GeoPlex. In addition, we demonstrate that AnySat successfully generalizes to new sensor configurations not present in its training set.

#### 2. Related Work

In this section, we review the dynamic field of selfsupervised learning in geospatial models, highlighting recent

efforts to enhance their adaptability to diverse inputs. Finally, we present the feature-predictive paradigm, which is instrumental to improve the versatility of EO models.

Self-Supervised Geospatial Models. The abundance of raw EO data makes it particularly suitable for self-supervised learning approaches [9, 13, 47, 70]. Generative models leverage the unique properties of EO data with adapted strategies such as spectral [18], temporal [21, 22], and spatio-temporal [38, 79], and hybrid [69] masking. Other approaches predict rotated [42] or rescaled [50, 56, 66] versions of the input data, or predict missing modalities from available ones [7, 23]. However, these models are often trained on specific combinations of modalities and are limited to those modalities during inference, which hinders their applicability as foundation models expected to adapt to diverse scenarios.

Versatile EO Models. Several approaches have been proposed to improve the generalizability of EO models. Some models address variability in spatial resolutions by training on images of different resolutions and generalizing to coarser scales [56], while others manage spectral variability by training on sensors with different spectral bands [77]. Temporal adaptability is achieved in models capable of handling both single-date images and image time series [7, 11, 31]. Attempts have also been made to generalize across modalities by training on data from different sensors [36, 38] or and even text or audio [60]. Despite these efforts, many models are still trained with a single scale and expect the input to have a certain shape, typically 224 × 224 pixels. They resize other inputs to fit the model architecture, leading to inefficiencies for smaller inputs [69, Tab 5]. A key obstacle preventing the creation of truly versatile generative self-supervised models is the requirement for multiple encoders, decoders, and augmentations to handle different configurations. In this paper, we explore feature-predictive architectures as a promising solution to this challenge.

Feature-Predictive Architectures. Self-supervised learning methods have achieved significant success in image analysis [16, 33, 53]. These approaches learn without labels using pretext tasks, which can be discriminative [29, 51], contrastive [15, 16, 30, 33], or generative, where the model predicts a degraded version of its input [34, 73]. Recent works have proposed performing reconstruction in feature space rather than input space (e.g., pixel space) [10, 78]. Among feature-predictive architectures, the Joint Embedding Predictive Architecture (JEPA) has shown particular promise [6] by learning to predict the features of masked parts of an input image. Feature space reconstruction based model can also be combined with contrastive objectives for improved stability and representation quality [10].

(∆m/δm)2 × (δm2 × Tm × Cm) (∆m/δm)2 × E

∆m × ∆m × Tm × Cm

E

| |[Figure 22]<br><br>|[Figure 23]<br><br>| |encode sub-patches|
|---|---|---|---|---|
|Ima|ge|Ima|ge| |
|[Figure 24]<br><br>[Figure 25]<br><br>| |[Figure 26]<br><br>| |ϕproj|
|Ima|ge|Ima|ge| |

|[Figure 27]<br><br>Ima|ge|split in sub-patches|
|---|---|---|
| | |size : δm pixels|

merge sub-patches

xmp

| |
|---|

ϕtrans

fmp

- Figure 2. Scale-Adaptive Patch Encoding. We consider a patch xmp of resolution ∆m = P/Rm pixels. We first split xmp into sub-patches of size δm pixels, which are mapped by a modality-specific projector ϕprojm to a E-dimensional embedding. Then, a shared spatial transformer module ϕtrans combines all sub-patches into a vector of size E. As the sub-patch size δm is fixed, the patch sizes ∆m only influences the number of input tokens to ϕtrans, allowing us to use the same network for different resolutions.

Because it bypasses the need for complex data augmentations or decoder networks, JEPA is particularly well-suited for massively multimodal applications such as Earth observation. SAR-JEPA [41] introduces the first implementation of JEPA concepts for EO, focusing exclusively on SAR data. In this paper, we combine JEPA with a versatile spatial encoder architecture, allowing a single model to handle diverse data scales, resolutions, and modalities.

#### 3. Method

We first describe our proposed architecture (Sec. 3.1) and self-supervised training procedure (Sec. 3.2). Then, we detail the fine-tuning and probing methods used for downstream tasks (Sec. 3.3). Our work focuses primarily on multi-dataset self-supervised training. However, for clarity, we initially describe the method for a single multimodal dataset, later generalizing it to multiple datasets.

##### 3.1. Architecture

Tiles with multimodal observations are first partitioned into spatially aligned patches. Unlike classical Vision Transformers [20], our model supports patches of varying sizes, accommodating the significant scale variations common in Earth Observation (EO) datasets. Each patch is embedded via a scale-adaptive patch encoder, after which a combiner network integrates representations from multiple modalities into a unified spatial embedding.

Formally, we consider a tile x of size S × S meters, observed through multiple modalities M. Each modality m ∈ M has its own resolution Rm (meters per pixel), temporal observations Tm (with Tm = 1 for single-date modalities), and number of channels Cm (e.g., spectral or polarization channels). The tile x observed in modality m is denoted xm and is represented as a tensor of shape (S/Rm) × (S/Rm) × Tm × Cm.

Spatially Consistent Patching. Tiles are partitioned into a set P of non-overlapping patches, each of size P ×P meters. An input token xmp represents the observation of patch p ∈ P in modality m. All modalities share the same spatial patch layout, ensuring spatial consistency across modalities. The

total number of tokens is thus |M| · (S/P)2. Although the patch size is constant across modalities, each token may have distinct tensor dimensions due to differing resolutions, temporal extents, and channel numbers.

Patch Encoding. We design a scale-adaptive patch encoder ϕpatch to map each input token xmp into a fixed-size vector fpm ∈ RE, regardless of modality resolution Rm or patch size P. The encoding scheme, illustrated in Fig. 2, involves three stages:

- (i) We first subdivide each token into fixed-size sub-patches of δm × δm pixels, flattening their spatial dimensions to vectors of size δm2 TmCm.
- (ii) Each flattened sub-patch is mapped to dimension E via a modality-specific MLP ϕprojm . For multi-temporal modalities (Tm > 1), a Lightweight Temporal Attention Encoder (LTAE)[26] collapses the temporal dimension.
- (iii) We add positional encodings based on ground sampling distance [56] to the sub-patch embeddings. A shared transformer network ϕtrans with B blocks aggregates the

sub-patch embeddings into a single representation fpm per modality using a CLS-like token.

Using sub-patches of fixed sizes δm allows ϕpatch to process patches of different patch sizes P without rescaling the input data. Indeed, changes in P only influence the number of input tokens processed by ϕtrans, which has no incidence on the embedding size.

Modality-Combiner Network. The combiner network ϕcomb merges embeddings fpm from all available modalities into a multimodal representation fp⋆ for each patch p ∈ P. We use the cross-attention-based architecture proposed by OmniSat [7, 3.1]: (i) We first add to each fpm an absolute positional encoding pos(p)—the same one used for subpatches.; (ii) The tokens go through a sequence of B selfattention blocks; (iii) We associate each patch with a token with a shared learned value and add positional encoding; and

- (iv) We compute the cross-attention between these tokens and the embeddings of the last self-attention block. This results in one embedding per patch fp⋆.

Lcon

dropped patches mask token fmask

| |
|---|

fp,mS fp,⋆S fp,⋆pred

⋆

xmp

| |
|---|

⋆

⋆

|[Figure 28]|
|---|

|[Figure 29]|
|---|

⋆ multimodal token

- ϕpatchS

- ϕpatchT

ϕpredS

- ϕcombS

- ϕcombT

⋆

⋆

drop token fdrop

|[Figure 30]|
|---|

|[Figure 31]|
|---|

GEOPLEX

⋆

|[Figure 32]|
|---|

|[Figure 33]|
|---|

|[Figure 34]|
|---|

|[Figure 35]|
|---|

STUDENT

[Figure 36]

[Figure 37]

Image

Image

|Ima|[Figure 38]<br><br>ge|
|---|---|
| | |

|Ima|ge|
|---|---|
|[Figure 39]| |

|Ima|ge|
|---|---|
| |[Figure 40]|

|Ima|ge|
|---|---|
|[Figure 41]| |

|Ima|ge|
|---|---|
| |[Figure 42]|

EMA EMA

Image

LJEPA

|Ima|ge|
|---|---|
| |[Figure 43]|

|Ima|ge|
|---|---|
|[Figure 44]| |

|Ima|ge|
|---|---|
|[Figure 45]| |

|[Figure 46]<br><br>Ima|ge||Ima|ge|
|---|---|
|[Figure 48]| |
|
|---|---|---|
| | | |
| | | |

|Ima|ge|
|---|---|
|[Figure 48]| |

[Figure 49]

[Figure 50]

Image Image

TEACHER

|Ima|ge|
|---|---|
|[Figure 51]| |

|Ima|ge|
|---|---|
|[Figure 52]| |

|Ima|ge|
|---|---|
|[Figure 53]| |

|Ima|ge|
|---|---|
| |[Figure 54]|

fp,mT fp,⋆T

⋆

⋆

|Ima|[Figure 55]<br><br>ge|
|---|---|
| | |

|[Figure 56]<br><br>Ima|ge|
|---|---|
| | |

positional encoding

⋆

|EMA|
|---|

Patches of size P

⋆

exponential moving average

- Figure 3. Architecture of AnySat. We begin each iteration by randomly selecting a dataset among GeoPlex and sampling a tile. Each

available modality is divided into spatially aligned patches of size P. The student network’s patch encoder ϕpatchS embeds each patch and we apply a contrastive loss to encourage spatial consistency across modalities. We then apply dropping and masking : some patches have all modalities removed (dropping), while others have only random modalities removed (masking). The remaining patches are merged in the modality combiner ϕcombS to form multimodal representations fS⋆ for the non-dropped patches. The predictor ϕpredS then reconstructs the embeddings of the dropped patches. Finally, the student network’s output is compared to the teacher’s, whose weights are an Exponential Moving Average (EMA) of the student’s weights and which processes the complete set of patches without masking or dropping.

##### 3.2. Training

We adapt the Joint Embedding Predictive Architecture (JEPA) framework [6] to multimodal Earth Observation, enabling self-supervised pretraining on datasets of varying modalities without labels. A student network operates on heavily masked inputs, aiming to predict embeddings generated by an unmasked teacher network whose parameters follow an Exponential Moving Average (EMA) of the student’s weights [33]. Training leverages two losses: a contrastive loss to enforce modality consistency and a JEPA loss for masked embedding prediction.

The student network consists of a patch encoder ϕpatchS ,

a modality combiner ϕcombS , and a predictor network ϕpredS with 3 self attention blocks. The teacher network includes a

patch encoder ϕpatchT and a modality combiner ϕcombT and no predictor. The student network first embeds all input tokens

xmp into vectors of size E using the patch encoder:

fp,mS = ϕpatchS (xmp ) . (1)

Contrastive Loss. For a fixed patch p, the observations xmp for m ∈ M capture different aspects of the same spatial region but share the same underlying semantics: the content

of p. Therefore, we expect the representations fpm,S to be consistent across modalities. We enforce this intuition with a contrastive loss inspired by OmniSat [7]. Specifically, we use a modified InfoNCE loss [52], where each token (p,m) is positively paired with those from the same spatial patch but different modalities:





exp ⟨fp,mS,fp,nS⟩/τ

−log |P||M|

n̸=m

, (2)

Lcon =

 

 

exp ⟨fp,mS,fq,nS⟩/τ

(p,m)∈P×M

n̸=m q̸=p

where τ is a temperature parameter, and ⟨·,·⟩ denotes the cosine similarity between embeddings.

Joint Embedding Predictive Architecture. We adapt the JEPA self-supervised learning framework [6] to the context of multimodal Earth Observation. Avoiding reconstruction in pixel space is particularly beneficial for EO data, which can be heavily influenced by factors such as weather, time of day, or acquisition angle. Reconstructing in latent space allows us to learn more consistent and semantically meaningful features. The training process proceeds as follows:

- • Patch Dropping. We apply JEPA’s masking strategy by randomly selecting five rectangular regions on the tile. Let K ⊂ P be the set of patches intersected by these rectangles, and K¯ = P \ K the remaining patches. We drop all the student’s tokens fp,mS for patches p ∈ K.
- • Modality & Temporal Masking: We randomly mask a subset L ⊂ K¯ × M of the remaining tokens, ensuring that at least one modality per patch remains unmasked. Masked token embeddings are replaced with a fixed value fmask ∈ RE, which is learned as a parameter of the network. We also randomly mask 50% of the timestamps of all time series.
- • Combiner: We input all tokens (masked or not) to the

student’s combiner ϕcombS , producing multimodal embeddings fp,⋆S for all p ∈ K¯:

fp,⋆S = ϕcomb({fp,mS}(p,m)̸∈L ∪ {fmask}(p,m)∈L) . (3)

- • Predictor: We replace each dropped patch p ∈ K with a fixed value fdrop ∈ RE. We add positional encodings to all tokens (including the dropped ones) and input them

to the predictor ϕpredS , yielding embeddings fp,⋆pred for all patches p ∈ P:

fp,⋆pred = ϕpredS ({fp,⋆S}p∈K¯ ∪ {fdrop}p∈K) . (4)

- • Teacher Encoding: The teacher network receives all input tokens xmp , embeds them using ϕpatchT , and combines

them with ϕcombT without any dropping, masking, or temporal dropout. The teacher outputs patch embeddings

fp,⋆T for all p ∈ P.

- • Loss Function: The training objective is the L2 distance between the student predictions and the teacher’s multimodal embeddings for the dropped patches:

1 |K| p∈K

fp,⋆pred − fp,⋆T 22 . (5)

LJEPA =

After training, we use the teacher network for downstream tasks and discard the student. Note that all modules are shared across all modalities except for the projection layers ϕprojm in the patch encoder ϕpatch.

Training with Multiple Datasets. The flexibility of AnySat enables us to train a single model simultaneously on several datasets of various sizes and scales with the same weights and without rescaling. We consider a set D of such datasets. Each dataset d ∈ D is characterized by the subset Md ⊂ M of its available modalities and Sd the size of its tiles. We also consider a batch size Bd and a set Pd of acceptable patch sizes, which depend on the nature of the data, the available resolution, and the tile size. We use the following procedure:

1. Randomly select a dataset d in D.

[Figure 57]

|[Figure 58]|
|---|

GeoPlex

S2NAIP

External datasets SICKLE

Planted FLAIR

| |
|---|

| |
|---|

BraDD-S1TS TS2Crop

| |
|---|

| |
|---|

Sen1Flood11 SO2SAT

TSAITS PASTIS

| |
|---|

| |
|---|

HLS BurnScar

Figure 4. Datasets Considered. GeoPlex is composed of 5 diverse dataset spanning the entire world, with a higher concentration in Europe and the US where open-data are more abundant. We also consider external evaluation datasets with a more diverse spread.

- 2. Randomly select a patch size P in Pd.
- 3. Randomly sample Bd tiles in d.
- 4. Process the tiles and backpropagate the loss. 3.3. Downstream Tasks

After pretraining, AnySat can be fine-tuned or probed for various downstream tasks, including classification and semantic segmentation.

Classification. For tile-level classification, we insert a [CLS] token into the combiner network’s cross-attention module. This token generates a tile-level embedding, subsequently mapped to label logits through a linear classifier.

Semantic Segmentation. For semantic segmentation, we predict labels at pixel-level resolution by first selecting a modality whose resolution is close to the annotation resolution. A dense feature map at the sub-patch scale (δm) is formed by concatenating sub-patch embeddings (outputs of ϕprojm ) with corresponding multimodal patch embeddings (outputs of ϕcomb). An MLP then maps these concatenated embeddings to logits of dimension δm × δm × N, where N is the number of semantic classes. Unfolding these logits yields pixel-level predictions. Using sub-patches results in higher-resolution predictions compared to methods that rely only on patch-level representations.

Probing. AnySat supports linear probing, where a simple linear classifier can be attached directly to the class token for classification or to the dense feature maps for segmentation. This approach avoids complex segmentation heads typically required in earlier methods [48], leveraging the dense features produced by our architecture.

New Sensor Configurations. AnySat can adapt to sensors with configurations differing from those in the training

Test sets of datasets from GeoPlex

FLAIR semseg (mIoU)

PLANTED classif (macro F1)

TreeSatAI-TS classif (weighted F1)

PASTIS-HD classif (macro F1)

PASTIS-HD semseg (mIoU)

| |
|---|

from scratch fine-tuned linear probing UperNet probing

65

58

75

70

66

| |
|---|

56

65

60

| |
|---|

64

60

54

55

36.1

55

| |
|---|

70

###### AnySat

AnySatOmniSatCROMADOFA

###### AnySat

AnySatOmniSatDOFAPSE+LTAE

AnySatUTAE-MMTSViTUTAE

ViViTS-MM

ViViT-S2

UT&T UNet

UTAE

6.1K free parameters

+

+

⋆

⋆

External datasets

HLS Burn Scar semseg (mIoU)

SICKLE semseg (IoU)

TS2Crop classif (OA)

Sen1Flood11 semseg (mIoU)

So2Sat semseg (mIoU)

BRADD-S1TS chgdet (IoU)

80 6.1K

- 90
- 91
- 92

90

29K 29K

###### 90 3M

###### 6.1K

###### 6.1K

90

60

47M

39M

80

29K

39M 33M

33M

70

85

29K

80

50

14K

51.4

80

70

60

70

AnySatPrithviv2PrithviCROMA

AnySatUTAE3DUNetConvLSTM

AnySat3DUNet-MM3DUNet-S2UTAE

AnySatCROMAPrithviSatlasNet

AnySatDOFACROMASatMAE

AnySatOS-CNNTAELSTM

⋆ models trained on private external data + access to additional context

Figure 5. Quantitative Evaluation. We evaluate AnySat across 9 open-access datasets and for four tasks: multilabel classification (classif), semantic segmentation (semseg), pixel-wise change detection (chgdet), and pixel-wise regression (regression). For clarity, we only visualize the four best performance per dataset, see Appendix for full results. We report the number of trainable parameters for probing evaluations.

datasets well as new sensors. During self-supervised pretraining, we learn a sensor-specific scalar value representing missing data, which is subsequently used wherever modality channels are absent during fine-tuning or probing. For sensors not featured in the training sensors, we randomly initialize a new projector and fine-tune it along the other free parameters. This effectively extends AnySat to previously unseen sensors but cannot be used in probing for sensors too different from the training mix.

#### 4. Experiments

##### 4.1. Datasets and Evaluation

We present the datasets used for training and evaluation, as well as our evaluation protocol.

GeoPlex. As argued by Roscher et al. [58], EO models benefit from high-quality, diverse, and curated data rather than extensive but uniform acquisitions. We follow this principle by compiling a collection of five multimodal datasets, each featuring different combinations of modalities, scales, and resolutions. GeoPlex comprises the training sets of the following datasets:

- • TreeSatAI-TS [3, 7]: A forest-centric dataset in Germany with Sentinel-1 & 2 time series and Very High Resolution (VHR) images at 0.2 m resolution.
- • FLAIR [25]: A French land cover dataset with Sentinel-

2 time series and VHR images with elevation data (0.2 m). To form multimodal patches, we crop the Sentinel-2 time series to match the extent of the VHR images (discarding 93.5% of pixels).

- • PLANTED [54]: A global forest dataset comprising time series from multiple sensors, including Sentinel1/2, Landsat-7, ALOS-2, and MODIS. Only 1.3 of the 2.3M images used in the paper are publicly available.
- • S2NAIP-URBAN [11]: An urban dataset in the continental US with VHR images (1.25m) and time series from Sentinel-1/2 and Landsat-8/9.
- • PASTIS-HD [7, 27]: A French crop mapping dataset with VHR images (1.5m) and Sentinel-1 & 2 time series. As PASTIS is evaluated in 5-fold cross-validation, there are no dedicated train and test sets. We include the entire dataset (without labels) in GeoPlex.

As illustrated in Fig. 4, GeoPlex spans 249K km2 across five continents and 171 billion pixels. The sampled tiles range in size from 0.36 to 164 hectares. GeoPlex includes 11 distinct modalities with resolutions ranging from 0.2 m to 250 m, with both VHR images and time series data:

###### • Very High Resolution Images:

- – Aerial: RGB+NIR (near-infrared) at 0.2 m
- – Aerial+NMS: RGB+NIR+Elevation at 0.2 m
- – NAIP: RGB+NIR at 1.25 m
- – SPOT6: RGB+NIR at 1.5 m.

###### • Time Series Data:

- – Sentinel-1: 3 channels (VV/VH polarization + ratio) at 10 m, Ascending & Descending Orbits
- – Sentinel-2: 10 channels at 10 m
- – ALOS-2: 3 channels (polarization) at 30 m
- – Landsat-7: 6 channels at 30 m
- – Landsat-8/9: 11 channels at 30 m
- – MODIS: 7 channels at 250 m.

We select the possible patch size per dataset, while we set the sub-patch size per modality 1 pixel for very high-resolution images and 10 pixels for time series data. See the Appendix for the complete characteristics of all datasets.

External Datasets. To showcase AnySat’s flexibility, we also consider 6 datasets not included in GeoPlex. AnySat can be directly fine-tuned or linearly probed on new datasets, even if their modality combination is not featured in GeoPlex. We consider the following datasets:

- • SICKLE [59]: A multimodal crop mapping dataset in India featuring Sentinel-1, Sentinel-2, and Landsat-8 time series. As the test set has not been released, we use the validation set.
- • BraDD-S1TS [39]: A change detection dataset comprising Sentinel-1 time series of the Amazon rainforest, aiming to segment deforested areas.
- • TimeSen2Crop [75]: A crop mapping dataset in Slovenia consisting of single-pixel Sentinel-2 time series, a modality not present in GeoPlex.
- • Sen11Flood1 [12]: A global flood mapping dataset with pixel annotations and single-date Sentinel-1 and 2 observations, a configuration not present in GeoPlex. Each tile covers 2600 hectares.
- • So2Sat [80]: A local climate zone classification dataset containing co-registered Sentinel-1 and Sentinel-2 imagery across multiple cities worldwide, with single-date observations—a configuration not present in GeoPlex.
- • HLS Burn Scar [55]: A dataset for burn scar detection using Harmonized Landsat-Sentinel (HLS) imagery, featuring time series data to identify post-fire affected areas and large tiles of 24K hectares.

Evaluation. We evaluate our model on the annotated datasets of GeoPlex (excluding S2NAIP-URBAN) and the 6 external datasets across three tasks: (i) Classification: TSAIT-TS, PASTIS-HD, PLANTED, TimseSenCrop, So2Sat; (ii) Semantic Segmentation: PASTIS-HD, FLAIR, SICKLE, Sen1Flood11, HLS Burn Scars; and (iii) Binary pixel-wise change detection: BraDD-S1TS.

We use three evaluation settings to evaluate the models: • From Scratch. The model is trained directly on the

labeled training set in a supervised manner.

- • Fine-tuning. The model is pretrained in a selfsupervised manner, then fine-tuned on the training set.
- • Linear Probing. The model is initially pretrained in a self-supervised manner, and a linear layer is fitted with the training set.

Competing Methods. We compare AnySat against stateof-the-art Earth Observation models. Most foundation models pre-trained on external data cannot be directly applied to target datasets with different input configurations. For example, the ScaleMAE and SatMAE models are trained on the Functional Map of the World [17] and limited to RGB bands, while CROMA is trained on single-date Sentinel-2 data. Since these specific modalities are not present in any of our evaluation datasets, we cannot directly evaluate these pretrained models. Instead, we modify the input layers of these models to match the target number of spectral bands.

##### 4.2. Results and Analysis

We evaluate our model on different datasets from and outside of GeoPlex with fine-tuning and linear probing.

Performance on GeoPlex’ Test Sets. We evaluate AnySat on the test sets of the GeoPlex datasets, as shown in Fig. 5, with detailed results provided in the Appendix. Despite using a single pretrained model, AnySat sets new state-ofthe-art results for TreeSatAI-TS (+0.9 weighted F1 score) and PASTIS-HD (+2.8 mIoU in classification and +0.2 in segmentation). AnySat also achieves near state-of-the-art performance on PLANTED [54], even though the ViViT models [5] were trained on a withheld dataset with nearly 80% more data of the same type. Similarly, our model performs close to the state-of-the-art on FLAIR, despite having access to only 6.5% of the extent of the Sentinel-2 tiles used by UT&T [25].

Pretraining on GeoPlex consistently improves performance, indicating that training on a collection of datasets with varied modalities leads to richer and more robust representations. The improvement is more pronounced for smaller datasets like TreeSatAI-TS and in classification tasks rather than segmentation. We attribute this to the amount of supervision available in larger datasets and dense annotations, which make pretraining less beneficial.

Performance on External Datasets. Fig. 5 shows that AnySat significantly outperforms the state-of-the-art for 6 external datasets, improving SICKLE by +3.6 mIoU, BraDDS1TS by +10.2 mIoU, and TimeSen2Crop by +11.0 OA. These gains highlight AnySat’s strong spatial generalization as GeoPlex primarily covers the northern hemisphere, while the external datasets have global coverage. We also evaluate AnySat on datasets from GeoBench [40] and PANGAEABench [49], see Tab. 1, AnySat performs on par or better

HLS Burns (mIoU) Spacenet7 (mIoU) Sen1Floods11 (mIoU) So2Sat (OA) PASTIS-HD Fold1 (mIoU)

free param.

Models

HLS Planet S1 +S2 S1 +S2 S1+S2+VHR FT Prob. FT Prob. FT Prob. FT Prob. Prob. FT Prob.

AnySat (ours) 125M 6K-3M 90.6 87.7• 58.8 58.1• - 91.1⋆ 59.1⋆ 65.9 42.7† Prihtvi2 [65] 630M - 90.5 - - - 89.9 - - - Prihtvi [38] 130M 39M 86.9 83.6 - 56.5 90.4 88.3 - - 33.9 CROMA [23] 350M 47M - 82.4 - 59.3 90.9 90.9 49.2⋆ - 32.3 SatMAE [18] 304M 31M - - - - - - 46.9⋆ - ScaleMAE [56] 350M 47M - 76.7 54.0 63.0 - 74.1 - - 24.6 DOFA [77] 151M 39M - 80.6 - 61.8 - 89.4 59.3⋆ - 30.0 RemoteCLIP [43] 168M 39M 65.1 76.6 - 57.8 - 74.3 - - 18.2 DINO-S12 [53] 54M 31M - 81.7 56.6 56.5 - 88.6 - 44.0 36.2 SatlasNet [37] 111M 33M - 80.0 - 61.9 - 90.3 - 29.1 17.5

- Table 1. Comparison to Foundation Models. We evaluate AnySat fine-tuned or linearly probed on datasets taken from the GeoBench [40] and PANGEA benchmarks [49]. Unseen sensor , unseen sensor configuration , : single date, FT: fine-tuning, Prob.: linear or UperNet probing. ⋆/ † /•: Probing requires 6K/33K/3M free parameters. The best performance is in bold whereas the second one is underlined.

than all foundation models across all benchmarks. Note that these methods use resizing (on PASTIS-HD and TreeSatAI) and sliding window aggregation (on SN7) instead of our proposed adaptive patch encoding.

Moreover, AnySat can be effectively linearly probed for semantic segmentation. It surpasses all specialized approaches on BraDD-S1TS when linearly probed, and likewise exceeds the performance of foundation models with fine-tuned UperNet segmentation heads on Sen1Flood11. Notably, a linearly probed AnySat outperforms a fine-tuned Prithvi2 [65] on Sen1Floods11 with 105 fewer free parameters. These findings underscore the expressive power of AnySat’s self-supervised features and confirm that it can be adapted to new tasks and datasets at minimal training cost and still deliver competitive performance.

Performance on New Sensor Configurations. We demonstrate AnySat’s robustness in handling sensor configurations not present in GeoPlex. For instance, SICKLE’s LandSat8 requires three additional bands beyond those used in S2NAIP’s LandSat8, while TimeSen2Crop provides only 9 of the 10 bands employed by our Sentinel-2 projector network. Applying the padding strategy described in Sec. 3.3, AnySat achieves state-of-the-art results on both datasets. We also evaluate AnySat on single-date Sentinel images (So2Sat, Sen1Flood11) and single-pixel time series (TimeSen2Crop), which were never part of GeoPlex, and again observe stateof-the-art performance. Finally, we test AnySat on the HLSBurnScar dataset [55]. As GeoPlex does not contain HLS data (but contains Sentinel and LandSat), we train a new projector for this new modality. AnySat outperforms all competing methods, including Prithvi [38], which was trained on 252M km2 of HLS imagery. In comparison, GeoPlex comprises only 249K km2 without any HLS data, further illustrating the strong generalization capability of AnySat.

Ablation Study. We evaluate the impact of several key design choices and report the results in Tab. 2. All results are presented for the Fold 5 of PASTIS-HD and for the classification and semantic segmentation tasks. We do not pretrain on the entire GeoPlex but use Fold 1 to 4 of PASTISHD in a self-supervised fashion.

- • Random Token Dropping. We replaced JEPA’s block masking strategy with purely random token dropping for the student network. This modification decreased classification performance but slightly improved segmentation results. In order to use a single model configuration for all tasks, we maintained a unified approach. Interestingly, block masking does not appear to be as critical for EO data than for natural images (see Table 6 in [6]).
- • No Contrastive Loss. We remove the contrastive loss

and retain only the reconstruction loss LJEPA. This substantially reduces the classification performance (−4.3 F1) but only a moderate decrease in segmentation performance (−0.2 mIoU). These findings suggest that the contrastive loss can help the feature-predictive approach learn more discriminative features, particularly benefiting classification tasks.

- • Naive Semantic Segmentation. We predict pixel-wise logits directly from the patch embeddings without utilizing subpatch features. This results in a decrease in segmentation performance by 2.4 mIoU, highlighting the importance of subpatches in providing fine-grained spatial information.

Inference and Training Times. Our model was pretrained on GeoPlex using 1760 GPU-hours on an NVIDIA H100 GPU. Fine-tuning takes between 10 and 40 hours, depending on the dataset size. Linear probing takes approximately 2 hours on BraDD-S1TS.

In terms of inference speed, AnySat processes one monodate tile from TreeSatAI [3] in 3ms on average, which is

- Table 2. Ablation. We evaluate the impact for several critical design choices of our model on the Fold 1 of PASTIS-HD.

classification segmentation macro F1 mIoU

Experiment

best configuration 72.0 63.6 random token dropping 71.3 64.1 no contrastive 67.7 63.4 naive semseg - 61.2

faster than ScaleMAE [56] (10ms) and comparable to DOFA [77] (3ms) and OmniSat [7] (2ms).

#### 5. Conclusion

We have presented AnySat, a versatile architecture designed to address the diversity of EO data in terms of resolutions, scales, and modalities. By leveraging a joint embedding predictive architecture and scale-adaptive spatial encoders, AnySat can be trained in a self-supervised manner on highly heterogeneous datasets. Pretrained on GeoPlex, a comprehensive collection of multimodal datasets with varying characteristics, our model achieved state-of-the-art performance across multiple datasets, tasks, and modalities.

A key advantage of AnySat is its ability to be applied and fine-tuned on a wide array of combinations of data types and scales with a single model. Moreover, new datasets can be easily incorporated into GeoPlex for self-supervised pretraining. Our goal is to generalize this approach to develop a versatile foundation model for environmental monitoring on a global scale.

#### Acknowledgement

This work was granted access to the HPC resources of IDRIS under the allocations AD011014719 and AD011014286R1 made by GENCI. We thank Jordi Inglada, Antoine Labatie, Dimitris Samaras, Yohann Perron, Vincent Lepetit for inspiring discussions and valuable feedback.

#### References

- [1] Lightning: LinearWarmupCosineAnnealingLR. https : / / lightning - flash . readthedocs . io / en / stable / api / generated / flash . core . optimizers . LinearWarmupCosineAnnealingLR . html. Accessed: 2024-11-20. 16
- [2] PyTorch: ReduceLROnPlateau. org / docs / stable / generated / torch . optim . lr _ scheduler . ReduceLROnPlateau . html # torch . optim . lr _ scheduler . ReduceLROnPlateau. Accessed: 2024-02-

29. 16

- [3] Steve Ahlswede, Christian Schulz, Christiano Gava, Patrick Helber, Benjamin Bischke, Michael F¨orster, Florencia Arias, J¨orn Hees, Beg¨um Demir, and Birgit Kleinschmit. TreeSatAI Benchmark Archive: A multi-sensor, multi-label dataset for tree species classification in remote sensing. Earth System Science Data Discussions, 2022. 6, 8, 17, 18
- [4] allenai.org. AI2-S2-NAIP. https://huggingface.co/datasets/allenai/s2-naip,

2024. [Online; accessed 01-Sept-2024]. 17, 18

- [5] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luˇci´c, and Cordelia Schmid. ViViT: A video vision transformer. In CVPR, 2021. 7, 15
- [6] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In CVPR, 2023. 2, 4, 8
- [7] Guillaume Astruc, Nicolas Gonthier, Clement Mallet, and Loic Landrieu. Omnisat: Self-supervised modality fusion for earth observation. In ECCV, 2024. 1, 2, 3, 4, 6, 9, 14, 15, 17, 18
- [8] Muhammad Awais, Muzammal Naseer, Salman Khan, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Foundational models defining a new era in vision: A survey and outlook. arXiv preprint arXiv:2307.13721,

2023. 1

- [9] Kumar Ayush, Burak Uzkent, Chenlin Meng, Kumar Tanmay, Marshall Burke, David Lobell, and Stefano Ermon. Geography-aware self-supervised learning. In ICCV, 2021. 2
- [10] Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. Data2vec: A general framework for self-supervised learning in speech, vision and language. In ICML, 2022. 2, 15
- [11] Favyen Bastani, Piper Wolters, Ritwik Gupta, Joe Ferdinando, and Aniruddha Kembhavi. SatlasPretrain: A large-scale dataset for remote sensing image understanding. In ICCV, 2023. 1, 2, 6, 16
- [12] Derrick Bonafilia, Beth Tellman, Tyler Anderson, and Erica Issenberg. Sen1Floods11: A georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1. In CVPR Workshop EarthVision, 2020. 7, 14, 16, 17, 18
- [13] Jules Bourcier, Gohar Dashyan, Karteek Alahari, and Jocelyn Chanussot. Learning representations of satellite images from metadata supervision. In ECCV, 2024. 2
- [14] Lorenzo Bruzzone and Sebastiano B Serpico. Classification of imbalanced remote-sensing data by neural networks. Pattern recognition letters, 1997. 16

- [15] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 2
- [16] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020. 2
- [17] Gordon Christie, Neil Fendley, James Wilson, and Ryan Mukherjee. Functional map of the world. In CVPR, 2018. 7
- [18] Yezhen Cong, Samar Khanna, Chenlin Meng, Patrick Liu, Erik Rozi, Yutong He, Marshall Burke, David Lobell, and Stefano Ermon. SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery. In NeurIPS, 2022. 2, 8, 15, 16
- [19] Bo Dang and Yansheng Li. MSResNet: Multiscale residual network via self-supervised learning for waterbody detection in remote sensing imagery. Remote Sensing, 2021. 16
- [20] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2020. 3
- [21] Iris Dumeur, Silvia Valero, and Jordi Inglada. Paving the way toward foundation models for irregular and unaligned satellite image time series. arXiv preprint arXiv:2407.08448, 2024. 2
- [22] Iris Dumeur, Silvia Valero, and Jordi Inglada. Selfsupervised spatio-temporal representation learning of satellite image time series. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2024. 2
- [23] Anthony Fuller, Koreen Millard, and James R Green. CROMA: Remote sensing representations with contrastive radar-optical masked autoencoders. In NeurIPS,

2023. 2, 8, 15, 16

- [24] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. In NeurIPS, 2024. 1
- [25] Anatol Garioud, Nicolas Gonthier, Loic Landrieu, Apolline De Wit, Marion Valette, Marc Poup´ee, S´ebastien Giordano, and Boris Wattrelos. FLAIR: A country-scale land cover semantic segmentation dataset from multi-source optical imagery. In NeurIPS Dataset and Benchmark, 2023. 6, 7, 14, 15, 17, 18
- [26] Vivien Sainte Fare Garnot and Loic Landrieu. Lightweight temporal self-attention for classifying satellite images time series. In Advanced Analytics

- and Learning on Temporal Data: ECML PKDD Workshop, 2020. 3, 15, 16
- [27] Vivien Sainte Fare Garnot and Loic Landrieu. Panoptic segmentation of satellite image time series with convolutional temporal attention networks. In ICCV, 2021. 6, 14, 15, 16, 17
- [28] Vivien Sainte Fare Garnot, Loic Landrieu, and Nesrine Chehata. Multi-modal temporal attention models for crop mapping from satellite time series. ISPRS Journal of Photogrammetry and Remote Sensing, 2022. 15, 18
- [29] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. In ICLR, 2018. 2
- [30] Jean-Bastien Grill, Florian Strub, Florent Altch´e, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. In NeurIPS, 2020. 2
- [31] Xin Guo, Jiangwei Lao, Bo Dang, Yingying Zhang, Lei Yu, Lixiang Ru, Liheng Zhong, Ziyuan Huang, Kang Wu, Dingxiang Hu, et al. Skysense: A multi-modal remote sensing foundation model towards universal interpretation for earth observation imagery. In CVPR,

2024. 1, 2, 15

- [32] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016. 15
- [33] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020. 2, 4, 15
- [34] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 2
- [35] Danfeng Hong, Bing Zhang, Xuyang Li, Yuxuan Li, Chenyu Li, Jing Yao, Pedram Ghamisi, Naoto Yokoya, Hao Li, Xiuping Jia, Antonio Plaza, Paolo Gamba, Jon Atli Benediktsson, and Jocelyn Chanussot. SpectralGPT: Spectral remote sensing foundation model. TPAMI, 2024. 15
- [36] Chia-Yu Hsu, Wenwen Li, and Sizhe Wang. Geospatial foundation models for image analysis: Evaluating and enhancing NASA-IBM Prithvi’s domain adaptability. International Journal of Geographical Information Science, 2024. 2
- [37] Jeremy Irvin, Lucas Tao, Joanne Zhou, Yuntao Ma, Langston Nashold, Benjamin Liu, and Andrew Y Ng. USat: A unified self-supervised encoder for multi-sensor satellite imagery. arXiv preprint arXiv:2312.02199, 2023. 8
- [38] Johannes Jakubik, S Roy, CE Phillips, P Fraccaro, D Godwin, B Zadrozny, D Szwarcman, C Gomes, G Nyir-

- jesy, B Edwards, et al. Foundation models for generalist geospatial artificial intelligence. URL https://arxiv. org/abs/2310.18660. 1, 2, 8, 15, 16
- [39] Kaan Karaman, V Sainte Fare Garnot, and Jan Dirk Wegner. Deforestation detection in the Amazon with Sentinel-1 SAR image time series. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 2023. 7, 14, 16, 17, 18
- [40] Alexandre Lacoste, Nils Lehmann, Pau Rodriguez, Evan Sherwin, Hannah Kerner, Bj¨orn L¨utjens, Jeremy Irvin, David Dao, Hamed Alemohammad, Alexandre Drouin, et al. Geo-bench: Toward foundation models for earth monitoring. NeurIPS, 2023. 7, 8
- [41] Weijie Li, Wei Yang, Tianpeng Liu, Yuenan Hou, Yuxuan Li, Zhen Liu, Yongxiang Liu, and Li Liu. Predicting gradient is better: Exploring self-supervised learning for sar atr with a joint-embedding predictive architecture. ISPRS Journal of Photogrammetry and Remote Sensing, 2024. 3
- [42] Zhihao Li, Biao Hou, Siteng Ma, Zitong Wu, Xianpeng Guo, Bo Ren, and Licheng Jiao. Masked angle-aware autoencoder for remote sensing images. arXiv preprint arXiv:2408.01946, 2024. 2
- [43] Fan Liu, Delong Chen, Zhangqingyun Guan, Xiaocong Zhou, Jiale Zhu, Qiaolin Ye, Liyong Fu, and Jun Zhou. Remoteclip: A vision language foundation model for remote sensing. IEEE Transactions on Geoscience and Remote Sensing, 2024. 8
- [44] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR. 16
- [45] Siqi Lu, Junlin Guo, James R Zimmer-Dauphinee, Jordan M Nieusma, Xiao Wang, Parker VanValkenburgh, Steven A Wernke, and Yuankai Huo. AI foundation models in remote sensing: A survey. arXiv preprint arXiv:2408.03464, 2024. 1
- [46] Rose M Rustowicz, Robin Cheong, Lijing Wang, Stefano Ermon, Marshall Burke, and David Lobell. Semantic segmentation of crop type in Africa: A novel dataset and analysis of deep learning methods. In CVPR Workshop EarthVision, 2019. 16
- [47] Oscar Manas, Alexandre Lacoste, Xavier Gir´o-i Nieto, David Vazquez, and Pau Rodriguez. Seasonal contrast: Unsupervised pre-training from uncurated remote sensing data. In ICCV, 2021. 2
- [48] Valerio Marsocci and Nicolas Audebert. Cross-sensor self-supervised training and alignment for remote sensing. arXiv preprint arXiv:2405.09922, 2024. 5
- [49] Valerio Marsocci, Yuru Jia, Georges Le Bellier, David Kerekes, Liang Zeng, Sebastian Hafner, Sebastian Gerard, Eric Brune, Ritu Yadav, Ali Shibli, et al. PANGAEA: A global and inclusive benchmark for geospatial foundation models. arXiv preprint arXiv:2412.04204, 2024. 7, 8

- [50] Mubashir Noman, Muzammal Naseer, Hisham Cholakkal, Rao Muhammad Anwer, Salman Khan, and Fahad Shahbaz Khan. Rethinking transformers pretraining for multi-spectral satellite imagery. In CVPR,

2024. 2

- [51] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In ECCV, 2016. 2
- [52] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 4
- [53] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. TLMR, 2023. 1, 2, 8, 15
- [54] Luis Miguel Pazos-Out´on, Cristina Nader Vasconcelos, Anton Raichuk, Anurag Arnab, Dan Morris, and Maxim Neumann. Planted: A dataset for planted forest identification from multi-satellite time series. International Geoscience and Remote Sensing Symposium,

2024. 6, 7, 15, 16, 17, 18

- [55] Christopher Phillips, Sujit Roy, Kumar Ankur, and Rahul Ramachandran. HLS foundation burnscars dataset, 2023. 7, 8, 16, 17, 18
- [56] Colorado J Reed, Ritwik Gupta, Shufan Li, Sarah Brockman, Christopher Funk, Brian Clipp, Kurt Keutzer, Salvatore Candido, Matt Uyttendaele, and Trevor Darrell. Scale-MAE: A scale-aware masked autoencoder for multiscale geospatial representation learning. In ICCV, 2023. 1, 2, 3, 8, 9, 15, 17
- [57] Esther Rolf, Jonathan Proctor, Tamma Carleton, Ian Bolliger, Vaishaal Shankar, Miyabi Ishihara, Benjamin Recht, and Solomon Hsiang. A generalizable and accessible approach to machine learning with global satellite imagery. Nature communications, 2021. 15
- [58] Ribana Roscher, Marc Russwurm, Caroline Gevaert, Michael Kampffmeyer, Jefersson A. Dos Santos, Maria Vakalopoulou, Ronny H¨ansch, Stine Hansen, Keiller Nogueira, Jonathan Prexl, and Devis Tuia. Better, not just more: Data-centric machine learning for Earth observation. IEEE Geoscience and Remote Sensing Magazine, 2024. 6
- [59] Depanshu Sani, Sandeep Mahato, Sourabh Saini, Harsh Kumar Agarwal, Charu Chandra Devshali, Saket Anand, Gaurav Arora, and Thiagarajan Jayaraman. SICKLE: A multi-sensor satellite imagery dataset annotated with multiple key cropping parameters. In WACV,

2024. 7, 14, 16, 17, 18

- [60] Srikumar Sastry, Subash Khanal, Aayush Dhakal, Adeel Ahmad, and Nathan Jacobs. TaxaBind: A unified embedding space for ecological applications. In WACV, 2025. 2

- [61] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. In NeurIPS Dataset and benchmark, 2022. 1
- [62] Hochreiter Sepp and Schmidhuber J¨urgen. Long shortterm memory. Supervised sequence labelling with recurrent neural networks, 2012. 16
- [63] Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-Kin Wong, and Wang-Chun Woo. Convolutional LSTM network: A machine learning approach for precipitation nowcasting. In NeurIPS, 2015. 16
- [64] Adam Stewart, Nils Lehmann, Isaac Corley, Yi Wang, Yi-Chia Chang, Nassim Ait Ait Ali Braham, Shradha Sehgal, Caleb Robinson, and Arindam Banerjee. SSSL4EO-l: Datasets and foundation models for Landsat imagery. NeurIPS, 36, 2024. 15
- [65] Daniela Szwarcman, Sujit Roy, Paolo Fraccaro, Thorsteinn El´ı G´ıslason, Benedikt Blumenstiel, Rinki Ghosal, Pedro Henrique de Oliveira, Joao Lucas de Sousa Almeida, Rocco Sedona, Yanghui Kang, et al. Prithvi-EO-2.0: A versatile multi-temporal foundation model for earth observation applications. arXiv preprint arXiv:2412.02732, 2024. 8, 16
- [66] Maofeng Tang, Andrei Cozma, Konstantinos Georgiou, and Hairong Qi. Cross-scale mae: A tale of multiscale exploitation in remote sensing. In NeurIPS, 2024. 2
- [67] Wensi Tang, Guodong Long, Lu Liu, Tianyi Zhou, Michael Blumenstein, and Jing Jiang. Omni-scale CNNs: A simple and effective kernel size configuration for time series classification. In ICLR, 2021. 16
- [68] Michail Tarasiou, Erik Chavez, and Stefanos Zafeiriou. ViTs for SITS: Vision transformers for satellite image time series. In CVPR, 2023. 15
- [69] Gabriel Tseng, Ivan Zvonkov, Mirali Purohit, David Rolnick, and Hannah Kerner. Lightweight, pre-trained transformers for remote sensing timeseries. arXiv preprint arXiv:2304.14065, 2023. 2, 15
- [70] Wei-Hsin Tseng, Ho`ang-Anˆ Lˆe, Alexandre Boulch, S´ebastien Lef`evre, and Dirk Tiede. CROCO: Crossmodal contrastive learning for localization of Earth observation data. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences,

2022. 2

- [71] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz

- Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 16
- [72] Elliot Vincent, Jean Ponce, and Mathieu Aubry. Satellite image time series semantic change detection: Novel architecture and analysis of domain shift. arXiv preprint arXiv:2407.07616, 2024. 16
- [73] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In ICML,

2008. 2, 16

- [74] Yi Wang, Nassim Ait Ali Braham, Zhitong Xiong, Chenying Liu, Conrad M Albrecht, and Xiao Xiang Zhu. SSL4EO-S12: A large-scale multi-modal, multitemporal dataset for self-supervised learning in Earth observation. IEEE Geoscience and Remote Sensing Magazine, 2023. 1
- [75] Giulio Weikmann, Claudia Paris, and Lorenzo Bruzzone. Timesen2crop: A million labeled samples dataset of Sentinel 2 image time series for crop-type classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2021. 7, 16, 17, 18
- [76] Piper Wolters, Favyen Bastani, and Aniruddha Kembhavi. Zooming out on zooming in: Advancing superresolution for remote sensing, 2023. 17, 18
- [77] Zhitong Xiong, Yi Wang, Fahong Zhang, Adam J Stewart, Jo¨elle Hanna, Damian Borth, Ioannis Papoutsis, Bertrand Le Saux, Gustau Camps-Valls, and Xiao Xiang Zhu. Neural plasticity-inspired foundation model for observing the Earth crossing modalities. arXiv preprint arXiv:2403.15356, 2024. 1, 2, 8, 9, 15, 16
- [78] Kun Yi, Yixiao Ge, Xiaotong Li, Shusheng Yang, Dian Li, Jianping Wu, Ying Shan, and Xiaohu Qie. Masked image modeling with denoising contrast. In ICLR,

2023. 2

- [79] Yuan Yuan, Lei Lin, Qingshan Liu, Renlong Hang, and Zeng-Guang Zhou. SITS-Former: A pre-trained spatiospectral-temporal representation model for sentinel2 time series classification. International Journal of Applied Earth Observation and Geoinformation, 2022. 2
- [80] Xiao Xiang Zhu, Jingliang Hu, Chunping Qiu, Yilei Shi, Jian Kang, Lichao Mou, Hossein Bagheri, Matthias H¨aberle, Yuansheng Hua, Rong Huang, et al. So2Sat LCZ42: A benchmark dataset for global local climate zones classification. arXiv preprint arXiv:1912.12171, 2019. 7, 16, 17, 18

## AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities

### Supplementary Material

PASTIS-HD classif (wF1)

SOTA

PASTIS-HD semseg(mIoU)

PLANTED semseg(mIoU)

AnySat

50

65 FLAIR

70

semseg LP (mIoU)

PASTIS-HD classif (mF1)

60

75

40

60

65

55

70

semseg(IoU)

TSatAI-TS classif(wF1)

SICKLE

90

80

74

76

85

75

HLS Burn Scarsemseg LP (mIoU)

chgdet (IoU)BRADD-SITS

58

85

- 91
- 92

90

90

100

TimeSen2Crop classif(OA)

65

classifLP(OA)So2Sat

Sen1Floods11 semseg LP (mIoU)

- Figure A. Overall Performance. We underline external datasets. LP stands for Linear Probing.

In this appendix, we provide detailed results in Sec. A, an extended ablation study in Sec. B, and provide implementation details in Sec. C. Finally, we provide more details on the datasets and experiments of the main paper in Sec. D

#### A. Detailed Results

We provide qualitative illustrations of our predictions and detailed quantitative results for the test sets of GeoPlex.

Qualitative Results. We present qualitative illustrations in Fig. B for four segmentation tasks: PASTIS, FLAIR, SICKLE, and BraDD-S1TS. AnySat predicts precise segmentations that closely follow the extents of buildings, trees, and parcels. Notably, the predictions do not display grid artifacts despite our segmentation head being a simple linear layer applied to each subpatch. This suggests that using subpatches of small sizes (e.g., 4 × 4 pixels for PASTIS and 10 × 10 pixels for FLAIR), combined with larger context through patch embeddings, is an effective strategy for producing smooth and consistent segmentation maps.

Quantitative Results. We provide in Tab. A and Tab. B the detailed performance of AnySat, with and without pretrain-

ing, and an extensive comparison with recent EO models. Pretraining on GeoPlex improves performance for smaller datasets (e.g., TreeSatAI-TS, PASTIS in classification), but this effect is more limited for segmentation datasets (FLAIR, PASTIS in segmentation) or larger ones like PLANTED. We hypothesize that this is due to the quantity of available supervision; for instance, FLAIR has over 20 billion individual labels. In the case of FLAIR, the pretrained model is 0.5 points behind training from scratch, which we attribute to stochastic noise, as our performance on the validation set is on par with training from scratch: 54.7 for pretrained vs. 54.8 from scratch.

#### B. Additional Ablation

We propose an additional experiment to evaluate the impact of one of our design choices.

No Modality or Temporal Masking. In this experiment, we remove the modality and temporal masking for the student encoder during pretraining. This modification results in a slight increase in segmentation performance by +0.4 mIoU but a decrease in classification performance by −0.6 F1 score. These ambiguous results are similar to the effects we observed with naive patch dropping. An advantage of including modality and temporal masking is that it reduces the memory requirements during training by up to 30%. Since our goal is to train a single model on several datasets aimed to be fine-tuned for multiple tasks, we keep a unique configuration and adopt this masking strategy.

#### C. Implementation Details

GeoPlex. See Tab. C for more details on the composition of GeoPlex. GeoPlex is composed of five distinct datasets—TSAI-TS, PASTIS-HD, FLAIR, PLANTED, and S2NAIP-URBAN—which collectively offer a rich combination of data types, including images, time series, and various modalities. These datasets span extensive geographical areas, ranging from 180 km² to over 211,000 km², and provide a wide array of spatial resolutions (from 0.2m to 250m), temporal resolutions (from 1 to 140 time steps), and spectral resolutions (from 3 to 10 bands). The inclusion of multiple satellite and aerial platforms, such as Sentinel-1/2, Landsat 7/8/9, SPOT6/7, and NAIP, ensures a robust and varied training set.

PASTID-HD [7, 27] FLAIR [25] SICKLE [59] BraDD-S1TS [39] Sen1Floods11[12]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

- S1-TS S1-TS S1-TS, first date S1 monodate

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

- S2-TS S2-TS S2-TS S1-TS, last date S2 monodate

[Figure 68]

[Figure 69]

[Figure 70]

VHR 1.5 m VHR 0.2 m LandSat8-TS

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

ground truth

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

1280 m 102.4 m 320 m 480 m 5120 m prediction

- Figure B. Illustration of Results. We represent the inputs, predictions, and ground truth for tiles from four datasets. The colormaps are taken directly from the papers. TS: time series, a single date has been chosen. S1/2 stands for Sentinel-1/2. For PASTIS-HD, white parcels are not annotated (void label).

- Table A. Model Performance on the Test Sets of GeoPlex. For time series, we denote by when a single date has been selected, and when seasonal medians have been concatenated in the channel dimension. AL stands for ALOS-2 and MO for MODIS. LP stands for linear probing

Model Pre-training Modalities TSAI-TS - multilabel classif. VHR S1 S2 wF1 AnySat (ours) GeoPlex ✓ ✓ ✓ 75.1 AnySat (ours) None ✓ ✓ ✓ 72.7 OmniSat [7] TSAI-TS ✓ ✓ ✓ 74.2 DOFA [77] DOFA ✓ 71.6 PSE+LTAE [26] None ✓ ✓ 71.2 PSE + ResNet [7] None ✓ 68.1 ScaleMAE [56] TSAI ✓ 62.5 SatMAE [18] TSAI ✓ 61.5 CROMA [23] TSAI ✓ 61.0 UT&T [25] ImageNet ✓ ✓ ✓ 56.7 MOSAIKS[57] TSAI 56.0 PRESTO [69] PRESTO 46.3 Model Pre-training Modalities PLANTED - classif. S1 S2 LS AL MO maF1 AnySat (ours) GeoPlex ✓ ✓ ✓ ✓ ✓ 61.5 AnySat (ours) None ✓ ✓ ✓ ✓ ✓ 61.2 ViViT [5, 54] None ✓ ✓ 62.2 ViViT [5, 54] None ✓ ✓ ✓ ✓ ✓ 59.3 FLAIR - semantic seg VHR S2 mIoU AnySat (ours) GeoPlex ✓ ✓ 55.1 AnySat (ours) None ✓ ✓ 55.6 UT&T [25] ImageNet ✓ ✓ 56.9 UNet [32] ImageNet ✓ 54.7 UTAE [27] None ✓ 36.1

PASTIS-HD - multilabel classif. VHR S1 S2 maF1 AnySat (ours) GeoPlex ✓ ✓ ✓ 72.8 AnySat (ours) None ✓ ✓ ✓ 65.5 OmniSat [7] PASTIS-HD ✓ ✓ ✓ 69.9 CROMA [23] PASTIS-HD 60.1 DOFA [77] DOFA ✓ 55.7 UT&T [25] ImageNet ✓ ✓ ✓ 53.5 UTAE [27] None ✓ ✓ 46.9 ScaleMAE [56] PASTIS-HD ✓ 42.2 PASTIS-HD - semantic seg VHR S1 S2 OA mIoU AnySat (ours) GeoPlex ✓ ✓ ✓ 85.0 66.5 AnySat (ours) None ✓ ✓ ✓ 84.8 66.3 SkySense [31] SkySense ✓ ✓ ✓ 85.9 UTAE-MM [28] None ✓ ✓ 84.2 66.3 TSViT [68] None ✓ 83.4 65.4 UTAE [27] None ✓ - 63.1

PASTIS-HD - semseg LP VHR S1 S2 mIoU AnySat LP (ours) GeoPlex ✓ ✓ ✓ 42.7 S12-DINO LP [53, 64] foundation ✓ ✓ ✓ 36.2 S12-MoCo LP [33, 64] foundation ✓ ✓ ✓ 34.5 S12-D2V LP [10, 64] foundation ✓ ✓ ✓ 34.3 SpectralGPT [35] foundation ✓ ✓ ✓ 35.4 Prithvi [38] foundation ✓ ✓ ✓ 33.9

- Table B. External Datasets. We evaluate our pretrained model on 4 external datasets, in the fine-tuning or linear probing settings.

stands for single-date observations. We report the number of trainable parameters for probing experiments.

SICKLE [59] L8 S1 S2 mIoU AnySat (fine-tune) ✓ ✓ ✓ 89.3 AnySat (linear 6.1K) ✓ ✓ ✓ 82.0 Unet3d [46, 59] ✓ ✓ ✓ 82.1 UTAE [27, 59] ✓ ✓ ✓ 51.4 BraDD-S1TS [39] S1 mIoU AnySat (fine-tune) ✓ 80.9 AnySat (linear 6.1K) ✓ 78.9 UTAE [27] ✓ 70.7 3D-UNet [46] ✓ 68.1 Conv-LSTM [63] ✓ 63.7 TimeSen2Crop [75] S2 OA AnySat (fine-tune) ✓ 92.2 AnySat (linear 14K) ✓ 70.3 OS-CNN [67, 73] ✓ 81.2 MLP+TAE [26, 72] ✓ 80.9 W.LSTM [14, 62] ✓ 78.2 Transformer [71] ✓ 78.1 MSResNet [19] ✓ 76.3 Sen1Floods11 [12] S1 S2 mIoU AnySat (linear 6.1K) 91.1 CROMA [23] (UperNet 47M) 90.9 CROMA [23] (fine-tune 350M) 90.9 Prithvi [38] (fine-tune 130M) 90.4 Prithvi2 [65] (fine-tune 630M) 90.4 SatlasNet [11] (UperNet 33M) 90.3 Prithvi [38] (UperNet 39M) 88.3

HLS Burn Scar [55] HLS mIoU AnySat (fine-tune) ✓ 90.6 AnySat (linear 3M) ✓ 87.7 Prithvi2 [65] (fine-tune 630M) ✓ 90.5 Prithvi [38] (fine-tune 130M) ✓ 86.9 Prithvi [38] (UperNet 39M) ✓ 83.6 CROMA [23] (UperNet 47M) ✓ 82.4 DOFA [77] (UperNet 39M) ✓ 80.6 So2Sat [80] S1 S2 OA AnySat (linear 29k) 59.1 DOFA [77] (linear) 59.3 CROMA [23] (linear) 49.2 SatMAE [18] (linear) 46.9

Network Architecture. AnySat’s architecture follows the Vision Transformer (ViT) template and has 125M learn-

able parameters, of which 73.6% are modality-agnostic and resolution-adaptive. The components of the model are:

- • Modality Projectors ϕprojm (33M parameters for 11 projectors). These modules are MLPs responsible for projecting the input data of each modality into a common feature space.
- • Spatial Transformer ϕtrans (45M parameters). Composed of three self-attention transformer blocks, this module captures the spatial relationships between subpatches for each modality and patch.
- • Modality Combiner ϕcomb (49M parameters). This module consists of three self-attention blocks followed by a cross-attention block, and merges the representations from different modalities into a unified feature vector for each patch.
- • Predictor ϕpred (29M parameters). Exclusive to the student, this module is a single self-attention block and predicts the teacher’s embeddings for the dropped patches.

Handling MODIS data. In the Planted dataset [54], MODIS observations are included, but their resolution (250 meters) is larger than the entire observed tile (120 meters). We treat these observations as context tokens: we concatenate their ϕpatch embeddings to the |M| · (S/P)2 tokens from all other modalities. We do not add positional encoding, and this token is not included in the contrastive loss.

Optimization Parameters. To better manage our memory usage, we adapt the batch size to the size of the samples of each dataset: TreesatAI-TD: 384, PASTIS-HD: 8, FLAIR: 96, PLANTED: 2048, S2NAIP: 16. We use 8 NVIDIA H100 for experiments on GeoPlex, PLANTED and Pastis-HD , and a smaller cluster of 3 A600 for TreeSatAI-TS and FLAIR.

Beyond the changes above, all optimization parameters are shared across all datasets. We used the AdamW [44] optimizer with a learning rate of 5 × 10−5 for all our experiments (pretraining and fine-tuning). We used a LinearWarmupCosineAnnealingLR [1] for classification and ReduceLROnPlateau [2] scheduler for pretraining and segmentation.

We set he contrastive temperature γ to 0.1 to n Eq. X. We used an EMA decay of 0.996. All other hyperparameters are shared with original JEPA implementation.

Position Encodings. We describe here our scale-adaptive positional encoding which allows us to use the same encoders for different resolutions, scales, and patch size. The input tokens to the modality combiner ϕcomb correspond to patches of size P ×P meters, while those to the spatial transformer ϕtrans represent subpatches of size (Rmδm)×(Rmδm) meters. Here, Rm varies per sensor modality m, and P is randomly chosen for each batch during training. To train a

- Table C. Considered Datasets. We present the detailed composition of GeoPlex, the collection of datasets used for self-supervised training, and our external evaluation datasets. For each dataset, we consider a set of acceptable patch sizes. img: img, t.s.: time series: t.s. S1/2: Sentinel-1/2. † upsampled from original acquisition resolution.

Sample Size (S)

Resolution Patch Size (P) Spatial (R) Temporal (T) Spectral (C) GeoPlex

Dataset Extent

Modalities

Aerial VHR 0.20m 1 4

50k × (1img + 2t.s.) 180 km² - 4.7 GPix

S = 60m P ∈ {10, 20, 30}m

- S1 10m 10-70 3
- S2 10m 10-70 10

TSAI-TS [3, 7]

SPOT6/7 1m† 1 4

2433 × (1img + 2t.s.) 3986 km² - 7.5 GPix

S = 1280m P ∈ {40, 80, 160}m

- S1 10m 140 3
- S2 10m 38-61 10

- PASTIS-HD [7, 27]

78k × (1img + 1t.s.) 815 km² - 20 GPix

S = 102.4m Aerial VHR 0.2m 1 5 P ∈ {10, 20, 50}m S2 10m 20-114 10

FLAIR [25]

S2 10m 8 10 S1 10m 8 3

1.3M × (5t.s.) 33,120 km² - 3.0 GPix

S = 120m P ∈ {30, 60}m

Planted [54]

Landsat 7 30m 20 3 ALOS-2 30m 4 3 MODIS 250m 60 7

NAIP 1.25m 1 4 S2NAIP- S2 10m 16-32 10 URBAN [4, 76] S1 10m 2-8 3

515k × (1img + 3t.s.) 211,063 km² - 136 GPix

S = 640m P ∈ {40, 80, 160}m

Landsat 8/9 10m† 4 8 External datasets

13k × (1t.s.) 2,995 km² - 1.2 GPix

S = 480m P = 10 m

BraDD-S1TS [39]

S1 10m 20-66 10

35k × (2t.s.) 3,584 km² - 3.6 GPix

S = 320m P = 10m

S2 10m 13-148 10 Landsat 8/9 10m† 8-34 8

Sickle [59]

1.2M × (1t.s.) 120 km² - 35 MPix

S = 10m P = 10m

TimeSen2Crop [75]

S2 10m 29 10

4.8k × (2img) 125,829 km² - 2.6 GPix

S = 5120m P = 80m

S2 10m 1 10 S1 10m 1 3

Sen1floods11 [12]

400k × (2 img) 41,029 km² - 82 GPix

S = 320m P = 10m

S2 10m 1 10 S1 10m 1 3

So2Sat [80]

804 × (1t.s.) 188,208 km² - 211 MPix

S = 15300m P = 240m

HLS 30m 1 6

HLS Burn Scar [55]

single scale-aware model capable of handling varying resolutions, we employ a scale-adaptive positional encoding inspired by Scale-MAE [56].

We use the same positional encodings in ϕcomb and ϕtrans. We first describe the positional encoding of a token by ϕcomb. We denote by posx the index of the token’s patch within its tile along the x-axis; similarly, posy along the y-axis. If the embeddings of the token have a dimension D, the positional encodings µx(posx,i) and equivalently µy(posy,i) are of size D/2. For i ∈ [0,D/2[ we have:

posx 10000Ei

g G

π 2

mod(i,2) , (A)

µx(posx,i) = sin

+

where g = P is the size in meter of the patch considered unit: patch of size for ϕcomb, and G is a reference length that we set to one meter. We compute µy(posy,i) similarly, and the positional encoding is the channelwise concatenation of both vectors. The positional encoding is directly added to the embeddings.

For ϕtrans, we define the positional encoding of each subpatch within its patch with the same formula, but set g to g = Rmδm, the size of the subpatch in meter.

#### D. Datasets and Tasks

Here, we provide more details about the datasets used to train and evaluate AnySat and their associated tasks. See Tab. C for an overview of the datasets used in GeoPlex.

TreeSatAI-TS [3, 7]: This multimodal dataset is designed for tree species identification and consists of 50,381 tiles, each covering an area of 60×60 meters, with multi-label annotations across 20 classes. All data were collected in Germany. The dataset includes Very High Resolution (VHR) images at 0.2 m with a NIR band, Sentinel-2 time series, and Sentinel-1 time series.

- PASTIS-HD [7, 28]: This crop mapping dataset supports classification, semantic segmentation, and panoptic segmentation. Each agricultural parcel is delineated at a resolution of 10 m and annotated across 18 crop types. The dataset contains 2,433 tiles with an extent of 1,280×1,280 m, including Sentinel-2 time series, Sentinel-1 time series (we use only the ascending orbit), and SPOT6 VHR imagery at 1.5 m resolution.

FLAIR [25]: This dataset combines VHR aerial imagery at a 0.2 m resolution with Sentinel-2 time series data and comprises 77,762 tiles acquired across metropolitan France. The VHR images include five channels: RGB, near-infrared, and a normalized digital surface model derived by photogrammetry. Each VHR pixel is annotated with one of 13 land cover classes.

PLANTED [54]: The PLANTED dataset is specifically designed for tree species identification and features 1,346,662 tiles of planted forest across the world. Each tile is associated with one of 40 distinct classes. This dataset integrates imagery from five different satellites with various resolutions: Sentinel-2 (10 m), Landsat-7 (30 m), MODIS (250 m), as well as radar time series from Sentinel-1 (10 m) and ALOS-2 (30 m). The time series are temporally aggregated at various intervals—seasonally, monthly, or yearly.

S2Naip-Urban [4, 76]: This dataset includes images captured at the same locations as the S2NAIP-Urban superresolution dataset [76], which is a subset of the extensive S2NAIP [4] dataset focused on urban areas. This split comprises 515,270 tiles, featuring imagery from NAIP at a 1.25 m resolution, Sentinel-2 and Sentinel-1 time series,

and Landsat-8/9 data rescaled to a 10 m resolution. We use this dataset for pretraining only because there are no official labels and evaluations.

BraDD-S1TS [39]: BraDD-S1TS (Brazilian Deforestation Detection) is a change detection dataset comprising

- Sentinel-1 time series of the Amazon rainforest, aiming to segment deforested areas. It includes 13,234 tiles covering regions with varying deforestation rates, providing pixelwise binary annotations for deforestation events occurring between the time series’ first and last radar image.

Sickle [59]: SICKLE is a multimodal crop mapping dataset from India containing 34,848 tiles with Sentinel1, Sentinel-2, and Landsat-8 time series. We use the paddy / non-paddy culture binary semantic segmentation task. As the test set has not been released by the authors, we perform our experiments on the validation set.

TimeSen2Crop [75]: TimeSen2Crop is a crop mapping dataset consisting of 1,212,224 single-pixel Sentinel-2 time series, a configuration not present in GeoPlex. It includes data from Slovenia with annotations for 16 different crop types.

Sen1floods11 [12]: Sen1Floods11 is a flood segmentation dataset featuring 4,831 pairs of Sentinel-1 and Sentinel-2 images, each annotated with dense flooded/not-flooded labels. The dataset spans diverse global regions, with each tile covering a 5120 × 5120 m area ( 2600 hectares) and containing a single acquisition date per sensor.

So2Sat [80]: So2Sat is a local climate zone classification dataset containing co-registered single-date Sentinel-1 and

- Sentinel-2 imagery across multiple cities worldwide. It comprises 400,673 image patches, each annotated with one of 17 local climate zone classes according to the LCZ scheme. An image represents a zone of size 320 × 320 m. So2Sat specifically targets urban morphology classification tasks for sustainable urban planning and climate studies.

HLS Burn Scar [55]: HLS Burn Scar is designed for postfire burn scar detection using Harmonized Landsat-Sentinel (HLS) imagery. It contains 804 tiles covering a 15.3 × 15.3 km area 23400 hectares) at 30m resolution and covering multiple wildfire events across diverse ecosystems in the United States.

