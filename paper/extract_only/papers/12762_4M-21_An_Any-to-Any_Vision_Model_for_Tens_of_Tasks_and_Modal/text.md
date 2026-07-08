# arXiv:2406.09406v2[cs.CV]14Jun2024

## 4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities

Roman Bachmann1†∗ O˘guzhan Fatih Kar1∗ David Mizrahi2†∗ Ali Garjani1 Mingfei Gao2 David Griffiths2 Jiaming Hu2 Afshin Dehghan2 Amir Zamir1

1Swiss Federal Institute of Technology Lausanne (EPFL) 2Apple https://4m.epfl.ch

RGB modalities Edge modalities

RGB

Color palette

SAM edges

Canny edges

|[Figure 1]|
|---|

[Figure 2]

[Figure 3]

[Figure 4]

Geometric modalities

Text modalities

Surface normals

3D human poses

T5-XXL embeddings

Depth

Caption

Web text

Any-to-any model

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Albany International Airport serves as the major air center for the Capital Region, Northeastern ...

Getting ready for my flight!

Transformer encoder

Transformer decoder

Semantic modalities

Metadata modalities

Semantic metadata

Geometric metadata

Image metadata

SAM instances

Semantic segmentation

Bounding boxes

[Figure 9]

[Figure 10]

[Figure 11]

# Humans: 7 # Instances: 12 Objectness: 40% Walkability: 40% Clutter score: 75% …

Orig. res.: 512x512 Colorfulness: 35% Contrast: 45% Brightness: 60% Saturation: 40% …

Geometric complexity: 55% Occlusion score: 25% …

Global feature modalities

Feature map modalities

DINOv2 features (dense)

DINOv2 features (global)

ImageBind features (global)

ImageBind features (dense)

CLIP features (dense)

[Figure 12]

[Figure 13]

[Figure 14]

Figure 1: We demonstrate training a single model on tens of highly diverse modalities without a loss in performance compared to specialized single/few task models. The modalities are mapped to discrete tokens using modality-specific tokenizers. The model can generate any of the modalities from any subset of them.

### Abstract

Current multimodal and multitask foundation models, like 4M [62] or UnifiedIO [59, 58], show promising results. However, their out-of-the-box abilities to accept diverse inputs and perform diverse tasks are limited by the (usually small) number of modalities and tasks they are trained on. In this paper, we develop a single any-to-any model trained on tens of highly diverse modalities and by performing co-training on large-scale multimodal datasets and text corpora. This includes training on images and text along with several semantic and geometric modalities, feature maps from recent state of the art models like DINOv2 and ImageBind, pseudo labels of specialist models like SAM and 4DHumans, and a range of new modalities that allow for novel ways to interact with the model and steer the generation, for example, image metadata or color palettes. A crucial step in this process is performing discrete tokenization on various modalities, whether they are image-like, neural network feature maps, vectors, structured data like instance segmentation or human poses, or data that can be represented as text.

Through this, we show the possibility of training one model to solve at least 3x more tasks/modalities than existing models and doing so without a loss in performance. In addition, this enables more fine-grained and controllable multimodal generation capabilities and allows studying the distillation of models trained on diverse data and objectives into one unified model. We scale the training to a three billion parameter and different datasets. The multimodal models and training code are open sourced at https://4m.epfl.ch.

*Equal contribution & corresponding authors. Randomized order. †Work partially done while at EPFL and Apple.

Preprint.

### 1 Introduction

Having a single neural network to handle a wide and varied range of tasks and modalities has been a longstanding goal. Such a model, especially when capable of any-to-any predictions, brings notable advantages, such as test-time computational efficiency, model size, and enabling modality fusion.

However, multitask learning has commonly faced significant challenges. For example, the training often suffers from negative transfer, leads to reduction in performance compared to single-task models, and typically requires careful strategies for balancing losses or gradients [46, 93, 82, 94, 31]. Moreover, training a single network on tasks and modalities that vary greatly in terms of dimensionality, data type, and value ranges presents additional complexities†. Recent notable efforts in the space of multimodal and multitask training, such as Pix2Seq [16, 17], OFA [88], 4M [62], or Unified-IO [59, 58] have made significant strides in unifying the representation space for conceptually different inputs and targets. A large part of their success can be attributed to transforming different modalities into a common representation, namely sequences of discrete tokens, and training relatively standard Transformer architectures on them. While these works show promising results, they are typically trained on a small set of modalities. This raises the question if increasing the set of tasks/modalities the models can solve will lead to a degradation of performance.

We build upon the multimodal masking pre-training scheme [62] and increase its capabilities by training on tens of highly diverse modalities. Concretely, we add SAM segments [47], 3D human poses and shapes from 4DHumans [35], canny edges extracted from RGB and SAM instances, color palettes, multiple types of image, semantic and geometric metadata, as well as T5-XXL [68] text embeddings, in addition to 7 more common modalities. On top of that, we include dense feature maps of the recent state of the art models DINOv2 [65] and ImageBind [33], as well as their global embedding vectors to enable multimodal retrieval abilities. Please see fig. 1 for an overview.

We are able to train a single unified model on diverse modalities by encoding them with modalityspecific discrete tokenizers (see fig. 3). For image-like modalities, e.g. RGB or edges, we train ViT-based [24] VQ-VAE [63] tokenizers to map the inputs into a small grid of discrete tokens. For modalities like 3D human poses or image embeddings, we train MLP-based discrete VAEs to compress them into a small set of discrete tokens. All other modalities that can be mapped to a text representation, such as captions or metadata, are encoded using a WordPiece tokenizer [23].

The resulting model demonstrates the possibility of training a single model on a large number of diverse modalities/tasks without any degradation in performance and significantly expands the out-ofthe-box capabilities compared to existing models. Adding all these modalities enables new potential for multimodal interaction, such as retrieval from and across multiple modalities, or highly steerable generation of any of the training modalities, all by a single model.

In short, we expand the capabilities of existing models across several key axes:

- • Modalities: Increase from 7 in the existing best any-to-any models [62] to 21 diverse modalities, enabling new capabilities like cross-modal retrieval, controllable generation, and strong out-of-the-box performance. This is one of the first times in the vision community that a single model can solve tens of diverse tasks in an any-to-any manner (see fig. 2), without sacrificing performance and especially do so without any of the conventional multitask learning difficulties [73, 46, 93, 82, 94, 31].
- • Diversity: Add support for more structured data, such as human poses, SAM instances, metadata, and color palettes for controllable generation.
- • Tokenization: Investigate discrete tokenization of diverse modalities such as global image embeddings, human poses, and semantic instances using modality-specific approaches.
- • Scale: Scale the model size to 3B parameters and dataset to 0.5B samples using [11].
- • Co-Training: Demonstrate co-training on vision and language modeling simultaneously.

†Modality vs task: “Modalities” usually denote the inputs to a model (e.g. sensory signals), and “tasks” usually denote the outputs (e.g. semantics). The adopted architecture in multimodal masked modeling enables a symmetric input-output structure, thus modalities and tasks are used interchangeably in this paper.

###### Input Predictions

Caption Bounding boxes

Surface Semantic Depth normals

Human poses

DINOv2 ImageBind Metadata Canny SAM SAM Color instances edges edges palette

RGB

CLIP

segmentation

|[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

|[Figure 18]|
|---|

|[Figure 19]|
|---|

|[Figure 20]|
|---|

|[Figure 21]|
|---|

|[Figure 22]|
|---|

|[Figure 23]|
|---|

|[Figure 24]|
|---|

|[Figure 25]|
|---|

|[Figure 26]|
|---|

|walkability: 0.08 g. complexity: 0.04 clutter score: 69 ...|
|---|

|[Figure 27]|
|---|

the view from the top of <person> lookout

RGB

|[Figure 28]|
|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

|[Figure 32]|
|---|

|[Figure 33]|
|---|

|[Figure 34]|
|---|

|[Figure 35]|
|---|

|[Figure 36]|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

|walkability: 0.16 g. complexity: 0.15 clutter score: 45 ...|
|---|

|[Figure 40]|
|---|

CaptionDepthboxes Semantic

<person> showing mountains as well as a family

|[Figure 41]|
|---|

|[Figure 42]|
|---|

|[Figure 43]|
|---|

|[Figure 44]|
|---|

|[Figure 45]|
|---|

|[Figure 46]|
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

|[Figure 49]|
|---|

|[Figure 50]|
|---|

|[Figure 51]|
|---|

|[Figure 52]|
|---|

|walkability: 0.18 g. complexity: 0.17 clutter score: 74 ...|
|---|

|[Figure 53]|
|---|

Bounding

the view from the pier at the entrance of the sea

|[Figure 54]|
|---|

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

|walkability: 0.00 g. complexity: 0.17 clutter score: 32 ...|
|---|

|[Figure 66]|
|---|

segmentation

the view from the top of the hill

|[Figure 67]|
|---|

|[Figure 68]|
|---|

|[Figure 69]|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

|[Figure 75]|
|---|

|[Figure 76]|
|---|

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|walkability: 0.14 g. complexity: 0.04 clutter score: 55 ...|
|---|

|[Figure 79]|
|---|

the view from the top of the

mountain

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|walkability: 0.26<br><br>g. complexity: 0.11 clutter score: 74 ...|
|---|

|[Figure 92]|
|---|

SAMImageBindCLIPnormals

the view from the top of the lake wanaka lookout

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

|[Figure 100]|
|---|

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|walkability: 0.18 g. complexity: 0.04 clutter score: 65 ...|
|---|

|[Figure 105]|
|---|

Surface

the road to the mountains

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

|[Figure 115]|
|---|

|[Figure 116]|
|---|

|walkability: 0.12<br><br>g. complexity: 0.43 clutter score: 71 ...|
|---|

|[Figure 117]|
|---|

|[Figure 118]|
|---|

Human

poses

the streets of lisbon, portugal

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

|[Figure 124]|
|---|

|[Figure 125]|
|---|

|[Figure 126]|
|---|

|[Figure 127]|
|---|

|[Figure 128]|
|---|

|[Figure 129]|
|---|

|walkability: 0.32 g. complexity: 0.01 clutter score: 75 ...|
|---|

|[Figure 130]|
|---|

|[Figure 131]|
|---|

DINOv2

the view from the top of the

mountain

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]|
|---|

|[Figure 136]|
|---|

|[Figure 137]|
|---|

|[Figure 138]|
|---|

|[Figure 139]|
|---|

|[Figure 140]|
|---|

|[Figure 141]|
|---|

|[Figure 142]|
|---|

|walkability: 0.48<br><br>g. complexity: 0.01 clutter score: 56 ...|
|---|

|[Figure 143]|
|---|

|[Figure 144]|
|---|

the art of the last of us part 2

|walkability: 0.06 g. complexity: 0.03 clutter score: 44 ...|
|---|

|[Figure 145]|
|---|

|[Figure 146]|
|---|

|[Figure 147]|
|---|

|[Figure 148]|
|---|

|[Figure 149]|
|---|

|[Figure 150]|
|---|

|[Figure 151]|
|---|

|[Figure 152]|
|---|

|[Figure 153]|
|---|

|[Figure 154]|
|---|

|[Figure 155]|
|---|

|[Figure 156]|
|---|

|[Figure 157]|
|---|

CannyMetadata

the blue lagoon

|[Figure 158]|
|---|

|[Figure 159]|
|---|

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

|[Figure 165]|
|---|

|[Figure 166]|
|---|

|walkability: 0.10<br><br>g. complexity: 0.06 clutter score: 75 ...|
|---|

|[Figure 167]|
|---|

|[Figure 168]|
|---|

|[Figure 169]|
|---|

|[Figure 170]|
|---|

edges

the turquoise lakes of the canadian rockies

|[Figure 171]|
|---|

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

|[Figure 177]|
|---|

|[Figure 178]|
|---|

|[Figure 179]|
|---|

|[Figure 180]|
|---|

|[Figure 181]|
|---|

|[Figure 182]|
|---|

|walkability: 0.06 g. complexity: 0.11 clutter score: 69 ...|
|---|

|[Figure 183]|
|---|

edges SAM

the view from the summit of mount rundle

|[Figure 184]|
|---|

|[Figure 185]|
|---|

|[Figure 186]|
|---|

|[Figure 187]|
|---|

|[Figure 188]|
|---|

|[Figure 189]|
|---|

|[Figure 190]|
|---|

|[Figure 191]|
|---|

|[Figure 192]|
|---|

|[Figure 193]|
|---|

|[Figure 194]|
|---|

|[Figure 195]|
|---|

|walkability: 0.16<br><br>g. complexity: 0.18 clutter score: 69 ...|
|---|

|[Figure 196]|
|---|

instances

the hotel at loch lomond

|[Figure 197]|
|---|

|[Figure 198]|
|---|

|[Figure 199]|
|---|

|[Figure 200]|
|---|

|[Figure 201]|
|---|

|[Figure 202]|
|---|

|[Figure 203]|
|---|

|[Figure 204]|
|---|

|[Figure 205]|
|---|

|[Figure 206]|
|---|

|[Figure 207]|
|---|

|walkability: 0.04<br><br>g. complexity: 0.11 clutter score: 36 ...|
|---|

|[Figure 208]|
|---|

|[Figure 209]|
|---|

palette

the three hills of the ancient city of te anau, new zealand

Color

- Figure 2: One-to-all generation. 4M-21 can generate all modalities from any given input modality and can benefit from chained generation [62]. Notice the high consistency among the predictions of all modalities for one input. Each row starts from a different modality coming from the same scene. Highlighted in green are new input/output pairs that 4M [62] cannot predict nor accept as input. Note that, while this figure shows predictions from a single input, 4M-21 can generate any modality from any subset of all modalities.

### 2 Method

We adopt the 4M pre-training scheme [62] as it has been shown to be a versatile approach that can be efficiently scaled to a diverse set of modalities. We keep the architecture and the multimodal masked training objective the same, but expand upon the model and dataset size, the types and number of modalities with which we train the model, and train jointly on multiple datasets. All modalities are first transformed into sequences of discrete tokens using modality-specific tokenizers (See fig. 3). During training, random subsets of these tokens are selected from all modalities as inputs and targets, and the objective is to predict one subset from the other. We rely on pseudo labeling to create a large pre-training dataset with multiple aligned modalities.

#### 2.1 Modalities

We train on a large and diverse set of modalities that we group into the following categories: RGB, geometric, semantic, edges, feature maps, metadata, and text modalities. Below we provide a summary of them (See fig. 1 and appendices D and E for details, and fig. 2 for generation examples).

RGB: We include both tokenized and pixel versions of RGB images to facilitate transfer learning. We also extracted their color palettes using PyPalette [2], at varying number of colors. This enables us to perform conditional generation using desired colors for better artistic control.

Geometric modalities: These contain surface normals, depth, and 3D human poses & shape which provide important information about the scene geometry. For the first two, we used Omnidata models from [26, 44] for pseudo labeling due to their strong generalization performance. For 3D human poses and shape, we leverage a recent state-of-the-art model, 4D-Humans [35].

Semantic modalities: We include semantic segmentation and bounding boxes to capture the scene semantics and leverage Mask2Former [19] and ViTDet [54] models for pseudo labeling. Next to these, we also incorporated pseudo labels extracted from Segment Anything Model [47] (SAM) as SAM instances for its strong object representation.

Edges: As recent generative methods such as ControlNet [98] showed, edges carry important information about the scene layout and semantics that are also useful for conditioning, abstraction, and sketching. We consider two types of edges, specifically Canny edges and SAM edges. The former is extracted from the RGB images with OpenCV [1]. As Canny edges may contain low-level information, e.g. shading edges, we also include edges extracted from SAM instances to get a more semantic boundary map. We tokenize Canny and SAM edges with a shared tokenizer.

Feature maps: We extract embeddings from CLIP [67], DINOv2 [65] and ImageBind [33] as they demonstrated strong transfer learning and retrieval capabilities. Previously, tokenized CLIP features were shown to be an effective target for masked image modelling [89, 62] that enables distilling a useful semantic representation of the scene. We follow a similar approach and tokenize the feature maps from pre-trained CLIP-B16, DINOv2-B14 and ImageBind-H14 models. We also included the global embeddings of DINOv2 and ImageBind models and tokenized them separately.

Metadata: We extract several useful pieces of information from the RGB images and other modalities, that can be categorized into semantic metadata, geometric metadata, and image processing metadata. For this, we use functionalities from Pillow [3] OpenCV [1], and Omnidata [26].

The following semantic metadata are extracted from bounding boxes, poses, and segmentation maps:

- • Crowdedness score: number of humans (extracted from 4DHumans instances)
- • SAM clutter score: number of SAM instances
- • COCO clutter score: number of COCO [55] instances
- • COCO instance diversity: number of unique COCO instance classes
- • Objectness score: % of pixels that belong to countable COCO semantic classes
- • Walkability score: % of pixels belonging to walkable COCO semantic classes such as ‘road’
- • Semantic diversity: number of unique COCO semantic classes
- • Caption length: length of the caption in characters, words, and sentences

These are aimed to capture the semantic regularities of the scene at a more holistic level as opposed to pixel-based representations.

Similarly, geometric metadata captures the scene geometry more globally. They are extracted from surface normals and depth maps:

- • Geometric complexity: angular variance of surface normals
- • Occlusion score: % of occlusion edges over a fixed threshold

Finally, image processing metadata contains several aspects of images such as original image height and width before cropping, which can be used as conditioning to generate higher quality images [66], brightness, contrast, saturation, entropy, and colorfulness [37]. Similar to color palette, these help with encoding low-level image representations into the model and enable more steerable generation.

Text: Large language models (LLMs) trained on large text corpora learn strong representations as shown by several works [23, 68, 85, 64]. We include captions from CC12M [15] and COYO700M [11] datasets, as well as web text from C4 [68] for language modeling. Next, we employ both a standard WordPiece [23] tokenizer for captions as [62] as well as caption embeddings obtained from a T5-XXL [68] encoder to capture better text representations, which have been shown to improve text-to-image generation fidelity [76, 13] (See fig. 4).

###### Spatial discrete VAE with diffusion decoder: RGB, normal, depth, edges

Spatial discrete VAE: Segmentation, CLIP, DINOv2, ImageBind, SAM inst.

Noised image

|[Figure 210]|
|---|

|[Figure 211]|
|---|

|[Figure 212]|
|---|

[Figure 213]

1 2 3 4 5 6 7 8 9

1 2 3 4 5 6 7 8 9

|VQ|
|---|

|VQ|
|---|

ViT encoder

Diffusion decoder

ViT encoder

ViT decoder

VQ-VAE quantization loss Diffusion loss

VQ-VAE quantization loss Reconstruction loss

MLP discrete VAE: Human poses, DINOv2 & ImageBind global tokens

Sequence tokenizer: Text, bounding boxes, metadata, color palette

|[Figure 214]|
|---|

|[Figure 215]|
|---|

A B C D E F G

In bamboo thickets, A red panda's gentle gaze, Autumn's calm embrace.

In bamboo thickets, A red panda's gentle gaze, Autumn's calm embrace.

|Memcodes|
|---|

MLP encoder MLP decoder

1 2 3 WordPiece WordPiece

...

Reconstruction loss

- Figure 3: Tokenization overview. We employ suitable tokenization schemes for different modalities based on their format and performance. For image-like modalities and feature maps, we use spatial VQ-VAEs [63] with optional diffusion decoders for detail rich modalities like RGB. For non-spatial modalities like global tokens or parameterized poses, we compress them to a fixed number of discrete tokens using Memcodes [60] with MLP encoders and decoders. All sequence modalities are encoded as text using WordPiece [23]. The shown examples are real tokenizer reconstructions. Notice the low reconstruction error. See appendix D for more details.

#### 2.2 Tokenization

Tokenization consists of converting modalities and tasks into sequences or sets of discrete tokens, thereby unifying their representation space. This is critical for training large multimodal models as it confers the following key benefits: 1) It enables training multimodal and multitask models with a single pre-training objective. After tokenization, all tasks are formulated as a per-token classification problem using the cross-entropy loss. This improves training stability, enables full parameter sharing, and removes the need for task-specific heads, loss functions, and loss balancing. 2) It makes generative tasks more tractable by allowing the model to iteratively predict tokens, either autoregressively [69, 91] or through progressive unmasking [14, 13]. 3) It reduces computational complexity by compressing dense modalities like images into a sparse sequence of tokens. This decreases memory and compute requirements, which is crucial when scaling up to larger dataset and model sizes.

We use different tokenization approaches to discretize modalities with different characteristics. See fig. 3 for an overview. To summarize, we mainly use three different types of tokenizers, as explained below. Please see appendices D and H for more details and insights on tokenizer design choices.

ViT tokenizer (with optional diffusion decoder): We trained modality-specific ViT [24] based VQ-VAE [63] tokenizers for image-like modalities such as edges and feature maps. The resulting tokens form a small grid of size 14 × 14 or 16 × 16, according to the pseudo-labeler patch size. The edge tokenizers use a diffusion decoder [78, 62] to get visually more plausible reconstructions.

MLP tokenizer: For human poses and global embeddings from DINOv2 and ImageBind, we use Bottleneck MLP [6] based discrete VAEs with Memcodes quantization [60] to tokenize them into a small number of tokens, e.g. 16.

Text tokenizer: We leverage a WordPiece [23] tokenizer which is used to encode not only text, but also other modalities such as bounding boxes, color palettes and metadata using a shared set of special tokens to encode their type and values (See appendix D.6 for details).

#### 2.3 Training details

Datasets: We perform the training in two stages, namely a 4M pre-training stage on a significantly larger image dataset, followed by a fine-tuning phase on a smaller dataset containing a larger number of modalities. Since the 4M-XL model showed signs of overfitting on sequence modalities when trained on CC12M [15], we re-trained the models on COYO700M [11], containing 50 times more samples. COYO700M was pseudo labeled with the same modalities used for 4M. To cut down on pseudo labeling cost when expanding the number of modalities, we decided to pseudo label CC12M instead of COYO700M, and fine-tune the models with both new and old modalities. To avoid overfitting the larger models, we co-train them with samples from COYO700M. In addition to the previously mentioned multimodal datasets, we also included the C4 [68] text corpus in training. We perform the training by randomly sampling elements of each batch from any of these datasets, given

Fine-grained multimodal conditioning control

Improved text understanding capabilities

Caption input: a metallic blue sphere to the left of a yellow box made of felt Caption input: a blue semi-truck and its trailer jumping over a row of motorcycles

Caption input: two football players warming up on the pitch

Caption input: a picture of two astronauts in a lush jungle

Caption input: a painting of two greek philosophers walking on an old street

Caption input: a painting of two clowns walking on the street with skyscrapers

|[Figure 216]|
|---|

|[Figure 217]|
|---|

|[Figure 218]|
|---|

|[Figure 219]|
|---|

|[Figure 220]|
|---|

|[Figure 221]|
|---|

|[Figure 222]|
|---|

|[Figure 223]|
|---|

|[Figure 224]|
|---|

|[Figure 225]|
|---|

|[Figure 226]|
|---|

|[Figure 227]|
|---|

|[Figure 228]|
|---|

|[Figure 229]|
|---|

|[Figure 230]|
|---|

|[Figure 231]|
|---|

|[Figure 232]|
|---|

|[Figure 233]|
|---|

|[Figure 234]|
|---|

|[Figure 235]|
|---|

|[Figure 236]|
|---|

|[Figure 237]|
|---|

|[Figure 238]|
|---|

|[Figure 239]|
|---|

|[Figure 240]|
|---|

|[Figure 241]|
|---|

|[Figure 242]|
|---|

|[Figure 243]|
|---|

Human pose input:

[Figure 244]

4M-7 (from caption) 4M-21 (from caption) 4M-21 (from T5-XXL emb.)

4M-7 (from caption) 4M-21 (from caption) 4M-21 (from T5-XXL emb.)

Caption input: a minimalist sketch of two stick figures

Caption input: a sketch of business people walking in the corridor of a modern office building

Caption input: an oil painting of two shepherds on a mountain meadow

Caption input: a colorful painting of bride and groom walking down the aisle

Caption input: a black background with a large yellow circle and a small red square Caption input: a green pepper to the left of a red pepper

|[Figure 245]|
|---|

|[Figure 246]|
|---|

|[Figure 247]|
|---|

|[Figure 248]|
|---|

|[Figure 249]|
|---|

|[Figure 250]|
|---|

|[Figure 251]|
|---|

|[Figure 252]|
|---|

|[Figure 253]|
|---|

|[Figure 254]|
|---|

|[Figure 255]|
|---|

|[Figure 256]|
|---|

|[Figure 257]|
|---|

|[Figure 258]|
|---|

|[Figure 259]|
|---|

|[Figure 260]|
|---|

|[Figure 261]|
|---|

|[Figure 262]|
|---|

|[Figure 263]|
|---|

|[Figure 264]|
|---|

|[Figure 265]|
|---|

|[Figure 266]|
|---|

|[Figure 267]|
|---|

|[Figure 268]|
|---|

|[Figure 269]|
|---|

|[Figure 270]|
|---|

|[Figure 271]|
|---|

|[Figure 272]|
|---|

4M-7 (from caption) 4M-21 (from caption) 4M-21 (from T5-XXL emb.)

4M-7 (from caption) 4M-21 (from caption) 4M-21 (from T5-XXL emb.)

Probing with grounded generation

Steerable multimodal data generation from metadata

Walkability # humans # SAM

Occlusion score

Original resolution

Metadata input:

Polygon input RGB generation Caption input

Contrast 10%

RGB generation

instances

|[Figure 273]|
|---|

|[Figure 274]|
|---|

|[Figure 275]|
|---|

1

3

10%

64x64

15%

|[Figure 276]|
|---|

a winter ride with family

|[Figure 277]|
|---|

|[Figure 278]|
|---|

|[Figure 279]|
|---|

|[Figure 280]|
|---|

|[Figure 281]|
|---|

|[Figure 282]|
|---|

|[Figure 283]|
|---|

|[Figure 284]|
|---|

SAM edges input

|[Figure 285]|
|---|

|[Figure 286]|
|---|

|[Figure 287]|
|---|

Caption input

40%

4

30

25%

256x256

60%

|[Figure 288]|
|---|

Caption input:

|[Figure 289]|
|---|

|[Figure 290]|
|---|

|[Figure 291]|
|---|

|[Figure 292]|
|---|

|[Figure 293]|
|---|

|[Figure 294]|
|---|

|[Figure 295]|
|---|

|[Figure 296]|
|---|

|[Figure 297]|
|---|

riding a blue car at sunset

a bowl of soup on a wooden table

a picture of a Swiss mountain scene

80%

20

200

85%

2048x2048

90%

|[Figure 298]|
|---|

|[Figure 299]|
|---|

|[Figure 300]|
|---|

|[Figure 301]|
|---|

|[Figure 302]|
|---|

|[Figure 303]|
|---|

|[Figure 304]|
|---|

|[Figure 305]|
|---|

|[Figure 306]|
|---|

|[Figure 307]|
|---|

|[Figure 308]|
|---|

|[Figure 309]|
|---|

a historical photo of a classic car

- Figure 4: Fine-grained & steerable multimodal generation. Top left: 4M-21 can generate variants of images that are grounded in any input modality, here human poses. Bottom left: This enables us to perform multimodal edits (e.g. editing the shape of a polygon or grounding generation with edges) and probe the learned representation. For example, by only changing the shape of the ellipse, 4M-21 renders the bowl from different angles. Top right: By pre-training on 21 types of modalities, including T5-XXL embeddings, and co-training with language modeling on a large text corpus, we show improved text understanding capabilities (even when the input is captions instead of language model embeddings). Bottom right: Compared to generating images from captions only, metadata provides a more direct and steerable way of controlling the multimodal data generation process, enabling exciting further research into generative dataset design.

a pre-determined set of sampling weights, and perform language modeling on them. Exact details on the training mixture are given in appendix E.2.

Architecture: We adopt 4M’s encoder-decoder based transformer architecture with additional modality embeddings to accommodate new modalities. Similar to 4M, besides RGB tokens, the encoder directly accepts RGB pixels with a learnable patch-wise projection to enable use as a ViT [24] backbone for transfer learning.

Masking strategy: We used both multimodal random [7, 62] and span masking [68] strategies that mask input and target tokens. We invoke dataset mixing ratios and Dirichlet sampling parameters, α, to ensure stable training on multiple modalities and datasets, as detailed in appendix E.2.

### 3 Multimodal capabilities

We demonstrate a broad range of capabilities unlocked by 4M-21, including steerable multimodal generation (Sec. 3.1), multimodal retrieval (Sec. 3.2) and strong out-of-the-box capabilities (Sec. 3.3). Please see the project website for more visualizations demonstrating these capabilities.

- 3.1 Steerable multimodal generation

##### 4M-21 can predict any training modality by iteratively decoding tokens [62, 14, 13]. This is shown in fig. 2 where we can generate all modalities from a given input modality in a consistent manner. Furthermore, as we can generate any of the training modalities from any subset of other modalities, both conditionally and unconditionally, it enables several ways to perform fine-grained and multimodal generation, as shown in fig. 4. This includes diverse capabilities such as performing multimodal edits, probing the learned representations, and steering multimodal data generation. Moreover, 4M-21 exhibits improved text understanding capabilities leading to geometrically and semantically plausible generations, both when conditioning on T5-XXL embeddings and on regular captions (fig. 4, top right).

Any-to-RGB retrieval Any-to-any retrieval Multimodal retrieval

Query Top-3 Retrievals

Query Top-3 Retrievals

Query Top-3 Retrievals

|[Figure 310]|
|---|

|[Figure 311]|
|---|

|[Figure 312]|
|---|

|[Figure 313]|
|---|

|[Figure 314]|
|---|

|[Figure 315]|
|---|

|[Figure 316]|
|---|

|[Figure 317]|
|---|

|[Figure 318]|
|---|

|[Figure 319]|
|---|

|[Figure 320]|
|---|

|[Figure 321]|
|---|

ski season is coming

|[Figure 322]|
|---|

|[Figure 323]|
|---|

|[Figure 324]|
|---|

|[Figure 325]|
|---|

|[Figure 326]|
|---|

|[Figure 327]|
|---|

|[Figure 328]|
|---|

|[Figure 329]|
|---|

|[Figure 330]|
|---|

|[Figure 331]|
|---|

|[Figure 332]|
|---|

|[Figure 333]|
|---|

|[Figure 334]|
|---|

|[Figure 335]|
|---|

|[Figure 336]|
|---|

|[Figure 337]|
|---|

|[Figure 338]|
|---|

|[Figure 339]|
|---|

|[Figure 340]|
|---|

|[Figure 341]|
|---|

|[Figure 342]|
|---|

|[Figure 343]|
|---|

|[Figure 344]|
|---|

|[Figure 345]|
|---|

|[Figure 346]|
|---|

|[Figure 347]|
|---|

|[Figure 348]|
|---|

|[Figure 349]|
|---|

|[Figure 350]|
|---|

|[Figure 351]|
|---|

|[Figure 352]|
|---|

|[Figure 353]|
|---|

|[Figure 354]|
|---|

|[Figure 355]|
|---|

|[Figure 356]|
|---|

|[Figure 357]|
|---|

|[Figure 358]|
|---|

a fancy mansion

brightness: 30/255

|[Figure 359]|
|---|

|[Figure 360]|
|---|

|[Figure 361]|
|---|

|[Figure 362]|
|---|

|[Figure 363]|
|---|

|[Figure 364]|
|---|

|[Figure 365]|
|---|

|[Figure 366]|
|---|

|[Figure 367]|
|---|

|[Figure 368]|
|---|

|[Figure 369]|
|---|

a fancy mansion

brightness: 200/255

|[Figure 370]|
|---|

|[Figure 371]|
|---|

|[Figure 372]|
|---|

|a bird’s eye view of sunset shores beach hotel|
|---|

|[Figure 373]|
|---|

|[Figure 374]|
|---|

|[Figure 375]|
|---|

|[Figure 376]|
|---|

|[Figure 377]|
|---|

|[Figure 378]|
|---|

|details of the neuschwanstein castle|
|---|

a fancy mansion

brightness: 200/255 walkability: 75%

- Figure 5: Different modes of multimodal retrieval. We perform multimodal retrievals by predicting global embeddings (here shown for DINOv2) from a given input (of any modality) using 4M-21 and comparing the cosine distances between the query and retrieval set embeddings. Left: Retrieving RGB images from distinctly different query modalities (here RGB, segmentation map, edges, depth map, color palette, and caption). Middle: Retrieving any modality using any other modality as the query input. Each query modality constrains the retrievals differently, e.g. here the RGB image and caption queries always yield Neuschwanstein castle retrievals. In contrast, for depth and semantic queries, the scene is more ambiguous, thus they retrieve other buildings with similar characteristics. Right: We can also combine any subset of modalities to define the query input, e.g. surface normals and a color palette, to better control the retrieval. See appendix B.2 for more results.

- 3.2 Multimodal retrieval

Our model can also perform multimodal retrievals by predicting global embeddings of DINOv2 and ImageBind from any (subset) of the input modalities. Once the global embeddings are obtained, the retrieval is done by finding the retrieval set samples with the smallest cosine distance to the query [65, 33]. As shown in fig. 5, this unlocks retrieval capabilities that were not possible with the original DINOv2 and ImageBind models such as retrieving RGB images or any other modality via using any other modality as the query. Furthermore, one can combine multiple modalities to predict the global embedding, resulting in better control over retrievals, as shown on the right.

|[Figure 379]|
|---|

|[Figure 380]|
|---|

|[Figure 381]|
|---|

|[Figure 382]|
|---|

|[Figure 383]|
|---|

|[Figure 384]|
|---|

|[Figure 385]|
|---|

|[Figure 386]|
|---|

|[Figure 387]|
|---|

|[Figure 388]|
|---|

|[Figure 389]|
|---|

|[Figure 390]|
|---|

|[Figure 391]|
|---|

|[Figure 392]|
|---|

|[Figure 393]|
|---|

|[Figure 394]|
|---|

|[Figure 395]|
|---|

|[Figure 396]|
|---|

|[Figure 397]|
|---|

|[Figure 398]|
|---|

|[Figure 399]|
|---|

|[Figure 400]|
|---|

|[Figure 401]|
|---|

|[Figure 402]|
|---|

|[Figure 403]|
|---|

|[Figure 404]|
|---|

|[Figure 405]|
|---|

|[Figure 406]|
|---|

|[Figure 407]|
|---|

|[Figure 408]|
|---|

|[Figure 409]|
|---|

|[Figure 410]|
|---|

|[Figure 411]|
|---|

|[Figure 412]|
|---|

|[Figure 413]|
|---|

|[Figure 414]|
|---|

|[Figure 415]|
|---|

|[Figure 416]|
|---|

|[Figure 417]|
|---|

|[Figure 418]|
|---|

|[Figure 419]|
|---|

|[Figure 420]|
|---|

|[Figure 421]|
|---|

|[Figure 422]|
|---|

|[Figure 423]|
|---|

|[Figure 424]|
|---|

|[Figure 425]|
|---|

|[Figure 426]|
|---|

|[Figure 427]|
|---|

|[Figure 428]|
|---|

|[Figure 429]|
|---|

|[Figure 430]|
|---|

|[Figure 431]|
|---|

|[Figure 432]|
|---|

|[Figure 433]|
|---|

|[Figure 434]|
|---|

|[Figure 435]|
|---|

|[Figure 436]|
|---|

|[Figure 437]|
|---|

|[Figure 438]|
|---|

|[Figure 439]|
|---|

|[Figure 440]|
|---|

|[Figure 441]|
|---|

|[Figure 442]|
|---|

|[Figure 443]|
|---|

|[Figure 444]|
|---|

|[Figure 445]|
|---|

|[Figure 446]|
|---|

|[Figure 447]|
|---|

|[Figure 448]|
|---|

|[Figure 449]|
|---|

|[Figure 450]|
|---|

|[Figure 451]|
|---|

|[Figure 452]|
|---|

|[Figure 453]|
|---|

|[Figure 454]<br><br>[Figure 455]|
|---|

|[Figure 456]|
|---|

|[Figure 457]|
|---|

|[Figure 458]|
|---|

|[Figure 459]|
|---|

|[Figure 460]|
|---|

|[Figure 461]|
|---|

|[Figure 462]|
|---|

|[Figure 463]|
|---|

|[Figure 464]|
|---|

|[Figure 465]|
|---|

|[Figure 466]|
|---|

PseudoLabelPredictionsPseudoLabelPredictionsPseudoLabelPredictions

RGB Input Predictions

Caption

Bounding boxes

Semantic segmentation

Depth CLIP

Surface normals

Human poses

DINOv2 ImageBind Metadata

Canny edges

SAM edges

SAM instances

Color palette

- Figure 6: Out-of-the-box vision tasks. Given an RGB image, 4M-21 can predict all tasks successfully, as can be seen from their high consistency with the pseudo labels. See fig. 7 for more results.

- Table 1: Out-of-the-box (zero-shot) performance. We show the performance for a common subset of tasks: surface normals and depth estimation on DIODE [86], semantic and instance segmentation on COCO [55], k-NN retrieval on ImageNet-1K [75], and 3D human keypoint estimation on 3DPW [87]. We compare to a set of strong baselines and specialist models, including our pseudo labelers. The model learned to solve all the tasks without a loss of performance, is significantly better than the baselines, and is competitive with pseudo labelers, while being a single model for all tasks. Compared to 4M-7, the 4M-21 model preserved its performance while solving 3x more tasks. ✗ denotes that a given model cannot solve the task out-of-the-box. * shows the tokenizer reconstruction quality and provides an estimate on the performance upper bound due to tokenization. See fig. 13 for qualitative comparisons. Best results are bolded, second best underlined.

Method Normals ↓ Depth ↓ Sem. seg. ↑ Inst. seg. ↑ IN1K kNN ↑ 3D human KP ↓

Omnidata [44] 22.5 0.68 ✗ ✗ ✗ ✗ M2F-B [19] ✗ ✗ 45.7 ✗ ✗ ✗ SAM [47] ✗ ✗ ✗ 32.9 ✗ ✗ DINOv2-B14 [65] ✗ ✗ ✗ ✗ 82.1 / 93.9 ✗ ImageBind-H14 [33] ✗ ✗ ✗ ✗ 81.1 / 94.4 ✗ 4D-Humans [35] ✗ ✗ ✗ ✗ ✗ 81.3

Pseudolabelers

OASIS [18] 34.3 ✗ ✗ ✗ ✗ ✗ MiDaS DPT [70] ✗ 0.73 ✗ ✗ ✗ ✗ M2F-S [19] ✗ ✗ 44.6 ✗ ✗ ✗ M2F-L [19] ✗ ✗ 48.0 ✗ ✗ ✗ HMR [43] ✗ ✗ ✗ ✗ ✗ 130.0 UnifiedIO-B [59] 35.7 1.00 32.9 ✗ ✗ ✗ UnifiedIO-L [59] 33.9 0.87 41.6 ✗ ✗ ✗ UnifiedIO-XL [59] 31.0 0.82 44.3 ✗ ✗ ✗ UnifiedIO 2-L [58] 37.1 0.96 38.9 ✗ ✗ ✗ UnifiedIO 2-XL [58] 34.8 0.86 39.7 ✗ ✗ ✗ UnifiedIO 2-XXL [58] 37.4 0.84 41.7 ✗ ✗ ✗

4M-7 B [62] 21.9 0.71 43.3 ✗ ✗ ✗ 4M-21 B 21.7 0.71 42.5 15.9 73.1 / 89.7 108.3

4M-7 L [62] 21.5 0.69 47.2 ✗ ✗ ✗ 4M-21 L 21.1 0.69 46.4 31.2 77.0 / 91.9 97.4

4M-7 XL [62] 20.6 0.69 48.1 ✗ ✗ ✗ 4M-21 XL 20.8 0.68 48.1 32.0 78.3 / 92.4 92.0 Tokenizer bound* 4.0 0.06 90.5 91.2 80.2 / 93.0 17.5

#### 3.3 Evaluating out-of-the-box capabilities

4M-21 is capable of performing a range of common vision tasks out-of-the-box, as demonstrated visually in fig. 6. In table 1, we evaluate the performance on DIODE [86] surface normal and depth estimation, COCO [55] semantic and instance segmentation, 3DPW [87] 3D human pose estimation, and do ImageNet-1K [75] kNN retrieval using predicted DINOv2 global tokens. We compare against the pseudo labeling networks, strong baselines, and the 4M model from [62] trained on 7 modalities. For surface normal estimation and semantic segmentation, we observed that ensembling multiple predictions significantly improves performance, see appendix F for more details and results.

Our model consistently achieves strong out-of-the-box performance, and often matches or even outperforms the pseudo labelers and other specialist baselines, while being a single model for all tasks. Notice the large performance gap with other multitask models like Unified-IO [59] and Unified-IO-2 [58]. For kNN retrieval, 4M-21 XL performance approaches the tokenizer bound, i.e. the retrieval performance using the DINOv2 tokenizer reconstructions. While the smaller models lag behind 4M models, we observe that 4M-21 XL is able to match the performance of 4M-7 XL, while being trained to solve three times more tasks. The trend over the model size needing to be larger is expected as the number of tasks increase.

### 4 Transfer experiments

To study the scaling characteristics of pre-training any-to-any models on a larger set of modalities, we train models across three different sizes: B, L, and XL. We then transfer their encoders to downstream tasks and evaluate on both unimodal (RGB) and multimodal (RGB + Depth) settings. The decoders are discarded for all transfer experiments, and we instead train task-specific heads. We perform self-comparisons in a similar manner to [62, 7], as well as comparing to a set of strong baselines.

Unimodal transfers. For unimodal transfers we leverage the RGB patch embeddings learned during the pre-training, as RGB pixel inputs are used alongside the tokenized modalities. For the XL models and DINOv2 g, we perform parameter-efficient fine-tuning using LoRA [40] instead of full fine-

- Table 2: Unimodal transfer study. We transfer 4M-21 and baselines to ImageNet-1K [75] classification, ADE20K [100] semantic segmentation, NYUv2 [80] depth estimation, and ARKitScenes [9] (ARKS) 3D object detection. We observe that 4M-21 1) does not lose performance for the transfer tasks that are similar to the seven modalities of 4M, i.e. first three columns of the results, while being able to solve many more, and 2) leads to improved performance for novel tasks that are more different from 4M modalities, e.g. 3D object detection (last column). The improvements are further verified in the multimodal transfer results (Table 3) showing the usefulness of new modalities. Best results per task are bolded, second best underlined.

Pre-training Enc. IN1K ADE20K NYUv2-D ARKS data param. Acc.↑ mIoU↑ δ1 acc.↑ AP3D↑

Method

84.2 46.1 89.1 30.9 DeiT III B [84] IN21K 85.4 49.0 87.4 36.1 MultiMAE B [7] IN1K 84.0 46.2 89.0 34.2 DINOv2 B [65] LVD142M 85.3 51.6 92.2 38.1 4M-7 B [62] CC12M 84.5 50.1 92.0 40.3 4M-7 B (Ours) COYO 84.4 49.4 91.4 38.6 4M-21 B CC12M+COYO+C4 84.5 50.1 90.8 42.4

MAE B [38] IN1K

86M

86.8 51.8 93.6 36.2 DeiT III L [84] IN21K 87.0 52.0 89.6 40.3 DINOv2 L [65] LVD142M 86.7 53.4 94.1 42.8 4M-7 L [62] CC12M 86.6 53.4 94.4 46.8 4M-7 L (Ours) COYO 86.7 53.5 94.3 45.2 4M-21 L CC12M+COYO+C4 86.5 53.4 93.7 47.0

MAE L [38] IN1K

303M

DINOv2 g [65] LVD142M 1.1B 88.0 58.7 92.5 45.3 4M-7 XL [62] CC12M

87.0 55.0 96.1 48.1 4M-7 XL (Ours) COYO 87.1 56.1 96.5 47.3 4M-21 XL CC12M+COYO+C4 87.1 56.0 96.5 48.4

1.2B

tuning, which significantly improves results for XL models. We did not observe similar performance gains for the smaller models. Further training details are described in appendix G.

We evaluate on ImageNet-1K classification [22, 75], ADE20K semantic segmentation [100], NYUv2 depth estimation [80], and ARKitScenes [9] 3D object detection tasks. Some transfer tasks are completely unseen during pre-training, e.g. object classification or 3D object detection, while others are included as different instantiations, e.g. absolute depth instead of relative depth, or using ADE20K instead of COCO classes. We follow the best practices and commonly used settings from other papers [62].

The results are shown in table 2. We make the following observations: 1) for the transfer tasks that are similar to the seven modalities of 4M, e.g. semantic segmentation or depth, 4M-21 does not lose performance due to being trained on many more modalities, 2) for novel transfer tasks like 3D object detection that are sufficiently different from 4M modalities, we observe an improved performance. Moreover, the performance improves with larger model sizes, showing promising scaling trends. These trends can be further seen in the multimodal transfer results, which we explain next.

Multimodal transfers. We perform multimodal transfers on NYUv2, Hypersim [72] semantic segmentation, and 3D object detection on ARKitScenes. We compare transfers using RGB images only, and RGB pixels + tokenized sensory depth as inputs. As table 3 shows, 4M-21 makes strong use of optionally available depth inputs and significantly improves upon the baselines.

Table 3: Multimodal transfer study. We transfer both 4M-21 and 4M (pre-trained on CC12M) to NYUv2 and Hypersim segmentation, and 3D object detection on ARKitScenes. All models are able to use optionally available depth when it is of high quality (Hypersim & ARKitScenes), while our model achieves the best results. Best results are bolded, second best underlined.

NYUv2-S Hypersim ARKitScenes mIoU ↑ mIoU ↑ AP3D↑

### 5 Related Work

Method RGB RGB-D RGB RGB-D RGB RGB-D

4M-7 B 56.6 57.5 40.2 43.9 40.3 46.5 4M-21 B 58.7 59.7 38.6 46.4 42.4 48.1

Multitask learning in vision involves training a single model to perform multiple visual tasks efficiently [12, 74]. Earlier methods [27, 61, 48] combined multiple dense vision tasks into a single model but faced challenges scaling to a larger variety

4M-7 L 61.2 61.4 48.7 50.5 46.8 49.5 4M-21 L 61.8 61.8 47.3 50.7 47.0 50.1

4M-7 XL 62.1 61.2 48.6 51.0 48.1 50.1 4M-21 XL 63.9 63.9 48.6 52.5 48.4 51.3

of tasks and modalities, limited by training instabilities and the need for careful task selection and loss balancing to reduce negative transfer [46, 96, 82, 94].

Recently, discrete tokenization has enabled a shift towards integrating numerous vision tasks into unified multimodal and multitask models such as Gato [71], OFA [88], Pix2Seq [16, 17], UnifiedIO [59, 58], 4M [62], and more [49, 79, 5, 39, 25, 42, 4, 92, 83, 102, 52, 99]. These methods first transform various modalities and tasks into sequences or sets of discrete tokens [23, 50, 63, 29, 90], and then train a single Transformer on these tokens using either a sequence modeling [71, 88, 59, 16, 17, 49, 79] or masked modeling objective [62, 89]. Some methods (e.g. Gato [71], UnifiedIO [59, 58]) perform co-training on multiple disjoint datasets and are capable of performing a wide range of tasks, but not jointly. In contrast, methods like 4M [62] train on a single aligned dataset through the use of pseudo labeling, enabling any-to-any modality prediction but on a typically more limited set of modalities. We significantly expand upon them by adding the ability to use this framework for an even greater amount of modalities and capabilities.

Furthermore, masked modeling has proven effective for learning useful representations in both NLP [23, 68] and vision [38, 8, 101, 28]. Extending it to multimodal domains [7, 34, 89, 62] enables strong cross-modal representations which is critical for multimodal learning. When combined with tokenization, masked modeling also enables generative applications [14, 53, 13, 81, 62]. Our work highlights the ability of masked modeling to expand to a much greater set of modalities than previously shown, improving upon the out-of-the-box and multimodal generation capabilities of previous works.

### 6 Limitations and Discussion

We developed an any-to-any model on tens of diverse modalities and tasks. This was achieved by mapping all modalities to discrete sets of tokens via modality-specific tokenizers and using a multimodal masked training objective [62]. We successfully scaled the training to 3 billion parameters and to 21 modalities and different datasets, without a degradation in performance compared to the existing expert single/few task models. This results in strong out-of-the-box capabilities as well new potential for multimodal interaction, generation, and retrieval, all by a single unified model. Below, we discuss limitations and future work.

Transfer/emergent capabilities: One hope from training a single network on several tasks is leading to a model that can solve novel tasks, often referred to as “transfer” or “emergent” capabilities. While, as we showed, a multitask model brings several key advantages even without transfer/emergence (e.g., efficiency, using a single model for broad out-of-the-box capabilities, modality fusion, etc.), we observe that the potential for transfer/emergence remains largely untapped. In general, compared to LLMs, vision/multimodal models have not shown exciting results in terms of transfer/emergence yet. We find this to be an important point to address in the future, e.g., via designing multitask architectures that have emergence, in contrast to out-of-the-box capabilities, as their main objective.

Better tokenization: Like any token-based model, ours can directly benefit from progress on tokenizers, e.g. higher reconstruction fidelity.

Co-training on partially aligned datasets: We showed the possibility of training on partially aligned datasets, e.g. text data from C4 and other modalities from CC12M, yet further investigations and a larger mixture of datasets are expected to bring stronger capabilities.

### References

- [1] OpenCV. https://opencv.org/ 4
- [2] PyPalette. https://github.com/adamgrieger/pypalette 3, 24
- [3] Python Pillow. https://python-pillow.org/ 4
- [4] Aghajanyan, A., Huang, B., Ross, C., Karpukhin, V., Xu, H., Goyal, N., Okhonko, D., Joshi, M., Ghosh, G., Lewis, M., Zettlemoyer, L.: CM3: A causal masked multimodal model of the internet. ArXiv abs/2201.07520 (2022) 10
- [5] Aghajanyan, A., Yu, L., Conneau, A., Hsu, W.N., Hambardzumyan, K., Zhang, S., Roller, S., Goyal, N., Levy, O., Zettlemoyer, L.: Scaling laws for generative mixed-modal language models. ArXiv abs/2301.03728 (2023) 10

- [6] Bachmann, G., Anagnostidis, S., Hofmann, T.: Scaling mlps: A tale of inductive bias. arXiv preprint arXiv:2306.13575 (2023) 5, 24
- [7] Bachmann, R., Mizrahi, D., Atanov, A., Zamir, A.: MultiMAE: Multi-modal multi-task masked autoencoders. European Conference on Computer Vision (2022) 6, 8, 9, 10, 28, 29
- [8] Bao, H., Dong, L., Piao, S., Wei, F.: BEiT: BERT pre-training of image transformers. In: International Conference on Learning Representations (2022) 10
- [9] Baruch, G., Chen, Z., Dehghan, A., Dimry, T., Feigin, Y., Fu, P., Gebauer, T., Joffe, B., Kurz, D., Schwartz, A., Shulman, E.: Arkitscenes - a diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. In: NeurIPS (2021), https://arxiv.org/pdf/2111.08897.pdf 9
- [10] Burgess, N., Milanovic, J., Stephens, N., Monachopoulos, K., Mansell, D.: Bfloat16 processing for neural networks. In: 2019 IEEE 26th Symposium on Computer Arithmetic (ARITH). pp. 88–91 (2019). https://doi.org/10.1109/ARITH.2019.00022 26
- [11] Byeon, M., Park, B., Kim, H., Lee, S., Baek, W., Kim, S.: COYO-700M: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset (2022) 2, 4, 5, 22, 25
- [12] Caruana, R.: Multitask learning. Machine Learning 28, 41–75 (1997). https://doi.org/10.1023/A:1007379606734 9
- [13] Chang, H., Zhang, H., Barber, J., Maschinot, A., Lezama, J., Jiang, L., Yang, M., Murphy, K.P., Freeman, W.T., Rubinstein, M., Li, Y., Krishnan, D.: Muse: Text-to-image generation via masked generative transformers. In: International Conference on Machine Learning (2023) 4, 5, 6, 10, 23
- [14] Chang, H., Zhang, H., Jiang, L., Liu, C., Freeman, W.T.: MaskGIT: Masked generative image transformer. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 11305–11315

(2022) 5, 6, 10

- [15] Changpinyo, S., Sharma, P.K., Ding, N., Soricut, R.: Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 3557–3567 (2021) 4, 5, 25, 31
- [16] Chen, T., Saxena, S., Li, L., Fleet, D.J., Hinton, G.: Pix2seq: A language modeling framework for object detection. In: International Conference on Learning Representations (2022) 2, 10, 24, 25
- [17] Chen, T., Saxena, S., Li, L., Lin, T.Y., Fleet, D.J., Hinton, G.E.: A unified sequence interface for vision tasks. Advances in Neural Information Processing Systems 35, 31333–31346 (2022) 2, 10
- [18] Chen, W., Qian, S., Fan, D., Kojima, N., Hamilton, M., Deng, J.: Oasis: A large-scale dataset for single image 3d in the wild. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 679–688 (2020) 8
- [19] Cheng, B., Misra, I., Schwing, A.G., Kirillov, A., Girdhar, R.: Masked-attention mask transformer for universal image segmentation. CVPR (2022) 4, 8, 23
- [20] Clark, K., Luong, M.T., Le, Q.V., Manning, C.D.: Electra: Pre-training text encoders as discriminators rather than generators. In: International Conference on Learning Representations (2020) 28, 29
- [21] Cubuk, E.D., Zoph, B., Shlens, J., Le, Q.V.: Randaugment: Practical automated data augmentation with a reduced search space. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops. pp. 702–703 (2020) 28
- [22] Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: ImageNet: A large-scale hierarchical image database. 2009 IEEE Conference on Computer Vision and Pattern Recognition pp. 248–255 (2009) 9
- [23] Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: BERT: Pre-training of deep bidirectional transformers for language understanding. In: North American Chapter of the Association for Computational Linguistics

(2019) 2, 4, 5, 10

- [24] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations

(2021) 2, 5, 6

- [25] Driess, D., Xia, F., Sajjadi, M.S., Lynch, C., Chowdhery, A., Ichter, B., Wahid, A., Tompson, J., Vuong, Q., Yu, T., et al.: Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378

(2023) 10

- [26] Eftekhar, A., Sax, A., Bachmann, R., Malik, J., Zamir, A.R.: Omnidata: A scalable pipeline for making multi-task mid-level vision datasets from 3d scans. 2021 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 10766–10776 (2021) 4
- [27] Eigen, D., Fergus, R.: Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture. In: Proceedings of the IEEE international conference on computer vision. pp. 2650–2658 (2015) 9
- [28] El-Nouby, A., Izacard, G., Touvron, H., Laptev, I., Jegou, H., Grave, E.: Are large-scale datasets necessary for self-supervised pre-training? ArXiv abs/2112.10740 (2021) 10
- [29] Esser, P., Rombach, R., Ommer, B.: Taming transformers for high-resolution image synthesis. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 12868–12878 (2020) 10
- [30] Feichtenhofer, C., Li, Y., He, K., et al.: Masked autoencoders as spatiotemporal learners. Advances in neural information processing systems 35, 35946–35958 (2022) 26
- [31] Fifty, C., Amid, E., Zhao, Z., Yu, T., Anil, R., Finn, C.: Efficiently identifying task groupings for multi-task learning. Advances in Neural Information Processing Systems 34, 27503–27516 (2021) 2
- [32] Ghiasi, G., Cui, Y., Srinivas, A., Qian, R., Lin, T.Y., Cubuk, E.D., Le, Q.V., Zoph, B.: Simple copy-paste is a strong data augmentation method for instance segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2918–2928 (2021) 29
- [33] Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K.V., Joulin, A., Misra, I.: ImageBind: One embedding space to bind them all. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 15180–15190 (2023) 2, 4, 7, 8, 23
- [34] Girdhar, R., El-Nouby, A., Singh, M., Alwala, K.V., Joulin, A., Misra, I.: OmniMAE: Single model masked pretraining on images and videos. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10406–10417 (2023) 10
- [35] Goel, S., Pavlakos, G., Rajasegaran, J., Kanazawa, A., Malik, J.: Humans in 4D: Reconstructing and tracking humans with transformers. In: ICCV (2023) 2, 4, 8, 23, 28
- [36] Goyal, P., Dollár, P., Girshick, R.B., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., He, K.: Accurate, large minibatch sgd: Training imagenet in 1 hour. ArXiv abs/1706.02677 (2017) 26, 28
- [37] Hasler, D., Suesstrunk, S.E.: Measuring colorfulness in natural images. In: Human vision and electronic imaging VIII. vol. 5007, pp. 87–95. SPIE (2003) 4, 23
- [38] He, K., Chen, X., Xie, S., Li, Y., Doll’ar, P., Girshick, R.B.: Masked autoencoders are scalable vision learners. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 15979– 15988 (2021) 9, 10, 23
- [39] Hu, A., Russell, L., Yeo, H., Murez, Z., Fedoseev, G., Kendall, A., Shotton, J., Corrado, G.: Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080 (2023) 10
- [40] Hu, J.E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Chen, W.: LoRA: Low-rank adaptation of large language models. ArXiv abs/2106.09685 (2021), https://api.semanticscholar.org/ CorpusID:235458009 8, 29
- [41] Huang, G., Sun, Y., Liu, Z., Sedra, D., Weinberger, K.Q.: Deep networks with stochastic depth. In: European conference on computer vision. pp. 646–661. Springer (2016) 28, 29
- [42] Huang, S., Dong, L., Wang, W., Hao, Y., Singhal, S., Ma, S., Lv, T., Cui, L., Mohammed, O.K., Liu, Q., Aggarwal, K., Chi, Z., Bjorck, J., Chaudhary, V., Som, S., Song, X., Wei, F.: Language is not all you need: Aligning perception with language models. ArXiv abs/2302.14045 (2023) 10
- [43] Kanazawa, A., Black, M.J., Jacobs, D.W., Malik, J.: End-to-end recovery of human shape and pose. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 7122–7131 (2018) 8
- [44] Kar, O.F., Yeo, T., Atanov, A., Zamir, A.: 3d common corruptions and data augmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18963–18974 (2022) 4, 8
- [45] Ke, L., Ye, M., Danelljan, M., Liu, Y., Tai, Y.W., Tang, C.K., Yu, F.: Segment anything in high quality

(2023) 23

- [46] Kendall, A., Gal, Y., Cipolla, R.: Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition pp. 7482–7491

(2017), https://api.semanticscholar.org/CorpusID:4800342 2, 10

- [47] Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., Dollár, P., Girshick, R.B.: Segment anything. ArXiv abs/2304.02643 (2023), https://api.semanticscholar.org/CorpusID:257952310 2, 4, 8, 23, 26
- [48] Kokkinos, I.: Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 6129–6138 (2017) 9
- [49] Kolesnikov, A., Susano Pinto, A., Beyer, L., Zhai, X., Harmsen, J., Houlsby, N.: UViM: A unified modeling approach for vision with learned guiding codes. Advances in Neural Information Processing Systems 35, 26295–26308 (2022) 10
- [50] Kudo, T., Richardson, J.: Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. arXiv preprint arXiv:1808.06226 (2018) 10
- [51] Levine, Y., Lenz, B., Lieber, O., Abend, O., Leyton-Brown, K., Tennenholtz, M., Shoham, Y.: {PMI}masking: Principled masking of correlated spans. In: International Conference on Learning Representations (2021), https://openreview.net/forum?id=3Aoft6NWFej 25
- [52] Li, H., Zhu, J., Jiang, X., Zhu, X., Li, H., Yuan, C., Wang, X., Qiao, Y., Wang, X., Wang, W., Dai, J.: Uni-perceiver v2: A generalist model for large-scale vision and vision-language tasks (2022) 10
- [53] Li, T., Chang, H., Mishra, S., Zhang, H., Katabi, D., Krishnan, D.: [mage]: MAsked generative encoder to unify representation learning and image synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2142–2152 (2023) 10
- [54] Li, Y., Mao, H., Girshick, R., He, K.: Exploring plain vision transformer backbones for object detection. In: Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part IX. pp. 280–296. Springer (2022) 4, 23
- [55] Lin, T.Y., Maire, M., Belongie, S.J., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft COCO: Common objects in context. In: European Conference on Computer Vision (2014) 4, 8
- [56] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2021) 23
- [57] Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (2019) 26, 28, 29, 30
- [58] Lu, J., Clark, C., Lee, S., Zhang, Z., Khosla, S., Marten, R., Hoiem, D., Kembhavi, A.: Unified-IO 2: Scaling autoregressive multimodal models with vision, language, audio, and action. ArXiv abs/2312.17172

(2023), https://api.semanticscholar.org/CorpusID:266573555 1, 2, 8, 10, 26, 27, 28

- [59] Lu, J., Clark, C., Zellers, R., Mottaghi, R., Kembhavi, A.: Unified-IO: A unified model for vision, language, and multi-modal tasks. In: The Eleventh International Conference on Learning Representations

(2023) 1, 2, 8, 10, 26, 27, 28

- [60] Mama, R., Tyndel, M.S., Kadhim, H., Clifford, C., Thurairatnam, R.: Nwt: Towards natural audio-to-video generation with representation learning. ArXiv abs/2106.04283 (2021), https://api. semanticscholar.org/CorpusID:235367982 5
- [61] Misra, I., Shrivastava, A., Gupta, A., Hebert, M.: Cross-stitch networks for multi-task learning. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3994–4003 (2016) 9
- [62] Mizrahi, D., Bachmann, R., Kar, O.F., Yeo, T., Gao, M., Dehghan, A., Zamir, A.: 4M: Massively multimodal masked modeling. In: Advances in Neural Information Processing Systems (2023) 1, 2, 3, 4, 5, 6, 8, 9, 10, 22, 24, 25, 26, 28, 29
- [63] van den Oord, A., Vinyals, O., Kavukcuoglu, K.: Neural discrete representation learning. Advances in neural information processing systems 30 (2017) 2, 5, 10
- [64] OpenAI: GPT-4 technical report (2023) 4

- [65] Oquab, M., Darcet, T., Moutakanni, T., Vo, H.Q., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.Y.B., Li, S.W., Misra, I., Rabbat, M.G., Sharma, V., Synnaeve, G., Xu, H., Jégou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: DINOv2: Learning robust visual features without supervision. ArXiv abs/2304.07193

(2023) 2, 4, 7, 8, 9, 23, 28

- [66] Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Muller, J., Penna, J., Rombach, R.: SDXL: Improving latent diffusion models for high-resolution image synthesis. ArXiv abs/2307.01952

(2023), https://api.semanticscholar.org/CorpusID:259341735 4, 23

- [67] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning (2021) 4, 23
- [68] Raffel, C., Shazeer, N.M., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research 21(1), 5485–5551 (2020) 2, 4, 5, 6, 10, 22, 23, 25
- [69] Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., Sutskever, I.: Zero-shot text-to-image generation. In: International Conference on Machine Learning (2021) 5
- [70] Ranftl, R., Bochkovskiy, A., Koltun, V.: Vision transformers for dense prediction. 2021 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 12159–12168 (2021) 8, 23
- [71] Reed, S., Zolna, K., Parisotto, E., Colmenarejo, S.G., Novikov, A., Barth-Maron, G., Gimenez, M., Sulsky, Y., Kay, J., Springenberg, J.T., et al.: A generalist agent. arXiv preprint arXiv:2205.06175 (2022) 10
- [72] Roberts, M., Paczan, N.: Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. 2021 IEEE/CVF International Conference on Computer Vision (ICCV) pp. 10892–10902 (2020) 9
- [73] Ruder, S.: An overview of multi-task learning in deep neural networks. ArXiv abs/1706.05098 (2017), https://api.semanticscholar.org/CorpusID:10175374 2
- [74] Ruder, S.: An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098

(2017) 9

- [75] Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M.S., Berg, A.C., Fei-Fei, L.: ImageNet large scale visual recognition challenge. International Journal of Computer Vision 115, 211–252 (2014) 8, 9
- [76] Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E.L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems 35, 36479–36494 (2022) 4, 23
- [77] Shazeer, N.: GLU variants improve transformer. ArXiv abs/2002.05202 (2020) 26
- [78] Shi, J., Wu, C., Liang, J., Liu, X., Duan, N.: DiVAE: Photorealistic images synthesis with denoising diffusion decoder. ArXiv abs/2206.00386 (2022) 5
- [79] Shukor, M., Dancette, C., Rame, A., Cord, M.: Unified model for image, video, audio and language tasks. arXiv preprint arXiv:2307.16184 (2023) 10
- [80] Silberman, N., Hoiem, D., Kohli, P., Fergus, R.: Indoor segmentation and support inference from RGBD images. In: European Conference on Computer Vision (2012) 9
- [81] Sohn, K., Ruiz, N., Lee, K., Chin, D.C., Blok, I., Chang, H., Barber, J., Jiang, L., Entis, G., Li, Y., et al.: Styledrop: Text-to-image generation in any style. arXiv preprint arXiv:2306.00983 (2023) 10
- [82] Standley, T., Zamir, A., Chen, D., Guibas, L., Malik, J., Savarese, S.: Which tasks should be learned together in multi-task learning? In: International Conference on Machine Learning. pp. 9120–9132. PMLR (2020) 2, 10
- [83] Team, C.: Chameleon: Mixed-modal early-fusion foundation models (2024) 10
- [84] Touvron, H., Cord, M., Jégou, H.: DeiT III: Revenge of the vit. In: European Conference on Computer Vision. pp. 516–533 (2022) 9

- [85] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023) 4
- [86] Vasiljevic, I., Kolkin, N., Zhang, S., Luo, R., Wang, H., Dai, F.Z., Daniele, A.F., Mostajabi, M., Basart, S., Walter, M.R., et al.: Diode: A dense indoor and outdoor depth dataset. arXiv preprint arXiv:1908.00463

(2019) 8

- [87] Von Marcard, T., Henschel, R., Black, M.J., Rosenhahn, B., Pons-Moll, G.: Recovering accurate 3d human pose in the wild using imus and a moving camera. In: Proceedings of the European conference on computer vision (ECCV). pp. 601–617 (2018) 8
- [88] Wang, P., Yang, A., Men, R., Lin, J., Bai, S., Li, Z., Ma, J., Zhou, C., Zhou, J., Yang, H.: Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In: International Conference on Machine Learning (2022) 2, 10
- [89] Wang, W., Bao, H., Dong, L., Bjorck, J., Peng, Z., Liu, Q., Aggarwal, K., Mohammed, O.K., Singhal, S., Som, S., Wei, F.: Image as a foreign language: BEiT pretraining for all vision and vision-language tasks. ArXiv abs/2208.10442 (2022) 4, 10
- [90] Yu, J., Li, X., Koh, J.Y., Zhang, H., Pang, R., Qin, J., Ku, A., Xu, Y., Baldridge, J., Wu, Y.: Vectorquantized image modeling with improved VQGAN. In: International Conference on Learning Representations (2022) 10
- [91] Yu, J., Xu, Y., Koh, J.Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B.K., Hutchinson, B., Han, W., Parekh, Z., Li, X., Zhang, H., Baldridge, J., Wu, Y.: Scaling autoregressive models for content-rich text-to-image generation. Transactions on Machine Learning Research (2022) 5
- [92] Yu, L., Shi, B., Pasunuru, R., Muller, B., Golovneva, O.Y., Wang, T., Babu, A., Tang, B., Karrer, B., Sheynin, S., Ross, C., Polyak, A., Howes, R., Sharma, V., Xu, P., Tamoyan, H., Ashual, O., Singer, U., Li, S.W., Zhang, S., James, R., Ghosh, G., Taigman, Y., Fazel-Zarandi, M., Celikyilmaz, A., Zettlemoyer, L., Aghajanyan, A.: Scaling autoregressive multi-modal models: Pretraining and instruction tuning. ArXiv abs/2309.02591 (2023) 10
- [93] Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., Finn, C.: Gradient surgery for multi-task learning. ArXiv abs/2001.06782 (2020), https://api.semanticscholar.org/CorpusID:210839011 2
- [94] Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., Finn, C.: Gradient surgery for multi-task learning. Advances in Neural Information Processing Systems 33, 5824–5836 (2020) 2, 10
- [95] Yun, S., Han, D., Oh, S.J., Chun, S., Choe, J., Yoo, Y.: Cutmix: Regularization strategy to train strong classifiers with localizable features. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 6023–6032 (2019) 28
- [96] Zamir, A.R., Sax, A., Shen, B.W., Guibas, L.J., Malik, J., Savarese, S.: Taskonomy: Disentangling task transfer learning. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition pp. 3712–3722 (2018) 10
- [97] Zhang, H., Cisse, M., Dauphin, Y.N., Lopez-Paz, D.: mixup: Beyond empirical risk minimization. In: International Conference on Learning Representations (2018) 28
- [98] Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image diffusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3836–3847 (2023) 4, 24
- [99] Zhang, Y., Gong, K., Zhang, K., Li, H., Qiao, Y., Ouyang, W., Yue, X.: Meta-transformer: A unified framework for multimodal learning (2023) 10
- [100] Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., Torralba, A.: Scene parsing through ADE20K dataset. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) pp. 5122–5130

(2017) 9

- [101] Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., Kong, T.: iBoT: Image BERT pre-training with online tokenizer. In: International Conference on Learning Representations (2022) 10
- [102] Zhu, X., Zhu, J., Li, H., Wu, X., Wang, X., Li, H., Wang, X., Dai, J.: Uni-Perceiver: Pre-training unified architecture for generic perception for zero-shot and few-shot tasks. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp. 16783–16794 (2021) 10

## Appendix

- A Code, Pre-trained Models & Interactive Visualizations 17
- B Multimodal Capabilities 17

- B.1 Additional multimodal generation & probing visualizations . . . . . . . . . . . . . 17
- B.2 Additional retrieval visualizations . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- C Additional Ablations 22

- C.1 Ablation of pre-training data and modalities . . . . . . . . . . . . . . . . . . . . . 22
- C.2 Ablation of ensembling the predictions . . . . . . . . . . . . . . . . . . . . . . . . 22

- D Multimodal Dataset & Tokenization Details 23

- D.1 Pseudo labeled multimodal training dataset . . . . . . . . . . . . . . . . . . . . . 23
- D.2 Tokenization of human poses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- D.3 Tokenization of SAM instances . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- D.4 Tokenization of global feature maps . . . . . . . . . . . . . . . . . . . . . . . . . 24
- D.5 Tokenization of dense feature maps . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- D.6 Tokenization of sequence modalities . . . . . . . . . . . . . . . . . . . . . . . . . 24
- D.7 Tokenization of Canny and SAM edges . . . . . . . . . . . . . . . . . . . . . . . 25

- E Training Details 25

- E.1 Modality-specific accommodations . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- E.2 Multidataset co-training and diversified multimodal masking strategy . . . . . . . . 25

- F Out-of-the-box Evaluation Details 26

- F.1 Surface normal and depth estimation on DIODE . . . . . . . . . . . . . . . . . . . 26
- F.2 Semantic and instance segmentation on COCO . . . . . . . . . . . . . . . . . . . 26
- F.3 kNN retrieval on ImageNet-1K . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- F.4 3D human pose prediction on 3DPW . . . . . . . . . . . . . . . . . . . . . . . . . 28

- G Transfer Evaluation Details 28
- H Investigating Different Tokenization Schemes 29
- I Broader Impact 31

- I.1 Computational costs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- I.2 Social impact . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

### A Code, Pre-trained Models & Interactive Visualizations

Please see our website for documented open-source code, pre-trained model and tokenizer weights, as well as an overview video and additional interactive visualizations.

### B Multimodal Capabilities

#### B.1 Additional multimodal generation & probing visualizations

Please see Figures 7, 8, 9, 10 for additional qualitative results on any-to-any generation, controlled generation, and text understanding capabilities of our model.

###### RGB Input Predictions

Semantic segmentation

Surface normals

Human poses

Canny edges

SAM edges

SAM instances

Color palette

Bounding boxes

Caption

Depth CLIP

DINOv2 ImageBind Metadata

SingleTaskExperts

|[Figure 467]|
|---|

|[Figure 468]|
|---|

|[Figure 469]|
|---|

|[Figure 470]|
|---|

|[Figure 471]|
|---|

|[Figure 472]|
|---|

|[Figure 473]|
|---|

|[Figure 474]|
|---|

|[Figure 475]|
|---|

|[Figure 476]|
|---|

|[Figure 477]|
|---|

|[Figure 478]|
|---|

| |
|---|

[Figure 479]

|[Figure 480]|
|---|

(PseudoLabels) Predictions

|[Figure 481]<br><br>|
|---|

|[Figure 482]|
|---|

|[Figure 483]|
|---|

|[Figure 484]|
|---|

|[Figure 485]|
|---|

|[Figure 486]|
|---|

|[Figure 487]|
|---|

|[Figure 488]|
|---|

|[Figure 489]|
|---|

|[Figure 490]|
|---|

|[Figure 491]|
|---|

|[Figure 492]|
|---|

|[Figure 493]|
|---|

|[Figure 494]|
|---|

|[Figure 495]|
|---|

SingleTaskExperts

|[Figure 496]|
|---|

|[Figure 497]|
|---|

|[Figure 498]|
|---|

|[Figure 499]|
|---|

|[Figure 500]|
|---|

|[Figure 501]|
|---|

|[Figure 502]|
|---|

|[Figure 503]|
|---|

|[Figure 504]|
|---|

|[Figure 505]|
|---|

|[Figure 506]|
|---|

|[Figure 507]|
|---|

| |
|---|

[Figure 508]

|[Figure 509]|
|---|

(PseudoLabels) Predictions

|[Figure 510]|
|---|

|[Figure 511]<br><br>[Figure 512]|
|---|

|[Figure 513]|
|---|

|[Figure 514]|
|---|

|[Figure 515]|
|---|

|[Figure 516]|
|---|

|[Figure 517]|
|---|

|[Figure 518]|
|---|

|[Figure 519]|
|---|

|[Figure 520]|
|---|

|[Figure 521]|
|---|

|[Figure 522]|
|---|

|[Figure 523]|
|---|

|[Figure 524]|
|---|

|[Figure 525]|
|---|

SingleTaskExperts

|[Figure 526]|
|---|

|[Figure 527]|
|---|

|[Figure 528]|
|---|

|[Figure 529]|
|---|

|[Figure 530]|
|---|

|[Figure 531]|
|---|

|[Figure 532]|
|---|

|[Figure 533]|
|---|

|[Figure 534]|
|---|

|[Figure 535]|
|---|

|[Figure 536]|
|---|

|[Figure 537]|
|---|

|[Figure 538]|
|---|

|[Figure 539]|
|---|

(PseudoLabels) Predictions

|[Figure 540]|
|---|

|[Figure 541]|
|---|

|[Figure 542]|
|---|

|[Figure 543]|
|---|

|[Figure 544]|
|---|

|[Figure 545]|
|---|

|[Figure 546]|
|---|

|[Figure 547]|
|---|

|[Figure 548]|
|---|

|[Figure 549]|
|---|

|[Figure 550]|
|---|

|[Figure 551]|
|---|

|[Figure 552]|
|---|

|[Figure 553]|
|---|

|[Figure 554]|
|---|

SingleTaskExperts

|[Figure 555]|
|---|

|[Figure 556]|
|---|

|[Figure 557]|
|---|

|[Figure 558]|
|---|

|[Figure 559]|
|---|

|[Figure 560]|
|---|

|[Figure 561]|
|---|

|[Figure 562]|
|---|

|[Figure 563]|
|---|

|[Figure 564]|
|---|

|[Figure 565]|
|---|

|[Figure 566]|
|---|

|[Figure 567]|
|---|

|[Figure 568]|
|---|

(PseudoLabels) Predictions

|[Figure 569]<br><br>|
|---|

|[Figure 570]|
|---|

|[Figure 571]|
|---|

|[Figure 572]|
|---|

|[Figure 573]|
|---|

|[Figure 574]|
|---|

|[Figure 575]|
|---|

|[Figure 576]|
|---|

|[Figure 577]|
|---|

|[Figure 578]|
|---|

|[Figure 579]|
|---|

|[Figure 580]|
|---|

|[Figure 581]|
|---|

|[Figure 582]|
|---|

|[Figure 583]|
|---|

SingleTaskExperts

|[Figure 584]|
|---|

|[Figure 585]|
|---|

|[Figure 586]|
|---|

|[Figure 587]|
|---|

|[Figure 588]|
|---|

|[Figure 589]|
|---|

|[Figure 590]|
|---|

|[Figure 591]|
|---|

|[Figure 592]|
|---|

|[Figure 593]|
|---|

|[Figure 594]|
|---|

|[Figure 595]|
|---|

|[Figure 596]|
|---|

|[Figure 597]|
|---|

(PseudoLabels) Predictions

|[Figure 598]|
|---|

|[Figure 599]|
|---|

|[Figure 600]|
|---|

|[Figure 601]|
|---|

|[Figure 602]|
|---|

|[Figure 603]|
|---|

|[Figure 604]|
|---|

|[Figure 605]|
|---|

|[Figure 606]|
|---|

|[Figure 607]|
|---|

|[Figure 608]|
|---|

|[Figure 609]|
|---|

|[Figure 610]|
|---|

|[Figure 611]|
|---|

|[Figure 612]|
|---|

SingleTaskExperts

|[Figure 613]|
|---|

|[Figure 614]|
|---|

|[Figure 615]|
|---|

|[Figure 616]|
|---|

|[Figure 617]|
|---|

|[Figure 618]|
|---|

|[Figure 619]|
|---|

|[Figure 620]|
|---|

|[Figure 621]|
|---|

|[Figure 622]|
|---|

|[Figure 623]|
|---|

|[Figure 624]|
|---|

|[Figure 625]|
|---|

|[Figure 626]|
|---|

(PseudoLabels) Predictions

|[Figure 627]<br><br>|
|---|

|[Figure 628]|
|---|

|[Figure 629]|
|---|

|[Figure 630]|
|---|

|[Figure 631]|
|---|

|[Figure 632]|
|---|

|[Figure 633]|
|---|

|[Figure 634]|
|---|

|[Figure 635]|
|---|

|[Figure 636]|
|---|

|[Figure 637]|
|---|

|[Figure 638]|
|---|

|[Figure 639]|
|---|

|[Figure 640]|
|---|

|[Figure 641]|
|---|

SingleTaskExperts

|[Figure 642]|
|---|

|[Figure 643]|
|---|

|[Figure 644]|
|---|

|[Figure 645]|
|---|

|[Figure 646]|
|---|

|[Figure 647]|
|---|

|[Figure 648]|
|---|

|[Figure 649]|
|---|

|[Figure 650]|
|---|

|[Figure 651]|
|---|

|[Figure 652]|
|---|

|[Figure 653]|
|---|

|[Figure 654]|
|---|

|[Figure 655]|
|---|

(PseudoLabels) Predictions

|[Figure 656]|
|---|

|[Figure 657]|
|---|

|[Figure 658]|
|---|

|[Figure 659]|
|---|

|[Figure 660]|
|---|

|[Figure 661]|
|---|

|[Figure 662]|
|---|

|[Figure 663]|
|---|

|[Figure 664]|
|---|

|[Figure 665]|
|---|

|[Figure 666]|
|---|

|[Figure 667]|
|---|

|[Figure 668]|
|---|

|[Figure 669]|
|---|

|[Figure 670]|
|---|

SingleTaskExperts

|[Figure 671]|
|---|

|[Figure 672]|
|---|

|[Figure 673]|
|---|

|[Figure 674]|
|---|

|[Figure 675]|
|---|

|[Figure 676]|
|---|

|[Figure 677]|
|---|

|[Figure 678]|
|---|

|[Figure 679]|
|---|

|[Figure 680]|
|---|

|[Figure 681]|
|---|

|[Figure 682]|
|---|

|[Figure 683]|
|---|

|[Figure 684]|
|---|

(PseudoLabels) Predictions

|[Figure 685]<br><br>|
|---|

|[Figure 686]|
|---|

|[Figure 687]|
|---|

|[Figure 688]|
|---|

|[Figure 689]|
|---|

|[Figure 690]|
|---|

|[Figure 691]|
|---|

|[Figure 692]|
|---|

|[Figure 693]|
|---|

|[Figure 694]|
|---|

|[Figure 695]|
|---|

|[Figure 696]|
|---|

|[Figure 697]|
|---|

|[Figure 698]|
|---|

|[Figure 699]|
|---|

- Figure 7: RGB-to-any generation. This is an extension of fig. 6 and visualizes the model’s out-ofthe-box capabilities on various vision tasks, compared to the pseudo labeler outputs.

Image metadata

Geometric metadata

Geometric complexity

Occlusion score

Original Res.

Brightness

Saturation

Colorfulness

Contrast

Entropy

40%

30%

128 x 128

10%

0%

15%

15%

3

|[Figure 700]|
|---|

|[Figure 701]|
|---|

|[Figure 702]|
|---|

|[Figure 703]|
|---|

|[Figure 704]|
|---|

|[Figure 705]|
|---|

|[Figure 706]|
|---|

|[Figure 707]|
|---|

|[Figure 708]|
|---|

|[Figure 709]|
|---|

|[Figure 710]|
|---|

|[Figure 711]|
|---|

50%

40%

768 x 768

40%

15%

40%

60%

5

|[Figure 712]|
|---|

|[Figure 713]|
|---|

|[Figure 714]|
|---|

|[Figure 715]|
|---|

|[Figure 716]|
|---|

|[Figure 717]|
|---|

|[Figure 718]|
|---|

|[Figure 719]|
|---|

|[Figure 720]|
|---|

|[Figure 721]|
|---|

|[Figure 722]|
|---|

|[Figure 723]|
|---|

80%

80%

2048 x 2048

85%

95%

65%

95%

9.5

|[Figure 724]|
|---|

|[Figure 725]|
|---|

|[Figure 726]|
|---|

|[Figure 727]|
|---|

|[Figure 728]|
|---|

|[Figure 729]|
|---|

|[Figure 730]|
|---|

|[Figure 731]|
|---|

|[Figure 732]|
|---|

|[Figure 733]|
|---|

|[Figure 734]|
|---|

|[Figure 735]|
|---|

Semantic metadata (semantic classes)

Walkability score

Semantic diversity

Objectness score

10%

3

20%

|[Figure 736]|
|---|

|[Figure 737]|
|---|

|[Figure 738]|
|---|

|[Figure 739]|
|---|

|[Figure 740]|
|---|

|[Figure 741]|
|---|

|[Figure 742]|
|---|

|[Figure 743]|
|---|

|[Figure 744]|
|---|

50%

10

40%

|[Figure 745]|
|---|

|[Figure 746]|
|---|

|[Figure 747]|
|---|

|[Figure 748]|
|---|

|[Figure 749]|
|---|

|[Figure 750]|
|---|

|[Figure 751]|
|---|

|[Figure 752]|
|---|

|[Figure 753]|
|---|

90%

25

100%

|[Figure 754]|
|---|

|[Figure 755]|
|---|

|[Figure 756]|
|---|

|[Figure 757]|
|---|

|[Figure 758]|
|---|

|[Figure 759]|
|---|

|[Figure 760]|
|---|

|[Figure 761]|
|---|

|[Figure 762]|
|---|

Semantic metadata (instances)

COCO clutter score

COCO instance diversity

SAM clutter score

Crowdedness score

5

40%

10

- 2
- 3

|[Figure 763]|
|---|

|[Figure 764]|
|---|

|[Figure 765]|
|---|

|[Figure 766]|
|---|

|[Figure 767]|
|---|

|[Figure 768]|
|---|

|[Figure 769]|
|---|

|[Figure 770]|
|---|

|[Figure 771]|
|---|

|[Figure 772]|
|---|

|[Figure 773]|
|---|

|[Figure 774]|
|---|

10

50%

100

|[Figure 775]|
|---|

|[Figure 776]|
|---|

|[Figure 777]|
|---|

|[Figure 778]|
|---|

|[Figure 779]|
|---|

|[Figure 780]|
|---|

|[Figure 781]|
|---|

|[Figure 782]|
|---|

|[Figure 783]|
|---|

|[Figure 784]|
|---|

|[Figure 785]|
|---|

|[Figure 786]|
|---|

20

80%

300

10

|[Figure 787]|
|---|

|[Figure 788]|
|---|

|[Figure 789]|
|---|

|[Figure 790]|
|---|

|[Figure 791]|
|---|

|[Figure 792]|
|---|

|[Figure 793]|
|---|

|[Figure 794]|
|---|

|[Figure 795]|
|---|

|[Figure 796]|
|---|

|[Figure 797]|
|---|

|[Figure 798]|
|---|

- Figure 8: Steerable multimodal generation using metadata. This is an extension of fig. 4 and shows our model’s capability of generating multimodal data by conditioning on a wide set of controls. The common caption for all examples is "a painting of a bridge in a lush forest".

Varying SAM polygon instances

SAM polygon instance inputs & RGB generations

|[Figure 799]|
|---|

|[Figure 800]|
|---|

|[Figure 801]|
|---|

|[Figure 802]|
|---|

|[Figure 803]|
|---|

|[Figure 804]|
|---|

|[Figure 805]|
|---|

###### Caption input

A framed painting of mountains inside a bedroom

|[Figure 806]|
|---|

|[Figure 807]|
|---|

|[Figure 808]|
|---|

|[Figure 809]|
|---|

|[Figure 810]|
|---|

|[Figure 811]|
|---|

|[Figure 812]|
|---|

Color palette input

|[Figure 813]|
|---|

|[Figure 814]|
|---|

|[Figure 815]|
|---|

|[Figure 816]|
|---|

|[Figure 817]|
|---|

|[Figure 818]|
|---|

|[Figure 819]|
|---|

|[Figure 820]|
|---|

Color palette input

|[Figure 821]|
|---|

Varying color palette

Color palette inputs & RGB generations

|[Figure 822]|
|---|

|[Figure 823]|
|---|

|[Figure 824]|
|---|

|[Figure 825]|
|---|

|[Figure 826]|
|---|

|[Figure 827]|
|---|

|[Figure 828]|
|---|

|[Figure 829]|
|---|

|[Figure 830]|
|---|

|[Figure 831]|
|---|

Caption input

Outside view of an apartment

Normals input

|[Figure 832]|
|---|

|[Figure 833]|
|---|

|[Figure 834]|
|---|

|[Figure 835]|
|---|

|[Figure 836]|
|---|

|[Figure 837]|
|---|

|[Figure 838]|
|---|

|[Figure 839]|
|---|

|[Figure 840]|
|---|

|[Figure 841]|
|---|

|[Figure 842]|
|---|

- Figure 9: Probing with grounded generation. This is an extension of fig. 4 and further shows our model’s capability on performing generation by conditioning on multimodal input. The top row varies SAM instances and combines them with a fixed caption and color palette input. The bottom row fixes the normals and caption inputs and varies the color palette.

Caption input: a photo of a teddy bear made of water Caption input: vibrant portrait painting of Salvador Dalí with a robotic half face

|[Figure 843]|
|---|

|[Figure 844]|
|---|

|[Figure 845]|
|---|

|[Figure 846]|
|---|

|[Figure 847]|
|---|

|[Figure 848]|
|---|

|[Figure 849]|
|---|

|[Figure 850]|
|---|

|[Figure 851]|
|---|

|[Figure 852]|
|---|

|[Figure 853]|
|---|

|[Figure 854]|
|---|

|[Figure 855]|
|---|

|[Figure 856]|
|---|

|[Figure 857]|
|---|

|[Figure 858]|
|---|

|[Figure 859]|
|---|

|[Figure 860]|
|---|

|[Figure 861]|
|---|

|[Figure 862]|
|---|

|[Figure 863]|
|---|

|[Figure 864]|
|---|

|[Figure 865]|
|---|

|[Figure 866]|
|---|

4M-7 (from caption) 4M-21 (from caption) 4M-21 (from T5-XXL emb.)

4M-7 (from caption) 4M-21 (from caption) 4M-21 (from T5-XXL emb.)

Caption input: a drawing of a house on a mountain Caption input: the silhouette of the Milllenium Wheel at dusk

|[Figure 867]|
|---|

|[Figure 868]|
|---|

|[Figure 869]|
|---|

|[Figure 870]|
|---|

|[Figure 871]|
|---|

|[Figure 872]|
|---|

|[Figure 873]|
|---|

|[Figure 874]|
|---|

|[Figure 875]|
|---|

|[Figure 876]|
|---|

|[Figure 877]|
|---|

|[Figure 878]|
|---|

|[Figure 879]|
|---|

|[Figure 880]|
|---|

|[Figure 881]|
|---|

|[Figure 882]|
|---|

|[Figure 883]|
|---|

|[Figure 884]|
|---|

|[Figure 885]|
|---|

|[Figure 886]|
|---|

|[Figure 887]|
|---|

|[Figure 888]|
|---|

|[Figure 889]|
|---|

|[Figure 890]|
|---|

4M-7 (from caption) 4M-21 (from caption) 4M-21 (from T5-XXL emb.)

4M-7 (from caption) 4M-21 (from caption) 4M-21 (from T5-XXL emb.)

Caption input: a stop sign with a blue background Caption input: a cloud in the shape of a teacup

|[Figure 891]|
|---|

|[Figure 892]|
|---|

|[Figure 893]|
|---|

|[Figure 894]|
|---|

|[Figure 895]|
|---|

|[Figure 896]|
|---|

|[Figure 897]|
|---|

|[Figure 898]|
|---|

|[Figure 899]|
|---|

|[Figure 900]|
|---|

|[Figure 901]|
|---|

|[Figure 902]|
|---|

|[Figure 903]|
|---|

|[Figure 904]|
|---|

|[Figure 905]|
|---|

|[Figure 906]|
|---|

|[Figure 907]|
|---|

|[Figure 908]|
|---|

|[Figure 909]|
|---|

|[Figure 910]|
|---|

|[Figure 911]|
|---|

|[Figure 912]|
|---|

|[Figure 913]|
|---|

|[Figure 914]|
|---|

4M-7 (from caption) 4M-21 (from caption) 4M-21 (from T5-XXL emb.)

4M-7 (from caption) 4M-21 (from caption) 4M-21 (from T5-XXL emb.)

Caption input: a painting of black and white with a red border Caption input: a giant gorilla at the top of the Empire State Building

|[Figure 915]|
|---|

|[Figure 916]|
|---|

|[Figure 917]|
|---|

|[Figure 918]|
|---|

|[Figure 919]|
|---|

|[Figure 920]|
|---|

|[Figure 921]|
|---|

|[Figure 922]|
|---|

|[Figure 923]|
|---|

|[Figure 924]|
|---|

|[Figure 925]|
|---|

|[Figure 926]|
|---|

|[Figure 927]|
|---|

|[Figure 928]|
|---|

|[Figure 929]|
|---|

|[Figure 930]|
|---|

|[Figure 931]|
|---|

|[Figure 932]|
|---|

|[Figure 933]|
|---|

|[Figure 934]|
|---|

|[Figure 935]|
|---|

|[Figure 936]|
|---|

|[Figure 937]|
|---|

|[Figure 938]|
|---|

4M-7 (from caption) 4M-21 (from caption) 4M-21 (from T5-XXL emb.)

4M-7 (from caption) 4M-21 (from caption) 4M-21 (from T5-XXL emb.)

##### Figure 10: Text understanding. This is an extension of fig. 4 and further demonstrates improved text understanding capabilities of our method compared to 4M for several caption inputs.

#### B.2 Additional retrieval visualizations

Please see Figures 11 and 12 for additional qualitative results on RGB-to-Any and Any-to-RGB retrievals.

Query Top-3 Retrievals Top-3 Retrievals Top-3 Retrievals Top-3 Retrievals

|[Figure 939]|
|---|

|[Figure 940]|
|---|

|[Figure 941]|
|---|

|[Figure 942]|
|---|

|[Figure 943]|
|---|

|[Figure 944]|
|---|

|[Figure 945]|
|---|

|[Figure 946]|
|---|

|[Figure 947]|
|---|

|[Figure 948]|
|---|

|[Figure 949]|
|---|

|[Figure 950]|
|---|

|[Figure 951]|
|---|

|[Figure 952]|
|---|

|[Figure 953]|
|---|

|[Figure 954]|
|---|

|[Figure 955]|
|---|

|[Figure 956]|
|---|

|[Figure 957]|
|---|

|[Figure 958]|
|---|

|[Figure 959]|
|---|

|[Figure 960]|
|---|

|[Figure 961]|
|---|

|[Figure 962]|
|---|

|[Figure 963]|
|---|

|[Figure 964]|
|---|

|[Figure 965]|
|---|

|[Figure 966]|
|---|

|[Figure 967]|
|---|

|[Figure 968]|
|---|

|[Figure 969]|
|---|

|[Figure 970]|
|---|

|[Figure 971]|
|---|

|[Figure 972]|
|---|

|[Figure 973]|
|---|

|[Figure 974]|
|---|

|[Figure 975]|
|---|

|[Figure 976]|
|---|

|[Figure 977]|
|---|

|[Figure 978]|
|---|

|[Figure 979]|
|---|

|[Figure 980]|
|---|

|[Figure 981]|
|---|

|[Figure 982]|
|---|

|[Figure 983]|
|---|

|[Figure 984]|
|---|

|[Figure 985]|
|---|

|[Figure 986]|
|---|

|[Figure 987]|
|---|

|[Figure 988]|
|---|

|[Figure 989]|
|---|

|[Figure 990]|
|---|

|[Figure 991]|
|---|

|[Figure 992]|
|---|

|[Figure 993]|
|---|

|[Figure 994]|
|---|

|[Figure 995]|
|---|

|[Figure 996]|
|---|

|[Figure 997]|
|---|

|[Figure 998]|
|---|

|[Figure 999]|
|---|

|[Figure 1000]|
|---|

|[Figure 1001]|
|---|

|[Figure 1002]|
|---|

|[Figure 1003]|
|---|

|[Figure 1004]|
|---|

|[Figure 1005]|
|---|

|[Figure 1006]|
|---|

|[Figure 1007]|
|---|

|[Figure 1008]|
|---|

|[Figure 1009]|
|---|

|[Figure 1010]|
|---|

|[Figure 1011]|
|---|

|[Figure 1012]|
|---|

|[Figure 1013]|
|---|

|[Figure 1014]|
|---|

|[Figure 1015]|
|---|

|[Figure 1016]|
|---|

|[Figure 1017]|
|---|

|[Figure 1018]|
|---|

|[Figure 1019]|
|---|

|[Figure 1020]|
|---|

|[Figure 1021]|
|---|

|[Figure 1022]|
|---|

|[Figure 1023]|
|---|

|[Figure 1024]|
|---|

|[Figure 1025]|
|---|

|[Figure 1026]|
|---|

|[Figure 1027]|
|---|

|[Figure 1028]|
|---|

|[Figure 1029]|
|---|

|[Figure 1030]|
|---|

|[Figure 1031]|
|---|

|[Figure 1032]|
|---|

|[Figure 1033]|
|---|

|[Figure 1034]|
|---|

|[Figure 1035]|
|---|

|[Figure 1036]|
|---|

|[Figure 1037]|
|---|

|[Figure 1038]|
|---|

|[Figure 1039]|
|---|

|[Figure 1040]|
|---|

|[Figure 1041]|
|---|

|[Figure 1042]|
|---|

|[Figure 1043]|
|---|

|[Figure 1044]|
|---|

|[Figure 1045]|
|---|

|[Figure 1046]|
|---|

|[Figure 1047]|
|---|

|[Figure 1048]|
|---|

|[Figure 1049]|
|---|

|[Figure 1050]|
|---|

|[Figure 1051]|
|---|

|[Figure 1052]|
|---|

|[Figure 1053]|
|---|

|[Figure 1054]|
|---|

|[Figure 1055]|
|---|

|[Figure 1056]|
|---|

|[Figure 1057]|
|---|

|[Figure 1058]|
|---|

|[Figure 1059]|
|---|

|[Figure 1060]|
|---|

|[Figure 1061]|
|---|

|[Figure 1062]|
|---|

|[Figure 1063]|
|---|

|[Figure 1064]|
|---|

|[Figure 1065]|
|---|

|[Figure 1066]|
|---|

|[Figure 1067]|
|---|

|[Figure 1068]|
|---|

- Figure 11: RGB-to-Any retrieval. This is an extension of fig. 5 and further demonstrates cross-modal retrieval capabilities of our model. Here our model successfully retrieves several modalities (RGB, depth, normals, segmentation) using the RGB image as the query input.

Depth-to-RGB retrieval Normals-to-RGB retrieval Segmentation-to-RGB retrieval Caption-to-RGB retrieval

Query Top-3 Retrievals Query Top-3 Retrievals

Query Top-3 Retrievals Query Top-3 Retrievals Query Top-3 Retrievals

|[Figure 1069]|
|---|

|[Figure 1070]|
|---|

|[Figure 1071]|
|---|

|[Figure 1072]|
|---|

|[Figure 1073]|
|---|

|[Figure 1074]|
|---|

|[Figure 1075]|
|---|

|[Figure 1076]|
|---|

|[Figure 1077]|
|---|

|[Figure 1078]|
|---|

|[Figure 1079]|
|---|

|[Figure 1080]|
|---|

|[Figure 1081]|
|---|

|<person> showing mountains as well as a family|
|---|

|[Figure 1082]|
|---|

|[Figure 1083]|
|---|

|[Figure 1084]|
|---|

|[Figure 1085]|
|---|

|[Figure 1086]|
|---|

|[Figure 1087]|
|---|

|[Figure 1088]|
|---|

|[Figure 1089]|
|---|

|[Figure 1090]|
|---|

|[Figure 1091]|
|---|

|[Figure 1092]|
|---|

|[Figure 1093]|
|---|

|[Figure 1094]|
|---|

|[Figure 1095]|
|---|

|[Figure 1096]|
|---|

|[Figure 1097]|
|---|

|[Figure 1098]|
|---|

|pile canvas print featuring the photograph colorful glass marbles by<br><br><person>|
|---|

|[Figure 1099]|
|---|

|[Figure 1100]|
|---|

|[Figure 1101]|
|---|

|[Figure 1102]|
|---|

|[Figure 1103]|
|---|

|[Figure 1104]|
|---|

|compaction in about 5 cms of snow left behind a snow tire, showing tread-snow interaction|
|---|

|[Figure 1105]|
|---|

|[Figure 1106]|
|---|

|[Figure 1107]|
|---|

|[Figure 1108]|
|---|

|[Figure 1109]|
|---|

|[Figure 1110]|
|---|

|[Figure 1111]|
|---|

|[Figure 1112]|
|---|

|[Figure 1113]|
|---|

|[Figure 1114]|
|---|

|[Figure 1115]|
|---|

|[Figure 1116]|
|---|

|forbidden fruit: a warning sign outside the kula lodge in <person>|
|---|

|[Figure 1117]|
|---|

|[Figure 1118]|
|---|

|[Figure 1119]|
|---|

|[Figure 1120]|
|---|

|[Figure 1121]|
|---|

|[Figure 1122]|
|---|

|[Figure 1123]|
|---|

|[Figure 1124]|
|---|

|[Figure 1125]|
|---|

|[Figure 1126]|
|---|

|[Figure 1127]|
|---|

|[Figure 1128]|
|---|

|[Figure 1129]|
|---|

|[Figure 1130]|
|---|

|[Figure 1131]|
|---|

|[Figure 1132]|
|---|

|[Figure 1133]|
|---|

|[Figure 1134]|
|---|

|topiary garden with many topiary animals topiary dog and bear|
|---|

|[Figure 1135]|
|---|

|[Figure 1136]|
|---|

|[Figure 1137]|
|---|

|[Figure 1138]|
|---|

|[Figure 1139]|
|---|

|[Figure 1140]|
|---|

|[Figure 1141]|
|---|

|[Figure 1142]|
|---|

|[Figure 1143]|
|---|

|[Figure 1144]|
|---|

|[Figure 1145]|
|---|

|[Figure 1146]|
|---|

|weekly meal planmain dishes and dessert recipe for the entire week. so yummy!|
|---|

|[Figure 1147]|
|---|

|[Figure 1148]|
|---|

|[Figure 1149]|
|---|

|[Figure 1150]|
|---|

|[Figure 1151]|
|---|

|[Figure 1152]|
|---|

|[Figure 1153]|
|---|

|[Figure 1154]|
|---|

|[Figure 1155]|
|---|

|[Figure 1156]|
|---|

|[Figure 1157]|
|---|

|[Figure 1158]|
|---|

|[Figure 1159]|
|---|

|[Figure 1160]|
|---|

|[Figure 1161]|
|---|

|[Figure 1162]|
|---|

|[Figure 1163]|
|---|

|[Figure 1164]|
|---|

|warsaw, poland: palace of culture<br><br>and science, the tallest building in poland|
|---|

|[Figure 1165]|
|---|

|[Figure 1166]|
|---|

|[Figure 1167]|
|---|

|[Figure 1168]|
|---|

|[Figure 1169]|
|---|

|[Figure 1170]|
|---|

|[Figure 1171]|
|---|

|[Figure 1172]|
|---|

|[Figure 1173]|
|---|

|[Figure 1174]|
|---|

|[Figure 1175]|
|---|

|[Figure 1176]|
|---|

|[Figure 1177]|
|---|

|[Figure 1178]|
|---|

|[Figure 1179]|
|---|

|firework explosion against a black sky at night|
|---|

|[Figure 1180]|
|---|

|[Figure 1181]|
|---|

|[Figure 1182]|
|---|

|[Figure 1183]|
|---|

|[Figure 1184]|
|---|

|[Figure 1185]|
|---|

|[Figure 1186]|
|---|

|[Figure 1187]|
|---|

|[Figure 1188]|
|---|

|[Figure 1189]|
|---|

|[Figure 1190]|
|---|

|[Figure 1191]|
|---|

|a bed or beds in a room at the roost room 12|
|---|

|[Figure 1192]|
|---|

|[Figure 1193]|
|---|

|[Figure 1194]|
|---|

|[Figure 1195]|
|---|

|[Figure 1196]|
|---|

|[Figure 1197]|
|---|

|[Figure 1198]|
|---|

|[Figure 1199]|
|---|

|[Figure 1200]|
|---|

|[Figure 1201]|
|---|

|[Figure 1202]|
|---|

|[Figure 1203]|
|---|

|[Figure 1204]|
|---|

|[Figure 1205]|
|---|

|[Figure 1206]|
|---|

|[Figure 1207]|
|---|

|vinci airports became one of the world’s top five airport operators|
|---|

|[Figure 1208]|
|---|

|[Figure 1209]|
|---|

|[Figure 1210]|
|---|

|[Figure 1211]|
|---|

|[Figure 1212]|
|---|

|[Figure 1213]|
|---|

|[Figure 1214]|
|---|

|[Figure 1215]|
|---|

|[Figure 1216]|
|---|

|[Figure 1217]|
|---|

|[Figure 1218]|
|---|

- Figure 12: Any-to-RGB retrieval. This is an extension of fig. 5 and further demonstrates cross-modal retrieval capabilities of our model. Here our model successfully retrieves RGB images when the query inputs are from depth, normals, segmentation, and caption modalities.

### C Additional Ablations

#### C.1 Ablation of pre-training data and modalities

For training 4M-21, we initialize the training using 4M models that we pre-trained on COYO700M [11]. We ablate in Table 4 different choices of training data and modalities. We can see that performing cotraining on C4 [68] and COYO700M [11] has the potential to slightly improve transfer performance on average.

- Table 4: Pre-training data and modality mixture ablation: We ablate different pre-training modality and dataset choices on B models. * represents the models initialized from the corresponding 4M models trained on COYO700M.

Pre-training ImageNet-1K ADE20K NYUv2 depth ARKitScenes data Top-1 acc. ↑ mIoU ↑ δ1 acc. ↑ AP3D ↑

Method

4M-7 B [62] CC12M 84.5 50.1 92.0 40.3 4M-7 B COYO700M 84.4 49.4 91.4 38.6 4M-7 B * CC12M 84.5 49.2 91.0 39.5

4M-21 B * CC12M 84.4 49.2 90.9 40.0 4M-21 B * CC12M+C4 84.6 49.5 90.4 41.2 4M-21 B * CC12M+COYO700M+C4 84.5 50.1 90.8 42.4

#### C.2 Ablation of ensembling the predictions

Unlike the deterministic pseudo labeler and other state of the art networks we compared against in table 1, our model can produce multiple prediction given the same RGB input through repeated sampling with a different seed. As shown in Table 5, ensembling ten samples of predicted surface normals and semantic segmentation maps can significantly improve the reported metrics. While ensembling improves upon these metrics, we note that the ensembled predictions can be comparatively blurrier around object edges than any individual sample.

- Table 5: Ensembling ablation: We ablate ensembling multiple predictions on DIODE normals and COCO semantic segmentation compared to no ensembling. As the results suggest, ensembling in all cases improves the quantitative results.

DIODE Normals COCO Semseg mean angle error ↓ mean IoU ↑

Method No Ensemble Ensemble No Ensemble Ensemble

4M-21 B 22.3 21.7 39.0 42.5 4M-21 L 21.7 21.1 43.8 46.4 4M-21 XL 21.3 20.8 46.5 48.1

### D Multimodal Dataset & Tokenization Details

#### D.1 Pseudo labeled multimodal training dataset

Similar to 4M, to have an aligned multimodal dataset, we pseudo label the CC12M dataset using strong specialized models for each task. The pseudo labeling of existing modalities is done in the same fashion as 4M, using Omnidata DPT-Hybrid [70] for surface normals and depth estimation, COCO Mask2Former [19] with a SwinB [56] backbone for semantic segmentation, COCO ViTDet ViT-H model [54] initialized from MAE weights [38] for bounding boxes, and CLIP-B16 [67] with ViT-B/16 visual backbone backbone for CLIP feature maps.

- 3D human poses. We use 4D-Humans [35] to extract 3D pose and shape parameterized by an SMPL model. For the images in CC12M without humans, we set the pose label to a “none" token. For the images with humans, we form a sequence by concatenating the bounding box, body pose, camera, and shape values in a sequence for each human instance. As data augmentation, we randomly shuffle the order of each component in the sequence.

SAM instances. Besides semantic segmentation and bounding boxes, SAM [47] instance segmentation also provides some level of semantic information from an image by clustering together semantically similar pixels in it. Unlike semantic segmentation, SAM instances are not restricted to a specific set of classes and can segment in more detail. We use the SAM H model and query it with points in a grid format to obtain the instances. We also considered the SAM-HQ [45] H model, however in the grid-point querying format, it yields very similar results to SAM. We found 32 × 32 query points to be the optimal choice both for pseudo labeling speed and quality.

DINOv2 and ImageBind global features & feature maps. We extract both dense feature maps and global embeddings, i.e. cls token embeddings, from DINOv2-B14 [65] and ImageBind-H14 [33] pre-trained models. For the latter, we only extracted the image embeddings, incorporating other modality embeddings such as thermal or audio could be interesting future work.

T5-XXL embeddings. Language model embeddings, such as from T5-XXL [68], have been shown to improve the generation fidelity and text understanding capabilities of text-to-image generative models [76, 13]. Consequently, we use the T5-XXL encoder to extract text embeddings from all CC12M captions, without any preprocessing of the text. Unlike other modalities, we do not convert these text embeddings to a sequence of discrete tokens or treat them as targets (similar to the RGB pixel modality variant). Instead, we only provide them as inputs using a linear projection from the T5-XXL embedding dimension (dT5-XXL = 4096) to our model’s embedding dimension.

Image metadata. From RGB images, we directly extract different types of metadata like the original height and width before cropping [66], brightness, contrast, saturation and entropy. We additionally extract a notion of colorfulness, following [37].

Semantic metadata. We compute the crowdedness score as the number of humans in the pseudo labeled human poses, the SAM clutter score as the number of SAM instances, the COCO clutter score as the number of COCO instances, the COCO instance diversity as the number of unique COCO instance classes, and the semantic diversity as the number of unique COCO semantic classes in an image. For caption length, we count the number of characters, words, and sentences. As objectness score, we count the percentage of pixels in the COCO semantic segmentation map that belong to countable classes (indices 87, 90, 96, 97, 98, 99, 100, 101, 102, 103, 105, 106, 109, 110,

111, 112, 113, 117, 118, 119, 122, 123, 124, 125, 126, 129, 131, 132), and for the walkability score we count classes such as ‘road’ (indices 87, 90, 97, 100, 102, 105, 106, 122, 123, 125, 126, 132).

Geometric metadata. To compute the occlusion score, we first generate occlusion edges from depth images by applying a Sobel filter, followed by counting the percentage of occlusion edge pixels that surpass a threshold of 0.3. As a notion of geometric complexity, we project surface normal pixels onto the unit sphere, and compute their angular variance. Note that images of indoor scenes or caves featuring large surfaces pointing in all different directions receive a high score in this metric, while ones with a more localized geometric variance get a somewhat lower score. Exploring other potential notions of geometric complexity can be an interesting future addition.

Color palette. For every RGB image, we extract between one and seven color palettes using PyPalette [2]. During training, we randomly sample one of the color palettes to enable users to input palettes with different levels of granularity.

SAM edges and canny edges. Edges are a convenient way of grounding image generation on shapes contained in images [98]. To pseudo label edges, we apply the OpenCV canny edge detector on SAM instance maps and RGB, to obtain SAM edges and canny edges respectively.

#### D.2 Tokenization of human poses

We use a BottleneckMLP [6] with 6 blocks and 1024 width to compress pose into 8 tokens. We use 1024 vocabulary size, and trained using smooth L1 loss for 15 epochs on CC12M training data. We also binned the global orientation, body shape, and bounding boxes into 1000 discrete bins similar to [16]. The final sequence is obtained by also adding identifiers, i.e. “bbox”, “pose”, “shape”, before the corresponding sub-sequence.

#### D.3 Tokenization of SAM instances

The SAM instance tokenizer is a ViT-based VQ-VAE that tokenizes 64 × 64 binary masks into 16 tokens using a vocabulary size of 1024. The tokenizer is trained using the cross-entropy loss for 24 epochs on CC12M training data, by resizing individual masks into a square aspect ratio image of 64 × 64 pixels. To preserve the SAM instances’ original location, width, and height in the image, their bounding boxes are extracted. The final sequence for each instance is formed by appending the identifier “polygon” to 4 numbers that specify the bounding box of the instance, along with the 16 token IDs.

#### D.4 Tokenization of global feature maps

Similar to human poses, we use BottleneckMLP with 6 blocks and 1024 width to compress DINOv2B14 and ImageBind-H14 global embeddings into 16 tokens. We use 8192 vocabulary size, and trained using cosine similarity loss for 15 epochs.

#### D.5 Tokenization of dense feature maps

We follow [62] and tokenize CLIP-B16, DINOv2-B14, and ImageBind-H14 dense feature maps into 196, 256, and 256 tokens, respectively, using a ViT-based VQVAE with 8192 vocabulary size and smooth L1 loss.

#### D.6 Tokenization of sequence modalities

We tokenize text, color palette, metadata, and bounding boxes using a WordPiece tokenizer by fitting it on all captions and 4000 “special value” tokens, with a joint vocabulary size of 30k. These special tokens are divided into four groups, each with 1000 values, i.e. v0=0, v0=1, ..., v0=999, v1=0, v1=1, ... v1=999, v2=0, v2=1, ...,

v2=999, v3=0, v3=1, ..., v3=999. For bounding boxes, we follow 4M [62] and represent xmin, ymin, xmax, ymax coordinates using v0, v1, v2, v3 tokens respectively. Other modalities are tokenized by binning their values into corresponding bins, e.g. color palette sequence is formed as color = c R = r G = g B = b R = r,... where c takes a value between 1 and 7 and

specifies the number of colors in the palette and r,g,b takes values between 0-255. We chose to model metadata using interleaved pairs of special tokens, where the first one specifies the type of metadata modality, and the second specifies its value. For example, a crowdedness score of 3 and a brightness of 120 would be specified as the sequence v1=5 v0=3 v1=10 v0=120. During training the number of metadata entries and their order is randomized. All of this results in a sequence prediction formulation, following [62, 16].

#### D.7 Tokenization of Canny and SAM edges

We use a VQ-VAE with a diffusion decoder, similar to [62] to tokenize the edge modalities. We use the same tokenizer as it reconstructs both edges similarly well.

### E Training Details

Please see Tab. 6 for an overview of pre-training settings. For more accurate model comparisons, the architecture and overall training objective of our B, L, and XL models are the same as those of 4M models. However, we do modify and improve various aspects of the training process that allow us to significantly increase the number of training modalities. These changes concern modality-specific accommodations to the masking strategy, the ability to co-train on several datasets, and the use of a more diversified multimodal masking strategy. We describe these modifications below:

#### E.1 Modality-specific accommodations

Positional and modality embeddings. As with 4M, 4M-21 incorporates both learnable modality embeddings and fixed sine-cosine positional embeddings for each modality. The positional embeddings are either 1D or 2D depending on the modality type.

Metadata grouping and chunk-based masking. To address the sparsity and number of different types of metadata, the metadata modalities are all grouped together as a single modality during training. This prevents the over-allocation of tokens to sparse metadata, enabling a more balanced distribution of the token budget across modalities. However, the standard span masking from T5 [68] and 4M [62] performs random uniform masking at the token level, which can lead to pre-training inefficiencies [51] and make conditioning on specific metadata difficult, as conditioning on just one of them would rarely occur during pre-training with this masking strategy. Instead, we propose to mask chunks of sequence (similar to PMI-Masking [51]), where the span masking is performed per chunk of metadata instead of at the token level.

#### E.2 Multidataset co-training and diversified multimodal masking strategy

Multi-dataset support. Unlike 4M which was only trained on a single aligned dataset, we train 4M-21 on multiple datasets simultaneously. This flexibility allows for the inclusion of datasets with varying numbers of modalities, which enables training on both large-scale datasets with a smaller number of modalities and smaller datasets with a larger diversity of modalities.

Sampling and masking strategies. Our data sampling process involves selecting a training dataset based on its sampling weight, followed by choosing a masking strategy from the dataset-specific mixture of masking strategies. Input and target tokens are then sampled using the selected strategy.

Co-training datasets. We co-train on several datasets to improve the model’s performance and the data diversity. These include CC12M [15], which comprises about 10 million text-image samples fully pseudo labeled with all 21 modalities, and accounts for 60% of our training samples. Additionally, we include COYO700M [11], with approximately 500 million text-image samples pseudo labeled with the 7 modalities of 4M, and accounts for 20% of our training samples. Lastly, the Colossal Clean Crawled Corpus (C4) [68], a large text-only dataset, is used for language model co-training, also making up 20% of our training samples.

Diverse mixture of masking strategies. As with 4M [62], the masking strategy is governed by Dirichlet distribution with parameter α. This distribution influences the sampling of tokens from modalities: a lower α results in samples dominated by one modality, while a higher α leads a more balanced representation across all modalities. For both CC12M and COYO datasets, we implement

multiple masking strategies to cater to specific training needs, and randomly sample from them for every sample in the batch:

- • All-to-all masking: Involves four masking strategies with symmetric input and target α set to 0.01, 0.1, 1.0, and 10.0 respectively.
- • RGB-to-all masking: Consists of only RGB tokens as input, with target α all set to 0.5.
- • Caption-biased masking: Includes two strategies, heavily skewed towards either unmasked captions or T5-XXL embeddings as input. These masking strategies are particularly beneficial for tasks involving text-to-image generation

- Table 6: Pre-training settings. Training configuration for 4M-21 used in the transfer experiments and generation results.

Configuration 4M-21 B 4M-21 L 4M-21 XL Weight initialization 4M (COYO) Training length (n tokens) 500B Warmup length (n tokens) 10B Optimizer AdamW [57] Opt. momentum β1,β2 = 0.9,0.95 Base learning rate [36] 1e-4 1e-4 2e-5 Batch size 8192 Weight decay 0.05 Gradient clipping ✗ ✗ 3.0 Learning rate schedule Cosine decay Feedforward activation SwiGLU [77]

Input token budget 256 Target token budget 256 Input and target α Mixture (see Sec. E.2) Masking strategy Mixture (see Sec. E.2) Dataset Mixture (see Sec. E.2)

Image resolution 2242 Augmentation None (Center Crop) Repeated sampling [30] 4 Data type bfloat16 [10]

### F Out-of-the-box Evaluation Details

Below, we provide further details on out-of-the-box evaluations we performed. Please also see fig. 13 for a qualitative comparison between our XL model and Unified-IO XL [59], as well as Unified-IO 2 XXL [58]. Furthermore, table 7 compares Unified-IO, Unified-IO 2, and our model’s out-of-thebox capabilities on surface normal estimation, depth estimation, and semantic segmentation. As demonstrated, our model outperforms Unified-IO and Unified-IO 2 in all the mentioned tasks.

#### F.1 Surface normal and depth estimation on DIODE

We follow the evaluation setup in [62] and evaluate on DIODE validation set at 224 × 224 input resolution.

#### F.2 Semantic and instance segmentation on COCO

We employ a similar approach as SAM [47] by querying our model on the bounding boxes to obtain the instances. To predict the instances, only the target bounding box is provided in the input final sequence, and the tokens are masked for our model to predict them.

DIODE depth

|[Figure 1219]|
|---|

|[Figure 1220]|
|---|

|[Figure 1221]|
|---|

|[Figure 1222]|
|---|

|[Figure 1223]|
|---|

|[Figure 1224]|
|---|

RGB Input

|[Figure 1225]|
|---|

|[Figure 1226]|
|---|

|[Figure 1227]|
|---|

|[Figure 1228]|
|---|

|[Figure 1229]|
|---|

|[Figure 1230]|
|---|

Ground Truth

|[Figure 1231]|
|---|

|[Figure 1232]|
|---|

|[Figure 1233]|
|---|

|[Figure 1234]|
|---|

|[Figure 1235]|
|---|

|[Figure 1236]|
|---|

4M-21 XL

|[Figure 1237]|
|---|

|[Figure 1238]|
|---|

|[Figure 1239]|
|---|

|[Figure 1240]|
|---|

|[Figure 1241]|
|---|

|[Figure 1242]|
|---|

Unified-IO XL

|[Figure 1243]|
|---|

|[Figure 1244]|
|---|

|[Figure 1245]|
|---|

|[Figure 1246]|
|---|

|[Figure 1247]|
|---|

|[Figure 1248]|
|---|

Unified-IO 2 XXL

DIODE surface normals

|[Figure 1249]|
|---|

|[Figure 1250]|
|---|

|[Figure 1251]|
|---|

|[Figure 1252]|
|---|

|[Figure 1253]|
|---|

|[Figure 1254]|
|---|

RGB Input

|[Figure 1255]|
|---|

|[Figure 1256]|
|---|

|[Figure 1257]|
|---|

|[Figure 1258]|
|---|

|[Figure 1259]|
|---|

|[Figure 1260]|
|---|

Ground Truth

|[Figure 1261]|
|---|

|[Figure 1262]|
|---|

|[Figure 1263]|
|---|

|[Figure 1264]|
|---|

|[Figure 1265]|
|---|

|[Figure 1266]|
|---|

4M-21 XL

|[Figure 1267]|
|---|

|[Figure 1268]|
|---|

|[Figure 1269]|
|---|

|[Figure 1270]|
|---|

|[Figure 1271]|
|---|

|[Figure 1272]|
|---|

Unified-IO XL

|[Figure 1273]|
|---|

|[Figure 1274]|
|---|

|[Figure 1275]|
|---|

|[Figure 1276]|
|---|

|[Figure 1277]|
|---|

|[Figure 1278]|
|---|

Unified-IO 2 XXL

COCO semantic segmentation maps

|[Figure 1279]|
|---|

|[Figure 1280]|
|---|

|[Figure 1281]|
|---|

|[Figure 1282]|
|---|

|[Figure 1283]|
|---|

|[Figure 1284]|
|---|

RGB Input

|[Figure 1285]|
|---|

|[Figure 1286]|
|---|

|[Figure 1287]|
|---|

|[Figure 1288]|
|---|

|[Figure 1289]|
|---|

|[Figure 1290]|
|---|

Ground Truth

|[Figure 1291]|
|---|

|[Figure 1292]|
|---|

|[Figure 1293]|
|---|

|[Figure 1294]|
|---|

|[Figure 1295]|
|---|

|[Figure 1296]|
|---|

4M-21 XL

|[Figure 1297]|
|---|

|[Figure 1298]|
|---|

|[Figure 1299]|
|---|

|[Figure 1300]|
|---|

|[Figure 1301]|
|---|

|[Figure 1302]|
|---|

Unified-IO XL

|[Figure 1303]|
|---|

|[Figure 1304]|
|---|

|[Figure 1305]|
|---|

|[Figure 1306]|
|---|

|[Figure 1307]|
|---|

|[Figure 1308]|
|---|

Unified-IO 2 XXL

- Figure 13: Comparing 4M-21 XL, Unified-IO XL [59], and Unified-IO 2 XXL [58] out-of-the-box. 4M-21 XL demonstrates strong generalization to inputs from different datasets and tasks out-of-thebox (zero shot), significantly improving over Unified-IO 1 and 2.

- Table 7: Out-of-the-box capabilties. Comparison between Unified-IO 2 and our model out-of-thetask capabilities across surface normal estimation, depth estimation, and semantic segmentation. We use the L1 score as the metric for surface normal and depth estimation, and mean IoU for semantic segmentation.

Method Normals ↓ Depth ↓ Sem. seg. ↑

Unified-IO B [59] 35.7 1.00 32.9 Unified-IO L 33.9 0.87 41.6 Unified-IO XL 31.0 0.82 44.3

Unified-IO 2 L [58] 37.1 0.96 38.9 Unified-IO 2 XL 34.8 0.86 39.7 Unified-IO 2 XXL 37.4 0.84 41.7

4M-21 B 21.7 0.71 42.5 4M-21 L 21.1 0.69 46.4 4M-21 XL 20.8 0.68 48.1

- F.3 kNN retrieval on ImageNet-1K We follow the evaluation setup from DINOv2 [65]and set k = 20 and temperature to 0.07.
- F.4 3D human pose prediction on 3DPW

We follow the evaluation implemented in the 4D-Humans [35] codebase, with the difference that we use 224 × 224 as input image resolution as opposed to 256 × 256.

G Transfer Evaluation Details

We provide the transfer settings in Tables 8, 9, 10. We also note that after an extensive hyper parameter search for the DINOv2-g baseline on NYUv2, using a ConvNeXt head, it achieved only 92.5 δ1 acc., which is lower than the reported 95.0 with frozen encoder and DPT head.

- Table 8: Image classification settings. Configuration for intermediate fine-tuning on ImageNet-21K and fine-tuning on ImageNet-1K, the settings follow MultiMAE [7] and 4M [62].

Configuration ImageNet-21K ImageNet-1K

Base Large XL Base Large XL Fine-tuning epochs 20 50 20 20 Warmup epochs 2 2 Optimizer AdamW [57] AdamW [57] Opt. momentum β1,β2 = 0.9,0.95 β1,β2 = 0.9,0.999 Base learning rate [36] 1e-4 1e-4 5e-5 1e-4 Batch size 4096 4096 4096 1024 Weight decay 0.05 0.05 Learning rate schedule Cosine decay Cosine decay Layer-wise lr decay [20] 0.75 0.85 0.85 0.75 0.85 0.85 Drop path [41] 0.1 0.2 0.4 0.1 0.2 0.4

Input resolution 2242 2242 Augmentation RandAug(9, 0.5) [21] RandAug(9, 0.5) [21] Random resized crop (0.5, 1) (0.08, 1) Label smoothing ε 0.1 0.1 Mixup [97] 0.1 0.1 Cutmix [95] 1.0 1.0

- Table 9: Semantic segmentation settings. Configuration for semantic segmentation fine-tuning on ADE20K, the settings follow MultiMAE [7] and 4M [62].

Configuration ADE20K

Base Large XL Fine-tuning epochs 64 64 128 Warmup epochs 1 Optimizer AdamW [57] Opt. momentum β1,β2 = 0.9,0.999 Learning rate 2e-4 2e-4 3e-4 Batch size 64 Weight decay 0.05 Learning rate schedule Cosine decay Layer-wise lr decay [20] 0.75 0.85 0.95 Drop path [41] 0.1 0.2 0.3 LoRA [40] rank / scale ✗ ✗ 64 / 1.0

Input resolution 5122 Augmentation Large-scale jitter (LSJ) [32] Color jitter ✓

- Table 10: Depth estimation settings. Configuration for depth estimation fine-tuning on NYUv2, the settings follow MultiMAE [7] and 4M [62].

NYUv2 Configuration Base Large XL Fine-tuning epochs 1000 Warmup epochs 100 Optimizer AdamW [57] Opt. momentum β1,β2 = 0.9,0.999 Learning rate 1e-4 1e-4 5e-5 Batch size 128 128 16 Weight decay 1e-4 Learning rate schedule Cosine decay Layer-wise lr decay [20] 0.75 0.85 0.9 Drop path [41] 0.1 0.2 0.0 LoRA [40] rank / scale ✗ ✗ 8 / 1.0 Input resolution 2562 Random crop ✓ Color jitter ✓

### H Investigating Different Tokenization Schemes

As we develop several tokenization strategies for each modality, ablating their performance against all possible design choices would be prohibitively expensive. Thus, we focus on one modality, namely SAM instances, and provide a more detailed look into the impact of different tokenization strategies. We study two approaches for SAM instances: path tokenization and mask tokenization.

Path tokenization: We represent each instance in the image as a list of polygon coordinates. Then we tokenize these coordinates using a Bottleneck MLP-based VQ-VAE tokenizer. To achieve a fixed-size input, the polygons are either simplified or extended to have the same number of corner points. We found that fixing the maximum number of corners to 128 results in a minimal change in the overall polygon shape, thus we use this value for all the path tokenization ablations.

Mask tokenization: In this scheme, we first convert each instance to a binary masks and resize them to a fixed mask size. Then, we tokenize them using a ViT-based VQ-VAE tokenizer, similar to the way we tokenize image-like and feature map modalities.

Path Tokenization Mask Tokenization

Number of Tokens

Vocabulary Size

Loss

Mask Size

100.0

100.0

100.0

100.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

97.5

97.5

97.5

97.5

95.0

95.0

95.0

95.0

92.5

92.5

92.5

92.5

IoU

90.0

90.0

90.0

90.0

87.5

87.5

87.5

87.5

85.0

85.0

85.0

85.0

32 × 32 64 × 64* 128 × 128

9 16* 25

512 1024* 2048

L1 MSE CE* Dice

- Figure 14: Ablating tokenization choices: We ablate the impact of different tokenization choices. Performance is reported as reconstruction IoU on CC12M validation set. * shows the mask tokenization configuration we used in the final tokenizer. See appendix H for details.

Ablations: We investigated L1 and MSE losses for both tokenization schemes, and additionally cross-entropy and Dice loss for the mask tokenization. We also investigated the effects of the total number of tokens, token vocabulary size, and mask size. To compare the performance of the resulting tokenizers, we use the IoU between the pseudo-labeled and reconstructed instances as our metric. fig. 14 illustrates the results of different ablated configurations. For each configuration, the remaining unspecified parameters are by default set to 16 for the number of tokens, 1024 for the vocabulary size, L1 for the loss, and 64 × 64 for the mask size. The ablations show that using mask tokenization with 16 tokens, 1024 vocabulary size, and 64 × 64 mask size performs well and sets a good balance between reconstruction quality and total sequence length.

In all ablations, the tokenizers are trained for 24 epochs starting with 5 warmup epochs using the AdamW [57] optimizer with β1,β2 = 0.9,0.999 and a batch size of 128. For all the experiments except the Dice loss, a learning rate of 1e-5 is used. Since using this learning rate for the Dice loss experiment resulted in instabilities, we reduced its learning rate to 1e-6. As demonstrated in fig. 15, increasing the number of tokens results in better reconstruction quality both for the mask tokenizer and the path tokenizer. Compared to L1 loss, the cross-entropy loss training obtains reconstructions with smoother edges and better coverage.

|[Figure 1309]|
|---|

|[Figure 1310]|
|---|

|[Figure 1311]|
|---|

|[Figure 1312]|
|---|

|[Figure 1313]|
|---|

|[Figure 1314]|
|---|

|[Figure 1315]|
|---|

|[Figure 1316]|
|---|

|[Figure 1317]|
|---|

|[Figure 1318]|
|---|

|[Figure 1319]|
|---|

|[Figure 1320]|
|---|

|[Figure 1321]|
|---|

|[Figure 1322]|
|---|

|[Figure 1323]|
|---|

|[Figure 1324]|
|---|

|[Figure 1325]|
|---|

|[Figure 1326]|
|---|

|[Figure 1327]|
|---|

|[Figure 1328]|
|---|

|[Figure 1329]|
|---|

|[Figure 1330]|
|---|

|[Figure 1331]|
|---|

|[Figure 1332]|
|---|

|[Figure 1333]|
|---|

|[Figure 1334]|
|---|

|[Figure 1335]|
|---|

|[Figure 1336]|
|---|

|[Figure 1337]|
|---|

|[Figure 1338]|
|---|

|[Figure 1339]|
|---|

|[Figure 1340]|
|---|

Instance ground truth

Mask tokenizer Path tokenizer L1 loss

3x3 tokens 4x4 tokens 4x4 tokens 9 tokens 16 tokens 25 tokens

Cross entropy L1 loss 5x5 tokens

- Figure 15: Different tokenization schemes for SAM instances. We compare different tokenization schemes to tokenize SAM instances for pre-training. Please see Sec. H for details.

### I Broader Impact

#### I.1 Computational costs

All models were trained on Nvidia A100 GPUs. The 4M-21 B model was trained for 2 days using 64 A100s. The 4M-21 L model was trained for 4 days using 128 A100s. The largest 4M-21 XL model required 11 days using 128 A100s. Fine-tuning and transfer learning experiments for each model used approximately 20% additional compute compared to its pre-training. Training the various tokenizers (RGB, depth, normals, CLIP, DINOv2, ImageBind, semantic segmentation, SAM edges, and Canny edge detection, SAM instances, and 3D human poses) required roughly 5 days using 8 A100s each, totaling approximately 60 A100-days. In total, the primary experiments reported in the paper used approximately 120’000 A100-hours, not including additional preliminary experiments and ablations. We estimate the total compute for the full research project, including preliminary and unreported experiments, to be 150’000 A100-hours.

#### I.2 Social impact

We are open sourcing our code and models to support researchers with the democratization of the tools and to enable transparent inspection and safeguarding. 4M-21 models are trained on publicly available datasets with some curation, e.g. people’s names are redacted in CC12M [15]. However, this process is still noisy, hence we advise caution when using the models for generation.

