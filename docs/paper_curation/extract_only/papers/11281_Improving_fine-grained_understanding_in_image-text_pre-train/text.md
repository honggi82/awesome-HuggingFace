## arXiv:2401.09865v1[cs.CV]18Jan2024

[Figure 1]

2024-1-19

# Improving fine-grained understanding in image-text pre-training

###### Ioana Bica1, Anastasija Ilić1, Matthias Bauer1, Goker Erdogan1, Matko Bošnjak1, Christos Kaplanis1, Alexey A. Gritsenko1, Matthias Minderer1, Charles Blundell1, Razvan Pas,canu1 and Jovana Mitrović1

1Google DeepMind

We introduce SPARse Fine-grained Contrastive Alignment (SPARC), a simple method for pretraining more fine-grained multimodal representations from image-text pairs. Given that multiple image patches often correspond to single words, we propose to learn a grouping of image patches for every token in the caption. To achieve this, we use a sparse similarity metric between image patches and language tokens and compute for each token a language-grouped vision embedding as the weighted average of patches. The token and language-grouped vision embeddings are then contrasted through a fine-grained sequence-wise loss that only depends on individual samples and does not require other batch samples as negatives. This enables more detailed information to be learned in a computationally inexpensive manner. SPARC combines this fine-grained loss with a contrastive loss between global image and text embeddings to learn representations that simultaneously encode global and local information. We thoroughly evaluate our proposed method and show improved performance over competing approaches both on image-level tasks relying on coarse-grained information, e.g. classification, as well as region-level tasks relying on fine-grained information, e.g. retrieval, object detection, and segmentation. Moreover, SPARC improves model faithfulness and captioning in foundational vision-language models.

### 1. Introduction

Contrastive pre-training from large-scale, noisy image-text datasets (Jia et al., 2021; Radford et al., 2021) has become a widely used paradigm for learning general vision representations useful for a wide range of downstream tasks as well as for learning vision encoders in multimodal foundation models (Alayrac et al., 2022; Chen et al., 2022; Li et al., 2022a). By aligning global image and text representations in a shared latent space using similar and dissimilar image-text pairs, these models achieve impressive performance on image-level vision tasks like classification (Radford et al., 2021), coarse-grained retrieval and visual question answering (Alayrac et al., 2022; Chen et al., 2022). On the other hand, these models have been shown to discard fine-grained visual information (Krojer et al., 2022) and work poorly on downstream tasks involving localization (Ranasinghe et al., 2022; Zhong et al., 2022), counting (Paiss et al., 2023) and understanding spatial relationships between objects (Parcalabescu et al., 2021) or object attributes (Yuksekgonul et al., 2022). These shortcomings are further exacerbated when these pretrained models are used in foundation models (Alayrac et al., 2022; Chen et al., 2022; Li et al., 2022a) or when they are used to initialize models for object detection (Minderer et al., 2022) or segmentation (Zhou et al., 2022).

A recent line of work has started to explore incorporating losses between image patch and text token embeddings (Huang et al., 2021; Mukhoti et al., 2023; Wang et al., 2022; Yao et al., 2021) to learn representations encoding more fine-grained details. Motivated by the idea of aligning patches corresponding to individual objects in the image to tokens corresponding to the words describing these objects, these local losses learn soft correspondences between image patches and text tokens from image-text pairs. While these models have achieved improved performance on fine-grained retrieval (Yao et al., 2021), image classification (Yao et al., 2021), object detection and segmentation (Mukhoti et al., 2023; Wang et al., 2022), they are computationally and memory

Corresponding author(s): [ioanab,anastasijailic,msbauer,gokererdogan,matko,kaplanis,agritsenko,mjlm,cblundell,razp,mitrovic]@google.com © 2024 Google DeepMind. All rights reserved

expensive, unstable during training (Yao et al., 2021) and/or rely on pretrained models to kickstart learning.

In this work, we propose SPARse Fine-grained Contrastive Alignment (SPARC), a novel objective for multimodal pretraining which learns representations that encode both coarsegrained/global and fine-grained/local information. We propose to build language-grouped vision embeddings by learning to aggregate (in an unsupervised way) image patches corresponding to individual words in the caption; this is motivated by the observation that usually multiple image patches correspond to one word in the caption. As a first step, SPARC computes the similarity between the patch and token embeddings of an individual image-text pair and enforces sparsity in the resulting similarity matrix. This sparsification enables only the most relevant image patches to be attributed to individual tokens. Next,

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Language-grouped vision embeddings

[Figure 6]

[Figure 7]

[Figure 8]

“cat” “and” “dog” “in” “basket”

“cat” “and” “dog”

Tokenembeddings

Tokenembeddings

[Figure 9]

·

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

[Figure 10]

[Figure 11]

·

+

[Figure 12]

·

“in” “basket” ·

Sparse cross-modal alignment

Finegrained loss

Figure 1 | For every text token, SPARC learns a corresponding language-grouped vision embedding as the alignmentweighted combination of patches that are most similar to that token. We calculate a sparse similarity metric between tokens and patches of individual image-text pairs (left) and use it to compute the resulting alignment weights (middle). We contrast the language-grouped vision embeddings with token embeddings in a fine-grained contrastive sequence-wise loss (right).

- as illustrated in Figure 1, for every text token, we compute the corresponding language-grouped vision embedding as the alignment-weighted sum of the patch embeddings, where the alignment weights are computed from the sparsified similarity matrix. The resulting language-grouped vision embeddings are contrasted with the token embeddings from the same image-text pair by optimizing for the similarity between individual tokens and their corresponding language-grouped vision embedding and dissimilarity to all other language-grouped vision embeddings. SPARC combines the resulting fine-grained/local contrastive loss with a global contrastive loss between image and text embeddings which enables it to simultaneously encode global and local information in the learned representations.

Through its design choices, SPARC addresses several shortcomings of existing methods for learning image representations with more fine-grained information. Firstly, several of these methods (Huang et al., 2021; Mukhoti et al., 2023; Yao et al., 2021) learn representations with fine-grained losses that compute similarities between all image patch embeddings and all text token embeddings in a batch. This approach is both computationally and memory intensive and does not scale to large batch sizes (which are needed for obtaining good performance for contrastive methods (Jia et al., 2021; Radford et al., 2021; Zhai et al., 2023b)). On the other hand, SPARC contrasts patch and token embeddings

- at the level of individual image-text pairs and does not use other examples from the batch to compute the similarity matrix which leads to more favourable computation and memory footprints and more easily scales to large batch sizes. Secondly, for learning soft correspondences between image patches and text tokens, prior work (Huang et al., 2021; Mukhoti et al., 2023; Wang et al., 2022) usually relies on building cross-modal weighted representations with weights computed as a softmax over patch and token embedding similarities. The winner-takes-all dynamics of softmax (Elfadel and Wyatt Jr, 1993; Peterson and Söderberg, 1989) strongly bias learning towards one-to-one mappings

between individual text tokens and image patches which often does not correspond to underlying data. For example, in an image of a dog, the token embedding for “dog” should be matched with all patch embeddings that correspond to the dog in the image and not just one/a few. Moreover, softmax can be problematic from a gradient flow perspective (Hoffmann et al., 2023; Shen et al., 2023; Zhai et al., 2023a) as it tends to lead to a low entropy distribution, where softmax saturates and therefore its Jacobian vanishes (Hoffmann et al., 2023). See Appendix A for a more detailed explanation. On the flip side, SPARC does not use softmax for calculating the alignment weights which allows it to learn a flexible one-to-many matching between individual tokens and the corresponding image patches and to avoid the winner-take-all dynamics of softmax. Thirdly, several of these approaches start from contrastively pre-trained vision-language models (Mukhoti et al., 2023) or from pre-trained language models (Huang et al., 2021; Wang et al., 2022). Moreover, existing fine-grained objectives have been developed in different communities (i.e. medical (Huang et al., 2021; Wang et al., 2022) vs. general vision (Mukhoti et al., 2023; Yao et al., 2021)) leveraging different types and sizes of datasets, architectures and pretraining setups. This makes it difficult to compare different approaches and assess the benefits of using individual fine-grained objectives.

To summarize, our main contributions are as follows:

- • We propose SPARC, a novel method for pre-training multimodal models on large-scale noisy image-text data which learns both coarse-grained and fine-grained information.
- • Through an extensive experimental evaluation, we show that SPARC significantly improves performance on both fine-grained and coarse-grained downstream tasks over competing methods.
- • For the first time in the literature, we perform a thorough like-for-like comparison on the benefits of different fine-grained objectives for large-scale pretraining of multimodal models.

### 2. Sparse Fine-grained Contrastive Alignment

Let B = {(𝒙1𝑣, 𝒙1𝑡 ), (𝒙2𝑣, 𝒙2𝑡 ), . . . , (𝒙𝐵𝑣, 𝒙𝑡𝐵)} be a mini-batch of image-text pairs. Let 𝑓𝑣(·) be the image encoder, 𝑓𝑡(·) the text encoder and 𝑔𝑣(·) and 𝑔𝑡(·) linear adaptors. For an image 𝒙𝑖𝑣, we denote the corresponding patches as (𝒙𝑖,𝑣1, 𝒙𝑖,𝑣2, . . . , 𝒙𝑖,𝑃𝑣 ) and the patch embeddings as (𝒗𝑖,1, 𝒗𝑖,2, . . . , 𝒗𝑖,𝑃) with 𝒗𝑖,𝑝 = 𝑔𝑣( 𝑓𝑣(𝒙𝑖,𝑝𝑣 )) ∈ ℝ𝑑; 𝑃 denotes the number of patch embeddings. We calculate the global vision embedding as 𝒗𝑖 = 𝑔𝑣(ℎ𝑣(avg_pool({ 𝑓𝑣(𝒙𝑖,𝑝𝑣 )}𝑃𝑝=1))) with ℎ𝑣 being a single non-linear layer that facilitates the encoding of different granularities of information. For the corresponding text 𝒙𝑖𝑡, we denote the tokens as (𝒙𝑖,𝑡1, 𝒙𝑖,𝑡2, . . . , 𝒙𝑖,𝐿𝑡

) with 𝐿𝑖 the number of tokens for sample 𝑖. The token embeddings (𝒕𝑖,1, 𝒕𝑖,2, . . . , 𝒕𝑖,𝐿𝑖) are computed as 𝒕𝑖,𝑙 = 𝑔𝑡( 𝑓𝑡(𝒙𝑖,𝑙𝑡 )) and the global text embedding 𝒕𝑖 is computed by average pooling { 𝑓𝑡(𝒙𝑖,𝑙𝑡 )}𝑙𝐿=𝑖1 and applying the adaptor 𝑔𝑡, i.e. 𝒕𝑖 = 𝑔𝑡(avg_pool({ 𝑓𝑣(𝒙𝑖,𝑙𝑡 )}𝑙𝐿=𝑖1).

𝑖

Global alignment: In order to learn global information, SPARC uses the global contrastive loss (Jia et al., 2021; Radford et al., 2021) which operates at the level of global image (𝒗) and global text embeddings (𝒕). Specifically, we learn image and text embeddings by maximizing the similarity to the corresponding text and image embeddings, while minimizing the similarity to other text and image embeddings in the batch, i.e. we optimize

∑︁𝐵

- 1

- 2𝐵

exp(𝜙(𝒗𝑖, 𝒕𝑖)/𝜏)

log

𝐿𝑔 = −

𝐵 𝑗=1 exp(𝜙(𝒗𝑖, 𝒕𝑗)/𝜏)

𝑖=1

𝑖∥2 · ∥¯𝒕¯𝒕𝑗𝑗∥2 and 𝜏 as temperature.

with 𝜙(𝒗𝑖, 𝒕𝑗) = ∥𝒗¯𝒗¯𝑖

exp(𝜙(𝒕𝑖, 𝒗𝑖)/𝜏)

+ log

𝐵 𝑗=1 exp(𝜙(𝒕𝑖, 𝒗𝑗)/𝜏)

, (1)

[Figure 13]

global vision embedding

Global alignment

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Vision encoder

global text embedding

Textencoder

17 ”a picture of a cat and dog”

sparsify and normalize

. language-grouped

93

vision embeddings

75

0

Similarity matrix sl,p Alignment weights al,p

Finegrained alignment

- Figure 2 | Overall architecture for SPARC. The global alignment loss maximizes the similarity between the global vision and global text embeddings, while minimizing the similarity with the other global embeddings in the batch. To obtain the finegrained alignment, we compute the similarity between the patch embeddings and the token embeddings and then sparsify and normalize the resulting similarity matrix to obtain alignment weights. These alignment weights are then used to group the patch embeddings. The resulting language-grouped vision embeddings are then contrasted to the token emebddings in a sequence-wise finegrained alignment loss.

Finegrained alignment: Motivated by the observation that usually multiple image patches correspond to one word in the caption, we propose to learn groupings of patches that correspond to individual text tokens. Specifically, for every token embedding we learn a corresponding languagegrouped vision embedding as an alignment-weighted combination of patches that encode that token in the visual domain. We propose to compute the alignment weights based on the similarity between token and patch embeddings of the corresponding image-text pair. To facilitate the grouping of appropriate patch embeddings given a text token we sparsify and min-max normalize the similarity matrix to compute the alignment weights. To learn language-grouped vision embeddings, we propose a fine-grained local loss that optimizes for the alignment between individual token embeddings and their corresponding language-grouped vision embeddings within a given image-text pair. Specifically, we propose a sequence-wise contrastive loss to optimize this fine-grained alignment within SPARC. Optimizing this loss (in addition to the global contrastive loss above) biases the learned representation to preserve detailed information about the image (as described by the caption) instead of just the global information sufficient to minimize the global contrastive loss.

For an image-text pair, let 𝑠𝑖,𝑙𝑝 represent the similarity between text token embedding 𝒕𝑖𝑙 and image patch embedding 𝒗𝑖𝑝, i.e. 𝑠𝑖,𝑙𝑝 = 𝒕𝑖𝑙 · 𝒗𝑖𝑝, where 𝑠𝑖,𝑙𝑝 ∈ ℝ𝐿×𝑅 and · is the inner product. Going forward we drop the example index 𝑖 for simplicity. To obtain alignment weights, for each token 𝑗, we first normalize 𝑠𝑙𝑝 to [0, 1] using min-max normalization across columns (i.e. patches):

𝑠𝑙𝑝 − min𝑘 𝑠𝑙𝑘 max𝑘 𝑠𝑙𝑘 − min𝑘 𝑠𝑙𝑘

(2)

ˆ𝑠𝑙𝑝 =

We sparsify the similarity matrix 𝑆 = (ˆ𝑠𝑗𝑘)1≤𝑗≤𝐿,1≤𝑘≤𝑃 to facilitate learning and to encourage each token to be aligned to a few of the patches, i.e.

˜𝑠𝑗𝑘 =

ˆ𝑠𝑗𝑘 if ˆ𝑠𝑗𝑘 ≥ 𝜎 0 otherwise

(3)

with 𝑃 the number of patch embeddings of an image and 𝜎 the sparsity threshold. We compute alignment weights as

˜𝑠𝑗𝑘

(4)

𝑎𝑗𝑘 =

𝑅 𝑟=1 ˜𝑠𝑗𝑟

where 𝑎𝑗𝑘 represents the weight of patch 𝑘 for computing the language-grouped vision embedding corresponding to token 𝑗. Note that this approach enables a flexible mapping between a token and arbitrarily many patch embeddings that encode that token in the visual domain, e.g. all of the image patches corresponding to “dog” can be matched to the token encoding “dog”. For every token 𝑡𝑙 we compute the corresponding language-grouped vision embedding 𝒄𝑙 as

∑︁𝑅

𝑎𝑙𝑟𝒗𝑟 (5)

𝒄𝑙 =

𝑟=1

as the alignment-weighted combination of patch embeddings with 𝑅 the number of patches with non-zero alignment weight.

To learn fine-grained information we propose to optimize the alignment between token embeddings and their corresponding language-grouped vision embeddings. Specifically we propose a fine-grained contrastive loss that operates over sequences of tokens and patches at the level of each image-text pair and does not require negatives from other image-text pairs. This considerably reduced computation and memory costs over previous methods (Huang et al., 2021; Yao et al., 2021) that require samples from the whole batch in order to compute their fine-grained losses. SPARC optimizes the following fine-grained alignment contrastive loss

  

###### ∑︁𝐵

###### ∑︁𝐿𝑖

exp(𝜙(𝒕𝑖𝑗, 𝒄𝑖𝑗)/𝜏)

exp(𝜙(𝒄𝑖𝑗, 𝒕𝑖𝑗)/𝜏)

- 1

- 2𝐵

1

+ log

, (6)

log

𝐿𝑓 = −

𝐿𝑖 𝑘=1 exp(𝜙(𝒄𝑖𝑗, 𝒕𝑖𝑘)/𝜏)

𝐿𝑖 𝑘=1 exp(𝜙(𝒕𝑖𝑗, 𝒄𝑖𝑘)/𝜏)

𝐿𝑖

𝑖=1

𝑗=1

which tries to maximize the similarity of every token embedding with its corresponding languagegrouped vision embedding and minimize the similarity to other language-grouped vision embeddings in the sequence and vice versa.

Overall objective: The overall SPARC objective is a weighted sum of the global contrastive loss and the finegrained alignment constrastive loss:

𝐿SPARC = 𝜆𝑔𝐿𝑔 + 𝜆𝑓 𝐿𝑓 (7) where 𝜆𝑔 and 𝜆𝑓 are hyperparameters. We provide the pseudo-code for SPARC in Appendix C.

Sparsity threshold. We choose the sparsity threshold 𝜎 to be equal to 1/𝑃 with 𝑃 the number of image patches. This choice is motivated by the consideration that every text token should attend to at least to one image patch. Since we use the min-max normalization the smallest similarity of 1/𝑃 is achieved when all patches are equally similar as the number of patches is constant. Note that this threshold naturally allows for the number of patches corresponding to one token to considerably vary between tokens within an image as well as across images; this enables the same class of objects (e.g. “dogs”) to be appropriately represented irrespective of the difference in sizes, scales and shapes across different instances within and across images. Note also that the threshold also allows for the decoupling of similarities of individual patches to different tokens as it allows for different number of zero entries in different rows of the similarity matrix; thus, whether and how much a patch is similar to a token, has no bearing to how similar it is to a different token which is useful e.g. in situations when we have more detailed captions (e.g. “large brown dog”) and/or when a single word is represented by multiple tokens.

### 3. Related work

Contrastive image-text pre-training CLIP (Radford et al., 2021) and ALIGN (Jia et al., 2021) popularized learning general visual representations by leveraging textual supervision from noisy largescale data scrapped from the internet. These methods learn representations through a contrastive objective that maximises the similarity between the representation of the whole image and the representation of the full text of matched image-text pairs and minimizes the similarity between the remaining image-text pairs within the batch. However, learning visual representations through matching the global image and text embeddings can result in a coarse visual representation that discards many fine-grained details (i.e all details that are not needed for differentiating the matching of global text embedding from the other text embeddings in the batch). To address this problem, FILIP (Yao et al., 2021) proposes a cross-modal late interaction mechanism, which optimizes the token-wise maximum similarity between image and text tokens through a contrastive objective. While this approach achieves a finer-grained alignment between image patches and words in the text, computing the token-wise similarity between all image patches and text tokens in the batch becomes memory inefficient for large batch sizes so they use several tricks during pre-training to address this issue. A related approach PACL (Mukhoti et al., 2023) starts from CLIP-pretrained vision and text encoders and trains on top of the frozen representations an adapter to obtain better fine-grained understanding. The adapter is a two-layer MLP with a residual connection and is trained through a contrastive objective that compares the global text embedding and a weighted global image embedding with the weights calculated using the cosine similarity between individual image patches and the global text embedding.

In a parallel stream of work, several methods have been proposed in the medical literature to learn visual representation using medical images - radiology report pairs from small scale datasets (consisting of up to 200k data points) (Dawidowicz et al., 2023; Huang et al., 2021; Wang et al., 2022). GLoRIA (Huang et al., 2021) builds localized visual representations by contrasting attention-weighted patch embeddings with the text tokens, where the attention weights are computed through softmax on the similarity matrix between the patch and token embeddings. Similarly to FILIP, the local objective in GLoRIA requires computing the similarity between all patch and token embeddings within the batch which is computationally intensive and does not scale to large batch sizes. Alternatively, MGCA (Wang et al., 2022) considers a token-wise fine-grained loss that employs a bidirectional multi-head attention strategy to learn the matching between image patch and token embedding. While this is more efficient to compute, learning these matchings through a bidirectional multi-head cross-attention strategy adds more parameters to the dual encoders, involves tuning several additional hyperparameters and suffers from the same problems with using softmax for computing the attention weights. MGCA also uses a domain-specific disease-level alignment loss that enforce a cluster assignment consistency to leverage inter-subject semantic correspondences. More recent methods (Dawidowicz et al., 2023) consider incorporating into the pre-training objective not only fine-grained losses similar to the ones used in GLoRIA and MGCA, but also domain-specific features and image views. Note that these methods from the medical literature start from a text encoder pre-trained with medical texts (Alsentzer et al.,

- 2019), while we consider the case of pre-training the image and text encoders jointly from scratch.

Fine-grained understanding in vision-language models Alternative approaches for improving the fine-grained capabilities of vision-language models require pre-trained modules, specialised networks and human annotations. One line of work, proposes matching image regions to textual descriptions through contrastive losses, where the image regions - text description pairs are obtained from human annotations (Li et al., 2022b) or by using region proposal networks (Ren et al., 2015) and various text matching approaches (Varma et al., 2023; Zhong et al., 2022). A separate line of work adds a cross-modal encoder (with significant extra parameters) on top of the dual image-text encoder

and uses captioning (Li et al., 2022a; Yu et al., 2022), masked language modelling (Li et al., 2021; Yang et al., 2022), image-text matching (Li et al., 2021; Yang et al., 2022; Zeng et al., 2021) and bounding box prediction losses (Zeng et al., 2021) (with bounding boxes obtained from humanannotations (Krishna et al., 2017; Kuznetsova et al., 2020; Shao et al., 2019)). For more related works see Appendix B.

### 4. Experiments

While there has been significant interest in learning fine-grained representations, the breadth of training setups used in the literature have made it difficult to compare different fine-grained objectives. Specifically the use of custom datasets (Yao et al., 2021) and pretrained language and/or vision models (Huang et al., 2021; Mukhoti et al., 2023; Wang et al., 2022) have made it difficult to discern the benefit of individual fine-grained losses on learning more detailed representations. In this work we want to enable a like-for-like comparison and understand the impact of SPARC and competing fine-grained losses on downstream performance. For this purpose, we reimplement all competing baselines: CLIP (Radford et al., 2021), FILIP (Yao et al., 2021), PACL (Mukhoti et al., 2023), MGCA (Wang et al., 2022) and GLoRIA (Huang et al., 2021), and use the same pretraining datasets, architecture and number of training steps when training with the different objectives; we pretrain randomly initialized networks. We thoroughly evaluate the learned representations across a broad range of tasks and datasets, ranging from coarse-grained image-level tasks like classification and retrieval to fine-grained tasks like object detection and semantic segmentation. Unlike some competing methods that improve fine-grained understanding at the cost of decreasing coarse-grained task performance, SPARC simultaneously boosts performance over both coarse- and fine-grained tasks across a number of different benchmarks.

#### 4.1. Experimental setup

Model architectures Following the literature, we use Vision Transformers (ViTs) (Dosovitskiy et al.,

- 2020) as image encoders and Transformers (Vaswani et al., 2017) as text encoders. We experiment with ViT-B/32, ViT-B/16 and ViT-L/14 and pair them with corresponding language models. See details in Appendix D. Datasets We train using large-scale datasets ALIGN (Jia et al., 2021), JFT (Sun et al., 2017; Zhai

- et al., 2022) and LTIP (Long Text & Image Pairs) (Alayrac et al., 2022). ALIGN has 1.8 billion images paired with noisy alt-text, JFT has of 4 billion images semi-automatically annotated with a class-hierarchy of 30k labels, while LTIP has 312 million higher-quality images - text pairs with richer image captions. See Appendix D for more details.

Pre-training details We resize images to the 224 × 224 resolution and tokenize the text with a 32k vocabulary sentencepiece tokenizer (Kudo and Richardson, 2018) while keeping a maximum number of 55 tokens for each caption. We train all models using the AdamW (Loshchilov and Hutter, 2017) optimizer, a cosine learning rate schedule with linear warm-up and weight decay regularization. We use a batch size of 16348 and we pre-train the ViT-B models for 200k steps (≈ 3.2 billion data points) and the ViT-L models for 250k steps (≈ 4.1 billion data points). See Appendix D for more hyperparameter details.

CLIP 66.7 66.2 58.9 71.5 63.2 42.6 15.1 51.7 FILIP 52.7 50.7 44.0 55.8 47.1 28.7 8.4 38.2 PACL 58.9 56.9 50.0 62.6 54.0 34.9 9.3 44.1 GloRIA 62.8 61.5 54.3 66.7 56.7 38.4 11.2 47.5 MGCA 66.0 64.5 56.4 69.5 62.0 41.1 14.7 51.7 SPARC (ours) 68.1 67.0 59.7 72.0 64.9 44.5 16.7 53.2

ViT-B/32

CLIP 71.6 70.9 63.7 74.8 71.1 48.5 32.2 56.8 FILIP 56.6 55.6 48.9 59.7 54.0 33.2 14.4 43.1 PACL 61.1 59.6 52.6 64.8 56.3 36.1 12.8 45.2 GloRIA 67.4 66.9 59.8 71.7 66.6 43.8 24.6 54.2 MGCA 69.6 69.3 62.2 73.6 68.8 46.1 29.0 55.0 SPARC (ours) 72.6 71.1 64.4 75.0 72.0 48.5 33.8 57.3

ViT-B/16

CLIP 77.3 75.9 69.5 79.1 78.8 59.6 52.5 64.5 MGCA 75.6 73.9 68.0 77.9 77.2 56.0 45.0 63.1 SPARC (ours) 78.2 76.9 70.6 80.0 79.3 59.7 51.9 65.4

ViT-L/4

Table 1 | Top-1 accuracy (in %) of zero-shot classification on ImageNet (IN) and its variants ImageNetV2 Threshold (IN-V2 Th), ImageNet-V2 Matched Frequency (In-V2 MF), ImageNet-V2 Top Images (IN-V2 TI), ImageNet-R (IN-R), ImageNet-C (IN-C), ImageNet-Sketch (IN-Sketch).

#### 4.2. Zero-shot image classification

We first evaluate SPARC on the coarse-grained task of zero-shot image classification. Specifically we test zero-shot classification on ImageNet (Russakovsky et al., 2015) and a number of datasets testing for specific capabilities like robustness to perturbations and various distribution shifts; we choose ImageNetV2 (Recht et al., 2019), ImageNet-R (Hendrycks et al., 2021), ImageNet-C (Hendrycks and Dietterich, 2019), ImageNet-A (Hendrycks et al., 2019) and ImageNet-Sketch (Wang et al., 2019) for this purpose. We follow a similar protocol to (Radford et al., 2021) for the evaluation, and compute results for both one prompt per example (i.e. the class label) in Table 1 and when using prompt ensembling in Table 2. For more details on the evaluation protocol please see Appendix D. From both

- Table 1 and Table 2 we see that SPARC outperforms or matches competing methods in all settings and across different ViT architectures. Specifically, SPARC shows very effective information encoding from larger patches as exhibited by the significant improvements over baselines for ViT B/32, especially on ImageNet-R, -C, -A and -Sketch showcasing the robustness to perturbations and adversarial examples. Moreover, we notice that while prompt ensembling improves performance of all methods on zero-shot image classification (which is in line with the literature) the performance gain from SPARC are still preserved in this evaluation setting.

Note that PACL (Mukhoti et al., 2023), GLoRIA (Huang et al., 2021) and MGCA (Wang et al., 2022) were developed with the use of pretrained language and/or vision encoders in mind, whereas here they are tested in a pretraining from scratch setting. From Table 1 and Table 2, we see that in the pretraining setting PACL and GLoRIA underperform CLIP, whereas MGCA shows more competitive performance to CLIP. On the other hand, FILIP (Yao et al., 2021), which was developed as a finegrained objective for pretraining from scratch, has proven highly unstable to train across a wide range of learning rates and weight decay parameters which lead to decreased performance. This training difficulty has also been noted in the original paper (Yao et al., 2021) (cf. in the Appendix

CLIP 69.0 68.8 60.4 73.4 62.4 44.6 15.8 52.4 FILIP 56.8 54.8 48.4 60.0 44.6 30.8 7.8 39.6 PACL 61.2 59.5 51.9 65.2 52.9 36.4 9.3 45.2 GloRIA 65.9 64.8 57.0 69.6 57.4 40.7 11.7 48.7 MGCA 68.6 67.4 59.2 72.6 61.0 43.5 14.1 50.9 SPARC (ours) 70.4 69.6 62.1 74.5 63.2 46.5 17.3 52.7

ViT-B/32

CLIP 73.9 73.6 66.1 77.1 68.8 50.4 32.5 57.3 FILIP 61.4 61.0 53.8 65.6 53.2 35.9 14.2 45.1 PACL 63.3 61.7 54.4 66.8 54.1 37.3 12.9 45.4 GloRIA 70.4 70.0 62.8 74.7 65.7 46.4 25.0 54.8 MGCA 72.7 72.7 65.3 76.3 67.6 48.4 29.8 55.5 SPARC (ours) 74.7 74.0 67.1 77.8 71.1 51.31 34.2 57.9

ViT-B/16

CLIP 79.2 78.5 71.8 81.6 78.5 61.3 51.5 65.1 MGCA 78.0 77.4 70.5 80.6 75.2 57.9 45.5 63.1 SPARC (ours) 79.7 78.9 72.6 81.9 79.8 61.3 53.4 65.9

ViT-L/4

- Table 2 | Top-1 accuracy (in %) of zero-shot classification using prompt ensembling on ImageNet (IN) and its variants ImageNet-V2 Threshold (IN-V2 Th), ImageNet-V2 Matched Frequency (In-V2 MF), ImageNet-V2 Top Images (IN-V2 TI), ImageNet-R (IN-R), ImageNet-C (IN-C), ImageNet-Sketch (IN-Sketch).

A.3. "...training is extremely unstable and the Nan loss easily happens."). In addition to that FILIP uses a number of additional tricks not present in a standard pretraining setup like image augmentations, backtranslation of captions and custom prompt ensembling.

#### 4.3. Image-Text retrieval

Next we evaluate SPARC on zero-shot cross-modal retrieval tasks, i.e image-to-text and text-to-image retrieval, on Flickr30k (Plummer et al., 2015) and MSCOCO (Lin et al., 2014). From Table 3, we see that SPARC outperforms all competing baselines across all metrics. While using fine-grained losses PACL and GLoRIA significantly underperforms the global contrastive objective CLIP, MGCA shows competitive performance to CLIP in the pretraining setting. Unfortunately, FILIP (Yao et al., 2021) again underperforms CLIP across all metrics. In an attempt to stabilize FILIP we combined it with CLIP and observed an improvement on image-to-text Flikr30k on ViT B/32 while being competitive on other benchmarks to CLIP. We provide these results in Appendix D.

#### 4.4. Evaluating faithfulness

We further examine fine-grained performance of SPARC through faithfulness—how consistent the model’s highest scoring caption is with the ground truth caption(s) (Ji et al., 2023). This is different from top-1 retrieval (R@1) which measures exact match retrieval and does not evaluate the ability of the models to faithfully describe the elements in the image. Faithfulness has been used in the LLM literature to assess the propensity of the model to hallucinate (Adlakha et al., 2023; Razumovskaia

- et al., 2023) as models with higher faithfulness more accurately capture the details of the ground truth while not inserting additional information (possible hallucinations). The lexical overlap metric

Flickr30k MSCOCO image-to-text text-to-image image-to-text text-to-image

Objective R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10

CLIP 79.2 95.1 97.2 66.5 88.0 93.1 53.5 78.2 86.7 38.4 64.8 74.9 PACL 65.5 86.8 92.2 49.8 76.5 84.7 37.6 65.1 75.7 26.5 50.6 61.8 GLoRIA 74.6 92.1 96.2 61.5 85.3 90.7 46.9 73.0 82.7 34.5 61.0 71.7 MGCA 81.5 93.9 96.8 64.4 86.5 92.0 54.5 78.6 86.8 37.7 63.7 74.0 FILIP 62.6 86.9 92.9 50.5 77.7 84.9 35.6 61.0 73.1 26.2 51.0 62.4 SPARC (ours) 82.5 96.2 97.6 67.7 88.2 93.0 55.0 79.1 87.3 39.7 65.9 75.7

ViT-B/32

CLIP 84.0 96.1 98.2 71.6 90.3 94.1 56.2 80.6 88.2 42.4 68.6 78.3 PACL 69.6 89.7 94.2 54.9 80.7 87.3 41.8 67.8 77.6 29.1 54.3 65.5 GLoRIA 78.0 95.5 98.0 68.4 88.9 93.2 49.7 75.4 84.6 38.9 65.1 75.2 MGCA 82.2 96.1 98.1 67.7 88.5 93.2 57.6 80.5 87.8 39.8 65.7 75.3 FILIP 69.0 89.8 94.0 55.8 81.5 87.9 40.2 66.0 76.3 29.5 55.3 66.3 SPARC (ours) 84.4 97.6 98.7 72.0 91.2 94.9 57.6 81.2 88.5 43.0 68.6 78.5

ViT-B/16

CLIP 84.7 96.9 98.4 73.7 91.8 95.4 58.6 82.6 89.1 44.8 70.5 79.5 MGCA 85.9 96.9 98.1 73.2 91.6 95.3 59.7 83.2 89.7 44.3 69.6 78.8 SPARC (ours) 86.9 97.3 98.6 74.4 91.7 95.4 58.9 82.9 89.7 45.6 71.1 80.1

ViT-L/14

- Table 3 | Results on zero-shot image-to-text and text-to-image retrieval on MSCOCO and Flickr30k datasets. R@i denotes Recall at i.

of K-Precision measuring the proportion of tokens in the top chosen caption that appear in the ground truth tokens has been shown to correlate well with human judgement (Adlakha et al., 2023). In Table 4 we report the K-Precision on the MSCOCO for all tokens (K-P), as well as K-Precision restricted to nouns and adjectives only (K-Pna), as these better encode the objects observed in the image. We evaluate all methods on two architectures and see that SPARC reduced hallucinations of objects (higher K-Pna) while also showing competitive performance to related methods when taking all tokens into account (as measured by K-P).

ViT-B/32 ViT-B/16 Method K-Pna K-P K-Pna K-P

CLIP 76.03 77.82 77.56 78.99 FILIP 63.3 66.83 66.05 70.09 PACL 3.36 26.26 4.09 27.31 GLoRIA 71.63 73.54 73.85 75.3 MGCA 75.79 77.98 77.66 80.03 SPARC (ours) 76.46 78.44 78.72 79.77

- Table 4 | All-token K-Precision (K-P) and the K-Precision restricted to nouns and adjectives (K-Pna) (in %) on MSCOCO.

#### 4.5. Fine-grained localization

We further examine SPARC by evaluating it on fine-grained tasks requiring precise localization such as open-vocabulary object detection and zero-shot semantic segmentation. For these evaluations, we use the ViT-B/16 architecture.

Open-vocabulary object detection. To first evaluate whether the improved fine-grained understanding learned with SPARC translates to tasks requiring fine-grained localization, we use SPARC as a backbone for object detection. Specifically, we used the OWL-ViT open-vocabulary object detector (Minderer et al., 2022) with a ViT-B/16 backbone. After SPARC pre-training, detection heads are added to the backbone and fine-tuned on Objects365 (Shao et al., 2019) and Visual Genome (Krishna et al., 2017) datasets following the approach in Minderer et al. (2022). We evaluate the resulting model on the large-vocabulary dataset LVIS (Gupta et al., 2019) which is well-suited for testing the transfer of knowledge from image-level pretraining. LVIS contains 1203 categories of objects, of which 307 “rare” categories are excluded from the training data to measure zero-shot transfer from pretraining. Moreover, we also evaluate detection on the 80 MSCOCO classes. We run detection training three times and report mean and standard deviation in Table 5. SPARC improves over CLIP +0.9% on LVIS and MSCOCO as measured by mean average precision and +3.1% on LVIS “rare” classes. Since LVIS “rare” classes are never seen during detection training data, the model has to rely on information transfer from the pretrained representations for these classes. The large improvement of SPARC over the baseline on LVIS APrare suggests that SPARC has learned more informative fine-grained representations.

LVIS MSCOCO Method APall APrare APall

CLIP 26.9 ± 0.12 22.0 ± 0.79 38.5 ± 0.19 SPARC (ours) 27.9 ± 0.11 25.1 ± 0.95 39.4 ± 0.13

- Table 5 | Mean Average precision (as mean ± standard deviation) on all and rare classes on LVIS and on all classes in MSCOCO.

Semantic Segmentation. Following related work (Mukhoti et al., 2023), we also perform zero-shot segmentation given a text label, i.e. we compute patch embeddings of a given image and calculate the cosine similarity of the patch embedding with the text embeddings of all the ground-truth classes (Mukhoti et al., 2023; Ranasinghe et al., 2022). We assign a matching class for each patch as the text that corresponds to the maximum cosine similarity of that patch. We then upsample the patches to match the resolution of the ground-truth segmentation and calculate for each class the Intersection over Union (IoU) between the predicted and ground-truth segmentations; we report the mean of the IoU scores over the classes present in the ground-truth image. More details about this evaluation can found in Appendix D. From Table 6 we see that SPARC strongly improves over other baselines, significantly surpassing the next best model by +4.34 mIoU on the PASCAL VOC (Everingham et al., 2015) dataset and by +1.2 mIoU on the PASCAL Context (Mottaghi et al., 2014) dataset. We visualize the predicted

Method Pascal VOC Pascal Context CLIP 23.02 20.45 FILIP 19.32 9.31 PACL 1.23 1.61 GLoRIA 22.64 15.26 MGCA 21.91 11.50 SPARC (ours) 27.36 21.65

Table 6 | Semantic Segmentation: mIoU of predicted and ground-truth segmentation on Pascal VOC and PASCAL Context datasets.

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

- Figure 3 | Qualitative results for zero-shot segmentation on Pascal VOC dataset. We illustrate the original image, pixel-level ground-truth labels and the the patch-level segmentation masks obtained from SPARC, GLoRIA and CLIP.

segmentation masks on the PASCAL VOC dataset in Figure 3. Whereas CLIP predicts the object to be present in many different parts of the image, SPARC achieves better object localization and predicts their shapes more accurately.

#### 4.6. SPARC backbones in vision language models

Vision backbones trained contrastively from imagetext paired data are often frozen and used in foundational vision-language models (VLMs) such as Flamingo (Alayrac et al., 2022). To understand whether the fine-grained performance improvements obtained from SPARC translate to better captioning performance in VLMs, we perform experiments where we compare using a CLIP backbone vs. a SPARC backbone in a Flamingo-style architecture (Alayrac et al., 2022). For this, we freeze the ViT-B/16 vision models trained with CLIP and SPARC and pair them with a frozen 400M parameter (pretrained) language model. On top of the frozen vision and language backbones, we train Perceiver Resampler cross-attention layers (Alayrac et al., 2022) to produce free-form text as output. More details about the training set-up can be found in Appendix D. We evaluate the models on captioning tasks on MSCOCO and Flickr30k datasets and we report results in Table 7.

Method MSCOCO Flickr30k

CLIP 24.3 12.9 SPARC (ours) 25.3 13.6

Table 7 | CIDEr score evaluating captioning performance of different vision backbones in a Flamingo-style (Alayrac et al., 2022) model.

#### 4.7. Ablations

To assess the benefits of the different components in SPARC on performance, we perform the following two ablations: removing the sparsity on the similarity matrix and using softmax instead to compute the alignment weights for grouping the patch embeddings. From the results in Table 8 on both fine-grained (MSCOCO retrieval) and coarse-grained (ImageNet zero-shot classification) tasks we notice that both components play a significant role in the model’s performance. In particular, using softmax results in the highest decrease in performance. See Appendix A for a detailed discussion of the problems with using softmax to compute the alignment weights.

MSCOCO (i2t) MSCOCO (t2i) ImageNet R@1 R@5 R@1 R@5 Top-1 acc.

SPARC 57.6 81.2 43.0 68.6 72.6

- - no sparsity 56.1 80.7 42.4 68.2 72.1
- - softmax 55.2 79.8 41.6 67.5 70.6

- Table 8 | Ablations for the ViT-B/16 SPARC model on the MSCOCO image-to-text (i2t) and text-toimage (t2i) retrieval and zero-shot classification on ImageNet.

#### 4.8. Memory consumption and FLOPS

To understand the computational and memory efficiency of the different methods, we also compute the FLOPS and peak memory usage for one update step for different batch size. Note that all methods are trained on 256 TPUs. In Figure 4 (a) we show the teraFLOPS (TFLOPS) and in Figure 4 (b) the peak memory usage (in MB) of the different methods for one update step when varying the batch size (B) from 2048 to 16384. Notice that GLoRIA (Huang et al., 2021) is as memory intensive at batch size 4096 as the other methods (e.g. CLIP) at batch size 16384. Thus, due to device constraints, we were only able to train GLoRIA with batch size 4096. Moreover, notice that for FILIP the TFLOPS used for one update step increases by more than 200% between B=8196 and B=16384, as opposed to the 100% increase for CLIP, SPARC and MGCA. In addition, for B=16384, both FILIP and PACL have 2x peak memory compared to CLIP, SPARC and MGCA. On the other hand, note that CLIP, SPARC and MGCA use the same order of magnitude of FLOPS and memory. To further highlight the differences between them, we plot the relative increase in TFLOPS in Figure 4 (c) and the relative increase in peak memory in Figure 4 (c) of SPARC and MGCA with respect to CLIP. Notice that for B=16384, i.e. the batch size we use for our experiments, the relative increase in TFLOPS and peak memory for SPARC is almost half the one for MGCA. We provide detailed numbers for the FLOPS (in TFLOPS) and of the Peak Memory (in MB) in Appendix D.6.

[Figure 24]

[Figure 25]

(a) (b)

[Figure 26]

[Figure 27]

(c) (d)

- Figure 4 | TFLOPS (a) and Peak Memory (b) used by all methods. Relative increase in TFLOPS (c) and Peak memory (d) when comparing SPARC and MGCA to CLIP.
- 5. Discussion

In this work we proposed a novel method Sparse Fine-grained Contrastive Alignment (SPARC) for fine-grained vision-language pretraining. SPARC simultaneously learns information at different levels of granularity by contrasting both image-level and caption-level embeddings and token and patch embeddings. SPARC learns to group patches based on similarity to tokens and contrast the resulting language-grounded patch embeddings with token embeddings. Unlike previous work this comparison is done within individual image-text pairs and does not require the computationally and memory expensive comparison of all patches and tokens within the full batch. Through extensive experimental evaluation we show that SPARC improves performance both on image-level tasks like classification and retrieval and more fine-grained tasks like object detection and segmentation that require localization. Moreover, SPARC improves model faithfulness and

While the simple sparsification of the similarity matrix in SPARC already improves performance, we believe that exploring different approaches to sparsification and learning patch groupings could lead to even more informative representations. Moreover, given that SPARC learns patch groupings based on the associated caption, exploring pretraining data with highly descriptive captions is another interesting line of future work. Also, leveraging bounding boxes and segmentation masks (in addition to image-text pairs) would facilitate learning patch groupings and improve learning efficiency since the similarity matrix could be pre-sparsified according to these signals. Another interesting avenue of future work is further exploring how SPARC encoders perform as part of multimodal foundational models like Flamingo (Alayrac et al., 2022), BLIP (Li et al., 2022a) and PALI (Chen et al., 2022).

### References

V. Adlakha, P. BehnamGhader, X. H. Lu, N. Meade, and S. Reddy. Evaluating correctness and faithfulness of instruction-following models for question answering. arXiv preprint arXiv:2307.16877, 2023.

J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022.

E. Alsentzer, J. R. Murphy, W. Boag, W.-H. Weng, D. Jin, T. Naumann, and M. McDermott. Publicly available clinical bert embeddings. arXiv preprint arXiv:1904.03323, 2019.

- X. Chen, X. Wang, S. Changpinyo, A. Piergiovanni, P. Padlewski, D. Salz, S. Goodman, A. Grycner,

- B. Mustafa, L. Beyer, et al. Pali: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794, 2022.

- G. Dawidowicz, E. Hirsch, and A. Tal. Limitr: Leveraging local information for medical image-text representation. ArXiv, abs/2303.11755, 2023. URL https://api.semanticscholar.org/ CorpusID:257636659.

A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

- I. M. Elfadel and J. L. Wyatt Jr. The" softmax" nonlinearity: Derivation using statistical mechanics and useful properties as a multiterminal analog circuit element. Advances in neural information processing systems, 6, 1993.

M. Everingham, S. M. A. Eslami, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The pascal visual object classes challenge: A retrospective. International Journal of Computer Vision, 111(1): 98–136, Jan. 2015.

- S. Geng, J. Yuan, Y. Tian, Y. Chen, and Y. Zhang. Hiclip: Contrastive language-image pretraining with hierarchy-aware attention. arXiv preprint arXiv:2303.02995, 2023.

- A. Gupta, P. Dollar, and R. Girshick. Lvis: A dataset for large vocabulary instance segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5356–5364, 2019.

D. Hendrycks and T. Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint arXiv:1903.12261, 2019.

D. Hendrycks, K. Zhao, S. Basart, J. Steinhardt, and D. Song. Natural adversarial examples.(2019). arXiv preprint cs.LG/1907.07174, 5(6), 2019.

D. Hendrycks, S. Basart, N. Mu, S. Kadavath, F. Wang, E. Dorundo, R. Desai, T. Zhu, S. Parajuli, M. Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8340–8349, 2021.

- D. T. Hoffmann, S. Schrodi, N. Behrmann, V. Fischer, and T. Brox. Eureka-moments in transformers: Multi-step tasks reveal softmax induced optimization problems. arXiv preprint arXiv:2310.12956, 2023.

- S.-C. Huang, L. Shen, M. P. Lungren, and S. Yeung. Gloria: A multimodal global-local representation learning framework for label-efficient medical image recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3942–3951, 2021.

Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang, A. Madotto, and P. Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38, 2023.

C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–4916. PMLR, 2021.

- R. Krishna, Y. Zhu, O. Groth, J. Johnson, K. Hata, J. Kravitz, S. Chen, Y. Kalantidis, L.-J. Li, D. A. Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017.

B. Krojer, V. Adlakha, V. Vineet, Y. Goyal, E. Ponti, and S. Reddy. Image retrieval from contextual descriptions. arXiv preprint arXiv:2203.15867, 2022.

- T. Kudo and J. Richardson. Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. arXiv preprint arXiv:1808.06226, 2018.

A. Kuznetsova, H. Rom, N. Alldrin, J. Uijlings, I. Krasin, J. Pont-Tuset, S. Kamali, S. Popov, M. Malloci, A. Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International Journal of Computer Vision, 128(7):1956–1981, 2020.

- J. Li, R. Selvaraju, A. Gotmare, S. Joty, C. Xiong, and S. C. H. Hoi. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, 34:9694–9705, 2021.

- J. Li, D. Li, C. Xiong, and S. Hoi. Blip: Bootstrapping language-image pre-training for unified visionlanguage understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR, 2022a.

- L. H. Li, P. Zhang, H. Zhang, J. Yang, C. Li, Y. Zhong, L. Wang, L. Yuan, L. Zhang, J.-N. Hwang, et al. Grounded language-image pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10965–10975, 2022b.

- T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.

- I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

- M. Minderer, A. Gritsenko, A. Stone, M. Neumann, D. Weissenborn, A. Dosovitskiy, A. Mahendran,

- A. Arnab, M. Dehghani, Z. Shen, et al. Simple open-vocabulary object detection. In European Conference on Computer Vision, pages 728–755. Springer, 2022.

- R. Mottaghi, X. Chen, X. Liu, N.-G. Cho, S.-W. Lee, S. Fidler, R. Urtasun, and A. Yuille. The role of context for object detection and semantic segmentation in the wild. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.

##### J. Mukhoti, T.-Y. Lin, O. Poursaeed, R. Wang, A. Shah, P. H. Torr, and S.-N. Lim. Open vocabulary semantic segmentation with patch aligned contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19413–19423, 2023.

- R. Paiss, A. Ephrat, O. Tov, S. Zada, I. Mosseri, M. Irani, and T. Dekel. Teaching clip to count to ten. arXiv preprint arXiv:2302.12066, 2023.

L. Parcalabescu, M. Cafagna, L. Muradjan, A. Frank, I. Calixto, and A. Gatt. Valse: A task-independent benchmark for vision and language models centered on linguistic phenomena. arXiv preprint arXiv:2112.07566, 2021.

C. Peterson and B. Söderberg. A new method for mapping optimization problems onto neural networks. International Journal of Neural Systems, 01(01):3–22, 1989.

B. A. Plummer, L. Wang, C. M. Cervantes, J. C. Caicedo, J. Hockenmaier, and S. Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE international conference on computer vision, pages 2641–2649, 2015.

- A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin,

- J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
- K. Ranasinghe, B. McKinzie, S. Ravi, Y. Yang, A. Toshev, and J. Shlens. Perceptual grouping in vision-language models. arXiv preprint arXiv:2210.09996, 2022.

K. Ranasinghe, B. McKinzie, S. Ravi, Y. Yang, A. Toshev, and J. Shlens. Perceptual grouping in contrastive vision-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5571–5584, 2023.

E. Razumovskaia, I. Vulić, P. Marković, T. Cichy, Q. Zheng, T.-H. Wen, and P. Budzianowski. Dial BeInfo for Faithfulness: Improving factuality of information-seeking dialogue via behavioural fine-tuning, 2023.

- B. Recht, R. Roelofs, L. Schmidt, and V. Shankar. Do imagenet classifiers generalize to imagenet? In International conference on machine learning, pages 5389–5400. PMLR, 2019.

S. Ren, K. He, R. Girshick, and J. Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28, 2015.

O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115:211–252, 2015.

S. Shao, Z. Li, T. Zhang, C. Peng, G. Yu, X. Zhang, J. Li, and J. Sun. Objects365: A large-scale, high-quality dataset for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8430–8439, 2019.

K. Shen, J. Guo, X. Tan, S. Tang, R. Wang, and J. Bian. A study on relu and softmax in transformer. arXiv preprint arXiv:2302.06461, 2023.

- C. Sun, A. Shrivastava, S. Singh, and A. Gupta. Revisiting unreasonable effectiveness of data in deep learning era. In Proceedings of the IEEE international conference on computer vision, pages 843–852, 2017.

M. Varma, J.-B. Delbrouck, S. Hooper, A. Chaudhari, and C. Langlotz. Villa: Fine-grained visionlanguage representation learning from real-world data. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22225–22235, 2023.

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

- F. Wang, Y. Zhou, S. Wang, V. Vardhanabhuti, and L. Yu. Multi-granularity cross-modal alignment for generalized medical visual representation learning. Advances in Neural Information Processing Systems, 35:33536–33549, 2022.

- H. Wang, S. Ge, Z. Lipton, and E. P. Xing. Learning robust global representations by penalizing local predictive power. In Advances in Neural Information Processing Systems, pages 10506–10518, 2019.

J. Xu, S. De Mello, S. Liu, W. Byeon, T. Breuel, J. Kautz, and X. Wang. Groupvit: Semantic segmentation emerges from text supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18134–18144, 2022.

J. Xu, J. Hou, Y. Zhang, R. Feng, Y. Wang, Y. Qiao, and W. Xie. Learning open-vocabulary semantic segmentation models from natural language supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2935–2944, 2023.

J. Yang, J. Duan, S. Tran, Y. Xu, S. Chanda, L. Chen, B. Zeng, T. Chilimbi, and J. Huang. Visionlanguage pre-training with triple contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15671–15680, 2022.

- L. Yao, R. Huang, L. Hou, G. Lu, M. Niu, H. Xu, X. Liang, Z. Li, X. Jiang, and C. Xu. Filip: Fine-grained interactive language-image pre-training. arXiv preprint arXiv:2111.07783, 2021.

J. Yu, Z. Wang, V. Vasudevan, L. Yeung, M. Seyedhosseini, and Y. Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022.

- M. Yuksekgonul, F. Bianchi, P. Kalluri, D. Jurafsky, and J. Zou. When and why vision-language models behave like bag-of-words models, and what to do about it? arXiv preprint arXiv:2210.01936, 2022.

- Y. Zeng, X. Zhang, and H. Li. Multi-grained vision language pre-training: Aligning texts with visual concepts. arXiv preprint arXiv:2111.08276, 2021.

- S. Zhai, T. Likhomanenko, E. Littwin, D. Busbridge, J. Ramapuram, Y. Zhang, J. Gu, and J. Susskind. Stabilizing transformer training by preventing attention entropy collapse. ICML, 2023a.

- X. Zhai, A. Kolesnikov, N. Houlsby, and L. Beyer. Scaling vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12104–12113, 2022.
- X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. Sigmoid loss for language image pre-training. International Conference on Computer Vision, 2023b.

Y. Zhong, J. Yang, P. Zhang, C. Li, N. Codella, L. H. Li, L. Zhou, X. Dai, L. Yuan, Y. Li, et al. Regionclip: Region-based language-image pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16793–16803, 2022.

- C. Zhou, C. C. Loy, and B. Dai. Extract free dense labels from clip. In European Conference on Computer Vision, pages 696–712. Springer, 2022.

### A. Problems with using softmax for obtaining alignment weights

𝑆𝑜𝑓𝑡𝑚𝑎𝑥 is ubiquitously used to normalise activations that should or could be interpreted as probabilities, as it is for example the case of attention/alignmnet weights. One potential reason behind this choice is the dominating practice of using 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 as the output activation function for classification tasks, being the canonical link function for multinomial outputs. Another appealing property is that it acts as a differentiable 𝑚𝑎𝑥-operator, allowing for a natural interpretation of selecting one class out of multiple.

However, 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 can be problematic from a gradient flow perspective (Hoffmann et al., 2023; Shen et al., 2023; Zhai et al., 2023a), and in this section we will expand this observation and the implications it might have on our specific use case. Also, intuitively from its role as a soften 𝑚𝑎𝑥 operator, softmax prefers to converge to peaky uni-modal distribution, selecting one out of 𝑘, and is less likely to represent multi-modal distributions. This is due to how gradients flow through the activation, leading to winner-takes-all dynamics (Elfadel and Wyatt Jr, 1993; Peterson and Söderberg, 1989) that ensures the peakyness and unimodality of the distribution represented.

If we assume 𝑎(h) = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(h)1, for some h ∈ R𝑘, then we can write the derivative as

𝜕a𝑖 𝜕h𝑗

a𝑖 − a2𝑖 iff 𝑖 = 𝑗

(8)

=

−a𝑖a𝑗 otherwise

Assume we have some loss 𝐿 which is a function of 𝑖 a𝑖V𝑖, i.e. some values V𝑖 ∈ R𝑛 that have been summarised using attention weights a𝑖.

Softmax gradients vanish at initialisation. Assume we have a large number of patches or tokens we want to attend over. In our notation, 𝑘 ≫ 0. At initialisation, all preactivation entries h𝑖 will be small numbers of similar magnitude. The attention weights will be uniformally distributed over the 𝑘 patches, leading to a𝑖 ≈ 1𝑘 ≪ 1,∀𝑖. Due to the weights being almost uniformally distributed, different observation will lead to randomly selecting a different patch. Therefore in expectation the gradient through the softmax on a particular token 𝑖 will be scaled by 𝑘12 which will vanish very fast to 0 as 𝑘 grows. Note that in the rare scenario that the system picks the 𝑖-th element, the gradient becomes 1

𝑘 which also vanishes to 0 as 𝑘 grows. If we consider a very large 𝑘, this ensures that we have a plateau at initialization that might be hard to escape (or might take many updates to do so). See also

(Hoffmann et al., 2023) for a similar observation.

Softmax exhibits winner-takes-all dynamics. This has been understood and seen as a desirable property early on, see for example (Peterson and Söderberg, 1989) and (Elfadel and Wyatt Jr, 1993). One way to intuitively justify this behaviour is to think of the effect of applying the softmax operation multiple time (i.e. study the dynamics of a system whose transition function is just softmax). As shown in (Peterson and Söderberg, 1989) Fig. 5, the corners of the simplex act as attractors of this dynamical system, where from any initial condition, the system very quickly converges to one of the corners. This is caused by the dynamics of the gradients. When a particular weight is pushed up, all other weights are pushed down due to the normalisation. The amount by which the weight is pushed depends on its magnitude. So if a particular weight is larger and correlates positively with the desired behaviour, it will be pushed up proportionally more than other weights that correlate positively. Note that the particular form of the function (including the exponentiation) play a role in the form the gradients take, and removing the exponentiation will change the behaviour. These types of dynamics, have the downside of leading the distribution induced by the softmax to be unimodal.

1By abuse of notation, we will use a ∈ R𝑘, where a = 𝑎(h) and use a𝑖 for the 𝑖-th dimension of vector a

That is, softmax will act, as the name of the activation indicates, as a max operator, preferring to learn a behaviour where it picks one out of 𝑘, rather than multiple equally relevant candidates.

Softmax saturates proportional to its certainty Assume ∃𝑖 such that ∀𝑗, 𝑗 ≠ 𝑖 we have 𝑎𝑖 ≫ 𝑎𝑗. This implies that 1− 𝑎𝑖 → 0 and 𝑎𝑗 < 1− 𝑎𝑖. The gradient for the 𝑖-th position, according to equation 8, will be 𝑎𝑖(1− 𝑎𝑖) and will go to zero as linearly as 𝑎𝑖 approaches 1. The gradient for any other position 𝑗, will go to 0 at the same rate, as it will be roughly 𝑎𝑗 which is bounded from above from 1 − 𝑎𝑖. Note that a step of size Δ on ℎ, due to the exponentiation and normalization of softmax, will make 𝑎𝑖 → 1 exponentially fast for constant change in ℎ.

### B. Additional related works

We further expand here the discussion on achieving fine-grained understanding in vision-language models (VLMs) through additional losses and modules.

In addition to the approaches described in Section 3, another line of work involves proposes modifying the underlying vision transformer architecture to build modules that lead to a hierarchical grouping of image regions: e.g. GroupViT (Xu et al., 2022), OVSegmentor (Xu et al., 2023), HiCLIP (Geng et al., 2023). While these methods propose architectural changes, the objective used for training still involves having a global contrastive loss. Conversely, in our work, we use the standard vision transformer architecture and propose instead changes to the training objective to achieve finegrained understanding.

Moreover, note that several of these approaches (Xu et al., 2023) and the other methods who add a cross-modal encoder on top of the dual image-text encoder (Li et al., 2021; Yang et al., 2022) with captioning/masked language modelling losses start training from pre-trained text encoders and/or vision encoder.

Similarly, (Ranasinghe et al., 2023) improve the semantic and spatial information in dual encoders trained contrastively by changing the patch embeddings aggregation methods from average pooling to max pooling and by starting training with both pre-trained vision and language encoders. In our work, we focus specifically on the set-up of training the dual encoders from scratch.

### C. SPARC pseudo-code

##### Listing 1 provides JaX-alike pseudo-code for the SPARC objective detailing the construction of both the global and the local losses.

- 1 # Models:
- 2 # vision_encoder
- 3 # language_encoder
- 4 # Inputs:
- 5 # image - [B, H, W, C]
- 6 # text - [B, N]
- 7 # Hyperparameters:
- 8 # similarity_threshold
- 9 # global_loss_weight
- 10 # local_loss_weight
- 11 # inverse_temperature
- 12
- 13 def pairwise_contrastive_loss(a, b, labels):
- 14 labels = eye(a.shape[0])
- 15 logits_ab = dot(a * b.T) * inverse_temperature
- 16 return softmax_cross_entropy(logits=logits_ab , labels=labels , reduction=’mean’)
- 17
- 18 def masked_pairwise_contrastive_loss(a, b, mask):
- 19 batch_size , seq_len , _ = a.shape[0]
- 20 mask_logits = einshape(’bnm ->(bn)m’, 1.0 - mask , n=seq_len)
- 21 labels = einshape(’ns->(bn)s’, eye(a.shape[1]), b=batch_size)
- 22 logits = einsum(’bmd ,bnd ->bmn’, a, b) * inverse_temperature
- 23 logits = einshape(’bnm ->(bn)m’, logits)
- 24 loss = softmax_cross_entropy(logits=logits - mask_logits * INF , labels=labels)
- 25 loss = sum(loss * mask) / sum(mask)
- 26 return loss
- 27
- 28 # ---------- GLOBAL LOSS ----------
- 29
- 30 # encoders include adapters
- 31 v_patch_embed = vision_encoder(image)
- 32 l_token_embed , language_mask = language_encoder(text)
- 33
- 34 v_embed = l2_normalize(mean(v_patch_embed , axis=1), axis=-1)
- 35 l_embed = l2_normalize(mean(l_token_embed , axis=1), axis=-1)
- 36
- 37 loss_vl = pairwise_contrastive_loss(v_embed , l_embed)
- 38 loss_lv = pairwise_contrastive_loss(l_embed , v_embed)
- 39
- 40 global_loss = 0.5 * (loss_vl + loss_lv) # (eq 1)
- 41
- 42 # ---------- LOCAL LOSS ----------
- 43
- 44 # similarity calculation
- 45 similarity = einsum(’btd ,bpd ->btp’, l_token_embed , v_patch_embed)
- 46
- 47 # min -max normalisation
- 48 similarity = (similarity - min(similarity , axis=-1)) /
- 49 (max(similarity , axis=-1) - min(similarity , axis=-1)) # (eq 2)
- 50
- 51 # thresholding
- 52 similarity = where(similarity < similarity_threshold , 0.0 , similarity) # (eq 3)
- 53
- 54 # alignment -weighting
- 55 v_align_weights = similarity / sum(similarity , axis=-1) # (eq 4)
- 56 l_grouped_v_patch_embed = einsum(’btp ,bpd ->btd’, v_align_weights , v_patch_embed) # (eq 5)
- 57
- 58 l_grouped_v_patch_embed = l2_normalize(l_grouped_v_patch_embed , axis=-1)
- 59 l_token_embed = l2_normalize(l_token_embed , axis=-1)
- 60
- 61 loss_vl_local = masked_pairwise_contrastive_loss(l_grouped_v_patch_embed , l_token_embed , language_mask)
- 62 loss_lv_local = masked_pairwise_contrastive_loss(l_token_embed , l_grouped_v_patch_embed , language_mask)
- 63
- 64 local_loss = 0.5 * (loss_vl_local + loss_lv_local) # (eq 6)
- 65
- 66 # ---------- TOTAL (SPARC) LOSS ----------
- 67
- 68 loss = global_loss_weight * global_loss + local_loss_weight * local_loss # (eq 7)

##### Listing 1 | Pseudo-code for SPARC.

### D. Experiments details

#### D.1. Model architectures

For the dual-encoder, we use the standard Vision Transformers (ViTs) (Dosovitskiy et al., 2020) as image encoders and Transformers (Vaswani et al., 2017) as text encoders. We perform experiments with ViT-B models with different patch sizes (ViT-B/32 and ViT-B/16) and a ViT-L model with patch size 14 (ViT-L/14). Thus, for the ViT-B image encoder, we use a model with 12 layers, 768 width and 12 attention heads, while for the ViT-L image encoder we use a model with 24 layers, 1024 width and 16 attention heads. For the language encoder, we use an architecture with 12 layers, 768 width and 12 attention heads. The linear adapters 𝑔𝑣(·) and 𝑔𝑡(·) project the vision and language embeddings respectively to a shared embedding space of dimensionality 512.

#### D.2. Datasets

As described in Section 4, we use the following datasets for pre-training: ALIGN (Jia et al., 2021), JFT (Sun et al., 2017; Zhai et al., 2022) and LTIP (Long Text & Image Pairs) (Alayrac et al., 2022). Note that for JFT, where the images were semi-automatically annotated with a class-hierarchy of 30k labels, we flatten the hierarchical label structure and use all the assigned labels to describe the image. We use a multi-step training strategy where we alternate sampling batches from each of the 3 large datasets; the gradient updates are then performed by aggregating the gradients from computing the loss on one batch from each of the datasets.

#### D.3. Baselines

Our implementation of baselines follow the publicly available code (where available2) with a few minor differences we outline here.

In the original MGCA implementation, token-wise cross-modal alignment (see Eqn. 5 in the original paper) uses the last-layer attention weight from a visual token to the [CLS] token (averaged across multiple heads) to weight the loss terms for different visual tokens (and vice versa for language tokens). In our implementation, since we do not use the [CLS] token but instead use average pooling to get the global language/vision embeddings, we omit this weighting operation.

In the original GLoRIA implementation, language tokens are aggregated for each word to ensure that contrasted language embeddings refer to complete words (see Section 3.2.1 in the original paper); however, to ensure fair comparison, we do not have this additional aggregation operation, and instead use language tokens directly in local losses. Additionally, in our experiments we found that it is crucial to normalize the pairwise vision-language embedding similarities (see Eqn. 3 in the original paper) by √𝐷 where 𝐷 is the embedding size. Without this normalization, we found training with GLoRIA to be unstable. Moreover, recall that GLoRIA requires computing similarities between all token embeddings and all patch embeddings in the batch. This is memory expensive and it was not possible (due to device memory constraints) for batch sizes of 16348. Consequently, we used a batch size of 4096 for Gloria and trained the models for 800k steps (to match the number of examples seen by the other baseline). See discussion in Section D.6 for detailed computation of FLOPs and memory usage of GLoRIA.

For FILIP [50] we follow the original paper and implement token dropping for FILIP which the authors propose in order to reduce the large memory consumption of their method. In the original paper the authors comment on the training difficulty in the original paper (cf. in the Appendix

2GLoRIA: https://github.com/marshuang80/gloria, MGCA: https://github.com/HKU-MedAI/MGCA

Objective IN IN-V2 Th IN-V2 MF IN-V2 TI IN-R IN-C IN-A IN-Sketch

CLIP 66.7 66.2 58.9 71.5 63.2 42.6 15.1 51.7 FILIP 52.7 50.7 44.0 55.8 47.1 28.7 8.4 38.2 CLIP + FILIP 66.5 65.8 58.2 71.1 63.0 42.3 15.1 51.3 SPARC (ours) 68.1 67.0 59.7 72.0 64.9 44.5 16.7 53.2

ViT-B/32

CLIP 71.6 70.9 63.7 74.8 71.1 48.5 32.2 56.8 FILIP 56.6 55.6 48.9 59.7 54.0 33.2 14.4 43.1 CLIP + FILIP 71.8 70.5 63.4 74.4 70.6 47.8 32.0 56.2 SPARC (ours) 72.6 71.1 64.4 75.0 72.0 48.5 33.8 57.3

ViT-B/16

- Table 9 | Top-1 accuracy (in %) of zero-shot classification on ImageNet (IN) and its variants ImageNetV2 Threshold (IN-V2 Th), ImageNet-V2 Matched Frequency (In-V2 MF), ImageNet-V2 Top Images (IN-V2 TI), ImageNet-R (IN-R), ImageNet-C (IN-C), ImageNet-Sketch (IN-Sketch). All methods have been trained on ALIGN, JFT, LTIP for the same number of training steps.

A.3.”...training is extremely unstable and the Nan loss easily happens.”). We observed similar training instability in our setup across a wide range of learning rates and weight decay parameters. This training instability leads to significant performance degradation compared to CLIP. We hypothesize that the non-standard additional tricks that FILIP uses such as image augmentations, backtranslation of captions and custom prompt ensembling could potentially improve training stability; note that we do not use these tricks in order to ensure a fair comparison across methods. Given FILIP’s training instability, we conducted a number of additional experiments combining CLIP and FILIP in order to better understand the training instability. Below in Tables 9 and 10 we present these results – as can be seen combining these two methods leads to some improvements on some benchmarks while some performance degradation on other benchmarks.

Finally, all methods in our paper use learned temperature parameters (instead of fixed temperatures as is done in the original MGCA and GLoRIA implementations) as our experiments showed that this significantly improved performance for all methods.

MSCOCO Flickr30k image-to-text text-to-image image-to-text text-to-image

Objective R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10 R@1 R@5 R@10

ViT-B/32

CLIP 53.5 78.2 86.7 38.4 64.8 74.9 79.2 95.1 97.2 66.5 88.0 93.1 FILIP 35.6 61.0 73.1 26.2 51.0 62.4 62.6 86.9 92.9 50.5 77.7 84.9 CLIP + FILIP 52.0 77.0 85.6 37.8 64.4 74.5 81.2 95.4 97.1 66.8 87.7 92.3 SPARC (ours) 55.0 79.1 87.3 39.7 65.9 75.7 82.5 96.2 97.6 67.7 88.2 93.0

ViT-B/16

CLIP 56.2 80.6 88.2 42.4 68.6 78.3 84.0 96.1 98.2 71.6 90.3 94.1 FILIP 40.2 66.0 76.3 29.5 55.3 66.3 69.0 89.8 94.0 55.8 81.5 87.9 CLIP + FILIP 54.9 79.0 87.4 41.3 67.7 77.5 82.7 97.0 98.4 71.1 90.5 94.7 SPARC (ours) 57.6 81.2 88.5 43.0 68.6 78.5 84.4 97.6 98.7 72.0 91.2 94.9

- Table 10 | Results on zero-shot image-to-text and text-to-image retrieval on MSCOCO and Flickr30k datasets. R@i denotes Recall at i. All methods have been trained on ALIGN, JFT, LTIP for the same number of training steps.

#### D.4. Hyperparameters details

We train all models using the AdamW (Loshchilov and Hutter, 2017) optimizer, a cosine learning rate schedule with linear warm-up of 2500 steps. For all methods, we sweep over learning rate and weight decay values in the following ranges: learning rate in [7𝑒 − 4, 9𝑒 − 4, 1.1𝑒 − 4] and weight decay in [0.1, 0.2, 0.3]. We use a batch size of 16348 (except for GLoRIA for which we use 4096 batch size) and we pre-train the ViT-B models for 200k steps (≈ 3.2 billion data points).

For the other SPARC hyperparameters, we set the global loss weight 𝜆𝑔 = 0.5 and we sweep the local loss weight in 𝜆𝑓 ∈ [0.5, 1.0, 5.0, 10.0]. Moreover, we use a learned temperature parameter 𝜏.

For baseline specific hyperparameters, we follow the publicly available code (where available) and the original papers. For MGCA (Wang et al., 2022), as described in the paper, we set the weighing of the different losses 𝜆1 = 1, 𝜆2 = 1, 𝜆3 = 1, the number of attention heads for computing the cross-modal embeddings to 1 with a 128 embedding dimension. For MGCA’s crossmodal prototype alignment loss, we use 500 prototypes with 𝜖 = 0.05 and 3 iterations for the Sinkhorn-Knopp clustering algorithm.

For FILIP, we implemented the token dropping procedure described in the paper and use 20% token dropping in our experiments.

For PACL, we closely follow the original paper in terms of implementation up to one notable detail – we include a learnable temperature parameter in the loss as we found this to significantly improve performance.

#### D.5. Prompt ensembling for zero-shot classification

Following Radford et al. (2021) and Yao et al. (2021) we use prompt templates to augment the label for classification tasks. We use the prompt templates format from Yao et al. (2021):

[prefix]{class label}, [suffix] (9)

For the [prefix], we use the templates from Radford et al. (2021). On the other hand, for the [suffix], we use the templates from Yao et al. (2021), which shows that adding the reference word ‘it’ at the end of the prompt, e.g. ‘I like it’, further improves performance.

#### D.6. Memory consumption and FLOPS for the different methods

We provide detailed numbers for the FLOPS (in TFLOPS) and of the Peak Memory (in MB) in Table 11.

FLOPS (TFLOPS) Peak memory (MB) Objective B = 2048 B = 4096 B = 8192 B = 16384 B = 2048 B = 4096 B = 8192 B = 16384 CLIP 1.15 2.29 4.57 9.14 4394 4452 5889 8578 PACL 1.2 2.46 5.24 12.8 4682 6267 9786 14785 GLoRIA 3.34 13.21 − − 8013 13840 − − MGCA 1.16 2.31 4.62 9.23 4412 4462 5936 8681 FILIP 1.37 3.17 8.09 27.25 4394 5230 8657 15463 SPARC (ours) 1.15 2.3 4.6 9.19 4408 4450 5914 8620

- Table 11 | TFLOPS and peak memory usage for one update step of each method for different batch sizes.

#### D.7. Semantic segmentation

For zero-shot semantic segmentation, we pass the patch embeddings through the extra dense layer and the adapter to compute the cosine similarity with the text embeddings for the ground-truth classes. Similarly to (Mukhoti et al., 2023) we compute the mean Intersection over Union (mIoU) only for the foreground classes.

#### D.8. SPARC backbones in vision language models

We train the Perceiver Resampler part of Flamingo (Alayrac et al., 2022) on the ALIGN (Jia et al., 2021), LTIP (Long Text & Image Pairs) (Alayrac et al., 2022) and VTP (Video & Text Pairs) (Alayrac et al., 2022) datasets. VTP consists of 27 million short videos paired with text descriptions, where each video if 22s on average. We use the AdamW optimizer, a cosine learning rate schedule with peak learning rate of 1𝑒 − 4, linear warmup with 5000 warm-up steps and 250k training steps in total.

#### D.9. SPARC vs CLIP Faithfulness Examples

To further understand the ability of SPARC and CLIP models to faithfully describe the elements in the image, we provide several qualitative examples. Thus, for MSCOCO, we chose examples where the top-1 retrieved caption for both SPARC and CLIP is not part of the ground truth captions, but where where SPARC has higher all-token K-Precision (Figure 5) and higher K-Precision restricted to nouns and adjectives (6). From these figure, we notice that captions retrieved using the CLIP representations describe objects that not present in the image (e.g. “several signs for bars” when there are none present) or get the number of objects wrong (e.g. "two motorcycles" when there is only one motorcycle). Alternatively, captions retrieved using the SPARC representations are more faithful to the image, but also provide more descriptive details (e.g. "young boy in white shirt", "dinner table with a place setting").

[Figure 28]

##### Figure 5 | SPARC vs CLIP vs Ground Truth for examples where SPARC has higher all-token K-Precision (K-P)

[Figure 29]

##### Figure 6 | SPARC vs CLIP vs Ground Truth for examples where SPARC has higher K-Precision restricted to nouns and adjectives (K-Pna)

