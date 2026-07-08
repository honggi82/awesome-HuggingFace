## VidLA: Video-Language Alignment at Scale

Mamshad Nayeem Rizve Fan Fei Jayakrishnan Unnikrishnan Son Tran Benjamin Z. Yao Belinda Zeng Mubarak Shah Trishul Chilimbi

{mnrizve, feiff, jayunn, sontran, benjamy, zengb, shahmub, trishulc}@amazon.com

# arXiv:2403.14870v1[cs.CV]21Mar2024

### Abstract

In this paper, we propose VidLA, an approach for videolanguage alignment at scale. There are two major limitations of previous video-language alignment approaches. First, they do not capture both short-range and long-range temporal dependencies and typically employ complex hierarchical deep network architectures that are hard to integrate with existing pretrained image-text foundation models. To effectively address this limitation, we instead keep the network architecture simple and use a set of data tokens that operate at different temporal resolutions in a hierarchical manner, accounting for the temporally hierarchical nature of videos. By employing a simple two-tower architecture, we are able to initialize our video-language model with pretrained image-text foundation models, thereby boosting the final performance. Second, existing video-language alignment works struggle due to the lack of semantically aligned large-scale training data. To overcome it, we leverage recent LLMs to curate the largest video-language dataset to date with better visual grounding. Furthermore, unlike existing video-text datasets which only contain short clips, our dataset is enriched with video clips of varying durations to aid our temporally hierarchical data tokens in extracting better representations at varying temporal scales. Overall, empirical results show that our proposed approach surpasses state-of-the-art methods on multiple retrieval benchmarks, especially on longer videos, and performs competitively on classification benchmarks.

### 1. Introduction

Vision-language alignment is crucial for solving many vision-language tasks like text-guided retrieval [41, 42, 47, 65, 74], visual question answering [11, 40, 88, 99], visual captioning [40, 42, 61, 86], etc. In the past few years, training on web-scale image-text pairs has significantly improved performance in many of these vision-language tasks [43, 44, 65, 99]. However, humans perceive the world as a continuous stream of images, which is also reflected in the increasing number of videos uploaded to the web

MSVD (T2V)

ActivityNet (T2V)

55.0

Vatex (T2V)

70.0

50.0

75.0

65.0

DiDeMo (T2V)

70.0

45.0

60.0

60.0 55.0

65.0

55.0

40.0

50.0

MSR-VTT (V2T)

60.0

60.0

55.0

45.0

50.0

45.0

MSR-VTT (T2V)

45.0 50.0 55.0 60.0

50.0 55.0

25.0

60.0 65.0 55.0

30.0

DiDeMo (V2T)

70.0 75.0 80.0 85.0

65.0 70.0

35.0

60.0 65.0

40.0

MSR-VTT (ZS T2V)

70.0

75.0 80.0

ActivityNet (V2T)

Vatex (V2T)

VidLA

MSVD (V2T)

Previous SOTA

Figure 1. Recall@1 performance on retrieval benchmarks compared to previous SoTA with ViT-B scale models.

daily. Despite the ubiquitous use of videos, when compared to image-language alignment, video-language alignment is significantly more challenging for two key reasons.

First, unlike image-language data, it is much harder to collect aligned video-language data at scale. To address this issue, most prior works utilize automatic speech recognition (ASR) systems [2, 66, 102] to extract textual transcripts and generate paired video-language data for largescale training [57, 93, 100]. However, it has been shown that transcripts often corresponds poorly with their associated visual contents [32, 56, 57, 72]. As a result, some recent works [12, 38, 46, 79] skipped large-scale videolanguage training and worked around by utilizing languagealigned image encoders, followed by lightly adapting them with temporal modules on small-scale video datasets with paired textual descriptions [6, 59]. However, training on such small-scale datasets often leads to overfitting [94] and does not encourage learning temporally diverse representations [38]. Second, since the vision transformer architecture lacks strong visual inductive bias such as that in CNN-type architectures, it requires large-scale pretraining data for effective generalization [21, 67]. In case of videos, this problem is amplified further due to the added temporal dimension. Facing this challenge, to be more efficient, existing works utilize factorized [4, 7, 60] or hierarchical [45, 48, 52, 68] space-time attention. However, neither of these solutions are optimal for large-scale video-language alignment, as factorized space-time attention overly focuses on aggregating redundant local spatio-

temporal information [45], while hierarchical space-time attention makes the use of pretrained language-aligned nonhierarchical image encoders [44, 65] challenging. Our work addresses both challenges in large-scale video-language alignment using large language models and a novel hierarchical temporal attention mechanism.

to model global spatio-temporal relationships in a temporally hierarchical manner, inspired from prior art [24, 104], we incorporate several Multi-Scale Temporal [MST] tokens that interact with all patch tokens at varying temporal resolutions to summarize the video semantic concepts at various temporal scales. To make this space-time attention operation more efficient, we update the patch tokens by attending over all the [MST] tokens and other patch tokens from the same frame. Finally, we utilize a [CLS] token to attentively aggregate information from all the [MST] and patch tokens. We utilize this aggregated spatio-temporal representation for video-language alignment training. Our hierarchical temporal attention design not only models local temporal relationships but also models global temporal relationships at different temporal hierarchies while utilizing strong pretrained image-text encoders.

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]|
|---|

| | |
|---|---|
|Text Encoder| |

| | |
|---|---|
|Video Encoder with Hierarchical Temporal Attention| |

[Figure 4]

[Figure 5]

[Figure 6]

A person is fishing near a rocky shore..

The speaker is having difficulty..

[Figure 7]

A person is

A person is

To summarize, in this work, we make two major technical contributions: (i) we propose several techniques to utilize LLMs to generate a large scale video-text dataset where the generated text has high semantic correlation with the visual content. (ii) we design a hierarchical temporal modeling approach that effectively incorporates pretrained imagetext encoders and handles videos of varying lengths in the training set to improve downstream performance as shown in Figure 1. We extensively evaluate the performance of our method on several video-text retrieval benchmarks to demonstrate its efficacy and the effectiveness of our data curation techniques and modeling approach. A summary of our approach is provided in Figure 2.

Captions Subtitles

Video Clips

- Figure 2. Figure summarizing our video-language alignment training approach with a two-tower architecture, where text encoder and video encoder with hierarchical temporal attention are trained with info-NCE losses to align video representations with subtitle and caption text representations simultaneously. We generate the captions using a multi-modal LLM and utilize an LLM to summarize the caption and subtitle texts.

In addition, having large-scale video-text data set is crucial for video-language alignment training. Towards that end, we construct a very large dataset, with ∼800M videotext pairs, to train video-language alignment model at scale. In this context, we propose several simple data curation strategies using LLMs [8, 14–16, 78] to improve the semantic correlation between textual description and associated visual content of large-scale video-language datasets. First, we utilize recent multi-modal large language models (MLLM) [17, 42, 49, 98, 105] to generate auxiliary captions grounded in visual content. Second, to scale our solution, we generate captions at a low frame rate, capitalizing on temporal redundancy of videos. Third, we augment the existing video-language datasets by incorporating videos of varying lengths to facilitate robust alignment across diverse temporal scales. We utilize LLMs for summarizing longer video descriptions, preventing training asymmetry when we sample the same number of frames for videos of all durations but use longer texts for longer videos. The LLM summarization also helps when long textual descriptions cover disparate concepts.

### 2. Related Works

Vision-Language Representation Learning Recently, image-language models [34, 44, 65, 74, 97, 99] have drawn huge attention because of their effectiveness in learning generic visual representations that are transferable to several downstream tasks like classification, retrieval, etc. This success can partly be attributed to the recently available large-scale image-text datasets [18, 69–71, 76]. However, in case of videos, there’s no large-scale aligned videolanguage dataset. Therefore, to perform video-language pretraining, most recent works [23, 29, 55, 62, 94] bootstrap from a pretrained image-language model and then perform some form of lightweight adaptation on the video datasets.

Adapting CLIP to Video Many video foundation model works aim to adapt CLIP for video representation. Most use a straightforward approach and encode frame samples with CLIP image encoder and pool across frames with various mechanisms [23, 29, 55] to represent the video. Other works insert additional temporal specific modelings such as divided space-time attention [7] or adapters [33] into CLIP vision transformer (ViT) layers [12, 62, 63, 79, 96]. Among others there are also works using novel parallel architectures [50] or using addition special tokens to capture temporal in-

To efficiently utilize the non-hierarchical image-text pretrained models while accounting for the temporally hierarchical nature of videos, we factorize the space-time attention operation into two parts: local and global temporal attention. First, we focus on modeling the local temporal relationships by treating videos as collections of temporal tubes of single patch tokens. This attention operation focuses on capturing fine-grained motion across frames. Next,

|Clip Duration # clips Length (s)|Subtitle # sent # words Summarized # words<br><br>|Caption # cap # words Summarized # words|
|---|---|---|
|Short 496M 13.2 Medium 212M 30.4 Long 100M 60.3<br><br>|2.1 31.6 31.6* 4.7 72.1 19.5 9.2 142.1 22.4<br><br>|1.0 10.9 10.9*<br>2.3 24.8 15.2 4.5 48.8 18.8<br>|

Table 1. Statistics of our curated training data set YT-VidLA-800M. *For short video clips, texts are not summarized.

teraction between frames [94].

onds) to videos in existing large-scale datasets [57, 59, 93]. The medium length clips on the other hand are on average 30 seconds long, which is similar in length to videos in common retrieval benchmarks [3, 92]. The longer clips are on average 60 seconds in duration. Overall, we extract around 500 million short video clips, 200 million medium length video clips, and 100 million long video clips as summarized in Table 1. Next, we discuss about our visually grounded auxiliary caption generation process.

Video-Language Datasets For image-language pretraining, web images paired with alt-text have demonstrated to be extremely effective and can scale up to billions of samples [11, 69, 75, 81]. Video dataset using similar alt-text such as WebVid [6] are often at a much smaller scale. Alternatively, VideoCC [59] dataset is generated by finding visually similar video clips from existing image text data. Video subtitle datasets on the other hand are much more widely available and easy to scale, leading to wide adoption [57, 93, 100, 101], however these type of videos are often very short clips segmented by sentence boundaries, and the subtitles are usually noisy and have poor visual grounding. In this work, instead of generating a new dataset, we propose a way to effectively use existing large scale video dataset to improve video text alignment.

Caption Generation To improve visual grounding in language supervision, we generate auxiliary captions for each clip using multi-modal LLMs. Particularly, we utilize BLIP-2 [42] to generate captions for the frames of the extracted video clips. To be efficient, capitalizing on the temporal redundancy of videos, we generate these captions at a very low frame-rate (∼ 0.1 FPS). To aggregate the framelevel captions to generate the video clip-level caption, we perform text summarization, which we discuss next.

### 3. Video-Language Pretraining Dataset

Text Summarization We use an LLM [78] to summarize the individual frame captions to obtain the caption for each video clip. Additionally, we summarize ASR subtitles from longer videos to obtain right abstraction for video-language alignment training. Furthermore, caption and subtitle summarization address another practical problem: it reduces the size of the input text, making it feasible to fit within the context length window of CLIP’s pretrained text encoder. After this operation, each video clip is paired with two summarized texts corresponding to ASR subtitle and generated caption. We present the statistics of YT-VidLA-800M before and after summarization in Table 1.

A key component of our Video-Language Alignment method summarized in Figure 2 is a high quality large scale video-language dataset. In order to be able to effectively train the video-text models with hierarchical temporal attention, and to allow the model to learn to align videos with different temporal scales, we need a dataset with videos of different lengths, and corresponding text annotations with varying levels of temporal abstraction. We describe our novel data curation scheme in detail below.

Source Videos We utilize 20 million videos from the YTTemporal-1B [100] dataset for creating video-text dataset since its the largest collection of publicly available videos. These videos cover a diverse set of video concepts, and have been filtered to ensure better visual groundedness as well as to protect user privacy. Unlike prior works which utilize the video frame-subtitle pairs from this dataset, we create a new dataset composed of video clips paired with subtitles and generated captions which we call YT-VidLA-800M. Next, we describe our multi-scale video clip extraction process.

### 4. Method

In VidLA, we utilize an extension of the two-tower architecture for image-language alignment from CLIP [65]. Particularly, we retain the CLIP text encoder architecture and extend CLIP’s vision encoder to improve its temporal modeling capability by introducing a novel attention mechanism illustrated in Figure 3. We provide details of our video encoder in the following.

Video Clip Extraction To extract video clips, first, we punctuate the ASR subtitles using a bert-based [20] punctuation model to split the full subtitle into sentences. Next, we split the videos at sentence boundaries to generate clips, where each clip covers one or more sentences. To facilitate video-language alignment across different temporal scales, we split each video into clips of varying temporal lengths. To be particular, our shortest clips are on average 13 seconds long in duration, which are similar in length (6-13 sec-

Preliminaries The vision encoder accepts as input a video clip v consisting of T RGB frames vt ∈ RH×W×3,t ∈ {0,1,...,T − 1} each of size H × W pixels sampled from the video. Following vision transformer [21] and pretrained image-language models [44, 65], we split each frame into non-overlapping patches of size P × P yielding N = HW/P2 patches for each frame. Each of the NT patches

|Spatial<br><br>Temporal<br><br>Patch Token Update|
|---|

|Spatial<br><br>Temporal<br><br>Spatial<br><br>Temporal<br><br>[MST]-0 Update [MST]-1 Update [MST] Token Update|
|---|

|Spatial<br><br>Temporal<br><br>[CLS] Token Update|
|---|

Spatial

Spatial

Spatial

Spatial

Spatial

Temporal

Spatially-Local Temporal Attention

Global Spatio-Temporal Attention

[CLS] Token [MST] Tokens Patch Tokens of Different Frames Query Token Key Token

- Figure 3. Figure summarizing the different tokens and the attention mechanisms used to update the tokens in our proposed Hierarchical Temporal Attention. This toy example uses N = 4 patches, T = 4 frames, U = 2 levels of temporal hierarchy , V = 1 [MST] token per level and temporal scale r = 2. Hierarchical temporal attention can be factorized into two parts. Spatially Local Temporal Attention (left): Patch tokens only attend to its neighbors across time. For instance, first patch token of the first frame gets updated by only attending to the first patch token of all the other frames. Global Spatio-temporal Attention (right): To capture global spatio-temporal semantics efficiently, we update the patch tokens by attending to other patch tokens from the same frame as well as all the [MST] tokens. The third and fourth column depict the hierarchical [MST] token update mechanism. Particularly, from the third column we observe that [MST]-0 gets updated by attending to all the patch tokens and other [MST] tokens of lower temporal resolution. The next column demonstrates the multi-scale [MST] attention mechanism where the second [MST] token, [MST]-1, only attends to patch tokens from a subset of frames with a higher stride. The [CLS] token acts as an aggregator and attentively pulls information from both [MST] and patch tokens.

is linearly mapped with a learnable matrix and then combined with learnable spatial and temporal position embeddings to obtain a sequence of TN patch tokens, represented by Z0 ∈ RTN×d, where d denotes the dimension of each token. We incorporate a set of UV [MST] tokens to capture summarized information at different temporal scales from the video where U represents the number temporal hierarchies and V represents the number of [MST] tokens at each temporal hierarchy. We also include a [CLS] token to capture the global representation of the video (see e.g., [19]). We create the final input sequence, Z0 ∈ R(1+UV +TN)×d, by prepending the learnable [CLS] token and UV additional [MST] tokens to the sequence of TN patch tokens. The sequence of input tokens are passed through L transformer layers. We use Zl to denote the sequence after the l-th layer. In each layer the sequence is treated with two steps of attention followed by an MLP layer as summarized in Figure 3 and detailed next.

an attention mask, M ∈ RTN×TN, formally defined as

0 if mod(|j − i|,N) = 0 −∞ otherwise.

Mi,j =

Spatially local temporal attention is then performed as

##### ZlSlT = MMSA(LN( Zl−1), M) + Zl−1 (1)

where LN(.) is layer normalization [5] operation and MMSA(.,.) is masked multi-head self-attention which can be expressed as MMSA(Z,M) := softmax(QKT/

##### √

d + M)V ∈ RTN×d; here Q,K,V are query, key, value embeddings of the sequence of input tokens Z obtained through linear projection and M is the input attention mask.

After the attention computation, we again prepend the [CLS] and [MST] tokens to the updated patch tokens, ZlSlT, to obtain the token sequence ZlSlT = [(Zl−1)[CLS],(Zl−1)[MST], ZlSlT].

Spatially Local Temporal Attention Inspired from a long line of works [24, 25, 73] that seek to model finegrained temporal motion for video understanding, we employ spatially local temporal attention. As the first operation in any l-th layer of the transformer, we remove the [CLS] and [MST] tokens from the sequence of input tokens to that layer, Zl−1, to apply this attention only on the patch tokens, Zl−1 ∈ RTN×d. To capture finegrained temporal motion during this attention operation, each patch token only attends to patch tokens from other frames in the same spatial position, effectively allowing attention only along the temporal dimension. This operation can be represented using

Global Spatio-Temporal Attention To efficiently model the global spatio-temporal semantics in a hierarchical manner, we utilize the hierarchical [MST] tokens for guiding the global spatio-temporal attention. We employ an asymmetric attention mechanism to update the [CLS], [MST], and patch tokens as illustrated in the second grid in Figure 3. To keep the attention operation computationally efficient, each patch token attends to all patch tokens from the same frame, and to all the UV [MST] tokens ∈ RUV ×d. The patch token updates can be expressed using an attention mask, M[PATCH] ∈ RTN×(1+UV +TN), defined as

M[PATCH] = [0,MG] where 0 is a TN × (1 + UV ) matrix of zeros and MG is a TN × TN matrix with

0 if N i = N j −∞ otherwise

MGi,j =

where ⌊.⌋ indicates the FLOOR function.

The update procedure for [MST] tokens is designed to capture the temporally hierarchical nature of video concepts. The attention mask for each [MST] token is determined by the hierarchy level of that token, ranging from 0 to U − 1, and the temporal scale r. Specifically, the [MST] tokens from a particular hierarchy level u attend to [MST] tokens from lower temporal hierarchies and to the [PATCH] tokens from every ru-th frame. As there are V [MST] tokens in each hierarchy level, the updates for the [MST] tokens can be expressed using another attention mask, M[MST] ∈ RUV ×(1+UV +TN) where the first V rows correspond to [MST] tokens of hierarchy level 0, followed by V rows of hierarchy level 1, and so on. The attention mask can be formally expressed as M[MST] = [−∞1, M[MST],self, M[MST],patch] where 1 is a UV × 1 vector of all 1’s, M[MST],self is a UV × UV matrix and M[MST],patch is a UV × TN matrix with

0 if V i ≥ V j −∞ otherwise

M[MST],self

i,j =

0 if mod N j ,r⌊

V ⌋ = 0 −∞ otherwise

i

patch

M[MST],

i,j =

Note that both patch and [MST] tokens do not attend to the [CLS] token to limit propagation of global information into the these local tokens. We update the [CLS] token by attending to all the patch and [MST] tokens. This asymmetric update ensures that the [CLS] token merely acts as an aggregator where it attentively pulls information from all tokens. We denote the attention mask for updating the [CLS] token as M[CLS] ∈ R1×(1+UV +TN). We set all the entries of M[CLS] to 0 to allow attention computation with all tokens. Finally, we vertically stack these attention masks, [M[CLS],M[MST],M[PATCH]], to generate the attention mask, M, for global spatio-temporal attention. The global spatiotemporal attention mechanism also includes an MLP and skip connection as summarized in the following,

ZlGST = MMSA(LN(ZlSlT),M)) + ZlSlT Zl = MLP(ZlGST) + ZlGST (2)

We propagate these updated token embeddings, Zl, to the next transformer layer. Finally, we use a linear projection of the [CLS] token from the last transformer layer as the video embedding for video-language alignment training.

Pretraining Objective For video-language alignment training, we use language supervision from both ASR subtitle, ts, and caption, tc. Let’s assume s ∈ RD, c ∈ RD and v ∈ RD are the encoded features vectors for subtitle, caption and video. We use the commonly used info-NCE loss [10] as the objective function for video-language alignment training. The overall objective function is

1 B

L =

B

1 B

(Livs) +

i=1

B

(Livc) (3)

i=1

where, Lvs and Lvc are info-NCE loss between video representation and the language representation from subtitle s and caption c respectively; for each loss,

exp(vi⊤ti/τ)

Livt = −log

− log

B j=1 exp(v⊤

i tj/τ)

exp(t⊤i vi/τ)

B j=1 exp(t⊤

i vj/τ)

where t ∈ {c,s}, B is the batch size and τ is the learnable temperature scale.

### 5. Experiments and Results

Implementation Details We initialize our text and video encoders form pretrained OpenAI CLIP [65] checkpoints. We randomly initialize the [MST] tokens. To ensure that the initializion of our video encoder is close to CLIP’s vision encoder, we initialize the projection matrices of spatially local temporal attention with zero. During training, we uniformly sample 12 frames from each video clip. We use multi-scale random crop [83] with a ratio of 1.0 and 0.8 to crop the video to 224 × 224 while preserving aspect ratio. We also apply random horizontal flip for augmentation. We train our models for 3 epochs. We use a initial learning rate of 2e − 5 with cosine decay to 4e − 8. For training, we utilize 128 A100 GPUs and set the batch size to 4096. We set the number of hierarchies, U, to 3, the number of [MST] tokens in each hierarchy, V , to 4, and the temporal scale r to 2. We provide additional training and finetuning implementation details in the Supplementary.

Video-Text Retrieval Datasets We evaluate our retrieval performance on MSR-VTT [92], DiDeMo [3], ActivityNet Captions [37], MSVD [9] and VATEX [89] datasets. On all these datasets, we finetune on the standard training split and test it on the standard test/val splits. Following prior works [6, 39, 91, 94], we concatenate the multiple descriptions to form a paragraph and perform paragraph-to-video retrieval on DiDeMo and ActivityNet Captions datasets.

Main Results We compare the retrieval performance of our proposed method VidLA with other recent works on MSR-VTT, DideMo, ActivityNet Captions, MSVD, and VATEX datasets and report the results in Table 2 and 3. We use VidLA-X/Y to denote the variant of our model that

|Method<br><br>MSR-VTT Text-to-Video R@1 R@5 R@10 Avg MdR↓ MnR↓|MSR-VTT Video-to-Text R@1 R@5 R@10 Avg MdR↓ MnR↓<br><br>|
|---|---|
|ClipBERT [39] 22.0 46.8 59.9 42.9 6.0 − Support Set [64] 30.1 58.5 69.3 52.6 3.0 − HD-VILA [93] 35.6 65.3 78.0 59.6 3.0 − All-in-One [80] 37.9 68.1 77.1 61.0 − − Frozen [6] 32.5 61.5 71.2 55.1 3.0 −|− − − − − − 28.5 58.6 71.6 52.9 3.0 − − − − − − − 37.5 66.1 77.4 60.3 − − − − − − − −<br><br>|

CLIP-ViT-B/32 CLIP4Clip [55] 44.5 71.4 81.6 65.8 2.0 15.3 42.7 70.9 80.6 64.7 2.0 11.6 CenterCLIP [103] 44.2 71.6 82.1 66.0 2.0 15.1 42.8 71.7 82.2 65.6 2.0 10.9 CLIP2TV [27] 46.1 72.5 82.9 67.2 2.0 15.2 43.9 73.0 82.8 66.6 2.0 11.1 CAMoE* [13] 47.3 74.2 84.5 68.7 2.0 11.9 49.1 74.3 84.3 69.2 2.0 9.9 DRL [87] 47.4 74.6 83.8 68.6 2.0 − 45.3 73.9 83.3 67.5 2.0 − STAN* [50] 49.0 74.8 83.5 69.1 2.0 − − − − − − − PIDRo [31] 48.2 74.9 83.3 68.8 2.0 12.6 47.4 74.8 84.1 68.8 2.0 8.7 Cap4Video [91] 49.3 74.3 83.8 69.1 2.0 12.0 47.1 73.7 84.3 68.4 2.0 8.7 UATVR* [22] 49.8 76.1 85.5 70.5 2.0 12.9 51.1 74.8 85.1 70.3 1.0 8.3 CLIPViP [94] 50.1 74.8 84.6 69.8 1.0 − − − − − − − CLIPViP* [94] 55.9 77.0 86.8 73.2 1.0 − − − − − − − VidLA-B/32 55.6 79.7 86.9 74.1 1.0 11.4 55.1 79.9 88.0 74.3 1.0 6.9 VidLA-B/32* 60.9↑5.0 81.6 89.4 77.3 1.0 8.7 60.8↑9.7 82.4 89.1 77.4 1.0 6.3

CLIP-ViT-B/16 BridgeFormer [28] 37.6 64.8 75.1 59.2 3.0 − − − − − − − CLIP2TV [27] 49.3 74.7 83.6 69.2 2.0 13.5 46.9 75.0 85.1 69.0 2.0 10.0 TS2-Net [51] 49.4 75.6 85.3 70.1 2.0 13.5 46.6 75.9 84.9 69.1 2.0 8.9 Cap4Video [91] 51.4 75.7 83.9 70.3 1.0 12.4 49.0 75.2 85.0 69.7 2.0 8.0 DRL* [87] 53.3 80.3 87.6 73.7 1.0 − 56.2 79.9 87.4 74.5 1.0 − STAN* [50] 54.6 78.4 85.1 72.7 1.0 − − − − − − − PIDRo* [31] 55.9 79.8 87.6 74.4 1.0 10.7 54.5 78.3 87.3 73.4 1.0 7.5 UATVR* [22] 53.5 79.5 88.1 73.7 1.0 10.2 54.5 79.1 87.9 73.8 1.0 7.6 CLIPViP [94] 54.2 77.2 84.8 72.1 1.0 − − − − − − − CLIPViP* [94] 57.7 80.5 88.2 75.5 1.0 − − − − − − − VidLA-B/16 58.0 81.1 87.8 75.6 1.0 10.4 56.1 80.5 88.7 75.1 1.0 6.8 VidLA-B/16* 61.1↑3.4 83.8 90.4 78.4 1.0 8.1 63.1↑6.9 84.7 90.8 79.5 1.0 6.1

Two Stage Models with Cross-Modal Fusion Re-Ranking

VINDLU†[12] 46.5 71.5 80.4 66.1 − − − − − − − − UMT† [46] 51.0 76.5 84.2 70.6 − − 49.0 77.0 84.7 70.2 − − InternVideo(ViT-L)†* [90] 55.2 79.6 87.5 74.1 − − 57.9 − − − − −

- Table 2. Retrieval performance on the MSR-VTT benchmark, metrics used are recall at (R@) 1, 5, 10, average recall (Avg), top candidate median rank (MdR) and mean rank (MnR). * indicates inference with dual-softmax. † indicates two-stage method with candidate reranking. Performance delta is calculated against SoTA two-tower methods.

uses ViT-X/Y as the vision encoder, e.g., VidLA-B/32 uses ViT-B/32 as the vision encoder. We present results with and without using dual-softmax [13] for score normalization prior to ranking at the inference stage. Our proposed method outperforms all prior works using a similar ViT backbone by a significant margin. Particularly, from results reported in Table 2, we observe that VidLA-B/32 outperforms the second best method, CLIP-ViP, by 5.5% on MSRVTT for text-to-video retrieval in terms of R@1 without dual-softmax. We notice similar improvement (3.8%) with ViT-B/16 backbone. We also notice a large improvement on the video-to-text retrieval task. Table 3, demonstrates a similar pattern on other four datasets. Particularly, we observe a larger improvement on datasets with longer videos such as ActivityNet Captions and DiDeMo, where our proposed method outperforms the second best method, CLIPViP, by 8.4% and 10.1% respectively. These results demonstrate that our proposed method not only outperforms the prior best method but also attains larger improvement if the downstream dataset is temporally longer.

### 6. Analysis and Discussion

We empirically validate our design choices on the model architecture, dataset temporal scales, language supervision as well as their combined effect by conducting a series of experiments to evaluate the model’s retrieval performance. In all experiments, unless otherwise specified, we use the VidLA-B/32 model pretrained on an 80M subset of the YTVidLA-800M dataset for 1 epoch, finetuned on MSR-VTT dataset. For these analysis experiments, we evaluate the retrieval performance without DSL. This 80M subset is constructed by sampling about 2M random source videos and then splitting them into short, medium and long clips as discussed in Section 3. For a fair comparison with other methods, we also utilize the same ViT-B/32 model as the vision encoder, initialized from the same CLIP checkpoint, and trained with the same compute and data budget.

Attention Design To analyze the effectiveness of [MST] guided hierarchical temporal attention mechanism, we conduct a series of experiments with different attention configurations and report the results in Table 4. The first two

|Method<br><br>DiDeMo R@1 R@5 R@10<br><br>|ActivityNet Captions R@1 R@5 R@10|MSVD R@1 R@5 R@10<br><br>|Vatex R@1 R@5 R@10|
|---|---|---|---|
|ClipBERT [39] 20.4 48.0 60.8 Support Set [64] − − − HD-VILA [93] 28.8 57.4 69.1 All-in-One [80] 32.7 61.4 73.5 Frozen [6] 34.6 65.0 74.7|21.3 49.0 63.5<br><br>29.2 61.6 −<br><br>28.5 57.4 94.0<br><br>22.4 53.7 67.7<br><br><br>− − −|− − −<br><br>28.4 60.0 72.9<br><br>− − − − − − 33.7 64.7 76.3<br><br>|− − − 45.9 82.4 90.4<br><br>− − − − − − − − −|

CLIP-ViT-B/32 CLIP4Clip [55] 43.4 70.2 80.6 40.5 72.4 − 46.2 76.1 84.6 − − − CenterCLIP [103] − − − 43.9 74.6 85.8 47.6 77.5 86.0 − − − CLIP2TV [27] 45.5 69.7 80.6 44.1 75.2 − 47.0 76.5 85.1 − − − CAMoE* [13] 43.8 71.4 − 51.0 77.7 − 49.8 79.2 87.0 − − − DRL [87] 47.9 73.8 82.7 44.2 74.5 86.1 48.3 79.1 87.3 63.5 91.7 96.5 STAN* [50] 51.3 75.1 83.4 − − − − − − − − − PIDRo* [31] 48.6 75.9 84.4 44.9 74.5 86.1 47.5 77.5 86.0 − − − UATVR [22] 43.1 71.8 82.3 − − − 46.0 76.3 85.1 61.3 91.0 95.6 CLIPViP [94] 48.6 77.1 84.4 51.1 78.4 88.3 − − − − − − CLIPViP* [94] 53.8 79.6 86.5 59.1 83.9 91.3 − − − − − − VidLA-B/32 56.9 82.2 89.2 61.3 84.8 91.3 48.6 77.9 85.7 66.5 86.2 88.4 VidLA-B/32* 62.2↑8.4 84.6 90.0 69.2↑10.1 88.2 93.3 52.7↑2.9 80.4 87.0 73.7↑7.2 87.6 89.1

CLIP-ViT-B/16 BridgeFormer [28] 37.0 62.2 73.9 − − − 52.0 82.8 90.0 − − − DRL [87] 49.0 76.5 84.5 46.2 77.3 88.2 50.0 81.5 89.5 65.7 92.6 96.7 UATVR [22] 45.8 73.7 83.3 − − − 49.7 79.0 87.3 64.5 92.6 96.8 Cap4Video [91] 52.0 79.4 87.5 − − − 51.8 80.8 88.3 66.6 93.1 97.0 CLIPViP [94] 50.5 78.4 87.1 53.4 81.4 90.0 − − − − − − CLIPViP* [94] 55.3 82.0 89.3 61.4 85.7 92.6 − − − − − − VidLA-B/16 61.1 83.7 89.1 65.2 87.4 92.8 51.5 79.9 86.9 69.2 87.1 88.9 VidLA-B/16* 64.8↑6.9 86.0 91.8 73.0↑10.8 89.9 93.6 55.9↑3.9 82.3 88.3 75.8↑9.2 88.3 89.3

Two Stage Models with Cross-Modal Fusion Re-Ranking

VINDLU†[12] 61.2 85.8 91.0 55.0 81.4 89.7 − − − − − − UMT† [46] 61.6 86.8 91.5 58.3 83.9 91.5 71.9 94.5 97.8 − − − InternVideo(ViT-L)* [90] 57.9 82.4 88.9 62.2 85.9 93.2 58.4 84.5 90.4 71.1 − −

- Table 3. Text-to-video Retrieval performance on the DiDeMo, ActivityNet Captions, MSVD and Vatex datasets. * indicates inference with dual-softmax. † indicates two-stage method with candidate re-ranking. Performance delta is calculated against SoTA two-tower methods.

|[MST] Hierarchy Local<br><br>|MSR-VTT Retrieval R@1 R@5 R@10 Avg|
|---|---|
|✗ ✗ ✗ ✓ ✗ ✗ ✓ ✓ ✗ ✓ ✗ ✓ ✓ ✓ ✓<br><br>|49.1 75.3 83.5 69.3<br><br>49.2 77.6 85.2 70.7<br><br><br>50.0 77.6 85.4 71.0<br><br>51.3 76.5 85.0 70.9 53.5 77.5 85.6 72.2<br><br><br>|

- Table 4. Comparison of retrieval performances on MSR-VTT dataset with different settings for [MST] token attention and the effect of spatially-local temporal attention.

|Multi-Scale<br><br>|MSR-VTT Retrieval R@1 R@5 R@10 Avg|
|---|---|
|✗ ✓<br><br>|51.9 78.2 85.6 71.9 53.5 77.5 85.6 72.2<br><br>|

- Table 5. Ablation study on the length distribution of videos in the pretraining dataset. Retrieval performance improves when the dataset is created with short, medium and long clips

chies in [MST] tokens. On the other hand, the fourth row shows the effectiveness of spatially-local temporal attention, where it provides a significant improvement in terms of R@1 retrieval performance over the seon. Finally, the last row confirms the efficacy of our proposed temporal attention mechanism, providing a substantial 4.4% improvement over the baseline. Overall, these results not only validate the effectiveness of our proposed attention mechanism but also highlight the efficacy of its individual components.

Temporal Scales in Pretraining Data To analyze the impact of incorporating multiple temporal scales in the proposed pretraining dataset, we compare a model pretrained on the 80M subset containing short, medium and long clips against a model trained on only short short clips from the same set of 2M videos. For a fair comparison, we train these models for same number of steps. We present the finetuned results in Table 5 and observe that including multiple scales in the pretraining dataset helps boost retrieval performance.

Retrieval Performance on Videos of Different Lengths To conduct a more finegrained analysis of the performance of our method, in the left plot of Figure 4, we compare

rows demonstrate the effectiveness of [MST] tokens, even without any temporal hierarchy. Third row demonstrates the effectiveness of introducing multiple temporal hierar-

54

VidLA

62

VidLA

TimeSformer

52

TimeSformer

60

Proxy Attention

Proxy Attention

Joint ST Attention

50

58

Joint ST Attention

56

48

R@1

54

R@1

46

52

44

50

48

42

46

40

44

1 10 80 # Training Samples in Millions

Short Medium Long Duration of Video Clips

- Figure 4. Retrieval performance on MSR-VTT compared to other attention mechanisms Left: R@1 numbers for validation videos separated into 3 bins of different durations. VidLA consistently improves over baselines for all video durations. Right: Scaling up the pretraining dataset improves the performance. Our architecture improves over other attention mechanisms at all data scales.

the performances of VidLA with respect to other attention methods on videos of different lengths. For this analysis, we report MSR-VTT R@1 results for three splits of videos in the validation set. Particulalry, we sort the videos by length and pick the shortest third for the short split, longest third for the long split and the remaining for the medium split. We observe that VidLA consistently outperforms other methods on all splits of different video lengths.

|Sub Cap Sum<br><br>|MSR-VTT Retrieval R@1 R@5 R@10 Avg|
|---|---|
|✓ ✓ ✗ ✗ ✓ ✓ ✓ ✗ ✓ ✓ ✓ ✓<br><br>|36.3 65.0 76.3 59.2 48.9 74.1 84.0 69.0 50.1 76.7 84.5 70.4 53.5 77.5 85.6 72.2<br><br>|

- Table 6. Comparison of finetuned retrieval performances on MSRVTT dataset with different language supervision during pretraining. We compare the effectiveness of using subtitles, captions and whether or not they are summarized.

Training Data Size It is well-known that performance of retrieval models scales with the pretraining data size in the contrastive learning setting. We study our model’s performance as a function of the pretraining dataset size by pretraining different models on datasets of sizes 80M, 10M and 1M. We report the results in the right plot on Figure 4 and compare the performance of VidLA with other attention methods. We notice that VidLA outperforms all the methods across all data scales.

Effect of Different Language Supervision To validate the efficacy of utilizing both subtitles and captions for language supervision, as well as the effectiveness of text summarization, we pretrain our model with different combinations of text sources and summarization. From the results presented in Table 6, we observe that the model’s performance is better with supervision from both subtitles and captions compared to using only one of the two. Additionally, removing

summarization significantly degrades performance. Without summarization, video-text alignment suffers due to increased verbosity in longer videos and the inability to leverage CLIP’s pretrained embedding layer due to increased context length.

|Method Frames<br><br>|K400 Views Top-1|Sth-sth-v2 Views Top-1<br><br>|
|---|---|---|
|TimeSformer-B/16 [7] 96 VideoMAE-B/16 [77] 16 VideoMAE-v2-B/16 [82] 16 ViViT-L/16 [4] 32 VideoSwin-B [53] 32 UMT-B/16800e [46] 8 VidLA-B/32 16 VidLA-B/16 16<br><br>|1 × 3 80.7 5 × 3 81.5<br><br>5 × 3 81.5<br><br>1 × 3 81.7 3 × 4 82.7 3 × 4 85.7<br><br>5 × 3 82.4 5 × 3 84.9<br><br><br><br><br>|1 × 3 62.4<br>2 × 3 70.8 2 × 3 71.2<br><br><br>1 × 1 65.9<br><br>1 × 3 69.6<br>2 × 3 70.8<br><br><br>2 × 3 67.9 2 × 3 69.9<br><br><br>|

Table 7. Comparison of finetuned classification performances on Kinetics-400 and Something-Something-v2. VidLA models using ViT-B backbones achieve competitive results in spite of being pretrained only for alignment.

Classification Results Even though our proposed method primarily focuses on video-language alignment, we evaluate the performance of our method on a related downstream task, i.e., action recognition. We add a classification head on top of the video encoder from VidLA and finetune it on the popular benchmark datasets Kinetics-400 [36] and Something-Something-V2 [30]. We report the results of the finetuned models in Table 7. Although VidLA was pretrained only for video-language alignment, we observe that VidLA performs competitively even against models such as VideoMAE that use dense pretraining objectives to promote the learning of finegrained features.

### 7. Conclusion

In this work, we propose a novel hierarchical temporal modeling architecture that captures temporal relationships at multiple temporal scales while remaining flexible to leverage image-text pretrained models. We also introduce an approach for utilizing LLMs to create the largest videolanguage dataset with better semantic alignment between video and language. We empirically validate the efficacy of our proposed hierarchical temporal attention mechanism as well as its design choices on data with varying temporal lengths and at different dataset sizes, demonstrating its advantage over other performant temporal modeling approaches. Our extensive experimentation also validates our data curation choices. Overall, our results highlight the importance of both high-quality large-scale training data as well as simple and scalable temporal architecture, and establishes VidLA as the new state-of-the-art on multiple video-text retrieval benchmarks while demonstrating its competitiveness on classification benchmarks.

### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022. 2
- [2] Dario Amodei, Sundaram Ananthanarayanan, Rishita Anubhai, Jingliang Bai, Eric Battenberg, Carl Case, Jared Casper, Bryan Catanzaro, Qiang Cheng, Guoliang Chen, et al. Deep speech 2: End-to-end speech recognition in english and mandarin. In International conference on machine learning, pages 173–182. PMLR, 2016. 1
- [3] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer vision, pages 5803–5812, 2017. 3, 5
- [4] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luˇci´c, and Cordelia Schmid. Vivit: A video vision transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6836–6846,

2021. 1, 8

- [5] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450,

2016. 4

- [6] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728– 1738, 2021. 1, 3, 5, 6, 7
- [7] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In ICML, volume 2, page 4, 2021. 1, 2, 8
- [8] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 2
- [9] David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies, pages 190–200,

2011. 5

- [10] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PMLR, 2020. 5
- [11] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794, 2022. 1, 3
- [12] Feng Cheng, Xizi Wang, Jie Lei, David Crandall, Mohit Bansal, and Gedas Bertasius. Vindlu: A recipe for effective video-and-language pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern

- Recognition, pages 10739–10750, 2023. 1, 2, 6, 7
- [13] Xing Cheng, Hezheng Lin, Xiangyu Wu, Fan Yang, and Dong Shen. Improving video-text retrieval by multi-stream corpus alignment and dual softmax loss. arXiv preprint arXiv:2109.04290, 2021. 6, 7, 2, 3
- [14] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2023. 2
- [15] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.
- [16] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022. 2
- [17] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning,

2023. 2

- [18] Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin Johnson. Redcaps: Web-curated image-text data created by the people, for the people. arXiv preprint arXiv:2111.11431,

2021. 2

- [19] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 4171–4186. Association for Computational Linguistics, 2019. 4
- [20] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 3
- [21] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 1, 3
- [22] Bo Fang, Wenhao Wu, Chang Liu, Yu Zhou, Yuxin Song, Weiping Wang, Xiangbo Shu, Xiangyang Ji, and Jingdong Wang. Uatvr: Uncertainty-adaptive text-video retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 13723–13733, October

2023. 6, 7

- [23] Han Fang, Pengfei Xiong, Luhui Xu, and Yu Chen. Clip2video: Mastering video-text retrieval via image clip. arXiv preprint arXiv:2106.11097, 2021. 2

- [24] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6202–6211, 2019. 2, 4, 1
- [25] Christoph Feichtenhofer, Axel Pinz, and Andrew Zisserman. Convolutional two-stream network fusion for video action recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1933– 1941, 2016. 4
- [26] Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, and Zicheng Liu. Violet: End-toend video-language transformers with masked visual-token modeling. arXiv preprint arXiv:2111.12681, 2021. 1
- [27] Zijian Gao, Jingyu Liu, Weiqi Sun, Sheng Chen, Dedan Chang, and Lili Zhao. Clip2tv: Align, match and distill for video-text retrieval. arXiv preprint arXiv:2111.05610,

2021. 6, 7

- [28] Yuying Ge, Yixiao Ge, Xihui Liu, Dian Li, Ying Shan, Xiaohu Qie, and Ping Luo. Bridging video-text retrieval with multiple choice questions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16167–16176, 2022. 6, 7
- [29] Satya Krishna Gorti, No¨el Vouitsis, Junwei Ma, Keyvan Golestan, Maksims Volkovs, Animesh Garg, and Guangwei Yu. X-pool: Cross-modal language-video attention for text-video retrieval. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5006–5015, 2022. 2
- [30] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The “something something” video database for learning and evaluating visual common sense. In ICCV, volume 1, page 5, 2017. 8
- [31] Peiyan Guan, Renjing Pei, Bin Shao, Jianzhuang Liu, Weimian Li, Jiaxi Gu, Hang Xu, Songcen Xu, Youliang Yan, and Edmund Y. Lam. Pidro: Parallel isomeric attention with dynamic routing for text-video retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11164–11173, October

2023. 6, 7

- [32] Tengda Han, Weidi Xie, and Andrew Zisserman. Temporal alignment networks for long-term video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2906–2916, 2022. 1
- [33] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International Conference on Machine Learning, pages 2790–2799. PMLR, 2019. 2
- [34] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904–

4916. PMLR, 2021. 2

- [35] Peng Jin, Hao Li, Zesen Cheng, Kehan Li, Xiangyang Ji, Chang Liu, Li Yuan, and Jie Chen. Diffusionret: Generative text-video retrieval with diffusion model. arXiv preprint

- arXiv:2303.09867, 2023. 2, 3
- [36] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, Mustafa Suleyman, and Andrew Zisserman. The kinetics human action video dataset, 2017. 8
- [37] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In Proceedings of the IEEE international conference on computer vision, pages 706–715, 2017. 5
- [38] Jie Lei, Tamara L Berg, and Mohit Bansal. Revealing single frame bias for video-and-language learning. arXiv preprint arXiv:2206.03428, 2022. 1
- [39] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and Jingjing Liu. Less is more: Clipbert for video-and-language learning via sparse sampling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7331–7341, 2021. 5, 6, 7
- [40] Chenliang Li, Haiyang Xu, Junfeng Tian, Wei Wang, Ming Yan, Bin Bi, Jiabo Ye, Hehong Chen, Guohai Xu, Zheng Cao, et al. mplug: Effective and efficient vision-language learning by cross-modal skip-connections. arXiv preprint arXiv:2205.12005, 2022. 1
- [41] Gen Li, Nan Duan, Yuejian Fang, Ming Gong, and Daxin Jiang. Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 11336–11344, 2020. 1
- [42] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 1, 2, 3
- [43] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR, 2022. 1
- [44] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, 34:9694–9705, 2021. 1, 2, 3
- [45] Kunchang Li, Yali Wang, Peng Gao, Guanglu Song, Yu Liu, Hongsheng Li, and Yu Qiao. Uniformer: Unified transformer for efficient spatial-temporal representation learning. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29,

2022. OpenReview.net, 2022. 1, 2

- [46] Kunchang Li, Yali Wang, Yizhuo Li, Yi Wang, Yinan He, Limin Wang, and Yu Qiao. Unmasked teacher: Towards training-efficient video foundation models. arXiv preprint arXiv:2303.16058, 2023. 1, 6, 7, 8
- [47] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pretraining for vision-language tasks. In Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXX 16, pages 121–

- 137. Springer, 2020. 1
- [48] Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. Mvitv2: Improved multiscale vision transformers for classification and detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4804–4814, 2022. 1
- [49] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023. 2
- [50] Ruyang Liu, Jingjia Huang, Ge Li, Jiashi Feng, Xinglong Wu, and Thomas H Li. Revisiting temporal modeling for clip-based image-to-video knowledge transferring. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6555–6564, 2023. 2, 6, 7
- [51] Yuqi Liu, Pengfei Xiong, Luhui Xu, Shengming Cao, and Qin Jin. Ts2-net: Token shift and selection transformer for text-video retrieval. In European Conference on Computer Vision, pages 319–335. Springer, 2022. 6
- [52] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3202–3211, 2022. 1
- [53] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3202–3211, June

2022. 8

- [54] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 1
- [55] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing, 508:293–304, 2022. 2, 6, 7, 1, 3
- [56] Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. End-to-end learning of visual representations from uncurated instructional videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9879– 9889, 2020. 1
- [57] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2630–2640, 2019. 1, 3
- [58] Liliane Momeni, Mathilde Caron, Arsha Nagrani, Andrew Zisserman, and Cordelia Schmid. Verbs in action: Improving verb understanding in video-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15579–15591, 2023. 2
- [59] Arsha Nagrani, Paul Hongsuck Seo, Bryan Seybold, Anja Hauth, Santiago Manen, Chen Sun, and Cordelia Schmid. Learning audio-video modalities from image captions. In European Conference on Computer Vision, pages 407–426. Springer, 2022. 1, 3
- [60] Daniel Neimark, Omri Bar, Maya Zohar, and Dotan Asselmann. Video transformer network. In Proceedings of the IEEE/CVF international conference on computer vi-

- sion, pages 3163–3172, 2021. 1
- [61] Van-Quang Nguyen, Masanori Suganuma, and Takayuki Okatani. Grit: Faster and better image captioning transformer using dual visual features. In European Conference on Computer Vision, pages 167–184. Springer, 2022. 1
- [62] Bolin Ni, Houwen Peng, Minghao Chen, Songyang Zhang, Gaofeng Meng, Jianlong Fu, Shiming Xiang, and Haibin Ling. Expanding language-image pretrained models for general video recognition. In European Conference on Computer Vision, pages 1–18. Springer, 2022. 2
- [63] Junting Pan, Ziyi Lin, Xiatian Zhu, Jing Shao, and Hongsheng Li. St-adapter: Parameter-efficient image-to-video transfer learning. Advances in Neural Information Processing Systems, 35:26462–26477, 2022. 2
- [64] Mandela Patrick, Po-Yao Huang, Yuki Asano, Florian Metze, Alexander Hauptmann, Joao Henriques, and Andrea Vedaldi. Support-set bottlenecks for video-text representation learning. arXiv preprint arXiv:2010.02824, 2020. 6, 7
- [65] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 1, 2, 3, 5
- [66] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International Conference on Machine Learning, pages 28492–

28518. PMLR, 2023. 1

- [67] Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. Do vision transformers see like convolutional neural networks? Advances in Neural Information Processing Systems, 34:12116– 12128, 2021. 1
- [68] Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan, Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed, Judy Hoffman, et al. Hiera: A hierarchical vision transformer without the bellsand-whistles. arXiv preprint arXiv:2306.00989, 2023. 1
- [69] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 2, 3
- [70] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.
- [71] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018. 2
- [72] Nina Shvetsova, Anna Kukleva, Xudong Hong, Christian

- Rupprecht, Bernt Schiele, and Hilde Kuehne. Howtocaption: Prompting llms to transform video annotations at scale. arXiv preprint arXiv:2310.04900, 2023. 1
- [73] Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. Advances in neural information processing systems, 27, 2014. 4
- [74] Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. Flava: A foundational language and vision alignment model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15638–15650, 2022. 1, 2
- [75] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023. 3
- [76] Bart Thomee, David A. Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. Commun. ACM, 59(2):64–73, jan 2016. 2
- [77] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in neural information processing systems, 35:10078–10093, 2022. 8
- [78] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 2, 3, 1
- [79] Junke Wang, Dongdong Chen, Zuxuan Wu, Chong Luo, Luowei Zhou, Yucheng Zhao, Yujia Xie, Ce Liu, Yu-Gang Jiang, and Lu Yuan. Omnivl: One foundation model for image-language and video-language tasks. Advances in neural information processing systems, 35:5696–5710,

2022. 1, 2

- [80] Jinpeng Wang, Yixiao Ge, Rui Yan, Yuying Ge, Kevin Qinghong Lin, Satoshi Tsutsui, Xudong Lin, Guanyu Cai, Jianping Wu, Ying Shan, et al. All in one: Exploring unified video-language pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6598–6608, 2023. 6, 7, 1
- [81] Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang. Git: A generative image-to-text transformer for vision and language. arXiv preprint arXiv:2205.14100, 2022. 3
- [82] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2: Scaling video masked autoencoders with dual masking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14549–14560, June 2023. 8
- [83] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. Temporal segment networks for action recognition in videos. IEEE transactions on pattern analysis and machine intelligence, 41(11):2740–2755, 2018. 5
- [84] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, and Luc Van Gool. Temporal seg-

- ment networks for action recognition in videos. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(11):2740–2755, 2019. 1
- [85] Mengmeng Wang, Jiazheng Xing, and Yong Liu. Actionclip: A new paradigm for video action recognition. arXiv preprint arXiv:2109.08472, 2021. 2
- [86] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. In International Conference on Machine Learning, pages 23318–23340. PMLR, 2022. 1
- [87] Qiang Wang, Yanhao Zhang, Yun Zheng, Pan Pan, and Xian-Sheng Hua. Disentangled representation learning for text-video retrieval. arXiv preprint arXiv:2203.07111,

2022. 6, 7, 2, 3

- [88] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for all vision and visionlanguage tasks. arXiv preprint arXiv:2208.10442, 2022. 1
- [89] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4581–4591, 2019. 5
- [90] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022. 6, 7, 1
- [91] Wenhao Wu, Haipeng Luo, Bo Fang, Jingdong Wang, and Wanli Ouyang. Cap4video: What can auxiliary captions do for text-video retrieval? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10704–10713, 2023. 5, 6, 7
- [92] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5288–5296, 2016. 3, 5
- [93] Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5036–5045, 2022. 1, 3, 6, 7
- [94] Hongwei Xue, Yuchong Sun, Bei Liu, Jianlong Fu, Ruihua Song, Houqiang Li, and Jiebo Luo. Clip-vip: Adapting pretrained image-text model to video-language representation alignment. arXiv preprint arXiv:2209.06430, 2022. 1, 2, 3, 5, 6, 7
- [95] Shen Yan, Tao Zhu, Zirui Wang, Yuan Cao, Mi Zhang, Soham Ghosh, Yonghui Wu, and Jiahui Yu. Videococa: Video-text modeling with zero-shot transfer from contrastive captioners. arXiv preprint arXiv:2212.04979, 2022. 1
- [96] Taojiannan Yang, Yi Zhu, Yusheng Xie, Aston Zhang, Chen Chen, and Mu Li. Aim: Adapting image mod-

- els for efficient video action recognition. arXiv preprint arXiv:2302.03024, 2023. 2
- [97] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive languageimage pre-training. arXiv preprint arXiv:2111.07783,

2021. 2

- [98] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023. 2
- [99] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022. 1, 2
- [100] Rowan Zellers, Jiasen Lu, Ximing Lu, Youngjae Yu, Yanpeng Zhao, Mohammadreza Salehi, Aditya Kusupati, Jack Hessel, Ali Farhadi, and Yejin Choi. Merlot reserve: Neural script knowledge through vision and language and sound. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16375–16387, 2022. 1, 3
- [101] Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu, Jae Sung Park, Jize Cao, Ali Farhadi, and Yejin Choi. Merlot: Multimodal neural script knowledge models. Advances in Neural Information Processing Systems, 34:23634– 23651, 2021. 3
- [102] Yu Zhang, James Qin, Daniel S Park, Wei Han, ChungCheng Chiu, Ruoming Pang, Quoc V Le, and Yonghui Wu. Pushing the limits of semi-supervised learning for automatic speech recognition. arXiv preprint arXiv:2010.10504, 2020. 1
- [103] Shuai Zhao, Linchao Zhu, Xiaohan Wang, and Yi Yang. Centerclip: Token clustering for efficient text-video retrieval. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 970–981, 2022. 6, 7, 2, 3
- [104] Bolei Zhou, Alex Andonian, Aude Oliva, and Antonio Torralba. Temporal relational reasoning in videos. In Proceedings of the European conference on computer vision (ECCV), pages 803–818, 2018. 2
- [105] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 2

### A. Implementation details

In this section, we provide additional implementation details of pretraining as well finetuning for both retrieval and classification tasks. We also include additional details regarding YT-VidLA-800M creation.

#### A.1. Additional pretraining details

For pretraining, we set the weight decay to 0.02. We use AdamW [54] optimizer and set β1 and β2 to 0.9 and 0.999 respectively. We set gradient clipping to 10.

#### A.2. Finetuning details for retrieval

For retrieval finetuning on all downstream datasets, we utilize 32 frames. For updating model parameters, we utilize AdamW optimizer with an initial learning rate of 1e − 5 and decay it to 1e − 6 using cosine annealing. We set the weight decay to 0.2. For VidLA-B/32 finetuning on all datasets, we set the batch size to 128, whereas, for VidLAB/16 finetuning, we set the batchsize to 196. Unlike pretraining, we do not perform any multi-scale spatial crop for finetuning, rather, we perform center crop while preserving aspect ratio. We train VidLA-B/32 model for 440 steps on MSVD dataset. We respectively train VidLA-B/32 for 2.5×, 2.5×, 2×, and 6× steps on MSR-VTT, DiDeMo, ActivityNet Captions, and VATEX datasets. For VidLAB/model, we train for 4×, 1.5×, 2×, 1×, and 6× steps on MSR-VTT, DiDeMo, ActivityNet Captions, MSVD and VATEX datasets, respectively.

#### A.3. Finetuning details for classification

For finetuning the video encoders from the pre-trained VidLA-B/32 for classification, we train for 100 epochs with a batch size of 512 for Kinetics-400 and 40 epochs with batch size 1024 for Something-Something-v2. For VidLAB/16, we train for 75 epochs and 40 epochs respectively for Kinetics-400 and Something-Something-v2, each with a batch size of 512. We use the dense sampling approach of [24] for Kinetics-400, and TSN [84] uniform sampling for Something-Something-v2.

#### A.4. Additional details for YT-VidLA-800M creation

For caption generation, we use Blip-2 ViT-L OPT2.7B [42]. We utilize Llama-2 13B Chat [78] model for caption and ASR subtitle summarization. We provide the summarization prompt in the following.

Summarize the following sentences into a single sentence, not exceeding 25 words. Do not output any additional text and use any external information. <input text>

### B. Zero-Shot Retrieval

We evaluate the zero-shot retrieval performance of VidLAB/16 on the MSR-VTT dataset. We present the results in Table 8. We compare the performance with both two-tower and two-stage methods, without specific constraints on vision backbone size (e.g., ViT-B/16). We observe that VidLA outperforms all the prior works by a significant margin that employ similar-sized architectures. To be more specific, VidLA outperforms the recent two-stage method UMT [46] by 6%. Surprisingly, we also find that VidLA outperforms InternVideo [90] by 0.8%, even though InternVideo utilizes a larger vision backbone. These results provide further evidence for the effectiveness of VidLA in the video-language alignment task.

Method R@1 Frozen [6] 18.7 VIOLET [26] 25.9 CLIP4Clip [55] 30.6 VINDLU† [12] 32.0 Singularity [38] 34.0 OmniVL [79] 34.6 VideoCoCa [95] 34.3 UMT-B† [46] 35.5 InternVideo(ViT-L)* [90] 40.7 VidLA-B/16 36.6 VidLA-B/16* 41.5

Table 8. Zero-shot retrieval results on MSR-VTT. * indicates inference with dual-softmax. † indicates two-stage method with candidate re-ranking.

### C. Zero-Shot Multiple Choice

The objective of the zero-shot multiple choice task is to find the correct answer from a list of options. We evaluate the performance of VidLA-B/16 on the MSR-VTT dataset for this task and report the results in Table 9. We observe that VidLA outperforms the previous state-of-the-art, InternVideo, by 0.9%. We also notice that VidLA outperforms All-in-one by 11.9%. Overall, these results further demonstrate the generalizability of our large-scale video-language alignment training.

Method Accuracy All-in-one [80] 80.3 InternVideo(ViT-B) [90] 91.3 VidLA-B/16 92.2

Table 9. Zero-shot multiple choice results on MSR-VTT.

### D. Zero-Shot Classification

We evaluate the performance of VidLA-B/16 on the zero-shot action classification task using the Kinetics-400

dataset. We only compare with methods that do not utilize Kinetics-400 data (labels and videos) for pretraining. To compare with prior art, we report results on both validation and test sets of Kinetics-400. We perform a ‘single-view’ evaluation and utilize uniformly sampled 32 frames. For evaluation, following CLIP [65], we obtain the embedding of all the class labels using the text encoder while utilizing the default prompt templates for Kinetics-400 classification. We consider the prediction as correct if the groundtruth label has the maximum similarity with the video embedding. We report the results in Table 10. We observe that our proposed method outperforms all prior works by a noticeable margin. To be precise, VidLA outperforms CLIP by around 11% on both validation and test sets. VidLA also outperforms, ActionCLIP [85] by 3.8%. We also notice that VidLA outperforms current state-of-the-art VFC [58] on both validation and test sets. Note that VFC [58] employs a verb focused contrastive loss during pretraining which greatly improves performance on action recognition task since the extracted verbs are, in principle, very similar to the downstream action recognition labels. Therefore, we also report a baseline score from VFC [58] which does not employ such a verb focused contrastive objective. VidLA outperforms the VFC-baseline by 3.5% on the Kinetics-400 test set. These results further validate the effectiveness of VidLA in action recognition, complementing the finetuned results reported in Table 7 of the main text.

Method Top-1 Accuracy Validation-Set

CLIP [65] 48.9 ActionCLIP [85] 56.4 VFC [58] 59.4 VidLA-B/16 60.2

Test-Set

Flamingo-3B [1] 45.2 Flamingo-80B [1] 49.1 Flamingo-9B [1] 49.7 CLIP [65] 47.9 VFC-Baseline [58] 55.6 VFC [58] 58.8 VidLA-B/16 59.1

Table 10. Zero-shot classification results on Kinetics-400.

### E. Video-to-Text Retrieval Results

We present video-to-text retrieval results on DiDeMo, ActivityNet Captions, MSVD, and VATEX datasets in Table 11, 12, 13, and 14 respectively. Similar to the text-tovideo retrieval results reported in the main text, we observe that VidLA outperforms prior art on video-to-text retrieval by a significant margin in most settings. However, InternVideo outperforms VidLA-B/16 on the VATEX dataset, likely due to its larger backbone architecture.

DiDeMo Retrieval R@1 R@5 R@10 Avg

Method

CLIP-ViT-B/32 CLIP4Clip [55] 42.5 70.6 80.2 64.4 CAMoE* [13] 45.5 71.2 − − DRL [87] 45.4 72.6 82.1 66.7 DiffusionRet* [35] 50.3 75.1 82.9 69.4 VidLA-B/32 55.0 82.8 88.9 75.6 VidLA-B/32* 62.2 85.1 91.1 79.5 CLIP-ViT-B/16 DRL [87] 49.9 75.4 83.3 69.5 InternVideo(ViT-L)* 59.1 − − − UMT† 59.5 84.9 90.5 78.3 VidLA-B/16 59.6 84.4 90.2 78.0 VidLA-B/16* 67.5 87.5 91.7 82.2

- Table 11. Video-to-text retrieval on DiDeMo dataset. * indicates inference with dual-softmax. † indicates two-stage method with candidate re-ranking.

Method

ActivityNet Captions Retrieval R@1 R@5 R@10 Avg

CLIP-ViT-B/32 CLIP4Clip [55] 42.5 74.1 85.8 67.5 CenterCLIP [103] 44.5 75.7 86.2 68.8 DRL [87] 42.2 74.0 86.2 67.5 CAMoE* [13] 49.9 77.4 − − DiffusionRet* [35] 47.4 76.3 86.7 70.1 VidLA-B/32 60.8 85.0 91.8 79.2 VidLA-B/32* 69.1 88.8 93.7 83.9 CLIP-ViT-B/16 DRL [87] 45.7 76.5 87.8 70.0 CenterCLIP [103] 46.7 77.1 88.0 70.6 UMT† 56.0 83.5 91.7 77.1 InternVideo(ViT-L)* 62.8 − − − VidLA-B/16 63.4 87.5 93.2 81.4 VidLA-B/16* 72.9 90.6 94.9 86.1

- Table 12. Video-to-text retrieval on ActivityNet Captions dataset.

* indicates inference with dual-softmax. † indicates two-stage method with candidate re-ranking.

### F. Effect of Image-Language Pretraining

A key objective of our work is to introduce minimal architectural modifications to effectively utilize pretrained image-language models. To analyze the effectiveness of image-language pretraining, we conduct experiments where we initialize the network weights randomly instead of using the CLIP weights. We evaluate the performance on three data scales and present the retrieval results on the MSRVTT dataset in Figure 5. Figure 5 demonstrates that initializing the model weights with CLIP weights significantly outperforms the case where we initialize the model weights randomly at all data scales. This finding further validates the efficacy of VidLA in effectively utilizing the pretrained image-language weights and highlights the challenge of training from scratch for video-language alignment, which

MSVD Retrieval R@1 R@5 R@10 Avg CLIP-ViT-B/32 CLIP4Clip [55] 62.0 87.3 92.6 80.6 DRL [87] 62.3 86.3 92.2 80.3 CAMoE* [13] 49.8 79.2 87.0 72.0 DiffusionRet [35] 61.9 88.3 92.9 81.0 CenterCLIP [103] 63.5 86.4 92.6 80.8 VidLA-B/32 66.5 92.4 95.6 84.8 VidLA-B/32* 77.4 96.0 98.7 90.7 CLIP-ViT-B/16 CenterCLIP [103] 68.4 90.1 95.0 84.5 DRL [87] 68.7 92.5 95.6 85.6 InternVideo(ViT-L)* 76.3 − − − UMT† 74.0 94.6 97.3 88.6 VidLA-B/16 68.8 93.4 96.2 86.1 VidLA-B/16* 80.3 97.2 98.4 92.0

Method

- Table 13. Video-to-text retrieval on MSVD dataset. * indicates inference with dual-softmax. † indicates two-stage method with candidate re-ranking.

Method

VATEX Retrieval R@1 R@5 R@10 Avg CLIP-ViT-B/32 DRL [87] 77.0 98.0 99.4 91.5 VidLA-B/32 79.8 99.5 99.8 93.0 VidLA-B/32* 85.0 99.8 99.9 94.9 CLIP-ViT-B/16 DRL [87] 80.1 98.5 99.5 92.7 InternVideo(ViT-L)* 87.2 − − − VidLA-B/16 81.1 99.5 99.9 93.5 VidLA-B/16* 85.6 99.8 100.0 95.1

- Table 14. Video-to-text retrieval on VATEX dataset. * indicates inference with dual-softmax. requires a large amount of training data.

50

40

CLIP Initialization

R@1

30

Random Initialization

20

10

1 10 80 # Training Samples in Millions

Figure 5. Effect of image-language pretraining on MSR-VTT.

### G. Analysis of [MST] Token Configuration

To investigate the impact of various design aspects of [MST] tokens, we systematically vary the number of hierarchies, U, and the temporal scale, r and evaluate the performance

on the MSR-VTT dataset. We conduct these experiments on the 80 million data scale, consistent with most of the analysis done in the main text. We report the results in Table 15.

The first and the second rows of Table 15 demonstrate that incorporating [MST] tokens for a single hierarchy, U = 1, leads to noticeable performance gains. The third and fourth rows further illustrate that increasing the number of hierarchies to 2 provides additional improvements, and setting the temporal scale to a higher value leads to better results. Finally, the last two rows indicate that adding one more hierarchy provides further improvement. However, we do not notice a significant difference between different temporal scales. Note that the [MST] token design in the last row is our default setting. The results in Table 15 provides further support for our hypothesis that incorporating hierarchy into the design of [MST] tokens plays a crucial role in improving video-language alignment performance.

|U-V -r<br><br>|MSR-VTT Retrieval R@1 R@5 R@10 Avg|
|---|---|
|✗<br><br>1-4-1<br><br>2-4-2<br><br>2-4-3 3-4-3 3-4-2<br><br><br>|49.1 75.3 83.5 69.3<br><br>51.3 76.5 85.0 70.9<br>52.1 76.6 84.2 71.0<br><br>52.3 76.9 85.7 71.6<br>53.2 77.0 85.6 71.9<br><br><br>53.5 77.5 85.6 72.2<br><br><br>|

- Table 15. Retrieval performance on MSR-VTT with different [MST] designs.

- H. Results on Hierarchical Dataset

We demonstrated VidLA’s efficacy on both short and long videos in Figure 4 (left). To further illustrate VidLA’s effectiveness in a strictly hierarchical setting, we constructed a temporally hierarchical test set with 1K short, mid and long clip triplets centered around the same timestamp. Zeroshot retrieval performance in Table 16 shows VidLA outperforms current state-of-the-art method, CLIPViP, even when trained on the same pretraining dataset (YT-VidLA-80M).

Method Short Medium Long CLIPViP-B/32 85.9 86.8 88.1 VidLA-B/32 86.8 90.5 92.5

Table 16. Text2Video retrieval R@1 on a 1k hierarchical test sets

- I. Analysis of Pretraining Dataset

To analyze the effect of different pretraining datasets, we pretrain VidLA-B/32 on diffrent pretraining datasets and later finetune on downstream datasets and report the results in Table 17. The first row reports the results where no pretaining on video data was performed and the model was only finetuned on the downstream dataset. From second

and third row we notice that training on YT-VidLA-10M outperforms training on WebVid-10M. The last two rows demonstrate the advantage of our large-scale training where the performance improves significantly with scale.

Dataset MSR-VTT DiDeMo ActivityNet WIT-400M 42.9 40.7 40.7 WebVid-10M 43.6 46.6 46.9 YT-VidLA-10M 49.3 46.3 49.7 YT-VidLA-80M 53.5 52.3 53.9 YT-VidLA-800M 55.6 56.9 61.3

Table 17. Text-to-Video retrieval R@1 with different pretraining datasets

