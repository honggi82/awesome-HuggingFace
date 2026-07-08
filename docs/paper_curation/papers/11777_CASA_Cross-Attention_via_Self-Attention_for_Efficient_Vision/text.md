## CASA: Cross-Attention over Self-Attention for Efficient Vision-Language Fusion

Moritz B¨ohle∗ Am´elie Royer∗ Juliette Marrie∗ Edouard Grave Patrick P´erez

moritz@kyutai.org amelie@kyutai.org juliette@kyutai.org egrave@kyutai.org patrick@kyutai.org

# arXiv:2512.19535v2[cs.CV]6Mar2026

### Abstract

Vision-language models (VLMs) are commonly trained by directly inserting image tokens from a pretrained vision encoder into the text stream of a language model. This allows text and image information to fully attend to one another within the model, but becomes rapidly costly for long multiimage conversations or streaming video applications, both in terms of memory and compute. VLMs leveraging crossattention (CA) are an efficient alternative to token insertion as image tokens are not added to the KV cache. Despite being introduced early on, multimodal CA models are scarce in the current VLM literature and often underperform their token insertion counterparts. In this work, we reinvestigate the effectiveness of cross-attention for visionlanguage modeling: (i) We analyze the core differences between the cross-attention and self-attention mechanisms, (ii) we train cross-attention VLMs both from a text-only LLM and by adapting a pretrained insertion-based VLM, showing that simple cross-attention is far more competitive with token insertion than previously reported, and (iii) we demonstrate the practical advantages of cross-attention on real-time video captioning, where it naturally maintains low latency and near-constant memory cost. For samples and code, please see our project page at kyutai.org/casa.

### 1. Introduction

Cross-attention (CA) has been employed early on as a lightweight mechanism for fusing multimodal information in transformers [1, 20, 39]. Most recent state-of-the-art VLMs have departed from cross-attention-based fusion, and instead insert visual embeddings into the language model’s input stream, directly interleaving them with the text embeddings [4, 8, 41, 52]. While insertion-based fusion is remarkably effective, it incurs high computational and memory costs which grow with the number of image tokens, becoming a bottleneck for high-resolution images, multiimage conversations, and streaming video applications.

∗ Equal contribution.

CA has recently regained interest as a naturally efficient alternative, particularly for streaming applications on long multimodal sequences [6, 23, 35, 51]. This has led to multiple architectural variants of cross-attention, in an effort to improve its performance, such as attention gating mechanisms [35, 51], learnable visual tokens [6], or updating visual embeddings across the depth of the network [23]. Nonetheless, CA-based VLMs currently lag behind insertion-based models on several tasks such as document and chart understanding, as shown in Figure 1. The causes of this performance gap remain poorly understood. In particular, it is unclear whether it reflects fundamental limitations of CA or instead arises from differences in training data and implementation choices.

Through this work, we methodically revisit CA as a fusion mechanism. To shed light on why current SotA CA models lag behind recent VLMs, we first analyze the efficiency trade-offs of token insertion and cross-attention showing that CA offers significant memory and speed advantages at both training and inference time (Tab. 1). We validate these findings in two controlled settings: training a CA-based VLM from scratch from a text-only LLM, and adapting a pretrained insertion-based VLM to crossattention. We find that vanilla cross-attention, without any bells and whistles, performs better than previously reported, narrowing the gap to token insertion to only a few percents on most benchmarks when compared in an identical training setting. This simple model also outperforms current stateof-the-art CA-based VLMs of larger size, showing the importance of using a comparable modern training pipeline when assessing the performance of CA models with respect to the literature. We further evaluate our CA model on the costly task of real-time video captioning, where CA maintains near-constant memory and latency over long video horizons. In summary, we provide a controlled and comprehensive comparison of cross-attention and token insertion for vision-language fusion. Our contributions are threefold: (i) We systematically analyze the core differences between the two mechanisms, identifying five key core design elements that progressively bridge cross-attention and token insertion (Section 3.2). (ii) We train cross-attention VLMs

[Figure 1]

- Figure 1. State-of-the-art cross-attention (CA) VLMs. We report benchmark performance normalized by Qwen2.5-VL 3B scores (hatched grey). We adapt Qwen2.5-VL 3B, a token-insertion-based VLM, to use cross-attention layers. The resulting model (red) retains most of the base model’s performance, and outperforms prior CA-based VLMs across model scales (2B–14B), in particular on highresolution document and chart understanding tasks (ChartQA, DocVQA).

both from a text-only LLM and by adapting a pretrained insertion-based VLM, showing that simple cross-attention is far more competitive with token insertion than previously reported. (iii) We demonstrate the practical advantages of cross-attention on real-time video captioning [5], where CA enables continuous visual updates with low latency and near-constant memory cost, while token insertion models quickly exhaust their memory budget. To foster reproducibility, we release our inference code and trained models.

### 2. Related work

Insertion-based fusion. Token insertion has become the dominant paradigm for training VLMs. For this, visual embeddings from a pretrained encoder are inserted directly into the language model’s input sequence, interleaving them with text embeddings. The visual and textual information thus interact through self-attention layers without any architectural changes. Recent token insertion models [4, 8, 26, 41, 53] have achieved strong multimodal performance with this strategy. However, the number of visual tokens rapidly grows with image resolution or video length, thus increasing the memory and computational costs at both training and inference time. A wide range of techniques has been explored to mitigate this cost, including compressing visual tokens via pixel unshuffling [36], token merging [22], query-based compression [20, 21, 53], or pooling [25, 42], inserting visual tokens in only a subset of layers [7], reducing the cost of attention operations [49], compressing the KV cache at inference [5, 34, 54], or modifying positional embeddings to handle longer contexts [4, 14]. These techniques are largely orthogonal to the choice of fusion mechanism, and could also complement CA to further reduce the number of keys and values in the CA layers.

Cross-attention-based fusion for VLMs was popularized by Flamingo [1], one of the first large-scale VLMs, in which a frozen LLM is conditioned on visual inputs through gated cross-attention. It was adopted in subse-

quent works [3, 19], and revisited in [6, 23, 51], which leverage the natural scalability of cross-attention for longcontext or streaming applications. However, current SotA cross-attention VLMs underperform token-insertion models, in particular on tasks requiring fine-grained visual understanding such as chart or document reading [23, 51]. Prior work has attempted to address this gap by, e.g., updating the visual representations throughout the depth of the model [23], adding register tokens [6], or introducing bespoke architectural changes [51]. In this work, we show that the performance gap is much lesser than expected when comparing token-insertion to cross-attention fusion in identical pipelines, even when using vanilla cross-attention.

Streaming visual understanding. Recently, the VLM literature has seen a gain of interest for streaming video understanding tasks, such as live video captioning or multimodal assistants [5, 47, 54]. Such tasks imply a strict memory and computational budget to execute in real-time and for as long as possible. To address the limitations of token insertion, it is possible to limit the KV cache size through compression [54], pruning of the oldest video frames [5, 34], or by adding condensed “visual memories” to the text stream [47, 57]. In contrast, cross-attention naturally lends itself to efficiently handle a dense stream of images as input. Closest to our work, StreamChat [23] adopts a cross-attention-based design that updates its visual keys and values at each decoding step to align the current text stream with the latest video frame. However, to improve the performance of its cross-attention mechanism, StreamChat applies additional dedicated FFN layers to the visual tokens throughout the network. In our ablations, we show that the additional updates of image embeddings indeed yield a small performance boost, but incur high memory and compute cost, in particular during training.

### 3. To Cross-Attend or Self-Attend?

Cross-attention’s practical benefits stem from the core idea that image tokens never directly enter the transformer’s in-

put token stream. Consequently, they are neither added to the KV cache, nor updated via the Feed Forward Network (FFN) layers, saving both compute and memory. However, these strengths may become a limitation when considering the model’s performance: Unlike token insertion, image embeddings are not updated throughout the depth of the network, and text tokens cannot directly attend to past images as they are not present in the KV cache. To better understand these distinctions and how they impact the performance-efficiency trade-off, we first formalize the CA and self-attention (SA) mechanisms (Section 3.1) and analyze their key differences (Section 3.2). Finally, we discuss how to implement CA for scalable multi-image training, allowing us to leverage sequence packing strategies and video inputs (Section 3.3).

#### 3.1. Preliminaries

Self-attention in token insertion models. Given text tokens x = x1...T and a tokenized image y = y1...N inserted at position K < T in the text stream, token insertion lets xT interact with image tokens through self-attention in a standard causal setting as:

SAins(xT|xi≤T,y) = MHA(xT|x1...K y1...N xK+1...T).

(1) MHA(q|k) is multi-head attention [40] with query q and k

- as keys and values.

Cross-attention. In contrast, CA injects visual information to text tokens as additive updates. In CA layers, text tokens xi>K which follow the image at timestep K attend to the corresponding image tokens as keys and values:

CA(xT|y) = 1T>K × MHA(xT|y1...N). (2)

In the remainder of the text, we will refer to the interval [K,T] as the “window” corresponding to the image y seen

- at time K. Simply speaking, each window is defined as the temporal positions between two input images. Note that

in this setting, text tokens xi>K do not attend to past images seen before y by design, and we will further elaborate on this locality property in Section 3.2. The output of (2) is then combined, typically via a sum, with the output of the corresponding SA layer in the same transformer block, which operates on text tokens only:

SAtext(xT|xi≤T) = MHA(xT|x1...T). (3) In practice, we only consider the scenario where crossattention layers are placed in parallel of self-attention layers in the same transformer block, as illustrated in Figure 2e: Both CA and SAtext operate as separate layers, with their own projections, but on the same inputs. Nevertheless, other designs have been explored in the literature, such as placing

- them after [23]. The attention patterns of Eq. (1), Eq. (2), and Eq. (3) are summarized in Figure 2.

#### 3.2. From Cross-Attention to Token Insertion

We identify five core differences between cross-attention and token insertion as fusion mechanisms, each with a different impact on model efficiency and performance. We discuss each of them below, emphasizing how, by combining these core aspects, we can progressively recover selfattention from cross-attention.

- (D1) Additional parameters. CA introduces new dedicated layers with additional learnable parameters. In contrast, SA uses the same projection weights to process image and text tokens without distinction. To eliminate this difference, we introduce a parameter-sharing variant of CA, denoted by CA , in which the query, key, value and output projections are shared between the CA and SA layers of a same transformer block. Like token insertion, this variant thus adds no new parameters. Further, since CA and SAtext operate in parallel on the same inputs, we only need to compute said projections once, thereby saving compute.
- (D2) Joint text-image attention and positional embeddings. Inserting images into the token stream allows text tokens to attend to both image and other text tokens in the same MHA operation. In addition, both text and image tokens receive temporal positional embeddings (e.g., RoPE). In contrast, in CA layers text tokens only attend to image tokens, and there is traditionally no information on the temporal position of image tokens relative to the text ones. To bridge this gap, we introduce CAt+v, where text (“t”) tokens attend to the last seen visual (“v”) tokens as well as to preceding text tokens in the same window:

##### CAt+v(xT | xK<i≤T,y) = MHA(xT | y1...N xK+1...T).

(4)

In other words CAt+v can be thought as the SA operation of (1) operating inside local windows, rather than across the whole image-text interleaved sequence.

(D3) Additional layers. Cross-attention operates as an additive residual update to the self-attention output within each transformer block, effectively doubling the number of attention layers. A natural way to reduce this overhead is to instead replace self-attention layers with cross-attention layers in a subset of transformer blocks. We denote this variant by CA . In our experiments, we replace every second SA layer with a CA layer, yielding efficiency gains with only a minor performance drop compared to CA. Note that not every layer can be replaced by CA, as no text-to-text attention would remain. In contrast, CAt+v layers also computes text-to-text attention, albeit in local windows, and thus constitutes an interesting fused hybrid between CA and SA. (D4) Image token updates. With token insertion, the image embeddings are updated through the network in the same way as text tokens: In every transformer block, they pass through a feed forward network (D4.1) and attend to

[Figure 2]

[Figure 3]

|CA Layer + Normalization<br><br>MLP Layer + Normalization<br><br>SA Layer + Normalization<br><br>⊕<br><br>| | |
|---|---|
| | |
<br><br>× L|
|---|

Tokenize + Embed

What is this? himgi A yellow house!

(a) Insertion (b) Cross-attention

[Figure 4]

[Figure 5]

Image Encoder

[Figure 6]

(e) Cross-attention model

(c) Text-only SA (d) CAt+v

- Figure 2. Different attention patterns: (a) full causal self-attention (SA) on image and text, used in standard insertion-based models, (b) block-wise cross-attention (CA), where each text query attends to all image tokens in its window, (c) text-only causal SA, used alongside CA to allow for text-to-text interaction, and (d) CAt+v, where each text token attends to all preceding text and image tokens in its window (cf. Sec. 3.1). In (e) we show the general architecture of the CA VLMs we investigate in this work.

Add new params

Update image tokens

Training Streaming Inference

Helium1-2B backbone

tokstxt/s Mem. FPS Mem. #KV CA ✓ ✗ 1817 32.9 6.8 6.1 kT + N CA CA + (D1) ✗ ✗ 1845 32.0 7.7 5.7 kT + N CAt+v CA + (D2) ✓ ✗ 1560 33.1 5.6 6.1 kT + (T +N) CA CA + (D3) ✗ ✗ 1995 29.0 7.6 5.7 kT or N CA+FFNs CA + (D4.1) ✓ ✓ 1504 57.8 6.6 6.1 kT + kN SAins CA + (D1-5) ✗ ✓ 1501 62.8 1.2 29.5 kT + kN

- Table 1. Efficiency of cross-attention vs. token insertion. We compare cross-attention (CA) and variants with (D1): shared parameters between CA and SA, (D2): text-to-text attention in CA layers, (D3): replacing SA layers with CA, and (D4.1): image updates through FFNs, and token insertion (SAins). In a single-image setting, SAins is recovered by adding (D1–5) components to CA. In all settings, we report whether new parameters are required, whether images embeddings evolve throughout the network, as well as memory and throughput measurements for inference and training. Measurements are performed on a single GPU. For inference, we imitate the streaming video captioning setup of Section 4.5 in a controlled setting. Specifically, we report for how many frames per second (FPS) the resulting model would be able to predict 5 text tokens each, while incorporating the new image information in a streaming fashion.

preceding tokens via self-attention as in Eq. (1) (D4.2). In contrast, in cross-attention models image embeddings do not receive any persistent updates, and only undergo different KV projections in every cross-attention layer. This lack of iterative updates may limit the model’s ability to further refine image representations and has motivated prior work to enhance CA-based fusion with image updates through dedicated FFNs [23]. Similarly, we denote by CA+FFNs a variant of CA where image embeddings are updated through the FFNs of the network. However, as shown in Table 1, this incurs significant memory costs at training and may require sacrifices to fit the memory constraints (e.g. using shorter training sequence lengths or lower image resolution).

(D5) Multi-images History. As mentioned in Section 3.1,

CA operates in local windows such that a text token may only attend to the last seen image. In contrast, text tokens in insertion-based models can attend to the whole history, including all past images. This is a key trade-off of CA: Not retaining past images in the KV cache may be detrimental to the model when dealing with multiple images (e.g. videos), but also enables lower memory usage at inference, as we will demonstrate in Section 4.5. While prior work has considered integrating multiple past image tokens as keysvalues in CA layers [51], we do not study this setting in this work, to retain the focus on efficient VLMs. Nevertheless, this lack of explicit visual history is not necessarily a dealbreaker: Recent work on context compression [30, 44, 47] suggests an elegant approach to tackle the lack of explicit image history in CA by adding special “gist tokens” in the

text stream immediately after every image. In CA layers, gist tokens only attend to the image tokens, while in SAtext they naturally interact with other text tokens in a standard causal manner. Simply put, gist tokens do not have any initial textual semantic meaning and are placed in the text stream as a way to retain a compressed representation of the image they directly follow. We implicitly rely on this mechanism in all our video experiments: Each window only contains the latest video frame, and a certain number of gist tokens is placed after each frame. In practice, we directly use the post-image delimiter, often present in chat templates of recent VLMs such as Qwen-VL, to serve this purpose. Thus, when generating an answer in a classical video QA setting, the model only has access to the last frame of the video, and all past gist tokens of previous windows. Interestingly, we find that even with this compressed visual history, the CA-based VLMs achieve surprisingly strong performance on video QA tasks.

Take-aways. The above discussion reveals that the transition from CA to token insertion can be decomposed into five key design choices (D1-D5). Notably, D1-D3 can be addressed with simple, lightweight modifications to the crossattention mechanism that can even reduce computational overhead. In contrast, D4 (updating image tokens through FFNs) incurs significant memory and compute costs during training, and avoiding D5 (retaining all past images in the KV cache) is a critical design choice for streaming applications, where the accumulation of image tokens introduces substantial memory and latency overhead. This is reflected in the training and inference costs reported in Table 1.

#### 3.3. Scalable Cross-attention Training

We train our models on a mixture of recent public visionlanguage datasets, primarily composed of question-answer pairs over single images. Due to the high disparity in sequence lengths between samples, we employ multimodal sequence packing, as commonly done in LLM training [11, 48] and modern VLM pipelines [4, 52].

To match the desired cross-attention inference behavior, where text tokens only attend to the image of their corresponding window, we employ the block-wise attention implementation of FlashAttention-2 [9] in the cross-attention layers during training. As illustrated in Figure 2, we define the attention blocks with the image insertion points acting as natural window delimiters. Importantly for CAt+v, where text tokens appear as both queries and key-values, placing the image anywhere other than at the beginning of a window would break causality between the text tokens during training as the attention mask is bottom-right aligned in the implementation of FlashAttention-2.

### 4. Experiments

#### 4.1. Experimental Setting

Below we provide a brief summary of our experimental setup. Please refer to Appendix A.1 for full details.

Training Data. We train our models on FineVision [43] and a subset of LLaVA-OneVision-1.5 [2]. Both are curated collections of publicly available image-text datasets covering a wide range of tasks such as captioning, document and chart reading, general VQA, etc. For video training, we further train our models with the aforementioned image-text data alongside LLaVA-Video-178K [58].

Backbones. We investigate both extending language-only models with visual understanding as well as adapting existing VLMs with CA. Specifically, we train our models in two settings: (i) Starting from Helium1-2B [17], a text-only LLM, we jointly finetune the backbone and CA layers; (ii) We adapt a frozen Qwen2.5-VL-3B [4] VLM by only learning the additional CA layers. In both scenarios, we use the vision encoder of Qwen2.5-VL [4]. We finetune its last 4 layers when training on image, and freeze it when further finetuning on videos. Finally, we initialize CA layers from the self-attention layers of the respective backbone.

Benchmarks. We evaluate our models on common VLM benchmarks on a variety of tasks: document (DocVQA [28]) and chart (ChartQA [27], InfoVQA [29]) understanding, text recognition (TextVQA [37], OCRBench [24]), and general QA (RealWorldQA [45], AI2D [16], GQA [15], MME [12]). Similarly, we evaluate video models on video understanding (MVBench [21], VideoMME [13], PerceptionTest [33]), temporal action reasoning (NExT-QA [46]), and long video understanding (MLVU [60]).

Training Compute. As detailed in Sec. 3.3, we use multimodal sequence packing with block-wise attention during image training, limiting the length of the resulting interleaved text-image sequences to 2048 text tokens and 20,480 image tokens per GPU. We train our models on 64 H100 GPUs with a batch size of 64 and 2 gradient accumulation steps. For token-insertion experiments, we halve the sequence lengths and double the batch size to fit the increased memory cost while processing the same number of tokens as CA models. We process images at native resolution up to 9522 pixels (1156 tokens per image) and downscale larger images, keeping their aspect ratio, to this target resolution. For videos, we use Rmax=5042 and extract 2 frames per second, with a maximum clip duration of 3 minutes, resulting in up to 46,080 image tokens per sequence. In terms of training horizon, we train for 40k training steps for Helium1 experiments, 25k for Qwen2.5-VL adaptation on images, and a further 15k for videos.

#### 4.2. From LLM to VLM with Cross-Attention

We first train the text-only Helium1-2B with additional CA layers. In Table 2 we compare a Helium1-based VLM trained with token insertion against different CA variants, all trained under the same conditions. For reference, we also report performance of proprietary VLMs (InternVL2.5 [8], Qwen2-VL [41], and Video-LLaMA3 [52]), an open-source VLM trained on public data (SmolVLM [26]) and recent CA-based VLMs (mPLUG-Owl3 [51], StreamChat [23]).

As shown in Table 2, the performance of the vanilla cross-attention model, as well as its variants, is very close to the token insertion model trained in the same setting, with an average 1.5 percent drop of performance. A significant gap only remains on ChartQA and InfographicVQA, both dealing with understanding complex graphics and figures. Compared to the literature, our CA models outperform the cross-attention-based mPLUG-Owl3 on most benchmarks, even surpassing its 7B variant, highlighting the importance of an up-to-date training pipeline to fairly compare CA models with token insertion. When comparing different variants of cross-attention, we first note that adding text tokens to the key-values of cross-attention (CAt+v) performs similarly to the vanilla CA on average, with a slight boost on general VQA benchmarks. Second, we observe that sharing CA and SA parameters (CA ) is detrimental to the model performance, but only by a small margin. This introduces an interesting efficiency-performance trade-off, as CA is more practical than CA, according to the results of Table 1. Finally, replacing every other self-attention layer with CA layers (CA ) provides another solid option to trim off compute while incurring a small drop in performance. The robustness of CA across these variants is encouraging from a practical standpoint: the most efficient ones (CA , CA ) process over 6× more frames per second and use over 5× less memory at inference than token insertion (Tab. 1), at only a minor performance cost. We further showcase these advantages on the task of live video captioning in Sec. 4.5.

#### 4.3. Adapting an Insertion VLM to Cross-Attention

The previous results show that CA is competitive with token insertion when both are trained from scratch. In practice, however, adapting strong pretrained VLMs is more efficient. We therefore investigate whether a pretrained insertion-based model can be efficiently converted to crossattention. Specifically, we adapt Qwen2.5-VL-3B [4] by replacing its token-insertion mechanism with CA layers. We train the added CA layers alongside the four last blocks of the visual encoder and keep all other parameters frozen.

Image evaluation. We report results in Table 3, directly comparing to the base model Qwen2.5-VL. As in Section 4.2, changing to a CA-based design incurs a moderate performance drop (6.8% on average). Consistent with the trends observed in the previous section, the most significant

gap occurs on InfographicVQA. Importantly, these results are obtained by training only the CA layers and the last image encoder blocks for 25k steps, while keeping all other parameters frozen. Despite this lightweight adaptation, the resulting CA model retains the vast majority of the base model’s capabilities while gaining the practical efficiency advantages of cross-attention at inference time.

Video evaluation. We further finetune our Qwen-CA model on videos [58] and report results in Table 4. As discussed in Section 3.2 (D5), we implement cross-attention with windows, such that each text token only attends to the most recent video frame. Thus, to preserve information from past images in the text stream, we rely on the postimage delimiter tokens, already inserted by Qwen-VL’s chat template after each frame, to act as gist tokens. In particular, this differs from past CA works, which generally insert multiple past frames as key-value inputs in cross-attention layers, thus increasing the cost of the CA operation. Despite this limited visual history, our CA model lands only 3.9 percent below the base model on average, suggesting that it effectively stores relevant visual information in the gist tokens. For reference, we also train a model in the standard evaluation setting, where all image frames are inserted in a window (second-to-last row). This leads to a performance boost (but a higher compute cost) and performs on-par or better than prior CA works of larger size.

#### 4.4. Ablation Experiments

For ease of reading, we group the 9 benchmarks into 3 categories in our ablation results: HRES, high-res. chart and document reading (DocVQA, InfoVQA, ChartQA), OCR, reading text in natural images (OCRBench, TextVQA), and VQA, general-knowledge visual understanding (RealWorldQA, AI2D, GQA, MME).

Cross-attention layer frequency. In Table 5 (left), we report results for training CA from the text-only Helium12B model [17] where CA layers replace SA every n layers, with n = 2 and 4. Note that the CA results of Table 2 use n = 2 by default. Reducing the number of CA layers (here, replacing SA every 4 layers instead of every 2) further shifts the efficiency–performance trade-off, providing a simple way to adapt to different compute budgets.

Updating image tokens. Recent CA-based VLMs [23] update the image embeddings by applying dedicated FFN layers to improve performance. We evaluate the benefits of this approach through a small-scale ablation with the image encoder frozen, as propagating the image tokens through FFNs substantially increases memory usage (cf. Table 1). As shown in Table 6, updating image tokens yields improves performance (≈2 percents on the average performance), in line with prior work [23], but comes with significantly higher memory usage during training (Table 1).

# train High-res Doc/Chart Scene Text Knowledge / General QA tokens CHART DOC INFO OCRB TEXT REALW AI2D GQA MME

Token Insertion – Proprietary

InternVL2.5 [8] 0.5T 79.2 88.7 60.9 804 74.3 60.1 74.9 59.5 2005 Qwen2-VL [41] 1.4T 73.5 90.1 65.5 767 79.7 62.9 69.9 59.8 1872 VideoLLaMA3 [52] 79.8 91.9 69.4 779 80.1† 67.3 78.2 62.7 1901

Token Insertion – Public data SmolVLM [26] 68.7 80.0 42.2† 729 73.0 51.0† 59.7† 49.2† 1568† InsertionHe-2B (ours) 0.13T 80.0 87.2 55.3 745 74.4 59.6 64.6 54.0 1676 Cross-attention SotA – Public data

mPLUG-Owl3 8B [51] 0.1T 59.2† 55.9† 36.8† 527† 69.0 63.9† 73.4 65.0 1940† StreamChat 7B [23] ⋄ ⋄ ⋄ ⋄ 72.4 61.7 76.6 62.4 1520 mPLUG-Owl3 2B 0.1T 48.5† 48.2† 28.1† 450† 62.6 56.9† 62.6 61.0 1551†

Cross-attention (ours) – Public data

CAHe-2B 0.13T 75.9 85.8 51.8 722 73.8 58.0 66.2 54.3 1731 CA He-2B 0.13T 74.1 84.0 50.6 696 73.4 56.5 64.2 51.7 1613 CA He-2B 0.13T 73.8 84.5 48.0 714 73.0 56.9 64.3 54.0 1607 CAtv,He-2B 0.13T 75.3 85.1 52.1 722 74.1 60.3 65.5 54.5 1720

† : Reproduced with the publicly available models at huggingface. ⋄: Results not available.

- Table 2. Comparing cross-attention and insertion-based models. Unless specified, all models are built on 2B LLMs. We use lmms-eval [55] for evaluation, and re-evaluate existing models when benchmark results are not provided in the original work. When trained under the same conditions, our CA model reaches performance close to that of a token insertion model, showing the potential of cross-attention as an efficient alternative. Nonetheless, a significant gap remains on infographic and chart understanding tasks, highlighting the benefits of token insertion for certain tasks.

Document/Chart Scene Text Knowledge / General QA CHART DOC INFO OCRB TEXT REALW AI2D GQA MME Qwen2.5-VL 3B

84.0 93.0 77.1 797 79.3 65.4 81.6 ⋄ 2157 83.1† 92.4† 75.1† 796† 79.6† 60.4† 79.6† 61.0† 2224†

CAQ2.5-VL 80.3 87.0 57.4 783 76.3 61.3 74.0 59.0 1910 CAtv,Q2.5-VL 81.3 87.1 56.8 775 76.3 63.8 74.4 58.6 1971

† : Reproduced with the publicly available model at huggingface.

- Table 3. Adapting a frozen Qwen2.5-VL by training additional CA layers (and the last 4 block of the image encoder) retains similar performance as the base model on most benchmarks. As for the experiment of Table 2, a large gap remains on the InfographicVQA benchmark, suggesting some tasks may benefitting from token insertion fusion mechanism than others.

Link to token compression. To reduce token insertion costs, it is common to compress the number of image tokens before inserting them in the text stream [20, 22, 25, 42, 53]. In Table 5 (right), we report results of training Helium1-2B with either full token insertion or a Q-Former-based compression [20], in which a small transformer block is applied to the N image tokens produced by the vision encoder, alongside Q ≪ N learnable queries; only the Q queries are

- then inserted in the LLM’s textual stream. For general VQA tasks, we find that even aggressive token compression has limited impact on the performance, but for tasks requiring

more detailed representations (HRES, OCR) we observe significant performance drops. Hence, compressed insertion can be a practical alternative to full token insertion if the task does not involve fine-grained visual detail. Nonetheless, as we discuss in Appendix C, even with token compression the memory cost and context length limit quickly become a bottleneck when dealing with long streaming video understanding. Finally, note that token compression is orthogonal and can be combined with CA to further reduce the number of tokens in the CA layers.

MAX WINDOWS

VIDEOMME NEXT PERCEP. MV

MLVU #FRAMES w/o sub w/ sub QA TEST BENCH

768

61.5 67.6 ⋄ 66.9 67.0 68.2 360 58.8† 63.6† 78.9† 67.0† 66.2† 65.2†

Qwen2.5VL 3B

✗

StreamChat 7B 40 ✗ 58.6 62.8 78.5 ⋄ 53.3 63.9 mPlug-Owl3 7B 128 ✗ 53.5 ⋄ 78.6 ⋄ 54.5 ⋄ mPlug-Owl3 7B† 180 ✗ 59.8† 65.2† 82.1† 65.6† 58.3† 65.6†

CAtv,Q2.5-VL 3B 180 ✗ 59.0 64.8 79.8 63.8 60.6 67.9 CAtv,Q2.5-VL 3B 180 ✓ 56.9 60.5 78.8 62.6 59.5 66.2

† : Reproduced with the publicly available model at huggingface.

- Table 4. Video benchmarks. We further finetune our Qwen2.5-VL model with CA layers on the LlavaVid dataset [2] for 15k training steps. Results are reported for CAt+v in a setting where each window contains a single frame (✓), using gist tokens to preserve past visual history,

- as described in Section 3.2. For fair comparison, we also report results in the more standard, yet less efficient, evaluation setting where all frames are placed in a single window (✗), as done in prior CA work. In both scenarios, adapting Qwen2.5-VL to use cross-attention layers only incur a small performance drop and performs on par with prior CA models, even those of larger size.

(a) CA t+v layer frequency (b) Impact of token compression CA

CA (every n layers) INSERTION QFORMER QFORMER

TASK n = 2 n = 4 (1156 tokens) (128) (32) HRES 70.8 68.4 67.8 74.2 62.7 56.0 OCR 73.1 72.0 71.1 74.5 70.8 67.0 VQA 60.4 58.8 57.0 60.6 57.8 56.9 AVG. 68.1 66.4 65.3 69.8 63.8 59.9

- Table 5. Ablation experiments on (a) CA and (b) token compression. (a) Effect of inserting CA at different layer intervals, in the same training setting as Table 2. (b) Impact of token compression, a common approach to improve the efficiency of token insertion. Reducing the number of image tokens quickly deteriorates performance on tasks involving high resolution images. Furthermore, as we show in Section 4.5, token compression is not sufficient to solve the memory bottleneck inherent to image token insertion, as the KV cache still grows during streaming inference.

MODEL \ BENCHMARK HRES OCR VQA AVG. CAt+v (He-2B) 55.7 64.1 44.1 54.7

+ FFN UPDATES 58.1 66.6 43.2 56.0

- Table 6. Updating image embeddings through the underlying LLM’s FFNs provides a modest increase in performance, but incurs non-negligible memory and compute costs. We also freeze the image encoder to accommodate the increased memory usage.

CA to process long text-image sequences, we now consider the task of live video captioning.

Specifically, we finetune our CAt+v-adapted Qwen2.5VL model trained on images only on Live-WhisperX-526K, a recent instruction-tuning dataset [5]. It is composed of video frames sampled at 2fps and interleaved with the text transcript of the video’s original audio. As discussed in Section 3.2, we design our CA layers such that text tokens only attend to the latest image in the current window. At inference, this means we simply continuously execute the model while replacing the key-value sources of the CA layers every half a second. We provide qualitative samples of live captioning results in Figure 3 and Appendix C.

Quantitative evaluation. We evaluate our LiveCC-tuned model on the LiveSports3K benchmark proposed in [5]. The dataset consists of videos of sports events (∼ 20 seconds long). The captions are evaluated using an LLM as a judge, following the evaluation protocol provided in LiveCC’s repository with GPT-4o acting as the judge. In Table 4(a), we report results for a CAt+v Qwen2.5-VL3B model trained on LiveCC and evaluated across training steps, and compare to the results reported in the original LiveCC paper. Notably, despite its smaller size (3B vs. 7B), our CA model obtains scores similar to those of LiveCC.

#### 4.5. Application to Live Video Captioning

Streaming video understanding is an important motivation for building efficient vision-language models. This tasks comes with two particular challenges: (i) Memory costs needs to be kept under control in order to handle longer videos, and (ii) for streaming applications, the model latency needs to stay below the frame rate to avoid accumulating delay over time. By design, CA is well equipped to address both of these issues. To illustrate the usefulness of

Real-world performance. As can be seen in Figure 4(b), the memory cost of token insertion methods increases more rapidly than for the CA-based model. While token compression reduces the cost of token insertion for short conversations, it cannot alone prevent the increased memory usage leading to OOM when the number of tokens is too high. In Figure 4, we also report the wall-time of the same models (recorded on a single H100 GPU) as a function of the number of frames inserted: Generation with insertion-based models becomes progressively slower as image tokens ac-

[Figure 7]

00:00 00:02 00:04 00:06 00:08 00:10 00:12 00:14 00:16 00:18

This video shows [00:01|0.0s] the Apollo 13 [00:02|0.0s] spacecraft during its [00:03|0.0s] 40 years of space camp missions at [00:05|0.0s] NASA’s Johnson Space Center [00:06|0.0s] in Houston Texas [00:07|0.0s] and it was filmed by NASA [00:08|0.0s] Space Camp Director [00:09|0.0s] Steve Smith who has been with us for over [00:10|0.0s] 25 years now [00:11|0.0s] at this point he had to go back home [00:12|0.0s] to his job as a [00:13|0.0s] director but we’re going [00:15|0.0s] with him here today [00:16|0.0s] so you can see how they’ve changed [00:17|0.0s] over time from [00:18|0.0s] Apollo era [00:19|0.0s] missions all right [00:20|0.0s] we’ll start

- Figure 3. Live captioning. We display captions generated by our CAtv,Q2.5-VL model. Each text span is annotated with the corresponding frame’s timestamp (top) and the model’s delay as [timestamp|delay]. As shown in Figure 4b, the model’s outputs are generated much faster than real-time with no noticeable increase over time and near-constant memory usage, as only new text tokens are added to the model’s KV cache; for further qualitative examples, including insertion-based VLMs, see Appendix C.

Model Training steps Winrate (%) CAt+v (Qwen2.5-VL) (Ours, 3B)

3k 25.4 6k 27.6 8k 33.7

12k 34.3 15k 36.3 17k 39.4 20k 39.0

Baselines as reported in LiveCC, 7B

LiveCC-7B-Base 43.2 LiveCC-7B-Instruct 41.5 Qwen2-VL-7B-LiveCC 33.7

(a) LLM-as-judge evaluation on LiveSports video captioning (b) Walltimes in the streaming video captioning setting for CA and token-insertion

[Figure 8]

- Figure 4. Quantitative results on LiveCC. (a) We evaluate our CAtv,Q2.5-VL model on the LiveSports captioning tasks proposed in [5], following their LLM-as-a-judge methodology. (b) We record the walltime as a function of the number of frames inserted in a streaming captioning scenario, for CA and token-compression techniques (Q-Former with different numbers of query tokens). While token compression mitigates the computational cost of token insertion for short videos, the CA model maintains low latency inference for longer times. In addition, token insertion tends to go out-of-memory quickly especially for larger numbers of tokens. Note that for better readability, we only plot a subset of markers, although the plotted measurements occur at every frame.

cumulate in the KV cache. In contrast, the CA-based model maintains high inference speed over much longer horizons. Together, these results show that the efficiency properties of CA translate into tangible advantages for real-world streaming, enabling low-latency captioning over extended time horizons where insertion-based models are impractical.

- 5. Conclusions

performs close to token insertion on most image and video benchmarks, even without employing architectural modifications proposed in prior work. As a result, our experiments on adapting a pretrained SotA insertion-based VLM to cross-attention yield a strong CA-based model, outperforming prior CA-based VLMs of larger size (Fig. 1). This adapted model naturally lends itself to streaming applications: by replacing the key-value sources of the CA layers at each new frame, it performs live video captioning with near-constant memory and latency, while token insertion models quickly exhaust their memory budget. Our results suggest that cross-attention deserves renewed consideration as a practical and competitive alternative for visionlanguage fusion, especially as applications move toward longer, streaming multimodal inputs.

We revisit CA as a fusion mechanism for VLMs, a design that has been largely set aside in favor of token insertion despite CA’s favorable efficiency properties. We identify five core design differences (D1-D5) that progressively bridge CA and token insertion, and analyze their respective impact on efficiency and performance. Training CA-based VLMs both from a text-only LLM and by adapting a pretrained insertion-based model, we find that simple cross-attention

Acknowledgements. This project is funded by Iliad Group, CMA CGM Group and Schmidt Sciences. The authors thank Alexandre D´efossez for his support and feedback throughout the project.

### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a Visual Language Model for Few-Shot Learning. NeurIPS,

2022. 1, 2

- [2] Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv:2509.23661, 2025. 5, 8, 13
- [3] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. OpenFlamingo: An Open-Source Framework for Training Large Autoregressive Vision-Language Models. arXiv:2308.01390, 2023. 2
- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL Technical Report. arXiv:2502.13923, 2025. 1, 2, 5, 6
- [5] Joya Chen, Ziyun Zeng, Yiqi Lin, Wei Li, Zejun Ma, and Mike Zheng Shou. LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale. In CVPR, 2025. 2, 8, 9, 13, 14
- [6] Kaibing Chen, Dong Shen, Hanwen Zhong, Huasong Zhong, Kui Xia, Di Xu, Wei Yuan, Yifei Hu, Bin Wen, Tianke Zhang, et al. EVLM: An Efficient Vision-Language Model for Visual Understanding. arXiv:2407.14177, 2024. 1, 2
- [7] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models. In ECCV, 2024. 2
- [8] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling. arXiv:2412.05271, 2024. 1, 2, 6, 7
- [9] Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In ICLR, 2024. 5
- [10] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and PixMo: Open Weights and Open Data for State-of-theArt Vision-Language Models. In CVPR, 2025. 13
- [11] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. 2407.21783, 2024. 5
- [12] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang,

- Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. ArXiv, abs/2306.13394, 2023. 5
- [13] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-MME: The FirstEver Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis. In CVPR, 2025. 5
- [14] Junqi Ge, Ziyi Chen, Jintao Lin, Jinguo Zhu, Xihui Liu, Jifeng Dai, and Xizhou Zhu. V2pe: Improving multimodal long-context capability of vision-language models with variable visual position encoding. ArXiv, abs/2412.09616, 2024. 2
- [15] Drew A Hudson and Christopher D Manning. GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering. In CVPR, 2019. 5, 13
- [16] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A Diagram Is Worth A Dozen Images. In ECCV, 2016. 5
- [17] Kyutai. Helium1: A modular and multilingual LLM, 2025. 5, 6
- [18] Hugo Lauren¸con, Andr´es Marafioti, Victor Sanh, and L´eo Tronchon. Building and better understanding vision-language models: insights and future directions. arXiv:2408.12637, 2024. 13
- [19] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Joshua Adrian Cahyono, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Otter: A Multi-Modal Model with In-Context Instruction Tuning. IEEE TPAMI, 2025. 2
- [20] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. In International Conference on Machine Learning (ICML), 2023. 1, 2, 7, 14
- [21] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. MVBench: A Comprehensive Multi-modal Video Understanding Benchmark. In CVPR, 2024. 2, 5
- [22] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, Yu Qiao, Yali Wang, and Limin Wang. VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling. arXiv:2501.00574, 2024. 2, 7
- [23] Jihao Liu, Zhiding Yu, Shiyi Lan, Shihao Wang, Rongyao Fang, Jan Kautz, Hongsheng Li, and Jose M Alvare. StreamChat: Chatting with Streaming Video. arXiv:2412.08646,

2024. 1, 2, 3, 4, 6, 7

- [24] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. OCRBench: On the Hidden Mystery of OCR in Large Multimodal Models. Science China Information Sciences, 67(12), 2024. 5
- [25] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language VideoLLaMA 3: Frontier Multimodal Foundation Models for Im-

- age and Video Understanding . In Proceedings of the Association for Computational Linguistics (ACL), 2024. 2, 7
- [26] Andr´es Marafioti, Orr Zohar, Miquel Farr´e, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, et al. SmolVLM: Redefining small and efficient multimodal models. arXiv:2504.05299, 2025. 2, 6, 7
- [27] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. In Findings of the association for computational linguistics: ACL, 2022. 5
- [28] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. DocVQA: A Dataset for VQA on Document Images. In WACV, 2021. 5
- [29] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. InfographicVQA. In WACV, 2022. 5
- [30] Jesse Mu, Xiang Lisa Li, and Noah D. Goodman. Learning to compress prompts with gist tokens. ArXiv, abs/2304.08467,

2023. 4

- [31] Ahmed Nassar, Andres Marafioti, Matteo Omenetti, Maksym Lysak, Nikolaos Livathinos, Christoph Auer, Lucas Morin, Rafael Teixeira de Lima, Yusik Kim, A Said Gurbuz, et al. SmolDocling: An ultra-compact visionlanguage model for end-to-end multi-modal document conversion. arXiv:2503.11576, 2025. 13
- [32] Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao, Man Jiang, Xiaomeng Zhao, et al. OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotation. In CVPR, 2025. 13
- [33] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception Test: A Diagnostic Benchmark for Multimodal Video Models. In NeurIPS, 2023. 5
- [34] Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Shuangrui Ding, Dahua Lin, and Jiaqi Wang. Streaming Long Video Understanding with Large Language Models. In NeurIPS, 2024. 2
- [35] Am´elie Royer, Moritz B¨ohle, Gabriel de Marmiesse, Laurent Mazar´e, Neil Zeghidour, Alexandre D´efossez, and Patrick P´erez. Vision-Speech Models: Teaching Speech Models to Converse about Images. arXiv:2503.15633, 2025. 1
- [36] Wenzhe Shi, Jose Caballero, Ferenc Husz´ar, Johannes Totz, Andrew P Aitken, Rob Bishop, Daniel Rueckert, and Zehan Wang. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. In CVPR, 2016. 2
- [37] Amanpreet Singh, Vivek Natarjan, Meet Shah, Yu Jiang, Xinlei Chen, Devi Parikh, and Marcus Rohrbach. Towards VQA Models That Can Read. In CVPR, 2019. 5
- [38] Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. WIT: Wikipedia-based Image Text Dataset for Multimodal Multilingual Machine Learning. In Proceedings of the 44th international ACM SIGIR

- conference on research and development in information retrieval, 2021. 13
- [39] Hao Hao Tan and Mohit Bansal. Lxmert: Learning crossmodality encoder representations from transformers. In Conference on Empirical Methods in Natural Language Processing, 2019. 1
- [40] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is All you Need. In NeurIPS, 2017. 3
- [41] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-VL: Enhancing VisionLanguage Model’s Perception of the World at Any Resolution. arXiv:2409.12191, 2024. 1, 2, 6, 7
- [42] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, et al. InternVideo2.5: Empowering Video MLLMs with Long and Rich Context Modeling. arXiv:2501.12386, 2025. 2, 7
- [43] Luis Wiedmann, Orr Zohar, Amir Mahla, Xiaohan Wang, Rui Li, Thibaud Frere, Leandro von Werra, Aritra Roy Gosthipaty, and Andr´es Marafioti. FineVision: Open Data Is All You Need, 2025. 5, 13
- [44] Linshan Wu, Jiaxin Zhuang, and Hao Chen. Voco: A simpleyet-effective volume contrastive learning framework for 3d medical image analysis. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22873–22882, 2024. 4
- [45] xAI. Grok-1.5 vision preview, 2024. Dataset obtained at https://huggingface.co/datasets/lmmslab/RealWorldQA. 5
- [46] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In CVPR, 2021. 5
- [47] Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Kelly Peng, Yao Lu, and Song Han. StreamingVLM: Real-Time Understanding for Infinite Video Streams. arXiv:2510.09608, 2025. 2, 4
- [48] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv:2505.09388, 2025. 5
- [49] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. ArXiv, abs/2406.06484, 2024. 2
- [50] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, et al. UReader: Universal OCR-free Visuallysituated Language Understanding with Multimodal Large Language Model. In Findings of the Association for Computational Linguistics: EMNLP 2023, 2023. 13
- [51] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mPLUGOwl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models. In ICLR, 2025. 1, 2, 4, 6, 7
- [52] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang,

- Hang Zhang, Xin Li, et al. VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding. arXiv:2501.13106, 2025. 1, 5, 6, 7
- [53] Hang Zhang, Xin Li, and Lidong Bing. Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding. arXiv:2306.02858, 2023. 2, 7
- [54] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, and Xiaojie Jin. Flash-VStream: Efficient Real-Time Understanding for Long Video Streams. In ICCV, 2025. 2
- [55] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, et al. Lmms-eval: Reality check on the evaluation of large multimodal models. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 881–916, 2025. 7
- [56] Liang Zhang, Anwen Hu, Haiyang Xu, Ming Yan, Yichen Xu, Qin Jin, Ji Zhang, and Fei Huang. TinyChart: Efficient chart understanding with program-of-thoughts learning and visual token merging. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1882–1898, 2024. 13
- [57] Pan Zhang, Xiaoyi Dong, Yuhang Cao, Yuhang Zang, Rui Qian, Xilin Wei, Lin Chen, Yifei Li, Junbo Niu, Shuangrui Ding, et al. InternLM-XComposer2.5-OmniLive: A Comprehensive Multimodal System for Long-term Streaming Video and Audio Interactions. arXiv:2412.09596, 2024. 2
- [58] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video Instruction Tuning With Synthetic Data, 2024. 5, 6, 13
- [59] Bo Zhao, Boya Wu, Muyang He, and Tiejun Huang. SVIT: Scaling up Visual Instruction Tuning. arXiv:2307.04087,

2023. 13

- [60] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, et al. MLVU: Benchmarking Multi-task Long Video Understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13691– 13701, 2025. 5

### A. Experimental Details

#### A.1. Training Details

Image data. We train our image models on FineVision [43] and a subset of LLaVA-OneVision-1.5 [2]. FineVision and LLaVA-OneVision-1.5 are curated collections of publicly available image–text datasets with 24M images covering captioning, chart reading, grounding and counting, mathematics, document understanding and OCR, and general VQA. In FineVision, we replace Doclingmatix [31] with Docmatix [18] and over-sample it by a factor of 6.

As the two datasets overlap substantially, we only retain a subset of LLaVA-OneVision-1.5 datasets not already included in FineVision: OmniDocBench [32], allenai-pixmo-docs [10], amc-aime, aops-forum, arxiv-figs, diagram, GQA [15], infographic-azuregpt4v, invoices, latex-ocr, llava-cot-100k, llava-wild, llrv-gpt4v, olympiads, oroikon-chart-captioning, rootsautomation, sherlock, SVIT [59], TinyChart [56], UReader-chart [50], UReaderocr [50], UReader-tr [50], viquae, vision-oritented, visual-chat, and WIT [38].

We process images at their native resolution using Qwen 2.5-VL’s visual encoder. During training, we use image resolution up to 9522 pixels. Note that the encoder applies pixel unshuffling, reducing the number of visual tokens by a factor of 4. For instance, a 9522 image yields 1156 image tokens, while a 5042 frame yields 324.

Video training data. LLaVA-Video-178K [58] is a video instruction-tuning dataset comprising 1.3M question–answer pairs over 178K videos of up to 3 minutes, annotated with open-ended and multiple-choice questions generated using GPT-4o and human input. In practice, we sample frames at 2 fps, except for clips shorter than 10s, for which we uniformly sample 20 frames. We process frames

- at lower resolutions than for single images (5042 pixels), to compensate for the higher number of images.

Training schedule. Training is performed on 64 NVIDIA H100 GPUs with a batch size of 128 for all CA models (batch size 64 and 2 gradient accumulation steps). For token insertion, the feasible sequence length per device is smaller due to higher memory constraints. We thus trade-off shorter sequence lengths (half shorter) with a higher batch size (4 gradient accumulatio steps insteas of 2) to maintain a total sequence length similar to that of CA models. Our CA training starting from Helium1-2B takes 25 hours (40k steps), while our adaptation of Qwen2.5-VL-3B takes 16 hours for the image-only training stage (25k steps), and 26 hours for image-video training (15k steps). Training our InsertionHe-2B model takes 24 hours (40k steps, 4 accumulation steps). Each GPU processes either a sequence of image samples (with multimodal sequence packing, as described in Sec. 3.3 or a single video sample. More specifically,

for each batch, we sample either a packed image sequence consisting of multiple question-answer pairs from the image training data (see above) or a single video sample, at a ratio 3:1 (image:video). The packed image sequence is limited by a maximum number of text and image tokens, specified in Table 7, which also reports the trained parameters for each training stage.

Optimization. We use a standard cross-entropy-based nexttoken-prediction loss applied only to the answer tokens of a given question-answer pair. We use AdamW with a constant learning rate schedule apart from a linear warmup and decay. The learning rate is set to 10−4 for training new parameters, and 10−5 for adapting existing parameters.

#### A.2. Architecture Details

Multimodal sequence packing. To train our tokeninsertion model with sequence packing, we also employ block-wise attention to guarantee that the self-attention operations are properly masked: Each question-answer sample in the packed sequence attends to itself without carrying over any textual context from preceding samples in the sequence. This makes the procedure equivalent to batched training, while being more efficient as it avoids padding samples to the maximum sequence length within the batch.

Attention windows in CA. As detailed in Sec. 3.2 and Fig. 2, the attention operation in CA layers acts in local attention windows, which are naturally delimited by image occurrences: Each window consists of a single image (or multiple consecutive images) followed by the associated text. Consequently, (i) during text-image training with packed multimodal sequences, windows in CA layers consist of question-answer pairs and their associated image(s). (ii) When training on LLaVA-Video-178K [58], we define a separate window for each frame as discussed in Sec. 4.3: Simply put, each window contains a single frame and their image delimiter tokens from the Qwen’s chat template, acting as gist tokens; except for the last window which contains the question and answer text tokens. In Table 4 we also report results closer to standard CA-based models, where all frames extracted from the video are included in a single window, preserving the complete visual history. (iii) When training on LiveCC [5], each window consists of a single frame (extracted at 2fps) and the corresponding closed captions for this timestamp, which is typically only a few tokens long. Nevertheless, the global coherence of the entire video script is preserved through the text-only tokens interactions in the self-attention layers.

Chat template. For Qwen2.5-VL, we use the model’s provided chat template, which includes pre- and post-image tokens (⟨VISION START⟩ and ⟨VISION END⟩), user–assistant turn delimiters, and a system prompt; we simply omit the insertion of image-token placeholders when training CA.

Trained params Max #Tokens Model Stage

Image encoder

LLM #Steps Image Text (4 last blocks)

Helium1-2B Image stage ✓ ✓ 40k 20,480 2048

Image training ✓ ✗ 25k 20,480 2048 Image-video training ✗ ✗ 15k 46,080 3072 LiveCC training ✗ ✗ 20k 30,720 3072

Qwen2.5-VL

- Table 7. Training configurations. Apart from newly introduced parameters (i.e., CA layers, Q-Former [20]) which are always trained at a base learning rate of 10−4, previously existing parameters (i.e., the image encoder and pretrained language model) are either frozen or trained with a learning rate of 10−5. For all experiments (except smaller-scale ablations), we train on 64 NVIDIA H100 GPUs with 2 gradient accumulation steps, i.e. the maximum number of tokens per gradient update is 128 × #{Image + text tokens}.

High-res Document/Chart Understanding Scene Text Understanding Knowledge / General QA CHARTQA DOCVQA INFOVQA OCRBENCH TEXTVQA REALWORLDQA AI2D GQA MME

InsertionHe-2B

504 78.0 75.2 44.0 682 69.6 54.9 64.5 52.1 1686 728 78.9 85.2 52.3 699 72.6 57.5 65.1 52.1 1690 952 80.0 87.2 55.3 745 74.4 59.6 64.6 54.0 1676

1400 79.6 87.8 58.8 744 74.5 54.2 64.5 54.1 1679

CAHe-2B

504 72.1 69.5 38.6 672 65.9 53.2 66.5 54.0 1735 728 75.9 83.5 49.2 712 72.7 58.2 66.2 54.3 1721 952 75.9 85.8 51.8 722 73.8 58.0 66.2 54.3 1731

1400 75.9 53.2 726 74.1 57.6 66.1 54.3 1716

- Table 8. Impact of image resolution at evaluation. We evaluate our CAt+v-Helium-2B and InsertionHe-2B models, for different input image resolutions. For comparison, in Table 2, these models are trained and evaluated at the same max resolution of 9522 pixels

For Helium1-2B, we use a minimal template in which user and assistant turns are wrapped with their respective start and end tokens, without any pre- or post-image tokens.

Image processing. As we rely on the vision encoder of Qwen2.5-VL for all of our models, we apply the corresponding preprocessing for images (patch size of 14, flattening, etc.). Note that for videos, however, we do not use Qwen2.5-VL’s video-processing strategy of temporally downscaling the video frames by a factor of 2, as we found it detrimental in our experiments. Instead, we process videos frame by frame, in the same way as images.

RoPE in CAt+v. We do not use any positional embeddings in standard cross-attention layers. For the CAt+v variant which includes text tokens as key-values, we apply the language backbone’s RoPE implementation within CAt+v layers, i.e. standard RoPE for Helium1-2B and multi-modal RoPE for Qwen2.5-VL with the temporal position of image tokens frozen (they do not advance across image tokens).

### B. Additional Results

#### B.1. Training Speed

In Figure 5 and Figure 6, we report the performance of our CA models trained from Helium and adapted from Qwen2.5-VL respectively, at different time during training. We find that both settings require few training steps to train the CA layers, as most performance gains occur within the

first 20k training steps for both models. For reference, we report our final results in Table 3 for 25k training steps and in Table 2 for 40k training steps.

#### B.2. Impact of image resolution

In Table 2, we report results on VLM benchmarks for CAHe-2B and InsertionHe-2B evaluated at maximum resolutions of 9522. In Table 8, we additionally show results when varying the maximum resolution at inference (4482, 6722, 8962, and 13442). As expected, resolution can be reduced with little to no loss of performance for general VQA tasks, whereas tasks involving high-resolution images (DocVQA and InfoVQA) exhibit a noticeable performance drop.

### C. Live Video Captioning

Finally, we experiment with the task of streaming video captioning to showcase CA’s practical benefits. To that end, we directly finetune the CAtv,Q2.5-VL model on the recent Live-Whisper-526K dataset[5] for up to 20,000 steps. We report a set of qualitative videos with caption subtitles generated with CAtv,Q2.5-VL-LiveCC on our project page: The captions are displayed at the time they are generated by the model, to mimic livestreaming conditions. For a static visualization of video excerpts, see Figure 3 and Figure 7.

CA (Helium1-2B) Benchmark performance vs. training step

###### ChartQA

###### DocVQA

###### InfoVQA

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

50

80

70

45

60

60

40

50

Score

Score

Score

35

40

40

30

30

25

20

20

20

0 10k 20k 30k 40k

0 10k 20k 30k 40k

0 10k 20k 30k 40k

Training step

Training step

Training step

###### OCRBench

###### TextVQA

###### RealWorldQA

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

57.5

70

70

55.0

60

60

52.5

50

50

Score

Score

Score

50.0

40

40

47.5

30

30

45.0

20

20

42.5

40.0

10

0 10k 20k 30k 40k

0 10k 20k 30k 40k

0 10k 20k 30k 40k

Training step

Training step

Training step

###### AI2D

###### GQA

MME

55

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

65

1600

50

60

1400

Score

Score

Score

55

45

1200

50

40

1000

45

35

800

0 10k 20k 30k 40k

0 10k 20k 30k 40k

0 10k 20k 30k 40k

Training step

Training step

Training step

Figure 5. Performance across training for CA (Helium1 - 2B) across all benchmarks

CA (Qwen2.5-VL 3B) Benchmark performance vs. training step

###### ChartQA

###### DocVQA

###### InfoVQA

87.5

57.5

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

80.0

85.0

55.0

77.5

82.5

52.5

75.0

80.0

72.5

50.0

Score

Score

Score

70.0

77.5

47.5

67.5

75.0

45.0

65.0

72.5

42.5

62.5

0 5k 10k 15k 20k 25k

0 5k 10k 15k 20k 25k

0 5k 10k 15k 20k 25k

Training step

Training step

Training step

###### OCRBench

###### TextVQA

###### RealWorldQA

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

78

76

- 56
- 57
- 58
- 59
- 60
- 61

76

74

74

Score

Score

Score

72

72

70

70

68

68

0 5k 10k 15k 20k 25k

0 5k 10k 15k 20k 25k

0 5k 10k 15k 20k 25k

Training step

Training step

Training step

###### AI2D

###### GQA

MME

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

- 56
- 57
- 58
- 59

1900

74

1850

1800

72

Score

Score

Score

1750

70

1700

1650

68

1600

0 5k 10k 15k 20k 25k

0 5k 10k 15k 20k 25k

0 5k 10k 15k 20k 25k

Training step

Training step

Training step

Figure 6. Performance across training for CA (Qwen2.5-VL 3B) across all benchmarks

[Figure 9]

00:00 00:02 00:04 00:06 00:08 00:10 00:12 00:14 00:16 00:18

###### [00:00|0.0s] This video shows [00:01|0.0s] a pangolin [00:02|0.0s] [00:03|-0.0s] moving around. [00:04|0.0s] [00:05|0.0s] It’s a [00:06|0.0s] [00:07|0.0s] member of the [00:08|0.0s] [00:09|-0.4s] tubulidae family. [00:10|0.0s] [00:11|-0.0s] It’s a [00:12|0.0s] [00:13|-0.1s] reptile. [00:14|0.0s] [00:15|0.0s] It’s [00:16|0.0s] [00:17|0.0s] a [00:18|0.0s] [00:19|-0.0s] mammal. [00:20|0.0s]

[Figure 10]

00:00 00:02 00:04 00:06 00:08 00:10 00:12 00:14 00:16 00:18

###### [00:00|0.0s] This video shows [00:01|-0.1s] a lioness [00:02|0.0s] [00:03|0.0s] and her cubs [00:04|0.0s] [00:05|-0.1s] searching for [00:06|0.0s] [00:07|0.0s] food in the [00:08|0.0s] [00:09|-0.2s] Kalahari Desert. [00:10|0.0s] [00:11|-0.0s] It’s a [00:12|0.0s] [00:13|-0.1s] hot summer day. [00:14|0.0s] [00:15|0.0s] The [00:16|0.0s] [00:17|-0.5s] lioness is looking for [00:18|-0.0s] [00:19|-0.0s] food for her [00:20|0.0s]

[Figure 11]

00:00 00:02 00:04 00:06 00:08 00:10 00:12 00:14 00:16 00:18

###### [00:00|0.0s] This video shows [00:01|0.0s] a 10 [00:02|0.0s] [00:03|-0.6s] ,000 square foot [00:04|0.0s] [00:05|-0.1s] apartment. [00:06|0.0s] [00:07|0.0s] It’s [00:08|0.0s] [00:09|0.0s] a [00:10|0.0s] [00:11|0.0s] two -story [00:12|0.0s] [00:13|0.0s] apartment [00:14|0.0s] [00:15|0.0s] building, [00:16|0.0s] [00:17|-0.0s] and [00:18|0.0s] [00:19|-1.2s] it’s 10 ,000 square [00:20|-0.5s]

Figure 7. Live captioning examples – CA. Similar to Figure 3 we show excerpts of video generated by a CAt+v model finetuned on LiveWhisperX. For full video samples, see the project page.

