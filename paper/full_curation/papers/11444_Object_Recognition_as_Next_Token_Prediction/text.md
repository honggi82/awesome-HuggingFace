# arXiv:2312.02142v4[cs.CV]31Mar2024

## Object Recognition as Next Token Prediction

Kaiyu Yue12* Bor-Chun Chen1 Jonas Geiping3 Hengduo Li1 Tom Goldstein2 Ser-Nam Lim4 1Meta 2University of Maryland 3ELLIS Institute & MPI-IS T¨ubingen 4University of Central Florida

### Abstract

Auto-Regression

[Figure 1]

so fa cat blank et

We present an approach to pose object recognition as next token prediction. The idea is to apply a language decoder that auto-regressively predicts the text tokens from image embeddings to form labels. To ground this prediction process in auto-regression, we customize a non-causal attention mask for the decoder, incorporating two key features: modeling tokens from different labels to be independent, and treating image tokens as a prefix. This masking mechanism inspires an efficient method − one-shot sampling − to simultaneously sample tokens of multiple labels in parallel and rank generated labels by their probabilities during inference. To further enhance the efficiency, we propose a simple strategy to construct a compact decoder by simply discarding the intermediate blocks of a pretrained language model. This approach yields a decoder that matches the full model’s performance while being notably more efficient. The code is available at github.com/kaiyuyue/nxtp.

Generative Decoder

Image Token Embeddings Probability

Figure 1. Object recognition as next token prediction using a generative decoder such as a transformer-based language model to auto-regressively predict object labels. Photo authorized with CC BY 4.0.

Note that CLIP predefines the gallery with a fixed number of object descriptions prior to inference. This requirement reveals that CLIP’s object embeddings cover only a portion of the textual space in practical scenarios, rather than its entirety. Additionally, enlarging the gallery has been shown to diminish its performance [19]. Given these observations, a question arises: Can we eliminate the predefined object labels or descriptions?

A direct strategy could use a generative model, particularly a large language model (LLM) [11, 87, 91, 92, 112–114], to decode labels from image embeddings. For instance, Flamingo [1, 3] employs a LLM to transform image embeddings into textual outputs for various vision tasks such as object recognition, image captioning, and visual question answering (VQA). But producing the desired results for a specific task needs several reference samples as fewshot prompts for the model. In other words, it requires predefined reference pivots to refine and align its predictions more precisely with the target task.

### 1. Introduction

This paper delves into a fundamental problem in computer vision − object recognition − translating an image into object labels. Generally speaking, the recognition framework comprises an image encoder and a decoder. The image encoder, either in the form of a convolutional neural network (CNN) [43, 60, 72, 106, 110] or a vision transformer (ViT) [28, 93, 120], produces image embeddings, while the decoder propagates them to predict object labels.

The most straightforward alternative is to skip any predefining procedure and align the LLM with the recognition task directly. This approach hinges on the fact that a LLM’s token embeddings represent the entire textual space, including all object labels. This is as opposed to predefining subsets, i.e., query galleries or reference pivots, of this space that potentially constrains the model’s capability.

If the decoder is a linear classifier [28, 43, 60, 72, 106, 110], it needs to be initialized with fixed object concepts. ResNet [43], for instance, initializes its final linear layer with 1K embeddings, a.k.a. weights, to represent 1K objects in ImageNet [25]. Such static weights, however, limit the model’s ability to recognize any object. This limitation can be mitigated using a language model [26, 114] as the decoder to generate a flexible set of object embeddings from input descriptions. For example, CLIP [93] encodes the object descriptions into dynamic weights by prompting with “a photo of a {L}”, where L could be any object name, and matches these weights with image embeddings to recognize objects.

Building on this concept, we propose a simple method that employs a language decoder to auto-regressively decode object labels token-by-token from image embeddings, as depicted in Figure 1. We operate a pretrained CLIP image encoder [93] to produce image embeddings, already aligned with text, and linearly transform them to match the language decoder’s embedding dimension.

*Work done during an internship at Meta AI. kaiyuyue@cs.umd.edu.

This auto-regressive framework, unlike the contrastive framework exemplified by CLIP [93], is trained to predict text embeddings from image embeddings, rather than aligning both. While related in spirit to recent vision-language models such as LiMBeR [81], LLaVA [68, 69], and BLIP-2 [64, 65], our method introduces differences and innovations:

First, our approach targets object recognition, as opposed to the chat-oriented VQA methods. We train on image-caption pairs, easier to collect and annotate than image-questionanswer triplets, and extract nouns from raw captions as reference labels to weakly supervise training. For inference, we generate text fragments as labels rather than sentences. In scenarios like recommendation systems [97] that require labels or tags, a simple label-only output is more concise than verbose sentences requiring further post-processing.

Second, our decoder has a different token modeling mechanism. Instead of decoding all input and output tokens in a conditional sequence as in LLMs, we ensure tokens from different labels to be independent, while tokens from the same label remain conditional. Naturally, all label tokens are conditional on image embeddings. This decoupling is based on the understanding that different labels in the same image are independent but their coexistence is determined by the underlying visual context. To this end, we customize a non-causal attention mask for our language decoder.

Further, the non-causal masking mechanism inspires a new sampling method, called one-shot sampling, to generate text tokens for labels. Instead of sampling tokens in sequence as in greedy search, beam search, and nucleus sampling [50], one-shot sampling simultaneously samples tokens of multiple labels in parallel and ranks them by their probabilities. This makes use of the strong parallelization capabilities of a transformer, leading to object recognition that is much more efficient than the aforementioned methods and does not suffer from repetition issues [35, 121].

Lastly, we put forth a straightforward strategy to enhance model efficiency of our recognition model. We hypothesize that only partial knowledge in LLMs is vital for recognition and focus on maximizing efficiency by not engaging the entire language model. To construct the decoder, we start with a pretrained LLM, e.g., LLaMA [112, 113], retain first six transformer blocks along with the final output layer, and drop the intervening blocks. This compact decoder matches the full model’s performance but is substantially more efficient, i.e., 4.5× faster in inference.

### 2. Related Work

Aligning Images and Text, including sentences, phrases, or words, in a shared space has been prevalent for imagetext matching [9, 23, 34, 49, 57, 59, 78, 108, 119], and foundational in contrastive frameworks [40, 75, 93], while others

are geared towards generating text descriptions from images [55, 56, 59, 78, 108, 115]. Then, integrating visual perception with LLMs [114] like GPT [11, 87, 91, 92] and LLaMA [112, 113] is gaining traction by treating image embeddings as language token embeddings, seamlessly fusing visual and textual information within the model [48, 105]. Such methods are being applied to tasks such as detection [14], few-shot recognition [1, 93], textual explainations [10], classification justification [45], bottleneck models [100, 122], reasoning [2, 42, 46, 77, 80, 103], and chat-based models [22, 64, 65, 68, 69, 81] for captioning and VQA.

Tackling Open-Vocabulary Tasks for recognition [93], detection [29, 38, 61, 82, 83, 123] and segmentation [29, 36] typically involves training on a set of base labels and then recognizing rare unseen labels. The cornerstone of openvocab approaches is the contrastive learning [41, 109] like CLIP [93], which employs a language model to encode labels to contrast with images. Therefore, open-vocab methods potentially inherit CLIP’s limitations discussed in Section 1 due to the predefined base and rare labels. CaSED [19] utilizes raw captions to form a vocabulary-free gallery, diverging from the gallery of predefined label vocabularies. However, its performance is heavily dependent on gallery selection, as demonstrated in Table 10 of [19], highlighting its limitations as a retrieval-based method.

We argue that by dramatically increasing the training data to cover a wide array of objects, the reliance on recognizing rare data and concepts can be heavily reduced. Our method aligns more with the open-world paradigm [6] that incrementally learns new labels over time, mirroring the way of data collection in the real world. In the application, given just an image, our model predicts labels with ranking probabilities, without relying on any predefined set of concepts.

### 3. Method

#### 3.1. Revisiting Object Recognition

We begin by briefly reviewing object recognition in its general formulation. Suppose that 2D images are fed into a backbone, e.g. ViT [28] in CLIP [93], which produces image embeddings1 Xv ∈ RM×D, where M is the spatial size and D is the embedding dimension. In a nutshell, the problem of recognition aims to decode object labels solely from Xv, translating image embeddings into the textual space.

In the past years, the core design of this translation employs a set of textual embeddings W ∈ RN×D to seek the optimal alignment with Xv:

arg max σ(Wf(Xv)⊤), (1)

1Bold capital letters denote a matrix X, and bold lower-case letters a column vector x. xi and xj represents the ith row and jth column of the matrix X respectively. Xij denotes the scalar in the ith row and jth column of the matrix X. All non-bold letters represent scalars.

Xv Xp

Lk

where σ is the softmax function and f is to transform Xv for aligning with W. For instance, linear classifiers such as ResNet [43] employ the average pooling as f to transform Xv to a single vector representation, and initiate W using a set of predefined concepts corresponding to object labels, e.g., N = 1000 for ImageNet [25]. The contrastive frameworks such as CLIP [93] embed a collection of predefined object descriptions into W, and apply an aggregation (like [CLS] embedding [28]) and linear projection as f on Xv.

###### Xv Xp

∞

[SEP]

M =

ind o ors cat so fa

0.0

Figure 2. Non-causal attention mask for prefixing image tokens Xv and decoupling tokens from different labels Lk to be independent at the [SEP] token.

Eq. 1 aims to maximize the alignment between f(Xv) and W. The space of W plays a critical role in this alignment as the diversity and richness of the embeddings in W directly affect the model’s ability to differentiate objects. The linear classifiers and contrastive frameworks, however, limit W to a predefined subset that potentially constrains the model’s capability to recognize any object. Our goal is to eliminate this limitation and extend W to the entire textual space.

where wt is the t-th token of L, and w<t is the sequence of tokens before the t-th token. To compute the conditional probability in Eq. 4, the transformer-based LLM in f employs a causal mask M [114] on the pairwise attention A to model the interdependence between tokens:

A ← A + M, M = tril(∞), (5)

#### 3.2. Auto-Regression for Recognition

where tril(∞) is with zeros in the lower triangle and infinity values in the upper triangle. This enforces the token wt to attend only to the preceding tokens w<t, i.e., making wt conditional on w<t, as shown in the left of Figure 2.

Recently, LLMs have significantly advanced in understanding and generating text [11, 87, 91, 92, 112–114]. Considering that their token embeddings are trained to represent the entire textual space, we define W with the token embeddings2 from a pretrained LLM, e.g., LLaMA [112, 113], featuring N = 32K textual tokens. Then Eq. 1 changes to predicting the token:

#### 3.3. Non-causal Masking

In general, an image contains multiple objects, and our goal is to predict them all. Suppose there are K objects, and we denote the output set of labels for the image as L = {L1,...,LK}, where k-th label has Tk + 1 tokens, including the special token [SEP] for the delimiter. Then the likelihood of this set of labels appearing in the image is the product of their probabilities:

P(w|Xv) = arg max σ(Wf(Xv)⊤), (2)

where w represents the most probable single token for Xv. In our method, f is a combination of linear projection and the pretrained LLM to project Xv in the textual space of W. That is, f is our language decoder.

Tk+1

K

K

To guide the language decoder in the recognition task, we prompt it with a short instruction − “the objects in the image are” − tokenized as Xp ∈ RP×D. Then we concatenate Xv and Xp to form our input token embeddings:

P(wtk|w<tk , X). (6)

P(L) =

P(Lk) =

t=1

k=1

k=1

Now Eq. 6 is not a standard auto-regression practiced in LLMs because wtk only needs to attend to the input tokens X and the preceding tokens w<tk from the same label Lk. This is supported by the understanding that the labels coexist in the same image due to the underlying visual context, but are independent of each other. Additionally, the image tokens Xv exhibit inherently spatial correlation, in contrast to the temporal correlation of natural language tokens. Therefore, we customize a non-causal attention mask M with two designs, illustrated in the right of Figure 2: a) We decouple the correlation between tokens from different labels at the [SEP] token to prevent these tokens from being attended to each other; b) We treat image tokens Xv as a prefix [27, 70, 94, 116–118], enabling the image tokens to see each other. Interestingly, our non-causal attention mask shares a similar design as the column mask in [95] but is developed from a different perspective, where the column mask is specifically for image-to-image attention.

X = Xv ⊕ [IMG] ⊕ Xp, (3)

where ⊕ is the concatenation operation and [IMG] is a special token to indicate the boundary.

Typically, a label consists of multiple tokens, e.g., “sofa” has two tokens [so] and [fa]. Without loss of generality, we assume a label L has T tokens. Now predicting L is equivalent to auto-regressively predicting its tokens:

T

P(wt|w<t, X), (4)

P(L) = P(w1, . . . , wT|Xv, Xp) =

t=1

2In general, LLMs have two sets of token embeddings, one for encoding input tokens and the other for predicting output tokens. Some LLMs like GPT-2 [92] share the same embeddings for both input and output tokens [90], while others like LLaMA [113] employ different embeddings. Our method defines W with the embeddings designated for output tokens.

#### 3.5. Truncating the Decoder

In the end, Eq. 6 is our final training objective. We use the cross-entropy loss for optimization, with weakly-supervised labels3 L extracted from the corresponding image captions.

Now, considering the language model LLaMA in our decoder f, we posit that a specific subset of language understanding in its numerous parameters is vital for recognition. This realization prompts us to focus on maximizing efficiency by not engaging the entire model. We construct our language decoder, initially based on the LLaMA 7B (version 1 or 2), by truncating it to the first 6 transformer blocks along with the final output layer, as depicted in Figure 4, while preserving its tokenizer and the pretrained 32K token embeddings for encoding the input. We designate this modified version as the truncated language decoder, denoted as Langtruncated in our experiments.

#### 3.4. One-Shot Sampling

The non-causal masking decouples the tokens from distinct labels, indicating that the first token of any label could be the next after X in the first sampling round. In other words, a higher probability for the first token, being sampled after input X, would result in a higher relevance of the label to the image. This inspires us to sample tokens of multiple labels in parallel, as shown in Figure 3.

cat art character door cart wall k c

cat art character door cart wall k c

freezing

top-k

ViT-L/14 LLaMA 7B

sampling

training

oon paper

Xv

truncating

Figure 4. Encoder and truncated decoder. We retain the first 6 transformer blocks along with the final output layer of the LLaMA 7B as our truncated decoder, and train with partial encoder blocks.

itten ute

ranked logits initial tokens final tokens w/

[SEP]

Figure 3. One-shot sampling for generating tokens of top-k labels in parallel. Once the model samples the [SEP] token, the label is completed. Otherwise, the model continues for unfinished labels.

### 4. Experiments

Data. We construct training datasets at two different scales for experiments. G3M: a training group of 3M(illion) pairs combines CC3M [104], COCO Captions [15, 67], SBU [88], which is mainly used for ablation studies. G70M: We gather 67M pairs from LAION-Synthetic-115M (slightly fewer than previous work due to missing URLs) [64, 102]. Combining it with G3M, we form a 70M-pair training group for scaling-up training. For evaluation, we use the validation split of CC3M, COCO Captions, and OpenImages V7 [7]. We parse the raw captions to obtain meaningful nouns as reference labels in both training and evaluation. The processing details are described in Section A.5.

Given input tokens X, we propagate them into the decoder and rank the output logits by their softmax probabilities. The top-k tokens, called initial tokens, decide the top-k labels to be generated. The efficacy of linking initial tokens to final labels is explored in Table 8, highlighting the promise of this straightforward approach. Then we sample the next token for the top-k initial tokens in parallel, using top-1 sampling, to generate k labels. If the sampled token is [SEP], the label is completed. Otherwise, the model continues to sample the next token for the unfinished labels. Finally, we report the probability of each label as the product of its token probabilities. We refer to this approach as one-shot sampling, which enables parallel sampling of multiple labels in one shot. The key to its parallelism lies in the non-causal masking mechanism, which also avoids the repetition issue [35, 121] typically faced in greedy and beam search, as it causes the model to focus uniformly on the same input tokens X across various labels.

Implementation. The inference augmentation for input images in CLIP [93] is applied in both training and evaluation. The input size is 2242. The image encoder is ViTL/14 [28] pretrained from CLIP [93], producing 256 token embeddings with the dimension of 1024, as Xv. Note that we drop its [CLS] token. The special token embedding of [IMG] is learned during training. The special token [SEP] is the comma (,), and 32K token embeddings for the input are fixed. The max number of input tokens is 512. No [EOS] token, i.e., the end of the sentence, is used in the input. We shuffle labels for each image in training.

To sum up, the one-shot sampling differs from other sampling methods in two essential aspects: a) It operates in parallel across multiple object labels, with each parallel branch processing a small number of tokens (roughly less than ten tokens), in contrast to the sequential sampling of other methods; b) It naturally aligns with the vision recognition task by representing the image as a spatially correlated entity, while other sampling methods depict the image

Training. AdamW [74] with the cosine annealing learning rate (LR) schedule [73] is applied in single-stage training. The multi-dimensional parameters apply a weight decay of 10−1. The global batch size is 512 with 32 NVIDIA A100-SXM4-80GB GPUs. The warm-up has 2K iterations. We jointly train four parts: the last 6 blocks of the image encoder ViT-L/14, the projection layer for Xv, the special

- as a sequence of tokens.

3Our learning approach is considered weakly-supervised as the labels are incomplete and imperfect derived from raw captions.

[IMG] token embedding, and the whole truncated language decoder, using a LR of 10−5 for 3 epochs, as shown in Figure 4, taking ∼5 hours on G3M and ∼5 days on G70M.

Evaluation. The n-gram overlap metrics, including BLEU [89] and ROUGE [66], are widely used to evaluate the quality of sentences generated by language models. However, these metrics are not suitable for evaluating the quality of results in recognition tasks. For example, “car” and “automobile” have the low n-gram similarity but are semantically alike. To quantify the semantic similarity between the generated labels and the reference labels, we adopt the concept from BERTScore [124] to formulate our evaluation metric4.

Formally, given a set of reference labels R with size M and a set of generated labels G with size N, we use the sentenceBERT [96] to encode R to a set of semantic embeddings R ∈ RM×D and G to G ∈ RN×D, where D is the embedding dimension. Then we compute the cosine similarity matrix S ∈ RM×N between R and G:

ri gj⊤ ∥ri∥∥gj∥

∈ R[−1,1]. (7)

Sij =

We compute the recall for the reference set R and the precision for the generated set G:

1 M

R =

M

1 N

max

Sij, P =

j

i=1

N

j=1

Sij, (8)

max

i

where max indicates the greedy matching strategy following [124]. Finally, we compute the F1 score as the harmonic mean of R and P:

2RP R + P

. (9)

F1 =

For each sample, we evaluate the top-k generated labels out of N and report the average R, P, and F1 over all samples. Note that, different models may have different numbers of generated labels N for each image. Especially, when N < k, we do not pad the matrix S with zeros to make N = k and penalize the model. Thus, the model with N < k will have a higher P compared to the model with N = k.

#### 4.1. Main Results

The comprehensive comparisons with other related methods, including CLIP [93], Open Flamingo [3], LLaVA [68, 69], BLIP-2 [65], InstructBLIP [22], and CaSED [19], are detailed in Table 1 with top-10 predictions, and Table A.10 with top-5 predictions.

Preliminary. We construct two galleries for CLIP: a) the base gallery, highlighted in gray, contains reference labels only from the corresponding test dataset, e.g., CC3M validation labels for CC3M evaluation. b) the extended gallery,

4The metric essentially measures the model’s accuracy, as explained in Section A.4.

| |k = 1| | | |
|---|---|---|---|---|
| | |k|= 3| |
| | | |k = 5|k|
| | | | | |

0.6

0.7

0.0 0.3

0.5

0.4

precision

precision

0.6

| |
|---|

| |
|---|

0.3

| |
|---|

0.5

= 10

0.2

0.5

1.0

0.1

0.0

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

0.4 0.5 0.6 0.7

recall

recall

1.0

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | |CLIP| |
| | | | | |Flamingo InstructB|-MPT LIP|
| | | | | |Ours| |
| | | | | | | |
| | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | |CLIP| |
| | | |Flamin Instruc Ours<br><br>|go-MPT tBLIP|
| | | | | |
| | | | | |

0.9

0.8

0.9

0.7

0.6

precision

0.8

precision

0.5

0.4

0.7

0.3

0.2

0.6

0.1

0.0

0.5

0.1 0.2 0.3 0.4 0.5 0.6 0.7

0.4 0.5 0.6 0.7

recall

recall

| | | | | | | |
|---|---|---|---|---|---|---|
| | | || |
|---|
| | | |
| | | | | | | |
| | | | || |
|---|
| | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

0.6

0.7

0.5

precision

precision

0.4

0.6

0.3

0.2

0.5

0.1

0.0

0.1 0.2 0.3 0.4 0.5 0.6

0.4 0.5 0.6

recall

recall

Figure 5. Precision-recall (PR) curves on CC3M, COCO, and OpenImages validation splits within 3 rows from top to bottom. The left column is the PR curves with different thresholds, i.e., [0.0, 0.3, 0.5, 1.0], applying on the similarity matrix S in Eq. 7. The right column is the PR curves with different top-k predictions, where k is [1, 3, 5, 10]. All figures share the same legend.

includes all reference labels from the G3M training group.

Regarding CaSED [19], its performance is significantly impacted by the search gallery composition. For a fair comparison, we evaluate CaSED using: a) the released gallery provided with the paper, in gray, featuring CLIP ViT-L/14 text embeddings from CC12M [104]; b) the extended gallery, comprising CLIP ViT-L/14 text embeddings from COCO, SBU, CC3M, and LAION-400M, which covers our G70M training group. CaSED can be considered a CLIP variant, with its defining aspect being the enhanced query gallery.

We evaluate other methods using their largest publicly available models. We employ two prompt types, list and caption, to generate object labels from them, detailed in Section A.6. Also, we use the instruct prompt for instruction-based methods, similar to its use for GPT-4V Preview [86] in A.1.

|method|models (vision + lang)|prompt<br><br>|data scale|# params (B)<br><br>|CC3M R P F1|COCO R P F1<br><br>|OpenImages R P F1<br><br>|
|---|---|---|---|---|---|---|---|
|CLIP [93] CaSED [19]<br><br>|ViT L-14 + CLIPlang ViT L-14 + Retrieval<br><br>|-|400M 12M<br><br>|0.43 0.43<br><br>|0.575 0.448 0.499 0.648 0.471 0.540|0.525 0.562 0.540 0.582 0.592 0.584<br><br>|0.510 0.462 0.480 0.534 0.470 0.494|
|CLIP [93] CaSED [19] Flamingoopen [3] Flamingoopen Flamingoopen Flamingoopen LLaVA1.0 [69] LLaVA1.0 LLaVA1.0 LLaVA1.5 [68] LLaVA1.5 LLaVA1.5 BLIP-2 [65] BLIP-2 InstructBLIP [22] InstructBLIP InstructBLIP|ViT L-14 + CLIPlang ViT L-14 + Retrieval<br><br>ViT L-14 + LLaMA 1 [112]<br><br>ViT L-14 + LLaMA 1<br><br>ViT L-14 + MPT [111]<br><br>ViT L-14 + MPT<br><br>ViT L-14 + LLaMA 2 [113]<br><br><br>ViT L-14 + LLaMA 2<br><br><br>ViT L-14 + LLaMA 2<br><br>ViT L-14 + Vicuna [16]<br><br>ViT L-14 + Vicuna ViT L-14 + Vicuna ViT g-14 + Flant5xxl [17]<br><br>ViT g-14 + Flant5xxl ViT g-14 + Flant5xxl ViT g-14 + Flant5xxl ViT g-14 + Flant5xxl<br><br>|list caption list caption list caption instruct list caption instruct list caption list caption instruct|400M 403M<br><br>2.1B 2.1B 2.1B 2.1B<br><br>753K 753K 753K 1.2M 1.2M 1.2M<br><br>129M 129M 129M 129M 129M<br><br>|0.43 0.43 8.34 8.34 8.13 8.13 13.3 13.3 13.3 13.4 13.4 13.4 12.2 12.2 12.3 12.3 12.3<br><br>|0.451 0.383 0.409 0.653 0.481 0.548<br><br>0.547 0.540 0.536<br>0.548 0.521 0.527 0.554 0.569 0.553 0.534 0.533 0.527 0.540 0.528 0.526 0.634 0.460 0.528 0.588 0.450 0.505 0.538 0.515 0.518 0.632 0.453 0.522 0.572 0.498 0.522 0.544 0.557 0.542 0.600 0.539 0.561 0.596 0.554 0.567 0.639 0.487 0.546 0.529 0.604 0.555<br><br><br>|0.429 0.483 0.450 0.616 0.629 0.620 0.549 0.721 0.618<br><br>0.553 0.697 0.611<br><br>0.556 0.793 0.646<br><br>0.554 0.754 0.633<br><br><br>0.580 0.803 0.666 0.688 0.668 0.675 0.638 0.631 0.632 0.591 0.783 0.665 0.679 0.649 0.661 0.630 0.716 0.659 0.494 0.871 0.623 0.600 0.893 0.714 0.613 0.897 0.725 0.690 0.662 0.673 0.569 0.879 0.686<br><br>|0.386 0.363 0.371<br><br>0.560 0.494 0.519 0.526 0.621 0.562 0.538 0.607 0.563 0.555 0.635 0.584<br><br>0.551 0.613 0.574<br><br>0.543 0.641 0.580<br><br>0.610 0.511 0.550 0.615 0.541 0.570<br><br>0.552 0.614 0.574<br><br>0.611 0.508 0.549 0.615 0.577 0.582 0.476 0.641 0.538 0.523 0.626 0.561<br><br><br>0.544 0.634 0.578 0.647 0.539 0.581<br><br><br><br><br>0.561 0.698 0.615<br>|
|Ours Ours|ViT L-14 + Langtruncated ViT L-14 + Langtruncated<br><br>|-<br><br>|3M 70M|1.78 1.78<br><br>|0.738 0.530 0.611 0.722 0.512 0.593<br><br>|0.700 0.712 0.702 0.765 0.757 0.758<br><br>|0.613 0.544 0.570 0.663 0.564 0.603<br><br>|

- Table 1. Comparison of different methods with top-10 predictions. Bold numbers are the best results and underlined numbers are the second best results, same for the following tables.

Analytic Comparisons. In the R column of Table 1, R remains consistent as the number of reference labels per sample is fixed, so unaffected by prediction count. Higher R suggests top-k predictions have higher semantic relevance to the reference labels. Our method outperforms others for top-10 predictions across all datasets, showing our approach’s ability to yield more relevant labels.

The P column is sensitive to the quantity of predictions; for instance, if we assess top-10 predictions but the model produces only five labels, the precision will be higher than that of the model yielding 10 predictions, according to Eq. 8. To better understand the P/R relationship, we plot two different precision-recall (PR) curves in Figure 5, calculated by adjusting the match threshold between references and predictions, and altering k for predictions.

The left column of Figure 5 derives from various thresholds on the similarity matrix S in Eq. 7 with top-10 predictions. The curves demonstrate a strong linear correlation due to the calculation of P and R from the best matches in S. A threshold of 0.7, for example, excludes pairs with lower similarity, reducing both P and R simultaneously. The rate

- at which P and R decline with increasing thresholds reflects the overall similarity of predictions to reference labels − a faster drop means the lower overall similarity. Our method, with the gradual descent of the curves, suggests better prediction quality across all test datasets. At a threshold of 1.0, non-zero values of P and R signify that the model’s predictions perfectly match the reference labels.

The right column of Figure 5 shows the PR curves for varying top-k predictions, with the inverse correlation between

P and R, indicating their trade-off. Our method outperforms others in both P and R at top-1 and -3, while at top-5, Flamingoopen and InstructBLIP saturate at the same level as top-10, even we double their sampling tokens for trying to generate more. This observation demonstrates that VQAbased models are suboptimal for the task due to the lack of the ability to generate diverse labels consistently. The plateau explain their highest P, but lower R and F1 in Table 1. Our method can achieve higher recall with increasing k, showing that it can consistently hold a P/R balance.

### 5. Ablation Studies

Truncating the Language Decoder. To test our conjecture that only a subset of knowledge in LLMs is vital for the task, we reduce the decoder’s size starting from LLaMA 7B. We have found that removing intermediate transformer blocks results in a compact decoder with comparable performance.

To begin, we need to determine which transformer blocks to remove out of the 32 blocks in LLaMA 7B. Drawing inspiration from [44], we initially fine-tuned the last third, i.e., 11 blocks, along with the final output layer. On the other hand, motivated by the observation that the language decoder takes image embeddings as the input with a novel domain, we fine-tune the first third of the blocks, i.e., 11 blocks, and the final output layer. This approach is premised on the hypothesis that the initial blocks might be better suited to learn the image embeddings. As evidenced by Table 2, indeed the first third of the LLaMA 7B emerges as the most significant segment. Therefore, we decided to remove blocks after the 11th block.

|f.t. part|CC3M R P F1<br><br>|COCO R P F1<br><br>|OpenImages R P F1|
|---|---|---|---|
|first third last third<br><br>|0.679 0.602 0.632 0.651 0.586 0.611|0.621 0.802 0.698 0.585 0.748 0.654<br><br>|0.559 0.593 0.569 0.550 0.587 0.562|

- Table 2. Partial fine-tuning (f.t.) results of LLaMA 7B with top-5 predictions, sampled by one-shot method. The first third encompasses the first 11 transformer blocks plus the final output layer, while the last third includes the last 11 blocks with the output layer.

|# params<br><br>|CC3M R P F1<br><br>|COCO R P F1<br><br>|OpenImages R P F1|
|---|---|---|---|
|7.05B - 32 3.00B - 11 1.78B - 6<br><br>|0.679 0.602 0.632 0.676 0.600 0.630 0.673 0.598 0.627|0.621 0.802 0.698<br><br>0.622 0.805 0.699<br><br><br>0.618 0.799 0.695<br><br>|0.559 0.593 0.569 0.561 0.598 0.572<br>0.560 0.595 0.570<br>|
|1.18B - 3 0.77B - 1|0.670 0.595 0.624 0.665 0.590 0.620<br><br>|0.615 0.795 0.692 0.610 0.790 0.688<br><br>|0.558 0.593 0.568 0.555 0.590 0.565|

- Table 3. Comparison of different language decoder sizes with top-5 predictions, sampled by one-shot method. The number of parameters counts both the image encoder (0.43B) and the language decoder. It is paired with the number of transformer blocks in our language decoder, e.g., 1.78B model has 6 blocks in the decoder, denoted as 1.78B - 6.

|decoder w/ LLaMA<br><br>|CC3M R P F1|COCO R P F1<br><br>|OpenImages R P F1<br><br>|
|---|---|---|---|
|3B [113] 7B → 2.6B<br><br>|0.718 0.522 0.599 0.745 0.532 0.615<br><br>|0.689 0.702 0.693 0.703 0.716 0.707<br><br>|0.612 0.546 0.571 0.615 0.546 0.572|

- Table 4. Comparison between truncated decoder and small language model at equivalent model size with top-10 predictions.

|sampling<br><br>|CC3M R P F1<br><br>|COCO R P F1|OpenImages R P F1<br><br>|
|---|---|---|---|
|greedy beam one-shot|0.661 0.604 0.624 0.641 0.590 0.608 0.673 0.598 0.627<br><br>|0.606 0.802 0.687 0.585 0.772 0.663 0.618 0.799 0.695<br><br>|0.549 0.599 0.565 0.530 0.577 0.546 0.560 0.595 0.570<br><br>|

- Table 5. Comparison of different sampling methods using top-5 predictions. The greedy and beam search sample up to 64 tokens, and takes first five generated labels as predictions.

Note that, we always retain the final output layer of LLaMA for generating the final logits. Initially, we truncate LLaMA 7B at the 11th block, as illustrated in Figure 4, resulting in a 3B model. Table 3 shows that the 3B model matches the full model in performance. To further explore the impact of the decoder size, we truncate the 3B model’s decoder by removing its last 5 transformer blocks to produce a 1.78B model and find it still performs comparably to the full model. Until the 0.77B model, which has only one transformer block, the performance has a noticeable drop but small.

The other way to construct the decoder is directly using relative small LLMs, e.g., LLaMA 3B [113]. Table 4 shows our truncated decoder outperforms LLaMA 3B at the same model scale, indicating that truncated decoders can be benefited from the better token embeddings of the larger LLMs. Plus, truncating enables models to flexibly balance accuracy and efficiency across different model scales as in Table 3.

Sampling Strategies. We investigate three deterministic token sampling methods: greedy search, 3-way beam search, and one-shot sampling. Greedy and beam search select the highest probability token, i.e., top-1, at each step. With our model, greedy and beam search suffer from the repetition issue, explained in Section 3.4. To mitigate it for the comparison, we follow [58] to penalize the logits x of the preceding generated tokens. The sampling distribution for the next token is

exp(xi/(τ · (i ∈ G))) j exp(xj/(τ · (j ∈ G)))

, (10)

p =

where τ = 1.2 is the penalization factor, (·) is the indicator function, and G is the set of preceding sampled tokens.

The results are shown in Table 5. One-shot sampling considers label count instead of token count in greedy and beam search. It generates more diverse labels without the repetition issue, explaining its superior performance in R and F1 over greedy and beam search, though with marginally reduced P, consistently in top-10 predictions (see Table A.6). Their top-10 comparisons show that, unlike one-shot sampling, increasing the number of tokens in greedy and beam search does not result in more diverse labels.

Note that our one-shot sampling could potentially encounter a competition issue, where if multiple plausible labels share the same initial token, it would sample one of them and omit the others. While sampling multiple times for the same token could mitigate this issue, in practice, its impact seems less critical than the repetition issue in sequential sampling. Plus, redundant tokenization can allow multiple labels with the same starting words being returned through different token combinations. This is tentatively indicated by our largescale predictions in Table 9.

- 3.00B

7.05B

#params

123.0 ms

226.0 ms

550.0 ms

586.0 ms

1005.0 ms

2235.0 ms

|w/ greedy search<br><br>w/ one-shot sampling|
|---|

Generation Efficiency. We combine the sampling methods with different decoder sizes to investigate their overall generation efficiency. As illustrated above, the 1.78B model is

- 4.5× faster than the 7B version in inference. Further, with one-shot sampling and truncated language model, our approach achieves 18.1× speed-up compared to the full model with greedy sampling. The inference time is measured by the average time of generating top-10 labels with one-shot sampling and 64 tokens with greedy search per image. The models run with a batch size of 1 and 16-bit Floating Point, i.e., FP16, on an A100 GPU. Attention is without kv-cache.

1.78B

Non-causal Masking. In Section 3.3, the non-causal masking considers two aspects: a) prefixing image embeddings

Xv in the input sequence, and b) decoupling tokens from different labels to be independent. The first ablation is to un-prefix the image embeddings as a sequential input. Table

truncated versions of LLaMA, namely 1.78B models of LLaMA 1 [112] and LLaMA 2 [113]. LLaMA 2 marginally outperforms LLaMA 1 trained on G3M, and has comparable results trained on G70M.

- 6 shows that the prefixing is beneficial for the performance, especially with the sequential sampling strategy, i.e., greedy search. For the one-shot sampling, the prefixing helps with a slight improvement on COCO.

Ranking Predictions. Our one-shot sampling method selects the final top-k labels based on the probabilities of their initial tokens. Table 8 demonstrates the effectiveness of this approach compared to using full label probabilities. Further details on ranking strategies can be found in A.2.

The second ablation is to model tokens conditionally from different labels, also shown in Table 6. Independent modeling is able to also provide marginal performance improvement with both greedy search and one-shot sampling, even though it provides significant gains in efficiency due to the parallelized decoding of all object labels.

Large-scale Prediction. We evaluate our method on largescale prediction, i.e., top-100 predictions, with the same settings as in Table 1. Table 9 shows our method’s consistent ability to predict diverse labels as the number of predictions increases, where R and F1 are improved, and P is decreased. Besides, CLIP [93] has a similar trend, but its performance is much lower than ours. Further, with inflating its gallery from base to the extended one, CLIP has a performance drop across all datasets, also observed in [19].

CC3M COCO OpenImages modeling R P F1 R P F1 R P F1 greedy search

baseline 0.662 0.577 0.611 0.602 0.754 0.667 0.539 0.559 0.543 + prefix 0.664 0.580 0.613 0.604 0.759 0.670 0.541 0.563 0.546 + indep. 0.668 0.600 0.625 0.609 0.797 0.688 0.548 0.588 0.561

one-shot sampling

[Figure 2]

[Figure 3]

baseline 0.677 0.601 0.630 0.611 0.790 0.687 0.556 0.592 0.567 + prefix 0.678 0.603 0.632 0.613 0.792 0.689 0.557 0.594 0.568 + indep. 0.679 0.602 0.632 0.621 0.802 0.698 0.559 0.593 0.569

- Table 6. Ablations for prefixing image embeddings and independent modeling of different labels with top-5 predictions, generated by greedy search and one-shot sampling.

CC3M COCO OpenImages

version R P F1 R P F1 R P F1 trained on G3M

- 1 0.673 0.598 0.627 0.618 0.799 0.695 0.560 0.595 0.570

- 2 0.673 0.599 0.627 0.620 0.803 0.698 0.560 0.598 0.572

trained on G70M

- 1 0.659 0.576 0.609 0.674 0.866 0.755 0.594 0.615 0.597

- 2 0.653 0.572 0.604 0.673 0.865 0.754 0.593 0.614 0.596

- Table 7. Comparison of truncating different LLaMA versions for the language decoder with top-5 predictions.

|ranking|CC3M R P F1<br><br>|COCO R P F1<br><br>|OpenImages R P F1|
|---|---|---|---|
|full<br><br>|0.673 0.598 0.627 0.673 0.598 0.627|0.618 0.799 0.695<br><br>0.619 0.800 0.695<br>|0.560 0.595 0.570 0.562 0.597 0.572<br><br>|

- Table 8. Comparison of different strategies for ranking top-

5 predictions. The first row ranks predictions using initial token probabilities, whereas the second row uses full label probabilities, derived by multiplying token probabilities.

|method<br><br>|CC3M R P F1<br><br>|COCO R P F1|OpenImages R P F1<br><br>|
|---|---|---|---|
|CLIP CLIP|0.752 0.360 0.483 0.615 0.332 0.427<br><br>|0.715 0.430 0.536 0.576 0.411 0.478<br><br>|0.666 0.387 0.485 0.506 0.334 0.399|
|ours<br><br>|0.868 0.394 0.538|0.930 0.499 0.649<br><br>|0.874 0.448 0.589|

- Table 9. Large-scale top-100 predictions with the same settings in Table 1. Different LLaMA Versions. In Table 7, we compare two

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

Figure 6. Qualitative results with top-10 predictions. The top bar is with the first prediction’s probability. The right gray column displays GPT-4V Preview [86]’s predictions. For extensive results of 336 images, refer to Section A.8.

### 6. Conclusion

We have presented an auto-regressive framework for object recognition based on next token prediction, efficiently generating labels with one-shot sampling in parallel and intuitively depending only on the number of required labels.

### A. Appendix

#### A.1. Compare with GPT-4V Preview

Since the GPT-4V(ision) Preview [86] is also able to generate object labels for images, we compare our method with it for the recognition task. The API parameters for the GPT4V Preview [86] are: input image size is 2562, temperature is zero for deterministic predictions, and detail is low with sampling 65 output tokens. The model version from API is gpt-4-1106-vision-preview. We prompt it to generate ten main object labels as its top-10 predictions with the following instruction:

the instruction for OpenAI GPT-4-vision-preview API5 Describe every detail in the image by listing ten main object labels. The answer should only contain the object labels separated by a comma, for example, “car, airplane, dog”.

Due to the API request limit, we are able to evaluate it on a subset of the COCO validation split, which contains 4359 out of 5000 images in total. We compare various methods in Table A.1 with top-10 predictions, showing that our method performs better than the GPT-4V Preview [86] across all metrics, and the GPT-4V Preview has the second-highest R. The PR-curves are illustrated in Figure A.1, indicating that our method has a better P/R trade-off. Since GPT-4V Preview consistently generates ten labels for each image, its P is also low compared to Flamingoopen and InstructBLIP.

|method<br><br>|prompt|COCO R P F1<br><br>|
|---|---|---|
|CLIP [93] Flamingoopen [3] w/ MPT [111] InstructBLIP [22] GPT-4V Preview [86]|list list instruct<br><br>|0.525 0.562 0.540 0.556 0.794 0.647 0.613 0.897 0.725 0.625 0.601 0.610<br><br>|
|Ours<br><br>|-<br><br>|0.765 0.756 0.758|

Table A.1. Comparison with top-10 predictions on COCO validation subset.

Cross-Validation. As we mentioned in Section 3.3, the reference labels extracted from the raw captions are imperfect and incomplete. To verify that our method generalizes well to predict plausible labels, we conduct a cross-validation on the COCO validation subset, treating the GPT-4V Preview’s predictions as reference labels to evaluate others. Table A.2 demonstrates that our method consistently matches the performance across all metrics as presented in Table 1, in which our method ranks first in R and F1. Again, the lower P for our method is due to the fact that our model predicts the required number of labels, while others with a higher P presumably predict less than ten labels. Regarding R, LLaVA1.0 [69] ranks second in performance.

5platform.openai.com/docs/guides/vision.

| |k = 1| | | |
|---|---|---|---|---|
| | |k|= 3<br><br>k =|5|
| | | | | |
| | | | |k|
| | | | | |
| | | | | |

1.0

0.9

| |
|---|

0.8

0.0 0.3

0.9

0.7

| |
|---|

| |
|---|

0.6

0.5 1.0

precision

precision

0.8

= 10

0.5

0.4

| |
|---|

0.7

0.3

CLIP

Flamingo-MPT

0.2

InstructBLIP

0.6

GPT-4V Preview

0.1

Ours

0.0

0.1 0.2 0.3 0.4 0.5 0.6 0.7

0.4 0.5 0.6 0.7

recall

recall

Figure A.1. Precision-recall (PR) curves on COCO validation subset. The same settings as in Figure 5.

|method<br><br>|prompt|COCO R P F1<br><br>|
|---|---|---|
|CLIP [93] CaSED [19] Flamingoopen [3] w/ MPT [111] LLaVA1.0 [69] LLaVA1.5 [68] BLIP-2 [65] InstructBLIP [22] GPT-4V Preview [86]<br><br>|list caption caption caption list instruct<br><br>|0.467 0.509 0.485 0.535 0.562 0.546 0.517 0.760 0.609 0.593 0.599 0.595 0.576 0.572 0.573 0.498 0.736 0.590<br><br>0.505 0.731 0.594<br>1.000 1.000 1.000<br>|
|Ours Ours w/ top-100<br><br>|-<br><br>|0.632 0.651 0.641 0.823 0.473 0.600|

Table A.2. Comparison with top-10 predictions on COCO validation subset, viewing GPT-4V Preview’s predictions as reference labels. Gray row shows our top-100 predictions.

#### A.2. Ranking Predictions

We ablate ranking strategies for the predictions produced by our model. Given an image, our model generates K labels L = {L1,...,LK}. Each label Lk has Tk + 1 tokens, including the special token [SEP] for the delimiter.

Ranking by CLIP Score. The first strategy is to rank the predictions by the CLIP score:

clip(Lk) = fCLIP(image, label Lk), (A.1)

where fCLIP is the CLIP model [93] with the image encoder of ViT-L/14 and the language encoder. The CLIP score is based on cosine distance in the embedding space.

Ranking by Probability. The second strategy is to rank the predictions by their probabilities in Eq. 6:

prob(Lk) =

Tk+1

P(wtk|w<tk , X), (A.2)

t=1

in which the probability of each label is the product of the individual probabilities of its tokens, including the delimiter token [SEP]. If greedy and beam search sample a particular label multiple times, we sum up the probabilities as its final probability.

Ranking by Perplexity. The third one is to rank the predictions by their perplexities. The perplexity is computed with the fixed length Tk + 1 for each label:

1 Tk + 1

ppl(Lk) = exp −

Tk+1

log P(wtk|w<tk , X) . (A.3)

t=1

If the greedy and beam search sample a particular label multiple times, we use its minimum perplexity to ensure optimal selection and accuracy.

Ranking by Cross-Modal Similarity Score. The last one is to rank predictions by their cross-modal similarity scores, computed with the image and label token embeddings:

1 Tk

sim(Lk) =

Tk

d(wtk, Xv), (A.4)

t=1

where d is the euclidean distance averaged over all the image token embeddings for each label token embedding wtk:

M

wtk · xvi wtk 2 · ∥xvi∥2

1 M

d(wtk, Xv) =

, (A.5)

2 − 2 ·

i=1

where M is the number of image tokens. This similarity is also called compatibility score to measure the compatibility between image and label embeddings, which motivates us to select the predictions that are compatible with the corresponding images. In other words, the closer the label token embeddings are to the image token embeddings, the more likely the label is the correct prediction.

Results. Table A.3 compares the above four ranking strategies using top-5 predictions across different sampling methods for our 1.78B model trained on G3M. The greedy and 3way beam search samples 64 tokens for each image. Since one-shot sampling yields ordered predictions, we sample 10 labels per image and utilize ranking strategies to select the final top-5 predictions.

The overall best ranking strategy is using probability for greedy search and one-shot sampling, and using CLIP score for beam search. For R, one-shot sampling with probability ranks first on CC3M and COCO, and the greedy search with probability leads on OpenImages. The greedy search with probability has a slightly higher P than one-shot sampling with probability, but the latter has a better overall F1.

For greedy search, the compatibility score has the same performance as the perplexity. For one-shot sampling, the compatibility score is better than the perplexity. Without a ranking strategy, one-shot sampling matches the performance of probability-based ranking, showing its effectiveness in using top-k initial tokens to decide the final top-k predictions.

No ranking strategy outperforms the CLIP score for both greedy and beam search, yet we apply CLIP score to other models like Flamingo, BLIP-2, InstructBLIP, and LLaVA.

greedy beam one-shot

ranking R P F1 R P F1 R P F1 CC3M

- 0.661 0.604 0.624 0.641 0.590 0.608 0.673 0.598 0.627 clip 0.646 0.604 0.617 0.630 0.594 0.605 0.643 0.588 0.608 prob 0.659 0.602 0.622 - - - 0.673 0.598 0.627

ppl 0.614 0.563 0.581 - - - 0.509 0.466 0.484 sim 0.611 0.564 0.581 0.598 0.557 0.571 0.594 0.531 0.556

COCO

- 0.606 0.802 0.687 0.585 0.772 0.663 0.618 0.799 0.695 clip 0.590 0.792 0.673 0.573 0.772 0.654 0.592 0.773 0.668 prob 0.603 0.796 0.683 - - - 0.619 0.800 0.695

ppl 0.578 0.748 0.649 - - - 0.528 0.640 0.577 sim 0.576 0.747 0.647 0.552 0.724 0.623 0.576 0.717 0.637

OpenImages

- 0.549 0.599 0.565 0.530 0.577 0.546 0.560 0.595 0.570 clip 0.540 0.598 0.560 0.525 0.580 0.544 0.543 0.591 0.559 prob 0.580 0.576 0.569 - - - 0.562 0.597 0.572

ppl 0.577 0.571 0.565 - - - 0.495 0.505 0.496 sim 0.575 0.571 0.564 0.509 0.553 0.524 0.527 0.547 0.532

- Table A.3. Comparison of different ranking strategies for various sampling methods with top-5 predictions. In the case of “-”, no ranking strategy is used, and one-shot sampling directly outputs the top-5 labels.

|ranking<br><br>|CC3M R P F1|COCO R P F1<br><br>|OpenImages R P F1|
|---|---|---|---|
|clip|0.545 0.568 0.549 0.551 0.574 0.555<br><br>|0.548 0.794 0.643 0.552 0.801 0.648|0.526 0.655 0.576<br><br>0.527 0.657 0.577<br>|

- Table A.4. Comparison of different ranking strategies with top-
- 5 predictions for Flamingoopen + MPT.

For BLIP-2, InstructBLIP, and LLaVA, whose outputs are sentences, the CLIP score is the only choice for ranking. But for Flamingo, since it has a same format as ours, we can test its performance without ranking strategy. Because it saturates at top-10, we only report its top-5 comparison. The results are shown in Table A.4, showing that the CLIP score is the optimal ranking strategy for those models.

#### A.3. Additional Results

In this section, we present additional results, mainly with top-10 predictions, for ablation studies.

Ablation on Truncating the Decoder. We compare the results of different truncating sizes of the language decoder with top-10 predictions in Table A.5. There is a small performance drop, 0.745 → 0.738 in R on CC3M, with truncating the decoder from 3B to 1.78B, while the performances on COCO and OpenImages remain the same.

Ablation on Sampling Methods. We compare sampling methods, i.e., greedy search, 3-way beam search, and oneshot sampling, with top-10 predictions in Table A.6. The results, consistent with those in Table 5, indicate that oneshot sampling surpasses greedy and beam search in R and

F1 scores but falls short in P when considering top-10 predictions. The reason is that greedy and beam search produce ∼7 labels average per image in top-10 due to the repetition issue. Figure A.2 (right side) demonstrates saturation around k = 7, accounting for their higher P in top-

- 10 predictions. This ablation study shows that greedy and beam search do not produce more diverse predictions with increasing number of tokens.

k = 1

0.6

| | | | | | |
|---|---|---|---|---|---|
| | | | | |0.3|
| | | | | | |
| | | |0.5| | |
| | | | | | |
| | |1.0| | | |
| | | | | | |

one-shot sampling

greedy search

0.0

beam search

0.7

0.5

precision

precision

0.4

k = 3

0.3

k = 5 k = 7 k = 10

0.6

0.2

0.1

0.2 0.3 0.4 0.5 0.6

0.4 0.5 0.6

recall

recall

Figure A.2. Precision-recall (PR) curves of different sampling methods on OpenImages validation split with top-10 predictions. The same settings as in Figure 5.

Ablation on LLaMA Versions. Table A.7 compares the results of different LLaMA versions for the language decoder with top-10 predictions. The top-10 results are consistent with Table 7, showing LLaMA 2 is slightly better than LLaMA 1 on G3M, and comparable on G70M.

Ablation on Embedding Models in Evaluation Metric. The evaluation metric is based on embedding models to compute the similarity Sij in Eq. 7. To verify the robustness of our method, we compare the results using CLIP ViT-L/14 [93] as the metric embedding model in Table A.8. Our results are from the 1.78B model trained on G70M, and the others are from the best settings in Table 1. Our method consistently outperforms others in R and F1 scores, and is competitive in P.

Ablation on Training Epochs. We conduct an ablation study on training epochs for our 1.78B model on G3M. Table A.9 shows the results with top-10 predictions, indicating that training more epochs improves the performance.

Additional Main Results. Table A.10 shows the main results with top-5 predictions, consistent with those in Table 1. The performance drop on CC3M for models trained on G3M versus G70M stems from a data distribution shift.

#### A.4. Evaluation Metric

The recall in evaluation metric Eq. 8 essentially represents the top-k accuracy, which is for recognition tasks [99].

For an image, ground-truth (GT) labels are G = {gi}Mi=1, ordered model predictions are P = {pj}Nj=1. The standard recall is defined as Recall = TP/(TP + FN).

|# params<br><br>|CC3M R P F1|COCO R P F1<br><br>|OpenImages R P F1|
|---|---|---|---|
|7.05B - 32 3.00B - 11 1.78B - 6<br><br>|0.748 0.534 0.617 0.745 0.532 0.615 0.738 0.530 0.611|0.699 0.710 0.702 0.703 0.716 0.707 0.698 0.712 0.702<br><br>|0.613 0.543 0.569 0.615 0.546 0.572<br>0.613 0.544 0.570<br>|
|1.18B - 3 0.77B - 1<br><br>|0.736 0.530 0.611 0.731 0.529 0.608|0.697 0.713 0.703 0.693 0.708 0.698<br><br>|0.612 0.547 0.571 0.609 0.547 0.569|

- Table A.5. Comparison of different language decoder sizes with top-10 predictions. The same settings as in Table 3.

|sampling<br><br>|CC3M R P F1|COCO R P F1<br><br>|OpenImages R P F1|
|---|---|---|---|
|greedy beam one-shot|0.708 0.568 0.621 0.681 0.557 0.604 0.738 0.530 0.611<br><br>|0.655 0.755 0.696 0.623 0.725 0.665 0.698 0.712 0.702<br><br>|0.582 0.574 0.569 0.557 0.552 0.546 0.613 0.544 0.570<br><br>|

- Table A.6. Comparison of different sampling methods with top10 predictions. The greedy and beam search sample 128 tokens for each image without ranking strategies.

CC3M COCO OpenImages

version R P F1 R P F1 R P F1 trained on G3M

- 1 0.738 0.530 0.611 0.698 0.712 0.702 0.613 0.544 0.570

- 2 0.740 0.531 0.612 0.700 0.714 0.705 0.614 0.547 0.571

trained on G70M

- 1 0.722 0.512 0.593 0.765 0.757 0.758 0.663 0.564 0.603

- 2 0.721 0.512 0.593 0.765 0.756 0.758 0.662 0.563 0.602

- Table A.7. Comparison of truncating different LLaMA versions for the language decoder with top-10 predictions.

|method<br><br>|CC3M R P F1|COCO R P F1<br><br>|OpenImages R P F1|
|---|---|---|---|
|CLIP Flamingo BLIP-2 InstBLIP<br><br>|0.799 0.746 0.771 0.842 0.842 0.841 0.864 0.838 0.850 0.883 0.827 0.853<br><br>|0.774 0.783 0.778 0.835 0.922 0.875 0.854 0.961 0.904 0.892 0.887 0.889<br><br>|0.762 0.725 0.742 0.838 0.863 0.849 0.822 0.864 0.841 0.878 0.842 0.859<br><br>|
|Ours|0.908 0.825 0.864<br><br>|0.915 0.911 0.913<br><br>|0.881 0.838 0.858<br><br>|

- Table A.8. Comparison with top-10 predictions using CLIP ViTL/14 [93] as the embedding model in evaluation metric.

|epoch|CC3M R P F1<br><br>|COCO R P F1<br><br>|OpenImages R P F1|
|---|---|---|---|
|1<br><br>2<br><br>3<br><br><br>|0.654 0.487 0.553 0.698 0.509 0.583 0.738 0.530 0.611|0.620 0.623 0.620 0.659 0.667 0.661 0.700 0.712 0.702<br><br>|0.591 0.520 0.548 0.604 0.528 0.558 0.613 0.544 0.570<br><br>|

- Table A.9. Comparison of different training epochs with top-10 predictions.

For recognition tasks, GT should either be TP (correctly identified) or FN (missed), i.e., TP + FN = |G| = M, then

TP |G|

TP M

TP TP + FN

. (A.6)

=

=

Recall =

For closed-set recognition, TP = Mi=1 I(gi ∈ P), where gi ∈ P is a greedy matching – correct prediction is exactly the same as gi with maximum semantic similarity, e.g., gi =

|method<br><br>|models (vision + lang)<br><br>|prompt<br><br>|data scale|# params (B)<br><br>|CC3M R P F1|COCO R P F1<br><br>|OpenImages R P F1<br><br>|
|---|---|---|---|---|---|---|---|
|CLIP [93] CaSED [19]<br><br>|ViT L-14 + CLIPlang ViT L-14 + Retrieval|-<br><br>|400M 12M<br><br>|0.43 0.43<br><br>|0.515 0.481 0.493 0.577 0.520 0.541<br><br>|0.468 0.590 0.523 0.533 0.666 0.590|0.460 0.485 0.467 0.490 0.506 0.492<br><br>|
|CLIP [93] CaSED [19] Flamingoopen [3] Flamingoopen Flamingoopen Flamingoopen LLaVA1.0 [69] LLaVA1.0 LLaVA1.0 LLaVA1.5 [68] LLaVA1.5 LLaVA1.5 BLIP-2 [65] BLIP-2 InstructBLIP [22] InstructBLIP InstructBLIP|ViT L-14 + CLIPlang ViT L-14 + Retrieval<br><br>ViT L-14 + LLaMA 1 [112]<br><br>ViT L-14 + LLaMA 1<br><br>ViT L-14 + MPT [111]<br><br>ViT L-14 + MPT<br><br>ViT L-14 + LLaMA 2 [113]<br><br><br>ViT L-14 + LLaMA 2<br><br><br>ViT L-14 + LLaMA 2<br><br>ViT L-14 + Vicuna [16]<br><br>ViT L-14 + Vicuna ViT L-14 + Vicuna ViT g-14 + Flant5xxl [17]<br><br>ViT g-14 + Flant5xxl ViT g-14 + Flant5xxl ViT g-14 + Flant5xxl ViT g-14 + Flant5xxl<br><br>|list caption list caption list caption instruct list caption instruct list caption list caption instruct|400M 403M<br><br>2.1B 2.1B 2.1B 2.1B<br><br>753K 753K 753K 1.2M 1.2M 1.2M<br><br>129M 129M 129M 129M 129M<br><br>|0.43 0.43 8.34 8.34 8.13 8.13 13.3 13.3 13.3 13.4 13.4 13.4 12.2 12.2 12.3 12.3 12.3|0.400 0.388 0.390 0.571 0.521 0.539 0.542 0.541 0.535 0.539 0.523 0.525<br><br>0.551 0.574 0.555<br><br>0.532 0.537 0.528 0.537 0.522 0.522 0.588 0.520 0.547 0.566 0.507 0.531 0.535 0.523 0.521 0.581 0.510 0.543<br><br>0.552 0.530 0.532<br><br><br>0.541 0.558 0.541 0.594 0.549 0.564 0.593 0.559 0.569 0.603 0.535 0.561 0.529 0.605 0.556<br><br>|0.385 0.489 0.427 0.532 0.683 0.596 0.541 0.726 0.616 0.547 0.712 0.614 0.552 0.801 0.648 0.551 0.762 0.635 0.574 0.790 0.659 0.601 0.755 0.667 0.600 0.746 0.662 0.581 0.800 0.666 0.600 0.751 0.664 0.589 0.786 0.667 0.482 0.842 0.606 0.600 0.894 0.714 0.613 0.897 0.725 0.604 0.752 0.667 0.569 0.881 0.686<br><br>|0.349 0.366 0.354 0.498 0.526 0.505 0.524 0.622 0.561 0.533 0.608 0.561 0.527 0.657 0.577<br><br>0.544 0.655 0.588<br><br>0.545 0.632 0.578 0.545 0.557 0.545 0.567 0.589 0.571<br><br><br>0.545 0.618 0.573 0.551 0.560 0.555 0.566 0.607 0.576 0.466 0.626 0.526 0.523 0.626 0.561<br>0.546 0.640 0.582 0.572 0.585 0.572 0.559 0.698 0.614<br><br><br>|
|Ours Ours<br><br>|ViT L-14 + Langtruncated ViT L-14 + Langtruncated|-<br><br>|3M 70M|1.78 1.78<br><br>|0.673 0.598 0.627 0.659 0.577 0.609<br><br>|0.618 0.799 0.695 0.674 0.866 0.755<br><br>|0.560 0.595 0.570 0.594 0.615 0.597|

Table A.10. Comparison of different methods with top-5 predictions. The same settings as in Table 1.

pj = cat, and I(·) is binary. This Recall is also called Exact Recall [124], also known as accuracy in image classification tasks [99]. In detail, to evaluate a classifier on ImageNet, each image has M = 1 GT label and N = 1000 class predictions, then Eq. A.6 becomes

top-k accuracy = Recall = I(g1 ∈ P1:k), (A.7)

For open-set recognition, TP = Mi=1 I(gi ∈ P), gi ∈ P is a greedy matching but I(·) is not binary because correct pre-

diction might not be exactly the same as gi. For instance, gi = cat, pj = kitty or feline or moggie are all correct with high semantic similarity, and pj = dog or desk are wrong with low semantic similarity. I(·) is continuous to represent degrees of semantic similarity between gi and pj. One common choice for I(·) is cosine similarity Sij between contextual embeddings of gi and pj, then Eq. A.6 becomes

M

1 M

maxj Sij, (A.8)

Recall =

i=1

which is a.k.a. BERT Recall [124]. For the open-set case, each image has M ≥ 1 GT labels and N ≥ 1 predictions, then top-k accuracy is

1 M

Recalltop-k =

M

1 M

I(gi ∈ P1:k) =

i=1

M

maxj∈[1,k] Sij.

i=1

(A.9)

The top-k refers to the k most relevant predictions of all possible labels in the world to the image.

#### A.5. Data Preprocessing

For an image, the paired caption is preprocessed using the steps summarized in the following table.

|step<br><br>|details|
|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br><br>|Lowercase the caption. Eliminate high-frequency noise words that lack meaningful content. The noise words removed in our work are [ person, persons, stock, image, images, background, ounce, illustration, front, photography, day ]. Keep only the letters, and a few special characters like spaces ( ), periods (.), commas (,), ampersands (&), and hyphens (-). Exclude all others, including numbers and words containing numbers. Use NLTK [8] to tokenize the caption into words. Then tag the words with their part-of-speech (POS) tags to filter out words that are not nouns. The noun tags used in this paper are [ NN, NNS ]. Lemmatize the words to their root forms. For example, the word “dogs” is lemmatized to “dog”.|

With this preprocessing, we obtain a set of meaningful noun words for each image and summarize the information in the following table, including the number of image-caption pairs and distinct nouns.

|statistics<br><br>|CC3M train val|COCO train val<br><br>|SBU train<br><br>|OpenImages val<br><br>|LAION train|
|---|---|---|---|---|---|
|# images # nouns|2.69M 12478 22890 4875<br><br>|118287 5000 15444 3834<br><br>|828816 132372|41686 3119<br><br>|67M 2.7M|

The training split contains 2,794,419 distinct nouns, while all validation splits have a total of 8,637 distinct nouns. The

number of overlapping nouns between the training and validation splits is 8,347, which is 97.8% of distinct nouns in validation splits.

#### A.6. Prompt Settings

For training, we adopt the prompt augmentation, which contains different prompt templates but with the same semantic meaning. In each training iteration, we randomly select one prompt from those templates for the batched images. For inference, we only use one simple prompt in all experiments. The prompt templates are listed as follows.

|setting|prompt templates<br><br>|
|---|---|
|training|The objects in the image are<br><br>The items present in the picture are<br><br>The elements depicted in the image are<br><br>The objects shown in the photograph are<br><br>The items visible in the image are<br><br>The objects that appear in the picture are<br><br>The elements featured in the image are<br><br>The items captured in the photograph are<br><br>The elements seen in the picture are<br><br>The items represented in the image are<br><br>|
|inference|The objects in the image are|

CC3M COCO OpenImages

# tokens R P F1 R P F1 R P F1 prompt: list

64 0.542 0.556 0.540 0.482 0.842 0.606 0.455 0.622 0.518 128 0.544 0.557 0.542 0.494 0.871 0.623 0.476 0.641 0.538 256 0.542 0.556 0.540 0.482 0.842 0.606 0.455 0.622 0.518

prompt: caption

64 0.601 0.539 0.561 0.600 0.893 0.714 0.523 0.626 0.562 128 0.609 0.539 0.561 0.600 0.893 0.714 0.523 0.626 0.562 256 0.600 0.539 0.560 0.601 0.894 0.714 0.512 0.643 0.562

- Table A.11. Different number of sampling tokens for BLIP-2 with top-10 predictions.

CC3M COCO OpenImages

# tokens R P F1 R P F1 R P F1 prompt: list

256 0.596 0.554 0.567 0.613 0.897 0.725 0.546 0.640 0.582 512 0.596 0.554 0.567 0.613 0.897 0.725 0.544 0.634 0.578

prompt: caption

256 0.639 0.487 0.546 0.690 0.662 0.673 0.647 0.539 0.581 512 0.639 0.487 0.546 0.690 0.662 0.673 0.647 0.539 0.581

- Table A.12. Different number of sampling tokens for InstructBLIP with top-10 predictions.

#### A.7. Number of Sampling Tokens in Comparison

We have various models to compare with ours. For a fair comparison, we need to take care of the maximum number of sampling tokens for each model to make sure that we can extract enough potential nouns words from their outputs. LLaVA [68, 69] has a maximum number of sampling tokens of 1024, which is already enough for the task. BLIP2 [65] has a maximum 32 in default, but we change it to 64 for top-5 and 128 for top-10. To verify this setting is fair for BLIP-2, we ablate the number of sampling tokens for BLIP2 with the caption prompt in Table A.11. For InstructBLIP [22], we use its default number of sampling tokens, which is 256 for top-5 and top-10. To verify the setting, we ablate the number of sampling tokens for InstructBLIP in Table A.12. Due to Flamingo [1, 3] has the same output format as ours, we keep the same maximum number of sampling tokens for it as ours for greedy search, i.e., 64 for top-5. We double the number to 128 for its top-10 predictions. For VQA methods, sampling more tokens for more potential predictions significantly increases time cost, esp. with beam search.

For comparison, we evaluate chat-based VQA models, i.e., BLIP-2 [65], InstructBLIP [22], and LLaVA [68, 69], with two types of prompt, which are

- 1) text completion: The objects in the image are,
- 2) and VQA: Describe every detail in the image. We refer to the text completion prompt as prompt: list and the VQA prompt as prompt: caption. After obtaining model outputs, we apply the rule from Section A.5 to extract nouns as predicted labels.

Especially, Flamingo [1, 3] has a unique prompt setting with few-shot instruction. For the caption type, we change the prompt setting to What objects are in the image?. Then we construct the prompt with 4-shot samples as in [1], which is listed as the following tables.

the list prompt type with few-shot samples for Flamingo

<image>The objects in the image are boy, bush, chair, clothes, grass, house, tree, sports ball.<|endofchunk|> <image>The objects in the image are bus, car, clouds, house, leaves, person, road.<|endofchunk|> <image>The objects in the image are giraffe, grass, tree.<|endofchunk|> <image>The objects in the image are cat, telecontroller, sofa.<|endofchunk|> <image>The objects in the image are

#### A.8. Visualizing Predictions

We visualize the top-10 predictions from our 1.78B model trained on G70M in Figure A.3-A.9 without cherry-picking. The image is paired with two columns: our predictions on the left, probability-indicating ranking bars on the right. The images sampled from COCO have gray column to show GPT-4V Preview’s [86] predictions, intuitively illustraing the strengths and weaknesses of our method with the applesto-apples comparison.

[Figure 18]

the reference images as few-shot samples for Flamingo

[Figure 19]

#### A.9. Discussion

In this section, we discuss the limitations of our method and experiments that we have tried but does not work well.

Less Is More. Our method’s performance heavily relies on the quality of the training data. More noisy data will hurt the performance, for example, models trained on the noisier CC12M [12] underperform compared to those trained on CC3M [104]. Moreover, high quality requires more human efforts, which is expensive, meaning to densely annotate all possible labels for each image. We might consider using GPT-4V [86] for generating high-quality labels, though it may be costly (API expenses) and subject to the hallucination issue. Exploring methods to train models with fewer labels for broader generalization could be intriguing.

Defining Labels. How to define the label for describing an object in an image? A label could be a word, which is used in this paper, but also could be a phrase or a sentence. We have tried to define the label with the noun phrase, which includes an adjective, for example, “gray car” and “cute boy”. However, these models underperformed, partly due to poor training data quality and the limitations of the parser for extracting noun phrases from captions. We also experimented with concrete nouns for training, but the results were unsatisfactory due to noisy reference labels produced by the parser, which needs a comprehensive filter to remove noise.

Evaluation. First, our evaluation has limitations due to the incomplete and imperfect nature of reference labels derived from raw captions. Second, we calculate P, R and F1 score based on the semantic similarity between the embeddings of predicted and reference labels from a pretrained language model. However, such a model-based semantic similarity brings noise and bias to the evaluation results due to the model imperfection. This motivates us to conduct the cross-validation experiments in Section A.1, which views GPT-4V’s [86] predictions as reference labels. Developing a reliable evaluation metric beyond human evaluation or model-based semantic similarity is an interesting topic.

Fine-Grained Recognition. Our method, though not designed for fine-grained recognition, could be adapted for such tasks. Currently, the method underperforms in this area due to the use of general, rather than fine-grained, training data. Improving performance may be possible by using more specific, fine-grained training data, which circles back to the initial question regarding the quality of training data.

Single-Label Prediction. Our method is optimized for topk predictions and exhibits lower performance in top-1 accuracy evaluations. Our approach encourages the model to predict multiple labels for an image, which is more realistic than predicting just one label because images generally contain multiple objects. Therefore, we do not focus on improving top-1 accuracy in this paper.

Competition Issue. We acknowledge the inherent competitive issue in our one-shot sampling, similar to the repetition issue observed in sequence-based methods like greedy and beam search. However, its results are still promising in experiments, which is likely due to redundant tokenization. Mitigating or analyzing the competition issue for the one-shot sampling could be our future research topic.

#### A.10. Other Related Works

Approaching object recognition as a natural language prediction, pioneered by [4, 31, 85], has been proposed before the deep learning era [63]. The motivation is primarily to assist journalists in annotating images for retrieval purposes [5, 79]. [85] slices an image into regions and predicts words using probabilistic models. [31] views recognition as a machine translation problem, aligning image regions with words using a lexicon, optimized by the EM algorithm [24].

Image Annotation and Multi-label Prediction. The evolution of image annotation or tagging closely mirrors that of multi-label prediction. Initial approaches develop on topic models [53] like latent Dirichlet allocation [5] and probabilistic latent semantic analysis [49, 84]. Mixture models [32, 52, 62] have also been explored to model the joint distributions over images and tags. Then SVM-based discriminative models [21, 47, 54] are proposed to predict tags. Later, the annotation task is treated as a retrieval problem [39, 76] based on nearest neighbors [20] or joint optimization [13]. The difficulty of collecting multi-label annotations inspires curriculum learning-based models [18, 30] and semi-supervised methods [33, 101, 107]. Now models with ranking-based losses [37] and transformer-based architecture [51, 71, 98, 125] are introduced for tagging images, but they are still closed-set recognition models trained on heavily-annotated/cleaned datasets. In contrast, our method is an open-set recognition model trained on raw data, which is at the real open-level with a large-scale prediction capability (top-100). In the figure below, our model correctly predicts the wild terms such as sora, cloudscape, text, logo, letter, art, and animation, assigning probabilities for ranking or filtering, while [125] does not.

Recognize Anything [122]

[Figure 20]

bar stool | bulletin board | canteen | ceiling | chair | coffee shop | table | restaurant | ﬂoor | retail | stool | store

Our Top-20 Predictions

| prob: 0.15203 - coffee | prob: 0.09728 - shop | prob: 0.09182 - counter | prob: 0.03848 - interior | prob: 0.03389 - bar | prob: 0.03215 - restaurant | prob: 0.02440 - table

| prob: 0.02245 - store | prob: 0.01950 - area | prob: 0.01905 - inside | prob: 0.01590 - starbucks | prob: 0.01313 - cafe | prob: 0.01220 - chair | prob: 0.01172 - ﬂoor

| prob: 0.01020 - cup | prob: 0.00879 - drink | prob: 0.00794 - room | prob: 0.00746 - customer | prob: 0.00635 - wood | prob: 0.00345 - bakery

Recognize Anything [125]

[Figure 21]

coffee | coffee cup | computer | control | cup | table | electronic | gadget | game | game controller | ipad | joystick | laptop | mug | tablet | pine cone | remote | sit | tablet computer | window | window sill

Our Top-20 Predictions

| prob: 0.01088 - nintendo | prob: 0.01051 - computer | prob: 0.00823 - mario | prob: 0.00819 - remote | prob: 0.00734 - control | prob: 0.00713 - sill | prob: 0.00393 - desk

| prob: 0.03786 - console | prob: 0.03552 - cup | prob: 0.02562 - top | prob: 0.02060 - mug | prob: 0.01817 - screen | prob: 0.01339 - video | prob: 0.01111 - star

| prob: 0.07281 - tablet | prob: 0.06749 - coffee | prob: 0.06593 - window | prob: 0.05811 - controller | prob: 0.05653 - game | prob: 0.04788 - switch | prob: 0.04065 - wii

Recognize Anything [125]

[Figure 22]

cloud | cloudy | ﬂoat | sea | sky

Our Top-20 Predictions

| prob: 0.00256 - storm | prob: 0.00200 - name | prob: 0.00200 - cloudscape | prob: 0.00190 - sun | prob: 0.00188 - art | prob: 0.00182 - animation | prob: 0.00179 - air

| prob: 0.54264 - cloud | prob: 0.09989 - word | prob: 0.07556 - sky | prob: 0.03171 - letter | prob: 0.01874 - sora | prob: 0.01388 - logo | prob: 0.01000 - text

| prob: 0.00719 - top | prob: 0.00719 - blue | prob: 0.00682 - title | prob: 0.00611 - photo | prob: 0.00430 - picture | prob: 0.00288 - sonora | prob: 0.00271 - middle

#### A.11. Acknowledgements

We thank Alessandro Conti, the primary author of CaSED [19], for supplying the text embedding galleries for CC3M, COCO, SBU, and LAION-400M. We also thank Damian Gessler for the help on downloading training datasets and solving cluster issues, and our group colleagues at Meta for the helpful discussions.

0.27

0.21

0.31

bench tree field

tree grass bench fence sky leaves field pathway bushes clouds

baby elephant gate zoo enclosure mother

elephant baby elephant fence tree shadow dirt grass sunlight enclosure vegetation

cake flower

wedding cake cake stand flowers petals icing layers tablecloth knife heart ribbon

|[Figure 23]|
|---|

|[Figure 24]|
|---|

|[Figure 25]|
|---|

top

middle

table wedding

park grass

rose layer tier stand pink

area

fence couple

top wood forest

trunk dirt

0.30

0.34

0.27

airplane cloud sky

airplane clouds sky undercarriage wings engines fuselage landing gear airliner blue sky

zebra dirt road

zebras road grass trees sky dirt ears tails stripes savannah

dog

hot dog bun cheese mustard plate table drink cup shadow reflection

|[Figure 26]|
|---|

|[Figure 27]|
|---|

|[Figure 28]|
|---|

plate hotdog

plane fly jet

tree group

top paper table mustard

herd

grass field wild

blue landing

bun topping

top passenger

path

cheese

0.58

0.15

0.25

bus building

bus ferris wheel reflection trees side mirror windshield wipers bus fleet license plate headlight pedestrian crossing sign road

pile bunch

apple banana pear fruit stem skin bunch yellow red green

elephant fence

elephant fence trees dirt elephant dung grass pipe leaves shadow ear

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

lot parking side decker street city tower top

banana table fruit top

dirt trunk

standing

zoo baby area

apple surface

group type

enclosure tusk

0.16

0.28

0.40

origami cranes heart-shaped container paper colors table shadows folds texture pattern transparency

paper heart lot

frisbee

dog frisbee grass tail ears nose eyes mouth collar paws

giraffe zoo enclosure

giraffes fence building trees shadows enclosure sky sunlight ears spots

|[Figure 32]|
|---|

|[Figure 33]|
|---|

|[Figure 34]|
|---|

dog mouth

grass

container table glass

baby group

ball playing orange

couple building

crane box basket bowl

field toy paw

fence head back

0.23

0.36

0.32

magazine plate toast banana shrimp coffee cup notebook pen table

orange bowl juice

oranges bowl spoon sink faucet window tile basket countertop curtain

coffee plate food table book

traffic light sign

traffic light red light skull symbol street van building sign trees shadow road markings

|[Figure 35]|
|---|

|[Figure 36]|
|---|

|[Figure 37]|
|---|

counter table plate

street road

cup banana picture

car stop

top slice fruit

middle side lane

top photo

spoon

0.38

0.46

0.36

dog top

dog lemon tree birdhouse wooden fence grass shadow shrub blue sky patio sunlight

foil oven food

oven aluminum foil baked dish oven rack oven light oven door heating element kitchen appliance glass door metal shelf

refrigerator

refrigerator floor tiles cabinet dining table chairs countertop wall ceiling door bookshelf

|[Figure 38]|
|---|

|[Figure 39]|
|---|

|[Figure 40]|
|---|

kitchen fridge room

bench fence ledge

silver pan aluminum inside tray microwave tin

table cabinet

deck edge table

chair freezer dining door

tree paw

0.34

0.37

0.16

zebra

zebra trees grass shadows leaves sunlight branches soil foliage underbrush

bus street

bus street buildings traffic lights pedestrian crossing road signs sidewalk windows sky street lamps

vas wall

wall vases flowers patterns shelves colors glass reflection artwork lighting

|[Figure 41]|
|---|

|[Figure 42]|
|---|

|[Figure 43]|
|---|

tree grass baby

stop city side road

glass

display flower bottle

ground adult dirt forest field couple

decker building

vase item type case

group pink

0.27

0.21

0.47

room living

sofa armchair coffee table lamp painting window blinds bookshelf cushion vase bamboo sticks

cow field

cows grass trees sky clouds fence shadows field horizon nature person stop sign street grass house sidewalk fire hydrant trees utility pole clouds elephant fence grass house trees sky pole wire window roof bridge streetlight car lights railing sign building road grass night sky fence bed pillows bedside table lamp curtains picture frame wall carpet bedspread phone clock numbers hands time circle white black metal sky brand name

hydrant fire

fire hydrant street cars trees traffic light sky building curb chain grass

|[Figure 44]|
|---|

|[Figure 45]|
|---|

|[Figure 46]|
|---|

couch window

sky

street blue road

pasture cloud grass

furniture table chair

side building

herd cattle

lot

middle

sofa floor

tree horse

water fountain

0.43

0.21

0.25

zebra zoo hay

zebra fence grass hay tree shadow enclosure dirt leaves sunlight

street sign stop

bench fish animal

teddy bear plush fish bench grass bow tie wooden slats shadow outdoor setting red collar orange color

|[Figure 47]|
|---|

|[Figure 48]|
|---|

|[Figure 49]|
|---|

grass baby

corner house

bear top toy

pile enclosure mother field adult

road building

stuffed chair basket clown

intersection fire side

0.41

0.24

0.32

grass bear field

sign stop

stop sign car traffic lights road sky utility pole trees grass bridge additional sign

bear grass fur ears eyes nose mouth whiskers head animal

elephant fence field

|[Figure 50]|
|---|

|[Figure 51]|
|---|

|[Figure 52]|
|---|

car truck

grass couple

ground camera

road side bridge street

standing top leg

zoo

area standing

eye mouth

animal building

top traffic

0.24

0.17

0.20

sandwich table drink

sandwich plastic container apple bottled drink table label lid straw vanilla flavor wood grain

night

cow camera tongue mouth

cow chain fence hay barn ear tag nose ring shadow light wooden beam

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

car street

food coffee

light bridge

barn hay head

container bag box cup

road highway

traffic city tunnel

ear face pen

bottle

0.28

0.30

0.18

sign

street sign one way sign building window reflection sky metal pole text arrow shield shape sign

bed room

restaurant

diner sign street sidewalk fence building sky lamppost windows roof

|[Figure 56]|
|---|

|[Figure 57]|
|---|

|[Figure 58]|
|---|

sign building

street building

pillow hotel sheet

way library madison side city pole arrow

corner street city sidewalk store entrance coop

blanket top

lamp window

nightstands

0.28

0.55

0.25

egg pizza

plane

airplane sky cloud tarmac traffic cone propeller wing tail fin hangar date stamp

clock building top sign sky

pizza egg spinach cheese crust asparagus black pepper plate table lighting

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|[Figure 61]|
|---|

airplane tarmac runway

top spinach topping

top jet ground propeller

green plate food

side street

cloudy

couple aircraft

wall kertul

vegetable pan

0.18

0.44

0.18

computer desk chair

chair desk telephone laptop mouse pen lamp cable notebook cushion

zebra grass

zebra grass bushes trees soil shadows leaves stripes ears tail

light traffic street

traffic light sky tree building car street shadow grass pole sidewalk

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

field tree

laptop

pole intersection

top table hotel room

dirt bush

building

wild animal middle grazing

road car city

mouse lamp

stoplight

0.22

0.16

0.28

tray

ground microwave middle food grass top

rice cooker microwave asphalt leaves fence vegetation sign discarded object grass dirt

window sill cat

cat window sunlight curtain windowsill glass shadow tree blinds wall

grilled fish carrots asparagus potato lemon slice plate tablecloth bread glass container parsley

|[Figure 65]|
|---|

|[Figure 66]|
|---|

|[Figure 67]|
|---|

plate carrot

orange top

potato table fish

windowsill table ledge

street toaster

bread vegetable

fence side

silver platter

camera sun

0.25

0.13

0.37

room living table

sofa bookshelf lamp table picture frames cabinet rug storage boxes wall art floor

bench

street lamp bench fire hydrant platform mountains sky clouds grass power line pole

horse train field

horse grass fence train trees sky clouds bushes railway tracks field

|[Figure 68]|
|---|

|[Figure 69]|
|---|

|[Figure 70]|
|---|

middle pole road

grass track

furniture cabinet couch picture lot

street

desert park post field side

middle

tree grazing

wall bookshelf

area distance

0.33

0.30

0.26

bench building

bench railing pavement building lamp post sky roof sliding board grass water

cat shoe top ground

cat shoes shoelaces wall floor sleeping whiskers fur pattern textile

giraffe grass field mother

giraffe grass bird fence sky shadow hill ears spots tail

|[Figure 71]|
|---|

|[Figure 72]|
|---|

|[Figure 73]|
|---|

chair restaurant sign

sneaker pair foot

baby standing

table sidewalk

adult middle

top

floor head

beach window

zoo couple

paw

0.24

0.26

0.28

truck street side

truck street buildings motorcycle bicycle signage shadows electrical wires windows sky

room living

sofa dining table chairs curtains coffee table rug bookshelf lamp wall art window

phone building

landline phone wall wires plug socket switchboard curtain metal pipe screws paint cracks

|[Figure 74]|
|---|

|[Figure 75]|
|---|

|[Figure 76]|
|---|

rug

wall telephone side cell

building back road horn door

furniture chair table

area television

street window

vehicle trailer

couch floor

top brick

0.24

0.15

0.23

sign street side

street sign wall no parking sign pole tree shadow traffic signal graffiti asphalt electrical wires

bus train

train bus building trees streetlight road sky bus stop windows railing

zebra

zebras fence grass trees tail stripes ears hooves soil shadow

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

zoo grass fence

city building street road

building wall pole

couple

enclosure field pair

singapore tram track

corner city

road middle

area top

car

0.22

0.32

0.42

donut

donut sprinkles hand napkin fingers pavement blurred background

bear sidewalk

teddy bear tree sidewalk curb street shadow red paint building sunlight chain-link fence

chair bear

teddy bear chair cushion armrest plaid fabric wood leather stripe pattern wall

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

someone doughnut

street

top

hand paper piece pink

tree stuffed ground

bow stuffed animal

pole road

sits plaid

sprinkle top bag

corner top

seat cushion

0.33

0.64

0.21

bunch banana counter

bananas stickers fruit bunch curvature yellow purple background shadow reflection container

bus road

vintage bus tree road building window headlight tire grille side mirror second bus

plate wine table glass

wine glasses cured meats cheese table plates napkin outdoor setting building hand cobblestone ground

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

street tree side

top metal

cheese food

orange building

kitchen plate table

meat bread bottle

lot truck van

row sink

top

0.17

0.43

0.24

street building

parking meter street building construction scaffolding sky road signs sidewalk fence power lines streetlight

zebra grass

zebras grass trees sky foliage savanna tails ears stripes hooves

umbrella building row

umbrellas sky building window antenna cable roof wall shadow sunlight

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

meter construction

field tree

sky group street

group herd wild plain savannah giraffe

pole city post

sun

lot side box

blue bunch

city

0.30

0.33

0.21

sign

street signs tree sky pole "no parking" sign leaves branches clear sky road name sign directional arrow sign

light traffic building street city intersection

clock building tower town street

building clock tower sky tree hill road street lamp window roof foliage

traffic light building tree sky window reflection street green light architecture foliage

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

street pole tree side

road sky

tree village

window pole side stop

lake corner

mountain

side church

city

0.32

0.28

0.28

orange table bench top wooden

orange wooden bench metal bolts concrete paint lines shadow texture pebbles blurred background color contrast

keyboard mouse top

keyboard mouse tabletop cable usb dongle shadow corner surface edge texture giraffes water trees grass path fence bushes rock rabbit sky table omelette plate cup coffee pot fork knife napkin shadow floor bear cutout warning sign trees person trash bin snow grass forest mountain shadow cat table remote control bookshelf books pen paper chair wall jacket stop sign street lights buildings cars windows sky road crosswalk trees street lamps computer monitors keyboard mouse headphones desk water bottle soda can screen papers cloth pizza cutting board cheese grater bowl eggs whipped cream can hot sauce bottle olive oil bottle kitchen counter scissors sheep grass wool ears eyes noses legs tails field flock

piano ceiling light curtains framed picture fireplace coffee table sofa floor chess set suitcase

fireplace room living table

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

computer desk table

piano

surface photo wood

chair furniture

counter apple

place television bedroom

pad laptop

middle sits

0.30

0.53

0.44

elephant trunk couple field grass dirt baby

elephants tusks trunks ears dirt vegetation bushes sky tails wrinkles

giraffe dirt water pond path tree zoo

giraffe tree field

giraffe trees grass sky branches savannah sunlight shrubs horizon leaves

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

grass bush

middle standing

road group

ground head adult

wild savannah sun

edge

0.20

0.12

0.56

plate fruit food table

ben boat

river building sky boat lights clouds pier trees windows reflection

clock building

clock building windows architectural detail cornice facade sky moulding ledge column

|[Figure 98]|
|---|

|[Figure 99]|
|---|

|[Figure 100]|
|---|

house

tower top sky

river clock

breakfast

cup coffee

parliament

church

tower london

side cloudy

top bowl omelette

building water

dome sun

0.19

0.20

0.39

bear car top

light street traffic

vase flower

vase flowers table tiles decorative skulls stems petals leaves lighting background blur

traffic light storm clouds building street sky traffic signal pole car window roof streetlight

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

table top pot

mailbox box sign dog

building city sky

shell orange glass

cloudy

truck vehicle side

intersection pole road

vas wooden

0.28

0.25

0.35

cat desk

track train

train railway tracks overhead lines highway cars grass trees sky fence light poles

hydrant

fire hydrant brick wall door sidewalk street light fixture shadow graffiti metal plate building airplane jet bridge tarmac airport sky luggage cart windows markings fence grass double oven kitchen appliance control panel glass doors racks digital display buttons handle ventilation slots black color teddy bear christmas tree lights ribbon pine needles ornaments branches wall ceiling garland

|[Figure 104]|
|---|

|[Figure 105]|
|---|

|[Figure 106]|
|---|

fire building

top table book

freight

car side

brick

side street

computer sits

road passing

door corner wall city

passenger bridge railroad

chair television keyboard

0.43

0.45

0.19

sign street stop

water plane

seaplane water mountains clouds boat reflection shoreline registration number propellers wing

plane

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

airport runway tarmac

airplane beach boat body

night side light

gate terminal

city building

jet top

top shore ocean

passenger window

road word

lake

0.29

0.24

0.34

computer desk keyboard headphone top table mouse

vehicle museum

motorcycle jeep airplane anti-aircraft gun fire wall sign spotlight painting floor

oven top stove microwave kitchen

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

display motorcycle

wall

car room army truck

door cabinet

monitor laptop screen

counter appliance wall

building

0.40

0.56

0.37

pizza counter

bear water

polar bear water rocks paw fur whiskers ears nose eyes cliff

tree

|[Figure 113]|
|---|

|[Figure 114]|
|---|

|[Figure 115]|
|---|

bear christmas

board top food

paw swimming

stuffed bow top pine ornament ribbon light

zoo pool river

kitchen plate oven knife

leg cub top

topping

0.25

0.15

0.46

field group sheep

plane airport

airplane tarmac jet bridge grass trees hills sky airport markings fence terminal building

parking lot motorcycle night bike

motorcycle parking lot street lights trees sky building curb painted lines asphalt grass

|[Figure 116]|
|---|

|[Figure 117]|
|---|

|[Figure 118]|
|---|

runway malaysia tarmac jet passenger ground top middle

grass herd flock

car street

bunch

grazing standing

light building dusk

pasture

0.24

0.12

0.30

van chip

van parking lot lays logo trees sky clouds asphalt windows wheel shadow

ben boat

river building sky boat lights clouds pier trees windows reflection

bathroom sink wall

bathroom sink mirror faucet tile wall trash can towel cabinet window picture frame

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

lot

house

food truck

river clock

mirror pink tile cabinet towel floor vanity

parking chevrolet

parliament

tower london

fry side delivery

building water

0.23

0.30

0.35

game controller

top cat

cat pillow bed bookshelf books wall blanket clothing room curtain

video game case wii remote brochure book paper table mario yoshi luigi text

bird window

window bird water rust handle latch wall reflection sign forest

|[Figure 122]|
|---|

|[Figure 123]|
|---|

|[Figure 124]|
|---|

bed orange book blanket pillow

top table wii

water boat sill top

book nintendo

ledge ocean

shelf couch

box item paper

body lake

sheet

0.12

0.21

0.33

vintage car airplane propeller tires wing windows sky tarmac antenna headlight

bulldog

dog car traffic lights sky building car window road signs car mirror clouds yield sign

airplane car plane top

clock street

clock building sky snow street sidewalk window streetlight tree roof

|[Figure 125]|
|---|

|[Figure 126]|
|---|

|[Figure 127]|
|---|

car dog

building town snow

head window

vintage couple model

truck back

sidewalk tower middle pole city

side street

sits building display

light

0.31

0.31

0.32

track train

train railway tracks overhead wires trees grass sky locomotive rust dirt vegetation

giraffe

giraffes fence grass trees rocks sky enclosure path shrubs purple flowers

giraffe field

giraffe sky clouds grass savanna horizon bushes dirt water shadow

|[Figure 128]|
|---|

|[Figure 129]|
|---|

|[Figure 130]|
|---|

zoo grass fence

grass standing

station engine railway

area field

sky middle camera top

side orange

enclosure tree dirt pond

car passing railroad

tree plain

0.26

0.16

0.28

bar flower

flowers vase window light bottle reflection shelf blinds plant shadow

plate shrimp food fry table tray

sandwich pickle plate vegetable tomato

sandwich pickles tomato slice plate potato chips greens bread cheese table shadow big ben building sky clock face light window tower silhouette dusk architecture zebras fence grass trees tail stripes ears hooves soil shadow horse dog grass halter sky field hill shadow ear muzzle

fried shrimp french fries bread collard greens plastic fork plastic cup hot sauce bottle tray paper napkin chicken wing

|[Figure 131]|
|---|

|[Figure 132]|
|---|

|[Figure 133]|
|---|

counter

vase mirror

vas light

food lettuce

sauce seafood

top room shelf

salad

fork meal

bread cheese

0.21

0.26

0.21

house street

house van street trees grass sky street sign curb shadows driveway

row motorcycle

motorcycles wheel building street sidewalk sign headlight windshield license plate reflection

tower ben

|[Figure 134]|
|---|

|[Figure 135]|
|---|

|[Figure 136]|
|---|

tree

street sidewalk building line

clock london sunset

home neighborhood

car truck

parliament night dusk

group scooter

sign grass road

bunch lot

sky england

0.30

0.17

0.23

building

bridge river building clock tower windows sky crane reflection railing street lamp

herd bison

bison snow river trees grass sky clouds forest shadows sunlight

zebra

|[Figure 137]|
|---|

|[Figure 138]|
|---|

|[Figure 139]|
|---|

river bridge

zoo grass fence

river snow

tower city water clock chicago

buffalo field animal

couple

enclosure field pair

tree grazing

view station

area top

group

0.27

0.29

0.29

bench park tree middle grass garden

bench grass trees pathway leaves metal armrests plants curb soil shrubbery

sandwich plate salad

plate sandwich salad bread chicken lettuce mixed greens soda can table fence pizza scooter model parchment paper countertop oven kitchen backsplash cheese tomato sauce crust pepperoni hot dog bun chili hand fingers sauce meat nails thumb wrist building ivy clock windows alley car dumpster street signs shadows doorway statue cross church sky clouds spires clock windows sun brickwork cat shoes shoelaces wall floor sleeping whiskers fur pattern textile sofa television armchair coffee table window curtains carpet picture frames floor lamp bookshelf baseball player baseball bat catcher umpire baseball helmet baseball glove stadium seats audience baseball uniform protective gear zebras grass savanna sky hill horizon wildlife herd field nature

horse

|[Figure 140]|
|---|

|[Figure 141]|
|---|

|[Figure 142]|
|---|

dog field

grass standing

lettuce chicken meat bread table

top middle

metal ground

retriever mouth brown

plant sidewalk

side food

0.34

0.26

0.43

car building

car boat building trees bus bicycle sign fence road grass zebra grass sand rocks trees enclosure fence sky antelope shadow bush sofa curtains plant table lamp painting carpet cushion window vase laptop robot table wheels cardboard box cables concrete floor black bag white wall metal pole giraffe mountain sky trees building fence enclosure roof windows vegetation refrigerator window chair door wall curtain electrical outlet floor ceiling cabinet

pizza top toy

pole post

parking meter yarn bombing window reflection building sidewalk bench door sign tree

|[Figure 143]|
|---|

|[Figure 144]|
|---|

|[Figure 145]|
|---|

boat lot ford road

sidewalk sign yarn

counter oven tray metal

street building

top truck

pan scooter

lamp corner

vehicle sits

motorcycle

light

0.20

0.42

0.39

zoo zebra sand dirt

dog

kitchen cabinet

kitchen cabinets refrigerator oven sink faucet range hood countertop drawer floor

|[Figure 146]|
|---|

|[Figure 147]|
|---|

|[Figure 148]|
|---|

hand sandwich

floor wood

hotdog someone

area group

door counter

bun plate

grass ground

steel stove

top meat towel

gravel animal

appliance flooring

0.25

0.35

0.65

room living

building brick clock

vase flower

flowers vase table jug wall tray stems petals leaves water

|[Figure 149]|
|---|

|[Figure 150]|
|---|

|[Figure 151]|
|---|

couch window

table top vas

photo street

furniture table chair

side entrance

purple glass plant

wall sofa

car wall vine

pot wooden

coffee

0.25

0.28

0.26

laptop top robot computer tank

statue church

boat water

boat chair backpack railing water hills sky rope deck horizon

|[Figure 152]|
|---|

|[Figure 153]|
|---|

|[Figure 154]|
|---|

cathedral building

backpack chair deck

jesus tower

vehicle floor table desk

top luggage ship ocean view

mary sky top sun

wheel

0.50

0.30

0.50

cat shoe top ground

clock

clock building windows door flag sky streetlamp steps handrails brickwork

giraffe zoo

|[Figure 155]|
|---|

|[Figure 156]|
|---|

|[Figure 157]|
|---|

building tower street

tree mountain

fence area top

sneaker pair foot

brick hotel

city side sign

park building

floor head

standing

paw

top

0.30

0.26

0.28

pigeon bird

pigeon bagel asphalt feather sunlight shadow pebble bird droppings wing beak

room living

refrigerator

|[Figure 158]|
|---|

|[Figure 159]|
|---|

|[Figure 160]|
|---|

kitchen room fridge

couch

bread ground

chair furniture

cabinet freezer

food

television cat window table dog

top doughnut

chair table floor door

group

piece parking

0.22

0.23

0.27

ground microwave middle food grass top

rice cooker microwave asphalt leaves fence vegetation sign discarded object grass dirt

baseball player

dog

hot dog bun cheese mustard plate table drink cup shadow reflection

|[Figure 161]|
|---|

|[Figure 162]|
|---|

|[Figure 163]|
|---|

plate hotdog

ball catcher

top paper table mustard

plate game field home

street toaster

bun topping

fence side

base dodger

cheese

0.47

0.27

0.33

elephant

elephants river trees rocks sand water foliage sky mud herd

zebra grass

panda

panda bamboo fur claws eyes nose ears mouth whiskers leaves

|[Figure 164]|
|---|

|[Figure 165]|
|---|

|[Figure 166]|
|---|

river water group

bear bamboo

field herd

grass

group plain wild animal

herd

tree zoo cub

stream baby bank

branch forest top

hole jungle

area savannah

0.17

0.24

0.43

shuttle

mountain

ear cat

|[Figure 167]|
|---|

|[Figure 168]|
|---|

|[Figure 169]|
|---|

space launch

river valley

rabbit bunny

pad rocket

stream footage

pink hat

grass field

sky lift

costume head scarf

satellite nasa cape

middle water range

fur

0.21

0.28

0.26

tree road

monkey top branch

claus santa

|[Figure 170]|
|---|

|[Figure 171]|
|---|

|[Figure 172]|
|---|

forest light path

christmas present gift vector character clause box decoration

tree sky log

wood middle

stump sits fence sun

sun photo

end

0.25

0.56

0.28

desk window

calculator pencil pen top

plate fork avocado lasagna

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

chair office

computer corner room wall home bedroom

number keyboard

bean casserole vegetable

photo computer

salad food quesadilla

table crayon

0.22

0.19

0.32

table thanksgiving

building street city

plant shelf

|[Figure 176]|
|---|

|[Figure 177]|
|---|

|[Figure 178]|
|---|

pumpkin

wall hanging pot planter item

candle decoration

side truck

dining room

car view

plate turkey dinner

traffic intersection road

basket room wood

0.17

0.27

0.37

grass sky

view

pie

|[Figure 179]|
|---|

|[Figure 180]|
|---|

|[Figure 181]|
|---|

car sun

chocolate glass plate

wind blowing

road photo

field cloud plant

cream top slice dessert table dish

cloud sky

mirror highway

beach reed tree

window

0.35

0.40

0.67

rug

pumpkin scissors table top paper craft

pill capsule glass

|[Figure 182]|
|---|

|[Figure 183]|
|---|

|[Figure 184]|
|---|

design series room word

jar bottle photo tablet

pattern text designer living floor

pair item

medicine surface pile

supply diy

0.37

0.24

0.24

painting shape color line

bridge sidewalk walkway

water

|[Figure 185]|
|---|

|[Figure 186]|
|---|

|[Figure 187]|
|---|

pig beach ocean

footage

piece abstract

river water

sand

animal island shore

city

wall paper art brown

street view road

swimming boar

0.16

0.33

0.29

building

cup spoon saucer

card butterfly

|[Figure 188]|
|---|

|[Figure 189]|
|---|

|[Figure 190]|
|---|

water croatia

wish flower way

city view

table

coffee tea top

greeting stampin'up handmade

sea town

boat hotel body

plate teacup

word paper

mug

0.09

0.39

0.23

ink drawing paper

lion rock cub group top sunset family hill

night highway

|[Figure 191]|
|---|

|[Figure 192]|
|---|

|[Figure 193]|
|---|

car road light view

pen pencil drawn

hand table

freeway traffic street

top sketch

africa mountain

city

0.51

0.10

0.30

earring

cone construction

cathedral

|[Figure 194]|
|---|

|[Figure 195]|
|---|

|[Figure 196]|
|---|

church town night

diamond pair gold

road

sign traffic street

city spire riga

crystal

drop stone silver cross

orange

side intersection

square building

corner

dusk

dangle

0.40

0.28

0.27

earring pair gold

soda

road mountain

|[Figure 197]|
|---|

|[Figure 198]|
|---|

|[Figure 199]|
|---|

can vector

tree forest

art coca

stone lavender

highway side

drink

purple drop pearl

style icon cola

middle pine view

gem amethyst

beverage

car

0.17

0.33

0.11

garden italy castle villa view

gold coast

rainbow

|[Figure 200]|
|---|

|[Figure 201]|
|---|

|[Figure 202]|
|---|

cloud cat top

surfer view ocean

unicorn cartoon

city beach tower

palace countryside

horse sky animal pink

house building

skyline water

lawn

0.23

0.28

0.19

flower

wall city night jerusalem

building church stone

|[Figure 203]|
|---|

|[Figure 204]|
|---|

|[Figure 205]|
|---|

invitation pink card

door courtyard

ribbon wedding

view street

tree window

paper grey bow

israel

car light

middle wall arch

book

tower

0.12

0.44

0.35

sunset

cliff coastline sunset

yankee cap era

|[Figure 206]|
|---|

|[Figure 207]|
|---|

|[Figure 208]|
|---|

rock beach ocean

grey baseball

sea view

sea photo

beach

hat logo

indonesia thailand coast philippine

bay ocean

mlb gray side

england dusk

0.30

0.40

0.24

rise game card board box expansion monster dungeon cthul chubo

castle sand beach

living room

|[Figure 209]|
|---|

|[Figure 210]|
|---|

|[Figure 211]|
|---|

wallpaper

sky photo ocean tower

sofa furniture

couch design decor table mirror

top desert turret

0.19

0.16

0.25

side pillow

car concept

rope anchor

|[Figure 212]|
|---|

|[Figure 213]|
|---|

|[Figure 214]|
|---|

bed

mercedes

boat

case word cover

door vehicle

water vector

show frankfurt

art

text couple

sea ocean

silver

grey sheet

auto motor

ship sailboat

0.16

0.31

0.33

shore mountain

view mountain

chair beach shore

|[Figure 215]|
|---|

|[Figure 216]|
|---|

|[Figure 217]|
|---|

lake beach water canoe

balcony

tree window

sea water

glass house deck hill valley

rock bench stone lounge ocean

boat kayak tree top

0.19

0.11

0.45

computer screen

sea

spoon

|[Figure 218]|
|---|

|[Figure 219]|
|---|

|[Figure 220]|
|---|

water coral plant

chocolate sauce syrup liquid

skull sign icon

ocean

bone monitor

reef group

top cocoa fork glass milk

symbol skeleton

algae tube worm

photo

0.52

0.23

0.21

drop water ocean vector

ice pool

shower shelf wall bathroom stone tile head

|[Figure 221]|
|---|

|[Figure 222]|
|---|

|[Figure 223]|
|---|

water

pond ground

sea wave

park block

droplet boat sky blue

sun winter

copper faucet fixture

lake

0.24

0.42

0.13

bird rock

fence

tree paper

|[Figure 224]|
|---|

|[Figure 225]|
|---|

|[Figure 226]|
|---|

flower field tree

beak puffin

card table

orange top standing cliff ground penguin

wood countryside

scene book cut doll picture bird

footage

foreground bush grass

0.37

0.12

0.49

cake birth

ocean

tree house

|[Figure 227]|
|---|

|[Figure 228]|
|---|

|[Figure 229]|
|---|

calf whale water

top tiger decoration

roof

trunk ground middle

dolphin mother

friend picture

baby adult

top home

character orange topper

sea orca

side building

0.24

0.32

0.22

pattern fabric photo

ship water ocean

beach rainbow

|[Figure 230]|
|---|

|[Figure 231]|
|---|

|[Figure 232]|
|---|

ocean sky sunset sea

color seamless

boat

sea body

royalty flower design

middle

shore print dune cloud

sky video cloud

lace vector

0.18

0.24

0.45

guitar

paper flower

wall poster museum

|[Figure 233]|
|---|

|[Figure 234]|
|---|

|[Figure 235]|
|---|

fender neck bass

craft sky

room display

piece project

head shot pickup string

painting

leaf child

exhibition exhibit gallery

art cloud

closeup stratocase

art

0.14

0.45

0.40

banana monkey

rope boat

water photo bottle

|[Figure 236]|
|---|

|[Figure 237]|
|---|

|[Figure 238]|
|---|

cucumber fruit leaf

water ocean

plastic lid glass

ship view

hand ground

sea deck

cap container

vegetable squirrel bite

sailboat photo

drop surface

0.14

0.36

0.26

abstract pattern color

field grass

apple plate oatmeal glass

|[Figure 239]|
|---|

|[Figure 240]|
|---|

|[Figure 241]|
|---|

bear walking

design triangle

standing walk boar

breakfast table food wine

shape line rainbow photo wall

forest animal camera

cereal bake

0.44

0.52

0.38

cinnamon bowl stick

candle church

ocean beach

|[Figure 242]|
|---|

|[Figure 243]|
|---|

|[Figure 244]|
|---|

row lit

word

table wooden spice cup surface top anise

poster text title

group dark light altar

quote cover

message icy

top night

0.20

0.12

0.63

sailboat river boat

building house

tile floor

|[Figure 245]|
|---|

|[Figure 246]|
|---|

|[Figure 247]|
|---|

door london

overlay bathroom

water skyline

flower window

text money

city building boston lake park

iron fence entrance england

word mosaic

title tilt

0.24

0.36

0.19

candle shelf table

house tree

rock water garden gravel

|[Figure 248]|
|---|

|[Figure 249]|
|---|

|[Figure 250]|
|---|

home garden middle

driftwood glass top plant branch log vase

stone ground pebble

yard

building bush grass

fire dirt bed

cottage

0.23

0.26

0.45

sign beach flower

hand bottle

cloud airplane window

|[Figure 251]|
|---|

|[Figure 252]|
|---|

|[Figure 253]|
|---|

plastic method

view wing

sand wedding

soap product

plane sky mountain

tulip word message stand top

liquid dish cleaning pack

jet top

0.14

0.28

0.31

city skyline

donkey cart

library book shelf

|[Figure 254]|
|---|

|[Figure 255]|
|---|

|[Figure 256]|
|---|

view building

road horse

row lot room

wagon dirt side carriage field trailer

sunset bangkok saigon photo thailand korea

middle hallway

floor school

0.19

0.25

0.21

track wave

turbine wind field

candle wooden

|[Figure 257]|
|---|

|[Figure 258]|
|---|

|[Figure 259]|
|---|

ocean train water

flower table

grass top sky

leaf decoration

sea coast

row generator hill park

petal rose pink

beach

crash storm

autumn

0.29

0.25

0.32

history phone smartphone iphone cell

tree flower

dog pool

|[Figure 260]|
|---|

|[Figure 261]|
|---|

|[Figure 262]|
|---|

palm footage

top water

city building foreground street bridge view

retriever swimming

apple android

laying float sits paw

picture technology

device

0.54

0.44

0.26

roll plate bread table sesame

plate corn

glass beer table

|[Figure 263]|
|---|

|[Figure 264]|
|---|

|[Figure 265]|
|---|

cob slice

pint drink wooden top apple

top piece

bun pastry

bowl onion

cheese top pizza

bunch leaf

bar thornbury

0.32

0.23

0.22

plate

audi car

cat table

|[Figure 266]|
|---|

|[Figure 267]|
|---|

|[Figure 268]|
|---|

fry sandwich

road sport

camera eye top

pork hamburger meat food

highway track

collar floor sits window wall

coupe driving

chicken side bun

blue silver

0.35

0.33

0.17

cage bird

car silver show sport

house step stair door plant

|[Figure 269]|
|---|

|[Figure 270]|
|---|

|[Figure 271]|
|---|

window top sits

field grass lot display crowd event

perch head feeder wire bar

home porch

building entrance tree

0.40

0.21

0.21

diamond plate

cream face box

meat beef

|[Figure 272]|
|---|

|[Figure 273]|
|---|

|[Figure 274]|
|---|

metal surface pattern

sale display

tube bottle

store market package plastic counter tray

steel photo

gel skin

eye moist hydra

sheet texture

floor

0.38

0.34

0.15

chocolate cup

lion grass camera rock ground tree zoo

wall

|[Figure 275]|
|---|

|[Figure 276]|
|---|

|[Figure 277]|
|---|

board blackboard

coffee drink table

cat drawing chalkboard bunch lot whiteboard art

mug bowl food milk

distance field wild

top

0.16

0.53

0.28

desert ford

bmw suv driveway

flower sunflower

|[Figure 278]|
|---|

|[Figure 279]|
|---|

|[Figure 280]|
|---|

car sand dune

yellow garden middle

car house street

hyundai

center plant

focus blue road

road lot

leaf petal

side hood

orange

hatch

0.36

0.26

0.18

bear panda

bmw

car building

|[Figure 281]|
|---|

|[Figure 282]|
|---|

|[Figure 283]|
|---|

car silver

rock zoo ground

masera ferrari sport street lot house roadster parking

road

tree sport

cub something

coupe view end park

top enclosure boulder

0.17

0.42

0.32

rig middle oil platform

cat eye top

dog bed

|[Figure 284]|
|---|

|[Figure 285]|
|---|

|[Figure 286]|
|---|

camera top collar eye head pink blanket face

table bed box

ocean drilling

water

camera sits floor laying

offshore sea gulf

0.24

0.29

0.36

group grass lemur

runway air plane

lizard

|[Figure 287]|
|---|

|[Figure 288]|
|---|

|[Figure 289]|
|---|

tree top sits

zoo raccoon

force aircraft

branch gecko camera

bunch top ground fence pile

sky ground landing

iguana wood photo

top propeller

0.33

0.29

0.24

track race car

truck

ship ocean water

|[Figure 290]|
|---|

|[Figure 291]|
|---|

|[Figure 292]|
|---|

fire building

ferrari racetrack

roof engine house lot

destroyer

warship battleship

corvette racing sport road gts

boat

street station

sea navy

side

sailing

0.26

0.17

0.21

gmc suv

table glass

log wood

|[Figure 293]|
|---|

|[Figure 294]|
|---|

|[Figure 295]|
|---|

bottle top cup

tree yukon wheel street

pile firewood

stack top lot

button

beer drink

road cadillac

tree bunch

coin badge

srx den

ground

0.19

0.17

0.29

silver

skylight

flower pink leaf

|[Figure 296]|
|---|

|[Figure 297]|
|---|

|[Figure 298]|
|---|

car display cadillac

room window

roof construction

garden plant petal

show auto

house building

vehicle sedan luxury

bloom bush group middle

ceiling home wall

srx

0.33

0.27

0.35

market garlic display

owl grass eye camera

dog grass field

|[Figure 299]|
|---|

|[Figure 300]|
|---|

|[Figure 301]|
|---|

pile bunch

air tongue mouth

top tree field bird

farmer store sale top head

park

ball lawn

sits ground

rottweiler

0.22

0.16

0.32

liquid orange candle blue

garden

plate cupcake frosting pink

|[Figure 302]|
|---|

|[Figure 303]|
|---|

|[Figure 304]|
|---|

door storage

metal

top glass paint

roof shed grass

top chocolate

cream

tray icing cake

cup yellow

side yard

color

greenhouse

0.33

0.31

0.25

engine train display

cherokee street

garden lettuce

|[Figure 305]|
|---|

|[Figure 306]|
|---|

|[Figure 307]|
|---|

jeep sidewalk

vegetable

leaf wooden

track locomotive

house car suv

plant ground bunch wood bed

steam sits

driveway

side building

parking building

museum

0.15

0.15

0.32

glass bead

plate cavia food

box

|[Figure 308]|
|---|

|[Figure 309]|
|---|

|[Figure 310]|
|---|

post building mailbox

gold

button round object

cream pancake

street side top brick city corner

sauce top

jewelry table sphere stone

berry chocolate

ice

0.26

0.33

0.06

tree

car sport

sea

|[Figure 311]|
|---|

|[Figure 312]|
|---|

|[Figure 313]|
|---|

baby branch

water reef fish

lot supercar mclaren parking

sloth rainforest forest leaf hang sits hanging

ocean

algae sponge

display building

scorpionfish worm glow

race motorcycle

0.16

0.79

0.55

dish pan

groundhog rock animal

beatles

|[Figure 314]|
|---|

|[Figure 315]|
|---|

|[Figure 316]|
|---|

cover poster

top oven cheese food casserole

top photo

band cartoon

postcard

art album book volume john

picture tree hog

sauce tomato lasagna

photograph

0.19

0.31

0.36

wood wall piece

bird wall

fence gate yard

|[Figure 317]|
|---|

|[Figure 318]|
|---|

|[Figure 319]|
|---|

top head blue jay hang sits beak rack

hole shelf

backyard

area garden

board wooden

metal

pair plank

tree house

plywood

iron

0.31

0.22

0.19

candy bowl

chicken turkey

field grass middle water plant tree

|[Figure 320]|
|---|

|[Figure 321]|
|---|

|[Figure 322]|
|---|

chocolate plate bean table

pan oven

top

tray roast

lot top pile

fence area reed

food sheet metal

bunch

rice

0.39

0.15

0.41

rack wire

field truck

flower pink bloom

|[Figure 323]|
|---|

|[Figure 324]|
|---|

|[Figure 325]|
|---|

cooling counter top

home

land vehicle

leaf plant bush

bread pretzel basket

rover house

tree garden petal bud

grass road hill

roll bun

0.21

0.10

0.32

tarmac airport

mountain plant tree power tower view water hill area top

plant

|[Figure 326]|
|---|

|[Figure 327]|
|---|

|[Figure 328]|
|---|

flower garden ground

plane runway

top jet passenger

leaf grass

pot purple spot hosta

terminal landing ground

0.39

0.17

0.43

pen

lot

battleship

|[Figure 329]|
|---|

|[Figure 330]|
|---|

|[Figure 331]|
|---|

tip metal silver

car porsche

water ship boat

turbo parking coupe hood

ballpoint fountain

warship

harbor destroyer

handle brush roller barrel

sale vehicle

dock ocean carrier

silver

0.39

0.15

0.29

car grass

field truck

case violin

|[Figure 332]|
|---|

|[Figure 333]|
|---|

|[Figure 334]|
|---|

jaguar vintage

home

display

land vehicle

museum glass cello bow

field sport show

rover house

grass road hill

string top wall

display lot park

0.27

0.16

0.28

motorcycle garage

board

pinball

|[Figure 335]|
|---|

|[Figure 336]|
|---|

|[Figure 337]|
|---|

beef steak meat piece

machine game room

wall floor

building room

arcade display

tray cut table slice top

blue ground

table top museum car

top bike

0.29

0.43

0.30

car garage

leaf

ocean sun bird

|[Figure 338]|
|---|

|[Figure 339]|
|---|

|[Figure 340]|
|---|

plant vein view

firebird shop hood

water sky sea

close shot

engine workshop

macro surface

airplane

pontiac chevrolet corvette

fly setting

top spot

boat

0.22

0.29

0.24

ferrari

dolphin water ocean group

lighthouse tower

|[Figure 341]|
|---|

|[Figure 342]|
|---|

|[Figure 343]|
|---|

car show

sky brick

display sport auto supercar floor motor lot

couple swimming

top cloudy

sea pair

building house middle light

whale pool

0.32

0.53

0.07

fruit tree

fiat car

paper plate food wrap table

|[Figure 344]|
|---|

|[Figure 345]|
|---|

|[Figure 346]|
|---|

branch

aba alfa

leaf plant

view maroon hatch

photo

tortilla taco

pod flower

studio sport fiorin

top burrito basket

garden bud

0.24

0.19

0.32

vegetable bean corn

car

leaf plant

|[Figure 347]|
|---|

|[Figure 348]|
|---|

|[Figure 349]|
|---|

ferrari road track

tree foliage garden

top towel

game race

squash plate

jungle

need screenshot

bush forest

tray stove

driving gts

branch stem

zucchini

0.40

0.37

0.27

fence

dune sand

bunch grape vine

|[Figure 350]|
|---|

|[Figure 351]|
|---|

|[Figure 352]|
|---|

dog wire tree

desert sky

cluster leaf tree

top

cloud sahara sunset

chain cage area gate

branch wine plant close

top

namibia mountain

dirt

0.16

0.27

0.45

rock water river

road

flower

|[Figure 353]|
|---|

|[Figure 354]|
|---|

|[Figure 355]|
|---|

tortoise turtle grass

leaf pink

peony

tree forest

vintage stem print

dirt middle ground

middle mountain

stream lake creek

shell side gravel

picture

tree plant

0.18

0.23

0.33

sink kitchen

flower table top display

fighter jet plane

|[Figure 356]|
|---|

|[Figure 357]|
|---|

|[Figure 358]|
|---|

brick stove

top airplane building

wall room wood

vase pot vas

ground sits museum tarmac

top fire

plant bunch center

oven

### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: A Visual Language Model for Few-Shot Learning. In NeurIPS, 2022. 1, 2, 13
- [2] Jacob Andreas and Dan Klein. Reasoning About Pragmatics With Neural Listeners and Speakers. In EMNLP, 2016. 2
- [3] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. OpenFlamingo: An Open-Source Framework for Training Large Autoregressive Vision-Language Models. In arXiv:2308.01390,

2023. 1, 5, 6, 9, 12, 13

- [4] Kobus Barnard and David Forsyth. Learning the Semantics of Words and Pictures. In ICCV, 2001. 14
- [5] Kobus Barnard, Pinar Duygulu, David Forsyth, Nando De Freitas, David M Blei, and Michael I Jordan. Matching Words and Pictures. In JMLR, 2003. 14
- [6] Abhijit Bendale and Terrance Boult. Towards Open World Recognition. In CVPR, 2015. 2
- [7] Rodrigo Benenson and Vittorio Ferrari. From Colouring-in to Pointillism: Revisiting Semantic Segmentation Supervision. arXiv:2210.14142, 2022. 4
- [8] Steven Bird, Ewan Klein, and Edward Loper. Natural Language Processing With Python: Analyzing Text With the Natural Language Toolkit. O’Reilly Media, Inc., 2009. 12
- [9] David M Blei and Michael I Jordan. Modeling Annotated Data. In ACM SIGIR, 2003. 2
- [10] Damian Borth, Rongrong Ji, Tao Chen, Thomas Breuel, and Shih-Fu Chang. Large-Scale Visual Sentiment Ontology and Detectors Using Adjective Noun Pairs. In ACM MM,

2013. 2

- [11] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language Models Are Few-Shot Learners. In NeurIPS,

2020. 1, 2, 3

- [12] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12M: Pushing Web-Scale Image-Text Pre-training to Recognize Long-Tail Visual Concepts. In CVPR, 2021. 14
- [13] Minmin Chen, Alice Zheng, and Kilian Weinberger. Fast Image Tagging. In ICML, 2013. 14
- [14] Ting Chen, Saurabh Saxena, Lala Li, David J Fleet, and Geoffrey Hinton. Pix2Seq: A Language Modeling Framework for Object Detection. In ICLR, 2022. 2
- [15] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft COCO Captions: Data Collection and Evaluation Server. arXiv:1504.00325, 2015. 4
- [16] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An Open-Source

- Chatbot Impressing GPT-4 with 90% ChatGPT Quality. https://lmsys.org/blog/2023-03-30-vicuna, 2023. 6, 12
- [17] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling InstructionFinetuned Language Models. arXiv:2210.11416, 2022. 6, 12
- [18] Elijah Cole, Oisin Mac Aodha, Titouan Lorieul, Pietro Perona, Dan Morris, and Nebojsa Jojic. Multi-Label Learning From Single Positive Labels. In CVPR, 2021. 14
- [19] Alessandro Conti, Enrico Fini, Massimiliano Mancini, Paolo Rota, Yiming Wang, and Elisa Ricci. VocabularyFree Image Classification. In NeurIPS, 2023. 1, 2, 5, 6, 8, 9, 12, 14
- [20] Thomas Cover and Peter Hart. Nearest Neighbor Pattern Classification. IEEE Trans. Inf. Theory, 1967. 14
- [21] Claudio Cusano, Gianluigi Ciocca, and Raimondo Schettini. Image Annotation Using SVM. In SPIE, 2003. 14
- [22] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. InstructBLIP: Towards GeneralPurpose Vision-Language Models With Instruction Tuning. In NeurIPS, 2023. 2, 5, 6, 9, 12, 13
- [23] Marie-Catherine De Marneffe, Bill MacCartney, Christopher D Manning, et al. Generating Typed Dependency Parses From Phrase Structure Parses. In LREC, 2006. 2
- [24] Arthur P Dempster, Nan M Laird, and Donald B Rubin. Maximum Likelihood From Incomplete Data via the EM Algorithm. J. R. Stat. Soc. Ser. B (Methodol.), 1977. 14
- [25] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A Large-Scale Hierarchical Image Database. In CVPR, 2009. 1, 3
- [26] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In NACCLHLT, 2019. 1
- [27] Shizhe Diao, Wangchunshu Zhou, Xinsong Zhang, and Jiawei Wang. Write and Paint: Generative Vision-Language Models Are Unified Modal Learners. In ICLR, 2022. 3
- [28] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale. In ICLR,

- 2021. 1, 2, 3, 4

[29] Yu Du, Fangyun Wei, Zihe Zhang, Miaojing Shi, Yue Gao, and Guoqi Li. Learning to Prompt for Open-Vocabulary Object Detection With Vision-Language Model. In CVPR,

- 2022. 2

- [30] Thibaut Durand, Nazanin Mehrasa, and Greg Mori. Learning a Deep Convnet for Multi-Label Classification With Partial Labels. In CVPR, 2019. 14
- [31] Pinar Duygulu, Kobus Barnard, Joao FG de Freitas, and David A Forsyth. Object Recognition as Machine Translation: Learning a Lexicon for a Fixed Image Vocabulary. In ECCV, 2002. 14

- [32] Shao Lei Feng, Raghavan Manmatha, and Victor Lavrenko. Multiple Bernoulli Relevance Models for Image and Video Annotation. In CVPR, 2004. 14
- [33] Rob Fergus, Yair Weiss, and Antonio Torralba. Semisupervised Learning in Gigantic Image Collections. In NeurIPS, 2009. 14
- [34] Andrea Frome, Greg S Corrado, Jon Shlens, Samy Bengio, Jeff Dean, Marc’Aurelio Ranzato, and Tomas Mikolov. DeViSE: A Deep Visual-Semantic Embedding Model. In NeurIPS, 2013. 2
- [35] Zihao Fu, Wai Lam, Anthony Man-Cho So, and Bei Shi. A Theoretical Analysis of the Repetition Problem in Text Generation. In AAAI, 2021. 2, 4
- [36] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Scaling Open-Vocabulary Image Segmentation With ImageLevel Labels. In ECCV, 2022. 2
- [37] Yunchao Gong, Yangqing Jia, Thomas Leung, Alexander Toshev, and Sergey Ioffe. Deep Convolutional Ranking for Multilabel Image Annotation. In ICLR, 2014. 14
- [38] Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui. Open-Vocabulary Object Detection via Vision and Language Knowledge Distillation. In ICLR, 2022. 2
- [39] Matthieu Guillaumin, Thomas Mensink, Jakob Verbeek, and Cordelia Schmid. TagProp: Discriminative Metric Learning in Nearest Neighbor Models for Image AutoAnnotation. In ICCV, 2009. 14
- [40] Tanmay Gupta, Arash Vahdat, Gal Chechik, Xiaodong Yang, Jan Kautz, and Derek Hoiem. Contrastive Learning for Weakly Supervised Phrase Grounding. In ECCV, 2020. 2
- [41] Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality Reduction by Learning an Invariant Mapping. In CVPR, 2006. 2
- [42] Chi Han, Hengzhi Pei, Xinya Du, and Heng Ji. Zero-Shot Classification by Logical Reasoning on Natural Language Explanations. In ACL, 2023. 2
- [43] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. In CVPR,

2016. 1, 3

- [44] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked Autoencoders Are Scalable Vision Learners. In CVPR, 2022. 6
- [45] Lisa Anne Hendricks, Zeynep Akata, Marcus Rohrbach, Jeff Donahue, Bernt Schiele, and Trevor Darrell. Generating Visual Explanations. In ECCV, 2016. 2
- [46] Lisa Anne Hendricks, Ronghang Hu, Trevor Darrell, and Zeynep Akata. Grounding Visual Explanations. In ECCV,

2018. 2

- [47] Tomer Hertz, Aharon Bar-Hillel, and Daphna Weinshall. Learning Distance Functions for Image Retrieval. In CVPR,

2004. 14

- [48] Sepp Hochreiter and J¨urgen Schmidhuber. Long ShortTerm Memory. In Neural Computation, 1997. 2
- [49] Thomas Hofmann. Unsupervised Learning by Probabilistic Latent Semantic Analysis. Machine Learning, 2001. 2, 14
- [50] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The Curious Case of Neural Text Degeneration. In ICLR, 2020. 2

- [51] Xinyu Huang, Youcai Zhang, Jinyu Ma, Weiwei Tian, Rui Feng, Yuejie Zhang, Yaqian Li, Yandong Guo, and Lei Zhang. Tag2Text: Guiding Vision-Language Model via Image Tagging. In ICLR, 2024. 14
- [52] Jiwoon Jeon, Victor Lavrenko, and Raghavan Manmatha. Automatic Image Annotation and Retrieval Using CrossMedia Relevance Models. In ACM SIGIR, 2003. 14
- [53] Yangqing Jia, Mathieu Salzmann, and Trevor Darrell. Learning Cross-Modality Similarity for Multinomial Data. In ICCV, 2011. 14
- [54] Thorsten Joachims. Optimizing Search Engines Using Clickthrough Data. In ACM SIGKDD, 2002. 14
- [55] Justin Johnson, Andrej Karpathy, and Li Fei-Fei. DenseCap: Fully Convolutional Localization Networks for Dense Captioning. In CVPR, 2016. 2
- [56] Andrej Karpathy and Li Fei-Fei. Deep Visual-Semantic Alignments for Generating Image Descriptions. In CVPR,

2015. 2

- [57] Andrej Karpathy, Armand Joulin, and Li F Fei-Fei. Deep Fragment Embeddings for Bidirectional Image Sentence Mapping. In NeurIPS, 2014. 2
- [58] Nitish Shirish Keskar, Bryan McCann, Lav R Varshney, Caiming Xiong, and Richard Socher. CTRL: A Conditional Transformer Language Model for Controllable Generation. arXiv:1909.05858, 2019. 7
- [59] Ryan Kiros, Ruslan Salakhutdinov, and Rich Zemel. Multimodal Neural Language Models. In ICML, 2014. 2
- [60] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet Classification With Deep Convolutional Neural Networks. In NeurIPS, 2012. 1
- [61] Weicheng Kuo, Yin Cui, Xiuye Gu, AJ Piergiovanni, and Anelia Angelova. F-VLM: Open-Vocabulary Object Detection Upon Frozen Vision and Language Models. In ICLR,

2023. 2

- [62] Victor Lavrenko, Raghavan Manmatha, and Jiwoon Jeon. A Model for Learning the Semantics of Pictures. In NeurIPS,

2003. 14

- [63] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep Learning. Nature, 2015. 14
- [64] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. In ICML, 2022. 2, 4
- [65] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping Language-Image Pre-training With Frozen Image Encoders and Large Language Models. In ICML, 2023. 2, 5, 6, 9, 12, 13
- [66] Chin-Yew Lin. ROUGE: A Package for Automatic Evaluation of Summaries. In ACL, 2004. 5
- [67] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft COCO: Common Objects in Context. In ECCV, 2014. 4
- [68] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved Baselines with Visual Instruction Tuning. arXiv:2310.03744, 2023. 2, 5, 6, 9, 12, 13

- [69] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning. In NeurIPS, 2023. 2, 5, 6, 9, 12, 13
- [70] Peter J Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz Kaiser, and Noam Shazeer. Generating Wikipedia by Summarizing Long Sequences. In ICLR,

2018. 3

- [71] Shilong Liu, Lei Zhang, Xiao Yang, Hang Su, and Jun Zhu. Query2Label: A Simple Transformer Way to Multi-Label Classification. arXiv:2107.10834, 2021. 14
- [72] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A Convnet for the 2020s. In CVPR, 2022. 1
- [73] Ilya Loshchilov and Frank Hutter. SGDR: Stochastic Gradient Descent With Warm Restarts. In ICLR, 2017. 4
- [74] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In ICLR, 2019. 4
- [75] Ziqiao Ma, Jiayi Pan, and Joyce Chai. World-To-Words: Grounded Open Vocabulary Acquisition Through Fast Mapping in Vision-Language Models. In ACL, 2023. 2
- [76] Ameesh Makadia, Vladimir Pavlovic, and Sanjiv Kumar. A New Baseline for Image Annotation. In ECCV, 2008. 14
- [77] Chengzhi Mao, Revant Teotia, Amrutha Sundar, Sachit Menon, Junfeng Yang, Xin Wang, and Carl Vondrick. Doubly Right Object Recognition: A Why Prompt for Visual Rationales. In CVPR, 2023. 2
- [78] Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, and Alan L Yuille. Explain Images With Multimodal Recurrent Neural Networks. arXiv:1410.1090, 2014. 2
- [79] Marjo Markkula and Eero Sormunen. End-User Searching Challenges Indexing Practices in the Digital Newspaper Photo Archive. Information Retrieval, 2000. 14
- [80] Sachit Menon and Carl Vondrick. Visual Classification via Description From Large Language Models. In ICLR, 2023. 2
- [81] Jack Merullo, Louis Castricato, Carsten Eickhoff, and Ellie Pavlick. Linearly Mapping From Image to Text Space. In ICLR, 2023. 2
- [82] Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, et al. Simple Open-Vocabulary Object Detection. In ECCV, 2022. 2
- [83] Matthias Minderer, Alexey Gritsenko, and Neil Houlsby. Scaling Open-Vocabulary Object Detection. In NeurIPS,

2023. 2

- [84] Florent Monay and Daniel Gatica-Perez. PLSA-Based Image Auto-Annotation: Constraining the Latent Space. In ACM MM, 2004. 14
- [85] Yasuhide Mori, Hironobu Takahashi, and Ryuichi Oka. Image-To-Word Transformation Based on Dividing and Vector Quantizing Images With Words. In First International Workshop on Multimedia Intelligent Storage and Retrieval Management, 1999. 14
- [86] OpenAI. GPT-4V(ision) System Card. OpenAI Blog, 2023. 5, 8, 9, 13, 14

- [87] OpenAI. GPT-4 Technical Report. arXiv:2303.08774,

2023. 1, 2, 3

- [88] Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2Text: Describing Images Using 1 Million Captioned Photographs. In NeurIPS, 2011. 4
- [89] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. BLEU: A Method for Automatic Evaluation of Machine Translation. In ACL, 2002. 5
- [90] Ofir Press and Lior Wolf. Using the Output Embedding to Improve Language Models. EACL, 2017. 3
- [91] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving Language Understanding by Generative Pre-training. OpenAI Blog, 2018. 1, 2, 3
- [92] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language Models Are Unsupervised Multitask Learners. OpenAI Blog, 2019. 1, 2, 3
- [93] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning Transferable Visual Models From Natural Language Supervision. In ICML, 2021. 1, 2, 3, 4, 5, 6, 8, 9, 11, 12
- [94] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the Limits of Transfer Learning With a Unified Text-To-Text Transformer. In JMLR, 2020. 3
- [95] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-Shot Text-To-Image Generation. In ICML,

2021. 3

- [96] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence Embeddings Using Siamese Bert-Networks. In EMNLP, 2019. 5
- [97] Paul Resnick and Hal R Varian. Recommender Systems. ACM Communications, 1997. 2
- [98] Tal Ridnik, Gilad Sharir, Avi Ben-Cohen, Emanuel BenBaruch, and Asaf Noy. ML-Decoder: Scalable and Versatile Classification Head. In WACV, 2023. 14
- [99] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. In IJCV, 2015. 11, 12
- [100] Khalid Saifullah, Yuxin Wen, Jonas Geiping, Micah Goldblum, and Tom Goldstein. Seeing in Words: Learning to Classify Through Language Bottlenecks. In ICLR Track on Tiny Papers, 2023. 2
- [101] Florian Schroff, Antonio Criminisi, and Andrew Zisserman. Harvesting Image Databases From the Web. TPAMI,

2010. 14

- [102] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. LAION400M: Open Dataset of Clip-Filtered 400 Million ImageText Pairs. In NeurIPS Workshop on Data-Centric AI, 2021. 4

- [103] Sarah Schwettmann, Neil Chowdhury, and Antonio Torralba. Multimodal Neurons in Pretrained Text-Only Transformers. In ICCV Workshop on CLVL, 2023. 2
- [104] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual Captions: A Cleaned, Hypernymed, Image Alt-Text Dataset for Automatic Image Captioning. In ACL, 2018. 4, 5, 14
- [105] Alex Sherstinsky. Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) Network. Physica D - Nonlinear Phenomena, 2020. 2
- [106] Karen Simonyan and Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. In ICLR, 2015. 1
- [107] Richard Socher and Li Fei-Fei. Connecting Modalities: Semi-supervised Segmentation and Annotation of Images Using Unaligned Text Corpora. In CVPR, 2010. 14
- [108] Richard Socher, Andrej Karpathy, Quoc V Le, Christopher D Manning, and Andrew Y Ng. Grounded Compositional Semantics for Finding and Describing Images With Sentences. In TACL, 2014. 2
- [109] Kihyuk Sohn. Improved Deep Metric Learning With MultiClass N-Pair Loss Objective. In NeurIPS, 2016. 2
- [110] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going Deeper With Convolutions. In CVPR, 2015. 1
- [111] MosaicML NLP Team. Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs. https://www.mosaicml.com/blog/mpt-7b, 2023. 6, 9, 12
- [112] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971, 2023. 1, 2, 3, 6, 8, 12
- [113] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. LLaMA 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288, 2023. 2, 3, 6, 7, 8, 12
- [114] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. In NeurIPS, 2017. 1, 2, 3
- [115] Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and Tell: A Neural Image Caption Generator. In CVPR, 2015. 2
- [116] Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang. GIT: A Generative Image-To-Text Transformer for Vision and Language. In TMLR, 2022. 3
- [117] Thomas Wang, Adam Roberts, Daniel Hesslow, Teven Le Scao, Hyung Won Chung, Iz Beltagy, Julien Launay, and Colin Raffel. What Language Model Architecture and Pretraining Objective Works Best for Zero-Shot Generalization? In ICML, 2022.
- [118] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. SimVLM: Simple Visual Lan-

guage Model Pretraining With Weak Supervision. In ICLR,

2022. 3

- [119] Jason Weston, Samy Bengio, and Nicolas Usunier. Wsabie: Scaling up to Large Vocabulary Image Annotation. Google,

2011. 2

- [120] Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying CLIP Data. In ICLR, 2024. 1
- [121] Jin Xu, Xiaojiang Liu, Jianhao Yan, Deng Cai, Huayang Li, and Jian Li. Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation. In NeurIPS, 2022. 2, 4
- [122] Yue Yang, Artemis Panagopoulou, Shenghao Zhou, Daniel Jin, Chris Callison-Burch, and Mark Yatskar. Language in a Bottle: Language Model Guided Concept Bottlenecks for Interpretable Image Classification. In CVPR, 2023. 2
- [123] Alireza Zareian, Kevin Dela Rosa, Derek Hao Hu, and Shih-Fu Chang. Open-Vocabulary Object Detection Using Captions. In CVPR, 2021. 2
- [124] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. BERTScore: Evaluating Text Generation With Bert. In ICLR, 2020. 5, 12
- [125] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize Anything: A Strong Image Tagging Model. arXiv:2306.03514, 2023. 14

