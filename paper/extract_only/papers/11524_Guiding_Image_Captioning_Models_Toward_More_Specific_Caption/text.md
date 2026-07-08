### Guiding image captioning models toward more specific captions

Simon Kornblith1 Lala Li1 Zirui Wang2* Thao Nguyen3† 1Google DeepMind 2Apple AI/ML 3University of Washington

## arXiv:2307.16686v1[cs.CV]31Jul2023

#### Abstract

Image captioning is conventionally formulated as the task of generating captions for images that match the distribution of reference image-caption pairs. However, reference captions in standard captioning datasets are short and may not uniquely identify the images they describe. These problems are further exacerbated when models are trained directly on image-alt text pairs collected from the internet. In this work, we show that it is possible to generate more specific captions with minimal changes to the training process. We implement classifier-free guidance [14] for an autoregressive captioning model by fine-tuning it to estimate both conditional and unconditional distributions over captions. The guidance scale applied at decoding controls a trade-off between maximizing p(caption|image) and p(image|caption). Compared to standard greedy decoding, decoding with a guidance scale of 2 substantially improves reference-free metrics such as CLIPScore (0.808 vs. 0.775) and caption→image retrieval performance in the CLIP embedding space (recall@1 44.6% vs. 26.5%), but worsens standard reference-based captioning metrics (e.g., CIDEr 78.6 vs 126.1). We further explore the use of language models to guide the decoding process, obtaining small improvements over the Pareto frontier of referencefree vs. reference-based captioning metrics that arises from classifier-free guidance, and substantially improving the quality of captions generated from a model trained only on minimally curated web data.

#### 1. Introduction

Image captioning is both a difficult task for computer vision systems to perform and a difficult task to evaluate. Although automated captioning metrics rank the best captioning systems higher than humans, human raters still show a strong preference for human-generated captions [20], suggesting shortcomings in both captioning models and metrics. One shortcoming relates to the lack of specificity in generated captions. Conventional maximum likelihood-

*Work performed while at Google. †Work performed as a student researcher at Google.

[Figure 1]

50

CLIPcaptionimageR@1

45

40

35

30

25 50 75 100 125 CIDEr

γ=1.0 a man riding a blue motorcycle on a dirt road

- γ=1.5 a man riding a blue motorcycle on a straw bale
- γ=2.0 rider on blue suzuki motorcycle near straw bales
- γ=3.0 racer on blue suzuki motorcycle leaning up against straw bales

= 1.0 = 1.2 = 1.5

- = 2.0

- = 3.0

- = 4.0

GT A person riding a baby blue motorcycle near haystacks

Figure 1. Using classifier-free guidance (γ > 1) results in more specific captions that are farther from the reference distribution. Left: Example of captions generated at different guidance scales for a single image. Right: Caption→image recall@1 with CLIP ViT-B/32 vs. CIDEr score, for captions generated with different guidance scales γ on MS-COCO. Higher scales improve retrieval accuracy at the expense of CIDEr.

based image captioning models attempt to generate captions such that the p(caption|image) is high. However, captions from the ground truth distribution are often nonspecific, e.g., human annotators will usually describe a German Shepard only as a dog. Moreover, previous work has emphasized “reference-based” captioning metrics that measure the match between generated captions and humanprovided ground truth captions [28, 23, 41]. These metrics intrinsically penalize captions that are more specific than ground truth.

In this work, we explore strategies to guide image captioning models to produce more specific captions by modifying the decoding distribution, and explore the tradeoffs in captioning metrics that result. We first investigate the application of classifier-free guidance (CFG) [14] to image captioning with autoregressive models. Classifierfree guidance increases p(image|caption) at the expense of p(caption|image). Although CFG hurts reference-based image captioning metrics such as BLEU [28], ROUGE [23], and CIDEr [41], it improves “reference-free” metrics that measure captions’ specificity via the similarity between the

image and the generated caption in the embedding space of image-text models [13] or caption→image retrieval performance. Qualitatively, we find that captions generated with CFG are more specific than both the ground truth captions and captions generated without CFG, but they are less grammatical, particularly at high CFG scales.

Beyond classifier-free guidance, we experiment with guiding image captioning models using the probability distribution obtained from a few shot-prompted language model (LM). We find that using a language model to guide a captioning model trained on MS-COCO [24] with descriptive manually written captions can allow it to achieve slightly better trade-offs between reference-free vs. reference-based captioning metrics than those observed with CFG. LM guidance also substantially improves the captions produced by a model trained exclusively on minimally curated web data. Although this model achieves a CIDEr score of only 21.8 without guidance, this CIDEr score improves to 57.4 when guided by a language model prompted with 20 captions from the MS-COCO training set.

In summary, our contributions are as follows:

- • We propose two strategies to guide image captioning models to produce more specific captions: classifier-free guidance and language model guidance.
- • We demonstrate that classifier-free guidance yields captions that are closer to the corresponding image in the embedding space of image-text models, but are farther from human-provided reference captions.
- • We show that language model guidance can alter caption styles, substantially improving captions produced by a model trained only on minimal curated web data and marginally improving the trade-off between captioning metrics observed with classifier-free guidance.

#### 2. Related work

Measuring specificity of captions. Early work using neural networks for image captioning found that models have a propensity to regurgitate captions from their training data, and as a result, the generated captions are not descriptive enough to uniquely identify images [42, 11]. To address this shortcoming, Lindh et al. [25] proposed to use caption→image recall with an image retrieval model to examine whether images can be retrieved from generated captions, and further attempt to differentiate through this retrieval process to train a captioning model. Their approach marginally improves retrieval accuracy, but worsens reference-based captioning metrics. More recent work has adopted approaches to evaluate the specificity of captions based on the CLIP image-text model [30]. Hessel et al. [13] propose CLIPScore, an image captioning metric based on the cosine similarity between CLIP embeddings of the image and the generated caption. Kasai et al. [20] re-

port that CLIPScore-based metrics align better with human judgments compared to reference-based captioning metrics.

Improving specificity of captions. Recent work has attempted to directly optimize CLIP-based losses that measure the similarity of captions with corresponding images in the CLIP embedding space, either on their own or jointly with CIDEr scores. Work that trains captioning models has generally approached this problem using reinforcement learning, and finds that adding these losses worsens standard reference-based captioning metrics but improves similarity and retrieval in the CLIP embedding space [16, 6, 50], similar to our observations regarding CFG. Wen et al. [43] attempt to generate prompts for text-to-image generative models that correspond to specific images without a captioning model, by directly optimizing the similarity between the text and image in the CLIP embedding space using a gradient-based discrete optimization procedure, but the resulting text is not grammatical.

Other work has attempted to generate more descriptive captions through different means. Dense captioning [45] aims to detect and caption all objects in an image, but concatenating all of these captions leads to long and unnatural captions, whereas CFG produces single-sentence captions. The Localized Narratives dataset [29] contains visually grounded captions for MS-COCO images collected through voice annotation. These captions are substantially more descriptive than the captions in the MS-COCO dataset and can be used for model training. Concurrent with our work, IC3 [5] proposes to generate multiple captions with an off-the-shelf captioning model and combine them using a language model. The resulting captions are longer, but achieve greater caption→image recall.

Captioning from uncurated data. In Section 4.2, we explore the use of LM guidance for captioning with access to uncurated image-text data from the web and a small number of captions but not images from the target distribution. This setting, which does not rely on aligned images and captions from the target distribution, is often referred to as “zero-shot” captioning, and previous work has pursued a number of alternative approaches. Flamingo [3] and CM3 [1] perform zero-shot captioning by pretraining on interleaved image/text data. MAGIC [38] and ZeroCap [40] generate captions using a combination of guidance from CLIP and a large language model. Other recent work adapts CLIP to perform captioning by training a text decoder using only captions, with no corresponding images [27, 22].

Classifier-free guidance. CFG is widely used in diffusion-based and autoregressive text-to-image models [26, 32, 34, 33, 12, 47]. Because of the popularity of the combination of CFG and diffusion, previous work that has performed image captioning with diffusion models has also examined the use of CFG. This work finds either no benefit to using CFG [44] or a small and inconsistent ben-

efit that appears to vary with minor changes in training settings [51]. However, these studies do not seek to generate more specific captions, and thus measure only referencebased captioning metrics, which we likewise find do not benefit from CFG. Concurrently with out work, [35] propose to use classifier-free guidance to improve prompt following in large language models.

#### 3. Methods 3.1. Classifier-free guidance for image captioning

Let x be an image caption and y be the corresponding image. A standard captioning model aims to model the likelihood p(x|y), factorized autoregressively in terms of the probability of each token given previous tokens

p(x|y) = p(xn|xn−1,...x1,y)...p(x1|y). (1) The network is trained so that its output distribution qθ(xn|xn−1,...x1,y) def= softmax(fθ(xn−1,...x1,y)) approximates p(xn|xn−1,...x1,y). At inference time, one typically uses beam search or greedy decoding to produce a caption that has a particularly high probability. In this work, we use greedy decoding because it is the more common choice and it is also simpler to implement.

Classifier-free guidance (CFG) [14] aims to generate outputs that maximize or otherwise achieve high values of

γ

p(x|y) p(x)

lθ,γ(x,y) def= p(x)

∝ p(x)p(y|x)γ (2)

where proportionality holds because p(x|y)/p(x) = p(y|x)/p(y) and p(y) is fixed. The parameter γ is called the guidance scale and controls the trade-off between maximization of p(x|y) and p(y|x). When γ = 1, lθ,γ(x,y) = p(x|y) and guidance has no effect. Setting γ > 1 inflates the probability of the image given the caption p(y|x) relative to the unconditional probability of the caption p(x).

Ho and Salimans [14] originally proposed CFG in the context of diffusion models, which estimate the score functions ∇log p(x|y) and ∇log p(x). Although lθ,γ(x,y) factorizes autoregressively, it is not a normalized probability distribution, so it is not entirely clear how one should sample tokens when performing autoregressive generation. Crowson [8] suggested to sample from

q˜θ,γ(xn|xn−1,...,x1,y) def= softmax(fθ(xn−1,...,x1,0)

+ γ(fθ(xn−1,...,x1,y) − fθ(xn−1,...,x1,0))), (3) where fθ(xn−1,...,x1,0) are logits generated by the model without conditioning, usually by passing zeros in place of the conditioning information. This formulation has been successfully applied in autoregressive image models [12, 47]. In our experiments, we adopt this formulation as well, but since we decode greedily, i.e., at each step we take the token that maximizes q˜θ,γ(xn|xn−1,...,x1,y) and thus lθ,γ(x,y), any form of normalization of the per-step

sampling distribution would produce the same captions. We provide pseudocode in Appendix A.1.

##### 3.2. Language model guidance

Inspired by classifier-free guidance, we consider language model (LM) guidance, which attempts to maximize

p(x|y)α p(x)β

lθ′ ,γ(x,y) def= q(x)

, (4)

where p(x) and p(x|y) are obtained from a captioning model as in CFG but q(x) is obtained from a language model that was trained independently (but with the same vocabulary) on a large text corpus. The quantity p(x|y)/p(x) = p(x,y)/(p(x)p(y)) measures the strength of the association between a caption and an image; its logarithm is the pointwise mutual information (PMI). LM guidance relies on the assumption that, even for large shifts in the prior distribution of captions p(x), the shift in PMI will be small. Empirically, we obtain better results by allowing different exponents for the numerator and denominator, with α > β. This decoupling resembles PMIk [9], which reduces the bias of PMI toward rare associations. We provide a more detailed derivation in Appendix A.2.

We investigate two applications of LM guidance. First, we combine a captioning model fine-tuned on MS-COCO with a LM prompted with manually written descriptive captions to alter the style of the captions the model produces. The manually written prompts are shown in Appendix A.4. Second, we combine a captioning model trained only on low-quality web data with a LM prompted with varying numbers of examples from the MS-COCO training set to evaluate the ability of LM guidance to elicit higher-quality captions without high-quality paired data. We randomly select a different set of captions for each minibatch of four test examples. In both cases, we separate the captions with two newlines. Because this format leads the LM to place probability mass on the newline token to end the caption, we transfer the probability mass from the newline token to the EOS token. See Appendix A.3 for pseudocode.

##### 3.3. Models and training

Our captioning model is a “bottleneck” variant of CoCaBase [46], which combines a contrastive loss with a captioning loss to simultaneously learn aligned image and text embeddings as well as a captioner. The architecture consists of an image encoder, a unimodal text decoder, and a multimodal text decoder, each of which are Transformers with 12 layers, 768 hidden dimensions, an MLP of size 3072, and 12 self-attention heads, matching BERTBASE [10] and GPT-1 [31]. The image encoder is a ViT-B/18 that processes 288 × 288 input and produces an embedding such that images are embedded close to their corresponding text.

CoCa’s multimodal text decoder processes the represen-

tations of the image encoder to produce a caption. Whereas [46] conditions the multimodal text decoder using crossattention to pooled representations, our bottleneck variant uses only the contrastive image embedding. Appendix A.5 shows a diagram of the resulting architecture. We adopt this bottleneck variant because of its simplicity and the conceptual appeal: When CFG is used, the captioner’s role is to invert the image embedding, providing a caption that, when embedded by the text encoder, lies close to it. However, as we show in Appendix B.1, this choice of the bottleneck model is not critical, and CFG is equally effective with the standard CoCa architecture with attention pooling.

For CFG experiments, we pretrain our model on an image-text dataset comprising images from the JFT-5B dataset [39, 48] paired with their corresponding label names substituted into a randomly selected prompt from the list provided by Radford et al. [30], web images paired with noisy alt text from the ALIGN dataset [17], and a small amount of data from other sources. We follow the same recipe as in [46], and do not mask conditioning information during pretraining.1 We then fine-tune on the combined MS-COCO train and Karpathy validation splits [18] using Adam with batch size 128. We linearly warm up to a learning rate of 1 × 10−5 over the first 1,000 steps and linearly decay to zero over the rest of training. We vary γ ∈ {1.0,1.2,1.5,2.0,3.0,4.0}, conditioning masking proportion in {0.0,0.25,0.5,0.75}, and numbers of steps in {5,000,10,000,20,000,50,000}. We report results from the model trained for 20,000 steps with masking proportion 0.5, which achieves near-optimal results, in Tables 1 and B.4, and sample example captions from it. To ensure that results generalize across datasets, we also experiment with a model fine-tuned on Conceptual Captions [36] for 100,000 steps with masking proportion 0.5.

For LM guidance experiments, we pretrain on the JFT5B and ALIGN datasets, again following the recipe of [46]. For zero-shot captioning experiments, we fine-tune this model on the same datasets for an additional 50,000 steps with conditioning masking proportion of 0.5 to improve our ability to sample unconditionally. For LM guidance on MS-COCO, we first fine-tune on ALIGN, JFT-5B images backcaptioned by an MS-COCO fine-tuned CoCa2B model, and a small amount of internal data before finetuning on MS-COCO. Our language model is a variant of Primer [37] with 2 billion parameters, trained on a similar dataset to that used to train PaLM [7].

##### 3.4. Evaluation

We adopt the standard reference-based captioning metrics BLEU-4, METEOR, ROUGE, and CIDEr, as well

1We find that passing an all-zero image embedding to the pretrained model yields samples that resemble the unconditional distribution, suggesting that it implicitly learns to model the unconditional distribution.

as reference-free captioning metrics based on CLIP ViTB/32 [30]. The first reference-free captioning metric is CLIPScore [13], which is defined as CLIP-S(c,v) = 2.5 · max(cos(c,v),0) where c and v are the CLIP embeddings of the caption and image respectively. The second reference-free metric measures the accuracy with which we can retrieve an image from the generated caption within a given test split by taking the k nearest neighbors of the caption in the CLIP embedding space. Because recall@k for k > 1 is highly correlated with recall@1 (R@5: r = 0.99, R@10: r = 0.98), we plot only recall@1. We additionally report RefOnlyCLIP-S, a reference-based metric that uses the CLIP text encoder to compute the similarity of CLIP embeddings of the generated captions with embeddings of ground truth captions, and RefCLIP-S, which takes the average of the per-image harmonic means of CLIP-S and RefOnlyCLIP-S [13]. Unless otherwise stated, all evaluation is performed on the MS-COCO Karpathy test split [18].

#### 4. Results

##### 4.1. Classifier-free guidance

We first investigate the trade-off between referencebased and reference-free image captioning metrics as a function of guidance scale. Because different guidance scales and metrics could conceivably benefit from different fine-tuning hyperparameter combinations, we plot all results from our hyperparameter grid in Figure 2. Although standard greedy decoding (γ = 1.0) produces the highest CIDEr, METEOR, ROUGE, and BLEU-4 scores, higher guidance weights consistently yield higher values of reference-free captioning metrics. In particular, γ = 3.0 offers both the best caption→image recall and the best CLIPScore.

Table 1 compares our results, obtained from a single model evaluated at different guidance scales, with previous work that reports either CLIPScore or CLIP ViT-B/32 caption→image retrieval performance. Although our model is trained with standard cross-entropy loss rather than a CLIP-based loss and our pretraining dataset is distinct from CLIP’s, sampling from our model with CFG yields higher CLIPScores than all other models trained without CLIPbased losses, and better CLIP caption→image retrieval even when compared with models that use CLIP-based losses.

We present examples of captions generated at different CFG scales in Figure 3. Higher CFG strengths lead to more descriptive captions. At γ = 1.0, the central object in the top left image is described as a “car” as in the ground truth caption, whereas at γ > 1.0 it is a “station wagon.” Similarly, at low CFG strengths, the birds in the center image are described simply as “birds,” whereas at γ = 2.0 they become “crested cranes.” However, at γ = 3.0, captions clearly become less grammatical, containing repeated

80

80

80

80

78

78

78

78

CLIPScore

76

76

76

76

74

74

74

74

72

72

72

72

= 1.0 = 1.2

70

70

70

70

10 20 30 BLEU-4

10 20 30 METEOR

20 40 60 ROUGE

0 50 100 CIDEr

- = 1.5

- = 2.0

- = 3.0

- = 4.0

50

50

50

50

40

40

40

40

CLIPR@1

Human

30

30

30

30

20

20

20

20

10 20 30 BLEU-4

10 20 30 METEOR

20 40 60 ROUGE

0 50 100 CIDEr

- Figure 2. Classifier-free guidance controls a trade off between reference-free and reference-based captioning metrics. Each point reflects a model trained with a different hyperparameter combination; each color represents a γ value used to decode. Models are evaluated with different guidance scales γ, using reference-free captioning metrics based on CLIP ViT-B/32 (y-axes; top: CLIPScore, bottom: recall@1) and reference-based captioning metrics (x-axes). The dashed line reflects the value of the reference-free captioning metric for the groundtruth captions obtained from MS-COCO.

|Model<br><br>|Reference-Based Metrics BLEU-4 METEOR ROUGE CIDEr RefOnlyCLIP-S<br><br>|Reference-Free Metrics CLIP-S R@1 R@5 R@10|RefCLIP-S|
|---|---|---|---|

Models trained with CLIP features or losses: CLIP-Captioner [4] 38.7 29.3 58.6 126.0 0.811 0.754 0.814 UMT-BITG [16] 37.3 28.2 57.9 122.6 0.772 X-LAN+SCST+GEG [50] 36.5 28.7 57.5 121.7 28.1 50.3 67.2 CIDEr + CLIP-S Reward [6] 37.7 28.8 58.3 124.6 0.772 24.4 50.2 63.1 CLIP-S Reward [6] 6.2 18.7 31.6 11.2 0.860 42.5 71.6 82.2 ZeroCap [40] 2.6 11.5 14.6 0.87 0.79 Models trained without access to CLIP: UMT-BITG w/o CLIP loss [16] 37.6 28.3 58.1 122.5 0.725 VinVL-large [49] 41.0 30.9 59.4∗ 140.9 0.91∗ 0.78∗ 0.84∗ Ours (γ = 1.0) 36.1 30.5 58.2 126.1 0.900 0.775 26.5 51.9 64.1 0.830 Ours (γ = 1.2) 35.1 30.0 57.5 124.1 0.899 0.785 31.3 57.4 69.3 0.835

- Ours (γ = 1.5) 31.5 28.4 54.4 113.2 0.891 0.796 36.6 64.0 75.0 0.838

- Ours (γ = 2.0) 20.9 23.3 43.0 78.6 0.862 0.808 44.6 71.7 81.7 0.831

- Ours (γ = 3.0) 11.5 17.1 29.4 41.7 0.820 0.808 49.4 75.7 84.7 0.811

- Ours (γ = 4.0) 6.5 12.3 18.4 17.3 0.766 0.782 44.7 71.3 80.9 0.771

- Table 1. Quantitative comparison of our approach with results from previous work that reports CLIP-based metrics. For VinVL-large, ∗ indicates metrics from [19].

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

γ=1.0: a car with a surfboard on top of it parked next to a car

- γ=1.5: a vintage station wagon with a surfboard on top
- γ=2.0: antique station wagons and a car buick stationwagon
- γ=3.0: buick woody woody stationwagon and surf green station wagon parked in front of car show establishment

GT: An old green and brown car with chrome trim.

γ=1.0: a knife is sitting on a cutting board next to a sliced carrot

- γ=1.5: a knife is sitting on a cutting board next to an orange
- γ=2.0: knife sitting on cutting board next to whole one
- γ=3.0: knife sitting on cutting board next to misshappi carrot on cutting board

GT: A knife sticking out of the side of a block of cheese.

γ=1.0: a herd of sheep grazing

on a grass covered road

- γ=1.5: sheep grazing on a highway with a truck on the road
- γ=2.0: sheep graze on freeway medians where grass is grown
- γ=3.0: grazing trucks blocking sheep on roadway grazing grass

GT: A large herd of sheep are grazing by the busy road.

γ=1.0: two birds standing in a cage with their heads in the air

- γ=1.5: two birds standing inside of a cage in a zoo
- γ=2.0: two crested cranes inside a wire cage
- γ=3.0: crested tantalus cranes caged together in birdcage enclosure

GT: Two birds who are looking out of the cage they are in.

γ=1.0: a view of a city with a clock tower in the background

- γ=1.5: a city with steeples and trees and buildings
- γ=2.0: spires of churches line a city skyline
- γ=3.0: spires steeples buildings trees church spires and trees

GT: A clock that is on the side of a tower.

γ=1.0: a kitchen with a mi-

crowave and a refrigerator

- γ=1.5: a kitchen with a microwave and a refrigerator
- γ=2.0: a kitchen with red appliances and white cupboards
- γ=3.0: appliances sit in a small empty dingroomy red and white kitchen

GT: A kitchen that has a tile floor, a refrigerator, a microwave, and a toaster.

γ=1.0: a bathroom with a large

mirror and a bathtub

- γ=1.5: a bathroom with a large mirror and a bathtub
- γ=2.0: a spacious bathroom with a large mirror and a large tub
- γ=3.0: spacious bathroom with chandelier over tub mirrors and tv

GT: A bathroom with a tub, sinks, lights and a television.

- Figure 3. Caption descriptiveness increases with CFG strength, but high CFG strengths produce agrammatical captions. Here we show examples of captions generated with different classifier-free guidance scales, for randomly selected images without human faces from the MS-COCO Karpathy test split. Captions labeled γ = 1.0 are obtained without CFG; γ > 1 uses CFG; GT = ground truth.

words (“woody woody”) and nonsense words (“misshappi”, “dingroomy”). Figure 4 shows captions obtained with and without CFG next to the top 5 closest images in the embedding space of CoCa 2B [46],2 where it is clear that CFG adds details to captions that help to distinguish them from other captions in the test split. We provide additional examples in Appendix C.

To provide additional quantitative assessments of the specificity of elicited captions, we perform two additional evaluations, described further in Appendix B.2. First, we generate captions for the Stanford Dogs [21] test set, which consists of 8,580 images in total of 120 breeds of dogs, and examine their properties. Without guidance, only 1.9% of captions contain one of the 120 breed names, whereas at γ = 2.0, 42.4% do. The percentage of these breed names that are correct changes little, from 61.7% without guidance to 58.5% at γ = 2.0. Second, we performed a human evaluation comparing captions of MS-COCO test set images obtained without guidance and at γ = 2.0. We asked subjects to select the caption that is “better” and “more descriptive” or to indicate that they are both equal. When we asked these questions separately, we found that the two sets of captions are statistically indistinguishable. However, when asking both questions on the same survey, we found that captions generated without guidance are slightly “better” (50.5% vs. 46.6%, p = 0.006, binomial test) but captions generated at γ = 2.0 are “more descriptive” (52.7% vs. 45.8%, p = 1 × 10−6).

To validate the reliability of our results, we further measure the impact of CFG on three additional datasets, nocaps [2], Flickr-8k [15], and Conceptual Captions (CC3M) [36], as well as with alternative retrieval models. nocaps is a test set for captioning models with objects not present in MS-COCO; Flickr-8k is a small captioning dataset collected using a different procedure than MSCOCO; and Conceptual Captions is a set of 3.3M captions collected from filtered alt-text. We fine-tune the bottleneck CoCa-Base model directly on CC3M, and use our model fine-tuned on MS-COCO to caption images on nocaps and Flickr-8K. As shown in Figure 5, we find trade-offs between reference-based and reference-free captioning metrics similar to those above. In Appendix B.3, we report reference-free captioning metrics on MS-COCO computed with two additional retrieval models: the pretrained CoCa 2B model from [46] and the fine-tuned CoCa Base model that we use to generate captions. With both models, CFG substantially increases recall, in line with results obtained with CLIP ViT-B/32.

Although CFG produces captions that are more successful at uniquely identifying images than decoding from the conditional distribution, caption lengths are similar for

2We use CoCa 2B rather than CLIP because, quantitatively and qualitatively, it provides better retrieval results both with and without guidance.

[Figure 9]

- Figure 4. Captions generated with CFG contain specific details that improve retrieval. For each reference image (far left), we show captions at γ = 1.0 (no guidance) and γ = 2.0. To the right, we show the closest images to each caption in the CoCa embedding space. Reference images are selected at random subject to the constraints that the closest image differs between γ values and there are no identifiable human faces.

0 50 100

70

72

74

76

78

80

CLIPScore

Fine-tune on COCO Eval on nocaps

0 50

70

72

74

76

78

80

Fine-tune on COCO Eval on Flickr-8K

50 100

74

76

78

Fine-tune on CC3M Eval on CC3M

0 50 100 CIDEr

40

50

60

CLIPR@1

0 50 CIDEr

20

30

40

= 1.0 = 1.2 = 1.5

- = 2.0

- = 3.0

= 4.0 Human

50 100 CIDEr

30

40

50

- Figure 5. CFG also yields trade-offs between captioning metrics on nocaps, Flickr-8K, and CC3M.

|γ<br><br>|Words|Characters|
|---|---|---|

|1.0 1.2 1.5 2.0 3.0 4.0|9.6 ± 1.4 9.6 ± 1.4 9.4 ± 1.4<br><br>9.3 ± 2.4<br><br>10.7 ± 7.6<br><br><br>19.9 ± 16.9<br><br>|44.2 ± 7.2<br><br>44.7 ± 7.4<br>45.7 ± 7.8<br><br><br>50.3 ± 18.6 69.0 ± 56.1<br><br>161.2 ± 140.0|
|---|---|---|

- Table 2. Moderate CFG scales do not substantially change caption lengths, although higher CFG scales result in longer captions. Numbers are mean ± standard deviation.

45

40

CLIPR@1

35

30

LM guidance CF guidance

25

20

90 95 100 105 110 115 120 125 130 CIDEr

Figure 6. Language model guidance produces captions that slightly exceed the Pareto frontier of CIDEr vs. caption→image retrieval accuracy on MS-COCO.

γ ∈ [1,2], as shown in Table 2. Thus, at low guidance strengths, CFG improves recall by making more efficient use of words, rather than by producing more verbose captions. Higher CFG strengths lead to longer captions but, as described above, these captions are agrammatical and contain nonsense words.

##### 4.2. Language model guidance

We first experiment with guiding a captioning model fine-tuned on MS-COCO to produce more descriptive captions using a language model prompted with manually written prompts. We first manually wrote a prompt containing 10 descriptive captions of COCO test set images (Appendix A.4). We then sweep over α ∈ {1,2,3,4,5,6,7,8,9,10,12,15} and β ∈ {0,α/4,α/2,3/4α,α}, and compare the resulting retrieval/CIDEr trade-off to that produced by the same model with CFG. We observe that it is possible to obtain small improvements upon the Pareto frontier provided by CFG, as shown in Figure 6. With α = 5, β = −5/2, LM guidance achieves CLIP ViT-B/32 R@1 of 39.6% and CIDEr of 114.4, whereas CFG with γ = 1.6 is worse on both metrics, achieving R@1 of 39.0% and CIDEr of 109.3.

We further experiment with using prompting to control the captioner using a manually written prompt of 25 captions in the form of “a photo of NUMBER OBJECTS” (e.g., “a photo of eight apples”; see Appendix A.4). With α = β = 1, the guided model is able to match this format and counts the number of objects in images (Figure 7).

We next investigate whether language model guidance can elicit better captions from a model trained only on

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

a photo of two dogs a photo of one bird a photo of four suitcases a photo of five sheep

- Figure 7. Captions generated with LM guidance with a prompt of 25 captions in the form of “a photo of NUMBER OBJECTS”. Examples are selected to show different numbers of objects.

0 20 40 Shots

2.5

5.0

7.5

10.0

12.5

15.0

CLIPR@1

0 20 40 Shots

0

20

40

60

CIDEr

LM guidance No guidance

- Figure 8. LM guidance substantially improves CIDEr and retrieval scores of a model trained solely on minimally curated web data and evaluated on MS-COCO. The x-axis shows the number of captions used to prompt the LM; we do not prompt with images.

low-quality data. Here, we use a CoCa model that is pretrained on image-alt text pairs from the web (the ALIGN dataset [17]) and classification labels converted to text (the JFT-5B dataset [48]), without any additional fine-tuning. Because the data distribution places higher probability mass on short, non-descriptive captions than on longer captions, the resulting model is of limited utility for captioning, and would generally need to be fine-tuned on another dataset such as MS-COCO before being applied to a captioning task. Rather than fine-tune, we use LM guidance to prompt the model with captions from the MS-COCO training set.

LM guidance substantially improves the quality of the captions produced by the original pretrained CoCa model without any clean parallel data. With LM guidance, we achieve CIDEr scores of 48.6 with 5 shots and 59.7 with 50 shots, far exceeding the CIDEr score of 21.8 obtained with no guidance. Figure 8 shows CIDEr and CLIP recall@1 scores for LM guidance of this pretrained CoCa model as a function of the number of shots, with α = β = 1. Table 3 compares classifier-free guidance and LM guidance. CFG yields higher CLIP-Scores and retrieval accuracy than LM guidance with α = β = 1, but LM guidance provides much higher CIDEr scores.

We compare captions generated with CFG to those generated with LM guidance for four images in Figure 9. In general, CFG produces agrammatical captions, whereas LM guidance produces grammatical captions but hallucinates details. For example, the image in the upper left shows two elephants and no zebras, but LM guidance leads to the caption “an elephant and a zebra in a field.”

|Model|Reference-Based Metrics BLEU-4 METEOR ROUGE CIDEr RefOnlyCLIP-S<br><br>|Reference-Free Metrics CLIP-S R@1 R@5 R@10<br><br>|RefCLIP-S|
|---|---|---|---|

Classifier-free guidance:

γ = 1.0 8.2 8.3 21.8 21.8 0.766 0.694 9.0 19.5 26.1 0.725 γ = 1.2 8.6 9.5 24.5 25.0 0.781 0.718 12.7 27.2 35.1 0.745

- γ = 1.5 8.9 10.0 25.6 25.2 0.780 0.728 16.7 33.8 43.0 0.750

- γ = 2.0 8.1 9.7 23.9 22.9 0.777 0.741 21.2 40.8 51.1 0.755

- γ = 3.0 7.1 8.7 20.0 18.5 0.767 0.753 25.8 47.8 58.3 0.756

- γ = 4.0 6.4 7.5 16.3 13.9 0.749 0.743 27.3 48.5 58.1 0.742 Language model guidance with α = β = 1: 2 captions 12.7 14.6 34.7 39.3 0.806 0.688 10.0 23.7 32.4 0.740 5 captions 15.0 16.6 39.1 48.6 0.827 0.712 12.4 27.5 37.5 0.763 10 captions 16.2 17.7 40.5 53.1 0.835 0.723 13.0 30.5 41.0 0.773 20 captions 17.4 18.4 41.6 57.4 0.839 0.728 14.4 32.2 42.7 0.777 50 captions 18.1 19.1 42.5 59.7 0.840 0.729 13.4 32.5 43.9 0.778 Other models trained without aligned MS-COCO images and captions: ZeroCap [40] 2.6 11.5 14.6 0.87 0.79 MAGIC [38] 12.9 17.4 39.9 49.3 Flamingo [3] 84.3 DeCap (560 captions) [22] 51.4 DeCap (full train set) [22] 24.7 25.0 91.2 CapDec (full train set) [27] 26.4 25.1 51.8 91.8

Table 3. Comparison of decoding strategies for a captioning model trained only on minimally curated web data (JFT-5B and ALIGN) and evaluated on MS-COCO. At the bottom, we report metrics for other models trained without aligned MS-COCO images and captions. These models may not be directly comparable since they use different pretraining data. DeCap and CapDec use all 560K captions in the MS-COCO training set to train their decoders; we include CIDEr for DeCap with 560 captions (0.1% of the training data) for comparison.

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

- γ=1.0: a photo of the small elephant.
- γ=2.0: elephants in the ruaha national park
- γ=3.0: elephants chobe np

LM: a elephant and a zebra in a field

GT: Two elephants standing on a grassy field next to a tree.

- γ=1.0: a photo of the small coffee.
- γ=2.0: a coffee in a video game.
- γ=3.0: a banana in a video game.

LM: a banana with a cup of coffee

GT: A close up of a banana next to a cup with liquid.

- γ=1.0: a photo of the large windsurfing.
- γ=2.0: windsurfing in tarifa
- γ=3.0: windsurfing wallpapers 1200x1024

LM: a windsurfer in the water

GT: A man riding a wind sail in the ocean filled with waves.

- γ=1.0: a photo of the large giraffe.
- γ=2.0: a giraffe in a video game.
- γ=3.0: giraffe standing photo 1

LM: a giraffe standing in a tall tree GT: A giraffe in a dry savannah with dry shrubs

Figure 9. Examples of captions generated from a model pretrained only on minimally curated data, for randomly selected images without human faces. Captions labeled γ = 1.0 are obtained without CFG; γ > 1 uses CFG; LM indicates LM guidance with α = β = 1 and 20 shots; GT indicates ground truth.

- 5. Conclusion

also inadequate, because captions that lie close to the image need not be grammatical and may contain gibberish. Our proposed methods leveraging classifier-free guidance and language model guidance modulate the trade-offs between these two goals, as captured by various reference-based and reference-free metrics.

There are several possible extensions to our work. First, our present experiments use only greedy decoding. Although greedy decoding appears to perform reasonably well in our setup, it may be suboptimal for LM guidance with prompts that impose greater structure upon the captions. If the LM is prompted to output either “there is a person in this image” or “there is no person this image”, greedy decoding is likely to fail even if the captioner properly scores the two possible captions, because when choosing between the tokens “a” and “no”, the captioner has no knowledge of the structure that the LM will impose on future tokens. Since beam search could explore both tokens, it may offer better results in this scenario. Second, our method could be combined with RL-based methods to increase similarity in a contrastive embedding space, which may further improve retrieval performance and CLIPScore. Finally, with a perfect captioning model, p(image|caption) should increase with γ. However, in practice we find that γ > 3 leads to a decrease in retrieval performance. This discrepancy suggests that the difference between the conditional and unconditional model distributions may be a noisy estimator of the pointwise mutual information. Although selecting γ is one way to regularize this estimator, there may also be strategies to regularize p(x|y)/p(x) at training time.

Our study indicates that it is possible to substantially improve the extent to which generated captions uniquely describe the images they correspond to, raising questions regarding the goal of image captioning and how it should be evaluated. As it is conventionally formulated, image captioning aims not to provide text that can substitute for an image, but to write the text that a human annotator would have written. This formulation penalizes captions that are more descriptive than ground truth, even when a human might prefer them. On the other hand, treating image captioning as a problem of generating a caption that lies close to the image in the embedding space of an image-text model is

#### Acknowledgements

We thank Kevin Clark, David Fleet, Geoffrey Hinton, and the rest of the Google DeepMind Toronto team for inspiration, comments, and discussion.

#### References

- [1] Armen Aghajanyan, Bernie Huang, Candace Ross, Vladimir Karpukhin, Hu Xu, Naman Goyal, Dmytro Okhonko, Mandar Joshi, Gargi Ghosh, Mike Lewis, et al. Cm3: A causal masked multimodal model of the internet. arXiv preprint arXiv:2201.07520, 2022. 2
- [2] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. nocaps: novel object captioning at scale. In Proceedings of the IEEE International Conference on Computer Vision, pages 8948–8957, 2019. 6
- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems, 2022. 2, 8
- [4] Manuele Barraco, Marcella Cornia, Silvia Cascianelli, Lorenzo Baraldi, and Rita Cucchiara. The unreasonable effectiveness of clip features for image captioning: An experimental analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4662–4670, 2022. 5
- [5] David M Chan, Austin Myers, Sudheendra Vijayanarasimhan, David A Ross, and John Canny. ic3: Image captioning by committee consensus. arXiv preprint arXiv:2302.01328, 2023. 2
- [6] Jaemin Cho, Seunghyun Yoon, Ajinkya Kale, Franck Dernoncourt, Trung Bui, and Mohit Bansal. Fine-grained image captioning with CLIP reward. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 517–527, Seattle, United States, July 2022. Association for Computational Linguistics. 2, 5
- [7] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. PaLM: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022. 4
- [8] Katherine Crowson. You can apply a similar trick to classifier-free guidance to autoregressive transformers to sample from a synthetic ”super-conditioned” distribution. https://twitter.com/RiversHaveWings/status/1478093658716966912,

2022. 3

- [9] B´eatrice Daille. Approche mixte pour l’extraction automatique de terminologie: statistiques lexicales et filtres linguistiques. PhD thesis, Ph. D. thesis, Universit´e Paris 7, 1994. 3
- [10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language

- Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. 3
- [11] Jacob Devlin, Hao Cheng, Hao Fang, Saurabh Gupta, Li Deng, Xiaodong He, Geoffrey Zweig, and Margaret Mitchell. Language models for image captioning: The quirks and what works. arXiv preprint arXiv:1505.01809, 2015. 2
- [12] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scenebased text-to-image generation with human priors. arXiv preprint arXiv:2203.13131, 2022. 2, 3
- [13] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7514–7528, Online and Punta Cana, Dominican Republic, Nov. 2021. Association for Computational Linguistics. 2, 4
- [14] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 1, 3
- [15] Micah Hodosh, Peter Young, and Julia Hockenmaier. Framing image description as a ranking task: Data, models and evaluation metrics. Journal of Artificial Intelligence Research, 47:853–899, 2013. 6
- [16] Yupan Huang, Hongwei Xue, Bei Liu, and Yutong Lu. Unifying multimodal transformer for bi-directional image and text generation. In Proceedings of the 29th ACM International Conference on Multimedia, pages 1138–1147, 2021. 2, 5
- [17] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, pages 4904–4916. PMLR,

2021. 4, 7

- [18] Andrej Karpathy and Fei-Fei Li. Deep visual-semantic alignments for generating image descriptions. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015, pages 3128–3137. IEEE Computer Society, 2015. 4
- [19] Jungo Kasai, Keisuke Sakaguchi, Ronan Le Bras, Lavinia Dunagan, Jacob Morrison, Alexander R Fabbri, Yejin Choi, and Noah A Smith. Bidimensional leaderboards: Generate and evaluate language hand in hand. arXiv preprint arXiv:2112.04139, 2021. 5
- [20] Jungo Kasai, Keisuke Sakaguchi, Lavinia Dunagan, Jacob Morrison, Ronan Le Bras, Yejin Choi, and Noah A Smith. Transparent human evaluation for image captioning. arXiv preprint arXiv:2111.08940, 2021. 1, 2
- [21] Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, and Li Fei-Fei. Novel dataset for fine-grained image categorization. In First Workshop on Fine-Grained Visual Categorization, IEEE Conference on Computer Vision and Pattern Recognition, Colorado Springs, CO, June 2011. 6
- [22] Wei Li, Linchao Zhu, Longyin Wen, and Yi Yang. Decap: Decoding clip latents for zero-shot captioning. In International Conference on Learning Representations, 2022. 2, 8

- [23] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004. 1
- [24] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 2
- [25] Annika Lindh, Robert J Ross, Abhijit Mahalunkar, Giancarlo Salton, and John D Kelleher. Generating diverse and meaningful captions. In International Conference on Artificial Neural Networks, pages 176–187. Springer, 2018. 2
- [26] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning, pages 16784–16804. PMLR, 2022. 2
- [27] David Nukrai, Ron Mokady, and Amir Globerson. Text-only training for image captioning using noise-injected clip. arXiv preprint arXiv:2211.00575, 2022. 2, 8
- [28] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. BLEU: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318,

2002. 1

- [29] Jordi Pont-Tuset, Jasper Uijlings, Soravit Changpinyo, Radu Soricut, and Vittorio Ferrari. Connecting vision and language with localized narratives. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part V 16, pages 647–664. Springer,

2020. 2

- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 2, 4
- [31] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018. 3
- [32] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125,

2022. 2

- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 2
- [34] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo-Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In Advances in Neural Information Processing Systems. 2

- [35] Guillaume Sanchez, Honglu Fan, Alexander Spangher, Elad Levi, Pawan Sasanka Ammanamanchi, and Stella Biderman. Stay on topic with classifier-free guidance. arXiv preprint arXiv:2306.17806, 2023. 3
- [36] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018. 4, 6
- [37] David R So, Wojciech Ma´nke, Hanxiao Liu, Zihang Dai, Noam Shazeer, and Quoc V Le. Primer: Searching for efficient transformers for language modeling. arXiv preprint arXiv:2109.08668, 2021. 4
- [38] Yixuan Su, Tian Lan, Yahui Liu, Fangyu Liu, Dani Yogatama, Yan Wang, Lingpeng Kong, and Nigel Collier. Language models can see: plugging visual controls in text generation. arXiv preprint arXiv:2205.02655, 2022. 2, 8
- [39] Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable effectiveness of data in deep learning era. In Proceedings of the IEEE international conference on computer vision, pages 843–852, 2017. 4
- [40] Yoad Tewel, Yoav Shalev, Idan Schwartz, and Lior Wolf. Zerocap: Zero-shot image-to-text generation for visualsemantic arithmetic. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17918–17928, 2022. 2, 5, 8
- [41] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. CIDEr: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4566–4575, 2015. 1
- [42] Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and tell: A neural image caption generator. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3156–3164, 2015. 2
- [43] Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery. arXiv preprint arXiv:2302.03668, 2023. 2
- [44] Shitong Xu. Clip-diffusion-lm: Apply diffusion model on image captioning. arXiv preprint arXiv:2210.04559, 2022. 2
- [45] Linjie Yang, Kevin Tang, Jianchao Yang, and Li-Jia Li. Dense captioning with joint inference and visual context. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2193–2202, 2017. 2
- [46] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022. 3, 4, 6, 14
- [47] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022. 2, 3
- [48] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In Proceedings of

- the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12104–12113, 2022. 4, 7
- [49] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Revisiting visual representations in vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5579– 5588, 2021. 5
- [50] Youyuan Zhang, Jiuniu Wang, Hao Wu, and Wenjia Xu. Distincive image captioning via clip guided group optimization. arXiv preprint arXiv:2208.04254, 2022. 2, 5
- [51] Zixin Zhu, Yixuan Wei, Jianfeng Wang, Zhe Gan, Zheng Zhang, Le Wang, Gang Hua, Lijuan Wang, Zicheng Liu, and Han Hu. Exploring discrete diffusion models for image captioning. arXiv preprint arXiv:2211.11694, 2022. 3

# Appendix

#### A. Additional details regarding our approach

##### A.1. Pseudocode for classifier-free guidance

Below, we provide pseudocode for greedy decoding with classifier-free guidance. Note that, in practice, we perform decoding in batches.

# captioner: Captioning model (returns token log probs) # img_embed: Image embedding # gamma: Classifier-free guidance scale # max_length: Maximum number of tokens in caption # BOS: Beginning of sequence token # EOS: End of sequence token

tokens = [BOS] for i in range(0, max_length):

# Eq. 3 (without the softmax, since it does not affect the argmax). cond_log_probs = captioner(tokens, img_embed) uncond_log_probs = captioner(tokens, zeros_like(img_embed)) scores = uncond_log_probs + gamma * (cond_log_probs - uncond_log_probs)

# Greedily take the next token. next_token = argmax(scores) tokens.append(next_token) if next_token == EOS: break

###### A.2. Derivation of language model guidance Assume that we have two joint distributions of captions x and images y, p(x,y) and q(x,y), and these distributions have the same pointwise mutual information between any image-caption pair, i.e. log qq(x()xq,y()y) = log pp(x()xp,y()y), and thus qq(x()xq,y()y) =

p(x,y) p(x)p(y). Starting with the leftmost expression from Eq. 2, there exists an expression that uses the joint distribution from p but only marginals of captions from q,

γ

γ

q(x) qq((xx|y))

= q(x) q q(x()xq,y()y)

(5)

γ

= q(x) p p(x()xp,y()y)

. (6) In Eq. 4, we further decouple the exponents for the numerator and denominator of the above equation. As we note, this decoupling is reminiscent of pmik. To see this relationship, first note that pp(x()xp,y()y) is the exponential of pmi(x,y) = log pp(x()xp,y()y). Replacing pmi(x,y) with pmik(x,y) = log p(x,y)

γ

k

k

p(x)p(y), Eq. 6 becomes q(x) p(x,y)

. Setting α = kγ and β = γ gives

p(x)p(y)

p(x,y)α p(x)βp(y)β

p(x|y)α p(x)βp(y)β−α ∝ q(x)

p(x|y)α p(x)β

q(x)

, (7) where the proportionality holds because p(y) is fixed.

= q(x)

##### A.3. Pseudocode for language model guidance

# captioner: Captioning model (returns token log probs) # lm: Language model (returns token log probs) # prompt_tokens: Tokenized prompt for language model # img_embed: Image embedding # alpha, beta: Cond/uncond exponents from Eq. 4 # max_length: Maximum number of tokens in caption # BOS: Beginning of sequence token

# EOS: End of sequence token # NEWLINE: Newline token

tokens = [BOS] for i in range(0, max_length):

# Log of Eq. 4. lm_log_probs = lm(concat(prompt_tokens, tokens)) cond_log_probs = captioner(tokens, img_embed) uncond_log_probs = captioner(tokens, zeros_like(img_embed)) scores = lm_log_probs + alpha * cond_log_probs - beta * uncond_log_probs

# Transfer probability mass from NEWLINE to EOS. scores[EOS] = logsumexp([scores[EOS], scores[NEWLINE]]) scores[NEWLINE] = -inf

# Greedily take the next token. next_token = argmax(scores) tokens.append(next_token) if next_token == EOS: break

##### A.4. Manually written prompts

Below, we include the manually written prompts that we use in our language model guidance experiments. Each caption is separated by two newlines.

###### A.4.1 Descriptive caption prompt

a bathroom with goldenrod circular patterned tiles contains a toilet bidet sink mirror tissue dispenser and hairdryer\n donuts being sorted on the conveyor belt of a device labeled donut robot in an industrial kitchen\n a green glass mug containing 3 toothbrushes and 1 tube of toothpaste sitting on a windowsill \n a man wearing sunglasses and a gray shirt poses with a woman wearing a white shirt next to a giraffe with a fence behind them\n a snow covered wooden bench in front of a fence with snow covered evergreen plants behind it \n two white horses pull a plow with a man in a white shirt and cyan cap and a man in a red shirt with sunglasses behind them next to a fence under a sky with cumulus clouds\n a man in a blue shirt and a small child in a red striped shirt play frisbee next to trees in a park\n a black clock tower with a lit up white clock face with roman numerals in front of a dilapidated five story warehouse after dusk\n a decorative pool flanked by palm trees in front of a stone clock tower next to a large ten story building with a bright advertisement on top in a city at night\n cows with gray bodies and white heads eating grass on a hill with a foggy mountain in the background\n

###### A.4.2 Counting prompt

a photo of four clouds\n a photo of one cat\n a photo of three horses\n a photo of seven candles\n a photo of sixteen keys\n a photo of one rat\n

a photo of five carrot sticks\n a photo of one turtle\n a photo of two boats\n a photo of one orange\n a photo of nine books\n a photo of ten fingers\n a photo of twelve eggs\n a photo of one microwave\n a photo of two children\n a photo of six leaves\n a photo of two monitors\n a photo of one toilet\n a photo of one house\n a photo of five pairs of pants\n a photo of eight apples\n a photo of eleven stars\n a photo of one hat\n a photo of two chairs\n a photo of seven coins\n a photo of three birds\n

##### A.5. Difference between attention pooling and bottleneck CoCa architecture

Yu et al. [46] perform attentional pooling over the token representations of the image encoder and pass the resulting tokens into the multimodal text decoder (Figure A.1 left). By contrast, our bottleneck architecture uses the same embedding for the contrastive loss and multimodal text decoder (Figure A.1 right). We create this bottleneck because a goal of our work is to invert contrastive embeddings, producing a caption that lies close to the contrastive image embedding when it is embedded by the text encoder. As we show below in Appendix B.1, this bottleneck is not necessary for CFG to yield improvements. The attention pooling architecture is equally compatible with our approach and yields slightly better performance.

CoCa (Yu et al., 2022)

Bottleneck CoCa (Ours)

Captioning Loss

Captioning Loss

two dogs running in a field [/s]

two dogs running in a field [/s]

Multimodal Text Decoder

Multimodal Text Decoder

Contrastive Loss

Contrastive Loss

attentional pooling cls-token

attentional pooling cls-token

Image Encoder

Unimodal Text Decoder

Image Encoder

Unimodal Text Decoder

[s] two dogs running in a field [CLS]

[s] two dogs running in a field [CLS]

[Figure 18]

[Figure 19]

pairs

pairs

“two dogs running in a field”

“two dogs running in a field”

,

,

image text

image text

Figure A.1. Comparison of CoCa architecture introduced by Yu et al. [46] (left) with our bottleneck CoCa architecture (right).

#### B. Additional experimental results

##### B.1. Attention pooling CoCa architecture

Classifier-free guidance yields similar qualitative results (and slightly better quantiative results) when using the standard CoCa architecture with attention pooling (Figure A.1 left) rather than the bottleneck architecture used in the main text (Fig-

- ure A.1 right). We fine-tune CoCa-Base for 20,000 steps with a max learning rate of 1 × 10−5 and a conditioning masking proportion of 0.5, following the same procedure that gave the near-optimal bottleneck model described in Section 3.3. Fig-
- ure B.1 plots reference-based metrics on the x-axis and reference-free metrics on the y-axis, showing a similar trade-off to

Figure 2. Table B.1 provides quantitative results demonstrating that the attention pooling architecture performs slightly better across both reference-based and reference-free evaluations. Nonetheless, we adopt the bottleneck architecture for our main experiments for the reasons described in Appendix A.5 above.

- 78

- 79

- 80

- 81

- 78

- 79

- 80

- 81

- 78

- 79

- 80

- 81

- 78

- 79

- 80

- 81

CLIPScore

= 1.0 = 1.2

10 20 30 BLEU-4

20 30 METEOR

20 40 60 ROUGE

50 100 CIDEr

- = 1.5

- = 2.0

- = 3.0

- = 4.0

50

50

50

50

CLIPR@1

Human

40

40

40

40

30

30

30

30

10 20 30 BLEU-4

20 30 METEOR

20 40 60 ROUGE

50 100 CIDEr

Figure B.1. Effect of classifier-free guidance on captioning metrics with the attention pooling CoCa model. All points reflect the same finetuned model; each color represents a γ value used to decode. Models are evaluated with different guidance scales γ, using reference-free captioning metrics based on CLIP ViT-B/32 (y-axes; top: CLIPScore, bottom: recall@1) and reference-based captioning metrics (x-axes). The dashed line reflects the value of the reference-free captioning metric for the ground-truth captions obtained from MS-COCO. See Figure 2 for results with the bottleneck model.

|Model<br><br>|Reference-Based Metrics BLEU-4 METEOR ROUGE CIDEr RefOnlyCLIP-S|Reference-Free Metrics CLIP-S R@1 R@5 R@10<br><br>|RefCLIP-S|
|---|---|---|---|

|Bottleneck (γ = 1.0) Bottleneck (γ = 1.2)<br><br>Bottleneck (γ = 1.5)<br><br>Bottleneck (γ = 2.0)<br><br>Bottleneck (γ = 3.0)<br><br>Bottleneck (γ = 4.0)<br><br><br>|36.1 30.5 58.2 126.1 0.900 35.1 30.0 57.5 124.1 0.899 31.5 28.4 54.4 113.2 0.891 20.9 23.3 43.0 78.6 0.862 11.5 17.1 29.4 41.7 0.820<br><br>6.5 12.3 18.4 17.3 0.766<br><br>|0.775 26.5 51.9 64.1 0.785 31.3 57.4 69.3 0.796 36.6 64.0 75.0 0.808 44.6 71.7 81.7 0.808 49.4 75.7 84.7 0.782 44.7 71.3 80.9<br><br>|0.830 0.835 0.838 0.831 0.811 0.771|
|---|---|---|---|
|Att. Pooling (γ = 1.0) Att. Pooling (γ = 1.2)<br><br>Att. Pooling (γ = 1.5)<br>Att. Pooling (γ = 2.0)<br>Att. Pooling (γ = 3.0)<br>Att. Pooling (γ = 4.0)<br>|36.8 30.9 59.0 130.3 0.901 36.3 30.6 58.4 129.1 0.901 32.7 29.0 55.3 118.0 0.892 22.1 24.0 44.3 84.6 0.861 12.2 17.5 30.7 45.7 0.816<br><br>7.2 12.1 19.7 20.7 0.767<br><br>|0.777 27.2 52.7 64.6 0.786 32.0 58.0 69.4 0.798 38.2 64.9 75.6<br><br>0.814 48.6 73.7 83.5<br>0.815 53.6 78.2 86.0 0.788 48.2 72.1 80.1<br>|0.832 0.837 0.840 0.833 0.812 0.773<br><br>|

Table B.1. Quantitative comparison of results obtained with bottleneck and attention pooling architectures.

##### B.2. Quantitative assessment of specificity

###### B.2.1 Evaluation on Stanford Dogs

|γ|% Containing Breed<br><br>|% Breeds Correct|
|---|---|---|

|1.0 1.2 1.5 2.0 3.0|1.9 6.2<br><br>15.9 42.4 67.0<br><br>|61.7 69.0 69.7 58.5 53.3<br><br>|
|---|---|---|

- Table B.2. We generate captions for the 8,580 captions in the Stanford Dogs test set and measure the percentage of the captions that contain the name of one of the 120 dog classes (“% Containing Breed”) and the percentage of those captions where that name is correct (“% Breeds Correct”).

###### B.2.2 Human evaluation

We performed a human evaluation in which we presented crowdsourcing workers with each image and the two possible captions. We experimented with asking subjects to pick the better caption and the more descriptive caption either on different forms or the same form. When asking subjects to pick only the better caption, we provided the following instructions:

Please answer a survey about comparing the quality of two captions for each image. We will present to you an image and ask which caption is better.

When asking subjects to pick the more descriptive caption, we instead provided the following instructions:

Please answer a survey about comparing the descriptiveness of two captions for each image. We will present to you an image and ask which caption is a more detailed description of the image. Please ignore grammatical errors that do not affect readability.

When asking both questions simultaneously, we instructed the subjects as follows:

Please answer a survey about comparing two captions for each image. We will present to you an image and ask a couple questions about:

- 1) descriptiveness: ”Which caption is a more detailed description of the image?”
- 2) quality: ”Which caption is better?”

In each case, subjects saw the image along with the two captions (in random order) as well as the option ”I’m indifferent.” Subjects clicked the radio button next to their preferred choice. We excluded 55 images for which the captions generated without guidance and at γ = 2.0 were identical, resulting in a total of 4,945 images. We obtained a single rating for each image in each condition.

Results are shown in Table B.3. When we asked which caption was “better” and which was “more descriptive” in separate surveys, we found that subjects preferred each caption at a statistically indistinguishable rate. When we asked subjects to pick the “better” and “more descriptive” captions in the same survey, we found that γ = 1.0 was more likely to be chosen as “better” whereas γ = 2.0 was more likely to be chosen as “more specific.” Comparing the odds ratios obtained with the two ways of posing the questions using Fisher’s exact test, we find that the difference between them is statistically significant (“better”: p = 0.004; “more descriptive”: p = 0.01) indicating that human judgments are significantly affected by whether the questions are posed on the same form or separately.

|Question γ = 1.0 γ = 2.0 Indifferent p-value|
|---|

|Separate forms:<br><br>Better 48.0% (2375) 49.8% (2461) 2.2% (109) p = 0.22 More descriptive 47.7% (2359) 49.5% (2446) 2.8% (140) p = 0.21<br><br>|
|---|
|Same form: Better 50.5% (2497) 46.6% (2306) 2.9% (142) p = 0.006 More descriptive 45.8% (2265) 52.7% (2606) 1.5% (74) p = 10−6<br><br>|

- Table B.3. Human evaluation results. We report the percentage and overall number of the 5,000 MS-COCO Karpathy test set images where subjects preferred captions generated at γ = 1.0 or γ = 2.0 or were indifferent, as well as the p-value for the null hypothesis that users are equally likely to select the captions generated at γ = 1.0 and γ = 2.0, computed by a binomial test. When p < 0.05, we bold-face the best result in each row. Otherwise, we bold-face both results.

##### B.3. Reference-free metrics with retrieval models

In Table B.4, we show cosine similarity between generated captions and image embeddings and caption→image retrieval accuracy for the CoCa 2B model and the CoCa-Base model fine-tuned on MS-COCO that was used to generate the captions. In both cases, we find that γ > 1 yields much better metrics than no guidance. Retrieval accuracies (but not cosine similarities) are directly comparable across models; both models offer better retrieval accuracy than CLIP ViT-B/32.

|γ|CoCa 2B Cos. R@1 R@5 R@10<br><br>|Captioning Model (CoCa Base) Cos. R@1 R@5 R@10|
|---|---|---|

|1.0 1.2 1.5 2.0 3.0 4.0<br><br>|0.125 40.1 65.3 75.1 0.128 46.5 72.0 80.3 0.131 55.5 78.9 86.4 0.135 64.9 86.4 91.3 0.134 66.5 87.0 91.4<br><br>0.126 60.3 81.8 87.5<br><br><br>|0.843 49.4 75.0 84.1 0.859 56.2 80.1 88.1 0.877 64.6 85.9 91.5 0.887 73.0 91.6 95.3 0.890 77.7 92.4 95.8 0.875 74.7 90.1 94.0|
|---|---|---|

###### Table B.4. CFG improves caption→image retrieval in the embedding spaces of CoCa models on MS-COCO. “Cos.” = mean cosine similarity between the image and text embeddings.

#### C. Additional examples

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

γ=1.0 a pair of skis sitting on a

γ=1.0 a herd of sheep grazing in

γ=1.0 a group of birds standing

γ=1.0 a living room with a couch

γ=1.0 a vase filled with red and

tiled floor

a field

on top of a wooden post

and a table

yellow flowers

- γ=1.5 pair of skis and ski boots on tiled floor
- γ=2.0 skis and pair of skis on linoleum floor
- γ=3.0 skis and pair of bottle opener sit on vct floor

- γ=1.5 a herd of sheep grazing in a field
- γ=2.0 sheep are gathered in a field near piles of hay
- γ=3.0 bales of sheep are gathered in formation near rocks

- γ=1.5 seagulls lined up on posts in a lake
- γ=2.0 seagulls lined up along a pond line
- γ=3.0 seagulls lined up along posts in shallow water

- γ=1.5 a living room with a couch and a window
- γ=2.0 living room with large window overlooking woods
- γ=3.0 livingroom with view out the window

- γ=1.5 tulips in a clear vase on a table
- γ=2.0 tulips in a clear glass vase on a tablecloth
- γ=3.0 tulips in a clear punchov glass setting on doily

GT Skis and ski boots sit together on a tiled floor.

GT Wood post lined up in the water with birds perched on them.

GT A living room in a remotely located home.

GT Fresh red and yellow tulips in a vase.

GT A herd of sheep standing on top of a grass covered field.

[Figure 25]

γ=1.0 a cat sitting on a blue chair with a white wall behind it

- γ=1.5 a cat sitting on a blue chair outside
- γ=2.0 calico colored cat sitting on blue chair outside
- γ=3.0 calico colored cat sitting on blue metal chair

GT A furry cat sits on a blue chair.

[Figure 26]

γ=1.0 a bathroom with a toilet sink and bathtub

- γ=1.5 a bathroom with blue and white tiles and a blue and white towel
- γ=2.0 bathroom with blue accents and blue and white towels
- γ=3.0 spotless uncroom bathroom with blue accents fisheye fisheye fisheye fisheye fisheyemmangles viewersquallly fisheye and fisheye lens

GT Bathroom with white

pedestal sink, bathtub and shower, and commode.

[Figure 27]

[Figure 28]

γ=1.0 a cat sitting on top of a

γ=1.0 a box of assorted donuts with a variety of toppings

desk next to a box

- γ=1.5 a cat sitting on top of a desk
- γ=2.0 a cat sitting on top of files on a cabinet
- γ=3.0 tortoiseshell mittedtabkat sitting inquisitive on papers

- γ=1.5 a box of assorted donuts with different toppings
- γ=2.0 six glazed and chocolate sprinkled doughnuts in a box
- γ=3.0 krispy box of dozen glazed and decorated doughnuts

GT Cat sitting next to remote control on small counter.

GT Half a dozen donuts from Krispy Kreme of various different flavors.

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

γ=1.0 a cat sitting on a desk

γ=1.0 a large brown and black insect on top of a laptop

γ=1.0 a red traffic light sitting on the side of a road

γ=1.0 a stone wall with a clock

tower and a stone wall

next to a laptop

- γ=1.5 a cat sitting on a desk next to a laptop
- γ=2.0 cat sitting on desk looking at lap top screen
- γ=3.0 calico laptop sitting on computer desk with calico cat sitting on top of screen

- γ=1.5 a bug sitting on the edge of a laptop
- γ=2.0 dragonfly perched on television outside on patio
- γ=3.0 dragonfly perched on television outside on cantilever table

- γ=1.5 a traffic light with a red pedestrian crossing sign on it
- γ=2.0 red traffic light sitting on the side of a street
- γ=3.0 pedestrian signal red on a black light pole

- γ=1.5 ruins of a building with people walking around
- γ=2.0 ruins at a castle in turkey
- γ=3.0 ruins at diocletianopolis roman ruins

GT A city made out of stone brick with large arches.

GT A bug sitting on the side of a laptop computer.

GT A red traffic light with a sad face drawn over it.

GT A cat standing on top of a laptop computer.

[Figure 33]

γ=1.0 a pizza that is sitting on a

pan

- γ=1.5 pepperoni pizza on metal pan with cutter
- γ=2.0 pepperoni pizza on metal pan with cutter
- γ=3.0 pepperoni steel traybake pepperoni steel tray pizza cutter pepperoni steel tray

GT A pan with three pieces of pepperoni pizza.

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

γ=1.0 a basket of bananas and

γ=1.0 a giraffe standing in a grassy area next to a rock wall

γ=1.0 a bird is standing on the

γ=1.0 a group of teddy bears sitting on a bed

γ=1.0 two black suitcases are

γ=1.0 a cat sitting on a bed next

sitting next to each other

to a blanket

ground in the grass

coconuts on a table

- γ=1.5 coconut and bananas in a basket with a banana inside
- γ=2.0 coconut basket with bananas and nuts in it
- γ=3.0 coconut basket bananas coconut husknus and husk laid out

- γ=1.5 weeds and rocks in a grassy area with dirt
- γ=2.0 weeds and rocks litter a gravel path in a grassy area
- γ=3.0 weeds and gravel strewn away along gravel trail strewn with bird rocks

- γ=1.5 three teddy bears sitting on a bed together
- γ=2.0 four teddy bears sitting on a bed together
- γ=3.0 cuddling teddy bears lay piled on a sofa

- γ=1.5 two suitcases with wheels on white background
- γ=2.0 two suitcases facing each other 3d illustration
- γ=3.0 cgi suitcases rendered cgi cgi cgi looks like luggages cgi cgie cgih cgih cgih cgih cgih cgih cgih cgih cgih cgih cgih cgih cgih travelshpinky like nexushxm gif 3dding

- γ=1.5 a cat sitting on a bed under a blanket
- γ=2.0 a tabby kitten sitting on top of a comforter on a bed
- γ=3.0 tabby kitten sitting on uncovered rumple drapes on unmade unmade unmade bed

- γ=1.5 a giraffe standing in a grassy area next to a rock wall
- γ=2.0 giraffe standing in enclosure near trees and rock wall
- γ=3.0 girafe confined motionless zoo confined wild confined into captivity

GT Three different teddy bear on a blanket on a chair.

GT A bird walking through some gravel as its baby chicks follow.

GT a basket with a few things of fruit in it

GT A brown and white cat lying on a bed

GT A giraffe walking through a lush green field.

GT A couple of pieces of very nice looking luggage.

[Figure 40]

γ=1.0 a cake with a dog and

horse on it

- γ=1.5 a cake with dogs and horses on it
- γ=2.0 cake decorated dog horse and dog motif with three horses
- γ=3.0 cake puppy horse dog dog and cats decorated for a first birthday

GT A cake that has paw prints and miniatures dogs on it.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

γ=1.0 a city street with a clock

γ=1.0 a table with a keyboard a cup of coffee and a keyboard

γ=1.0 two cats sitting on a rug in

γ=1.0 a sandwich and a drink in a basket on a table

γ=1.0 a plate of food with a sandwich and a drink

γ=1.0 a wooden bench sitting in

γ=1.0 a hot dog and a mustard bottle on a table

tower and cars

a room

the middle of a forest

- γ=1.5 a city street at night with cars and buildings
- γ=2.0 cars are driving down a busy city street at night
- γ=3.0 ginza at night with cars lights and edifice in asia

- γ=1.5 two cats sitting on a rug in a room
- γ=2.0 two cats sitting on rug in room with orange carpet
- γ=3.0 cats sitting next to each other on patterned carpet

- γ=1.5 a sandwich and a drink in a basket on a table
- γ=2.0 sandwich basket with drink and pickle relish
- γ=3.0 sandwich basket drink relish relish pickle hot dog and drink

- γ=1.5 tater tots and a sandwich and tater tots are on a paper plate
- γ=2.0 tater tots toast and a beer on a restaurant table
- γ=3.0 tater tots toast club sandwich tater tots and beer on a restaurant table

- γ=1.5 a bench sitting in the middle of a hedge
- γ=2.0 hedges and bench in a forested area
- γ=3.0 hedges hedge bench hedges bush hedges

- γ=1.5 a hotdog and mustard are on wax paper next to a counter
- γ=2.0 hot dog and mustard candles on wax paper
- γ=3.0 dug hot dog and mustard candles on wax paper under counter

- γ=1.5 a keyboard coffee cup and glasses on a table
- γ=2.0 keyboard coffee sunglasses pen and cup on outdoor table
- γ=3.0 keyboard coffee sunglasses pen wallet keyboard starbucks cup on outdoor table

GT The traffic and people on a commercial street corner at night

GT A black cat and an orange cat are sitting on the floor.

GT A bench out by a hedge by the woods

GT A hotdog with toppings served in a red basket

GT A tray with a cheese and meat sandwich with tater tots.

GT Two hot dogs sitting on top of tissue paper.

GT a keyboard on a table with a toothbrush a book some sunglasses and coffee

Figure C.1. Additional examples of captions generated with classifier-free guidance at different strengths.

