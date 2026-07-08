[Figure 1]

## Parrot Captions Teach CLIP to Spot Text

Yiqi Lin1* Conghui He1*† Alex Jinpeng Wang2* Bin Wang1* Weijia Li3 Mike Zheng Shou2

1 Shanghai AI Laboratory 2 Show Lab, National University of Singapore 3 Sun Yat-Sen University

# arXiv:2312.14232v3[cs.CV]1Feb2024

https://linyq17.github.io/CLIP-Parrot-Bias/

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

##### Simply Do Text Spotting?

[Figure 6]

[Figure 7]

Alignment !!!

1 2 3 4

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

5 6 7 8 9

(a) Images from LAION-2B

- 1). Year of Yes: How to Dance It Out, Stand In the Sun and Be Your Own Person by Shonda Rhimes.
- 2). 10 Ways to Teach Your Toddler to Listen- excellent advice from Dr. B, a school psychologist.
- 3). Download 2018 Ford F150 3.5L Ecoboost vs 5.0L V8 Coyote Drag Race! It's Kunes Country Prize Fights! Video.
- 4). Kids Again (feat. Amy Allen) by Artist Vs Poet.
- 5). Modoc: The True Story of the Greatest Elephant That Ever Lived, Ralph Helfer.
- 6). Everton Mints 150g Jar.
- 7). National Association of Student Financial Aid Administrators Presents 2015 NASFAA What You Need to Know About Financial Aid.
- 8). Hake's - BILL GRAHAM FILLMORE EAST CONCERT POSTER FEATURING MOTHERS OF INVENTION.
- 9). \"\u2022 #LEGO #NINJAGO [ \"\"Nobody is perfect. I am nobody, so I am perfect\"\" ] #quote \u2022 #Kai #KaiSmith \u2022 My Edit. Hope you'll like it! :-)\".

##### CLIP Score Top5% in LAION-2B

(b) Image Paired Captions

Figure 1. In LAION-2B [33], image-text pairs with the Top-5% highest similarity score are most dominant by visual text! These samples have dense concurrent text appearing in captions and images (text form in pixels). We refer to their captions as Parrot Captions as they raise a question: Dose CLIP Simply Parroting Text in Images for Vision-Language Alignment? The concurrent text is spotted by the OCR model and highlighted with color in image-text pairs. (Best view in color)

#### Abstract

Despite CLIP being the foundation model in numerous vision-language applications, the CLIP suffers from a severe text spotting bias. Such bias causes CLIP models to ‘Parrot’ the visual text embedded within images while dis-

*Equal contribution. †Corresponding author.

regarding the authentic visual semantics. We uncover that in the most popular image-text dataset LAION-2B, the captions also densely parrot (spell) the text embedded in images. Our analysis shows that around 50% of images are embedded with visual text content, and around 30% of captions words are in these embedded visual content. Based on such observation, we thoroughly inspect the different re-

leased versions of CLIP models and verify that the visual text is the dominant factor in measuring the LAION-style image-text similarity for these models. To examine whether these parrot captions shape the text spotting bias, we train a series of CLIP models with LAION subsets curated by different parrot-caption-oriented criteria. We show that training with parrot captions easily shapes such bias but harms the expected visual-language representation learning in CLIP models. This suggests that it is urgent to revisit either the design of CLIP-like models or the existing imagetext dataset curation pipeline built on CLIP score filtering.

#### 1. Introduction

Recently, contrastive learning models [17, 30, 33] pretrained with large-scale image-text pair data has led to numerous vision-language modeling task breakthroughs. Due to its efficiency and simplicity, the pioneering work CLIP [30] now serves as a foundation model in various applications [20, 26, 31, 48]. However, several works [2, 12] have shown that the CLIP models have perpetuating biases towards visual text [19, 25], color [35, 43], gender [40], etc. In this paper, we focus on probing the visual text bias in CLIP, i.e., the capacity of spotting text in images. Most of the previous cues [25, 30, 35] attribute the sources of biases to the noisy pre-training data. Therefore, we begin by taking a close look at the most popular dataset, LAION-2B [33].

Considering the massive scale of the image-text data, it is non-trivial to assess the bias simply with a rough estimation. To this end, we first do image clustering on the whole dataset and rank each cluster by the CLIP scores to analyze the most preferred types of image-text pairs under CLIP score measurement. As shown in Fig. 1, we surprisingly observe that a decent number of samples with top CLIP scores have dense concurrent text appearing in the captions and the images in the form of pixels. These samples break the assumption that the CLIP models leverage text supervision to align the visual and language concepts. We refer to these captions as Parrot Captions as they provide another shortcut to achieve the same goal by teaching the CLIP to do text spotting even without perceiving the actual visual concepts. To understand the underlying impact, we analyze the parrot captions from three perspectives: dataset, widely used released models, and model training.

Our main contributions are three-fold:

- 1. Captions in LAION-2B have a significant bias towards describing visual text content embedded in the images. We provide thorough profiling using off-theself text spotting models on the LAION-2B dataset and show that over 50% of the images are embedded with visual text content. Moreover, by examining the spotted text content and the paired caption in each image-

text pair, we find that over 90% of the captions at least have one concurrent word and reach at least around 30% words overlap between the caption and spotted text from images. This finding suggests that the basic assumption of image-text semantic alignment in CLIP does not stand its ground when training with LAION-style data.

- 2. Released CLIP models have strong text spotting bias almost in every style of web images, resulting in the CLIP-filtering datasets inherently biased towards visual text dominant data. We investigate the OpenAI released CLIP model’s behaviors in the LAION-2B dataset by examining the difference between alignment scores before and after text removal. The results show that the CLIP model predictions densely correlate the visual text embedded in images with their parrot captions. Next, we further study the preference of the text spotting capacity on text content in the CLIP and OpenCLIP models. Note that the CLIP is trained on WIT-400M while the OpenCLIP uses the LAION-2B dataset. Therefore, we use synthetic images embedded with specific rendered text to avoid overfitting in OpenCLIP. Our analysis shows that the OpenCLIP model is more biased towards text spotting than the CLIP model. We believe that the parrot caption plays a lurking role in training these released CLIP models and is the source of text spotting capacity instead of emergence [41] in language models.
- 3. CLIP models easily learn text spotting capacity from parrot captions while failing to connect the visionlanguage semantics, just like a text spotting parrot. We sample different LAION-2B subsets curated by textorientated criteria, including the embedded text ratio, the concurrent word ratios, and the relative CLIP score from text removal to train CLIP models under the same setting. The results show that using parrot captions data, the CLIP model can learn strong text spotting capacity but lose most of the zero-shot generalization ability on image-text downstream tasks. Lastly, we argue that the existing data curation pipeline built on the CLIP score and the contrastive fashion urgently needs to be reexamined by considering such hidden parrot captions.

#### 2. Related Work

###### 2.1. Contrastive Vision-Language Pre-training

Modeling vision and language by aligning the embedding similarity between paired image-text data [17, 30, 33] has shown great potential for transferable to downstream vision-language tasks. The pre-training techniques mainly contain the vision encoder [9, 15] for image embedding encoding, text encoder [8] for text embedding modeling, and cross-modal contrastive learning [17, 21, 30, 46] for learning a joint embedding space of vision and language. The pioneering work CLIP [30] leverages 400 million noisy

Algorithm 1 Pseudocode of Detecting Co-Emb. Text (Rate)

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

# caption: captions from LAION-2B dataset. # ocr_text: text spotted by OCR model. cap_words = set(caption.split()) ocr_words = set(ocr_text.split()) co_emb_text = intersection(cap_words, ocr_words) co_emb_text_rate = len(co_emb_text) / len(cap_words)

Raw Image

All-Emb. Text Removal

Co-Emb. Text Removal

Random Removal

Syn-Emb. Text Template

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Waterfall restaurants in the Philippines WTF fun facts

searching advanced selection strategy to improve the data quality from common crawl data pool gains a growing interest [10]. Recently, several works [5, 24, 29] suggest that introducing text-related filtering methods improves the pretraining dataset quality. In DiHT [29], the data curation steps include filtering out the image-text pairs with high OCR confidence and matching text ratio. Moreover, [5, 24] mainly focus on studying the importance of filtering out the text-dominate images utilizing OCR models to improve pretraining dataset quality. Maini et al. [24] also draw the observation that 40% of LAION’s image text is highly correlated with the caption, but only performing a small pilot study on 500 samples with manual judgment. Differently, this paper makes the first attempt to reveal the source of text spotting capacity in CLIP is the data bias and the consequences of such bias in existing commonly used datasets.

Caption Co-Emb. Text Mask

All-Emb. Text Mask

Random Sampled Mask

Syn-Emb. Text Template

- Figure 2. Visualization of defined terminologies. Co-emb. text is highlighted in the caption with colors.

image-text pairs to learn transferable visual representation from text supervision and show impressive zero-shot performance for various vision-language tasks. Following the CLIP, several vision-language models such as ALIGN [17], BASIC [28], and Open-CLIP [33] are proposed, and the CLIP models have been replicated on various datasets including WIT [30], LAION [33], COYO [4], and DataComp [10]. We mainly profile the LAION-2B [33] dataset due to its large scale and wide usage [26, 31] and two versions of pre-trained models, CLIP and OpenCLIP. Note that the 2 billion image-text pairs in the LAION-2B dataset are filtered by OpenAI released CLIP models, making the OpenCLIP connect to CLIP closely.

#### 3. Terminology

The data processing on images in Sec. 4, 5, 6 mainly cover clustering, text spotting (OCR), and text inpainting. Firstly, we cluster all images based on feature similarity. For each image-text pair, we then use the pre-trained text spotting model to detect and recognize the text print in image pixels. The mask images in Fig. 2 are the spotted text area. Next, we match the spotted text with the caption using Algorithm 1 to obtain the concurrent words and their ratio in captions. Lastly, we use inpainting to remove the text from the image for the CLIPs’ pattern ablation. To avoid confusion, we define these concepts as follows,

###### 2.2. Studying of CLIP Behaviors

Despite the strong zero-shot and transferable performance of CLIP, the perpetuating biases [1, 12, 19, 40, 45] in CLIP are still not well investigated due to its large-scale noisy training data. Much research [25, 35, 42, 43] focuses on revealing or enhancing the downstream performance with discovered bias in CLIP. For example, colorful masks [43] or red circles [35] applied to images can improve the zeroshot performance on visual grounding or keypoint localization tasks. In studying visual text content bias, [12] shows the multimodal neurons of CLIP not only respond to visual content and the visual text embedded in the image. Another work [19] shows that image recognition in CLIP can be strongly dominated by the visual text embedded in the image. To disentangle such bias, [25] attempts to separate the text spotting representation in pre-trained CLIP by training representation projection. Meanwhile, LoGoPrompt [34] enhances the classification performance by utilizing the visual text content as auxiliary prompts as input. Also, CLIPPO [39] shows that directly aligning the image and synthetic images embedded with the captions can perform similarly to CLIP without a text-specific encoder.

- • Embedded Text: text spotted by OCR models from the images. To study the correlation of embedded text with captions, we define different kinds of embedded text as,

- – All-Emb. Text: all the text is spotted from an image.
- – Co-Emb. Text: spotted text concurrently appears in the image’s corresponding captions.
- – Syn-Emb. Text: synthetic text rendered in an image with a fixed font and a blank background.

Fig. 2 shows examples of spotted embedded text by binary mask and the rendering results of synthetic text.

- • Co-Emb. Text Rate (CoTR): the word set IoU of CoEmb. text and captions (See Algorithm. 1).
- • Parrot Caption: captions with CoTR > 0.
- • Image w/ or w/o Embedded Text: spotted text results of a given image are none-empty or empty.
- • Text Removal Image: do inpainting in the specific spot-

###### 2.3. Data Curation with Text Removal

Due to the successful practice of data curation in LAION datasets [32, 33] on scaling up the image-text datasets,

[Figure 22]

(a) The ratio of different OCR-oriented data types in LAION-2B clusters.

[Figure 23]

(b) Top CLIP score samples visualization from 50 clusters with

ratio over 80%.

| |
|---|

- Figure 3. (a): Based on the OCR prediction results, the image-text pairs are divided into three types:

| |
|---|

image without visual embedded text content;

| |
|---|

the spotted text from the image has no concurrent text with the caption;

| |
|---|

the spotted text at least share one concurrent word with the caption. The clusters are merged from 4000 into 100 for a better view. (b): In the clusters with high

| |
|---|

ratio, the top CLIP score samples contain various text sources, such as posters, book covers, advertisements, TV show screenshots, and even PowerPoint slides.

ted text area (All-Emb., Co-Emb., or Random). The random is implemented by sampling other image’s text areas. For the different inpainting results, see Fig. 2.

- • Relative Scores (RSA/RSC): the difference of the CLIP score between images modified by different inpainting operations while keeping the same captions. RSA and RSC are the short for the relative scores before and after removing All-Emb. text and Co-Emb. text.
- • Image Clusters: image partitions based on K-Means.
- • CLIP and OpenCLIP: the CLIP models are trained on WIT-400M [30] and LAION-2B [33] dataset.
- • N-gram Vocabulary (Vocab): the set of all contiguous N word sequences extracted from a text corpus, such as the collection of all captions or embedded text.

- 4. Profiling LAION-2B Data

ing captions to bring out our observations.

###### 4.1. Implementation Details

Clustering with CLIP Features: We first train K-Means (implemented by Faiss [18]) on the LAION-400M [32] subset using ViT-B-32 [9] CLIP features to speed up the clustering process. Due to the large memory consumption, we reduce the feature dimensions from 512 to 256 using PCA (implemented by scikit-learn [27]). Then, we scan and partition the whole dataset using trained K-Means with the same feature extraction pipeline.

Text Spotting and Matching: To detect and recognize text across various scenes, we adopt DeepSolo [44] as our textspotting model and use the pre-trained checkpoints with the ViTAEv2-S [47] backbone in default setting. The output format of the text spotting model is a sequence of polygons of text location and their recognized characters. Despite its strong performance, we empirically find that DeepSolo can not handle the crowd scenes well (with more than 100 separate words) but is only a small proportion of the dataset (∼2%). To identify the correlation between the spotted text and captions, we use Algorithm 1 to calculate the CoEmb. text rate in each image-text pair. Considering the predictions that the text spotting model might miss or misspell

To better profile the image-text pair data on a billion scale, we first cluster all the images based on CLIP features into 4,000 clusters and sort each cluster with CLIP scores. After obtaining all the cluster labels, we use the SOTA text spotting model [44] to get the visual text content on all the collected images. Finally, we aggregate all the modelpredicted results and compare them with their correspond-

|Num. of Total Img. Num. of Img. w/ Emb. Text Num. of Img. w/ Co-Emb. Text|1,985,284,122 1,083,896,427 663,600,432<br><br>|
|---|---|
|Co-Emb. Text Rate (in Total)<br><br>– (in Img. w/ Emb. Text) Fuzzy Co-Emb. Text Rate (in Total)<br><br>– (in Img. w/ Emb. Text)<br><br><br>|15.42% 28.24% 30.46% 55.79%|

Table 1. Overall correlation statistic between spotted text and captions in the LAION-2B. More than 50% of images are embedded with text, and 30% of caption words are printed in images!

words in complex scenes, we also use Levenshtein distance to calculate the fuzzing similarity and reported in Tab. 1.

###### 4.2. Statistic and Observations from LAION-2B

The overall statistics of the 2 billion image-text pairs are reported in Tab. 1. In summary, the images embedded with visual text content reach a surprisingly high proportion of 54.60% in the investigated data. Around 15% of words in the dataset captions are Co-Emb. text, and the proportion of Co-Emb. text can further reach 30% when considering the fuzzy matching results of the spotted text and captions. This suggests that the CLIP models trained on these data might lead to a high bias toward text spotting.

To better visualize the data distribution, we provide cluster-specific statics results and top CLIP score samples of text-dominated clusters in Fig. 3. We divide all images into 100 clusters based on visual similarity and classify

- them into three data types according to the OCR results. Every cluster contains more or less images embedded with text. Combined with sample visualization, we observe that in the LAION collected data, the parrot captions cover various scenes. In the subsets of images embedded with text, around 60% of captions at least precisely parrot one concurrent word (Co-Emb. Text Rate > 0) appearing in the image. It suggests that the data collection pipeline of LAION [33] has a strong bias to introduce parrot captions from web data.

To better understand Co-Emb. Text, we provide a more thorough analysis of the word counting and text size of parrot captions. As shown in Fig. 4a, the results show that a large proportion of the Co-Emb. Text only takes a few words. However, we also find a large number of captions that are almost full parrot captions (see areas around the heatmap diagonal). Next, in Fig. 4b, we investigate the correlation between the size of concurrent words and CLIP score. The results show that the large text size does not usually lead to a higher score; meanwhile, the small text size can also dominate the score. One possible reason is the text content and input resolution may matter more for CLIP. Moreover, we discover that the larger text is more likely to be parroted in the captions, as shown in Fig. 4c.

[Figure 24]

[Figure 25]

[Figure 26]

0.10

AreaRatio:Box/Image

0.08

NumCo-EmbWord

CLIPScore

0.06

0.04

0.02

0.00

Co-Emb Others

Num Caption Word

Text Area Ratio: Sum(Boxes) / Image

Text Type

(a) (b)

(c)

Figure 4. (a): Visualization of the number of caption words and associated spotted concurrent words based on precise word matching. (b): Distribution of total area of concurrent words placed in the image and its ViT-B CLIP score. (c): Distribution of text size of the single concurrent word and other spotted word.

#### 5. Inspecting Pre-Trained CLIP Models

It is important to note that the LAION-2B dataset collection pipeline uses the CLIP score from OpenAI’s model to filter out the image-text pair below 0.28. Therefore, we inspect these two released CLIP models [30, 33] to answer better why LAION data contains such a high proportion of parrot captions. Specifically, the OpenAI’s CLIP model is trained on the WIT dataset (out-of-domain model), and the OpenCLIP is trained on LAION-2B (in-domain model). We first study whether the embedded text is the key factor in CLIP filtering by ablating the embedded text using text inpainting. Moreover, we further investigate whether the text spotting capacity prefers specific text content by examining synthetic images with Syn-Emb. text.

###### 5.1. Ablation of Embedded Text Removal

Text Removal via Inpainting: Given the OCR predicted results, we use the fast marching method [38] to do text inpainting in the area of the spotted text polygons. Accordingly, we generate two versions of text removal results for each image with embedded text, i.e., All-Emb. text removal and Co-Emb. text removal, as the parrot caption prediction is imperfect due to the limitation of OCR models. We also generate random inpainting images with randomly sampled spotted text polygons from other images to ablate the information loss caused by image inpainting. The spotted text masks and inpainting results are shown in Fig. 2.

Results: Based on the OCR predicted results and text inpainting operations, we can obtain six types of LAION images, including •): images without any embedded text (OCR results are empty); •): images with any embedded text (OCR results are none-empty); ×): images removed All-Emb. text (Inpaint all the areas of OCR predicted text); ■): images removed Co-Emb. text (Inpaint the areas of concurrent text in OCR predicted text and captions); ×): images with random inpainting by other image’s All-Emb. text area, and ■): images randomly inpainted by other image’s Co-Emb. text area. Then, we calculate the CLIP scores of all the groups of images and their paired captions using

[Figure 27]

[Figure 28]

Avg.Image-TextSimilarityScores

RelativeSimilarityScores

S(■)-S(×)

S(●)-S(×) S(●)-S(■)

Cluster Index Per Sample Comparison

Figure 5. Left: Mean CLIP scores of image-text pairs with different text removal operations depicted in Sec 5.1, and the data are grouped by cluster the same as Fig. 3. Right: Overall relative CLIP score distribution by comparing different text removal operations.

[Figure 29]

[Figure 30]

|Setup<br><br>|CLIP Score|
|---|---|
|• Raw w/o Emb. Text<br><br>• Raw w/ Emb. Text<br><br><br>|0.3223 ± 0.0078 0.3358 ± 0.0094|
|× Random All-Emb. Text Removal × All-Emb. Text Removal|0.3260 ± 0.0057 0.2974 ± 0.0197<br><br>|
|■ Random Co-Emb. Text Removal<br><br>■ Co-Emb. Text Removal<br><br><br>|0.3341 ± 0.0051 0.2993 ± 0.0146|

Frequency

Frequency

Table 2. Mean CLIP score of different setups of text removal.

OpenAI released CLIP model (ViT-B-32). Fig. 5 reports the mean scores of different types of images in each cluster and raises four observations as follows,

1-gram Index Text Length

(a) Statistic of 1-gram vocabularies.

[Figure 31]

- • The images embedded with text achieve higher CLIP scores in most clusters than those without embedded text.
- • The CLIP scores significantly drop once we remove the text from the images compared to its random inpainting baseline. It indicates that the parrot captions correlate highly with the CLIP score measurement.
- • The text spotting mechanism of CLIP might be similar to Bags-of-Words [45]. Most of the relative CLIP scores (S(■) - S(×)) between images removed Co-Emb. text and All-Emb. text are positive, as shown in the right of Fig. 5. The straightforward reason is the images lose more visual information due to the larger in-painting area, while another possible reason is the imperfect text spotting prediction or the corner cases in the matching algorithm leaking parts of the concurrent text in images.
- • Not all the samples are dominated by the embedded text, as some samples achieve higher scores after removing text, indicating the embedded text also can be a distractor.

Image-TextSimilarityScores

1-gram Frequency Groups

(b) Grouped by 1-gram frequency intervals.

Figure 6. OpenCLIP more bias than the CLIP model. Grouped score distributions of prompting CLIP and OpenCLIP models with N-gram Syn-Emb. text and synthetic images.

formation loss in inpainted regions.

###### 5.2. Prompting with Syn-Emb. Text

Discussion: Due to the text removal, the image distribution may shift from the CLIP training set. Therefore, we provide two random removal baselines to examine the effect of distribution shift. In Tab. 2, we report the mean scores of different setups. Results show that the random baselines are very close to the raw image baseline, indicating that the CLIP model is robust to the distribution shift caused by in-

Generating Synthetic Images from N-gram Vocabulary: To investigate the CLIP models’ text spotting preference, we adopt a similar strategy in [25] to use synthetic images to embed specific text content by rendering text in a blank background. For each text, we use four fore-background style rendering templates (black-white, black-grey, whitegrey, and white-black), as shown in Fig. 2. Different from

the uniformly sampling letters in [25], we generate the text content from the N-gram vocabulary built from captions and Co-Emb. text to study the text spotting pattern. We only select the top frequent 400,000 grams for each vocabulary. The statistics of 1-gram vocabulary are shown in Fig. 6a, which is a long-tail distribution. Next, we calculate the synthetic images and their rendered text similarity on released ViT-B-32 CLIP and OpenCLIP models.

Results: Firstly, we examine whether the CLIP models prefer recognizing more commonly seen words (with high frequency in vocabulary). Therefore, we group the 1-gram results based on their frequency interval in the whole vocabulary, as shown in Fig. 6b. The results show that the OpenCLIP model clearly has a stronger text spotting capacity than CLIP, i.e., more biased towards text spotting. We also observe that all the CLIP models are more sensitive to the vocabulary built from the concurrent words. Interestingly, both CLIP and OpenCLIP models have slightly higher scores on the less frequent grams. Secondly, considering the long-tail grams might contain more characters, we further group the 1-gram and 2-gram results based on their text length in Fig. 7a and Fig. 7b. Note that the CoEmb. text is not regularly arranged in the images, making it hard to extract continuous word sequences. Results show that all the models are better at spotting the longer words, possibly due to the tokenizer used in the text encoder, making them more discriminative. Meanwhile, in the groups of 2-gram samples, the scores gradually drop when spotting the extremely long text, indicating that the spotting capacity of CLIP models is possibly built on word-by-word.

#### 6. Training CLIP on Emb. Text Curated Data

Next, we dive deeper into the parrot captions by training CLIP models on LAION-2B subsets selected by different embedded-text-oriented criteria under the same setting.

###### 6.1. Experiment Setups

Training Setting and Implementation Details: We use the open-source software OpenCLIP [16] for all the CLIP model training. Our experiments are conducted on ViTB [9] and RN50 [15] using 8 NVIDIA A100 GPUs for training. We use 4,096 batch size for 3M and 8,192 for 12M scale subsets. Other settings remain the same as [33].

Evaluation: We follow the DataComp benchmark [10] using 38 zero-shot classification and retrieval tasks as evaluation. We report the average performance (Avg.) of the DataComp benchmark and two subset track performances, ImageNet (IN) and Retrieval (Ret.). To evaluate the text spotting capacity, we use a synthetic benchmark the same as in Sec. 5.2 and a real-world benchmark sampled from LAION-2B as the validation set. In the synthetic benchmark, we calculate the similarity of all the 1-gram synthetic image-text pairs from caption vocabulary and report all the

[Figure 32]

Image-TextSimilarityScores

- Text Length (1-gram Caption Vocab)

(a) Grouped by Caption 1-gram length.

[Figure 33]

- Text Length (2-gram Caption Vocab)

Image-TextSimilarityScores

(b) Grouped by Caption 2-gram length.

Figure 7. CLIPs are better at spotting the longer words. Score distributions of N-gram Syn-Emb. text grouped by text length.

|Data|Model<br><br>|IN Ret. Avg.|
|---|---|---|
|3M Random 3M w/o Emb. Text 3M w/ Emb. Text Only<br><br>|RN50 RN50 RN50<br><br>|0.204 0.222 0.256 0.228 0.232 0.282 0.071 0.139 0.164<br><br>|
|3M Random 3M w/o Emb. Text 3M w/ Emb. Text Only|ViT-B ViT-B ViT-B<br><br>|0.131 0.148 0.210 0.162 0.164 0.234 0.052 0.111 0.153<br><br>|
|12M Random 12M w/o Emb. Text 12M w/ Emb. Text Only<br><br>|RN50 RN50 RN50|0.360 0.354 0.354 0.409 0.361 0.372 0.129 0.192 0.218<br><br>|
|12M Random 12M w/o Emb. Text 12M w/ Emb. Text Only|ViT-B ViT-B ViT-B<br><br>|0.314 0.299 0.351 0.370 0.318 0.364 0.129 0.172 0.225<br><br>|

Table 3. Ablation of images embedded with or without text. The model trained on data without embedded text performs best on all tasks, while the data with embedded text damages the generalization capacity of learned representations.

trained model results in Fig 8. For the real-world benchmark, we sample 1M image-text pairs without any embedded text and 1M samples dominated by the parrot caption (the relative scores between raw and Co-Emb. text removal images higher than 0.2). Fig. 9 aggregates the mean scores of the 2M evaluation set and also reports the mean scores of applying text removal on the 2M evaluation set results.

[Figure 34]

Image-TextSimilarityScores

Rand w/o Emb.

w/ Emb. Only

CoTR =0.0

CoTR ≥0.3

CoTR ≥0.5

CoTR ≥0.8

CoTR =1.0

RSA <0.0

RSA ≥0.0

RSA ≥0.1

RSA ≥0.2

RSA ≥0.3

RSC <0.0

RSC ≥0.0

RSC ≥0.1

RSC ≥0.2

RSC ≥0.3

Figure 8. CLIP models learn text spotting well from parrot captions. Benchmarking text spotting capacity of CLIP models with 1-gram caption vocabulary synthetic images dataset as the same as Sec. 5.2.

|Data (3M)|Model|IN Ret. Avg.|
|---|---|---|
|CoTR = 0.0 CoTR ≥ 0.3 CoTR ≥ 0.5 CoTR ≥ 0.8<br><br>CoTR = 1.0<br><br><br>|RN50 RN50 RN50 RN50 RN50<br><br>|0.193 0.229 0.247 0.031 0.110 0.137 0.021 0.099 0.124 0.012 0.082 0.096 0.012 0.074 0.102<br><br>|
|CoTR = 0.0 CoTR ≥ 0.3 CoTR ≥ 0.5 CoTR ≥ 0.8<br><br>CoTR = 1.0<br><br><br>|ViT-B ViT-B ViT-B ViT-B ViT-B<br><br>|0.132 0.164 0.206 0.029 0.084 0.130 0.021 0.082 0.119<br><br>0.012 0.076 0.104<br>0.013 0.076 0.103<br>|

|Data (3M)<br><br>|Model|Avg.S(•)<br><br>|IN Ret. Avg.|
|---|---|---|---|
|RSC < 0.0<br><br>RSC ≥ 0.0<br><br>RSC ≥ 0.1<br><br>RSC ≥ 0.2<br><br>RSC ≥ 0.3<br><br><br>|RN50 RN50 RN50 RN50 RN50<br><br>|0.326 0.345 0.354 0.364 0.380|0.125 0.171 0.209 0.062 0.129 0.168 0.014 0.091 0.106 0.008 0.084 0.104 0.005 0.058 0.084<br><br>|
|RSC < 0.0<br><br>RSC ≥ 0.0<br><br>RSC ≥ 0.1<br><br>RSC ≥ 0.2<br><br>RSC ≥ 0.3<br><br><br>|ViT-B ViT-B ViT-B ViT-B ViT-B|0.326 0.345 0.354 0.364 0.380<br><br>|0.079 0.129 0.174 0.045 0.119 0.149 0.018 0.091 0.116 0.008 0.076 0.106 0.004 0.059 0.091<br><br>|

- Table 4. Ablation of different Co-Emb. Text Rate(CoTR). The fewer parrot captions, the better downstream task performance.

|Data (3M)|Model<br><br>|Avg.S(•)|IN Ret. Avg.|
|---|---|---|---|
|RSA < 0.0<br><br>RSA ≥ 0.0<br>RSA ≥ 0.1<br>RSA ≥ 0.2<br>RSA ≥ 0.3<br>|RN50 RN50 RN50 RN50 RN50<br><br>|0.319 0.339 0.351 0.360 0.376<br><br>|0.181 0.220 0.239 0.126 0.180 0.215 0.041 0.123 0.148 0.017 0.094 0.109 0.009 0.075 0.097<br><br>|
|RSA < 0.0<br><br>RSA ≥ 0.0<br>RSA ≥ 0.1<br>RSA ≥ 0.2<br>RSA ≥ 0.3<br>|ViT-B ViT-B ViT-B ViT-B ViT-B<br><br>|0.319 0.339 0.351 0.360 0.376|0.123 0.159 0.198 0.079 0.129 0.185 0.031 0.103 0.134 0.012 0.080 0.103 0.006 0.070 0.096<br><br>|

- Table 5. Ablation of models trained on subsets sampled by different RSA. RSA denotes the relative similarity (S(•) − S(×)) of raw S(•) and removed All-Emb. text S(×) images.
- 6.2. Ablation Study on Data Curation

Table 6. Ablation of models trained on subsets sampled by different RSC. RSC denotes the relative similarity (S(•) − S(■)) of raw S(•) and removed Co-Emb. text S(■) images.

[Figure 35]

Avg.Image-TextSimilarityScores

CLIP Baseline

CoTR: [=0;≥0.3;≥0.5;≥0.8;=1]

RSA: [<0;≥0;≥0.1;≥0.2;≥0.3]

RSC: [<0;≥0;≥0.1;≥0.2;≥0.3]

Figure 9. Comparison of mean similarity of LAION-2B subset for text spotting capacity validation. Models trained with more parrot captions are better at aligning the image with parrot captions but perform worse at aligning images without embedded text.

Curation I: Embedded Text in Images. To study the impact of embedded text on overall pre-train data quality, we sample three subsets: random baseline, images without any embedded text, and images all embedded with text from LAION-2B. The subsets include 3M and 12M scales. The results in Tab. 3 show that images embedded with

text generally reduce the pre-training dataset quality as all performance tracks significantly decrease. Meanwhile, in Fig. 8, the model trained with the images embedded with text achieves the strongest text spotting capacity compared to the random and images without embedded text baselines. Curation II: Co-Emb. Text Rate (CoTR). Tab. 3 reports

|BLIP Data (3M)<br><br>|Visual Question Answering (Acc) VQAv2 TextVQA ST-VQA<br><br>|Image Captioning (CIDEr)<br><br>COCO TextCaps<br><br>|Text-to-Image Retrieval(R@1) COCO TextCaps<br><br>|Image-to-Text Retrieval(R@1)<br><br>COCO TextCaps<br><br>|
|---|---|---|---|---|
|Rand w/ Emb. Text w/o Emb. Text<br><br>|71.07 15.36 10.48 68.94 19.05 12.65 71.22 13.65 9.29<br><br>|115.6 53.7 108.9 92.1<br><br>116.2 41.5<br><br><br>|48.91 56.34 42.89 70.1<br><br>49.96 31.83<br><br><br>|65.46 72.45 58.5 81.42<br><br>66.5 48.7<br><br><br>|
|CoTR = 0.0 CoTR ≥ 0.3 CoTR ≥ 0.5 CoTR ≥ 0.8<br><br>CoTR = 1.0<br><br><br>|71.11 13.97 9.75<br><br>67.4 19.28 11.81 67.02 19.64 12.38 66.38 18.50 12.00 66.18 18.47 12.80<br><br>|116.3 44.6 104.9 96.9 102.7 94.1<br><br>100.9 91.6<br><br>101.2 91.3<br><br><br>|49.55 38.05 37.78 67.28 35.94 65.24 34.13 62.65 33.55 61.83<br><br>|66.08 54.57 51.98 78.2 50.32 76.94<br><br>46.9 73.56 46.62 73.05<br><br>|
|RSA < 0.0<br><br>RSA ≥ 0.0<br><br>RSA ≥ 0.1<br><br>RSA ≥ 0.2<br><br>RSA ≥ 0.3<br><br><br>|70.79 14.16 9.64<br><br>70.03 18.76 11.81 68.14 19.48 13.33 66.01 21.06 11.85 64.20 18.44 12.04<br><br>|115.7 44.9 111.9 84.5 105.6 96.1<br><br>98.7 94.4 95.26 91.1<br><br>|48.25 36.85 46.25 68.61 39.96 68.13 33.03 64.17 26.64 60.11<br><br>|64.72 54.7<br><br>62.92 81.23 54.64 79.37 47.12 75.33<br><br>37.3 70.24<br><br>|
|RSC < 0.0<br><br>RSC ≥ 0.0<br><br>RSC ≥ 0.1<br><br>RSC ≥ 0.2<br><br>RSC ≥ 0.3<br><br><br>|70.13 15.19 10.74 68.86 20.12 13.75 67.35 20.54 12.84<br><br>62.62 20.32 13.14<br><br>63.75 18.94 13.03<br><br><br>|112.2 46.7 107.8 93.5 103.4 96.9<br><br>98.7 92.8 92.9 88.7<br><br>|46.8 41.95 42.0 69.78 36.4 66.69<br><br>30.08 61.38 24.23 58.35<br><br>|63.24 58.05 57.42 80.92 51.02 77.79 42.96 71.98 34.72 68.95<br><br>|

Table 7. BLIP downstream tasks performance of pre-training on different curated 3M subsets. The gray color represents tasks requiring the model to read the text from images.

the CLIP models trained on parrot captions with different CoTR. We first select all the images with embedded text and

- then sample images based on the CoTR depicted at Algorithm 1 with different thresholds. With increasing CoTR, all the zero-shot benchmark performance drops significantly. Despite the images in the subset (CoTR = 0) all embedded with text, the pre-trained model performs similarly to the random baseline in 3. It indicates that the parrot caption is more crucial than embedded text in reducing the pre-trained data quality. For the text spotting capacity, Fig. 8 and 9 show that the increasing CoTR does not lead to stronger text spotting capacity, possibly due to the average length of captions decreasing in higher CoTR data.

Curation III: Relative Score from Text Removal. Given the observations in Sec. 5.1, we further select a series of subsets based on the relative score of images before and after text removal. The subsets with higher relative scores are more dominant embedded text (RSA) or parrot captions (RSC) in CLIP score measurement. In Tab. 5 and 6, we report the zero-shot performance of models trained on subsets with different relative score thresholds. The CLIP pre-trained with higher RSA or RSC both get worse downstream performance. Importantly, the average raw CLIP scores S(•) of these subsets have a positive correlation with RSA or RSC, indicating using CLIP scores from a biased pre-trained model as the data filtering strategy can be unreliable. When comparing the RSA and RSC, the results show that the samples dominated by the latter, i.e., parrot captions, are less informative for CLIP training. Moreover, Fig. 8 and. 9 show that the text spotting capacity of CLIP can be further improved by training on the samples using relative scores as data curation criteria against CoTR.

###### 6.3. Ablation Study on Text-Oriented Tasks

Inspired by [11], we further investigate the model behavior on downstream tasks requiring reading text, including Visual Question Answering (VQA), Image Captioning, and Text-Image Retrieval. Specifically, for the text-oriented tasks, we use Text VQA [37] and ST-VQA [3] for VQA, and TextCaps [36] for captioning and retrieval. Moreover, we also provide the same tasks on the datasets that only require the model to see, i.e., the natural image dataset. Similarly, we use VQAv2 [13] for VQA and COCO [7] for captioning and retrieval. We choose BLIP [22] for the ablation study instead of CLIP as it can be directly applied to all these tasks. We first pre-train the BLIP on different subsets with 10 epochs and then finetune 10 epochs for VQA, and 5 epochs for captioning and retrieval. As shown in Tab.7, training BLIPs to spot text can boost their performance on the downstream tasks requiring the model to read but impede the performance of downstream tasks only requiring the model to see, which are consistent with the observation on classification tasks. Nevertheless, when BLIPs mainly focus on reading, e.g. (RSA ≥ 0.3), their text-oriented and natural downstream performance also decreases. In other words, the parrot captions can benefit the text-orient downstream tasks while requiring careful data mixing trade-off.

#### 7. Profiling More Image-Text Dataset

MMC4. Multimodal C4 (MMC4) [49] is a popular imagetext interleaved dataset also built on the CLIP feature matching. A linear assignment algorithm is used to place images into longer bodies of text using CLIP features. Therefore, we profile the MMC4 dataset with the same

|Number of Total Images Number of Images w/ Emb. Text Image w/ Emb. Text Rate<br><br>|527156206 264618807 50.20%|
|---|---|
|Co-Emb. Text Rate (in Total)<br><br>– (in Images w/ Emb. Text) Fuzzy Co-Emb. Text Rate (in Total)<br>– (in Images w/ Emb. Text)<br>|2.88% 15.70% 5.75% 31.28%<br><br>|

- Table 8. Overall parrot captions statistic in MMC4 [49].

|Number of Total Images Number of Images w/ Emb. Text Image w/ Emb. Text Rate<br><br>|9,230,079 3,421,152 37.06%|
|---|---|
|Co-Emb. Text Rate (in Total)<br><br>– (in Images w/ Emb. Text) Fuzzy Co-Emb. Text Rate (in Total)<br><br>– (in Images w/ Emb. Text)<br><br><br>|6.21%<br><br>15.94%<br>16.75% 43.01%<br>|

- Table 9. Overall parrot captions statistic in CC12M [6].

pipeline in Sec. 4 to investigate whether parrot captions commonly exist. Note that, we calculate the assigned text for Co-Emb. text statistics. As shown in Tab. 8, we found that the image distribution in MMC4 is similar to LAION2B, with around 50% of images containing embedded text. Meanwhile, the average captions of MMC4 are much longer than those of LAION-2B, resulting in a lower CoTR than LAION-2B. Nevertheless, how the correlation between embedded text and images affects the interleaved dataset still needs further investigation, which we left for future work.

CC12M. Conceptual 12M (CC12M) [6] dataset is built on a rule-based system without using CLIP models from webpages annotations. We further profile the CC12M to ablate the origin of parrot captions in different data curation pipelines. Tab. 9 shows that the text is commonly embedded in the web images, while the Co-Emb. text rate is more lower than the LAION-2B [33] dataset. Therefore, there is still room for improving the data collection pipelines based on the CLIP model filtering.

#### 8. A Simple Fix with Text-Orientated Filtering

To provide an alternative solution for existing released CLIP models, we further construct a less text-biased LAION-2B subset by filtering the dataset with OCR results. Specifically, we first remove all images with detected text. Then, we filter the image-text pairs with a CLIP score greater than 0.3 and an aesthetics score greater than 0.45 to obtain highquality data. Finally, we perform deduplication in each cluster based on the K-means label predicted in Sec. 4 to obtain the compact filtering dataset with 107,166,507 (100M) samples. Given the curated subset, we train a CLIP model

2). S(●):2.87/S(×):3.93↑

1). S(●):3.69/S(×):4.57↑

Captions:

[Figure 36]

[Figure 37]

- 1). Spider Webs Halloween Treat Tote Bag.
- 2). Young girl doing handstand on meadow at summer evening Stock Image.
- 3). KEEP CALM AND AUDITION FOR THE XMAS FACTOR CHECK YOUR EMAILS.
- 4). NIKE AIR MORE UPTEMPO '96 PRM FLAX PHANTOM.

4). S(●):3.41/S(×):1.93↓

3). S(●):4.72/S(×):2.28↓

[Figure 38]

[Figure 39]

Figure 10. Embedded texts play different roles for visual concepts. S(•) and S(×) denote the CLIP score before and after removing All-Emb. text. OCR results are masked with color.

from scratch following the setting used in [33]. The performance of our trained CLIP is reported in Tab. 10. It indicates that the CLIP model can achieve a high performance while without introducing such text spotting bias. Nevertheless, due to the imperfect OCR results, the subset inevitably contains some parrot captions and brings costly scalability, which we also left for future work. The pre-trained models 1 and the filtered subset 2 are released on OpenDataLab [14].

#### 9. Discussion and Conclusion

The popularity of vision-language contrastive loss stems from its efficiency and simplicity. However, the analysis and experiments we presented show that the embedded text in images and their parrot captions plant significant textspotting bias due to such contrastive fashions. Firstly, almost half of the captions in the widely used LAION-2B dataset are biased towards parroting the embedded text in images. Secondly, the pre-trained CLIP models have strong preferences for the image-text pair with parrot captions, which achieve higher similarity scores than those without. Finally, using data biasing to parrot captions, we can easily train a CLIP model with a strong text-spotting bias. Our work demonstrates the emergency of reviewing the impact of parrot captions in the entire ecosystem of CLIP models.

Here, we further showcase some examples to provide a more complete perspective on their negative impact and functionalities. On the one hand, as shown in Fig. 10, when the embedded text is not directly relevant to visual content, like ‘EMILY’ on the bag and watermark, this text plays a

- 1https://github.com/opendatalab/CLIP-Parrot-Bias
- 2https : / / openxlab . org . cn / datasets / opendatalab -

linyiqi/LAION-text-debiased-100M

|Metric|Ours<br><br>|CLIP OpenCLIP|DC medium DC large|
|---|---|---|---|
|Data|100M (LAION)<br><br>|400M (WIT) 2B (LAION)|128M (DC) 1.28B (DC)<br><br>|
|Sync. Score ↓|0.163 ± 0.065|0.317 ± 0.030 0.368 ± 0.042<br><br>|0.268 ± 0.024 0.338 ± 0.034|
|IN IN dist. shifts VTAB Retrieval Avg. 38 datasets<br><br>|0.526 0.404 0.481 0.421 0.443|0.633 0.666 0.485 0.522 0.526 0.565 0.501 0.560 0.525 0.565<br><br>|0.176 0.459 0.152 0.378 0.259 0.426 0.219 0.419 0.258 0.437|

- Table 10. Comparison of our debiased model and the released pre-trained models. We evaluate on our proposed synthetic (Sec. 5.2) and Datacomp [10] benchmark. For the synthetic benchmark, we use the 1-gram vocabulary built from captions and report the mean and std of the synthetic image-text similarity (Sync. S). We also report the performance of CLIP model trained on medium and large Datacomp [10](DC) pools with no filtering.

strong distractor for the CLIP alignments. On the other hand, parts of web media content and the concept propagation inherently are presented by embedded text, such as slogans and brand logos in Fig. 10. Therefore, our future endeavor involves building a bias-aware data curation pipeline and a more robust training function to mitigate such issues. Again, we urge the community to acknowledge and address these issues with greater attention.

#### References

- [1] Sandhini Agarwal, Gretchen Krueger, Jack Clark, Alec Radford, Jong Wook Kim, and Miles Brundage. Evaluating clip: towards characterization of broader capabilities and downstream implications. arXiv preprint arXiv:2108.02818,

2021. 3

- [2] Hugo Berg, Siobhan Mackenzie Hall, Yash Bhalgat, Wonsuk Yang, Hannah Rose Kirk, Aleksandar Shtedritski, and Max Bain. A prompt array keeps the bias away: Debiasing visionlanguage models with adversarial learning. arXiv preprint arXiv:2203.11933, 2022. 2
- [3] Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Mar¸cal Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In ICCV, pages 4291–4301, 2019. 9
- [4] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/ kakaobrain/coyo-dataset, 2022. 3
- [5] Liangliang Cao, Bowen Zhang, Chen Chen, Yinfei Yang, Xianzhi Du, Wencong Zhang, Zhiyun Lu, and Yantao Zheng. Less is more: Removing text-regions improves clip training efficiency and robustness. arXiv preprint arXiv:2305.05095,

2023. 3

- [6] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pretraining to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3558–3568, 2021. 10, 1
- [7] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick.

- Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 9
- [8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 2
- [9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2, 4, 7
- [10] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. arXiv preprint arXiv:2304.14108, 2023. 3, 7, 11, 2
- [11] Roy Ganz, Oren Nuriel, Aviad Aberdam, Yair Kittenplon, Shai Mazor, and Ron Litman. Towards models that can see and read. arXiv preprint arXiv:2301.07389, 2023. 9
- [12] Gabriel Goh, Nick Cammarata, Chelsea Voss, Shan Carter, Michael Petrov, Ludwig Schubert, Alec Radford, and Chris Olah. Multimodal neurons in artificial neural networks. Distill, 6(3):e30, 2021. 2, 3
- [13] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, pages 6904–6913, 2017. 9
- [14] Conghui He, Wei Li, Zhenjiang Jin, Bin Wang, Chao Xu, and Dahua Lin. Opendatalab: Empowering general artificial intelligence with open datasets. https://opendatalab. com, 2022. 10
- [15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016. 2, 7
- [16] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. 7
- [17] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom

- Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICML, pages 4904– 4916. PMLR, 2021. 2, 3
- [18] Jeff Johnson, Matthijs Douze, and Herv´e J´egou. Billionscale similarity search with GPUs. IEEE Transactions on Big Data, 7(3):535–547, 2019. 4
- [19] Yoann Lemesle, Masataka Sawayama, Guillermo VallePerez, Maxime Adolphe, H´el`ene Sauz´eon, and Pierre-Yves Oudeyer. Language-biased image classification: evaluation based on semantic representations. arXiv preprint arXiv:2201.11014, 2022. 2, 3
- [20] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Rene Ranftl. Language-driven semantic segmentation. In ICLR, 2022. 2
- [21] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. NeurIPS, 34:9694–9705,

2021. 2

- [22] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, pages 12888–12900. PMLR, 2022. 9
- [23] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 1
- [24] Pratyush Maini, Sachin Goyal, Zachary C Lipton, J Zico Kolter, and Aditi Raghunathan. T-mars: Improving visual representations by circumventing text feature learning. arXiv preprint arXiv:2307.03132, 2023. 3
- [25] Joanna Materzy´nska, Antonio Torralba, and David Bau. Disentangling visual and written concepts in clip. In CVPR, pages 16410–16419, 2022. 2, 3, 6, 7
- [26] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2, 3
- [27] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. JMLR, 12:2825–2830, 2011. 4, 1
- [28] Hieu Pham, Zihang Dai, Golnaz Ghiasi, Kenji Kawaguchi, Hanxiao Liu, Adams Wei Yu, Jiahui Yu, Yi-Ting Chen, Minh-Thang Luong, Yonghui Wu, et al. Combined scaling for zero-shot transfer learning. Neurocomputing, 555: 126658, 2023. 3
- [29] Filip Radenovic, Abhimanyu Dubey, Abhishek Kadian, Todor Mihaylov, Simon Vandenhende, Yash Patel, Yi Wen, Vignesh Ramanathan, and Dhruv Mahajan. Filtering, distillation, and hard negatives for vision-language pre-training. In CVPR, pages 6967–6977, 2023. 3
- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language super-

- vision. In ICML, pages 8748–8763. PMLR, 2021. 2, 3, 4, 5, 1
- [31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 2, 3
- [32] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 3, 4
- [33] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NeurIPS, 35:25278– 25294, 2022. 1, 2, 3, 4, 5, 7, 10
- [34] Cheng Shi and Sibei Yang. Logoprompt: Synthetic text images can be good visual prompts for vision-language models. In ICCV, pages 2932–2941, 2023. 3
- [35] Aleksandar Shtedritski, Christian Rupprecht, and Andrea Vedaldi. What does clip know about a red circle? visual prompt engineering for vlms. arXiv preprint arXiv:2304.06712, 2023. 2, 3
- [36] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension. In ECCV, pages 742–758. Springer, 2020. 9
- [37] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, pages 8317–8326, 2019. 9
- [38] Alexandru Telea. An image inpainting technique based on the fast marching method. Journal of graphics tools, 9(1): 23–34, 2004. 5
- [39] Michael Tschannen, Basil Mustafa, and Neil Houlsby. Clippo: Image-and-language understanding from pixels only. In CVPR, pages 11006–11017, 2023. 3
- [40] Jialu Wang, Yang Liu, and Xin Eric Wang. Are genderneutral queries really gender-neutral? mitigating gender bias in image search. arXiv preprint arXiv:2109.05433, 2021. 2, 3
- [41] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682,

2022. 2

- [42] Yichen Xu, Zihan Xu, Wenhao Chai, Zhonghan Zhao, Enxin Song, and Gaoang Wang. Devil in the number: Towards robust multi-modality data filter. arXiv preprint arXiv:2309.13770, 2023. 3
- [43] Yuan Yao, Ao Zhang, Zhengyan Zhang, Zhiyuan Liu, TatSeng Chua, and Maosong Sun. Cpt: Colorful prompt tuning for pre-trained vision-language models. arXiv preprint arXiv:2109.11797, 2021. 2, 3
- [44] Maoyuan Ye, Jing Zhang, Shanshan Zhao, Juhua Liu, Tongliang Liu, Bo Du, and Dacheng Tao. Deepsolo: Let

- transformer decoder with explicit points solo for text spotting. In CVPR, pages 19348–19357, 2023. 4
- [45] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why visionlanguage models behave like bags-of-words, and what to do about it? In ICLR, 2022. 3, 6
- [46] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In CVPR, pages 18123–18133, 2022. 2
- [47] Qiming Zhang, Yufei Xu, Jing Zhang, and Dacheng Tao. Vitaev2: Vision transformer advanced by exploring inductive bias for image recognition and beyond. IJCV, pages 1–22,

2023. 4

- [48] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. IJCV, 130(9):2337–2348, 2022. 2
- [49] Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, and Yejin Choi. Multimodal c4: An open, billion-scale corpus of images interleaved with text. arXiv preprint arXiv:2304.06939, 2023. 9, 10

[Figure 40]

## Parrot Captions Teach CLIP to Spot Text

### Supplementary Material

Mean CLIP scores (Whole Dataset) 0.3021 Mean CLIP scores (All-Emb. Text Removal) 0.2912 Mean CLIP scores (Co-Emb. Text Removal) 0.2892

- Table 11. The mean CLIP scores of CC12M, which are obtained from ViT-B-32 models. The text removal operations are the same as Sec. 5, while the results are from the whole dataset.

|Data (3M)|Model|IN Ret. Avg.<br><br>|Sync. S|
|---|---|---|---|
|Random w/o Emb. Text w/ Emb. Text Only|RN50 RN50 RN50<br><br>|0.205 0.253 0.229<br><br>0.206 0.248 0.231 0.161 0.232 0.210<br><br><br>|0.186 0.121 0.220<br><br>|
|Random w/o Emb. Text w/ Emb. Text Only|ViT-B ViT-B ViT-B<br><br>|0.142 0.193 0.206 0.151 0.190 0.214 0.113 0.165 0.196<br><br>|0.127 0.096 0.148<br><br>|

- Table 12. Comparison of dataset quality on sampled subsets. The subsets are sampled the same as Sec. 6.2 Curation I. The Sync.S denotes the average CLIP score of syn-emb. text benchmark in Sec. 6.2.

#### 10. More Experiments on CC12M Dataset

We further examine the CC12M [6] dataset with the same process in Sec. 5 and 6 to investigate whether parrot captions dominate the CLIP models’ behavior in different data curation pipelines. The overall CLIP scores statistics and experiments are shown in Tab. 11 and Tab. 12. The results show that the captions in CC12M are less correlated with text embedded in the images based on the mean CLIP scores. The CLIP training results on different curated subsets indicate that the embedded text in CC12M only slightly reduces the pre-training dataset quality, while the images with embedded text still harm the downstream performance. In summary, the CC12M dataset constructed without CLIP score is not significantly biased towards parrot captions.

#### 11. Technical Details

For the PCA used in feature processing, we use Incremental PCA implemented by scikit-learn [27]. For the K-Means model, we train for 300 iterations and redo 10 times to select the best model. For the CLIP model training, we used PyTorch DDP and amp precision to train models on a single machine with 8 NVIDIA A100 GPUs. We used AdamW [23] as an optimizer, with β1 = 0.9 and β2 = 0.98 for all models. We used a cosine decay schedule with a linear warmup. We used a resolution of 224 ×224 images for

|Data (12M)|Model<br><br>|IN Ret. Avg.|
|---|---|---|
|CoTR = 0.0 CoTR ≥ 0.3 CoTR ≥ 0.5 CoTR ≥ 0.8<br><br>CoTR = 1.0<br><br><br>|RN50 RN50 RN50 RN50 RN50|0.349 0.367 0.348 0.055 0.115 0.159 0.037 0.102 0.135 0.019 0.084 0.102 0.017 0.080 0.112<br><br>|
|CoTR = 0.0 CoTR ≥ 0.3 CoTR ≥ 0.5 CoTR ≥ 0.8<br><br>CoTR = 1.0<br><br><br>|ViT-B ViT-B ViT-B ViT-B ViT-B<br><br>|0.302 0.303 0.320 0.059 0.104 0.165 0.040 0.098 0.141 0.021 0.078 0.117 0.021 0.081 0.114<br><br>|

- Table 13. Ablation of different Co-Emb. Text Rate(CoTR). The fewer parrot captions, the better downstream task performance.

|Data (3M)|Model<br><br>|Avg.S(•)|IN Ret. Avg.|
|---|---|---|---|
|RSA < 0.0<br><br>RSA ≥ 0.0<br><br>RSA ≥ 0.1<br><br>RSA ≥ 0.2<br><br>RSA ≥ 0.3<br><br><br>|RN50 RN50 RN50 RN50 RN50|0.319 0.339 0.351 0.360 0.376<br><br>|0.327 0.349 0.336 0.245 0.294 0.292 0.078 0.159 0.179 0.028 0.101 0.125 0.016 0.083 0.109<br><br>|
|RSA < 0.0<br><br>RSA ≥ 0.0<br>RSA ≥ 0.1<br>RSA ≥ 0.2<br>RSA ≥ 0.3<br>|ViT-B ViT-B ViT-B ViT-B ViT-B<br><br>|0.319 0.339 0.351 0.360 0.376<br><br>|0.277 0.285 0.313 0.211 0.241 0.285 0.068 0.133 0.180 0.024 0.090 0.120 0.011 0.076 0.103<br><br>|

- Table 14. Ablation of models trained on subsets sampled by differentRSA.RSAdenotestherelativesimilarity(S(•)−S(×)) of raw S(•) and removed All-Emb. text S(×) images.

|Data (3M)<br><br>|Model|Avg.S(•)<br><br>|IN Ret. Avg.|
|---|---|---|---|
|RSC < 0.0<br><br>RSC ≥ 0.0<br><br>RSC ≥ 0.1<br><br>RSC ≥ 0.2<br><br>RSC ≥ 0.3<br><br><br>|RN50 RN50 RN50 RN50 RN50<br><br>|0.326 0.345 0.354 0.364 0.380|0.125 0.171 0.209 0.062 0.129 0.168 0.014 0.091 0.106 0.008 0.084 0.104 0.005 0.058 0.084<br><br>|
|RSC < 0.0<br><br>RSC ≥ 0.0<br>RSC ≥ 0.1<br>RSC ≥ 0.2<br>RSC ≥ 0.3<br>|ViT-B ViT-B ViT-B ViT-B ViT-B<br><br>|0.326 0.345 0.354 0.364 0.380|0.079 0.129 0.174 0.045 0.119 0.149 0.018 0.091 0.116 0.008 0.076 0.106 0.004 0.059 0.091<br><br>|

- Table 15. Ablation of models trained on subsets sampled by different RSC. RSC denotes the relative similarity (S(•) − S(■)) of raw S(•) and removed Co-Emb. text S(■) images.

pre-training. The training loss is the InfoNCE loss [30].

[Figure 41]

Image-TextSimilarityScores

Rand w/o Emb.

w/ Emb. Only

CoTR =0.0

CoTR ≥0.3

CoTR ≥0.5

CoTR ≥0.8

CoTR =1.0

RSA <0.0

RSA ≥0.0

RSA ≥0.1

RSA ≥0.2

RSA ≥0.3

RSC <0.0

RSC ≥0.0

RSC ≥0.1

RSC ≥0.2

RSC ≥0.3

Figure 11. Benchmarking text spotting capacity of CLIP models on 12M scales with 1-gram caption vocabulary synthetic images dataset.

[Figure 42]

[Figure 43]

[Figure 44]

Frequency: [>20000] Frequency: [5000, 20000] Frequency: [5000, 3000]

Figure 12. The word clouds of 1-gram caption vocabulary.

#### 12. Frequent Words in N-gram Vocabulary

mance on the text-oriented task, such as MINST.

Fig. 12 shows the word clouds of 1-gram caption vocabulary. To better visualize the long-tail distribution, the word clouds are drawn from three different frequency intervals. It shows that the 1-gram text becomes more complex and longer when the frequency is lower.

#### 15. Sample Visualization

- In Fig. 16, we visualize samples with top CLIP scores in 250 randomly sampled clusters from the original 4000 clusters. Each cluster is associated with a certain concept or object.
- In Fig. 17 and Tab. 16, we show more examples with parrot captions and the text spotted results.

#### 13. Curation Ablation on 12M scales

We further provide the ablation study on 12M scales in Tab. 13, 14 and 15. All the results are consistent with the 3M scale results. Fig. 11 reports the text spotting capacity of models on 12M scales using the synthetic benchmark the same as Sec. 6.2. It shows that training with more parrot caption samples does not lead to a stronger text spotting performance in synthetic benchmarks.

#### 14. Full Tracks Evaluation on DataComp

In Fig. 13, 14, and 15 we report all dataset results on DataComp [10] of the ablation study. In most vision-centric datasets, the model performance is consistent with the average performance. Meanwhile, the results also indicate that the model with stronger text capacity achieves better perfor-

[Figure 45]

- (a) RN50

[Figure 46]

- (b) ViT-B-32

- Figure 13. Full tracks DataComp evaluation of Curation I: Embedded Text in Images.

[Figure 47]

- (a) RN50

[Figure 48]

- (b) ViT-B-32

- Figure 14. Full tracks DataComp evaluation of Curation II: Co-Emb. Text Rate (CoTR).

[Figure 49]

- (a) RN50 of RSA

[Figure 50]

- (b) ViT-B-32 of RSA

[Figure 51]

- (c) RN50 of RSC

[Figure 52]

- (d) ViT-B-32 of RSC

Figure 15. Full tracks DataComp evaluation of Curation III: Relative Score from Text Removal.

[Figure 53]

- (a) Cluster 0 to 50

[Figure 54]

- (b) Cluster 50 to 100

[Figure 55]

- (c) Cluster 100 to 150

[Figure 56]

- (d) Cluster 150 to 200

[Figure 57]

- (e) Cluster 200 to 250

###### Figure 16. Top CLIP scores sample visualization of each clustering. Each column is from the same cluster.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

(1) (2) (3) (4)

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

(5) (6) (7) (8)

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

(9) (10) (11) (12) (13)

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

(14) (15) (17)

(16)

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

(18)

(19) (20) (21)

Figure 17. More sample visualization of parrot captions and the corresponding captions are listed in Tab. 16

ID Type Content

- 1

Captions BEST DOCUMENTARY - Christian Film Festival - 2017 (1).png

Co-Emb. BEST DOCUMENTARY Christian Film Festival 2017

- 2

Captions how-to-make-bubbles-that-bounce

Co-Emb. how to make bubbles that bounce

- 3

Captions Denver Broncos Carbon Small Over Small Metal Acrylic Cut License Plate Frame

Co-Emb. Denver Broncos

- 4

Captions Title details for The Great Derangement by Amitav Ghosh - Available

Co-Emb. The Great Derangement by Amitav Ghosh

- 5

Captions Byron Wheelstand Contest 2017 Co-Emb. Byron Wheelstand Contest 2017

- 6

Captions dental marketing and practice management ideas for January - winter dental marketing ideas betty hayden consulting

Co-Emb. dental practice management ideas for January betty hayden consulting

- 7

Captions Best Sheep Trainer Alive Frosted Glass Mug

Co-Emb. Best Sheep Trainer Alive

- 8

Captions [THQ VIETNAM] VICO AUTOMATIC WASHING POWDER 3KG X 4 PACKS

Co-Emb. VICO AUTOMATIC WASHING POWDER

- 9

Captions Nobody Knows But You by Anica Mrose Rissi

Co-Emb. Nobody Knows But You Anica Mrose Rissi

- 10

Captions Bon Jovi Poster from Arco Arena on 15 Mar 93: 11 x 17

Co-Emb. Bon Jovi Arco Arena 15

- 11

Captions Enriching our Vision of Reality de Alister Mcgrath

Co-Emb. Alister Mcgrath Enriching our Vision of Reality

- 12

Captions March Favorites 2014 — FreshExpectations

Co-Emb. March Favorites 2014

- 13

Captions Wine Sense: The Art of Appreciating Wine by Steve Shipley Co-Emb. Wine Sense: The Art of Appreciating Wine by Steve Shipley

- 14

Captions Flute is my super power Tote Bag

Co-Emb. Flute is my super power

- 15

Captions My Son Lives To Protect Rabbits Travel Speaker

Co-Emb. My Son Lives To Protect Rabbits

- 16

Captions Poster: KEEP CALM AND LOVE WILL SINGE

Co-Emb. KEEP CALM AND LOVE WILL SINGE

- 17

Captions Olympic Games - The Myth Audiobook by Tina Angelou Narrated by Danae Phelps

Co-Emb. Olympic Games The Myth Tina Angelou

- 18

Captions 11/16/2012 never forget - 11/16/2012 never forget Twinkie RIP

Co-Emb. 11/16/2012 never forget Twinkie

- 19

Captions SYSTEM OF A DOWN : TOXICITY [ECO STYLE] (CD)

Co-Emb. SYSTEM OF A DOWN TOXICITY

- 20

Captions Text reads: how to create healthy habits for swim parents. A blue text box below reads: 7 habits for swim parents to learn

Co-Emb. how to create healthy habits for swim parents 7 habits for swim parents to learn

- 21

Captions Be Mine Wall Clock

Co-Emb. Be Mine

Table 16. Captions and the Co-Emb. Text shown in Fig 17.

