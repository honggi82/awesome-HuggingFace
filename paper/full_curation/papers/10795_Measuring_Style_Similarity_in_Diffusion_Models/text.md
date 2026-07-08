# arXiv:2404.01292v1[cs.CV]1Apr2024

## Measuring Style Similarity in Diffusion Models

Gowthami Somepalli⋆♦, Anubhav Gupta⋆♦, Kamal Gupta ♦, Shramay Palta ♦,

Micah Goldblum ♣, Jonas Geiping ♠, Abhinav Shrivastava ♦, Tom Goldstein♦

♣ New York University ♠ ELLIS Institute, MPI for Intelligent Systems

♦ University of Maryland, College Park

### Abstract

Generative models are now widely used by graphic designers and artists. Prior works have shown that these models remember and often replicate content from their training data during generation. Hence as their proliferation increases, it has become important to perform a database search to determine whether the properties of the image are attributable to specific training data, every time before a generated image is used for professional purposes. Existing tools for this purpose focus on retrieving images of similar semantic content. Meanwhile, many artists are concerned with style replication in text-to-image models. We present a framework for understanding and extracting style descriptors from images. Our framework comprises a new dataset curated using the insight that style is a subjective property of an image that captures complex yet meaningful interactions of factors including but not limited to colors, textures, shapes, etc.We also propose a method to extract style descriptors that can be used to attribute style of a generated image to the images used in the training dataset of a text-to-image model. We showcase promising results in various style retrieval tasks. We also quantitatively and qualitatively analyze style attribution and matching in the Stable Diffusion model. Code and artifacts are available at https://github.com/learn2phoenix/CSD.

[Figure 1]

Figure 1: Original artwork of 6 popular artists and the images generated in the style of these artists by three popular text-to-image generative models. The numbers displayed below each image indicates the similarity of generated image with artist’s style using proposed method. A high similarity score suggests a strong presence of the artist’s style elements in the generated image. Based on our analyses, we postulate that three artists on the right were removed (or unlearned) from SD 2.1 while they were present in MidJourney and SD 1.4. Please refer to Section 2 for more details.

⋆Equal contribution. Correspondence: gowthami@cs.umd.edu.

### 1 Introduction

Diffusion-based image generators like Stable Diffusion [49], DALL-E [47] and many others [1, 7, 39, 43] learn artistic styles from massive captioned image datasets that are scraped from across the web [54]. Before a generated image is used for commercial purposes, it is wise to understand its relationship to the training data, and the origins of its design elements and style attributes. Discovering and attributing these generated images, typically done with image similarity search, is hence becoming increasingly important. Such dataset attribution serves two purposes. It enables users of generated images to understand potential conflicts, associations, and social connotations that their image may evoke. It also enables artists to assess whether and how generative models are using elements of their work.

Despite a long history of research [60], recovering style from an image is a challenging and open problem in Computer Vision. Many existing retrieval methods [8, 45, 46] for large training datasets focus primarily on matching semantic content between a pair of images. Understanding the origin of the style present in a generated image, however, is much less well understood. To address this gap, we propose a self-supervised objective for learning style descriptors from images. Standard augmentation-based SSL pipelines (e.g. SimCLR and variants) learn feature vectors that are invariant to a set of augmentations. Typically, these augmentations preserve semantic content and treat style as a nuisance variable. In contrast, we choose augmentations that preserve stylistic attributes (colors, textures, or shapes) while minimizing content. Unfortunately, SSL is not enough, as style is inherently subjective, and therefore a good style extractor should be aligned with human perceptions and definitions of style. For this reason, we curate a style attribution dataset, LAION-Styles, in which images are associated with the artist that created them.

By training with both SSL and supervised objectives, we create a high-performance model for representing style. Our model, CSD, outperforms other large-scale pre-trained models and prior style retrieval methods on standard datasets. Using CSD, we examine the extent of style replication in the popular open-source text-to-image generative model Stable Diffusion [49], and consider different factors that impact the rate of style replication.

To summarize our contributions, we (1) propose a style attribution dataset LAION-Styles, associating images with their styles, (2) introduce a multi-label contrastive learning scheme to extract style descriptors from images and show the efficacy of the scheme by zero-shot evaluation on public domain datasets such as WikiArt and DomainNet (3) We perform a style attribution case study for one of the most popular text-to-image generative models, Stable Diffusion, and propose indicators of how likely an artist’s style is to be replicated.

### 2 Motivation

We present a case study that shows how style features can be used to interrogate a generative model, and provide utility to either artists or users. We consider the task of analyzing a model’s ability to emulate an artist’s style, and of attributing the style of an image to an artist. We begin by curating a list of 96 artists, primarily sourced from the WikiArt database, supplemented by a few contemporary artists who are notably popular within the Stable Diffusion community1. For each artist, we compute a prototype vector by averaging the embeddings of their paintings using our proposed feature extractor, CSD ViT-L. Next, we generate an image for each artist using Stable Diffusion 2.1 with a prompt in the format A painting in the style of <artist_name>. We compute the dot product similarity between each generated image’s embedding and the artist’s prototype. This process was repeated multiple times for each artist, and we plot mean results in Fig. 2. We refer to this quantity as the General Style Similarity (GSS) score for an artist, as it measures how similar a generated image is to a typical image from that artist while using our style representation model. We also plot an analogous style similarity score, but using “content-constrained” prompts. For instance, one prompt template is A painting of a woman doing <Y> style of <X> where X is the name of the artist and Y is some setting like reading a book or holding a baby etc. See Sec. 7 for all templates.

Each point in Fig. 2 represents an artist. Notice that GSS scores are highly correlated with contentconstrained scores, indicating that our feature vectors represent style more than semantic content. Our findings reveal that SD 2.1 is much more capable of emulating some artists than others. Artists like

1https://supagruen.github.io/StableDiffusion-CheatSheet/

Leonid Afremov, Georges Seurat exhibit high style similarity scores, and visual inspection of generated images confirms that indeed their style is emulated by the model (Fig. 1 - Original artwork vs SD 2.1). On the other end of the spectrum, artists such as Ruan Jia and Greg Rutkowski showed low similarity scores, and likewise the generated images bear little resemblance to the artists’ work. Interestingly, after completing this study, we discovered that Greg Rutkowski’s work was excluded from the training data for the Stable Diffusion 2.1 model, as reported by [11].

This demonstrates that the Style Similarity score can be used by artists to quantify how well a model emulates their style, or it can be used by users to ascertain whether a generated image contains stylistic elements associated with a particular artist. After a thorough inspection of the generations from 96 artists, we hypothesize that a single-image Style Similarity score below

0.9

| |amano<br><br>antoine blanchard<br><br>anton fadeev<br><br>carne griﬃths<br><br>dante gabriel rossetti<br><br>ferdinand knab<br><br>georges seurat<br><br>greg rutkowski<br><br>gustav klimt<br><br>ivan bilibin<br><br>leonid afremov<br><br>roy lichtenstein<br><br>ruan jia<br><br>thomas cole<br><br>vincent van gogh<br><br>wadim kashin<br><br>william turner<br><br>| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

ConstrainedStyleSimilarity-woman

0.8

0.7

- 0.5 indicates the absence of the artist’s style, whereas a score above 0.8 strongly indicates its presence.

In Figure 1, we show original artworks for 6 artists, and generations from MidJourney [39], Stable Diffusion 2.1 and Stable Diffusion

- 1.4 [49] for each of these artists. The 3 artists on the left side have high GSS while the ones on the right side have low GSS. Below each generated image, we display the similarity against the artist’s prototype vector. We see high image similarity scores in the Midjourney generations and qualitatively these images look stylistically similar to artists’ original artworks. We also see the interesting cases of Greg Rutkowski, Ruan Jia, and Amano whose style is captured by Stable Diffusion 1.4, while being notably absent in Stable Diffusion 2.1. This finding is in line with reports suggesting that some of these artists were removed from the training data of Stable Diffusion 2.1 [11]. Based on this analysis, we postulate that Ruan Jia, Wadim Kashin, Anton Fadeev, Justin Gerard, and Amano were also either excluded from the training data or post-hoc unlearned/removed from Stable Diffusion 2.1.

0.6

0.5

0.4

0.4 0.5 0.6 0.7 0.8 0.9 General Style Similarity

Figure 2: Style similarity of Stable Diffusion 2.1 generated images against the artist’s prototypical representation. On the X-axis, the similarities are depicted when the prompt is not constrained, while the Y-axis represents similarity when the prompt is constrained to generate an image of a “woman” in the artist’s style.

### 3 What is style?

The precise definition of “style” remains in contention, but many named artistic styles (e.g., cubism, impressionism, etc...) are often associated with certain artists. We leverage this social construct, and define style simply as the collection of global characteristics of an image that are identified with an artist or artistic movement. These characteristics encompass various elements such as color usage, brushstroke techniques, composition, and perspective.

Related work. Early computer vision algorithms attempted to model style using low-level visual features like color histograms, texture patterns, edge detection, and shape descriptors. Other computational techniques involve rule-based systems, such as the presence of specific compositional elements, the use of specific color palettes, or the presence of certain brushstroke patterns to identify specific style characteristics [19, 20, 24, 25, 29, 32, 35, 37, 51, 55, 59, 65, 67].

Modern studies have focused on the task of transferring style from one image to another[13, 17, 22, 34, 41, 62, 68]. Some works have also concentrated on style classification [2, 4, 10, 15, 27, 28, 30, 33, 38, 48, 53]. A limited number of studies address in-the-wild style quantification, matching, and retrieval [14, 23, 36, 50, 66]. In their seminal work, Gatys et al. [17] introduced Gram Matrices as style descriptors and utilized an optimization loop to transfer style. Another approach proposed by Luan et al. [34] includes a photorealism regularization term to prevent distortions in the reconstructed image. Zhang et al. [68] formulated style transfer using Markov random fields. Beyond Gram-based style representation, Chu et al. [10] explored various other types of correlations and demonstrated performance variations.

In a recent work by Lee et al [31], two separate neural network modules were used – one for image style and another for image content – to facilitate image style retrieval. In the most recent related research, Wang et al.[63] developed an attribution model trained on synthetic style pairs, designed to identify stylistically similar images. In contrast to this approach, our method leverages real image pairs, curated automatically through their caption annotations. Despite our training dataset being approx. 16% the size of training data used in [63], we demonstrate that our model significantly outperforms this method on many zero-shot style matching tasks in the later sections.

### 4 Creating a dataset for style attribution

While many large web datasets now exist, we need one that contains wide variations in artistic styles, and also labels that be used for downstream style retrieval evaluation. Some large-scale datasets specifically designed to handle such a challenge, like BAM [64], are not available in the public domain and others like WikiArt [52] are not large enough to train a good style feature extractor. In the following section, we propose a way to curate a large style dataset out of the LAION [54] Aesthetics 6+ dataset.

LAION-Styles: A dataset for style distillation. We curate our own dataset as a subset of LAION [54]. We start off with the 12M image-text pairs with predicted aesthetics scores of 6 or higher. Note that this dataset is extremely unbalanced, with some popular artists appearing much more frequently than others. Also, a large number of images are duplicated within the dataset which is a major issue for the text-to-image models trained on this data [57, 58]. Furthermore, the image captions within the data are often noisy and are often missing a lot of information. We address these challenges and propose a new subset of LAION-Aesthetics consisting of 511,921 images, and 3840 style tags, where each image can have one or more tags. We use this dataset for training our models.

We begin with a bank of styles collated in previous work for image understanding with the CLIP Interrogator [44]. This bank of styles was curated based on typical user prompts for Stable Diffusion. We combine the bank of artists, mediums, and movement references, to a shortlist of 5600 tags. We then search for these tags in the 12M LAION-Aesthetics captions and shortlist the images that have at least one of the tags present. We further filter out the tags which have over 100,000 hits in the dataset since human inspection found that they refer to common phrases like ‘picture’ or ‘photograph’ that do not invoke a distinct style. After discarding images with an unavailable URL, we are left with about 1 million images and 3840 tags. We further deduplicate the images using SSCD [45] with a threshold of 0.8 and merge the tags of images that are near copies of each other. As a by-product, the deduplication also helps with the missing tags in the images, since we can simply merge the text labels of duplicate images. After deduplication, we are left with 511,921 images.

### 5 Contrastive Style Descriptors (CSD)

Self-Supervised Learning. Many successful approaches to SSL [61] use a contrastive [9] approach, where two views (or augmentations) of the same image in the dataset are sampled and passed to a deep network to get their respective image descriptors. The network is trained to maximize the similarity of two views of the same image and minimize agreement with other images in the batch. Standard choices for augmentations include color jitter, blurring, grayscaling, etc., as these alter the image’s visual properties while preserving content. While these are good augmentations for object recognition tasks, they train the network to ignore image attributes associated with style.

Our approach relies on a training pipeline with two parts. First, we use contrastive SSL, but with a set of augmentations that are curated to preserve style. Second, we align our model with human perceptions of style by training on our labelled LAION-Styles dataset described in Section 4.

Proposed Approach. We seek a model for extracting image descriptors that contain concise and effective style information. To be a useful, the model should be invariant to semantic content and capable of disentangling multiple styles.

Given a dataset of N labeled images {xi,li}Ni=1, where each image can have one or more labels from a set of L labels, we define the label vector of the ith image as li = (c1,c2,...,cL), where each ck ∈ {0,1}. As mentioned in the previous section, our multi-label dataset consists of N = 511,921 images and L = 3,840 style tags. We consider a mini-batch of B images. Each of the images are

passed to a Vision Transformer (ViT) [12] backbone, and then projected to a d−dimensional vector. We consider two variants of ViT (ViT-B and ViT-L).

Our style descriptors fViT(xi) ∈ Rd are then used to create a matrix of pairwise cosine similarity scores si,j = cos(fViT(xi),fViT(xj)). In order to compute our multi-label contrastive loss (MCL), we also compute the groundtruth similarity scores as sˆi,j = liTlj , where is the indicator function that returns 1 if any of the labels of the images i,j match. Our final loss term reduces to:

LMCL = −sˆi,j log

- exp(si,j)/τ)

k̸=j

- exp(si,k)/τ)

, (1)

where τ is the temperature fixed during the training.

Since our supervised dataset is modest in size, we add a self-supervised objective. We sample two “views” (augmentations) of each image in a batch and add a contrastive SSL term. Standard SSL training routines (e.g., MoCo, SimCLR, BYOL etc.) choose augmentations so that each pair of views has the same semantic content, but different style content. These augmentations typically include Resize, Horizontal Flips, Color Jitter, Grayscale, Gaussian Blur, and Solarization [6, 18]. For our purposes, we depart from standard methods by excluding photometric augmentations (Gaussian Blur, Color Jitter), as they alter the style of the image. We keep the following spatial augmentations Horizontal Flips, Vertical Flips, Resize and Rotation as they keep style intact.

The overall loss function is a simple combination of the multilabel contrastive loss and self-supervised loss L = LMCL + λLSSL. During inference, we use the final layer embedding and the dot product to compute style similarity between any two images. In our experiments, we found that initializing weights to CLIP [46] ViT-B and ViT-L improves performance.

### 6 Results

Training details. We present the results for two variants of our model CSD ViT-B and CSD ViT-L version. Both the models are initialized with respective CLIP variant checkpoints and are finetuned for 80k iterations on the LAION-Styles dataset on 4 A4000/A5000 GPUs. We use an SGD optimizer with momentum 0.9 and learning rate of 0.003 for the projection layer and 1e − 4 for the backbone. Our mini-batch size per GPU is 16. We use λ = 0.2 and τ = 0.1 for the final model. The training takes about 8 hours for the base model and around 16 hours for the large model. See the Appendix for more details and ablations.

Task. We perform zero-shot evaluation across multiple datasets on a style-retrieval task. Following [3, 26], we split each dataset into two parts: Database and Query. Given a query image at test time, we evaluate whether we can find the ground-truth style in its nearest neighbors from the database.

Baselines. We compare our model against a recent style attribution model GDA [63] which is trained via fine-tuning on paired synthetic style data, and VGG [17, 56] Gram Matrices which are often used for neural style transfer applications. Further we compare with CLIP [46] models supervised with free-form text captions, and with other self-supervised models such as DINO [8], MoCo [21], SSCD [45]. We use the embeddings from the last layer for each of these models except for VGG where we use the Gram Matrix [17] of the last layer. We skipped evaluations of [14, 23, 50] since both pre-trained models and training data are not available.

Metrics. We do nearest neighbour searches for k ∈ [1,10,100], and report Recall@k, mAP@k. We use the standard definitions of these metrics from the retrieval literature. Like [5], we define positive recall as the existence of a correct label in top-N matches and no recall when none of the top-N matches share a label with the query. Similarly, mAP is defined as average over precision at each rank in N, and then averaged over all queries.

Evaluation Datasets. DomainNet [42] consists of an almost equal number of images from six different domains: Clipart, Infograph, Painting, Quickdraw, Real, and Sketch. Upon examination, we observed a strong stylistic resemblance between the Quickdraw and Sketch domains, leading us to exclude Quickdraw from our analysis. The dataset’s content information was utilized to categorize the images into two main clusters of content classes. This clustering was achieved through the application of word2vec [40]. The images within the smaller cluster were designated as part of the

- Table 1: mAP and Recall metrics on DomainNet and WikiArt datasets. Our model consistently performs the best in all cases except one, against both self-supervised and style attribution baselines.

DomainNet WikiArt DomainNet WikiArt

(mAP@k) (mAP@k) (Recall@k) (Recall@k)

Method 1 10 100 1 10 100 1 10 1 10 100 VGG Gram [16] - - - 25.9 19.4 11.4 - - 25.9 52.7 80.4 DINO ViT-B/16 [8] 69.4 68.2 66.2 44.0 33.4 18.9 69.4 93.7 44.0 69.4 88.1 DINO ViT-B/8 [8] 72.2 70.9 69.3 46.9 35.9 20.4 72.2 93.8 46.9 71.0 88.9 SSCD RN-50 [45] 67.6 65.9 62.0 36.0 26.5 14.8 67.6 95.0 36.0 62.1 85.4 MOCO ViT-B/16 [21] 71.9 71.1 69.6 44.0 33.2 18.8 72.0 94.0 44.0 69.0 88.0 CLIP ViT-B/16 [46] 73.7 73.0 71.3 52.2 42.0 26.0 73.7 94.5 52.2 78.3 93.5 GDA CLIP ViT-B [63] 62.9 61.6 59.3 25.6 21.0 14.1 62.9 92.3 25.6 56.6 83.8 GDA DINO ViT-B [63] 69.5 68.1 66.1 45.5 34.6 19.7 69.5 93.4 45.5 75.8 89.0 GDA ViT-B [63] 67.1 65.6 64.2 42.6 32.2 18.2 67.1 93.6 42.6 67.6 87.1 CSD ViT-B (Ours) 78.3 77.5 76.0 56.2 46.1 28.7 78.3 94.3 56.2 80.3 93.6 CLIP ViT-L [46] 74.0 73.5 72.2 59.4 48.8 31.5 74.0 94.8 59.4 82.9 95.1 CSD ViT-L (Ours) 78.3 77.8 76.5 64.56 53.82 35.65 78.3 94.5 64.56 85.73 95.58

Query set (20,000 images), and images in the bigger cluster to Database (206768 images). And the second dataset we evaluate all the models is, WikiArt[52]. It consists of 80096 fine art images spread across 1119 artists and 27 genres. We randomly split the dataset into 64090 Database and 16006 Query images. We use the artist as a proxy for the style since there is large visual variation within each genre for them to be considered as independent styles. Under this setting, WikiArt is a challenging retrieval task as the chance probability of successful match is just .09% while it is 20% for DomainNet.

#### 6.1 Analyses and observations

In Table 1, we report the metrics for all the baselines considered and the proposed CSD method using k nearest neighbors on 2 datasets - DomainNet and WikiArt. Note that while style and content are better separated in the case of DomainNet, WikiArt consists of more fine-grained styles and has more practical use cases for style retrieval. Loosely speaking, mAP@k determines what percentage of the nearest neighbors that are correct predictions, while the recall determines what percentage of queries has a correct match in the top-k neighbors. Our model CSD consistently outperformed all the pretrained feature extractors as well as the recent attribution method GDA [63] on both WikiArt and DomainNet evaluations. Note that all models are evaluated in a zero-shot setting. We see the most gains in the WikiArt dataset which is more challenging with chance probability of only 0.09%. When we look at mAP@1, which is same as top-1 accuracy, our base model outperforms the next best model by 5% points on WikiArt and 4.6% points on DomainNet. Our large model out-performs the closest large competitor by similar margins. Given the complexity of the task, these improvements are non-trivial.

We attribute the improvements to a couple of factors, (1) The multi-label style contrastive loss on our curated LAION-Styles dataset is quite helpful in teaching the model right styles (2) We hypothesize that these SSL models become invariant to styles because the way they were trained, but we are careful to not strip that away in our SSL loss component by carefully curating non-photometric augmentations in training.

Error Analysis. Even though our model outperforms the previous baselines, our top-1 accuracy for the WikiArt style matching task is still at 64.56. We tried to understand if there is a pattern to these errors. For example, our model is consistently getting confused between impressionist painters claude monet, gustave loiseau, and alfred sisley, all of whom painted many landscapes. They depicted natural scenes, including countryside views, rivers, gardens, and coastal vistas. Another example is pablo picasso and georges braque, who are both cubist painters. Given the impracticality of analyzing all 1,119 artists in the dataset, we opted for a macroscopic examination by categorizing errors at the art movement level. This approach is visualized in the heatmap presented in Fig. 3. In the heatmap, we see most of the errors concentrated along the diagonal, indicating that while the model often correctly identifies the art movement, it struggles to pinpoint the exact artist. There are instances of off-diagonal errors where the model incorrectly

identifies both the artist and their art movement. For example, many Post Impressionism and Realism paintings are assigned to Impressionism artists. Upon closer examination, it becomes apparent that they closely align in terms of historical timeline and geographical origin, both being from Europe. This analysis indicates the nuanced nature of style detection and retrieval in Computer Vision. It suggests that the upper limit for accuracy in this task might be considerably lower than 100%, even for a typical human evaluator, due to the inherent subtleties and complexities involved.

Abstract Expressionism 00

[Figure 2]

Action painting 01 Analytical Cubism 02

Art Nouveau Modern 03

80

Baroque 04 Color Field Painting 05

Contemporary Realism 06

Cubism 07 Early Renaissance 08

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

Human SSCD RN50 Dino ViT-B

Expressionism 09

60

Fauvism 10 High Renaissance 11

Impressionism 12 Mannerism Late Renaissance 13

GDA CLIP ViT-B GDA Dino ViT-B CLIP ViT-B

Minimalism 14 Naive Art Primitivism 15

40

New Realism 16 Northern Renaissance 17

CSD ViT-B CLIP ViT-L

Pointillism 18

Pop Art 19 Post Impressionism 20

CSD ViT-L

20

Realism 21

Rococo 22 Romanticism 23

0 20 40 60

Accuracy

Symbolism 24 Synthetic Cubism 25

Figure 4: Human study on Style Retrieval: Turns out untrained humans are worse than many feature extractors on matching images from the same artist.

Ukiyo e 26

0

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26

Figure 3: Confusion Matrix of errors in WikiArt: Art movements are predicted correctly. Errors occur in cases where movements share the same historical timeline and/or are derived from the same earlier movement.

#### 6.2 Human Study

To understand how our models compare to untrained humans, we conducted a small survey on style matching on 30 humans (excluding the authors). Following the convention in other papers [14, 28, 50, 62] and this paper, we assume, 2 images from same artist can be considered stylistic matches. For each query image, we gave 4 answer images out of which only one is from the same artist and hence is the right answer, so chance accuracy is 25%. We used the Artchive dataset introduced in [63] to create this survey and we collected 3 responses per item to break any ties. We present the results in Fig 4. Most interestingly, untrained humans are worse than many feature extractors at this task. SSCD is the only feature extractor that did worse than humans. Our model, CSD outperforms all the baselines on this dataset as well. This underpins the difficulty of style matching and also highlights the superior performance of our feature extractor.

### 7 Studying style in the wild: Analysis of Stable Diffusion

In the previous section, we have quantitatively shown that our model Contrastive Style Descriptors outperforms many baselines on style matching task. Now we try to address the question, Can we do style matching on Stable Diffusion generated images? To answer this question, we first curated multiple synthetic image collections using Stable Diffusion v 2.1 [49] and then compared them against the “ground truth” style matches on LAION-Style dataset.

- Table 2: mAP and Recall of SD 2.1 generated synthetic datasets based on Simple prompts and User-generated prompts

Simple prompts User-generated Simple prompts User-generated (mAP@k) (mAP@k) (Recall@k) (Recall@k) Method 1 10 100 1 10 100 1 10 100 1 10 100 GDA - DINO 11.6 10.2 7.6 4.45 4.59 4.24 11.6 28.1 52.83 4.45 25.22 67.18 CSD-ViTB 17.53 16.56 12.68 5.85 5.96 5.58 17.53 38.65 61.85 5.85 29.26 74.2 CLIP ViT-L/14 22.3 20.4 16.1 6.1 5.7 5.1 22.3 44.5 66.2 6.1 26.0 71.7 CSD (Ours) 24.5 23.3 18.5 5.7 5.9 5.6 24.5 47.2 67.5 5.7 26.5 71.8

Creating synthetic style dataset. The first challenge in curating synthetic images through prompts is the choice of prompts to be used for the generation. There have been no in-depth quantitative studies of the effect of prompts on generation styles. For this analysis, we chose 3 types of prompts.

[Figure 3]

- Figure 5: Nearest “style” neighbors. For each generated image (referred to as SD Gen), we show the top 5 style neighbors in CSD using our feature extractor. The green and red box around the image indicates whether or not the artist’s name used to generate the SD image was present in the caption of the nearest neighbor.

- 1. User-generated prompts: We used a Stable Diffusion Prompts2 dataset of 80,000 prompts filtered from Lexica.art. We used the test split and then filtered the prompts to make sure at least one of the keywords from the list we curated in Section 4 is present. We then sampled 4000 prompts from this subset for query split generation.
- 2. Simple prompts: We randomly sampled 400 artists which appeared most frequently in usergenerated prompts we analysed. We format the prompt as A painting in the style of <artist-name>, and we generate 10 images per prompt by varying the initialization seed.
- 3. Content-constrained prompts: We wanted to understand if we can detect style when we constrain the model to generate a particular subject/human in the style of an artist. For this, we used the prompt A painting of a woman in the style of <artist-name> or A painting of a woman reading in the style of <artist-name> etc., a total of 5 variations per subject repeated two times. We experimented with subjects, woman,dog and house in this study. We provide the exact templates in the appendix.

Table 3: mAP and Recall of SD 2.1 generated synthetic datasets based on Content-constrained prompts

Dog House Woman Dog House Woman

(mAP@k) (mAP@k) (mAP@k) (Recall@k) (Recall@k) (Recall@k)

Method 1 10 100 1 10 100 1 10 100 1 10 100 1 10 100 1 10 100 GDA-DINO 2.28 2 1.6 3.9 3.6 2.7 2.2 2.3 2 2.28 8.68 28.73 3.9 12.3 32 2.2 10.1 28.9 CSD-ViTB 4.5 4.31 3.61 4.6 4.29 3.87 7.55 7.83 6.42 4.5 14.36 34.88 4.6 15.03 39 7.55 20.1 42.46

CLIP ViT-L/14 2.3 2.2 1.9 4.5 4.2 3.6 7.4 7.1 6.2 2.2 9.8 29.9 4.5 13.8 35.3 7.4 19.0 41.6 CSD (Ours) 4.9 4.8 4.2 6.4 6.2 5.4 10.8 10.1 8.6 4.9 14.5 34.5 6.4 17.8 40.6 10.8 23.4 44.2

We generate 4000 images for each prompt setting using Stable Diffusion 2.1. There is only one style keyword in simple and content-constrained prompts, which we also use as a ground truth label for matching tasks. However, user-generated prompts can have multiple style labels within the caption, and we consider all of them as ground-truth labels.

Style retrieval on generated images. In Tab. 2, we show the retrieval results for Simple and Usergenerated prompts. We also compare our results with the second-best performing model in the previous section, CLIP ViT-L, and a recent style attribution model GDA [63]. We observe that our method outperforms CLIP on Simple prompt dataset. For User-generated prompts, the performance metrics are closer to CLIP model, but it’s important to note that these prompts are inherently more complex. This complexity results in a different label distribution in the query set for the two types of prompts we examine, leading to varied metric ranges in each case. Additionally, our method consistently outperforms both baselines in content-constrained scenarios, as evidenced in Table 3. This indicates the robustness and effectiveness of our approach in dealing with a variety of prompt complexities and content specifications. We refer the reader to Appendix to understand a few caveats of this quantitative study.

Qualitative Results. In Fig. 5, we showcase a selection of Stable Diffusion-generated images alongside their top 5 corresponding matches from LAION-Styles, as determined by the CSD ViT-L

###### 2https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts

[Figure 4]

- Figure 6: Top row: Images generated by Stable Diffusion. Middle and Bottom rows: Top matches retrieved by CLIP vs CSD (ours) respectively. CLIP is consistently biased towards image content, for instance retrieving image of a dog in the Column 1, 3, 4, or image of mother and baby in Column 7 or 8. Our method emphasize less on the content but more on the image styles. Please refer to the Appendix for the prompts.

feature extractor. The left section of the figure displays images generated from User-generated prompts, while the right section includes images created from Simple prompts. To aid in visual analysis, matches that share a label with the query image are highlighted in green. We can clearly see that the query image and the matches share multiple stylistic elements, such as color palettes and certain artistic features like motifs or textures. We observed that in generations based on usergenerated prompts, perceivable style copying typically occurs only when the prompts are shorter and contain elements that are characteristic of the artist’s typical content.

In Figure 6, we present several content-constrained prompt generations and their top-1 matches based on the CLIP ViT-L/14 model versus our CSD model. We observe that the CSD model accurately matches the correct artists to queries even when there is no shared content, only style. This is evident in columns 1, 3, 4, 7, and 8, where our model,CSD matches the correct style elements despite the subjects in the images being quite different. In contrast, the CLIP model still prioritizes content, often leading to mismatches in style.

[Figure 5]

- Figure 7: Does the diffusion model prefer some styles over others? When a prompt contains two style tags, we find that SD 2.1 strongly favors the style that it can best reproduce, we suspect because of a prevalence of the style in training data. In each block, the General Style Similarity(GSS) of the left side artist (red color) is less than the right side artist (blue color). (Ref Fig. 2). The generated image is more biased towards the artist with high GSS score.

Does the model prioritize some artists over others in the prompt? So far in the study we primarily concentrated on the impact of including an artist’s name in a style transfer prompt. In this section, we present preliminary findings in scenarios where prompts feature two artists. This is inspired by real-world user prompts from Stable Diffusion and Midjourney, where prompts often include multiple artists. We used the prompt in the style A painting in the style of <X> and <Y>, where X and Y represent different artists. For this study, we selected five artists with varying General Style Similarity (GSS) scores (referenced in Sec. 2). The artists, ranked by descending GSS scores, are Carne Griffiths, Roy Lichtenstein, Gustav Klimt, Pablo Picasso, and Ivan Bilibin. Note that most of the chosen artists have distinct styles that significantly differ from one another. We chose SD-XL

Turbo for this analysis because it is trained on deduplicated data, reducing bias towards frequently featured artists in the training dataset.

The results for each pair of artists are presented in Fig. 7. Interestingly, even without specific instructions to generate a female subject, most outputs depicted women, reflecting the common subject matter of the artists studied. We also calculated the style similarity scores for each generated image, comparing them to the prototypical styles of the artists in the prompts. In most cases, the style of the artist with the higher GSS score dominated the generated image. To test for potential bias towards the artist positioned first in the prompt (X), we conducted two trials with reversed artist positions. The results were generally consistent, with the dominant style remaining unchanged. However, in the case of Pablo Picasso and Gustav Klimt, this pattern did not hold; the model favored Picasso’s cubist style over Klimt’s nouveau style, possibly due to the small difference in their GSS scores. While this is not an extensive study, a consistent trend emerged: styles of artists with higher GSS scores, like Leonid Afremov and Carne Griffiths, predominantly influenced the combined style. We leave the comprehensive study on this topic to future work.

Which styles do diffusion models most easily emulate? In Fig. 2, we saw that General Style Similarity(GSS) and content-constrained style similarity scores are correlated, however, we see a few artists diverging away from the identity line. How far the score is diverging away from the identity line reflects how far outof-distribution the ‘subject’ in the prompt is for that artist. This could serve as an indicator of the generalization capability of the artist’s style. We hypothesize that artists who painted diverse subjects may have styles that generalize better to out-of-distribution (OOD) objects. To that end, we computed intra-cluster style similarity among all the artists’ paintings. We plot the difference in General style similarity scores and content-constrained similarity scores and the artist-level intra-cluster similarity in Fig. 8. For this experiment, we selected ‘dog’ as the subject, a choice informed by the observation that many artists predominantly painted women, landscapes, or cityscapes. Thus, ‘dog’ represents a subject that is OOD, as evidenced by lower scores for the dog category compared to women or houses in Tab. 3. Additionally, we limited our analysis to artists with GSS greater than 0.7 to ensure the model’s proficiency in reproducing the artist’s style in an unconstrained scenario. We see a high correlation of 0.568 between these 2 variables. It seems painters of diverse subjects are more likely to have their style replicated for out of distribution objects.

antoine blanchard

fra angelico

.....000005010015020 (General-Constrained)SS

alphonse mucha

leonid afremov

viktor vasnetsov

carne griﬃths

lucian freud pablo picasso

roy lichtenstein

vincent van gogh

jean arp

takashi murakami

edward hopper

0.6 0.7 0.8 Intra cluster similarity

Figure 8: Style generalization to new subjects: The X-axis represents the diversity of the artists’ paintings. The Y-axis shows the difference between General Style Similarity and Content-constrained Style Similarity on dog subjects.

In the top right corner of the figure, we note Antoine Blanchard, known for his Paris cityscapes, and Fra Angelico, who primarily focused on religious themes, including biblical scenes, saints, and other religious iconography. Conversely, in the lower left, we find Jean Arp and Pablo Picasso, whose work is characterized by abstract and non-traditional styles, encompassing a wide array of subjects. We conducted a qualitative verification to ascertain that style transfer was effective for artists located in the bottom right corner of the figure. Although this evaluation is not comprehensive, it serves as a preliminary investigation that may provide insights into the factors contributing to the generalization of style in diffusion models.

### 8 Conclusion

This study proposes a framework for learning style descriptors from both labeled and unlabeled data. After building a bespoke dataset, LAION-Styles, we train a model that achieves state-of-the-art performance on a range of style matching tasks, including DomainNet, WikiArt, and LAION-Styles.

Then, we show the substantive practical utility of this model through an investigative analysis on the extent of style copying in popular text-to-image generative models. Here, we show the model is capable of determining the factors that contribute to the frequency of style replication. Through cross-referencing of images with style copies and their original prompts, we have discover that the degree of style copying is increasing with prompt complexity. Such complex prompts lead to greater style copying compared to simple one-line prompts. This finding sheds light on the interplay between textual prompts and style transfer, suggesting that prompt design can influence the level of style copying in generative models. Finally, note that the definition of style used in this work is strictly based on artist attribution. We chose this definition because it can be operationalized and used in dataset construction. This definition is certainly not a golden truth, and we look forward to future studies using alternative, or extended definitions.

### 9 Acknowledgements

This work was made possible by the ONR MURI program and the AFOSR MURI program. Commercial support was provided by Capital One Bank, the Amazon Research Award program, and Open Philanthropy. Further support was provided by the National Science Foundation (IIS-2212182), and by the NSF TRAILS Institute (2229885).

### References

- [1] Adobe. Firefly, 2023. URL https://www.adobe.com/sensei/generative-ai/firefly. html.
- [2] Siddharth Agarwal, Harish Karnick, Nirmal Pant, and Urvesh Patel. Genre and style based painting classification. In 2015 IEEE Winter Conference on Applications of Computer Vision, pages 588–594. IEEE, 2015.
- [3] Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas Pajdla, and Josef Sivic. Netvlad: Cnn architecture for weakly supervised place recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5297–5307, 2016.
- [4] Zechen Bai, Yuta Nakashima, and Noa Garcia. Explain me the painting: Multi-topic knowledgeable art description generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5422–5432, 2021.
- [5] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728–1738, 2021.
- [6] Randall Balestriero, Mark Ibrahim, Vlad Sobal, Ari Morcos, Shashank Shekhar, Tom Goldstein, Florian Bordes, Adrien Bardes, Gregoire Mialon, Yuandong Tian, et al. A cookbook of self-supervised learning. arXiv preprint arXiv:2304.12210, 2023.
- [7] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.
- [8] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the International Conference on Computer Vision (ICCV), 2021.
- [9] Sumit Chopra, Raia Hadsell, and Yann LeCun. Learning a similarity metric discriminatively, with application to face verification. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), volume 1, pages 539–546. IEEE, 2005.
- [10] Wei-Ta Chu and Yi-Ling Wu. Image style classification based on learnt deep correlation features. IEEE Transactions on Multimedia, 20(9):2491–2502, 2018.
- [11] Decrypt. Greg rutkowski removed from stable diffusion but brought back by ai artists, March 2024. URL https://decrypt.co/150575/ greg-rutkowski-removed-from-stable-diffusion-but-brought-back-by-ai-artists.

- [12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
- [13] Vincent Dumoulin, Jonathon Shlens, and Manjunath Kudlur. A learned representation for artistic style. arXiv preprint arXiv:1610.07629, 2016.
- [14] Siddhartha Gairola, Rajvi Shah, and PJ Narayanan. Unsupervised image style embeddings for retrieval and recognition tasks. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3281–3289, 2020.
- [15] Noa Garcia and George Vogiatzis. How to read paintings: semantic art understanding with multi-modal retrieval. In Proceedings of the European Conference on Computer Vision (ECCV) Workshops, pages 0–0, 2018.
- [16] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576, 2015.
- [17] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2414–2423, 2016.
- [18] Jonas Geiping, Micah Goldblum, Gowthami Somepalli, Ravid Shwartz-Ziv, Tom Goldstein, and Andrew Gordon Wilson. How much data are augmentations worth? an investigation into scaling laws, invariance, and implicit regularization. arXiv preprint arXiv:2210.06441, 2022.
- [19] James Jerome Gibson. The senses considered as perceptual systems. 1966.
- [20] Daniel J Graham, James M Hughes, Helmut Leder, and Daniel N Rockmore. Statistics, vision, and the analysis of artistic style. Wiley Interdisciplinary Reviews: Computational Statistics, 4

(2):115–123, 2012.

- [21] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020.
- [22] Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In Proceedings of the IEEE international conference on computer vision, pages 1501–1510, 2017.
- [23] Xun Huang, Ming-Yu Liu, Serge Belongie, and Jan Kautz. Multimodal unsupervised imageto-image translation. In Proceedings of the European conference on computer vision (ECCV), pages 172–189, 2018.
- [24] James M Hughes, Daniel J Graham, and Daniel N Rockmore. Stylometrics of artwork: uses and limitations. In Computer Vision and Image Analysis of Art, volume 7531, pages 91–105. SPIE, 2010.
- [25] James M Hughes, Daniel J Graham, C Robert Jacobsen, and Daniel N Rockmore. Comparing higher-order spatial statistics and perceptual judgements in the stylometric analysis of art. In 2011 19th European Signal Processing Conference, pages 1244–1248. IEEE, 2011.
- [26] Qing-Yuan Jiang, Yi He, Gen Li, Jian Lin, Lei Li, and Wu-Jun Li. Svd: A large-scale short video dataset for near-duplicate video retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5281–5289, 2019.
- [27] Akshay Joshi, Ankit Agrawal, and Sushmita Nair. Art style classification with self-trained ensemble of autoencoding transformations. arXiv preprint arXiv:2012.03377, 2020.
- [28] Sergey Karayev, Matthew Trentacoste, Helen Han, Aseem Agarwala, Trevor Darrell, Aaron Hertzmann, and Holger Winnemoeller. Recognizing image style. arXiv preprint arXiv:1311.3715, 2013.

- [29] Sara Lawrence-Lightfoot and Jessica Hoffmann Davis. The art and science of portraiture. John Wiley & Sons, 2002.
- [30] Adrian Lecoutre, Benjamin Negrevergne, and Florian Yger. Recognizing art style automatically in painting with deep learning. In Asian conference on machine learning, pages 327–342. PMLR, 2017.
- [31] Seungmin Lee, Dongwan Kim, and Bohyung Han. Cosmo: Content-style modulation for image retrieval with text feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 802–812, June 2021.
- [32] Jia Li, Lei Yao, Ella Hendriks, and James Z Wang. Rhythmic brushstrokes distinguish van gogh from his contemporaries: findings via automated brushstroke extraction. IEEE transactions on pattern analysis and machine intelligence, 34(6):1159–1176, 2011.
- [33] Xin Lu, Zhe Lin, Xiaohui Shen, Radomir Mech, and James Z Wang. Deep multi-patch aggregation network for image style, aesthetics, and quality estimation. In Proceedings of the IEEE international conference on computer vision, pages 990–998, 2015.
- [34] Fujun Luan, Sylvain Paris, Eli Shechtman, and Kavita Bala. Deep photo style transfer. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4990– 4998, 2017.
- [35] Zhaoliang Lun, Evangelos Kalogerakis, and Alla Sheffer. Elements of style: learning perceptual shape style similarity. ACM Transactions on graphics (TOG), 34(4):1–14, 2015.
- [36] Shin Matsuo and Keiji Yanai. Cnn-based style vector for style image retrieval. In Proceedings of the 2016 ACM on International Conference on Multimedia Retrieval, pages 309–312, 2016.
- [37] Robert AJ Matthews and Thomas VN Merriam. Distinguishing literary styles using neural networks. In Handbook of neural computation, pages G8–1. CRC Press, 2020.
- [38] Orfeas Menis-Mastromichalakis, Natasa Sofou, and Giorgos Stamou. Deep ensemble art style recognition. In 2020 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE, 2020.
- [39] Midjourney. Midjourney, n.d. URL https://www.midjourney.com/home.
- [40] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26, 2013.
- [41] Taesung Park, Jun-Yan Zhu, Oliver Wang, Jingwan Lu, Eli Shechtman, Alexei Efros, and Richard Zhang. Swapping autoencoder for deep image manipulation. Advances in Neural Information Processing Systems, 33:7198–7211, 2020.
- [42] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source domain adaptation. In Proceedings of the IEEE International Conference on Computer Vision, pages 1406–1415, 2019.
- [43] Pablo Pernias, Dominic Rampas, Mats Leon Richter, Christopher Pal, and Marc Aubreville. Würstchen: An efficient architecture for large-scale text-to-image diffusion models. In The Twelfth International Conference on Learning Representations, 2023.
- [44] pharmapsychotic. Clip interrogator. https://github.com/pharmapsychotic/ clip-interrogator, 2023.
- [45] Ed Pizzi, Sreya Dutta Roy, Sugosh Nagavara Ravindra, Priya Goyal, and Matthijs Douze. A selfsupervised descriptor for image copy detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14532–14542, 2022.
- [46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

- [47] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021.
- [48] Catherine Sandoval Rodriguez, Margaret Lech, and Elena Pirogova. Classification of style in fine-art paintings using transfer learning and weighted image patches. In 2018 12th International Conference on Signal Processing and Communication Systems (ICSPCS), pages 1–7. IEEE, 2018.
- [49] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.
- [50] Dan Ruta, Saeid Motiian, Baldo Faieta, Zhe Lin, Hailin Jin, Alex Filipkowski, Andrew Gilbert, and John Collomosse. Aladin: all layer adaptive instance normalization for fine-grained style similarity. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11926–11935, 2021.
- [51] Robert Sablatnig, Paul Kammerer, and Ernestine Zolda. Hierarchical classification of paintings using face-and brush stroke models. In Proceedings. Fourteenth International Conference on Pattern Recognition (Cat. No. 98EX170), volume 1, pages 172–174. IEEE, 1998.
- [52] Babak Saleh and Ahmed Elgammal. Large-scale classification of fine-art paintings: Learning the right metric on the right feature. arXiv preprint arXiv:1505.00855, 2015.
- [53] Catherine Sandoval, Elena Pirogova, and Margaret Lech. Two-stage deep learning approach to the classification of fine-art paintings. IEEE Access, 7:41770–41781, 2019.
- [54] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion5b: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:2210.08402, 2022.
- [55] Jorge Miguel Silva, Diogo Pratas, Rui Antunes, Sérgio Matos, and Armando J Pinho. Automatic analysis of artistic paintings using information-based measures. Pattern Recognition, 114: 107864, 2021.
- [56] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.
- [57] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Diffusion art or digital forgery? investigating data replication in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.
- [58] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Understanding and mitigating copying in diffusion models. Advances in Neural Information Processing Systems, 36:47783–47803, 2023.
- [59] Bhargav Srinivasa Desikan, Hajime Shimao, and Helena Miton. Wikiartvectors: style and color representations of artworks for cultural analysis via information theoretic measures. Entropy, 24(9):1175, 2022.
- [60] Joshua Tenenbaum and William Freeman. Separating style and content. Advances in neural information processing systems, 9, 1996.
- [61] Matthew Walmer, Saksham Suri, Kamal Gupta, and Abhinav Shrivastava. Teaching matters: Investigating the role of supervision in vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.
- [62] Jianbo Wang, Huan Yang, Jianlong Fu, Toshihiko Yamasaki, and Baining Guo. Fine-grained image style transfer with visual transformers. In Proceedings of the Asian Conference on Computer Vision, pages 841–857, 2022.

- [63] Sheng-Yu Wang, Alexei A Efros, Jun-Yan Zhu, and Richard Zhang. Evaluating data attribution for text-to-image models. arXiv preprint arXiv:2306.09345, 2023.
- [64] Michael J Wilber, Chen Fang, Hailin Jin, Aaron Hertzmann, John Collomosse, and Serge Belongie. Bam! the behance artistic media dataset for recognition beyond photography. In Proceedings of the IEEE international conference on computer vision, pages 1202–1211, 2017.
- [65] John Willats and Frédo Durand. Defining pictorial style: Lessons from linguistics and computer graphics. Axiomathes, 15(3):319–351, 2005.
- [66] Daan Wynen, Cordelia Schmid, and Julien Mairal. Unsupervised learning of artistic styles with archetypal style analysis. Advances in Neural Information Processing Systems, 31, 2018.
- [67] Lei Yao, Jia Li, and James Z Wang. Characterizing elegance of curves computationally for distinguishing morrisseau paintings and the imitations. In 2009 16th IEEE International Conference on Image Processing (ICIP), pages 73–76. IEEE, 2009.
- [68] Wei Zhang, Chen Cao, Shifeng Chen, Jianzhuang Liu, and Xiaoou Tang. Style transfer via image component analysis. IEEE Transactions on multimedia, 15(7):1594–1601, 2013.

Measuring Style Similarity in Diffusion Models Appendix

- A Human Evaluation

We used 30 human subjects (excluding authors) to evaluate human-level performance on the task of style matching. Participation in the study was voluntary, and none of the subjects had any prior familiarity with the task. The authors manually vetted the data presented to the subjects for the absence of any offensive or inappropriate visuals. The subjects were informed that their responses would be used to compare the human performance with ML models. We did not collect any personally identifiable information and secured an exempt status from IRB at our institute for this study.

- B List of artists in style analysis

The following are the artists in the style analysis discussed in Sec. 2 - roy lichtenstein, justin gerard, amedeo modigliani, leonid afremov, ferdinand knab, kay nielsen, gustave courbet, thomas eakins, ivan shishkin, viktor vasnetsov, ivan aivazovsky, frederic remington, frederic edwin church, marianne north, salvador dali, pablo picasso, robert delaunay, ivan bilibin, rembrandt, frans hals, dante gabriel rossetti, max ernst, diego rivera, andy warhol, wadim kashin, caspar david friedrich, jan matejko, albert bierstadt, vincent van gogh, cy twombly, amano, anton fadeev, gian lorenzo bernini, mark rothko, mikhail vrubel, hieronymus bosch, katsushika hokusai, alphonse mucha, winslow homer, george stubbs, taro yamamoto, richard hamilton, carne griffiths, edward hopper, jan van eyck, francis picabia, michelangelo, arkhip kuindzhi, isaac levitan, gustave dore, antoine blanchard, john collier, paul klee, caravaggio, m.c. escher, leonardo da vinci, alan bean, greg rutkowski, jean arp, marcel duchamp, thomas cole, takashi murakami, thomas kinkade, raphael, hubert robert, john singer sargent, fra angelico, gustav klimt, ruan jia, harry clarke, william turner, claude monet, gerhard richter, frank stella, francisco goya, giuseppe arcimboldo, otto dix, lucian freud, jamie wyeth, rene magritte, titian, john atkinson grimshaw, man ray, albert marquet, mary cassatt, georges seurat, fernando botero, martin johnson heade, william blake, ilya repin, john william waterhouse, edmund dulac, peter paul rubens, frank auerbach, frida kahlo.

- C Model Ablations

We conducted several ablations on the choice of the backbone for the model, the λ hyperparameter in the loss formulation and the temperature hyperparameter, τ. We present the results on the Wikiart dataset for various configurations in Tab. 4. In the first ablation, we can see that CLIP ViT-B as the initialization for CSD gives the best performance. In the second ablation, we study the loss hyperparameter λ. λ = 0 model is trained only on Multi-label Contrastive loss, while λ = ∞ refers to model with only SSL loss. We see the best outcome when we combine both losses in the training. In the final ablation we see the effect of temperature hyperparameter. We clearly see the best outcome at τ = 0.1, which is conventionally the temperature used in other papers as well.

- D Dataset Curation Details

- D.1 LAION Deduplication

During our initial retrieval experiments, we found out that most of the top-N nearest neighbors that were being returned with respect to a query image were essentially the same images. We performed deduplication of this dataset (∼ 1.3 million images) by computing the SSCD [45] embedding of all the Database images and then computing the similarity of each image with all the other images and

Table 4: Model Ablations: We present results on Wikiart dataset. All the models are trained for same number of iterations. The baseline hyperparameters are λ = 0.2 and τ = 0.1 and backbone initialization with CLIP ViT-B. ⋆ refers to the CSD ViT-Base variant we discussed in the main paper.

mAP@k Recall@k Ablation Variant 1 10 100 1 10 100

SSCD RN-50 33.2 24.8 14.11 33.2 58.8 83.8 CLIP RN-50 51.8 44.2 25.2 51.8 77.0 92.1 DINO ViT-B 49.8 39.6 24.4 49.8 76.3 92.6 CLIP ViT-B ⋆ 56.2 46.1 28.7 56.2 80.3 93.6

Architecture,pre-training style

λ = ∞ 49.1 40.2 23.9 49.1 70.3 85.5 λ = 1 52.3 48.1 26.2 52.3 80.1 91.4 λ = 0.2 ⋆ 56.2 46.1 28.7 56.2 80.3 93.6 λ = 0.1 54.9 45.5 28.3 54.9 78.2 92.1 λ = 0 51.8 44.9 26.6 51.8 79.5 90.3

Loss Hyperparameter

τ = 0.01 55.1 44.4 27.3 55.1 79.7 93.6 τ = 0.1 ⋆ 56.2 46.1 28.7 56.2 80.3 93.6 τ = 0.5 42.3 39.9 20.3 42.3 70.6 90.1 τ = 1 36.2 28.7 18.0 36.2 64.4 86.3

Temperature

then recursively removing the duplicate images, keeping only one image in any connected graph. The labels corresponding to the removed images were merged with the labels of the master image such that the master image has labels of all its children, the children being removed from the

Database. We considered 2 images to be the same if the inner product between their SSCD embeddings was greater than 0.8. After the filtering, we are left with ∼ 0.5 million images

### E Stable Diffusion Analysis Extended

- E.0.1 Content-constrained templates. The following templates are used in generating content-constrained SD synthetic datasets.

- • Woman-constrained prompts: (1) A painting of a woman in the style of <artist>, (2) A painting of a woman holding an umbrella in the style of <artist>, (3) A painting of a woman wearing a hat in the style of <artist>, (4) A painting of a woman holding a baby in the style of <artist>, (5) A painting of a woman reading a book in the style of <artist>
- • Dog-constrained prompts: (1) A painting of a dog in the style of <artist>, (2) A painting of dog playing in the field in the style of <artist>, (3) A painting of a dog sleeping in the style of <artist>, (4) A painting of two dogs in the style of <artist>, (5) A portrait of a dog in the style of <artist>
- • House-constrained prompts: (1) A painting of a house in the style of <artist>, (2) A painting of a house in a forest in the style of <artist>, (3) A painting of a house in a desert in the style of <artist>, (4) A painting of a house when it’s raining in the style of <artist>, (5) A painting of a house on a crowded street in the style of <artist>

#### E.0.2 Fig 5 prompts.

Please find the prompts to generate the images presented in Fig. 6. The prompts are from left to right in the order.

- 1. A painting of a dog in the style of Van Gogh
- 2. A painting of dog playing in the field in the style of Georges Seurat
- 3. A painting of a dog sleeping in the style of Leonid Afremov
- 4. A painting of a dog sleeping in the style of Carne Griffiths

- 5. A painting of a woman holding an umbrella in the style of Katsushika Hokusai
- 6. A painting of a woman holding an umbrella in the style of Wassily Kandinsky
- 7. A painting of a woman holding a baby in the style of Amedeo Modigliani
- 8. A painting of a woman holding a baby in the style of Alex Gray

#### E.0.3 First observations on synthetic datasets.

When we scan through the generations, we find that for simple prompts the SD 2.1 model is borrowing the contents along with the artistic styles of an artist. For example, for the prompt A painting in the style of Warhol SD 2.1 always generated a version of Marilyn Diptych’s painting. Similarly, all the prompts of Gustav Klimt generated “The Kiss" even at different random seeds. We observed that some artist names are strongly associated with certain images and we believe this is due to dataset memorization as also discussed in [57, 58].

Another interesting issue is, for certain artists, if the prompt content diverges too vastly from their conventional “content", the SD model completely ignores the content part sometimes and only generates the “content" typical of the artist. For example, even when the prompt is A painting of a woman in the style of Thomas Kinkade, the SD model still outputs an image with charming cottages, tranquil streams, or gardens. The SD model sometimes completely ignored the content element in the prompt.

#### E.0.4 Some Caveats.

We emphasize that the above results should be taken with a grain of salt. Firstly, the LAION dataset, which we used, is inherently noisy, despite the sanitization steps we implemented as outlined in Section 4. The captions within this dataset frequently have issues. For example, tags relating to an artist or style might be absent, leading to inaccuracies in our evaluation. Even when the model correctly maps to the appropriate images in the dataset, these missing tags can cause correct results to be wrongly categorized as incorrect. Secondly, the curated style list from CLIP Interrogator is noisy. There are frequent re-occurrences of the same artist with different spellings in the style list. For example, if Van Gogh vs.Vincent Van Gogh ended up as different ‘style’ classes, and led to a few meaningless “errors.” Lastly, we assume that the model strictly adheres to the prompts during generation. However, our observations indicate that in a few cases, the SD model tends to ignore the style component of a prompt and focuses predominantly on the content. This divergence results in what would have been positive matches being classified as negatives. These factors collectively suggest that while the results are informative, they should be interpreted with an understanding of the underlying limitations and potential sources of error in the data and model behavior.

#### E.0.5 Qualitative results extended.

We present a few content-constrained generations and their respective top-2 matches using CSD ViT-L feature extractor in Fig. 9. We provide extended versions of Fig. 5 in Fig. 10 and Fig. 11. These figures depict the results obtained from Stable Diffusion (SD) generations using both “simple" artist prompts and “user-generated" prompts, along with their respective top-10 matches. The first column corresponds to the SD generation, while the subsequent columns display the identified matches.

To aid in the interpretation of these matches, we employed color-coded boxes to indicate the accuracy of the match. Specifically, green boxes represent true-positive matches, while red boxes indicate false-positive matches. However, it is important to note that the ground-truth labels assigned to the matched images may occasionally be incorrect because the ground-truth labels are generated from the LAION caption which may not always contain the artist’s name. Our analysis reveals several instances of such mislabeling, particularly evident in Fig. 11. Notably, numerous images that bear striking stylistic resemblance to the generated images are erroneously labeled as false positives.

These findings underscore the challenges involved in accurately assessing style copying in SD and emphasize the need for further exploration and refinement of evaluation methods.

[Figure 6]

[Figure 7]

[Figure 8]

###### (a) A woman in style of <Y> (b) A dog in style of <Y> (c) A house in style of <Y>

- Figure 9: Top row: Images generated by Stable Diffusion [49] using the prompt - A <X> in the style of <Y> where X comes from the set {woman, dog, house} and Y in order are Frida Kahlo, Josephine Wall, Gustav Klimt, Takashi Murakami, Picasso, Carne Griffiths, Leonid Afremov, Antoine Blanchard, Thomas Kinkade. Next two rows: top three style neighbors of the generated images from the LAION aesthetics datasets [54] as predicted by our model. The green and red box around the image indicate whether it was a true or false positive prediction based on whether the caption of the LAION image contained the name of the artist Y (used to generate the images).

[Figure 9]

###### Figure 10: Simple caption generations and matches: First column is SD generation, and the rest of the columns are top-10 matches in the LAION-Style database. The green box represents the correct match while the red box represents the incorrect match.

[Figure 10]

###### Figure 11: User-generated caption generations and matches: First column is SD generation, and the rest of the columns are top-10 matches in the LAION-Style database. The green box represents the correct match while the red box represents the incorrect match.

