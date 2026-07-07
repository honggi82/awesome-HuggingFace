## arXiv:2406.08478v2[cs.CV]18Jun2024

### What If We Recaption Billions of Web Images with LLaMA-3?

###### Xianhang Li⋆1 Haoqin Tu⋆1 Mude Hui⋆1 Zeyu Wang⋆1 Bingchen Zhao⋆2 Junfei Xiao3 Sucheng Ren3 Jieru Mei3 Qing Liu4 Huangjie Zheng5 Yuyin Zhou1 Cihang Xie1

⋆equal technical contribution

1UC Santa Cruz 2University of Edinburgh 3JHU 4Adobe 5UT Austin

[Figure 1]

[Figure 2]

[Figure 3]

Original: Cafe Van goch" stock image Ours: The first image shows the exterior of a cafe with a yellow and green facade. The cafe has a large glass window with red frames, and the sign above it reads 'Cafe Van Gogh'. There are several hanging lights with a warm glow, and the tables outside are set with white tablecloths, plates, and glasses. The chairs are black with a woven seat. The floor is wooden, and there is a small potted plant…

Original: Moyie BC railway Ours: A yellow train with red and blue accent s is traveling along a railway track that runs p arallel to a body of water. The train is surroun ded by a landscape with autumnal colors, fea turing trees with yellow and orange leaves. Th e railway track is bordered by a grassy area wi th some bushes and a dirt mound on the left side. The water is calm, reflecting the train an d the trees.

Original: Deluxe Twin Room Ours: A modern hotel room with a large bed covered in white and red bedding, a nightstand with a lamp, and a chair with a small table in front of it. The room has a large window with purple curtains, a white ceiling with a light fixture, and a dark wall with a patterned wallpaper. There are two colorful rugs on the floor, and the room is well-lit with natural light coming through the window.

[Figure 4]

[Figure 5]

[Figure 6]

Original: Western Kingbird Ours: A small, gray and yellow bird with a black beak and black eyes is perched on a brown branch. The bird has a fluffy appearance with a mix of gray and yellow feathers on its body. The background is a soft, out-offocus green, suggesting a natural environment.

Original: Deluxe Twin Room Ours: A slice of cake with a caramel

glaze and white frosting is placed on a white plate. The cake is garnished with a dollop of whipped cream on top

and a sprinkle of powdered sugar. The plate is on a white surface, and there is a watermark in the bottom right corner that reads 'Recipe Spain.com'."

Word distributions of our recaptions and the original captions.

[Figure 7]

[Figure 8]

[Figure 9]

Original: Eldora Speedway (OH) 6/8 Ours: A group of race cars is speeding on a di

Original :Buckle / gesp“The trike Brotherhood” Ours: A silver motorcycle charm with intricate

Original: siesta key florida picture Ours: A bustling beach scene with numerous people enjoying the sun and sand. The beach is lined with colorful umbrellas and sun loungers, and the water is a vibrant turquoise. The sky is partly cloudy, and the overall atmosphere is lively and crowded.

rt track. The lead car is red with the name 'Lu cas' on the side, followed by a blue car with t he number '21' and a black car with the numb er '2'. The cars are kicking up dust as they rac

details is attached to a red background with the words 'The Tike Brotherhood' and 'Blood & Honor' engraved in a cursive script. The charm

features a motorcycle with a sidecar, and the background has a wood grain texture."

e around the track.

Figure 1: Examples of the original caption and our recaption in DataComp-1B, and word distributions.

Preprint. Under review.

#### Abstract

Web-crawled image-text pairs are inherently noisy. Prior studies demonstrate that semantically aligning and enriching textual descriptions of these pairs can significantly enhance model training across various vision-language tasks, particularly text-to-image generation. However, large-scale investigations in this area remain predominantly closed-source. Our paper aims to bridge this community effort, leveraging the powerful and open-sourced LLaMA-3, a GPT-4 level LLM. Our recaptioning pipeline is simple: first, we fine-tune a LLaMA-3-8B powered LLaVA1.5 and then employ it to recaption ∼1.3 billion images from the DataComp-1B dataset. Our empirical results confirm that this enhanced dataset, Recap-DataComp1B, offers substantial benefits in training advanced vision-language models. For discriminative models like CLIP, we observe enhanced zero-shot performance in cross-modal retrieval tasks. For generative models like text-to-image Diffusion Transformers, the generated images exhibit a significant improvement in alignment with users’ text instructions, especially in following complex queries. Our project page is https://www.haqtu.me/Recap-Datacomp-1B/.

#### 1 Introduction

The exponential growth in data availability is one of the most paramount factors in driving the monumental successes of deep learning over the past decade [13, 32, 6, 57, 19, 17]. Typically, this data is sourced through web crawling with simple filtering mechanisms in place. While such an approach has facilitated large-scale data collection, exemplified by collections like LAION-400M [57] and LAION-5B [57] with billions of image-text records, it has inadvertently compromised data quality. As illustrated in Figure 1, these internet-crawled image-text pairs frequently exhibit misalignments between images and their corresponding textual content, and often, the textual descriptions are brief and lack detailed information.

To mitigate the noise present in web-crawled data, enhancements through post-processingimplemented via human-in-the-loop systems [61, 70] or automated pipelines [57, 28, 27]—are crucial, which help to train the advanced vision-language foundation models. Notably, both the close-sourced DALL-E 3 [41] and SORA [42] incorporate advanced captioning techniques to re-label their training datasets, a crucial step highlighted in their technical reports. Despite various efforts to open-source and replicate these methodologies [9, 28, 27, 35, 69, 16, 51], the community continues to face significant challenges in accessing high-quality, well-aligned image-text data at scale (e.g., at the billion level) for training advanced vision-language foundation models.

This paper endeavors to contribute to this community initiative, inspired specifically by the release of LLaMA-3 [62], a model demonstrating GPT-4-level capabilities across a variety of linguistic tasks. Additionally, recent studies have shown that leveraging LLaMA-3 can significantly enhance model performance on vision-language tasks [34, 65], comparable to those achieved by GPT-4V [1]. In response, we employ LLaMA-3 to develop our advanced captioner model. Our approach is straightforward: we first train a LLaMA-3-powered Llava model to act as an image captioner, which is then utilized to recaption the entire DataComp-1B dataset. As depicted in Figure 1, the resulting dataset, dubbed Recap-DataComp-1B, features enhanced textual descriptions and improved alignment with corresponding images, clearly surpassing its web-crawled counterparts. These quality enhancements are further quantitatively verified in Section 4.

Comprehensive evaluations highlight the significant improvements that Recap-DataComp-1B contributes to the training of advanced vision-language foundation models. Notably, this dataset enables CLIP models to achieve significant enhancements in their zero-shot cross-modal retrieval capabilities. It also enhances the alignment between generated images and text instructions in text-to-image generative models pre-trained on our dataset. We hope that the release of Recap-DataComp-1B will catalyze further developments in advanced vision-language foundation models, particularly encouraging the development within the open-source community.

Better CLIP Models

# { {

Tasks

[Figure 10]

[Figure 11]

[Figure 12]

|Retrieval|
|---|

Image-Caption

[Figure 13]

[Figure 14]

|Classification|
|---|

Recaption

[Figure 15]

[Figure 16]

|… …|
|---|

[Figure 17]

[Figure 18]

[Figure 19]

LLaVALLaMA3

[Figure 20]

[Figure 21]

[Figure 22]

|Image Generation|
|---|

DataComp-1B

Recaptioned DataComp-1B

|… …|
|---|

Better Diffusion Models

- Figure 2: The illustration of our recaptioning pipeline on DataComp-1B. We use LLaMA-3-powered LLaVA to reception images, which enables us to train stronger CLIP models and Text-to-Image Diffusion models.

#### 2 Related works

Vision-Language Foundation Models. CLIP [47] is one of the pioneering foundation models to connect image and text. By training on millions, and even billions, of image-text pairs [6, 14, 17, 19, 56–59], CLIP markedly showcases excessively strong zero-shot capacities, and furthermore, lays the cornerstone for building more advanced vision-language foundation models [3, 28, 27, 63, 35, 34, 10, 4, 65]. Apart from discriminative vision-language understanding, text-to-image generation models [15, 40, 41, 45, 48–50, 53, 68] have transformed the field of AI-generated content, facilitating the creation of high-quality images from natural language descriptions.

Enhancing Image-Text Data. Web-crawled image-text data [57, 19, 17] commonly face the problems of image-text misalignment and the low-quality of textual descriptions. Typically, there are two popular ways for improving the quality of these image-text pairs: 1) data filtering removes misaligned image-text pairs using various methods such as cleaning strategies [56, 19, 64], pretrained models [28, 57, 19], and human-assisted systems [61, 70, 77]; 2) data recaptioning improves the textual quality of image-text pair via generating new captions, which is the focus of this paper. To recaption data, LaCLIP [16] utilizes large language models (LLMs) like ChatGPT to rewrite the original captions; Nguyen et al. [39] employ BLIP2 [27] to recaption images. More recently, advanced large multimodal models have been applied to further enhance the quality of image captioning. For example, ShareGPT4V [9] employs GPT-4V to create highly descriptive captions from carefully crafted prompts and corresponding image inputs; the resulting dataset has significantly benefited the training of various models [7, 76, 12, 31, 18]. However, scaling such prompting with GPT-4V to billions of records is less practical, as it will drastically increase the monetary cost (of intensively calling OpenAI APIs) by more than 10,000×.

Our paper mostly follows the approach presented in [8, 38, 76, 12], where advanced open-source multimodal models like LLaVA [35] are employed for recaptioning purposes. However, our approach is distinguished by two major aspects: 1) we strongly enhance the LLM module in LLaVA, i.e., building with LLaMA-3; and 2) our recaptioning efforts are executed on a billion-scale dataset.

#### 3 Recaptioning Pipeline

Our recaptioning pipeline is centered around the advanced LLM LLaMA-3 [62], which achieves exceptionally strong performance in language understanding, reasoning, code generation, math problems, etc. [11, 60]. Specifically, we utilize the LLaVA framework [35] to fully harness its capabilities for visual understanding. We describe the detailed training procedures below.

###### 3.1 Model details

Model Configuration. We follow the setup of LLaVA-1.5 [33] to build our captioner model, except that we use LLaMA-3-8B as the language decoder because of its superior performance. The visual branch of CLIP ViT-L/14 [46] is used as the vision encoder. Two trainable MLP layers are employed on top of the vision encoder to project visual features into the language embedding space.

2-Stage Training. We also follow LLaVA-1.5 [33] for model training. Essentially we conduct instruction-tuning on the pre-trained LLM with its original auto-regressive training objective. In the

Table 1: Performance comparison of LLaVA. Model LLaVA-1.5-7B LLaVA-1.5-13B LLaVA-1.5-LLaMA3-8B (ours) GPT-4V

MMMU 33.6 36.4 45.2 56.8 MM-Vet 33.9 36.3 37.8 44.6

first stage, only the projection MLP is trained; in the second stage, we fine-tune both the projection MLP and the language decoder. Note that the vision encoder remains frozen all the time. Following the protocols in LLaVA [33], 558k image-text pairings filtered from LAION [56], CC [6],and SBU [43] are used as training data in the first stage; then 665k instructions-following data from LLaVA-1.5 [33], containing image-grounded conversation, image descriptions, and image-based complex reasoning tasks, are used for the second stage of training. To help our model generate higher-quality captions, we use the image-text pairs from HQ-Edit dataset [21] for further tuning.

Evaluations. To probe the visual understanding and reasoning ability of our LLaVA-1.5-LLaMA38B model, we opt for two comprehensive multi-modal evaluation benchmarks, MMMU [72] and MM-Vet [71]. These benchmarks assess a broad range of capabilities such as recognition, spatial awareness, OCR, knowledge, and language generation. As reported in Table 1, on both benchmarks, our LLaVA-1.5-LLaMA3-8B model surpasses the vanilla LLaVA-1.5-7B model by a significant margin. These results also substantially beat the considerably larger LLaVA-1.5-13B model, clearly demonstrating the superior visual understanding and reasoning ability of our model.

###### 3.2 Recaptioning DataComp-1B

With this advanced LLaVA model, we next use it to generate captions in a scalable and detailed manner, given the visual input, and the following text prompt:

"Please generate a detailed caption of this image. Please be as descriptive as possible."

As for the dataset, we opt for DataComp-1B [19], a widely accessible, large-scale vision-language dataset comprising ∼1.3 billion web-crawled image-text pairs. To ensure its quality, DataComp-1B is already a curated subset from a much larger collection of 12.8 billion image-text pairs and has been subjected to rigorous preprocessing which includes safety checks, deduplication, CLIP score filtering, and image-based filtering. Despite these efforts, the quality of the original captions in DataComp-1B still exhibits relatively low quality.

We apply our well-trained LLaVA-1.5-LLaMA3-8B model to recaption the entire DataComp-1B dataset. Specifically, captions are generated auto-regressively via greedy decoding, with the maximum output token length set to 128. We term this newly recaptioned dataset Recap-DataComp-1B.

#### 4 Analyzing Recap-DataComp-1B

This section collects and presents a quantitative analysis of our generated captions on DataComp1B. We primarily focus on two aspects: 1) the inherent features of the captions, including word distributions and average lengths; and 2) the semantic quality of the captions, evaluated in terms of the matching similarity between images and captions and the inherent textual quality of the captions.

###### 4.1 Word & Length Distribution

We begin our analysis by comparing the word frequency distributions between our recaptioned content and that from the original DataComp-1B, as illustrated in Figure 1, analyzing a randomly sampled subset of approximately 0.35 billion instances. Our findings reveal that the recaptioned content displays a considerably richer vocabulary, capturing 82.86% tokens of the word collections from both ours and the original caption data. Additionally, there is a noticeable variety in the usage of nouns and adjectives in our captions (e.g., “white” and “background”). We argue that this increased lexical diversity is a direct consequence of the extended length of our data. We thus present the distribution of caption lengths in Figure 3 to highlight this difference. On average, our recaptioned data demonstrates a longer sequence length of 49.43, whereas the original DataComp captions have a much shorter length of 10.22. These observations validate that our Recap-DataComp-1B surpasses the original DataComp-1B version in terms of both caption length and diversity.

- Figure 3: Average length distributions of both the original captions and our recaptioned data in DataComp-1B.

###### 4.2 GPT-4V & CLIP Evaluations

Next, we evaluate the semantic quality of recaptioned content using two models: 1) CLIP [47], which measures the semantic similarity between captions and images, and 2) GPT-4V [2], which assesses the fluency and alignment of captions with the given images.

For the CLIP evaluation, we analyze a subset of 180,000 image-text pairs. Interestingly, we note that, when using the standard CLIP-large model with ∼428M parameters for this measurement, our recaptioned content performs just comparably to the original captions (49.57 vs. 50.43). We attribute this result primarily to the limitations of the standard CLIP model, which is trained on ‘short’ captions and may inadequately capture the nuances in semantic similarity for longer captions. To probe deeper into semantic alignment between long captions and images, we utilize the LongCLIP-Large model [76], which is specifically fine-tuned to handle longer captions. With this setup, the LongCLIP score of our newly generated caption impressively attains 89.91, nearly 9× higher than the LongCLIP score of the original DataComp captions (i.e., only 10.09).

In addition, to evaluate both the textual quality and the alignment of the captions with their corresponding images, we randomly select 10,000 instances for GPT-4V [2] evaluation, employing the prompting strategy outlined below (CAPTION is the textual input), as per [44, 26].

|GPT-4V Evaluation Instruction: [Image Caption] CAPTION<br><br>Rate whether the caption is of high-quality and fluent and correctly matches the given image. The rating should be 1-5, where 1 is incorrect and not fluent at all, and 5 is correct and very fluent. Try to just give a numerical rating.<br><br>Your response should be in the format: Rating: (int)|
|---|

We can observe that our recaptioned content achieves markedly superior ratings, registering an average rating increase of 0.43 (from 3.71 to 4.14). Together with the findings from Section 4.1, this confirms the superior quality of our newly generated captions, in terms of length, vocabulary diversity, semantics, and image-text alignment.

#### 5 CLIP

CLIP [47] stands as a widely utilized vision-language model, where an image encoder and a text encoder are jointly trained to predict correct matches across entire batches of image-text pairs. In this section, we delve into the advantages of training CLIP models with our Recap-DataComp-1B dataset. We anticipate that CLIP models trained on this dataset will exhibit superior zero-shot cross-modal retrieval capabilities and enhanced text understanding, especially with long and complex textual inputs, given the improved quality of our recaptions.

###### Table 2: Recap-CLIP model configurations used in our paper.

Embed Vision Transformer Text Transformer # params (M) model dim layers width heads layers width heads vision text total S/16 384 12 384 6 12 384 6 22 33 55 B/16 512 12 768 12 12 512 8 86 53 141 L/16 768 24 1024 16 12 768 12 303 109 414 H/14 1024 32 1280 16 24 1024 16 631 334 967

- Table 3: Train with mixed captions. We choose Recap-CLIP-B/16 for this ablation. Larger p represents a higher ratio of the original caption. We report top-1 zero-shot classification accuracy on ImageNet-1K and top-1 recall for retrieval tasks. *concat: Concat two types of captions.

ImageNet-1K COCO R@1 Flickr30K R@1 Validation I→T T→I I→T T→I

mixed ratio p

- 0.0 36.0 53.0 34.1 74.1 53.5

- 0.1 58.4 60.9 40.5 83.9 65.5

- 0.2 62.5 61.7 41.4 85.8 65.7

- 0.3 65.1 62.7 42.6 86.3 67.0

- 0.4 66.7 62.6 42.5 87.4 67.7

- 0.5 67.2 61.9 42.7 85.9 66.7

- 0.6 68.0 62.2 42.4 86.0 67.4

- 0.7 68.4 60.7 42.3 86.3 66.9

- 0.8 69.2 61.5 42.2 85.2 66.9

- 0.9 69.2 60.6 41.1 86.0 65.7 1.0 69.7 57.3 37.7 84.2 63.0

*concat 43.3 57.8 35.6 80.2 56.4

###### 5.1 Experiment settings

Training. For reference, we term the CLIP model trained on our Recap-DataComp-1B dataset as Recap-CLIP. Our training pipeline primarily follows CLIPA [30, 29], which incorporates a twostate training, i.e., a pre-training process with a small image size followed by a fine-tuning stage incorporating a larger image resolution. We set the text token length to 128 to accommodate the learning of long captions presented in Recap-DataComp-1B. We conduct experiments using three model scales: S/16, B/16, and L/16, with details listed in Table 2. The AdamW [37] optimizer is used for training. In the pre-training phase, the model is trained with 2.56 billion samples with a reduced image size of 112, including a warm-up phase involving 51.2 million samples. The batch size and base learning rate are set to 32,768 and 8e-6, respectively. For the subsequent fine-tuning phase, we increase the image size to 224 and train the model on 128 million samples with a 25.6 million sample warm-up. Here, we adjust the batch size to 16,384 and the learning rate to 4e-7.

Evaluation. The efficacy of Recap-CLIP is gauged via several metrics. We evaluate zero-shot image classification on the ImageNet-1K dataset [52] and assess zero-shot cross-modal retrieval performance using the validation set of MSCOCO 2014 [32] and the test set of Flickr30K [66]1, following the established practices [47, 30, 74, 75].

We present our results from three aspects. First, we explore the impacts of differing mix ratios between original captions and our enhanced recaptions on CLIP performance. Next, we analyze the effects of enlarging the size of the CLIP text encoder. Lastly, we investigate the text understanding capability of our Recap-CLIP, via testing on VG-Attribute [73], which evaluates attributes understanding ability, and Urban1K [76], which tests the model’s ability to handle long text.

###### 5.2 Training with Mixed Captions

As pointed out by DALL-E 3 [41], blending both the briefgenuine captions and the long informative generated captions can effectively prevent the model from unwanted overfitting to recaption data. Therefore, we hereby first study the effect of varying mix ratios between the original captions and our

1We employ the widely used Karpathy split [24] of MSCOCO and Flickr30K.

- Table 4: Train with larger text encoder. We set p = 0.8 for recaption-based models. We report zero-shot top-1 accuracy on ImageNet-1K and top-1 recall on COCO and Flickr30K.

ImageNet-1K COCO R@1 Flickr30K R@1 Validation I→T T→I I→T T→I

vision encoder text encoder re-caption

small 60.7 49.2 30.1 73.5 53.3 small 60.2 53.7 34.3 78.6 57.9 base 61.7 +1.5% 56.4 +2.7% 34.8 +0.5% 79.7 +1.1% 59.1 +1.2%

S/16

base 69.7 57.3 37.7 84.2 63,0 base 69.2 61.5 42.2 85.2 66.9 large 69.8 +0.6% 62.9 +1.4% 42.8 +0.6% 86.7 +1.5% 67.3 +0.4%

B/16

large 74.1 60.2 41.9 86.0 68.5 large 73.8 64.3 46.1 88.3 70.5 huge 74.2 +0.4% 66.0 +1.7% 46.6 +0.5% 89.9 +1.6% 72.7 +2.2%

L/16

recaptions on the training of the Recap-CLIP B/16 model, as detailed in Table 2. Specifically, for each sample in a training batch, we randomly sample the original caption with probability 0 ≤ p ≤ 1 and our captions with probability 1 − p, referring to the mixed ratio:

Original with probability p Recaption with probability 1 − p

Caption =

This strategy ensures that each batch contains a mixture of our recaption and the original captions controlled by probability p. The randomness allows each sample to encounter different captions across training epochs, potentially enhancing the model’s generalization.

Main results. Our findings are summarized in Table 3. We observe that reducing the mixed ratio p (i.e., increasing the proportion of our recaption data) initially leads to an improvement followed by a decline in cross-modal retrieval performance. This initial improvement suggests that highquality recaptioned data effectively enhances contrastive learning. However, the subsequent decrease indicates that the original captions from DataComp-1B provide necessary training regularization, preventing the model from overly adapting to the specific qualities of the recaption data. Interestingly, we also observe that the performance of CLIP is relatively insensitive to certain variations in the mix ratio p, as evidenced by the consistent enhancement over the baseline (i.e. p=1.0) across all four different cross-modal retrieval metrics when varying p from 0.2 (80% recaption data) to 0.9 (10% recaption data). For instance, setting p at 0.9 and 0.2 both yields a similar performance enhancement of ∼3.5%, with the peak performance occurring at p=0.5, which delivers a substantial ∼5% boost.

But meanwhile, we note that incorporating our recaptions (negatively) affects the zero-shot classification task, exemplified by the consistent performance degradation across varying p values from

- 0 to 0.9. The phenomenon is similarly observed in the recent work [76] where they note directly fine-tuning on long text can significantly hurt the CLIP performance and therefore propose several techniques for enhancing learning with long texts. In this study, given our primary focus is on assessing the quality of Recap-DataComp-1B, we choose the ratio p = 0.8 to strike a promising balance between the classification performance (i.e., only marginally drops 0.5%) and the cross-modal retrieval performance (i.e., with a significant 3.4% boost on average) for later ablations.

###### 5.3 Training with Larger Text Encoder

We hereby investigate how the size of the text encoder affects models trained on a mixture of the original captions and our recaptions (with p = 0.8). Specifically, we keep the architectural configuration of the vision branch as in Table 2 and only twitch the text encoder. For instance, in the case of the S/16 model, we change from a smaller text encoder with 33M parameters to a larger, base-sized one with 53M parameters.

Main Results Our results, as shown in Table 4, highlight that enlarging the text encoder can further enhance performance across all model scales. The average improvement for adopting a larger text encoder in retrieval tasks is 1.4%, 1.0%, and 1.5% for small, base, and large models, respectively, suggesting that larger text encoders can help the CLIP model learn better from semantically rich captions.

- Table 5: Larger text encoder with different mixed ratios. We choose Recap-CLIP-B/16 with large text encoder for this ablation.

mixed ratio p

ImageNet-1K COCO R@1 Flickr30K R@1 Validation I→T T→I I→T T→I

- 0.5 68.5 64.3 43.4 86.8 67.8

- 0.6 69.2 64.4 43.2 87.5 68.8

- 0.7 69.3 63.2 42.7 88.0 68.2

- 0.8 69.8 62.9 42.8 86.7 67.3

Table 6: Comparison on the Urban-1K and VG-Attribute benchmark.

method re-caption

Urban-1K VG

I→T T→I Attribute

|OpenAI-CLIP-B/16 [47] OpenCLIP-B/16 [22]| |67.4 53.3 62.6 62.5 63.1 59.9|
|---|---|---|
|Recap-CLIP-B/16| |53.2 50.9 57.1 85.0 +31.8% 87.3 +36.4% 66.4 +9.1%<br><br>|

Recap-CLIP-L/16

69.8 64.6 60.1 89.0 +19.2% 91.8 +27.2% 66.8 +6.7%

Moreover, we re-assess the balanced ratio of recaption data using a larger text encoder. Specifically, we gradually increase the ratio of recaption data from 20% to 50%, utilizing the Recap-CLIP-B/16 model with the large text encoder. The results are presented in Table 5. Compared to the prior results where an optimal ratio is achieved at p = 0.8, using a larger text encoder can further push this optimal ratio to p = 0.6. In other words, this result concludes that, compared to the vanilla version, a stronger cross-modal retrieval performance can be achieved if 1) more recaptions are used and 2) a larger text encoder is used.

- 5.4 More evaluations on text understanding

Recent works demonstrate that CLIP models suffer from poor long context understanding and delicate attribute understanding [73, 76]. Given the long, enriched, and better-aligned captions, we expect Recap-CLIP to exhibit better text understanding capability. Thus, we evaluate our Recap-CLIP model on two benchmarks: (1) Urban1K [76], a long-caption image-text retrieval benchmark that contains

- 1k urban images and corresponding GPT-4V captions; (2) VG-Attribution [73], a modified version of Visual Genome [25] to test model abilities to attribute properties to objects. The results are shown in Tab. 6.

We observe consistent significant improvement if the model is trained on our Recap-Datacomp-1B dataset. For both text-to-image and image-to-text retrieval on Urban-1K dataset, our Recap-CLIP models surpass the vanilla baseline by at least 19% and sometimes up to an astonishingly high 36%. On the VG-attribution dataset, it is worth noting that our Recap-CLIP brings a performance boost very close to that of the NegCLIP fine-tuning [73] (e.g. ∼9% vs. 10%), a lightweight downstream fine-tuning process designed to boost CLIP ability to understand attribute and order. Nonetheless, it is noteworthy that our Recap-CLIP is naturally equipped with better text understanding ability, without any specific targeted fine-tuning, indicating the importance of better captions in web-scale data.

- 6 Text-to-Image Generation

It has been known to the research community that training with generated (high-quality) pseudocaptions improves text-to-image generative models in terms of generation quality and prompt following ability [8, 7, 5], primarily due to the low information and high noise density presented in the original web-crawled captions. Therefore, we evaluate the quality of our generated captions by training Text-to-Image (T2I) generative models on Recap-DataComp-1B for further justification. We expect the enriched information in the generated descriptions to better align the visual content in images, and thus improve the performance of the T2I models.

Table 7: Text-to-Image evaluation on COCO-30K results of DiT-BASE/4, trained with different mix ratios on Recap-DataComp-1B. Note for GPT-4V Score, we use a subset of 3K for the evaluation.

Training Evaluation mixed ratio p

Raw Our COCO-Recap FID↓ CLIP Score↑ FID↓ CLIP Score↑ Recap-Clip Score↑ GPT-4V Score↑

0.00 37.6 29.2 27.8−8.4 32.5+3.1% 28.3+8.4% 2.53+1.1 0.05 38.5 29.1 27.9 32.5 28.0 2.51 0.10 36.0 29.7 27.2 32.7 28.2 2.51 0.15 35.8 29.9 28.2 33.0 28.1 2.45 0.20 35.8 29.8 28.4 32.7 28.0 2.53 0.50 35.3 29.3 30.2 31.9 26.7 2.13

- 0.75 31.3 29.4 32.7 31.2 25.8 1.89

- 1.00 32.5 28.9 36.2 29.3 19.9 1.40

Training. We adopt Diffusion Transformers (DiT) [45] as our T2I model, where the text condition is firstly extracted with a CLIP text encoder [47], and then injected into each DiT block with the cross-attention design. Specifically, we follow the image preprocessing pipeline in DiT [45], where the images are preprocessed to have a square resolution of 256. The model is trained on visual latent extracted using a pretrained auto-encoder with a downsampling ratio of 8 [50]. Similar to the setup in previous experiments, the training text consists of a mixture of raw captions from Datacomp-1B, with a specified proportion p, and the rest of the captions replaced by refined captions from RecapDatacomp-1B. Moreover, the training batch size is 2048, and the AdamW optimizer [37] is used with a constant 1e-4 learning rate, without any warm-up schedule or weight decay. We name the resulting model Recap-DiT.

Evaluation. For sampling, we set the classifier-free guidance scale as 10 and use 250 DDPM steps to generate 30k images with captions from MSCOCO and our improved generated captions for zero-shot generation evaluation. We calculate Fréchet Inception Distance (FID) [20] with the reference images from MSCOCO [32] and CLIP score with both OpenAI ViT-B/32 model [47] and our own RecapCLIP ViT-L/16 model, following the established pipeline in prior T2I works [5, 67, 54, 23, 36, 78, 55]. Additionally, following the GPT-4V metric introduced in Section 4.2, we randomly select a subset of 3,000 our generated images for GPT-4V evaluation.

Main results. We report our observations in Tab. 7. Interestingly, when using raw COCO captions to generate 30,000 images for evaluation, the model trained with data integrated with our RecapDatacomp (for p < 1) demonstrates a better CLIP score, indicating improved vision-language alignment. However, there is no significant improvement observed in terms of FID. Our hypothesis is that the model adapts to the more informative and descriptive prompts, and could unleash its full potential only when similar informative testing prompts are provided.

Therefore, in another setting, we evaluate images generated using our LLaVA-1.5-LLaMA3-8B recaptioned version of the raw COCO captions. Here, we observe consistent and significant improvements in both FID and CLIP scores, particularly when more than half of the recaptioned data are integrated into the training dataset. Notably, models trained on Recap-Datacomp-1B (p = 0) surpass those trained on the vanilla Datacomp-1B (p = 1) by a large margin, with improvements observed in FID (-8.4), CLIP score (+3.1), Recap-CLIP score (+8.4), and GPT-4V score (+1.1). These observations justify that Recap-Datacomp-1B better reveals the potential of text-to-image models in generating images with high visual quality and improved alignment with textual conditions.

Larger models. We further train a larger model, DiT-L/2, for 1 epoch with a mixed ratio of p = 0.0, while keeping other training parameters unchanged. The model achieves an FID of 25.14 and a CLIP Score of 34.82. In Figure 4, we visually compare the generated results of DiT-L/2 and DiT-B/4 at p = 0.0. It is evident that although the quantitative scores may not show substantial improvement, as we scale up the model, there is a noticeable enhancement in the alignment between the generated images and the corresponding text, i.e., this improved alignment results in higher-quality images that are able to capture and express more intricate details. These results confirm that DiT models trained on our recaption DataComp-1B exhibit robust scalability for text-to-image generative tasks.

##### DiT-B/4 DiT-L/2 DiT-B/4 DiT-L/2

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

A group of elephants is grazing in a grassy field with trees in the background.

A giraffe stands in a lush green field with tall grass and trees in the background.

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

A black dog is standing in a sunny backyard with a variety of potted plants and flowers around it. The dog is wearing a blue collar and is looking towards the right side of the image.

Two brown bears are playfully splashing water at each other in a river.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

A white kitchen with a wooden cabinet above the sink, a white door with a window, and a white wall. The sink is filled with various items including a bottle, a cup, and a spoon. There is a paper towel roll on the counter, and the floor appears to be tiled.

A silver bus with a futuristic design is parked on the side of the road. The bus features a prominent pink

and black color scheme with a metallic finish. There is one people standing near the bus. The bus has a large window that covers most of its side, and there are

two small windows on the front.

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

- Figure 4: Visual comparison of generate results from DiT-L/2 and DiT-B/4 at p = 0.0, DiT-L/2 has better text comprehension and image generation than DiT-B/4. We mark entities in the instruction.

#### 7 Conclusion

This paper introduces Recap-DataComp-1B, a large-scale image dataset paired with detailed textual descriptions, generated using the LLaMA-3-powered Llava model. Our comprehensive analysis reveals that, compared to the original, web-crawled textual data, these generated descriptions align more accurately with their corresponding images and are more detailed. Utilizing Recap-DataComp1B for training resulted in consistent enhancements across various models, notably CLIP, particularly in image-to-text and text-to-image retrieval tasks, and in text-to-image Diffusion models, specifically in their ability to follow more closely to user-provided text instructions. By providing this highquality, publicly available, large-scale image-text dataset, we hope to inspire ongoing research and development that will push the boundaries of developing vision-language foundation models, more particularly in the open-source community.

A small airplane is flying low over a grassy field with a dense forest in the background. The sun is shining through the trees, creating a warm glow. The airplane is positioned in the center of the image, with its wings spread and the cockpit clearly visible. The field appears to be a landing strip, and the forest is composed of tall, straight trees that create a natural barrier.

A person in a black riding jacket and white pants is riding a brown horse with a white patch on its face. The rider is wearing a black helmet and is positioned in the center of the image. The horse is galloping across a grassy field with yellow flowers scattered throughout. In the background, there are trees and a wooden fence.

###### Acknowledge

This work is partially supported by a gift from Adobe, TPU Research Cloud (TRC) program, Google Cloud Research Credits program, AWS Cloud Credit for Research program, Edinburgh International Data Facility (EIDF) and the Data-Driven Innovation Programme at the University of Edinburgh.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4V(ision) system card. OpenAI Research Blog, 2023.
- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In NeurIPS, 2022.
- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond. arXiv preprint arXiv:2308.12966, 2023.
- [5] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2023.
- [6] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR, 2021.
- [7] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-\sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024.
- [8] Junsong Chen, Jincheng YU, Chongjian GE, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-$\alpha$: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In ICLR, 2024.
- [9] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023.
- [10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023.
- [11] Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E. Gonzalez, and Ion Stoica. Chatbot arena: An open platform for evaluating llms by human preference. arXiv preprint arXiv:2403.04132, 2024.
- [12] Xiangxiang Chu, Limeng Qiao, Xinyu Zhang, Shuang Xu, Fei Wei, Yang Yang, Xiaofei Sun, Yiming Hu, Xinyang Lin, Bo Zhang, et al. Mobilevlm v2: Faster and stronger baseline for vision language model. arXiv preprint arXiv:2402.03766, 2024.
- [13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.
- [14] Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin Johnson. Redcaps: Web-curated image-text data created by the people, for the people. arXiv preprint arXiv:2111.11431, 2021.
- [15] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. In NeurIPS, 2021.
- [16] Lijie Fan, Dilip Krishnan, Phillip Isola, Dina Katabi, and Yonglong Tian. Improving clip training with language rewrites. In NeurIPS, 2024.
- [17] Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, and Vaishaal Shankar. Data filtering networks. arXiv preprint arXiv:2309.17425, 2023.
- [18] Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Youqiang Zhang, and Junshi Huang. Dimba: Transformer-mamba diffusion models. arXiv preprint arXiv:2406.01159, 2024.

- [19] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. arXiv preprint arXiv:2304.14108, 2023.
- [20] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017.
- [21] Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit: A high-quality dataset for instruction-based image editing. arXiv preprint arXiv:2404.09990, 2024.
- [22] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip. arXiv preprint arXiv:2212.07143, 2021.
- [23] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up GANs for text-to-image synthesis. In CVPR, 2023.
- [24] Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In CVPR, 2015.
- [25] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. In IJCV, 2017.
- [26] Tony Lee, Yifan Mai, Chi Heem Wong, Josselin Somerville Roberts, Michihiro Yasunaga, Faarzan Kaiyom, Rishi Bommasani, and Percy Liang. The first steps to holistic evaluation of vision-language models, May 2024.
- [27] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 2023.
- [28] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, 2022.
- [29] Xianhang Li, Zeyu Wang, and Cihang Xie. CLIPA-v2: Scaling CLIP Training with 81.1% Zero-shot ImageNet Accuracy within a $10,000 Budget; An Extra $4,000 Unlocks 81.8% Accuracy. arXiv preprint arXiv:2306.15658, 2023.
- [30] Xianhang Li, Zeyu Wang, and Cihang Xie. An Inverse Scaling Law for CLIP Training. In NeurIPS, 2024.
- [31] Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Jinfa Huang, Junwu Zhang, Munan Ning, and Li Yuan. Moe-llava: Mixture of experts for large vision-language models. arXiv preprint arXiv:2401.15947, 2024.
- [32] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014.
- [33] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following, 2023.
- [34] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge. https://llava-vl.github.io/blog/2024-01-30-llava-next/, January 2024.
- [35] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2024.
- [36] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, and Qiang Liu. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. arXiv preprint arXiv:2309.06380, 2023.
- [37] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [38] Guansong Lu, Yuanfan Guo, Jianhua Han, Minzhe Niu, Yihan Zeng, Songcen Xu, Zeyi Huang, Zhao Zhong, Wei Zhang, and Hang Xu. Pangu-draw: Advancing resource-efficient text-to-image synthesis with time-decoupled training and reusable coop-diffusion. arXiv preprint arXiv:2312.16486, 2023.
- [39] Thao Nguyen, Samir Yitzhak Gadre, Gabriel Ilharco, Sewoong Oh, and Ludwig Schmidt. Improving multimodal datasets with image captioning. In NeurIPS, 2024.

- [40] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.
- [41] OpenAI. Dall·e 3 system card. OpenAI Research Blog, 2023.
- [42] OpenAI. Video generation models as world simulators. OpenAI Research Blog, 2024.
- [43] Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. In NeurIPS, 2011.
- [44] Piotr Padlewski, Max Bain, Matthew Henderson, Zhongkai Zhu, Nishant Relan, Hai Pham, Donovan Ong, Kaloyan Aleksiev, Aitor Ormazabal, Samuel Phua, et al. Vibe-eval: A hard evaluation suite for measuring progress of multimodal language models. arXiv preprint arXiv:2405.02287, 2024.
- [45] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023.
- [46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021.
- [47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.
- [48] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv::2204.06125, 2022.
- [49] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021.
- [50] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.
- [51] Noam Rotstein, David Bensaid, Shaked Brody, Roy Ganz, and Ron Kimmel. Fusecap: Leveraging large language models to fuse visual data into enriched image captions. arXiv preprint arXiv:2305.17718, 2023.
- [52] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. In IJCV, 2015.
- [53] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. In NeurIPS, 2022.
- [54] Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, and Timo Aila. StyleGAN-T: Unlocking the power of GANs for fast large-scale text-to-image synthesis. arXiv preprint arXiv:2301.09515, 2023.
- [55] Axel Sauer, Dominik Lorenz, A. Blattmann, and Robin Rombach. Adversarial diffusion distillation. ArXiv, abs/2311.17042, 2023.
- [56] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. In NeurIPS, 2022.
- [57] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.
- [58] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In ACL, 2018.
- [59] Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. Wit: Wikipediabased image text dataset for multimodal multilingual machine learning. In SIGIR, 2021.
- [60] Ingrid Stevens. Llama 3’s performance benchmark values explained. https://medium.com/. Accessed: 2024-06-05.

- [61] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, LiangYan Gui, Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models with factually augmented rlhf. arXiv preprint arXiv:2309.14525, 2023.
- [62] Meta LLaMA Team. Introducing Meta Llama 3: The most capable openly available LLM to date. https://ai.meta.com/blog/meta-llama-3/, 2024.
- [63] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. Simvlm: Simple visual language model pretraining with weak supervision. In ICLR, 2022.
- [64] Hu Xu, Saining Xie, Xiaoqing Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data. In ICLR, 2023.
- [65] Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, and Gao Huang. LLaVA-UHD: an lmm perceiving any aspect ratio and high-resolution images. arXiv preprint arXiv:2403.11703, 2024.
- [66] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. In TACL, 2014.
- [67] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. In TMLR, 2022.
- [68] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation. In TMLR, 2022.
- [69] Qiying Yu, Quan Sun, Xiaosong Zhang, Yufeng Cui, Fan Zhang, Xinlong Wang, and Jingjing Liu. Capsfusion: Rethinking image-text data at scale. arXiv preprint arXiv:2310.20550, 2023.
- [70] Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. arXiv preprint arXiv:2312.00849, 2023.
- [71] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. In ICML, 2024.
- [72] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024.
- [73] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why vision-language models behave like bags-of-words, and what to do about it? In ICLR, 2022.
- [74] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023.
- [75] Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In CVPR, 2022.
- [76] Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. Long-clip: Unlocking the long-text capability of clip. arXiv preprint arXiv:2403.15378, 2024.
- [77] Lei Zhang, Fangxun Shu, Sucheng Ren, Bingchen Zhao, Hao Jiang, and Cihang Xie. Compress & align: Curating image-text data with human knowledge. arXiv preprint arXiv:2312.06726, 2023.
- [78] Mingyuan Zhou, Zhendong Wang, Huangjie Zheng, and Hai Huang. Long and short guidance in score identity distillation for one-step text-to-image generation. arXiv preprint arXiv:2406.01561, 2024.

