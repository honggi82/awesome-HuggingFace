# arXiv:2402.19479v1[cs.CV]29Feb2024

## Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers

Tsai-Shien Chen1,2,∗ Aliaksandr Siarohin1 Willi Menapace1,3,∗ Ekaterina Deyneka1 Hsiang-wei Chao1 Byung Eun Jeon1 Yuwei Fang1 Hsin-Ying Lee1 Jian Ren1 Ming-Hsuan Yang2 Sergey Tulyakov1

1Snap Inc. 2University of California, Merced 3University of Trento

https://snap-research.github.io/Panda-70M

HDVILA-100M

Panda-70M (Ours)

|“It is a close-up shot of a brown and white english bulldog with wrinkles on its face, sitting on a person's lap.”<br><br>[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>“It is a red and purple betta fish swimming in a tank with gravel and plants.”<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>“A person is adding chicken broth to a pot of quinoa on a stove.”<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>“A basketball player is dribbling the ball and shooting it into the hoop.”<br><br>[Figure 10]<br><br>[Figure 11]<br><br>[Figure 12]|
|---|

|“He thought he was gonna get shows terrible communication on the teams part.”<br><br>[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>“We're gonna cook this all together stirring it constantly for just a minute until it smells nice and fragrant.”<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>“Yeah, now everybody thought that we couldn't replace cat; yeah, because you're such animal lovers.”<br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>“It's good that we have aquariums they bring this wonderful experience into our homes for special moments”<br><br>[Figure 22]<br><br>[Figure 23]<br><br>[Figure 24]|
|---|

Figure 1. Comparison of Panda-70M to the existing large-scale video-language datasets. We introduce Panda-70M, a large-scale video dataset with captions that are annotated by multiple cross-modality vision-language models. Compared to text annotations in existing dataset [80], captions in Panda-70M more precisely describe the main object and action in videos (highlighted in green). Besides, videos in Panda-70M are semantically coherent, high-resolution, and free from watermarks. More samples can be found in Appendix E.

### Abstract

The quality of the data and annotation upper-bounds the quality of a downstream model. While there exist large text corpora and image-text pairs, high-quality video-text data is much harder to collect. First of all, manual labeling is more time-consuming, as it requires an annotator to watch an entire video. Second, videos have a temporal dimension, consisting of several scenes stacked together, and showing multiple actions. Accordingly, to establish a video dataset with high-quality captions, we propose an automatic approach leveraging multimodal inputs, such as textual video description, subtitles, and individual video frames. Specifically, we curate 3.8M high-resolution videos from the pub-

∗ This work was done while interning at Snap.

licly available HD-VILA-100M dataset. We then split them into semantically consistent video clips, and apply multiple cross-modality teacher models to obtain captions for each video. Next, we finetune a retrieval model on a small subset where the best caption of each video is manually selected and then employ the model in the whole dataset to select the best caption as the annotation. In this way, we get 70M videos paired with high-quality text captions. We dub the dataset as Panda-70M. We show the value of the proposed dataset on three downstream tasks: video captioning, video and text retrieval, and text-driven video generation. The models trained on the proposed data score substantially better on the majority of metrics across all the tasks.

Table 1. Comparison of Panda-70M and other video-language datasets. We split the datasets into two groups: the group at the top is annotated by ASR, and the group at the bottom is labeled with captions.

Dataset Year Text Domain #Videos Avg/Total video len Avg text len Resolution HowTo100M [52] 2019 ASR Open 136M 3.6s 134.5Khr 4.0 words 240p ACAV [32] 2021 ASR Open 100M 10.0s 277.7Khr - YT-Temporal-180M [87] 2021 ASR Open 180M - - - HD-VILA-100M [80] 2022 ASR Open 103M 13.4s 371.5Khr 32.5 words 720p MSVD [13] 2011 Manual caption Open 1970 9.7s 5.3h 8.7 words LSMDC [58] 2015 Manual caption Movie 118K 4.8s 158h 7.0 words 1080p MSR-VTT [79] 2016 Manual caption Open 10K 15.0s 40h 9.3 words 240p DiDeMo [3] 2017 Manual caption Flickr 27K 6.9s 87h 8.0 words ActivityNet [11] 2017 Manual caption Action 100K 36.0s 849h 13.5 words YouCook2 [93] 2018 Manual caption Cooking 14K 19.6s 176h 8.8 words VATEX [73] 2019 Manual caption Open 41K ∼10s ∼115h 15.2 words Panda-70M (Ours) 2024 Automatic caption Open 70.8M 8.5s 166.8Khr 13.2 words 720p

- 1. Introduction We enter an era where the size of computing and data are indispensable for large-scale multimodal learning. Most breakthroughs are achieved by large-scale computing infrastructure, large-scale models, and large-scale data. Due to these integral components, we have powerful text-toimage [4, 57, 59, 61, 83] and image-to-text models [2, 36, 43, 53]. Scaling the model size or the compute is challenging and expensive; however, it requires a finite amount of engineering time. Scaling the data is relatively more challenging, as it takes time for a human to analyze each sample.

Especially, compared to image-text pairs [10, 12, 62], video-text pairs are even harder to obtain. First, annotating videos is more time-consuming, as an annotator needs to watch the entire video before labeling. Second, videos often contain multiple scenes stitched together and consist of temporally varying content. Finally, meta-information, such as subtitles, video description, and voice-over, is often too broad or not correctly aligned in time or cannot precisely describe a video. For example, several 100M-scale datasets, such as HD-VILA-100M [80] and HowTo100M [52], are annotated by automatic speech recognition (ASR). However, as shown in Figure 1, the subtitles usually fail to include the main content and action presented in the video. This limits the value of such datasets for multimodal training. We summarize the datasets available to the community in Table 1. Some are low-resolution, some are annotated by ASR, some contain data from a limited domain, some are small-scale, and some offer short captions.

In this work, we present a large-scale dataset containing 70M video clips with caption annotations. It includes high-resolution videos from an open domain with rich captions averaging 13.2 words per caption. While manually annotating 70M videos is prohibitively expensive, we opt for automatic annotation. Our key insight is that a video typically comes with information from several modalities that can assist automatic captioning. This includes the title,

description, subtitles of the video, individual static frames, and the video itself. The value of this data cannot be fully maximized when only partially used. In comparison, we propose to utilize different combinations of multimodal data as inputs to various cross-modality captioning models. To substantiate this idea, we conduct a numerical analysis based on a human evaluation (the details are provided in Appendix B.3). If we use multiple cross-modality models to caption some video samples and evaluate the results by showing them to humans, we see that there is no single model able to generate good captions for more than 31% of videos. However, if we jointly collect all the captions from different models, we observe that 84.7% of videos can be annotated with at least one good caption.

To establish the dataset with this mindset, we begin by using 3.8M high-resolution long videos collected from HDVILA-100M [80] and process them through the following three steps. First, we design a semantics-aware video splitting algorithm to cut long videos into semantically consistent clips while striking the balance between semantics coherence and the duration of the video clips. Second, we use a range of cross-modality teacher models, including image captioning models [37] and image/video visual-questionanswering (VQA) models [38, 88, 94] with additional text inputs, such as video description and subtitles, to predict several candidate captions for a clip. Lastly, we collect a 100K video subset, where human annotators act as an oracle to select the best caption for each video. We use this dataset to finetune a fine-grained video-to-text retrieval model [39] which is then applied to the whole dataset to select the most precise caption as the annotation. Running multiple teacher models is computationally expensive and time-consuming. To pursue efficient video captioning at scale in the future, we train a student model to distill the knowledge from the teachers. The student model adopts a two-branch architecture which can take both visual and textual inputs to benefit the captioning from multimodal information.

[Figure 25]

- Figure 2. Video captioning pipeline. Given a long video, we first split it into several semantically coherent clips. Subsequently, we utilize a number of teacher models with different multimodal inputs to generate multiple captions for a video clip. Lastly, we finetune a fine-grained retrieval model to select the caption that best describes the video clip as the annotation.

Extensive experiments demonstrate that pretraining with the proposed Panda-70M1 can benefit several downstream tasks, including video captioning, video and text retrieval, and text-to-video generation. We also show that training a student model in a knowledge distillation manner facilitates learning a strong student model which can outperform any teacher model by more than 7.7% preference ratio as in Ta-

- ble 3, where the performance can be further enhanced by additional text inputs, like video description and subtitles.

- 2. Related Work Vision-Language Datasets. Training with millions or even billions of image-text pairs [10, 31, 55, 62, 86] has been shown to be effective in learning powerful image foundation models [2, 6, 21, 25, 27, 82]. With this work, our goal is to build a large video-language dataset containing rich captions. We compare related datasets in Table 1. Several precedent video-language datasets [3, 11, 13, 58, 73, 79, 93] contain data tackling various tasks, such as action recognition, video understanding, VQA, and retrieval. However, manually annotating data is costly and limits the scale of such datasets (typically they contain less than 120K samples). To alleviate the lack of data, the works of [52, 80, 87] propose to automatically annotate data with subtitles, generated by ASR. While this approach significantly increases the dataset scale reaching 100M of samples, the subtitles, unfortunately, do not precisely describe the main video content, as shown in Figure 1. In comparison, in this work, we propose an automatic captioning pipeline with the inputs of multimodal data that enables us to scale up the dataset of high-quality video-caption pairs to a 70M scale. Vision-Language Models learn the correlation between visual data (images or videos) and linguistic signals (words or sentences) and can be applied to several downstream applications, including text-driven image or video generation [4, 8, 29, 57, 59, 61, 65, 83, 92], captioning [2, 36, 37, 43, 63, 81], VQA [14, 38, 53, 88, 94] and re-

1We call our dataset Panda, drawing an analogy to Panda Po, who learns from multiple martial arts teachers.

trieval [17, 39, 49]. We utilize several vision-language models for the annotation of Panda-70M. BLIP-2 [37] introduces an efficient vision-language pretraining that can facilitate image captioning. We use BLIP-2 as one of the teachers and input a randomly sampled video frame for captioning. MiniGPT-4 [94] is an image VQA model that learns a projection layer to align a large language model (LLM) and a visual encoder. In addition to a video frame, we also input a prompt with extra text information, such as video description and subtitles, and ask the model to summarize all multimodal inputs. For the video modality, Video-LLaMA [88] and VideoChat [38] are both video VQA models and learn to extract LLM-compatible visual embeddings. We use both models and ask them to caption a video with prompt input. Besides, Unmasked Teacher [39] is a video foundation model which can facilitate video understanding. We finetune it to implement fine-grained retrieval and use it to select the more precise caption as the annotation.

Video Annotation through Multi-modal Models. With the aforementioned development on vision-language models, some concurrent works [7, 72, 75] also leverage these models for video captioning. VideoFactory [72] employs BLIP-2 [37] to caption video clips. However, as reported in Appendix B.3, the performance of a single BLIP-2 model is suboptimal. More similar to our captioning pipeline, InternVid [75] and Stable Video Diffusion [7] also use multiple captioning models which are followed by an LLM for summarization. In practice, we found the LLM would propagate errors from noisy outputs of vision-language models.

### 3. Methodology

To build Panda-70M, we utilize 3.8M high-resolution long videos collected from HD-VILA-100M [80]. We then split them into 70.8M semantically coherent clips as described in Section 3.1. Section 3.2 shows how multiple cross-modality teacher models are used to generate a set of candidate caption annotations. Next, we finetune a fine-grained retrieval model to select the most accurate caption as detailed in Section 3.3. Finally, in Section 3.4, we describe our approach to

Table 2. Comparison of splitting algorithms. We split 1K long videos by three algorithms and test the semantics consistency of the output clips by the proposed Max Running LPIPS. Our splitting strikes a better balance for the trade-off between semantics consistency and clip length.

training a student captioning model using Panda-70M. The high-level view of our approach is shown in Figure 2.

- 3.1. Semantics-aware Video Splitting A desired video sample in a video-captioning dataset should have two somewhat contradictory characteristics. On the one hand, the video should be semantically consistent, so the video samples can better benefit the downstream tasks, such as action recognition, and the caption can also more accurately express its semantics content without ambiguity. On the other hand, the video cannot be too short or fragmentary to contain meaningful motion content, which is beneficial to tasks, like video generation.

Method Max running LPIPS↓ Avg Video Len Sub. Align [52, 80] 0.408 11.8s PySceneDetect [1] 0.247 4.1s Our Splitting 0.256 7.9s

[Figure 26]

[Figure 27]

Pretrained UMT Finetuned UMT Annotators

To achieve both goals, we design a two-stage semanticsaware splitting algorithm to cut a long video into semantically coherent clips. In the first stage, we split the video based on shot boundary detection [1], as the semantics often change when a new scene starts. In the second stage, we stitch adjacent clips if they are incorrectly separated by the first stage, ensuring the videos do not end up being too short. To do so, we use ImageBind [25] to extract embeddings of video frames and merge the adjacent clips if the frame embeddings from two clips are similar. We also implement additional procedures to handle: 1) long videos without any cut-scenes, 2) videos using complex transitions, such as fade-in and fade-out effects, which are not usually detected as cut-scenes, and 3) removal of redundant clips to increase the diversity of the dataset. More details of the splitting algorithm are in Appendix A. Notably, while our dataset focuses on fine-grained video-text pairs with consistent semantics, users can still acquire long videos with multiple cut-scenes by concatenating consecutive clips and captions, as these clips are split from the same long video.

Selective rate

Figure 3. Distributions of the selective rate of teacher models. We plot the distributions of the selective rate of eight teachers on 1,805 testing videos. The results are based on the selection of the pretrained (red) or finetuned (green) Unmasked Teacher [39] and human annotators (blue).

We start with a large pool including 31 captioning models. The introduction of the model pool is in Appendix B.1. Since running the inference of all models on 70M video clips is computationally expensive, we construct a short list of eight well-performing models based on a user study. The list is shown in the y-axis of Figure 3. More details of this process are in Appendix B.3. Briefly, the models are composed of five base models with different pretraining weights and input information. The five base models include VideoLLaMA [88] (video VQA), VideoChat [38] (video VQA), VideoChat Text [38] (natural language model which textualizes the video content), BLIP-2 [37] (image captioning), and MiniGPT-4 [94] (image VQA). To implement video captioning by cross-modality teacher models, we formulate distinct captioning processes tailored to each modality. For example, for the VQA models, in addition to visual data, we also input a prompt with additional text information and ask the models to summarize all multimodal inputs into one sentence. Details on the captioning process of each teacher model are described in Appendix B.2.

To quantitatively verify the semantic consistency of a video clip, we introduce Max Running LPIPS, which highlights the most significant perceptual change within a video clip. Formally, given an n-second video clip, we subsample the video frames each second and denote the keyframes as {f1,...,fn}. The Max Running LPIPS is formulated as:

##### max({LPIPS(fi,fi+1) | i ∈ [1,n − 1]}). (1)

where LPIPS(·,·) is the perceptual similarity [89] of two images. As in Table 2, our splitting achieves a better semantics consistency than the splitting based on the alignment of subtitles sentences [52, 80], while maintaining longer video length than the vanilla shot boundary detection [1].

- 3.2. Captioning with Cross-Modality Teachers Videos in HD-VILA-100M [80] contain rich multimodal information beneficial for captioning. Specifically, besides the video itself, there are also useful texts (e.g., video title, description, and subtitles) and images (e.g., individual video frames). Driven by this insight, we propose to use several captioning models with the inputs of different modalities.

We hypothesize that teacher models using different modality data perform well on different kinds of videos. For example, video models can perform better on videos with complex dynamics due to the additional modules to handle temporal information. On the other hand, image models can accurately caption the videos with rare and uncommon objects, since they were trained using large-scale datasets of

image-text pairs [62]. Finally, for videos that are visually hard to understand, VQA models have leverage as they can employ additional textual clues.

Video Clip Input

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

This hypothesis can be supported by a numerical evaluation. Specifically, we conduct a user study where the participants are asked to select the best caption from eight candidates. We plot the selective rate of each teacher model in

Text Input (Optional)

|[Title] Old VS New - 1966 Ford Mustang GT & 2018 Ford Mustang | Just a Quick Look [Subtitles] Today, we're gonna take a quick look at the 1966 Ford Mustang with a 289… [Description] Lets check out this beautiful 1966 Ford Mustang GT 289 in the showroom!|
|---|

- Figure 3 (blue bars). The results show that the best captions are generated by different teacher models. Moreover, the highest selective rate of an individual teacher model (i.e., BLIP-2 with opt6.7b [90]) is only 17.85%. This fact expresses the limited captioning capability of a single model on a wide variety of videos.

[Figure 36]

[Figure 37]

Tokenizer & Embedding

Video Encoder

K/V

K/V

[Figure 38]

[Figure 39]

[Figure 40]

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

Q

Q

Video Q-Former & Linear Projection

Text Q-Former

- 3.3. Fine-grained Video-to-Text Retrieval Given multiple candidate captions for a video, we seek the one that best aligns with the video content. An intuitive idea is to use the available generic video-to-text retrieval models [25, 39] to pick such a caption. Unfortunately, we find that they usually fail to pick the optimal result. One reason is that generic models are trained using contrastive learning objectives [15, 82] and learn to distinguish one sample from other completely unrelated samples2. In contrast, in our case, all candidate captions are highly relevant to the video sample and require the model to discern subtle distinctions within each caption for optimal performance.

gradient

Learned Queries

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

| |
|---|
| |
| |
| |

Text Representation Video Representation

[Figure 41]

Large Language Model

"A red mustang in a showroom with american flags on the wall."

Figure 4. Architecture of student captioning model.

selected captions (blue bars). We run the finetuned UMT on the whole dataset to select the best caption as the annotation as elaborated in Appendix C.3.

To tailor the retrieval model to our “fine-grained” retrieval scenario, we collect a subset of 100K videos, for which human annotators select the caption containing the most correct and detailed information about the main content of the video. We then finetune Unmasked Teacher [39] (UMT) on this dataset. We implement hard negative mining [16, 35] for contrastive loss, where the seven captions not selected by annotators compose the hard negative samples and are assigned a larger training weight. We describe the details of the dataset collection and finetuning of UMT in Appendix C.1 and C.2 respectively.

#### 3.4. Multimodal Student Captioning Model

While the aforementioned captioning pipeline can generate promising captions, the heavy computational demands hinder its capability to expand the dataset to an even larger scale. Indeed, one needs to run 8 + 1 different models to annotate a single video clip. To deal with this problem, we learn a student captioning model on Panda-70M to distill the knowledge from multiple teacher models.

As shown in Figure 4, the student model includes visual and text branches, leveraging multimodal inputs. For the vision branch, we use the same architecture as VideoLLaMA [88] to extract LLM-compatible video representation. For the text branch, a straightforward design is to directly input text embedding into the LLM. However, this will lead to two problems: first, the text prompt with video description and subtitles can be too long, dominating the decision of the LLM and burdening heavy computation; second, the information from the description and subtitles is often noisy and not necessary to align with the content of the video. To tackle this, we add a text Q-former to extract the text representation with fixed length and better bridge the video and text representations. The Q-former has the same architecture as the Query Transformer in BLIP-2 [37]. During training, we block the gradient propagation from the text branch to the vision branch and train the visual encoder only based on the video input. More details about the architecture and training of the student model are in Appendix D.

We quantitatively evaluate the retrieval performance of UMTs with and without finetuning on the validation set. The experiments indicate that a finetuned UMT can achieve 35.90% R@1 accuracy which significantly outperforms a pretrained UMT which has 21.82% R@1. Notably, we conducted a human agreement evaluation by asking two other persons to re-perform the annotation and comparing the results with the original annotations. The average human agreement score is only 44.9% R@1 showing that the task is subjective when more than one caption is equally good. Alternatively, if we consider the captions selected by any of the three persons as good captions (i.e., a video might have multiple good captions), UMT achieves 78.9% R@1. Besides, in Figure 3, we show that a finetuned UMT (green bars) can select the captions distributed similarly to human-

2Negative samples for contrastive learning [15] are usually randomly sampled from the within-batch data and show no association to the anchor.

- Table 3. Zero-shot video captioning (%). We compare Video-LLaMA [88] with official weight (pretrained on 2.5M videos and 595K images) and our Panda-2M pretraining weight. We also test our student model (with vision branch only) trained on the complete Panda70M dataset. We report BLEU-4 (B-4) [54], ROUGE-L (R) [40], METEOR (M) [5], CIDEr (C) [69], and BERTScore (BERT) [91] on two benchmarks MSR-VTT [79] and MSVD [13].

Method Pretraining Data

MSR-VTT MSVD B4↑ R↑ M↑ C↑ BERT↑ B4↑ R↑ M↑ C↑ BERT↑

Video-LLaMA [88] 2.5M vid + 595K img 5.8 30.0 15.9 14.3 84.5 12.7 43.0 23.6 38.5 87.3 Video-LLaMA [88] Panda-2M (Ours) 23.5 48.6 26.7 29.1 87.2 31.2 59.9 34.7 47.0 89.8

Student (Ours) Panda-70M (Ours) 25.4 50.1 27.7 31.5 87.9 32.8 61.2 35.3 49.2 90.2

- Table 4. Comparison of the teacher(s) and student captioning models (%). We conduct a user study to compare single teacher, all teacher, and two student models (with and without text).

[Figure 42]

Model Preference Ratio↑

Video-LLaMA [88] (pretrain) 9.4 Video-LLaMA [88] (finetune) 7.0 VideoChat [38] 7.7 VideoChat Text [38] 3.3 BLIP-2 [37] (opt2.7b) 10.7 BLIP-2 [37] (opt6.7b) 9.0 BLIP-2 [37] (flant5xl) 9.9 MiniGPT-4 [94] 3.1

Student (video input) (Ours) 18.4 Student (video+text inputs) (Ours) 21.4

Figure 5. Qualitative comparison of video captioning. We visualize a sample from the testing set of Panda-70M and show its annotation (bottommost). We also show the captions predicted from three models, including Video-LLaMA [88] with official weight and the student models with video-only or video and text inputs.

All Teachers (Ours) 23.3

### 4. Experiments

We visualize the samples of Panda-70M in Appendix E. To quantitatively evaluate the effectiveness of Panda-70M, we test its pretraining performance on three downstream applications: video captioning in Section 4.1, video and text retrieval in Section 4.2, and video generation in Section 4.3. The training details of the downstream models adhere to the official codebases unless explicitly specified.

textual information for a fair comparison. For the student model, in addition to the video, we also randomly input the metadata and subtitles into the model during training.

Downstream datasets and evaluation metrics. We test zero-shot video captioning on two benchmarks: MSRVTT [79] and MSVD [13]. MSR-VTT contains 10K videos with 20 manually annotated captions for each video; we report the results on the 2,990 testing split. MSVD consists of 1,970 videos with a total of 80K descriptions; we report the numbers on the 670 testing videos. Note that we do not use any training or validation videos from the downstream datasets. To quantitatively evaluate the quality of output captions, we follow the common protocols [41, 48, 78] and report BLEU-4 [54], ROGUE-L [40], METEOR [5], and CIDEr [69]. All the metrics are computed using the pycocoevalcap [42] package. We also compute BERTScore [91] to evaluate the contextual similarity for each token in the ground truth and the predicted captions. The results are reported in Table 3. For a fair comparison, we do not input any additional text information to the student model during the inference on the downstream datasets. In Figure 5, we also showcase a video sample from the testing set of Panda-70M and the predicted captions for the qualitative comparison.

#### 4.1. Video Captioning

Experiment setup. To evaluate the performance of video captioning, we use Video-LLaMA [88] with the vision branch only as the base model. We compare two pretraining weights: the official weight, which is jointly trained on

- 2.5M video-text pairs and 595K image-text pairs [43], and the weight trained on our Panda-2M from scratch. Panda2M is a randomly sampled subset of Panda-70M and shares the same amount of training samples as the official weight. We also train our student model with both video and text branches on complete Panda-70M for better captioning performance. For all models, we use the same backbone, using Vicuna-7B [18] as the large-language model, ViT [22] and Q-Former [37] as the video encoder, and the linear projection layer from MiniGPT-4 [94]. For Panda-2M pretraining, we only use the video and caption data without using other

- Table 5. Video and text retrieval (%). We compare the Unmasked Teacher [39] with the official checkpoint (pretrained on 2.5M videos and 3M images) and our Panda-5M pretraining. We evaluate their performance on zero-shot and finetune text-to-video (T2V) and videoto-text (V2T) retrieval. We report R@1, R@5, and R@10 accuracy on three benchmarks: MSR-VTT [79], DiDeMo [3], and MSVD [13].

MSR-VTT DiDeMo MSVD

Method Pretraining Data

R@1↑ R@5↑ R@10↑ R@1↑ R@5↑ R@10↑ R@1↑ R@5↑ R@10↑ Zero-shot T2V / V2T Retrieval

AlignPrompt [34] 2.5M vid + 3M img 24.1 / - 44.7 / - 55.4 / - 23.8 / - 47.3 / - 57.9 / - - / - - / - - / BridgeFormer [24] 2.5M vid + 3M img 26.0 / - 46.4 / - 56.4 / - 25.6 / - 50.6 / - 61.6 / - 43.6 / - 74.9 / - 84.9 / -

UMT [39] 2.5M vid + 3M img 30.2 / 33.3 51.3 / 58.1 61.6 / 66.7 33.6 / 32.1 58.1 / 57.3 65.5 / 66.7 66.3 / 44.4 85.5 / 73.3 89.3 / 82.4 UMT [39] Panda-5M (Ours) 37.2 / 36.3 58.1 / 61.0 69.5 / 69.7 34.2 / 33.4 58.4 / 57.9 66.5 / 65.8 71.2 / 37.2 88.4 / 65.1 92.7 / 75.6

Finetune T2V / V2T Retrieval CLIP4Clip [49] 400M img 44.5 / 40.6 71.4 / 69.5 81.6 / 79.5 43.4 / 42.5 70.2 / 70.6 80.6 / 80.2 46.2 / 62.0 76.1 / 87.3 84.6 / 92.6 X-CLIP [50] 400M img 49.3 / 48.9 75.8 / 76.8 84.8 / 84.5 50.4 / 66.8 80.6 / 90.4 - / - 47.8 / 47.8 79.3 / 76.8 - / InternVideo [74] 146M vid + 100M img 55.2 / 57.9 - / - - / - 57.9 / 59.1 - / - - / - 58.4 / 76.3 - / - - / -

UMT [39] 2.5M vid + 3M img 53.3 / 51.4 76.6 / 76.3 83.9 / 82.8 59.7 / 59.5 84.9 / 84.5 90.8 / 90.7 53.7 / 77.2 80.5 / 91.6 86.8 / 94.8 UMT [39] Panda-5M (Ours) 58.4 / 58.5 80.9 / 81.0 86.9 / 87.0 60.6 / 58.9 86.0 / 84.6 92.4 / 90.4 57.5 / 81.3 83.6 / 93.7 89.5 / 96.6

As in Table 3, Video-LLaMA with Panda-2M pretraining weight achieves significantly superior performance compared to the official weight. Numerically, our pretraining weight yields 17.7% and 18.5% improvement respectively on MSR-VTT and MSVD in terms of B-4. Besides, in Figure 5, we can find that the caption from the original VideoLLaMA contains irrelevant and generic information, such as date and location. In comparison, our prediction better aligns with the video content.

Can the student perform better than its teacher? In Section 3.4, we learn a student model in a knowledge distillation manner. To evaluate the performance of the student model, we conduct a user study where participants are asked to select the best caption from ten candidates for each video. Ten captions are predicted from eight teacher models and two student models (with and without text inputs). We collect the results from five participants to reduce the personal subjective bias. Each participant saw the same 200 videos, which were randomly sampled from the testing set and had not been seen during the training of the student model and UMT. We report the preference ratio of each model and the R@1 accuracy of the finetuned UMT (i.e., all teachers) in Table 4. We can observe that the student model outperforms any individual teacher model and achieves a comparable performance with all teacher models.

Can multimodal inputs leverage video captioning? Our student model supports both video and text inputs. In Ta-

- ble 4, we show that the student model with both video and text inputs outperforms the model with video input only by

- 3.0% preference ratio. Qualitatively, we show the predictions with and without text inputs in Figure 5. While the prediction with pure video input can include partial content of the video, like “cactus”, the model with both video and text inputs can more comprehensively include keywords such as “succulents” and “different species” from the video title, description, and subtitles.

#### 4.2. Video and Text Retrieval

Experiment setup. We use Unmasked Teacher [39] as the base model to evaluate the performance on video and text retrieval. The standard protocols [17, 24, 34, 39, 78] jointly use 3M images from CC3M [64] and 2.5M videos as the pretraining datasets. Thus, we randomly sample a Panda5M subset, which shares the same number of training samples as the standard pretraining dataset for a fair comparison. For both datasets, we use the same backbone composed of ViT-L/16 [21] and BERTlarge [20]. We use the official weights for the standard datasets pretraining and train the model from scratch for our Panda-5M.

Downstream datasets and evaluation metric. We test both zero-shot and finetune retrieval on three benchmarks: MSR-VTT [79], DiDeMo [3], and MSVD [13]. For MSRVTT, we follow the common protocol [34, 85] to evaluate on 1K testing split, which is not the same as the testing videos for captioning in Section 4.1. For DiDeMo [3], it contains 10K Flickr videos with a total of 40K dense captions. As in the previous standard [24, 33, 44], we evaluate paragraph-to-video retrieval by concatenating all sentence descriptions of one video into a single query. We report the results on the 1K testing set. For MSVD [13], we report the results on the 670 testing videos. We employ the standard metric and report R@1, R@5, and R@10 accuracy on both text-to-video and video-to-text retrieval in Table 5.

We can observe that pretraining with our Panda-5M outperforms the official weight in both zero-shot and finetune retrieval settings. Especially, our pretraining yields 7.0%, 0.6%, and 4.9% lifts in terms of R@1 of zero-shot textto-video retrieval on MSR-VTT [79], DiDeMo [3], and MSVD [13] respectively. Besides, pretraining UMT [39] with our Panda-5M also outperforms the existing state-ofthe-art methods [49, 50, 74] which are pretrained with much more vision-text data pairs (i.e., >100M).

- Table 6. Zero-shot text-to-video generation. We compare the zero-shot text-to-video generation of AnimateDiff [26] with the official weight (pretrained on 2.5 M videos) and our Panda-2M pretraining. We report FVD [68] on UCF101 [66] and CLIP similarity (CLIPSim) [76] on MSR-VTT [79]. We only compare with the models trained with less than 10M videos.

UCF101 MSR-VTT FVD↓ CLIPSim↑

Method (#) P-T Videos

CogVideo [30] 5M 701.6 MagicVideo [92] 10M 699.0 LVDM [28] 18K 641.8 0.2751 ModelScope [70] 10M 639.9 0.3000 VideoLDM [8] 10M 550.6 -

AnimateDiff [26] 2.5M 499.3 0.2869 AnimateDiff [26] Panda2M (Ours) 421.9 0.2880

#### 4.3. Text-to-Video Generation

Experiment setup. To evaluate the effectiveness of textto-video generation, we use AnimateDiff [26] as the base model and compare two weights: the officially released weight, which is trained on 2.5M text-video pairs, and the weight trained on our Panda-2M, a 2.5M subset of Panda70M. We follow the official codebase and use Stable Diffusion v1.5 [59] (SD) as the base text-to-image (T2I) generator. During training, we fix T2I modules and only train the motion modeling modules. For each training video, we sample 16 frames with a stride of 4, and then resize and center-crop to 256 × 256px resolution.

Downstream datasets and evaluation metrics. To evaluate the models, we follow the evaluation protocols [8, 23, 65, 72] for zero-shot evaluation on UCF101 [66] and MSR-VTT [79]. Specifically, we generate 16-frame videos in 256 × 256px resolution. For UCF101 [66], we produce a text prompt for each class [23] and generate 10,000 videos which share the same class distribution as the original dataset [8, 72]. We compute Fr´echet Video Distance (FVD) [68] on the I3D embeddings [84]. For MSRVTT [79], we generate a video sample for each of the 59,800 test prompts [23, 65] and compute CLIP similarity (CLIPSim) [76]. We report the numbers in Table 6. We also show the generated video samples in Figure 6. To visualize the results, we follow the official codebase and replace SD T2I with personalized Dreambooth weight [60], TUSUN3. Note that the test prompt and the video sample from the AnimtateDiff with the official weight (top row in Figure 6) are directly from the project page of AnimateDiff.

Panda-2M pretraining consistently shows superior performance on both metrics compared to the official weight. As highlighted, our pretraining yields 77.4 lower FVD on UCF101 and outperforms state-of-the-art models pretrained on a dataset within a 10M scale in terms of FVD. Qualita-

3https://civitai.com/models/33194/pallass-catmanul-lora

“Cut tusuncub walking in the snow, blurry, looking at viewer, depth of field, blurry background, full body, solo, cute …”

|[Figure 43]|[Figure 44]|[Figure 45]|[Figure 46]|
|---|---|---|---|

AnimateDiff

Panda-2M(Ours) Original

|[Figure 47]|[Figure 48]|[Figure 49]|[Figure 50]|
|---|---|---|---|

Pretrainingwith

Figure 6. Qualitative results of text-to-video generation. We visualize the videos generated by the AnimateDiff [26] with official weight (top) and our Panda-2M pretraining (bottom). Note that the test prompt and the video sample of the original AnimateDiff (top) are directly from the project website of AnimateDiff.

tively, our pretraining weights can generate the video with a more meaningful motion and photorealistic appearance and do not include a watermark.

### 5. Conclusion and Limitations

This paper introduces Panda-70M, a large-scale video dataset with caption annotations. The dataset includes highresolution and semantically coherent video samples. To caption 70M videos, we propose an automatic pipeline that can leverage multimodal information, such as video description, subtitles, and individual static video frames. We demonstrate that pretraining with Panda-70M can facilitate three downstream tasks: video captioning, video and text retrieval, and text-to-video generation.

Despite showing impressive results, the proposed dataset is still bound by a few limitations. First, we collect the videos from HD-VILA-100M [80], where most of the samples are vocal-intensive videos. Hence, the major categories of our dataset are news, television shows, documentary films, egocentric videos, and instructional and narrative videos. As our annotation pipeline does not require the presence of video subtitles, we list the collection of more unvocal videos as an important extension of this work.

Second, we focus on a fine-grained dataset where the video samples are semantically consistent so the caption can accurately express its semantics content without ambiguity. Nevertheless, it would limit the content diversity within a single video and also reduce average video duration, which might be hurtful to the downstream tasks, such as long video generation [9] and dense video captioning [71, 81]. Future efforts in building datasets with long videos and dense captions can benefit these downstream applications.

Risk mitigation. Prior to the release of the dataset, we used the internal automatic pipeline to filter out the video samples with harmful or violent language and texts that include drugs or hateful speech. We also use the NLTK framework to replace all people’s names with “person”.

### References

- [1] Pyscenedetect. https : / / github . com / Breakthrough/PySceneDetect. 4, 1
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. NeurIPS, 2022. 2, 3
- [3] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In ICCV, 2017. 2, 3, 7
- [4] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint, 2022. 2, 3
- [5] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In ACL workshop, 2005. 6
- [6] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. BEit: BERT pre-training of image transformers. In ICLR, 2022. 3
- [7] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3
- [8] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023. 3, 8
- [9] Tim Brooks, Janne Hellsten, Miika Aittala, Ting-Chun Wang, Timo Aila, Jaakko Lehtinen, Ming-Yu Liu, Alexei Efros, and Tero Karras. Generating long videos of dynamic scenes. NeurIPS, 2022. 8
- [10] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/ kakaobrain/coyo-dataset, 2022. 2, 3
- [11] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In CVPR,

2015. 2, 3

- [12] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12M: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR,

2021. 2

- [13] David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In ACL, 2011. 2, 3, 6, 7
- [14] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechu Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. arXiv preprint, 2023. 3
- [15] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020. 5

- [16] Tsai-Shien Chen, Wei-Chih Hung, Hung-Yu Tseng, Shao-Yi Chien, and Ming-Hsuan Yang. Incremental false negative detection for contrastive learning. In ICLR, 2022. 5
- [17] Feng Cheng, Xizi Wang, Jie Lei, David Crandall, Mohit Bansal, and Gedas Bertasius. Vindlu: A recipe for effective video-and-language pretraining. In CVPR, 2023. 3, 7
- [18] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, 2023. 6, 3
- [19] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instructionfinetuned language models. arXiv preprint, 2022. 3
- [20] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint, 2018. 7, 5
- [21] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 3, 7, 5
- [22] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19358–19369, 2023. 6
- [23] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, MingYu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. In ICCV, 2023. 8
- [24] Yuying Ge, Yixiao Ge, Xihui Liu, Dian Li, Ying Shan, Xiaohu Qie, and Ping Luo. Bridging video-text retrieval with multiple choice questions. In CVPR, 2022. 7
- [25] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In CVPR, 2023. 3, 4, 5, 1
- [26] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint, 2023. 8
- [27] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 3
- [28] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint, 2023. 8
- [29] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint, 2022. 3

- [30] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint, 2022. 8
- [31] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICML, 2021. 3
- [32] Sangho Lee, Jiwan Chung, Youngjae Yu, Gunhee Kim, Thomas Breuel, Gal Chechik, and Yale Song. Acav100m: Automatic curation of large-scale datasets for audio-visual video representation learning. In ICCV, 2021. 2
- [33] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and Jingjing Liu. Less is more: Clipbert for video-and-language learning via sparse sampling. In CVPR,

2021. 7

- [34] Dongxu Li, Junnan Li, Hongdong Li, Juan Carlos Niebles, and Steven CH Hoi. Align and prompt: Video-and-language pre-training with entity prompts. In CVPR, 2022. 7
- [35] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. NeurIPS, 2021. 5
- [36] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML,

2022. 2, 3

- [37] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint, 2023. 2, 3, 4, 5, 6
- [38] Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint, 2023. 2, 3, 4, 6
- [39] Kunchang Li, Yali Wang, Yizhuo Li, Yi Wang, Yinan He, Limin Wang, and Yu Qiao. Unmasked teacher: Towards training-efficient video foundation models. ICCV, 2023. 2, 3, 4, 5, 7
- [40] Chin-Yew Lin and Franz Josef Och. Automatic evaluation of machine translation quality using longest common subsequence and skip-bigram statistics. In ACL, 2004. 6
- [41] Kevin Lin, Linjie Li, Chung-Ching Lin, Faisal Ahmed, Zhe Gan, Zicheng Liu, Yumao Lu, and Lijuan Wang. Swinbert: End-to-end transformers with sparse attention for video captioning. In CVPR, 2022. 6
- [42] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 6
- [43] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint, 2023. 2, 3, 6
- [44] Yang Liu, Samuel Albanie, Arsha Nagrani, and Andrew Zisserman. Use what you have: Video retrieval using representations from collaborative experts. arXiv preprint, 2019. 7
- [45] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021. 3

- [46] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint, 2017. 5, 7
- [47] Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In ICLR, 2017. 7
- [48] Huaishao Luo, Lei Ji, Botian Shi, Haoyang Huang, Nan Duan, Tianrui Li, Jason Li, Taroon Bharti, and Ming Zhou. Univl: A unified video and language pre-training model for multimodal understanding and generation. arXiv preprint,

2020. 6

- [49] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing, 2022. 3, 7
- [50] Yiwei Ma, Guohai Xu, Xiaoshuai Sun, Ming Yan, Ji Zhang, and Rongrong Ji. X-clip: End-to-end multi-grained contrastive learning for video-text retrieval. In ACM MM, 2022. 7
- [51] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint, 2023. 2, 3
- [52] Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In ICCV, 2019. 2, 3, 4
- [53] OpenAI. Gpt-4 technical report. arXiv preprint, 2023. 2, 3
- [54] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In ACL, 2002. 6
- [55] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 3
- [56] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 2020. 3
- [57] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021. 2, 3
- [58] Anna Rohrbach, Marcus Rohrbach, Niket Tandon, and Bernt Schiele. A dataset for movie description. In CVPR, 2015. 2, 3
- [59] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 3, 8
- [60] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 8
- [61] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 2022. 2, 3

- [62] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NeurIPS, 2022. 2, 3, 5
- [63] Paul Hongsuck Seo, Arsha Nagrani, Anurag Arnab, and Cordelia Schmid. End-to-end generative pretraining for multimodal video captioning. In CVPR, 2022. 3
- [64] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In ACL,

2018. 7

- [65] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data. In ICLR, 2023. 3, 8
- [66] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. 8
- [67] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint, 2023. 3
- [68] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint, 2018. 8
- [69] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In CVPR, 2015. 6
- [70] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint, 2023. 8
- [71] Teng Wang, Ruimao Zhang, Zhichao Lu, Feng Zheng, Ran Cheng, and Ping Luo. End-to-end dense video captioning with parallel decoding. In ICCV, 2021. 8
- [72] Wenjing Wang, Huan Yang, Zixi Tuo, Huiguo He, Junchen Zhu, Jianlong Fu, and Jiaying Liu. Videofactory: Swap attention in spatiotemporal diffusions for text-to-video generation. arXiv preprint, 2023. 3, 8
- [73] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, highquality multilingual dataset for video-and-language research. In ICCV, 2019. 2, 3
- [74] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint,

2022. 7

- [75] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. arXiv preprint arXiv:2307.06942, 2023. 3

- [76] Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. Godiva: Generating open-domain videos from natural descriptions. arXiv preprint, 2021. 8
- [77] Jialian Wu, Jianfeng Wang, Zhengyuan Yang, Zhe Gan, Zicheng Liu, Junsong Yuan, and Lijuan Wang. Grit: A generative region-to-text transformer for object understanding. arXiv preprint, 2022. 3
- [78] Haiyang Xu, Qinghao Ye, Ming Yan, Yaya Shi, Jiabo Ye, Yuanhong Xu, Chenliang Li, Bin Bi, Qi Qian, Wei Wang, et al. mplug-2: A modularized multi-modal foundation model across text, image and video. arXiv preprint, 2023. 6, 7
- [79] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In CVPR, 2016. 2, 3, 6, 7, 8
- [80] Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In CVPR, 2022. 1, 2, 3, 4, 8
- [81] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. In CVPR,

2023. 3, 8

- [82] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. arXiv preprint,

2022. 3, 5

- [83] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation. TMLR, 2022. 2, 3
- [84] Sihyun Yu, Jihoon Tack, Sangwoo Mo, Hyunsu Kim, Junho Kim, Jung-Woo Ha, and Jinwoo Shin. Generating videos with dynamics-aware implicit generative adversarial networks. arXiv preprint arXiv:2202.10571, 2022. 8
- [85] Youngjae Yu, Jongseok Kim, and Gunhee Kim. A joint sequence fusion model for video question answering and retrieval. In ECCV, 2018. 7
- [86] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision. arXiv preprint, 2021. 3
- [87] Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu, Jae Sung Park, Jize Cao, Ali Farhadi, and Yejin Choi. Merlot: Multimodal neural script knowledge models. NeurIPS,

2021. 2, 3

- [88] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint, 2023. 2, 3, 4, 5, 6
- [89] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 4
- [90] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab,

- Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint, 2022. 5, 3
- [91] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. arXiv preprint, 2019. 6
- [92] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint,

2022. 3, 8

- [93] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In AAAI, 2018. 2, 3
- [94] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint, 2023. 2, 3, 4, 6

## Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers Supplementary Material

## Table of Contents

###### A. Details of Semantics-Aware Video Splitting Algorithm 1

- A.1. Stage1: Splitting based on Shot Boundary Detection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
- A.2. Stage2: Stitching based on Semantics Similarity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2

###### B. Details of Teacher Captioning Models: Pool, Inference, and Selection 2

- B.1. Introduction of 31 Captioning Models Pool . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- B.2. Inference of Cross-Modality Teacher Model for Video Captioning . . . . . . . . . . . . . . . . . . . . . . . . 3
- B.3. Selecting 8 Captioning Models based on a Human Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . 3

###### C. Details of Fine-Grained Video-to-Text Retrieval: Dataset, Training, and Inference 5

- C.1. Collection of Dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- C.2. Finetuning of Retrieval Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- C.3. Inference of Retrieval Model on Panda-70M . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

###### D. Details of Student Captioning Model: Architecture and Training 6

- D.1. Model Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- D.2. Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

###### E. Visualization of Panda-70M Dataset 7

- E.1. Category: Animal . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- E.2. Category: Scenery . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- E.3. Category: Food . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- E.4. Category: Sports Activity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- E.5. Category: Vehicles . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- E.6. Category: Tutorial and Narrative . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- E.7. Category: News and TV Shows . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- E.8. Category: Gaming and 3D Rendering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

### A. Details of Semantics-Aware Video Splitting Algorithm

- In Section 3.1, we propose a video splitting algorithm to cut a long video into several semantically coherent clips. The algorithm includes two stages, splitting and stitching, for which the details are described in Appendix A.1 and A.2.

#### A.1. Stage1: Splitting based on Shot Boundary Detection

We first split a long video by PySceneDetect [1]. Specifically, we use ContentDetector with cutscene threshold of 25 and min scene len of 15 frames. Next, we design a two-step post-processing algorithm to handle 1) long videos with complex transitions, such as fade-in and fade-out effects, that cannot be reliably detected by PySceneDetect and 2) unedited footage that does not contain any cut-scenes but has semantic changes within the same clip.

To handle both cases, we propose creating artificial scene cuts each 5 seconds for clips without cut-scene. That is, if a video clip is longer than 5 seconds, we cut out the first 5 seconds as a new clip and recursively apply the same procedure to the remaining part. Since we are only interested in semantically consistent video clips, we extract the ImageBind [25] features of the frames near the beginning or the end. If the features of these two frames are dramatically different we remove that clip. Specifically, given a n-frame video clip C, we extract the features f(CA) and f(CB) for the number 0.1 × n and 0.9 × n frames, denoted as CA and CB. We only keep the video clips if satisfying ∥f(CA) − f(CB)∥ ≤ 1.0. As such, we can exclude video clips with transition effects or significant semantics changes within a clip.

Ratio(%)

|[Figure 51]|
|---|

Video Duration (second)

###### Figure 7. Distribution of video duration of Panda-70M.

Table 7. Overview of 31 teacher models. 31 teacher models are composed of 6 base models with various weights and input information. Input data includes vision (V), subtitles (S), and metadata (M). Vision data is either a video or a static video frame, depending on the type of base model. Metadata includes the title and the description of a video. For example, V-S-M for MiniGPT-4 means MiniGPT-4 with the inputs of a video frame, subtitles, and metadata.

Input Information

Base Model Type Weights

# of Models V V-S V-M V-S-M

Video-LLaMA [88] Video VQA pretrain / finetune ✓ ✓ ✓ ✓ 8 VideoChat [38] Video VQA 7B ✓ ✓ ✓ ✓ 4 VideoChat Text [38] NLP-based Video VQA - ✓ ✓ ✓ ✓ 4 Video-ChatGPT [51] Video VQA - ✓ ✓ ✓ ✓ 4 BLIP-2 [37] Image Captioning opt2.7b / opt6.7b / flant5xl ✓ ✗ ✗ ✗ 3 MiniGPT-4 [94] Image VQA 7B / 13B ✓ ✓ ✓ ✓ 8

#### A.2. Stage2: Stitching based on Semantics Similarity

The first stage introduces many short consecutive clips with the same semantic content. To this end, we propose an additional procedure to merge the clips with the same semantic content. Formally, given two adjacent clips C1 and C2 in sequence, we concatenate them into a clip if ∥f(CB1 ) − f(CA2 )∥ ≤ 0.6.

Finally, we perform a post-processing to stabilize the quality and diversity of the video clips with the following steps:

- • First, we exclude the clips shorter than 2 seconds or clips that contain only slight motion (i.e., ∥f(CA)−f(CB)∥ ≤ 0.15). For the videos longer than 60 seconds, we only retain the first 60 seconds.
- • Next, we represent each clip by the average of ImageBind features extracted from stage1 (Section A.1) and only keep the video clips that are semantically different (i.e., Euclidean distance > 0.3) from the precedent clips to increase the diversity of the video samples.
- • Finally, we trim out the first and last 10% of a video clip as we notice that the beginning and the ending of a clip usually contain unstable camera movement or transition effects.

With the proposed splitting algorithm, we split 3,790,459 long videos into 70,817,169 clips with an average clip duration of 8.477 seconds. We plot the distribution of video length in Figure 7.

### B. Details of Teacher Captioning Models: Pool, Inference, and Selection

- In Section 3.2, we propose to use multiple cross-modality teacher models for captioning. Specifically, we start with a large pool including 31 captioning models. We elaborate on the composition of the model pool and how we implement them for video captioning in Appendix B.1 and B.2 respectively. As running the inference of the models to 70M videos is computationally expensive, we select only 8 models as the representative, based on a human evaluation. We will describe more details about this process in Appendix B.3.

|You are given some information about a video and will be asked to summarize the video (or the given video frame). The subtitles of the video: “(video subtitles)” Some descriptions of the video: [“(video title)”, “(video description)”] Please faithfully summarize the video (or the video frame) in one sentence.|
|---|

Figure 8. Prompt template of the VQA models.

#### B.1. Introduction of 31 Captioning Models Pool

The primary reason to utilize cross-modality teacher models is to leverage multimodal data that would benefit video captioning. As such, we consider the base models, including image/video visual-question-answering (VQA) and image captioning models. Specifically, we employ Video-LLaMA [88], VideoChat [38], VideoChat Text [38], Video-ChatGPT [51], BLIP2 [37], and MiniGPT-4 [94] as the base models. Based on these models, we collect 31 captioning models in total using different weights and input information. We list the summary of all captioning models in Table 7.

#### B.2. Inference of Cross-Modality Teacher Model for Video Captioning We list the inference details of each base model as follows:

- • Video-LLaMA [88] is a video VQA model. We only use the vision branch and do not use the audio one. The model uses Vicuna-7B [18] as the LLM to implement VQA. We use two official weights, including the pretraining weight, which is trained on 2.5M video-text pairs and LLaVA-CC3M [43], and the finetuning weight, which is further finetuned on instruction-tuning data from [38, 43, 94].
- • VideoChat [38] and Video-ChatGPT [51] are video VQA models. We use Vicuna-7B as the LLM and follow the official codebase for the rest of the configuration.
- • VideoChat Text [38] is a natural-language processing (NLP)-based video VQA model. The model would textualize the video content into video tags, dense captions, and a general caption respectively by three models [45, 56, 77]. As such, users can have a conversation with a chatbot and discuss the video based on the extracted textual content. The original codebase uses ChatGPT-4 [53] as the chatbot, which is, however, not freely released to the public. Thus, we replace it with LLaMA [67] for large-scale captioning.
- • BLIP-2 [37] is a language-image pretraining model. We only use it for image captioning and do not input texts. We use the weights pretraining with different LLMs, including OPT [90] (opt2.7b and opt6.7b) and FlanT5 [19] (flant5xl).
- • MiniGPT-4 [94] is an image VQA model. We use two variants respectively with Vicuna-7B and Vicuna-13B as LLMs.

To implement cross-modality teacher models for video captioning, we design the algorithms specifically for the models of different modalities. For an image model, given an n-frame video clip, we randomly sample a video frame in-between number 0.3 × N and 0.7 × N frames as the input. For a VQA model, in addition to the visual data, we also input a text prompt that could include additional textual information, such as video title, description, and subtitles, to assist video captioning. Specifically, we use the prompt template in Figure 8 if we would like to include the information of either metadata or subtitles or both for captioning. In contrast, we use a dummy prompt: “Please faithfully summarize the video (or image) in one sentence.” if we only input the vision data for captioning.

#### B.3. Selecting 8 Captioning Models based on a Human Evaluation

Running 31 captioning models on 70M videos requires significant computation resources. Hence, we propose to find a well-performing subset of the models by a two-step algorithm, including a human evaluation and model selection algorithm. Human evaluation. First, we conduct a user study by showing the output captions of each model to humans. Specifically, we randomly sample 1K video clips and perform the inference of 31 captioning models on each video. Next, the human annotators are asked to select “every good caption”, where a good caption is defined as: “the caption cannot contain any wrong information and needs to cover the main action OR all of the main objects presented in the video.” If none of the captions is a good caption, the annotators are asked to select the “All Bad” option. We randomly shuffle 31 captions to minimize the annotator’s bias on the order of the captions. Considering that a human is hard to focus on reading all 31 caption sentences at the same time, we split the captions into three groups. The annotator will see the same video three times with at most 11 captions once. We show the interface of this user study in Figure 9 and plot the results in Figure 10.

Algorithm of model selection. In the second step, we collect a list of 8 captioning models as the representative to reduce the computation for large-scale captioning. Intuitively, one may opt for the models exhibiting the top 8 performance. Nonethe-

|[Figure 52]|
|---|

###### Figure 9. Screenshot of the user study interface.

SelectiveRate(%)

|[Figure 53]| | | | | | | | |
|---|---|---|---|---|---|---|---|---|

Video-LLaMA

Video-LLaMA

VideoChat VideoChat Text Video-ChatGPT BLIP-2

MiniGPT-4

MiniGPT-4

All Bad

pretrain

finetune

opt2.7b/6.7b/flant5xl

7B

13B

Vision

| |
|---|

Vision-Subtitles

| |
|---|

Vision-Metadata

| |
|---|

Vision-Subtitles-Metadata

| |
|---|

Figure 10. Ratio of an individual captioning model to predict a good caption. Each bar represents an individual model and is colored by its input information. We highlight the 8 selected teacher models with gray. Note that we also report the ratio of “All Bad” at rightmost.

less, such behavior does not align with the philosophy of our captioning algorithm. Specifically, our algorithm utilizes multiple cross-modality models to cover good captioning on various types of videos and only retrieves one best caption as the annotation for each video (as described in Section 3.3). Accordingly, we propose to use the set of models that can jointly cover a good caption for most video samples. The algorithm starts by selecting the best-performing model (i.e., BLIP-2 with opt6.7b). Next, we only consider the videos that the previously selected model(s) cannot generate a good caption and then greedily find the model that performs best on those videos. We recursively collect the models under this mindset until we make the list of 8 captioning models. The 8 selected models are highlighted in Figure 10.

Additional findings. From Figure 10, we can also observe that a single captioning model can predict a good caption for at most 30.8% of the videos. In comparison, all 31 captioning can jointly predict at least one good caption for 84.7% of the videos (based on the “All Bad” ratio of 15.3%). This fact supports our motivation to use multiple cross-modality teacher models to jointly predict the captions for a video. Last but not least, according to our statistics, using 8 selected teacher captioning models can jointly predict a good caption for 76.8% of the videos which shows comparable performance with all

- 31 models while significantly reducing the computational requirements.

|[Figure 54]<br><br>13.93 8.15<br><br>5.27<br><br>6.34<br><br>10.54 10.23<br><br>9.85 6.51<br><br>UMT matching scores > 0.43 UMT matching scores < 0.43<br><br>| |
|---|
|
|---|

[Figure 55]

Figure 11. Distribution of the source teacher models of the captions in Panda-70M.

### C. Details of Fine-Grained Video-to-Text Retrieval: Dataset, Training, and Inference

- In Section 3.3, we mention that the available generic retrieval models [25, 39] cannot pick the best caption from 8 candidates predicted by our teacher models. The main reason is that all of the candidate captions are highly relevant to the video sample and require the model to discern subtle distinctions within each caption for optimal performance. To better perform our “fine-grained” retrieval task, we first annotate a subset of video samples by manually selecting the best caption as detailed in Appendix C.1. Next, we finetune Unmasked Teacher [39] (UMT) and run the inference of the model on all video samples respectively in Appendix C.2 and C.3.

#### C.1. Collection of Dataset

We randomly sample 100K video samples from our dataset and ask human annotators to select “the best caption” for each video. At the beginning of the task, the annotator will read the task description as follows:

“You are presented with a short video clip and a set of textual summaries that describe this clip. Choose the textual summary that is the most faithful and descriptive of the content of the video clip. Imagine you are talking on the phone with your friend and you need to describe the video to him.”

Note that this task is different from the user study in Appendix B.3, where a human is asked to select “every good caption”. But, we also randomly shuffle the captions and provide an “All Bad” option if all of the captions contain wrong information. We filter out 12,064 videos with the “All Bad” option selected and split the dataset into 86,131 and 1,805 videos for training and validation. We plot the selective rate of each teacher model on the validation set in Figure 3 (blue bar).

#### C.2. Finetuning of Retrieval Model

We finetune Unmasked Teacher [39] as the text retrieval model on the training set. We use the larger model configuration, consisting of ViT-L/16 [21] and BERTlarge [20], and initialize the model with the weights pretrained on 25M image-text and video-text pairs. We follow the original codebase and only use the video-text contrastive (VTC) and video-text matching (VTM) loss functions for finetuning. For VTC, we implement hard negative mining [16, 35] which guides the model focusing on distinguishing the selected caption (i.e., the positive sample) and the other 7 captions (i.e., the hard negative samples). Specifically, we set the training weights of the positive and hard negatives as 1 while the weights of other negatives (i.e., captions from other videos) as 0.01. For the training videos, we randomly sample 12 video frames and apply RandomResizedCrop transformation with scale [0.5,1.0] to get the video with the resolution of 224 × 224px. We use the AdamW [46] optimizer with a learning rate of 2e−5, β = [0.9,0.999], and a weight decay of 0.02. We set the batch size of 32 and last the training for 10 epochs. The model is funtuned on 8 Nvidia A100 GPUs (80GB).

#### C.3. Inference of Retrieval Model on Panda-70M

With the finetuned UMT, we automatically retrieve the best caption as the annotation for all 70M videos. We illustrate the distribution of the finetuned UMT’s selection in Figure 11 and the caption length in Figure 12. We also plot the word cloud of the randomly sampled 100K caption annotations in Figure 13 to highlight the rich content within the annotated captions.

Ratio(%)

|[Figure 56]|
|---|

<5 >40

Caption Length (#words)

Figure 12. Distribution of caption length of Panda-70M.

[Figure 57]

Figure 13. Word cloud of 100K caption samples in Panda-70M.

In addition to the retrieval result, UMT also predicts a matching score for the video-text pair. In practice, we find the score is highly correlated to the alignment of the contents within the video-text pair. A score higher than 0.43 usually represents a strong association between the video and the caption. Numerically, 89.6% of the samples in Panda-70M have matching scores higher than 0.43.

### D. Details of Student Captioning Model: Architecture and Training

#### D.1. Model Architecture

Figure 4 shows the architecture of the student captioning model. The model includes a vision branch and a text branch for additional subtitle and metadata inputs.

The vision branch shares the same design as Video-LLaMA [88]. Specifically, given an 8-frame video with the resolution of 224 × 224px, a visual encoder first individually encodes each video frame into multiple frame-level features with the dimension of 32 × 768. The visual encoder is composed of a frozen pretrained visual encoder, including a ViT-G/14 from EVA-CLIP [22] and a Q-former [37]. Subsequently, the temporal fusion module aggregates multiple frame-level features into a single 32 × 768 video representation. The module includes a position embedding layer to inject temporal information into video frames and a video Q-former to fuse the frame-level features. Finally, the model projects the video representation into a 32 × 4096 feature by a linear layer.

For the text branch, given a prompt with an arbitrary length, the model first tokenizes the prompt and embeds each token into a feature vector with 4096 length by a pretrained embedding layer [18]. Considering that the number of token embedding might be large for a longer prompt and the information of the prompt might not well align with the video content, we then design a text Q-Former to extract a fixed and shorter length of text embedding and at the same time, better bridge the feature

of the input video and text prompt. Specifically, the text Q-Former takes the inputs of the 32 × 4096 video representation as the queries and multiple token embedding as the key and value. The module then outputs a 32 × 4096 text representation. Finally, we combine the multimodal inputs by concatenating the text and video representations in sequence to get a 64×4096 feature and input it to the LLM to predict the video caption.

#### D.2. Training Details

The training data includes a video-caption pair and additional text information (i.e., the metadata and subtitles). For the video data, we randomly sample 8 frames and apply the same video reading algorithm as in Appendix C.2. For the text branch, we embed the extra text information into the prompt. To learn a captioning model that can take both video-only and video-text inputs, we drop part of the text inputs at random. Formally, we use the prompt template in Figure 8 and employ the metadata or/and subtitles information with the probability of 0.5 (the sampling for metadata and subtitles are independent).

We use the AdamW [46] optimizer. The learning rate is initialized as 1e−6 and linearly warmed up to 1e−4 within the first 2,500 steps and gradually decreased to 5e−5 based on cosine annealing strategy [47]. We set β = [0.9,0.99] and use a weight decay of 0.05. We train the model on the whole Panda-70M with a batch size of 48 and last the training for 300K steps. The model is trained on 48 Nvidia A100 GPUs (80GB).

- E. Visualization of Panda-70M Dataset In the following subsections, we visualize video-text pairs in Panda-70M by category.

#### E.1. Category: Animal

|[Figure 58]|[Figure 59]|[Figure 60]|[Figure 61]|[Figure 62]|[Figure 63]|
|---|---|---|---|---|---|

”A person is holding a long haired dachshund in their arms.”

|[Figure 64]|[Figure 65]|[Figure 66]|[Figure 67]|[Figure 68]|[Figure 69]|
|---|---|---|---|---|---|

”A group of dolphins are swimming in the ocean.”

|[Figure 70]|[Figure 71]|[Figure 72]|[Figure 73]|[Figure 74]|[Figure 75]|
|---|---|---|---|---|---|

”A rhino and a lion are fighting in the dirt.”

|[Figure 76]|[Figure 77]|[Figure 78]|[Figure 79]|[Figure 80]|[Figure 81]|
|---|---|---|---|---|---|

”A cat laying on a rug with a leash around its neck.”

#### E.2. Category: Scenery

|[Figure 82]|[Figure 83]|[Figure 84]|[Figure 85]|[Figure 86]|[Figure 87]|
|---|---|---|---|---|---|

”There is a beach with waves and rocks in the foreground, and a city skyline in the background.”

|[Figure 88]|[Figure 89]|[Figure 90]|[Figure 91]|[Figure 92]|[Figure 93]|
|---|---|---|---|---|---|

”An aerial view of a freeway intersection at dusk.”

|[Figure 94]|[Figure 95]|[Figure 96]|[Figure 97]|[Figure 98]|[Figure 99]|
|---|---|---|---|---|---|

”The waves are crashing on the beach and the water is foamy.”

|[Figure 100]|[Figure 101]|[Figure 102]|[Figure 103]|[Figure 104]|[Figure 105]|
|---|---|---|---|---|---|

”There is a field of reeds blowing in the wind against a cloudy sky.”

#### E.3. Category: Food

|[Figure 106]|[Figure 107]|[Figure 108]|[Figure 109]|[Figure 110]|[Figure 111]|
|---|---|---|---|---|---|

”Someone is frying dough balls in a pan with oil.”

|[Figure 112]|[Figure 113]|[Figure 114]|[Figure 115]|[Figure 116]|[Figure 117]|
|---|---|---|---|---|---|

”A person is using a chef's knife to chop fresh parsley on a wooden cutting board.”

|[Figure 118]|[Figure 119]|[Figure 120]|[Figure 121]|[Figure 122]|[Figure 123]|
|---|---|---|---|---|---|

”A person is making a pie crust on a table.”

|[Figure 124]|[Figure 125]|[Figure 126]|[Figure 127]|[Figure 128]|[Figure 129]|
|---|---|---|---|---|---|

”There are sausages cooking on a grill, and a person is using tongs to turn them over.”

#### E.4. Category: Sports Activity

|[Figure 130]|[Figure 131]|[Figure 132]|[Figure 133]|[Figure 134]|[Figure 135]|
|---|---|---|---|---|---|

”A female gymnast is practicing her skills on a climbing wall.”

|[Figure 136]|[Figure 137]|[Figure 138]|[Figure 139]|[Figure 140]|[Figure 141]|
|---|---|---|---|---|---|

”A group of young girls are playing soccer on a green grass field with a goal in the background.”

|[Figure 142]|[Figure 143]|[Figure 144]|[Figure 145]|[Figure 146]|[Figure 147]|
|---|---|---|---|---|---|

”A man paddles a canoe on a wave in the ocean.”

|[Figure 148]|[Figure 149]|[Figure 150]|[Figure 151]|[Figure 152]|[Figure 153]|
|---|---|---|---|---|---|

”A skateboarder performs a trick in a skate park.”

#### E.5. Category: Vehicles

|[Figure 154]|[Figure 155]|[Figure 156]|[Figure 157]|[Figure 158]|[Figure 159]|
|---|---|---|---|---|---|

”An orange dodge challenger parked in front of a house.”

|[Figure 160]|[Figure 161]|[Figure 162]|[Figure 163]|[Figure 164]|[Figure 165]|
|---|---|---|---|---|---|

”It is a rally car driving on a dirt road in the countryside, with people watching from the side of the road.”

|[Figure 166]|[Figure 167]|[Figure 168]|[Figure 169]|[Figure 170]|[Figure 171]|
|---|---|---|---|---|---|

”A remote control monster truck is driving on rough terrain.”

|[Figure 172]|[Figure 173]|[Figure 174]|[Figure 175]|[Figure 176]|[Figure 177]|
|---|---|---|---|---|---|

”A blue off-road truck is driving on a sand dune and jumping into the air.”

#### E.6. Category: Tutorial and Narrative

|[Figure 178]|[Figure 179]|[Figure 180]|[Figure 181]|[Figure 182]|[Figure 183]|
|---|---|---|---|---|---|

”A person in blue gloves is connecting an electrical supply to an injector.”

|[Figure 184]|[Figure 185]|[Figure 186]|[Figure 187]|[Figure 188]|[Figure 189]|
|---|---|---|---|---|---|

”A person is welding a piece of metal using a welding torch, and the metal is glowing red hot.”

|[Figure 190]|[Figure 191]|[Figure 192]|[Figure 193]|[Figure 194]|[Figure 195]|
|---|---|---|---|---|---|

”A person is using an electric drill to make a hole in a piece of cardboard.”

|[Figure 196]|[Figure 197]|[Figure 198]|[Figure 199]|[Figure 200]|[Figure 201]|
|---|---|---|---|---|---|

”A person is making a green clay model of a monster using different tools.”

#### E.7. Category: News and TV Shows

|[Figure 202]|[Figure 203]|[Figure 204]|[Figure 205]|[Figure 206]|[Figure 207]|
|---|---|---|---|---|---|

”The columns of the temple of mars ultor in rome, italy are surrounded by trees and buildings.”

|[Figure 208]|[Figure 209]|[Figure 210]|[Figure 211]|[Figure 212]|[Figure 213]|
|---|---|---|---|---|---|

”A large pile of lava blocking a road.”

|[Figure 214]|[Figure 215]|[Figure 216]|[Figure 217]|[Figure 218]|[Figure 219]|
|---|---|---|---|---|---|

”Two men hugging each other in front of a trophy.”

|[Figure 220]|[Figure 221]|[Figure 222]|[Figure 223]|[Figure 224]|[Figure 225]|
|---|---|---|---|---|---|

”A rocket launches into space on the launch pad.”

#### E.8. Category: Gaming and 3D Rendering

|[Figure 226]|[Figure 227]|[Figure 228]|[Figure 229]|[Figure 230]|[Figure 231]|
|---|---|---|---|---|---|

”A man in a spartan armor kneeling down.”

|[Figure 232]|[Figure 233]|[Figure 234]|[Figure 235]|[Figure 236]|[Figure 237]|
|---|---|---|---|---|---|

”A screenshot of a minecraft game showing a snowy landscape.”

|[Figure 238]|[Figure 239]|[Figure 240]|[Figure 241]|[Figure 242]|[Figure 243]|
|---|---|---|---|---|---|

”A 3d rendering of a zoo with animals and a train.”

|[Figure 244]|[Figure 245]|[Figure 246]|[Figure 247]|[Figure 248]|[Figure 249]|
|---|---|---|---|---|---|

”The luxury yacht is sailing on calm waters with a beautiful sunset in the background.”

