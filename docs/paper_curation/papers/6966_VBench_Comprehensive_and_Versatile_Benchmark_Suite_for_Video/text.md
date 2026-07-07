## VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models

Ziqi Huang∗, Fan Zhang∗, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen,

Ying-Cong Chen, Limin Wang, Dahua Lin , Yu Qiao , Ziwei Liu

#### arXiv:2411.13503v1[cs.CV]20Nov2024

Abstract—Video generation has witnessed significant advancements, yet evaluating these models remains a challenge. A comprehensive evaluation benchmark for video generation is indispensable for two reasons: 1) Existing metrics do not fully align with human perceptions; 2) An ideal evaluation system should provide insights to inform future developments of video generation. To this end, we present VBench, a comprehensive benchmark suite that dissects “video generation quality” into specific, hierarchical, and disentangled dimensions, each with tailored prompts and evaluation methods. VBench has several appealing properties: 1) Comprehensive Dimensions: VBench comprises 16 dimensions in video generation (e.g., subject identity inconsistency, motion smoothness, temporal flickering, and spatial relationship, etc). The evaluation metrics with fine-grained levels reveal individual models’ strengths and weaknesses. 2) Human Alignment: We also provide a dataset of human preference annotations to validate our benchmarks’ alignment with human perception, for each evaluation dimension respectively. 3) Valuable Insights: We look into current models’ ability across various evaluation dimensions, and various content types. We also investigate the gaps between video and image generation models. 4) Versatile Benchmarking: VBench++ is designed to evaluate a wide range of video generation tasks, including text-to-video and image-to-video. We introduce a high-quality Image Suite with an adaptive aspect ratio to enable fair evaluations across different image-to-video generation settings. Beyond assessing technical quality, VBench++ evaluates the trustworthiness of video generative models, providing a more holistic view of model performance. 5) Full Open-Sourcing: We fully open-source VBench++ at https://github.com/Vchitect/VBench, including all prompts, the Image Suite, evaluation methods, generated videos, and human preference annotations. We also continually add new video generation models to the VBench++’s leaderboard at https://huggingface.co/spaces/Vchitect/VBench Leaderboard to drive forward the field of video generation.

Index Terms—Video Generative Models, Evaluation Benchmark

✦

1 INTRODUCTION

# I

MAGE generation models have made rapid progress in the past few years, such as Variational Autoencoders

(VAEs) [1], Generative Adversarial Networks (GANs) [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], vector quantized (VQ) based approaches [13], [14], [15], and diffusion models [16], [17], [18]. This fuels recent explorations in video generation [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], which goes beyond static imagery and models the dynamics and kinematics of real-world scenes. With the growth of video generation models, there arises a critical need for effective evaluation methods. The evaluation should be able to accurately reflect human perception of generated videos, providing reliable measures of a model’s performance. Additionally, it should reflect each model’s specific strengths and weaknesses, offering insights that inform the data, training, and architectural choices of future video generation models.

- • ∗equal contributions. corresponding authors.
- • Email: Ziqi Huang ziqi002@ntu.edu.sg and Fan Zhang zhangfan2@pjlab.org.cn
- • Z. Huang, N. Chanpaisit, C. Si, Y. Jiang and Z. Liu are with S-Lab, Nanyang Technological University. F. Zhang, X. Xu, Y. He, J. Yu, Z. Dong, Q. Ma, Y. Wang, X. Chen, L. Wang, D. Lin and Y. Qiao are with Shanghai Artificial Intelligence Laboratory. X. Xu and Y. Chen are also with Hong Kong University of Science and Technology (Guangzhou). L. Wang is also with Nanjing University. D. Lin is also with The Chinese University of Hong Kong.

However, existing metrics for video generation such as Inception Score (IS) [39], Fr´echet inception distance (FID) [40], Fr´echet Video Distance (FVD) [41], [42], and CLIPSIM [43] are inconsistent with human judgement [44], [45]. Meanwhile, the Video Quality Assessment (VQA) methods [46], [47], [48], [49], [50], [51], [52], [53], [54] are primarily designed for real videos, thereby neglecting the unique challenges posed by generative models, such as artifacts in synthesized videos. Hence, there is a pressing need for an evaluation framework that aligns closely with human perception, and specifically designed for the characteristics of video generation models.

To this end, we introduce VBench, a comprehensive benchmark suite for evaluating video generation model performance. VBench has three appealing properties: 1) comprehensive evaluation dimensions, 2) human alignment, and 3) valuable insights.

First, our framework includes an evaluation dimension suite that employs a hierarchical and disentangled approach to the decomposition of “video generation quality”. This suite systematically breaks down the evaluation into two primary dimensions at a coarse level: Video Quality and Video-Condition Consistency. Each of these dimensions is further subdivided into more granular criteria. This hierarchical separation ensures that each dimension isolates and evaluates a single aspect of video quality, without interference from other variables, as illustrated in Figure 1. Recognizing video generation’s unique challenges, we have

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

### VBENCH VBENCH++

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

###### Image-to-Video (I2V)

Evaluation Dimension Suite

Evaluation Method Suite

Prompt Suite

Image Suite

###### Evaluation Dimension Suite

- • comprehensive, hierarchical, fine-grained, objective
- • revealing individual models' strengths and weaknesses

- • Adaptive aspect ratio
- • Diverse and fair content
- • High resolution (mainly 4K+)
- • Paired text prompts

……

separate prompts for:

for each dimension:

Subject Consistency Background Consistency

VBench’s Video Quality dimensions

Video Quality

- • each evaluation dimension
- • each content category

• designated model and pipeline

Temporal Quality

Image-to-Video Generation Quality

……

Temporal Flickering Motion Smoothness

[Figure 12]

Video-Image Subject Consistency

[Figure 13]

Video Quality

Video-Condition Consistency

Dynamic Degree

Video-Image Background Consistency

Alignment Verification

Aesthetic Quality

Video-Text Camera Motion

Frame-Wise Quality

Generated Videos

Imaging Quality

Object Class

[Figure 14]

……

[Figure 15]

Video Generation Quality

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Multiple Objects

[Figure 20]

###### Trustworthiness

###### More Models Alignment

Semantics

Human Action

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Color Spatial Relationship Scene

I2V & Trustworthiness: human alignment is thoroughly ablated and validated.

###### non-technical aspects of video generation models

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

VideoCrafter-2.0

Video-Text Consistency

VideoCrafter-1.0

Human Preference Annotation

……

SVD-XT

SEINE

Video-Condition Consistency

Culture Fairness

ConsistI2V

Overall Consistency

[Figure 31]

Pika Show-1

video generation models

sampled videos

Open-Sora

Gender Bias

Temporal Style

Alignment GeneratedVideos Verification

Trustworthiness

Gen-2

Style

I2VGen-XL

……

Skin Tone Bias

[Figure 32]

DynamiCrafter Animate-Anything

Appearance Style

Safety

- Fig. 1: Overview of VBench++. We propose VBench++, a comprehensive and versatile benchmark suite for video generative models. We design a comprehensive and hierarchical Evaluation Dimension Suite to decompose “video generation quality” into multiple well-defined dimensions to facilitate fine-grained and objective evaluation. For each dimension and each content category, we carefully design a Prompt Suite as test cases, and sample Generated Videos from a set of video generation models. For each evaluation dimension, we specifically design an Evaluation Method Suite, which uses a carefully crafted method or designated pipeline for automatic objective evaluation. We also conduct Human Preference Annotation for the generated videos for each dimension and show that VBench++ evaluation results are well aligned with human perceptions. VBench++ can provide valuable insights from multiple perspectives. VBench++ supports a wide range of video generation tasks, including text-to-video and image-to-video, with an adaptive Image Suite for fair evaluation across different settings. It evaluates not only technical quality but also the trustworthiness of generative models, offering a comprehensive view of model performance. We continually incorporate more video generative models into VBench++ to inform the community about the evolving landscape of video generation.

tailored evaluation dimensions to its specific characteristics. For example, in terms of Video Quality, maintaining consistent subject identity (e.g., a teddy bear) in generated videos is crucial, and is a problem rarely encountered in real-world videos. Additionally, Video-Condition Consistency is vital for conditional video generation tasks, requiring its dedicated evaluation criteria. For each evaluation dimension, we carefully prepared around 100 text prompts as test cases for text-to-video (T2V) generation, and devised specialized evaluation methods tailored to each dimension. In addition to multi-dimensional evaluations, we also assess T2V models across diverse content categories. We organized prompt suites for eight distinct types, such as animal, architecture, human, and scenery, allowing for a separate evaluation within each category. This exploration reveals variable competencies in T2V generation across different content types, highlighting areas of proficiency and those requiring further enhancement.

system enables detailed feedback on the strengths and weaknesses of video generation models across various ability aspects. This approach not only ensures a comprehensive evaluation of existing models but also provides valuable insights into the training of advanced video generation models, guiding architectural and data choices for improved video generation outcomes. Additionally, VBench can be readily applied to evaluate image generation models, and thus we investigate the disparities between video and image generation models. In Section 5, we discuss in detail on various observations and insights drawn from VBench evaluations.

More recently, alongside T2V generation, image-to-video (I2V) generation has gained increasing popularity [35], [55], [56], [57], [58], [59], [60], [61], [62], [63], [64], [65], [66], [67], [68], [69], [70]. In I2V, a video is generated based on an input image and, optionally, a text prompt. I2V enables the animation of still images by accepting visual conditions, producing aesthetically appealing results that leverage the high quality of the input image. However, evaluating I2V models can be challenging, as the choice of input images and the configured video resolution can significantly impact the generation results. To address these challenges, we introduce a high-quality and fair Image Suite that supports adaptive resolution and aspect ratio, designed to highlight each I2V model’s strengths across different settings. We also develop evaluation methods for assessing video-image consistency and video-text consistency in the context of I2V. These new evaluation dimensions are compatible with VBench’s existing dimensions, providing a comprehensive evaluation for I2V models. The newly proposed evaluation methods have been carefully aligned with human perception through extensive human annotation and experiments.

Second, we systematically demonstrate that our evaluation method suite is closely aligned with human perception in every fine-grained evaluation dimension. We collected human preference annotations for each dimension. Specifically, we use various T2V models to sample videos from our prompt suites. Then given two videos sampled from the same prompt, we ask human annotators to indicate preferences according to each VBench dimension respectively. We show that VBench evaluations highly correlate with human preferences. Additionally, the human preference annotations can be utilized for multiple purposes, such as fine-tuning generation or evaluation models to enhance alignment with human perceptions.

Third, VBench’s multi-dimensional and multicategorical approach can provide valuable insights to the video generation community. Our multi-dimensional

[Figure 33]

JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 3

I2V Subject

I2V Background

Imaging Quality

Subject Consistency

Background Consistency

Safety

Temporal Flickering

Skin Bias

Camera Motion

Aesthetic Quality

Motion Smoothness

Gender Bias

Dynamic Degree

Culture Fairness

Aesthetic Quality

Overall Consistency

Subject Consistency

Dynamic Degree

Imaging Quality

Temporal Style

Background Consistency

Motion Smoothness

Object Class

Appearance Style

Multiple Objects

Scene

Temporal Flickering

Human Action

Spatial Relationship

Color

|DynamiCrafter-1024 SEINE-512x320 I2VGen-XL Animate-Anything ConsistI2V VideoCrafter-I2V<br><br>|
|---|

| |
|---|

LaVie

ModelScope CogVideo Show-1 VideoCrafter-0.9 VideoCrafter-2.0

(a) Text-to-Video Generative Models. We visualize the evaluation results of four text-to-video generation models in 16 VBench dimensions. For comprehensive numerical results, please refer to Table 2.

(b) Image-to-Video Generative Models. We visualize the evaluation results of six image-to-video generation models. See Table 3 for comprehensive numerical results.

(c) Trustworthiness of Video Generative Models. We visualize the trustworthiness of video generative models, along with other dimensions. For comprehensive numerical results, please refer to Table 4a.

- Fig. 2: VBench++ Evaluation Results. We visualize the evaluation results of text-to-video and image-to-video generative models using VBench++. We normalize the results per dimension for clearer comparisons.

Beyond evaluating the technical capabilities of various video generative models, we believe it is crucial to consider the trustworthiness of these models—specifically, how well they generate content that is fair across different cultures and demographics, and how effectively they avoid producing harmful or offensive content. These considerations are especially important when adopting models for downstream applications, such as social media broadcasting and the education sector. To address this, we have integrated four trustworthiness dimensions into the VBench++ framework, providing benchmarking and evaluation methods that are carefully designed and aligned with human perception.

We are open-sourcing VBench++, including its evaluation dimension suite, evaluation method suite, prompt suite, generated videos, and the dataset of human preference annotations. We also encourage more video generation models to participate in the VBench++ challenge.

In contrast to the earlier VBench presented at CVPR 2024 [71], the enhanced VBench++ is a more versatile framework. 1) We now support the evaluation of both text-tovideo and image-to-video generation. The newly introduced high-quality Image Suite offers a robust evaluation benchmark, featuring adaptive aspect ratios, and diverse and fair content. 2) Additionally, beyond assessing the technical quality of the generated videos, we also evaluate the trustworthiness of each generative model, offering a comprehensive perspective on model characteristics. 3) Both image-to-video and trustworthiness evaluation frameworks have been carefully aligned with human perception through extensive annotation and experiments. 4) Furthermore, we continually expand the VBench++ leaderboard by adding 32 new models to the 4 models initially presented at CVPR 2024. To better support evaluating long video generative models, we also open-sourced VBench-Long. We maintain open access to evaluation videos and data, ensuring that the community benefits from our ongoing efforts.

2 RELATED WORKS

- 2.1 Video Generative Models

Recently, diffusion models [16], [17], [18], [72] have achieved significant progress in image synthesis [73], [74], [75], [76], [77], [78], [79], and enabled a line of works towards video generation [20], [21], [23], [24], [25], [26], [27], [56], [62], [80], [81], [82], [83], [84], [85], [86], [87], [88], [89]. Many recent diffusion-based works [20], [24], [26], [27] are text-to-video (T2V) models. Other guidance modalities are also available, including image-to-video [57], [90], [91], [92], videoto-video [93], [94], [95], [96], [97], and a variety of control maps [87], [98], [99], [100], [101], [102], [103] such as pose, depth, and sketch. The boom of video generation models requires a comprehensive evaluation system to inform their current capabilities and guide future developments, and VBench takes the initiative in providing a comprehensive benchmark suite for fine-grained and human-aligned evaluation.

- 2.2 Evaluation of Visual Generative Models

Existing video generation models typically use metrics like Inception Score (IS) [39], Fr´echet inception distance (FID) [40], Fr´echet Video Distance (FVD) [41], and CLIPSIM [43] for evaluation. The UCF-101 [104] dataset’s class labels often serve as text prompts for IS, FID, and FVD, whereas MSR-VTT [105]’s human-labeled video captions are used for CLIPSIM. Despite covering various real-world scenarios, these prompts lack diversity and specificity, limiting accurate and fine-grained evaluation of video generation. For text-to-image (T2I) models, several benchmarks [75], [106], [107], [108], [109], [110], [111] are proposed to assess various capabilities like compositionality [106] and editing ability [107], [109]. However, video generative models still lack comprehensive evaluation benchmarks for detailed and human-aligned feedback. Our work differs from concurrent research [112], [113] in three key ways: 1) We have created

16 distinct evaluation dimensions, each with specialized prompts for precise assessment; 2) We have empirically validated that every dimension aligns closely with human perception; 3) Our multi-dimensional and multi-categorical evaluation offers valuable and comprehensive insights into video generation.

Evaluation of Image-to-Video Generation. Similar to T2V models, current image-to-video (I2V) models [56], [57], [60], [61], [63], [65] typically assess performance on MSRVTT [105] and UCF-101 [104] using metrics like FVD [41], [42], IS [39], and CLIPSIM [43]. Some models [35], [55], [59], [67] rely on a small set of generated images for evaluation, and heavily depending on manual assessment [35], [55], [56], [57], [58], [59], [60], [61]. AIGCBench and AnimateBench [114], [115] mainly use CLIP-based scores for I2V evaluation. TC-Bench [116] only evaluates specific areas of I2V. These existing evaluation approaches are not comprehensively verified against human perception and fail to account for varying default resolutions of different models. Our approach evaluates I2V models using an adaptiveresolution strategy, ensuring each model is compared at its optimal settings. Additionally, we provide comprehensive evaluation dimensions, each aligned with human perception through extensive experiments and human annotation.

Evaluation of Visual Generation Trustworthiness. Visual generative models should ensure their trustworthiness through cultural fairness, human bias reduction, and content safety [117], [118], [119], [120], [121], [122], [123], [124], [125], [126], [127], [128], [129], [130], [131]. Prior studies have addressed biases in culture representation [118], [132] and human attributes like gender and race [110], [133], [134], [135], [136], [137], and implemented safety measures [76], [120], [138], [139] like classifiers and checkers [76], [140], [141] to filter out harmful content. We take the initiative to evaluate trustworthiness in video generation, and introduce human-aligned benchmarks and evaluation pipelines.

##### 3 VBENCH++ SUITE

In this section, we introduce the main components of VBench++. In Section 3.1, we present our rationale for designing the 16 evaluation dimensions, as well as each dimension’s definition and evaluation method. We then elaborate on the prompt suites we use in Section 3.2.1, and the newly proposed Image Suite in Section 3.3. To validate VBench++’s alignment with human perception, we conduct human preference annotation for each dimension (see Section 3.4). The experiments and the insights drawn from VBench++ will be detailed in Section 4 and Section 5.

###### 3.1 Evaluation Dimension Suite

We first introduce our evaluation dimensions and their corresponding evaluation methods.

Existing evaluation metrics like FVD [41] often conclude video generation model performance to a single number. This oversimplifies the evaluation and has several risks. First, a single number can obscure an individual model’s strengths and weaknesses, and it fails to provide insights into specific areas where a model excels or underperforms.

This makes it challenging to derive insights for future architectural and training designs based on single-valued metrics. Second, the notion of “high-quality video generation” is complex and multifaceted, with individuals prioritizing different video attributes based on the intended application. For instance, some may prioritize the absence of temporal flickering, while others may consider fidelity to the text prompt as the most significant, with less emphasis on flickering. Therefore, in contrast with performing singlevalued evaluations of video generation quality, we propose a disaggregated approach by decomposing the brand notion of “video generation performance” into multiple discrete dimensions for fine-grained evaluation.

Specifically, we break “video generation quality” down into 16 disentangled dimensions in a top-down manner, with each evaluation dimension assessing one aspect of video generation quality. On the top level, we evaluate T2V performance from two broad perspectives: 1) Video Quality — “Without considering alignment with the text prompt, does the video alone look good?”, which focuses on the perceptual quality of the synthesized video, and does not consider the input condition (e.g., text prompt), and 2) Video-Condition Consistency — “Is the video consistent with what the user wants to generate?”, which focuses on whether the synthesized video is consistent with the guiding condition that the user provides (e.g., the text prompt for T2V generation). Under both “Video Quality” and “Video-Condition Consistency”, we further break the coarse-grained dimensions into more finegrained dimensions, as shown in Figure 1.

3.1.1 Video Quality

We split “Video Quality” into two disentangled aspects, “Temporal Quality” and “Frame-Wise Quality”, where the former only considers the cross-frame consistency and dynamics, and the latter only considers the quality of each individual frame without taking temporal quality into concern. For “Temporal Quality”, we further devise five evaluation dimensions, where each focusing on a different aspect of temporal quality.

Temporal Quality - Subject Consistency. For a subject (e.g., a person, a car, or a cat) in the video, we assess whether its appearance remains consistent throughout the whole video. To this end, we calculate the DINO [142] feature similarity across frames.

Temporal Quality - Background Consistency. We evaluate the temporal consistency of the background scenes by calculating CLIP [43] feature similarity across frames.

Temporal Quality - Temporal Flickering. Generated videos can exhibit imperfect temporal consistency at local and highfrequency details. We take static frames and compute the mean absolute difference across frames.

Temporal Quality - Motion Smoothness. Subject Consistency and Background Consistency focus on temporal consistency of the “look” instead of the smoothness of “movement and motion”. We believe it is important to evaluate whether the motion in the generated video is smooth, and follows the physical law of the real world. We utilize the motion priors in the video frame interpolation model [143] to evaluate the smoothness of generated motions.

Temporal Quality - Dynamic Degree. Since a completely static video can score well in the aforementioned temporal

quality dimensions, it is important to also evaluate the degree of dynamics (i.e., whether it contains large motions) generated by each model. We use RAFT [144] to estimate the degree of dynamics in synthesized videos.

Frame-Wise Quality - Aesthetic Quality. We evaluate the artistic and beauty value perceived by humans towards each video frame using the LAION aesthetic predictor [145]. It can reflect aesthetic aspects such as the layout, the richness and harmony of colors, the photo-realism, naturalness, and artistic quality of the video frames.

Frame-Wise Quality - Imaging Quality. Imaging quality refers to the distortion (e.g., over-exposure, noise, blur) presented in the generated frames, and we evaluate it using the MUSIQ [146] image quality predictor trained on the SPAQ [147] dataset.

- 3.1.2 Text-to-Video: Video-Condition Consistency

We mainly dissect “Video-Condition Consistency” into “Semantics” (i.e., the type of the entities and their attributes) and “Style” (i.e., whether the generated video is consistent with user-requested style), with each decomposed into more fine-grained dimensions.

Semantics - Object Class. We use GRiT [148] to detect the success rate of generating the specific class of objects depicted in the text prompt.

Semantics - Multiple Objects. Other than generating a single object of a particular class, the ability to compose multiple objects from different classes in the same frame is also an essential ability in video generation. We detect the success rate of generating all the objects specified in the text prompt within each video frame.

Semantics - Human Action. Human action is an important aspect in human-centric video generation. We apply UMT [149] to evaluate whether human subjects in generated videos can accurately execute the specific actions mentioned in the text prompts.

Semantics - Color. To evaluate whether synthesized object colors align with the text prompt, we use GRiT [148] for color captioning and comparison with expected colors.

Semantics - Spatial Relationship. Other than classes and attributes of synthesized objects, we also evaluate whether their spatial relationship follows what is specified by the text prompt. We focus on four primary types of spatial relationships, and perform rule-based evaluation similar to [106].

Semantics - Scene. We need to evaluate whether the synthesized video is consistent with the intended scene described by the text prompt. For example, when prompted “ocean”, the generated video should be “ocean” instead of “river”. We use Tag2Text [150] to caption the generated scenes, and then check its correspondence with scene descriptions in the text prompt.

Style - Appearance Style. Apart from semantics consistency with the text prompt, another important pillar in video-condition consistency is style. There are many styles that alter the look, color, and texture of synthesized video frames, such as “oil painting style”, “black and white style”, “watercolor painting style”, “cyberpunk style”, “black and white” etc. We calculate the CLIP [43] feature similarity between synthesized frames and these style descriptions.

Style - Temporal Style. Apart from appearance styles, videos also have temporal styles like various motion speed and focus shifts. We use ViCLIP [151] to calculate the video feature and the temporal style description feature similarity to reflect temporal style consistency.

Overall Consistency. We further use overall video-text consistency computed by ViCLIP [151] on general text prompts as an aiding metric to reflect both semantics and style consistency.

- 3.1.3 Image-to-Video Dimensions In addition to T2V models, we also evaluate I2V models based on two aspects: consistency with the input conditions and video quality. For video quality, we apply the same criteria used for text-to-video models, as described in Sec. 3.1.1. For the consistency dimensions, we devise three specific dimensions, detailed as follows. I2V Subject: Video-Image Subject Consistency. We evaluate the consistency between the subject in the input image and the corresponding subject in the generated video by calculating feature similarities. Specifically, we employ DINOv1 [142] to extract features from both the input image and the individual video frames. Different image-to-video models handle the input image in various ways: some models, such as DynamiCrafter [56], may position the input image randomly within the generated video, while others use it as the first frame. Consequently, the input image’s position within the video varies. Furthermore, variations caused by camera movement and subject motion introduce additional discrepancies between the input image and the video frames over time. To address these challenges, we calculate both the similarity between the input image and each video frame, as well as the similarity between consecutive video frames. The final score is obtained by taking a weighted average of these similarity measurements. I2V Background: Video-Image Background Consistency. We evaluate the consistency between the input image and the generated video frames, particularly in cases where the image focuses on the scene rather than a distinct subject. For feature extraction, we use DreamSim [152], which is highly sensitive to changes in the background. The final consistency score is calculated using the same approach as that used to assess subject consistency between images and videos. Camera Motion: Video-Text Camera Motion Consistency. In I2V models, text prompts describing camera motion are used to guide camera movements in the generated video. We focus on seven core types of camera movements: ”pan left”, ”pan right”, ”tilt up”, ”tilt down”, ”zoom in”, ”zoom out”, and ”static”. To assess the consistency of the generated camera motion with the input prompt, we use CoTracker [153], which tracks points along the four edges of the video and predicts the camera motion type based on carefully designed and validated heuristics.
- 3.1.4 Trustworthiness In addition to assessing technical quality, we prioritize evaluating the ”trustworthiness” of video generative models. Within “trustworthiness”, we consider “Culture Fairness” (i.e., can the model accurately generate scenes across diverse cultural groups?), ”Human Bias” (i.e., whether the generated

[Figure 34]

[Figure 35]

【Prompt】The bund Shanghai, Van Gogh style｜【Question】Is the style of the video in the Van Gogh style?

search tags

| | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | |

[Figure 36]

[Figure 37]

###### Compare

|A is better<br><br>B is better Same quality<br><br><br>|
|---|

playback speed

shortcuts

mark as invalid

clear annotations

Fig. 4: Interface for Human Preference Annotation. Top: prompt and question. Right: choices that annotators can make. Bottom left: control for stop and playback.

Fig. 3: Prompt Suite Statistics. The two graphs provide an overview of our prompt suites. Left: the word cloud to visualize word distribution of our prompt suites. Right: the number of prompts across different evaluation dimensions and different content categories.

videos exhibit bias across different human groups?) and “Safety” (i.e., whether the generated videos contain unsafe contents like nudity or violence?).

Culture Fairness. We evaluate models’ ability to generate scenes across various cultures. Specifically, We insert cultural keywords into text prompts and assess the fairness of generating the same scene across different cultural contexts. To evaluate how well the generated videos align with the per-scene cultural prompts, we use ViCLIP [151].

Human Bias - Gender Bias. Given specific occupations or descriptions of individuals, we evaluate the extent to which the model exhibits gender bias when generating human faces. For each frame, we use RetinaFace [154] for face detection, followed by computing the BLIP2 [155] similarity between the cropped facial image and the text prompts “male” or “female” to determine the gender attribute. If multiple faces are detected within a single frame, the frame is marked invalid to ensure accuracy, particularly in cases where unintended individuals may appear in the background. For example, in the prompt "The portrait of a nurse", a hospital scene might unintentionally include patients or doctors, complicating the identification of the intended subject for generation. The final gender classification for a video is derived by aggregating results across multiple frames. To measure the model’s gender bias, we sample each prompt 10 times and compute the L-1 distance between the gender classification results and the uniform gender distribution. For instance, if the male-tofemale ratio is 7:3, the bias score would be calculated as 1 − ∥(7/10,3/10) − (1/2,1/2)∥1 = 0.6.

Human Bias - Skin Tone Bias. We evaluate whether the model demonstrates a preference for specific skin tones using prompts that are skin-tone-agnostic. We classify skin tones based on the Fitzpatrick Scale [156], which includes six categories: "Pale white", "Fair", "Olive", "Moderate brown", "Dark brown" and "Black". We use RetinaFace [154] for facial region detection (similar to the ”Gender Bias” dimension), and CLIP [43] for skin tone classification. To quantify bias, we calculate the L2 distance between the detected distribution of skin tones and the uniform distribution. For more reliable classification, we merge adjacent tones from the six original categories into three broader classes. For instance, if the detected skin tone distribution is 3:5:2, the bias score would be calculated as 1 − ∥(3/10,5/10,2/10) − (1/3,1/3,1/3)∥2 = 0.784.

Safety. Seemingly harmless text prompts may unintentionally suggest unsafe visual content, potentially bypassing prompt filters [138], [157], which can result in models generating unexpected inappropriate content. To evaluate model safety, we use three safety detection models: NudeNet [140], the Stable Diffusion Safety Checker [76], and the Q16 classifier [141], to comprehensively detect a wide range of unsafe content. NudeNet specializes in detecting nudity by focusing on regions such as the "Genitalia", "Breasts", "Buttocks", and "Anus". The Stable Diffusion Safety Checker assesses Not-Safe-For-Work (NSFW) content and offers a broader safety assessment [158] than NudeNet. Additionally, the Q16 classifier is employed to detect other harmful content, such as self-harm and violence. A frame is deemed unsafe if flagged by any of these models. If more than 50% of the frames in a video are marked as unsafe, the entire video is classified as unsafe. The thresholds are carefully determined through experiments with human preference feedbacks.

3.2 Prompt Suite 3.2.1 Prompt Suite for Text-to-Video

The sampling procedure of current diffusion-based video generation models [24], [26], [27] is time-consuming (e.g., 90 seconds per video for LaVie [26], and more than 2 minutes per video for CogVideo [19]). Therefore, we need to control the amount of test cases for efficient evaluation. Meanwhile, we need to maintain the diversity and comprehensiveness of our prompt suite, so we design compact yet representative prompts in terms of both the evaluation dimensions and the content categories. We visualize our prompt suite distributions in Figure 3.

Prompt Suite per Dimension. For each VBench evaluation dimension, we carefully designed a suite of around 100 prompts as test cases. The prompt suite is carefully curated to probe the specific ability relevant to the dimension tested. For example, for the “Subject Consistency” dimension which aims to evaluate the consistency of subjects’ appearances throughout the video, we ensure every prompt has a movable subject (e.g., animals or vehicles) performing non-static actions, where their consistency might be compromised due to inconsistency introduced by their movements or changing locations. In “Object Class” dimension, we ensure the existence of a specific class of object in every prompt. For

- TABLE 1: Image Suite Resolution Distribution. The Image Suite primarily consists of images of 4K resolution or higher. We present statistics based on two types of resolution classifications: one based on image area (i.e., the total number of pixels), and the other based on the length of both sides.

|Loose Definision of Image Resolution<br><br>|Image Area (W×H) Percentage|Image Side Length (W and H) Percentage|
|---|---|---|
|<1K<br><br>[1K, 2K)<br><br>[2K, 4K) [4K, 8K)<br><br><br>≥8K<br><br>|<1920×1080 0.0% [1920×1080, 2560×1440) 3.4% [2560×1440, 3840x2160) 6.8% [3840×2160, 7680×4320) 85.6%<br><br>≥7680×4320 4.2%|W<1920 or H<1080 0.3% (1920≤W and 1080≤H) and (W<2560 or H<1440) 5.4%<br><br>(2560≤W and 1440≤H) and (W<3840 or H<2160) 23.1% (3840≤W and 2160≤H) and (W<7680 or H<4320) 68.7%<br><br>7680≤W and 4320≤H 2.5%|

“Human Action”, each test prompt contains a human subject performing a well-defined action from the Kinetics-400 dataset [159], where 100 representative actions are selected with minimal semantic overlaps among themselves.

Prompt Suite per Category. When designing prompts for each dimension, the focus was to showcase models’ ability in that specific dimension. We further incorporate prompt suites for eight content categories to provide insights into the performance across varied content types. To this end, we prepare a collection of human-curated prompts from the Internet and divide them into 8 distinctive categories following YouTube’s categorization. Subsequently, we feed both the category labels and prompts into a Large Language Model (LLM) [160], obtaining multilabel outputs for each caption. We select 800 prompts and manually clean their labels to serve as per-category prompt suites. Finally, we obtain 100 prompts for each of these eight categories: Animal, Architecture, Food, Human, Lifestyle, Plant, Scenery, and Vehicles.

- 3.2.2 Prompt Suite for Trustworthiness

We design concise yet comprehensive prompts that address various dimensions of trustworthiness, ensuring broad coverage across diverse scenarios within each dimension.

Culture Fairness. We follow Huntington’s theory [161] and categorize cultures into nine major classes: 1 African, Buddhist, Chinese, Christian, Greco-Roman, Hindu, Islamic, Japanese, and Latin American. We select 14 distinct scenarios (i.e., wedding, dance, architecture, palace, holiday celebration, art, landscape, students, male dressing, female dressing, interior design, greeting, praying, and dinner), where each scenario typically has prominent visual differences across culture. Each cultural category is paired with each scenario, resulting in totally 126 text prompts.

Human Bias. We use a shared prompt suite to evaluate “Gender Bias” and “Skin Tone Bias”. To measure these biases effectively, we follow HEIM [131] and design text prompts that are semantically agnostic to gender or skin tone, and test for distribution skewness in the sampled videos in terms of skin tone or gender. These prompts describe humans from one of the six aspects: Occupation, Location, Emotion, Appearance, Behavior and Clothing. For each aspect, we design 15 text prompts, resulting in a total of 90 prompts.

1. The original Huntington’s classification includes African, Buddhist, Chinese, Hindu, Islamic, Japanese, Latin American, Orthodox and Western. We made with slight modifications for to achieve a more robust culture classification.

For example, we have 15 different occupations, human in 15 different locations, and faces with 15 different emotions etc. All prompts follow the format "The portrait of ...", such as "an artist", or "a person cooking a meal", "a person wearing yoga clothes".

Safety. We focus on text prompts that appear harmless but could potentially lead to the generation of unsafe videos. This is because text filters can readily screen out explicitly unsafe text prompts [138], [157], while subtler cues within seemingly innocent prompts presents a more complex challenge for safe video generation. Following Safe Latent Diffusion [120], we categorize harmful content into seven classes: hate, harassment, violence, self-harm, sexual content, shocking images, and illegal activity. For each class, we use GPT-4 [162] to assist with prompt generation and manually screen the outputs to ensure no harmful language is present. Additionally, we incorporate generalized descriptions of real historical events and scenes to avoid referencing specific sensitive individuals or events, resulting in totally 90 carefully curated prompts.

###### 3.3 Image Suite

To evaluate I2V models, we develop a comprehensive and high-quality Image Suite. This suite is adaptable to various resolutions and aspect ratios and offers diversity in both foreground and background content categories.

High Resolution. We curate images from Pexels [163] and Pixabay [164], both known for providing high-quality and royalty-free photos. Most of the selected images have an original resolution of 4K or higher. We detail the resolution distribution of our Image Suite in Table 1.

Adaptive Aspect Ratios. Since different I2V models use varying default resolutions (e.g., SVD [55] uses 1024×576 and ConsistI2V [60] uses 256×256), we propose that models should be evaluated at their default resolution and aspect ratio to avoid potential degradation in video quality from resolution changes. To facilitate this, we introduce a pipeline that converts images to various resolutions and aspect ratios while preserving their main content for fair evaluations. The pipeline has two stages: image selection and image cropping. In the selection stage, we identify images suitable for cropping that allow the main content to remain intact. Specifically, we select images where the main subject does not occupy a large portion of the frame, which permits full retention of content during cropping. In the cropping stage, we first generate two crops with extreme aspect ratios commonly used by I2V models: 1:1 for a square format and 16:9 for a widescreen format. After determining the 16:9 and 1:1 bounding boxes of the original image, we calculate

[Figure 38]

- (a) Cropping Pipeline for Portrait Images.

[Figure 39]

- (b) Cropping Pipeline for Landscape Images.

Fig. 5: Image Suite Pipeline for Adaptive Aspect Ratio Cropping. We provide a pipeline that crops images to various aspect ratios while preserving key content. (a) Portrait Images. If the original image’s width is less than its height, it is first cropped to a 1:1 ratio (red bounding box), followed by a second crop to a 16:9 aspect ratio (yellow bounding box). Additional crops interpolate between the 1:1 red box and the 16:9 yellow box to produce other common ratios (1:1, 7:4, 8:5, 16:9). (b) Landscape Images. If the original image’s width is greater than its height, we first crop the image to a 16:9 aspect ratio (red bounding box), and further crop the 16:9 image to a 1:1 aspect ratio (yellow bounding box). We then perform additional crops between the 16:9 red box and 1:1 yellow box to obtain the common aspect ratios (1:1, 7:4, 8:5, 16:9).

[Figure 40]

Fig. 6: Content Distribution of Image Suite. Our image suite encompasses a wide variety of content to ensure a comprehensive evaluation.

camera control instructions to the end of the prompt, such as “camera pans left” or “camera zooms in.”

In summary, our Image Suite provides several advantages: adaptive aspect ratios for testing different imageto-video models at their default resolutions; diverse and balanced content for comprehensive evaluation; high resolution, primarily 4K and above, to support tasks requiring detailed, high-quality images; and carefully crafted text prompts customized for each image.

###### 3.4 Human Preference Annotation

We perform human preference labeling on massive generated videos. The primary goal is to validate VBench evaluation’s alignment with human perception in each of the 16 evaluation dimensions, and the verification results will be detailed in Section 4.2. In addition to T2V evaluation dimensions, we further conduct extensive human annotations for I2V dimensions and trustworthiness dimensions, ensuring that all newly proposed evaluation methods align with human judgment.

bounding boxes for intermediate aspect ratios based on these crops to ensure the main content remains centered and unaltered. This process is illustrated in Figure 5.

Diverse and Fair Content for both Foreground and Background. Our image suite encompasses a wide variety of content to ensure a comprehensive evaluation. Foreground content includes diverse categories such as Human, Animal, Plant, Food, Transportation, and Others. Background images cover categories like Architecture, Scenery, Indoor, and Abstract. The content distribution statistics are shown in Figure 6. To ensure fairness, we maintained diversity within each category. For example, in the Human category, we considered factors such as the number of individuals, age, race, and gender.

Text Prompts Paired with Images. Each image is paired with a tailored text prompt to guide the video generation process. We start by generating initial captions using models like CoCa [165] and BLIP2 [155]. These captions are then manually reviewed and refined: phrases like ”an image of” or ”a picture of” are removed, descriptions are adjusted to better reflect the image content, and motion details are added. The refined prompts are used as the benchmark for the Video-Image Consistency dimensions. For the Video-Text Camera Motion Consistency dimension, we append specific

3.4.1 Human Annotations for Text-to-Video Evaluations

Text-to-Video Data Preparation. Given a text prompt pi, and four video generation models to be evaluated {A,B,C,D}, we use each model to generate a video, forming a “group” of videos Gi,j = {Vi,A,j,Vi,B,j,Vi,C,j,Vi,D,j}. For each prompt pi, we sample five such groups of videos {Gi,0,Gi,1,Gi,2,Gi,3,Gi,4}. For each group, we pair the videos up in pair-wise combinations, yielding six pairs: (VA,VB), (VA,VC), (VA,VD), (VB,VC), (VB,VD), (VC,VD), and ask human annotators to indicate their preferred video for each pair. Within the VBench evaluation framework, a prompt suite of N prompts produces N × 5 × 6 pairwise video comparisons. The video order within each pair is randomized to ensure unbiased annotation.

Human Labeling Rules. Specifically, the human annotators are asked to only consider the specific evaluation dimension of interest and select the preferred video. For example, in Figure 4, for the Appearance Style dimension, the question is “Is the style of the video in the Van Gogh style?”, and human annotators are instructed to only focus on whether the generated video’s style belongs to the Van Gogh style and should not consider other quality aspects of the generated video, such as potential issues like the degree of temporal flickering. In the example in this figure, video A resembles

- TABLE 2: Text-to-Video Evaluation Results per Dimension. This table compares the performance of video generative models across each of the 16 VBench dimensions. We continuously expand the VBench++ Leaderboard by evaluating 32 additional models beyond the 4 models initially presented in the CVPR 2024 paper. A selection of these newly evaluated models is presented in the table below. A higher score indicates relatively better performance for a particular dimension. We also provide two specially built baselines, i.e., Empirical Min and Max (the approximated achievable min and max scores for each dimension), as references.

|Models|Consistency<br><br>Subject<br><br>|Consistency<br><br>Background|Flickering<br><br>Temporal<br><br>|Smoothness<br><br>Motion<br><br>|Degree<br><br>Dynamic|Quality<br><br>Aesthetic<br><br>|Quality<br><br>Imaging<br><br>|Class<br><br>Object|
|---|---|---|---|---|---|---|---|---|
|LaVie [26] ModelScope [20], [27]<br><br>VideoCrafter-0.9 [24] CogVideo [19]<br><br>VideoCrafter-1.0 [62] Show-1 [25]<br><br>VideoCrafter-2.0 [36] Gen-2 [166]<br><br><br>AnimateDiff-v2 [86] Latte-1 [26] Pika-1.0 [167] Kling [168] Gen-3 [169] CogVideoX-2B [170] CogVideoX-5B [170]<br><br>|91.41% 89.87% 86.24%<br>92.19%<br><br><br>95.10%<br><br>95.53%<br>96.85%<br>97.61%<br><br>95.30% 88.88%<br>96.94%<br><br>98.33%<br><br>97.10%<br><br><br><br><br>96.78% 96.23%<br>|97.47%<br><br>95.29%<br><br>92.88%<br><br>96.20%<br><br>98.04% 98.02% 98.22%<br><br>97.61%<br><br><br><br><br>97.68%<br><br>95.40%<br><br>97.36% 97.60%<br><br>96.62%<br><br><br>96.63% 96.52%<br><br>|98.30% 98.28%<br><br>97.60%<br><br>97.64%<br><br>98.93%<br><br>99.12%<br><br><br>98.41%<br><br>99.56%<br><br><br>98.75%<br><br>98.89%<br><br>99.74%<br><br><br>99.30% 98.61% 98.89% 98.66%<br><br><br>|96.38%<br><br>95.79% 91.79%<br>96.47%<br><br>95.67%<br><br>98.24%<br><br>97.73%<br><br>99.58%<br><br><br>97.76% 94.63% 99.50% 99.40% 99.23% 97.73%<br><br>96.92%<br><br><br><br><br>|49.72% 66.39% 89.72% 42.22% 55.00% 44.44% 42.50% 18.89% 40.83% 68.89% 47.50% 46.94% 60.14% 59.86% 70.97%<br><br>|54.94% 52.06% 44.41% 38.18%<br><br>62.67% 57.35%<br>63.13%<br><br><br>66.96%<br>67.16%<br><br><br>61.59%<br>62.04% 61.21%<br>63.34%<br><br><br>60.82%<br>61.98%<br>|61.90% 58.57%<br><br>57.22%<br><br>41.03% 65.46%<br><br>58.66%<br><br><br>67.22% 67.42% 70.10% 61.92% 61.87%<br><br>65.62%<br><br>66.82%<br><br><br>61.68%<br><br>62.90%<br>|91.82%<br><br>82.25%<br><br>87.34%<br><br>73.40% 78.18% 93.07% 92.55% 90.92% 90.90%<br><br>86.53%<br><br>88.72%<br><br>87.24%<br><br><br><br><br>87.81%<br><br>83.37%<br><br><br><br><br>85.23%|
|Empirical Min Empirical Max|14.62% 100.00%<br><br>|26.15% 100.00%|62.93% 100.00%<br><br>|70.60% 99.75%<br><br>|0.00% 100.00%<br><br>|0.00% 100.00%|0.00% 100.00%<br><br>|0.00% 100.00%<br><br>|

|Models|Objects<br><br>Multiple|Action<br><br>Human<br><br>|Color|Relationship<br><br>Spatial<br><br>|Scene|Style<br><br>Appearance<br><br>|Style<br><br>Temporal<br><br>|Consistency<br><br>Overall|
|---|---|---|---|---|---|---|---|---|
|LaVie [26] ModelScope [20], [27]<br><br>VideoCrafter-0.9 [24] CogVideo [19]<br>VideoCrafter-1.0 [62] Show-1 [25]<br>VideoCrafter-2.0 [36] Gen-2 [166]<br><br><br>AnimateDiff-v2 [86] Latte-1 [26] Pika-1.0 [167] Kling [168] Gen-3 [169] CogVideoX-2B [170] CogVideoX-5B [170]|33.32%<br><br>38.98% 25.93% 18.11% 45.66% 45.47% 40.66% 55.47% 36.88%<br><br>34.53%<br><br><br>43.08% 68.05% 53.64% 62.63% 62.11%<br><br>|96.80%<br><br>92.40%<br>93.00% 78.20%<br><br><br>91.60% 95.60%<br><br>95.00%<br><br>89.20%<br><br>92.60%<br><br>90.00% 86.20%<br><br><br>93.40%<br><br>96.40%<br><br><br><br><br>98.00%<br>99.40%<br>|86.39%<br><br>81.72%<br><br>78.84%<br><br>79.57%<br><br>93.32%<br><br>86.35%<br><br>92.92%<br><br>89.49%<br><br>87.47% 85.31%<br><br>90.57%<br><br><br><br><br>89.90%<br><br>80.90%<br><br><br>79.41%<br><br>82.81%<br>|34.09%<br><br>33.68%<br><br>36.74% 18.24% 58.86% 53.50% 35.86% 66.91%<br><br>34.60%<br><br><br><br><br>41.53% 61.03% 73.03%<br><br>65.09%<br><br>69.90%<br><br>66.35%<br><br><br>|52.69% 39.26% 43.36% 28.24% 43.75%<br><br>47.03% 55.29%<br>48.91% 50.19% 36.26%<br>49.83%<br>50.86% 54.57%<br>51.14%<br><br><br>53.20%<br>|23.56%<br><br>23.39%<br><br>21.57%<br><br>22.01%<br><br>24.41%<br><br>23.06%<br><br>25.13% 19.34%<br><br>22.42%<br><br>23.74%<br><br>22.26% 19.62%<br><br>24.31%<br><br><br>24.80%<br><br><br><br><br>24.91%<br><br><br>|25.93% 25.37% 25.42% 7.80% 25.54% 25.28%<br><br>25.84% 24.12%<br>26.03% 24.76% 24.22% 24.17% 24.71%<br><br><br>24.36%<br>25.38%<br>|26.41%<br><br>25.67%<br><br>25.21%<br><br>7.70%<br><br>26.76%<br><br>27.46%<br><br>28.23%<br><br><br>26.17%<br><br>27.04%<br><br><br>27.33%<br><br><br>25.94%<br><br>26.42%<br><br><br>26.69%<br><br>26.66%<br><br>27.59%<br>|
|Empirical Min Empirical Max<br><br>|0.00% 100.00%|0.00% 100.00%<br><br>|0.00% 100.00%<br><br>|0.00% 100.00%|0.00% 82.22%<br><br>|0.09% 28.55%<br><br>|0.00% 36.40%|0.00% 36.40%<br><br>|

the Van Gogh better than video B, and the annotator is expected to select “A is better”. For every dimension, we carefully prepare instructions and train the human annotators to understand the definition of the dimension, and perform multiple quality assurance protocols via a prelabeling trial, and two rounds of post-labeling checks.

- 3.4.2 Human Annotations for Image-to-Video Evaluations The process for I2V human annotaions is similar to T2V’s. For each I2V evaluation dimension, annotators are provided with the input images, text prompts, and the corresponding generated videos. They are then asked to indicate their preference between two videos. Detailed documentation and examples are provided to ensure annotators grasp each dimension’s requirements. Multiple rounds of quality checks are conducted to guarantee the quality of the annotations.
- 3.4.3 Human Annotations for Trustworthiness Evaluations Trustworthiness dimensions focus on aspects such as bias, fairness, and safety in generated videos, often requiring

analysis across a batch of outputs. Annotators are tasked with selecting the group of videos that performs better in terms of fairness, bias mitigation, or safety. For each text prompt, 10 videos are sampled from each model. Annotators compare two groups of videos, and select the group that better satisfies the trustworthiness criteria. Extensive quality checks and re-labeling are performed to ensure high-quality annotations.

4 EXPERIMENTS

4.1 Per-Dimension Evaluation 4.1.1 Text-to-Video Evaluation

We initially adopted the video generation models LaVie [26], ModelScope [20], [27], VideoCrafter [24], and CogVideo [19] for VBench evaluation, and 32 more models have been added to the VBench++ Leaderboard as they become available.

- TABLE 3: Image-to-Video Evaluation Results. This table compares the performance of seven I2V models across VBench++’s I2V dimensions. A higher score indicates relatively better performance for a particular dimension.

|Models|Subject<br><br>I2V|Background<br><br>I2V|Motion<br><br>Camera|Consistency<br><br>Subject<br><br>|Consistency<br><br>Background<br><br>|Flickering<br><br>Temporal|Smoothness<br><br>Motion|Degree<br><br>Dynamic<br><br>|Quality<br><br>Aesthetic|Quality<br><br>Imaging|
|---|---|---|---|---|---|---|---|---|---|---|
|DynamiCrafter-1024 [56] SEINE-512x320 [57] I2VGen-XL [64] Animate-Anything [63] ConsistI2V [60] VideoCrafter-I2V [62] SVD-XT-1.1 [55]|96.71% 94.85%<br><br>96.74%<br><br>98.54% 94.69% 90.97%<br><br>97.51%<br>|96.05%<br><br>94.02%<br><br>95.44%<br><br>96.88%<br><br>94.57% 90.51%<br><br>97.62%<br>|35.44% 23.36% 13.32% 12.56% 33.60% 33.58% -<br><br>|95.69%<br><br>94.20%<br><br>96.36% 98.90%<br><br>95.27%<br><br><br>97.86% 95.42%<br>|97.38%<br><br>97.26%<br><br>97.93%<br><br>98.19%<br><br><br>98.28%<br><br><br>98.79%<br><br><br>96.77%<br><br>|97.63%<br><br>96.72%<br><br>98.48% 98.14%<br><br>97.56%<br>98.19%<br>99.17%<br><br><br>|97.38%<br><br>96.68%<br><br>98.31% 98.61%<br><br>97.38%<br><br>98.00%<br><br><br><br><br>98.12%|47.40% 34.31% 24.96% 2.68% 18.62% 22.60% 43.17%<br><br>|66.46%<br><br>58.42% 65.33%<br><br>67.12%<br><br>59.00%<br><br>60.78% 60.23%<br><br><br><br><br>|69.34%<br>70.97%<br><br>69.85% 72.09% 66.92%<br><br>71.68%<br><br>70.23%<br><br><br>|

- TABLE 4: Evaluation Results for Model Trustworthiness. This table compares the trustworthiness of image and video generative models. A higher score indicates relatively better performance for a particular dimension.

Culture Fairness

Culture Fairness

| | |
|---|---|
| | |

| | |
|---|---|
| | |

- (a) T2V Results for Trustworthiness.

|Models|Fairness<br><br>Culture<br><br>|Bias<br><br>Gender<br><br>|Bias<br><br>Skin|Safety|
|---|---|---|---|---|
|LaVie [26] ModelScope [20], [27] Show-1 [171] VideoCrafter0.9 [24] VideoCrafter2.0 [36] CogVideo [19]<br><br>|81.59% 81.75% 79.21% 74.76% 84.92% 49.29%|22.91% 36.70% 16.68% 39.57% 14.25% 21.59%<br><br>|13.38% 28.44% 20.61% 17.56% 30.94% 15.08%<br><br>|50.11%<br><br>41.22% 43.89%<br>42.00% 54.33% 42.11%<br>|

- (b) T2I Results for Trustworthiness.

Gender Bias

Gender Bias

Safety

Safety

Skin Bias

Skin Bias

| |
|---|

| |
|---|

LaVie

ModelScope CogVideo VideoCrafter-0.9 VideoCrafter-2.0 Show-1

LaVie

ModelScope CogVideo VideoCrafter-0.9 VideoCrafter-2.0 Show-1 SD1.4 SD2.1 SDXL

(a) Trustworthiness of Video Generative Models.

(b) Trustworthiness of Video vs. Image Models.

|Models|Fairness<br><br>Culture|Bias<br><br>Gender<br><br>|Bias<br><br>Skin<br><br>|Safety|
|---|---|---|---|---|
|Stable Diffusion 1.4 [172]<br>Stable Diffusion 2.1 [172] SDXL [77]<br>|84.13%<br><br>88.33%<br><br>89.37%<br><br><br>|40.27% 34.93% 38.42%|30.03% 33.20% 38.29%<br><br>|57.56% 59.11% 65.00%|

Fig. 7: Trustworthiness of Visual Generative Models. We visualize the trustworthiness evaluation results of visual generative models. For comprehensive numerical results, please refer to Table 4.

For every dimension, we calculate the VBench scores using the evaluation method suite described in Section 3.1, and show the results using Figure 2a and Table 2. We additionally designed three reference baselines, namely Empirical Max, Empirical Min, and WebVid-Avg. The first two approximate the maximum / minimum scores that videos might be able to achieve, and WebVid-Avg reflects the WebVid10M [173] dataset quality in each VBench dimension.

Empirical Max. For most dimensions, to approximate the maximum achievable values, we first retrieve WebVid10M [173] videos according to our prompt suites. We use CLIP [43] to extract text features of both WebVid-10M’s captions and our prompts. For each prompt, we retrieve the top-5 WebVid-10M videos according to text feature similarity with the given prompt. Given that the generated videos are usually 2 seconds in length, we randomly select a 2-second segment from each retrieved video and sample frames at 8 frames per second (FPS). For each dimension, we use the retrieved videos according to its prompt suite and report the highest-scoring video’s result as Empirical Max.

Empirical Min. To approximate the minimum achievable values, we use randomly generated 2-second Gaussian noise clips to calculate results for the “Video-Condition Consistency” dimensions. For most “Video Quality” dimensions, we select frames from real videos and design frame concatenation for each dimension, approximating the minimum score achievable for each VBench dimension.

WebVid-Avg. Similar to Empirical Max, we compute the average for each dimension on retrieved WebVid-10M [173] videos. This baseline could reflect the average perdimension quality of the commonly used video generation

training dataset WebVid-10M, and provide a reference for model performances. The comparison against WebVid-Avg and Empirical Max is visualized in Figure 11 (b).

- 4.1.2 Image-to-Video Evaluation

We initially evaluate the following I2V models: DynamiCrafter-1024 [56], SEINE-512x320 [57], I2VGenXL [64], Animate-Anything [63], ConsistI2V [60], VideoCrafter-I2V [62], and SVD-XT-1.1 [55]. For every I2V dimension, we compute scores using the evaluation method suite described in Section 3.1.3. The results are presented in Figure 2b and Table 3.

- 4.1.3 Trustworthiness Evaluation

For trustworthiness, we initially evaluate these models: LaVie [26], ModelScope [20], [27], Show1 [171], VideoCrafter-0.9 [24], VideoCrafter-2.0 [36], and CogVideo [19]. For each trustworthiness dimension, we compute scores using the evaluation method suite described in Section 3.1.4. The evaluation results are presented in Table 4a and Figure 7a.

###### 4.2 Validating Human Alignment of VBench++

To validate that our evaluation method can faithfully reflect human perception, we performed a large-scale human annotation for each dimension, as mentioned in Section 3.4. We show the correlation between VBench++ evaluation results and human preference annotations in Figure 8, 9, and 10.

Win Ratio. Given the human labels, we calculate the win ratio of each model. During pairwise comparisons, if a

Subject Consistency

Background Consistency

Temporal Flickering

Motion Smoothness

Dynamic Degree

Aesthetic Quality

Imaging Quality

Object Class

1.0

1.0

1.0

1.0

1.0

1.0

1.0

1.0

= 0.9651

= 0.9412

= 0.8873

= 0.9980

= 0.8209

= 0.9865

= 0.9216

= 0.8037

0.5

0.5

0.5

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0

0.0

0.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

VBench

Multiple Objects

Human Action

Color

Spatial Relationship

Scene

Appearance Style

Temporal Style

Overall Consistency

1.0

1.0

1.0

1.0

1.0

1.0

1.0

1.0

= 0.9898

= 0.8915

= 0.6073

= 0.9759

= 0.9407

= 0.9965

= 0.9753

= 0.9327

0.5

0.5

0.5

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0

0.0

0.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

0.5 1.0

Human

LaVie win ratio

ModelScope win ratio VideoCrafter win ratio CogVideo win ratio Fitting

| | | |
|---|---|---|
| | | |

- Fig. 8: Human Alignment of Text-to-Video (T2V) Evaluations. Our experiments show that VBench evaluations across all dimensions closely match human perceptions. Each plot shows the alignment verification result of a specific VBench dimension. In each plot, a dot represents the human preference win ratio (horizontal axis) and VBench evaluation win ratio (vertical axis) for a particular video generation model. We linearly fit a straight line to visualize the correlation, and calculate the Spearman’s correlation coefficient (ρ) for each dimension.

0.0 0.5 1.0

Human

0.0

0.5

1.0

VBench

= 0.9421

I2V Subject

Fitting

0.0 0.5 1.0

Human

0.0

0.5

1.0

VBench

= 0.8827

I2V Background

Fitting

0.0 0.5 1.0

Human

0.0

0.5

1.0

VBench

= 0.9987

Camera Motion

Fitting

Animate-Anything DynamiCrafter-1024 VideoCrafter I2VGen-XL SEINE-512x320 ConsistI2V SVD-XT 1.1

- Fig. 9: Human Alignment of Image-to-Video (I2V) Evaluations. Experiments show that our Image-to-Video (I2V) evaluations across all dimensions closely match human perceptions. Each plot shows the alignment verification result of a specific evaluation dimension, similar to Figure 8.

Culture Fairness

Gender Bias

Skin Bias

Safety

1.0

1.0

1.0

1.0

= 0.9310

= 0.8814

= 0.6864

= 0.9241

VBench

VBench

VBench

VBench

0.5

0.5

0.5

0.5

Fitting

Fitting

Fitting

Fitting

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

Human

Human

Human

Human

LaVie ModelScope VideoCrafter-0.9 CogVideo Show-1 VideoCrafter-2.0

Fig. 10: Human Alignment of Trustworthiness Evaluations. Experiments show that our trustworthiness evaluations across all dimensions align well with human perceptions. Each plot shows the alignment verification result of a specific evaluation dimension, similar to Figure 8.

model’s video is selected as better, then the model scores 1 and the other model scores 0. If there is a tie, then both models score 0.5. For each model, the win ratio is calculated as the total score divided by the total number of pair-wise comparisons participated.

T2V Evaluation. For each T2V evaluation dimension, we calculate the model win ratio based on (1) VBench evaluation results, and (2) human annotation results, respectively, and compute their correlations, as shown in Figure 8. We observe that VBench’s per-dimension evaluation results are highly correlated with human preference annotations.

I2V Evaluation. For each I2V dimension, we calculated the model win ratios based on both VBench++ evaluation results and human preference annotations, and their correlations. The results are visualized in Figure 9.

Trustworthiness Evaluation. Similarly, we validate human alignment for each trustworthiness dimension. The results are presented in Figure 10.

###### 4.3 Per-Category Evaluation

We evaluate the T2V models across eight different content categories, by generating videos based on Prompt Suite per Category described in Section 3.2.1, and then calculating their performance across different evaluation dimensions. Figure 12 visualizes the evaluation results of each model in terms of the eight content categories.

###### 4.4 Video Generation V.S. Image Generation

We conduct a comparative analysis of the frame-wise generation capability exhibited by text-to-video (T2V) models and text-to-image (T2I) models with two primary objectives: first, to assess the extent to which T2V models have successfully inherited the frame-wise generative capability of the T2I models; and second, to investigate the framewise generation capability gap between existing T2I and T2V models. As an initial exploration into this problem, we compare video generation models with three image generation models, namely Stable Diffusion (SD) 1.4 [76], SD2.1 [76], and SDXL [77]. We choose 10 VBench dimensions that can encompass frame-wise generation capabilities, and sample frames from all the image and video generation models according to Prompt Suite per Evaluation Dimension described in Section 3.2.1. Figure 11 (a) visualizes the evaluation results of T2V versus T2I models.

Trustworthiness. We additionally perform comparisons of model trustworthiness of T2V and T2I models in Table 4b and Figure 7b.

##### 5 INSIGHTS AND DISCUSSIONS

In this section, we discuss the observations and insights we draw from our comprehensive evaluation experiments.

· Trade-off across Ability Dimensions. We have noticed a trade-off in video generation models between 1) temporal consistency (Subject Consistency, Background Consistency,

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

- JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 12

[Figure 46]

der synthesis, impacting both spatial and temporal dimensions, probably because poor temporal modeling results in distorted and blurred imagery. This highlights the need for improved handling of dynamic motions in video generation models.

· Challenges of Data Quantity in Handling Complex Categories like Human. The WebVid-10M dataset [173] allocates 26% of its content to the Human category, which is the largest share among the eight categories. However, the Human category exhibits one of the poorest results among eight categories (see Figure 12). This suggests that merely increasing data volume may not significantly enhance performance in complex categories like Human. A potential approach could involve integrating human-related priors or controls, such as skeletons, to better capture the articulated nature of human appearances and movements.

(a) T2V vs. T2I (b) T2V vs. WebVid-Avg & Max

- Fig. 11: More Comparisons of Video Generation Models with Other Models and Baselines. We use VBench to evaluate other models and baselines for further comparative analysis of T2V models. (a) Comparison with text-to-image (T2I) generation models. (b) Comparison with WebVid-Avg and Empirical Max baselines.

· Prioritizing Data Quality Over Quantity in Large-Scale Datasets. For Aesthetic Quality, Figure 12 shows that the Food category almost always tends to have the highest scores among all categories. This is corroborated by the WebVid-10M dataset [173], where Food ranks highest in Aesthetic Quality according to VBench evaluation, despite comprising just 11% of the total data. This observation suggests that at million scales, data quality might hold greater importance than quantity. Furthermore, VBench’s evaluation dimensions can be potentially useful for cleaning datasets in specified quality dimensions.

Temporal Flickering, Motion Smoothness) and 2) Dynamic Degree. Models strong in temporal consistency often have a lower Dynamic Degree, as these two aspects are somewhat complementary (see Figure 2a and Table 2). For example, LaVie excels in Background Consistency and Temporal Flickering but has a low Dynamic Degree, probably because generating relatively static scenes can “cheat” to get high temporal consistency scores. Conversely, VideoCrafter shows a high Dynamic Degree but suffers from poor performance in all temporal consistency dimensions. This trend highlights the current challenge for models to achieve temporal consistency with dynamic content of large motions. Future research should focus on enhancing both aspects simultaneously, as improving only one might indicate compromising the other.

· Compositionality: T2I versus T2V. As shown in Figure 11 (a), T2V models significantly underperform in Multiple Objects and Spatial Relationship compared to T2I models (especially SDXL [77]), which highlights the need to enhance compositionality (i.e., correctly composing multiple objects in the same frame). We believe possible solutions might be: 1) curating training data incorporating multiple objects with corresponding captions explicitly depicting this compositionality, or 2) adding intermediate spatial control modules or modalities during video synthesis. Furthermore, the disparity of the text encoders might also account for the performance gap. As T2I models leverage bigger (OpenCLIP ViT-H for SD2.1 [76]) or more sophisticated (CLIP ViT-L & OpenCLIP ViT-G for SDXL [77]) text encoders compared with T2V models (e.g., CLIP ViT-L alone for LaVie), more representative text embeddings could be featuring more accurate object composition comprehension.

· Uncovering Hidden Potential of T2V Models in Specific Content Categories. Our analysis reveals that the capabilities of some models vary significantly across different content types. For instance, for Aesthetic Quality, CogVideo scores well for Food (see Figure 12 rightmost chart), whereas it underperforms in others like Animal and Vehicles. The average results across various prompts might suggest a lower overall “Aesthetic Quality” (as seen in Figure 2a), but CogVideo demonstrates relatively strong aesthetics in at least the Food category. This suggests that with tailored training data and strategies, CogVideo could potentially match other models in aesthetics by improving such ability in other content types. Therefore, we recommend evaluating video generation models not just based on ability dimensions but also considering specific content categories to uncover their hidden potential.

· Video-Image Consistency versus Between-Frame Consistency. For I2V models of comparable levels of Dynamic Degree, we observe a trade-off between 1) the consistency of generated videos with the input image, and 2) the consistency between frames within the generated videos. For example, in the I2V Background dimension, the ranking of the three models with similar Dynamic Degree is: I2VGen-XL [64] > ConsistI2V [60] > VideoCrafter-I2V [62], which is inversely related with their ranking in Background Consistency: I2VGen-XL [64] < ConsistI2V [60] < VideoCrafter-I2V [62]. This observation might suggest that an I2V model’s ability to align closely with the input image may limit its capacity to maintain temporal consistency across video frames. Conversely, relaxing the focus on input image alignment might enhance temporal consistency in video semantics.

· Bottleneck in Temporally Complex Categories Affecting Spatial and Temporal Performance. For spatially complex categories (e.g., Animal, LifeStyle, Human, Vehicles), models all perform relatively poorly mainly in Aesthetic Quality (shown in Figure 12). This is likely due to the challenges in synthesizing harmonious color schemes, articulated structures, and appealing layouts amidst complex elements. On the other hand, for categories involving complex and intense motions like Human and Vehicle, performance is relatively poor across all dimensions. This suggests that motion complexity and dynamic intensity significantly hin-

· Limited Performance in Camera Motion Control. The performance in the Camera Motion dimension of current

[Figure 47]

- JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 13

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

- Fig. 12: VBench++ Results across Eight Content Categories (best viewed in color). For each chart, we plot the VBench++ evaluation results across eight different content categories, benchmarked by our Prompt Suite per Category. The results are linearly normalized between 0 and 1 for better visibility across categories.

I2V models remain relatively low, with the highest classification accuracy reaching only 35.44%. This highlights the need for substantial improvement in current models to accurately follow camera motion instructions. Potential solutions include expanding training datasets with videos featuring prominent camera motions paired with detailed annotations, or integrating explicit camera control mechanisms into the models.

Natural Science Foundation of China (62076119, 61921006, 62102150); the Science and Technology Commission of Shanghai Municipality (23YF1461900, 23QD1400800); and Shanghai Artificial lntelligence Laboratory.

##### REFERENCES

- [1] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv preprint arXiv:1312.6114, 2013.
- [2] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. WardeFarley, S. Ozair, A. C. Courville, and Y. Bengio, “Generative adversarial nets,” in NeurIPS, 2014.
- [3] M. Mirza and S. Osindero, “Conditional generative adversarial nets,” arXiv preprint arXiv:1411.1784, 2014.
- [4] A. Brock, J. Donahue, and K. Simonyan, “Large scale GAN training for high fidelity natural image synthesis,” arXiv preprint arXiv:1809.11096, 2018.
- [5] T. Karras, T. Aila, S. Laine, and J. Lehtinen, “Progressive growing of GANs for improved quality, stability, and variation,” in ICLR, 2018.
- [6] T. Karras, S. Laine, and T. Aila, “A style-based generator architecture for generative adversarial networks,” in CVPR, 2019.
- [7] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila, “Analyzing and improving the image quality of StyleGAN,” in CVPR, 2020.
- [8] T. Karras, M. Aittala, S. Laine, E. H¨ark¨onen, J. Hellsten, J. Lehtinen, and T. Aila, “Alias-free generative adversarial networks,” in NeurIPS, 2021.
- [9] Y. Jiang, Z. Huang, X. Pan, C. C. Loy, and Z. Liu, “Talk-to-Edit: Fine-grained facial editing via dialog,” in ICCV, 2021.
- [10] J. Fu, S. Li, Y. Jiang, K.-Y. Lin, C. Qian, C. C. Loy, W. Wu, and Z. Liu, “Stylegan-human: A data-centric odyssey of human generation,” in ECCV, 2022.
- [11] J. Fu, S. Li, Y. Jiang, K.-Y. Lin, W. Wu, and Z. Liu, “Unitedhuman: Harnessing multi-source data for high-resolution human generation,” in ICCV, 2023.
- [12] Y. Jiang, Z. Huang, T. Wu, X. Pan, C. C. Loy, and Z. Liu, “Talkto-edit: Fine-grained 2d and 3d facial editing via dialog,” IEEE TPAMI, 2023.
- [13] A. Van Den Oord, O. Vinyals et al., “Neural discrete representation learning,” in NeurIPS, 2017.
- [14] P. Esser, R. Rombach, and B. Ommer, “Taming transformers for high-resolution image synthesis,” in CVPR, 2021.
- [15] Y. Jiang, S. Yang, H. Qju, W. Wu, C. C. Loy, and Z. Liu, “Text2human: Text-driven controllable human image generation,” ACM TOG, 2022.
- [16] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in NeurIPS, 2020.
- [17] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics,” in ICML, 2015.
- [18] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in ICLR, 2021.

· Trustworthiness in Video Generative Models. Among the models evaluated, those primarily developed by industrial companies (e.g., ModelScope [20], [27] and VideoCrafter2.0 [36]) demonstrate relatively stronger performance in trustworthiness compared to models originating from academic research (e.g., CogVideo [19] and Show-1 [171]). A possible explanation is that industrial organizations might have more diverse in-house data and emphasize internal reviews focused on trustworthiness during model development. We encourage users and researchers to exercise caution when using video generation models, keeping safety and ethical considerations in mind.

##### 6 CONCLUSION

With the growing focus on video generation, comprehensive evaluation of these models is essential to assess current advancements and guide future research. In this work, we take the first step forward and propose VBench, a comprehensive benchmark suite for evaluating video generation models. On top of VBench, we propose VBench++ to support image-to-video evaluation, and trustworthiness evaluation. We also continually include newly released models to our leaderboard to drive forward the field of video generation. With its multi-dimensional, human-aligned, insight-rich, and versatile properties, VBench++ could play vital roles for evaluating future video generation models and inspiring further advancements in video generation. We believe that VBench++ is a significant contribution to the video generation and evaluation community.

Acknowledgement. This study is supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOET2EP20221-0012), NTU NAP, and under the RIE2020 Industry Alignment Fund - Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s); the National Key R&D Program of China (2022ZD0160201); National

- [19] W. Hong, M. Ding, W. Zheng, X. Liu, and J. Tang, “CogVideo: Large-scale pretraining for text-to-video generation via transformers,” arXiv preprint arXiv:2205.15868, 2022.
- [20] Z. Luo, D. Chen, Y. Zhang, Y. Huang, L. Wang, Y. Shen, D. Zhao, J. Zhou, and T. Tan, “VideoFusion: Decomposed diffusion models for high-quality video generation,” in CVPR, 2023.
- [21] U. Singer, A. Polyak, T. Hayes, X. Yin, J. An, S. Zhang, Q. Hu, H. Yang, O. Ashual, O. Gafni et al., “Make-a-video: Text-to-video generation without text-video data,” arXiv preprint arXiv:2209.14792, 2022.
- [22] J. Z. Wu, Y. Ge, X. Wang, S. W. Lei, Y. Gu, W. Hsu, Y. Shan, X. Qie, and M. Z. Shou, “Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation,” arXiv preprint arXiv:2212.11565, 2022.
- [23] A. Blattmann, R. Rombach, H. Ling, T. Dockhorn, S. W. Kim, S. Fidler, and K. Kreis, “Align your latents: High-resolution video synthesis with latent diffusion models,” in CVPR, 2023.
- [24] Y. He, T. Yang, Y. Zhang, Y. Shan, and Q. Chen, “Latent video diffusion models for high-fidelity video generation with arbitrary lengths,” arXiv preprint arXiv:2211.13221, 2022.
- [25] D. J. Zhang, J. Z. Wu, J.-W. Liu, R. Zhao, L. Ran, Y. Gu, D. Gao, and M. Z. Shou, “Show-1: Marrying pixel and latent diffusion models for text-to-video generation,” arXiv preprint arXiv:2309.15818, 2023.
- [26] Y. Wang, X. Chen, X. Ma, S. Zhou, Z. Huang, Y. Wang, C. Yang, Y. He, J. Yu, P. Yang et al., “Lavie: High-quality video generation with cascaded latent diffusion models,” arXiv preprint arXiv:2309.15103, 2023.
- [27] J. Wang, H. Yuan, D. Chen, Y. Zhang, X. Wang, and S. Zhang, “Modelscope text-to-video technical report,” arXiv preprint arXiv:2308.06571, 2023.
- [28] L. Yu, Y. Cheng, K. Sohn, J. Lezama, H. Zhang, H. Chang, A. G. Hauptmann, M.-H. Yang, Y. Hao, I. Essa et al., “Magvit: Masked generative video transformer,” in CVPR, 2023.
- [29] L. Yu, J. Lezama, N. B. Gundavarapu, L. Versari, K. Sohn, D. Minnen, Y. Cheng, A. Gupta, X. Gu, A. G. Hauptmann et al., “Language model beats diffusion–tokenizer is key to visual generation,” arXiv preprint arXiv:2310.05737, 2023.
- [30] D. Kondratyuk, L. Yu, X. Gu, J. Lezama, J. Huang, R. Hornung, H. Adam, H. Akbari, Y. Alon, V. Birodkar et al., “Videopoet: A large language model for zero-shot video generation,” arXiv preprint arXiv:2312.14125, 2023.
- [31] A. Gupta, L. Yu, K. Sohn, X. Gu, M. Hahn, L. Fei-Fei, I. Essa, L. Jiang, and J. Lezama, “Photorealistic video generation with diffusion models,” arXiv preprint arXiv:2312.06662, 2023.
- [32] O. Bar-Tal, H. Chefer, O. Tov, C. Herrmann, R. Paiss, S. Zada, A. Ephrat, J. Hur, Y. Li, T. Michaeli et al., “Lumiere: A spacetime diffusion model for video generation,” arXiv preprint arXiv:2401.12945, 2024.
- [33] W. Menapace, A. Siarohin, I. Skorokhodov, E. Deyneka, T.-S. Chen, A. Kag, Y. Fang, A. Stoliar, E. Ricci, J. Ren et al., “Snap video: Scaled spatiotemporal transformers for text-to-video synthesis,” arXiv preprint arXiv:2402.14797, 2024.
- [34] W. Wang, J. Liu, Z. Lin, J. Yan, S. Chen, C. Low, T. Hoang, J. Wu, J. H. Liew, H. Yan et al., “Magicvideo-v2: Multi-stage highaesthetic video generation,” arXiv preprint arXiv:2401.04468, 2024.
- [35] R. Girdhar, M. Singh, A. Brown, Q. Duval, S. Azadi, S. S. Rambhatla, A. Shah, X. Yin, D. Parikh, and I. Misra, “Emu video: Factorizing text-to-video generation by explicit image conditioning,” arXiv preprint arXiv:2311.10709, 2023.
- [36] H. Chen, Y. Zhang, X. Cun, M. Xia, X. Wang, C. Weng, and Y. Shan, “Videocrafter2: Overcoming data limitations for highquality video diffusion models,” arXiv preprint arXiv:2401.09047, 2024.
- [37] R. Villegas, M. Babaeizadeh, P.-J. Kindermans, H. Moraldo, H. Zhang, M. T. Saffar, S. Castro, J. Kunze, and D. Erhan, “Phenaki: Variable length video generation from open domain textual descriptions,” in ICLR, 2022.
- [38] S. Ge, T. Hayes, H. Yang, X. Yin, G. Pang, D. Jacobs, J.-B. Huang, and D. Parikh, “Long video generation with time-agnostic vqgan and time-sensitive transformer,” in ECCV, 2022.
- [39] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, X. Chen, and X. Chen, “Improved techniques for training gans,” in NeurIPS, 2016.
- [40] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “GANs trained by a two time-scale update rule converge to a local nash equilibrium,” in NeurIPS, 2017.

- [41] T. Unterthiner, S. van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly, “Towards accurate generative models of video: A new metric & challenges,” arXiv preprint arXiv:1812.01717, 2018.
- [42] ——, “FVD: A new metric for video generation,” in ICLRW, 2019.
- [43] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in ICML, 2021.
- [44] M. Ding, W. Zheng, W. Hong, and J. Tang, “Cogview2: Faster and better text-to-image generation via hierarchical transformers,” in NeurIPS, 2022.
- [45] M. Otani, R. Togashi, Y. Sawai, R. Ishigami, Y. Nakashima, E. Rahtu, J. Heikkil¨a, and S. Satoh, “Toward verifiable and reproducible human evaluation for text-to-image generation,” in CVPR, 2023.
- [46] H. Wu, C. Chen, L. Liao, J. Hou, W. Sun, Q. Yan, J. Gu, and W. Lin, “Neighbourhood representative sampling for efficient end-toend video quality assessment,” arXiv preprint arXiv:2210.05357, 2022.
- [47] H. Wu, C. Chen, J. Hou, L. Liao, A. Wang, W. Sun, Q. Yan, and W. Lin, “Fast-vqa: Efficient end-to-end video quality assessment with fragment sampling,” in ECCV, 2022.
- [48] H. Wu, E. Zhang, L. Liao, C. Chen, J. H. Hou, A. Wang, W. Sun, Q. Yan, and W. Lin, “Exploring video quality assessment on user generated contents from aesthetic and technical perspectives,” in ICCV, 2023.
- [49] ——, “Towards explainable video quality assessment: A database and a language-prompted approach,” in ACM MM, 2023.
- [50] H. Wu, L. Liao, A. Wang, C. Chen, J. H. Hou, E. Zhang, W. Sun, Q. Yan, and W. Lin, “Towards robust text-prompted semantic criterion for in-the-wild video quality assessment,” arXiv preprint arXiv:2304.14672, 2023.
- [51] H. Wu, L. Liao, C. Chen, J. H. Hou, E. Zhang, A. Wang, W. Sun, Q. Yan, and W. Lin, “Exploring opinion-unaware video quality assessment with semantic affinity criterion,” in ICME, 2023.
- [52] H. Wu, C. Chen, L. Liao, J. Hou, W. Sun, Q. Yan, and W. Lin, “Discovqa: Temporal distortion-content transformers for video quality assessment,” IEEE TCSVT, 2023.
- [53] D. Li, T. Jiang, and M. Jiang, “Quality assessment of in-the-wild videos,” in ACM MM, 2019.
- [54] Z. Tu, Y. Wang, N. Birkbeck, B. Adsumilli, and A. C. Bovik, “Ugc-vqa: Benchmarking blind video quality assessment for user generated content,” IEEE TIP, 2021.
- [55] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts et al., “Stable video diffusion: Scaling latent video diffusion models to large datasets,” arXiv preprint arXiv:2311.15127, 2023.
- [56] J. Xing, M. Xia, Y. Zhang, H. Chen, X. Wang, T.-T. Wong, and Y. Shan, “Dynamicrafter: Animating open-domain images with video diffusion priors,” arXiv preprint arXiv:2310.12190, 2023.
- [57] X. Chen, Y. Wang, L. Zhang, S. Zhuang, X. Ma, J. Yu, Y. Wang, D. Lin, Y. Qiao, and Z. Liu, “Seine: Short-to-long video diffusion model for generative transition and prediction,” arXiv preprint arXiv:2310.20700, 2023.
- [58] X. Chen, Z. Liu, M. Chen, Y. Feng, Y. Liu, Y. Shen, and H. Zhao, “Livephoto: Real image animation with text-guided motion control,” arXiv preprint arXiv:2312.02928, 2023.
- [59] J. Yu, X. Cun, C. Qi, Y. Zhang, X. Wang, Y. Shan, and J. Zhang, “Animatezero: Video diffusion models are zero-shot image animators,” arXiv preprint arXiv:2312.03793, 2023.
- [60] W. Ren, H. Yang, G. Zhang, C. Wei, X. Du, S. Huang, and W. Chen, “Consisti2v: Enhancing visual consistency for imageto-video generation,” arXiv preprint arXiv:2402.04324, 2024.
- [61] C. Wang, J. Gu, P. Hu, S. Xu, H. Xu, and X. Liang, “Dreamvideo: High-fidelity image-to-video generation with image retention and text guidance,” arXiv preprint arXiv:2312.03018, 2023.
- [62] H. Chen, M. Xia, Y. He, Y. Zhang, X. Cun, S. Yang, J. Xing, Y. Liu, Q. Chen, X. Wang, C. Weng, and Y. Shan, “Videocrafter1: Open diffusion models for high-quality video generation,” arXiv preprint arXiv:2310.19512, 2023.
- [63] Z. Dai, Z. Zhang, Y. Yao, B. Qiu, S. Zhu, L. Qin, and W. Wang, “Fine-grained open domain image animation with motion guidance,” arXiv preprint arXiv:2311.12886, 2023.
- [64] S. Zhang, J. Wang, Y. Zhang, K. Zhao, H. Yuan, Z. Qin, X. Wang, D. Zhao, and J. Zhou, “I2vgen-xl: High-quality image-

- to-video synthesis via cascaded diffusion models,” arXiv preprint arXiv:2311.04145, 2023.
- [65] W. Weng, R. Feng, Y. Wang, Q. Dai, C. Wang, D. Yin, Z. Zhao, K. Qiu, J. Bao, Y. Yuan et al., “Art-v: Auto-regressive text-to-video generation with diffusion models,” in CVPR, 2024.
- [66] Y. Zeng, G. Wei, J. Zheng, J. Zou, Y. Wei, Y. Zhang, and H. Li, “Make pixels dance: High-dynamic video generation,” in CVPR, 2024.
- [67] X. Guo, M. Zheng, L. Hou, Y. Gao, Y. Deng, C. Ma, W. Hu, Z. Zha, H. Huang, P. Wan et al., “I2v-adapter: A general imageto-video adapter for video diffusion models,” arXiv preprint arXiv:2312.16693, 2023.
- [68] D. J. Zhang, D. Li, H. Le, M. Z. Shou, C. Xiong, and D. Sahoo, “Moonshot: Towards controllable video generation and editing with multimodal conditions,” arXiv preprint arXiv:2401.01827, 2024.
- [69] X. Shi, Z. Huang, F.-Y. Wang, W. Bian, D. Li, Y. Zhang, M. Zhang, K. C. Cheung, S. See, H. Qin et al., “Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling,” arXiv preprint arXiv:2401.15977, 2024.
- [70] X. Wang, Z. Zhu, G. Huang, B. Wang, X. Chen, and J. Lu, “Worlddreamer: Towards general world models for video generation via predicting masked tokens,” arXiv preprint arXiv:2401.09985, 2024.
- [71] Z. Huang, Y. He, J. Yu, F. Zhang, C. Si, Y. Jiang, Y. Zhang, T. Wu, Q. Jin, N. Chanpaisit, Y. Wang, X. Chen, L. Wang, D. Lin, Y. Qiao, and Z. Liu, “VBench: Comprehensive benchmark suite for video generative models,” in CVPR, 2024.
- [72] P. Dhariwal and A. Nichol, “Diffusion models beat GANs on image synthesis,” in NeurIPS, 2021.
- [73] A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen, “GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models,” arXiv preprint arXiv:2112.10741, 2021.
- [74] S. Gu, D. Chen, J. Bao, F. Wen, B. Zhang, D. Chen, L. Yuan, and B. Guo, “Vector quantized diffusion model for text-to-image synthesis,” in CVPR, 2022.
- [75] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. Denton, S. K. S. Ghasemipour, B. K. Ayan, S. S. Mahdavi, R. G. Lopes et al., “Photorealistic text-to-image diffusion models with deep language understanding,” arXiv preprint arXiv:2205.11487, 2022.
- [76] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in CVPR, 2022.
- [77] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Muller,¨ J. Penna, and R. Rombach, “Sdxl: Improving latent diffusion models for high-resolution image synthesis,” arXiv preprint arXiv:2307.01952, 2023.
- [78] Z. Huang, K. C. Chan, Y. Jiang, and Z. Liu, “Collaborative diffusion for multi-modal face generation and editing,” in CVPR, 2023.
- [79] Z. Huang, T. Wu, Y. Jiang, K. C. Chan, and Z. Liu, “ReVersion: Diffusion-based relation inversion from images,” arXiv preprint arXiv:2303.13495, 2023.
- [80] Z. Xing, Q. Feng, H. Chen, Q. Dai, H. Hu, H. Xu, Z. Wu, and Y.-G. Jiang, “A survey on video diffusion models,” arXiv preprint arXiv:2310.10647, 2023.
- [81] W. Harvey, S. Naderiparizi, V. Masrani, C. Weilbach, and F. Wood, “Flexible diffusion modeling of long videos,” arXiv preprint arXiv:2205.11495, 2022.
- [82] J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet, “Video diffusion models,” arXiv preprint arXiv:2204.03458, 2022.
- [83] J. Ho, W. Chan, C. Saharia, J. Whang, R. Gao, A. Gritsenko, D. P. Kingma, B. Poole, M. Norouzi, D. J. Fleet et al., “Imagen video: High definition video generation with diffusion models,” arXiv preprint arXiv:2210.02303, 2022.
- [84] D. Zhou, W. Wang, H. Yan, W. Lv, Y. Zhu, and J. Feng, “Magicvideo: Efficient video generation with latent diffusion models,” arXiv preprint arXiv:2211.11018, 2022.
- [85] S. Ge, S. Nah, G. Liu, T. Poon, A. Tao, B. Catanzaro, D. Jacobs, J.-B. Huang, M.-Y. Liu, and Y. Balaji, “Preserve your own correlation: A noise prior for video diffusion models,” in ICCV, 2023.
- [86] Y. Guo, C. Yang, A. Rao, Y. Wang, Y. Qiao, D. Lin, and B. Dai, “Animatediff: Animate your personalized text-to-image diffusion models without specific tuning,” in ICLR, 2024.
- [87] L. Khachatryan, A. Movsisyan, V. Tadevosyan, R. Henschel, Z. Wang, S. Navasardyan, and H. Shi, “Text2video-zero: Text-

- to-image diffusion models are zero-shot video generators,” arXiv preprint arXiv:2303.13439, 2023.
- [88] Y. Jiang, S. Yang, T. L. Koh, W. Wu, C. C. Loy, and Z. Liu, “Text2Performer: Text-driven human video generation,” in ICCV, 2023.
- [89] X. Ma, Y. Wang, G. Jia, X. Chen, Z. Liu, Y.-F. Li, C. Chen, and Y. Qiao, “Latte: Latent diffusion transformer for video generation,” arXiv preprint arXiv:2401.03048, 2024.
- [90] S. Yin, C. Wu, J. Liang, J. Shi, H. Li, G. Ming, and N. Duan, “Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory,” arXiv preprint arXiv:2308.08089, 2023.
- [91] P. Esser, J. Chiu, P. Atighehchian, J. Granskog, and A. Germanidis, “Structure and content-guided video synthesis with diffusion models,” in ICCV, 2023.
- [92] T.-S. Chen, C. H. Lin, H.-Y. Tseng, T.-Y. Lin, and M.-H. Yang, “Motion-conditioned diffusion model for controllable video synthesis,” arXiv preprint arXiv:2304.14404, 2023.
- [93] J. H. Liew, H. Yan, J. Zhang, Z. Xu, and J. Feng, “Magicedit: Highfidelity and temporally coherent video editing,” arXiv preprint arXiv:2308.14749, 2023.
- [94] C. Qi, X. Cun, Y. Zhang, C. Lei, X. Wang, Y. Shan, and Q. Chen, “Fatezero: Fusing attentions for zero-shot text-based video editing,” arXiv preprint arXiv:2303.09535, 2023.
- [95] S. Yang, Y. Zhou, Z. Liu, and C. C. Loy, “Rerender a video: Zero-shot text-guided video-to-video translation,” arXiv preprint arXiv:2306.07954, 2023.
- [96] H. Ouyang, Q. Wang, Y. Xiao, Q. Bai, J. Zhang, K. Zheng, X. Zhou, Q. Chen, and Y. Shen, “Codef: Content deformation fields for temporally consistent video processing,” in CVPR, 2024.
- [97] W. Chai, X. Guo, G. Wang, and Y. Lu, “Stablevideo: Textdriven consistency-aware diffusion video editing,” arXiv preprint arXiv:2308.09592, 2023.
- [98] J. Zhang, H. Yan, Z. Xu, J. Feng, and J. H. Liew, “Magicavatar: Multimodal avatar generation and animation,” arXiv preprint arXiv:2308.14748, 2023.
- [99] X. Wang, H. Yuan, S. Zhang, D. Chen, J. Wang, Y. Zhang, Y. Shen, D. Zhao, and J. Zhou, “Videocomposer: Compositional video synthesis with motion controllability,” arXiv preprint arXiv:2306.02018, 2023.
- [100] Y. Ma, Y. He, X. Cun, X. Wang, Y. Shan, X. Li, and Q. Chen, “Follow your pose: Pose-guided text-to-video generation using pose-free videos,” arXiv preprint arXiv:2304.01186, 2023.
- [101] Y. Zhang, Y. Wei, D. Jiang, X. Zhang, W. Zuo, and Q. Tian, “Controlvideo: Training-free controllable text-to-video generation,” arXiv preprint arXiv:2305.13077, 2023.
- [102] W. Chen, J. Wu, P. Xie, H. Wu, J. Li, X. Xia, X. Xiao, and L. Lin, “Control-a-video: Controllable text-to-video generation with diffusion models,” arXiv preprint arXiv:2305.13840, 2023.
- [103] J. Karras, A. Holynski, T.-C. Wang, and I. KemelmacherShlizerman, “Dreampose: Fashion image-to-video synthesis via stable diffusion,” arXiv preprint arXiv:2304.06025, 2023.
- [104] K. Soomro, A. R. Zamir, and M. Shah, “Ucf101: A dataset of 101 human actions classes from videos in the wild,” arXiv preprint arXiv:1212.0402, 2012.
- [105] J. Xu, T. Mei, T. Yao, and Y. Rui, “Msr-vtt: A large video description dataset for bridging video and language,” in CVPR, 2016.
- [106] K. Huang, K. Sun, E. Xie, Z. Li, and X. Liu, “T2i-compbench: A comprehensive benchmark for open-world compositional textto-image generation,” arXiv preprint arXiv: 2307.06350, 2023.
- [107] S. Wang, C. Saharia, C. Montgomery, J. Pont-Tuset, S. Noy, S. Pellegrini, Y. Onoe, S. Laszlo, D. J. Fleet, R. Soricut et al., “Imagen editor and EditBench: Advancing and evaluating textguided image inpainting,” arXiv preprint arXiv:2212.06909, 2022.
- [108] T. Lee, M. Yasunaga, C. Meng, Y. Mai, J. S. Park, A. Gupta, Y. Zhang, D. Narayanan, H. B. Teufel, M. Bellagente et al., “Holistic evaluation of text-to-image models,” arXiv preprint arXiv:2311.04287, 2023.
- [109] S. Basu, M. Saberi, S. Bhardwaj, A. M. Chegini, D. Massiceti, M. Sanjabi, S. X. Hu, and S. Feizi, “Editval: Benchmarking diffusion based text-guided image editing methods,” arXiv preprint arXiv:2310.02426, 2023.
- [110] E. M. Bakr, P. Sun, X. Shen, F. F. Khan, L. E. Li, and M. Elhoseiny, “Hrs-bench: Holistic, reliable and scalable benchmark for text-toimage models,” in ICCV, 2023.
- [111] M. Ku, T. Li, K. Zhang, Y. Lu, X. Fu, W. Zhuang, and W. Chen, “Imagenhub: Standardizing the evaluation of conditional image generation models,” in ICLR, 2024.

- [112] Y. Liu, X. Cun, X. Liu, X. Wang, Y. Zhang, H. Chen, Y. Liu, T. Zeng, R. Chan, and Y. Shan, “Evalcrafter: Benchmarking and evaluating large video generation models,” in CVPR, 2024.
- [113] Y. Liu, L. Li, S. Ren, R. Gao, S. Li, S. Chen, X. Sun, and L. Hou, “Fetv: A benchmark for fine-grained evaluation of open-domain text-to-video generation,” in NeurIPS, 2023.
- [114] F. Fan, C. Luo, W. Gao, and J. Zhan, “Aigcbench: Comprehensive evaluation of image-to-video content generated by ai,” BenchCouncil Transactions on Benchmarks, Standards and Evaluations, p. 100152, 2024.
- [115] Y. Zhang, Z. Xing, Y. Zeng, Y. Fang, and K. Chen, “Pia: Your personalized image animator via plug-and-play modules in textto-image models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [116] W. Feng, J. Li, M. Saxon, T.-j. Fu, W. Chen, and W. Y. Wang, “Tcbench: Benchmarking temporal compositionality in text-to-video and image-to-video generation,” arXiv preprint arXiv:2406.08656, 2024.
- [117] Y. Wan, A. Subramonian, A. Ovalle, Z. Lin, A. Suvarna, C. Chance, H. Bansal, R. Pattichis, and K.-W. Chang, “Survey of bias in text-to-image generation: Definition, evaluation, and mitigation,” arXiv preprint arXiv:2404.01030, 2024.
- [118] F. Bianchi, P. Kalluri, E. Durmus, F. Ladhak, M. Cheng, D. Nozza, T. Hashimoto, D. Jurafsky, J. Zou, and A. Caliskan, “Easily accessible text-to-image generation amplifies demographic stereotypes at large scale,” in Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency, 2023, pp. 1493–1504.
- [119] A. Jha, V. Prabhakaran, R. Denton, S. Laszlo, S. Dave, R. Qadri, C. K. Reddy, and S. Dev, “Beyond the surface: A global-scale analysis of visual stereotypes in text-to-image generation,” arXiv preprint arXiv:2401.06310, 2024.
- [120] P. Schramowski, M. Brack, B. Deiseroth, and K. Kersting, “Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 22522–22531.
- [121] H. Luo, Z. Deng, R. Chen, and Z. Liu, “Faintbench: A holistic and precise benchmark for bias evaluation in text-to-image models,” arXiv preprint arXiv:2405.17814, 2024.
- [122] S. Zhai, H. Chen, Y. Dong, J. Li, Q. Shen, Y. Gao, H. Su, and Y. Liu, “Membership inference on text-to-image diffusion models via conditional likelihood discrepancy,” arXiv preprint arXiv:2405.14800, 2024.
- [123] M. Hall, S. J. Bell, C. Ross, A. Williams, M. Drozdzal, and A. R. Soriano, “Towards geographic inclusion in the evaluation of textto-image models,” arXiv preprint arXiv:2405.04457, 2024.
- [124] Y. Qu, X. Shen, Y. Wu, M. Backes, S. Zannettou, and Y. Zhang, “Unsafebench: Benchmarking image safety classifiers on realworld and ai-generated images,” arXiv preprint arXiv:2405.03486, 2024.
- [125] M. Saxon, Y. Luo, S. Levy, C. Baral, Y. Yang, and W. Y. Wang, “Lost in translation? translation errors and challenges for fair assessment of text-to-image models on multilingual concepts,” arXiv preprint arXiv:2403.11092, 2024.
- [126] M. Chen, Y. Liu, J. Yi, C. Xu, Q. Lai, H. Wang, T.-Y. Ho, and Q. Xu, “Evaluating text-to-image generative models: An empirical study on human image synthesis,” arXiv preprint arXiv:2403.05125, 2024.
- [127] L. Weidinger, M. Rauh, N. Marchal, A. Manzini, L. A. Hendricks, J. Mateos-Garcia, S. Bergman, J. Kay, C. Griffin, B. Bariach et al., “Sociotechnical safety evaluation of generative ai systems,” arXiv preprint arXiv:2310.11986, 2023.
- [128] M. Ventura, E. Ben-David, A. Korhonen, and R. Reichart, “Navigating cultural chasms: Exploring and unlocking the cultural pov of text-to-image models,” arXiv preprint arXiv:2310.01929, 2023.
- [129] L. Struppek, D. Hintersdorf, F. Friedrich, P. Schramowski, K. Kersting et al., “Exploiting cultural biases via homoglyphs in text-toimage synthesis,” Journal of Artificial Intelligence Research, vol. 78, pp. 1017–1068, 2023.
- [130] Y. Zhang, L. Jiang, G. Turk, and D. Yang, “Auditing gender presentation differences in text-to-image models,” arXiv preprint arXiv:2302.03675, 2023.
- [131] T. Lee, M. Yasunaga, C. Meng, Y. Mai, J. S. Park, A. Gupta, Y. Zhang, D. Narayanan, H. Teufel, M. Bellagente et al., “Holistic evaluation of text-to-image models,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- [132] A. Basu, R. V. Babu, and D. Pruthi, “Inspecting the geographical representativeness of images from text-to-image models,” in

- Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 5136–5147.
- [133] C.-Y. Chuang, V. Jampani, Y. Li, A. Torralba, and S. Jegelka, “Debiasing vision-language models via biased prompts,” arXiv preprint arXiv:2302.00070, 2023.
- [134] Y. Kim, S. Mo, M. Kim, K. Lee, J. Lee, and J. Shin, “Bias-to-text: Debiasing unknown visual biases through language interpretation,” arXiv preprint arXiv:2301.11104, 2023.
- [135] R. He, C. Xue, H. Tan, W. Zhang, Y. Yu, S. Bai, and X. Qi, “Debiasing text-to-image diffusion models,” arXiv preprint arXiv:2402.14577, 2024.
- [136] X. Shen, C. Du, T. Pang, M. Lin, Y. Wong, and M. Kankanhalli, “Finetuning text-to-image diffusion models for fairness,” in The Twelfth International Conference on Learning Representations, 2024.
- [137] J. Cho, A. Zala, and M. Bansal, “Dall-eval: Probing the reasoning skills and social biases of text-to-image generation models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3043–3054.
- [138] OpenAI, “Dalle 3 system card,” https://cdn.openai.com/ papers/DALL E 3 System Card.pdf, 2022.

- [139] X. Li, Y. Yang, J. Deng, C. Yan, Y. Chen, X. Ji, and W. Xu, “Safegen: Mitigating unsafe content generation in text-to-image models,” arXiv preprint arXiv:2404.06666, 2024.
- [140] notAI tech, “Nudenet,” https://github.com/notAI-tech/ NudeNet, 2023.
- [141] P. Schramowski, C. Tauchmann, and K. Kersting, “Can machines help us answering question 16 in datasheets, and in turn reflecting on inappropriate content?” in Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, 2022, pp. 1350–1361.
- [142] M. Caron, H. Touvron, I. Misra, H. J´egou, J. Mairal, P. Bojanowski, and A. Joulin, “Emerging properties in self-supervised vision transformers,” in ICCV, 2021.
- [143] Z. Li, Z.-L. Zhu, L.-H. Han, Q. Hou, C.-L. Guo, and M.-M. Cheng, “Amt: All-pairs multi-field transforms for efficient frame interpolation,” in CVPR, 2023.
- [144] Z. Teed and J. Deng, “Raft: Recurrent all-pairs field transforms for optical flow,” in ECCV, 2020.
- [145] LAION-AI, “aesthetic-predictor,” https://github.com/ LAION-AI/aesthetic-predictor, 2022.
- [146] J. Ke, Q. Wang, Y. Wang, P. Milanfar, and F. Yang, “MUSIQ: multiscale image quality transformer,” CoRR, vol. abs/2108.05997,

2021. [Online]. Available: https://arxiv.org/abs/2108.05997

- [147] Y. Fang, H. Zhu, Y. Zeng, K. Ma, and Z. Wang, “Perceptual quality assessment of smartphone photography,” in CVPR, 2020.
- [148] J. Wu, J. Wang, Z. Yang, Z. Gan, Z. Liu, J. Yuan, and L. Wang, “Grit: A generative region-to-text transformer for object understanding,” arXiv preprint arXiv:2212.00280, 2022.
- [149] K. Li, Y. Wang, Y. Li, Y. Wang, Y. He, L. Wang, and Y. Qiao, “Unmasked teacher: Towards training-efficient video foundation models,” arXiv preprint arXiv:2303.16058, 2023.
- [150] X. Huang, Y. Zhang, J. Ma, W. Tian, R. Feng, Y. Zhang, Y. Li, Y. Guo, and L. Zhang, “Tag2text: Guiding vision-language model via image tagging,” arXiv preprint arXiv:2303.05657, 2023.
- [151] Y. Wang, Y. He, Y. Li, K. Li, J. Yu, X. Ma, X. Chen, Y. Wang, P. Luo, Z. Liu, Y. Wang, L. Wang, and Y. Qiao, “Internvid: A large-scale video-text dataset for multimodal understanding and generation,” arXiv preprint arXiv:2307.06942, 2023.
- [152] S. Fu*, N. Tamir*, S. Sundaram*, L. Chai, R. Zhang, T. Dekel, and P. Isola, “Dreamsim: Learning new dimensions of human visual similarity using synthetic data,” arXiv preprint arXiv:2306.09344, 2023.
- [153] N. Karaev, I. Rocco, B. Graham, N. Neverova, A. Vedaldi, and C. Rupprecht, “Cotracker: It is better to track together,” arXiv:2307.07635, 2023.
- [154] J. Deng, J. Guo, E. Ververas, I. Kotsia, and S. Zafeiriou, “Retinaface: Single-shot multi-level face localisation in the wild,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 5203–5212.
- [155] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19730–19742.
- [156] T. B. Fitzpatrick, “The validity and practicality of sun-reactive skin types i through vi,” Archives of dermatology, vol. 124, no. 6, pp. 869–871, 1988.

- [157] Y. Yang, R. Gao, X. Yang, J. Zhong, and Q. Xu, “Guardt2i: Defending text-to-image models from adversarial prompts,” arXiv preprint arXiv:2403.01446, 2024.
- [158] LAION-AI, “Nsfw detector,” https://github.com/LAION-AI/ CLIP-based-NSFW-Detector/blob/main/safety settings.yml, 2023.

- [159] W. Kay, J. Carreira, K. Simonyan, B. Zhang, C. Hillier, S. Vijayanarasimhan, F. Viola, T. Green, T. Back, P. Natsev et al., “The kinetics human action video dataset,” arXiv preprint arXiv:1705.06950, 2017.
- [160] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing et al., “Judging llm-as-a-judge with mtbench and chatbot arena,” arXiv preprint arXiv:2306.05685, 2023.
- [161] S. P. Huntington, “The clash of civilizations?” in The new social theory reader. Routledge, 2020, pp. 305–313.
- [162] OpenAI, “GPT-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.
- [163] “Pexels, royalty-free stock footage website,” https://www. pexels.com, accessed: 2024-03-01.
- [164] “Pixabay, royalty-free stock footage website,” https://pixabay. com/, accessed: 2024-03-01.
- [165] J. Yu, Z. Wang, V. Vasudevan, L. Yeung, M. Seyedhosseini, and Y. Wu, “Coca: Contrastive captioners are image-text foundation models,” arXiv preprint arXiv:2205.01917, 2022.
- [166] “Gen-2,” Accessed September 25, 2023 [Online] https:// research.runwayml.com/gen2, 2023. [Online]. Available: https: //research.runwayml.com/gen2
- [167] “Pika 1.0,” Accessed December 28, 2023 [Online] https://www. pika.art/, 2023. [Online]. Available: https://www.pika.art/
- [168] “Kling,” Accessed June 6, 2024 [Online] https://klingai. kuaishou.com/, 2024. [Online]. Available: https://klingai. kuaishou.com/
- [169] “Gen-3,” Accessed June 17, 2024 [Online] https://runwayml. com/research/introducing-gen-3-alpha, 2024. [Online]. Available: https://runwayml.com/research/introducing-gen-3-alpha
- [170] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng et al., “Cogvideox: Text-to-video diffusion models with an expert transformer,” arXiv preprint arXiv:2408.06072, 2024.
- [171] D. J. Zhang, J. Z. Wu, J.-W. Liu, R. Zhao, L. Ran, Y. Gu, D. Gao, and M. Z. Shou, “Show-1: Marrying pixel and latent diffusion models for text-to-video generation,” arXiv preprint arXiv:2309.15818, 2023.
- [172] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” 2021.
- [173] M. Bain, A. Nagrani, G. Varol, and A. Zisserman, “Frozen in time: A joint video and image encoder for end-to-end retrieval,” in ICCV, 2021.

