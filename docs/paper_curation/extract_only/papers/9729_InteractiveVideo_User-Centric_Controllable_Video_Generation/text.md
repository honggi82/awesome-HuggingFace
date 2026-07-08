## InteractiveVideo: User-Centric Controllable Video Generation with Synergistic Multimodal Instructions

Yiyuan Zhang1∗ Yuhao Kang2∗ Zhixin Zhang2* Xiaohan Ding3 Sanyuan Zhao2 Xiangyu Yue1

- 1 Multimedia Lab, The Chinese University of Hong Kong
- 2 Beijing Institute of Technology 3Tencent AI Lab

# arXiv:2402.03040v1[cs.CV]5Feb2024

yiyuanzhang.ai@gmail.com, kangyuhao@bit.edu.cn, xyyue@ie.cuhk.edu.hk https://invictus717.github.io/InteractiveVideo

User Personalization

[Figure 1]

[Figure 2]

[Figure 3]

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

Input Editing “purple flowers are trembling, a bee is flying”

Input Editing “yellow butterfly is flying to the cat's face”

Personalize

User Prompt Editing

###### User Content Editing

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

“a ship is moving cross the river”

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Text Image

Input Paint a boat “seaside, boat, the boat is leaving”

Input

Drag Paint

User Trajectory Editing

###### User Semantic Editing

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Input Paint Color “flowers are shaking in the wind”

Input Drag “car, running”

Figure 1. Interactive Video Generation We propose a user-centric framework that effectively synergizes users’ multimodal instructions. Users can easily edit key components in the video generation process, leading to high-quality video and increased user satisfaction.

### Abstract

users’ multimodal instructions into generative models, thus facilitating a cooperative and responsive interaction between user inputs and the generative process. This approach enables iterative and fine-grained refinement of the generation result through precise and effective user instructions. With InteractiveVideo, users are given the flexibility to meticulously tailor key aspects of a video. They can paint the reference image, edit semantics, and adjust video motions until their requirements are fully met. Code, models, and demo are available at https://github.com/ invictus717/InteractiveVideo.

We introduce “InteractiveVideo”, a user-centric framework for video generation. Different from traditional generative approaches that operate based on user-provided images or text, our framework is designed for dynamic interaction, allowing users to instruct the generative model through various intuitive mechanisms during the whole generation process, e.g. text and image prompts, painting, drag-and-drop, etc. We propose a Synergistic Multimodal Instruction mechanism, designed to seamlessly integrate

*Equal Contribution

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

“Clouds are moving”

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

“the tide is flooding”

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

“Tide is flooding”

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

“The clouds are becoming dark”

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

“Snow is melting”

- Figure 2. Comparison between Gen-2 and InteractiveVideo. For each case, the first row is the generation results of Gen-2, and the second row is our results. (More comparison results with Pika Labs, I2VGen-XL [48], and Gen-2 can be found in Appendix Figures 7, 8, and 9.)

### 1. Introduction

Video generation has attracted significant attention due to its promising future in the AI-Generated Content field [3, 18, 26, 40, 43] and its potential to enhance the efficiency of movie creation and serve as a new infrastructure technology for the film industry [7, 14, 29, 51]. The advancement of diffusion models [13, 30, 31, 47] has infused the field of video generation with new potential [3, 4, 26]. The success of Gen-1 [8], MagicVideo [51], and Align your latents [1] has significantly inspired further exploration of high-quality

visual content generation for videos.

As the capabilities of video generation models improve, user expectations for the generated videos are concurrently elevated, leading to an increased demand for videos that accurately meet their specific requirements. Existing video generation models typically utilize a reference image, known as the image condition, and a textual description, referred to as the text condition, as inputs. Enhancing video generation foundation models by making them larger, more advanced, and more sophisticated could potentially fulfill

user requirements more effectively by enhancing the understanding of image and text conditions, thereby producing videos of superior quality. However, our objective is to empower existing video generation models to more accurately fulfill user requirements from a different angle - by equipping models with the capability to interpret complex, multidimensional human instructions. This approach is driven by the observation that the current conditioning mechanisms (image and text) fall short of capturing the full spectrum of user intentions. 1) The text condition may not be informative enough. Even though existing video generation models support long and detailed text prompts, it is difficult to precisely depict complex video motions and dynamics using only text. As a result, it becomes challenging for generative models to fully interpret the intended video content. 2) The conditional image does not contain temporal information. The absence of optical flow and temporal consistency can easily lead to the introduction of unsatisfactory artifacts in the video generation process. Moreover, 3) there is a significant demand from users for the customization of videos, which entails the intuitive manipulation of video contents, semantics, and motions. In response to these challenges, we propose a novel approach that improves the ability of existing video generation models to better understand human intentions and generate videos guided by more detailed and multifaceted human instructions.

Recently, remarkable progress in large language models has drawn widespread attention across the community [2]. One key to the success of large language models is learning from human feedback through reinforcement learning [5, 20, 23, 32] which significantly improves the performance of language models and leads to superior generation results. Pioneers in the visual content generation field have also introduced human feedback to generate high-quality images [41]. Nonetheless, the intricacy, diversity, and level of control required for video generation far surpass those needed for single-image generation, making it a highly significant yet relatively underexplored challenge.

To address these challenges, we propose InteractiveVideo, a user-centric video generation framework that empowers users to actively participate in the generation process through multimodal instructions, enabling control over video content, semantics, and motions. Users can customize a video through various manipulations such as painting and dragging, as illustrated in Figure 1. More specifically, we propose a Synergistic Multimodal Instruction mechanism that empowers generative models to interpret and act upon users’ editing and revision instructions across various facets, such as video content, regional semantics, object motion, subjects, and the overall dynamics of the video. In our framework, we capture user interactions in the form of image, text, motion, and trajectory prompts, and we incorporate these user instructions as independent conditions

of probabilistic models. As a result, InteractiveVideo is a training-free framework that can be easily and flexibly applied to different fundamental generative models. It is worth noting that our proposed framework seamlessly integrates with existing generative models and practical techniques, such as Stable Diffusion [25], DreamBooth [27], and LoRA [17], thus expanding the video generation capabilities with our interactive framework.

In this framework, user interactions are involved through four distinct types of instructions which can be employed independently or collaboratively to effectively guide the video generation process. The four types of instructions are:

- • Image Instruction: the image condition or prompt for image-to-video generation.
- • Content Instruction: a textual description of the visual elements and the painting edits of the user to control the video content.
- • Motion Instruction: a textual description specifying the desired movements and dynamics of elements within the video.
- • Trajectory Instruction: user-defined motion trajectories for specific video elements, expressed through interactions such as dragging.

By incorporating these detailed and multidimensional human instructions, we can generate videos that better align with the unique preferences and requirements of users.

We compare our InteractiveVideo with the advanced video generation solutions, e.g. Gen-21, I2VGen-XL [48], and Pika Labs. Comparison results in Figure 2, 7, 8, 9 show the superiority of InteractiveVideo with higher quality, better flexibility, and richer controllability. Our InteractiveVideo paves the way for a novel paradigm in visual content generation, integrating user interactions to enable highly customized video generation. This empowers users to effortlessly obtain high-quality videos they desire through intuitive manipulation and effective interactions.

In summary, our contributions are as follows:

- • Framework Deisgn: we propose a novel interactive framework that empowers users to precisely control video generation by intuitive manipulations.
- • Generation Algorithm: we propose a Synergistic Multimodal Instructions mechanism, which integrates user prompts as probabilistic conditions and enables interaction without the need for additional training.
- • High-quality Video Generation: our generation results demonstrate superiority over state-of-the-art video generation methods, including Gen-2, I2VGen-XL [48], and Pika Labs.

1https://research.runwayml.com/gen2

### 2. Related Work

- 2.1. Video Generation

Initial attempts in video generation primarily leveraged Generative Adversarial Networks (GANs) [9, 16, 21, 24, 29, 33, 37, 44] and Variational Autoencoders (VAEs) [21, 22, 42]. These methods, however, faced considerable challenges in effectively modeling the intricate spatio-temporal dynamics necessary for text-driven video generation, leaving the problem largely unsolved. Subsequent innovations shifted towards diffusion models [13, 30, 31, 47] to enhance diversity and fidelity in video outputs [1, 3, 7, 14, 15, 18, 26, 40, 43, 45, 46, 51] and to scale up pre-training data and model architecture [6, 14, 16, 28, 36, 49]. Recent efforts have introduced spatio-temporal conditions [4, 7, 34, 43], for instance, through VideoComposer [34], Gen-1 [7] and DragNUWA [43]. These methods aim to provide a more controlled generation process but still encounter constraints in achieving flexible and user-satisfied video synthesis.

- 2.2. Models Guided by Human Feedback

The idea of learning from human feedback, initially investigated in reinforcement learning and agent alignment contexts [5, 20], was subsequently applied to large language models [23, 32]. This approach has significantly improved the generation of textual outputs that are helpful, honest, and harmless. In the field of visual content generation, particularly in video generation and editing, a similar goal is pursued. [19, 38, 41] demonstrates the great potential of human guidance for the visual content generation field.

These works collectively highlight the growing trend of incorporating human feedback in various forms of generative models, extending its utility from text-based to visual content generation. However, learning from human feedback for video generation remains under-explored owing to its complicated elements of motion, subjects, and spatialtemporal dynamics. We aim to fill this gap, providing a training-free and user-friendly solution for elevating existing video generative models with effective human guidance and generating more user-satisfying and higher-quality videos.

- 3. Methodology

##### 3.1. Preliminary

As shown in Figure 3, InteractiveVideo realizes controllable video generation with two generative pipelines based on latent diffusion models - 1) the text-to-image (T2I) pipeline Pimg and 2) the image-to-video (I2V) pipeline Pvideo. The framework outputs a video containing NF

F }. We denote the Image Instruction by x ∈ RC×H×W, the Content Instruction by y, the Motion Instruction by y′, and the Trajectory Instruction by r. More specifically, the Trajectory Instruction is represented by start and end points and region masks, which indicate the desired moving trajectories of specific objects. The whole pipeline can be formulated as

frames {v1,v2,··· ,vN

F } = Pvideo(Pimg(x,y),y′,r). (1)

{v1,v2,··· ,vN

In practice, we may implement Pimg with any off-theshelf T2I model as long as it takes a text condition and an image condition as inputs. We use x˜ to denote its generated image, i.e., the intermediate image, which is the input to the I2V model.

We then use x˜ as the image condition of the I2V pipeline and the Motion Instruction y′ as the text condition. We may use any off-the-shelf I2V diffusion models which require image and text conditions. Let E be the image encoder of the I2V model, z0 = E(x˜) be the corresponding latent code, ϵt be the predicted noise at step t, the classic (i.e., interaction-free) video denoising process can be denoted with

zt = √α¯tz0 + √1 − α¯t · ϵt, (2)

where α¯t is a parameter related to the variance schedule [12].

##### 3.2. Synergistic Multimodal Instructions

We control the video diffusion process with users’ multimodal instructions via altering the predicted noise according to the users’ operations. Conceptually, with R denoting the function that changes ϵt according to users’ operations, our interaction-controlled video diffusion process can be represented by

zt = √α¯tz0 + √1 − α¯t · R(ϵt). (3)

The proposed concrete implementation for the function R(·) is realized by treating user interactions as denoising residuals. Since the intermediate image x˜ is utilized as the condition image of Pvideo, it is seen as the “interface” between user and video generation model. Consequently, our framework empowers users to interact with the target video by introducing their interactions as new generation conditions of the video denoising process.

Specifically, we transform users’ operations into denoising residuals to eventually control the video diffusion process. Formally, in the video denoising process, assume the original intermediate image is x˜ and the corresponding latent code is z0 = E(x˜). Once the user has operated on the

InteractiveVideo by Synergistic Multimodal Instructions

Motion Instruction

Content Instruction

Text

Image

Video

“opening mouth ”

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

..

[Figure 121]

[Figure 122]

“A cute white cat ”

Input Control

#### x y

Trajectory Instruction

Paint Drag

Image Instruction

Condition Image

Edit Edit

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

“Flowers ”

Image Diffusion Video Diffusion

- Figure 3. Framework Illustration. In InteractiveVideo, users can utilize multimodal instructions to interact with generative models on video content, motion, and trajectory.

image (e.g., painted some lines or set some trajectories2), the intermediate image changes accordingly, and we denote the resultant intermediate image as x˜′ and the corresponding latent code becomes z0′ = E(x˜′). We use z0′ to predict the noise in the video diffusion process. Formally, let t be the time step, ϵt be the noise predicted with zt−1 and ϵ′t be the noise predicted with zt′−1, the noise we use is given by

distribution produced by users’ typical operations. Specifically, the eventual i-th video frame vi′ can be computed as:

vi′ = Conv2D(SiLU(GroupNorm(vi − x˜))). (5)

### 4. Experiments

In this section, we present features of the InteractiveVideo framework including personalization (§ 4.1), finegrained video editing (§ 4.2), and precise motion control

ϵˆt = λ · ϵt + (1 − λ) · ϵ′t , (4)

where λ is a hyper-parameter to balance the learned noise residual and human instructions. Then we use ϵˆt, instead of the original ϵt, in the denoising process to generate the eventual video.

- (§ 4.3). Besides, we also conduct quantitative analysis
- (§ 4.4) on generation quality and user study on the satisfaction rates with our framework. Then, we demonstrate the generation efficiency (§ 4.5) of our framework. 4.1. Personalizing a Video

Note that after the user operations on the image, the human-crafted discrepancy may affect the temporal coherence of the resulting video. This is because the user operations may have deviated the intermediate image from the distribution on which the I2V model was trained (e.g., the user has drawn a twisted yellow curve to create a sun in the sky, which is unusual in the training data of the I2V model). To solve this problem, upon the completion of the video diffusion process, we post-process the resultant video following AnimateDiff [10]. Every single frame is aligned with the intermediate image via a Group Normalization [39] layer, a SiLU [11] activation, and a 2D convolutional layer adopted from AnimateDiff or PIA [50], as such structures are found to generalize well to our common

Existing methods [7, 10, 34] have made significant progress in the animation of static images into videos. However, these methods are limited to animating objects or scenes already present in the original static images, and encounter difficulties when it comes to generating a video with objects or scenes absent from the referenced images. In other words, existing methods have limited ability to control video content, especially when users want to add or animate previously unseen objects or scenes.

With InteractiveVideo, we enable video content manipulation by incorporating abundant elements. In Figure 4, we demonstrate that our framework supports the users to customize the video content freely. For example, we use a brush to paint sketches of birds, waves, and polar lights in Figure 4 (a), (b), and (c), respectively. The added objects are seamlessly integrated and animated throughout the entire

2Painting and trajectory drawing affect the intermediate image in different ways. The former makes a difference on the intermediate image through the T2I pipeline as it changes the very beginning Image Instruction. The latter moves the handle points within the specified region to the target points and changes the optical flow of the intermediate image.

Input User Edit Generated Video

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

- (a)
- (b)
- (c)

“Morning, birds, birds are moving ”

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

“Sea, the wave is crashing”

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

“Dark night, the polar lights are floating”

- Figure 4. Video Content Manipulation with InteractiveVideo. In (a), (b), and (c), we present the content manipulation by adding birds, waves, and polar lights. Then, these added objects are driven in the whole video. We use these results to show the flexibility of our framework for video content creation.

video. Seen from the following frames, InteractiveVideo enables users to create a video of satisfactory temporal consistency even though the referenced image does not directly contain the objects. Meanwhile, such cases also demonstrate the versatility and adaptability of our framework in creating diverse and engaging video content, highlighting its potential for a wide range of applications in content creation and editing.

- ure 5 (b), and the logo in Figure 5 (c) can be easily modified. The generated videos are of high quality, featuring realistic motion, appropriate light reflection, and visually appealing textures. 4.3. Precise Motion Control

Motion control, particularly precise motion control, poses a significant challenge in the field of video generation due to the complexity of modeling spatial-temporal patterns. The primary difficulty lies in maintaining the temporal consistency of generated videos, especially when handling substantial motion. This issue mainly stems from the limited temporal receptive field of 1D temporal attention, which struggles to accommodate the full range of motionrelated changes over time. As a result, ensuring smooth and consistent representation of motion in generated videos remains a considerable obstacle in this field. Differently, InteractiveVideo excels in precise motion control, which we will discuss from three aspects as follows:

1) Large Motion. As shown in the first two rows of Fig-

- ure 6 first two rows, we present the large motion control by turning around characters in both realistic and cartoon styles. The details of turning around the female character are impressive, with the motion of her hair appearing highly realistic.

##### 4.2. Fine-grained Video Editing

Another significant limitation of current generation methods is the challenge of performing precise regional editing. During the generation process, models have difficulty interpreting natural language references such as “left”, “right”, “up”, and “down”. This makes it hard to accurately edit regional semantics, which is crucial for user experience.

Fortunately, InteractiveVideo overcomes this limitation by enabling intuitive manipulation in the intermediate image. As illustrated in Figure 5 (a), it is difficult for users to edit the color of a specific tree or control the color of a particular cluster of falling leaves using existing methods. In contrast, our framework allows users to perform finegrained semantic editing on any region. For example, after the editing process, the trees in Figure 5 (a), clouds in Fig-

Input User Edit Generated Video

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

- (a)
- (b)
- (c)

“Windy, red leaves are flying in the wind”

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

“View, colorful clouds are moving”

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

“The batman is taking off his disguise, he is superman”

- Figure 5. Fine-grained Video Editing with InteractiveVideo. In (a), (b), and (c), we perform fine-grained regional semantic editing on changing colors and appearances of specific objects, These results show the outstanding controllability of our framework for video generation.

- 2) Precise Motion. As seen in the third row of Figure 6,

the adorable corgi holding the “INTERACTIVE VIDEO” brand displays several different charming gestures, including wagging its tail, smiling with an open mouth, turning its head, and shaking its ears.

- 3) Multi-Object Motion. The last two rows in Figure

- Figure 6 showcase the ability of InteractiveVideo to control multi-object motion. Our framework precisely controls the movements of both the cute girl and the lovable dog. When adjusting the dog’s head, its tail also wags, and the girl naturally lowers her hand. While controlling these two objects, the girl smiles sweetly, and the dog turns its head to the other side.

Evaluation Metrics. We evaluate generation quality by considering image and text alignment, using CLIP scores to measure cosine similarity between embeddings. Image alignment compares input images and video frame embeddings, while text alignment examines text and frame embedding similarities.

CLIP Score User Study Satisfaction Rate

Methods

Image Text Image Text (%) VideoComposer[35] 225.3 62.85 0.180 0.110 43.5 AnimateDiff[10] 218.0 63.31 0.295 0.220 51.6 PIA [50] 225.9 63.68 0.525 0.670 52.5 InteractiveVideo [Ours] 234.6 65.31 0.745 0.813 72.8

Table 1. Quantitative comparison on AnimateBench.

##### 4.4. Quantitative Analysis

User Study. To substantiate the enhancement of our method in terms of visual quality and user experience, we carried out a user study comparing our approach with other video models. This study utilized 40 prompts from AnimateBench, which feature a variety of scenes, styles, and objects. Compared to existing video generation methods, our InteractiveVideo notably outperforms in terms of human preference scores and delivers state-of-the-art performance in user satisfaction rates. These quantitative results, coupled with the user study, effectively demonstrate the significance and superiority of the interactive generation paradigm and

AnimateBench. Since InteractiveVideo is a general framework for open-domain video generation, we use AnimateBench for comparison. We assessed the text-based video generation capability using 105 unique cases with varying content, styles, and concepts. These cases were created using seven distinct text-to-image models, with five images per model for thorough comparison. Additionally, we crafted three motion-related prompts for each image to evaluate motion controllability across different methods, focusing on potential single-shot image motions.

### 5. Responsible AI and Ethic Claim

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

In developing InteractiveVideo, our research rigorously adheres to the principles of Responsible AI and ethical guidelines. This innovative framework for video generation is designed with a strong commitment to ethical AI practices, ensuring that user interactions with the system - through text, images, and direct manipulation - are processed with the utmost integrity and transparency. The implementation of our Synergistic Multimodal Instruction mechanism is a testament to our dedication to these principles. It not only facilitates a seamless integration of diverse user inputs but also ensures that the AI operates within ethical boundaries, avoiding biases and respecting user intent. By empowering users to interactively manipulate the video generation process, InteractiveVideo promotes not just creativity but also responsibility in AI use. This approach aligns with our commitment to uphold ethical standards in AI, ensuring that InteractiveVideo serves as a model for responsible innovation in the realm of AI-driven content creation

“1Man, he is turning his body”

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

“1Girl, she is turning gradually”

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

“The dog is smiling”

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

### 6. Conclusion and Discussion

“The dog is turning its head”

In summation, we introduce InteractiveVideo, a novel paradigm shift in the domain of video generation that champions a user-centric approach over the conventional methodologies reliant on pre-defined images or textual prompts. This framework is distinguished by its capacity to facilitate dynamic, real-time interactions between the user and the generative model, enabled by a suite of intuitive interfaces including, but not limited to, text and image prompts, manual painting, and drag-and-drop capabilities. Central to our framework is the innovative Synergistic Multimodal Instruction mechanism, a testament to our commitment to integrating multifaceted user interaction into the generative process cohesively and efficiently. This mechanism augments the interactive experience and significantly refines the granularity with which users can influence the generation outcomes. The resultant capability for users to meticulously customize key video elements to their precise preferences, coupled with the consequent elevation in the visual quality of the generated content, underscores the transformative potential of InteractiveVideo in the landscape of video generation technologies.

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

“The girl is smiling, the dog is turning its head”

- Figure 6. Precise Motion Control of InteractiveVideo. Our framework shows strong controllability in large motion control, precise gesture control, and multi-object motion control.

user-centric designs.

##### 4.5. Generation Efficiency

InteractiveVideo takes only 16GB CUDA memory in the inference process, and it runs on a single RTX 4090. Besides, in Table 2, we also report the latency of InteractiveVideo. It is worth noting that InteractiveVideo can generate a video within about 12 seconds though it requires two independent diffusion models for better controllability.

Discussion on the Computational Efficiency. Notwithstanding the promising advancements heralded by InteractiveVideo, the adoption of a user-centric generative approach is not devoid of challenges. Paramount among these is the imperative to ensure the framework’s accessibility and intuitive usability across a broad spectrum of users, alongside maintaining the generative models’ efficacy and computational efficiency amidst diverse and dynamic input scenarios. Future research endeavors might fruitfully focus on

Process Image Instruction Content Instruction Motion Instruction Trajectory Instruction Time 19.34ms 31.47ms 12.22s 77.35 ms

Table 2. Latency Analysis of InteractiveVideo.

the refinement of these models to enhance scalability and the development of adaptive algorithms capable of more accurately interpreting and actualizing user intentions.

Future Works. We may delve into several promising directions. Enhancing the AI’s understanding of complex user inputs, such as emotional intent or abstract concepts, could lead to more nuanced and contextually relevant video generation. Additionally, exploring the integration of real-time feedback loops where the model suggests creative options based on user input history could further personalize the user experience. Investigating the application of this framework in virtual and augmented reality environments opens up new dimensions for immersive content creation. Furthermore, extending the framework’s capabilities to include collaborative generation where multiple users can interact and contribute to a single generative process may revolutionize co-creation in digital media.

Further Applications. The potential applications of InteractiveVideo extend well into the realms of education, where bespoke video content could significantly enrich the pedagogical experience and entertainment, particularly in the creation of interactive narratives. As we continue to iterate upon and enhance this framework, the scope for its application appears limitless, heralding a future in which video generation transcends mere content creation to become a conduit for deep, interactive engagement between creators and their digital canvases.

## Appendix

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

“The girl is smiling, the dog is turning its head”

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

“The man is turning gradually”

- Figure 7. Comparison with existing methods on motion control. We compare InteractiveVideo (4th row) with Pika Labs (1st row), I2VGen-XL (2nd row), and Gen-2 (3rd row).

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

“The clouds are moving”

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

“The tide is flooding”

- Figure 8. Comparison with existing methods on landscapes. We compare InteractiveVideo (4th row) with Pika Labs (1st row), I2VGenXL (2nd row), and Gen-2 (3rd row).

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

“The clouds are becoming dark”

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

“The snow is melting”

- Figure 9. Comparison with existing methods on dynamic scenes. We compare InteractiveVideo (4th row) with Pika Labs (1st row), I2VGen-XL (2nd row), and Gen-2 (3rd row).

### References

- [1] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 2, 4
- [2] Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Kaijie Zhu, Hao Chen, Linyi Yang, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. A survey on evaluation of large language models. arXiv preprint arXiv:2307.03109, 2023. 3
- [3] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter1: Open diffusion models for high-quality video generation, 2023. 2, 4
- [4] Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable text-to-video generation with diffusion models,

2023. 2, 4

- [5] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017. 3, 4
- [6] Xiaohan Ding, Yiyuan Zhang, Yixiao Ge, Sijie Zhao, Lin Song, Xiangyu Yue, and Ying Shan. Unireplknet: A universal perception large-kernel convnet for audio, video, point cloud, time-series and image recognition. arXiv preprint arXiv:2311.15599, 2023. 4
- [7] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. arXiv preprint arXiv:2302.03011, 2023. 2, 4, 5
- [8] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023. 2
- [9] Tsu-Jui Fu, Licheng Yu, Ning Zhang, Cheng-Yang Fu, JongChyi Su, William Yang Wang, and Sean Bell. Tell me what happened: Unifying text-guided video completion via multimodal masked video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10681–10692, 2023. 4
- [10] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 5, 7
- [11] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016. 5
- [12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 4
- [13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 2, 4

- [14] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2, 4
- [15] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. arXiv preprint arXiv:2204.03458, 2022. 4
- [16] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 4
- [17] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 3
- [18] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-toimage diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023. 2, 4
- [19] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning textto-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023. 4
- [20] Jan Leike, David Krueger, Tom Everitt, Miljan Martic, Vishal Maini, and Shane Legg. Scalable agent alignment via reward modeling: a research direction. arXiv preprint arXiv:1811.07871, 2018. 3, 4
- [21] Yitong Li, Martin Min, Dinghan Shen, David Carlson, and Lawrence Carin. Video generation from text. In Proceedings of the AAAI conference on artificial intelligence, 2018. 4
- [22] Gaurav Mittal, Tanya Marwah, and Vineeth N Balasubramanian. Sync-draw: Automatic video generation using deep recurrent attentive architectures. In Proceedings of the 25th ACM international conference on Multimedia, pages 1096– 1104, 2017. 4
- [23] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730–27744, 2022. 3, 4
- [24] Yingwei Pan, Zhaofan Qiu, Ting Yao, Houqiang Li, and Tao Mei. To create what you tell: Generating videos from captions. In Proceedings of the 25th ACM international conference on Multimedia, pages 1789–1798, 2017. 4
- [25] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 3
- [26] Ludan Ruan, Yiyang Ma, Huan Yang, Huiguo He, Bei Liu, Jianlong Fu, Nicholas Jing Yuan, Qin Jin, and Baining Guo. Mm-diffusion: Learning multi-modal diffusion models for joint audio and video generation. In Proceedings of

- the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10219–10228, 2023. 2, 4
- [27] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 3
- [28] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 4

- [29] Ivan Skorokhodov, Sergey Tulyakov, and Mohamed Elhoseiny. Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3626–3636, 2022. 2, 4
- [30] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Computer Vision, pages 2256–2265. PMLR, 2015. 2, 4
- [31] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2, 4
- [32] Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021, 2020. 3, 4
- [33] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual description. arXiv preprint arXiv:2210.02399, 2022. 4
- [34] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. arXiv preprint arXiv:2306.02018, 2023. 4, 5
- [35] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. arXiv preprint arXiv:2306.02018, 2023. 7
- [36] Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. Godiva: Generating open-domain videos from natural descriptions. arXiv preprint arXiv:2104.14806, 2021. 4
- [37] Chenfei Wu, Jian Liang, Lei Ji, Fan Yang, Yuejian Fang, Daxin Jiang, and Nan Duan. N¨uwa: Visual synthesis pretraining for neural visual world creation. In European Conference on Computer Vision, pages 720–736. Springer, 2022. 4
- [38] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: Better aligning text-

- to-image models with human preference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2096–2105, 2023. 4
- [39] Yuxin Wu and Kaiming He. Group normalization. In Proceedings of the European conference on computer vision (ECCV), pages 3–19, 2018. 5
- [40] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors.

- 2023. 2, 4

[41] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. arXiv preprint arXiv:2304.05977,

- 2023. 3, 4

- [42] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 4
- [43] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023. 2, 4
- [44] Sihyun Yu, Jihoon Tack, Sangwoo Mo, Hyunsu Kim, Junho Kim, Jung-Woo Ha, and Jinwoo Shin. Generating videos with dynamics-aware implicit generative adversarial networks. arXiv preprint arXiv:2202.10571, 2022. 4
- [45] Sihyun Yu, Kihyuk Sohn, Subin Kim, and Jinwoo Shin. Video probabilistic diffusion models in projected latent space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18456–18466,

2023. 4

- [46] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation, 2023. 4
- [47] Qinsheng Zhang, Molei Tao, and Yongxin Chen. gddim: Generalized denoising diffusion implicit models. arXiv preprint arXiv:2206.05564, 2022. 2, 4
- [48] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023. 2, 3
- [49] Yiyuan Zhang, Kaixiong Gong, Kaipeng Zhang, Hongsheng Li, Yu Qiao, Wanli Ouyang, and Xiangyu Yue. Metatransformer: A unified framework for multimodal learning. arXiv preprint arXiv:2307.10802, 2023. 4
- [50] Yiming Zhang, Zhening Xing, Yanhong Zeng, Youqing Fang, and Kai Chen. Pia: Your personalized image animator via plug-and-play modules in text-to-image models. arXiv preprint arXiv:2312.13964, 2023. 5, 7
- [51] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022. 2, 4

