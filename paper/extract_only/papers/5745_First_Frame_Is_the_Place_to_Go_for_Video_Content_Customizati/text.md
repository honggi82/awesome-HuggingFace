# arXiv:2511.15700v2[cs.CV]23Mar2026

## First Frame Is the Place to Go for Video Content Customization

Jingxi Chen1∗†, Zongxia Li1∗†, Zhichao Liu, Guangyao Shi2, Xiyang Wu1, Fuxiao Liu4, Cornelia Fermüller1, Brandon Y. Feng3†, Yiannis Aloimonos1 1University of Maryland 2USC 3MIT 4NVIDIA http://firstframego.github.io

First Frame Robot Manipulation

|[Figure 1]|
|---|

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

First Frame Aerial View Simulation

|[Figure 6]|
|---|

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

First Frame Multi-product Demonstration

|[Figure 11]|
|---|

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

First Frame Driving Simulation

|[Figure 16]|
|---|

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

First Frame Filmmaking

|[Figure 21]|
|---|

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Figure 1. FFGo is a lightweight add-on that invokes the innate capabilities of pre-trained video generation models, such as Wan2.2 [33], to treat the first frame as a compositional blueprint, enabling natural subject mixing and interaction throughout the video. Given a single input image with multiple elements and a guiding text prompt, FFGo generates coherent, customized videos across diverse applications including robotic manipulation, driving/aerial/underwater simulation, multi-product demonstration, and filmmaking. It requires no architectural modifications, and achieves strong subject-mixing performance with just 20–50 LoRA fine-tuning examples.

### Abstract

later reuse during generation. Leveraging this insight, we show that it’s possible to achieve robust and generalized video content customization in diverse scenarios, using only 20–50 training examples without architectural changes or large-scale finetuning. This unveils a powerful, overlooked capability of video generation models for reference-based video customization.

What role does the first frame play in video generation models? Traditionally, it’s viewed as the spatial-temporal starting point of a video, merely a seed for subsequent animation. In this work, we reveal a fundamentally different perspective: video models implicitly treat the first frame as a conceptual memory buffer that stores visual entities for

*Equal contributions † Corresponding author.

### 1. Introduction

Recent advances in video generation models [1, 2, 4, 19, 24, 28, 33, 38, 43–45] have made them powerful tools for content creation, filmmaking, simulation, and other applications involving creative visual experiences. A key application of these models is reference-based video generation, where one or more reference inputs are used to compose and synthesize visually consistent videos. This capability is essential for real-world scenarios such as film production and simulation, where customization and controllability are critical. Unlike standard Text-to-Video (T2V) models that rely solely on textual prompts, reference-based generation allows users to guide video synthesis through visual references, enabling finer control over generated video contents to be customized for the specific user guidance.

The simplest case is using a single reference image, known as the Image-to-Video (I2V) paradigm. I2V models animate a given image to generate videos that maintain visual consistency with the reference content. While effective as a basic form of video customization, this single-image setup constrains both the spatial diversity and content composition of the generated video. To overcome these limitations, recent research has focused on developing multi-reference video generation models [6, 8, 15, 20, 34], which can incorporate multiple reference images to achieve richer and more flexible video content customization.

Existing multi-reference video generation models generally follow two strategies: 1) Architectural modification of pre-trained video generation models to accommodate additional reference inputs. 2) Fine-tuning on large-scale, task-specific video customization datasets, such as inserting humans into animations or product demonstration videos. However, fine-tuning often leads to performance degradation and loss of generalization in the adapted models. Since video customization typically occurs post-training, the diversity and quality of the adaptation datasets are far lower than those used during pre-training. Consequently, fine-tuned models tend to overfit to specific customization scenarios, forgetting the broad generative priors learned during pre-training, effectively transforming once generalist video models into narrow, task-specialized systems.

Is it possible to incorporate content from multiple reference images into pre-trained video generation models without modifying their architecture or relying on large-scale video training datasets?

To address this question, we investigate existing video generation models and uncover a previously overlooked yet fundamental capability: their innate ability to incorporate multiple reference concepts into the generation process without any architectural modifications or large-scale fine-tuning. As in Figure 2, standard video generation models have potentials to naturally embed visual concepts from multiple references into the first frame, functioning as a memory

buffer, and then fuse them consistently through scene transitions during generation. However, this ability is difficult to invoke directly: prompt engineering alone often leads to unstable outcomes and struggles to precisely preserve object identities or control visual composition. To transform this observation into a practical capability for multi-referencebased video customization, we propose a simple yet effective approach that reliably activates this innate ability. Using only 20-50 training video examples for lightweight training, our method enables the model to select visual concepts in the first frame and transition scenes coherently, achieving generalized multi-reference video customization. Importantly, this is done without modifying the model architecture or compromising the rich generative priors of the pre-trained video generation model.

To summarize our contributions in this work:

- • We investigate the innate and general ability of video generation models, showing that the first frame not only serves as the spatiotemporal start of generation but also acts as a conceptual buffer, which enables multi-reference-based video content customization.
- • We develop a simple yet effective pipeline that leverages Vision-Language Models (VLMs) for high-quality training data curation and achieves invocation through just 20–50 examples via few-shot LoRA adaptation.
- • We evaluate and compare our proposed invocation addon, FFGo, with state-of-the-art video generation models across diverse applications, including filmmaking, generalized multi-object interaction, driving/aerial/underwater simulation, and robotic manipulation, etc.

### 2. Related Work

Video Generation and Video Content Customization. Video generation models [1, 24, 28, 33, 38, 43] are a powerful class of generative models that synthesize videos conditioned on user-provided text prompts, primarily for creative content generation. Most state-of-the-art models are based on diffusion frameworks [5, 27, 30] with U-Netbased denoising backbones, and more recently, Diffusion Transformers (DiT). A key limitation of these pre-trained models is their reliance solely on text prompts, which restricts controllability, particularly in real-world applications such as product demonstrations or simulations, where visual references are often required for precise customization. To address this, recent works in video content customization [6, 8, 11, 14, 15, 20, 37] explore extending video generation models to accept additional visual inputs. These approaches typically require: 1) architectural modifications, and 2) large-scale training on specialized datasets of customized videos. However, both requirements come with significant drawbacks. Architectural changes often compromise model efficiency and compatibility, while task-specific fine-tuning on limited domains may degrade the general

First Frame Veo 3

|[Figure 26]<br><br>[Figure 27]|
|---|

|[Figure 28]<br><br>[Figure 29]|
|---|

[Figure 30]

First Frame Sora 2

|[Figure 31]<br><br>[Figure 32]|
|---|

|[Figure 33]<br><br>[Figure 34]|
|---|

[Figure 35]

First Frame Wan 2.2

|[Figure 36]<br><br>[Figure 37]|
|---|

|[Figure 38]<br><br>[Figure 39]|
|---|

[Figure 40]

Text Prompt <transition> + Text Prompt

- Figure 2. In this figure, we illustrate a general yet under-explored observation: video generation models possess an innate ability to perform subject mixing via scene transitions from a mixed-subject first frame. As shown, the red-boxed results (without the transition phrase: <transition>) contrast with the blue-boxed results (with a carefully chosen <transition> e.g., “The camera view suddenly zoom in to show") revealing significant differences in composition. However, this phenomenon faces three key limitations that hinder practical use: 1) The prompt engineering process for <transition> is highly manual, time-consuming, and model/video-dependent. 2) Scene transitions are often unstable. 3) Object identity is often lost, resulting in changes in appearance or the disappearance of reference objects.

knowledge acquired during large-scale pretraining, where data is more diverse, higher in quality, and broader in scope.

The Innate Abilities of Pre-trained Generative Models Pre-trained generative models are typically trained on massive and diverse datasets, which endows them with generalpurpose capabilities that often extend beyond their original design goals. These emergent properties, what we refer to as innate abilities, remain under-explored in current research, despite growing evidence of their utility in real-world applications. For example, recent work [12] demonstrates that with just a few fine-tuning samples and using Low-Rank Adaptation (LoRA) [10] based fine-tuning, a pre-trained image generation model can be prompted to generate gridaligned images with coherent content. In the video domain, study [38] havs shown that pre-trained image-to-video (I2V) models can perform various frame-level perception tasks such as edge detection, segmentation, and super-resolution, despite not being explicitly trained for them. Motivated by these findings, we explore whether similar innate abilities exist in pre-trained video generation models related to video content customization, and how they might be invoked through minimal adaptation, without architectural changes or large-scale retraining.

Vision-language models (VLMs) for multimodal data curation. Vision-language models (VLMs) have become essential across a wide range of multi-modal applications, demonstrating strong capabilities in video understand-

ing [9, 17, 23], video captioning [16, 41, 46], and visual recognition tasks such as object detection and 3D scene understanding [3, 13, 18, 26, 32]. Recent progress in unified VLMs, e.g., Gemini-2.5-Pro [7], Qwen2.5-Omni [40], GPT4o [13] have further extended these capabilities by enabling seamless processing and generation across images, videos, and text within a shared representation space. These models operate over a unified multimodal input space, capable of processing images, text instructions, and videos, and generating outputs across the same modalities, including images, textual responses, and videos [21, 36, 42, 47]. In particular, unified VLMs demonstrate strong instruction-following capabilities for image editing tasks [39] and video understanding tasks [9], enabling us to leverage them for curating high-quality training data.

### 3. Proposed Approach

#### 3.1. Pipeline Overview

Overview of our pipeline is shown in Figure 3. It consists of three main components: 1) Dataset Curation, which utilizes VLMs to generate high-quality paired training data inputs for corresponding videos. 2) Few-shot LoRA Adaptation, which invokes the model’s innate fusion and transition abilities, and 3) Clean Customized Video Inference, which enables generalized multi-reference video generation.

[Figure 41]

Dataset Curation:

[Figure 42]

[Figure 43]

[Figure 44]

Cake

SAM 2 Element Identification

[Figure 45]

Christmas hat

Man Unified VLM

Training/Inference

Vmix: Video of F frames

[Figure 46]

<transition> + “The video features a person in a Santa hat and festive sweater decorating a

Fc Fg

Ctrans

layered cake. The cake, positioned on a turntable, is a focal point as the person

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

meticulously works on its design.” LoRA

[Figure 53]

|[Figure 54]|
|---|

[Figure 55]

Imix

I2V Model

[Figure 56]

[Figure 57]

- Figure 3. The overview of our proposed pipeline FFGo, consists of 1) Dataset Curation for getting the high quality finetuning data from existing videos, 2) Few-shot LoRA Adaptation for training/inference to invoke the I2V model’s innate ability in fusing the subjects in the first frame and perform a scene trasition to generate a video Vmix following subjects in the first mixing frame Imix and the text prompt.

#### 3.2. Dataset Curation

Previous subject mixing models such as SkyReels-A2 [8], VACE [15] modified model architectures to take in multiple reference images and uses millions of training samples to train models to mix subjects, where the subjects are mainly with humans and does not generalize to other applications such as autonomous driving, robot manipulation. Instead of modifying the models’ architecture and using millions of training data to learn subject mixing, we leverage the models’ pre-trained abilities to learn subject mixing without modifying its architecture. To do so, we need to prepare a dataset where the input aligns with the base model (Wan2.2 [33])’s input, which is an image and a text prompt.

Training Data Selection. We gather two thousand videos from varies sources include Veo 3 showcases [38], HOIGen1M [22] and licensed short videos. Each of the videos is ranging from six to twenty seconds long, we crop all the videos to keep the first 81 frames. Next, we carefully go over a subset of the videos, and pick the high quality ones that have clear interaction or combinations of different elements, with clear and complete boundaries that can be easily segmented. Then for each video we think it is high quality, we write down the elements we want to segment out from the video in text form. For example, in a video where a man wearing a Chrismas hat is making a cake, we write down

the element names the man, cake, Chrismas hat, which will be used to segment out later. We use the first frame of the selected videos to extract elements and process to be the input for the training data. We sample a total of 50 carefully selected videos with human annotations of objects to be segmented out.

Element Extraction using unified VLMs. Given an image I, and a set of elements in textual representation O, our goal is to use a unified VLM to (1) recognize and extract and generate each of the individual O from I, (2) remove all O from I and generate a complete background of I. Specifically, we prompt Gemini-2.5-Pro [7] to do this process. At them end, we use SAM 2 [29] to remove the white backgrounds of the individually extracted or generated elements and keep an RGBA layer for each O, where only the element has RGB layer. Given the set of elements O in RGBA and a background, we combine them into a single image, with the background always on the right and the elements always on the left with automatic resize to fit the elements.

Prompt generation using VLMs. Given a set of RGBA elements, and a ground truth video, we prompt Gemini-2.5Pro to generate a prompt that not only describe the video, but also focuses on the appearance and interaction of these elements. Thus, we get a final training dataset with combined images with their respective background and elements and a

caption that describes how these elements fit into the video and background.

#### 3.3. Few-shot LoRA Adaptation

As shown in Figure 3, with only 20–50 training examples, we apply a simple LoRA-based few-shot adaptation to a pretrained model, enabling it to learn subject mixing by fusing subjects in the first frame and performing a scene transition. Given a pre-trained I2V model gθ, input image I, and text prompt C, the standard generation process is: V = gθ(I,C), where V is a video of F frames. In our adapted pipeline: The input image is replaced by a subject-mixed image Imix. The prompt is modified by appending a special identifier [31] to indicate a transition Ctrans =< transition > +C. A LoRA-based weight update ∆θ is applied to the model. LoRA learns an updated weight matrix θ defined as θ = θ + ∆θ, where θ represents the original pre-trained weights and ∆θ is a learnable low-rank update. Instead of learning a full-rank matrix of size d × k, LoRA factorizes the update as: θ = θ + αAB, where α is a scaling factor, A ∈ Rd×r , B ∈ Rr×k, and r is a low-rank dimension much smaller than both d and k. The target output video Vmix exhibits both subject mixing and a temporal scene transition. Specifically, the video has frame length F = {Fc,Fg} , where: Fc represents the temporal compression frames, namely the temporal compression ratio (e.g., Fc = 4 in the Wan2.2), Fg contains the generated subject-mixed content, following a sudden transition after frames Fc. Thus, the adapted video generation process becomes: Vmix = gθ+∆θ(Imix,Ctrans).

#### 3.4. Clean Customized Video Inference

During the inference, to generate a content customized video, the processing is very simple, since the model will generate the Vmix has frame length F = {Fc,Fg} with both subject mixing and a temporal scene transition. The users can easily cut off the first Fc frames to get the clean subject mixing videos. As an example, in Wan2.2 as the base model, to generate a F = 81 frame video, the first Fc = 4 frames will be discarded and the rest Fg = 77 frames will be the clean customized content videos.

### 4. Experimental Results

The results and comparisons are best viewed through video examples. We include video results in our project webpage.

#### 4.1. Datasets and Implementation Details

Few-shot Training Data. We carefully selected 50 videos featuring object-object, human-human, or human-object interactions with clearly defined boundaries suitable for segmentation. These were curated from a pool of 2,000 videos sampled from HOIGen-1M [22], licensed video clips, and Veo 3 samples [38]. Each curated video contains 81 frames.

The dataset spans four main categories: human-object interaction (60%), human-human interaction (14%), element insertion (20%), and robot manipulation (6%). Details of the curated training set are provided in the Supplementary Materials.

LoRA Adaptation Training Details We use Wan2.2-I2VA14B [33] as our base model. For LoRA adaptation, we train with a LoRA rank of 128 and introduce a unique transition phrase (An arbitrary phrase, as long as it is unique, following a similar intuition to DreamBooth [31]), <transition>: “ad23r2 the camera view suddenly changes.” This phrase serves as a prompt to trigger the model’s innate ability for subject selection and scene transition from the first frame. Since Wan2.2-I2V-A14B employs two separate denoising transformers for low- and high-noise regimes, we train each independently for 5 hours using 2 NVIDIA H200 GPUs, with a batch size of 4.

#### 4.2. Evaluation Strategy

In this section, we outline our evaluation strategy to demonstrate the effectiveness of our proposed model and its performance relative to baseline methods.

Test set spanning diverse application scenarios. To rigorously evaluate the effectiveness and generalization of our approach for video content customization, we curate a test set of 50 reference sets, each composed of materials from different sources, spanning diverse applications such as robot manipulation, filmmaking, aerial/driving/underwater simulation, and product demonstrations. Compared to prior works [8, 15], our test set offers two key advantages: (1) it covers a broader range of real-world customization scenarios beyond the typical human-human or human-object interactions; and (2) it supports up to 5 reference subjects, exceeding the 3-subject limit (e.g., object1, object2, scene) seen in previous works.

Baseline Models We compare our method against three strong baselines built on the Wan architecture with 14 billion parameters: Wan2.2-14B-I2V [33], VACE [15], and SkyReels-A2 [8]. Wan2.2-14B-I2V is our base I2V model, to which we apply our lightweight adaptation for invoking its innate subject mixing and scene transition capabilities. As shown in Figures 5, 6 and 7, our adaptation significantly enhances its performance in video content customization while preserving the original generation quality (Figure 4). VACE and SkyReels-A2 represent state-of-the-art video customization models trained on millions of high-quality examples. Despite their scale, we demonstrate that our method can outperform both using only 50 training videos across a wide range of scenarios.

#### 4.3. Qualitative Comparison

In this section, we qualitatively compare our proposed method with baseline models across diverse scenarios.

Model Overall Quality ↑ Object Identity ↑ Scene Identity ↑ Avg. Rank ↓ % Ranked 1st ↑ Wan2.2-I2V-A14B 2.09 3.32 3.01 3.27 3.4% SkyReels-A2 2.34 2.89 3.43 3.02 4.3% VACE 3.00 3.50 3.66 2.50 11.1% Ours (FFGo) 4.28 4.53 4.58 1.21 81.2%

Table 1. User Study Results: We report ratings and rankings from 200 annotations across 40 users. FFGo consistently outperforms all baseline models across evaluation aspects, despite being trained on only 50 examples, demonstrating significantly better generalization across diverse application scenarios.

Comparison with the Base Model. We first compare our adapted model with the base model (Wan2.2-I2V-A14B) to demonstrate the effectiveness of our approach in video content customization, as shown in Figures 5, 6, and 7. For the base model, we use the mixed image input along with a fixed transition phrase (<transition>) that empirically performs best. As observed, the base model often animates elements independently, and the referred objects tend to disappear post-transition. In contrast, our adapted model consistently preserves object identities and performs coherent scene transitions, indicating a substantial improvement in handling reference-based video customization.

Preserving Pre-trained Knowledge in the Base Model. As demonstrated earlier, our add-on significantly enhances the base model’s performance in reference-based video customization. Crucially, it also preserves the base model’s original pre-trained knowledge. Since our method is designed to invoke the model’s innate capabilities rather than overwrite them, it retains the core generative priors embedded through pre-training. This preservation is illustrated in Figure 4. In the rare successful cases where the base model maintains all object identities and executes a coherent scene transition, our results closely mirror the base output, in this instance, replicating the motion and positioning of the wingsuit performer and the car. This fidelity underscores an important advantage: our method enables customization without compromising the valuable general knowledge encoded during pre-training. Given that post-training data for customization is often narrower in scope and lower in quality than pre-training corpora, it is critical for video content customization approaches to integrate new reference inputs while preserving the strengths of the base model. Our add-on achieves exactly this, offering a lightweight yet effective pathway to transform generalpurpose I2V models into powerful, user-controllable video customization systems.

Comparison with State-of-the-Art Video Customization Baselines. We compare our method with two Wan-based state-of-the-art baselines: VACE and SkyReels-A2. 1) As shown in Figures 5 and 7, both baselines are trained on millions of videos for specific customization tasks involving the composition of three elements: human, object, and scene. However, this design leads to overfitting, limiting their generalization to novel application scenarios. In con-

Text Prompt References

[Figure 58]

[Figure 59]

A high-angle, third-person aerial tracking shot

[Figure 60]

follows a silver Tesla Cybertruck as it drives along

a winding, S-curved asphalt road through a dense, green forest. Flying in perfect formation

directly above the vehicle is a person wearing a

blue and black wingsuit. The camera maintains its high vantage point, capturing both the Cybertruck

navigating the curves of the road below and the

wingsuit flyer mirroring its exact path in the air.

Wan2.2-I2V-A14B

[Figure 61]

[Figure 62]

[Figure 63]

Ours

[Figure 64]

[Figure 65]

[Figure 66]

Figure 4. As shown in the figure, in rare cases where the base model Wan2.2-I2V-A14B successfully performs a scene transition while preserving all reference object identities, the output closely resembles ours. This demonstrates that our add-on approach effectively retains the base model’s pre-trained generative capabilities.

trast, our method, trained on just 50 examples, activates the pre-trained base model’s innate capabilities while preserving its general knowledge, enabling superior performance across diverse use cases. 2) Figure 6 highlights a key limitation of these baselines: their architecture only supports up to three reference inputs. As a result, they fail to handle scenarios with five references. In contrast, our approach has no such architectural constraint, as our multi-reference capability stems from the the utilization of the first frame as conceptual memory buffer, not architectural modification.

#### 4.4. Quantitative Comparison

To quantitatively evaluate our method and compare it with baselines, we conducted a user study on the full test set across all methods. Details of the user study setup are provided in the Supplementary Materials. We built an online interface where users are shown the prompt and a reference image containing all foreground objects and background context. The system then displays four generated videos in randomized order. Users were asked to: 1) Rank the four videos from best (1st place) to worst (4th place). 2) Rate each video on three aspects: Object Identity: with a score range from 1 (worst) to 5 (best), How well are foreground object identities preserved from the reference image?. Scene Identity: with a score range from 1 (worst) to 5 (best), How accurately is the background retained? and Overall Quality: with a score range from 1 (worst) to 5 (best),

Wan2.2-I2V-A14B

Text Prompt

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Professional-quality video with rich details. The

video features a charming Teddy Bear sipping

apple juice from a bottle using its hand, while delicately holds a vibrant red rose using its

hand, admiring its beauty, perhaps as an offering or a gesture of affection.

References

[Figure 71]

SkyReels-A2

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

VACE

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Ours

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

- Figure 5. Qualitative comparison with baseline methods. This test scenario involves generalized multi-object interactions. As shown in the figure, our method best preserves the identities of input objects and the scene, while generating a customized video with coherent motion that aligns with the text prompt description.

Wan2.2-I2V-A14B

A video scene unfolds in a vast, golden wheat

field under a blue sky, with a red smoke flare billowing from a supply crate in the mid-ground.

In the foreground, the character Wukong, in his ornate armor, stands next to a soldier wearing a

helmet and tactical gear. The soldier holds a blue iPhone, gesturing with it, while Wukong holds a VR headset. The two characters are positioned side-by-side, appearing to be in an

animated discussion, comparing and talking about the two different technology products

they are holding. SkyReels-A2

VACE

Ours

References

Text Prompt

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

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

- Figure 6. Qualitative comparison with baseline methods. This scenario evaluates performance with an excessive number of references, five in total (four objects and one scene). VACE and SkyReels-A2, due to their architecture-based limitations, support only up to three references and fail to include all four reference objects in the generated video. In contrast, our model successfully fuses all four objects into a coherent, customized video with natural interactions. Notably, our model also enables precise selection via text prompt (e.g., blue iPhone), preserving key visual traits such as the triple-camera design while modifying appearance (e.g., changing the color to blue).

Wan2.2-I2V-A14B

Text Prompt

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

A man wearing a short-sleeved shirt gently

hands over a toy rocket to another man dressed in a black blazer over a dark graphic

T-shirt. The man dressed in a black blazer looks at the rocket with mild curiosity and intrigue.

References

[Figure 111]

SkyReels-A2

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

VACE

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Ours

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

- Figure 7. Qualitative comparison with baseline methods. This scenario evaluates generalized human-object interactions involving multiple humans. While both VACE and SkyReels-A2 excel in customized single human-object video generation, they struggle in more complex multi-human scenarios where interactions are mediated by shared objects. In such cases, both baselines fail to maintain object integrity and coherent interaction. In contrast, our method reliably generates consistent videos with preserved object identity and realistic multi-human interactions.

How realistic and coherent is the video overall?. Table 1 presents the results. Our method (FFGo) outperformed all baselines across every metric, rank, object identity, scene identity, and overall quality, despite using only a lightweight adaptation. Notably, over 80% of users selected our results as their top choice, indicating strong alignment with real user preferences. Importantly, FFGo transforms the base model Wan2.2-I2V-A14B from the lowest-performing baseline to the top performer in user evaluations. This highlights the strength of our add-on approach, which achieves state-of-theart customization performance without architectural changes or large-scale training.

### 5. Limitations

While our adaptation effectively invokes the innate ability of pre-trained I2V models for video content customization via the first frame, several limitations remain. Although it is theoretically possible to incorporate an arbitrary number of reference subjects in the first frame, in practice, increasing the number of subjects reduces the resolution available to each, making identity preservation more difficult. Another challenge is selective control: as the number of reference subjects grows, it becomes harder to reliably target specific objects using only text prompts. Empirically, we find our method performs well up to four subjects plus a reference

scene (five references in total), beyond which identity preservation and prompt-based selection degrade. We believe these limitations are not fundamental and can be addressed through engineering improvements. For example, using multiple start frames as an extended conceptual memory buffer could allow for higher-capacity reference encoding. We leave such enhancements to future work.

### 6. Conclusions

In this work, we propose a fundamentally different perspective on the role of the first frame in video generation models. Contrary to the standard view that treats it merely as the spatiotemporal starting point of an animation, we show that the first frame functions as a conceptual memory buffer, capable of storing and fusing disjoint reference subjects for downstream generation. Building on this insight, we introduce a lightweight, add-on method to invoke this overlooked innate ability for video content customization. Without modifying the model architecture or requiring large-scale finetuning, our few-shot adaptation turns a base video generation model into a state-of-the-art video customization system. We demonstrate strong performance across a wide range of real-world scenarios and validate our approach through a comprehensive user study, showing clear alignment with user preferences.

Acknowledgment: We greatly acknowledge NSF’s support under awards OISE 2020624 and BCS 2318255.

### References

- [1] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2
- [2] Ryan Burgert, Yuancheng Xu, Wenqi Xian, Oliver Pilarski, Pascal Clausen, Mingming He, Li Ma, Yitong Deng, Lingxiao Li, Mohsen Mousavi, et al. Go-with-the-flow: Motioncontrollable video diffusion models using real-time warped noise. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13–23, 2025. 2
- [3] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024. 3
- [4] Jingxi Chen, Brandon Y Feng, Haoming Cai, Tianfu Wang, Levi Burner, Dehao Yuan, Cornelia Fermuller, Christopher A Metzler, and Yiannis Aloimonos. Repurposing pre-trained video diffusion models for event-based video interpolation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12456–12466, 2025. 2
- [5] Jingxi Chen, Yixiao Zhang, Xiaoye Qian, Zongxia Li, Cornelia Fermuller, Caren Chen, and Yiannis Aloimonos. From inpainting to layer decomposition: Repurposing generative inpainting models for image layer decomposition. arXiv preprint arXiv:2511.20996, 2025. 2
- [6] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Kwot Sin Lee, Ivan Skorokhodov, Kfir Aberman, Jun-Yan Zhu, Ming-Hsuan Yang, and Sergey Tulyakov. Multi-subject open-set personalization in video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6099–6110, 2025. 2
- [7] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 3, 4
- [8] Zhengcong Fei, Debang Li, Di Qiu, Jiahua Wang, Yikun Dou, Rui Wang, Jingtao Xu, Mingyuan Fan, Guibin Chen, Yang Li, et al. Skyreels-a2: Compose anything in video diffusion transformers. arXiv preprint arXiv:2504.02436, 2025. 2, 4, 5
- [9] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025. 3
- [10] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3,

2022. 3

- [11] Chi-Pin Huang, Yen-Siang Wu, Hung-Kai Chung, Kai-Po Chang, Fu-En Yang, and Yu-Chiang Frank Wang. Videomage:

- Multi-subject and motion customization of text-to-video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17603–17612, 2025. 2
- [12] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024. 3
- [13] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 3
- [14] Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si, Dahua Lin, Yu Qiao, Chen Change Loy, and Ziwei Liu. Videobooth: Diffusion-based video generation with image prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6689–6700, 2024. 2
- [15] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17191–17202, 2025. 2, 4, 5
- [16] Zongxia Li, Xiyang Wu, Hongyang Du, Fuxiao Liu, Huy Nghiem, and Guangyao Shi. A survey of state of the art large vision language models: Benchmark evaluations and challenges. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 1587–1606, 2025. 3
- [17] Zongxia Li, Xiyang Wu, Guangyao Shi, Yubin Qin, Hongyang Du, Tianyi Zhou, Dinesh Manocha, and Jordan Lee Boyd-Graber. Videohallu: Evaluating and mitigating multimodal hallucinations on synthetic video understanding. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 3
- [18] Zongxia Li, Wenhao Yu, Chengsong Huang, Rui Liu, Zhenwen Liang, Fuxiao Liu, Jingxi Che, Dian Yu, Jordan Boyd-Graber, Haitao Mi, et al. Self-rewarding visionlanguage model via reasoning decomposition. arXiv preprint arXiv:2508.19652, 2025. 3
- [19] Zongxia Li, Hongyang Du, Dawei Liu, Xiyang Wu, Lantao Yu, Jingxi Chen, Fuxiao Liu, Xiaomin Wu, Jing Xie, Chengsong Huang, Yicheng He, and Guangyao Shi. Multimodal video generation models with audio: Present and future. https://doi.org/10.13140/RG.2.2.26531.31528, 2026. 2
- [20] Tingting Liao, Chongjian Ge, Guangyi Liu, Hao Li, and Yi Zhou. Character mixing for video generation. arXiv preprint arXiv:2510.05093, 2025. 2
- [21] Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Mitigating hallucination in large multi-modal models via robust instruction tuning. arXiv preprint arXiv:2306.14565, 2023. 3
- [22] Kun Liu, Qi Liu, Xinchen Liu, Jie Li, Yongdong Zhang, Jiebo Luo, Xiaodong He, and Wu Liu. Hoigen-1m: A large-scale dataset for human-object interaction video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24001–24010, 2025. 4, 5, 12
- [23] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He,

- Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 3
- [24] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, et al. Sora: A review on background, technology, limitations, and opportunities of large vision models. arXiv preprint arXiv:2402.17177, 2024. 2
- [25] Ilya Loshchilov, Frank Hutter, et al. Fixing weight decay regularization in adam. arXiv preprint arXiv:1711.05101, 5

(5):5, 2017. 15

- [26] Anish Madan, Neehar Peri, Shu Kong, and Deva Ramanan. Revisiting few-shot object detection with vision-language models. Advances in Neural Information Processing Systems, 37:19547–19560, 2024. 3
- [27] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

- 2023. 2

[28] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

- 2024. 2

- [29] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714,

2024. 4

- [30] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [31] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500–22510,

2023. 5, 16

- [32] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, Yilin Zhao, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, et al. Eagle: Exploring the design space for multimodal llms with mixture of encoders. arXiv preprint arXiv:2408.15998, 2024. 3
- [33] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

- 1, 2, 4, 5

[34] Lizhen Wang, Zhurong Xia, Tianshu Hu, Pengrui Wang, Pengfei Wei, Zerong Zheng, Ming Zhou, Yuan Zhang, and Mingyuan Gao. Dreamactor-h1: High-fidelity human-product demonstration video generation via motion-designed diffusion transformers. arXiv preprint arXiv:2506.10568, 2025.

- 2

- [35] Xierui Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multi-subject zero-shot im-

age personalization with layout guidance. arXiv preprint arXiv:2406.07209, 2024. 12

- [36] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 3
- [37] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6537–6549, 2024. 2
- [38] Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025. 2, 3, 4, 5, 12
- [39] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 3
- [40] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215, 2025. 3
- [41] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10714–10726, 2023. 3
- [42] Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305, 2023. 3
- [43] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2
- [44] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyang Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identitypreserving text-to-video generation by frequency decomposition. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12978–12988, 2025.
- [45] Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and William T Freeman. Physdreamer: Physics-based interaction with 3d objects via video generation. In European Conference on Computer Vision, pages 388–406. Springer, 2024. 2
- [46] Yue Zhao, Long Zhao, Xingyi Zhou, Jialin Wu, Chun-Te Chu, Hui Miao, Florian Schroff, Hartwig Adam, Ting Liu, Boqing Gong, et al. Distilling vision-language models on millions of videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13106–13116,

2024. 3

- [47] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma,

###### Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 3

## First Frame Is the Place to Go for Video Content Customization Supplementary Material

### Contents

- 1. Introduction 2
- 2. Related Work 2
- 3. Proposed Approach 3

- 3.1. Pipeline Overview . . . . . . . . . . . . . . 3
- 3.2. Dataset Curation . . . . . . . . . . . . . . . 4
- 3.3. Few-shot LoRA Adaptation . . . . . . . . . 5
- 3.4. Clean Customized Video Inference . . . . . 5

- 4. Experimental Results 5

- 4.1. Datasets and Implementation Details . . . . 5
- 4.2. Evaluation Strategy . . . . . . . . . . . . . . 5
- 4.3. Qualitative Comparison . . . . . . . . . . . 5
- 4.4. Quantitative Comparison . . . . . . . . . . . 6

- 5. Limitations 8
- 6. Conclusions 8

- A. Video Results 12
- B. Comparison with Two-Stage Baselines 12
- C. Details about Training and Testing Set 12

- C.1. Training Dataset Curation Details . . . . . . 12
- C.2. Test Set Curation . . . . . . . . . . . . . . . 13

- D. Details about User Study 13

- D.1. User Study Platform . . . . . . . . . . . . . 13
- D.2. User Interface Details . . . . . . . . . . . . 14

- E. More Training and Inference Details 15
- F. Generalization to First-Frame Layouts 15
- G. Visual Consistency Across Generated Videos from Different Reference Sources 15
- H. Automatic Quantitative Metrics 15
- I. Explanation of the Transition Phrase 16

### A. Video Results

Please refer to our project page: http : / / firstframego . github . io for video results, which clearly demonstrate the effectiveness of our method and its comparison with baseline models.

Ours: The shark joins the party later, as directed by the text prompt

First Frame (Ours)

|[Figure 126]|
|---|

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

First Frame (MS-Diffusion) MS-Diffusion+I2V: Loses temporal composition controllability in the generated video

|[Figure 131]|
|---|

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Figure 8. Comparison with MS-Diffusion + I2V.

### B. Comparison with Two-Stage Baselines

Fig. 8 shows a comparison with a representative two-stage baseline: First composing an image layout by an image composition model like MS-Diffusion [35] and then using an I2V model for animation. Our approach offers two key advantages: 1) Our method is fully end-to-end; 2) Temporal Control via Text Prompts: More importantly, our method preserves fine-grained temporal control. For example, prompts like “a shark joins the party later” are faithfully realized. In contrast, two-stage pipelines lose this capability, as the fixed first frame dictates the entire spatial layout, limiting temporal flexibility.

### C. Details about Training and Testing Set C.1. Training Dataset Curation Details

Our training corpus is sourced from three datasets: one randomly selected folder from HOIGen-1M [22] (≈ 2,000 clips), all five Veo 3 demonstration videos [38], and 200 licensed short videos. This yields 2,205 candidate clips.

We manually curate the data to select videos with clearly separable foregrounds, humans or manipulable objects, set against uncluttered backgrounds. Only clips depicting cleanly segmentable single- or multi-object interactions are retained.

This filtering results in 50 high-quality training examples (Figure 11), distributed across four scene types: human–object interaction (60%), human–human interaction (14%), element insertion (20%), and robot manipulation (6%).

Training Data Processing. After curation, all clips are standardized to 81 frames for consistent training. From each video, we extract the first frame as a reference image and manually tag all foreground entities of interest, e.g., cake, party hat, male presenter, mouse. Using a prompt-to-prompt workflow with Gemini-2.5-Pro, we then perform:

• Object Extraction: Apply Prompt 9 to generate highfidelity renditions of each tagged entity, preserving their original appearance and scale. We refine results using

SAM 2 or Adobe Photoshop to isolate each object as an RGBA layer.

• Background Cleanup: Use Prompt 10 to produce a clean companion image with all tagged objects removed, yielding a pristine background plate.

This paired set of object cut-outs and object-free backgrounds forms the compositional basis of the training first frame.

Caption Generation. We use Gemini-2.5-Pro to generate rich, element-aware captions for each training sample, based on the individual object cut-outs, clean background plate, and the full 81-frame video. These inputs are paired with a structured prompt template (Fig. 12) to ensure consistency and relevance.

Element Composition for First Frame. For each training clip, we synthesize a 1280×720 reference canvas: all foreground cut-outs are vertically tiled on the left half, while the clean background is centered on the right (see Fig. 11). This composite serves as both the conditioning input and the initial frame, guiding the video generation model to blend the elements into a coherent sequence.

#### C.2. Test Set Curation

We manually curated a diverse test set of foreground objects and backgrounds from our self-collected images. Each object was segmented using SAM 2 or Adobe Photoshop and saved as an RGBA cut-out. These cut-outs were then composited with their respective backgrounds on a 1280×720 canvas, following the same layout used in training.

For each object-background pair, we drafted an initial prompt and refined it using Gemini-2.5-Pro with the template shown in Fig. 13. This process produced 50 high-quality prompts paired with composite reference images, forming our final test set.

### Object Extraction Task Prompt Template

Prompt – Given the input image, extract the subset

{IDENTIFIED OBJECT} (i.e., only the specified foreground objects)— return them alone with no resizing, compression, or background so the output resolution exactly matches the original image.

Figure 9. Prompt for extracting identified foreground objects using a unified VLM.

### D. Details about User Study

To ensure a smooth user study and annotation experience, we developed an HTML-based interface for participants

### Object Removal Task Prompt Template

Prompt – Given the input image, remove the subset {IDENTIFIED OBJECTS} entirely. Return the edited image only—it must preserve the source resolution (no scaling or compression) and contain neither the specified objects nor any artifacts of their removal.

- Figure 10. Prompt and specifications for the object removal task.

[Figure 136]

- Figure 11. Our training dataset comprises four categories: human–object interaction (60%), human–human interaction (14%), element insertion (20%), and robot manipulation (6%).

to annotate and submit data. In this section, we describe the hiring platform, the job posting, and the design of the annotation interface.

#### D.1. User Study Platform

We recruit participants through Prolific,1 a research platform designed for user studies. Prolific offers an AI user study beta program that targets participants with experience in generative AI annotation.

To ensure quality, we apply screening filters to select

1https://www.prolific.com/

## Training Data Prompt Generation Prompt Template

##### Task Description

You are given a video and several images. Generate a descriptive caption for the video that prominently features the components shown in the images. Wrap your final text in <caption>...</caption> tags. The caption must highlight the significance and role of these components throughout the video, while omitting filler such as “The scene unfolds with a whimsical and heartwarming narrative, emphasizing the simple joys of life through the Teddy Bear’s endearing actions”.

##### Examples of Descriptive Captions

- 1. Film quality, professional quality, rich details. The video begins to show the surface of a pond, and the camera slowly zooms in to a close-up. The water surface begins to bubble, and then a blonde woman is seen coming out of the lotus pond soaked all over, showing the subtle changes in her facial expression.
- 2. A professional male diver performs an elegant diving maneuver from a high platform. Full-body side view captures him wearing bright red swim trunks in an upside-down posture with arms fully extended and legs straight and pressed together. The camera pans downward as he dives into the water below.

Figure 12. Prompt template used to generate captions for our training data.

## Video-Prompt Enhancement Output

##### Task Description

You will be given a prompt and several images for video generation. You task is to make the prompt richer in description so the model can understand better. Enclose your caption within <caption></caption> tags. The caption must emphasize the significance and role of these components (and some description of each component) throughout the video. Your caption should exclude unnecessary information such as “The scene unfolds with a whimsical and heartwarming narrative, emphasizing the simple joys of life through the Teddy Bear’s endearing actions”.

##### Example of a Descriptive Caption

- 1. Film quality, professional quality, rich details. The video begins to show the surface of a pond, and the camera slowly zooms in to a close-up. The water surface begins to bubble, and then a blonde woman is seen coming out of the lotus pond soaked all over, showing the subtle changes in her facial expression.
- 2. A professional male diver performs an elegant diving maneuver from a high platform. Full-body side view captures him wearing bright red swim trunks in an upside-down posture with arms fully extended and legs straight and pressed together. The camera pans downward as he dives into the water below.

Prompt to Optimize {Insert your test prompt to optimize here}

Figure 13. Prompt template for test prompt enhancement.

participants with prior video annotation experience and fluent English proficiency, as understanding nuanced textual prompts is crucial for this task.

We hire 40 participants, each tasked with annotating five video sets, where each set contains generated outputs from four different models. The annotation process takes approximately 15 minutes per participant. Each is compensated $5.50, reflecting the expected time and effort.

Our recruitment post and task instructions are shown in

- Figure 14.

#### D.2. User Interface Details

Participants first arrive at a login screen, where they enter their unique Prolific ID to match their responses with task-completion records. After authentication, they are presented with the textual prompt used to generate the videos, along with a composite reference image showing the required foreground objects and background. Below, four candidate videos are displayed in a randomized order to eliminate position bias. Participants then rank the videos based on overall quality, as shown in Figure 15a.

Next, participants scroll down to rate each video on three

### Annotation Task Instructions

You will be presented with five sets of short, AI-generated videos (5 s, no audio). Each set contains:

- • Prompt – textual description of the intended video (scene, objects, motion).
- • Reference Image – split into two halves:

- – Left side: foreground objects that should appear in the video.
- – Right side: background scene to be integrated with the objects.

- • Generated Videos (4 total) – four model outputs attempting to fuse the objects with the background. Your Task for Each Set

- Step 1: Overall Ranking

- • Watch all four videos carefully.
- • Rank them from best to worst based on overall quality and faithfulness to the prompt.
- • Assign unique ranks (1 = best, 4 = worst).

- Step 2: Aspect Ratings After ranking, rate each video on a 1–5 scale (1 = very poor, 5 = excellent):

- • Object Identity – How well do objects retain their identity?
- • Scene / Background Identity – How well is the background preserved?
- • Video Quality – Overall realism and temporal coherence. Notes
- • Evaluate all four videos in every set before submitting answers.
- • There are five sets in total (20 videos).

Figure 14. Recruitment post for our user study.

criteria, Object Identity, Scene Identity, and Overall Quality, using a 5-point Likert scale (Figure 15b).

### E. More Training and Inference Details

We train LoRA modules of rank 128 for both high- and lownoise regime transformers in the base model Wan2.2-I2VA14B. Training videos are resized to a resolution of 1344 × 768 with 81 frames. We use a batch size of 4 and optimize with AdamW [25], setting the learning rate to 1 × 10−4, ϵ = 1 × 10−10, and a weight decay of 3 × 10−2.

During inference, videos are generated at a resolution of 1280 × 720 with 81 frames, following the standard output format of Wan2.2-I2V-A14B based models.

### F. Generalization to First-Frame Layouts

Although training uses a fixed first-frame layout (a), our model can generalize to unseen layouts in some cases. As shown in Fig. 16, we evaluate three novel layouts (b), (c), and (d), beyond the training layout of cut-outs on the left and background on the right. The results suggest that our model interprets the first frame contextually rather than relying solely on the seen training layout.

Model CLIP-I↑ DINO-I↑ CLIP-cap↑ Motion Smoothness ↑ Dynamic Degree↑ Aesthetic Quality↑ Imaging Quality↑

Wan2.2-I2V-A14B 0.66 0.42 33.2 0.96 9.75 0.82 0.61 VACE 0.68 0.46 33.6 0.97 7.78 0.92 0.65 SkyReels-A2 0.66 0.43 33.1 0.96 12.93 0.91 0.72 Ours 0.67 0.46 34.00 0.98 14.64 0.85 0.73

Table 2. Concept and VBench Automatic Quantitative Metrics

### G. Visual Consistency Across Generated Videos from Different Reference Sources

For all test results presented in the paper, the reference inputs are drawn from different sources rather than a single video. Visual consistency is maintained in the generated videos due to the pre-trained models’ learned priors. For instance, in Fig. 17, fine-grained shadows cast by a hand and bottle are correctly rendered on the teddy bear (shown by arrows).

### H. Automatic Quantitative Metrics

In Table 2, we show standard automatic concept and VBench quantitative metrics: CLIP-I, DINO-I, CLIP-cap (text alignment), Motion Smoothness, Dynamic Degree (motion intensity), Aesthetic Quality, and Imaging Quality, to compare with baselines. These standard metrics further validate the effectiveness of our method.

[Figure 137]

[Figure 138]

(a) Users rank the overall quality of the four candidate videos. (b) Users rate three specific aspects with a Likert scale 1-5.

- Figure 15. Web-based annotation interface used in our user study. Part (a) collects a global quality ranking, while part (b) gathers detailed aspect-wise ratings for each video.

(a) Same First Frame Layout as in Training

|[Figure 139]|
|---|

[Figure 140]

[Figure 141]

[Figure 142]

First Frame Layout

[Figure 143]

|[Figure 144]|
|---|

[Figure 145]

(b) Unseen First Frame Layout: Swapping Left and Right

[Figure 146]

[Figure 147]

[Figure 148]

|[Figure 149]|
|---|

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

(c) Unseen First Frame Layout: Random Order of References

|[Figure 154]|
|---|

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

(d) Unseen First Frame Layout: Overlapping of References

- Figure 16. Generalization to spatial layouts in the first frame.

References from Different Sources Generated Video with Consistent Visual Effects

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Figure 17. References from different sources.

### I. Explanation of the Transition Phrase

The transition phrase <transition> (e.g., “ad23r2 the camera view suddenly changes”) serves as a unique trigger in the text prompt. Paired with LoRA training, it enables the base model to learn to invoke latent abilities for scene cuts and reference fusion when encountered during inference. The choice of trigger can be arbitrary, as long as it is unique. This design is inspired by the use of unique trigger phrases in DreamBooth [31], but serves a fundamentally different purpose.

