# arXiv:2503.20822v1[eess.IV]26Mar2025

## Synthetic Video Enhances Physical Fidelity in Video Synthesis

Qi Zhao1 Xingyu Ni2,1 Ziyu Wang3,1 Feng Cheng1 Ziyan Yang1 Lu Jiang1* Bohan Wang4*

1ByteDance Seed 2Peking University 3ShanghaiTech University 4National University of Singapore

[Figure 1]

Figure 1. Our synthetic-data-enhanced video generation model is capable of producing videos depicting human dancing (rows 1), scenes featuring large camera orbiting around the object (row 2), and animals against solid-color backgrounds for matting (row 3).

### Abstract

We investigate how to enhance the physical fidelity of video generation models by leveraging synthetic videos derived from computer graphics pipelines. These rendered videos respect real-world physics, such as maintaining 3D consistency, and serve as a valuable resource that can potentially improve video generation models. To harness this potential, we propose a solution that curates and integrates synthetic data while introducing a method to transfer its physical realism to the model, significantly reducing unwanted artifacts. Through experiments on three representative tasks emphasizing physical consistency, we demonstrate its efficacy in enhancing physical fidelity. While our model still lacks a deep understanding of physics, our work offers one of the first empirical demonstrations that synthetic video enhances physical fidelity in video synthesis. Website: https://kevinz8866.github.io/simulation/

*Corresponding author

### 1. Introduction

Video generation models [9, 17, 25, 35, 36] have demonstrated strong capabilities in producing high-quality and visually compelling videos of real-world scenarios. Despite their remarkable progress, these generation videos often struggle to respect the underlying physical laws of the real world, indicating a significant gap in applications where physical fidelity is essential [30, 65, 66]. For instance, while a video generation model can generate realistic-looking objects or humans within a scene, it may fail to maintain 3D consistency when the camera moves or when the subjects undergo deformation.

In this paper, we explore whether synthetically generated videos can enhance the physical fidelity of video generation models. Specifically, we utilize synthetic videos rendered through modern computer-generated imagery (CGI) production pipelines used in gaming and film, such as Blender [59] and Unreal Engine [19]. By utilizing standard computer graphics techniques, we can generate highquality, physically consistent video content at scale. CGI production pipelines generate videos via precise 3D asset modeling, animation, and rendering based on predetermined physical rules [13]. This approach allows for highly

accurate scene configuration and ensures that the rendered videos intrinsically respect real-world physics, provided the setups and parameters are properly specified. As such, synthetic video is highly configurable, allowing precise control over scene setup, objects, and motion. Additionally, ground-truth descriptions can be easily obtained based on the specifications of the 3D environment.

However, training video generation models using synthetic video data presents several challenges. Synthetic videos inherit an appearance gap, making them easily distinguishable from real videos. Further, the limited availability of 3D assets, together with the complexity of their composition, restricts the diversity of synthetic video content. As a result, leveraging synthetic video to enhance model understanding remains an active area of research [32, 39, 40]. Regarding video generation, to the best of our knowledge, no prior work has specifically explored the use of synthetic videos to enhance video generation models.

Therefore, we present an investigation into how synthetic video enhances the physical fidelity of video generation models. As a pilot study, we examine three representative tasks known to be challenging even for state-of-the-art video generation models. Figure 1 illustrates their generated videos which include: 1) Large human motion generation, where significant movements cause noticeable shape deformations in body parts, such as breakdance or backflip. 2) Wide-angle camera rotation, where the camera spins around a specific axis, capturing a broader field of view of the object or actions. 3) Video layer decomposition, where the model must generate a subject or motion against a green screen background. This task evaluates whether the model can effectively disentangle the subject from the background during generation. These tasks are not exhaustive but serve as a reasonable starting point for studying physical fidelity in video generation.

We propose a solution that uses synthetic videos to enhance video generation models. At the data level, based on computer graphics techniques, we begin by constructing a synthetic video generation pipeline that offers diverse scene configurations, assets, and animations. Next, we explore the curation and integration of synthetic videos to transfer their physical fidelity to the video generation model. Through extensive analysis and ablations, we identify key factors that govern how well synthetic videos transfer physical fidelity to real-world video generation, including visual distribution, asset quality, rendering quality, the role of synthetic captions and the best blending strategy of synthetic videos with their real counterparts.

At the model level, we propose a novel approach SimDrop to reduce the introduction of undesirable rendering artifacts into the final generation model by training a synthetic reference model that solely captures the visual patterns of synthetic video data. We show that with classifier-

free guidance [27], the reference model can work in auxiliary with the generation model to remove the visual artifacts from synthetic data but keeps the physical fidelity.

To verify the effectiveness of our solution, we employ two measurements inspired by related works [1, 31], assessing fidelity in terms of 3D consistency and human pose integrity. While these measurements are not perfect, they offer meaningful indicators of the physical fidelity of video generation. Additionally, human evaluations are incorporated to ensure alignment with human perception. Our experiments demonstrates that by carefully crafting and integrating synthetic video data, video generation models can significantly reduce collapse and distortion in human motion and improve 3D consistency [1] of objects under large camera movements. Moreover, our approach enables models to generate backgrounds of uniform color while maintaining clearly separated, dynamically moving objects in the foreground. It is worth noting that while our model improves physical fidelity, it still lacks an understanding of the underlying principles of physics, leaving significant room for further improvement

In summary, we make the following contributions:

- • We present a computer graphics-based synthesis pipeline to generate videos for training video generation models.
- • We identify key factors in curating synthetic video data and propose strategies for effectively training video generation models on these datasets.
- • To the best of our knowledge, our work provides one of the first empirical demonstrations that incorporating synthetic video data can improve the physical fidelity of video generation models.

### 2. Synthetic Video Generation using Computer Graphics Techniques

Augmenting datasets with synthetic data has been widely adopted in the field of machine learning. Specifically, standard CGI production pipelines, such as those implemented in Blender [59] or Unreal Engine [19], have long been employed to synthesize highly controlled and visually realistic image and video data. By explicitly modeling objects, cameras, environments, and illumination, they offer fine-grained control over every aspect of a scene, enabling the generation of large-scale, diverse, and visually realistic video datasets.

Our data synthesis pipeline is built on such a CGI production pipeline. We build a procedural 3D scene generator driven by a carefully chosen set of parameters, enabling diverse 3D scene generation. Then, we couple it with the open-source rendering engines Unreal Engine and Blender to generate high-quality video outputs. Based on the three aforementioned challenging tasks, we focus on generating videos containing a single object per scene and aim to maximize diversity in both appearance and motion. Following

[Figure 2]

- Figure 2. Visualization of the pipeline to augment video generation model with synthetic video data. We first plan the synthetic videos and generation descriptive tags for each elements (e.g. object, character, motion, etc). Then we combine the element descriptions to form the caption for synthetic videos. During training, we mix the synthetic videos with real-world video data to improve physics fidelity in challenging video generation tasks.

standard practice, we consider a 3D scene to include four key components: (1) the 3D object, (2) the camera, (3) the lighting conditions, and (4) the environment. Each component is fully customizable through a set of predefined parameters, as detailed in Appendix A.1. Then, our pipeline automatically converts the parameters into a 3D scene and renders them into a video. Next, we will explain how we effectively sample the parameter space to achieve our goal.

### 3. Method

Our goal is to investigate how data augmentation with synthetic videos can enhance a video generation model to produce physically consistent videos. As a pilot study, this paper focuses on three specific generation tasks, each representing a challenging generation task even for stateof-the-art video generation models: large human motion, wide-angle camera rotation, and video layer decomposition. We assess quality primarily based on physical fidelity (see Sec. 3.4 for metric definition), rather than the commonly used visual fidelity or aesthetics.

Training video generation models with synthetic video data presents challenges due to the distributional gap between synthetic and real videos. Our method addresses the gap between synthetic and real videos through three key techniques: data curation, a captioning strategy, and a novel training approach. Figure 2 provides an overview of our method, illustrating how these components work together to enhance video generation. In the following, Section 3.1 presents the curation of the synthetic video pixels. Section 3.2 explains how we caption the synthetic videos. Lastly, Section 3.3 details our strategy and method to incorporate the synthetic data.

Training Data Human Motion Collapse Rate

- (a) Random 87%

- (b) Forward shot only 42%

- (c) Forward + following shot 23%

Table 1. Randomly chosen camera configurations (a-b) lead to high collapse rate for generated videos. Using configuration (c) aligning with the real world greatly reduce the rate.

#### 3.1. Curating Synthetic Pixels

This section explores strategies for narrowing the gap between real and synthetic videos by refining synthesis configurations – including camera, background, object, lighting, and other visual factors – as well as a study examining the impact of visual appearance brought by asset quality and rendering quality.

Synthesis configurations Our generation tasks require producing videos that maintain 3D consistency for objects and ensure body coherence in human motion. To achieve this, we synthesize videos that emphasize these aspects by incorporating large object deformations (e.g., human dance) and significant camera rotations (e.g., orbiting around objects). Additionally, it is beneficial to incorporate characteristics of real videos such as common camera setups. For instance, professional videographers often capture a subject’s upper body from frontal angles when filming humans. To align with this practice, we ensure that a significant portion of our synthetic data follows similar configurations.

To demonstrate this, we examine the effectiveness of synthetic videos with different camera configurations: random, forward-shot only, frontal, and following shots. As shown in Table 1, we find that synthetic videos incorporating both forward and following shots, which closely align with real-world camera setups, significantly enhance the

Training Data Gym Layer Spin shot Default 83.3% 95% 85% Low-quality asset - 92.5% 22.5% Low-cost rendering 41.7% 17.5% -

Table 2. Success rates illustrating how asset and rendering quality in synthetic videos affect physical fidelity. When asset or rendering quality is low, the physical fidelity in these synthetic videos is less likely to transfer effectively to video generation models.

[Figure 3]

- Figure 3. Visualizations of synthetic videos highlighting both good- and poor-quality 3D assets (a) and rendering (b).

video generation model. This approach notably reduces the collapse rate – defined as the proportion of generated videos that exhibit body collapse – leading to more physically realistic outputs.

We find that synthesizing objects against a clean background allows the model to focus on the subject without diverting capacity to modeling the noisy backgrounds that are inevitable in most real videos. However, using a monotonous background with little variation can lead to overfitting or undesirable associations between the background and the foreground objects. To address this, we adopt a simple yet effective approach by incorporating diverse backgrounds with variations in color, texture, transparency, lighting conditions, and environments (e.g., indoor and outdoor settings). A similar strategy is applied to the camera and object (see Appendix A.1). Empirically, we find that this increased diversity leads to stronger model performance, particularly in previously unseen scenarios.

Appearance Gap Ideally, we would like the appearance of a rendered video to match that of real videos. However, achieving this is challenging as the appearance gap arises from multiple factors. First, real videos are captured by physical cameras, which introduces imperfections such as lens distortions. Second, inaccuracies in the rendered materials and object shapes in virtual environments create additional discrepancies. Finally, rendering algorithms themselves approximate real-world lighting physics, further contributing to the mismatch. In principle, one could hire a large team of skilled artists to overcome these discrepancies, but such a process would be highly resource-intensive.

To this end, we explore several rendering settings to balance this trade-off. Our experiments indicate that both lowquality 3D assets and low-cost rendering quality (Figure 3 top) significantly decrease the success rate of the generated videos, as shown in Table 2. When generating videos with spin shot, the success rate is greatly decreased. For the layer decomposition task, even though the success rate of generating a pure color background remains high, the objects that appear in the output videos often look cartoonish (See Appendix A.2). Table 2 also illustrates that ensuring sufficient quality in both the 3D assets and the rendering settings (Figure 3 bottom) is essential to achieve a high success rate.

#### 3.2. Crafting Captions for Synthetic Videos

Conventional pipelines for building large-scale videocaption datasets is to collect videos first and then generate captions using Vision-Language Models (VLMs). In contrast, as synthetic videos are created from a cross combination of 3D objects, scenes, and camera movements during video synthesis, we caption each element separately and then merge the descriptions into a final caption for the rendered video. This method is efficient and accurate: if we have N objects, M scenes, and C camera setups, it requires only (N + M + C) captions, whereas an existing approach would need to caption N × M × C distinct videos. Such decomposition also improves accuracy and the granularity of the generated caption, as VLMs may produce inconsistent or vague descriptions when confronted with challenging lighting conditions or camera viewpoints in real videos. In contrast, our method ensures consistency by keeping descriptions for the same element regardless of final scene.

As synthetic and real videos exhibit distinct visual characteristics, we hypothesize that embedding special tags (e.g., “animated” or “rendered,” as shown in Figure 11) within synthetic video captions helps the model distinguish the two domains and transfer only the desired physical fidelity into the generation. Through our ablations, we find that explicitly tagging synthetic data promotes more effective cross-domain knowledge transfer (See Sec. 4.3).

#### 3.3. Training with Synthetic Videos

We employ a diffusion transformer model based on the MMDiT architecture [20], trained on real videos at native resolutions [14] within the latent space of a variational autoencoder (VAE) [33]. The model is pretrained using the flow-matching objective.

To enhance the physical fidelity for video generation, we explore incorporating synthetic video data. While training on a mix of synthetic and real videos can improve fidelity, its effectiveness depends on the synthetic-to-real ratio and training steps. Too much synthetic data risks introducing artifacts, while too little yields minimal improvement. Similarly, excessive training can cause overfitting, whereas in-

sufficient training fails to leverage synthetic data effectively.

Even with a well-tuned mixing ratio, synthetic video can still introduce distinctive patterns and artifacts in the generated outputs. To mitigate this, we draw inspiration from [21, 51, 57] and propose SimDrop. Based on [27], we can guide the diffusion generation process toward the overlapping distribution of synthetic and real videos while reducing the influence of synthetic artifacts.

SimDrop begins by training a reference model, Vσ, which aims to capture the unique patterns (e.g., blinkering, animated facials) of synthetic data that pair with rendering engines rather than the clearly defined visual concepts like objects or scenes. Therefore, in training the reference model, we build different captions that only ignores the desired aspect of the synthetic videos (e.g., human motion). This reference model then work in auxiliary with the generation model Vθ trained on mixture of synthetic and real data. Then the reference model can output only the visual patterns but not interfering the objects or human body formation in the video during inference. Formally, let lk denote the denoised latent at step k, and we have:

lk = Vθ(lk−1,t) − αf Vσ,lk−1,t,ˆ nˆ + βf Vθ,lk−1,t,n , where t, tˆ (respectively n, nˆ) are positive (negative) prompts for the synthetic-mixed and reference models, and fθ(Vθ,l,t,n) = Vθ(l,t) − Vθ(l,n). The terms α and β control the influence of each guidance. Using the special tags discussed in Section 3.2, we can incorporate them into the negative prompts. Adding such tags to negative prompts further offers additional benefits, albeit limited.

#### 3.4. Evaluating Physical Fidelity

Since there is no common standard on evaluating physical fidelity in videos, we adopt the following metrics to assess the physical fidelity, inspired by related work [1, 31, 53]. Although these quantitative metrics are not perfect, combining them with human evaluation can provide useful signals. Human pose estimation confidence We employ a stateof-the-art human vision model, Sapiens [31], to evaluate the physical fidelity of the generated human motion. We use a 2B-parameter, 17-keypoint checkpoint to estimate the pose of single-human motion outputs from each model on a perframe basis. The average confidence score ϵconf per keypoint per frame ranges from 0 to 1. Based on the assumption of human vision models, a motion sequence with more realistic body structures and clearer poses gives a higher confidence score.

3D reconstruction error Based on a widely used 3D sparse reconstruction tool, COLMAP [53, 54], we evaluate the physical fidelity of the static objects in the videos with large camera motion. Using a single pinhole camera model and a sequential feature matching mode, COLMAP reconstructs the scene from the video frames. Similar to

the work of [1], we use the following metrics as indicators of physical fidelity: (1) the number of matched feature points (N), (2) the average track length (T ), and (3) the average re-projection error (ϵ). In general, if the video frames generated by a model provide a greater diversity of camera viewpoints yet still maintain the 3D consistency, the number of matched feature points tends to increase. However, the mean track length of each feature point is expected to decrease due to the faster camera motion. Furthermore, a model that is more physically consistent will yield a lower re-projection error in the resulting 3D reconstruction.

Human evaluation For each prompt, we generate two videos of different random seeds. For human evaluation, we instruct our annotators to examine the outputs from different models side-by-side for each prompt strictly following the guideline that focus on physical fidelity of the video. In general, a successfully generated video refers to one that follow the text prompt without visible artifacts. For large human motion, we let the human annotators focus on the integrity of human body, such as limbs, hands, and neck, during the large motion. For large camera motion, the annotators will determine whether the camera motion is performed according to the prompt and examine the object quality. For the layer decomposition task, annotators judge the videos based on two criteria: object quality and background quality. The details of the guideline is in Appendix A.4. Afterwards, we compute the successful rate and average the results across all human evaluators.

### 4. Experiments

To evaluate the effectiveness of synthetic videos for improving the physical fidelity of video generation models, we assess the trained model on three text-to-video tasks: (a) large human motion (dancing and gymnastics), (b) camera spin shots, and (c) layer decomposition (e.g., a moving animal over a solid-colored background). For each task, we use specific text prompts to test the model’s ability to accurately generate the video content. During evaluation, we focus on examining only the physical fidelity of the generated videos, and our criteria do not include aesthetics.

#### 4.1. Implementation Details

Synthetic Video Dataset Following the strategy we discussed in Sec. 3, we first render 32,847 videos of static objects with diverse camera movements and scene setups using Blender and 18,364 videos of humans performing diverse motions captured in simple indoor scenes with different background colors using Unreal Engine for the experiments. Additionally, we plan to release over 1.5M synthetic videos on static objects and 300K synthetic videos on human motions that are outside of the scope of this research to facilitate future research.

Experiment Setup To verify whether synthetic videos can

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

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

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

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

- Figure 4. Visualizations of the videos generated by our improved model, trained using synthetic data. Rows 1,2 highlight wide-angle camera motion; rows 3 display layer decomposition; and rows 4,5,6 demonstrate large human motion.

benefit video generation models, we combine the resulting synthetic videos with real-world videos to train the video generation model with 8B parameters, pretrained with only real-world video data. In line with the tasks we aim to improve, we adopt the optimal strategy found in Sec. 3. To evaluate our trained model, we create 10 prompts each for gymnastics, dancing, camera motion, and layer decomposition, resulting in 40 prompts in total. Please refer to Appendix A.3 for details. We inference videos output with resolution of 1280×720 and duration of 5s at 24 fps. We use the same negative prompts for all inference queries. We compare the outputs of our model trained with synthetic data against the outputs of both the original checkpoint and some of leading commercial models [9, 36, 52] at the same setting and follows the evaluation method in Sec. 3.4.

#### 4.2. Results

Large human motion Our model generates videos of humans performing dancing and gymnastics with significantly reduced limb collapse. As shown in Table 3, the user study indicates that our model produces fewer artifacts in generated videos compared with other models, including three leading generation models and our pretarined model named “Base Model”. In particular, our video generation model greatly improves the success rate of gymnastics movements, while other video generation models generate significantly fewer successful cases. The human pose estimation confidence scores, discussed in Sec. 3.4), further support the findings from the user study. Although the synthetic videos used for training have less realistic shading, the model still

ϵconf User Study

Model

Gym↑ Dance↑ Gym↑ Dance↑

Kling 1.6 [36] 0.715 0.812 10% 43% Runway Gen-3α [52] 0.672 0.809 4% 14% Sora [9] 0.722 0.813 15% 44% Base Model 0.779 0.818 9% 30% Our Model 0.791 0.837 61% 86%

Table 3. The average confidence score of human pose estimation and user study results on the large human motion task.

[Figure 52]

Figure 5. Visualization of video frames with large human motion generated by our model. The shadow of human body follows the human motion.

learns correct human body deformation from the synthetic data and preserves the base model’s realism. Figure 4 visualizes frames of our generated videos. We can see that our model produces visually plausible shadows, a feature that other video generative models have struggled to achieve.

##### Wide-angle camera rotation Our model can produce large

Model N ↑ T ↓ ϵproj↓ ϵˆproj↓ User Study↑

Kling 1.6 [36] 13,328 36.34 0.972 0.298 20% Runway Gen-3α [52] 13,199 36.21 1.181 0.361 26% Sora [9] 14,443 33.62 1.244 0.318 25% Base Model 16,548 31.84 1.159 0.437 20% Our Model 42,895 12.93 1.077 0.135 80%

Table 4. 3D reconstruction metrics and user study results on the large camera motion task. Note that the re-projection error ϵproj is computed over all extracted feature points, whereas ϵˆproj only considers the 1,000 points with the smallest error in each case. The latter metric offers a fairer comparison for methods that produce a significantly higher volume of feature points.

camera spins around static objects and animals, as illustrated in Figure 4. Our training set contains abundant synthetic videos featuring such camera rotations around daily objects. However, the objects used in our testing prompts – including food, animals, and landscapes – lie entirely outside the distribution of our training data. Nonetheless, the model successfully learns the general concept of extensive camera movements and adapts it to previously unseen objects while preserving a high level of realism. As reported in Table 4, the user study indicates that our model’s success rate in producing the intended camera motion is significantly higher than that of other methods. Furthermore, 3D reconstruction metrics, discussed in Sec. 3.4), confirm that objects generated by our model exhibit the strongest geometric consistency across different video frames. Our approach yields the largest number of feature points and the shortest track lengths, indicate that our videos have the largest camera motion and meanwhile maintains best 3D consistency. When these feature points are projected back onto 2D images, our model’s error ϵˆproj is more than twice as small compared with other approaches, demonstrating the enhanced physical fidelity of our generated videos.

Layer decomposition As shown in Table 5, While the baseline models largely fail, our model can produce outputs with a clear separation of the foreground object and the background when tasked to generate videos on pure color backgrounds. This decomposition is beneficial for compositing objects onto arbitrary backgrounds. Similarly to the large camera motion scenario, our model shows this capability to objects not present in the training dataset. Figure 4 shows an example in which the requested object appears cleanly over a green background, suggesting that the model has learned to decompose the scene effectively. Furthermore, the model can even generate dynamic objects and human motion in layers, decomposing the scene in both spatial and temporal dimensions. Neither the original pretrained model nor other commercial models achieve such a clear separation of layers.

Model Layer Decomposition↑

Kling-1.6 [36] 4% Runway-gen3α [52] 1% Sora [9] 4% Base Model 26% Our Model 84%

- Table 5. User study results on the layer decomposition task. With synthetic data augmentation, our model greatly outperforms leading commercial models and the original pretrained model.

Caption Type Uprock↑ Spin↑ Freeze↑

- a) Generic 2% 16% 0%

- b) Fine-grained 98% 84% 66%

- Table 6. Fine-grained captions on human motion achieve better successful rate than generic captions on the large human motion task. “Uprock”, “Spin”, “Freeze” are particular dance moves.

Caption Type Dance Move

- a) No Special Tags 12.5%

- b) Special Tags 90%

- c) Special Tags+Special NP 92.5%

- Table 7. Experiment results on the effect of special tags in synthetic data captioning. Without special tags to differentiate the visual style of the synthetic videos, the video generated models will more likely to generate animated characters or collapsed human motions after training. Also, adding the special tags in negative prompts during generation will help although marginally.

#### 4.3. Ablation Studies

Ablations on synthetic captions We perform experiments of different synthetic caption setups to verify our design in Sec. 3.2. Experiments in Table 6 studies if the finegrained captions help video generation model better learn human motion than their generic counterparts, typically from a zero-shot VLM inference. We observe that for various dance moves (“Uprock”, “Spin”, “Freeze”), having fine-grained caption (Figure 11) greatly reduce the video generation model to generate videos that include collapse and distortion of human body during large motions. Table 7 summarizes the experiments on embedding special tags in synthetic captions to distinguish synthetic videos from the real videos. We found that without special tags, the video generation model is much likely to output videos of animated visual or collapsed human body. We further added the special tags in negative prompts, but found only marginal improvements.

Ablations on training with synthetic videos To examine the mix rate of the synthetic and real videos. we perform the experiment summarized in Table 8. We found that higher mix rate share the same effect as longer training steps and

#Steps

3000 5000 10000 15000

Percentage

10% synthetic videos 20% 25% 40% 60% 50% synthetic videos 55% 75% 85% 80%

- Table 8. Ablation results on synthetic data mix rate and training steps. Here we measure the success rate which the trained foundation model generates videos that follows the prompts but does not include visual patterns in the synthetic videos. We found that large proportion and longer training steps help transferring the properties in synthetic videos to the video generation model. However, performance will saturate and failure cases will include visual patterns of synthetic data.

over-training the model on synthetic data will not lead to more performance increase. Instead, more patterns from synthetic data will appear in the final output. We also verify the design of SimDrop in Table 9. We found that using the captions in training the reference model to prompt them will achieve the best result in terms of the visually preferred cases rated by humans. It also reports the impact of the hypereparameter α value.

α Good Same Bad G-B↑

- 0.1 26.32% 71.05% 2.63% 23.69%

- 0.2 39.47% 52.63% 7.89% 31.58%

- Table 9. Experiment results on SimDrop. Here, we compare the output videos with SimDrop with the models without SimDrop. Evaluators will choose the best out of two videos side-by-side. We then compute the winning/same/losing rate against the baseline.

### 5. Related Work

Video generation Conditional video generation is a challenging task aiming to synthesize temporally coherent and visually realistic video sequences from structured inputs such as images and text prompts. Current video generation models can be broadly categorized into Generative Adversarial Networks (GANs) [5, 8, 24, 38], autoregressive models [34, 37, 68, 69], and diffusion models [17, 26, 28, 35, 67, 73]. These architectures in video generation usually inherit their success in image generation [10, 20, 48, 70]. In recent years, rapid advancements in video generation, represented by Sora [9], have been significantly driven by the availability of large-scale webcollected video datasets and the development of scalable model architectures such as DiT [20]. State-of-the-art commercial models [25, 35, 36, 52] have demonstrated the ability to generate highly realistic videos. These models leverage extensive training data to improve motion fluency, scene reality and overall aesthetic quality in generating videos.

Physics in video generation Despite the effort in scaling data and model size, problems remain in the physics of

generated videos after researchers’ evaluation [6, 30, 47]. Yet for video generation models, physics appears learnable directly from video data [11, 22, 45, 61] and is crucial for these foundation models to serve as world models [1, 9, 16, 60]. Therefore, there are growing number of works [3, 40] in improving physics-grounding in video generation and beyond [7, 44]. They mainly propose model modifications by adding additional supervisory signals [12, 29, 41, 63], and mainly tailored for a certain aspect of physics such as motion [12, 42, 43] or sound in the videos [58]. While such methods show more physically coherent results, they often require modifications to the diffusion architecture itself and rely on manually specified control signals. Our work focuses on physical fidelity and differentiates by proposing a data-centric approach without modifying the diffusion model architecture and harness the potential of 3D rendering engines [13]. Our method build synthetic video data that can benefit video generation models regardless of their architectures and improves on diverse aspects of physics fidelity.

Synthetic data in AI Synthetic Data from simulation engines has been widely applied in advancing many fields of AI, such as autonomous driving [62, 74] and embodied agents [50, 56, 72], or scene generation [55]. At the intersection of synthetic data and video, most work focus on understanding [32, 64, 71] and only a few early work [3] explore how synthetic video data can help video generation in particular tasks such as camera control [4] or motion [23, 42]. We are the first work to systematically study how synthetic videos from simulation engines can help improve the physics fidelity of video generation model.

### 6. Conclusion

In this study, we investigate how to use synthetic video data generated by CGI production pipelines (Blender [59] and Unreal Engine [19]) to enhance physical fidelity of video generation models. We verify our method on three tasks necessitating realistic physical behavior, where our model achieves superior results through synthetic data enhancement. Our results demonstrate that the physical fidelity of video generation can be enhanced using synthetic video. Note that while our method improves physical fidelity and aligns more closely with human perception, it still lacks an understanding of the underlying principles of physics, leaving significant room for further improvement.

Going beyond, future work may consider generating more intricate physical effects [40], including complex interactions among multiple objects and physically based fluid simulations. Moreover, while we only focus on the RGB color channel in this work, the synthetic rendering pipeline offer much more information(e.g., depth, normals, alpha masks) that could serve as supervisory signals, otherwise not easily obtainable in real datasets.

### 7. Acknowledgment

We thank Ceyuan Yang, Liangke Gui, and Shanchuan Lin for their insightful discussions on this project. We also appreciate Zhibei Ma and Renfei Sun for their support in building the engineering foundation.

### References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 2, 5, 8
- [2] Autotroph. Blender market. https : / / blendermarket.com/, 2024. 12
- [3] Yunhao Ba, Guangyuan Zhao, and Achuta Kadambi. Blending diverse physical priors with neural networks. arXiv preprint arXiv:1910.00201, 2019. 8
- [4] Jianhong Bai, Menghan Xia, Xintao Wang, Ziyang Yuan, Xiao Fu, Zuozhu Liu, Haoji Hu, Pengfei Wan, and Di Zhang. Syncammaster: Synchronizing multi-camera video generation from diverse viewpoints. arXiv preprint arXiv:2412.07760, 2024. 8
- [5] Yogesh Balaji, Martin Renqiang Min, Bing Bai, Rama Chellappa, and Hans Peter Graf. Conditional gan with discriminative filter generation for text-to-video synthesis. In IJCAI, page 2, 2019. 8
- [6] Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, KaiWei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. arXiv preprint arXiv:2406.03520, 2024. 8
- [7] Daniel M Bear, Elias Wang, Damian Mrowca, Felix J Binder, Hsiao-Yu Fish Tung, RT Pramod, Cameron Holdaway, Sirui Tao, Kevin Smith, Fan-Yun Sun, et al. Physion: Evaluating physical prediction from vision in humans and machines. arXiv preprint arXiv:2106.08261, 2021. 8
- [8] Tim Brooks, Janne Hellsten, Miika Aittala, Ting-Chun Wang, Timo Aila, Jaakko Lehtinen, Ming-Yu Liu, Alexei Efros, and Tero Karras. Generating long videos of dynamic scenes. Advances in Neural Information Processing Systems, 35:31769–31781, 2022. 8
- [9] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. https://openai.com/research/videogeneration - models - as - world - simulators,

2024. 1, 6, 7, 8

- [10] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11315–11325, 2022. 8
- [11] Pradyumna Chari, Chinmay Talegaonkar, Yunhao Ba, and Achuta Kadambi. Visual physics: Discovering physical laws from videos. arXiv preprint arXiv:1911.11893, 2019. 8

- [12] Hila Chefer, Uriel Singer, Amit Zohar, Yuval Kirstain, Adam Polyak, Yaniv Taigman, Lior Wolf, and Shelly Sheynin. Videojam: Joint appearance-motion representations for enhanced motion generation in video models. arXiv preprint arXiv:2502.02492, 2025. 8
- [13] Celso M De Melo, Antonio Torralba, Leonidas Guibas, James DiCarlo, Rama Chellappa, and Jessica Hodgins. Nextgeneration deep learning based on simulators and synthetic data. Trends in cognitive sciences, 26(2):174–187, 2022. 1, 8
- [14] Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36:2252–2274, 2023. 4
- [15] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13142–13153, 2023. 12
- [16] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. Advances in neural information processing systems, 36:9156–9172, 2023. 8
- [17] Abul Ehtesham, Saket Kumar, Aditi Singh, and Tala Talaei Khoei. Movie gen: Swot analysis of meta’s generative ai foundation model for transforming media generation, advertising, and entertainment industries. arXiv preprint arXiv:2412.03837, 2024. 1, 8
- [18] Epic Games. Metahuman. https : / / www . unrealengine . com / en - US / metahuman, 2024. 12
- [19] Epic Games. Unreal engine 5. https://www. unrealengine.com/en-US/unreal-engine-5,

2024. 1, 2, 8, 13

- [20] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 4, 8

- [21] Felix Friedrich, Manuel Brack, Lukas Struppek, Dominik Hintersdorf, Patrick Schramowski, Sasha Luccioni, and Kristian Kersting. Fair diffusion: Instructing text-toimage generation models on fairness. arXiv preprint arXiv:2302.10893, 2023. 5
- [22] Quentin Garrido, Nicolas Ballas, Mahmoud Assran, Adrien Bardes, Laurent Najman, Michael Rabbat, Emmanuel Dupoux, and Yann LeCun. Intuitive physics understanding emerges from self-supervised pretraining on natural videos. arXiv preprint arXiv:2502.11831, 2025. 8
- [23] Daniel Geng, Charles Herrmann, Junhwa Hur, Forrester Cole, Serena Zhang, Tobias Pfaff, Tatiana Lopez-Guevara, Carl Doersch, Yusuf Aytar, Michael Rubinstein, et al. Mo-

- tion prompting: Controlling video generation with motion trajectories. arXiv preprint arXiv:2412.02700, 2024. 8
- [24] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 8
- [25] Google. Veo2 - google deepmind. https://deepmind. google/technologies/veo/veo-2/, 2024. 1, 8
- [26] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Fei-Fei Li, Irfan Essa, Lu Jiang, and Jos´e Lezama. Photorealistic video generation with diffusion models. In European Conference on Computer Vision, pages 393–411. Springer, 2024. 8
- [27] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 2, 5
- [28] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 8
- [29] Achuta Kadambi, Celso de Melo, Cho-Jui Hsieh, Mani Srivastava, and Stefano Soatto. Incorporating physics into datadriven computer vision. Nature Machine Intelligence, 5(6): 572–580, 2023. 8
- [30] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. arXiv preprint arXiv:2411.02385, 2024. 1, 8
- [31] Rawal Khirodkar, Timur Bagautdinov, Julieta Martinez, Su Zhaoen, Austin James, Peter Selednik, Stuart Anderson, and Shunsuke Saito. Sapiens: Foundation for human vision models. In Proceedings of the 18th European Conference on Computer Vision, pages 206–228, Berlin, Heidelberg, 2024. Springer-Verlag. 2, 5
- [32] Yo-whan Kim, Samarth Mishra, SouYoung Jin, Rameswar Panda, Hilde Kuehne, Leonid Karlinsky, Venkatesh Saligrama, Kate Saenko, Aude Oliva, and Rogerio Feris. How transferable are video representations based on synthetic data? In Advances in Neural Information Processing Systems, pages 35710–35723. Curran Associates, Inc., 2022. 2, 8
- [33] Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013. 4
- [34] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jos´e Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023. 8
- [35] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 1, 8
- [36] Kuaishou. Kling video model. https://kling. kuaishou.com/en, 2024. 1, 6, 7, 8
- [37] Manoj Kumar, Mohammad Babaeizadeh, Dumitru Erhan, Chelsea Finn, Sergey Levine, Laurent Dinh, and

- Durk Kingma. Videoflow: A conditional flow-based model for stochastic video generation. arXiv preprint arXiv:1903.01434, 2019. 8
- [38] Yitong Li, Zhe Gan, Yelong Shen, Jingjing Liu, Yu Cheng, Yuexin Wu, Lawrence Carin, David Carlson, and Jianfeng Gao. Storygan: A sequential conditional gan for story visualization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6329–6338,

2019. 8

- [39] Junwei Liang, Lu Jiang, and Alexander Hauptmann. Simaug: Learning robust representations from simulation for trajectory prediction. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIII 16, pages 275–292. Springer, 2020. 2
- [40] Daochang Liu, Junyu Zhang, Anh-Dung Dinh, Eunbyung Park, Shichao Zhang, and Chang Xu. Generative physical ai in vision: A survey. arXiv preprint arXiv:2501.10928,

2025. 2, 8

- [41] Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. Physgen: Rigid-body physics-grounded imageto-video generation. In European Conference on Computer Vision (ECCV), 2024. 8
- [42] Jiaxi Lv, Yi Huang, Mingfu Yan, Jiancheng Huang, Jianzhuang Liu, Yifan Liu, Yafei Wen, Xiaoxin Chen, and Shifeng Chen. Gpt4motion: Scripting physical motions in text-to-video generation via blender-oriented gpt planning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 1430–1440, 2024. 8
- [43] Joanna Materzy´nska, Josef Sivic, Eli Shechtman, Antonio Torralba, Richard Zhang, and Bryan Russell. Newmove: Customizing text-to-video models with novel motions. In Proceedings of the Asian Conference on Computer Vision, pages 1634–1651, 2024. 8
- [44] Willi Menapace, St´ephane Lathuili`ere, Aliaksandr Siarohin, Christian Theobalt, Sergey Tulyakov, Vladislav Golyanik, and Elisa Ricci. Playable environments: Video manipulation in space and time. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3584–3593, 2022. 8
- [45] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsense-based benchmark for video generation. arXiv preprint arXiv:2410.05363, 2024. 8
- [46] Meta Reality Labs Research. Digital twin catalog. https: //www.projectaria.com/datasets/dtc/, 2024. 12
- [47] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models learn physical principles from watching videos? arXiv preprint arXiv:2501.09038, 2025. 8
- [48] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

- 2023. 8

[49] Poly Haven. Poly haven. https://polyhaven.com/,

- 2024. 13

- [50] Xavier Puig, Kevin Ra, Marko Boben, Jiaman Li, Tingwu Wang, Sanja Fidler, and Antonio Torralba. Virtualhome: Simulating household activities via programs. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8494–8502, 2018. 8
- [51] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 5
- [52] RunwayML. Gen-3 alpha. https://runwayml.com/ research/introducing-gen-3-alpha, 2024. 6, 7, 8
- [53] Johannes Lutz Sch¨onberger and Jan-Michael Frahm. Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 5
- [54] Johannes Lutz Sch¨onberger, Enliang Zheng, Marc Pollefeys, and Jan-Michael Frahm. Pixelwise view selection for unstructured multi-view stereo. In European Conference on Computer Vision (ECCV), 2016. 5
- [55] Yu Shang, Yuming Lin, Yu Zheng, Hangyu Fan, Jingtao Ding, Jie Feng, Jiansheng Chen, Li Tian, and Yong Li. Urbanworld: An urban world model for 3d city generation. arXiv preprint arXiv:2407.11965, 2024. 8
- [56] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 8
- [57] Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. Styledrop: Text-to-image generation in any style. In 37th Conference on Neural Information Processing Systems (NeurIPS). Neural Information Processing Systems Foundation, 2023. 5
- [58] Kun Su, Kaizhi Qian, Eli Shlizerman, Antonio Torralba, and Chuang Gan. Physics-driven diffusion models for impact sound synthesis from videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9749–9759, 2023. 8
- [59] The Blender Fundation. Blender. https://www. blender.org/, 2024. 1, 2, 8, 13
- [60] Xiaofeng Wang, Zheng Zhu, Guan Huang, Boyuan Wang, Xinze Chen, and Jiwen Lu. Worlddreamer: Towards general world models for video generation via predicting masked tokens. arXiv preprint arXiv:2401.09985, 2024. 8
- [61] Jiajun Wu, Erika Lu, Pushmeet Kohli, Bill Freeman, and Josh Tenenbaum. Learning to see physics via visual deanimation. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2017. 8
- [62] Ziyang Xie, Zhizheng Liu, Zhenghao Peng, Wayne Wu, and Bolei Zhou. Vid2sim: Realistic and interactive simulation from video for urban navigation. arXiv preprint arXiv:2501.06693, 2025. 8
- [63] Qiyao Xue, Xiangyu Yin, Boyuan Yang, and Wei Gao. Phyt2v: Llm-guided iterative self-refinement for

- physics-grounded text-to-video generation. arXiv preprint arXiv:2412.00596, 2024. 8
- [64] Honghui Yang, Di Huang, Wei Yin, Chunhua Shen, Haifeng Liu, Xiaofei He, Binbin Lin, Wanli Ouyang, and Tong He. Depth any video with scalable synthetic data. arXiv preprint arXiv:2410.10815, 2024. 8
- [65] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-video: Customized video generation with userdirected camera movement and object motion. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 1
- [66] Sherry Yang, Jacob Walker, Jack Parker-Holder, Yilun Du, Jake Bruce, Andre Barreto, Pieter Abbeel, and Dale Schuurmans. Video as the new language for real-world decision making. arXiv preprint arXiv:2402.17139, 2024. 1
- [67] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 8
- [68] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10459–10469, 2023. 8
- [69] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023. 8
- [70] Bowen Zhang, Shuyang Gu, Bo Zhang, Jianmin Bao, Dong Chen, Fang Wen, Yong Wang, and Baining Guo. Styleswin: Transformer-based gan for high-resolution image generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11304–11314,

2022. 8

- [71] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 8
- [72] Qi Zhao, Haotian Fu, Chen Sun, and George Konidaris. Epo: Hierarchical llm agents with environment preference optimization. arXiv preprint arXiv:2408.16090, 2024. 8
- [73] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. https://github.com/hpcaitech/OpenSora, 2024. 8
- [74] Yunsong Zhou, Michael Simon, Zhenghao Mark Peng, Sicheng Mo, Hongzi Zhu, Minyi Guo, and Bolei Zhou. Simgen: Simulator-conditioned driving scene generation. Advances in Neural Information Processing Systems, 37: 48838–48874, 2024. 8

| |Property Name|Choice|Description<br><br>|
|---|---|---|---|
|Camera<br><br>|Camera Focus Type|Follow|The camera focus follows the object.|
| | |Fixed|The camera focus is static in the world space.|
| |Camera Focus Position|Upper, Center, Lower<br><br>|The camera focus is at the upper/center/lower part of the object.|
| |Camera Movement Type|Truck, Dolly, Pedestal, Tilt, Pan, Spin, Following, Zoom<br><br>|The basic camera movement types.|
| |Camera Movement Value<br><br>|Scalar|How much the camera moves.|
| |Camera Initial Position|3D Position<br><br>|The initial position of the camera.|
| |Camera Focal Length<br><br>|Scalar<br><br>|The scalar controls how much percentage of the object is visible on the screen.|
|LightandEnvironment|Scene Type<br><br>|Env|The environment is given by a HDR environmental map. The map will also be used as the light source.|
| | |Basic|The environment is an indoor room which color is controlled by “Scene Color” and has two light sources.|
| | |Empty<br><br>|The environment is empty but has two light sources or one environmental map as the light source.|
| |Scene Color<br><br>|RGB color|The color for the indoor room when presented.|
| |Light Position<br><br>|3D position|The position of the light when presented.|
| |Light Color|Scalar|The color temperature of the light when presented.|
| |Light Intensity<br><br>|Scalar|The intensity of the light when presented.|
| |Ambient Light Intensity<br><br>|Scalar|Ambient light intensity. The ambient light exists when the lights are used.<br><br>|
|Render|Background Color<br><br>Render Engine Render Quality<br><br>|RGBA color Blender/Unreal High/Low<br><br>|The background color of the location where the scene is empty. The quality of the rendering. We have two presets of rendering setting.|

andEnvironmentCamera

Table 10. The parameters used for controlling our rendering pipeline.

[Figure 53]

Figure 6. 3D scene setup in Blender and Unreal Engine. The wireframes and corresponding rendering outputs.

### A. Appendix

#### A.1. Details of Synthetic Data Generation

Following a standard CGI production pipeline for creating videos, our synthetic video generation framework comprises two main modules: (1) 3D scene setup and (2) rendering. Below, we provide a detailed overview of these modules and the specific parameters that govern them.

##### A.1.1. 3D Scene Setup

As discussed in Sec. 2, we focus on generating videos featuring a single object per scene. To achieve this, we build a procedural 3D scene generator driven by a carefully chosen set of parameters, enabling the production of a wide variety of synthetic videos. A typical 3D scene is composed of four main components: (1) the 3D object, (2) the camera, (3) the lighting conditions, and (4) the environment. We adopt this composition in our generator. Each component in our generator is controlled by a set of parameters, which we detail below.

3D Object. As we target single-object videos, we seek to

include 3D assets that are both high-quality and highly varied. To this end, we collect assets from Objaverse 1.0 [15], Digital Twin Catalog [46], Blender Market [2], and Metahuman [18]. These sources collectively provide diverse asset categories and styles. We further filter assets from Objaverse based on categories, polygon count, view count, user ratings, and VLM to ensure overall quality. For other sources, we retain all assets since they are already curated with high fidelity.

Camera. We represent the camera using a set of parameters that capture real-world usage scenarios (see Table 10). These parameters include:

- • Camera movement type: Determines the camera’s trajectory around the object. In our experiments, we select one movement type at a time and quantify its extend using a parameter “Camera Movement Value”.
- • Initial position and focus: Specifies where the camera starts and how it focuses on the primary object.
- • Focal length: Adjusts the camera’s field of view relative to how much of the screen the object occupies.

Such parameterization allows us to mimic various camera

[Figure 54]

Figure 7. Examples of our synthetic video data. We render the synthetic videos with diverse background to alleviate the potential biases in synthetic videos.

behaviors from the real world.

Lighting and Environment. For simplicity, we jointly model the environment and its lighting conditions (see Table 10). Our parameterization supports three main configurations:

- • HDR environment map: Provides both the background and primary light source. We use environment maps from Poly Haven [49].
- • Solid-color indoor room: Uses two light sources (Figure 6) for illumination: one positioned above the object and another placed elsewhere in the scene.
- • Empty scene: Lit by either an environment map or two lights for more controlled illumination with empty surroundings.

Although these settings may appear simple, they cover a wide range of lighting conditions and backdrop variations, thereby maintaining diversity while keeping the primary object prominent.

##### A.1.2. Rendering Setup

We employ two open-source rendering engines to generate high-quality video outputs:

- • Unreal Engine (Lumen): We use Unreal Engine 5.4.4 with Lumen as our renderer with maximal render-quality settings to achieve realistic rendering effects [19].
- • Blender (Cycles): We use Blender 4.2 and Cycles renderer configured with carefully chosen parameters to balance rendering speed and visual fidelity [59]

These engines offer robust rendering pipelines and physically based shading models, ensuring that our synthetic data closely reflects real-world lighting conditions.

##### A.1.3. Random Sampling of Parameter Space

To produce a large and diverse set of synthetic videos, we define a configuration (“config”) file containing all relevant parameters described above. Figure 7 show some examples of synthetic videos with diverse setups. Our 3D scene generator parses this config file and sets up the scene. Then, the

[Figure 55]

Figure 8. Example outputs from video generation models trained on synthetic datasets with low-quality assets. The resulting objects frequently exhibit cartoonish or animated characteristics, diverging from the intended original visual style.

rendering engines render the scene into a video. For largescale generation, we employ random sampling over each parameter’s prescribed probability distribution, guided by the key insights from Sec. 3. Each sampling step produces a unique config file, which is then rendered into a separate synthetic video. This process enables us to generate a vast set of diverse synthetic videos with minimal manual intervention.

#### A.2.MoreAblationExperimentsandVisualizations

In this section, we provide additional visualizations of the data curation experiments and the ablation studies. Figure 8 and Figure 9 show the effect of using poor quality asset and rendering respectively. Figure 10 shows the effect of excessive training on synthetic data. Color patterns are introduced into the generation model. Figure 11 gives an example of fine-grained and generic captions and an example of using special tags. Figure 12 and Figure 13 show the comparison between videos from generation with and without SimDrop. Lastly, Figure 14 showcases the layer decompostion videos can use to separate out dynamic objects (e.g. animals, fluids) to enable video matting. Finally, Figure 15 shows more generated videos across all three tasks.

#### A.3. Evaluation Prompts

##### Large Human Motion

Dancing:

- • A dancer practicing at home
- • In a street setting, a teenager is performing breakdance moves, including leaning back, balancing on one leg, and rhythmically moving arms.
- • An attractive man energetically dances, featuring lively movements. He crosses his arms and vigorously moves his legs, imitating horse riding and other whimsical actions.
- • A young woman gracefully pirouettes on one foot, her other leg bent elegantly and arms outstretched for balance

[Figure 56]

- Figure 9. Visualization of generated outputs from video generation models trained with synthetic videos of low quality assets in large camera motion task. The objects in these generated videos more likely to appear static or animated.

[Figure 57]

- Figure 10. Visualization of over training video generation models trained with synthetic videos. Visual patterns such as color tone are more likely to appear in generated videos.

[Figure 58]

- Figure 11. A comparison of generating captions for synthetic videos using existing methods (Generic Caption) and our method (Fine-Grained Caption). We also show a comparison of captions with special tags and without special tags.

and flair. She transitions through various spins, showcasing a dynamic dance routine that blends elements of northern soul dancing. She dances in a bustling urban plaza, or a serene beach at sunset, or a lively street festival, or, a beautifully lit dance studio. Each setting captures the fluidity and energy of her movements, adding depth and variety to her performance.

• A young woman is performing breakdance moves, including leaning back and balancing on one leg while engaging arms rhythmically.

[Figure 59]

- Figure 12. A comparison showcasing the effect of SimDrop. Row 1 is the result without SimDrop and Row 2 is the video with the method. The color tone in row two is significantly more better and without color pattern from the synthetic data.

[Figure 60]

- Figure 13. A comparison showcasing the effect of SimDrop. Row 1 is the result without SimDrop and Row 2 is the video with the method. The human faces in row two is significantly more realistic and appealing.

[Figure 61]

- Figure 14. Example of background editing. Our layer generation enables easy background replacement via green-screen matting.

- • A woman dancing on grassland during sunset
- • On a beach, an Ultraman from Japanese TV show is spinning around on one foot while keeping other leg bent and arms extended for balance and style. It performs multiple spins, emphasizing a dance move commonly associated with northern soul dancing.
- • In a bright dance room, a young woman is performing

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

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

Figure 15. More visualization of generated videos for large camera motion (row 1,2), layer decomposition (row 3,4), and large human motion(row 5,6).

a dance with enthusiastic movements The person crosses arms and moves legs energetically, mimicking riding a horse and performing other playful gestures.

- • A young woman is performing a breakdance move, starting with a dynamic step and then transitioning into a series of fluid body movements and rhythmic steps.
- • A handsome man initiates with a dynamic step followed by a series of fluid body motions and rhythmic steps.

Gymnastics:

- • In a bright dance room, A man executes a backflip by initially crouching low, launching himself upwards, rotating backwards in midair before returning to a standing position on his feet.
- • In a well-lit dance studio, A woman performs a gymnastics moves to flip her body. Her backflip is to first crouch low, then rotating upwards and backward in midair, eventually landing back in a standing position.
- • A man performs a backflip by first squatting down, then launching itself into the air, flipping backward, and finally landing back onfeet on grassland under sunshine.
- • In a sunny grassland, a woman executes a backflip by initially crouching, then springing into the air, rotating backward, and ultimately landing on her feet.
- • A female athlete performs a backflip by first squatting down, then launching itself into the air, flipping backward, and finally landing back onfeet during the floor execrise event at the Olympic Games.
- • During the floor exercise event at the Olympic Games,

a male athlete performs a stunning backflip. He begins by squatting down low, gathering his strength and focus. With a powerful burst of energy, he launches himself into the air, his body gracefully arching as he flips backward. The sunlight glints off his muscular form as he completes the rotation, and he lands solidly on his feet, his expression a mix of concentration and triumph.

- • A man Moves with dynamic energy, shifting from a standing position to a deep crouch, then rotating her body midair before landing upright on the sunlit grassland.
- • A woman is moving dynamically, transitioning from a standing position to a deep crouch and then rotating body mid-air before returning to an upright stance on grassland under sunshine.
- • During the floor exercise event at the Olympic Games, a female athlete moves with dynamic precision. She transitions from a standing position to a deep crouch, then launches herself into the air, rotating her body mid-flight before landing gracefully back on her feet.
- • At the Olympic Games’ floor exercise event, a male athlete showcases his agility by swiftly dropping into a deep crouch from a standing position. He then propels himself into the air, executing a mid-air rotation, and lands back on his feet with precision and grace.

Large Camera Motion

- • A lion standing on the grass. spin shot.
- • An astronaut riding a horse, high definition, 4k. spin shot.
- • A panda swimming underwater. spin shot.

- • Video of sailboat on a lake during sunset. spin shot.
- • Variety of succulent plants on a garden. spin shot.
- • A birthday cake in the plate. spin shot.
- • Big cargo ship passing on the shore. spin shot.
- • Time lapse video, sunrise of the Great Wall. spin shot.
- • A tree with Halloween decoration. spin shot.
- • A Labrador dog wearing glasses and casual clothes is lying on the bed reading. spin shot.

Layer Decomposition

- • A lion standing in a green background.
- • A lion running in a green background.
- • Turtle swimming in a green background.
- • An african penguin walking in a green background.
- • Variety of succulent plants in a green background.
- • Leaves swaying in the wind in a green background.
- • A stack of dried leaves burning in a green background.
- • Big cargo ship like in the movies passing in a green background.
- • Helicopter landing in a green background.
- • A young woman is performing breakdance moves, including leaning back and balancing on one leg while engaging arms rhythmically in a light blue background.

#### A.4. Human Evaluation Details

Our user study videos are available on the project website. We invite the community to also rate the videos.

Large Human Motion For large human motions, we asks our human raters to examine how many out of the generated videos in each video show no collapse in human body structure. Specifically, we ask them to focus on the limbs and torso areas. The detailed rules are as following: 1. Does the video include the full body of the person (all four limbs) for more than 2 seconds? 2. Is the video bascially showing what is specified by the prompt, including background and motion? 3. Does the person in the video looks animated? 4. Is there limbs or torso addition/missing from the video? 5. Is there transition of body parts that are obviously unnatural (e.g. switching body parts at the same location)?

Please Note: 1. DO NOT focus your judgement on these part of the human body: hands, feet, or face 2. DO NOT judge the asethetics or naturalness of the human motion, please just focus on human body integrity

Large Camera Motion For Large camera motion, we instruct the human raters to focus on the object and the degree which the picture rotates. The detailed rules are as following: If any of the following question is yes, please mark the video as 0 1. If the object appear in the video is corrupt, unnatural, or animated 2. If the background is not of pure color as instructed by the prompt

Layer Decomposition For layer decompostion, we instruct the human raters to focus on the object and the background quality. The detailed rules are as following: If any of the following question is yes, please mark the video as 0

1. If the object appear in the video does not spin at all. 2. If the object appear in the video spins but the background does not move with the object 3. If the object appear in the video corrupts, becomes unnatural or looks animated.

