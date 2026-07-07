# arXiv:2603.23462v1[cs.CV]24Mar2026

RealMaster: Lifting Rendered Scenes into Photorealistic Video

DANA COHEN-BAR1,2 IDO SOBOL2,3 RAPHAEL BENSADOUN2 SHELLY SHEYNIN2 ORAN GAFNI2 OR PATASHNIK1 DANIEL COHEN-OR1 AMIT ZOHAR2

1TEL AVIV UNIVERSITY 2REALITY LABS, META 3TECHNION

[Figure 1]

Fig. 1. RealMaster lifts synthetic-looking rendered video into photorealistic video, faithfully re-realizing the original scene.

often retain a sterile appearance that lacks the high-frequency detail of real-world footage, often falling into the uncanny valley (see Fig. 1, top). Bridging this gap would enable a compelling new paradigm: using video diffusion models as a learned second-stage renderer atop fast 3D engines, combining the control of traditional graphics with the photorealism of generative models.

State-of-the-art video generation models produce remarkable photorealism, but they lack the precise control required to align generated content with specific scene requirements. Furthermore, without an underlying explicit geometry, these models cannot guarantee 3D consistency. Conversely, 3D engines offer granular control over every scene element and provide native 3D consistency by design, yet their output often remains trapped in the “uncanny valley”. Bridging this sim-to-real gap requires both structural precision, where the output must exactly preserve the geometry and dynamics of the input, and global semantic transformation, where materials, lighting, and textures must be holistically transformed to achieve photorealism. We present RealMaster, a method that leverages video diffusion models to lift rendered video into photorealistic video while maintaining full alignment with the output of the 3D engine. To train this model, we generate a paired dataset via an anchor-based propagation strategy, where the first and last frames are enhanced for realism and propagated across the intermediate frames using geometric conditioning cues. We then train an IC-LoRA on these paired videos to distill the high-quality outputs of the pipeline into a model that generalizes beyond the pipeline’s constraints, handling objects and characters that appear mid-sequence and enabling inference without requiring anchor frames. Evaluated on complex GTA-V sequences, RealMaster significantly outperforms existing video editing baselines, improving photorealism while preserving the geometry, dynamics, and identity specified by the original 3D control.

To bridge this gap, the task of sim-to-real translation aims to transform rendered video into photorealistic sequences. A natural approach is to leverage recent advances in video editing, where largescale generative models have demonstrated impressive capabilities in modifying video content while preserving temporal coherence. However, sim-to-real translation poses a fundamentally different challenge than standard video editing. Unlike typical editing tasks, which involve local modifications or global stylization, sim-to-real requires simultaneously satisfying two seemingly conflicting objectives: structural precision, where the output must exactly preserve the input’s geometry, motion, and dynamics down to fine details; and global semantic transformation, where materials, lighting, and textures must be holistically transformed to achieve true photorealism. Because the input is already near-photorealistic, details cannot be abstracted away as in conventional style transfer; the model must preserve fine details while adding the high-frequency nuances that characterize real-world footage. In practice, we find that existing video editing methods struggle with this tension. When applied to sim-to-real, they either fail to recognize the synthetic nature of the input and leave it largely unchanged, or they change too much and fail to preserve important details from the original.

1 Introduction

Recent advancements in large-scale generative models have enabled the synthesis of video with extraordinary photorealism. However, these models remain difficult to steer with precision: they rely on text prompts or reference images rather than explicit 3D representations, limiting their capacity to control individual scene elements or guarantee geometric consistency across frames.

In this work, we present RealMaster, a method for sim-to-real video translation. Specifically, we train a model that lifts rendered video into photorealistic video while preserving the underlying scene structure and dynamics. A central component of our approach is a sparse-to-dense propagation strategy that constructs

In contrast, traditional 3D engines offer precise user control and enforce geometric consistency by design. Yet, despite decades of progress in rendering, the sim-to-real gap persists: synthetic outputs

high-quality training supervision directly from rendered sequences. Given a rendered video, we first edit the first and last frames to serve as photorealistic visual anchors. We then propagate their appearance across the sequence using a conditional video model guided by edge cues, producing a photorealistic video that remains aligned with the original rendered input. This process yields paired rendered–photorealistic video data. We then train an IC-LoRA on these video pairs, distilling the behavior of the propagation pipeline into a model that generalizes beyond its limitations and can directly perform the sim-to-real task at inference time. By leveraging the foundation model as a strong prior, the network learns to discount imperfections in the synthetic data and produce high-quality outputs that remain faithful to the input rendered video.

We evaluate the effectiveness of RealMaster through extensive experiments on diverse sequences from the GTA-V virtual environment. This setting provides a challenging testbed due to its complex lighting transitions, high-speed motion, intricate geometric details, and the presence of multiple interacting characters. As shown in Fig. 1, RealMaster produces photorealistic videos that preserve the structure and dynamics of the source scenes under these challenging conditions. Our quantitative and qualitative results further demonstrate that RealMaster significantly outperforms state-of-the-art video editing baselines in both preservation of the input and photorealism, successfully resolving the trade-off between structural precision and global transformation that limits existing methods.

- 2.2 Video Generation and Controllability

Recent breakthroughs in diffusion-based generative models [Ho et al. 2020; Song et al. 2021] have redefined video synthesis. Foundation models such as Stable Video Diffusion [Blattmann et al. 2023], Gen-

- 2 [Esser et al. 2023], Lumiere [Bar-Tal et al. 2024], CogVideoX [Yang et al. 2024], MovieGen [Polyak et al. 2024], Wan [Wan et al. 2025] and LTX-2 [HaCohen et al. 2026] produce high-resolution, cinematic sequences.

In parallel to these advances in video generation, a growing body of work studies controllability through explicit conditioning. ControlNet [Zhang and Agrawala 2023] introduced a paradigm for conditioning image diffusion models on spatial control signals such as depth, edges, and human pose. Subsequent work extends structural conditioning to video diffusion by providing these signals across time, including depth-conditioned generation [Luo et al. 2023], temporally sparse constraints [Guo et al. 2024], and trainingfree ControlNet-style control for text-to-video [Zhang et al. 2024].

Complementary to structural conditioning, exemplar-based approaches use in-context visual examples to guide generation. InContext LoRA [Huang et al. 2024] demonstrates this for text-toimage diffusion transformers, showing that the model can learn to leverage structured exemplars provided in the context during generation, and that this capability can be further strengthened through lightweight fine-tuning.

2.3 Video Editing

Diffusion-based video generation models have been extended to video editing through two main paradigms. Early work largely operates in a zero-shot manner, enabling text-guided manipulation without requiring task-specific paired training data [Cong et al.

- 2023; Geyer et al. 2023; Liu et al. 2023; Qi et al. 2023; Singer et al.
- 2024; Wang et al. 2023; Wu et al. 2023; Yang et al. 2023]. In contrast, more recent approaches leverage large-scale training to support general-purpose video editing capabilities across a wide range of edits [Bai et al. 2025; DecartAI 2025; Jiang et al. 2025; Molad et al. 2023; Polyak et al. 2024; Qin et al. 2023].

A complementary line of work focuses on first-frame editing followed by propagation [Ceylan et al. 2023; Ku et al. 2024; Ouyang et al. 2024b,a], where sparse edits are transferred across time using a conditional video model. This paradigm is most closely related to our approach, as it similarly aims to maintain temporal coherence while applying targeted appearance changes.

However, despite strong performance on creative edits, existing video editing methods struggle on sim-to-real translation. When applied to rendered videos, they either fail to recognize the synthetic appearance and thus produce minimal changes, or they introduce large visual edits that fail to preserve the underlying scene structure and character identity. This limitation highlights a fundamental tension in sim-to-real translation: the task requires both global appearance transformation and strict input preservation—objectives that current video editing methods struggle to optimize jointly.

- 3 Method

2 Related Work

- 2.1 Sim-to-Real Translation

The mapping of rendered content to photorealistic domains is fundamentally distinct from artistic style transfer. This problem was first explored in classical example-based synthesis, most notably the Image Analogies framework [Hertzmann et al. 2001], which introduced non-parametric mappings between paired images to transfer complex textures. Building on this logic, Johnson et al. [2011] developed CG2Real, leveraging large-scale image retrieval to inject real-world statistics into computer-generated images. While these early methods established the importance of data-driven anchors, they relied on manual feature matching and lacked the robust generative priors inherent in modern foundation models.

Subsequent efforts shifted toward deep generative architectures that replace manual matching with learned representations. Early image-to-image translation via conditional GANs [Isola et al. 2017; Liu et al. 2017; Yi et al. 2017; Zhu et al. 2017] refined these analogies into global mappings but often struggled with the photometric precision required for sim-to-real tasks. To bridge this gap, Chen et al. [2018] and Richter et al. [2021] demonstrated that incorporating engine-specific G-buffers, including depth and surface normals, significantly improves geometric grounding in complex sequences. Recent work [Wang et al. 2025] explores zero-shot diffusion-based realism enhancement for synthetic videos, demonstrating promising results on egocentric driving data. In this work, we study sim-to-real translation for videos containing rendered humans, where preserving character identity and articulated motion introduces additional challenges compared to primarily rigid-object scenes.

Our goal is to transform rendered 3D engine outputs into photorealistic video while preserving the underlying scene structure and

[Figure 2]

- Fig. 2. Overview of RealMaster. Our method consists of two stages: (1) Synthetic-to-Realistic Data Generation: Given a synthetic video, we edit sparse keyframes and propagate their appearance across the sequence using VACE, conditioned on edge maps from the input video, to create paired synthetic–realistic training data. (2) Model Training: We fine-tune an IC-LoRA over a text-to-video diffusion model on the paired data, enabling direct sim-to-real video translation at inference time.

dynamics. We achieve this through a two-stage approach: first, we construct high-quality paired training data via a data generation pipeline. Then, we train an IC-LoRA adapter that distills the data generation pipeline behavior into a model with improved generalization beyond the pipeline’s inherent constraints. An overview of our method is shown in Fig. 2.

- 3.1 Data Generation Pipeline

A central challenge in sim-to-real video translation is the lack of paired data aligning rendered engine outputs with corresponding photorealistic videos. To address this, we develop a pipeline that directly constructs photorealistic counterparts from rendered videos.

Image-based sim-to-real translation is more mature and reliable than its video equivalent. We therefore adopt a sparse-to-dense strategy: we edit a small set of keyframes using an image editing model to establish the target photorealistic appearance, and then propagate this appearance to intermediate frames using a video model with structural conditioning.

Keyframe Enhancement. Given a rendered video sequence, we first translate the first and last frames into the photorealistic domain using an off-the-shelf image editing model [Wu et al. 2025]. These enhanced keyframes serve as appearance anchors that define the target photorealistic look for the full sequence.

Edge-Based Keyframe Propagation. To propagate keyframe appearance to intermediate frames, we utilize VACE [Jiang et al. 2025], a video generative model that conditions generation on reference frames and structural signals.

Specifically, we extract edge maps from the input video and use VACE to generate the full video conditioned on the photorealistically edited keyframes and the corresponding edge maps. Edge

conditioning anchors generation to the input’s structure and motion, allowing VACE to propagate the keyframe appearance while preserving scene layout and dynamics across intermediate frames.

- 3.2 Model Training

We train a lightweight LoRA adapter that distills our data generation pipeline into a single model for sim-to-real video translation. Specifically, we adopt an IC-LoRA architecture on top of a pre-trained text-to-video diffusion backbone. During training, we concatenate clean reference tokens from the rendered input video with noisy tokens and optimize the model to denoise toward the corresponding photorealistic target. Training is lightweight, requiring only a small paired dataset and a few hours of fine-tuning on a single GPU.

At inference time, the resulting model avoids several constraints imposed by the pipeline. First, the pipeline requires access to both the first and last frames of a sequence, which makes streaming or autoregressive generation impractical. Second, because edits are anchored to sparse keyframes, the pipeline struggles to preserve the appearance and identity of objects and characters that emerge mid-sequence. Third, the image editing model can over-edit anchor frames, causing deviations from the input scene.

Overall, the trained model removes these inference-time constraints, enabling temporally coherent sim-to-real translation while preserving scene structure and character identity.

- 3.3 Implementation Details

For data generation, we sample clips from the SAIL-VOS [Hu et al. 2019] training set, upsampling them from 8fps to 16fps by repeating each frame to obtain 81-frame sequences at 800 × 1200 resolution. We edit the keyframes using Qwen-Image-Edit [Wu et al. 2025] and propagate their appearance to intermediate frames using VACE [Jiang et al. 2025] conditioned on edge maps. To improve identity consistency in the generated pairs, we filter out clips whose minimum ArcFace [Deng et al. 2019] cosine similarity between faces

[Figure 3]

- Fig. 3. Qualitative Results. We show representative GTA-V video sequences together with their edits produced by our method. These translated sequences demonstrate our method’s ability to produce photorealistic video while maintaining strict temporal coherence. Note the consistent appearance of materials, lighting, and fine details across frames. Best viewed zoomed in.

detected in the source and edited videos falls below 0.4. This process yields a training set of 1,216 clips.

For model training, we fine-tune Wan2.2 T2V-A14B [Wan et al. 2025]usingaLoRAadapterwitha rank of 32. Following IC-LoRA [Huang et al. 2024], we encode the rendered input as clean reference tokens with their timestep fixed to 𝑡=0, sharing positional encoding with the noisy tokens being denoised.

- 4 Experiments

input and edited videos. This metric captures high-level semantic and structural consistency between the rendered input and the photorealistic output.

For realism assessment, we use GPT-4o to rate the photorealism of edited frames on a scale from 1 to 10. For each video, we uniformly sample five frames and report the average score. We conduct this evaluation under two settings: (i) GPT-RSno-ref, where only the edited frame is provided to GPT-4o, and (ii) GPT-RSwith-ref, where the corresponding input frame is provided alongside the edited frame. This allows us to assess realism both in isolation and relative to the rendered input.

We perform a series of experiments to evaluate RealMaster. First, we compare our approach against strong baselines for video editing and sim-to-real translation. Second, we conduct ablation studies to assess the impact of key design choices in our approach.

To evaluate temporal consistency, we adopt the Temporal Flickering and Motion Smoothness metrics from VBench [Huang et al. 2023]. Temporal Flickering measures frame-to-frame visual instability, capturing abrupt appearance changes across consecutive frames, while Motion Smoothness assesses the coherence of motion over time.

- 4.1 Experimental Setup

We use a subset of 100 clips sampled uniformly from the SAIL-VOS validation set for our experiments. SAIL-VOS is recorded at 8fps, and we upsample it to 16fps by repeating each frame. The validation set contains diverse GTA-V scenarios featuring multiple interacting characters and visually complex scenes with many objects. We evaluate all methods using both automatic metrics and human evaluation. Both assess key aspects such as photorealism, input preservation, and temporal consistency.

Baselines. We compare our method against three strong video editing methods: Runway-Aleph [Runway 2025], LucyEdit [DecartAI 2025] and Editto [Bai et al. 2025]. Among these, Editto is explicitly trained for sim-to-real translation using synthetic-real pairs.

4.2 Qualitative Results

Automatic Metrics. We evaluate identity consistency, structure preservation, realism, and temporal consistency using complementary automatic metrics. To measure identity consistency, we compute the mean ArcFace similarity between faces detected in the input and edited videos. Specifically, we uniformly sample five frames per video, match the detected faces between the input and edited frames, and report the average cosine similarity of their ArcFace embeddings. We assess structure preservation by measuring the ℓ2 distance between DINO features extracted over all frames of the

As shown in Fig. 3, our method transforms rendered videos toward the photorealistic domain. The results preserve scene structure and motion, as well as character identity and appearance, while improving material and lighting realism.

These improvements are demonstrated in dynamic, cluttered scenes with multiple interacting characters, camera motion, and frequent occlusions, showing that the method successfully enhances realism despite challenging conditions that stress both structural precision and global semantic transformation.

[Figure 4]

- Fig. 4. Qualitative comparison with baseline methods. We compare our method against Runway-Aleph, LucyEdit, and Editto on three videos from the benchmark. The baselines either alter the original scene content, leading to identity drift and color shifts, or fail to produce sufficiently photorealistic results. In contrast, our method preserves scene structure and identity while improving the photorealism.

Method Realism Faithfulness Visual Quality

vs. Editto 63% 94% 78% vs. LucyEdit 93% 85% 93% vs. Runway-Aleph 64% 88% 70%

Overall 73% 89% 80%

- Fig. 5. User study. We report the percentage of trials where participants preferred RealMaster over each baseline for realism, faithfulness to the original video, and overall visual quality.

Table 1. Quantitative comparison against baselines. We compare our method against baseline approaches using automatic metrics on our benchmark.

Method GPT-RSno-ref↑ GPT-RSref↑ ArcFace↑ DINO↓ Temp. Flicker↑ Mot. Smooth.↑

Editto 5.104 3.838 0.204 41.79 0.972 0.972 Runway-Aleph 4.98 5.33 0.300 38.04 0.976 0.972 LucyEdit 3.48 4.20 0.375 36.68 0.976 0.986

RealMaster 5.296 7.33 0.473 30.28 0.976 0.973

- 4.4 User Study

To further validate our results, we conduct a user preference study comparing our method against the three baselines. In each trial, participants view the original rendered input together with two enhanced outputs (RealMaster vs. one baseline) and answer three questions assessing realism, faithfulness to the original video, and overall visual quality. In total, we collect 675 pairwise comparisons from 45 participants across the benchmark. As shown in Fig. 5, our method is preferred over all baselines across all three metrics.

- 4.5 Ablation Studies

Fig. 4presentsaqualitativecomparison with the baselines. Runway-

Aleph can improve realism but shifts object colors and does not preserve character identity. LucyEdit pushes the output toward a more game-like appearance than the input and alters many details of the original scene. Editto, despite training on paired synthetic–real data, deviates significantly from the content of the original scene. In contrast, RealMaster preserves structure and identity while substantially improving visual realism.

4.3 Quantitative Comparison

We conduct ablation studies to compare alternative design choices in our data generation pipeline and to quantify the additional gains from training a model on the generated data. For each sequence, we edit the first and last frames and explore different strategies for propagating their appearance to intermediate frames using VACE. Specifically, we compare: (i) editing additional anchor frames at regular intervals (one every 0.5 seconds) and conditioning VACE on these anchors, (ii) conditioning VACE on depth maps, and (iii) conditioning VACE on edge maps (our default pipeline). Finally, we compare these pipeline variants to our full method (RealMaster), which trains a LoRA-adapted model on data generated with edgebased propagation.

As shown in Table 1, our method outperforms all baselines on most evaluated metrics. It achieves the highest scores on both GPTRSno-ref and GPT-RSref, indicating superior photorealism both in isolation and relative to the rendered input. It also obtains the best ArcFace score and the lowest DINO score, demonstrating improved preservation of character identity and structural fidelity.

For temporal consistency, our method is competitive with the strongest baselines. It matches the best Temporal Flickering score and achieves comparable Motion Smoothness. While LucyEdit attains a slightly higher Motion Smoothness score, it does so by blurring the video, which reduces high-frequency detail and can inflate smoothness metrics while degrading structural precision.

In Fig. 6, we show qualitative comparisons of these propagation variants. Multiple anchors often introduce flickering, as inconsistencies across independent image edits are amplified during interpolation. Depth provides coarse geometric guidance but can miss

Overall, these results indicate that our method provides a better balance between photorealism, identity and structure preservation, and temporal consistency for sim-to-real video translation.

Frame 1 Frame 2 Frame 3

[Figure 5]

[Figure 6]

[Figure 7]

DepthDepthEdges(ours)Edges(ours)SourceSourceMult.AnchorsMult.Anchors

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

- Fig. 6. Data generation ablation. We ablate sparse-to-dense propagation for training pair generation, comparing multiple-anchor editing, depth conditioning, and edge conditioning for VACE. Multiple-anchor editing leads to temporal flickering and fluctuations in identity. Depth conditioning loses facial expression and facial structure, often failing to preserve identity. In contrast, edge conditioning preserves facial details more reliably and produces the most stable results across the sequence.

high-frequency cues important for identity and facial expressions. Edges more reliably preserve object boundaries and fine facial details, improving structural precision in the generated training pairs..

In Fig. 7, we compare inference with our trained model to direct use of the data generation pipeline. The model generalizes to cases where the pipeline fails, such as when an object first appears between the two boundary anchors, for which the pipeline has no appearance supervision. It also better preserves the appearance of the rendered input and avoids changes that are sometimes overly aggressive from the image editing model.

Table 2 quantifies these trends. Multiple Anchors and Depth perform worse on ArcFace and DINO, which reflect preservation of

Frame 1 Frame 2 Frame 3

[Figure 29]

[Figure 30]

[Figure 31]

SourceDataSourceDataRealMasterRealMaster

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

Fig. 7. Model vs. data pipeline comparison. We compare direct use of the data generation pipeline to inference with our trained model. Top: Our trained model produces a more faithful translation, preserving object identity, the color palette, and lighting. Bottom: The data generation pipeline fails when new objects (e.g., gloves) appear mid-sequence, since it relies only on two boundary anchors to define appearance.

identity and scene structure. Edges, which is the pipeline used to generate training pairs, yields the strongest pipeline scores on these metrics while maintaining stable Temporal Flickering and Motion Smoothness. Training RealMaster on the generated data pairs further improves all metrics, with the largest gains in structure and temporal consistency.

Table 2. Ablation study results. We compare multiple variants of the data generation pipeline and our trained model. The data variants include Multiple Anchors, which introduces anchor-frame edits every 0.5 seconds instead of two boundary anchors, and Depth and Edges, which condition VACE on depth maps or edge maps, respectively. RealMaster denotes the trained diffusion model learned from edge-based data. All variants are evaluated on the SAIL-VOS validation set.

Variant ArcFace↑ DINO↓ Temp. Flicker↑ Motion Smooth.↑

Multiple anchors 0.357 33.983 0.950 0.969 Depth 0.334 34.27 0.952 0.954 Edges 0.468 32.29 0.954 0.954

###### RealMaster 0.473 30.28 0.976 0.973

[Figure 47]

Fig. 8. Adding Weather Effects. RealMaster can add weather effects to a given scene by changing the textual prompt, despite not being trained for this capability. The model synthesizes dynamic phenomena such as falling rain droplets and snow accumulation.

- 5 Additional Applications

Beyond standard sim-to-real translation, our approach enables capabilities that would require significant effort to achieve in traditional rendering pipelines.

Dynamic Weather Effects. Video diffusion models inherently capture rich priors about natural phenomena, including weather dynamics. By simply modifying the text prompt at inference time, our model can introduce dynamic weather effects such as rain or snow into rendered scenes. Fig. 8 shows an example of this capability. These effects include realistic details that are challenging to synthesize in 3D engines, such as wet surface reflections, falling raindrops, and snow accumulation. Traditional simulators require careful modeling of these phenomena, including particle systems, shader modifications, and environmental lighting adjustments. In contrast, our approach provides these capabilities through the learned priors of the video model, without any additional engineering effort.

### Cross-SimulatorGeneralization. Althoughourmodelistrained

exclusively on data from SAIL-VOS, where the underlying engine is GTA-V, it generalizes to rendered videos from other simulators. We demonstrate this by applying the same trained model to scenes from the CARLA-LOC dataset [Han et al. 2024], which is collected in the CARLA driving simulator [Dosovitskiy et al. 2017] and has significantly different characteristics. Unlike SAIL-VOS, which features third-person views of people, videos in CARLA-LOC are captured from an egocentric driving perspective and focus on vehicles rather than pedestrians. CARLA uses a different rendering engine with its own lighting and material models. As shown in Fig. 9, our model successfully transforms these scenes into photorealistic video while preserving the original structure and dynamics, despite never seeing CARLA data during training. This cross-simulator generalization

[Figure 48]

Fig. 9. Generalization to a new dataset. We apply RealMaster, trained on SAIL-VOS, directly to CARLA videos without additional training. The latter uses a different rendering engine and features egocentric driving scenes with vehicles, in contrast to the third-person, character-centric scenes in the former. As can be seen, the model generalizes well to this different setting.

suggests that the model learns a general mapping from rendered to real appearance, rather than overfitting to the specific visual characteristics of the training domain.

6 Discussion, Limitations and Future Work

We have presented a framework for sim-to-real video translation that lifts rendered scenes into photorealistic video while preserving underlying scene structure and dynamics. Our work is grounded in the view that sim-to-real is not merely an instance of video editing or stylization, but a problem defined by the need to reconcile two competing objectives: exact structural fidelity and global photorealistic transformation. Seen through this lens, the limitations of existing approaches stem not from incidental design choices, but from an inherent imbalance between these goals. By treating generative video models not as free-form generators, but as learned secondstage renderers operating atop explicit 3D engines, our framework separates structural control from visual realization, enabling the injection of rich real-world appearance priors without sacrificing the determinism and editability that motivate graphics pipelines.

More broadly, our results suggest that realism in generated video is not solely a matter of appearance, but of consistency maintained over time. Preserving identity, materials, and fine scale details across frames proves as critical as improving texture or lighting. Achieving such consistency requires explicit inductive bias that anchors generation to the underlying rendered structure, rather than relying on implicit regularization. Our findings also highlight the importance of data construction, where rendered structure constrains paired supervision to ensure realism is learned without compromising geometric or temporal fidelity.

Despite these advances, our approach has several limitations. First, the realism of the output is ultimately bounded by the capabilities of current image editing models, which provide photorealistic anchors during data construction; as a result, the output may still

fall short of full photorealism.. In addition, while our method preserves motion and dynamics present in the rendered input, it does not explicitly reason about motion itself. In particular, complex human body locomotion, articulated gestures, and fine grained pose dynamics are inherited from the simulator rather than modeled or refined by our approach, which may limit realism in scenarios where the underlying animation is itself implausible.

Several research directions appear promising. A real-time streaming variant could enable causal sim-to-real translation with low latency, supporting interactive applications. Another direction is to move beyond appearance and address motion realism more directly. Incorporating learned priors over body dynamics and gestures could help correct rigid or synthetic motion, further narrowing the gap between simulated and real-world video.

Acknowledgments

We thank Ita Lifshitz and Daniel Garibi for their valuable contributions and support.

References

Qingyan Bai, Qiuyu Wang, Hao Ouyang, Yue Yu, Hanlin Wang, Wen Wang, Ka Leong Cheng, Shuailei Ma, Yanhong Zeng, Zichen Liu, Yinghao Xu, Yujun Shen, and Qifeng Chen. 2025. Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset. arXiv preprint arXiv:2510.15742 (2025). arXiv:2510.15742 [cs.CV] doi:10.48550/arXiv.2510.15742

Omer Bar-Tal et al. 2024. Lumiere: A Space-Time Diffusion Model for Video Generation. arXiv:2401.12945 (2024). Andreas Blattmann, Tim Dockhorn, Sumith Kulal, et al. 2023. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. arXiv:2311.15127 (2023).

Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. 2023. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 23206–23217.

Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun. 2018. Learning to See in the Dark. In CVPR.

Yuren Cong, Mengmeng Xu, Christian Simon, Shoufa Chen, Jiawei Ren, Yanping Xie, Juan-Manuel Perez-Rua, Bingbing Ni, Cihang Xie, and Andrea Vedaldi. 2023. FLATTEN: Optical Flow-Guided Attention for Consistent Text-to-Video Editing. arXiv preprint arXiv:2310.05922 (2023).

DecartAI. 2025. Lucy Edit: Open-Weight Text-Guided Video Editing. arXiv preprint

(2025).

Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. 2019. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 4690–4699. doi:10. 1109/CVPR.2019.00482

Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun.

2017. CARLA: An Open Urban Driving Simulator. arXiv:1711.03938 [cs.LG] https: //arxiv.org/abs/1711.03938

Patrick Esser et al. 2023. Structure and Content-Guided Video Synthesis with Diffusion Models. arXiv:2302.03011 (2023). Michal Geyer et al. 2023. TokenFlow: Consistent Diffusion Features for Consistent Video Editing. arXiv:2307.10373 (2023). Yuwei Guo et al. 2024. SparseCtrl: Adding Sparse Controls to Video Diffusion Models. arXiv:2311.16933 (2024).

Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, Eitan Richardson, Guy Shiran, Itay Chachy, Jonathan Chetboun, Michael Finkelson, Michael Kupchick, Nir Zabari, Nitzan Guetta, Noa Kotler, Ofir Bibi, Ori Gordon, Poriya Panet, Roi Benita, Shahar Armon, Victor Kulikov, Yaron Inger, Yonatan Shiftan, Zeev Melumian, and Zeev Farbman. 2026. LTX-2: Efficient Joint Audio-Visual Foundation Model. arXiv preprint arXiv:2601.03233 (2026). arXiv:2601.03233 [cs.CV] doi:10.48550/arXiv.2601.03233

Yuhang Han, Zhengtao Liu, Shuo Sun, Dongen Li, Jiawei Sun, Chengran Yuan, and Marcelo H. Ang Jr. 2024. CARLA-Loc: Synthetic SLAM Dataset with Full-stack Sensor Setup in Challenging Weather and Dynamic Environments. arXiv:2309.08909 [cs.RO] https://arxiv.org/abs/2309.08909

Aaron Hertzmann, Charles E. Jacobs, Nuria Oliver, Brian Curless, David H. Salesin, and William T. Freeman. 2001. Image Analogies. In Proceedings of the 28th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH). 327–340.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising Diffusion Probabilistic Models. NeurIPS (2020).

Yuan-Ting Hu, Hong-Shuo Chen, Kaiwen Hui, Jia-Bin Huang, and Alexander G. Schwing. 2019. SAIL-VOS: Semantic Amodal Instance Level Video Object Segmentation – A Synthetic Dataset and Baselines. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 3100–3110. doi:10. 1109/CVPR.2019.00322

Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. 2024. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775 (2024).

Ziqi Huang, Yinan He, Yufei Ma, Xiaolong Wang, Yu Wang, Yang Liu, Hao Li, Zheng-Jun Zha, and Lei Zhang. 2023. VBench: Comprehensive Benchmark Suite for Video Generative Models. arXiv preprint arXiv:2311.17982 (2023).

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. 2017. Image-to-Image Translation with Conditional Adversarial Networks. In CVPR. Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. 2025. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598 (2025).

Micah K. Johnson, Kevin Dale, Shai Avidan, Hanspeter Pfister, William T. Freeman, and Wojciech Matusik. 2011. CG2Real: Improving the Realism of Computer Generated Images using a Large Collection of Photographs. IEEE Transactions on Visualization and Computer Graphics (TVCG) 17, 9 (2011), 1273–1285.

Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. 2024. Anyv2v: A tuning-free framework for any video-to-video editing tasks. arXiv preprint arXiv:2403.14468 (2024).

Ming-Yu Liu, Thomas Breuel, and Jan Kautz. 2017. Unsupervised Image-to-Image Translation Networks. In NeurIPS.

Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. 2023. Video-P2P: Video Editing with Cross-attention Control. arXiv preprint arXiv:2303.04761 (2023). arXiv:2303.04761 [cs.CV] doi:10.48550/arXiv.2303.04761

Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. 2023. Videofusion: Decomposed diffusion models for high-quality video generation. arXiv preprint arXiv:2303.08320 (2023).

Eyal Molad et al. 2023. Dreamix: Video Diffusion Models are General Video Editors. arXiv:2302.01329 (2023).

Hao Ouyang, Qiuyu Wang, Yuxi Xiao, Qingyan Bai, Juntao Zhang, Kecheng Zheng, Xiaowei Zhou, Qifeng Chen, and Yujun Shen. 2024b. Codef: Content deformation fields for temporally consistent video processing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8089–8099.

Wenqi Ouyang, Yi Dong, Lei Yang, Jianlou Si, and Xingang Pan. 2024a. I2vedit: Firstframe-guided video editing via image-to-video diffusion models. In SIGGRAPH Asia 2024 Conference Papers. 1–11.

Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. 2024. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720 (2024).

Chenyang Qi, Xiaodong Cun, Shangchen Zhang, et al. 2023. FateZero: Fusing Attentions for Zero-shot Text-based Video Editing. In ICCV.

Bosheng Qin, Juncheng Li, Siliang Tang, Tat-Seng Chua, and Yueting Zhuang. 2023. InstructVid2Vid: Controllable Video Editing with Natural Language Instructions. arXiv preprint arXiv:2305.12328 (2023). arXiv:2305.12328 [cs.CV] doi:10.48550/arXiv. 2305.12328

Stephan R. Richter, Hassan Abu AlHaija, and Vladlen Koltun. 2021. Enhancing Photorealism Enhancement. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) (2021).

Runway. 2025. Introducing Runway Aleph. https://runwayml.com/research/ introducing-runway-aleph. Accessed: 2026-01-21.

Uriel Singer, Amit Zohar, Yuval Kirstain, Shelly Sheynin, Adam Polyak, Devi Parikh, and Yaniv Taigman. 2024. Video editing via factorized diffusion distillation. In European Conference on Computer Vision. Springer, 450–466.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Stefano Ermon, Sander Dieleman, and Jiquan Ngiam. 2021. Score-Based Generative Modeling through Stochastic Differential Equations. In International Conference on Learning Representations (ICLR).

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. 2025. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025).

Wen Wang, Yan Jiang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. 2023. Zero-Shot Video Editing Using Off-The-Shelf Image Diffusion Models. arXiv preprint arXiv:2303.17599 (2023). arXiv:2303.17599 [cs.CV] doi:10.48550/arXiv.2303.17599

Yifan Wang, Liya Ji, Zhanghan Ke, Harry Yang, Ser-Nam Lim, and Qifeng Chen. 2025. Zero-shot Synthetic Video Realism Enhancement via Structure-aware Denoising. arXiv preprint arXiv:2511.14719 (2025).

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng,

Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. 2025. Qwen-Image Technical Report. arXiv:2508.02324 [cs.CV] https://arxiv.org/abs/2508.02324

Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, et al. 2023. Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Video Editing. In ICCV.

Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. 2023. Rerender A Video: ZeroShot Text-Guided Video-to-Video Translation. In SIGGRAPH Asia 2023 Conference Papers. ACM, 1–11.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, and Jie Tang. 2024. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. arXiv preprint arXiv:2408.06072 (2024). arXiv:2408.06072 [cs.CV] doi:10.48550/arXiv.2408.06072

Zili Yi, Hao Zhang, Peng Tan, and Minglun Gong. 2017. DualGAN: Unsupervised Dual Learning for Image-to-Image Translation. In ICCV. Lvmin Zhang and Maneesh Agrawala. 2023. Adding Conditional Control to Text-toImage Diffusion Models. In ICCV.

Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. 2024. ControlVideo: Training-free Controllable Text-to-Video Generation. In International Conference on Learning Representations (ICLR). https://arxiv.org/abs/ 2305.13077

Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. 2017. Unpaired Imageto-Image Translation using Cycle-Consistent Adversarial Networks. In ICCV.

[Figure 49]

##### Fig. 10. Additional Qualitative Results.

[Figure 50]

- Fig. 11. Additional Qualitative Results.
- Fig. 12. Additional qualitative comparisons with baseline methods.

[Figure 51]

- Fig. 13. Additional generalization results on the CARLA-LOC dataset

[Figure 52]

## Supplementary Material

- A Failure Cases

We identify two main failure modes of RealMaster, illustrated in Fig. 14. First, when the scene contains many small, distant objects, the model tends to be overly conservative, producing only subtle photorealistic changesthat are hard to notice at full-frame resolution. This behavior is inherited from the image editing model, QwenImage-Edit, used in the data generation pipeline, which similarly struggles to enhance small objects. Second, scenes with fast camera or character motion lead to temporal artifacts in the output. This limitation is inherited from the base video diffusion model, which was not designed to handle large inter-frame displacements.

[Figure 53]

[Figure 54]

Fig. 14. Failure cases. Top: overly conservative output on a scene with small, distant objects. Bottom: temporal artifacts caused by fast camera and character motion.

- B Additional Implementation Details

This section provides additional implementation details for the data generation pipeline and for LoRA training.

B.1 Data generation pipeline

We sample 81-frame clips from the SAIL-VOS training set and upsample videos from 8fps to 16fps by repeating each frame at 800 × 1200 resolution. For each clip, we edit the first and last frames with QwenImage-Edit using the prompt "make it look photorealistic" and treat them as appearance anchors. We then use VACE to propagate anchor appearance to the intermediate frames, conditioned on an edge representation of each input frame.

To improve identity consistency, we filter generated pairs using ArcFace. We retain a clip only if the mean ArcFace cosine similarity between detected faces in the rendered input and in the generated output exceeds 0.4. Out of 3,050 initial clips, this filtering retains

Table 3. Training Hyperparameters. Summary of the configuration used for fine-tuning RealMaster.

#### Hyperparameter Value

Base Model Wan2.2 T2V-A14B LoRA Rank 32 Optimizer AdamW Learning Rate 1 × 10−4 Batch Size 8 Total Training Steps 1,200 Resolution 800 × 1200 Frames per Clip 81 Training Hardware 1 × H200

1,216 training clips, removing approximately 60% of the generated data.

- B.2 Model training

We fine-tune Wan2.2 T2V-A14B using an IC-LoRA training setup. We encode the rendered input clip as clean reference tokens with the timestep fixed to 𝑡=0, and share positional encoding with the noisy target tokens that are denoised toward the photorealistic target.

We provide a detailed summary of the hyperparameters used for training RealMaster in table 3.

- B.3 Baseline configurations

For all three baselines, Runway-Aleph, LucyEdit, and Editto, we use the prompt "make the video look photorealistic". All other settings follow the default configurations provided by the respective authors.

C Evaluation Metric Details C.1 GPT-RS

We use GPT-4o as a rubric-based judge to rate photorealism. We report two variants. GPT-RSwith-ref provides the rendered input frame as a reference and asks the judge to consider both photorealism and faithfulness. GPT-RSno-ref provides only the edited frame and asks the judge to score photorealism only. In both cases, the model returns valid JSON with a single integer key rating in the range 1 to 10.

You are an expert evaluator of GTA-to-photoreal image translation. You will be shown TWO images:

- 1) The original GTA game frame
- 2) The edited image produced by a model attempting photorealism Your task: Evaluate how successful the edited image is as a faithful,

photorealistic transformation of the original GTA frame.

Faithfulness requirements:

- - Same scene layout and camera viewpoint
- - Same object positions, object colors and proportions
- - No hallucinated, removed, or swapped objects
- - No major geometric changes (bending, drifting, resizing)

Photorealism focus:

- - Geometry stability (warping, melting, bending)
- - Lighting and shadows (direction, contact, consistency)

- - Materials and textures (plastic look, over-smoothing, repetition)
- - Fine detail (grain, sharpness, depth of field)
- - Text and signage (legible, stable, non-gibberish)
- - Neural artifacts (halos, ghosting, ringing, checkerboard)

Score the QUALITY OF THE EDIT, considering BOTH:

- 1) Faithfulness to the original GTA image
- 2) Photorealism of the edited image Scale (1-10): 10 = Faithful and indistinguishable from real camera footage 8-9 = Faithful with minor realism flaws visible on close inspection 6-7 = Mostly faithful; noticeable synthetic artifacts or small

inconsistencies 4-5 = Partially faithful; clear mismatches or strong realism artifacts

1-3 = Unfaithful or failed transformation (hallucinations, scene changes, or severe artifacts)

Return valid JSON only with a single key: rating (integer 1-10).

- Image 1 is the original GTA frame.
- Image 2 is the edited output attempting photorealism. Rate how successful the edit is.

You are an expert in video synthesis results. Your task is to judge whether the provided image could plausibly be

a real

camera frame. Score photorealism only. Ignore aesthetics or artistic quality. Scale (1-10, photorealism): 10 = Indistinguishable from real camera footage 8-9 = Looks real at a glance; minor flaws on close inspection 6-7 = Mixed realism; noticeable synthetic artifacts but partially

plausible

4-5 = Clearly synthetic; CG or translation artifacts are obvious, but image is coherent

1-3 = Obviously synthetic or broken; severe artifacts make it

unmistakably fake Focus on:

- - Geometry stability (warping, melting, bending)
- - Lighting and shadows (direction, contact, consistency)
- - Materials and textures (plastic look, over-smoothing, repetition)
- - Fine detail (grain, sharpness, depth of field)
- - Text and signage (legible, stable, non-gibberish)
- - Neural artifacts (halos, ghosting, ringing, checkerboard) Be conservative: if multiple strong artifacts exist, do not score

above 7. Return valid JSON only with a single key: rating (integer 1-10).

This image is a frame produced by a model that converts GTA gameplay into realistic video. Rate how realistic it looks as real camera footage.

