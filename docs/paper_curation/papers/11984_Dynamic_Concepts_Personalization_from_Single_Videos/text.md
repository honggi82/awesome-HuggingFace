# arXiv:2502.14844v1[cs.GR]20Feb2025

Dynamic Concepts Personalization from Single Videos

RAMEEN ABDAL OR PATASHNIK IVAN SKOROKHODOV WILLI MENAPACE ALIAKSANDR SIAROHIN SERGEY TULYAKOV DANIEL COHEN-OR KFIR ABERMAN

SNAP RESEARCH

https://snap-research.github.io/dynamic_concepts

[Figure 1]

Fig. 1. We personalize a video model to capture dynamic concepts – entities defined not only by their appearance but also by their unique motion patterns, such as the fluid motion of ocean waves or the flickering dynamics of a bonfire (left). This enables high-fidelity generation, editing, and the composition of these dynamic elements into a single video, where they interact naturally (right).

remains an open problem. Unlike images, videos introduce an additional temporal dimension, making personalization significantly more challenging. In particular, video concepts are inherently dynamic, encompassing both appearance and motion, which must be learned and represented cohesively.

Personalizing generative text-to-image models has seen remarkable progress, but extending this personalization to text-to-video models presents unique challenges. Unlike static concepts, personalizing text-to-video models has the potential to capture dynamic concepts – entities defined not only by their appearance but also by their motion. In this paper, we introduce Setand-Sequence, a novel framework for personalizing Diffusion Transformers (DiTs)–based generative video models with dynamic concepts. Our approach imposes a spatio-temporal weight space within an architecture that does not explicitly separate spatial and temporal features. This is achieved in two key stages. First, we fine-tune Low-Rank Adaptation (LoRA) layers using an unordered set of frames from the video to learn an identity LoRA basis that represents the appearance, free from temporal interference. In the second stage, with the identity LoRAs frozen, we augment their coefficients with Motion Residuals and fine-tune them on the full video sequence, capturing motion dynamics. Our Set-and-Sequence framework resulting in a spatio-temporal weight space effectively embeds dynamic concepts into the video model’s output domain, enabling unprecedented editability and compositionality, and setting a new benchmark for personalizing dynamic concepts.

In this work, we propose a novel approach to personalizing generative text-to-video models, focusing on the idea of dynamic concepts – objects or subjects characterized by their entangled appearance and motion. Describing dynamic concepts through text is challenging, hence, embedding them into the output domain of video model and representing them as tokens, facilitates a wide range of editing and compositional tasks.

Our work is built upon the state-of-the-art Diffusion Transformers (DiTs) architecture for video generation [36, 38], which processes spatial and temporal tokens simultaneously. Unlike video generators using factorized spatial and temporal modeling (UNet-based) [6] suffering from rigid artifacts, this joint spatio-temporal modeling is necessary for high-quality video generation [38]. While DiTs achieve superior quality, they lack the innate inductive bias to disentangle spatial and temporal features. This absence of built-in separation poses a challenge for embedding dynamic concepts effectively in the model’s weight space.

1 Introduction

The advent of generative models has revolutionized content creation, enabling the synthesis and manipulation of high-quality visual media with remarkable fidelity. Recent advances in text-toimage [45, 49, 54] and text-to-video [4] models have further expanded these capabilities, opening up unprecedented opportunities for creative expression and personalization.

Furthermore, directly fine-tuning low-rank adapters (LoRAs) on a single video often fails to capture both appearance and motion, resulting in a non-reusable representation that fails to generalize across diverse contexts or support meaningful compositionality of dynamic concepts. To address these limitations, we introduce a novel framework called Set-and-Sequence, designed to impose a spatialtemporal structure in the weight space, enabling the representation of dynamic concepts in the weights space.

Personalization has been a well-established area of research in the image domain, allowing models to learn user-specific concepts that can be customized, edited, and composed into diverse contexts [16, 51]. However, while video models have shown improvement in quality and capability, the task of personalizing these models

[Figure 2]

- Fig. 2. Set-and-Sequence framework operates in two stages: (i) Identity Basis: We train LoRA Set Encoding on a unordered set of frames extracted from the video, focusing only on the appearance of the dynamic concept to achieve high fidelity without temporal distractions. (ii) Motion Residuals: The Basis of the Identity LoRAs is frozen and the coefficient part is augmented with coefficients of LoRA Sequence Encoding trained on the temporal sequence of full video clip, allowing the model to capture the motion dynamics of the concept.

The proposedSet-and-Sequence framework operates in two stages:

(i) Identity Basis: We train LoRAs on a static unordered set of frames extracted from the video, focusing solely on the appearance of the dynamic concept to achieve high fidelity without temporal distractions. (ii) Motion Residuals: The Basis of the Identity LoRAs is frozen and the coefficient part is augmented with new coefficients trained on the temporal sequence of full video clip, allowing the model to capture the motion dynamics of the concept.

This two-stage approach unlocks transformative capabilities in video generation. For the first time, we demonstrate seamless scene composition and adaptation with preserved motion and appearance. Tasks such as blending disparate video components—e.g., combining the fluid motion of ocean waves with the flickering dynamics of a bonfire—are achieved with unprecedented fidelity as shown in Fig. 1 and in the supplementary video. Moreover, our framework enables intuitive editing of camera motion, refining expressions, and introducing localized changes, all driven by text prompts. These advancements represent a significant leap in compositionality, scalability, and adaptability, setting a new benchmark for personalized generative video models.

2 Related Work

- 2.1 Foundational Video Models.

Foundational video models, such as Imagen Video [23], Sora [38], CogVideoX [60], Veo2 [12], MovieGen [40] and others have made significant strides in synthesizing visually stunning and semantically aligned videos from textual descriptions. They were originally based on U-net-like [50] architectures [6, 20, 24, 55] and were extending image generators to video synthesis by training additional temporal layers to model dynamics. However, in the pursuit of greater scalability, the community switched to transformer-based

backbones with joint spatio-temporal modeling (e.g., [26, 36, 38]), which quickly became the dominant paradigm for large-scale video generation (e.g., [29, 38, 40, 60]). While these models excel at generating coherent content, they primarily rely on generic motion trajectories, limiting their ability to capture nuanced human expressions, individualized mannerisms, or complex dynamic interactions within a shared scene [36, 38]. These limitations highlight the need for methods capable of personalization, dynamic scene composition, and precise editing in generative video models. To address these challenges, we build on the video DiT (DiT version of Snap Video [36]) architecture and extend its capabilities with our proposed Set-and-Sequence framework, enabling the representation and compositionality of dynamic concepts with unprecedented fidelity and adaptability.

2.2 Video Personalization and Motion Representation.

While personalization in image generation has seen significant advancements—enabling identity preservation, stylization, and tailored manipulation [16, 27, 31, 51, 52]—video personalization remains relatively underexplored. In the video domain, personalization methods predominantly build upon UNet-based architectures [3, 22, 34, 59, 62, 66], inheriting their shortcomings. Furthermore, approaches in this domain can be broadly categorized into three domains. First, works like Token Flow [18] focus on video stylization [7, 28, 30, 63]. Second, methods like DreamVideo [58] and others [5, 35, 64] emphasize extracting motion dynamics from several videos to perform motion transfer. Third, approaches like Customize-a-Video [47] , Fate/Zero [41], and DreamMix [37] perform local editing on single videos by optimizing specific parts. Although promising, these methods, such as Customize-a-Video [47] are architecture specific and operate on the assumption that motion

[Figure 3]

- Fig. 3. Local and Global Editing. Our Set-and-Sequence framework enables text-driven edits of dynamic concepts while preserving both their appearance and motion. Edits can be global (e.g., background and lighting) or local (e.g., clothing and object replacement), ensuring high fidelity to the original dynamic concepts.

and appearance are disentangled, optimizing distinct LoRA [25] modules or layers for each. This rigid separation often leads to artifacts, losing fidelity and contextual realism. Moreover, they primarily target applications like motion transfer, diverting focus from video personalization that captures the inherent entanglement of appearance and motion in dynamic concepts. To solve this, we introduce a shared spatio-temporal weight space that cohesively encodes dynamic concepts using a two-stage LoRA [25] training.

- 2.3 Scene Composition in Video Models.

Scene composition and dynamic editing remain significant challenges in video synthesis due to the complexities of maintaining temporal coherence and contextual fidelity. Approaches like BreakA-Scene [2] enable concept-level blending but are limited to static, image-like representations, relying heavily on predefined masks and cross-attention mechanisms. In video models, scene composition often involves generating composed images using personalized text-to-image methods [42, 57] and then applying image-to-video techniques to synthesize motion dynamics on top of the static image [6, 9, 10, 19, 21, 46]. However, these models face several inherent limitations. First, they depend on powerful image composition models capable of blending multiple objects into a cohesive scene effectively ignoring the motion [6, 10]. Second, they lack awareness of object-specific attributes, such as viewpoints, dynamic evolution of motion, and spatial relationships [6, 10]. Third, they fail to account for nuanced expressions and intricate motion patterns that cannot be adequately captured through text descriptions alone [9, 10]. These shortcomings render such models incompatible with the goals of video personalization and advanced compositionality. For the first time, we demonstrate advanced compositionality by merging disparate dynamics, such as fire and water, while capturing both appearance as well as motion from single videos. Our approach overcomes the limitations of previous methods, offering a unified framework that enables personalization of dynamic concepts.

[Figure 4]

Fig. 4. Stylization. Top: Stylization of dynamic concepts achieved by reweighting the identity basis. Bottom: Stylization and motion editing performed using prompt derived from the video in the top row.

3 Method

We propose Set-and-Sequence (See Fig. 2), a novel framework for personalizing text-to-video models using dynamic concepts extracted from single-video examples. Our approach learns these dynamic

[Figure 5]

- Fig. 5. Dynamic Concepts Composition. Composition results achieved by our framework showcasing seamless integration of dynamic concepts. with each concept color-coded for clarity. For a more comprehensive demonstration, refer to the supplementary videos.

concepts as a decomposition of appearance and motion into a unified spatio-temporal weight space inspired by the state-of-the-art generators [36, 38]. We impose this weight space in DiT-based diffusion architecture [39], an architecture that does not explicitly separate spatial and temporal features unlike UNet-based architectures [47], resulting in seamless compositionality, editing, and adaptation. Central to our framework is a two-stage learning technique. In the first stage, Identity Basis Learning, we train Low-Rank Adaptation (LoRA) layers on an unordered set of video frames, extracting a static, motion-independent identity basis that captures the appearance of the concept. In the second stage, Motion Residual Encoding, the identity basis is augmented with motion dynamics by fine-tuning coefficients on the full video sequence. We employ additional regularizations and employ text conditioning at each stage, using static prompts for appearance learning and a combination of static and dynamic prompts for encoding motion dynamics. At inference time, this enables intuitive reprompting, recomposing, and editing of content using only text descriptions, facilitating advanced personalization and dynamic scene composition.

- 3.1 Preliminaries

Video Diffusion with Flow Matching Loss. Our framework builds on a video diffusion model trained with a flow matching loss [1, 32]. This objective aligns the predicted and true velocity fields and is defined as:

2

𝜕x𝑡 𝜕𝑡

, (1)

Lflow = Ex,𝑡 v𝜃 (x𝑡,𝑡) −

2

where x𝑡 represents the perturbed data at time 𝑡, v𝜃 is the predicted velocity field, and 𝜕𝜕𝑡x𝑡 is the true data flow.

Low-RankAdaptation(LoRA). LoRAfine-tunesapretrainedmodel by introducing low-rank updates to its weight matrices:

W′ = W + ΔW, ΔW = AB, (2)

where W is the original weight matrix, A ∈ R𝑚×𝑟 and B ∈ R𝑟×𝑛 are low-rank matrices with rank 𝑟 ≪ 𝑚𝑖𝑛(𝑚,𝑛).

LoRA’s parameter efficiency and adaptability make it an ideal choice for disentangling identity and motion in video data.

3.2 Stage 1: Identity Basis Learning

In the first stage, we extract static identity features from an unordered set of frames as images in the input video. This stage creates a time-independent identity representation, forming the foundation for subsequent motion encoding. By decomposing and separating identity from motion, it enables the independent editing of appearance and motion during inference. The LoRA weight modification for this stage is defined as:

### W′ = W + A1B1, (3)

where A1 ∈ R𝑚×𝑟 and B1 ∈ R𝑟×𝑛 represent the low-rank parameters capturing the identity. Static text tokens Tstatic are used to describe the subject’s appearance (e.g., as an illustration in Fig. 2, “a [v] person”). In practice, for efficient editing and composition; appearance, background and expression information is also included in the static prompts to make it detailed (See supplementary materials). The [v] token is initialized with zeros. The resulting conditional velocity field is defined as:

v𝜃 (x𝑡,𝑡;Tstatic). (4)

The identity-specific flow matching loss ensures accurate reconstruction of static features. The learned parameters A1 and B1 are obtained by solving the following optimization problem:

2

𝜕x𝑡 𝜕𝑡

(A1, B1) = arg min A1,B1

Ex,𝑡 v𝜃 (x𝑡,𝑡;A1, B1, Tstatic) −

. (5)

2

This stage creates a robust, low-dimensional basis for identity representation.

[Figure 6]

- Fig. 6. Comparison with baselines. Comparison of our method with baseline approaches (NewMove [35], DreamVideo [58], DB-LoRA [51, 53], and DreamMix [37]) on two editing scenarios: changing the background and shirt, and adding a glass. Our method demonstrates superior adherence to the prompt while preserving the subject identity, outperforming the baselines.

- 3.3 Stage 2: Motion Residual Encoding Building upon the static identity basis established in Stage 1, the

second stage introduces an additional low-rank matrix B2, encoding motion as a residual deformation on top of the identity. This stage captures the temporal evolution of motion dynamics, enabling independent manipulation and composition of motion during inference. The weight modification for this stage is defined as:

W′ = W + A1B1 + A1B2, (6)

where A1 and B1 remain fixed to preserve identity, and B2 ∈ R𝑟×𝑛 encodes motion-specific deformations. Motion encoding uses a union of static and motion-specific text tokens:

Tmotion := Tstatic ∪ Tdynamic, (7)

where Tdynamic describes motion attributes (e.g., as illustrated in the Fig. 2, "... in [u] motion"). In practice, we also augment the dynamic part with the course action e.g., "... dancing with legs up" and camera motion e.g., "... as the camera zooms in" (See supplementary materials). The predicted velocity field is conditioned on both text components:

v𝜃 (x𝑡,𝑡;Tmotion). (8)

The flow matching loss for motion encoding ensures that the motion is reconstructed as a deformation. The learned parameters B2 is obtained by solving the following optimization problem:

B2 = argmin

B2

Ex,𝑡 v𝜃 (x𝑡,𝑡;A1, B1, B2, Tmotion) −

𝜕x𝑡 𝜕𝑡

2

2

. (9)

This stage ensures that motion is complementary to identity, enabling robust, adaptable representations for dynamic content.

- 3.4 Regularization

Regularization is integral to our framework, ensuring robust training, preventing overfitting, and maintaining efficient representations of identity and motion. We employ four distinct regularization strategies to achieve these goals:

- 3.4.1 Prior Preservation [48, 51]. To ensure fidelity to the pretrained base model, we regularize both training stages to reconstruct videos generated by the base model. We sample videos with text having two parts i.e. appearance and motion, for example, in case of humans, "A man in a blue t-shirt. He is walking in a park". These prompts generalize for non-human dynamic concepts as well.

- 3.4.2 High Dropout Regularization for High-Rank LoRA. Training on a single video introduces challenges of underfitting with lowrank LoRA and overfitting with high-rank LoRA. To address these issues, we adopt a high-rank LoRA configuration combined with selective dropout applied exclusively to the B matrix in the LoRA update. Dropout regularization is applied to B LoRAs in both stages as follows:

B′ = B ⊙ M, (10) where M is a binary mask with dropout probability 𝑝 (e.g., 𝑝 = 0.8). This selective dropout ensures that Aℎ remains stable, providing a consistent basis for encoding appearance and motion. By introducing sparsity in the learned coefficients, this approach mitigates overfitting while encouraging exploration of diverse parameter combinations facilitating applications like dynamic concept composition (See Fig. 1).

- 3.4.3 Context-Aware Regularization. To enhance robustness and generalization,weincorporatetexttoken masking and self-conditioning as complementary regularization techniques.

Text Token Masking: Random tokens in the input text are masked during training, requiring the model to infer the missing information from the remaining context. This prevents overfitting to specific token patterns and improves adaptability to diverse or incomplete prompts which we leverage for editing and re-composition.

Self-Conditioning: Inspired by [8], intermediate model outputs are reintroduced as inputs during subsequent steps, enabling iterative refinement. This feedback loop improves temporal consistency, ensuring stable identity and motion across frames.

- 4 Experiment Settings 4.1 Evaluation Dataset

We evaluate our framework on a curated dataset of human-centric videos, as identity preservation poses a significant challenge in editing and composition. Unlike general objects, humans are highly sensitive to subtle inconsistencies in appearance, motion, and expressions, making deviations immediately perceptible. The dataset includes five distinct identities performing actions such as dancing and walking, as well as scenario with two identities interacting in the same video. This setup tests our framework’s ability to preserve identity, maintain motion coherence, and achieve compositional

[Figure 7]

- Fig. 7. Ablation. Ablation of design choices on the editing task of adding a different shirt and background. Low-rank LoRA (LoRA-1) results in underfitting, failing to capture sufficient detail, while high-rank LoRA (LoRA-8) overfits, compromising adaptability. Our two-stage approach with added regularization achieves a balanced trade-off, preserving both fidelity and editability.

fidelity compared to existing baselines. We evaluate our method using this dataset on the local and global editing tasks.

- 4.2 Baselines.

We evaluate our method against several baselines, including stateof-the-art UNet-based approaches, architecture-agnostic methods, and LoRA-based variations as part of an ablation study. Among UNet-based models, we compare against DreamVideo [58] and NewMove [35] which are state-of-the-art frameworks for video personalization. Additionally, code for methods like Customize-a-Video [47] is unavailable, restricting direct comparison. We adapt UNet-based DreamMix [37] (Code not available) to the DiT architecture, enabling mixed image-video training to benchmark its performance. To ensure a broader evaluation, we include architecture-agnostic methods like DreamBooth LoRA [51, 53] and Textual Inversion [17]. In our ablation study, we compare several LoRA setups to highlight the efficacy of our approach.

- 4.3 Evaluation Metrics

We employ the following metrics to quantitatively assess the quality of the generated videos:

Semantic Alignment. We utilize CLIP-Text Similarity [43] (C-T) to measure the alignment between the generated video and the input text prompt. This metric computes the cosine similarity between the text embedding and the aggregated embeddings of all video frames, providing a global assessment of semantic consistency.

Identity Preservation. Maintaining identity consistency is crucial, especially in videos featuring human subjects. We utilize ArcFace Identity Similarity (ID) [13] to measure how well the identity of a person is preserved between the generated and reference videos.

Reconstruction Fidelity. To quantify pixel-level fidelity, we compute the Mean Squared Error (MSE) between corresponding frames in the generated and reference videos.

Temporal Coherence. Ensuring smooth transitions and motion consistency across frames is critical. For Temporal Coherence (TC), we compute as CLIP image embeddings on all generated frames and

report the average cosine similarity between all pairs of consecutive frames.

5 Experiment Results

- 5.1 Quantitative Evaluation

In order to show the effectiveness of our approach, we first ablate with the baselines and show that our two stage approach is essential for the model to integrate the dynamic concepts into the prior. Table 1 and Fig. 7 illustrates the results of our ablation study. They are evaluated on two editing tasks i.e. "changing the shirt and background" and "holding a glass". When using a low-rank LoRA, we observe a significant loss in identity preservation due to underfitting, as the rank is insufficient to model the complexity of dynamic concepts. Conversely, a high-rank LoRA overfits, resulting in diminished adaptability to new prompts. In contrast, our two-stage approach strikes a balance by using the same rank to separately train an Identity Basis and Motion Residual. This enables better adaptation to novel prompts while preserving both appearance and motion. Finally, with the added dropout regularization discussed in Sec 3.4, our framework achieves seamless integration of local edits (e.g., clothing changes) and global edits (e.g., background replacement) while maintaining motion fidelity, as shown in the supplementary videos.

To evaluate the effectiveness of our method with state-of-the-art approaches, we compare the results of various baseline methods in Fig. 6. Each method is provided with the same editing prompt, and we assess their ability to preserve both the identity and adherence to the textual description. As shown in the figure, our framework achieves high fidelity in identity preservation and text adherence, significantly outperforming other methods. For motion preservation and coherence, we provide additional results in the supplementary videos, highlighting the seamless integration of dynamic motion with edits. Table 2 shows the quantitative analysis of the results, where our two-stage set-and-sequence results in a better trade-off compared to other methods.

- 5.2 Qualitative Results

Our framework demonstrates significant advancements in both editing and composition of dynamic concepts, setting a new benchmark in personalized video generation. In this section, we explore two primary applications: editing and composition.

5.2.1 Editing. Our framework excels in both local and global editing, as demonstrated in Fig. 3. The core objective is to capture intricate expressions and mannerisms while adapting dynamic concepts to new scenarios. For example, in Fig. 3, our method models complex athletic movements that current video generation models fail to replicate. These results are further demonstrated in the supplementary videos, where the coherency of motion and appearance across edits is observed. Moreover, our framework supports a wide range of edits, such as changing expressions, age, and camera angles, or seamlessly integrating new objects and backgrounds into the scene. This level of control is achieved without compromising motion fidelity or identity preservation. For instance, by adjusting the weights of the LoRAs or modifying associated text prompts, our framework can produce creative outputs such as Pixar-style (See

Fig. 4) characterizations or adapt the same dynamic concept to a completely different context. The ability to selectively adapt parts of a video, as shown in the supplementary videos, further emphasizes the flexibility of our approach.

- 5.2.2 Composition. One of the most significant contributions of our framework is the ability to compose dynamic concepts in novel and diverse settings. Leveraging the shared spatio-temporal weight space and our regularization techniques, we use single examples of multiple dynamic concepts and jointly train them using a unified identity basis and associated motion residual module. Here, after the first stage, identity basis jointly represents multiple concepts and in the second stage, the motion deformations for each dynamic concept is learned jointly. For instance, we demonstrate the composition of multiple entities, such as intricate movements, ocean waves and a bonfire, within the same scene. Fig. 5 illustrates how our method ensures coherence across these entities while maintaining their distinct characteristics. Additionally, we address challenges such as identity leakage (see Fig. 8), which arises when semantically similar concepts are combined (humans in our case). To mitigate this, we employ a simple yet effective strategy of additionally training on stitched videos as a regularization. These videos are trained less frequently and although not necessary, removing backgrounds in the such stitched videos helps further in composing results involving two or more humans. More examples are provided in the supplementary videos.

- Table 1. Ablation of Baselines. Table evaluating Mean Square Error (MSE), Identity Preservation (ID), CLIP-T (C-T), and Temporal Coherency (TC) on the editing task. Our method demonstrates better reconstruction-edibility trade-off.

Method MSE ↓ ID ↑ C-T ↑ TC ↑

LoRA-1 0.0432 0.622 0.226 0.9974 LoRA-8 0.0223 0.703 0.224 0.9969 + Two-Stage 0.0461 0.629 0.250 0.9971 + Reg 0.0221 0.680 0.239 0.9972

- Table 2. Editing Task Evaluation. Table evaluating Mean Square Error (MSE), Identity Preservation (ID), CLIP-T (C-T), and Temporal Coherency (TC) on the editing task. Our method achieves a superior reconstructioneditability trade-off compared to the competing approaches.

## Method MSE ↓ ID ↑ C-T ↑ TC ↑

Tex-Inv 0.0714 0.145 0.201 0.9927 DB-LoRA 0.0223 0.703 0.224 0.9969 NewMove 0.2223 0.270 0.204 0.9914 DreamVideo 0.2021 0.118 0.218 0.9657 DreamMix 0.0429 0.579 0.226 0.9965 Ours 0.0221 0.680 0.239 0.9972

[Figure 8]

Fig. 8. Stitched Example. To address identity leakage when generating multiple identities and motions from multiple videos (Second Column), we augment training with stitched examples by combining videos side by side to generate new compositions with preserved identities (Third Column).

5.3 User Study

To evaluate the quality of identity preservation, motion fidelity, and adherence to prompts on the editing task, we conducted a user study with 10 participants. We omit UNet based methods due to the overall lower quality (See Supplementary Videos). Participants were presented with pairs of videos generated by different methods and were asked to separately select the video that performed better in terms of identity preservation, motion fidelity, and adherence to the prompt. The results of the user study, summarized in Table 3, demonstrate that our method consistently outperforms competing approaches by achieving a better tradeoff on the editing task.

Table 3. User Study. User study results comparing methods on Identity Preservation (ID), Motion Preservation (MP), Adherence to Prompt (AP), and Overall Preference of the edits (OP). Preference is computed in percentages.

## Method IP MP AP OP

Ours vs DreamMix 87% 88% 98% 100% Ours vs LoRA-1 99% 95% 94% 100% Ours vs LoRA-8 (DB-LoRA) 78% 75% 98% 98% Ours vs Two-Stage 86% 97% 76% 90%

- 6 Limitations

While our framework achieves state-of-the-art performance in video personalization and dynamic concept modeling, it does have limitations. The training process, which involves LoRA optimization with additional regularization, can be computationally intensive. An encoder based approach would be an ideal solution for future work. Additionally, while the method captures most motions with high fidelity, it may struggle with high-frequency or highly complex motion patterns, such as erratic or rapid movements, where temporal consistency could be further improved. These challenges present opportunities for future work to enhance efficiency, speed, and robustness in handling more dynamic scenarios.

- 7 Conclusion

We introduced a novel framework for personalized video generation that captures dynamic concepts using a two-stage Set-and-Sequence paradigm i.e. the first stage of identity encoding and then learning coupled motion residuals on the top. By embedding these concepts into this unified spatio-temporal weight space, our method achieves

high fidelity in appearance preservation, motion coherence, and text adherence, surpassing state-of-the-art baselines. The evaluations demonstrated versatility of our framework in editing and composition, while maintaining identity and motion fidelity. The ability to compose and adapt dynamic concepts in novel ways highlights the transformative potential of our approach. This work addresses long-standing challenges in video personalization and sets a new benchmark for personalized and compositional video generation.

Acknowledgments

We thank Gordon Guocheng Qian and Kuan-Chieh (Jackson) Wang for their feedback and support.

- 8 Architecture and Training Details

We build our framework as a video diffusion model operating in the latent space of a video autoencoder.

The latent representation is based on a causal video autoencoder following the architecture of MAGVITv2 [61]. Our autoencoder presents a high compression ratio of 8 × 16 × 16 on the time and spatial dimensions, respectively, with a bottleneck dimensionality of 32 channels. Full model details are given in Table 4.

The video diffusion backbone consists in a 11.5B parameters DiT [39] detailed in Table 5. The model is organized into 32 DiT Blocks with a hidden dimensionality of 4096 channels, each of which consists of a self-attention layer, followed by a cross-attention layer to attend to text conditioning, and a final MLP with an expansion factor of 4×. To further reduce the input dimensionality, a ViTlike [14] 1×2×2 input patchification operation is applied, increasing the effective video compression factor to 8 × 32 × 32. This allows modeling of a 121 frames 1024×576px video using only 9216 tokens, and enables the use of full 3D self-attention for high-quality motion modeling [40] without incurring a large computational penalty associated with its quadratic cost, which is further reduced by the use of a 6144 tokens attention window. Each self-attention block consists of 32 attention heads with QK-Normalization [15] and is augmented with 3D-RoPE [56] embeddings, separately applied to the attention head channels in a ratio of 2 : 1 : 1 for the temporal and spatial dimensions, respectively. Text prompts are encoded by the T5 [44] model and combined with video tokens through the crossattention layers. Following [39], diffusion timestep information is injected through modulation.

We perform pretraining of the model by jointly training it on a mixture of image and video data with a resolution of 512 or 1024 px, aspect ratios of 16 : 9 and 9 : 16 for videos, and 16 : 9, 1 : 1, and 9 : 16 for image content. We adopt a progressive training strategy on the video temporal dimension, progressively extending the number of frames from 17 to 121, corresponding to 5 seconds at our fixed framerate of 24 fps. During the pretraining stage, we use the AdamW [33] optimizer with a fixed learning rate of 1𝑒 − 4, 10k steps warmup, a weight decay of 0.01, 𝛽 = [0.9, 0.99], 𝜖 = 1𝑒 − 8 and a total of 822k training steps. Model training is accelerated through flash attention [11] with bf16 precision, and is distributed on 256 H100 GPUs using FSDP [65].

The training details of our Set-and-Sequence approach are summarized in Table 6. The model is trained in two stages. For a single

video, Stage I of the two-stage approach without regularization is trained for 150 steps, while Stage II is trained for 400 steps, requiring a total of approximately 90 minutes to converge. However, for our final method, which incorporates dropout regularization, convergence is slower. The number of training steps varies based on the complexity and number of videos. For a single video, Stage I is trained for 600 steps and Stage II for 900 steps. For multiple dynamic concepts and videos with complex motions, such as athletic dance sequences, Stage II requires extended training of 2k to 2.5k steps. To optimize text tokens effectively while avoiding overfitting, we use a lower learning rate of 1𝑒 − 5. We observe that our method is able to generalize without optimizing for these special tokens as well. Our training uses the AdamW optimizer [33] with a constant learning rate of 1𝑒 − 4. To ensure stable training, we set 𝛽 = [0.9, 0.99], apply a weight decay of 0.01, and use gradient clipping with a value of 0.05. Additionally, text prompt tokens are randomly dropped with a probability of 0.1. All experiments are conducted on NVIDIA A100 GPUs with 80GB of memory, using a batch size of 8. We use a cfg value of 8 to generate the results.

## Autoencoder MAGVIT

Base channels 16 Channel multiplier [1, 4, 16, 32, 64] Encoder blocks count [1, 1, 2, 8, 8] Decoder blocks count [4, 4, 4, 4, 4] Stride of frame [1, 2, 2, 2, 1] Stride of h and w [2, 2, 2, 2, 1] Padding mode replicate Compression rate 8 × 16 × 16 Bottleneck channels 32 Use KL divergence ✓ Use adaptive norm ✓(decoder only)

Table 4. Autoencoder and MAGVIT specifications.

## Backbone DiT

Input channels 32 Patch size 1 × 2 × 2 Latent token channels 4096 Positional embeddings 3D-RoPE DiT blocks count 32 Attention heads count 32 Window size 6144 (center) Normalization Layer normalization Use flash attention ✓ Use QK-normalization ✓ Use self conditioning ✓ Self conditioning prob. 0.9 Context channels 1024

Table 5. Backbone and DiT specifications.

Optimizer AdamW Learning rate 1 × 10−4 LR scheduler constant Beta [0.9, 0.99] Weight decay 0.01 Gradient clipping 0.05

- Dropout (Stage I) 0.8
- Dropout (Stage II) 0.5

Table 6. Training stages and optimization settings.

- 9 Prompts

Providing detailed prompts at the initialization stage is crucial for achieving high-quality editing and composition. Prompts describe not only the appearance but also their environment and dynamic behavior in detail. This allows the model to align the text description with the corresponding video frames effectively, leading to better identity and motion preservation during editing and composition.

For example, in the dancer case shown in Figure 2 of the main paper, the prompt explicitly defines the appearance, surroundings, and action: "A [v] man in black track pants, gray shirt, and cap near a road on a bridge with hands down and legs up. The man is performing [u] dancing motion with hands and legs." Here, the description of the attire (black track pants, gray shirt, and cap) ensures the model captures the subject’s visual identity, while the mention of the environment (a road on a bridge) provides spatial context. The prompt also includes details about the motion ("dancing motion with hands and legs"), guiding the model to encode temporal dynamics accurately.

References

- [1] Michael S Albergo and Eric Vanden-Eijnden. 2022. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571 (2022).
- [2] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel Cohen-Or, and Dani Lischinski.

2023. Break-A-Scene: Extracting Multiple Concepts from a Single Image. arXiv preprint arXiv:2305.16311 (2023).

- [3] Jianhong Bai, Tianyu He, Yuchi Wang, Junliang Guo, Haoji Hu, Zuozhu Liu, and Jiang Bian. 2024. UniEdit: A Unified Tuning-Free Framework for Video Motion and Appearance Editing. arXiv preprint arXiv:2402.13185 (2024).
- [4] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, et al. 2024. Lumiere: A spacetime diffusion model for video generation. In SIGGRAPH Asia 2024 Conference Papers. 1–11.
- [5] Xiuli Bi, Jian Lu, Bo Liu, Xiaodong Cun, Yong Zhang, WeiSheng Li, and Bin Xiao.

2024. CustomTTT: Motion and Appearance Customized Video Generation via Test-Time Training. arXiv preprint arXiv:2412.15646 (2024).

- [6] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al.

2023. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023).

- [7] Shengqu Cai, Duygu Ceylan, Matheus Gadelha, Chun-Hao Huang, Tuanfeng Wang, and Gordon. Wetzstein. 2024. Generative Rendering: Controllable 4DGuided Video Generation with 2D Diffusion Models. In CVPR.
- [8] Ting Chen and Lala Li. 2023. FIT: Far-reaching Interleaved Transformers. arXiv preprint arXiv:2305.12689 (2023).
- [9] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Yuwei Fang, Ivan Skorokhodov, Jun-Yan Zhu, Kfir Aberman, Ming-Hsuan Yang, and Sergey Tulyakov.

2024. VideoAlchemy: Open-set Personalization in Video Generation. https: //openreview.net/forum?id=popKM1zAYa

- [10] Zuozhuo Dai, Zhenghao Zhang, Yao Yao, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. 2023. AnimateAnything: Fine-Grained Open Domain Image Animation with Motion Guidance. arXiv:2311.12886 [cs.CV]

- [11] Tri Dao. 2024. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In International Conference on Learning Representations (ICLR).
- [12] Google DeepMind. 2024. VEO2. https://deepmind.google/technologies/veo/veo-2/

(2024).

- [13] Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, and Stefanos Zafeiriou. 2022. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 10 (oct 2022), 5962–5979. https://doi.org/10.1109/tpami.2021.3087709
- [14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. OpenReview.net.
- [15] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. 2024. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. OpenReview.net.
- [16] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. 2022. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618

(2022).

- [17] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. 2023. An image is worth one word: Personalizing text-to-image generation using textual inversion. In ICLR.
- [18] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. 2023. TokenFlow: Consistent Diffusion Features for Consistent Video Editing. arXiv preprint arxiv:2307.10373 (2023).
- [19] Litong Gong, Yiran Zhu, Weijie Li, Xiaoyang Kang, Biao Wang, Tiezheng Ge, and Bo Zheng. 2024. AtomoVideo: High Fidelity Image-to-Video Generation. arXiv:arXiv:2403.01800 [cs.CV]
- [20] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. 2023. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725 (2023).
- [21] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi.

2024. LTX-Video: Realtime Video Latent Diffusion. arXiv preprint arXiv:2501.00103

(2024).

- [22] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, and Jie Zhang. 2024. ID-Animator: Zero-Shot Identity-Preserving Human Video Generation. arXiv:2404.15275 [cs.CV] https://arxiv.org/abs/2404.15275
- [23] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. 2022. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303 (2022).
- [24] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. 2022. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868 (2022).
- [25] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685 [cs.CL] https://arxiv.org/abs/2106.09685
- [26] Allan Jabri, David J. Fleet, and Ting Chen. 2023. Scalable adaptive computation for iterative generation. In Proceedings of the 40th International Conference on Machine Learning (Honolulu, Hawaii, USA) (ICML’23). JMLR.org, Article 594, 21 pages.
- [27] Maxwell Jones, Sheng-Yu Wang, Nupur Kumari, David Bau, and Jun-Yan Zhu. 2024. Customizing Text-to-Image Models with a Single Image Pair. arXiv:2405.01536 [cs.CV] https://arxiv.org/abs/2405.01536
- [28] Ozgur Kara, Bariscan Kurtkaya, Hidir Yesiltepe, James M. Rehg, and Pinar Yanardag. 2024. RAVE: Randomized Noise Shuffling for Fast and Consistent Video Editing with Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
- [29] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. 2024. HunyuanVideo: A Systematic Framework For Large Video Generative Models. arXiv preprint arXiv:2412.03603

(2024).

- [30] Feng Liang, Bichen Wu, Jialiang Wang, Licheng Yu, Kunpeng Li, Yinan Zhao, Ishan Misra, Jia-Bin Huang, Peizhao Zhang, Peter Vajda, et al. 2023. FlowVid: Taming Imperfect Optical Flows for Consistent Video-to-Video Synthesis. arXiv preprint arXiv:2312.17681 (2023).
- [31] Chang Liu, Viraj Shah, Aiyu Cui, and Svetlana Lazebnik. 2024. UnZipLoRA: Separating Content and Style from a Single Image. arXiv:2412.04465 [cs.CV] https://arxiv.org/abs/2412.04465
- [32] Xingchao Liu, Chengyue Gong, and Qiang Liu. 2022. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 (2022).

- [33] Ilya Loshchilov and Frank Hutter. 2019. Decoupled Weight Decay Regularization. In International Conference on Learning Representations. https://openreview.net/ forum?id=Bkg6RiCqY7
- [34] Ze Ma, Daquan Zhou, Chun-Hsiao Yeh, Xue-She Wang, Xiuyu Li, Huanrui Yang, Zhen Dong, Kurt Keutzer, and Jiashi Feng. 2024. Magic-Me: Identity-Specific Video Customized Diffusion. arXiv:2402.09368 [cs.CV] https://arxiv.org/abs/2402.09368
- [35] Joanna Materzyńska, Josef Sivic, Eli Shechtman, Antonio Torralba, Richard Zhang, and Bryan Russell. 2024. NewMove: Customizing text-to-video models with novel motions. In Proceedings of the Asian Conference on Computer Vision. 1634–1651.
- [36] Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov, Ekaterina Deyneka, TsaiShien Chen, Anil Kag, Yuwei Fang, Aleksei Stoliar, Elisa Ricci, Jian Ren, et al. 2024. Snap video: Scaled spatiotemporal transformers for text-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7038–7048.
- [37] Eyal Molad, Eliahu Horwitz, Dani Valevski, Alex Rav Acha, Yossi Matias, Yael Pritch, Yaniv Leviathan, and Yedid Hoshen. 2023. Dreamix: Video Diffusion Models are General Video Editors. arXiv:2302.01329 [cs.CV] https://arxiv.org/ abs/2302.01329
- [38] OPENAI. 2024. SORA. https://openai.com/sora/ (2024).
- [39] William Peebles and Saining Xie. 2023. Scalable Diffusion Models with Transformers. IEEE, 4172–4182.
- [40] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. 2024. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720 (2024).
- [41] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. 2023. FateZero: Fusing Attentions for Zero-shot Text-based Video Editing. arXiv:2303.09535 (2023).
- [42] Guocheng Qian, Kuan-Chieh Wang, Or Patashnik, Negin Heravi, Daniil Ostashev, Sergey Tulyakov, Daniel Cohen-Or, and Kfir Aberman. 2024. Omni-ID: Holistic Identity Representation Designed for Generative Tasks. arXiv:2412.09694 [cs.CV] https://arxiv.org/abs/2412.09694
- [43] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al.

2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.

- [44] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res. 21, 1, Article 140 (Jan. 2020), 67 pages.
- [45] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 (2022).
- [46] Weiming Ren, Harry Yang, Ge Zhang, Cong Wei, Xinrun Du, Stephen Huang, and Wenhu Chen. 2024. ConsistI2V: Enhancing Visual Consistency for Image-to-Video Generation. arXiv preprint arXiv:2402.04324 (2024).
- [47] Yixuan Ren, Yang Zhou, Jimei Yang, Jing Shi, Difan Liu, Feng Liu, Mingi Kwon, and Abhinav Shrivastava. 2024. Customize-a-video: One-shot motion customization of text-to-video diffusion models. arXiv preprint arXiv:2402.14780 (2024).
- [48] Daniel Roich, Ron Mokady, Amit H Bermano, and Daniel Cohen-Or. 2022. Pivotal tuning for latent-based editing of real images. ACM Transactions on Graphics (TOG) 42, 1 (2022), 1–13.
- [49] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. 10684–10695.
- [50] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer, 234–241.
- [51] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR. 22500–22510.
- [52] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. 2023. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. arXiv preprint arXiv:2307.06949 (2023).
- [53] Simo Ryu. 2023. DreamboothLoRA. https://github.com/cloneofsimo/lora
- [54] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. In NIPS. 36479–36494.
- [55] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. 2022. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792

(2022).

- [56] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu.

2024. RoFormer: Enhanced transformer with Rotary Position Embedding. Neurocomputing 568 (2024), 127063. https://doi.org/10.1016/j.neucom.2023.127063

- [57] Kuan-Chieh Wang, Daniil Ostashev, Yuwei Fang, Sergey Tulyakov, and Kfir Aberman. 2024. MoA: Mixture-of-Attention for Subject-Context Disentanglement in Personalized Image Generation. arXiv:2404.11565 [cs.CV] https: //arxiv.org/abs/2404.11565
- [58] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. 2024. DreamVideo: Composing Your Dream Videos with Customized Subject and Motion. In CVPR.
- [59] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. 2023. Tune-AVideo: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation. arXiv:2212.11565 [cs.CV] https://arxiv.org/abs/2212.11565
- [60] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2024. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. arXiv preprint arXiv:2408.06072 (2024).
- [61] Lijun Yu, Jose Lezama, Nitesh Bharadwaj Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A Ross, and Lu Jiang. 2024. Language Model Beats Diffusion - Tokenizer is key to visual generation. In The Twelfth International Conference on Learning Representations. https://openreview. net/forum?id=gzqrANCF4g
- [62] David Junhao Zhang, Dongxu Li, Hung Le, Mike Zheng Shou, Caiming Xiong, and Doyen Sahoo. 2024. Moonshot: Towards Controllable Video Generation and Editing with Multimodal Conditions. arXiv:2401.01827 [cs.CV] https://arxiv.org/ abs/2401.01827
- [63] Yuxin Zhang, Fan Tang, Nisha Huang, Haibin Huang, Chongyang Ma, Weiming Dong, and Changsheng Xu. 2023. MotionCrafter: One-Shot Motion Customization of Diffusion Models. arXiv preprint arXiv:2312.05288 (2023).
- [64] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. 2023. MotionDirector: Motion Customization of Text-to-Video Diffusion Models. arXiv preprint arXiv:2310.08465

(2023).

- [65] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. 2023. PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel. arXiv:2304.11277 [cs.DC] https://arxiv.org/abs/2304.11277
- [66] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou.

2024. StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation. arXiv:2405.01434 [cs.CV] https://arxiv.org/abs/2405.01434

