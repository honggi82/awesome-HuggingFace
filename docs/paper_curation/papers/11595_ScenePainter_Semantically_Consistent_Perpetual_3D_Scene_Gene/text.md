### ScenePainter: Semantically Consistent Perpetual 3D Scene Generation with Concept Relation Alignment

Chong Xia, Shengjun Zhang, Fangfu Liu, Chang Liu, Khodchaphun Hirunyaratsameewong, Yueqi Duan† Tsinghua University

# arXiv:2507.19058v1[cs.CV]25Jul2025

###### Previous Works

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

SceneScapeWonderJourney

Input Image

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

###### ScenePainter (Ours)

[Figure 12]

[Figure 13]

Semantic Consistency Visual Diversity

Input Image

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Figure 1. We propose ScenePainter, which aims to generate semantically consistent yet visually diverse 3D view sequences starting from a single view. We show painted images from the view of a moist and shadowed street with a receding 3D camera motion. ScenePainter can generate plausible and consistent content given the first view while maintaining diversity in forms and appearances.

#### Abstract

Perpetual 3D scene generation aims to produce longrange and coherent 3D view sequences, which is applicable for long-term video synthesis and 3D scene reconstruction. Existing methods follow a “navigate-and-imagine” fashion and rely on outpainting for successive view expansion. However, the generated view sequences suffer from semantic drift issue derived from the accumulated deviation of the outpainting module. To tackle this challenge, we propose ScenePainter, a new framework for semanti-

cally consistent 3D scene generation, which aligns the outpainter’s scene-specific prior with the comprehension of the current scene. To be specific, we introduce a hierarchical graph structure dubbed SceneConceptGraph to construct relations among multi-level scene concepts, which directs the outpainter for consistent novel views and can be dynamically refined to enhance diversity. Extensive experiments demonstrate that our framework overcomes the semantic drift issue and generates more consistent and immersive 3D view sequences. Project Page: https://xiac20. github.io/ScenePainter/.

†Corresponding author.

#### 1. Introduction

With the rapid development of diffusion models, 3D generation has become one of the most significant problems in 3D computer vision. One main branch perpetual 3D scene generation has garnered more attention from academia and industry for their potential in large 3D scene construction and long video synthesis. Given an initial single image or a scene description, perpetual 3D scene generation aims to generate a series of 3D scene views, which are consistent in geometry and semantic relations. To achieve this, previous works mainly follow a ’navigate and imagine’ paradigm, which can be decomposed into the iterative processes of unprojecting, rendering, and outpainting. The geometry consistency problem has been well solved by previous works due to the improved monocular depth estimation and effective 3D scene representations like point cloud, mesh or 3D gaussian splatting [17].

However, the semantic consistency problem has not received enough attention, and causes serious semantic drift issue derived from the accumulated deviation and artifacts of the native outpainting process. One of previous state-of-the-art models WonderJourney [36] frequently encounters semantic drift issue, shifting from the arid desert to the vibrant lake or shifting from lush green scenes to snow-covered landscapes. Although this work is dedicated to generating diverse views, the scene inconsistency problem remains particularly pronounced and affects the scene quality. In addition, the semantic drift issue is bypassed in earlier works like InfiniteNature-Zero [20] and SceneScape [10], as they maintain subtle camera movements between adjacent frames for smooth geometric transformation, which hinders the generation of novel objects and leads to monotonous scenes. Therefore, generating semantically consistent and content-rich scene views is a key challenge in 3D scene generation as shown in Figure 1.

Towards this goal, we propose ScenePainter, a two-stage framework for semantically consistent yet visually diverse 3D scene generation illustrated in Figure 2. In the first stage of concept relation construction, we propose a scene-level customization method to achieve a thorough comprehension of the current scene. Unlike previous single image or multiconcept customization methods, scenes contain richer concepts and more complicated relations among them, such as the relative spatial position of objects, semantic distribution, spatial layout and overall style, which makes achieving satisfactory customization results particularly challenging. To solve this problem, we first extract multilevel scene concepts rather than simply multiple objects, and then leverage a hierarchical graph structure dubbed SceneConceptGraph to further construct the relations among concepts. For each concept-relation pair in the SceneConceptGraph, we take it as dedicated textual embeddings for text-to-image models to restore the original image region referenced by the rela-

tion. In this way, we obtain scene-specific textual embeddings of each concept-relation pair and optimized text-toimage model, which jointly direct the subsequent outpainting model with scene-specific prior transmission.

In the second stage of concept relation refinement, we follow the mainstream 3D generation framework and align the native outpainting model with extracted scene-specific prior to achieve semantically consistency and mitigate generation deivation. To be specific, we initialize the outpainter with the aforementioned optimized text-to-image model and guide the outpainting process with updated conceptrelation pair. Further, generating nearly identical scenes would lead to repetition and monotony, thus we strive to enhance visual diversity while ensuring semantic consistency. During the outpainting process, we can dynamically add new objects, change existing objects, or smoothly transition to other scenes with test-time refinement of the SceneConceptGraph. In this way, users can easily enrich and expand the current scene while enabling novel creation, thereby achieving both consistency and diversity simultaneously. Our main contributions are as follows:

- • We design a scene-level customization method, which constructs the concepts and relations of the scene and generates consistent scene views.
- • We propose a new 3D scene generation framework ScenePainter, which aligns the outpainting model with scenespecific prior for consistent and diverse scene expansion.
- • Extensive experiments demonstrate that our framework produces more consistent and vivid 3D view sequences in comparison to previous state-of-the-art works.

- 2. Related Work
- 3D Scene Generation. Due to the recent success of image diffusion models [13, 22, 24], 3D scene generation has seen rapid developments. Existing methods can be mainly divided into two branches. One branch of methods [29, 32, 37–39] employ 2D panorama to represent panoramic environments and lift it to 3D space. However, limited by the representative capacity of 2D panoramas, these methods often result in blurry renderings, ambiguity and gaps in 3D scene. Moreover, they fail to generate arbitrary 3D scene following specific camera trajectory or interactive user instructions, which limits their applications.

The other branch of methods [5, 7, 16] seeks to create perpetual scenes by leveraging a “navigate-and-imagine” paradigm, which renders the partial image, outpaints for a complete next view and repeats the process. The foundational works, Infinite Images [16], Infinite Nature [21] and InfiniteNature-Zero [20], first propose the autoregressive pipeline. SceneScape [10] further exploits the generative power and incrementally constructs a single cave-like environment. A recent development, WonderJourney [36],

Stage1: Concept Relation Construction

Stage2 : Concept Relation Refinement

SceneConceptGraph

[Figure 20]

[Figure 21]

|[Figure 22]| |
|---|---|
| | |

|[Figure 23]|
|---|

[Figure 24]

[Figure 25]

update

Text-to-Image Model

update

concept-relation extraction

concept-relation extraction

Mask

Mask

[Figure 26]

## { }

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

unproject

Outpaint Model

render

[Figure 31]

###### scene-specific initialization

Rendered Image Outpainted Image

Initial Image

3D Representation

- Figure 2. Overall framework of our approach ScenePainter. We propose a two-stage framework that constructs and refines scene concept relations with the graph structure SceneConceptGraph, and aligns the outpainting model with the scene-specific prior during the ongoing painting process. We use colored dots and regions to represent scene concept nodes and their masks, and black and red lines in SceneConceptGraph to denote the initial and newly added relation edges. Here, we simplify the update process in the second stage, which is similar to the one in the first stage.

focuses on creating diverse view sequences combining offthe-shelf models. While these approaches explore scene generation, they either generate diverse scene with obvious semantic drift issue or generate scenes with limited content and viewpoints. In contrast, our approach generates sceneconsistent and content-rich view sequences, achieving the balance between consistency and diversity.

Customized Generation. Customized image generation has attracted increasing attention due to its ability to create consistent images with user preferences. Two fundamental methods in text-to-image model customization are Textual Inversion (TI) [11] and DreamBooth (DB) [25], with TI focusing on optimizing text embeddings, while DB finetuning the whole image diffusion model. Other notable approaches include Custom Diffusion [19], which fine-tunes only specific layers, and LoRA [14, 26], which restricts updates to rank 1 matrices. Expanding on these techniques, various studies [12, 27, 28, 31, 34] have explored single subject or multi-object customization from a single or several reference images. Break-A-Scene [1] extracts multiple concepts from one single image and employs masks to indicate different objects. To the best of our knowledge, we are the first work to achieve scene-level customization by constructing scene graph and utilizing concept-relation pair textual embeddings, thus creating consistent scene views.

Video Generation. Expanded upon basic text-to-image diffusion models [13, 22, 24], recent works on text-to-video and image-to-video generation [3, 4, 8] have seen rapid advancements by incorporating temporal dynamics, enabling the production of high-quality and diverse video content. However, many of these approaches [15, 30, 33] focus on the dynamics of the subject with a relatively static viewpoint, such as human walking or pets running. Even with delicate text prompt, these methods are reluctant to create more scene content with large viewpoint transition. In this work, we mainly focus on long-range 3D view sequences generation with free-moving camera trajectories and rich content, which may potentially serve as keyframes for synthesizing long-term 3D videos.

#### 3. Approach

In this section, we present our method ScenePainter for 3D scene generation. At first, we review the general pipeline of perpetual 3D scene generation and briefly illustrate the framework of our ScenePainter in Section 3.1. Next, we provide a detailed introduction to our two main stages: Concept Relation Construction in Section 3.2 and Concept Relation Refinement in Section 3.3.

##### 3.1. Overview of ScenePainter

Perpetual 3D scene generation aims to synthesize a series of consistent yet diverse 3D views, starting from an arbitrary single image and following a long-range specified camera trajectory. The general pipeline mainly consists of three iterative processes: unproject, render, and outpaint, which could be further explained as lifting current 2D image to 3D representation, rendering partial image at next view camera and outpainting for a complete next image. The entire process could be modular, leveraging pre-trained monocular depth estimator, outpainting model and optional vision language model. To mathematically describe this, we suppose the generated view stream denoted as I = {I1,I2,...,IT} with each component Ii derived from Ii−1 as:

Di−1 = DepthEstimator(Ii−1), Pi−1 = unproject(Ii−1,Di−1),

(1)

Si = Si−1 ∪ Pi−1, I¯i,Mi = render(Si,Ci), Ii = outpaint(I¯i,Mi,Ti).

Here Di, Ci, I¯i, Mi, Ti represent the estimated depth, specified camera trajectory, partial image and mask for outpainting, and optional text prompt at ith frame respectively. To ensure geometry consistency, previous works focus on the representation of single 2D image in 3D space and the unified 3D scene representation to establish appropriate geometric structures, corresponding to Pi and Si as mentioned above. These methods mainly contribute on specific depth refinement strategies and effective 3D representations such as point cloud [36], mesh [10] and gaussian surfels [9, 35].

However, beyond geometric consistency, semantic consistency is also a crucial and challenging problem in 3D scene generation. Merely depending on the partial image and optional text prompt, it is difficult for the off-theshelf outpainting model to provide semantically consistent painted results that accurately match with the existing partial scene, which could be attributed to the limited performance and inherent diversity of the outpainter. Moreover, even though there may be slight deviation in semantic comprehension each time, the semantic drift error will be accumulated and amplified with iterative processing, leading to a significant difference from the original scene definition. This is referred to as the semantic transfer issue. When the first and last images are placed together, they appear to be two views from totally different and separate scenes.

In order to solve this issue, we propose ScenePainter, which constructs unified and comprehensive understanding of the scene concept relations based on the initial scene definition and aligns the scene-specific prior of the outpainter with it to ensure semantic consistency. During the subsequent view generation process, we continuously refine the

understanding of scene relationships while ensuring that the outpainter is updated in parallel, allowing the scene to evolve and enrich in a controllable way. The whole procedure could be depicted as follows:

M¯ 0,G0 = construct(I0), M0 = BLD(M¯0),

(2)

Ii+1 = Mi(Ii,Ti,Gi), Gi+1,Mi+1 = refine(Ii+1,Gi,Mi),

(3)

where M¯0 and M′ denote the initial optimized text-toimage generation model and the converted outpainting model by Blended Latent Diffusion (BLD) [2] for scene expansion. G refers to the specific multi-layer graph structure for concept relations extraction dubbed as SceneConceptGraph, as illustrated in Figure 2 . In the following parts, we will explain the construction and refinement process of scene concept relations in detail.

##### 3.2. Concept Relation Construction

Given an initial scene view I0, we construct a relation graph among multi-level scene concepts, G =< V,R >, where V and R represent the sets of concept vertices and relation edges respectively. The V contains three-layer concept node sets, i.e. V = {V1,V2,V3}. The first-level node set V1 contains only a single concept, which represents the overall environment and style, typically denoted as v0. The secondlevel node set V2 encompasses all the areas with the same category which we wish to focus on, such as a forest or a group of buildings. The third-level node set V3 represents the most fundamental objects in the scene. Specifically, for objects with uniqueness, we assume that they belong to second-level node set V2, where a single object represents its corresponding category region. And for objects with diverse appearances but belong to the same type, we categorize them as third-level nodes, and the common region formed by multiple objects is assigned to the secondlevel node. Besides, for each concept, we store its corresponding region mask and for each relation edge, we consider the combined area of the two connected concepts as its mask. We maintain the region affiliation mapping function, so that for any third-level node, the corresponding secondlevel node can be identified, which can be formulated as:

∀vi ∈ V3, ∃vj ∈ V2, s.t. vj = region(vi) (4)

The relation edge set R consists of three types of relations: R1 connects vertices in V1 and V2 and represents the spatial layout and style of category regions in the whole scene; R2 connects vertices both in V2 and represents the relative spatial layout and semantic connection of different category regions; R3 connects node in V3 and corresponding region node in V2 represents the relative spatial layout within the category region.

Considering the fact that concepts and relations in the scene have much more complicated and comprehensive meanings than plain text description, we decide to adopt learning-based method to optimize the concept and relation embeddings, which is widely used in customization generation. To be specific, we aim to extract N + M textual handles {vi}Ni=1 and {ri}Mi=1 from off-the-shelf text-to-image models, s.t. the textual handle represents the corresponding concept or relation. And the optimized handles can then be correspondingly considered as text prompts to guide the synthesis of new scene views and novel combinations of scene concepts and relations. Unlike previous objectcentric customization that utilizes basic object handles as text prompt, we focus on relation among concepts and take each relation-concept pair as text guidance, which could be denoted as < vi,r(i,j),vj >.

In terms of specific customization strategies, two classic methods are: Textual Inversion (TI) [11], which extracts text embeddings but fails to preserve their identity, and DreamBooth (DB) [25], which fine-tunes the entire diffusion model but lacks diversity. As a result, we optimize both the textual handles and model weights, combining the basic two customization strategies in two different phases as [1]. The combination enables the textual handles and the diffusion model to collaborate in iterative updates and construct a personalized understanding. The optimized textto-image generation model with scene-specific prior would serve as the initialization parameters of the outpainter in the following view generation process by the strategy Blended Latent Diffusion [2].

The handles are optimized using a combination of a masked reconstruction diffusion loss, a scene-specific prior preservation loss, and a cross-attention loss. For each relation-concept handle pair < vi,r(i,j),vj > with masks mi and mj, we keep the two concepts unchanged and outpaint the remaining part to generate novel views with similar scene environment to serve as training samples for scene-specific prior preservation and scene diversity. Then we supervise the reconstruction quality within the union region of the two concepts and the cross-attention map between each handle and its corresponding mask to get the handle focus on the regions it refers to. The total loss formulation can be summarized as:

Lrec = Ez,s,ϵ∼N(0,1),t ∥ϵ ⊙ mu − ϵθ(zt,t,ps) ⊙ mu∥22 ,

Lprior = Ez,s,ϵ∼N(0,1),t ∥ϵ − ϵθ(zt,t,ps)∥22 , Lattn = Ez,h,t ∥CAθ(h,zt) − mh∥22 , Ltotal = Lrec + λpriorLprior + λattnLattn,

(5) where zt is the noisy latent at time step t, ps is the text prompt, mu is the union of the handle masks mh, ϵ is the

added noise, ϵθ is the denoising network, and CAθ(h,zt) is the cross-attention map between the handle h and the noisy latent zt. λprior and λattn are hyperparameters to balance the combination of the total loss function.

After the first stage, we construct the SceneConceptGraph which represents concepts and relations with dedicated textual embeddings and at the same time, we prepare the initialization weights for the outpainter with customized scene-specific prior.

##### 3.3. Concept Relation Refinement

During the view sequence generation stage, relying on the graph construction of relations among scene concepts, the outpainting module tends to generate novel views consistent with the initial scene definition, which share a unified overall style, similar object characteristics and harmonized spatial layout. Moreover, in terms of scene diversity and editability, unlike previous works that adopt predefined or automatically generated text prompt as outpainting guidance and suffer from the limited prompt fidelity of the outpainter model leading to semantic drift issue, our model are more friendly to user-specified text prompt, with simply describing a handle or several handles to change the appearance or spatial location, muting a handle to keep it from appearing in the next frame and detailed description about new objects to generate novel concepts.

Furthermore, in order to dynamically refine the SceneConceptGraph with new concept relations and adaptively align the scene understanding of the outpainter for more consistent subsequent views, we take a test-time training approach to simultaneously update new text embeddings and align the outpainter. Unlike the construction process that focuses on all three types of concepts and relations, we only concentrate on one relation edge between the firstlevel concept i.e. the overall environment and second-level concept specified by users in the refinement stage for efficiency and real-time capability. To be specific, for the addition of a new concept, we assign new concept holder vn and relation holder r(0,n), and then take the relation-concept pair < v0,r(0,n),vn > as textual embeddings for training. And for the change of an existing concept vi, we take the relation-concept pair < v0,r(0,i),vi > accordingly. We use segmentation models to get the assigned concept mask and optimize both the textual handles and model weights with masked diffusion loss and cross-attention loss as mentioned before. The generated view sequences tend to maintain consistent relationships while making versatile changes.

#### 4. Experiments 4.1. Comparison Methods

To evaluate the performance of our approach, we compare with previous methods in two tasks: Single Image

Input Image InstantBooth IP-Adapter CustomDiffusion Break-A-Scene Ours

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

- Figure 3. Comparison with other customization methods. InstantBooth and IP-Adapter are designed for the whole image customization without concept mask, while Custom Diffusion and Break-A-Scene focus on multi-concept customization. All these methods fail to achieve satisfactory scene-level customization.

[Figure 38]

- Figure 4. Comparison with WonderJourney [36] and SceneScape [10]. We simply combine multi-concept customization method BreakA-Scene and 3D views generation framework WonderJourney as the baseline BAS+WJ for a comprehensive comparison.

Customization and 3D Views Generation. For the Single Image Customization task, we leverage the constructed SceneConceptGraph with optimized text-to-image model and choose four preeminent customization approaches to compare the quality and fidelity of the generated image: InstantBooth [27], IP-Adapter [34], Custom Diffusion [19]and Break-A-Scene [1]. For the 3D Views Generation task, we maintain both construction and refinement processes and select two state-of-the-art methods to compare the performance of generated view sequences: SceneScape [10] and WonderJourney [36]. We collect a dataset

of 30 scenes, including nature, village, city, indoor room, or fantasy scenes, etc. and based on the scene dataset, we conduct extensive qualitative and quantitative comparison in the following parts.

##### 4.2. Qualitative Comparison

Single Image Customization. We show qualitative comparisons with several mainstream customization strategies in Figure 3. We can observe that previous works fail to preserve the scene identity or be almost identical to the original scene. By contrast, our model is able to generate

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

WonderJourney Ours

- Figure 5. 3D representation comparison with WonderJourney [36]. We use the same first view and fixed camera path for evaluation. The generated 3D view sequences are shown in the supplementary material.

Input Image “no [chair]” “add (bookshelf) ” “blue [blanket] and tall [chair]”

scene views with great fidelity and novel combination ways of scene concepts, which achieves high-quality scene-level customization and provides the scene-specific prior for the outpainting model in the following scene generation stage.

“<[picture],[chair]>”

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Figure 6. Diversity and instruction fidelity of our approach. We use square brackets to denote defined concepts, parentheses to denote new objects, and angle brackets to indicate the relations between concepts.

3D Views Generation. We present qualitative examples of the generated 3D scene views compared to previous stateof-the-art methods in Figure 4. It shows that with the generation going on, SceneScape [10] fails to create content-rich views, and WonderJourney [36] suffers heavily from the semantic drift issue. Even with multi-concept customization on the first view, the simple combination baseline BAS+WJ still encounters problems including monotonous layouts, shifted overall style, and disorder in geometry due to the absence of the multi-level concepts and relations customization and test-time refinement. In contrast, our model generates more consistent and visually diverse scene views.

##### 4.3. Quantitative Comparison

Single Image Customization. To quantitatively assess our method and the baseline, we focus on scene fidelity evaluation, which measures the preservation of initial scene details in generated images. We adopt CLIP-I [23] and DINO [6] as evaluation metrics. These two metrics are the average pairwise cosine similarity of CLIP or ViT-S/16 DINO embeddings between the initial image and the generated image. As shown in Table 1, our method achieves the best scores in both CLIP-I and DINO metrics, indicating its high fidelity and strong scene-level customization capacity.

Furthermore, we present the intermediate 3D representation generated by WonderJourney and ours in Figure 5, which demonstrates that even though we adopt a similar unprojecting and rendering pipeline as WonderJourney, the inconsistency in objects and surroundings would cause the distortion in 3D geometric structure. Moreover, we provide a diverse example in Figure 6 to illustrate how to generate the desired 3D views based on user instructions. More results of our 3D view sequences, constructed 3D scenes, and synthesized 3D videos can be found in the supplementary document and demo video.

3D Views Generation. Since perpetual 3D scene generation is a new task without an existing dataset for quantitative evaluation, we conduct a user study on the subset of our collected dataset and focus on user preference evaluation. We generate scene views following each approach’s own setup and utilize visual quality, diversity and consistency as the evaluation metric. We show a side-by-side comparison

- Table 1. Quantitative comparison of scene fidelity based on DINO and CLIP-I metrics. Our method achieves the best scores.

Method DINO ↑ CLIP-I ↑

InstantBooth 0.785 0.832 IP-Adapter 0.899 0.922 Custom Diffusion 0.835 0.888 Break-A-Scene 0.877 0.896 ScenePainter (Ours) 0.931 0.952

- Table 2. Quantitative comparison of human preference based on visual quality, diversity and consistency criteria.

Qual. Div. Con.

Ours over WonderJourney 89.3% 83.4% 92.6% Ours over SceneScape 92.1% 97.6% 93.8%

- Table 3. Quantitative comparison results of the ablation study on the loss function and the graph structure.

Method DINO ↑ CLIP-I ↑

Ours w/o Lrec 0.523 0.568 Ours w/o Lprior 0.894 0.913 Ours w/o Lattn 0.723 0.768 Ours w/o V1 0.774 0.795 Ours w/o V3 0.857 0.886 Ours w/o R 0.634 0.672 ScenePainter (Ours) 0.924 0.949

of scene views generated by ours and SceneScape [10] or WonderJourney [36] and then ask one binary choice question for users to decide. As shown in Table 2, our model is strongly preferred over both baselines for all three metrics. WonderJourney generates wildly imaginative scenes, thus achieving acceptable diversity but notably low consistency caused by semantic deviation, whereas SceneScape generates more consistent but monotonous scenes due to the limited outpainting white space. Overall, our method achieves a well diversity-consistency balance and generates immersive 3D scene views.

- 4.4. Ablation Study

Input Image w/o Lrec w/o Lprior w/o Lattn Ours

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

- Figure 7. Qualitative comparison results of the ablation study on the construction loss function.

[Figure 53]

[Figure 54]

Input Image w/o V1 w/o V3 w/o R Ours

[Figure 55]

[Figure 56]

[Figure 57]

- Figure 8. Qualitative comparison results of the ablation study on the SceneConceptGraph structure.

overfitting to the initial scene image layout and construction without relations causes distorted layouts and chaotic geometric structures.

##### 4.5. Implementation Details

In the concept relation construction and refinement stage, we leverage the Stable Diffusion [24] model as the base text-to-image and outpainting model and SAM [18] for mask segmentation. We customize the base model with the concept-relation pair prompt in two phases. In the first phase, we follow Textual Inversion and train text encoder with 400 steps, 1×10−6 learning rate. In the second phase, we follow DreamBooth and train the whole diffusion model with 400 steps, 1 × 10−4 learning rate. Then during the refinement process, we train the UNet module simply with 50 steps, 1e-4 learning rate for test-time efficiency. The training is done using a single NVIDIA A6000 GPU with about 5 minutes for construction and 25 seconds for refinement. For the entire 3D scene generation pipeline, we follow the similar unprojecting and rendering process proposed by WonderJourney with our customized model for outpainting and we adopt Blended Latent Diffusion [2] for converting the text-to-image generation model to the outpainting model for scene-specific prior transfer.

#### 5. Conclusion

In this paper, we propose ScenePainter, a new framework designed for semantically consistent and visually diverse 3D scene generation. To be specific, we construct and dynamically maintain the hierarchical graph focusing on the relations and concepts of the scene representation. Then we take it as dedicated textual embeddings to align and refine the outpainter’s scene-specific prior with initial scene presentation for consistent expansion. Extensive experiments show that compared to previous methods, our framework effectively eliminates the semantic drift problem and produces a series of more consistent and vivid 3D scene views.

We further conduct ablation studies to evaluate the validity of our method on the construction loss function Equation

- 5 and the structure of our proposed SceneConceptGraph. From Figure 7 and Table 3, we observe that Lrec mainly ensures visual quality, Lprior inspires the generation of objects with different forms and Lattn prevents unreasonable object distributions and combinations. It can also be shown in Figure 8 and Table 3 that removing the first-level concept the whole environment leads to the general style transfer, removing the third-level concept the individual object causes

#### 6. Acknowledgement

This work was supported by the National Natural Science Foundation of China under Grant 62206147.

#### References

- [1] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel CohenOr, and Dani Lischinski. Break-a-scene: Extracting multiple concepts from a single image. In SIGGRAPH Asia 2023 Conference Papers, pages 1–12, 2023. 3, 5, 6
- [2] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. ACM transactions on graphics (TOG), 42

(4):1–11, 2023. 4, 5, 8

- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3
- [4] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators, 2024. 3
- [5] Shengqu Cai, Eric Ryan Chan, Songyou Peng, Mohamad Shahbazi, Anton Obukhov, Luc Van Gool, and Gordon Wetzstein. Diffdreamer: Towards consistent unsupervised single-view scene extrapolation with conditional diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2139–2150, 2023. 2
- [6] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 7
- [7] Lucy Chai, Richard Tucker, Zhengqi Li, Phillip Isola, and Noah Snavely. Persistent nature: A generative model of unbounded 3d worlds. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20863–20874, 2023. 2
- [8] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 3
- [9] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee. Luciddreamer: Domain-free generation of 3d gaussian splatting scenes. arXiv preprint arXiv:2311.13384, 2023. 4
- [10] Rafail Fridman, Amit Abecasis, Yoni Kasten, and Tali Dekel. Scenescape: Text-driven consistent scene generation. Advances in Neural Information Processing Systems, 36, 2024. 2, 4, 6, 7, 8
- [11] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 3, 5

- [12] Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7323–7334, 2023. 3
- [13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2, 3
- [14] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 3
- [15] Yash Jain, Anshul Nasery, Vibhav Vineet, and Harkirat Behl. Peekaboo: Interactive video generation via maskeddiffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8079– 8088, 2024. 3
- [16] Biliana Kaneva, Josef Sivic, Antonio Torralba, Shai Avidan, and William T Freeman. Infinite images: Creating and exploring a large photorealistic virtual space. Proceedings of the IEEE, 98(8):1391–1407, 2010. 2
- [17] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 2

- [18] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 8
- [19] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023. 3, 6
- [20] Zhengqi Li, Qianqian Wang, Noah Snavely, and Angjoo Kanazawa. Infinitenature-zero: Learning perpetual view generation of natural scenes from single images. In European Conference on Computer Vision, pages 515–534. Springer, 2022. 2
- [21] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, and Angjoo Kanazawa. Infinite nature: Perpetual view generation of natural scenes from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14458–14467, 2021. 2
- [22] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2, 3
- [23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 7

- [24] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3, 8
- [25] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 3, 5
- [26] Simo Ryu. Low-rank adaptation for fast text-toimage diffusion fine-tuning. 2022. URL https://github. com/cloneofsimo/lora. 3
- [27] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without testtime finetuning. In CVPR, pages 8543–8552, 2024. 3, 6
- [28] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-toimage generation. arXiv preprint arXiv:2303.09522, 2023. 3
- [29] Hai Wang, Xiaoyu Xiang, Yuchen Fan, and Jing-Hao Xue. Customizing 360-degree panoramas through text-to-image diffusion models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4933–4943, 2024. 2
- [30] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 3
- [31] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15943–15953, 2023. 3
- [32] Shuai Yang, Jing Tan, Mengchen Zhang, Tong Wu, Yixuan Li, Gordon Wetzstein, Ziwei Liu, and Dahua Lin. Layerpano3d: Layered 3d panorama for hyper-immersive scene generation. arXiv preprint arXiv:2408.13252, 2024. 2
- [33] Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, and Tali Dekel. Space-time diffusion features for zero-shot text-driven motion transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8466–8476, 2024. 3
- [34] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 3, 6

- [35] Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T Freeman, and Jiajun Wu. Wonderworld: Interactive 3d scene generation from a single image. arXiv preprint arXiv:2406.09394, 2024. 4
- [36] Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, et al. Wonderjourney:

- Going from anywhere to everywhere. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6658–6667, 2024. 2, 4, 6, 7, 8
- [37] Cheng Zhang, Qianyi Wu, Camilo Cruz Gambardella, Xiaoshui Huang, Dinh Phung, Wanli Ouyang, and Jianfei Cai. Taming stable diffusion for text to 360 panorama image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6347– 6357, 2024. 2
- [38] Haiyang Zhou, Xinhua Cheng, Wangbo Yu, Yonghong Tian, and Li Yuan. Holodreamer: Holistic 3d panoramic world generation from text descriptions. arXiv preprint arXiv:2407.15187, 2024.
- [39] Shijie Zhou, Zhiwen Fan, Dejia Xu, Haoran Chang, Pradyumna Chari, Tejas Bharadwaj, Suya You, Zhangyang Wang, and Achuta Kadambi. Dreamscene360: Unconstrained text-to-3d scene generation with panoramic gaussian splatting. In European Conference on Computer Vision, pages 324–342. Springer, 2025. 2

