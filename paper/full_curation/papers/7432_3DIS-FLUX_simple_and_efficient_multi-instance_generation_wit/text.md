## 3DIS-FLUX: SIMPLE AND EFFICIENT MULTIINSTANCE GENERATION WITH DIT RENDERING

# arXiv:2501.05131v1[cs.CV]9Jan2025

Dewei Zhou1, Ji Xie1, Zongxin Yang2, Yi Yang1 ∗ 1RELER, CCAI, Zhejiang University 2DBMI, HMS, Harvard University {zdw1999,sanaka87,yangyics}@zju.edu.cn {Zongxin Yang}@hms.harvard.edu

∗ corresponding author project page: https://limuloo.github.io/3DIS/

Layout 3DIS

###### 3DIS-FLUX

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

rendering w/ SD2 rendering w/ SDXL

rendering w/ FLUX

generated scene

car

person

person

table

pizza

(a) Comparison between 3DIS-FLUX and 3DIS

[Figure 5]

[Figure 6]

[Figure 7]

| | | |
|---|---|---|
| |1| |
| |3| |

|2|
|---|

generated scene depth

1) A adj. sign 2) A adj. cat 3) A girl

1) A ‘3DIS’ sign. 2) A robot cat. 1) A ‘FLUX’ sign. 2) A LEGO cat.

[Figure 8]

[Figure 9]

black lighting dog

fire car

Biden

Trump

golden frisbee

table

blue ice dog

red fire dog

pizza

(b) More results of 3DIS-FLUX

Figure 1: Images generated using our 3DIS-FLUX. Based on the user-provided layout, 3DIS (Zhou et al., 2024c) generates a scene depth map that precisely positions each instance and renders their fine-grained attributes without the need for additional training, using a variety of foundational models. Specifically, 3DIS-FLUX employs the state-of-the-art FLUX model for rendering, which is capable of producing superior image quality and offering enhanced control.

ABSTRACT

The growing demand for controllable outputs in text-to-image generation has driven significant advancements in multi-instance generation (MIG), enabling users to define both instance layouts and attributes. Currently, the state-of-the-art methods in MIG are primarily adapter-based. However, these methods necessitate retraining a new adapter each time a more advanced model is released, resulting in significant resource consumption. A methodology named Depth-Driven Decoupled Instance Synthesis (3DIS) has been introduced, which decouples MIG into two distinct phases: 1) depth-based scene construction and 2) detail rendering with widely pre-trained depth control models. The 3DIS method requires adapter training solely during the scene construction phase, while enabling various models to perform training-free detail rendering. Initially, 3DIS focused

on rendering techniques utilizing U-Net architectures such as SD1.5, SD2, and SDXL, without exploring the potential of recent DiT-based models like FLUX. In this paper, we present 3DIS-FLUX, an extension of the 3DIS framework that integrates the FLUX model for enhanced rendering capabilities. Specifically, we employ the FLUX.1-Depth-dev model for depth map controlled image generation and introduce a detail renderer that manipulates the Attention Mask in FLUX’s Joint Attention mechanism based on layout information. This approach allows for the precise rendering of fine-grained attributes of each instance. Our experimental results indicate that 3DIS-FLUX, leveraging the FLUX model, outperforms the original 3DIS method, which utilized SD2 and SDXL, and surpasses current state-of-the-art adapter-based methods in terms of both performance and image quality. Project Page: https://limuloo.github.io/3DIS/.

- 1 INTRODUCTION

With the rapid development of Diffusion Models (Ho et al., 2020; Song et al., 2020; Zhou et al., 2023; Zhao et al., 2024b; Lu et al., 2024a;b; Xie et al., 2024), contemporary models (Rombach et al., 2022; 2023; Podell et al., 2023; BlackForest, 2024) are capable of generating high-quality images. At the same time, there is a growing demand for more control over the generation process (Zhang

- et al., 2023; Liang et al., 2024; Lu et al., 2023; Zhao et al., 2024a). One prominent area of research that has garnered increasing attention is Multi-Instance Generation (MIG) (Zhou et al., 2024a;b; Wang et al., 2024; Li et al., 2023), which seeks to ensure precise alignment of each instance’s position and attributes with user-defined specifications during the generation of multiple instances.

Current strategies for Multi-Instance Generation (MIG) can be broadly classified into three categories: 1) Training-free methods, such as Multi-Diffusion (Bar-Tal et al., 2023) and RAGDiffusion (Chen et al., 2024b), which employ multiple sampling steps for each instance and later merge them based on layout information to achieve spatial control. BoxDiff (Xie et al., 2023) and TFLCG (Chen et al., 2024a) define a score function to guide the model’s sampling process, thereby enabling control over the layout. 2) Adapter-based methods, exemplified by MIGC (Zhou et al., 2024a) and InstanceDiffusion (Wang et al., 2024), which introduce layout information by training additional attention layers atop pre-trained models. 3) Text encoder fine-tuning methods, such as Reco (Yang et al., 2023) and Ranni (Feng et al., 2024), which incorporate layout information directly into the input text and subsequently fine-tune the text encoder and the whole model to embed this spatial context into the generated output.

Adapter-based methods are currently widely used due to their ability to provide strong control without the need to train the entire model. However, these approaches face two main challenges: 1) the need to retrain on different models. For example, methods like MIGC (Zhou et al., 2024a), GLIGEN (Li et al., 2023), and InstanceDiffusion (Wang et al., 2024) were initially trained on SD1.5, but as more advanced models such as SDXL and SD3 emerged, techniques like IFAdapter (Wu et al., 2024) and CreatiLayout (Zhang et al., 2024) had to be retrained accordingly. This process is resource-intensive and can be particularly difficult for users with limited GPU access. 2) Largescale instance-level annotations are often hard to obtain. Each instance generation can be viewed as a Text-to-Image task, and high-quality instance-level data is generally more challenging to acquire than high-quality image-level data.

To address the aforementioned challenges, the Depth-Driven Decouple Instance Synthesis (3DIS) (Zhou et al., 2024c) approach introduces a novel framework for Multi-Instance Generation. Instead of directly generating RGB images, 3DIS first trains a layout-to-depth model to produce a scene depth map. Then, 3DIS utilizes widely pre-trained depth control models, which only require image-level annotations, to generate images based on the layout provided by the generated scene depth map. Finally, 3DIS employs a training-free method to precisely render the attributes of each instance. The 3DIS approach offers two key advantages: 1) It requires the training of only a depth generation model, which can ignore many fine-grained attributes during training and does not necessitate high-quality visual fidelity. 2) The training-free rendering in 3DIS enables the use of various pre-trained models and better preserves the generative capabilities of large pre-trained models.

The original 3DIS paper focused on the training-free rendering approach using models based on U-Net architectures, such as SD1.5 (Rombach et al., 2022), SD2 (Rombach et al., 2023), and

##### Stage 1 Generate a coarse-grained scene depth map

[Figure 10]

1 Layout Adapter

[Figure 11]

- 1) A red knife
- 2) A orange cup
- 3) A yellow cup
- 4) A blue cup
- 5) A purple sandwich
- 6) A black sandwich

3

2 4

5

[Figure 12]

Fine-tuned Text-to-Depth Model

6

Scene Depth Map

##### Stage 2 Rendering fine-grained instance details

- 1) A red knife
- 2) A orange cup
- 3) A yellow cup
- 4) A blue cup
- 5) A purple sandwich
- 6) A black sandwich

- 4 3 1
- 5

2

[Figure 13]

[Figure 14]

Flux Detail Renderer

6

[Figure 15]

refine layout

[Figure 16]

FLUXdepth

[Figure 17]

[Figure 18]

Segment Model

High Quality Image

up-sample

[Figure 19]

[Figure 20]

Trainable Frozen

Figure 2: The overview of 3DIS-FLUX. In line with 3DIS, the 3DIS-FLUX approach decouples image generation into two distinct stages: the creation of a scene depth map and the training-free rendering of high-quality RGB images using various generative models. 3DIS-FLUX utilizes the Layout-to-Depth model from 3DIS to generate the scene depth map, and subsequently employs the FLUX-depth model to render images based on the depth map. During this process, 3DIS-FLUX incorporates an Attention Controller to ensure the accurate fine-grained attributes of each instance.

SDXL (Podell et al., 2023). With the advancement of diffusion model techniques, the Diffusion Transformer (DiT) (Li et al., 2024) architecture has demonstrated superior capabilities compared to traditional U-Net models. Specifically, FLUX has not only achieved significant improvements in image quality but has also enhanced control capabilities beyond those of previous models. As 3DIS is a flexible framework capable of quickly adapting to various new foundational models, we have extended it to propose 3DIS-FLUX, which leverages FLUX for training-free rendering, enabling stronger control and higher-quality image generation.

The overview of 3DIS-FLUX is depicted in Fig. 2. Consistent with the original 3DIS framework, 3DIS-FLUX initially generates a scene depth map using a layout-to-image model. Subsequently, we utilize the FLUX.1-depth-dev model for depth-to-image conversion. Through the application of the FLUX-depth model, alignment of the layout with the scene depth map is ensured. However, ensuring the accuracy of individual instance attributes still poses a challenge. To overcome this, we introduce a training-free Detail Renderer to achieve precise instance rendering in the Joint Attention (Liu

- et al., 2024; Dalva et al., 2024; Shin et al., 2024) of the FLUX models. Specifically, we ensure that image tokens corresponding to each instance only attend to their respective text tokens. In the early steps, we mandate that each instance’s image tokens only focus on their own image tokens. Moreover, since FLUX employs a T5 text encoder pre-trained exclusively on textual data—yielding embeddings devoid of visual information—we find that strict constraints on the attention map of text tokens within Joint Attention are crucial for successful rendering. Specifically, we restrict text tokens of each instance from attending to text tokens of other instances.

We conducted experiments on the COCO-MIG (Zhou et al., 2024a) benchmark. The results show that by using the more powerful FLUX model for rendering, 3DIS-FLUX achieved a 6.9% improvement in Instance Success Ratio (ISR) compared to the previous 3DIS-SDXL. Compared to the training-free state-of-the-art (SOTA) method Multi-Diffusion, 3DIS-FLUX’s improvement in ISR surpassed 41%. Against the SOTA adapter-based method, InstanceDiffusion, 3DIS-FLUX achieved a 12.4% higher ISR. Additionally, rendering with the FLUX model enabled our approach to demonstrate superior image quality compared to other methods.

- 2 METHOD

- 2.1 PRELIMINARIES

FLUX (BlackForest, 2024) is a recent state-of-the-art Diffusion Transformer (DiT) model that generates higher-quality images compared to previous models and demonstrates powerful text control capabilities. Given an input text, FLUX first encodes it into a text embedding using the T5 text encoder (Raffel et al., 2020). This text embedding is then concatenated with the image embedding to perform joint attention. After several iterations of joint attention (Liu et al., 2024; Shin et al., 2024; Dalva et al., 2024), the FLUX model decodes the output image embedding to produce a high-quality image that corresponds to the input text.

- 2.2 PROBLEM DEFINITION

Multi-Instance Generation (MIG) requires the generation model to simultaneously produce multiple instances, ensuring that their positions and attributes align with the user’s specifications. Given a layout P = {p1,p2,...,pn} and the textual descriptions of the instances T = {t1,t2,...,tn}, MIG demands that each instance i be generated at the specified position pi, while visually matching the description ti. Additionally, the user provides a global text c that describes the entire scene, and the generated image must be consistent with this global text.

- 2.3 OVERVIEW

Fig. 2 illustrates the overview of 3DIS-FLUX. Similar to the original 3DIS, 3DIS-FLUX decouples Multi-Instance Generation into two stages: generating the scene depth map and rendering finegrained details. In the first stage, 3DIS-FLUX employs the layout-to-depth model from 3DIS (Zhou et al., 2024c) to generate the corresponding scene depth map based on the user-provided layout. In the second stage, 3DIS-FLUX uses the FLUX.1-depth-dev (BlackForest, 2024) model to generate an image from the scene depth map, thereby controlling the layout of the generated image. To further ensure that each instance is rendered with accurate fine-grained attributes, 3DIS-FLUX incorporates a detail renderer, which constrains the attention mask during joint attention between the image and text embeddings based on the layout information.

- 2.4 FLUX DETAIL RENDERER

Motivation. Given a scene depth map generated in the first stage, FLUX.1-depth-dev model (BlackForest, 2024) is capable of producing high-quality images that adhere to the specified layout. In scenarios involving only a single instance, users can achieve precise rendering by describing the instance through a single global image text. However, challenges arise when attempting to render multiple instances accurately with just one global text description. For example, in the case illustrated in Fig. 2, rendering each “cup” in the scene depth map with designated attributes using a description such as “a photo of an orange cup, a yellow cup, and a blue cup” proves difficult. This approach frequently results in color inconsistencies, such as a cup intended to be blue being rendered as orange, with additional examples illustrated in Fig. 4. Consequently, integrating spatial constraints into the joint attention process of the FLUX model is essential for the accurate rendering of multiple instances. To overcome these challenges, we introduce a simple yet effective FLUX detail renderer that significantly enhances the precision of such renderings.

Preparation. To render multiple instances simultaneously according to the user’s descriptions, we encode not only the global image text c into fcT5 but also the instance descriptions {t1,t2,...,tn} into {f1T5,f2T5,...,fnT5}. These encoded features are concatenated to form the final text embedding F = concat(fcT5,f1T5,...,fnT5), which is then input into the FLUX model’s joint attention mechanism. Based on the user-provided layout P = {p1,p2,...,pn}, we determine the correspondence between image tokens and text tokens during the joint attention process. Since a scene depth map has already been generated in the first stage, we can opt to use the SAM (Kirillov et al., 2023) model to further optimize the user’s layout for more accurate rendering, as illustrated in Fig. 2.

Controlling the Attention of Image Embedding. The FLUX model generates images through multi-step sampling. 1) The early steps determine the primary attributes of each instance. There-

fore, it is essential to strictly avoid attribute leakage by ensuring that the image token corresponding to instance i can only attend to the image tokens within the pi region during joint attention and can only attend to its corresponding text token fiT5. 2) In the later steps, to ensure the quality of the generated image, we relax this constraint: each image token can attend to all other image tokens. Additionally, while attending to its corresponding text token fiT5, it can also attend to the global text token fcT5. We control these two phases by setting a threshold γ.

Controlling the Attention of Text Embedding. In the FLUX model, the T5 text encoder (Raffel et al., 2020), which was pretrained solely on textual data, is employed to extract text encodings. This contrasts with previous methods that utilized the CLIP text encoder (Radford et al., 2021), which was pretrained using both text and image data. Notably, we observed that during the joint attention process, the T5 text embeddings inherently lack significant semantic information. If unconstrained, they are prone to inadvertently introducing incorrect semantic information. For instance, as demonstrated in Fig. 5, when the T5 text embeddings of “black car” and “green parking meter” are concatenated and input into FLUX’s joint attention mechanism, allowing the “green parking meter” tokens to attend to the “black car” tokens results in the parking meter being predominantly black. Concurrently, it was found that FLUX was unable to successfully render the “black car” at this stage. Therefore, it is imperative to impose constraints on the attention masks of the text tokens during joint attention to avoid such semantic discrepancies. We have discovered that imposing strict attention mask constraints on the text tokens of instances throughout all steps does not significantly affect the quality of the final generated image. Therefore, throughout all steps, we restrict the text token corresponding to fiT5 during joint attention to only focus on the image tokens within the pi area and to only attend to the text token of fiT5 itself. For the text tokens of the global text tokenfcT5, we do not apply significant constraints.

- 3 EXPERIMENT

- 3.1 IMPLEMENT DETAILS

During the layout-to-depth phase, we employ the same method as used in the original 3DIS (Zhou et al., 2024c) approach. To incorporate depth control in image generation, we utilize the FLUX.1depth-dev model (BlackForest, 2024). During the image generation process, we employ a sampling strategy of 20 steps. For images with a resolution of 512, the parameter γ is set to 4. As the resolution increases, γ is adjusted accordingly: it is set to 3 for images of 768 resolution and reduced to 2 for images of 1024 resolution.

- 3.2 EXPERIMENT SETUP

Baselines. We compared our proposed 3DIS method with state-of-the-art Multi-Instance Generation approaches. The methods involved in the comparison include training-free methods: BoxDiffusion (Xie et al., 2023) and MultiDiffusion (Bar-Tal et al., 2023); and adapter-based methods: GLIGEN (Li et al., 2023), InstanceDiffusion (Wang et al., 2024), and MIGC (Zhou et al., 2024a).

Evaluation Benchmarks. We conducted experiments on the COCO-MIG (Zhou et al., 2024a) benchmark to assess a model’s ability to control the position of instances and precisely render fine-grained attributes for each generated instance. For a comprehensive evaluation, each model generated 750 images in the benchmark.

Evaluation Metrics. We used the following metrics to evaluate the model: 1) Mean Intersection over Union (MIoU), measuring the overlap between the generated instance positions and the target positions; 2) Instance Success Ratio (ISR), calculating the proportion of instances that are correctly positioned and possess accurate attributes.

- 3.3 COMPARISON

Comparison with SOTA Methods. The results presented in Tab. 1 demonstrate that the 3DIS method not only exhibits strong positional control capabilities but also robust detail-rendering capabilities. Notably, the entire process of rendering instance attributes is training-free for 3DIS. Compared to the previous state-of-the-art (SOTA) training-free method, MultiDiffusion, 3DIS-

Table 1: Quantitative results on proposed COCO-MIG-BOX (§3.3). Li means that the count of instances needed to generate in the image is i.

Instance Success Ratio↑ Mean Intersection over Union↑ Method L2 L3 L4 L5 L6 AVG L2 L3 L4 L5 L6 AVG Adapter rendering methods

GLIGEN [CVPR23] 41.3 33.8 31.8 27.0 29.5 31.3 33.7 27.6 25.5 21.9 23.6 25.2 InstanceDiff [CVPR24] 61.0 52.8 52.4 45.2 48.7 50.5 53.8 45.8 44.9 37.7 40.6 43.0

MIGC [CVPR24] 74.8 66.2 67.4 65.3 66.1 67.1 63.0 54.7 55.3 52.4 53.2 54.7

training-free rendering

TFLCG [WACV24] 17.2 13.5 7.9 6.1 4.5 8.3 10.9 8.7 5.1 3.9 2.8 5.3 BoxDiff [ICCV23] 28.4 21.4 14.0 11.9 12.8 15.7 19.1 14.6 9.4 7.9 8.5 10.6

MultiDiff [ICML23] 30.6 25.3 24.5 18.3 19.8 22.3 21.9 18.1 17.3 12.9 13.9 15.8 3DIS (SD1.5) 65.9 56.1 55.3 45.3 47.6 53.0 56.8 48.4 49.4 40.2 41.7 44.7 3DIS (SD2.1) 66.1 57.5 55.1 51.7 52.9 54.7 57.1 48.6 46.8 42.9 43.4 45.7 3DIS (SDXL) 66.1 59.3 56.2 51.7 54.1 56.0 57.0 50.0 47.8 43.1 44.6 47.0

3DIS-FLUX (FLUX) 76.4 68.4 63.3 58.1 58.9 62.9 67.3 61.2 56.4 52.3 52.7 56.2 vs. MultiDiff +46 +43 +39 +40 +39 +41 +45 +43 +39 +39 +39 +40

rendering w/ off-the-shelf adapters

3DIS+GLIGEN 49.4 39.7 34.5 29.6 29.9 34.1 43.0 33.8 29.2 24.6 24.5 28.8 vs. GLIGEN +8.1 +5.9 +2.7 +2.6 +0.4 +2.8 +9.3 +6.2 +3.7 +2.7 +0.9 +3.6 3DIS+MIGC 76.8 70.2 72.3 66.4 68.0 69.7 68.0 60.7 62.0 55.8 57.3 59.5

###### vs. MIGC +2.0 +4.0 +4.9 +1.1 +1.9 +2.6 +5.0 +6.0 +6.7 +3.4 +4.1 +4.8

3DIS (RGB, SD2)

Layout MIGC 3DIS (depth, SD1.5) 3DIS (RGB, SDXL)

3DIS (RGB, FLUX)

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

| |
|---|

Layout Instance Diffusion Ours (depth, SD1.5) Ours (RGB, SDXL)

Ours (RGB, FLUX)

Ours (RGB, SD2)

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

| |
|---|

### Figure 3: Qualitative results on the COCO-MIG (§3.3).

FLUX achieves a 41% improvement in the Instance Success Ratio (ISR). Additionally, when compared with the SOTA adapter-based method, Instance Diffusion, which requires training for rendering, 3DIS-FLUX shows a 12.4% increase in ISR. Importantly, the 3DIS approach is not mutually exclusive with existing adapter methods. For instance, combinations like 3DIS+GLIGEN and 3DIS+MIGC outperform the use of adapter methods alone, delivering superior performance. Fig. 3 offers a visual comparison between 3DIS and other SOTA methods, where it is evident that 3DIS not only excels in scene construction but also demonstrates strong capabilities in instance detail rendering. Furthermore, 3DIS is compatible with a variety of base models, offering broader applicability compared to previous methods.

Comparison of Rendering Across Different Models. As shown in Tab. 1, employing a more robust model significantly enhances the success rate of rendering. For instance, rendering with the FLUX model achieves a 9.9% higher Instance Success Ratio compared to using the SD1.5 model.

- 3.4 ABLATION STUDY

FLUX Detail Renderer. Results from Fig. 4 indicate that without employing a Detail Renderer to manage the Joint Attention process of the FLUX model, it becomes challenging to successfully render each instance in multi-instance scenarios. Additionally, data from Tab. 2 demonstrates that

Layout w/ Detail Renderer w/o Detail Renderer Layout w/ Detail Renderer w/o Detail Renderer

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

### Figure 4: Ablation Study on the FLUX Detail Renderer.

Layout w/ T2T control w/o T2T control Layout w/ T2T control w/o T2T control

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

### Figure 5: Ablation Study on Controlling Text-to-Text Attention in the FLUX Detail Renderer.

the introduction of a Detail Renderer enhances the Instance Success Ratio (ISR) by 17.8% and the Success Ratio (SR) by 12.2%. Moreover, the results from Fig. 4 also suggest that incorporating a Detail Renderer does not significantly compromise image quality.

Controlling the Attention of Image Embedding. Results from Tab. 2 show that in the Joint Attention mechanism, controlling each image token to focus solely on its corresponding instance description token (i.e., I2T control) is crucial for successfully rendering each instance, resulting in a 19.1% increase in the Instance Success Ratio (ISR). Additionally, restricting each instance’s image tokens to only attend to other image tokens belonging to the same instance (i.e., I2I control) in the earlier steps of the process also leads to a significant improvement, enhancing the ISR by 7.5%.

Table 2: Ablation study on FLUX detail renderer (§3.4).

|method|ISR ↑ MIOU ↑ SR ↑<br><br>|
|---|---|

|w/o I2I control w/o I2T control w/o T2I control w/o T2T control w/o detail renderer w/ all| |55.4 49.9 21.1 43.8 40.4 14.4 63.4 56.2 28.0 46.6 42.4 16.7 45.1 41.3 17.5 62.9 56.2 29.7<br><br>|
|---|---|---|

Controlling the Attention of Text Embedding. In contrast to models such as SD1.5 (Rombach et al., 2022), SD2 (Rombach et al., 2023), and SDXL (Podell et al., 2023), which utilize CLIP (Radford et al., 2021) as their text encoder, FLUX employs a T5 text encoder (Raffel et al., 2020). This encoder is exclusively pre-trained on textual data, resulting in embeddings that contain no visual information. Therefore, it becomes intuitively important in the joint attention mechanism to impose constraints on text tokens within multi-instance contexts. As demonstrated by the results in Tab. 2 and Fig. 5, the absence of constraints on text tokens within joint attention mechanisms—permitting a text token from one instance to attend to text tokens from other instances—significantly undermines the rendering success rate, evidenced by a substantial decrease in ISR by 16.3%. Furthermore, our analysis reveals that adding constraints, where each instance’s text tokens are restricted to only attend to their corresponding image tokens, does not result in a significant improvement.

- 4 CONCLUSION

We introduce 3DIS-FLUX, an extension of the prior 3DIS framework. The original 3DIS explored a training-free rendering approach using only the U-net architecture. In contrast, 3DIS-FLUX harnesses the state-of-the-art DiT model, FLUX, for rendering. Experiments conducted on the COCO-MIG dataset demonstrate that rendering with the more robust FLUX model allows 3DISFLUX to significantly outperform the previous 3DIS-SDXL method, and even surpass state-of-theart Adapter-based MIG approaches. The success of 3DIS-FLUX underscores the flexibility of the 3DIS framework, which can rapidly adapt to a variety of newer, more powerful models. We envision that 3DIS will enable users to utilize a broader spectrum of foundational models for multi-instance generation and expand its applicability to more diverse applications.

REFERENCES

Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. arXiv preprint arXiv:2302.08113, 2023.

BlackForest. Black forest labs; frontier ai lab, 2024. URL https://blackforestlabs.ai/. Minghao Chen, Iro Laina, and Andrea Vedaldi. Training-free layout control with cross-attention

guidance. WACV, 2024a.

Zhennan Chen, Yajie Li, Haofan Wang, Zhibo Chen, Zhengkai Jiang, Jun Li, Qian Wang, Jian Yang, and Ying Tai. Region-aware text-to-image generation via hard binding and soft refinement. arXiv preprint arXiv:2411.06558, 2024b.

Yusuf Dalva, Kavana Venkatesh, and Pinar Yanardag. Fluxspace: Disentangled semantic editing in rectified flow transformers. arXiv preprint arXiv:2412.09611, 2024.

Yutong Feng, Biao Gong, Di Chen, Yujun Shen, Yu Liu, and Jingren Zhou. Ranni: Taming text-toimage diffusion for accurate instruction following. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4744–4753, 2024.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33: 6840–6851, 2020.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023.

Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. CVPR, 2023.

Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, Dayou Chen, Jiajun He, Jiahao Li, Wenyue Li, Chen Zhang, Rongwei Quan, Jianxiang Lu, Jiabin Huang, Xiaoyan Yuan, Xiao-Ting Zheng, Yixuan Li, Jihong Zhang, Chao Zhang, Mengxi Chen, Jie Liu, Zheng Fang, Weiyan Wang, Jinbao Xue, Yang-Dan Tao, Jianchen Zhu, Kai Liu, Si-Da Lin, Yifu Sun, Yun Li, Dongdong Wang, MingDao Chen, Zhichao Hu, Xiao Xiao, Yan Chen, Yuhong Liu, Wei Liu, Dingyong Wang, Yong Yang, Jie Jiang, and Qinglin Lu. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. ArXiv, abs/2405.08748, 2024. URL https://api. semanticscholar.org/CorpusID:269761491.

Chao Liang, Fan Ma, Linchao Zhu, Yingying Deng, and Yi Yang. Caphuman: Capture your moments in parallel universes. In CVPR, 2024.

Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Chase Lambert, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving textto-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024.

Shilin Lu, Yanzhu Liu, and Adams Wai-Kin Kong. Tf-icon: Diffusion-based training-free crossdomain image composition. In ICCV, 2023.

Shilin Lu, Zilan Wang, Leyang Li, Yanzhu Liu, and Adams Wai-Kin Kong. Mace: Mass concept erasure in diffusion models. CVPR, 2024a.

Shilin Lu, Zihan Zhou, Jiayou Lu, Yuanzhi Zhu, and Adams Wai-Kin Kong. Robust watermarking using generative priors against image editing: From benchmarking to advances. arXiv preprint arXiv:2410.18775, 2024b.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, A. Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Stable diffusion version 2, 2023. URL https://stability.ai/news/ stable-diffusion-v2-release.

Chaehun Shin, Jooyoung Choi, Heeseung Kim, and Sungroh Yoon. Large-scale text-to-image model with inpainting is a zero-shot subject-driven image generator. arXiv preprint arXiv:2411.15466, 2024.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In Proc. of ICLR, 2020.

Xudong Wang, Trevor Darrell, Sai Saketh Rambhatla, Rohit Girdhar, and Ishan Misra. Instancediffusion: Instance-level control for image generation, 2024.

Yinwei Wu, Xianpan Zhou, Bing Ma, Xuefeng Su, Kai Ma, and Xinchao Wang. Ifadapter: Instance feature control for grounded text-to-image generation. arXiv preprint arXiv:2409.08240, 2024.

Jinheng Xie, Yuexiang Li, Yawen Huang, Haozhe Liu, Wentian Zhang, Yefeng Zheng, and Mike Zheng Shou. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. ICCV, 2023.

Rui Xie, Ying Tai, Chen Zhao, Kai Zhang, Zhenyu Zhang, Jun Zhou, Xiaoqian Ye, Qian Wang, and Jian Yang. Addsr: Accelerating diffusion-based blind super-resolution with adversarial diffusion distillation. arXiv preprint arXiv:2404.01717, 2024.

Zhengyuan Yang, Jianfeng Wang, Zhe Gan, Linjie Li, Kevin Lin, Chenfei Wu, Nan Duan, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Reco: Region-controlled text-to-image generation. In CVPR, 2023.

Hui Zhang, Dexiang Hong, Tingwei Gao, Yitong Wang, Jie Shao, Xinglong Wu, Zuxuan Wu, and Yu-Gang Jiang. Creatilayout: Siamese multimodal diffusion transformer for creative layout-toimage generation. arXiv preprint arXiv:2412.03859, 2024.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, pp. 3836–3847, 2023.

Chen Zhao, Weiling Cai, Chenyu Dong, and Chengwei Hu. Wavelet-based fourier information interaction with frequency diffusion adjustment for underwater image restoration. CVPR, 2024a.

Chen Zhao, Chenyu Dong, and Weiling Cai. Learning a physical-aware diffusion model based on transformer for underwater image enhancement. arXiv preprint arXiv:2403.01497, 2024b.

Dewei Zhou, Zongxin Yang, and Yi Yang. Pyramid diffusion models for low-light image enhancement. In IJCAI, 2023.

Dewei Zhou, You Li, Fan Ma, Zongxin Yang, and Yi Yang. Migc: Multi-instance generation controller for text-to-image synthesis. CVPR, 2024a.

Dewei Zhou, You Li, Fan Ma, Zongxin Yang, and Yi Yang. Migc++: Advanced multi-instance generation controller for image synthesis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024b.

Dewei Zhou, Ji Xie, Zongxin Yang, and Yi Yang. 3dis: Depth-driven decoupled instance synthesis for text-to-image generation. arXiv preprint arXiv:2410.12669, 2024c.

