# arXiv:2410.23775v3[cs.CV]5Nov2024

## IN-CONTEXT LORA FOR DIFFUSION TRANSFORMERS

TECHNICAL REPORT

Lianghua Huang Wei Wang Zhi-Fan Wu

Yupeng Shi Huanzhang Dou Chen Liang Yutong Feng Yu Liu Jingren Zhou

Tongyi Lab*

### ABSTRACT

Recent research [Huang et al., 2024] has explored the use of diffusion transformers (DiTs) for task-agnostic image generation by simply concatenating attention tokens across images. However, despite substantial computational resources, the fidelity of the generated images remains suboptimal. In this study, we reevaluate and streamline this framework by hypothesizing that text-to-image DiTs inherently possess in-context generation capabilities, requiring only minimal tuning to activate them. Through diverse task experiments, we qualitatively demonstrate that existing textto-image DiTs can effectively perform in-context generation without any tuning. Building on this insight, we propose a remarkably simple pipeline to leverage the in-context abilities of DiTs: (1) concatenate images instead of tokens, (2) perform joint captioning of multiple images, and (3) apply task-specific LoRA tuning using small datasets (e.g., 20 ∼ 100 samples) instead of fullparameter tuning with large datasets. We name our models In-Context LoRA (IC-LoRA). This approach requires no modifications to the original DiT models, only changes to the training data. Remarkably, our pipeline generates high-fidelity image sets that better adhere to prompts. While task-specific in terms of tuning data, our framework remains task-agnostic in architecture and pipeline, offering a powerful tool for the community and providing valuable insights for further research on product-level task-agnostic generation systems. We release our code, data, and models at https://github.com/ali-vilab/In-Context-LoRA.

Keywords In-context LoRA · Diffusion transformers · Image generation

### 1 Introduction

The advent of text-to-image models has significantly advanced the field of visual content generation, enabling the creation of high-fidelity images from textual descriptions [Ramesh et al., 2021, 2022, Esser et al., 2021, Rombach et al., 2022, Saharia et al., 2022a, Betker et al., 2023, Podell et al., 2023, Esser et al., 2024, Baldridge et al., 2024, Labs, 2024]. Numerous methods now offer enhanced control over various image attributes, allowing for finer adjustments during generation [Zhang et al., 2023, Ye et al., 2023, Huang et al., 2023, Ruiz et al., 2023, Wang et al., 2024a, Hertz et al., 2024]. Despite these strides, adapting text-to-image models to a broad spectrum of generative tasks—particularly those requiring coherent image sets with complex intrinsic relationships—remains a open challenge. In this work, we introduce a task-agnostic framework designed to adapt text-to-image models to diverse generative tasks, aiming to provide a universal solution for versatile and controllable image generation.

∗Emails: Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yutong Feng, Yu Liu, Jingren Zhou {xuangen.hlh, ww413411, wuzhifan.wzf, fengyutong.fyt, ly103369, jingren.zhou}@alibaba-inc.com, and Yupeng Shi (shiyupeng.syp@taobao.com). Huanzhang Dou (hzdou@zju.edu.cn, Zhejiang University) and Chen Liang (liangchen2022@ia.ac.cn, Institute of Automation, Chinese Academy of Sciences) contributed to this work during internships at Tongyi Lab.

[Figure 1]

[Figure 2]

In-Context LoRA for Diffusion Transformers A PREPRINT

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

- Figure 1: In-Context LoRA Generation Examples. Three tasks from top to bottom: portrait photography, font design, and home decoration. For each task, four images are generated simultaneously within a single diffusion process using In-Context LoRA models that are tuned specifically for each task.

[Figure 13]

In-Context LoRA for Diffusion Transformers A PREPRINT

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

- Figure 2: In-Context LoRA Generation Examples. Three tasks from top to bottom: sandstorm visual effect, portrait illustration, and visual identity design. For each task, an image pair is generated simultaneously within a single diffusion process. The further application of SDEdit for image-conditional generation will be discussed later in the paper.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

- Figure 3: FLUX Text-to-Image Generation Examples. Examples of text-to-image generation across six tasks using FLUX.1-dev, highlighting the creation of multi-panel images with varied relational attributes. Key observations include:

(1) The original text-to-image model can already generate multi-panel outputs with coherent consistency in identity, style, lighting, and font, though some minor imperfections remain. (2) FLUX.1-dev shows a strong capability in interpreting combined prompts that describe multiple panels, as further detailed in Appendix A.

Recent efforts, such as the Group Diffusion Transformers (GDT) framework [Huang et al., 2024], have explored reformulating visual generative tasks as a group generation problem. In this context, a set of images with arbitrary intrinsic relationships is generated simultaneously within a single denoising diffusion process, optionally conditioned on another set of images. The core idea of GDT involves concatenating attention tokens across images—both the conditional ones and those to be generated—while ensuring that the tokens of each image attend exclusively to their corresponding text tokens. This approach allows the model to adapt to multiple tasks in a task-agnostic, zero-shot manner without any fine-tuning or gradient updates.

However, despite its innovative architecture, GDT exhibits relatively low generation fidelity, often underperforming compared to the original pretrained text-to-image models. This limitation prompts a re-examination of the underlying assumptions and methodologies employed in adapting text-to-image models for complex generative tasks.

In this work, we make a pivotal assumption: text-to-image models inherently possess in-context generation capabilities. To validate this, we directly apply existing text-to-image models to a variety of tasks that require generating sets of images with diverse relationships. As illustrated in Figure 3, using the FLUX.1-dev model [Labs, 2024] as an example, we observe that the model can already perform different tasks, albeit with some imperfections. It maintains consistent attributes such as subject identities, styles, lighting conditions, and color palettes while modifying other aspects like poses, 3D orientations, and layouts. Moreover, the model demonstrates the ability to interpret and follow descriptions of multiple images within a single merged prompt, as detailed in Appendix A.

These surprising findings lead us to several key insights:

- 1. Inherent In-Context Learning: Text-to-image models already possess in-context generation abilities. By appropriately triggering and enhancing this capability, we can leverage it for complex generative tasks.
- 2. Model Reusability Without Architectural Modifications: Since text-to-image models can interpret merged captions, we can reuse them for in-context generation without any changes to their architecture. This involves simply altering the input data rather than modifying the model itself.
- 3. Efficiency with Minimal Data and Computation: High-quality results can be achieved without large datasets or prolonged training times. Small, high-quality datasets coupled with minimal computational resources may be sufficient.

Building upon these insights, we design an extremely simple yet effective pipeline for adapting text-to-image models to diverse tasks. Our approach contrasts with GDT in the following ways:

- 1. Image Concatenation: We concatenate a set of images into a single large image instead of concatenating attention tokens. This method is approximately equivalent to token concatenation in diffusion transformers (DiTs), disregarding differences introduced by the Variational Autoencoder (VAE) component.
- 2. Prompt Concatenation: We merge per-image prompts into one long prompt, enabling the model to process and generate multiple images simultaneously. This differs from the GDT approach, where each image’s tokens cross-attend exclusively to its text tokens.
- 3. Minimal Fine-Tuning with Small Datasets: Instead of performing large-scale training on hundreds of thousands of samples, we fine-tune a Low-Rank Adaptation (LoRA) of the model using a small set of just 20 ∼ 100 image sets. This approach significantly reduces the computational resources required and largely preserves the original text-to-image model’s knowledge and in-context capabilities.

The resulting model is remarkably simple, requiring no modifications to the original text-to-image models. Adaptation is achieved solely by adjusting a small set of tuning data according to specific task needs. To support image-conditional generation, we employ a straightforward technique: we mask one or multiple images in the concatenated large image and prompt the model to inpaint them using the remaining images. We directly utilize SDEdit [Meng et al., 2021] for this purpose.

Despite its simplicity, we find that our method can adapt to a diverse array of tasks with high quality. Figures 1 and 2 illustrate example outputs for various tasks, while Figures 4–12 present more specific cases by task, and Figures 13 and 14 demonstrate image-conditional generation results. While our approach requires task-specific tuning data, the overall framework and pipeline remain task-agnostic, allowing adaptation to a wide variety of tasks without modifying the original model architecture. This combination of minimal data requirements and broad applicability offers a powerful tool for the generative community, designers, and artists. We acknowledge that developing a fully unified generation system remains an open challenge and leave it as future work. To facilitate further research, we release our data, models, and training configurations at the project page2.

### 2 Related Work

#### 2.1 Task-Specific Image Generation

Text-to-image models have achieved remarkable success in generating high-fidelity images from complex textual prompts [Ramesh et al., 2021, 2022, Esser et al., 2021, Rombach et al., 2022, Saharia et al., 2022a, Betker et al., 2023, Podell et al., 2023, Chen et al., 2023, Esser et al., 2024, Baldridge et al., 2024, Labs, 2024]. However, they often lack fine-grained controllability over specific attributes of the generated images. To address this limitation, numerous works have been proposed to enhance control over aspects such as layouts [Zheng et al., 2023, Huang et al., 2023], poses [Zhang et al., 2023], identities Huang et al. [2023], Ye et al. [2023], Li et al. [2024], Wang et al. [2024a], color palettes [Huang et al., 2023], styles Hertz et al. [2024], Huang et al. [2023], regions [Meng et al., 2021, Lugmayr et al., 2022, Xie et al., 2022, Huang et al., 2023], handles [Pan et al., 2023, Shi et al., 2023, Liu et al., 2024a], distorted images [Saharia et al., 2022b, Kawar et al., 2022, Xia et al., 2023, Li et al., 2023] and lighting conditions [Zhang et al., 2023]. Some methods even support the simultaneous generation of multiple images, akin to our approach [Zhou et al., 2024a, Liu et al., 2024b, Yang et al., 2024, Chern et al., 2024, Huang et al., 2024].

2Project page: https://ali-vilab.github.io/In-Context-Lora-Page/

Despite these advancements, these models typically employ task-specific architectures and pipelines, limiting their flexibility and generalizability. Each architecture is tailored to individual tasks, and the capabilities developed for one task are not easily composable or extendable to arbitrary new tasks. This contrasts with recent progress in natural language processing [Radford et al., 2019, Brown, 2020, Touvron et al., 2023a,b, Dubey et al., 2024, Team et al., 2024], where models are designed to perform multiple tasks within a single architecture and can generalize beyond the tasks they were explicitly trained on.

- 2.2 Task-Agnostic Image Generation

To overcome the constraints of task-specific models, recent research has aimed at creating task-agnostic frameworks that support multiple controllable image generation tasks within a single architecture [Ge et al., 2023, Zhou et al., 2024b, Sheynin et al., 2024, Sun et al., 2024, Wang et al., 2024b]. For example, Emu Edit [Sheynin et al., 2024] integrates a broad array of image editing functions, while models like Emu2 [Sun et al., 2024], Emu3 [Wang et al., 2024b], TransFusion [Zhou et al., 2024b], Show-o [Xie et al., 2024], and OmniGen [Xiao et al., 2024] perform diverse tasks, from procedural drawing to subject-driven generation, within a unified model. Emu3 further extends this capability by supporting text, image, and video generation under a single framework. These works represent substantial advancements in the unified or task-agnostic generation.

In contrast to these models, we propose that existing text-to-image architectures already possess inherent in-context capabilities. This eliminates the need for developing new architectures, and enabling high-quality generation with minimal additional data and computational resources. Our approach not only enhances efficiency but also delivers superior generation quality across a wide array of tasks.

- 3 Method

- 3.1 Problem Formulation

Following the approach of Group Diffusion Transformers [Huang et al., 2024], we frame most image generation tasks as producing a set of n ≥ 1 images, conditioned on another set of m ≥ 0 images and (n + m) text prompts. This formulation encompasses a broad range of academic tasks, such as image translation, style transfer, pose transfer, and subject-driven generation, as well as practical applications like picture book creation, font design and transfer, storyboard generation, and more [Huang et al., 2024]. The correlations among both the conditional images and the generated images are implicitly maintained through the per-image prompts.

Our approach slightly modifies this framework by using a single consolidated prompt for the entire image set. This prompt typically begins with an overall description of the image set, followed by individual prompts for each image. This unified prompt design is more compatible with existing text-to-image models and allows the overall description to naturally convey the task’s intent, much like how clients communicate design requirements to artists.

- 3.2 Group Diffusion Transformers

We begin with the base framework, Group Diffusion Transformers (GDT) [Huang et al., 2024]. In GDT, a set of images are generated simultaneously within a single diffusion process by concatenating attention tokens across images in each Transformer self-attention block. This approach enables each image to "see" and interact with all other images in the set. Text conditioning is introduced by having each image attend to its corresponding text embeddings, allowing it to access both the content of other images and relevant text guidance.

GDT is trained on hundreds of thousands of image sets, enabling it to generalize across tasks in a zero-shot manner.

- 3.3 In-Context LoRA

Although GDT demonstrates zero-shot task adaptability, its generation quality falls short, often underperforming compared to baseline text-to-image models. We propose enhancements to improve this framework.

Our starting point is the assumption that base text-to-image models inherently possess some in-context generation capabilities for diverse tasks, even if quality varies. This is supported by results in Figure 3, where the model effectively generates multiple images (sometimes with conditions) across different tasks. Based on this insight, extensive training on large datasets is unnecessary; we can instead activate the model’s in-context abilities with carefully curated, high-quality image sets.

Another observation is that text-to-image models can generate coherent multi-panel images from a single prompt containing descriptions of multiple panels (see Figure 3 Appendix A). Thus, we can simplify the architecture by using consolidated image prompts instead of requiring each image to attend exclusively to its respective text tokens. This allows us to reuse the original text-to-image architecture without any structural modifications.

Our final framework design generates a set of images simultaneously by directly concatenating them into a single large image during training, while consolidating their captions into one merged prompt with an overarching description and clear guidance for each panel. After generating the image set, we split the large image into individual panels. Furthermore, since text-to-image models already demonstrate in-context capabilities, we don’t fine-tune the entire model. Instead, we apply Low-Rank Adaptation (LoRA) on a small set of high-quality data to trigger and enhance these capabilities.

To support conditioning on an additional set of images, we employ SDEdit, a training-free method, to inpaint a set of images based on an unmasked set, all concatenated within a single large image.

### 4 Experiments

#### 4.1 Implementation Details

We build our approach on the FLUX.1-dev text-to-image model [Labs, 2024] and train an In-Context LoRA specifically for our tasks. We select a range of practical tasks, including storyboard generation, font design, portrait photography, visual identity design, home decoration, visual effects, portrait illustration, and PowerPoint template design, among others. For each task, we collect 20 to 100 high-quality image sets from the internet. Each set is concatenated into a single composite image, and captions for these images are generated using Multi-modal Large Language Models (MLLMs), starting with an overall summary followed by detailed descriptions for each image. Training is conducted on a single A100 GPU for 5,000 steps with a batch size of 4 and a LoRA rank of 16. For inference, we employ 20 sampling steps with a guidance scale of 3.5, matching the distillation guidance scale of FLUX.1-dev. For image-conditional generation, SDEdit is applied to mask images intended for generation, enabling inpainting based on the surrounding images.

#### 4.2 Results

We present qualitative results demonstrating the versatility and quality of our model across various tasks. Given the wide diversity of tasks, we defer a unified quantitative benchmark and evaluation to future work.

#### 4.2.1 Reference-Free Image-Set Generation

In this setting, image sets are generated solely from text prompts, with no additional image input. Examples from a range of tasks are presented in Figures 4–12. Our method achieves high-quality results across a spectrum of image-set generation tasks.

#### 4.2.2 Reference-Based Image-Set Generation

In this setting, image sets are generated using both a text prompt and an input image set (with at least one reference image). SDEdit is applied to mask certain images, enabling inpainting based on the remaining ones. Results for image-conditioned generation are presented in Figure 13, with common failure cases shown in Figure 14. Although effective across multiple tasks, visual consistency across images is sometimes lower compared to text-conditioned generation. This discrepancy may result from SDEdit’s unidirectional dependency between masked and unmasked images, whereas text-only generation allows bidirectional dependencies among images, enabling mutual adjustment of conditions and outputs. This suggests potential for improvement, such as incorporating a trainable inpainting method, which we leave for future exploration.

[Figure 32]

[Figure 33]

[Figure 34]

In-Context LoRA for Diffusion Transformers A PREPRINT

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

- Figure 4: Film Storyboard Generation. Each set of three images is generated simultaneously using In-Context LoRA. A placeholder character name wrapped in "<" and ">" uniquely references the character’s identity across the images, ensuring consistent portrayal throughout the storyboard.

[Figure 47]

[Figure 48]

In-Context LoRA for Diffusion Transformers A PREPRINT

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

- Figure 5: Portrait Photography. Each set of four images is generated simultaneously using In-Context LoRA. Consistent subject identities are maintained across all images within each set, as illustrated in the figure.

[Figure 59]

[Figure 60]

In-Context LoRA for Diffusion Transformers A PREPRINT

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

- Figure 6: Home Decoration. Each set of four images is generated simultaneously using In-Context LoRA, showcasing a consistent decoration style across all images within each set.

In-Context LoRA for Diffusion Transformers A PREPRINT

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

- Figure 7: Couple Profile Generation. Each pair of images is generated simultaneously using In-Context LoRA, maintaining consistent style and identity features across both images in each set.

[Figure 83]

[Figure 84]

In-Context LoRA for Diffusion Transformers A PREPRINT

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

- Figure 8: Font Design. Each set of four images is generated simultaneously using In-Context LoRA, ensuring a consistent font style throughout all images in each set.

[Figure 103]

[Figure 104]

In-Context LoRA for Diffusion Transformers A PREPRINT

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

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

- Figure 9: PowerPoint Template Design. Each set of four images is generated simultaneously using In-Context LoRA, achieving a cohesive and unified presentation style across all slides within each set.

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

##### Figure 10: Visual Identity Design. Each pair of images is generated simultaneously using In-Context LoRA, ensuring a cohesive and consistent visual identity across both images in each pair.

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

##### Figure 11: Sandstorm Visual Effect. Each pair of images is generated using In-Context LoRA, demonstrating strong consistency between the "before" and "after" sandstorm effect images. For examples of image-conditional generation, please refer to Figure 13.

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

- Figure 12: Portrait Illustration. Each pair of images is generated using In-Context LoRA, maintaining consistent identity, clothing, expression, similar pose, and atmosphere between the "before" and "after" illustration versions. Rather than directly copying the original photo, the illustration artistically enhances key features, adding expressive emphasis. For additional examples of image-conditional generation, please refer to Figure 13.

[Figure 151]

[Figure 152]

In-Context LoRA for Diffusion Transformers A PREPRINT

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

- Figure 13: Image-Conditional Generation. Examples of image-conditional generation using In-Context LoRA across multiple tasks with training-free SDEdit. In some instances, such as the Application of Sandstorm Visual Effect case, inconsistencies may arise between input and output images, including changes in the motor driver’s identity and attire. Addressing these inconsistencies is left for future work.

[Figure 168]

[Figure 169]

In-Context LoRA for Diffusion Transformers A PREPRINT

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

- Figure 14: Failure Cases of Image-Conditional Generation. Examples of portrait identity transfer failure using In-Context LoRA with SDEdit. We observe that SDEdit for In-Context LoRA tends to be unstable, often failing to preserve identity. This may stem from a discrepancy between SDEdit’s unidirectional dependency on input-to-output mapping and the bidirectional nature of In-Context LoRA training. Addressing this issue is left for future work.

### References

Lianghua Huang, Wei Wang, Zhi-Fan Wu, Huanzhang Dou, Yupeng Shi, Yutong Feng, Chen Liang, Yu Liu, and Jingren

Zhou. Group diffusion transformers are unsupervised multitask learners. arXiv preprint arXiv:2410.15027, 2024. Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever.

Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021. Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image

generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022. Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022a.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

Jason Baldridge, Jakob Bauer, Mukul Bhutani, Nicole Brichtova, Andrew Bunner, Kelvin Chan, Yichang Chen, Sander Dieleman, Yuqing Du, Zach Eaton-Rosen, et al. Imagen 3. arXiv preprint arXiv:2408.07009, 2024.

Black Forest Labs. Flux: Inference repository. https://github.com/black-forest-labs/flux, 2024. Accessed: 2024-10-25.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500–22510, 2023.

Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024a.

Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4775–4785, 2024. Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided

image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.

Guangcong Zheng, Xianpan Zhou, Xuewei Li, Zhongang Qi, Ying Shan, and Xi Li. Layoutdiffusion: Controllable diffusion model for layout-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22490–22499, 2023.

Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11461–11471, 2022.

Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, and Kun Zhang. Smartbrush: Text and shape guided object inpainting with diffusion model, 2022.

Xingang Pan, Ayush Tewari, Thomas Leimkühler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. Drag your gan: Interactive point-based manipulation on the generative image manifold. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023.

Yujun Shi, Chuhui Xue, Jiachun Pan, Wenqing Zhang, Vincent YF Tan, and Song Bai. Dragdiffusion: Harnessing diffusion models for interactive point-based image editing. arXiv preprint arXiv:2306.14435, 2023.

Haofeng Liu, Chenshu Xu, Yifei Yang, Lihua Zeng, and Shengfeng He. Drag your noise: Interactive point-based editing via diffusion semantic propagation, 2024a.

Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image superresolution via iterative refinement. IEEE transactions on pattern analysis and machine intelligence, 45(4):4713–4726, 2022b.

Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. In Advances in Neural Information Processing Systems, 2022.

Bin Xia, Yulun Zhang, Shiyin Wang, Yitong Wang, Xinglong Wu, Yapeng Tian, Wenming Yang, and Luc Van Gool. Diffir: Efficient diffusion model for image restoration, 2023. URL https://arxiv.org/abs/2303.09472.

Xin Li, Yulin Ren, Xin Jin, Cuiling Lan, Xingrui Wang, Wenjun Zeng, Xinchao Wang, and Zhibo Chen. Diffusion models for image restoration and enhancement–a comprehensive survey. arXiv preprint arXiv:2308.09388, 2023.

Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent self-attention for long-range image and video generation. arXiv preprint arXiv:2405.01434, 2024a.

Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, Yanfeng Wang, and Weidi Xie. Intelligent grimm - open-ended visual storytelling via latent diffusion models. In The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6190–6200, 2024b.

Shuai Yang, Yuying Ge, Yang Li, Yukang Chen, Yixiao Ge, Ying Shan, and Yingcong Chen. Seed-story: Multimodal long story generation with large language model. arXiv preprint arXiv:2407.08683, 2024.

Ethan Chern, Jiadi Su, Yan Ma, and Pengfei Liu. Anole: An open, autoregressive, native large multimodal models for interleaved image-text generation. arXiv preprint arXiv:2407.06135, 2024.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are

unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. Tom B Brown. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste

Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, et al. Gemini: A family of highly capable multimodal models, 2024.

Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, and Ying Shan. Making llama see and draw with seed tokenizer. arXiv preprint arXiv:2310.01218, 2023.

Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model, 2024b.

Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8871–8879, 2024.

Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14398–14409, 2024.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024b.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024.

### A Prompts of Figure 3

Below are the detailed prompts used for each subfigure in Figure 3:

- (a) Portrait Photography. This four-panel image captures a young boy’s adventure in the woods, expressing curiosity and wonder. [TOP-LEFT] He crouches beside a stream, peering intently at a group of frogs jumping along the rocks, his face full of excitement; [TOP-RIGHT] he climbs a low tree branch, arms stretched wide as he balances, a big grin on his face; [BOTTOM-LEFT] a close-up shows him kneeling in the dirt, inspecting a bright yellow mushroom with fascination; [BOTTOM-RIGHT] the boy runs through a clearing, his arms spread out like airplane wings, lost in the thrill of discovery.
- (b) Product Design. The image showcases a modern and multifunctional baby walker designed for play and growth, featuring its versatility and attention to detail; [TOP-LEFT] the first panel highlights the walker’s sleek form with several interactive toys on the tray, focusing on its overall structure and functionality, [TOP-RIGHT] the second panel provides a side view emphasizing the built-in lighting around the play tray, illustrating both style and safety features, [BOTTOM-LEFT] the third panel displays a rear view of the walker showcasing the comfortable seat and adjustable design elements, underlining comfort and adaptability, [BOTTOM-RIGHT] while the final panel offers a close-up of the activity center with various colorful toys, capturing its playful appeal and engagement potential.
- (c) Font Design. The four-panel image emphasizes the versatility of a minimalist sans-serif font across various elegant settings: [TOP-LEFT] displays the word “Essence” in muted beige, featured on a luxury perfume bottle with a marble backdrop; [TOP-RIGHT] shows the phrase “Pure Serenity” in soft white, set against an image of serene, rippling water; [BOTTOM-LEFT] showcases “Breathe Deep” in pale blue, printed on a calming lavender candle, evoking a spa-like atmosphere; [BOTTOM-RIGHT] features “Elegance Defined” in charcoal gray, embossed on a sleek hardcover notebook, emphasizing sophistication and style.
- (d) Sandstorm Visual Effect. The two-panel image showcases a biker speeding through a desert landscape before and after a sandstorm effect, capturing a powerful transformation; [TOP] the first panel presents a biker riding along a dirt path, with the vast desert and blue sky stretching out behind them, conveying a sense of freedom and adventure, while [BOTTOM] the second panel introduces a violent sandstorm, with grains of sand swirling around the biker and partially obscuring the landscape, transforming the scene into a chaotic and thrilling visual spectacle.
- (e) Visual Identity Design. This two-panel image captures the essence of a visual identity design and its adaptable application, showcasing both the original concept and its practical derivative use; [LEFT] the left panel presents a bright and engaging graphic featuring a stylized gray koala character triumphantly holding a large wedge of cheese on a vibrant yellow background, using bold black outlines to emphasize the simplicity and playfulness of the design, while [RIGHT] the right panel illustrates the design’s extension to everyday objects, where the same koala and cheese motif has been skillfully adapted onto a circular coaster with a softer yellow tone, accompanied by a matching mug bearing smaller graphics of the koala and cheese, both items resting elegantly on a minimalist white table, highlighting the versatility and cohesive appeal of the visual identity across different mediums.
- (f) Portrait Illustration. This two-panel image showcases a transformation from a photographic portrait to a playful illustration; [LEFT] the first panel displays a man in a navy suit, white shirt, and brown shoes, sitting on a wooden bench in an urban park, his hand resting casually on his lap; [RIGHT] the illustration panel transforms him into a cartoon-like character, with smooth lines and exaggerated features, including oversized shoes and a vibrant blue suit, set against a minimalist park backdrop, giving the scene a lively and humorous feel.

