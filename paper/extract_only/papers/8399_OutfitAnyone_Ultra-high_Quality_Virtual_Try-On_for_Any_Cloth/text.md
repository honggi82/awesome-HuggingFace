## OutfitAnyone: Ultra-high Quality Virtual Try-On for Any Clothing and Any Person

### Ke Sun1∗ Jian Cao1∗ Qi Wang1 Linrui Tian1 Xindi Zhang1 Lian Zhuo1 Bang Zhang1 Liefeng Bo1 Wenbo Zhou2 Weiming Zhang2 Daiheng Gao2,3 1Intelligent Computing, Tongyi, Alibaba Group 2USTC 3Formation.ai

# arXiv:2407.16224v1[cs.CV]23Jul2024

[Figure 1]

Figure 1. We introduce OutfitAnyone, a diffusion-based framework for 2D Virtual Try-On. By far, it has garnered over 5,000 stars on GitHub and ranked within the top 20 among all the Hugging Face spaces.

### Abstract

Virtual Try-On (VTON) has become a transformative technology, empowering users to experiment with fashion without ever having to physically try on clothing. However, existing methods often struggle with generating highfidelity and detail-consistent results. While diffusion models, such as Stable Diffusion 1/2/3, have shown their capability in creating high-quality and photorealistic images, they encounter formidable challenges in conditional generation scenarios like VTON. Specifically, these models strug-

*These authors contributed equally to this work

gle to maintain a balance between control and consistency when generating images for virtual clothing trials.

OutfitAnyone addresses these limitations by leveraging a two-stream conditional diffusion model, enabling it to adeptly handle garment deformation for more lifelike results. It distinguishes itself with scalability—modulating factors such as pose, body shape and broad applicability, extending from anime to in-the-wild images. OutfitAnyone’s performance in diverse scenarios underscores its utility and readiness for real-world deployment. For more details and animated results, please see https: //humanaigc.github.io/outfit-anyone/.

### 1. Introduction

The concept of Virtual Try-On (VTON) is centered around the ability to digitally simulate how a piece of clothing would appear on an individual, using their photograph and an image of the clothing item. This technology holds the promise of significantly enriching the online shopping experience. However, the effectiveness of most VTON methods is often limited to scenarios where there is minimal variation in body posture and shape. A critical unresolved challenge lies in the accurate non-rigid transformation of clothing to conform to a specific body shape without causing any distortion to the garment’s patterns and textures [4, 9, 13, 30].

Currently, two primary approaches are being explored to address these challenges. The first is the template-based 3D Virtual Try-On (VTON), which has proven to be effective in tackling these issues, as demonstrated in various studies [8, 20, 21, 31]. The underlying technology of these methods involves converting 2D images into 3D textures for clothing mesh models. The crux of creating 3D textures from 2D images lies in establishing accurate correspondences between the catalog images and the UV textures, either manually using techniques like the Thin-Plate-Spline (TPS) warping [2] or automatically through the As-Rigid-As-Possible (ARAP) deformation [28] loss and the differentiable neural rendering [17].

While 3D Virtual Try-On (VTON) struggles to achieve the level of realism and diversity needed for a wide range of garments, it also faces the issue of longer processing times during inference, which is a significant disadvantage compared to 2D counterparts.

TryOnDiffusion [35], pioneering as the first VTON technique to harness the power of diffusion models, stands out as a quintessential algorithm in the second approach. It skillfully navigates significant occlusions, diverse poses, and alterations in body contours, meticulously maintaining the fine details of garments with high-resolution clarity. However, TryOnDiffusion can only solve single garment Try-On, which makes it utterly impractical for real-world usage.

Drawing inspiration from the innovative Parallel-UNet design featured in TryOnDiffusion, we’ve crafted OutfitAnyone, a cutting-edge technology dedicated to delivering ultra-high definition virtual try-ons for a wide array of clothing on any individual. It’s the ultimate solution for handling large occlusions, a variety of poses and body shapes, and an extensive range of garments.

At the heart of our approach lies a conditional Diffusion Model that meticulously processes images of the model, the clothing, and accompanying textual prompts, harnessing garment images as a guiding influence. The architecture of the network is bifurcated into two distinct pathways, each independently handling the model and clothing data.

These pathways merge within a sophisticated fusion network that adeptly integrates the intricacies of the garment onto the model.

which encompasses two pivotal components: the Zeroshot Try-on Network that generates the initial try-on visuals, and the Post-hoc Refiner that meticulously refines the clothing and skin textures in the final output images.

In summary, we contribute OutfitAnyone, which encompasses two pivotal components: the Zero-shot Try-on Network that generates the initial try-on visuals, and the Post-hoc Refiner that meticulously refines the clothing and skin textures in the final output images.

Specifically, The characteristics of OutfitAnyone can be summarized as follows:

- • a) Cutting-edge Realism: Our OutfitAnyone method sets a new industry standard for Virtual Try-On, delivering industry-leading, high-quality results.
- • b) High Robustness: The method we propose can support virtual try-on for anyone, any outfits, any body shape and any scenario.
- • c) Flexible Control: We support various pose and body shape guidance methods, including (openpose [3], SMPL [19], densepose [12]).
- • d) High Quality: We support flexible sizes VTON synthesizing, from 384 (width) × 684 (height) to 1080 (width) × 1920 (height).

### 2. Related Works

GAN-Based Virtual Try-on. Traditional approaches to virtual try-on technology [1, 6, 7, 14, 16, 30] typically involve a two-step process that leverages Generative Adversarial Networks (GANs) [10]. Initially, a precise warping module is utilized to adjust the shape of in-shop clothing to match the desired silhouette. Subsequently, the second phase deploys a GAN-driven generator to seamlessly integrate the reshaped attire onto the subject’s image.

The realism of these GAN-based methods hinges largely on the proficiency of the initial warping phase, prompting a concerted effort to bolster the warping module’s proficiency in managing non-rigid deformations. Beyond this, the quality of GANs has faced some skepticism, particularly in the last two years, as diffusion models have showcased their prowessful capability to generate data from all kinds (image, video, 3D and etc).

Diffusion-Based Virtual Try-On. In contrast to GANbased models, diffusion models have taken considerable leaps forward in the realm of high-fidelity conditional image generation [23, 25, 27, 29, 33]. The process of imagebased virtual try-on can be viewed as a specialized subset of the general image editing or restoration task, tailored to the specific conditions set by the provided garment and model.

DCI-VTON [11] and LADI-VTON [22] stand out as two seminal works that aim to bridge the gap between

traditional GAN-based methodologies and diffusion models. These works utilize explicit warping modules to generate deformed garments and subsequently leverage diffusion models to seamlessly integrate these garments with images of the reference person.

TryonDiffusion [35] has taken a bold step by forgoing the integration of garment warping modules into its process. This decision marks a significant departure from the traditional approach, as it removes the necessity for explicit warping and feature alignment mechanisms. It’s commendable that TryonDiffusion has paved the way for an alternative in VTON technology, one that doesn’t rely on the inclusion of meticulously crafted warping modules. This innovation has alleviated the substantial workload for researchers and engineers who previously had to invest significant effort in fine-tuning warping modules within the network—a task that has now become obsolete.

While TryonDiffusion has not yet fully exploited the potential of current Large Language Models (LLMs) and has opted to disregard text as an input method, MMTryOn [34] adeptly leverages the prowess of expansive multi-modal models. This sophisticated strategy facilitates a more refined and versatile interaction with diverse apparel styles and Virtual Try-On contexts. Simultaneously, the concurrent research by OOTDiffusion [32] ingeniously employs CLIP [24] to encode garment labels for upper body, lower body, and full-body try-on applications, enhancing the precision and adaptability of the system.

### 3. Overall Framework

As shown in the Fig. 2, we developed a framework incorporating ReferenceNet, which effectively maintains the integrity of pattern and texture information from clothing images when they are used as conditions in the main generation pipeline. This design ensures that both the fit and visual details of the clothing are accurately preserved throughout the generation process. Internally, the network segregates into two streams for independent processing of model and garment data. These streams converge within a fusion network that facilitates the embedding of garment details onto the model’s feature representation. On this foundation, we have established OutfitAnyone, comprising two key elements: the Zero-shot Try-on Network for initial try-on imagery, and the Post-hoc Refiner for detailed enhancement of clothing and skin texture in the output images.

#### 3.1. Clothing Feature Injection

Stable Diffusion (SD) [25] and its enhanced iteration SDXL [23] both employ a pretrained autoencoder for complexity reduction, which comprises an encoder and a decoder. To extend such framework for solving the virtual Try-On problem, it is crucial to maintain consistency in clothing appearance with additional clothing image condi-

tion input. Therefore, the input clothing image is fed into the encoder to extract its corresponding features in the latent space. Subsequently, we have engineered a specialized apparel feature processing network, ReferenceNet, which mirrors the architecture of the U-Net [26] found in the original SD model. Both networks were initialized with identical pretrained parameters to ensure consistency. The integration of spatial attention and cross attention layers enabled the successful incorporation of apparel-related features into the denoising pipeline, thereby significantly enhancing the quality of Try-On image generation.

#### 3.2. Classifier-Free Guidance

In the context of original SD, classifier-free guidance [15] is a technique used to control the generation process without relying on an external classifier. This method leverages a single diffusion model trained on both conditional and unconditional data. By adjusting the scale of guidance, it can steer the generation process towards producing images that align with a given text prompt.

In our virtual Try-On framework, we identified the clothing image as the pivotal control element, underscoring its significance over textual prompts. Consequently, we have tailored the unconditional classifier guidance to utilize a blank clothing image, while the conditional guidance is informed by the actual clothing image provided. we are able to harness the guidance scale effectively, thereby delivering more precise and consistent generation outcomes.

#### 3.3. Background and Lighting Retention

In order to maintain consistency in lighting and background between the generated image and the original image, previous works such as TryonDiffusion [35] have employed a person clothing segmentation model to obtain the clothing mask from the model image. This mask is then slightly expanded, and the corresponding area on the model image is erased. This partially erased image is inputted, and the generation model learns inpainting to fill the clothing area based on this image and the given clothing image. Such approach works well for swapping similar styles of clothing without requiring extensive data. However, it is not suitable for swapping clothes with significant style differences, such as changing from shorts to a long skirt or from tight-fitting to loose clothing. The reason is that the area of the original clothing mask may limit the generation of new clothing and the mask shape might cause undesired coupling with the style.

Our method involves first detecting the bounding box of the person in the model image and then erasing everything except the face and hands. This approach avoids undesired coupling between the mask shape and style, and provides a large enough area to support swapping both upper and lower garments. However, increasing the generated background

[Figure 2]

- Figure 2. Method overview: OutfitAnyone processes input consisting of a model, garment, and related prompts through a dual-path conditional diffusion model. This model bifurcates into two distinct pathways, each dedicated to handling the model and garment data independently. The two streams eventually merge within a fusion network, which effectively integrates the garment details into the model’s feature representation. To elaborate, we extract features: openpose (can be replaced by densepose or SMPL) and initmask from the model, and then concatenate these features with the model image. This composite data is then fed into our Dual-Path SD model, which guarantees not only the high-quality retention but also the restoration of the garment’s features. Importantly, the feature spaces for both models and garments are aligned, which significantly accelerates the convergence process (with a visible try-on effect achievable within just 6k iterations). Significantly, the prompt, although it may not align perfectly with the spatial pixels, plays a crucial role in preserving semantic-level information.

[Figure 3]

- Figure 3. Refiner takes the coarse output from the dual-path conditional diffusion model as its starting point and further enhances it through our subsequent refinement process.

body silhouette. However, this approach fails during complete outfit changes, necessitating an additional Pose and Shape Guider for guidance.

In terms of controlling the pose of the person, while previous studies like ControlNet [33] have demonstrated impressive results, they necessitate additional training phases and parameters. In contrast, we have embraced a more streamlined architectural approach. As detailed in Section 3.3, to ensure a consistent background and lighting, we incorporated condition images that reflect the pose and shape. All these components can be concatenated together and fed into a denoising U-Net as input. The control images could include skeleton images, dense pose images, or images rendered using the SMPL [18] model that correspond to the target image. In our experiments, we achieved pose and shape control effects similar to ControlNet, all without the need for additional parameters or training stages.

area might result in significant background differences. In such cases, we can employ a precise person segmentation model to extract the generated person and paste them back into the original background.

#### 3.4. Pose and Shape Guider

Traditional methods preserved body shape fidelity when swapping single clothing items by selectively replacing image parts, retaining the torso for generating a reasonable

#### 3.5. Detail Refiner

In our pursuit to create a virtual try-on experience for any clothing and any person, we aimed to incorporate a diverse range of clothing styles and human subjects in our dataset. However, ensuring variety in the dataset while maintaining high image quality and detail proved challenging. To address this, we selected the highest-quality images from the dataset and paired them with model-generated images from

[Figure 4]

Figure 4. Virtual Try-On with different Outfits.

the initial version of the virtual try-on, which lacked clear and high-quality details. See Fig. 3 for schematic diagram.

By doing so, we constructed a task-specific dataset comprising pairs of high- and low-quality images. Subsequently, we repurposed the virtual try-on framework, employing low-quality images as input and their high-quality counterparts as targets, to train the diffusion model in recovering fine, realistic details effectively.

### 4. Results

In this section, we demonstrate the robust performance of our method, which supports single and multi-piece virtual outfit changes for any clothing, shape, person, and background variations. Remarkably, our technique extends its capabilities to facilitate virtual outfit alterations on animated figures not originally included in our training datasets.

#### 4.1. Any Outfits

As shown in Fig. 4, OutfitAnyone not only supports singleitem clothing virtual try-on, but also allows simultaneous changes for complete outfits, including upper and lower garments. Furthermore, it effectively generates appropriate and realistic try-on results for various clothing styles, including long and short-sleeved tops, trousers and shorts, as well as dresses and similar garments. Compared to prior approaches, OutfitAnyone demonstrates superior adaptability and efficacy in managing an extensive variety of clothing styles and ensembles.

#### 4.2. Any Person

OutfitAnyone rightly caters to virtual try-on for models of diverse skin tones, ages, and genders, as illustrated in Fig. 5 and Fig. 6. Moreover, it adeptly handles selfie images from everyday users, which often vary greatly in quality

and lighting from professional model photos. Despite these differences, OutfitAnyone consistently delivers convincing outfit transformation results, as shown in the final column of Fig. 13.

Furthermore, our technology extends its prowess to animated characters not included in our training data, as showcased in Fig. 7. This capability underscores that our model transcends mere rote learning and mimicry; it has acquired genuine understanding and the intelligent capacity to apply outfit changes effectively across various contexts!

[Figure 5]

Figure 5. Virtual Try-On for kids.

#### 4.3. Any Body Shape

Our framework incorporates an additional channel for pose and shape guidance, which extracts densepose-like data (SMPL, openpose are also supported in our work) that mirrors the body’s contours. This information is instrumental in directing the final generated model to replicate the exact body shape of the original image. As demonstrated in

[Figure 6]

Figure 6. Virtual Try-On for brown-skinned people.

[Figure 7]

- Figure 7. Virtual Try-On for anime character.

[Figure 8]

- Figure 8. Virtual Try-On for different body shape.

[Figure 9]

Figure 9. The Refiner model significantly enhances the realism compared to the original model.

Fig. 8, our method excels at preserving the original model’s body shape even after a comprehensive outfit change, across a variety of body shapes.

#### 4.4. Any Background

OutfitAnyone demonstrates exceptional robustness across a variety of backgrounds and lighting scenarios. It generates reasonable clothing lighting effects in complex outdoor scenes, maintaining good performance across diverse indoor and outdoor backdrops, as evidenced in Fig. 5, Fig. 11 and the final column of Fig. 13. This adaptability demonstrates its effectiveness under various environmental conditions and real-world contexts.

[Figure 10]

Figure 10. Virtual Try-On for bizzare fashion.

[Figure 11]

Figure 11. Virtual Try-On for ordinary outdoor image.

#### 4.5. Refiner

[Figure 12]

Figure 12. Pre Refiner vs Post Refiner.

As mentioned in Sec. 3.5, we proposed utilizing a selfloop refiner model to enhance the realism of virtual try-on results. Fig. 9 shows that this refiner model significantly boosts the clarity and texture fidelity of the rendered images, while Fig. 12 underscores its ability to preserve sharp, localized details. This additional refinement step in OutfitAnyone is crucial for achieving a more vivid and convincing virtual try-on experience.

#### 4.6. Comparsion

We compared our method with popular community methods like OOTDiffusion [32] (with 5k stars on GitHub) and IDMVTON [5] (with 3k stars on GitHub), our model demonstrated noticeably better performance, particularly in challenging scenarios. As shown in the Fig. 13, OutfitAnyone excels even when dealing with ordinary users’ photos that have complex backgrounds and lighting conditions, which typically make it more difficult to achieve satisfactory virtual try-on results. This highlights the superior robustness of our approach in handling real-world situations, maintaining high-quality performance under various circumstances.

#### 4.7. Fashion Design Assistor

Indeed, our OutfitAnyone model proves to be a versatile and helpful resource for fashion designers. By employing its capabilities in generating unique and trendy clothing designs,

[Figure 13]

Figure 13. Comparsion betwwen our model and other models in the community. their results come from their huggingface space with the default parameters.

it can inspire designers to explore new styles and ideas. Moreover, when provided with only a single upper garment, our model can generate suggestions for potential lower garment designs, offering additional creative possibilities and facilitating the design process, As shown in the Fig. 10. Although there are minor details that require attention, we believe that with an increase in training data and optimization of the model, even better results can be achieved.

### 5. Conclusion

Since its initial release at the end of 2023, OutfitAnyone has undergone several iterations, building upon versions SD 1.5 and SDXL. Its original open-source version has ranked 14th on the huggingface space, placing it in the top 0.01% of the entire huggingface space (among over 200,000+ projects) https://huggingface. co/spaces/HumanAIGC/OutfitAnyone, garnering widespread recognition and attention. We are grateful for the emergence of powerful diffusion techniques (SD, SDXL, DDPM/DDIM/DPM, ControlNet and etc) and Google’s seminal exploration into virtual try-on: TryonDiffusion, which has allowed us to carve out a distinctive, unique, and mature path for virtual Try-On. In summary, OutfitAnyone has the honour of providing a benchmark application for the practical deployment of AI-generated content (AIGC).

### References

- [1] Shuai Bai, Huiling Zhou, Zhikang Li, Chang Zhou, and Hongxia Yang. Single stage virtual try-on via deformable attention flows. In European Conference on Computer Vision, pages 409–425. Springer, 2022. 2
- [2] S. Belongie, J. Malik, and J. Puzicha. Shape matching and object recognition using shape contexts. IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(4):509–522, 2002. 2
- [3] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh. Realtime multi-person 2d pose estimation using part affinity fields. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7291–7299, 2017. 2
- [4] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14131–14140, 2021. 2
- [5] Yisol Choi, Sangkyung Kwak, Kyungmin Lee, Hyungwon Choi, and Jinwoo Shin. Improving diffusion models for virtual try-on. arXiv preprint arXiv:2403.05139, 2024. 8
- [6] Haoye Dong, Xiaodan Liang, Xiaohui Shen, Bochao Wang, Hanjiang Lai, Jia Zhu, Zhiting Hu, and Jian Yin. Towards multi-pose guided virtual try-on network. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9026– 9035, 2019. 2
- [7] Ruili Feng, Cheng Ma, Chengji Shen, Xin Gao, Zhenjiang Liu, Xiaobo Li, Kairi Ou, Deli Zhao, and ZhengJun Zha. Weakly supervised high-fidelity clothing model generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3440–3449, 2022. 2
- [8] Daiheng Gao, Xu Chen, Xindi Zhang, Qi Wang, Ke Sun, Bang Zhang, Liefeng Bo, and Qixing Huang. Cloth2tex: A customized cloth texture generation pipeline for 3d virtual try-on. In 2024 International Conference on 3D Vision (3DV), pages 602–

611. IEEE, 2024. 2

- [9] Xin Gao, Zhenjiang Liu, Zunlei Feng, Chengji Shen, Kairi Ou, Haihong Tang, and Mingli Song. Shape controllable virtual try-on for underwear models. In Proceedings of the 29th ACM International Conference on Multimedia, pages 563–572, 2021. 2
- [10] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 2
- [11] Junhong Gou, Siyu Sun, Jianfu Zhang, Jianlou Si, Chen Qian, and Liqing Zhang. Taming the power of

- diffusion models for high-quality virtual try-on with appearance flow. In Proceedings of the 31st ACM International Conference on Multimedia, pages 7599– 7607, 2023. 2
- [12] Rıza Alp G¨uler, Natalia Neverova, and Iasonas Kokkinos. Densepose: Dense human pose estimation in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7297– 7306, 2018. 2
- [13] Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, and Larry S Davis. Viton: An image-based virtual try-on network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7543– 7552, 2018. 2
- [14] Sen He, Yi-Zhe Song, and Tao Xiang. Style-based global appearance flow for virtual try-on. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3470–3479, 2022. 2
- [15] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598,

2022. 3

- [16] Sangyun Lee, Gyojung Gu, Sunghyun Park, Seunghwan Choi, and Jaegul Choo. High-resolution virtual try-on with misalignment and occlusion-handled conditions. In European Conference on Computer Vision, pages 204–219. Springer, 2022. 2
- [17] Shichen Liu, Tianye Li, Weikai Chen, and Hao Li. Soft rasterizer: A differentiable renderer for imagebased 3d reasoning. The IEEE International Conference on Computer Vision (ICCV), 2019. 2
- [18] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multi-person linear model. ACM transactions on graphics (TOG), 34(6):1–16, 2015. 4
- [19] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multi-person linear model. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pages 851–866. 2023. 2
- [20] Sahib Majithia, Sandeep N Parameswaran, Sadbhavana Babar, Vikram Garg, Astitva Srivastava, and Avinash Sharma. Robust 3d garment digitization from monocular 2d images for 3d virtual try-on systems. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3428–3438,

2022. 2

- [21] Aymen Mir, Thiemo Alldieck, and Gerard Pons-Moll. Learning to transfer texture from clothing images to 3d humans. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7023–7034, 2020. 2
- [22] Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara.

- Ladi-vton: Latent diffusion textual-inversion enhanced virtual try-on. In Proceedings of the 31st ACM International Conference on Multimedia, pages 8580– 8589, 2023. 2
- [23] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2, 3
- [24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 3
- [25] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 2, 3
- [26] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015. 3
- [27] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic textto-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 2
- [28] Olga Sorkine and Marc Alexa. As-rigid-as-possible surface modeling. In Symposium on Geometry processing, pages 109–116. Citeseer, 2007. 2
- [29] Omost Team. Omost github page, 2024. 2
- [30] Bochao Wang, Huabin Zheng, Xiaodan Liang, Yimin Chen, Liang Lin, and Meng Yang. Toward characteristic-preserving image-based virtual try-on network. In Proceedings of the European conference on computer vision (ECCV), pages 589–604, 2018. 2
- [31] Yi Xu, Shanglin Yang, Wei Sun, Li Tan, Kefeng Li, and Hui Zhou. 3d virtual garment modeling from rgb images. In 2019 IEEE International Symposium on Mixed and Augmented Reality (ISMAR), pages 37–45. IEEE, 2019. 2
- [32] Yuhao Xu, Tao Gu, Weifeng Chen, and Chengcai Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. arXiv preprint arXiv:2403.01779, 2024. 3, 8

- [33] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. 2, 4
- [34] Xujie Zhang, Ente Lin, Xiu Li, Yuxuan Luo, Michael Kampffmeyer, Xin Dong, and Xiaodan Liang. Mmtryon: Multi-modal multi-reference control for high-quality fashion generation. arXiv preprint arXiv:2405.00448, 2024. 3
- [35] Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. Tryondiffusion: A tale of two unets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4606–4615, 2023. 2, 3

