# arXiv:2508.10868v2[cs.CV]3Sep2025

## TexVerse: A Universe of 3D Objects with High-Resolution Textures

### Yibo Zhang1,2 Li Zhang1,3 Rui Ma2∗ Nan Cao1,4

1Shanghai Innovation Institute 2Jilin University 3Fudan University 4Tongji University https://github.com/yiboz2001/TexVerse

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

Figure 1: We introduce TexVerse, a large-scale 3D asset dataset featuring high-resolution textures.

∗Corresponding author.

### Abstract

We introduce TexVerse, a large-scale 3D dataset featuring high-resolution textures. While recent advances in large-scale 3D datasets have enhanced high-resolution geometry generation, creating high-resolution textures end-to-end remains underexplored due to the lack of suitable datasets. TexVerse fills this gap with a curated collection of over 858K unique high-resolution 3D models sourced from Sketchfab, including more than 158K models with physically based rendering (PBR) materials. Each model encompasses all of its high-resolution variants, bringing the total to 1.6M 3D instances. TexVerse also includes specialized subsets: TexVerse-Skeleton, with 69K rigged models, and TexVerse-Animation, with 54K animated models, both preserving original skeleton and animation data uploaded by the user. We also provide detailed model annotations describing overall characteristics, structural components, and intricate features. TexVerse offers a high-quality data resource with wide-ranging potential applications in texture synthesis, PBR material development, animation, and various 3D vision and graphics tasks.

### 1 Introduction

Three-dimensional (3D) digital assets have become deeply integrated into modern life and industry, permeating fields from gaming and film to embodied artificial intelligence. However, creating highquality 3D assets is often complex, time-consuming, and costly, requiring precise mesh structures, high-resolution textures, and reliance on specialized skills and intricate toolchains. As the demand for 3D digital experiences continues to grow, the automated generation of high-quality 3D assets has emerged as a key focus for both industry and academia in recent years.

Benefiting from the release of large-scale 3D datasets [7, 8], 3D generation techniques have advanced rapidly in recent years. Current state-of-the-art methods [4, 12, 13, 14, 17] have achieved remarkable breakthroughs in the domain of high-resolution geometry generation. However, in the field of texture and physically based rendering (PBR) material generation, existing approaches [5, 16, 19, 20] still rely on generating low-resolution results followed by post-processing techniques, such as super-resolution, to enhance resolution and quality. The capability for end-to-end generation of high-resolution textures and PBR materials remains significantly underexplored.

This observation has sparked our interest. We recognize a clear gap in the community regarding the availability of accessible, large-scale, high-resolution texture datasets for 3D objects. Currently, the only open-source datasets with sample sizes exceeding 100K are Objaverse (818K) and Objaverse-XL (10.2M). Taking Objaverse as an example, its models are sourced from artist creations on Sketchfab [1], yet roughly half of the objects have textures with a maximum resolution below 1024 pixels or lack textures entirely. Even for objects labeled in the original metadata as having higher-resolution textures (e.g., 4096), Objaverse only provides versions limited to 1024 resolution, presenting a significant constraint. While Objaverse-XL is vastly larger in scale, its data, primarily sourced from the web (e.g., GitHub), exhibit considerable heterogeneity in quality and pose substantial challenges for data cleaning. The recently introduced Digital Twin Catalog (DTC) dataset [9], which includes approximately 2,000 high-precision scanned 3D objects with 4K PBR materials, offers superior quality but lacks the scale required to support the data-driven generation demand.

To fill this gap, we introduce TexVerse, a large-scale 3D dataset featuring high-resolution textures. Sketchfab hosts approximately 1.6 million freely downloadable 3D models. We curate TexVerse by filtering models with texture resolutions of at least 1024 pixels, excluding those tagged or described with terms related to “NoAI” and retaining only models licensed under distributable Creative Commons licenses. TexVerse encompasses 858,669 distinct high-resolution 3D models, of which 158,518 incorporate PBR materials, all standardized in the .glb format. Additionally, for each model, we also collect all of its high-resolution variants (e.g., the 4096 and 1024 versions of a model with a maximum resolution of 8192), yielding a total of 1,661,101 3D instances. For the rigged and animated categories of models, we further obtain the original user-uploaded file format to prevent the loss of skeletons and animations during the format conversion of Sketchfab. These are named as the TexVerse-Skeleton and TexVerse-Animation datasets, comprising 69,138 rigged models and 54,430 animated models, respectively, all with high-resolution textures. Furthermore, we provide 856,312 detailed annotations to the model generated by GPT-5 [2], which encompass the general description,

Table 1: Comparison with commonly used 3D datasets. Objaverse [8], Objaverse-XL[7] and TexVerse consist of both synthetic objects and real scans, with only a subset containing PBR materials. Within Objaverse and Objaverse-XL, only a subset provide high-resolution textures.

Dataset # Objects Type PBR Material High-Resolution Texture

ShapeNet [3] 51K synthetic ✗ ✗ 3D-Future [11] 10K synthetic ✗ ✓ ABO [6] 8K synthetic ✓ ✓ OmniObject3D [18] 6K real ✗ ✗ GSO [10] 1K real ✗ ✓ DTC [9] 2K real ✓ ✓ Objaverse [8] 818K both (✓)* (✓)* Objaverse-XL [7] 10M both (✓)* (✓)* TexVerse (Ours) 858K both (✓)* ✓

structural composition and detailed characteristics. We believe TexVerse will drive the community forward in areas such as high-resolution texture generation, PBR material synthesis, animation, and a wide range of 3D vision and graphics tasks.

### 2 Related work

We provide a comparison of our TexVerse dataset to existing commonly used 3D datasets in Table 1. Synthetic datasets constitute a significant portion of existing resources. ShapeNet [3] comprises approximately 51,000 mesh models with intricate geometric structures but limited texture resolution. Likewise, 3D-FUTURE [11] and ABO [6] focus on furniture and consumer goods, though their scales remain relatively modest. Objaverse [8] includes around 818,000 artist-created 3D models, primarily sourced from Sketchfab. However, nearly half of these models suffer from low-resolution textures, with some lacking textures entirely, and even those labeled as high-resolution are limited to 1024 resolution, restricting their use in tasks requiring high precision. Its expanded version, Objaverse-XL [7], encompasses 10 million objects from web (e.g., GitHub), but its significant quality heterogeneity presents substantial data cleaning challenges. In contrast, real-world datasets, limited by the difficulties of acquiring high-quality 3D data in the wild, are typically smaller in scale. For instance, GSO [10], OmniObject3D [18], and the recent DTC [9] offer high-quality scanned models but are restricted to a few thousand objects, insufficient for large-scale data-driven applications. As presented in Table 1, our TexVerse dataset, comprising 858,669 objects, includes both synthetic objects and real scan. All objects feature high-resolution textures, with a subset incorporating PBR materials, providing strong support for advanced 3D research.

### 3 TexVerse

TexVerse is a large-scale 3D dataset featuring high-resolution textures. The objects are sourced from Sketchfab, an online 3D marketplace where users can upload and share models for both free and commercial use. We conducted a comprehensive survey of Sketchfab, identifying approximately 1.6 million freely downloadable 3D models uploaded between 2012 and 2025. Using metadata provided by Sketchfab, we first filtered for high-resolution textured models with texture resolutions of at least 1024 pixels, excluding models tagged or described with terms related to “NoAI”. Then, we obtained models which under the distributable Creative Commons license using Sketchfab’s API. The resulting dataset, formatted in the .glb format, comprises 858,669 unique high-resolution textured 3D objects across various resolution levels. For each model, we also collected all of its high-resolution variants (e.g., the 4096 and 1024 versions of a model with a maximum resolution of 8192), yielding a total of 1,661,101 3D instances.

[Figure 17]

[Figure 18]

(a) Overview of high-level TexVerse categories (b) Word cloud of TexVerse metadata tags

[Figure 19]

[Figure 20]

858K

72.1K

Metalness

818K

TexVerse

800K

Specular

TexVerse

60K

(uid in Objaverse)

600K

Objaverse

40.1K

40K

Objaverse

TexVerse

400K

24.9K

20K

200K

158K

12.7K

81K 69K

34K 54K 41K

2.1K 3.6K 2.5K 0.7K

0

0

1024 2048 4096 8192

Total PBR Rigged

Animated

Resolution

Overview Texture distribution

PBR material distribution

(*: Only provide versions up to 1024)

(c) Statistics of TexVerse metadata

- Figure 2: TexVerse statistics. (a) Distribution of high-level TexVerse categories; (b) Word cloud of metadata tags; (c) Metadata statistics, including metadata comparison between TexVerse and Objaverse, texture resolution distribution, and PBR material distribution.

#### 3.1 Metadata

For each object in the TexVerse, we provide metadata extracted from the information supplied by its creator upon uploading to Sketchfab: uid, name, description, user-name, tags, categories, thumbnail-url, vertex-count, face-count, max-texture, pbr-type, is-rigged, animation-count, license.

#### 3.2 Statistics

We conduct detailed analyses of TexVerse, as presented in Fig. 2. Fig. 2(a) illustrates the category distribution, (b) displays the word cloud of tags, and (c) offers a static analysis of critical metadata alongside the comparison with Objaverse dataset.

Compared with Objaverse dataset The Objaverse dataset, similar to ours, consists of 818K Sketchfab models uploaded between 2012 and 2022. As shown in Fig. 2(c, middle), TexVerse demonstrates clear superiority in terms of high-resolution textures. Approximately half of the objects in Objaverse have textures with a maximum resolution below 1024 pixels, or lack textures entirely. Even for objects labeled in the original metadata as having higher-resolution textures (e.g., 4096), Objaverse only provides versions limited to 1024 resolution. We provide some examples in Fig. 3. In contrast, TexVerse significantly outperforms Objaverse across all high-resolution texture levels. In addition, Fig. 2(c, left) shows that TexVerse comprises over 858,000 objects, with nearly 60% representing novel models not present in Objaverse. It also surpasses Objaverse in the amount of critical metadata, underscoring its potential to advance a wide range of downstream tasks, such as geometry generation and skeleton generation tasks.

Rigged and animated models. TexVerse comprises 69,138 rigged models (i.e., models with a digital skeleton used to animate 3D models) and 54,430 animated models, maintaining high-resolution textures. However, we observe that during Sketchfab’s conversion of user-uploaded raw format into the standardized .glb format, essential information such as skeletons or animations is often discarded. To mitigate this issue, we also acquire the original file formats uploaded by users for rigged and animated models, which we designate as TexVerse-Skeleton and TexVerse-Animation.

4096

1024

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

8192

1024

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

(a) Model from Objaverse (b) Model from TexVerse

- Figure 3: Objaverse only provides versions up to 1024 resolution for objects labeled as having higher-resolution textures in the metadata, whereas we provide genuine high-resolution versions. The UIDs are d4d12479b5bb4bfaa72dbcf1955d5eb7 and d5e6b6a11da646f68a5fcba661dcae99.

PBR material TexVerse comprises 158,518 models with high-resolution physically based rendering (PBR) materials, following two standard workflows: Metalness and Specular. To qualify as PBR, each material must include a texture in the roughness or glossiness channel, as well as in either the metalness or specular channel, depending on the chosen workflow. The distribution of PBR materials across different texture resolutions is shown in Fig. 2(c, right). We provide an example featuring 4K resolution PBR materials, as shown in Fig. 4.

License All models in our dataset are released under distributable Creative Commons licenses. Over 80% of models are under CC BY or CC0 licenses, enabling flexible use in both academic and commercial settings.

#### 3.3 Model annotation

We provide 856,312 rich model annotations generated by GPT-5 [2] from the corresponding thumbnails, with the annotation process illustrated in Fig. 5. Following the instruction template inspired by [15], GPT-5 is instructed to generate annotations in a fixed three-sentence structure: first, provide an overall description; second, list the components of the object and their spatial relationships; and third, describe each component’s detailed attributes, such as text, texture, color, and shape. This standardized format ensures that the resulting descriptions are both comprehensive and consistent.

### 4 Conclusion

We present TexVerse, a large-scale 3D dataset featuring high-resolution textures. TexVerse includes 858,669 unique 3D models, of which 158,518 incorporate PBR materials. For rigged and animated models, we preserve the original user-uploaded file in the TexVerse-Skeleton and TexVerse-Animation datasets to maintain skeleton and animation information. Additionally, we provide 856,312 detailed model annotations generated by GPT-5, encompassing overall descriptions, structural compositions, and intricate feature details. We believe TexVerse will effectively support future research in fields such as high-resolution texture generation, PBR material synthesis, animation, and a broad spectrum of 3D vision and graphics applications.

Limitations and future work We determine model resolution based on metadata from Sketchfab, which may contain occasional annotation errors. Additionally, while we provide models with highresolution textures, further filtering and cleaning are needed to ensure consistent geometric quality and texture clarity. In future work, we will focus on addressing these limitations by developing data validation processes, improving the quality of geometric and texture data through targeted cleaning, and developing more robust annotation methods to enhance dataset reliability.

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

Rendering Albedo Metallic Roughness Normal

- Figure 4: Data example with 4K resolution PBR materials. Please zoom in to check the detail.

Given a thumbnail of a 3D asset, please describe the asset in the image, strictly following the format of 3 sentences: The first sentence should describe the overall information using the format 'A/An ****'; The second sentence should list the components of the object and their spatial relationships; The third sentence should describe all the detail of each component, such as meaningful text, texture, color, and shape. Please ensure the entire description consists of at least 3 sentences and covers all required detail. You can refer to the following example: Example: An antique rotary dial telephone installed on a wooden base. The phone features a handset and a coiled cord, with a circular dial on the front of the base and a small drawer with a handle underneath the dial. The overall appearance of the phone is gold, while the base is a reddish-brown color, and both the dial and the drawer handle are gold.

[Figure 40]

[Figure 41]

A humanoid beast warrior in ornate armor. The creature has a muscular, animalistic body with a snarling face, horns, and glowing eyes, wearing segmented armor pieces on its shoulders, forearms, and thighs, and standing on a wooden plank surface. The armor is decorated with intricate orange and black patterns, the fur is gray with detailed texturing, the claws are sharp and metallic-looking, and the glowing eyes emit a fierce yellow light.

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

A vintage hand-crank drill used for manual woodworking and metalworking. The tool consists of a metal drill bit at the front held by a textured chuck, a central gear mechanism connected to a rotating side handle, and a smooth wooden grip at the rear for stability. The drill bit is metallic silver with a spiral flute, the chuck is dark and textured for grip, the frame is painted an aged off-white with minor rust and paint chips, the side handle is rounded and made of worn wood with visible grain, and the rear wooden grip is polished but weathered with scratches and dents.

- Figure 5: Model annotation. Powered by GPT-5, we provide comprehensive model annotations that capture overall characteristics, structural components, and nuanced features.

### References

- [1] Sketchfab. https://sketchfab.com/. 2
- [2] Gpt-5. https://openai.com/gpt-5/, 2025. 2, 5
- [3] Angel X. Chang, Thomas A. Funkhouser, Leonidas J. Guibas, Pat Hanrahan, Qi-Xing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, Jianxiong Xiao, Li Yi, and Fisher Yu. Shapenet: An information-rich 3d model repository. arXiv preprint, 2015. 3
- [4] Yiwen Chen, Zhihao Li, Yikai Wang, Hu Zhang, Qin Li, Chi Zhang, and Guosheng Lin. Ultra3d: Efficient and high-fidelity 3d generation with part attention. arXiv preprint, 2025. 2
- [5] Wei Cheng, Juncheng Mu, Xianfang Zeng, Xin Chen, Anqi Pang, Chi Zhang, Zhibin Wang, Bin Fu, Gang Yu, Ziwei Liu, and Liang Pan. Mvpaint: Synchronized multi-view diffusion for painting anything 3d. In CVPR, 2025. 2
- [6] Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F. Yago Vicente, Thomas Dideriksen, Himanshu Arora, Matthieu Guillaumin, and Jitendra Malik. ABO: dataset and benchmarks for real-world 3d object understanding. In CVPR, 2022. 3
- [7] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. Objaverse-xl: A universe of 10m+ 3d objects. In NeurIPS, 2023. 2, 3
- [8] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, 2023. 2, 3
- [9] Zhao Dong, Ka Chen, Zhaoyang Lv, Hong-Xing Yu, Yunzhi Zhang, Cheng Zhang, Yufeng Zhu, Stephen Tian, Zhengqin Li, Geordie Moffatt, Sean Christofferson, James Fort, Xiaqing Pan, Mingfei Yan, Jiajun Wu, Carl Yuheng Ren, and Richard A. Newcombe. Digital twin catalog: A large-scale photorealistic 3d object digital twin dataset. In CVPR, 2025. 2, 3
- [10] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas Barlow McHugh, and Vincent Vanhoucke. Google scanned objects: A highquality dataset of 3d scanned household items. In ICRA, 2022. 3
- [11] Huan Fu, Rongfei Jia, Lin Gao, Mingming Gong, Binqiang Zhao, Stephen J. Maybank, and Dacheng Tao. 3d-future: 3d furniture shape with texture. IJCV, 2021. 3
- [12] Xianglong He, Zi-Xin Zou, Chia-Hao Chen, Yuan-Chen Guo, Ding Liang, Chun Yuan, Wanli Ouyang, Yan-Pei Cao, and Yangguang Li. Sparseflex: High-resolution and arbitrary-topology 3d shape modeling. arXiv preprint, 2025. 2
- [13] Zeqiang Lai, Yunfei Zhao, Haolin Liu, Zibo Zhao, Qingxiang Lin, Huiwen Shi, Xianghui Yang, Mingxin Yang, Shuhui Yang, Yifei Feng, Sheng Zhang, Xin Huang, Di Luo, Fan Yang, Fang Yang, Lifu Wang, Sicong Liu, Yixuan Tang, Yulin Cai, Zebin He, Tian Liu, Yuhong Liu, Jie Jiang, Linus, Jingwei Huang, and Chunchao Guo. Hunyuan3d 2.5: Towards high-fidelity 3d assets generation with ultimate details. arXiv preprint, 2025. 2
- [14] Zhihao Li, Yufei Wang, Heliang Zheng, Yihao Luo, and Bihan Wen. Sparc3d: Sparse representation and construction for high-resolution 3d shapes modeling. arXiv preprint, 2025. 2
- [15] Sitong Su, Xiao Cai, Lianli Gao, Pengpeng Zeng, Qinhong Du, Mengqi Li, Heng Tao Shen, and Jingkuan Song. Gt23d-bench: A comprehensive general text-to-3d generation benchmark. arXiv preprint, 2024. 5
- [16] Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. Unique3d: High-quality and efficient 3d mesh generation from a single image. In NeurIPS, 2024. 2

- [17] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Yikang Yang, Yajie Bao, Jiachen Qian, Siyu Zhu, Xun Cao, Philip Torr, and Yao Yao. Direct3d-s2: Gigascale 3d generation made easy with spatial sparse attention. arXiv preprint, 2025. 2
- [18] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, Dahua Lin, and Ziwei Liu. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. In CVPR, 2023. 3
- [19] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. CLAY: A controllable large-scale generative model for creating high-quality 3d assets. ACM TOG, 2024. 2
- [20] Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, Huiwen Shi, Sicong Liu, Junta Wu, Yihang Lian, Fan Yang, Ruining Tang, Zebin He, Xinzhou Wang, Jian Liu, Xuhui Zuo, Zhuo Chen, Biwen Lei, Haohan Weng, Jing Xu, Yiling Zhu, Xinhai Liu, Lixin Xu, Changrong Hu, Tianyu Huang, Lifu Wang, Jihong Zhang, Meng Chen, Liang Dong, Yiwen Jia, Yulin Cai, Jiaao Yu, Yixuan Tang, Hao Zhang, Zheng Ye, Peng He, Runzhou Wu, Chao Zhang, Yonghao Tan, Jie Xiao, Yangyu Tao, Jianchen Zhu, Jinbao Xue, Kai Liu, Chongqing Zhao, Xinming Wu, Zhichao Hu, Lei Qin, Jianbing Peng, Zhan Li, Minghui Chen, Xipeng Zhang, Lin Niu, Paige Wang, Yingkai Wang, Haozhao Kuang, Zhongyi Fan, Xu Zheng, Weihao Zhuang, YingPing He, Tian Liu, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, Jingwei Huang, and Chunchao Guo. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint, 2025. 2

