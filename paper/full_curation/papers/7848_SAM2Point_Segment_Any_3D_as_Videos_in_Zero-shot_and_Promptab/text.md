[Figure 1]

## SAM2POINT: SEGMENT ANY 3D AS VIDEOS IN ZERO-SHOT AND PROMPTABLE MANNERS

Ziyu Guo1∗, Renrui Zhang2,3∗†, Xiangyang Zhu4∗, Chengzhuo Tong4 Peng Gao4, Chunyuan Li3, Pheng-Ann Heng1

1CUHK MiuLar Lab 2CUHK MMLab 3ByteDance 4Shanghai AI Laboratory {ziyuguo, renruizhang}@link.cuhk.edu.hk

# arXiv:2408.16768v1[cs.CV]29Aug2024

∗ Equal contribution † Project lead

ABSTRACT

We introduce SAM2POINT, a preliminary exploration adapting Segment Anything Model 2 (SAM 2) for zero-shot and promptable 3D segmentation. SAM2POINT interprets any 3D data as a series of multi-directional videos, and leverages SAM 2 for 3D-space segmentation, without further training or 2D-3D projection. Our framework supports various prompt types, including 3D points, boxes, and masks, and can generalize across diverse scenarios, such as 3D objects, indoor scenes, outdoor scenes, and raw LiDAR. Demonstrations on multiple 3D datasets, e.g., Objaverse, S3DIS, ScanNet, Semantic3D, and KITTI, highlight the robust generalization capabilities of SAM2POINT. To our best knowledge, we present the most faithful implementation of SAM in 3D, which may serve as a starting point for future research in promptable 3D segmentation.

Live Demo: https://huggingface.co/spaces/ZiyuG/SAM2Point Code: https://github.com/ZiyuGuo99/SAM2Point

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

3D Object

Indoor Scene Outdoor Scene Raw LiDAR

[Figure 6]

[Figure 7]

SAM2POINT

3D Point Prompt

3D Box Prompt

3D Mask Prompt

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Figure 1: The Segmentation Paradigm of SAM2POINT. We introduce a zero-shot and promptable framework for robust 3D segmentation via SAM 2 (Ravi et al., 2024). It supports various userprovided 3D prompt, and can generalize to diverse 3D scenarios. The 3D prompt and segmentation results are highlighted in red and green, respectively.

Table 1: Comparison of SAM2POINT and Previous SAM-based Methods (Yang et al., 2023b; Cen et al., 2023; Xu et al., 2023a; Zhou et al., 2024). To our best knowledge, SAM2POINT presents the most faithful implementation of SAM (Kirillov et al., 2023) in 3D, demonstrating superior implementation efficiency, promptable flexibility, and generalization capabilities for 3D segmentation.

3D Prompt 3D Scenario Point Box Mask Object Indoor Outdoor Raw LiDAR

Method Zero-shot Project-free

SAM3D ✓ - - - - - ✓ - SA3D ✓ - - - - - ✓ ✓ SAMPro3D ✓ - - - - - ✓ - Point-SAM - ✓ ✓ - - ✓ ✓ ✓ -

SAM2POINT ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

- 1 INTRODUCTION

Segment Anything Model (SAM) (Kirillov et al., 2023) has established a superior and fundamental framework for interactive image segmentation. Building on its strong transferability, follow-up research further extends SAM to diverse visual domains, e.g., personalized objects (Zhang et al., 2023b; Liu et al., 2023d), medical imaging (Ma et al., 2024; Mazurowski et al., 2023), and temporal sequences (Yang et al., 2023a; Cheng et al., 2023). More recently, Segment Anything Model 2 (SAM 2) (Ravi et al., 2024) is proposed for impressive segmentation capabilities in video scenarios, capturing complex real-world dynamics.

Despite this, effectively adapting SAM for 3D segmentation still remains an unresolved challenge.

We identify three primary issues within previous efforts, as compared in Table 1, which prevent them from fully leveraging SAM’s advantages:

- • Inefficient 2D-3D Projection. Considering the domain gap between 2D and 3D, most existing works represent 3D data as its 2D counterpart as input for SAM, and back-project the segmentation results into 3D space, e.g., using additional RGB images (Yang et al., 2023b; Yin et al., 2024; Xu et al., 2023a), multi-view renderings (Zhou et al., 2023b), or Neural Radiance Field (Cen et al., 2023). Such modality transition introduces significant processing complexity, hindering efficient implementation.
- • Degradation of 3D Spatial Information. The reliance on 2D projections results in the loss of fine-grained 3D geometries and semantics, as multi-view data often fails to preserve spatial relations. Furthermore, the internal structures of 3D objects cannot be adequately captured by 2D images, significantly limiting segmentation accuracy.
- • Loss of Prompting Flexibility. A compelling strength of SAM lies in its interactive capabilities through various prompt alternatives. Unfortunately, these functionalities are mostly disregarded in current methods, as users struggle to specify precise 3D positions using 2D representations. Consequently, SAM is typically used for dense segmentation across entire multi-view images, thereby sacrificing interactivity.
- • Limited Domain Transferability. Existing 2D-3D projection techniques are often tailored to specific 3D scenarios, heavily dependent on in-domain patterns. This makes them challenging to apply to new contexts, e.g., from objects to scenes or from indoor to outdoor environments. Another research direction (Zhou et al., 2024) aims to train a promptable network from scratch in 3D. While bypassing the need for 2D projections, it demands substantial training and data resources and may still be constrained by training data distributions.

In this project, we introduce SAM2POINT, adapting SAM 2 for efficient, projection-free, promptable, and zero-shot 3D segmentation. As an initial step in this direction, our target is not to push the performance limit, but rather to demonstrate the potential of SAM in achieving robust and effective 3D segmentation in diverse contexts. Specifically, SAM2POINT exhibits three features as outlined:

- • Segmenting Any 3D as Videos. To preserve 3D geometries during segmentation, while ensuring compatibility with SAM 2, we adopt voxelization to mimic a video. Voxelized 3D

- data, with a shape of w ×h×l ×3, closely resembles the format of videos of w ×h×t×3. This representation allows SAM 2 for zero-shot 3D segmentation while retaining sufficient spatial information, without the need of additional training or 2D-3D projection.
- • Supporting Multiple 3D Prompts. Building on SAM 2, SAM2POINT supports three types of prompts: 3D points, bounding boxes, and masks. Starting with a user-provided 3D prompt, e.g., a point (x, y, z), we divide the 3D space into three orthogonal directions, generating six corresponding videos. Then, the multi-directional segmentation results are integrated to form the final prediction in 3D space, allowing for interactive promptable segmentation.
- • Generalizable to Various Scenarios. With our concise framework, SAM2POINT demonstrates strong generalization capabilities in diverse 3D scenarios with varying point cloud distributions. As showcased in Figure 1, our approach can effectively segment single objects, indoor scenes, outdoor scenes, and raw LiDAR, highlighting its superior transferability across different domains.

- 2 SAM2POINT

The detailed methodology of SAM2POINT is presented in Figure 2. In Section 2.1, we introduce how SAM2POINT efficiently formats 3D data for compatibility with SAM 2 (Ravi et al., 2024), avoiding complex projection process. Then, in Section 2.2, we detail the three types of 3D prompt supported and their associated segmentation techniques. Finally, in Section 2.3, we illustrate four challenging

- 3D scenarios effectively addressed by SAM2POINT.

- 2.1 3D DATA AS VIDEOS

Given any object-level or scene-level point cloud, we denote it by P ∈ Rn×6, with each point as p = (x,y,z,r,g,b). Our aim is to convert P into a data format that, for one hand, SAM 2 can directly process in a zero-shot manner, and, for the other, the fine-grained spatial geometries can be well preserved. To this end, we adopt the 3D voxelization technique. Compared to RGB image mapping (Yang et al., 2023b; Yin et al., 2024; Xu et al., 2023a), multi-view rendering (Zhou et al., 2023b), and NeRF (Cen et al., 2023) in previous efforts, voxelization is efficiently performed in 3D space, thereby free from information degradation and cumbersome post-processing.

In this way, we obtain a voxelized representation of the 3D input, denoted by V ∈ Rw×h×l×3 with each voxel as v = (r,g,b). For simplicity, the (r,g,b) value is set according to the point nearest to the voxel center. This format closely resembles videos with a shape of w×h×t×3. The main difference is that, video data contains unidirectional temporal dependency across t frames, while 3D voxels are isotropic along three spatial dimensions. Considering this, we convert the voxel representation as a series of multi-directional videos, inspiring SAM 2 to segment 3D the same way as videos.

- 2.2 PROMPTABLE SEGMENTATION

For flexible interactivity, our SAM2POINT supports three types of prompt in 3D space, which can be utilized either separately or jointly. We specify the prompting and segmentation details below:

- • 3D Point Prompt, denoted as pp = (xp,yp,zp). We first regard pp as an anchor point in 3D space to define three orthogonal 2D sections. Starting from these sections, we divide the 3D voxels into six subparts along six spatial directions, i.e., front, back, left, right, up, and down. Then, we regard them as six different videos, where the section serves as the first

frame and pp is projected as the 2D point prompt. After applying SAM 2 for concurrent segmentation, we integrate the results of six videos as the final 3D mask prediction.

- • 3D Box Prompt, denoted as bp = (xp,yp,zp,wp,hp,lp), including 3D center coordinates and dimensions. We adopt the geometric center of bp as the anchor point, and represent the 3D voxels by six different videos as aforementioned. For video of a certain direction, we project bp into the corresponding 2D section to serve as the box point for segmentation. We also support 3D box with rotation angles, e.g., (αp, βp, γp), for which the bounding rectangle of projected bp is adopted as the 2D prompt.

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

###### Diverse 3D Scenarios Various 3D Prompt 3D Data as Videos Zero-shot Segmentation

[Figure 16]

[Figure 17]

[Figure 18]

3D Point Prompt

3D Box Prompt

3D Mask Prompt

3D Object

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

##### Segment Anything Model 2

Indoor Scene

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Voxelized Representation

Outdoor Scene

[Figure 28]

Divide 3D Space by Anchor Point

Multi-directional Videos

[Figure 29]

[Figure 30]

Raw LiDAR

- Figure 2: The Detailed Methodology of SAM2POINT. We convert any input 3D data into voxelized representations, and utilize user-provided 3D prompt to divide the 3D space along six directions, effectively simulating six different videos for SAM 2 to perform zero-shot segmentation.

• 3D Mask Prompt, denoted as Mp ∈ Rn×1, where 1 or 0 indicates the masked and unmasked areas. We employ the center of gravity of the mask prompt as the anchor point, and divide 3D space into six videos likewise. The intersection between the 3D mask prompt and each section is utilized as the 2D mask prompt for segmentation. This type of prompting can also serve as a post-refinement step to enhance the accuracy of previously predicted 3D masks.

- 2.3 ANY 3D SCENARIOS

With our concise framework design, SAM2POINT exhibits superior zero-shot generalization performance across diverse domains, ranging from objects to scenes and indoor to outdoor environments. We elaborate on four distinct 3D scenarios below:

- • 3D Object, e.g., Objaverse (Deitke et al., 2023), with a wide array of categories, possesses unique characteristics across different instances, including colors, shapes, and geometries. Adjacent components of an object might overlap, occlude, or integrate with each other, which requires models to accurately discern subtle differences for part segmentation.
- • Indoor Scene, e.g., S3DIS (Armeni et al., 2016) and ScanNet (Dai et al., 2017), are typically characterized by multiple objects arranged within confined spaces, like rooms. The complex spatial layouts, similarity in appearance, and varied orientations between objects pose challenges for models to segment them from backgrounds.
- • Outdoor Scene, e.g., Semantic3D (Hackel et al., 2017), differs from indoor scenes, primarily due to the stark size contrasts of objects (buildings, vehicles, and humans) and the larger scale of point clouds (from a room to an entire street). These variations complicates the segmentation of objects whether at a global scale or a fine-grained level.
- • Raw LiDAR, e.g., KITTI (Geiger et al., 2012) in autonomous driving, is distinct from typical point clouds for its sparse distribution and absence of RGB information. The sparsity demands models to infer missing semantics for understanding the scene, and the lack of colors enforces models to only rely on geometric cues to differentiate between objects. In SAM2POINT, we directly set the RGB values of 3D voxels by the LiDAR intensity.

- 3 DISCUSSION AND INSIGHT

Building on the effectiveness of SAM2POINT, we delve into two compelling yet challenging issues within the realm of 3D, and share our insights on future multi-modality learning.

- 3.1 HOW TO ADAPT 2D FOUNDATION MODELS TO 3D?

The availability of large-scale, high-quality data has significantly empowered the development of large models in language (Brown et al., 2020; Touvron et al., 2023; Zhang et al., 2023a), 2D vision (Liu et al., 2023b; Team et al., 2023; Chen et al., 2024), and vision-language (Gao et al., 2024; Liu et al., 2023a; Li et al., 2024; Zhang et al., 2024) domains. In contrast, the 3D field has long struggled with a scarcity of data, hindering the training of large 3D models. As a result, researchers have turned to the alternative of transferring pre-trained 2D models into 3D.

The primary challenge lies in bridging the modal gap between 2D and 3D. Pioneering approaches, such as PointCLIP (Zhang et al., 2022), its V2 (Zhu et al., 2022), and subsequent methods (Ji et al., 2023; Huang et al., 2023), project 3D data into multi-view images, which encounter implementation inefficiency and information loss. Another line of work, including ULIP series (Xue et al., 2022; 2023), I2P-MAE (Zhang et al., 2023c), and others (Liu et al., 2023c; Qi et al., 2023; Guo et al., 2023a), employs knowledge distillation using 2D-3D paired data. While this method generally performs better due to extensive training, it suffers from limited 3D transferability in out-of-domain scenarios. Recent efforts have also explored more complex and costly solutions, such as joint multi-modal spaces (e.g., Point-Bind & Point-LLM (Guo et al., 2023b)), larger-scale pre-training (Uni3D (Zhou et al., 2023a)), and virtual projection techniques (Any2Point (Tang et al., 2024)).

From SAM2POINT, we observe that representing 3D data as videos through voxelization may offer an optimal solution, providing a balanced trade-off between performance and efficiency. This approach not only preserves the spatial geometries inherent in 3D space with a simple transformation, but also presents a grid-based data format that 2D models can directly process. Despite this, further experiments are necessary to validate and reinforce this observation.

- 3.2 WHAT IS THE POTENTIAL OF SAM2POINT IN 3D DOMAINS?

To the best of our knowledge, SAM2POINT presents the most accurate and comprehensive implementation of SAM in 3D, successfully inheriting its implementation efficiency, promptable flexibility, and generalization capabilities. While previous SAM-based approaches (Yang et al., 2023b; Xu et al., 2023a; Yin et al., 2024) have achieved 3D segmentation, they often fall short in scalability and transferability to benefit other 3D tasks. In contrast, inspired by SAM in 2D domains, SAM2POINT demonstrates significant potential to advance various 3D applications.

For fundamental 3D understanding, SAM2POINT can serve as a unified initialized backbone for further fine-tuning, offering strong 3D representations simultaneously across 3D objects, indoor scenes, outdoor scenes, and raw LiDAR. In the context of training large 3D models, SAM2POINT can be employed as an automatic data annotation tool, which mitigates the data scarcity issue by generating large-scale segmentation labels across diverse scenarios. For 3D and language-vision learning, SAM2POINT inherently provides a joint embedding space across 2D, 3D, and video domains, due to its zero-shot capabilities, which could further enhance the effectiveness of models like Point-Bind (Guo et al., 2023b). Additionally, in the development of 3D large language models (LLMs) (Hong et al., 2023; Xu et al., 2023b; Wang et al., 2023; Guo et al., 2023b), SAM2POINT can function as a powerful 3D encoder, supplying LLMs with 3D tokens, and leveraging its promptable features to equip LLMs with promptable instruction-following capabilities.

- 4 DEMOS

In Figures 3-7, we showcase demonstrations of SAM2POINT in segmenting 3D data with various 3D prompt on different datasets (Deitke et al., 2023; Armeni et al., 2016; Dai et al., 2017; Hackel et al., 2017; Geiger et al., 2012). For further implementation details, please refer to our open-sourced code.

3D Point Prompt

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

3D Box Prompt

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

3D Mask Prompt

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

- Figure 3: 3D Object Segmentation with SAM2POINT on Objaverse (Deitke et al., 2023). The 3D prompt and segmentation results are highlighted in red and green, respectively.

3D Point Prompt

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

3D Box Prompt

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

3D Mask Prompt

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

### Figure 4: 3D Indoor Scene Segmentation with SAM2POINT on S3DIS (Armeni et al., 2016). The

- 3D prompt and segmentation results are highlighted in red and green, respectively.

3D Point Prompt

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

3D Box Prompt

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

3D Mask Prompt

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

- Figure 5: 3D Indoor Scene Segmentation with SAM2POINT on ScanNet (Dai et al., 2017). The 3D prompt and segmentation results are highlighted in red and green, respectively.

3D Point Prompt

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

3D Box Prompt

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

3D Mask Prompt

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

- Figure 6: 3D Outdoor Scene Segmentation with SAM2POINT on Semantic3D (Hackel et al., 2017). The 3D prompt and segmentation results are highlighted in red and green, respectively.

3D Point Prompt

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

3D Box Prompt

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

3D Mask Prompt

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

- Figure 7: 3D Raw LiDAR Segmentation with SAM2POINT on KITTI (Geiger et al., 2012). The 3D prompt and segmentation results are highlighted in red and green, respectively.

- 5 CONCLUSION

In this project, we propose SAM2POINT, which leverages Segment Anything 2 (SAM 2) to 3D segmentation with a zero-shot and promptable framework. By representing 3D data as multidirectional videos, SAM2POINT supports various types of user-provided prompt (3D point, box, and mask), and exhibits robust generalization across diverse 3D scenarios (3D object, indoor scene, outdoor environment, and raw sparse LiDAR). As a preliminary investigation, SAM2POINT provides unique insights into adapting SAM 2 for effective and efficient 3D understanding. We hope our method may serve as a foundational baseline for promptable 3D segmentation, encouraging further research to fully harness SAM 2’s potential in 3D domains.

REFERENCES

Iro Armeni, Ozan Sener, Amir R Zamir, Helen Jiang, Ioannis Brilakis, Martin Fischer, and Silvio Savarese. 3d semantic parsing of large-scale indoor spaces. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1534–1543, 2016.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33:1877–1901, 2020.

Jiazhong Cen, Zanwei Zhou, Jiemin Fang, Wei Shen, Lingxi Xie, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian, et al. Segment anything in 3d with nerfs. Advances in Neural Information Processing Systems, 36:25971–25990, 2023.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24185–24198, 2024.

Yangming Cheng, Liulei Li, Yuanyou Xu, Xiaodi Li, Zongxin Yang, Wenguan Wang, and Yi Yang. Segment and track anything. arXiv preprint arXiv:2305.06558, 2023.

Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5828–5839, 2017.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13142–13153, 2023.

Peng Gao, Renrui Zhang, Chris Liu, Longtian Qiu, Siyuan Huang, Weifeng Lin, Shitian Zhao, Shijie Geng, Ziyi Lin, Peng Jin, et al. Sphinx-x: Scaling data and parameters for a family of multi-modal large language models. ICML 2024, 2024.

Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In Conference on Computer Vision and Pattern Recognition (CVPR), 2012.

Ziyu Guo, Renrui Zhang, Longtian Qiu, Xianzhi Li, and Pheng-Ann Heng. Joint-mae: 2d-3d joint masked autoencoders for 3d point cloud pre-training. arXiv preprint arXiv:2302.14007, 2023a.

Ziyu Guo, Renrui Zhang, Xiangyang Zhu, Yiwen Tang, Xianzheng Ma, Jiaming Han, Kexin Chen, Peng Gao, Xianzhi Li, Hongsheng Li, et al. Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following. arXiv preprint arXiv:2309.00615, 2023b.

Timo Hackel, Nikolay Savinov, Lubor Ladicky, Jan D Wegner, Konrad Schindler, and Marc Pollefeys. Semantic3d. net: A new large-scale point cloud classification benchmark. arXiv preprint arXiv:1704.03847, 2017.

Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. arXiv, 2023.

Tianyu Huang, Bowen Dong, Yunhan Yang, Xiaoshui Huang, Rynson WH Lau, Wanli Ouyang, and Wangmeng Zuo. Clip2point: Transfer clip to point cloud classification with image-depth pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22157–22167, 2023.

Jiayi Ji, Haowei Wang, Changli Wu, Yiwei Ma, Xiaoshuai Sun, and Rongrong Ji. Jm3d & jm3d-llm: Elevating 3d representation with joint multi-modal cues. arXiv preprint arXiv:2310.09503, 2023.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4015–4026, 2023.

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023b.

Minghua Liu, Ruoxi Shi, Kaiming Kuang, Yinhao Zhu, Xuanlin Li, Shizhong Han, Hong Cai, Fatih Porikli, and Hao Su. OpenShape: Scaling Up 3D Shape Representation Towards Open-World Understanding. arXiv preprint arXiv:2305.10764, 2023c.

Yang Liu, Muzhi Zhu, Hengtao Li, Hao Chen, Xinlong Wang, and Chunhua Shen. Matcher: Segment anything with one shot using all-purpose feature matching. arXiv preprint arXiv:2305.13310, 2023d.

Jun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, and Bo Wang. Segment anything in medical images. Nature Communications, 15(1):654, 2024.

Maciej A Mazurowski, Haoyu Dong, Hanxue Gu, Jichen Yang, Nicholas Konz, and Yixin Zhang. Segment anything model for medical image analysis: an experimental study. Medical Image Analysis, 89:102918, 2023.

Zekun Qi, Runpei Dong, Guofan Fan, Zheng Ge, Xiangyu Zhang, Kaisheng Ma, and Li Yi. Contrast with Reconstruct: Contrastive 3D Representation Learning Guided by Generative Pretraining. In International Conference on Machine Learning, 2023.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.

Yiwen Tang, Renrui Zhang, Jiaming Liu, Dong Wang, Zhigang Wang, Shanghang Zhang, Bin Zhao, and Xuelong Li. Any2point: Empowering any-modality large models for efficient 3d understanding. ECCV 2024, 2024.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Zehan Wang, Haifeng Huang, Yang Zhao, Ziang Zhang, and Zhou Zhao. Chat-3d: Data-efficiently tuning large language model for universal dialogue of 3d scenes. arXiv preprint arXiv:2308.08769, 2023.

Mutian Xu, Xingyilang Yin, Lingteng Qiu, Yang Liu, Xin Tong, and Xiaoguang Han. Sampro3d: Locating sam prompts in 3d for zero-shot scene segmentation. arXiv preprint arXiv:2311.17707,

- 2023a.

Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Pointllm: Empowering large language models to understand point clouds. arXiv preprint arXiv:2308.16911,

- 2023b.

Le Xue, Mingfei Gao, Chen Xing, Roberto Martín-Martín, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. ULIP: Learning Unified Representation of Language, Image and Point Cloud for 3D Understanding. arXiv preprint arXiv:2212.05171, 2022.

Le Xue, Ning Yu, Shu Zhang, Junnan Li, Roberto Martín-Martín, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. ULIP-2: Towards Scalable Multimodal Pre-training for 3D Understanding. arXiv preprint arXiv:2305.08275, 2023.

Jinyu Yang, Mingqi Gao, Zhe Li, Shang Gao, Fangjing Wang, and Feng Zheng. Track anything: Segment anything meets videos. arXiv preprint arXiv:2304.11968, 2023a.

Yunhan Yang, Xiaoyang Wu, Tong He, Hengshuang Zhao, and Xihui Liu. Sam3d: Segment anything in 3d scenes. arXiv preprint arXiv:2306.03908, 2023b.

Yingda Yin, Yuzheng Liu, Yang Xiao, Daniel Cohen-Or, Jingwei Huang, and Baoquan Chen. Sai3d: Segment any instance in 3d scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3292–3302, 2024.

Renrui Zhang, Ziyu Guo, Wei Zhang, Kunchang Li, Xupeng Miao, Bin Cui, Yu Qiao, Peng Gao, and Hongsheng Li. PointCLIP: Point Cloud Understanding by CLIP. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 8552–8562, 2022.

Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023a.

Renrui Zhang, Zhengkai Jiang, Ziyu Guo, Shilin Yan, Junting Pan, Xianzheng Ma, Hao Dong, Peng Gao, and Hongsheng Li. Personalize segment anything model with one shot. arXiv preprint arXiv:2305.03048, 2023b.

Renrui Zhang, Liuhui Wang, Yu Qiao, Peng Gao, and Hongsheng Li. Learning 3d representations from 2d pre-trained models via image-to-point masked autoencoders. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21769–21780, 2023c.

Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, et al. Mavis: Mathematical visual instruction tuning. arXiv preprint arXiv:2407.08739, 2024.

Junsheng Zhou, Jinsheng Wang, Baorui Ma, Yu-Shen Liu, Tiejun Huang, and Xinlong Wang. Uni3d: Exploring unified 3d representation at scale. arXiv preprint arXiv:2310.06773, 2023a.

Yuchen Zhou, Jiayuan Gu, Xuanlin Li, Minghua Liu, Yunhao Fang, and Hao Su. Partslip++: Enhancing low-shot 3d part segmentation via multi-view instance segmentation and maximum likelihood estimation. arXiv preprint arXiv:2312.03015, 2023b.

Yuchen Zhou, Jiayuan Gu, Tung Yen Chiang, Fanbo Xiang, and Hao Su. Point-sam: Promptable 3d segmentation model for point clouds. arXiv preprint arXiv:2406.17741, 2024.

Xiangyang Zhu, Renrui Zhang, Bowei He, Ziyu Guo, Ziyao Zeng, Zipeng Qin, Shanghang Zhang, and Peng Gao. PointCLIP V2: Prompting CLIP and GPT for Powerful 3D Open-world Learning. arXiv preprint arXiv:2211.11682, 2022.

