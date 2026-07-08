# arXiv:2408.02629v1[cs.CV]5Aug2024

## VIDGEN-1M: A LARGE-SCALE DATASET FOR TEXTTO-VIDEO GENERATION

Zhiyu Tan1 Xiaomeng Yang2 Luozheng Qin2 Hao Li∗1 1 Fudan University 2 Shanghai Academy of AI for Science https://sais-fuxi.github.io/projects/vidgen-1m

[Figure 1]

The video shows a highway winding through a lush green landscape. The road is surrounded by dense trees and vegetation on both sides. The sky is overcast, and the mountains in the distance are partially obscured by clouds. The highway appears to be in good condition, with clear lane markings. There are several vehicles traveling on the road, including cars and trucks. The colors in the video are predominantly green from the trees and grey from the road and sky.

[Figure 2]

The video shows a cartoon monkey standing on a wooden dock in a pond. The monkey is holding a red bucket and appears to be feeding the swans. There are several swans in the water, and they are all white with orange beaks. The background consists of green trees and a small building with a red roof. The water is blue, and the sky is clear.

[Figure 3]

The video shows a person riding a scooter at a skate park. The person is wearing a helmet and a black t-shirt. They ride up a ramp and perform a trick in the air before landing back on the ramp. The person then rides away from the ramp. The skate park has various ramps and obstacles, and there are trees in the background.

Figure 1: A snapshot of the video-text pairs in VidGen-1M.

ABSTRACT

The quality of video-text pairs fundamentally determines the upper bound of textto-video models. Currently, the datasets used for training these models suffer from significant shortcomings, including low temporal consistency, poor-quality captions, substandard video quality, and imbalanced data distribution. The prevailing video curation process, which depends on image models for tagging and manual rule-based curation, leads to a high computational load and leaves behind unclean data. As a result, there is a lack of appropriate training datasets for textto-video models. To address this problem, we present VidGen-1M, a superior training dataset for text-to-video models. Produced through a coarse-to-fine curation strategy, this dataset guarantees high-quality videos and detailed captions with excellent temporal consistency. When used to train the video generation model, this dataset has led to experimental results that surpass those obtained with other models.

∗Corresponding Author.

- 1 INTRODUCTION

Recently, there have been significant advancements in text-to-video models, such as Latte (Ma et al., 2024), SORA, OpenSora (Zheng et al., 2024), and Window Attention Latent Transformer (W.A.L.T) (Gupta et al., 2023). However, training on current video-text datasets faces challenges, including unstable training and poor performance. These issues stem from several problems inherent in existing video-text datasets: 1)Low Quality Captions: The captions exhibit poor consistency with the videos and lack detailed captions. As demonstrated in DALLE-3 (Betker et al., 2023), training text-to-image models on descriptive synthetic captions (DSC) significantly improves the performance on text-to-image alignment, which is also called ”prompt following”. While existing text-to-video datasets, such as HD-VILA-100M (Xue et al., 2022) and Pandas-70M (Chen et al., 2024), particularly lack precise and detailed captions, the average length of captions in these datasets is less than 15 words. These captions often fail to capture object motion, actions, and camera movements, hindering the model’s ability to effectively learn the semantic and temporal information in the videos. 2)Low Quality Videos: The existing dataset suffers from poor video quality and aesthetics, resulting in models trained on this dataset being unable to generate high-quality videos. 3)Temporal Inconsistency: The scene splitting algorithm fails to accurately detect scene transitions within videos, leading to instability in training models based on such data. 4)Data Imbalance: Current datasets, primarily composed of videos sourced from the internet, are often dominated by indoor human scenes, leading to significant data imbalance issues.

There are two main challenges in curating a dataset suitable for text-to-video generation tasks. First, the existing curation process relies on either image models or optical flow models. Image models, such as the CLIP (Radford et al., 2021), lack the capability to capture temporal relationships in videos. Conversely, using optical flow scores to curate videos with fast camera movements and static scenes is inaccurate. Second, compared to the data curation process for image-text pairs, the process for video-text pairs is significantly more complex and computationally intensive, posing major challenges for the academic community.

To tackle the aforementioned challenges, we introduce a multi-stage data curation process composed of three stages: coarse curation, captioning, and fine curation. During the coarse curation stage, we utilize existing models to perform scene splitting and tagging on the videos. Based on these tags, we filter and sample the videos to create a curated dataset. This process ensures a balanced distribution across different categories and styles while reducing the number of videos that need to be processed in the subsequent computationally intensive curating phase. During the captioning stage, we employ a video captioning model to generate descriptive synthetic captions (DSC) for the videos. In the fine curation stage, we employ a large language model (LLM) to refine video captions. This process addresses errors from the coarse curation stage, such as incorrectly filtered videos (e.g., those with scene transitions) and errors in caption generation (e.g., captions missing the EOS token).

In this work, we introduce a large-scale dataset comprising 1 million video clips with descriptive synthetic captions (DSC). The dataset features high-quality, open-domain videos accompanied by rich captions averaging 89.2 words each. These captions not only ensure stronger text-video alignment but also accurately capture the dynamic elements of the videos. Furthermore, the improved temporal consistency of the videos mitigates instability during model training. Additionally, the distribution of videos across different categories and styles is balanced. Training text-to-video models on our proposed dataset achieves superior performance compared to existing methods.

In summary, our main contributions can be summarized as follows:

- • We introduce a high-quality video dataset specifically designed for training text-to-video models.
- • We propose a multi-stage curation method that achieves precise, high-quality curated data with limited computational resources.
- • We release our text-to-video model, which generates high-quality videos that surpass the performance of existing state-of-the-art methods.

Dataset Year Text Domain #Videos AVL ATL Res

MSVD (Chen & Dolan, 2011) 2011 Human Open 2K 9.7s 8.7 words LSMDC (Rohrbach et al., 2015) 2015 Human Movie 118K 4.8s 7.0 words 1080p UCF101 (Soomro et al., 2012) 2015 Human Action 13K 7.2s 4.3 words 240p MSR-VTT (Xu et al., 2016) 2016 Human Open 10K 15.0s 9.3 words 240p DiDeMo (Anne Hendricks et al., 2017) 2017 Human Flickr 27K 6.9s 8.0 words ActivityNet (Caba Heilbron et al., 2015) 2017 Human Action 100K 36.0s 13.5 words YouCook2 (Zhou et al., 2018) 2018 Human Cooking 14K 19.6s 8.8 words VATEX (Wang et al., 2019) 2019 Human Open 41K ∼10s 15.2 words HowTo100M (Miech et al., 2019) 2019 ASR Open 136M 3.6s 4.0 words 240p YT-Temporal-180M (Zellers et al., 2021) 2021 ASR Open 180M - - HD-VILA-100M (Xue et al., 2022) 2022 ASR Open 103M 13.4s 32.5 words 720p Panda-70M (Chen et al., 2024) 2024 Auto Open 70.8M 8.5s 13.2 words 720p VIDGEN-1M (Ours) 2024 Auto Open 1M 10.6s 89.3 words 720p

- Table 1: Comparison of our dataset and other video-text datasets. “AVL” and “ATL” are abbreviations for “Average Video Length” and “Average Text Length”, respectively.

- 2 RELATED WORK

- 2.1 VIDEO-TEXT DATASET

To facilitate the development of video understanding and generation, researchers build a large volume of video-text datasets that vary in video length, resolution, domain, and scale. For instance, UCF101 (Soomro et al., 2012) is originally an action recognition dataset consisting of 13,320 videos, which can be classified into 101 categories. Formulate a unified text conditions for each category, UCF101 is widely used for benchmarking text-to-video generation. MSVD (Chen & Dolan, 2011) and MSRVTT (Xu et al., 2016) are two open-domain video-text datasets popular in video retrieval. These datasets collect videos first and then annotate these videos with human annotators. However, due to the heavy cost of human annotation, they are usually limited by scale, usually at thousands scale. To alleviate this and expand video-text datasets to million scale, How2100M, HD-VILA100M (Xue et al., 2022) and YT-Temporal-180M (Zellers et al., 2021) propose to automatically annotate videos with subtitles generated by ASR models. Meanwhile, Webvid scrapes 10.7 million videos along with text annotation. While Panda-70M Chen et al. (2024) collects 70 million high-resolution and semantically coherent video samples.

These large scale video-text datasets surly lay the cornerstone for the adavancement of text-to-video generation. However, they are limited by low quality captions, low video quality, temporal inconsistency and data imbalance. To alleviate these challenges, we meticulously curate Panda-70M in a coarse-to-fine way. Owing to our comprehensive and effective data curation, VidGen-1M features high video quality, high video-text consistency and balanced and diverse video content, which significantly differ it with previous works.

- 2.2 TEXT-TO-VIDEO GENERATION MODEL

To investigate the best practice for designing video generation models, researchers has made a series of progress. SVD (Blattmann et al., 2023) first utilize SDXL (Podell et al., 2023) to generate images conditioning on the input text, and then generate videos based on the generated images. MAGVIT2 (Yu et al., 2023) is a VQGAN (Esser et al., 2021) model that addresses the problem of codebook size and utilization by employing the lookup-free technique and training a large codebook. Specifically, it maps videos into quantized video token sequences and generates videos in an autoregressive manner. WALT (Gupta et al., 2023) proposes to patchify input videos to lower the training costs. Meanwhile, Latte (Ma et al., 2024) initializes its parameters from Pixart (Chen et al.), and investigate the training effciency of 4 DiT variants. SORA sparks the revolution of text-to-video generation, emerging a series of DiT (Peebles & Xie, 2023)-based video generation models, such as OpenSora (Zheng et al., 2024) and Mira (Ju et al., 2024). Except for these open source models, there are also some commercial video generation models that exhibit strong generation performance, such as kling, dreamachine, and vidu.

- 3 METHOD

In the construction of VidGen, we harnessed 3.8 million high-resolution, long-duration videos derived from the HD-VILA dataset. These videos were subsequently split into 108 million video clips. Following this, we tagged and sampled these video clips. The VILA model was then utilized for video captioning. Lastly, to rectify any data curating errors from the preceding steps, we deployed the LLM for further caption curating.

- 3.1 COARSE CURATION

To achieve efficient data curation with limited computational resources, we first employ a coarse curation approach. This involves scene splitting, video tagging, filtering, and sampling to reduce the computational load in subsequent stages of captioning and fine curation.

- 3.1.1 SCENE SPLITTING

Motion inconsistencies, such as scene changes and fades, are frequently observed in raw videos. However, since motion inconsistencies directly cut off the video semantics, text-to-video models are significantly sensitive to and confused by them, leading to heavy impairment on training efficiency. To alleviate their impairment, we follow the prior research (Blattmann et al., 2023; Chen et al., 2024; Zheng et al., 2024) to utilize PySceneDetect (Castellano, 2014) in a cascading manner to detect and remove scene transitions in our raw videos.

- 3.1.2 TAGGING

In order to construct a dataset suitable for training text-to-video models, the data must meet the following criteria: high-quality videos, balanced categories, and strong temporal consistency within the videos. To achieve this goal, we first need to tag each splitted video clip. Subsequently, these tags serve as the basis for curating and sampling.

Video Quality. The visual quality of videos is of paramount importance for the efficient training of text-to-video models. In order to enhance the quality of generated videos in text-to-video generation, we adopt a strategy of filtering out videos with low aesthetic appeal and high OCR scores. In this context, we employ the LAION Aesthetics model to predict and evaluate aesthetic scores, thereby ensuring a superior quality of training data. Particularly, the aesthetics models can also filter out visually abnormal videos, such as videos with irregular color distribution or weird visual elements.

Temporal Consistency. Incorrect scene splitting in videos can significantly impair the effectiveness of model training. High temporal consistency is a crucial characteristic required for the training data in text-to-video models. To ensure this, we utilize the CLIP model to extract visual features and assess temporal consistency. This assessment is achieved by calculating the cosine similarity between the starting and ending frames of video clips, thereby providing a quantitative measure of continuity and coherence.

Category The HD-VILA-100M video dataset displays significant imbalances across its categories, resulting in less than optimal performance of video generation models for these categories. To tackle this issue, we deploy predefined category tags to label each video, with the assistance of the CLIP model. Specifically, we extract the CLIP image features from the initial, middle, and final frames of each video, compute their average, and then determine the similarity between these averaged image features and the textual features associated with each tag. This methodology enables us to assign the most fitting tags to each video.

Motion. We employ the RAFT (Teed & Deng, 2020) model to predict the optical flow score of videos. As both static videos and those with excessively fast motion are detrimental for training text-to-video models, we filter out these videos based on their optical flow scores.

- 3.1.3 SAMPLING

By employing tags associated with visual quality, temporal consistency, category, and motion, we undertook the task of filtering and sampling. The curated data distribution across multiple dimensions in our dataset is depicted in Figure 2. This figure clearly indicates that videos characterized

[Figure 4]

[Figure 5]

[Figure 6]

(a) Category distribution (b) Aesthetics distribution (c) Text-Video alignment distribution

[Figure 7]

[Figure 8]

[Figure 9]

(d) Temproal consistency distribution (e) OCR distribution (f) Motion distribution

Figure 2: Distribution of curated data.

by low quality, static scene, excessive motion speed, and those demonstrating inadequate alignment between text and video along with poor temporal consistency were systematically removed. Concurrently, we have ensured a relatively even distribution of samples across diverse categories.

- 3.2 CAPTIONING

Dataset VN/DN VV/DV Avg N Avg V Pandas-70M 16.1% 19.2% 4.3 1.9 Ours 20.3% 41.1% 22.5 15.9

- Table 2: Statistics of noun and verb concepts for different datasets. VN: valid distinct nouns (appearing more than 10 times); DN: total distinct nouns; Avg N: average noun count per video. VV: valid distinct verbs (appearing more than 10 times); DV: total distinct verbs; Avg N: average verbs count per video.

The quality of video captions exerts a critical influence on text-to-video model, while the captions in the HD-VILA-100M dataset demonstrate several problems, including misalignment between text and video, inadequate descriptions, and limited vocabulary use. To enhance the information density of the captions, we employ the cutting-edge vision-language model, VILA (Lin et al., 2024). Owing to the remarkable video captioning ability of VILA, we have significantly enhanced caption quality. After captioning, we apply clip score to filter out the text-video pairs with low similarity.

We present a vocabulary analysis in Table 2, where we identify valid distinct nouns and valid distinct verbs as those that appear more than 10 times in the dataset. Utilizing the VILA model on the HDVILA-100M dataset, we have generated the enhanced HD-VILA-100M dataset. In the Panda-70M dataset, there are 270K distinct nouns and 76K distinct verbs; however, only 16.1% and 19.2% of these meet the validity criteria, respectively. Captions generated using VILA substantially enhance the valid ratio as well as the average count of nouns and verbs per video, thereby increasing the conceptual density.

Video Caption Pairs ST FLG Redup

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

The video shows a white boat moving through the water at a high speed, creating a large wake behind it. The boat is moving from the left to the right of the frame, and the wake it leaves behind is white and frothy. The water is a deep blue color, and the sky is clear and blue. In the background, there is a hilly coastline with green vegetation.

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

This video is an animated cartoon featuring a group of cats. The setting is a green field with a small pond. The first frame shows a small cat running towards the pond. The second frame shows a larger cat and a smaller cat standing by the pond. The third frame shows the small cat jumping into the pond. The fourth frame shows the small cat swimming in the pond. The fifth frame shows the small cat jumping out of the pond. The sixth frame shows the small cat running towards the larger cat. The seventh frame shows the small cat jumping into the larger cat's arms. The eighth frame shows the larger cat holding the small cat. The ninth frame shows the larger cat and the smaller cat smiling. The tenth frame shows the larger cat and the smaller cat hugging.

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

The video opens with a close-up shot of a small green frog sitting on a white surface. The frog has a smooth texture and is looking directly at the camera. The next shot shows a brown frog being held in a person's hand. The frog has a rough texture and is looking forward. The video includes text overlays that read "THIS RARE FROG IS THE UNLIKELY SYMBOL OF THE BATTLE OVER ENDANGERED SPECIES.

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

The video shows a woman in a purple blazer and a striped shirt sitting in front of a blue background. She is speaking to the camera and appears to be in a news studio. There is a caption at the bottom of the screen that reads \\"BREAKING BREAKING BREAKING BREAKING BREAKING BREAKING BREAKING BREAKING BREAKING BREAKING BREAKING

- Figure 3: Caption Curation Result. Employing Llama3.1 to curate video captions, as produced by VILA, results in a significant improvement in the quality of the training dataset. This is specifically evidenced through improvements in video temporal consistency and alignment between text and video. Consequently, this method facilitates notable improvements in the performance of the textto-video model.

- 3.3 FINE CURATION

In the stages of coarse curation and captioning, filtering for text-image alignment and temporal consistency using the CLIP score can remove some inconsistent data, but it is not entirely effective. Consequently, issues such as scene transitions in video, and two typical description errors occur in video captions: 1) Failed generating eos token, where the model fails to properly terminate the generation process, leading to looping or repetitive token generation, and 2) Frame-level generation, where the model lacks understanding of the dynamic relationships between frames and generates isolated descriptions for each frame, resulting in captions that lack coherence and fail to accurately reflect the video’s overall storyline and action sequence.

To address the mentioned data curating issues, one potential solution is manual annotation, but this approach is prohibitively expensive. Fortunately, with recent advancements in large language models, this problem can be resolved. Errors in captions generated by Multi-Modal Language Models (MLLMs) can be identified by analyzing specific patterns, such as scene transitions, repetitive content, and frame-level descriptions, using a Language Model (LLM). Models like LLAMA3 have shown exceptional proficiency in these tasks, making them a viable alternative to manual annotation.

In our endeavor to isolate and remove video-text pairings that exhibit discrepancies in both textvideo alignment and temporal consistency, we leveraged the cutting-edge Language Model (LLM), LLAMA3.1, to scrutinize the respective captions. The application of the fine curation has facilitated a marked improvement in the quality of the text-video pairs, as evidenced in Figure 3. Our study primarily centers around three critical factors: Scene Transition (ST), Frame-level Generation (FLG), and Reduplication (Redup).

- 4 EXPERIMENTS

- 4.1 IMPLEMENTATION DETAILS.

Experiment setup. To evaluate the effectiveness of our text-to-video training dataset, we performed a comprehensive evaluation using the base model, composed of both spatial and temporal attention blocks. To accelerate the training process, we initially conducted extensive pre-training on a large collection of low-resolution 256 × 256 images and videos. Following this, we carried out joint training on our VidGen-1M dataset using 512 × 512 px resolution images and 4-second videos.

- 4.2 EXPERIMENT RESULTS 4.2.1 QUALITATIVE EVALUATION.

As displayed in Figure4, our model’s ability to generate superior-quality videos is a testament to the robustness of the high-resolution VidGen-1M dataset. This dataset’s high quality is reflected in the realism and detail of the generated videos, reinforcing its effectiveness in training our model. A noteworthy characteristic of our generated videos is their strong ”prompt following” ability, which is a direct outcome of the high consistency between video-text pairs in the training data. This consistency ensures that the model can accurately interpret the textual prompts and generate corresponding video content with high fidelity. The first example further underlines the high quality of the VidGen1M dataset. The generated video demonstrates remarkable realism - from the diver’s hair floating underwater to the motion of the bubbles. These details, which showcase significant temporal consistency and adhere to real-world physics, highlight the model’s capability to generate believable and visually accurate video content.

The VidGen-1M dataset’s quality has far-reaching implications for the field of computer vision, particularly for text-to-video generation. By providing high-resolution and temporal consistency training data, VidGen-1M enables models to generate more realistic and high-quality videos. This can lead to advancements in video generation techniques, pushing the boundaries of what is currently possible. Furthermore, the high-quality data provided by VidGen-1M could potentially streamline the model training process. With more accurate and detailed training data, models can learn more effectively, potentially reducing the need for extensive computational resources and time-consuming training periods. In this way, VidGen-1M not only improves the outcomes of text-to-video generation but also contributes to more efficient and sustainable model training practices.

- 5 CONCLUSION

In this paper, we introduce a high-quality video-text dataset features high video quality, high caption quality, high temporal consistency and high video-text alignment, specifically designed for the training of text-to-video generation models. The aforementioned various high quality features arise from our meticulously data curation procedure, which efficiently ensures high data quality in a coarse-tofine manner. To verify the effectiveness of VidGen-1M, we train a text-to-video generation model on it. The results are promising, the model trained on VidGen-1M achieves remarkably better FVD scores on zero-shot UCF101, compared with state-of-the-art text-to-video models. To bootstrap the development of high performance video generation models, we will release VidGen-1M, along with the related codes and the models trained on it, to the public.

REFERENCES

Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In Proceedings of the IEEE international conference on computer vision, pp. 5803–5812, 2017.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

[Figure 26]

Prompt: In clear blue water, a diver in a black wetsuit and yellow mask, holding a camera and surrounded by bubbles, is filming the vast underwater world.

[Figure 27]

Prompt: A brightly lit rocket trails smoke as it ascends into the stark night sky.

[Figure 28]

Prompt: In a dimly lit, green and murky aquarium, variously sized yellow

### and orange fish swim and hide among rocks and plants.

[Figure 29]

#### Prompt: A man in a red cap and grey shirt stands on a beach, speaking to the camera, with the calm sea showing small waves and rocks visible in the water, and a setting sun in a clear sky with a few clouds.

- Figure 4: Qualitative Evaluation: The model we’ve developed can generate videos from natural language prompts at a resolution of 512 × 512. These videos are 4 seconds long and play at 8 frames per second. Notably, our model can generate photorealistic videos that maintain temporal consistency and align accurately with the textual prompt.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pp. 961–970, 2015.

Brandon Castellano. Pyscenedetect. https://github.com/Breakthrough/ PySceneDetect, 2014.

David Chen and William B Dolan. Collecting highly parallel data for paraphrase evaluation. In Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies, pp. 190–200, 2011.

Junsong Chen, YU Jincheng, GE Chongjian, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In The Twelfth International Conference on Learning Representations.

Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13320–13331, 2024.

Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873–12883, 2021.

Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, and Jos´e Lezama. Photorealistic video generation with diffusion models. arXiv preprint arXiv:2312.06662, 2023.

Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions, 2024. URL https://arxiv.org/abs/2407.06358.

Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pretraining for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26689–26699, 2024.

Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024.

Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 2630–2640, 2019.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Anna Rohrbach, Marcus Rohrbach, Niket Tandon, and Bernt Schiele. A dataset for movie description. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3202–3212, 2015.

Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.

Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pp. 402–419. Springer, 2020.

Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4581–4591, 2019.

Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5288–5296, 2016.

Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, and Baining Guo. Advancing high-resolution video-language representation with large-scale video transcriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5036–5045, 2022.

Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion– tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.

Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu, Jae Sung Park, Jize Cao, Ali Farhadi, and Yejin Choi. Merlot: Multimodal neural script knowledge models. Advances in neural information processing systems, 34:23634–23651, 2021.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, March 2024. URL https://github.com/hpcaitech/Open-Sora.

Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.

