# arXiv:2503.16660v1[cs.CV]20Mar2025

## When Less is Enough: Adaptive Token Reduction for Efficient Image Representation

Eduard Allakhverdov AIRI Moscow, Russia MIPT Dolgoprudny, Russia

allakhverdov@2a2i.org

Elizaveta Goncharova AIRI Moscow, Russia

goncharova@airi.net

Andrey Kuznetsov AIRI Moscow, Russia

kuznetsov@airi.net

### Abstract

Vision encoders typically generate a large number of visual tokens, providing information-rich representations but significantly increasing computational demands. This raises the question of whether all generated tokens are equally valuable or if some of them can be discarded to reduce computational costs without compromising quality. In this paper, we introduce a new method for determining feature utility based on the idea that less valuable features can be reconstructed from more valuable ones. We implement this concept by integrating an autoencoder with a Gumbel-Softmax selection mechanism, that allows identifying and retaining only the most informative visual tokens. To validate our approach, we compared the performance of the LLaVA-NeXT model, using features selected by our method with randomly selected features. We found that on OCR-based tasks, more than 50% of the visual context can be removed with minimal performance loss, whereas randomly discarding the same proportion of features significantly affects the model capabilities. Furthermore, in general-domain tasks, even randomly retaining only 30% of tokens achieves performance comparable to using the full set of visual tokens. Our results highlight a promising direction towards adaptive and efficient multimodal pruning that facilitates scalable and low-overhead inference without compromising performance.

### 1. Introduction

In recent years, vision encoders have become important components for various downstream tasks, providing universal representation of visual features. These encoders are trained to effectively compress raw pixel information into latent embeddings. Depending on their training objectives, vision encoders can encapsulate different types of informa-

[Figure 1]

Figure 1. Comparison of feature selection methods on Newton’s Principia text: original image (left), random feature selection retaining 40% of tokens (middle), and our proposed feature selector retaining 40% of tokens (right).

tion in their hidden states. However, it is widely recognized that many of these encoded features contain redundant or irrelevant information for downstream tasks [25, 27, 31]. Therefore, reducing the number of output features produced by vision encoders is an important and challenging task especially now, as encoders increasingly serve as funda-

mental mechanisms for visual understanding in multimodal models [2, 15, 31].

Multimodal models that process visual inputs typically condition on outputs of a Vision Transformer (ViT) [5], appending a long vision-derived prefix to the input of a Large Language Model (LLM) via a projection layer. Although this method gives promising results, handling large context length (especially when processing high-resolution images) remains a significant challenge. Moreover, previous studies have observed that not all ViT outputs equally contribute to downstream task performance [4]; many tokens can be redundant, noisy, or simply irrelevant [34]. Therefore, selectively identifying and retaining only the most informative features can significantly decrease the number of tokens while maintaining model performance.

To address this issue, we propose a novel method to select the most informative visual features from the encoder output using an autoencoder-based approach implemented with Gumbel-Softmax sampling. Our method identifies features that are essential to preserve crucial visual information, allowing us to accurately reconstruct the original feature set. We show that this training procedure not only efficiently identifies valuable features, but also provides highly interpretable results. Furthermore, we illustrate how our approach can be seamlessly integrated during inference in multimodal models, significantly reducing the visual prefix length without compromising performance.

In experiments conducted with the LLaVA-NeXT [15] and LLaVA-OneVision [13] models, we demonstrate that features selected using the the proposed approach contain essential information for the model to provide the correct answer to most of the analyzed tasks. Notably, our method reduces visual context length by up to 50% with minimal performance degradation in most benchmarks — and achieves reductions of up to 90% in certain tasks.

The contributions of our paper can be summarized as follows:

- • We propose a novel, interpretable method for selecting the most informative features from vision encoders.
- • We demonstrate how our approach serves as an effective in-situ feature reduction method for existing multimodal models without requiring further fine-tuning.
- • We empirically confirm that retaining as little as 50% of the original visual features can be sufficient to maintain near-baseline performance on multiple multimodal benchmarks.

### 2. Related Work

Several approaches to reduce redundancy in token-based visual representations have been proposed in recent literature. We will discuss the most relevant ones below.

#### 2.1. Token Pruning

Token pruning is a widely used method to remove irrelevant or noisy features from Vision Transformers while preserving the most valuable information. Many token pruning methods rely on attention scores to guide feature selection [30]. To further improve versatility, some approaches propose to retain not only the most relevant but also the most diverse set of tokens, thereby preserving a richer representation of the image [20]. Maintaining high quality embeddings is critical, especially in detection and segmentation tasks, as they require detailed and precise image representation [19].

Token pruning methods are often tailored to specific vision tasks. For example, Kinfu and Vidal [11] proposes three specialized pruning methods for human pose estimation. Two of these methods use a lightweight pose estimation network to guide patch selection, while the third uses learnable joint tokens to explicitly ensure that the selected patches retain key body joint information. The overall goal is to preserve the most informative embeddings needed for accurate pose estimation.

#### 2.2. Token Generation and Merging

Another line of research involves generating or merging tokens to obtain a compact yet informative subset from the original set of tokens. TokenLearner [28] merges hidden representations within Vision Transformers to produce a concise output containing only eight tokens that encapsulate internal functionality. Similarly, Token Merger [6] introduces the concept of “meta-tokens,” adaptively merging similar tokens to retain essential information. In [12], learnable decoupled embeddings are used, allowing end-toend token to be merged without relying solely on similarity measures. In addition, Resizable-ViT [37] introduces a flexible module that learns token-length labels, identifying the most informative tokens for detailed analysis.

Some models adopt a fixed, stage-wise downsampling strategy. For example, the Pyramid Vision Transformer (PVT) [33] divides the input image into increasingly coarser token maps at deeper layers, mimicking the pyramidal structure of CNNs. Unlike adaptive token merging [6], PVT statically reduces sequence length at each stage, thereby reducing computational cost for high-resolution inputs. Although this hierarchical design has proven effective for tasks such as detection and segmentation, it does not allow dynamic pruning or merging of tokens based on their content.

In addition, Li et al. [14] explores the broader implications of tokenization in Vision Transformers, proposing convolutional patterns to enhance visual representation quality.

While these token pruning and merging methods show promising results on traditional computer vision tasks (clas-

sification, segmentation, detection), they usually lack versatility. Most existing approaches cannot be directly transferred to multimodal models, which must leverage as much relevant visual information as possible in conjunction with textual inputs.

#### 2.3. Vision Context Reduction in Multimodal Models

Reducing the size of the visual context is particularly important in multimodal models, where visual tokens significantly increase the context length required by the LLM. However, visual tokens vary in their relevance depending on the query. Various strategies have been developed to address this problem. For example, interpolation-based methods reduce the number of visual tokens by attempting to preserve critical visual information from the entire feature subset (e.g., LLaVA-OneVision [13]). InternVL series of models [3] use a pixel-unshuffle operation typically applied to high-resolution images. Other popular approaches use trainable modules that either compress visual tokens or introduce learnable queries to extract important visual information, such as Perceiver IO [9] and BLIP-2 [16].

In our work, we propose a task-agnostic method to directly select the most informative visual features from the encoder outputs without fine-tuning or dependence on textual input. Therefore, our approach can be seamlessly applied to both purely visual tasks and multimodal scenarios.

### 3. Useful Feature Selection

The Transformer architecture has been successfully used as a backbone for vision encoders [5], providing hidden representations suitable for a wide range of vision tasks. However, due to the inherent design of the self-attention mechanism in Transformers, neighboring tokens naturally contain information about each other. Consequently, we assume that information may be duplicated redundantly in different regions of the output feature tensor. In particular, some visual representations could potentially be composed entirely of information already present in other tokens. If such redundant representations exist, they can be identified and removed without causing significant performance degradation in vision-related tasks.

This hypothesis naturally raises two critical questions: how can one quantitatively measure whether one set of features contains more information than another, and how can one select the optimal subset of features?

#### 3.1. Feature Subset Comparison

For any image I, the corresponding feature set F has dimensions (L,C). As mentioned earlier, tokens identified for potential exclusion have the characteristic property that they can be reconstructed from the remaining visual tokens in the set. Thus, if there exists an optimal reconstruction

- function R, which takes a pruned subset of features as input Fpr (where the superscript pr denotes pruned) with dimensions (Lpr,C) and returns a reconstructed set Frec with dimensions (L,C), and if a proximity function dist is defined between two tensors, then one subset is considered superior to the other if it allows for a more accurate reconstruction of the discarded visual tokens.

Formally, subset F1pr is superior to subset F2pr if:

dist(R(F1pr),F) < dist(R(F2pr),F). (1)

- 3.2. How to Select the Optimal Set?

To select the most informative features, we aim to find a function S, referred to as the optimal selector, which takes F as input and returns a pruned subset Fpr. We train this selector in a way similar to the autoencoder:

min

θ,ψ

dist Rψ(Sθ(F)), F + Lpr. (2)

In this formulation, the first term provides a high-quality reconstruction of the original feature set from the pruned subset. The additional term Lpr penalizes the selector Sθ if it trivially selects all tokens (acting as an identity function), thereby encouraging a more concise but informative subset.

- 4. Method 4.1. Implementation Details

In this section, we present the implementation details of the approach described in Sec. 3, which consists of two main components: Feature Selector and Feature Reconstructor.

##### 4.1.1. Feature Selector Architecture

Feature Selector S consists of three Transformer layers and a Gumbel-Softmax-based [10] head. The head creates a binary mask, as shown in Fig. 2, where zeros indicate visual tokens to be removed and ones indicate tokens to be retained.

During training, feature embeddings corresponding to zeros in the binary mask are replaced by a shared learnable embedding Emasked (this embedding will be reconstructed later by the component described in Sec. 4.2). During inference, embeddings corresponding to zeros are simply discarded, while those corresponding to ones are kept for downstream task. For example, they can be used as image representations in Vision-Language models, as shown in our experiments in Sec. 5.

For more flexibility during inference, one can choose to use logits from the linear layer instead of a hard binary mask. Based on these logits, the user can select a fixed number of the most informative features. This is exactly the approach that is used in our experiments, which we describe in Sec. 5.

[Figure 2]

Figure 2. Illustration of the Feature Selector in training mode. It uses three Transformer layers and a Gumbel-Softmax head to generate a binary mask where zeros mark tokens for removal and ones for retention. During training, the masked embeddings are replaced by a shared learnable embedding. During inference, the masked embeddings are discarded, while the retained ones are used for downstream tasks, such as image representations in Vision-Language models.

#### 4.2. Feature Reconstructor Architecture

Feature Reconstructor R also consists of three Transformer layers. Its primary objective is to restore the tokens that were replaced with a learned embedding Emasked, as illustrated in Fig. 3.

#### 4.3. Loss Function

As described in Sec. 3.2, the optimization objective is formulated as the sum of two terms: (1) a reconstruction loss and (2) a regularization term that aims to minimize the amount of information required for reconstruction. In principle, we would expect the reconstruction loss to approach zero while the regularization term converges to the fraction of useful visual tokens. However, in practice we did not observe the expected behavior. We found that the optimizer is more likely to converge to a local minimum where the regularization term drops to zero, thereby avoiding the token utilization penalty.

To resolve this issue, we modify the regularization term as follows.

Lpr → max(Lpr, p), (3) where p is a predetermined proportion of useful features.

In other words, whenever Lpr falls below p, the regularization penalty is effectively disabled, allowing for further optimization of the reconstruction loss. Empirically, we observe that Lpr initially decreases to p and then fluctuates

[Figure 3]

Figure 3. Illustration of Feature Reconstructor’s functionality. Its primary objective is to restore the tokens that were replaced with a learned representation.

around it during the remainder of training, while the reconstruction loss continues to decrease.

#### 4.4. Training

As shown in Figs. 2 and 3, our approach is similar to VQVAE [32], except that we use a set of input features instead of a learned dictionary, and the latent representation may vary in size.

We train both Rψ and Sθ following the framework introduced in Sec. 3.2. Specifically, we choose the l2 norm for the distance function dist and compute Lpr using the mask generated by Sθ.

Feature Selector. The feature selector Sθ processes the original feature tensor F and outputs a subset of selected features Fpr along with a binary mask M, referred to as the “Gumbel mask,” as illustrated in Fig. 2. Formally, this can be expressed as:

###### Fpr, M = Sθ(F), (4)

where the mask M specifies which spatial locations of the input tensor F are retained (marked as ones) and which are discarded (marked as zeros). The output Fpr, labeled as “Selected features” in Fig. 2, is formed by replacing the discarded feature vectors with a shared learnable representation (shown as blue hatched vectors).

Feature Reconstructor. The reconstructor is defined by: Frec = Rψ(Fpr), (5)

with Frec denoting the “Reconstructed tensor” shown in Fig. 3.

Regularization Term. Regularization term is computed directly from the mask:

Lpr =

H,W

Mh,w HW

. (6)

h=0,w=0

Overall Objective. Incorporating the modified regularization from Sec. 4.3, the overall optimization problem can be defined as follows:

min

θ,ψ

Frec − F

2

+ max Lpr, p . (7)

All components are fully differentiable, and we optimize them using gradient descent.

#### 4.5. Dataset

For our training dataset, we sampled 100K images from the COCO dataset [17]. Each image was pre-processed with a specific vision encoder for which the selector was trained. The resulting feature representations were used as training data.

### 5. Experiments

To evaluate the performance of the proposed feature selector, we integrated it with vision encoders used as backbones in two multimodal models: LLaVA-NeXT (visual encoder of CLIP [26]) and LLaVA-OneVision (visual encoder of SigLIP [36]). Our primary goal was to investigate whether the selector described in Sec. 4.1.1 can be applied directly (i.e., without further fine-tuning) to existing multimodal architectures, and to determine whether selecting informative features positively impacts model inference. To this end, we evaluated both models augmented with our trained feature selector under different pruning factors, comparing their performance to a baseline using random feature selection with the same pruning factors across multiple multimodal benchmarks.

#### 5.1. LLaVA-NeXT: Our Selector vs. Random Selector

Figures 4 and 5 present the experimental results obtained using the LLaVA-NeXT model in various multimodal benchmarks. For clarity, the results are organized into two groups. The first group (Fig. 4) contains OCR-like benchmarks involving large images, including DocVQA [24],

ChartQA [22], InfoVQA [23], TextVQA [29], and MMBench [18]. While MMBench does not consist entirely of OCR-based tasks, it includes complex tasks that require image comprehension. The second group (Fig. 5) includes the other benchmarks that evaluate the ability of models to solve various academic tasks and perform scene understanding (AI2D [7], GQA [8], MMMU [35], MMStar [1], and ScienceQA [21]).

In each figure, the horizontal axis represents the proportion of features retained (i.e., the parameter p defined in Sec. 4.3), while the vertical axis indicates the evaluation metric obtained on each benchmark. There we compare the performance of the original model using all available features (green dashed line) with the model using features selected by our trained selector (orange line) and randomly selected features (blue line). We also measure the original model without showing images (red dashed line) to understand how important it is to see the image when solving this benchmark.

From the OCR-like benchmark results (Fig. 4), two primary conclusions can be drawn:

- 1. The trained selector significantly outperforms random selection across all tested retention ratios.
- 2. Up to 50% of visual features can be discarded using the trained selector with only negligible performance degradation. From the remaining benchmarks (see Fig. 5), we observe

two additional outcomes:

- 1. The quality of the model’s responses remains largely unaffected by the number of visual features retained and closely matches the baseline with all features.
- 2. In these tasks, the choice of feature selection method (trained vs. random) has minimal impact on the quality of the response.

#### 5.2. LLaVA-OneVision: Our Selector vs. Random Selector

The LLaVA-OneVision model includes a built-in compression mechanism, which reduces the size of the visual feature tensor through interpolation. In the LLaVA-OneVision implementation, interpolation compression is activated from a predefined reference size, and the compression ratio varies depending on the input image dimensions. For example, larger documents typically result in compression ratios of 1.7 to 2.

Since our main goal is to demonstrate that our proposed selector effectively identifies informative visual features, we disabled the default compression mechanism in LLaVAOneVision instead of combining two distinct compression methods.

During preliminary experiments, we noticed that if the visual context exceeds the internally fixed context length of the model, the performance tends to degrade.

[Figure 4]

Figure 4. Comparison of LLaVA-NeXT performance with our selector (orange) and random selector (blue) on text-based benchmarks. The green dashed line represents the baseline performance using all features. The red dashed line represents the model’s performance without image input.

Therefore, we limited our experiments to the range p ∈ [0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ≈ 11.7], ensuring that the length of the retained visual features does not exceed approximately 11.7 the original input length.

Similar to the LLaVA-NeXT evaluations, the results for LLaVA-OneVision are divided into two categories, as depicted in Figs. 6 and 7.

For OCR-based benchmarks involving high-resolution images (Fig. 6), our trained selector consistently outperforms random selection, proving its ability to effectively preserve informative visual features.

In contrast, for non-OCR benchmarks (Fig. 7), the results are very similar to those observed previously with LLaVANeXT (Fig. 5), indicating that the performance is largely unaffected by the choice of selection method in these sce-

[Figure 5]

Figure 5. Comparison of LLaVA-NeXT performance with our selector (orange) and random selector (blue) on non-text benchmarks. The green dashed line represents the baseline performance using all features. The red dashed line represents the model’s performance without image input.

narios.

### 6. Discussion

Given the experimental results presented in Sec. 5, an important question arises: why does a trained feature selector significantly improve performance on OCR-like tasks involving high-resolution images, but not consistently on other tasks? The following analysis attempts to explain this discrepancy.

The strongest difference in performance depending on the percentage of visible tokens is observed on ChartQA, DocVQA, and TextVQA, InfoVQA benchmarks. These tasks have a common characteristic: to provide the correct answer, the model must identify the detailed visual information in the image, without extensive logical reasoning afterwards. In other words, these tasks rely heavily on visual perception and minimally on subsequent cognitive reasoning, making the advantage of selectively retaining informative features particularly pronounced.

In contrast, MMBench also requires careful perception of visual details, but solving them additionally requires logical reasoning. As a result, the overall performance depends

[Figure 6]

Figure 6. Comparison of LLaVA-OneVision performance with our trained selector (orange) and random selection (blue) on text-based (OCR-like) benchmarks. The green dashed line represents the baseline performance with all features retained. The red dashed line represents the model’s performance without image input.

not only on the quality of visual input, but also significantly on the reasoning capability of the language model, thereby reducing the relative advantage provided by selective feature sampling. However, we still observe that our feature selector outperforms the random one.

For the above benchmarks, we see a consistent pattern where metrics improve substantially as the percentage of visible features increases from 10% to approximately 50%. Beyond this threshold, performance gains become increasingly insignificant.

The tasks represented by the MMMU and MMStar datasets (in particular, the logical reasoning, mathematical reasoning, and science & technology categories) have the highest reasoning complexity. Here, the language model’s reasoning abilities play a predominant role, effectively over-

[Figure 7]

Figure 7. Comparison of LLaVA-OneVision performance with our trained selector (orange) and random selection (blue) on non-text benchmarks. The green dashed line represents the baseline performance with all features retained. The red dashed line represents the model’s performance without image input.

shadowing the impact of feature selection strategies.

Finally, tasks from AI2D, GQA, ScienceQA, and certain categories in MMStar (e.g., coarse perception, fine-grained perception, and instance reasoning) rely on a general understanding of the scene rather than isolated visual details. For these benchmarks, comprehensive feature coverage across the image proves more important than the selective retention of individually informative features. Consequently, random selection can perform comparable to or even outperform intelligent sampling in some cases due to more uniform coverage of the entire scene (e.g., in MMStar’s coarse perception and instance reasoning tasks).

It is also interesting to note the performance of models inferenced without a visible image, and the different percentages of sampled image features. As we can see for the MMStar benchmark, there is a gap between the no image and the entire image observed by the model, accounting for 10-20% performance improvement on average. Retaining only 10% of the features maintains performance comparable to using the entire image. For the MMMU benchmark, the behavior is noticeably different. We can observe (see Figs. 5 and 7) that there is no significant difference between no image and the entire image. Performance on this

[Figure 8]

Figure 8. Images from three benchmarks illustrating cases where the vision-language model gives correct answers or makes errors. The first column shows the model’s responses using the full visual context, the second column uses a randomly selected set of features, and the third column uses the features selected by our selector. (1) DocVQA: to answer the question selecting the correct features is crucial. (2) MMMU (math): to answer this question, both visual understanding and logical reasoning are important, but the model fails to reason correctly. (3) MMstar: the image details are less important, and the language model plays a dominant role.

benchmark depends primarily on the quality of the language model rather than on the ability to perceive visual information.

In Fig. 8, we provide examples of the images from three different benchmarks, illustrating the sampling mechanism and the types of questions for which the vision-language model either produces correct answers or makes errors.

### 7. Conclusion

In this paper, we proposed a novel method to select informative features from visual encoders. Our method is based on trainable VAE module with the Gumbel-Softmax built-in the ViT model. The proposed method allows to reduce the number of output vision tokens, while maximally preserving the most informative ones. To validate the effectiveness of our approach, we demonstrated how the proposed module can serve as an efficient feature reduction strategy for

multimodal models. Experimental results show that up to 50% of the visual context can be removed with minimal performance degradation for the LLaVA-NeXT and LLaVAOneVision models.

Furthermore, our feature selection method demonstrates high interpretability, as illustrated by qualitative examples given in Fig. 1, where the selected features clearly correspond to specific objects and shapes in the images.

However, we acknowledge certain limitations of the proposed method, including limited compatibility with interpolation-based feature compression methods that are widely used in modern multimodal models. To address this limitation, we plan to study joint fine-tuning of the feature selector and the language model as future work. Such fine-tuning can potentially mitigate compatibility issues that arise when combining different compression strategies (e.g., interpolation-based and Gumbel-based selections).

We believe that these results open promising new directions for research aimed at extracting informative representations from visual contexts. Potential applications of this approach include faster inference, reduced memory consumption, and improved resilience to noisy visual inputs.

### References

- [1] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models?, 2024. 5
- [2] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks, 2024. 2
- [3] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Kaipeng Zhang, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling, 2025. 3
- [4] Alessio Devoto, Federico Alvetreti, Jary Pomponi, Paolo Di Lorenzo, Pasquale Minervini, and Simone Scardapane. Adaptive layer selection for efficient vision transformer finetuning, 2024. 2
- [5] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale, 2021. 2, 3

- [6] Zhanzhou Feng and Shiliang Zhang. Efficient vision transformer via token merger. IEEE Transactions on Image Processing, 32:4156–4169, 2023. 2
- [7] Tuomo Hiippala, Malihe Alikhani, Jonas Haverinen, Timo Kalliokoski, Evanfiya Logacheva, Serafina Orekhova, Aino Tuomainen, Matthew Stone, and John A. Bateman. Ai2drst: a multimodal corpus of 1000 primary school science diagrams. Language Resources and Evaluation, 55(3):661–688,

2020. 5

- [8] Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering, 2019. 5
- [9] Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier H´enaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, and Jo¯ao Carreira. Perceiver io: A general architecture for structured inputs & outputs, 2022. 3
- [10] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax, 2017. 3
- [11] Kaleab Alamayehu Kinfu and Ren´e Vidal. Efficient vision transformer for human pose estimation via patch selection. In British Machine Vision Conference, 2023. 2
- [12] Dong Hoon Lee and Seunghoon Hong. Learning to merge tokens via decoupled embedding for efficient vision transformers, 2024. 2
- [13] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer, 2024. 2, 3
- [14] Changzhen Li, Jie Zhang, Yang Wei, Zhilong Ji, Jinfeng Bai, and Shiguang Shan. Patch is not all you need, 2023. 2
- [15] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models, 2024. 2
- [16] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models, 2023. 3
- [17] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Doll´ar. Microsoft coco: Common objects in context, 2015. 5
- [18] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player?, 2024. 5
- [19] Yifei Liu, Mathias Gehrig, Nico Messikommer, Marco Cannici, and Davide Scaramuzza. Revisiting token pruning for object detection and instance segmentation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024. 2
- [20] Sifan Long, Zhen Zhao, Jimin Pi, Shengsheng Wang, and Jingdong Wang. Beyond Attentive Tokens: Incorporating Token Importance and Diversity for Efficient Vision Transformers . In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10334–10343, Los Alamitos, CA, USA, 2023. IEEE Computer Society. 2

- [21] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering, 2022. 5
- [22] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning, 2022. 5
- [23] Minesh Mathew, Viraj Bagal, Rub`en P´erez Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V Jawahar. Infographicvqa, 2021. 5
- [24] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Docvqa: A dataset for vqa on document images, 2021. 5
- [25] Muzammal Naseer, Kanchana Ranasinghe, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Intriguing properties of vision transformers, 2021. 1
- [26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 5
- [27] Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. Do vision transformers see like convolutional neural networks?, 2022. 1
- [28] Michael Ryoo, AJ Piergiovanni, Anurag Arnab, Mostafa Dehghani, and Anelia Angelova. Tokenlearner: Adaptive space-time tokenization for videos. In Advances in Neural Information Processing Systems, pages 12786–12797. Curran Associates, Inc., 2021. 2
- [29] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read, 2019. 5
- [30] Quan Tang, Bowen Zhang, Jiajun Liu, Fagui Liu, and Yifan Liu. Dynamic token pruning in plain vision transformers for semantic segmentation. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 777– 786, 2023. 2
- [31] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Ziteng Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms,

2024. 1, 2

- [32] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning,

2018. 4

- [33] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 548– 558, 2021. 2
- [34] Jiawei Yang, Katie Z Luo, Jiefeng Li, Congyue Deng, Leonidas Guibas, Dilip Krishnan, Kilian Q Weinberger, Yonglong Tian, and Yue Wang. Denoising vision transformers,

2024. 2

- [35] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming

- Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi, 2024. 5
- [36] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training,

2023. 5

- [37] Qiqi Zhou and Yichen Zhu. Make a long image short: Adaptive token length for vision transformers. In Machine Learning and Knowledge Discovery in Databases: Research Track, pages 69–85, Cham, 2023. Springer Nature Switzerland. 2

