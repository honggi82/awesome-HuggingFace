## DC3DO: Diffusion Classifier for 3D Objects

Nursena Koprucu1,*, Meher Shashwat Nigam2,*, Shicheng (Luke) Xu5,*, Biruk Abere,*3, Gabriele Dominici,*4, Andrew Rodriguez2, Sharvaree Vadgama6, Berfin Inal6, Alberto Tono7,8 1Max Planck Institute for Intelligent Systems, Germany, 2Georgia Tech, USA, 3University of Gondar, Ethiopia 4Universit`a della Svizzera Italiana, Switzerland, 5Google LLC, USA, 6University of Amsterdam, Netherlands 7Stanford University, USA, 8Computational Design Institute, USA

# arXiv:2408.06693v1[cs.CV]13Aug2024

### Abstract

[Figure 1]

Inspired by Geoffrey Hinton’s emphasis on generative modeling (“To recognize shapes, first learn to generate them”), we explore the use of 3D diffusion models for object classification. Leveraging the density estimates from these models, our approach, “Diffusion Classifier for 3D Objects”, dubbed DC3DO, enables zero-shot classification of 3D shapes without additional training. Our method achieves on average 12.5% improvement compared with its multi-view counterparts, demonstrating superior multimodal reasoning compared to discriminative approaches. DC3DO uses a class-conditional diffusion model trained on ShapeNet. We run inferences on chairs and cars pointclouds. This work underscores the potential of generative models in 3D object classification. Code available https://github.com/SGI-2023/3D-Building-Classification.

### 1. Introduction

Figure 1. Dataset classes for classification. We performed 3D Classifications tests on cars, chairs, and airplanes. We used multiview and point cloud representations only for chairs and cars.

Recent advancements in deep generative models have yielded state-of-the-art (SOTA) performance in both classification and out-of-distribution (OOD) classification for images [18]. Deep generative models are increasingly being utilized for discriminative tasks, demonstrating superior effectiveness across various domains, including images [12], text [7], and tabular data [11, 30]. This progression builds upon the foundational work of Hinton [8], inspired by Oliver Selfridge’s “Pandemonium” model [36]. While these early researchers focused on generation within the image domain, one could argue that creation should originate in 3D space before extending to the image or text domains. By first generating 3D objects, we can enhance downstream tasks, not only in object classification but also in image and text classification.

struggle to handle the complexity and variability inherent in 3D data, we adopted a diffusion approach [9]. Diffusion models [38], a recent class of likelihood-based generative models, have shown significant promise in various tasks [10, 31, 34] by transforming random noise into coherent data samples through an iterative noising and denoising process.

Furthermore, today’s work in diffusion models [5, 32, 37] showed unmatched results not only on generative tasks [27] but also in classification tasks [26, 28]. Diffusion models belong to a class of generative models that model the data distribution of the dataset, similar to VAEs [17], GANs [2, 45], EBMs [47], Score-based models [51].

The classification of 3D shapes is increasingly important in fields such as computer vision, robotics, and virtual reality, hence this research. Since traditional methods often

Therefore, a question arises, Can we use diffusion mod-

els for 3D classification tasks? More critically, given their remarkable ability to generate original objects beyond the initial dataset distribution [27, 29, 44, 54, 58], how do these models perform for out-of-distribution (OOD) data. While these models have excelled on standard benchmarks, they often struggle with novel OOD data, a limitation attributed to biased training datasets that fail to encompass the full spectrum of real-world possibilities. This has been attributed to the biased training data that does not represent all real-world possibilities [13]. These deep generative models can synthesize strikingly realistic and diverse images, objects, and text and they have shown better performances in zero-shot [14, 15, 35], few-shot classification tasks [37].

In this research, we explore the application of Denoising Diffusion Probabilistic Models (DDPMs) [18] for classifying 3D shapes. Traditional classification methods often fall short with 3D data, requiring novel approaches. Furthermore, 3D data are represented as point clouds [27, 52, 59], voxels [4, 46], signed distance functions [29, 44, 55] and multi-view formats [40]. Inspired by LION [54], this research adopts point cloud and voxel [54, 59] combined with latent representations [29] and diffusion models: DC3DO. DC3DO focuses on leveraging the generative capabilities of diffusion models for zero-shot classification [19]. We compared it against a direct extenstion of the 2D conterpart performed on images [18]. In a dynamic data landscape, the ability to classify data into previously unseen categories, such as architectural structures [39, 42, 43], is of paramount importance. Diffusion models, with their inherent generative strengths, are particularly well-suited for this challenge. Furthermore, by advancing beyond traditional 2D prior models [22–24] and incorporating the LION model [54], renowned for generating high-fidelity 3D shapes, we enhance the effectiveness of the Diffusion Classifier in performing discriminative tasks, particularly in 3D object classification. Therefore, our contributions to this field are three-fold:

- • Novel Method for 3D shape classification: We introduce DC3DO to classify 3D shapes with a diffusion model.
- • Comparative analysis: We compare our method against multiview 3D representations using a 2D diffusion classifier [18]. We adapted MVCNN’s [40] with its view pooling method to a more U-Net and diffusion classifierfriendly method for a fair comparison.

In these unsupervised settings, diffusion models [9] are trained using the objectives of Variational Inference, specifically focusing on maximizing the evidence lower bound (ELBO) [41] of the log-likelihood, as described in [18]. This involves adding noise ϵ to a sample, using a neural network to predict the noise, and adopting Mean Squared Error (MSE) or L2 loss to compare this predicted noise against a white gaussian noise [1].

Given an input x and a finite set of classes c that we want to choose from, we can use the model to compute classconditional likelihoods pθ(x | ci), where i indicated the class number. Then, by selecting an appropriate prior distribution p(ci) and applying Bayes’ theorem, we can obtain predicted class probabilities p(ci | x). For conditional diffusion models that use an auxiliary input, like a class index for class-conditioned models or a prompt for text-to-image models, we can achieve this by leveraging the ELBO as an approximate class-conditional log-likelihood log p(x | ci). We repeatedly add noise and compute an estimate of the expected noise reconstruction losses (also called ϵ-prediction loss) for every class, as described in [18], please refer to Figure 2.

### 2. Related Work

Multimodal large language models (LLMs) strengths are leveraged in many current works [6, 16, 33, 48]. LLMs can handle diverse tasks through conversational interaction, specifically in the context of 3D objects. Typically, this is achieved by training a 3D shape encoder and aligning it with other modalities (e.g., text, images, and audio). The entire pipeline is then fine-tuned during an instruction-tuning phase, resulting in a model that is better aligned with user requests for specific 3D tasks. This fine-tuning stage is conducted using synthetic datasets or captioning datasets. These approaches highlight the vast potential of integrating 3D shapes into foundation models, although they still necessitate the fine-tuning of large models. Other methods, such as 3DAxiesPrompts [21], enhance images and prompts with additional artifacts to be able to exploit the 2D vision abilities of LLM for 3D objects.

PEVA-Net [20] employs a pre-trained CLIP model in a multiview pipeline to classify 3D objects in zero-shot or few-shot environments. It leverages CLIP’s zero-shot classification abilities for each view of the 3D object, subsequently aggregating these results to make the final prediction. Although this approach effectively exploits the zeroshot capabilities of vision-language models (VLMs), transforming 3D shapes into multiview images is an oversimplification that can lead to suboptimal results.

TAMM [57] demonstrates that when aligning 3D object representations with other modalities, the image modality contributes less significantly than the text modality. To address this, their method learns to separate visual features from semantic features within the 3D object representation, enabling a more effective alignment with the other modalities and enhancing performance in downstream tasks. These findings suggest that the alignment between modalities for integrating 3D representations into existing methods can sometimes be inadequate [50]. Regarding 3D representation learning, Zhang et al. [56] takes a different approach and incorporates 2D guidance. Their work, dubbed I2P-

MAE [56], learns advanced 3D representations, achieving state-of-the-art performance on 3D tasks and significantly lowering the need for large-scale 3D datasets. On the contrary concurrent work, DiffCLIP [37] demonstrates that the integration of CLIP and diffusion models for 3D classification facilitates zero-shot classification, achieving state-of-the-art results. This methodology utilizes a pretraining pipeline that incorporates a Point Transformer for few-shot 3D point cloud classification, wherein the CLIP model extracts style-based features of the class, synergistically combined with image features. While DiffCLIP [37] used Point Transformer we used LION, a latent point-voxel [25, 54, 59] representations that leverages a hierarchical two stages diffusion process with state of the art generative performances. Following the line of latent and implicit representations, Xin et al. [53] used a Classifier Score Distillation (CSD) method, which utilizes an implicit classification model for generation.

### 3. Methodology

In this section, we present and compare two distinct approaches for 3D object classification: Multi-View Diffusion Classifier (Section 3.1) and DC3DO (Section 3.2). The first approach, the Multi-View Diffusion Classifier, is designed to harness the power of diffusion models while maintaining the architecture of Diffusion Classifier [18]. The second approach, DC3DO, integrates the advanced generative capabilities of LION [54] with diffusion-based classification, targeting zero-shot classification of complex 3D shapes like cars and chairs.

Our goal is to thoroughly assess the performance of these methods in comparison to traditional and state-ofthe-art techniques. The Multi-View Diffusion Classifier (MVDC) offers an alternative to the widely-used MVCNN by employing a majority vote mechanism across multiple 3D views. On the other hand, DC3DO leverages LION’s robust generative framework (Section 3.3).

#### 3.1. Multi-View Diffusion Classifier (MVDC)

3D objects can be effectively represented as a series of images, providing a straightforward baseline for extending previous work [18] to the 3D domain. By simply aggregating multiple views of the same object, we can adapt existing diffusion-based classification techniques for 3D shapes. For our experiments, we utilized the ShapeNet dataset [3], focusing on a specific subset of 200 models per class. This selection was made due to the computational intensity of performing 1000 diffusion steps (t) per image (Xi), which is particularly challenging in environments with limited GPU resources (poor-gpus settings). Especially if each object is represented by 36 views taken from cameras view 10degree intervals around a circumference encircling the 3D object, adding to the computational complexity.

To classify a single object, our method proposes a majority vote scheme. Let X = {X1,X2,...,Xn} represent the set of n views of a 3D shape. Each view Xi is processed individually by the diffusion model to produce a corresponding prediction yi = f (Xi), where f(·) denotes the classification function of the diffusion model. Unlike MVCNN [40], which aggregates these views into a global representation through view pooling, our approach maintains the predictions {y1,y2,...,yn} independently.

The final classification decision y∗ is made by selecting the most common prediction among the individual predictions, formulated as:

###### y∗ = mode(y1,y2,...,yn)

This majority vote approach retains the architectural integrity of diffusion models while emphasizing simplicity and interpretability. While MVCNN’s view pooling may enhance performance by combining features, our goal is to assess the classification power of diffusion models in their unaltered form. Here the 2D Images are processed and encoded into feature maps i ∈ RH×W×C, where H is the height, W is the width, and C is the number of channels in the image (512 × 512 × 3 as the highest resolution in these experiments).

#### 3.2. Diffusion Classifier for 3D Objects (DC3DO)

DC3DO consists on the main contribution of our research. Our model combines LION [54] with diffusion classifier [18] for zero-shot classification. By utilizing LION’s ability to generate diverse 3D shapes and feeding them into the diffusion classifier, we achieve precise categorization of 3D cars and chairs. This section details DC3DO, emphasizing the advantages of diffusion models, the processing of 3D shapes, and the integration of the LION model to enhance classification accuracy.

#### 3.3. Integrating LION with Diffusion Classifiers

LION leverages a hierarchical latent space to effectively capture both global and local features of 3D structures, see Figure 2. This hierarchical approach ensures comprehensive encoding of both macro and micro features of 3D object structures.

##### 3.3.1 Integration with Hierarchical Latent Space

LION’s hierarchical latent space encodes 3D point clouds x ∈ R3×N, where x consists of N points (2048) with xyzcoordinates in R3, into a dual-layered latent representation. This representation includes:

• Global Latent Space: This vector-valued latent space, denoted as z0 ∈ RD

z, captures the overall structure and large-scale features of the building. It captures the overall

[Figure 2]

Figure 2. Methods comparison. We extended the Diffusion Classifier [18] paper to a multi-view [40] settings and we compare with our DC3DO model, based on [54]

spatial structure of the 3D shape, including its large-scale features.

##### • Local Point-Structured Latent Space: This latent space, denoted as h0 ∈ R(3+D

h)×N, represents a point cloud-structured latent consisting of N points with xyzcoordinates in R3 and additional Dh latent features per point. This layer captures detailed and fine-grained features.

- 3.3.2 Integration with Diffusion Models The integration process involves several key steps:

Encoding The 3D point cloud data x is encoded into the global latent space using LION’s PVCNN encoder. We found that the global latents contained enough information about the shape and high level features of the object, for the purpose of classification. Also, it is a much smaller latent space compared to the the local point structured latent space, making it easier to work with - considering the diffusion process requires multiple inference steps per sample.

Diffusion Process After encoding, the data undergoes a first diffusion process (global latent). This involves a fixed forward procedure where Gaussian noise is iteratively added (1000 steps) to the latent representations z0 and h0,

resulting in the diffused latents zt and ht. The forward diffusion process is defined as:

zt = αtz0 + σtϵ, ϵ ∼ N(0,I) (1)

ht = αth0 + σtϵ, ϵ ∼ N(0,I) (2) where:

- • z0 and h0 are the initial latent representations capturing the global and local features of the 3D shape, respectively.
- • zt and ht are the diffused latent representations at timestep t.
- • αt and σt are coefficients that control the amount of the original signal and the noise added at each timestep, respectively.
- • ϵ is the Gaussian noise sampled from a standard normal distribution N(0,I), which introduces randomness to the latent representations.

Denoising and Classification In the pipeline, a deep neural network, conditioned on class labels c, performs the denoising of the perturbed data. The denoising process involves reversing the forward diffusion to retrieve the latent representations zˆ0 and hˆ0 that best match the original data distribution x. Specifically, the network learns

to approximate the posterior distributions qϕ(z0|x,c) and qϕ(h0|x,z0,c) by minimizing the reconstruction error.

The classification is then performed by evaluating the likelihood pθ(x0 | c) of the denoised data zˆ0 and hˆ0 belonging to specific classes. This likelihood is computed as follows:

pθ(x0 | c) =

T

p(xT)

x1:T

t=1

pθ(xt−1 | xt,c) dx1:T (3)

where

- • pθ(x0 | c) is the class-conditional likelihood of the original data x0 given the class label c.
- • xT represents the final diffused state, typically modeled as a standard Gaussian distribution.
- • pθ(xt−1 | xt,c) denotes the learned reverse process that denoises the data at each timestep t, conditioned on the class label c.

The model assesses the denoised data to determine the most likely building category by evaluating which classspecific denoising process best corresponds to the introduced noise, ultimately assigning the data point to the class with the highest likelihood.

- 3.3.3 Text-Conditioned Diffusion

Our model employs multi-modal [49] approach to integrate diverse data modalities, providing a comprehensive approach to 3D building classification. The diffusion process is condition on a text prompt ”[C]” (”car”, ”chair”, and ”airplane” ). We added an additional text prompt to provide a diffusion process to classify the model as ”noncar”, ”non-chair”, and ”non-airplane”.

The integrated modalities include:

- • 3D Point Cloud Data: The primary representation of 3D shapes, capturing spatial distribution and structural details.
- • Textual Descriptions: Supplementary information describing the architectural features and styles of buildings. These are encoded into vector representations t ∈ Rd.

By integrating these modalities, our model achieves a more informed representation of the data, which improves classification accuracy and robustness. The integration of LION with diffusion models utilizes the combined strengths of both techniques, allowing for precise and reliable classification of 3D building structures.

- 4. Experimental Results

#### 4.1. MVDC - 2D Results

In our baseline evaluation, we utilized a multi-view diffusion classifier on the ShapeNet dataset, focusing on three classes: cars, chairs, and airplanes. This approach uses multiple views of 3D shapes to enhance classification accuracy

by taking advantage of the rich spatial information in the dataset. The process involved encoding 3D shapes into latent representations using a pretrained VAE, adding Gaussian noise, and employing a UNet model for denoising and classification. This way, we captured the intricate details of 3D shapes and effectively categorized them by adaptively selecting the most promising samples based on predicted errors, optimizing overall classification performance.

Table 1. Zero-shot classification accuracy (%) of DC3DO on ShapeNet for cars, airplanes, and chairs. We performed the comparison only on the first 200 subsamples models, each with 6 views and 200 sampling steps. For the Multi View Diffusion we used only the six frontal views, and image resolutions of 64 × 64.

Method Accuracy Car Chair

MVDC (100 models) 65.7% 32.3% MVDC (200 models) 64.8% 31.5%

DC3DO-100m (ours) 100% 36% DC3DO-200m (ours) 100% 49%

MVCNN [40] utilized 36 fixed cameras, with objects placed in a canonical pose. The cameras were positioned at uniform intervals, with a 10-degree rotation between each, defined by their position parameters (X,Y,Z). To manage computational constraints, we downscaled the images to 64×64 and reduced the number of views per object from 36 to 6. The selected views were the frontal ones, corresponding to camera angles of 10◦,20◦,30◦,340◦,350◦, and 360◦, in line with the ShapeNet view settings.

As explained in [37], frontal camera positions generally yield higher accuracy. Therefore, we focused on these 6 specific camera positions for our experiments.

To compute the accuracy of multi-view classification, we employed a majority vote mechanism across the n views of each 3D mesh, where n = 6. Let yi ∈ {0,1} represent the binary prediction for each view Xi, where 1 corresponds to the prediction “car” and 0 corresponds to “not car”. The final classification y∗ for each object is determined as:

y∗ =

1 if ni=1 yi ≥ n2 0 otherwise.

In this setup, if the number of votes for ”car” (i.e., n i=1 yi ) is greater than or equal to 3, the object is classified as ”car.” For each class c, we performed a binary classification, distinguishing between class c and all other classes. The accuracy Ac for each class c is calculated as:

Number of correctly classified objects in class c Total number of objects in class c

Ac =

.

Finally, the mean per-class accuracy A¯ is computed by averaging the binary classification accuracies across all classes:

C

1 C

A¯ =

Ac,

c=1

where C is the total number of classes.

#### 4.2. DC3DO Inference

Since DC3DO is based on LION, we took its model weights publicly available. LION has been trained on specific classes from the ShapeNet dataset, with the model weights for the “chairs” and “cars” categories publicly available to the research community. Consequently, we utilized these pretrained models in our experiments. Due to computational constraints, we set the number of diffusion steps for both the Multi-View Diffusion Classifier and LION to 200 steps.

For our experiments, we employed a batch size of 1 for the “cars” and “chairs” categories. Currently, it takes approximately 20 seconds to classify each object.

|Model<br><br>|Render<br><br>|Accuracy|
|---|---|---|
|Chair 142<br><br>Chair 143<br><br><br>Chair 188<br><br>|[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>|35%<br><br>34%<br><br>33%|

- Table 2. Best 3 chairs per accuracy

|Model<br><br>|Render|Accuracy|
|---|---|---|
|Chair 116<br><br>Chair 70<br><br>Chair 87<br><br>|[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]|20%<br><br>20%<br><br>20%|

- Table 3. Worst 3 chairs per accuracy

#### 4.3. Ablation Studies

To gain deeper insights into the contributions of different components in our model, we conducted ablation studies by systematically disabling specific features and quantifying their impact on performance.

For the MVDC model, let S × S denote the image size and n the number of views. We observed that the diffusion process time T increases non-linearly with both S and n. Specifically, T ∝ S2 × n, indicating that larger image sizes S and an increased number of views n result in significantly slower processing.

We ran inference at various image sizes to study its run time performance and relationship to the model performance. First, we confirmed that the inference time grows exponentially with larger image size, for a 512 × 512 resolution image and 500 sampling steps, the processing time was approximately 1.5 minutes per image, making it infeasible to evaluate at larger scale. Moreover, when we reduced the image size to S = 64×64 or S = 128×128, the classifier’s performance degraded severely. The model exhibited a tendency to collapse, consistently predicting a single class c regardless of the input views, suggesting that the classifier lost its ability to differentiate between classes under reduced image resolutions.

Table 4. Ablation studies about image resolutions. Inference time and accuracy analysis of MVDC model on 3 classes from shapenet: Airplane, Car, and Chair. The sample size is fixed at 200 steps.

Image Inference Accuracy Resolution Time Airplane Car Chair 64 × 64 1h03m 66.7% 64.8% 31.5% 128 × 128 2h13m 33.7% 66.7% 67.0% 256 × 256 7h05m 99.3% 98.7% 99.%

### 5. Limitations

One of the primary limitations of our approach is the computational cost. The 3D diffusion process currently requires approximately 20 minutes per object on a T4 GPU, making it a time-intensive task. Similarly, the multi-view approach, while effective, is also relatively slow due to the independent processing of each view.

Regarding the Multi-View Diffusion Classifier, a significant limitation is that the views are processed individually and then aggregated through a majority vote, rather than being combined into a global latent vector as in the approach used by MVCNN [40]. This method of independent view processing may not fully capture the holistic structure of 3D

[Figure 9]

[Figure 10]

[Figure 11]

(a) Car #184 Acc: 30%

(b) Car #39 Acc: 28%

(c) Car #142 Acc: 27%

[Figure 12]

[Figure 13]

[Figure 14]

(d) Car #149 Acc: 19%

(e) Car #152 Acc: 20%

(f) Car #60 Acc: 19%

Figure 3. Visual Comparison. comparison of best(a-c) and worst(d-f) performing cars with accuracy percentages

shapes, which could be better represented through a more integrated multi-view approach.

Due to time and computational constraints, we limited our experiments to 200 shapes per category. With access to more powerful GPUs and additional resources, future work could extend these experiments to a larger number of objects, potentially providing more comprehensive results.

### 6. Discussion and Future Work

The high classification accuracy on ID data indicates that the model effectively captures the distinguishing features of various 3D objects.

The hierarchical latent space of LION played a crucial role in accurately representing both global and local features of 3D shapes, contributing to the model’s overall performance compared the multi-view (see Section 5 for more details). The diffusion process further enhanced the model’s ability to denoise and classify complex 3D structures, providing a reliable mechanism for zero-shot classification.

These results highlight the potential of integrating generative models like LION with diffusion classifiers for advanced 3D shape analysis and classification tasks, particularly in scenarios involving diverse and unseen data. In fact, in this work, we delved into 3D diffusion models and present our method that enables zero-shot classification of 3D shapes in a robust manner. For future works, we wish to explore 3D diffusion capabilities in state-of-the-art multimodal methods such as ULIP-2 [50], integrated with PointBERT [52] architectures similar to the concurrent work [37]. We believe this will enhance the performance of these architectures and make them capable of 3D understanding.

### 7. Conclusion

In this paper, we propose a model that seamlessly integrates LION [54] with a diffusion classifier [18] to achieve accurate classification of 3D cars and chairs. The model’s success is driven by the hierarchical latent space and diffusion process, which together enable precise representation and classification of complex 3D shapes from the ShapeNet dataset [3]. Our approach, named DC3DO, demonstrates a 12.5% improvement on average compared to multi-view methods, highlighting the potential of generative models in 3D object classification. This work suggests that future research could adapt generative models to discriminative tasks, potentially leading to enhanced classification and regression performance.

### 8. Acknowledgments

We express our gratitude to Alexander C. Li and Le Xue for their insightful initial discussions. We also extend our thanks to the mentors and volunteers of the Summer Geometric Initiative (SGI) at MIT for their invaluable guidance. A special thank you to Professor Justin Solomon for his coordination and funding support. We are also grateful to Google Cloud for providing credits and sponsorship, and to Paul Guerrero for generously sharing the render file used in the previous SGI.

### References

- [1] Abdalla G. M. Ahmed, Jing Ren, and Peter Wonka. Gaussian blue noise. ACM Trans. Graph., 41(6), 2022. 2
- [2] Eric Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and Gordon Wetzstein. pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis. 2020. 1
- [3] Angel X. Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Mano-

- lis Savva, Shuran Song, Hao Su, Jianxiong Xiao, Li Yi, and Fisher Yu. Shapenet: An information-rich 3d model repository. arXiv pre-print, 2015. 3, 7
- [4] Christopher B Choy, Danfei Xu, JunYoung Gwak, Kevin Chen, and Silvio Savarese. 3d-r2n2: A unified approach for single and multi-view 3d object reconstruction. Proceedings of the European Conference on Computer Vision (ECCV),

2016. 2

- [5] Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis. CoRR, abs/2105.05233, 2021. 1
- [6] Ziyu Guo, Renrui Zhang, Xiangyang Zhu, Yiwen Tang, Xianzheng Ma, Jiaming Han, Kexin Chen, Peng Gao, Xianzhi Li, Hongsheng Li, and Pheng-Ann Heng. Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following, 2023. 2
- [7] Xizewen Han, Huangjie Zheng, and Mingyuan Zhou. Card: Classification and regression diffusion models, 2022. 1
- [8] Geoffrey E. Hinton. To recognize shapes, first learn to generate images. In Computational Neuroscience: Theoretical Insights into Brain Function, pages 535–547. Elsevier, 2007. 1
- [9] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. 1, 2
- [10] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models, 2022. 1
- [11] Tianyu Huang, Yihan Zeng, Zhilu Zhang, Wan Xu, Hang Xu, Songcen Xu, Rynson WH Lau, and Wangmeng Zuo. Dreamcontrol: Control-based text-to-3d generation with 3d self-prior. arXiv preprint arXiv:2312.06439, 2023. 1
- [12] Tao Huang, Jiaqi Liu, Shan You, and Chang Xu. Active generation for image classification, 2024. 1
- [13] Ali Jahanian, Lucy Chai, and Phillip Isola. On the ”steerability” of generative adversarial networks, 2020. 2
- [14] Ajay Jain, Ben Mildenhall, Jonathan T. Barron, Pieter Abbeel, and Ben Poole. Zero-shot text-guided object generation with dream fields, 2021. 2
- [15] Ajay Jain, Ben Mildenhall, Jonathan T. Barron, Pieter Abbeel, and Ben Poole. Zero-shot text-guided object generation with dream fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages pp.867–876, 2022. 2
- [16] Jiayi Ji, Haowei Wang, Changli Wu, Yiwei Ma, Xiaoshuai Sun, and Rongrong Ji. Jm3d & jm3d-llm: Elevating 3d understanding with joint multi-modal cues, 2024. 2
- [17] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. International Conference on Learning Representations (ICLR), 2014. 1
- [18] Alexander C. Li, Mihir Prabhudesai, Shivam Duggal, Ellis Brown, and Deepak Pathak. Your diffusion model is secretly

- a zero-shot classifier, 2023. 1, 2, 3, 4, 7

[19] Alexander C. Li, Mihir Prabhudesai, Shivam Duggal, Ellis Brown, and Deepak Pathak. Your diffusion model is secretly

- a zero-shot classifier, 2023. 2

- [20] Dongyun Lin, Yi Cheng, Shangbo Mao, Aiyuan Guo, and Yiqun Li. Peva-net: Prompt-enhanced view aggregation network for zero/few-shot multi-view 3d shape recognition,

2024. 2

- [21] Dingning Liu, Xiaomeng Dong, Renrui Zhang, Xu Luo, Peng Gao, Xiaoshui Huang, Yongshun Gong, and Zhihui Wang. 3daxiesprompts: Unleashing the 3d spatial task capabilities of gpt-4v, 2023. 2
- [22] Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Jiayuan Gu, and Hao Su. One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion. arXiv preprint arXiv:2311.07885, 2023. 2
- [23] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-2-3-45: Any single image to 3d mesh in 45 seconds without pershape optimization, 2023.
- [24] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object, 2023. 2
- [25] Zhijian Liu, Haotian Tang, Yujun Lin, and Song Han. Pointvoxel cnn for efficient 3d deep learning. In Conference on Neural Information Processing Systems (NeurIPS), 2019. 3
- [26] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution, 2024. 1
- [27] Shitong Luo and Wei Hu. Diffusion probabilistic models for 3d point cloud generation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 1, 2
- [28] Chenlin Meng, Kristy Choi, Jiaming Song, and Stefano Ermon. Concrete score matching: Generalized score matching for discrete data, 2023. 1
- [29] Gimin Nam, Mariem Khlifi, Andrew Rodriguez, Alberto Tono, Linqi Zhou, and Paul Guerrero. 3d-ldm: Neural implicit 3d shape generation with latent diffusion models. arXiv pre-print, 2022. 2
- [30] Ryan Po, Wang Yifan, and Vladislav Golyanik et al. Compositional 3d scene generation using locally conditioned diffusion. In ArXiv, 2023. 1
- [31] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion, 2022. 1
- [32] Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, and Supasorn Suwajanakorn. Diffusion autoencoders: Toward a meaningful and decodable representation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages pp.10619– 10629, 2022. 1
- [33] Zekun Qi, Runpei Dong, Shaochen Zhang, Haoran Geng, Chunrui Han, Zheng Ge, Li Yi, and Kaisheng Ma. Shapellm: Universal 3d object understanding for embodied interaction,

2024. 2

- [34] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents, 2022. 1
- [35] Aditya Sanghi, Pradeep Kumar Jayaraman, Arianna Rampini, Joseph Lambourne, Hooman Shayani, Evan Ather-

- ton, and Saeid Asgari Taghanaki. Sketch-a-shape: Zero-shot sketch-to-3d shape generation, 2023. 2
- [36] Oliver Selfridge. Pandemonium: A paradigm for learning.

1958. 1

- [37] Sitian Shen, Zilin Zhu, Linqian Fan, Harry Zhang, and Xinxiao Wu. Diffclip: Leveraging stable diffusion for language grounded 3d classification. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 3596–3605, 2024. 1, 2, 3, 5, 7
- [38] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. CoRR, abs/1503.03585, 2015. 1
- [39] Fedorova Stanislava, Tono Alberto, Nigam Meher Shashwat, Zhang Jiayao, Ahmadnia Amirhossein, Bolognesi Cecilia Maria, and L. Michels Dominik. Synthetic 3d data generation pipeline for geometric deep learning in architecture. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, XLIII-B2-2021: pp.337–344, 2021. 2
- [40] Hang Su, Subhransu Maji, Evangelos Kalogerakis, and Erik G. Learned-Miller. Multi-view convolutional neural networks for 3d shape recognition. In Proc. ICCV, 2015. 2, 3, 4, 5, 6
- [41] Thomas M. Sutter, Imant Daunhawer, and Julia E. Vogt. Generalized multimodal elbo, 2021. 2
- [42] Alberto Tono, Hannah Tono, and Andrea Zani. Encoded memory: Artificial intelligence and deep learning in architecture. Impact of Industry 4.0 on Architecture and Cultural Heritage, 2020. 2
- [43] Alberto Tono, Meher Shashwat Nigam, Amirhossein Ahmadnia, Stanislava Fedorova, and Cecilia Bolognesi. Limitations and review of geometric deep learning algorithms for monocular 3d reconstruction in architecture. Augmented reality and Artificial intelligence: Cultural Heritage and Innovative Design, 2021. 2
- [44] Alberto Tono, Heyaojing Huang, Ashwin Agrawal, and Martin Fischer. Vitruvio: Conditional variational autoencoder to generate building meshes via single perspective sketches. Automation in Construction, 166:105498, 2024. 2
- [45] Jiajun Wu, Chengkai Zhang, Tianfan Xue, William T. Freeman, and Joshua B. Tenenbaum. Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling. CoRR, abs/1610.07584, 2016. 1
- [46] Jiajun Wu, Yifan Wang, Tianfan Xue, Xingyuan Sun, William T Freeman, and Joshua B Tenenbaum. Marrnet: 3d shape reconstruction via 2.5d sketches. Advances in Neural Information Processing Systems (NeurIPS), 2017. 2
- [47] Jianwen Xie, Yifei Xu, Zilong Zheng, Song-Chun Zhu, and Ying Nian Wu. Generative pointnet: Deep energy-based learning on unordered point sets for 3d generation, reconstruction and classification, 2021. 1
- [48] Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Pointllm: Empowering large language models to understand point clouds, 2023. 2
- [49] Shicheng Xu. Multi-task 3d building understanding with multi-modal pretraining, 2023. 5

- [50] Le Xue, Ning Yu, Shu Zhang, Artemis Panagopoulou, Junnan Li, Roberto Mart´ın-Mart´ın, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. Ulip-2: Towards scalable multimodal pre-training for 3d understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 27091– 27101, 2024. 2, 7
- [51] Guandao Yang, Xun Huang, Zekun Hao, Ming-Yu Liu, Serge Belongie, and Bharath Hariharan. Pointflow: 3d point cloud generation with continuous normalizing flows, 2019. 1
- [52] Xumin Yu, Lulu Tang, Yongming Rao, Tiejun Huang, Jie Zhou, and Jiwen Lu. Point-bert: Pre-training 3d point cloud transformers with masked point modeling, 2021. 2, 7
- [53] Xin Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, SongHai Zhang, and Xiaojuan Qi. Text-to-3d with classifier score distillation. arXiv preprint arXiv:2310.19415, 2023. 3
- [54] Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, and Karsten Kreis. Lion: Latent point diffusion models for 3d shape generation, 2022. 2, 3, 4, 7
- [55] Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, and Karsten Kreis. Lion: Latent point diffusion models for 3d shape generation. Advances in Neural Information Processing Systems (NeurIPS), 2022. 2
- [56] Renrui Zhang, Liuhui Wang, Yu Qiao, Peng Gao, and Hongsheng Li. Learning 3d representations from 2d pre-trained models via image-to-point masked autoencoders, 2022. 2, 3
- [57] Zhihao Zhang, Shengcao Cao, and Yu-Xiong Wang. Tamm: Triadapter multi-modal learning for 3d shape understanding,

2024. 2

- [58] Xin-Yang Zheng, Hao Pan, Peng-Shuai Wang, Xin Tong, Yang Liu, and Heung-Yeung Shum. Locally attentional sdf diffusion for controllable 3d shape generation. ACM Transactions on Graphics (SIGGRAPH), 42(4), 2023. 2
- [59] Linqi Zhou, Yilun Du, and Jiajun Wu. 3d shape generation and completion through point-voxel diffusion. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. 2, 3

