## DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing

# arXiv:2306.14435v6[cs.CV]7Apr2024

Yujun Shi1 Chuhui Xue2 Jun Hao Liew2 Jiachun Pan1 Hanshu Yan2 Wenqing Zhang2 Vincent Y. F. Tan1 Song Bai2 1National University of Singapore 2 ByteDance Inc.

shi.yujun@u.nus.edu vtan@nus.edu.sg songbai.site@gmail.com

User Edit DragGAN DragDiffusion User Edit DragGAN DragDiffusion

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

Figure 1. DRAGDIFFUSION greatly improves the applicability of interactive point-based editing. Given an input image, the user clicks handle points (red), target points (blue), and draws a mask specifying the editable region (brighter area). All results are obtained under the same user edit for fair comparisons. Project page: https://yujun-shi.github.io/projects/dragdiffusion.html.

### Abstract

Accurate and controllable image editing is a challenging task that has attracted significant attention recently. Notably, DRAGGAN developed by Pan et al. (2023) [31] is an interactive point-based image editing framework that achieves impressive editing results with pixel-level precision. However, due to its reliance on generative adversarial networks (GANs), its generality is limited by the capacity of pretrained GAN models. In this work, we extend this editing framework to diffusion models and propose a novel approach DRAGDIFFUSION. By harnessing large-scale pretrained diffusion models, we greatly enhance the applicability of interactive point-based editing on both real and diffusion-generated images. Unlike other diffusion-based editing methods that provide guidance on diffusion latents of multiple time steps, our approach achieves efficient yet accurate spatial control by optimizing the latent of only one time step. This novel design is motivated by our observations that UNet features at a specific time step provides

sufficient semantic and geometric information to support the drag-based editing. Moreover, we introduce two additional techniques, namely identity-preserving fine-tuning and reference-latent-control, to further preserve the identity of the original image. Lastly, we present a challenging benchmark dataset called DRAGBENCH—the first benchmark to evaluate the performance of interactive point-based image editing methods. Experiments across a wide range of challenging cases (e.g., images with multiple objects, diverse object categories, various styles, etc.) demonstrate the versatility and generality of DRAGDIFFUSION. Code and the DRAGBENCH dataset: https://github.com/YujunShi/DragDiffusion

### 1. Introduction

Image editing with generative models [8, 14, 20, 29, 32, 35] has attracted extensive attention recently. One landmark work is DRAGGAN [31], which enables interactive point-

based image editing, i.e., drag-based editing. Under this framework, the user first clicks several pairs of handle and target points on an image. Then, the model performs semantically coherent editing on the image that moves the contents of the handle points to the corresponding target points. In addition, users can draw a mask to specify which region of the image is editable while the rest remains unchanged.

Despite DRAGGAN’s impressive editing results with pixel-level spatial control, the applicability of this method is being limited by the inherent model capacity of generative adversarial networks (GANs) [11, 18, 19]. On the other hand, although large-scale text-to-image diffusion models [36, 40] have demonstrated strong capabilities to synthesize high quality images across various domains, there are not many diffusion-based editing methods that can achieve precise spatial control. This is because most diffusion-based methods [14, 20, 29, 32] conduct editing by controlling the text embeddings, which restricts their applicability to editing high-level semantic contents or styles.

To bridge this gap, we propose DRAGDIFFUSION, the first interactive point-based image editing method with diffusion models [16, 36, 40, 43]. Empowered by large-scale pre-trained diffusion models [36, 40], DRAGDIFFUSION achieves accurate spatial control in image editing with significantly better generalizability (see Fig. 1).

Our approach focuses on optimizing diffusion latents to achieve drag-based editing, which is inspired by the fact that diffusion latents can accurately determine the spatial layout of the generated images [27]. In contrast to previ-

- ous methods [3, 9, 32, 50], which apply gradient descent on latents of multiple diffusion steps, our approach focuses on optimizing the latent of one appropriately selected step to conveniently achieve the desired editing results. This novel design is motivated by the empirical observations presented in Fig. 2. Specifically, given two frames from a video simulating the original and the “dragged” images, we visualize the UNet feature maps of different diffusion steps using principal component analysis (PCA). Via this visualization, we find that there exists a single diffusion step (e.g., t = 35 in this case) such that the UNet feature maps at this step alone contains sufficient semantic and geometric information to support structure-oriented spatial control such as drag-based editing. Besides optimizing the diffusion latents, we further introduce two additional techniques to enhance the identity preserving during the editing process, namely identity-preserving fine-tuning and referencelatent-control. An overview of our method is given in Fig. 3.

It would be ideal to immediately evaluate our method on well-established benchmark datasets. However, due to a lack of evaluation benchmarks for interactive point-based editing, it is difficult to rigorously study and corroborate the efficacy of our proposed approach. Therefore, to facilitate such evaluation, we present DRAGBENCH—the first

[Figure 13]

Figure 2. PCA visualization of UNet feature maps at different diffusion steps for two video frames. t = 50 implies the full DDIM inversion, while t = 0 implies the clean image. Notably, UNet features at one specific step (e.g., t = 35) provides sufficient semantic and geometric information (e.g., shape and pose of the cat, etc.) for the drag-based editing.

benchmark dataset for drag-based editing. DRAGBENCH is a diverse collection comprising images spanning various object categories, indoor and outdoor scenes, realistic and aesthetic styles, etc. Each image in our dataset is accompanied with a set of “drag” instructions, which consists of one or more pairs of handle and target points as well as a mask specifying the editable region.

Through extensive qualitative and quantitative experiments on a variety of examples (including those on DRAGBENCH), we demonstrate the versatility and generality of our approach. In addition, our empirical findings corroborate the crucial role played by identity-preserving finetuning and reference-latent-control. Furthermore, a comprehensive ablation study is conducted to meticulously explore the influence of key factors, including the number of inversion steps of the latent, the number of identitypreserving fine-tuning steps, and the UNet feature maps.

Our contributions are summarized as follows: 1) we present a novel image editing method DRAGDIFFUSION, the first to achieve interactive point-based editing with diffusion models; 2) we introduce DRAGBENCH, the first benchmark dataset to evaluate interactive point-based image editing methods; 3) Comprehensive qualitative and quantitative evaluation demonstrate the versatility and generality of our DRAGDIFFUSION.

### 2. Related Work

Generative Image Editing. Given the initial successes of generative adversarial networks (GANs) in image generation [11, 18, 19], many previous image editing methods have been based on the GAN paradigm [2, 8, 13, 23, 31, 33, 41, 42, 46, 52, 53]. However, due to the limited model capacity of GANs and the difficulty of inverting the real images into GAN latents [1, 7, 26, 35], the generality of these methods would inevitably be constrained.

[Figure 14]

- Figure 3. Overview of DRAGDIFFUSION. Our approach constitutes three steps: firstly, we conduct identity-preserving fine-tuning on the UNet of the diffusion model given the input image. Secondly, according to the user’s dragging instruction, we optimize the latent obtained from DDIM inversion on the input image. Thirdly, we apply DDIM denoising guided by our reference-latent-control on zˆt to obtain the final editing result zˆ0. Figure best viewed in color.

Recently, due to the impressive generation results from large-scale text-to-image diffusion models [36, 40], many diffusion-based image editing methods have been proposed [4–6, 14, 20, 24, 27, 28, 30, 32, 47]. Most of these methods aim to edit the images by manipulating the prompts of the image. However, as many editing attempts are difficult to convey through text, the prompt-based paradigm usually alters the image’s high-level semantics or styles, lacking the capability of achieving precise pixel-level spatial control. [9] is one of the early efforts in exploring better controllability on diffusion models beyond the prompt-based image editing. In our work, we aim at enabling a even more versatile paradigm than the one studied in [9] with diffusion models—interactive point-based image editing.

Point-based editing. To enable fine-grained editing, several works have been proposed to perform point-based editing, such as [8, 31, 49]. In particular, DRAGGAN has demonstrated impressive dragging-based manipulation with two simple ingredients: 1) optimization of latent codes to move the handle points towards their target locations and 2) a point tracking mechanism that keep tracks of the handle points. However, its generality is constrained due to the limited capacity of GAN. FreeDrag [25] propose to improve DRAGGAN by introducing a point-tracking-free paradigm. In this work, we extend the editing framework of DRAGGAN to diffusion models and showcase its generality over different domains. There is a work [30] concurrent to ours that also studies drag-based editing with diffusion models. Differently, they rely on classifier guidance to transforms the editing signal into gradients.

LoRA in Diffusion Models. Low Rank Adaptation (i.e., LoRA) [17] is a general technique to conduct parameter-

efficient fine-tuning on large and deep networks. During LoRA fine-tuning, the original weights of the model are frozen, while trainable rank decomposition matrices are injected into each layer. The core assumption of this strategy is that the model weights will primarily be adapted within a low rank subspace during fine-tuning. While LoRA was initially introduced for adapting language models to downstream tasks, recent efforts have illustrated its effectiveness when applied in conjunction with diffusion models [12, 39]. In this work, inspired by the promising results of using LoRA for image generation and editing [20, 38], we also implement our identity-preserving fine-tuning with LoRA.

### 3. Methodology

In this section, we formally present the proposed DRAGDIFFUSION approach. To commence, we introduce the preliminaries on diffusion models. Then, we elaborate on the three key stages of our approach as depicted in Fig. 3: 1) identity-preserving fine-tuning; 2) latent optimization according to the user-provided dragging instructions; 3) denoising the optimized latents guided by our reference-latent-control.

#### 3.1. Preliminaries on Diffusion Models

Denoising diffusion probabilistic models (DDPM) [16, 43] constitute a family of latent generative models. Concerning a data distribution q(z), DDPM approximates q(z) as the marginal pθ(z0) of the joint distribution between Z0 and a collection of latent random variables Z1:T. Specifically,

##### pθ(z0) = pθ(z0:T)dz1:T, (1)

where pθ(zT) is a standard normal distribution and the transition kernels pθ(zt−1|zt) of this Markov chain are all Gaussian conditioned on zt. In our context, Z0 corresponds to image samples given by users, and Zt corresponds to the latent after t steps of the diffusion process.

[36] proposed the latent diffusion model (LDM), which maps data into a lower-dimensional space via a variational auto-encoder (VAE) [22] and models the distribution of the latent embeddings instead. Based on the framework of LDM, several powerful pretrained diffusion models have been released publicly, including the Stable Diffusion (SD) model (https://huggingface.co/stabilityai). In SD, the network responsible for modeling pθ(zt−1|zt) is implemented as a UNet [37] that comprises multiple self-attention and cross-attention modules [48]. Applications in this paper are implemented based on the public Stable Diffusion model.

#### 3.2. Identity-preserving Fine-tuning

Before editing a real image, we first conduct identitypreserving fine-tuning [17] on the diffusion models’ UNet (see panel (1) of Fig. 3). This stage aims to ensure that the diffusion UNet encodes the features of input image more accurately (than in the absence of this procedure), thus facilitating the consistency of the identity of the image throughout the editing process. This fine-tuning process is implemented with LoRA [17]. More formally, the objective function of the LoRA is

Lft(z,∆θ) = Eϵ,t ∥ϵ − ϵθ+∆θ(αtz + σtϵ)∥22 , (2)

where θ and ∆θ represent the UNet and LoRA parameters respectively, z is the real image, ϵ ∼ N(0,I) is the randomly sampled noise map, ϵθ+∆θ(·) is the noise map predicted by the LoRA-integrated UNet, and αt and σt are parameters of the diffusion noise schedule at diffusion step t. The fine-tuning objective is optimized via gradient descent on ∆θ.

Remarkably, we find that fine-tuning LoRA for merely 80 steps proves sufficient for our approach, which is in stark contrast to the 1000 steps required by tasks such as subject-driven image generation [12, 38]. This ensures that our identity-preserving fine-tuning process is extremely efficient, and only takes around 25 seconds to complete on an A100 GPU. We posit this efficiency is because our approach operates on the inverted noisy latent, which inherently preserve some information about the input real image. Consequently, our approach does not require lengthy fine-tuning to preserve the identity of the original image.

#### 3.3. Diffusion Latent Optimization

After identity-preserving fine-tuning, we optimize the diffusion latent according to the user instruction (i.e., the handle and target points, and optionally a mask specifying the ed-

itable region) to achieve the desired interactive point-based editing (see panel (2) of Fig. 3).

To commence, we first apply a DDIM inversion [44] on the given real image to obtain a diffusion latent at a certain step t (i.e., zt). This diffusion latent serves as the initial value for our latent optimization process. Then, following along the similar spirit of [31], the latent optimization process consists of two steps to be implemented consecutively. These two steps, namely motion supervision and point tracking, are executed repeatedly until either all handle points have moved to the targets or the maximum number of iterations has been reached. Next, we describe these two steps in detail.

Motion Supervision: We denote the n handle points at

the k-th motion supervision iteration as {hki = (xki ,yik) : i = 1,...,n} and their corresponding target points as

{gi = (˜xi,y˜i) : i = 1,...,n}. The input image is denoted as z0; the t-th step latent (i.e., result of t-th step DDIM inversion) is denoted as zt. We denote the UNet output feature maps used for motion supervision as F(zt), and the feature vector at pixel location hki as Fhk

(zt). Also, we denote the square patch centered around hki as Ω(hki ,r1) = {(x,y) : |x−xki | ≤ r1,|y−yik| ≤ r1}. Then, the motion supervision loss at the k-th iteration is defined as:

i

n

Lms(ˆztk) =

(ˆztk) − sg(Fq(ˆztk)) 1

Fq+d

i

i=1 q∈Ω(hki ,r1)

+ λ (ˆztk−1 − sg(ˆzt0−1)) ⊙ ( −M) 1 , (3)

where zˆtk is the t-th step latent after the k-th update, sg(·) is the stop gradient operator (i.e., the gradient will not be back-

propagated for the term sg(Fq(ˆztk))), di = (gi − hki )/∥gi − hki ∥2 is the normalized vector pointing from hki to gi, M is the binary mask specified by the user, Fq+d

(ˆztk) is obtained via bilinear interpolation as the elements of q + di may not be integers. In each iteration, zˆtk is updated by taking one gradient descent step to minimize Lms:

i

∂Lms(ˆztk) ∂zˆtk

zˆtk+1 = zˆtk − η ·

, (4)

where η is the learning rate for latent optimization.

Note that for the second term in Eqn. (3), which encourages the unmasked area to remain unchanged, we are working with the diffusion latent instead of the UNet features. Specifically, given zˆtk, we first apply one step of DDIM denoising to obtain zˆtk−1, then we regularize the unmasked region of zˆtk−1 to be the same as zˆt0−1 (i.e., zt−1).

Point Tracking: Since the motion supervision updates zˆtk, the positions of the handle points may also change. Therefore, we need to perform point tracking to update the handle points after each motion supervision step. To achieve this goal, we use UNet feature maps F(ˆztk+1) and

F(zt) to track the new handle points. Specifically, we update each of the handle points hki with a nearest neighbor search within the square patch Ω(hki ,r2) = {(x,y) : |x − xki | ≤ r2,|y − yik| ≤ r2} as follows:

hki+1 = arg min

q∈Ω(hki ,r2)

Fq(ˆztk+1) − Fh0

(zt)

i

. (5)

1

#### 3.4. Reference-latent-control

After we have completed the optimization of the diffusion latents, we then denoise the optimized latents to obtain the final editing results. However, we find that na¨ıvely applying DDIM denoising on the optimized latents still occasionally leads to undesired identity shift or degradation in quality comparing to the original image. We posit that this issue arises due to the absence of adequate guidance from the original image during the denoising process.

To mitigate this problem, we draw inspiration from [6] and propose to leverage the property of self-attention modules to steer the denoising process, thereby boosting coherence between the original image and the editing results. In particular, as illustrated in panel (3) of Fig. 3, given the denoising process of both the original latent zt and the optimized latent zˆt, we use the process of zt to guide the process of zˆt. More specifically, during the forward propagation of the UNet’s self-attention modules in the denoising process, we replace the key and value vectors generated from zˆt with the ones generated from zt. With this simple replacement technique, the query vectors generated from zˆt will be directed to query the correlated contents and texture of zt. This leads to the denoising results of zˆt (i.e., zˆ0) being more coherent with the denoising results of zt (i.e., z0). In this way, reference-latent-control substantially improves the consistency between the original and the edited images.

### 4. Experiments

#### 4.1. Implementation Details

In all our experiments, unless stated otherwise, we adopt the Stable Diffusion 1.5 [36] as our diffusion model. During the identity-preserving fine-tuning, we inject LoRA into the projection matrices of query, key and value in all of the attention modules. We set the rank of the LoRA to 16. We fine-tune the LoRA using the AdamW [21] optimizer with a learning rate of 5 × 10−4 and a batch size of 4 for 80 steps.

During the latent optimization stage, we schedule 50 steps for DDIM and optimize the diffusion latent at the 35th step unless specified otherwise. When editing real images, we do not apply classifier-free guidance (CFG) [15] in both DDIM inversion and DDIM denoising process. This is because CFG tends to amplify numerical errors, which is not ideal in performing the DDIM inversion [29]. We use the Adam optimizer with a learning rate of 0.01 to optimize

the latent. The maximum optimization step is set to be 80. The hyperparameter r1 in Eqn. 3 and r2 in Eqn. 5 are tuned to be 1 and 3, respectively. λ in Eqn. 3 is set to 0.1 by default, but the user may increase λ if the unmasked region has changed to be more than what was desired.

Finally, we apply our reference-latent-control in the upsampling blocks of the diffusion UNet at all denoising steps when generating the editing results. The execution time for each component is detailed in Appendix D.

#### 4.2. DRAGBENCH and Evaluation Metrics

Since interactive point-based image editing is a recently introduced paradigm, there is an absence of dedicated evaluation benchmarks for this task, making it challenging to comprehensively study the effectiveness of our proposed approach. To address the need for systematic evaluation, we introduce DRAGBENCH, the first benchmark dataset tailored for drag-based editing. DRAGBENCH is a diverse compilation encompassing various types of images. Details and examples of our dataset are given in Appendix A. Each image within our dataset is accompanied by a set of dragging instructions, comprising one or more pairs of handle and target points, along with a mask indicating the editable region. We hope future research on this task can benefit from DRAGBENCH.

In this work, we utilize the following two metrics for quantitative evaluation: Image Fidelity (IF) [20] and Mean Distance (MD) [31]. IF, the first metric, quantifies the similarity between the original and edited images. It is calculated by subtracting the mean LPIPS [51] over all pairs of original and edited images from 1. The second metric MD assesses how well the approach moves the semantic contents to the target points. To compute the MD, we first employ DIFT [45] to identify points in the edited images corresponding to the handle points in the original image. These identified points are considered to be the final handle points post-editing. MD is subsequently computed as the mean Euclidean distance between positions of all target points and their corresponding final handle points. MD is averaged over all pairs of handle and target points in the dataset. An optimal “drag” approach ideally achieves both a low MD (indicating effective “dragging”) and a high IF (reflecting robust identity preservation).

#### 4.3. Qualitative Evaluation

In this section, we first compare our approach with DRAGGAN on real images. We employ SD-1.5 for our approach when editing real images. All input images and the user edit instructions are from our DRAGBENCH dataset. Results are given in Fig. 4. As illustrated in the figure, when dealing with the real images from a variety of domains, DRAGGAN often struggles due to GAN models’ limited capacity. On the other hand, our DRAGDIFFUSION can convincingly de-

[Figure 15]

###### Figure 4. Comparisons between DRAGGAN and DRAGDIFFUSION. All results are obtained under the same user edit for fair comparisons.

[Figure 16]

###### Figure 5. Editing results on diffusion-generated images with (a) Stable-Diffusion-1.5, (b) Counterfeit-V2.5, (c) Majicmix Realistic, (d) Interior Design Supermix.

[Figure 17]

Figure 6. Ablating the number of inversion step t. Effective results are obtained when t ∈ [30, 40].

liver reasonable editing results. More importantly, besides achieving the similar pose manipulation and local deformation as in DRAGGAN [31], our approach even enables more types of editing such as content filling. An example is given in Fig. 4 (a), where we fill the grassland with the pool using drag-based editing. This further validates the enhanced versatility of our approach. More qualitative comparisons are provided in Appendix F.

Next, to show the generality of our approach, we perform drag-based editing on diffusion-generated images across a spectrum of variants of SD-1.5, including SD-1.5 itself, Counterfeit-V2.5, Majicmix Realistic, Interior Design Supermix. Results are shown in Fig. 5 These results validate our approach’s ability to smoothly work with various pretrained diffusion models. Moreover, these results also illustrate our approach’s ability to deal with drag instructions of different magnitudes (e.g., small magnitude edits such as the left-most image in Fig. 5 (d) and large magnitude edits such as the left-most image in Fig. 5 (c)). Additional results with more diffusion models and different resolutions can be found in Appendix F.

#### 4.4. Quantitative Analysis

In this section, we conduct a rigorous quantitative evaluation to assess the performance of our approach. We begin by comparing DRAGDIFFUSION with the baseline method DRAGGAN. As each StyleGAN [19] model utilized in [31] is specifically designed for a particular image class, we employ an ensemble strategy to evaluate DRAGGAN. This strategy involves assigning a text description to characterize the images generated by each StyleGAN model. Before editing each image, we compute the CLIP similarity [34] between the image and each of the text descriptions associated with the GAN models. The GAN model that yields the highest CLIP similarity is selected for the editing task.

Furthermore, to validate the effectiveness of each component of our approach, we evaluate DRAGDIFFUSION in the following two configurations: one without identitypreserving fine-tuning and the other without referencelatent-control. We perform our empirical studies on the DRAGBENCH dataset, and Image Fidelity (IF) and Mean Distance (MD) of each configuration mentioned above are reported in Fig. 8. All results are averaged over the DRAGBENCH dataset. In this figure, the x-axis represents MD

and the y-axis represents IF, which indicates the method with better results should locate at the upper-left corner of the coordinate plane. The results presented in this figure clearly demonstrate that our DRAGDIFFUSION significantly outperforms the DRAGGAN baseline in terms of both IF and MD. Furthermore, we observe that DRAGDIFFUSION without identity-preserving fine-tuning experiences a catastrophic increase in MD, whereas DRAGDIFFUSION without reference-latent control primarily encounters a decrease in IF. Visualization on the effects of identity-preserving fine-tuning and reference-latent-control are given in Fig. 9, which corroborates with our quantitative results.

#### 4.5. Ablation on the Number of Inversion Step

Next, we conduct an ablation study to elucidate the impact of varying t (i.e., the number of inversion steps) during the latent optimization stage of DRAGDIFFUSION. We set t to be t = 10,20,30,40,50 steps and run our approach on DRAGBENCH to obtain the editing results (t = 50 corresponds to the pure noisy latent). We evaluate Image Fidelity (IF) and Mean Distance (MD) for each t value in Fig. 7(a). All metrics are averaged over the DRAGBENCH dataset.

In terms of the IF, we observe a monotonic decrease as t increases. This trend can be attributed to the stronger flexibility of the diffusion latent as more steps are inverted. As for MD, it initially decreases and then increases with higher t values. This behavior highlights the presence of a critical range of t values for effective editing (t ∈ [30,40] in our figure). When t is too small, the diffusion latent lacks the necessary flexibility for substantial changes, posing challenges in performing reasonable edits. Conversely, overly large t values result in a diffusion latent that is unstable for editing, leading to difficulties in preserving the original image’s identity. Given these results, we chose t = 35 as our default setting, as it achieves the lowest MD while maintaining a decent IF. Qualitative visualization that corroborates with our numerical evaluation is provided in Fig. 6.

#### 4.6. Ablation Study on the Number of Identitypreserving Fine-tuning Steps

We run our approach on DRAGBENCH under 0, 20, 40, 60, 80, and 100 identity-preserving fine-tuning steps, respectively (0 being no fine-tuning). The outcomes are assessed using IF and MD, and the results are presented in Fig.7 (b).

(a) (b) (c)

- Figure 7. Ablation study on (a) the number of inversion step t of the diffusion latent; (b) the number of identity-preserving fine-tuning steps; (c) Block No. of UNet feature maps. Mean Distance (↓) and Image Fidelity (↑) are reported. Results are produced on DRAGBENCH.

35 40 45 50 55 Mean Distance

0.70

0.75

0.80

0.85

ImageFidelity(1-LPIPS)

DragGAN

DragDiffusion w/o fine-tune

DragDiffusion w/o ref-latent-control

DragDiffusion

- Figure 8. Quantitative analysis on DRAGGAN, DRAGDIFFUSION and DRAGDIFFUSION’s variants without certain components. Image Fidelity (↑) and Mean Distance (↓) are reported. Results are produced on DRAGBENCH. The approach with better results should locate at the upper-left corner of the coordinate plane.

[Figure 18]

- Figure 9. Qualitative validation on effectiveness of identitypreserving fine-tuning and reference-latent-control.

This phenomenon shows that lengthy fine-tuning of LoRA would no longer significantly improve the performance of our approach. Considering the experimental results, we conduct identity-preserving fine-tuning for 80 steps by default to balance between effectiveness and efficiency. Visualizations that corroborate our quantitative evaluation are presented in the Appendix G.

#### 4.7. Ablation Study on the UNet Feature Maps

Finally, we study the effect of using different blocks of UNet feature maps to supervise our latent optimization. We run our approach on the DRAGBENCH dataset with the feature maps output by 4 different upsampling blocks of the UNet Decoder, respectively. The outcomes are assessed with IF and MD, and are shown in Fig. 7(c). As can be seen, with deeper blocks of UNet features, IF consistently increases, while MD first decreases and then increases. This trend is because feature maps of lower blocks contain coarser semantic information, while higher blocks contain lower level texture information [10, 47]. Hence, the feature maps of lower blocks (e.g., block No. of 1) lack finegrained information for accurate spatial control, whereas those of higher blocks (e.g., block No. of 4) lack semantic and geometric information to drive the drag-based editing. Our results indicate that the feature maps produced by the third block of the UNet decoder demonstrate the best performance, exhibiting the lowest MD and a relatively high IF. Visualizations that corroborate our quantitative evaluation are presented in the Appendix H.

### 5. Conclusion and Future Works

In this work, we extend interactive point-based editing to large-scale pretrained diffusion models through the introduction of a novel method named DRAGDIFFUSION. Furthermore, we introduce the DRAGBENCH dataset, which aims to facilitate the evaluation of the interactive pointbased editing methods. Comprehensive qualitative and quantitative results showcases the remarkable versatility and generality of our proposed method. Limitations of our

All results are averaged over the DRAGBENCH dataset.

Initially, as the number of fine-tuning steps increases, MD exhibits a steep downward trend while IF shows an upward trend. This reflects that identity-preserving finetuning can drastically boost both the precision and consistency of drag-based editing. However, as the fine-tuning progresses, both MD and IF subsequently begins to plateau.

approach are further discussed in Appendix E, and we leave making the drag-based editing more robust and reliable on diffusion models as our future work.

### References

- [1] Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2stylegan: How to embed images into the stylegan latent space? In Proceedings of the IEEE/CVF international conference on computer vision, pages 4432–4441, 2019. 2
- [2] Rameen Abdal, Peihao Zhu, Niloy J Mitra, and Peter Wonka. Styleflow: Attribute-conditioned exploration of stylegangenerated images using conditional continuous normalizing flows. ACM Transactions on Graphics (ToG), 40(3):1–21,

2021. 2

- [3] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 843–852,

2023. 2

- [4] Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. Text2live: Text-driven layered image and video editing. In European Conference on Computer Vision, pages 707–723. Springer, 2022. 3
- [5] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023.
- [6] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. arXiv preprint arXiv:2304.08465, 2023. 3, 5
- [7] Antonia Creswell and Anil Anthony Bharath. Inverting the generator of a generative adversarial network. IEEE transactions on neural networks and learning systems, 30(7):1967– 1974, 2018. 2
- [8] Yuki Endo. User-controllable latent transformer for stylegan image layout editing. arXiv preprint arXiv:2208.12408,

- 2022. 1, 2, 3

[9] Dave Epstein, Allan Jabri, Ben Poole, Alexei A. Efros, and Aleksander Holynski. Diffusion self-guidance for controllable image generation. arXiv preprint arXiv:2306.00986,

- 2023. 2, 3

- [10] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arxiv:2307.10373, 2023. 8
- [11] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2014. 2
- [12] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized lowrank adaptation for multi-concept customization of diffusion models. arXiv preprint arXiv:2305.18292, 2023. 3, 4

- [13] Erik H¨ark¨onen, Aaron Hertzmann, Jaakko Lehtinen, and Sylvain Paris. Ganspace: Discovering interpretable gan controls. Advances in neural information processing systems, 33:9841–9850, 2020. 2
- [14] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 1, 2, 3
- [15] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 5
- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 2, 3
- [17] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR),

2022. 3, 4

- [18] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 2
- [19] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8110–8119, 2020. 2, 7
- [20] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023. 1, 2, 3, 5
- [21] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of the International Conference on Learning Representations (ICLR), 2015. 5
- [22] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 4
- [23] Thomas Leimk¨uhler and George Drettakis. Freestylegan: Free-view editable portrait rendering with the camera manifold. arXiv preprint arXiv:2109.09378, 2021. 2
- [24] Jun Hao Liew, Hanshu Yan, Daquan Zhou, and Jiashi Feng. Magicmix: Semantic mixing with diffusion models. arXiv preprint arXiv:2210.16056, 2022. 3
- [25] Pengyang Ling, Lin Chen, Pan Zhang, Huaian Chen, and Yi Jin. Freedrag: Point tracking is not you need for interactive point-based image editing. arXiv preprint arXiv:2307.04684, 2023. 3
- [26] Zachary C Lipton and Subarna Tripathi. Precise recovery of latent vectors from generative adversarial networks. arXiv preprint arXiv:1702.04782, 2017. 2
- [27] Jiafeng Mao, Xueting Wang, and Kiyoharu Aizawa. Guided image synthesis via initial image editing in diffusion model. arXiv preprint arXiv:2305.03382, 2023. 2, 3
- [28] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 3

- [29] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023. 1, 2, 5
- [30] Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. Dragondiffusion: Enabling drag-style manipulation on diffusion models. arXiv preprint arXiv:2307.02421,

2023. 3

- [31] Xingang Pan, Ayush Tewari, Thomas Leimk¨uhler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. Drag your GAN: Interactive point-based manipulation on the generative image manifold. arXiv preprint arXiv:2305.10973, 2023. 1, 2, 3, 4, 5, 7
- [32] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. arXiv preprint arXiv:2302.03027, 2023. 1, 2, 3
- [33] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Text-driven manipulation of stylegan imagery. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2085–2094,

2021. 2

- [34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 7
- [35] Daniel Roich, Ron Mokady, Amit H Bermano, and Daniel Cohen-Or. Pivotal tuning for latent-based editing of real images. ACM Transactions on Graphics (TOG), 42(1):1–13,

2022. 1, 2

- [36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 2, 3, 4, 5
- [37] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015. 4
- [38] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500– 22510, 2023. 3, 4
- [39] Simo Ryu. Low-rank adaptation for fast text-to-image diffusion fine-tuning. https : / / github . com / cloneofsimo/lora, 2022. 3
- [40] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2, 3

- [41] Yujun Shen and Bolei Zhou. Closed-form factorization of latent semantics in gans. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1532–1540, 2021. 2
- [42] Yujun Shen, Jinjin Gu, Xiaoou Tang, and Bolei Zhou. Interpreting the latent space of gans for semantic face editing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9243–9252, 2020. 2
- [43] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265. PMLR, 2015. 2, 3
- [44] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In Proceedings of the International Conference on Learning Representations (ICLR),

2021. 4

- [45] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. arXiv preprint arXiv:2306.03881,

2023. 5

- [46] Ayush Tewari, Mohamed Elgharib, Gaurav Bharaj, Florian Bernard, Hans-Peter Seidel, Patrick P´erez, Michael Zollhofer, and Christian Theobalt. Stylerig: Rigging stylegan for 3d control over portrait images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6142–6151, 2020. 2
- [47] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1921–1930, 2023. 3, 8
- [48] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 4
- [49] Sheng-Yu Wang, David Bau, and Jun-Yan Zhu. Rewriting geometric rules of a gan. ACM Transactions on Graphics (TOG), 41(4):1–16, 2022. 3
- [50] Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, and Jian Zhang. Freedom: Training-free energy-guided conditional diffusion model. arXiv preprint arXiv:2303.09833,

2023. 2

- [51] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 5
- [52] Jiapeng Zhu, Ceyuan Yang, Yujun Shen, Zifan Shi, Deli Zhao, and Qifeng Chen. Linkgan: Linking gan latents to pixels for controllable image synthesis. arXiv preprint arXiv:2301.04604, 2023. 2
- [53] Jun-Yan Zhu, Philipp Kr¨ahenb¨uhl, Eli Shechtman, and Alexei A Efros. Generative visual manipulation on the natural image manifold. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part V 14, pages 597–613. Springer, 2016. 2

[Figure 19]

Figure 10. Examples of our DRAGBENCH dataset. Each image is accompanied by a set of drag-based editing instruction.

### A. Details About DRAGBENCH Dataset

We have collected 205 images and provided 349 pairs of handle and target points in total. Images in our DRAGBENCH are classified into the following 10 categories: animals, art works, buildings (city view), buildings (countryside view), human (head), human (upper body), human (full body), interior design, landscape, other objects. All human-related images are selected from Midjourney generation results to avoid potential legal concerns. All the other images are real images downloaded from unsplash (https://unsplash.com/), pexels (https://www.pexels.com/zh-cn/), and pixabay (https://pixabay.com/). Some examples of our dataset is given in Fig. 10.

### B. Links to the Stable Diffusion’s Finetuned Variants Used by Us

Here, we provide links to the fine-tuned variants of Stable Diffusion used by us: Counterfeit-V2.5 (https://huggingface.co/gsdf/Counterfeit-V2.5), Majixmix Realistic (https://huggingface.co/emilianJR/majicMIX realistic), Realistic Vision (https://huggingface.co/SG161222/Realistic Vision V2.0), Interior Design Supermix (https://huggingface.co/stablediffusionapi/interiordesignsuperm), DVarch (https://huggingface.co/stablediffusionapi/dvarch).

[Figure 20]

Figure 11. Limitation of DRAGDIFFUSION. Occasionally, some of the handle points cannot precisely reach the desired target.

### C. More Details on Editing Diffusion-Generated Images

Here we introduce more details about editing diffusion-generated images. Firstly, different from editing real images, we do not need to conduct LoRA fine-tuning before latent optimization. This is because the purpose of LoRA fine-tuning is to help better encode the features of the original image into the diffusion UNet. However, for diffusion-generated images, the image features are already well-encoded as the diffusion model itself can generate these images. In addition, during the latent optimization stage, we do not have to perform DDIM inversion as the diffusion latents are readily available from the generation process of the diffusion models.

Another details we need to attend to is the presence of classifier-free guidance (CFG) when editing generated images. As described in the main text, when editing real images, we turn off the CFG as it pose challenges to DDIM inversion. However, when editing generated images, we inevitably have to deal with CFG, as it is one of the key component in diffusion-based image generation. CFG introduces another forward propagation pass of the UNet during the denoising process with a negative text embedding from null prompt or negative prompt. This makes a difference during our latent optimization stage, as now we have two UNet feature maps (one from the forward propagation with positive text embedding and the other one from the negative text embedding) instead of only one. To deal with this, we concatenate these two feature maps along the channel dimension and then use the combined feature maps to supervise latent optimization. This simple strategy have been proven to be effective as shown in our empirical results.

### D. Execution Time

Given a real image with the resolution of 512 × 512, the execution time of different stages in DRAGDIFFUSION on a A100 GPU is as follows: LoRA fine-tuning is around 25 seconds, latent optimization is around 10 to 30 seconds depending on the magnitude of the drag-instruction, the final Latent-MasaCtrl guided denoising is negligible comparing to previous steps (about 1 to 2 seconds)

### E. Limitations

As shown in Fig. 11, the limitation of our DRAGDIFFUSION is that, occasionally, some of the handle points cannot precisely reach the desired target. This is potentially due to inaccurate point-tracking or difficulties in latent optimization when multiple pairs of handle and target points are given. We leave making the drag-based editing on diffusion models more robust and reliable as our future work.

### F. More Qualitative Results

To start with, we provide more qualitative comparisons between DRAGDIFFUSION and DRAGGAN in Fig. 13. These results consistently showing that our approach demonstrate much better versatility than DRAGGAN.

Next, we demonstrate results of applying DRAGDIFFUSION on images generated by two more fine-tuned variants of Stable-Diffusion-1.5, namely Realistic-Vision and DVarch. Results are shown in Fig. 12. These results along with the results in the main text corroborate the generality of our approach on different diffusion models.

Finally, we provide more results on generated images beyond the 512 × 512 resolution as in the main text. These results are shown in Fig. 14, which further demonstrate the versatility of DRAGDIFFUSION.

[Figure 21]

Figure 12. Editing results on diffusion-generated images with (a) Realistic Vision; (b) DVarch.

### G. Visual Ablation on the Number of Identity-preserving fine-tuning steps

In the main paper, we ablate on the effect of the number of identity-preserving fine-tuning steps (denoted by n in this section). We show through numerical experiments that n ≥ 80 produce ideal results in Fig. 7 (b) of the main text. In this section, we provide visualization that corroborate with conclusions in the main text, showing setting n ≥ 80 produces editing results that are free from artifacts such as distorted faces and scenes, unexpected hands, etc. Results are shown in Fig. 15.

### H. Visual Ablation on the UNet Feature Maps

In the main paper, we have studied the effect of using UNet feature maps produced by different blocks of UNet decoder for our approach. In this section, we provide visualization that corroborates with conclusions in the main text. Results are shown in Fig. 16. According to the results, using the 1-st block of feature maps will lead to unfavorable preservation of local details due to lack of fine-grained information. This corresponds to the low Image Fidelity (IF) and high Mean Distance (MD) as in main text Fig. 7 (c) when Block number is 1.

On the other hand, as the 4-th block of UNet feature maps only contains low-level information, the editing results is almost the same as the original real image, indicating ineffective editing. This corresponds to the high IF and high MD as in main text Fig. 7 (c) when Block number is 4.

Finally, using the 2-nd or the 3-rd block of UNet feature maps can can yield reasonable editing. However, if observing more closely, we can see that using the 3-rd block of features yields slightly better preservation of local details (e.g. more reasonable headwrap in Fig. 16 (a) and better details of buildings by the river in the Fig. 16 (b)). Correspondingly, in main text Fig. 7 (c), we also show using UNet feature maps output by the 3-rd block can yield better results (lower MD and higher IF).

[Figure 22]

###### Figure 13. Additional comparisons between DRAGGAN and DRAGDIFFUSION. All images are from our DRAGBENCH dataset. Both approaches are executed under the same drag-based editing instruction. Zoom in to check the details.

[Figure 23]

###### Figure 14. Editing results from DRAGDIFFUSION beyond 512 × 512 resolution. Results are produced by perform drag-based edits on images generated by Counterfeit-V2.5. The resolution of images in the first row are 768 × 512, while the images in the second row are 512 × 1024.

User Edit

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

- Figure 15. Visual ablation study on the number of identity-preserving fine-tuning steps (denoted as n). Zoom in to view details. From the visualization, we see that setting n < 80 can produce undesired artifacts in the dragging results (e.g., distorted faces and scenes, unexpected hands, etc.). On the other hands, n ≥ 80 normally produces reasonable results without artifacts.

User Edit Block No.=1 Block No.=2 Block No.=3 Block No.=4

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

worse preservation of local details best results ineffective editing

- (a)
- (b)

- Figure 16. Visual ablation study on the block number of the UNet feature map. Zoom in to view details. As in the figure, using feature maps of the 2-nd and 3-rd blocks produce reasonable results. However, if observing more closely, we can see that using the 3-rd block of features yields slightly better preservation of local details (e.g. more reasonable headwrap in (a) and better details of buildings by the river in (b)).

art works landscape city countryside animals head upper body full body interior design other objects

DRAGGAN 0.71 0.84 0.74 0.79 0.72 0.91 0.33 0.31 0.57 0.71 DRAGDIFFUSION 0.88 0.88 0.89 0.88 0.87 0.85 0.89 0.95 0.90 0.87

Table 1. Comparisons of Image Fidelity (1-LPIPS) on DRAGBENCH on each category ( ).

art works landscape city countryside animals head upper body full body interior design other objects

DRAGGAN 59.51 47.60 41.94 46.96 60.12 65.14 82.98 37.01 75.65 58.25 DRAGDIFFUSION 30.74 36.55 26.18 43.21 39.22 36.43 39.75 20.56 24.83 39.52



Table 2. Comparisons of Mean Distance on DRAGBENCH on each category (

).

### I. Detailed Comparisons on DRAGBENCH by Category

In the main paper Fig. 8, we report the Mean Distance (MD) and Image Fidelity (IF) averaging over all samples in DRAGBENCH. In this section, we provide detailed comparisons between DRAGGAN and DRAGDIFFUSION on each category in DRAGBENCH. Comparisons in terms of IF (i.e., 1-LPIPS) and MD are given in Tab. 1 and Tab. 2, respectively. According to our results, DRAGDIFFUSION significantly outperforms DRAGGAN in every categories of DRAGBENCH.

