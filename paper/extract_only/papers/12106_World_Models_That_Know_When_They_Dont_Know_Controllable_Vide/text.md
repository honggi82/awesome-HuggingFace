# arXiv:2512.05927v2[cs.CV]10Mar2026

## World Models That Know When They Don’t Know:

##### Controllable Video Generation with Calibrated Uncertainty

###### Zhiting Mei1∗, Tenny Yin1, Micah Baker1, Ola Shorinwa1∗, Anirudha Majumdar1 1Princeton University

∗Equal contribution.

Recent advances in generative video models have led to significant breakthroughs in high-fidelity controllable video synthesis, conditioned on text, robot actions, etc. Despite their impressive capabilities, video models often hallucinate — generating future video frames that are misaligned with physical reality — which raises serious concerns in many downstream applications. Exacerbating this issue, video models also lack the ability to assess and express their confidence, impeding hallucination mitigation. To address this challenge, we propose C3, an uncertainty quantification (UQ) method for training continuous-scale calibrated controllable video models for dense confidence estimation at the subpatch level, precisely localizing the uncertainty in each generated video frame. Our UQ method introduces three core innovations to empower video models to estimate their uncertainty. First, our method develops a novel framework that trains video models for correctness and calibration via strictly proper scoring rules. Second, we estimate the video model’s uncertainty in latent space, avoiding training instability and prohibitive training costs associated with pixel-space approaches. Third, we map the dense latent-space uncertainty to interpretable pixel-level uncertainty in the RGB space for intuitive visualization, providing high-resolution uncertainty heatmaps that identify untrustworthy regions. Through extensive experiments on large-scale robot datasets (Bridge and DROID) and real-world evaluations, we demonstrate that our method not only provides calibrated uncertainty estimates within the training distribution, but also enables effective out-of-distribution detection.

Keywords: Controllable Video Models, Uncertainty Quantification, Trustworthy Video Synthesis. Website: c-cubed-uq.github.io Code: github.com/irom-princeton/c-cubed

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Object Distortion Object Collapse Object Switch Color Change Object Disappear

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

[Figure 17]

actions

Controllable Video Generation

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

o

- n

- t

i

- n u
- o

- u s C

r

o

- o

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

C

Proper Scoring Rule

[Figure 27]

d a

Interpretable

e

t

C3

Accuracy

rb

i

l

a

n

C

t l

l

e

- a
- b

l

Confidence

Figure 1 We present C3, the first method for training video models that know when they don’t know. Using proper scoring rules, C3 generates dense confidence predictions at the subpatch (channel) level that are physically interpretable and aligned with observations.

Banner

##### 1 Introduction

State-of-the-art (SOTA) controllable generative video models [1–4] are capable of synthesizing high-fidelity videos with rich visual content across diverse task settings. Video models offer significant promise as high-

fidelity embodied world models that address some of the most important challenges in robotics, although their applications to robotics are still in their infancy. Concretely, video models enable photorealistic simulation of complex dynamical interactions (e.g., simulation of deformable bodies) with the potential for continuous learning by scaling the training data (See [5] for a recent survey on applications of video models in robotics). However, video models have a high propensity to hallucinate, i.e., to generate future video frames that are physically inconsistent, posing a significant hurdle in applications that demand trustworthy video generation. Despite their tendency to hallucinate, video models lack the fundamental capacity to express their uncertainty, which hinders their trustworthiness. To the best of our knowledge, only one existing work attempts to quantify the uncertainty of video models [6]. However, the resulting estimates only capture task-level uncertainty, failing to resolve the model’s uncertainty spatially and temporally at the frame-level. Given that robotics applications broadly require fine-grained frame-level decision-making, we believe that dense spatio-temporal uncertainty quantification is critical for practical adoption of video models in robotics.

To address this critical challenge, we present C3, an uncertainty quantification (UQ) method for calibrated controllable video synthesis, enabling subpatch-level confidence prediction at any resolution in video generation accuracy, i.e., at continuous scales (see Figure 1). Our work is centered on three core contributions. First, we introduce a novel framework for training video models for both accuracy and calibration, founded on proper scoring rules as loss functions, effectively teaching video models to quantify their uncertainty during the video generation process. We demonstrate that the resulting uncertainty estimates are well-calibrated (i.e., neither underconfident nor overconfident) using benchmark robot learning datasets, e.g., the Bridge [7] and DROID [8] datasets.

Second, we derive our UQ method directly in the latent space of the video model. This key design choice circumvents the high computation costs associated with video generation in the (higher-dimensional) pixel space. Further, operating in the latent space streamlines applicability of our proposed method to a wide range of SOTA latent-space video model architectures [1–3], without requiring specialized knowledge or adaptation for implementation. Moreover, we compute dense uncertainty estimates at the subpatch level for high-resolution UQ, with more fine-grained detail compared to patch-level UQ representations.

Third, we decode latent-space uncertainty into interpretable pixel-space confidence estimates via temporal RGB heatmaps for intuitive visualization. We show that the uncertainty heatmaps are well-aligned with physical intuition, with areas of greater uncertainty associated with hallucinations — highlighting untrustworthy areas of the video. Further, we show that the model’s confidence estimates are negatively correlated with the error between the generated video and the ground-truth video, which is also consistent with intuition.

Finally, we demonstrate the effectiveness of C3 in detecting out-of-distribution inputs (i.e., environment conditions and actions) in video generation through real-world experiments on a WidowX 250 robot. Particularly, we show that C3 provides calibrated uncertainty estimates, even when the quality of the generated video is significantly compromised given the distribution shift at test time.

##### 2 Related Work

Video Generation Models. Research breakthroughs in video generation have led to significant advances in the capabilities of generative video models in recent years. While early video generation models were limited to generating short-duration videos (only a few frames) with small temporal changes, SOTA models can generate seconds-long videos (hundreds of frames) with impressive photorealistic detail. Early methods [9–11] synthesize novel videos by applying local (pixel-level) transformations to input images, composing these transformations to capture more complex temporal changes. However, these methods are limited to localized scene updates generally centered around a target object in the scene video. Moreover, these methods lack sufficient expressivity to generate photorealistic videos. Subsequent work [12–14] employs generative adversarial networks [15], while others [16–18] utilize variational inference with variational autoencoders [19] for video generation but fail to sufficiently address limited expressiveness and mode collapse. Addressing these limitations, SOTA methods [1–3, 20] leverage diffusion-based or flow-based modeling [21–23] for highfidelity video generation. More recent work [1, 24, 25] builds upon these methods to enable video generation conditioned on robot actions.

[Figure 28]

Patchify

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

###### VAE x DiT … DiT

###### z

FinalLayer

Unpatchify

VAE

at:t+H

c

Time+Action Embedding

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Confidence

Unpatchify

DiT … DiT

RGB Map

Latent Space

UQ Probe fϕ

Figure 2 Model Architecture. C3 enables simultaneous video generation and uncertainty quantification, quantifying the model’s confidence in its accuracy using a UQ probe acting on the video latents. High uncertainty regions (red) show hallucinations.

Uncertainty Quantification of Video Models. Uncertainty quantification of large language models (LLMs) has been been extensively studied (see [26] for a review of UQ methods for LLMs); however, only a few papers have explored uncertainty quantification of image or video generation models. Like traditional methods for UQ of deep neural networks [27], prior work on UQ of generative image models applies Bayesian methods to image diffusion models to estimate epistemic and aleatoric uncertainty using a variance-decomposition-based approach [28]. Another approach [29] takes an ensemble-based UQ perspective, estimating the uncertainty of image diffusion models using the mutual information over the distribution of the weights of an ensemble of the diffusion models. Some other methods [30] extract language descriptions from the generated images, facilitating uncertainty quantification of image generation models using established UQ methods for language models. Extending these methods to uncertainty quantification of video generation models is not trivial, given the spatio-temporality of videos and the significant computation costs of classical UQ methods. One existing approach [6] directly considers uncertainty quantification of video models. However, this method only provides a composite confidence estimate for each generated video and thus fails to provide more informative dense confidence estimates at the frame-level or pixel-level. Robotics applications generally require fine-grained frame-level decision-making, which benefits from dense spatio-temporal uncertainty quantification. Our work explores this research direction specifically.

Architecture

##### 3 Uncertainty Quantification of Video Models

For simplicity, we limit the discussion of our UQ method to video diffusion or flow-based models, given their SOTA performance. We provide a brief review of these models in Appendix F.1. However, we note that our proposed method readily applies to other video model architectures, such as GAN-based/RNN-based video models, with relatively straightforward adaptations. We adopt the latent diffusion transformer (DiT) architecture, described by:

x = Encode (v, g), xˆ ∼ DiT(x,a), oˆ = Decode(xˆ), (1)

where v denotes the input video frames, g denotes other conditioning inputs (e.g., text or action), x ∈ U denotes the encoded conditioning inputs in the latent video space U, and a ∈ A represents the action sequence carried out by the agent.

###### 3.1 Confidence Prediction

We introduce C3, a method for uncertainty quantification of controllable (action-conditioned) video generation models that provides dense estimates of the model’s confidence in the accuracy of each video frame at the subpatch level, conditioned on input actions. Concretely, we train the video model Vθ : U × A → U × U to generate accurate video frames with corresponding dense confidence estimates:

xˆ,qˆ ∼ Vθ(v,g,a), (2)

where qˆ ∈ U is the confidence prediction. Each element in qˆ corresponds to the model’s confidence in the accuracy of the associated subpatch.

Traditional UQ methods, such as Monte Carlo-based methods or ensemble-based techniques, generally require multiple forward passes or multiple instances of the model to estimate uncertainty. However, video models typically have billions of parameters, making these methods too computationally expensive and generally intractable. To overcome these challenges, we take a novel approach to UQ of video models. First, we pose UQ as a classification problem over the accuracy of the generated video, seeking to assess the model’s confidence in the video accuracy. This key choice eliminates inductive biases associated with restricting the predicted accuracy to a specific class of probability distributions, which could hinder calibration.

Given the high computational cost of video generation, we design a transformer-based UQ probe fϕ : U → U to estimate the video model’s confidence directly in latent space. We integrate fϕ within the video generation framework for simultaneous video generation and uncertainty quantification during training and inference,

- as summarized in Figure 2. However, we note that both components can also be trained independently. For more efficient training, we generate the videos in latent space using a vector-quantized variational autoencoder (VQ-VAE) with spatio-temporal convolution and attention layers to map input video frames to a lower-dimensional latent space. Specifically, we roll-out the forward and reverse diffusion processes in latent space, before mapping the generated latent video to the pixel space using a decoder. In this work, we utilize pre-trained VQ-VAEs [1, 31, 32], self-supervised with a reconstruction objective to compress RGB videos into a compact latent space. Further, we leverage diffusion forcing [33] for independent per-sample noise schedules, in line with prior work.

For action-conditioned video generation, we compute action embeddings from input actions using a multilayer perceptron and sum the resulting action embeddings with the timestep embedding computed using frequency-space encodings. We feed the resulting embeddings c to the DiT. Given the input video frame and action, we extract the internal features z of the DiT from the penultimate layer, which is passed into fϕ, alongside the action and timestep embedding, to predict the subpatch (channel-wise) confidence qˆ. This confidence represents the probability that each subpatch of the generated latent video is accurate with respect to an element-wise boolean function acc, which we elaborate in Section 3.2.

###### 3.2 Model Architectures

The definition of the accuracy function acc induces different model architectures. To demonstrate our method’s amenability to different realizations, we consider three possible architectural instantiations of C3. We define acc in terms of a distance function d. In this work, we use the L1 loss:

d(xˆ,x⋆) := |xˆ − x⋆|, (3)

although other distance metrics can also be used, e.g., the squared deviation. However, we emphasize that in our setting, all p-norms simplify to the L1 loss in Equation (3) since d is applied element-wise, making them equivalent. We use the L1 loss for simplicity. Given d, the accuracy function maps the generated videos to a binary-valued output space of the same dimensions as U, where each element is in {0,1}, given by the boolean operator:

acc(xˆ,x⋆,ε) := d(xˆ,x⋆) ≤ ε, (4)

based on the errors between the ground-truth and generated videos, given a threshold ε. The technique used in specifying ε induces a range of model architectures, namely: (i) fixed-scale classification models, (ii) multi-class classification models, and (iii) continuous-scale classification models, which we describe in the following subsections. We train all variants of our model with proper scoring rules for calibration. Appendix F.2 provides a brief overview of proper scoring rules.

Fixed-scale classification model (FSC). The FSC model predicts the accuracy of generated videos at a fixed accuracy resolution during training and inference, and thus requires the specification of a single error threshold ε. By requiring only a single value of ε, FSC models are typically faster to train than other models at the cost of generality to a range of resolutions. In practice, we select an appropriate value of ε for the task domain. As is standard in classification problems, we train fϕ to predict the logits and use the sigmoid function σ to map these values to valid confidence estimates qˆ that lie within [0,1]:

qˆ = σ(fϕ(z,c)), (5)

where fϕ is the confidence probe, z is the latent internal feature, and c is the latent action/time embedding. We optimize the parameters of fϕ with the Brier score loss function, given by: BS = Ey(ˆq − y)2 with ground-truth accuracy y and predicted confidence qˆ for the prediction over a single subpatch. We sum over all subpatches in computing the loss for each video. With a slight overload in notation, y,qˆ denote each component in y,qˆ, respectively.

Multi-class classification model (MCC). Inspired by the effectiveness of classical UQ methods for LLMs [26], we pose video model UQ as a multi-class classification problem by discretizing the output space of predictions into confidence bins, with the corresponding acc defined by:

acc(xˆ,x⋆,Oi) := εi ≤ d(xˆ,x⋆) < εi, (6)

where Oi represents the i-th bin with lower-bound εi and upper-bound εi. For each subpatch of the generated video, the MCC model predicts its confidence that the corresponding subpatch is accurate with respect to the accuracy thresholds associated with the bin. Like the FSC model, the MCC model predicts the logits for each bin, which is subsequently mapped to valid confidence (probability) values qˆ using the softmax. We optimize fϕ with the cross-entropy loss function, which is a strictly proper scoring rule given by: CE = Ey − k yk log qk , with ground-truth value y and predicted confidence qk for the k-th bin.

Continuous-scale binary classification model (CS-BC). To demonstrate the expressiveness of our approach, we train a continuous-scale model for any-resolution confidence prediction, conditioning fϕ on an accuracy threshold ε specified at inference. During training, we uniformly sample a set of εv at each iteration to ensure sufficient coverage of the thresholds. (εv is the linearly scaled version of ε specifying the deviation between predicted and ground-truth deviations. See Appendix F.1 for the full derivation.) In practice, for faster training, we discretize the space of ε using an adaptive hierarchical technique, by first dividing ε into uniform bins and further adaptively subdividing bins for higher resolution. When domain knowledge is available, more informed discretization or sampling schemes can be used for more efficient training. Like the preceding models, the CS-BC model predicts logits for each subpatch, which is mapped to confidence estimates using the sigmoid function:

qˆ = σ(fϕ(z,c,ε)), (7) taking the conditioning input threshold ε. In our experiments, we explore training the models with both Brier score and binary cross entropy, given by: BCE = Ey y log q + (1 − y)log(1 − q) , which are proper scoring rules.

End-to-end training. We train the video generation and uncertainty quantification modules independently end-to-end using the loss function:

Lθ,ϕ = Lθ + Lϕ, (8) where θ represents the parameters of the DiTs for video generation and ϕ represents the parameters of the UQ probe. We apply a stop-gradient operator between the video generation DiTs and the UQ probe. In our ablations in Appendix D, we explore the effects of backpropagating gradients from the UQ probe fϕ to the video generation DiTs. To optimize the loss function, we use stochastic gradient descent with a cosine-annealing decay schedule applied to the learning rate.

Proposition 1 (Uncertainty Decomposition). Given the input actions and video frames, the predicted confidence qˆ provides a calibrated measure of uncertainty of the video diffusion model in the generated video, provided that ϕ converges to an optimal solution.

We provide the proof in Appendix G.

###### 3.3 Decoding Latent Confidence Predictions

Like the latent video x, the predicted confidence qˆ is not immediately interpretable; hence, we decode the predicted confidence from the latent space to the pixel (RGB) space for better visualization. However, simply utilizing pre-trained video tokenizers as decoders would generally yield equally uninterpretable outputs, since these pre-trained decoders are trained to map RGB embeddings from the latent space to the pixel space. As a result, we define a color map in latent space by encoding monochromatic RGB video frames into the

latent space. For simplicity, we construct a latent color map from red-only, green-only, and blue-only video frames; however, higher-resolution color maps can also be constructed. We map the confidence estimates to latent RGB video frames by interpolating between the video frames in the latent color map. Subsequently, we map the latent RGB video frames for the predicted confidence to pixel space using the same tokenizer used in decoding the latent video x. In Section 4, we demonstrate that the resulting uncertainty heatmaps are well-aligned with intuition, identifying areas of the generated video that contain hallucinations.

##### 4 Experiments

We evaluate the performance of C3 in uncertainty quantification of action-conditioned video models, specifically examining its calibration, interpretability, and out-of-distribution detection capabilities via the following questions: (i) Is C3 underconfident, calibrated, or overconfident? (ii) Are C3’s uncertainty estimates interpretable? (iii) Can C3 detect OOD inputs at inference? We provide additional experiments and ablations in the Appendix.

Datasets. We conduct experiments on the Bridge dataset [7], a standard benchmark dataset for roboticsoriented video models. The Bridge dataset consists of real-world robot trajectories collected in 24 environments on a WidowX 250 robot arm with a fixed RGB camera, capturing broad environment variations across different robot manipulation tasks. In addition, we present additional results using the DROID dataset [8] in Appendix C. The DROID dataset consists of trajectories collected on a Panda robot arm with a Robotiq gripper, featuring greater coverage of tasks with multi-view camera observations collected using a wrist camera and two scene cameras.

Metrics. In order to empirically evaluate the calibration of C3, we use metrics that capture deviation from perfect calibration, including expected calibration error (ECE) and maximum calibration error (MCE), see Equation (18).

###### 4.1 Are C3’s uncertainty estimates calibrated?

We examine the calibration of the uncertainty estimates computed by C3 in dynamics prediction with controllable video models, specifically in robot manipulation tasks which constitutes a major application domain for these models. We train the three model architectures: CS-BC, MCC, and FSC (discussed in Section 3.2) on the train dataset split of the Bridge dataset and evaluate the trained models on the test split. To assess calibration, first, we generate videos and their corresponding dense confidence predictions conditioned on the initial video frame and the entire action trajectory for each sample in the test dataset. Subsequently, we compute the ECE and the MCE for each model, measuring the deviations from perfect calibration. We provide additional details on the evaluation procedure in Appendix E.

[Figure 45]

Calibration Errors. In Figure 3, we show the average ECE and MCE of each model across all the test videos. For the continuous-scale model CS-BC, we compute the average errors across ten equally-spaced error thresholds εv, spanning the observable latentspace prediction error domain as visualized in Figure 19. In contrast, the FSC model does not take in an accuracy scale for conditioning; as a result, we compute the ECE and MCE at the fixed-scale used in training the model. (For more informative evaluation and more comparative results, we select the fixed scale to lie within the range of εv used by the CS-BC model.) Similarly, we compute the ECE and MCE for all classes (accuracy scales) in the MCC model and report the average values in Figure 3. The results indicate that C3 produces well-calibrated uncertainty estimates across all models. Although all

Figure 3 Average calibration error. All three architectures have low ECE and MCE.

#### conﬁdence_heatmap_high_threshold

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

###### Ground-truthGeneratedComposited

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

- Figure 4 Interpretability of C3’s confidence estimates. While attempting to pick up the pot, the video model hallucinates a green object appearing within the robot’s gripper in a way that violates the law of physics. The object further undergoes non-casual deformation and changes to its color. C3 localizes these hallucinations, highlighting that the video model is highly uncertain about these hallucinations.

models achieve relatively the same MCE, their performance on the ECE differs. This small difference can be explained by the tradeoff between continuous-scale calibration and fixed-scale calibration. The improved expressiveness and flexibility of continuous-scale calibration might come at the cost of a marginal reduction in calibration at a single (specific) scale. The converse holds for the fixed-scale model, which is less expressive and flexible. The multi-class classification variant lies between these extremes on the tradeoff curve. We emphasize that the superior calibration performance of C3 arises from the use of proper scoring rules.

###### 4.2 Are C3’s uncertainty estimates interpretable?

Here, we examine the interpretability of the video model’s uncertainty estimates in video trajectory prediction in robot manipulation.

Qualitative Results. In Figure 4, we show the ground-truth and generated videos from the video model, along with a visualization of the video model’s confidence using a confidence heatmap, which transforms the model’s confidence predictions to the RGB color space using a color map. In the composited uncertainty map, the red regions represent areas of high uncertainty, corresponding to locations where the model is unsure if the generated video matches the ground-truth video.

In Figure 4, we observe that the video model is uncertain about the robot’s interaction with the pot in the video. After attempting to grasp the pot, the video model hallucinates a green object appearing within the robot’s gripper. As the interaction proceeds, the object morphs in non-physical ways with unrealistic changes to its shape and color. C3 identifies these hallucinations, revealing that the video model is highly uncertain about its hallucinations, as indicated by the red regions in the composited uncertainty map. These results underscore the interpretability of our proposed UQ method.

Quantitative Results. We assess the correlation between the estimated confidence of the video model and the error between the ground-truth and generated latent videos using the Shepherd’s Pi correlation [34], which is a robust correlation method that uses bootstrapping to identify outliers that would otherwise skew the correlation coefficient. For calibrated models, one would expect a negative correlation between the estimated

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

###### GeneratedCompositedGround-truth

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

15 Hallucination

- Figure 5 Capturing Hallucinations. The video model hallucinates the robot picking up a carrot, which appears out of nowhere. C3 is able to capture the high uncertainty corresponding to these hallucinations.

conf_heatmap_artifacts

confidence and the error between the ground-truth and generated videos, generally indicating an increase in the uncertainty of the video model as the video error increases. As expected, for the FSC and CS-BC models, we observe a statistically significant negative correlation of −0.373 and −0.172 between the confidence estimates and the absolute errors in the generated video at a 99% significance level, respectively. However, for the MCC model, we obtain a positive correlation coefficient due to the inadequate supervision of rightmost bins which correspond to greater latent error values, given that most of the generated video patches have a notably small latent error. For a more informed analysis, we examine the correlation of the confidence estimates of the MCC model with the maximum bin edge set at 0.2. We find that the confidence error is negatively correlated with the video error at the 99% significance level, with a coefficient of −0.130.

We provide additional visualizations showing the remarkable ability of C3 in capturing hallucinations, i.e., in localizing regions of the generated video where the model inserts artifacts such as previously non-existent objects or morphed objects. In Figure 5, the video model hallucinates the robot picking up a carrot out of nowhere. C3 localizes the corresponding areas as regions of high uncertainty. Similarly, C3 reveals that the video model is uncertain about the interaction dynamics of the plushy toy in Figure 6. After the initial grasp, the toy deforms and changes its color in ways that violate the laws of physics. C3 highlights the corresponding regions as areas of high uncertainty. Further, our proposed method is able to capture uncertainty from occlusions, which is shown in Figure 7. While grasping the red spoon, the robot’s arm occludes some areas of the scene. As the task proceeds, C3 identifies the occluded areas behind the robot as high-uncertainty regions, in line with intuition.

###### 4.3 Detecting OOD Inputs at Inference

Here, we explore the performance of C3 in out-of-distribution (OOD) detection at inference time, noting the importance of calibrated uncertainty estimates in reliable OOD detection. Concretely, a trustworthy video model would express higher uncertainty when given a task that lies outside of the training distribution, reflecting its lack of knowledge of the scene and object dynamics. We examine the calibration of the uncertainty estimates computed by our method in these settings through real-world experiments on a WidowX 250 robot in the Bridge setup within a toy kitchen environment.

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

GeneratedCompositedGround-truth

###### GeneratedCompositedGround-truth

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

- Figure 6 Uncertainty in Object Interactions. The plushy toy deforms in non-casual ways that violate physical laws. C3 shows that the video model is highly uncertain about these interaction dynamics.

Figure 7 Occlusions. The robot arm occludes some areas of the scene while grasping the red spoon. C3 identifies the occluded areas as high-uncertainty regions, in line with intuition.

Occlusion

red_around_grasped_object 6

We consider OOD conditions across five axes: background, lighting, environment clutter, target object (task), and robot end-effector, creating environment settings that are noticeably different from those seen in the Bridge dataset. For the background axis, we introduce novel background objects into the scene, e.g., computer accessories and sport equipment. For lighting, we vary the RGB value of the environment lighting. We add more objects to the scene in the environment clutter setting and introduce novel target objects or objects in unseen configurations for grasping in the target object test setup. Along the end-effector axis, we create OOD conditions by modifying the appearance of the end-effector by attaching lightweight objects (e.g., a towel, plushy toys, etc.) to the robot without noticeably altering the robot dynamics. In these settings, we collect 50 ground-truth trajectories (10 trajectories per setting) and generate videos from the video model using the associated robot actions.

In Figure 8, we show the ground-truth and generated videos and composited uncertainty maps for one trajectory in the Background and Lighting categories, with additional visualizations provided in Appendix A. For unfamiliar background objects (e.g., a skeleton), we observe that the model becomes uncertain about the dynamics between the robot and the background object as it approaches the background object, which can be seen in the generated video. C3 localizes this uncertainty, accurately delineating more confident video patches from less confident ones. Likewise, we see that the video model struggles to generate accurate videos under unseen lighting conditions, with an observable degradation in the video quality over time. Our method again captures the increasing uncertainty of the video model spatio-temporally.

###### 4.4 Ablations

We conduct ablation studies on the effects of simultaneous video generation and uncertainty quantification, alignment between latent space error and video quality metrics, and end-to-end training without stop-gradient. We discuss additional ablations in Appendix D.

Effects on Video Generation Quality. We report standard video quality metrics (SSIM, PSNR, LPIPS) for our UQ video model (C3) compared to the vanilla video model without UQ in Table 1. Our model achieves marginally better scores on the perceptual metrics, showing that our method does not degrade video quality. In head-to-head rankings, our model outperforms the vanilla model with scores of 62.5%, 58.8%, and 62.5%

Background Lighting

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

GeneratedCompositedGround-truth

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

- Figure 8 OOD detection. C3 is able to accurately localize hallucinations (identified by the red regions) in OOD scenarios where the model is presented with inputs outside of its training distribution.

Table 1 Perceptual Quality of the Vanilla Video Model without UQ

Table 2 Comparison of Video Latent Error to Perceptual Metrics.

Metric Vanilla C3 (Ours)

Method Correlation Significance (↑)

SSIM 0.75 0.76 PSNR 18.4 18.6 LPIPS 0.28 0.28

SSIM −0.54 99% PSNR 0.83 99% LPIPS 0.73 99%

on SSIM, PSNR, and LPIPS.

experiments_ood_calibration

Validity between Latent Space Error and Video Quality Metrics. Table 2 shows the correlation between the L1 latent space error and standard perceptual metrics. We find that the L1 latent error is strongly correlated with SSIM, PSNR, and LPIPS at the 99% significance level. Note that one would expect a negative correlation between the L1 latent error and SSIM and PSNR, and a positive one between the L1 latent error and LPIPS. Our results underscore that the L1 latent error is well-aligned with standard perceptual metrics, without the additional cost of decoding the video latents and computing these perceptual metrics.

End-to-End Training without Stop-Gradient. We examine the calibration of C3 when training the video model and the UQ probe fϕ end-to-end without a stop-gradient operator between both models. In other words, we update the parameters of the video model with the gradient of fϕ, assessing the existence of any training synergies from joint training. We train the CS-BC model with and without the stop-gradient operator and find no significant difference in the calibration of the resulting confidence estimates. We observe a difference of about 5e−3 and 3e−3 in the ECE and MCE, respectively, highlighting that both variants achieve the same level of calibration. However, we note that backpropagating the gradient of fϕ through the video model incurs additional computation overhead. As a result, the stop-gradient operation provides a computational edge, especially in large video models with billions of parameters.

##### 5 Conclusion

We present a method for calibrated controllable video synthesis that trains video models to know when they don’t know. We use proper scoring rules as loss functions to achieve both accuracy and calibration in video generation. By quantifying uncertainty in the latent space, our proposed method overcomes computation challenges and training instability associated with pixel-space approaches. Furthermore, we map latent-space uncertainty to interpretable pixel-space confidence estimates that are well-aligned with human intuition. We show that our method is able to precisely localize hallucinations in generated videos. Likewise, we demonstrate the calibration of the confidence estimates computed by our method across different robot embodiments and tasks, and further show the effectiveness of C3 in detecting out-of-distribution inputs at inference time.

##### 6 Limitations and Future Work

Although we demonstrate the calibration of C3 in OOD scenarios, its theoretical calibration guarantees only hold within the training data distribution through the use of proper scoring rules. Specifically, the diversity of the training data and the presence of a distribution shift at inference influences the observed calibration of the confidence estimates. We emphasize that this limitation is not unique to our approach. Nevertheless, we reiterate that our results show that our method produces calibrated uncertainty estimates even in OOD settings. Future work will explore training strategies for better coverage of the test distribution. Additionally, long-duration temporal consistency of the confidence estimates computed by our method is limited by the history length of the conditioning inputs of the video model. With smaller historical contexts, our method may lose track of uncertain video patches over time. Long-duration video generation remains an open research problem, which will be explored in future work. Looking forward, we believe that as applications of video models in robotics mature, rigorous uncertainty quantification will be crucial to their practical effectiveness.

##### Acknowledgments

The authors were partially supported by the NSF CAREER Award #2044149, the Office of Naval Research (N00014-23-1-2148), and a Sloan Fellowship.

##### References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [2] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [3] Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, et al. Open-sora 2.0: Training a commercial-level video generation model in 200 k. arXiv preprint arXiv:2503.09642, 2025.
- [4] DeepMind. Veo-3: A text-to-video generation system with audio. Technical Report Tech Report, DeepMind / Google, 2025.
- [5] Zhiting Mei, Tenny Yin, Ola Shorinwa, Apurva Badithela, Zhonghe Zheng, Joseph Bruno, Madison Bland, Lihan Zha, Asher Hancock, Jaime Fern´andez Fisac, et al. Video generation models in robotics-applications, research challenges, future directions. arXiv preprint arXiv:2601.07823, 2026.
- [6] Zhiting Mei, Ola Shorinwa, and Anirudha Majumdar. How confident are video models? empowering video models to express their uncertainty. arXiv preprint arXiv:2510.02571, 2025.
- [7] Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao, Philippe HansenEstruch, Quan Vuong, Andre He, Vivek Myers, Kuan Fang, Chelsea Finn, and Sergey Levine. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning (CoRL), 2023.
- [8] Alexander Khazatsky, Karl Pertsch, et al. Droid: A large-scale in-the-wild robot manipulation dataset. 2024.
- [9] Xu Jia, Bert De Brabandere, Tinne Tuytelaars, and Luc V Gool. Dynamic filter networks. Advances in neural information processing systems, 29, 2016.
- [10] Chelsea Finn, Ian Goodfellow, and Sergey Levine. Unsupervised learning for physical interaction through video prediction. Advances in neural information processing systems, 29, 2016.
- [11] Ziwei Liu, Raymond A Yeh, Xiaoou Tang, Yiming Liu, and Aseem Agarwala. Video frame synthesis using deep voxel flow. In Proceedings of the IEEE international conference on computer vision, pages 4463–4471, 2017.
- [12] Aidan Clark, Jeff Donahue, and Karen Simonyan. Adversarial video generation on complex datasets. arXiv preprint arXiv:1907.06571, 2019.
- [13] Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Generating videos with scene dynamics. Advances in neural information processing systems, 29, 2016.
- [14] Alex X Lee, Richard Zhang, Frederik Ebert, Pieter Abbeel, Chelsea Finn, and Sergey Levine. Stochastic adversarial video prediction. arXiv preprint arXiv:1804.01523, 2018.
- [15] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.
- [16] Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan, Roy H Campbell, and Sergey Levine. Stochastic variational video prediction. arXiv preprint arXiv:1710.11252, 2017.
- [17] Mohammad Babaeizadeh, Mohammad Taghi Saffar, Suraj Nair, Sergey Levine, Chelsea Finn, and Dumitru Erhan. Fitvid: Overfitting in pixel-level video prediction. arXiv preprint arXiv:2106.13195, 2021.
- [18] Bohan Wu, Suraj Nair, Roberto Martin-Martin, Li Fei-Fei, and Chelsea Finn. Greedy hierarchical variational autoencoders for large-scale video prediction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2318–2328, 2021.
- [19] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.
- [20] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

- [21] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022.
- [22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [23] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [24] Julian Quevedo, Ansh Kumar Sharma, Yixiang Sun, Varad Suryavanshi, Percy Liang, and Sherry Yang. Worldgym: World model as an environment for policy evaluation, 2025. URL https://arxiv.org/abs/2506.00613.
- [25] Yanjiang Guo, Lucy Xiaoyang Shi, Jianyu Chen, and Chelsea Finn. Ctrl-world: A controllable generative world model for robot manipulation, 2025. URL https://arxiv.org/abs/2510.10125.
- [26] Ola Shorinwa, Zhiting Mei, Justin Lidard, Allen Z Ren, and Anirudha Majumdar. A survey on uncertainty quantification of large language models: Taxonomy, open research challenges, and future directions. ACM Computing Surveys, 2025.
- [27] Moloud Abdar, Farhad Pourpanah, Sadiq Hussain, Dana Rezazadegan, Li Liu, Mohammad Ghavamzadeh, Paul Fieguth, Xiaochun Cao, Abbas Khosravi, U Rajendra Acharya, et al. A review of uncertainty quantification in deep learning: Techniques, applications and challenges. Information fusion, 76:243–297, 2021.
- [28] Matthew Chan, Maria Molina, and Chris Metzler. Estimating epistemic and aleatoric uncertainty with a single model. Advances in Neural Information Processing Systems, 37:109845–109870, 2024.
- [29] Lucas Berry, Axel Brando, and David Meger. Shedding light on large generative networks: Estimating epistemic uncertainty in diffusion models. In The 40th Conference on Uncertainty in Artificial Intelligence, 2024.
- [30] Gianni Franchi, Nacim Belkhir, Dat Nguyen Trong, Guoxuan Xia, and Andrea Pilzer. Towards understanding and quantifying uncertainty for text-to-image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8062–8072, 2025.
- [31] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22563–22575, 2023.
- [32] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jose´ Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023.
- [33] Boyuan Chen, Diego Martı´ Monso´, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.
- [34] Guillaume A Rousselet and Cyril R Pernet. Improving standards in brain-behavior correlation analyses. Frontiers in human neuroscience, 6:119, 2012.
- [35] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [36] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [37] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015.
- [38] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [39] Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American statistical Association, 102(477):359–378, 2007.
- [40] W Brier Glenn et al. Verification of forecasts expressed in terms of probability. Monthly weather review, 78(1): 1–3, 1950.

- [41] X Chen, M Hong, S Liu, and R Sun. On the convergence of a class of adam-type algorithms for non-convex optimization. In 7th International Conference on Learning Representations, ICLR 2019, 2019.
- [42] Meixuan He, Yuqing Liang, Jinlan Liu, and Dongpo Xu. Convergence of adam for non-convex objectives: Relaxed hyperparameters and non-ergodic case. arXiv preprint arXiv:2307.11782, 2023.

Clutter Object End-Effector

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

GeneratedCompositedGround-truth

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

- Figure 9 OOD detection, C3 is able to accurately localize hallucinations (identified by the red regions) in OOD scenarios where the model is presented with inputs outside of its training distribution.

### Appendix

##### A Detecting OOD Inputs at Inference

Extreme Lighting

GeneratedCompositeGround-truth

[Figure 133]

[Figure 134]

[Figure 135]

We provide additional visualizations of the ground-truth and generated videos and confidence predictions under different OOD conditions in Figures 9 and 10. In cluttered environments, the video model fails to accurately predict the interaction between the robot and the objects in the scene, which C3 correctly identifies. Lastly, when an unfamiliar object is attached to the robot end-effector, the video model becomes uncertain about the dynamics of the robot, leading to hallucinations in the generated video. Our method identifies these hallucinations as regions of high uncertainty. Under extreme lighting conditions, the video model hallucinates a recoloring of the scene in an attempt to match the training data distribution. C3 reveals the model’s uncertainty, specifically identifying regions with edited colors as areas of high uncertainty. In Figure 11, we provide the reliability diagram of C3 in OOD environments. We observe that our method remains well-calibrated with only a very small drop in calibration compared to its performance under nominal conditions. As stated in Section 4.2, C3 achieves low calibration errors, with an ECE and MCE of 9.98e−2 and 1.71e−1, respectively.

experiments_ood_calibration

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

- Figure 10 Extreme Lighting. C3 identifies hallucinations where the model tries to reset the lighting.

[Figure 142]

- Figure 11 Reliability diagram, showing strong calibration.

##### B Calibration on the Bridge Dataset

We assess underconfidence vs. overconfidence of our proposed UQ method using reliability diagrams, which visualize the calibration error associated with the uncertainty estimates across different confidence bins. In Figure 12, we show the reliability diagram for each model averaged across all thresholds. The dashed line in each plot traces the path of perfect calibration, while the crossshaped markers indicate the density of samples in each bin. Across all models, we observe that with C3, the video models are wellcalibrated, i.e., neither underconfident nor overconfident. Notably, the models tend to be more conservative when unsure about the accuracy of the generated video, as visualized by the bars in the [0.3,0.7] confidence bins exceeding the

experiments_ood_calibration

[Figure 143]

- Figure 12 Aggregated reliability diagrams. All methods are well-calibrated. Note that FSC provides the best calibration but lacks flexibility, given its dependence on a fixed error threshold. In contrast, MCC and CSBC are more general models that trade off marginal calibration for broader effectiveness.

[Figure 144]

- Figure 13 Reliability diagrams for FSC and CS-BC. Both models achieve similar calibration at threshold εv = 0.5.

dashed line. This emergent behavior aligns well with trustworthiness in safety-critical applications, with a greater propensity for the model to express doubt when unsure about the accuracy of the generated video. Additionally, note that FSC provides the best calibration but lacks flexibility, since it relies on a fixed error threshold. In contrast, MCC and CSBC are more general architectures that trade off marginal levels of calibration for broader expressiveness. Further, in Figure 13, we compare calibration of the CS-BC and FSC models at the fixed scale (εv = 0.5) used in training the FSC model. We find that both models are well-calibrated, producing relatively the same reliability diagrams.

In Figure 14, we provide additional results for CS-BC, showing calibration across different accuracy threshold levels. Overall, we see that C3 is well-calibrated across each error threshold, with the top of the confidence bins tracing the diagonal line. Further, we observe greater uncertainty at lower thresholds (e.g., εv = 0.2) which aligns with the intuition that lower accuracy thresholds are generally associated with greater uncertainty in the accuracy of the prediction given the tightness of the threshold. Conversely, as the threshold increases, the sample densities gradually shift right toward the higher confidence region, aligning with the intuition that larger thresholds afford greater confidence in the accuracy of the generated videos. Moreover, we observe that

- at extremely low values of εv (εv ≤ 0.3), C3 tends to be underconfident, signified by the histograms going above the line of perfect calibration. Further, note that the model’s degree of underconfidence decreases as the accuracy threshold increases. Overall, our observations are well-aligned with safety. The video model tends to be more conservative at very low accuracy thresholds, mitigating false negatives, i.e., inaccurate patches that are identified as highly confident regions, which could otherwise lead to harmful consequences.

##### C Evaluations on the DROID Dataset

We conduct additional experiments verifying the effectiveness of C3 on the DROID dataset [8], which covers a much wider range of tasks and environments compared to the Bridge dataset. We train the CS-BC model on this dataset and evaluate the calibration and interpretability of its confidence estimates. We compute the

[Figure 145]

- Figure 14 Reliability diagram for each threshold. C3 is well-calibrated across all accuracy thresholds, with some degree of conservativeness at very low thresholds.

calibration errors across the test dataset with ECE and MCE values of 7.28e−2 and 1.74e−1, respectively. Note that the ECE is again close to the lower bound of the range of calibration errors, highlighting the well-calibrated nature of C3. In summary, we find that our method is well-calibrated across a wide range of video prediction robotics problems, with broad amenability to multi-view camera inputs in diverse environments. Moreover, the results indicate that C3 remains effective across different robot embodiments, despite the notable difference between the Panda robot in the DROID dataset and the WidowX robot in the Bridge dataset.

Interpretability. Next, we examine the interpretability of the confidence estimates computed by C3. First, we compute the correlation between the predicted confidence and the absolute error between ground-truth and generated latent videos. Similar to the Bridge dataset, a negative correlation between both quantities indicates greater interpretability of the uncertainty estimates. On the DROID dataset, we observe a negative correlation coefficient of −0.149 with a significance level greater than 99%, showing a desirable alignment between the model’s estimated confidence and the observed accuracy of the generated videos. In other words, with C3, the video model tends to be more uncertain about the generated videos when it’s more likely wrong. Our results can be explained by the use of proper scoring rules to achieve both calibration and accuracy.

[Figure 146]

We provide visualizations of the ground-truth and generated videos, along with the estimated confidence, highlighting the calibration and interpretability of C3’s uncertainty predictions. First, in Figure 15, we show the reliability diagram of C3 computed across the videos in the test dataset. Similar to the Bridge dataset, we observe a near-perfect calibration of the confidence estimates.

From Figure 16, we observe that the video model is able to successfully generate multi-view videos, capturing the evolution of the task from two side-camera views (shown in the first-two rows) in addition to a wrist-camera view (shown in the third row). However, we also see a degradation in the relative quality of the generated videos compared to the Bridge dataset, reflecting the increased difficulty associated with multi-view video generation compounded with the greater diversity of the DROID dataset. This observation is particularly conspicuous in the wrist-camera view generated by the video model where the details of the scene quickly fade into a blurry background. Despite this degradation in video quality, C3 is still able to produce interpretable, calibrated uncertainty estimates at a fine-grained level, localizing non-confident regions of the video in each camera view. In particular, we see that in the right-camera view of the generated video (second row), our method captures hallucinations of the robot’s gripper that appear in the video—the gripper morphs and elongates. Likewise, C3 correctly identifies the inaccurate blurry background in the wrist-camera view (third row) as a region of high uncertainty.

Figure 15 Reliability diagram on DROID, showing calibration.

t = 3 t = 5 t = 6 t = 7

t = 8

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

Ground-truth

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

GeneratedComposited

droid_hallucination

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

- Figure 16 Hallucination. C3 identifies hallucinations in the generated videos from the DROID dataset as areas of high uncertainty.

##### D Additional Ablations

We perform additional ablations to study the effects of different scoring rules, diffusion forcing, and end-to-end training without the stop-gradient operation, with respect to calibration of the confidence estimates computed by C3.

Proper Scoring Rules. We examine the calibration of video models trained with the binary cross-entropy loss function and the Brier loss function. Since both loss functions are proper scoring rules, one would expect both models to achieve similar calibration performance. We train two variants of the CS-BC model using these loss functions and observe similar calibration levels between both models, in line with the preceding expectation. The results indicate a negligible difference in the ECE of about 3e−4 and a similarly negligible difference in the MCE of about 6e−3.

Further, we visualize the reliability diagrams associated with each model in Figure 17. We see very similar trends in the reliability diagrams across all confidence bins. Specifically, the models are well-calibrated but more conservative when unsure about the accuracy of the generated video. These findings underscore the general applicability of our proposed framework to strictly proper scoring rules.

[Figure 162]

- Figure 17 Reliability diagram for ablation on proper scoring rules. C3 remains well-calibrated with the BCE and Brier scores.

Diffusion forcing. We explore diffusion forcing in generating the confidence maps during video prediction and compute the ECE and MCE to evaluate calibration. We visualize the reliability diagrams of the CS-BC model with and without diffusion forcing in Figure 18, showing that diffusion forcing degrades the calibration of C3. We hypothesize that this observation could be due to the effects of the recurrence in diffusion forcing, which leads to a notable increase in the conservatism of the confidence estimates. With diffusion forcing, the ECE rises to about 3.3e−1 with an associated increase in the MCE to about 5.54e−1. A comprehensive analysis of the effects of diffusion forcing on confidence prediction lies beyond the scope of this paper; hence, we leave it to future work.

[Figure 163]

- Figure 18 Reliability diagram for ablation on diffusion forcing. Diffusion forcing increases underconfidence of C3, degrading calibration.

Additional Baselines. Established UQ baselines are prohibitively expensive to implement for large generative models, e.g., deep ensembles would require significantly greater GPU VRAM and compute time. Nonetheless, we compare our method to an approximate implementation of deep ensembles, where we exploit the video model stochasticity for multiple generations, using the resulting variance among generated videos as an uncertainty signal. Further, we compare against a heuristic baseline using the raw diffusion noise as uncertainty. Since these baselines do not estimate uncertainty with a probability distribution, standard calibration metrics for classification (e.g., ECE/MCE) do not apply. Consequently, we examine the correlation between latent space error and estimated uncertainty to evaluate calibration, which is standard for these estimates. Table 3 shows their correlation coefficients and significance levels α. C3 measures confidence and should ideally be negatively correlated with error, while ensemble and heuristics measure uncertainty and thus should be positively correlated. C3 outperforms the heuristic baseline (whose uncertainty estimates are not statistically significant) and is more competitive than the ensemble, given its lower computation cost while achieving comparable calibration.

Table 3 Comparison to other UQ Baselines (⋆Ensembles are broadly impractical due to high cost.)

Method Correlation Significance (↑)

C3 (ours) −0.36 99% Ensemble⋆ 0.42 99% Heuristic −0.01 43%

##### E Additional Implementation Details

Model Training and Evaluation. We implement the video generation model using a diffusion transformer architecture with 49 transformer layers, each with 4 heads and an embedding dimension of 512. We use the Stable Video Diffusion (SVD) VAE [35] for encoding the videos into the latent space, extracting video patches with no temporal compression. Note that we use SVD for its generality although our approach is amenable to other (larger) models. Using a learning rate of 1e−5 with a cosine decay scheduler, we train the video model for 50k iterations with a batch size of 4 for the Bridge dataset and a batch size of 2 for the DROID dataset, on 8 NVIDIA L40 GPUs. We use an input video resolution of 256 × 256 for both datasets. We stack the multi-view camera inputs in the DROID dataset to construct a single video frame as input.

We train on the entire train split of the Bridge dataset; however, given the large size of the DROID dataset, we only train on a subset (TRI), covering both success and failure videos, across a broad range of tasks and environments. When training the CS-BC model, we randomly sample thresholds from a discrete set of 28 threshold values constructed linearly from 0.1 to 1 with adaptive (denser) spacing at lower thresholds between 0.1 and 0.3 to more effectively capture the fine-grained signal existent in this subrange. We define the output bins of the MCC model using the same set of threshold values. We evaluate the video models on 110 and 83 trajectories in the test split of the Bridge and DROID datasets, respectively.

When evaluating the CS-BC model at inference time, we use 10 linearly-spaced values of εv ranging from 0.1 to 1.0. Except otherwise noted in our qualitative evaluations, we visualize the confidence predictions at the midpoint of the range corresponding to threshold values of 0.5 or 0.6. We compute the aggregate calibration errors for each model using standard implementations over 20 bins and compute the video accuracy using the ℓ1 loss.

Visualizing Latent-Space Errors. We visualize the latent-space error between representative generated videos and the corresponding ground-truth videos in the RGB space in Figure 19, to aid understanding of the accuracy resolutions discussed in this work. Intuitively, one would expect a calibrated video model to more confidently identify regions with very low or high errors as accurate or inaccurate, respectively, and to be more uncertain about other regions. For alignment with human intuition, we construct a latent-space color map with three basis colors, with the extreme points of the color map defined by blue and green, corresponding to the minimum point (low-error region) and the maximum point (high-error region) of the error span, respectively. We use the color red to represent the middle region of the error span, which induces an interpretable confidence heatmap, discussed later in this section. Notably, Figure 19 shows that an error span over the range [0,1] is sufficient to capture essentially all observable error values. Specifically, the resulting color maps contain almost no green region, associated with the maximum end of the error span. We find that most of the error values lie within the selected error span, further justifying the accuracy resolutions utilized in our experiments.

t = 5 t = 15 t = 21

t = 8 t = 12 t = 20

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Ground-truthGenerated

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

- 0

- 1

- Figure 19 Latent space error. We visualize the latent-space video error in the RGB space, showing the observable range of the errors.

latent_space_error

##### F Preliminaries

We provide relevant background material on video models and proper scoring rules.

###### F.1 Video Diffusion/Flow-Based Models

Video diffusion models (and more generally flow-based video generation models) [1–3, 20, 36] have emerged as the dominant model architecture for controllable, high-fidelity video generation, capturing fine-grained scene detail and longer video durations compared to alternative architectures. Video diffusion models learn a data distribution pθ(x) over video samples x ∈ U by first destroying the underlying structure in the training data through a forward diffusion process and subsequently restoring the structure through a reverse diffusion process [22, 37], where U represents the space of videos and x consists of a sequence of video frames.

In denoising diffusion probabilistic models (DDPM) [22], the forward diffusion process adds Gaussian noise to the training data following a Markov chain, while the reverse diffusion process recovers the target data by denoising pure noise xT ∼ N(0,I) through the procedure:

q(xt | x0) ∼ N(xt;√α¯tx0,(1 − α¯t)I), (9)

where αt := 1 − βt and α¯t := ts=1 αs, with Σθ(xt,t) = βtI. For more stable training, the learning problem is reformulated as a noise prediction problem, where the learned model predicts ϵt = (√1 − α¯t)−1xt −

√α¯tµ and optimizes the loss function:

0 ∥ϵ − ϵθ,t(xt,t)∥22 , (10) via stochastic gradient descent.

Lθ = Et,ϵ,x

In practice, we use denoising diffusion implicit models (DDIMs) [38], which generalize DDPMs to a nonMarkovian forward diffusion process. In effect, DDIMs decouple the forward and reverse diffusion timesteps and leverage this feature to accelerate the video generation process during inference. Concretely, with DDIMs, we can train the diffusion model using longer forward diffusion timesteps and generate new videos using shorter reverse diffusion timesteps, significantly reducing the generation time and computation overhead. Rather than predicting the noise ϵ, we predict the velocity v with a diffusion transformer.

We train the video model using a robot dataset D = {((Ij,t,Ij,t+1,aj,t), ∀t ∈ [Tj]), j = 1,...,N}, (11)

consisting of N trajectories. Each data sample consists of the current observation Ij,t ∈ RH×W×C, the next observation Ij,t+1 ∈ RH×W×C, and the corresponding action aj,t ∈ Rm, for the j’th trajectory of length Tj.

At inference, we sample new video frames using:

√α¯tx˜0(t) √1 − α¯t

xt−1 := √α¯t−1 x˜0(t) + 1 − α¯t−1

xt −

, (12)

√1 − α¯t vθ(xt,c) denotes the value of x0 predicted at timestep t and c denotes the action and timestep embeddings.

where x˜0(t) = √α¯t xt −

Remark 1 (Velocity-space Accuracy). The distance function in Equation (3) requires executing the reverse diffusion process to generate the latent video x, which is computationally expensive during training. To address this challenge, we express the distance function in terms of the predicted and ground-truth velocities, vθ and v⋆, respectively. By manipulating Equation (12) algebraically, we derive the relation:

d(x,x⋆) = α¯t(1 − α¯t−1) − α¯t−1(1 − α¯t) d(v,v⋆), (13) with the corresponding boolean function acc given by:

acc(v,v⋆) := d(v,v⋆) ≤ εv, (14) where εv = √ ε

√

.

α¯t(1−α¯t−1)−

α¯t−1(1−α¯t)

Notably, quantifying accuracy in the velocity-space only requires a simple linear transformation. Given acc in Equation (14), we train ϵθ and fϕ without the reverse diffusion process for greater training efficiency.

###### F.2 Proper Scoring Rule

For a random variable Y following the distribution P(Y ), a scoring rule S evaluates a prediction q of the probability distribution of Y by assigning a real-valued score, which could be interpreted as a penalty or reward, providing a measure of the quality of the predicted distribution. A scoring rule is proper if:

E

S(p(Y ),y) ≤ E

y∼p(Y )

y∼p(Y )

S(q,y), (15)

for all q [39]. Note that a proper scoring rule assesses a larger penalty for all predictions that are not equal to the underlying probability distribution of Y . Intuitively, the proper score is minimized when the predicted distribution q matches the true probability of Y . The scoring rule is strictly proper if equality holds in Equation (15) if and only if p(Y ) = q. Some examples of proper scoring rules include the Brier Score (BS) [40], Cross Entropy (CE), and Binary Cross Entropy (BCE), discussed in the paper.

In addition, we note that the expected calibration error (ECE) and maximum calibration error (MCE) are the standard metrics used to measure deviation from perfect calibration:

M

|Bm|

ECE :=

n |acc(Bm) − conf(Bm)|, (16) MCE := max

m=1

|acc(Bm) − conf(Bm)|, (17)

m∈{1,...,M}

where Bm represents bin m with cardinality |Bm|, and n represents the total number of samples across all bins.

##### G Proofs

Proposition 1 (Uncertainty Decomposition). Given the input actions and video frames, the predicted confidence qˆ provides a calibrated measure of uncertainty of the video diffusion model in the generated video, provided that ϕ converges to an optimal solution.

Proof. The proof of this proposition follows immediately from the definition of a proper scoring rule in Equation (15). Hence, from Equation (15), the minimum of the right-hand side (RHS) of Equation (15) is unique and in particular, is attained when q = p(y). By optimizing ϕ to minimize the RHS of Equation (15), we have that qˆ → p(y), assuming convergence, which is guaranteed under relatively weak conditions [41, 42]. Further, we have that:

P[Y = 1 | Q = qˆ] = E[Y | Q = qˆ] = qˆ, (18)

where the last equality follows from the fact that qˆ = p(y), upon convergence. The result in Equation (18) indicates calibration of the predicted confidence qˆ.

| |
|---|

