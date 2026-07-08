## Quantum Denoising Diffusion Models

# arXiv:2401.07049v1[quant-ph]13Jan2024

Michael K¨olle, Gerhard Stenzel, Jonas Stein, Sebastian Zielinski, Bj¨orn Ommer, Claudia Linnhoff-Popien LMU Munich

michael.koelle@ifi.lmu.de

### Abstract

In recent years, machine learning models like DALLE, Craiyon, and Stable Diffusion have gained significant attention for their ability to generate high-resolution images from concise descriptions. Concurrently, quantum computing is showing promising advances, especially with quantum machine learning which capitalizes on quantum mechanics to meet the increasing computational requirements of traditional machine learning algorithms. This paper explores the integration of quantum machine learning and variational quantum circuits to augment the efficacy of diffusion-based image generation models. Specifically, we address two challenges of classical diffusion models: their low sampling speed and the extensive parameter requirements. We introduce two quantum diffusion models and benchmark their capabilities against their classical counterparts using MNIST digits, Fashion MNIST, and CIFAR10. Our models surpass the classical models with similar parameter counts in terms of performance metrics FID, SSIM, and PSNR. Moreover, we introduce a consistency model unitary single sampling architecture that combines the diffusion procedure into a single step, enabling a fast one-step image generation.

### 1. Introduction

Image generation remains a vital topic in computer vision and graphics [9, 18], encompassing tasks from synthetic data creation to artistic endeavors. Generative models, like Stable Diffusion, have found applications ranging from image editing to aiding multi-modal models like GPT-4 in providing visual responses to human queries [36, 40]. While Denoising Diffusion Models (DDMs) have recently seen significant progress, they face notable challenges such as high computational demands and the necessity for extensive parameter tuning [7, 17, 40]. Recent advancements in quantum computing present opportunities to alleviate some of these challenges [3, 10]. Specifically, quantum machine learning (QML) uses quantum principles to enhance effi-

[Figure 1]

Figure 1. Diffusion process (τ = 10) of models Q-Dense, QDDPM [20], U-Net and Dense on MNIST digits. Samples from every second step are depicted.

ciency for classical machine learning tasks [3, 28, 52].

In this paper, we combine QML with DDMs to form quantum denoising diffusion models (QDDMs). This synthesis retains the image generation effectiveness of DDMs while benefiting from the efficiencies of quantum computing. By merging these two powerful domains, we push the frontiers of what is currently achievable in image generation, setting new benchmarks for quality and efficiency. We introduce a novel quantum U-Net design, employing quantum convolutions to further refine image quality. Additionally, we leverage the inherent unitary properties of quantum circuits to optimize QDDMs’ sampling time, introducing our unitary single-sample consistency model architecture. We evaluate our models together with classical deep convolutional networks and U-Nets on the datasets MNIST digits, Fashion MNIST and CIFAR10 using FID, SSIM, and PSNR performance metrics. Furthermore, we showcase the single-shot image generation capabilities on simulator and on real IBMQ hardware. Our results show that QDDMs hold a competitive edge over classical DDMs in producing high-quality images with fewer parameters. Finally, we present a detailed empirical analysis, on the strengths and

limitations of QDDMs, setting the stage for potential future explorations in this promising intersection of quantum and machine learning. In summary, our key contributions include:

- • The inception of two novel quantum diffusion architectures: Q-Dense and QU-Net.
- • The introduction the unitary single-sample consistency model architecture.

### 2. Related Work

#### 2.1. Diffusion Models

Diffusion models, as first introduced by [44], present a unique approach to training generative models. Rather than the adversarial battle seen in GANs [11], diffusion models focus on the steady transformation of noise into meaningful data. While the inception of this technique showed promising results [7], subsequent improvements like Denoising Diffusion Implicit Models (DDIM) emerged [17, 33, 45]. In contrast to traditional diffusion models, which sample each intermediary step in a Markov chain fashion, DDIM identifies and removes noise earlier, bypassing certain sampling iterations [45]. In this work, we primarily follow the methodology detailed in Ho et al. [17].

#### 2.2. Variational Quantum Circuits

Quantum machine learning (QML) aims to harness the capabilities of quantum computing to meet the increasing computational requirements of traditional machine learning algorithms [3, 10]. Variational quantum circuits (VQC) are foundational to QML, serving as function approximators similar to classical neural networks. These circuits utilize parameterized unitary quantum gates on qubits [1], leveraging the principles of quantum mechanics such as superposition, entanglement, and interference. These gates derive their parameters from rotation angles, which are trainable via conventional machine learning methods. A VQC’s architecture consists of three components.

The first component embeds image and guidance data into qubits. For image data, we employ amplitude embedding, this method encodes 2n features (pixel values) into n qubits, representing each feature as a normalized amplitude of the quantum state [32, 47]. For label embedding, we use angle embedding, which only encodes n features into n qubits but uses less quantum gates [55]. The second component consists of multiple variational layers, similar to hidden layers in classical networks. We design our circuits with strongly entangling layers, following the approach of Schuld et al. [43]. We also apply data re-uploading, reembedding parts of the input in-between variational layers, which aids in more complex feature learning [38]. Lastly, we extract the output by measuring the quantum system, causing the system’s superposition to collapse. Given n

qubits, this allows us to derive 2n joint probabilities of the output states. Quantum simulators further enable the extraction of the circuit’s state vector, which we use to build the combined unitary matrix in Sec. 3.4.

It’s important to highlight that while VQCs can efficiently manage high-dimensional input with just log2(N) qubits [28], they still encounter issues such as high qubit costs and error rates in the current Noisy Intermediate-Scale Quantum (NISQ) era [39]. However, anticipated advancements in these domains hold promise for QML’s pivotal role ahead [10, 39].

#### 2.3. Quantum Diffusion Models

To the best of our knowledge, the model QDDPM by Dohun Kim et al. currently stands as the sole quantum diffusion method for image generation [20]. They designed a single-circuit model with timestep-wise layers that take unique parameters for each iteration, and shared layers consistent across all iterations. This model shines in its spaceefficiency, needing only log2(pixels) qubits and thus exhibiting logarithmic space complexity for image generation. To counteract the vanishing gradient issue, they constrained the circuit depth. For entanglement, they utilized special unitary (SU) gates, targeting two qubits simultaneously. While ”SU(4)” groups offer benefits like known differentiation, their parameter efficiency per gate is lacking, as they are using 15 parameters per group. Given the constrained circuit depth, their model produces images that are somewhat recognizable but miss the intricacies of the originals (refer to Fig. 1).

### 3. Quantum Denoising Diffusion Models 3.1. Dense Quantum Circuits

In our work, we employed a dense quantum circuit (or strongly entangling circuit) as the foundational component of our quantum models. The term “dense” refers to the extensive entanglement among qubits in the circuit. This design choice is reminiscent of the nomenclature in classical deep learning, where the term “dense” or “fully connected” describes layers where every neuron is connected to every other neuron in adjacent layers.

The architecture of our dense quantum model is a follows. As detailed in Sec. 2.2, we chose amplitude embedding for input embedding due to its space-efficiency. Given that we are training on simulators, we bypassed the initial preprocessing steps outlined in [32]. Instead, we directly initialized the normalized data to the state of the quantum circuit. We encode the normalized class indices for guidance by utilizing angle embedding. This is achieved by adding an additional qubit (ancilla) and performing a rotation around the x-axis by an angle of class index ×

#classes. Our circuit’s variational component consist of sev-

2π

eral strongly entangling layers [43], resulting in a a total of #layers×3×#qubits trainable parameters. We then calculate the joint probabilities of a qubit subset, measuring the likelihood of the output being in states |00...00⟩ to |11...11⟩. If our output vector surpasses the input vector in size, we truncate the excess, eliminating unused measured probabilities. To align the output within the input data’s range, we scale the obtained probabilities using the input data’s euclidean norm.

#### 3.2. Quantum U-Net

Our Quantum U-Net (QU-Net), draws inspiration from classical U-Nets, particularly those without attention layers and upscaling features (shown in Fig. 2).

[Figure 2]

Figure 2. QU-Net architecture and quantum convolution, embedding a flattened slice into a dense quantum circuit.

Opposed to the well-known blocks with two or more classical convolutions each, we incorporate only one quantum convolution layer per block, as we observed prolonged execution times when using more convolutions. Quantum convolutions are our novel approach to use the flexibility of convolutions in quantum machine learning, allowing us to embed any slice of shape cin × k × k into a dense quantum circuit (Sec. 3.1) with max(log2(cin × k × k),log2(cout)) wires (Fig. 2) and measure cout outputs (with c being input and output channels, and k being the kernel size), thus differing from existing solutions like Quanvolution [15] and Quantum CNNs [4, 35].

#### 3.3. Guidance

Diffusion models can be extended with guidance, introducing auxiliary data during both training and inference. This

process, represented as pθ(xt−1|xt,c) (or pθ(ϵt−1|xt,c)), uses c as the guiding data [34].

For our dense quantum circuit, normalized class labels are embedded as rotation angles into an additional ancilla qubit, ensuring distinct quantum state representation. For instance, labels 0 and 1 correspond to angles 0 and π. Classical dense networks traditionally introduce inputs via an extra neuron per layer, enhancing performance and increasing parameter count.

Contrarily, U-Nets, due to their architecture, implement a mask encoding for labels. This mask, defined as mask(c) = 0.1 · sin(c + height/20), subtly alters the input image with strategically placed pixel value stripes, facilitating label identification. For quantum U-Nets, despite the necessity for normalized inputs in quantum convolutions, this masking technique remains effective. However, datasets with extensive class variety might demand alternative strategies.

#### 3.4. Unitary Single Sampling

The unitary nature of quantum gates and circuits allows us to combine the iterative application of Uτ during a diffusion step into one unitary matrix U (Fig. 3). This enables us to create synthetic images, using a single-shot of the circuit U, bridging the gap between quantum diffusion models and classic consistency models [46]. Additionally, this approach can be faster than executing multiple iterations of a classical diffusion model or even faster than executing each gate individually, depending on transpliation process and quantum hardware.

[Figure 3]

Figure 3. Unitary Single Sampling architecture.

Training the single sampling model demands an alternate loss computation. Instead of typical measurement probabilities, we are interested in the post-circuit quantum state. Training directly on quantum computers is currently infeasible as quantum state tomography for state reconstruction scales exponentially with system size [5]. Therefore, we utilize a noise-free quantum simulator. Loss gets evalu-

ated by comparing the post-circuit state pθ( |xt−1⟩||xt⟩) and the less noisy image |xt⟩, both represented as 2n-length complex vectors, using metrics like Mean Absolute Error (MAE).

For efficient sampling, we employ the trained parameters in the concatenated circuit Un. Alternatively, we can determine a singular Udiffusion matrix from the matrix form of Un. Precomputing this matrix allows for more streamlined sampling [46].

### 4. Experimental Setup

#### 4.1. Datasets

We use three well-known datasets to evaluate and compare our models: the MNIST digits [6], Fashion MNIST [54], and a grayscale version of CIFAR-10 [22].

The MNIST dataset features 28×28 pixel grayscale images of digits (0-9). To probe scalability, we utilize the original as well as downscaled (8×8) and upscaled (32×32) versions. For specific experiments tailored to quantum circuit embedding, we narrow our focus to digits 0 and 1. Fashion MNIST, in contrast, presents a multitude of intra-class and inter-class variations. A pivotal challenge is that models must interpret extensive regions of an image, beyond just the central figures, as peripheral areas do not consistently register as zero. The CIFAR10 dataset stands out due to its varied backgrounds and the challenges brought forth by reduced edge contrast in grayscale. We’ve adopted a grayscale version created by averaging the RGB channels of its 32 × 32 pixel images.

#### 4.2. Metrics

To gauge the quality of our generated images, we employ three metrics: the Fr´echet Inception Distance (FID) [16, 27], the Structural Similarity Index Measure (SSIM) [53], and the Peak Signal-to-Noise Ratio (PSNR).

The FID serves as a tool to gauge the resemblance between original and generated data. It achieves this by calculating the Wasserstein-2 distance between Gaussian distributions of activations derived from the Inception-v3 model [48]. A noteworthy aspect of FID is that lower scores suggest a closer resemblance between datasets [16]. SSIM is defined as

2σxy + c2 σx2 + σy2 + c2

2µxµy + c1 µ2x + µ2y + c1 ·

, (1)

SSIM(x,y) =

and offers a measure of similarity between images x and y. Higher SSIM scores are indicative of more significant image resemblance. Lastly, the PSNR stands as a metric to quantify the noise levels within an image. Superior image quality is represented by higher PSNR values.

#### 4.3. Baselines

Our benchmarking process contrasts our models against the architectures Deep Convolutional Networks (DCNs), UNets, and Quantum Denoising Diffusion Probabilistic Models (QDDPM).

DCNs are hierarchical models that utilize convolutional layers to extract progressively complex spatial features from input data [12, 23, 25]. By leveraging spatial invariance through weight sharing and pooling operations, DCNs can discern intricate patterns in large-dimensional datasets. Their depth and specialized architectures, such as residual and inception modules, enable the capture of both low-level image details and high-level semantic information, making them integral to advanced computer vision tasks.

The U-Net architecture [13, 19, 41] is acclaimed for its capabilities in image segmentation. The U-Net’s encoder captures local features, while its decoder combines these insights with broader context through skip connections, promoting gradient flow and information transfer [8, 49]. Strategies such as zero-padding and interpolation techniques ensure image sizes remain consistent across the iterative diffusion model process [7, 33]. Further refining image quality, attention layers become particularly beneficial when integrated with natural language embeddings [40, 50].

Lastly, we compare to the quantum state-of-the-art QDDPM by Dohun Kim et al. [20] as described in Sec. 2.3. This model is specially designed for the 8 × 8 MNIST dataset, incorporating six gates per layer, which results in a total of 990 parameters over τ = 10 timesteps. When adapted for 16 × 16 images, the model requires a greater number of parameters, totaling 4920. It is important to note that our comparisons are qualitative in nature, as we lack common metrics for evaluation.

#### 4.4. Model Training and Evaluation

We build our quantum models using the PennyLane framework [2]. For training our quantum models, we use the PyTorch integration of PennyLane which facilitates classical backpropagation for the gradients w.r.t. the rotation angles. On actual quantum hardware, parameter-shift differentiation calculates gradients by re-evaluating circuits with perturbed parameters [31, 42]. To enhance convergence and stabilize training, parameter remapping confines values within the range [−π,π] or [0,2π] [24]. We used the classical optimization algorithm Adam [21] and minimize the Mean Squared Error (MSE) between the generated image pθ(xt) = xt−1 and xt−1, sourced from a noiseaugmented training dataset. Notably, for the unitary singlesample model, we adopt the mean absolute error (MAE) due to its native PyTorch implementation for complex tensors. All runs were conducted on identical hardware with Intel Core ® i9-9900 CPUs and 64 GB of RAM. In our two preliminary studies (Sec. 8), we explore the relationship between model hyperparameters and metrics. Additionally, we performed a hyperparameter search focusing on learning rate and batch size. The detailed hyperparameter settings for each model are available in Sec. 10. Regarding our inpainting task experiments [29, 37], we evaluate the mod-

els without specific training for this purpose, using MSE to assess image fidelity. Challenges emerged when masks hid essential features: unguided models predominantly depended on existing pixels, whereas guided models benefitted from label guidance.

### 5. Experiments

In our experiments, we assess the effectiveness and efficiency of our quantum models across various datasets and conditions. We utilize diverse datasets, including MNIST Digits 8 × 8, Fashion MNIST 28 × 28, and CIFAR10 32 × 32, to explore our models’ performance under varying data complexities and dimensionalities. Our novel Unitary Single Sampling approaches are tested in several scenarios: MNIST Digits 8 × 8 both unguided without ancilla and guided with ancilla, MNIST Digits 32 × 32 unguided without ancilla, and MNIST Digits 8 × 8 unguided without ancilla on IBM Q hardware. Detailed results of these experiments, highlighting key findings and observations, are presented in the following sections.

#### 5.1. MNIST Digits

We analyze performance by creating models with varying layer sizes for each dataset and measure their complexity via the sum of trainable parameters. For guided MNIST 8×8 images, models encompassed approximately 1000 parameters. All trained models with their respective configuration can be found in Tab. 2 and Tab. 3.

Quantum-wise, we employed a dense circuit (Q-Dense) with 47 layers and 7 qubits, utilizing 6 qubits for image embedding and measurement, plus an additional one for label embedding. A preliminary comparison among quantum models revealed small advantages via data re-uploading [38], but introduced a challenge due to embedding gates altering the circuit’s quantum state. The best model (7 reuploads, red-line) scored around 10 FID points lower than the no re-upload model, although the difference dwindled with an increasing number of re-uploads.

For our comparison, we use fully-connected classical networks with 1000 trainable parameters and a U-Net of total depth 2, having 3 channels in the first block and 6 in the second. We compared our models to U-Nets with a depth of 3, and 2 or 4 channels in the initial block, even though they exceeded the 1000 parameter limit.

The Q-Dense model significantly surpassed its classical equivalents, having the same number of parameters, and showed exceptional performance especially when τ values were in the range of 3 to 5. This is particularly remarkable considering that all models were trained with τ = 10, highlighting the Q-Dense models’ advanced ability to learn from the original data distribution. However, a drawback is observed when excessive iterations are performed, leading

Model

300

Dense

Q-Dense

250

U-Net

U-Net (7 size)

200

FIDScore

150

100

50

0 2 4 6 8 10 Tau

Figure 4. FID scores on MNIST 8x8 with guided models. τ denotes the diffusion steps. Lavender line illustrates larger U-Net capabilities for reference.

to a decline in FID score caused by ongoing modifications to the input image, which ultimately produces artifacts.

The purple line represents the largest U-Net model (with channels sized 4, 8, and 16). This model outperformed all other architectures and has more than seven times the number of trainable parameters. Quantum models outperformed classical models with similar parameter counts, achieving FID scores around 100, thus 20 points better than classical models. The quantum models, along with the largest U-Net, exhibited slightly more consistent lower score-variance than other classical models across all runs, indicating more consolidated knowledge.

##### 5.1.1 Inpainting

We used MSE to evaluate the inpainting capabilities of models with ≈ 1000 parameters, testing various masks and noise conditions across multiple scenarios, illustrated in Fig. 5. Notably, the dense quantum circuit produced visually consistent samples with minor artifacts while maintaining high overall quality. Despite presenting a better FID score, the deeper quantum U-Net performed worse compared to its shallower counterpart.

In a experiment, where the original pixel get reset after the inpainting, most models showed declining performance after initial steps, with only the deep convolutional network maintaining consistent Fig. 5, albeit low-quality, output. Sample quality consistently held across varied masks and predictive models. In conclusion, our quantum models successfully performed knowledge-transfer tasks without specific inpainting training. They achieved satisfactory inpainting results and MSE scores, which were only marginally

QU-Net Q-Dense U-Net

[Figure 4]

[Figure 5]

[Figure 6]

- Figure 5. Inpainting samples with a small mask on the top half, resetting the bottom after each of the 10 iterations.

lower than those of classical networks, despite the classical networks having twice as many parameters.

- 5.2. Fashion MNIST

We trained models with approximately 4000 parameters on a subset of the Fashion dataset, focusing on the ”TShirt/Top” and ”Trouser” classes due to their relative structural similarity and middle-ground complexity between the MNIST Digits and CIFAR dataset. This dataset, containing more outliers and variances than MNIST Digits and featuring images of 28 × 28 pixels, necessitated notably longer training times for models to stabilize.

0 2 4 6 8

275

300

325

350

375

400

FID

Q-Dense (4k) Q-Dense (2k) QU-Net (4k)

U-Net (2k) U-Net (4k)

0 2 4 6 8

0.025

0.050

0.075

0.100

0.125

SSIM

- Figure 6. Model sample quality on the Fashion dataset, evaluated using FID and SSIM.

Upon stabilization, the models exhibited good performance, with the larger dense quantum circuit achieving the top FID score of 280, as depicted in Fig. 6. Our QU-Net and a classical U-Net achieved comparable FID scores. In terms of structural similarity, the Q-Dense models surpassed the (Q)U-Nets.

Examining the samples (Fig. 7), dense quantum circuits generated well-defined images with noticeable noise, while classical U-Nets produced less noisy, albeit less discernible, shapes. The quantum models achieved higher scores in SSIM, focusing on general structure, while FID, sensitive to noise, offered mixed results, thus rendering inconclusive the performance comparison between the small dense quantum circuit and the U-Net. PSNR scores mirrored SSIM results but also highlighted a lower performance in the quantum U-Nets, with the smaller QU-Net particularly affected

###### Q-Dense 2k

###### Q-Dense 4k

|[Figure 7]|
|---|

|[Figure 8]|
|---|

Batch

Batch

###### U-Net 2k

###### U-Net 4k

|[Figure 9]|
|---|

|[Figure 10]|
|---|

Batch

Batch

###### QU-Net 2k

QU-Net 4k

|[Figure 11]|
|---|

|[Figure 12]|
|---|

Batch

Batch

Figure 7. Chosen samples from the Fashion dataset, displaying every third τ and selected for optimal FID score per architecture.

by substantial artifacts along image edges. Discussion on potential solutions to observed issues with dense quantum circuits is available in Sec. 7.

#### 5.3. CIFAR10

We compared models on the CIFAR10 dataset, illustrating the limitations of our quantum models: low output fidelity, potential for mode collapse, and slow execution times. Low output fidelity arose primarily from the measurement process’s mathematical properties, wherein the output was always normalized (Sec. 2.2). Applying our approach of multiplying the output by the norm of the input state was only effective for homogeneous datasets like MNIST Digits. For non-homogeneous datasets like CIFAR10, this approach could result in over- or underexposed images. We observed a heightened risk of mode collapse due to the vast difference in modes within the CIFAR10 dataset. Our models summed their losses across all batches during training, meaning that dataset outliers could distort the gradient landscape, hindering learning of the full distribution. We employed varying guided model configurations for quantum models and a ≈ 1800 parameter U-Net, despite seemingly

Table 1. Average metrics on CIFAR

Model FID PSNR SSIM U-Net 395.47 8.95 0.026 QU-Net 271.03 9.60 0.086 Q-Dense 399.38 10.06 0.061

small for the dataset, to manage training time.

As evidenced in Tab. 1, quantum U-Nets outperformed classical U-Nets with the same parameter count, and some smaller QU-Nets achieved superior FID scores. Conversely, the dense quantum models, limited by their higherdimensional input state and therefore the number of layers to prevent memory issues, exhibited the weakest performance.

In the generated samples, small QU-Nets with 1000 parameters generated mostly large-scale structures, exhibiting a high variance. The larger QU-Net with 4000 parameters generated more detailed images with finer structures. The classical U-Net (2000 parameters) produced styles intermediate between the two quantum U-Nets. Meanwhile, the dense quantum model performed the weakest, displaying little discernible structure in the samples.

Despite the evident limitations of the dense quantum model, quantum U-Nets demonstrated superiority over classical U-Nets, affirming the advantages of our quantum convolution layers over classical convolutions. Nonetheless, they also exhibited significant drawbacks, such as notably higher training and sampling times.

Upon comparing the training and sampling speeds of models, simplified U-Nets (134 seconds per training epoch, 859 seconds per sampling run) proved much slower than QDense models (2 and 9 seconds) and exhibited better scaling with the number of parameters. The U-Nets, hindered by a high parallel batch size and a sequential backward pass, were notably influenced by the high parallel complexity. We mitigated this with caching to expedite the training process, despite the requisite higher memory amount.

In conclusion, while U-Nets’ sublinear scaling was not beneficial due to a bottleneck in the simulator framework, leading to exceedingly slow execution, preliminary experiments with caching the matrix representations of the quantum layers demonstrated potential for future improvement.

### 6. Unitary Single-Sampling

In this exploration of our novel unitary single-sampling models, we benchmarked them against each other in terms of sample quality, training duration, and speedup, rather than against conventional models. We selected a subset of the MNIST dataset featuring 8 × 8 and 32 × 32 images labeled 0 and 1, and trained versions with guided, unguided,

and unguided with ancilla qubit approaches. Moreover, we executed some models on IBMQ’s 7-qubit quantum hardware since our bit-efficient models required only log2(8×8) qubits.

Matrix transformations of certain trained circuits showcased the performance enhancement of condensed representation. We examined various noise initializations for inference and their resultant impact on image quality, utilizing amplitude embedding to encode random noise for inputting initial noise xτ. Single-pass models, requiring only a single forward pass during sampling and predicting data reconstruction pθ(xt−1|xt), exhibited their advantage when treating the concatenation of repetitions as a singular circuit.

[Figure 13]

[Figure 14]

(a) Without ancilla (b) With ancilla

- Figure 8. Samples from undirected single-sample models on the MNIST 8 × 8 dataset.

MNIST Digits 8 × 8 The 56 quantum layer, undirected model without ancilla qubits, was transformed into matrix representation, streamlining the sampling process from 1008τ quantum gate applications to a single matrix multiplication and vastly accelerating the sampling process. Conversion took under 5 seconds for τ = 1 and was executed once due to class-independence, while repeated multiplications for other τ values took mere microseconds in the PyTorch framework. The model successfully generated discernible digits despite low quality, with its performance surpassing that of its 28-layer counterpart. Although models produced better samples with 8 < τ < 14, both were capable of generating distinguishable samples from both classes with adequate training.

0 2 4 6 8 10

100

150

200

250

FID

# Layers

47 95 476

0 2 4 6 8 10

0.0

0.1

0.2

0.3

SSIM

- Figure 9. FID and SSIM scores of guided single-sample models on the MNIST 8 × 8 dataset

| |[Figure 15]|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

[Figure 16]

0

2

4

6

8

10

(a) Simulator (b) IBMQ

- Figure 10. Unitary Single-Sampling Samples on Simulator (guided) and on IBMQ (unguided).

Directed single-sample models of various architectures were trained, revealing a trend where larger models benefited from added guidance, as evidenced by a 17% FID score improvement and an SSIM score boost, whereas smaller models could be impeded by excessive re-uploads. Evidently, single-sample models generally profited from a larger number of trainable parameters across all metrics, as depicted in Fig. 10a. These models successfully generated recognizable digits from a normal distribution without requiring custom multinomial distributions.

MNIST Digits 32 × 32 Given the 10-qubit image representation requirement, we limited our evaluation to unguided models without ancillas due to simulator constraints. As with the 8 × 8 dataset, the 133-layer model (approx. 4000 parameters) outperformed the 66-layer model. However, both models struggled to represent the training manifold, consistently generating only zeros. Notably, the larger model produced recognizable digits up to τ = 16, underscoring the scalability of our approach. Despite limitations in class representation, these models retained some diffusion model properties, requiring only τ matrix multiplications for results, unlike traditional models.

MNIST Digits 8×8 on IBMQ We ran our unguided noancilla model on IBMQ’s quantum hardware. As Fig. 10b shows, it produced distinguishable samples. Despite inherent quantum hardware noise, with 10000 shots, our fullyquantum circuit successfully executed a diffusion model in roughly 40 seconds, excluding transpilation and queuing. Classical post-processing was minimal. Our diffusion models’ inherent noise robustness proved beneficial, acting as intrinsic error correction, making them suitable for the NISQ era. As lower-noise quantum hardware emerges, our model’s quality will likely improve. However, experiments on older calibrated devices yielded less recognizable digits.

### 7. Conclusion

In our research, we explored quantum denoising diffusion models, introducing the Q-Dense and QU-Net architecture. Furthermore, we introduced a quantum consistency model called unitary single-sampling, which consolidates the diffusion process into one unitary matrix, enabling one-step image generation. We benchmarked our models on unguided, guided, and inpainting tasks using datasets like MNIST digits, Fashion MNIST, and CIFAR10, employing FID, SSIM, and PSNR metrics. We compared our models qualitatively to the quantum state-of-the-art, classical deep convolutional networks and U-Nets.

Our results show that our models vastly outperform the only other quantum denoising model by Dohun Kim et al. [20]. Additionally, our quantum models surpassed similarly-sized classical models and matched the efficacy of models twice their size. However, in inpainting tasks, classical models still hold an edge. We demonstrated the one-step generation capabilities of the first working unitary single-sampling model, both on quantum simulators and IBMQ hardware.

In future studies, we aim to enhance variational quantum circuits by streamlining simulations using cached matrices, allowing for quicker GPU-parallel execution. Adopting 16-bit float precision could notably reduce RAM usage [14, 51], considering the demonstrated success in classical machine learning and prevalent GPU support for FP16. We’re also keen to explore diffusion patching [30] which leverages pixel neighborhoods as channels, a method that might significantly boost execution speed, especially with RGB images. A deeper probe into optimal data embedding methods (Sec. 3.3) compared against classical models could yield insights into quantum knowledge representation. Furthermore, refining dense quantum circuits with a customized entangling circuit may offer superior spatial locality. Lastly, introducing classical components for postprocessing in our models might present a pathway to circumvent quantum state normalization constraints and bolster overall performance.

### Acknowledgements

This paper was partially funded by the German Federal Ministry of Education and Research through the funding program “quantum technologies — from basic research to market” (contract number: 13N16196).

### References

[1] Adriano Barenco, Charles H. Bennett, Richard Cleve, David P. DiVincenzo, Norman Margolus, Peter Shor, Tycho Sleator, John A. Smolin, and Harald Weinfurter. Elementary gates for quantum computation. Physical Review A, 52(5): 3457–3467, 1995. 2

- [2] Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, Shahnawaz Ahmed, Vishnu Ajith, M. Sohaib Alam, Guillermo Alonso-Linaje, B. AkashNarayanan, and Ali Asadi. PennyLane: Automatic differentiation of hybrid quantum-classical computations, 2022. 4
- [3] Jacob Biamonte, Peter Wittek, Nicola Pancotti, Patrick Rebentrost, Nathan Wiebe, and Seth Lloyd. Quantum machine learning. Nature, 549(7671):195–202, 2017. 1, 2
- [4] Iris Cong, Soonwon Choi, and Mikhail D. Lukin. Quantum convolutional neural networks. Nature Physics, 15(12): 1273–1278, 2019. 3
- [5] Marcus Cramer, Martin B Plenio, Steven T Flammia, Rolando Somma, David Gross, Stephen D Bartlett, Olivier Landon-Cardinal, David Poulin, and Yi-Kai Liu. Efficient quantum state tomography. Nature communications, 1(1): 149, 2010. 3
- [6] Li Deng. The MNIST database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6):141–142, 2012. 4, 2
- [7] Prafulla Dhariwal and Alex Nichol. Diffusion models beat GANs on image synthesis, 2021. 1, 2, 4
- [8] Michal Drozdzal, Eugene Vorontsov, Gabriel Chartrand, Samuel Kadoury, and Chris Pal. The importance of skip connections in biomedical image segmentation, 2016. 4
- [9] Mohamed Elasri, Omar Elharrouss, Somaya Al-ma’adeed, and Hamid Tairi. Image generation: A review. Neural Processing Letters, 54, 2022. 1
- [10] Thomas Gabor, Leo S¨unkel, Fabian Ritz, Thomy Phan, Lenz Belzner, Christoph Roch, Sebastian Feld, and Claudia Linnhoff-Popien. The holy grail of quantum artificial intelligence: Major challenges in accelerating the machine learning pipeline, 2020. 1, 2
- [11] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 2
- [12] Jiuxiang Gu, Zhenhua Wang, Jason Kuen, Lianyang Ma, Amir Shahroudy, Bing Shuai, Ting Liu, Xingxing Wang, Li Wang, Gang Wang, Jianfei Cai, and Tsuhan Chen. Recent advances in convolutional neural networks, 2017. 4
- [13] Steven Guan, Amir A Khan, Siddhartha Sikdar, and Parag V Chitnis. Fully dense U-Net for 2-d sparse photoacoustic tomography artifact removal. IEEE journal of biomedical and health informatics, 24(2):568–576, 2019. 4
- [14] Suyog Gupta, Ankur Agrawal, Kailash Gopalakrishnan, and Pritish Narayanan. Deep learning with limited numerical precision, 2015. 8
- [15] Maxwell Henderson, Samriddhi Shakya, Shashindra Pradhan, and Tristan Cook. Quanvolutional neural networks: Powering image recognition with quantum circuits, 2019. 3
- [16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium, 2018. 4
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, pages 6840–6851. Curran Associates, Inc., 2020. 1, 2

- [18] He Huang, Philip S. Yu, and Changhu Wang. An introduction to image synthesis with generative adversarial nets,

2018. 1

- [19] Huimin Huang, Lanfen Lin, Ruofeng Tong, Hongjie Hu, Qiaowei Zhang, Yutaro Iwamoto, Xianhua Han, Yen-Wei Chen, and Jian Wu. U-Net 3+: A full-scale connected UNet for medical image segmentation. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1055–1059. IEEE, 2020. 4
- [20] Dohun Kim and Seokhyeong Kang. Quantum denoising diffusion probabilistic models for image generation. In Korean Conference on Semiconductors, 2023. 1, 2, 4, 8
- [21] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017. 4
- [22] Alex Krizhevsky. Learning multiple layers of features from tiny images. University of Toronto, 2012. 4
- [23] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Communications of the ACM, 60(6):84–90, 2017. 4
- [24] Michael K¨olle, Alessandro Giovagnoli, Jonas Stein, Maximilian Balthasar Mansky, Julian Hager, and Claudia LinnhoffPopien. Improving convergence for quantum variational classifiers using weight re-mapping, 2022. 4
- [25] Yann LeCun, Koray Kavukcuoglu, and Cl´ement Farabet. Convolutional networks and applications in vision. In Proceedings of 2010 IEEE international symposium on circuits and systems, pages 253–256. IEEE, 2010. 4
- [26] Richard Liaw, Eric Liang, Robert Nishihara, Philipp Moritz, Joseph E Gonzalez, and Ion Stoica. Tune: A research platform for distributed model selection and training. arXiv:1807.05118, 2018. 1, 2
- [27] Shaohui Liu, Yi Wei, Jiwen Lu, and Jie Zhou. An improved evaluation framework for generative adversarial networks,

2018. 4

- [28] Seth Lloyd, Masoud Mohseni, and Patrick Rebentrost. Quantum algorithms for supervised and unsupervised machine learning, 2013. 1, 2
- [29] Andreas Lugmayr, Martin Danelljan, Andr´es Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. CoRR, abs/2201.09865, 2022. 4
- [30] Troy Luhman and Eric Luhman. Improving diffusion model efficiency through patching, 2022. 8
- [31] K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii. Quantum circuit learning. Physical Review A, 98(3), 2018. 4
- [32] Mikko M¨ott¨onen, Juha J. Vartiainen, Ville Bergholm, and Martti M. Salomaa. Transformation of quantum states using uniformly controlled rotations, 2004. 2
- [33] Alex Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models, 2021. 2, 4
- [34] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models, 2022. 3
- [35] Seunghyeok Oh, Jaeho Choi, and Joongheon Kim. A tutorial on quantum convolutional neural networks (QCNN), 2020. 3

- [36] OpenAI. Gpt-4 technical report, 2023. 1
- [37] Deepak Pathak, Philipp Kr¨ahenb¨uhl, Jeff Donahue, Trevor Darrell, and Alexei A. Efros. Context encoders: Feature learning by inpainting. CoRR, abs/1604.07379, 2016. 4
- [38] Adri´an P´erez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, and Jos´e I. Latorre. Data re-uploading for a universal quantum classifier. Quantum, 4:226, 2020. 2, 5
- [39] John Preskill. Quantum computing in the NISQ era and beyond. Quantum, 2:79, 2018. 2
- [40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2021. 1, 4
- [41] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. UNet: Convolutional networks for biomedical image segmentation, 2015. 4
- [42] Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, and Nathan Killoran. Evaluating analytic gradients on quantum hardware. Physical Review A, 99(3), 2019. 4
- [43] Maria Schuld, Alex Bocharov, Krysta M. Svore, and Nathan Wiebe. Circuit-centric quantum classifiers. Physical Review A, 101(3), 2020. 2, 3
- [44] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265. PMLR, 2015. 2
- [45] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2020. 2
- [46] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models, 2023. 3, 4
- [47] Andrew Steane. Quantum computing. Reports on Progress in Physics, 61(2):117–173, 1998. 2
- [48] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1–9, 2015. 4
- [49] Tong Tong, Gen Li, Xiejie Liu, and Qinquan Gao. Image super-resolution using dense skip connections. In Proceedings of the IEEE international conference on computer vision, pages 4799–4807, 2017. 4
- [50] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2017. 4
- [51] Ganesh Venkatesh, Eriko Nurvitadhi, and Debbie Marr. Accelerating deep convolutional networks using low-precision and sparsity, 2016. 8
- [52] Guillaume Verdon, Michael Broughton, and Jacob Biamonte. A quantum algorithm to train neural networks using low-depth circuits, 2019. 1
- [53] Z. Wang, E.P. Simoncelli, and A.C. Bovik. Multiscale structural similarity for image quality assessment. In The ThritySeventh Asilomar Conference on Signals, Systems and Computers, 2003, pages 1398–1402 Vol.2, 2003. 4
- [54] Han Xiao, Kashif Rasul, and Roland Vollgraf. FashionMNIST: a novel image dataset for benchmarking machine learning algorithms, 2017. 4, 2

[55] Xiao-Ming Zhang, Tongyang Li, and Xiao Yuan. Quantum state preparation with optimal circuit depth: Implementations and applications. Phys. Rev. Lett., 129:230504, 2022. 2

## Quantum Denoising Diffusion Models Supplementary Material

### 8. Preliminary Studies

In two preliminary studies, we examine the relationship between hyperparameters and sample quality metrics like FID, PSNR, and SSIM for the QU-Net architecture, as well as the impact of input scaling.

0.0

channels

- -0.0
- -0.6 -0.2 0.8 0.4 -0.8 0.6 0.4 -0.7 0.9

channels

-0.70.1 0.3 0.6-0.1 0.6-0.1-0.70.0 0.0 0.0 0.0 0.1 0.2

FID PSNR SSIM

FID PSNR SSIM

layers

channelsFIDPSNRSSIM

channelsFIDPSNR

(a) Q-U-Net Correlation

(b) U-Net Correlation

Figure 11. Hyperparameter-sample quality correlation for QUNets and U-Nets on the Fashion dataset.

For quantum U-Nets, besides batch size and learning rate optimizations, we primarily consider the number of layers L and initial channels C. Analysis reveals that a higher sampling step count τ correlates with better sample quality. More channels show mixed results, and increasing the number of layers L marginally improves SSIM and PSNR.

Classic U-Nets, in contrast, display a strong correlation between channel number and sample quality across all metrics. From these findings, quantum U-Nets benefit from a larger τ, more channels for PSNR, and additional layers for SSIM.

Lastly, we analyze optimal input distributions for our unguided single-sample quantum models, computing them numerically as z = U−1 · x due to the invertibility of the unitary diffusion matrix U. This analysis shows that the real and imaginary parts of input vector z follow normal distributions N(µ = 0.4,σ = 0.24) and N(µ = 0,σ = 0.14) respectively. When manipulated by our diffusion model, they produce a non-uniform training data distribution x concentrated around 0 and 1 for dark and bright digit parts, respectively.

### 9. Quantum Architectures

In this work, we introduced two quantum architectures for denosing diffusion models. Each architecture has a guided and unguided variant as seen in Fig. 13. The guided variants all utilize a extra qubit called ”ancilla”, where the label is

embedded. Fig. 13b details how the label is embedded using a RX rotation (angle embedding) and how subsequent variational layers extend to the ancilla qubit. Both variants use strongly entangling layers as variational layers. In Fig. 12 we depicted a example of multiple strongly entangling layers, each trainable parameter θij is associated with the ith qubit and the j-th rotational gate, where j ranges over {0,1,2}. For clarity, indices indicating the layer are omitted in the graphic representation. The entangling CNOT gates target the qubit at position (i+l) mod n, where l is the layer number. This scheme ensures circular entanglement in the first layer (l = 1), with control and target qubits being adjacent, except for the last CNOT gate that completes the circle. In the second layer (l = 2), the control and target qubits are separated by one qubit. For instance, the target qubit for the first qubit (i = 0) in this layer is the third qubit (i = 2).

### 10. Hyperparameters

Before we conducted our experiments, we ran a hyperparameter search for all tested model variants. We primarily focused our hyperparameter optimization on the learning rate, since it is often the most influencial hyperparameter. We used the bayesian optimization algorithm from the RayTune [26] library. All model variants with their respective hyperparameters are detailed in Tab. 2 for our classical models and Tab. 3 for our quantum models.

[Figure 17]

Figure 12. Example of the strongly entangling layers part of the VQC, where the red line denotes the layer boundary.

|Guided Target Dataset Model Model shape LR|
|---|
|yes Data MNIST digits 8 × 8 Deep Dense<br><br>65 and 64 neurons, fully connected<br><br>0.0025 U-Net channel count 4,8,16 0.0057<br><br>U-Net channel count 2,4,8 0.0032<br><br>U-Net channel count 3,6 0.0047<br><br><br>no Data MNIST digits 8 × 8 Deep Convolutional 1,10,101 0.00759 Deep Convolutional 1,6,6,6,6,1 0.00013 U-Net channel count 2,4,8 0.00507<br><br>no Noise MNIST digits 8 × 8 U-Net channel count 2,4,8 0.01339<br><br>Deep Convolutional 1,9,10,1 0.00532 yes Noise MNIST 28 × 28 U-Net channel count 3,6 0.01719<br><br>U-Net channel count 2,4,8 0.00641 yes Noise Fashion MNIST Deep Convolutional 1,8,8,8,8,8,8,7,1 0.01016<br><br>U-Net channel count 7,14 0.01270<br><br>|

- Table 2. Learning rates of non-quantum models. The optimal learning rates have been chosen by a Bayesian optimization algorithm, facilitated by the RayTune [26] library. MNIST digits 8 × 8 refer to the downsampled 8 × 8 images, MNIST digits 32 × 32 denotes upsampled images to 32 × 32. MNIST 28x28 refers to the original MNIST handwritten digits dataset [6], Fashion MNIST dataset [54] uses 32 × 32 images.

[Figure 18]

[Figure 19]

(a) Unguided circuit (b) Guided circuit

Figure 13. Unguided and guided single-sampling quantum circuits.

|Guided Target Dataset Model Model shape LR|
|---|
|yes Data MNIST digits 8 × 8 Q-Dense 47 layers 0.00097 Q-Dense Re-Up 47 layers, 3 re-ups 0.00360 Q-Dense Re-Up 47 layers, 5 re-ups 0.00345 Q-Dense Re-Up 47 layers, 7 re-ups 0.00362<br><br>no Data MNIST digits 8 × 8 Q-Dense 50 layers 0.00065 Q-U-Net Simple<br><br>Channel count 2,4,8 8 Layers each<br><br>0.00023 Q-U-Net Simple<br><br>Channel count 2,4,8 12 Layers each<br><br>0.00815 no Noise MNIST digits 8 × 8 Q-Dense 55 layers 0.00160<br><br>Q-U-Net Simple<br><br>Channel count 2,4,8 8 Layers each<br><br>0.00113 Q-U-Net Simple<br><br>Channel count 2,4,8 12 Layers each<br><br>0.00912<br><br>yes Noise MNIST digits 28 × 28 Q-Dense 30 layers 0.00409 Q-Dense 60 layers 0.00211 Q-U-Net Simple<br><br>Channel count 2,4,8 9 Layers each<br><br>0.00287 Q-U-Net Simple<br><br>Channel count 2,4,8 19 Layers each<br><br>0.01479<br><br>yes Noise Fashion Q-Dense 121 layers 0.00014 Q-Dense 60 layers 0.00723 Q-U-Net Simple<br><br>Channel count 3,6,12 8 Layers each<br><br>0.00051 Q-U-Net Simple<br><br>Channel count 3,6,12 12 Layers each<br><br>0.00029<br><br>no Data MNIST digits 8 × 8 Single Sample 476 Layers, with ancilla 0.00012 no MNIST digits 8 × 8 Single Sample 47 Layers, with ancilla 0.00322 no MNIST digits 8 × 8 Single Sample 55 Layers, no ancilla 0.00172 no MNIST digits 8 × 8 Single Sample 555 Layers, no ancilla 0.00002 no MNIST digits 8 × 8 Single Sample 111 Layers, no ancilla 0.00419 no MNIST digits 8 × 8 Single Sample 95 Layers, with ancilla 0.00104 yes MNIST digits 8 × 8 Single Sample 476 Layers 0.00022 yes MNIST digits 8 × 8 Single Sample 476 layers, 10 re-ups 0.00016 yes MNIST digits 8 × 8 Single Sample 47 Layers 0.00721 yes MNIST digits 8 × 8 Single Sample 47 layers, 10 re-ups 0.00220 yes MNIST digits 8 × 8 Single Sample 95 Layers 0.00099 no MNIST digits 32 × 32 Single Sample 66 Layers, no ancilla 0.00212 no MNIST digits 32 × 32 Single Sample 133 Layers, no ancilla 0.00190<br><br>|

###### Table 3. Learning rates of quantum models. If not further specified, quantum layers are strongly entangling layers as described in [43]. The datasets are the same as in Tab. 2. All models used a batch size of 20 (except the quantum U-Nets, which used a batch size of 10), as higher batch sizes, while slightly improving the progress per iteration, lead to a higher number of crashes due to insufficient memory.

