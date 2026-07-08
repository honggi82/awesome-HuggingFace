# arXiv:2507.17189v1[cs.LG]23Jul2025

## Met2Net: A Decoupled Two-Stage Spatio-Temporal Forecasting Model for Complex Meteorological Systems

Shaohan Li1,2∗ Hao Yang1∗ Min Chen1,2,3† Xiaolin Qin2,3† 1Chengdu University of Information Technology 2Chengdu Institute of Computer Applications, Chinese Academy of Sciences 3University of Chinese Academy of Sciences

#### Abstract

The increasing frequency of extreme weather events due to global climate change urges accurate weather prediction. Recently, great advances have been made by the end-to-end methods, thanks to deep learning techniques, but they face limitations of representation inconsistency in multivariable integration and struggle to effectively capture the dependency between variables, which is required in complex weather systems. Treating different variables as distinct modalities and applying a two-stage training approach from multimodal models can partially alleviate this issue, but due to the inconformity in training tasks between the two stages, the results are often suboptimal. To address these challenges, we propose an implicit twostage training method, configuring separate encoders and decoders for each variable. In detailed, in the first stage, the Translator is frozen while the Encoders and Decoders learn a shared latent space, in the second stage, the Encoders and Decoders are frozen, and the Translator captures inter-variable interactions for prediction. Besides, by introducing a self-attention mechanism for multivariable fusion in the latent space, the performance achieves further improvements. Empirically, extensive experiments show the state-of-the-art performance of our method. Specifically, it reduces the MSE for near-surface air temperature and relative humidity predictions by 28.82% and 23.39%, respectively. The source code is available at https:// github.com/ShremG/Met2Net.

#### 1. Introduction

Global climate change and the increasing frequency of extreme weather events make accurate weather prediction crucial [10]. While traditional statistical and physical models are effective in some areas, they often struggle with complex nonlinear relationships and high-dimensional

*Equal contribution. †Corresonding authors.

Spatial Temporal

4

| | |Spatial| | |Temporal| | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

1.00

2

DifferenceValue

NormalizedValue

0.75

0

0.50

0.25

2

0.00

4

TCC_S T2M_S TCC_T T2M_T

TCC_S T2M_S TCC_T T2M_T

(a) Data distributions.

(b) First-order difference.

0.20

1.2

TAU_Single TAU_Multi

TAU_Single TAU_Multi

| |
|---|

| |
|---|

Two-Stage

Two-Stage

0.18

1.0

Ours

Ours

MAE

MAE

0.16

0.8

0.14

0.6

0.22 0.24 0.26

1.0 1.2 1.4 1.6 1.8

RMSE

RMSE

(c) T2M variable.

(d) TCC variable.

Figure 1. (a) Spatiotemporal distributions of the meteorological variables TCC and T2M highlight their heterogeneity. (b) Firstorder differences illustrate variations in trends and rates. (c) and (d) show that incorporating multiple variables did not enhance and even degraded TAU’s performance, while the two-stage training scheme also produced suboptimal results.

data. Recently, advancements in artificial intelligence have driven the rapid development of spatiotemporal prediction models, significantly improving weather forecasting accuracy [16, 17, 31, 32, 50, 51]. The success of these methods is attributed not only to their focusing on temporal nonstationarity but also to their incorporation of spatial dependency to capture more complex weather patterns.

However, variables such as temperature, humidity, wind speed, and precipitation interact with each other, forming a complex climate system [1, 5, 7, 8, 12]. The prevailing methods aim to analyze a single variable, fail to capture these dynamic relationships, thus leading to prediction errors. Therefore, integrating multiple meteorological variables enables the model to better understand weather patterns, identify potential extreme weather events, and provide more reliable warnings and decision support in re-

|Encoder<br><br>Translator<br><br>Decoder<br><br>X<br>Y'<br><br>Zx<br><br>Z'y<br>|
|---|

|Encoder<br><br>Translator<br><br>Decoder<br><br>X<br><br>X'<br><br>Zx<br><br>Z'y<br><br>Stage 1 Stage 2<br><br>Zx<br><br>Encoder<br><br>Decoder<br><br>Y'<br><br>X|
|---|

|Encoder<br><br>Translator<br><br>Decoder<br><br>X<br>Y'<br><br>Zx<br><br>Z'y<br><br><br>Encoder<br><br>Translator<br><br>Decoder<br><br>X<br>Y'<br><br>Zx<br><br>Z'y<br><br><br>Stage 1 Stage 2<br><br>|
|---|

(a) E2E.

(b) Generative.

(c) Ours.

- Figure 2. (a) End-to-End (E2E) training strategy. (b) Generative model training strategy, where task inconsistency exists between the two stages. (c) Met2Net training strategy, which ensures task consistency across both stages. The Translator module is responsible for learning spatiotemporal features. The Translator module is responsible for learning spatiotemporal features.

sponse to climate change [8, 12].

During implementation, we found that simply incorporating multiple meteorological variables into the existing spatiotemporal prediction framework did not significantly improve accuracy; in some cases, it even decreased. As shown in Figure 1c and 1d, the experimental results confirm this performance decline. Different from image data with RGB channel, which could be regarded as multi-variables, the meteorological variables demonstrate a high degree of divergence [21, 52]. As an illustration, we present the distribution of each variable’s values across spatial and temporal dimensions, as shown in Figure 1a, 1b. (Detailed experimental setting is given in Appendix.) These distribution divergences make it challenging for simple end-to-end training to effectively integrate these variables [9, 37, 41], leading to (i) representation inconsistency. The reason is that when a single model processes highly divergent data, it can easily lead to a loss of the original characteristics of the data as features across different dimensions are forcibly integrated into the same space, resulting in an inconsistency between the representation and the input variables.

To address representation inconsistency, two-stage training approaches, inspired by multi-modal generation models [11, 47, 59], treat different meteorological variables as distinct modalities. This strategy, widely used in modern generative models like Latent Diffusion [47] and autoregressive for image [30, 62] and video [24, 61], maps diverse modalities into a shared latent space, transforming heterogeneity into homogeneity [43] and enhancing model performance. However, its application to spatiotemporal prediction remains rare.

When we adapted this two-stage strategy, results were unsatisfactory (Figure 1c, 1d), which we attribute to (ii) task inconformity. In generative tasks, the first stage typically employs Auto-encoders or Variational Auto-encoders [6, 27] to reconstruct data, mapping it into a latent space. The second stage then uses models like diffusion to gener-

ate representations from noise [36, 38, 63], continuing the reconstruction process. However, in spatiotemporal prediction, while the first stage aligns with generative training, the second stage shifts to a prediction task rather than denoising reconstruction. This mismatch hinders the effectiveness of learned representations, leading to suboptimal accuracy, as shown in Figure 1c and 1d.

To address this issue, we propose Met2Net, a decoupled two-stage spatio-temporal forecasting model for complex meteorological systems, as shown in Figure 2c. Specifically, to eliminate representation inconsistency and task inconformity, we freeze the gradients of the Translator in the first stage and those of the Encoder and Decoder in the second stage, ensuring alignment between training objectives. Momentum updates [18] adjust the frozen components in both stages, maintaining stability and enabling smooth parameter adaptation. Additionally, a prediction loss in the latent space ensures a unified learning objective, keeping both stages focused on forecasting rather than shifting from reconstruction to prediction. This design allows Met2Net to function within a single continuous training cycle, eliminating explicit stage separation. To further improve performance, dedicated encoders and decoders preserve variable-specific features, while a self-attention mechanism in the Translator enables effective multivariable integration. By addressing both representation inconsistency and task inconformity, Met2Net achieves more accurate and stable multivariate spatiotemporal prediction. To summarize, our contribution in this work is as follows:

- • We propose an implicit two-stage training paradigm that freezes module gradients at specific stages and employs momentum updates. By introducing prediction loss in the latent space, the approach effectively aligns training objectives, enhancing spatiotemporal prediction.
- • We treat each meteorological variable as an independent modality, using separate encoders and decoders, and introduce a self-attention mechanism in the Translator to capture inter-variable relationships.
- • We construct a general multivariate spatiotemporal prediction dataset to evaluate the robustness and generalization of our method. Empirical results show that Met2Net achieves SOTA performance, reducing the MSE of T2M and R by 28.82% and 23.39%, resectively. Additionally, it improves hurricane trajectory prediction accuracy, demonstrating its effectiveness in weather forecasting.

#### 2. Related works

Spatiotemporal predictive learning. The advancements in recurrent-based models have greatly enhanced our understanding of spatiotemporal predictive learning. The ConvLSTM model [48] combines convolutional networks with LSTM to capture spatiotemporal patterns. Following this, PredRNN [53] and PredRNN++ [54] used spatiotemporal

LSTM (ST-LSTM) and gradient highway units to better capture temporal dependencies and reduce gradient vanishing. MIM [56] uses the difference between hidden states to improve handling of nonstationarity. E3D-LSTM [55] adds 3D convolutions to LSTM to improve feature extraction across time and space. PhyDNet [16] separates partial differential equation (PDE) dynamics from unknown factors using a recurrent physical unit. MAU [4] introduces a motion-aware unit to capture motion-related information. Lastly, PredRNNv2 [58] uses a curriculum learning strategy and memory decoupling loss to improve performance. Unlike recurrent-based methods, which have high computational costs in spatiotemporal prediction, SimVP [14] introduces a non-recurrent spatiotemporal prediction framework that reduces computational costs while achieving competitive performance. Subsequently, non-recurrent models such

- as TAT [39], TAU [50], DMVFN [20], and Wast [40] made further progress by incorporating triplet attention, temporal attention, dynamic multiscale voxel flow, and a 3D wavelet framework. However, these methods are based on singlevariable spatiotemporal training, and although they have achieved top-tier performance in weather prediction, they fail to fully exploit the potential of multivariable data in weather forecasting. To address this, we propose a solution that independently encodes and decodes each variable and introduce an implicit two-stage training strategy.

Weather forecasting. Weather forecasting, as a specific application of spatiotemporal prediction, has also benefited from these advancements, leading to significant improvements in prediction accuracy and computational efficiency. Han et al. [17] applied the SimVP model for radar extrapolation and achieved competitive performance in precipitation prediction scenarios. Chen et al. [8] introduced the TAU model, integrating satellite data to alleviate the issue of severe abrupt changes in precipitation scenarios. Eusebi et al. [12] utilized PINN to fuse barometric pressure for hurricane reconstruction. Jiang et al. [22] employed GNN to integrate multiple variables, further enhancing the accuracy of wind speed predictions. Models such as Pangu [1], FuXi [7], and Fengwu [5] incorporate a wide range of meteorological variables, achieving near-perfect predictions of typhoon paths and highlighting their capability for mediumrange forecasting. This underscores the necessity and effectiveness of integrating multiple variables and multimodal data in weather forecasting.

Two stage training methods. Two-stage training is widely used in generative tasks, where models like Stable Diffusion [47], MAGVIT [62], and VideoPoet [24] map diverse modalities into a shared latent space for improved representation learning. Typically, the first stage employs Auto-encoders (AE) or Variational Auto-encoders

(VAE) [6, 27] for data reconstruction, while the second stage utilizes diffusion models [36, 38, 63] to generate latent representations from noise. Recent research on twostage training focuses on enhancing the generality of representations learned in the first stage through large-scale pretraining [3, 15, 18, 42]. While this improves transferability across tasks, it requires extensive data, limiting its feasibility in scenarios with constrained datasets. However, in spatiotemporal prediction, the challenge lies not only in obtaining general representations but also in addressing task inconformity, where the second stage shifts from reconstruction to forecasting. This misalignment reduces the effectiveness of learned representations for prediction, leading to suboptimal accuracy. Our work tackles this issue by designing a training strategy that better aligns the objectives of both stages, improving prediction performance without requiring large-scale pretraining.

#### 3. Method

##### 3.1. Preliminaries

We formally define the multivariable spatiotemporal predictive learning problem as follows. Given a multivariable time sequence Xtt−T+1 = {xti}tt−T+1 at time t with the past T frames, we aim to predict the subsequent T′ frames Y t+T

′

′

t+1 = {yit+1}t+T

t+1 from time t + 1, represents a collection of N variables, each with channels C, height H, and width W. In practice, we represent the time sequences as tensors, i.e., Xtt−T+1 ∈ RT×N×C×H×W and Y t+T

′

′×N×C×H×W. For example, the wind variable has two channels, U and V, and its dimensions can be represented as RT×1×2×H×W.

t+1 ∈ RT

In this paper, we denote the Encoder by E, the Decoder by D, and the Translator by T . We denote the true input data by X and the label by Y . The reconstructed or predicted corresponding data are represented by X′ and Y ′, respectively. Zx represents the encoded state of X processed by E, and Zy represents the encoded state of Y processed by E. Zy′ denotes the predicted state of Zx processed by T .

##### 3.2. Overview

As shown in Figure 3, training is divided into two stages. In the first stage, the Translator is frozen while the Encoder and Decoder are trained to focus on spatial feature compression and reconstruction. In the second stage, the Encoder and Decoder are frozen, and a latent space prediction loss is introduced to train the Translator for inter-variable relationship modeling and prediction. During inference, only the fully trained modules are used, excluding those with frozen gradients. Thus, while this approach adds parameters and computation during training, the parameter count and computational load during inference remain the same as a standard end-to-end model. To better handle multi-variable sce-

|Training Pipeline (B×T)×C×H×W<br><br>[Figure 1]<br><br>Encoder🔥1<br><br>Encoder🔥2<br><br><br>(B×T)×C×H×W<br><br>Translatorm<br><br>❄<br><br>B×N×(T×C)×H'×W' Decoder🔥1<br><br>Decoder🔥1<br><br>Encoderm 2<br><br>❄<br><br>Translator🔥<br><br>Decoderm 1<br><br>❄<br><br>Train❄Frozen Momentum Update Stack<br><br>Stage Ⅰ<br>Stage Ⅱ<br><br><br>Decoderm 2<br><br>❄<br><br>Slice<br><br>Encoderm 1<br><br>❄<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>[Figure 8]<br><br>T2M<br>U10<br><br><br>T2M<br>U10<br><br><br>U10 U10<br><br>T2M T2M|Encoder 1 Encoder 2<br><br>Translator<br><br>Decoder 1 Decoder 2<br><br>Inference pipeline|
|---|---|

🔥

- Figure 3. Training pipeline and Inference pipeline. The training pipeline consists of two stages: In Stage I, the Translator (blue snowflake icon) is frozen while the Encoder and Decoder (orange flame icon) are trained. In Stage II, the Encoder and Decoder are frozen, and the Translator is trained. Momentum updates, represented by dashed arrows, are applied to the frozen parameters at the end of each stage. Multiple encoders and decoders are used for different meteorological variables, enabling independent feature extraction and reconstruction. The inference pipeline uses the trained Encoder, Translator, and Decoder to generate the final output. narios, each variable has its own Encoder-Decoder pair.

where L1 and L2 represent the loss functions for the first and second stages, respectively, and both are defined as Mean Squared Error (MSE) in this study. Y represents the target data, Y ′ is the model’s output data, Zy′ is the output of the second stage T , and Zy is the output of Y after being processed by E.

##### 3.3. Implicit two-stage training strategy

Inspired by the two-stage training approach used in multimodal models in the generative domain [24, 47, 62], we propose an implicit two-stage training strategy. This strategy involves freezing the gradients of different components

##### 3.4. Multiple meteorological variables fusion

- at various stages within a single training process and using momentum updates to adjust parameters. While ensuring consistency of objectives between the two stages, this method retains the advantages of traditional two-stage training strategies, such as effective module transferability and efficient modality fusion.

In meteorological forecasting, integrating multiple variables is essential in that meteorological factors, as different physical quantities, are mutually coupled and influence each other within an actual geophysics dynamic system [22, 26, 33, 35, 57]. Merging different variables can be regarded as multimodal fusion, and we thus, enlighten by the latest techniques in multimodal fusion [2, 19, 24, 44], design independent E and D for each meteorological variable and then used a T in the latent space to achieve the fusion and prediction.

In implementation, we should stop gradients of Translator (T ) during the first stage and gradients of Encoder (E) & Decoder (D) during the second stage. However, the Translator and Encoder & Decoder cannot be optimized alternatively with naive stop-gradient operation. To address this issue, we introduce the momentum update method to update the corresponding parameters. Formally, denoting the parameters of fm and f as θm and θ, we update θm by:

Each meteorological variable is assigned a pair of E and D with the same configuration. For E and D, we use a twodimensional architecture. E independently compresses each variable into the latent space along the spatial dimensions, while D restores the embeddings from the latent space back to the original space. Formally, denoting the two input meteorological variables as X1, X2, the fusion steps read

θm ← αθm + (1 − α)θ (1) Here α ∈ [0,1] is a momentum coefficient. f represents the components E, D, and T whose parameters are updated through backpropagation, while fm represents the corresponding components with frozen gradients.

Z1 = E1(X1),Z2 = E2(X2) (3)

During the training process, the objective function for the first stage is consistent with the end-to-end model, aiming to minimize the gap between Y ′ and Y . The objective for the second stage is to minimize the gap between Zy′ and Zy. Ultimately, our loss function can be expressed as:

Z = Stack(Z1,Z2) (4) Zˆ′ = T (Z) (5)

1 = D1(Slice(Zˆ′)),Xˆ′

Xˆ′

2 = D2(Slice(Zˆ′)) (6) Here, Stack(·) and Slice(·) represent stacking and slicing

L = L1(Y ′,Y ) + L2(Zy′ ,Zy) (2)

- Table 1. Quantitative comparison of predictions across multiple meteorological variables, including UV10, T2M, TCC, and R, measured using MSE, MAE, and RMSE metrics. # Params denote the parameters of the model during the inference stage. N represents the number of models required to predict the four variables. ↓ indicates lower is better.

|Method Date|#Params. N<br><br>(M)|UV10 MSE↓ MAE↓ RMSE↓<br><br>|T2M MSE↓ MAE↓ RMSE↓<br><br>|TCC MSE↓ MAE↓ RMSE↓|R MSE↓ MAE↓ RMSE↓<br><br>|
|---|---|---|---|---|---|
|ConvLSTM[48] NeurIPS’2015 PredRNN[53] NeurIPS’2017 PredRNN++[54] ICML’2018 MAU[4] NeurIPS’2021 SimVP[14] CVPR’2022 ConvNeXt[34] CVPR’2022 HorNet[45] NeurIPS’2022 TAU[50] CVPR’2023 Wast[40] AAAI’2024 Met2Net Ours<br><br>|14.98 4 23.57 4 38.40 4<br><br>5.46 4 14.67 4 10.09 4 12.42 4 12.22 4 14.46 4<br><br>8.65 1<br><br>|1.8976 0.9215 1.3775 1.8810 0.9068 1.3715 1.8727 0.9019 1.3685 1.9001 0.9194 1.3784 1.9993 0.9510 1.4140 1.6914 0.8698 1.3006 1.5539 0.8254 1.2466 1.5925 0.8426 1.2619 - - 1.5055 0.8196 1.2270|1.5210 0.7949 1.2330 1.3310 0.7246 1.1540 1.4580 0.7676 1.2070 1.2510 0.7036 1.1190 1.2380 0.7037 1.1130 1.2770 0.7220 1.1300 1.2010 0.6906 1.0960 1.1620 0.6707 1.0780 1.0980 0.6338 1.0440 0.8271 0.5770 0.9094<br><br>|0.0494 0.1542 0.2223 0.0550 0.1588 0.2346 0.0548 0.1544 0.2341 0.0496 0.1516 0.2226 0.0477 0.1503 0.2183 0.0474 0.1487 0.2178 0.0469 0.1475 0.2166 0.0472 0.1460 0.2173<br><br>- 0.1452 0.2150 0.0422 0.1370 0.2054|35.1460 4.0120 5.9280 37.6110 4.0960 6.1330 45.9930 4.7310 6.7820 34.5290 4.0040 5.8760 34.3550 3.9940 5.8610 33.1790 3.9280 5.7600 32.0810 3.8260 5.6640 31.8310 3.8180 5.6420<br><br>- 3.6940 5.5690 24.3854 3.3209 4.9382<br><br>|

- Table 2. The statistics of datasets. The training or testing set

has Ntrain or Ntest samples, composed by T or T′ images with the shape (C, H, W). The weather refers to the multivariate prediction dataset. The subscript Ldenotes low spatial resolution, HA represents high-altitude variables, Hand indicates high spatial resolution. The MvMmfnist is a dataset we constructed for the general multivariate spatiotemporal prediction scenario.

Ntrain Ntest C H W T T′

WeatherL 52559 17495 5 32 64 12 12 WeatherHA 54019 2883 12 32 64 4 4 WeatherH 52559 17495 5 64 128 12 12 ERA5 43801 8737 4 128 128 12 12 UV10 52559 17495 1 32 64 12 12 T2M 52559 17495 1 32 64 12 12 TaxiBJ 19627 1334 2 32 32 4 4 MvMmfnist 10000 10000 3 64 64 10 10

operations, respectively. Xˆ′

1, Xˆ′

2 represent the model’s prediction results for the two meteorological variables.

- 3.5. Translator

V ∈ RN×T×H×W is the value term, extracted by another 2D-CNN, Za ∈ RN×T×H×W is the fused embeddings. We denote the above operations using VA(·) (Variable Attention). Next, we use a spatiotemporal feature extractor written as ST(·) to extract spatiotemporal features and map them into predictions, as

###### T (Z) = ST(VA(Z)) (8)

In this paper, we utilize TAU Block[50] as the practical implementation of ST(·). We discuss the impact of using different blocks on the results in the appendix.

#### 4. Experiment

In this section, we present experiments demonstrating the effectiveness of our method across two tasks: multivariate weather prediction and general spatiotemporal prediction. For multivariate weather prediction, we tested low- and high-resolution spatial data as well as high-altitude data, and we conducted a hurricane trajectory prediction experiment using real ERA5 data. For general spatiotemporal prediction, both single-variable and multivariate experiments were conducted to assess the method’s generalization and versatility. The datasets used are listed in Table 2.

To model the complex dynamic dependencies of multiple meteorological variables, we utilize an attention mechanism to effectively capture the correlations between variables [13] and achieve effective decoupling between multiple variables.

Before processing with the T , the embeddings of each variable for all time steps need to be arranged sequentially along the channel dimension. For a spatiotemporal signal with N variables, the length of T, and the spatial resolution of (H,W), which is denoted by Z ∈ RN×T×H×W, the attention matrix over variable axis reads as:

The datasets WeatherL and WeatherH use UV10 (10meter u and v components of wind), T2M (2-meter temperature), TCC (total cloud cover), and R (relative humidity), for a total of four variables, with the two components of wind considered as a single variable. The WeatherHA dataset uses U (u component of wind), V (v component of wind), T (temperature), and R (relative humidity), for a total of four variables. Additionally, the ERA5 dataset includes MSL (mean sea level pressure), U10, V10, and T2M, also consisting of four variables. For the ERA5 dataset, we performed spatial cropping within the range of 110◦E–142◦E and 12.25◦N–44◦N. The dataset was split into three subsets: 2017–2021 for training, 2022 for validation, and 2023 for testing.

QK⊺ √

), (7)

A = softmax(

d

where Q,K ∈ RN×D×H×W are query and key, which are extracted by two different 2D-CNNs, and √

d is a scaling term. After the softmax(·) function, A ∈ RN×N demonstrates the correlations between different variables. Subsequently, the fusion is performed by Za = AV , where

[Figure 9]

[Figure 10]

(a) t=1. (b) t=12.

- Figure 4. Visualization of prediction results for different lead times. (a) Results at a forecast time of 1 hour. The background in white represents the absolute error (|Y ′ − Y |) for each model. (b) Results at a forecast time of 12 hours. Across both 1-hour and 12-hour forecasts, the error of the TAU model is consistently higher than that of our Met2Net model.

- Table 3. Quantitative comparison of predictions across multiple high-altitude variable meteorological variables, including R, T, U, and V, measured using MSE, MAE, and RMSE metrics. ↓ indicates lower is better.

Method Date R T U V

MSE↓ MAE↓ RMSE↓ MSE↓ MAE↓ RMSE↓ MSE↓ MAE↓ RMSE↓ MSE↓ MAE↓ RMSE↓

ConvLSTM[48] NeurIPS’2015 368.15 13.49 19.19 6.3030 1.7695 2.5107 30.002 3.8923 5.4774 30.789 3.8238 5.5488 PredRNN[53] NeurIPS’2017 354.57 13.17 18.830 5.5966 1.6411 2.3657 27.484 3.6776 5.2425 28.973 3.6617 5.3827 PredRNN++[54] ICML’2018 363.15 13.25 19.056 5.6471 1.6433 2.3763 28.396 3.7322 5.3288 29.872 3.7067 5.4655 MAU[4] NeurIPS’2021 363.36 13.50 19.062 5.6287 1.6810 2.3725 27.582 3.7409 5.2519 27.929 3.6700 5.2848 SimVP[14] CVPR’2022 370.03 13.58 19.236 6.1068 1.7554 2.4712 28.782 3.8435 5.3649 29.094 3.7614 5.3939 ConvNeXt[34] CVPR’2022 367.39 13.52 19.168 6.1749 1.7448 2.4849 29.764 3.8688 5.4556 31.326 3.8435 5.5969 HorNet[45] NeurIPS’2022 353.02 13.02 18.789 5.5856 1.6198 2.3634 28.192 3.7142 5.3096 30.028 3.7148 5.4798 TAU[50] CVPR’2023 342.63 12.80 18.510 4.9042 1.5341 2.2145 24.719 3.5060 4.9719 25.456 3.4723 5.0454 Met2Net Ours 337.98 12.74 18.38 4.1376 1.3688 2.0341 22.832 3.3449 4.7783 23.900 3.3597 4.8888

1 2 3 4 5 6 7 8 9 10 11 12

Layer

0.5

0.6

0.7

0.8

0.9

1.0

CKAValue

Encoder

Translator

Decoder

TAU Ours

- Figure 5. Met2Net effectively resolves representation inconsistency by maintaining higher CKA values across layers compared to TAU. The TAU model exhibits a gradual increase in CKA values through the Translator section but maintains relatively lower CKA values overall. In contrast, Met2Net sustains higher CKA values across all layers, particularly in the Encoder and Translator sections, indicating improved representation consistency.

The optimizer used was Adam [23], with a training batch size of 16, running for 50 epochs, and a momentum parameter α set to 0.999. We evaluated the test set using three metrics: MSE, MAE, and RMSE.

##### 4.2. Multiple Meteorological Variables Prediction

Low-Resolution. Table 1 presents a quantitative comparison between our method and existing models for lowresolution multivariable meteorological prediction. Our method achieves SOTA performance across all major metrics, including MSE, MAE, and RMSE. Compared to TAU, our approach reduces MSE by 5.46%, 28.82%, 10.59%, and 23.39% for the four variables, respectively. To illustrate these differences more intuitively, Figure 4 presents a visual comparison between models, where Met2Net demonstrates smaller spatial prediction errors across the four variables, highlighting its advantage in accuracy.

High-Altitude Variable. To further validate the robustness of our approach under varying atmospheric conditions, we conducted an experiment on high-altitude meteorological prediction. Table 3 compares the performance of different models in this task, showing that our method maintains SOTA performance. This experiment evaluates model performance at three altitude levels—150, 500, and 850 hPa—using 6-hour intervals. Specifically, the previous 4

##### 4.1. Setup

The baseline experiments were conducted using the OpenSTL [51] library, with all meteorological variable data sourced from the WeatherBench [46] dataset. The settings followed the OpenSTL configuration, where each variable inputs 12 frames and predicts the subsequent 12 frames.

- Table 4. Quantitative comparison of predictions across multiple high-resolution meteorological variables, including UV10, T2M, TCC, and R, measured using MSE, MAE, and RMSE metrics. ↓ indicates lower is better.

Method Date UV10 T2M TCC R

MSE↓ MAE↓ RMSE↓ MSE↓ MAE↓ RMSE↓ MSE↓ MAE↓ RMSE↓ MSE↓ MAE↓ RMSE↓

ConvLSTM[48] NeurIPS’2015 1.5691 0.8653 1.2518 1.1385 0.6774 1.0654 0.0450 0.1431 0.2121 28.9580 3.6594 5.3769 PredRNN[53] NeurIPS’2017 1.4083 0.8037 1.1859 1.1862 0.6695 1.0873 0.0482 0.1444 0.2195 30.1590 3.7334 5.4875 MAU[4] NeurIPS’2021 1.5781 0.8578 1.2554 1.2493 0.7097 1.1161 0.0446 0.1451 0.2111 30.7881 3.7989 5.5442 SimVP[14] CVPR’2022 1.7654 0.9332 1.3279 1.2750 0.7333 1.1278 0.0448 0.1451 0.2115 32.7849 4.0132 5.7217 ConvNeXt[34] CVPR’2022 1.4804 0.8399 1.2160 1.2155 0.7048 1.1010 0.0457 0.1455 0.2136 31.9944 3.8841 5.6518 TAU[50] CVPR’2023 1.3038 0.7887 1.1411 1.0599 0.6592 1.0282 0.0434 0.1403 0.2082 28.2291 3.6307 5.3090 Met2Net Ours 1.2745 0.7835 1.1283 0.8936 0.6194 0.9446 0.0378 0.1270 0.1944 21.5415 3.1606 4.6382

- Table 5. Quantitative comparison of predictions performance on the ERA5 dataset, including MSL, U10, V10, and T2M, measured using MSE, MAE, and R2 metrics. ↓ indicates lower is better, while ↑ indicates higher is better.

Method Date MSL U10 V10 T2M

MSE↓ MAE↓ R2↑ MSE↓ MAE↓ R2↑ MSE↓ MAE↓ R2↑ MSE↓ MAE↓ R2↑

ConvLSTM[48] NeurIPS’2015 16577.42 90.8390 0.9677 1.7236 0.9238 0.9397 1.8728 0.9813 0.9060 2.0504 0.9236 0.9873 ConvNeXt[34] CVPR’2022 15618.98 92.7848 0.9696 1.5358 0.8495 0.9462 1.6811 0.9020 0.9156 1.2860 0.7347 0.9920 HorNet[45] NeurIPS’2022 14585.94 88.7772 0.9716 1.4964 0.8443 0.9476 1.6184 0.8896 0.9188 1.1962 0.7087 0.9926 MogaNet[29] ICLR’2024 18617.73 100.8613 0.9638 1.5582 0.8756 0.9454 1.6547 0.9084 0.9170 1.4031 0.7618 0.9913 TAU[50] CVPR’2023 16541.85 93.7000 0.9678 1.4979 0.8404 0.9476 1.6121 0.8824 0.9191 1.2808 0.7205 0.9920 Met2Net Ours 9645.28 70.3976 0.9812 1.2070 0.7518 0.9577 1.3124 0.7954 0.9341 0.8176 0.5866 0.9949

presents a quantitative comparison across multiple highresolution variables—UV10, T2M, TCC, and R—using MSE, MAE, and RMSE as evaluation metrics. Met2Net demonstrates competitive performance across all metrics, achieving the lowest error values, which underscores its robustness and accuracy in high-resolution prediction tasks.

115°E

120°E

125°E

130°E

135°E

140°E

| | | |
|---|---|---|
| | | |

Truth

42°N 42°N

| | | |
|---|---|---|
| | | |

TAU

| | |
|---|---|
| | |

Ours

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

37°N 37°N

| | | |
|---|---|---|
| | | |
| | | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Tracking Tropical Cyclones. Our method was evaluated on Typhoon Mawar’s trajectory prediction using the ERA5 dataset, which includes MSL, U10, V10, and T2M variables. As shown in Figure 6, our model closely follows the ground truth while outperforming the TAU model, particularly in the early and mid-stages. Notably, it achieves higher accuracy with a 3-hour lead time while using fewer parameters. The quantitative results in Table 5 further confirm its superiority, with the lowest MSE and MAE and the highest R2 across all variables, demonstrating its efficiency and effectiveness in tropical cyclone forecasting.

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

32°N 32°N

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

27°N 27°N

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

22°N 22°N

| | | |
|---|---|---|
| | | |

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |

| | |
|---|---|
| | |

17°N 17°N

Further analysis. To demonstrate the effectiveness of our method in addressing representation inconsistency, we analyzed the Centered Kernel Alignment (CKA) [25] values across layers for two meteorological variables: T2M and TCC. As shown in Figure 5, this analysis compares CKA values for both variables in the TAU and Ours models, focusing on the Encoder, Translator, and Decoder sections. CKA values, which measure similarity between layer representations, reflect each model’s alignment with the internal structure of T2M and TCC, where higher values indicate stronger representation consistency. The TAU model shows a gradual increase in CKA values through the Translator section, whereas the Ours model maintains consistently higher values, particularly in the Encoder and Trans-

115°E

120°E

125°E

130°E

135°E

140°E

- Figure 6. Comparison of predicted and ground truth tracks of Typhoon Mawar (3-hour lead time). The figure shows the observed trajectory alongside predictions from TAU and Met2Net. Met2Net more closely follows the ground truth, demonstrating improved accuracy in typhoon trajectory forecasting.

frames (representing one day’s data) are used to predict the next 4 frames, simulating a forecast of the following day’s weather based on current data.

High-Resolution. To assess the model’s ability to capture fine-scale meteorological patterns, we conducted an experiment on high-resolution meteorological prediction. Table 4

lator sections. This suggests that the Ours model better preserves representation consistency across variables.

- Table 6. Quantitative comparison on the MvMmfnist. The learning rate was optimized by selecting the best value from {1e−4, 1e−3, 5e−3}.

Method MSE↓ MAE↓ SSIM↑ PSNR↑ SimVP 155.52 397.10 0.8812 19.2825 ConvNeXt 174.97 434.63 0.8682 18.7304 TAU 147.69 385.13 0.8833 19.5075 Met2Net 132.89 328.67 0.9086 19.9069

##### 4.3. General Spatiotemporal Prediction

To evaluate the generalization capability of our model, we conducted experiments on single-variable and multivariable spatiotemporal prediction datasets. TaxiBJ was chosen for single-variable prediction, representing urban traffic flow, while a three-channel MvMmfnist dataset was created for multivariable prediction. These experiments aim to assess the model’s ability to handle diverse data types and spatiotemporal patterns, validating its applicability across different domains.

MvMmfnist. The dataset simulates the complexity of multivariate spatiotemporal prediction scenarios. It contains three channels, each functioning as an independent moving dataset. To better represent real-world scenarios, such as multivariate weather prediction, the channels are interrelated yet retain some independence. Specifically, all channels share the same movement trajectory: the second channel’s moving object is from the Fashion MNIST [60] dataset instead of MNIST [49], while the third channel is generated by flipping the foreground and background of the first channel. This design introduces both unique features and interrelationships among the channels. Experiments on this dataset, with results shown in Table 6, indicate that the Met2Net achieved state-of-the-art performance, demonstrating its effectiveness in multivariate spatiotemporal prediction scenarios. All models were trained using the Adam optimizer for 200 epochs, with the best-performing epoch selected for evaluation, ensuring a fair comparison.

TaxiBJ. Table 7 shows experimental results on the TaxiBJ dataset, where the only difference between our method, Met2Net, and the TAU model is the training strategy, with all other configurations kept identical. Results indicate that Met2Net achieves comparable performance to TAU across all metrics, including MSE, MAE, SSIM, and PSNR. This similarity suggests that our training approach provides advantages in multivariable prediction tasks without compromising performance in single-variable scenarios.

- Table 7. Quantitative comparison on the Taxibj. The subscript S in BaselineS indicates the single-variable model.

Method MSE↓ MAE↓ SSIM↑ PSNR↑ ConvLSTM[48] 0.3358 15.32 0.9836 39.45 PredRNN[53] 0.3194 15.31 0.9838 39.51 PredRNN++[54] 0.3348 15.37 0.9834 39.47 MAU[4] 0.3268 15.26 0.9834 39.52 SimVP[14] 0.3282 15.45 0.9835 39.45 MogaNet[28] 0.3114 15.06 0.9847 39.70 HorNet[45] 0.3186 15.01 0.9843 39.66 TAU[50] 0.3108 14.93 0.9848 39.74 Met2NetS 0.3164 14.82 0.9851 39.81

- Table 8. Ablation study of the proposed method on MSE. MED (Multiple Encoder and Decoder). ITS (Implicit two-stage). The subscript S in BaselineS indicates the single-variable model.

Method UV10 T2M TCC R

BaselineS 1.5925 1.1620 0.0472 31.8310 Baseline 1.8347 2.2247 0.0439 29.0749 +MED 1.8327 1.6708 0.0445 28.8003 +VA(·) 1.6287 1.5432 0.0428 26.6849 +ITS 1.5055 0.8247 0.0422 24.3854

##### 4.4. Ablation Study

Table 8 presents ablation study results for the proposed method in multivariable prediction. Using the TAU model as a baseline for multivariate meteorological forecasting, results indicate that model performance declines when handling multiple variables compared to single-variable predictions. This suggests that complex variable interactions challenge model precision, highlighting the need for optimized multivariable fusion strategies.

Starting from the baseline model, we progressively introduced Multi-Encoder-Decoder (MED), Variable Attention (VA), and Implicit Two-Stage Training (ITS). Results show a consistent MSE decrease across all variables, with substantial improvements after incorporating MED and VA, indicating that these enhancements decouple variable relationships and significantly improve accuracy. Further gains with ITS confirm the effectiveness of our training strategy.

#### 5. Conclusion

In this paper, we proposed a novel implicit two-stage training paradigm to improve spatiotemporal prediction tasks involving multiple meteorological variables. By treating each variable as an independent modality and incorporating separate encoders, decoders, and a self-attention mechanism, our framework, Met2Net, effectively captures complex interactions between variables and enhances prediction accuracy. We also constructed a general multivariate spatiotemporal prediction dataset, demonstrating the robustness and generalization of our method across different tasks.

#### Acknowledgement

This research was partly supported by the Sichuan Science and Technology Achievement Transfer and Transformation Demonstration Project (2024ZHCG0026), the Sichuan Science and Technology Program (2024NSFJQ0035), and the Talents Program supported by the Organization Department of the Sichuan Provincial Party Committee.

#### References

- [1] Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian. Accurate medium-range global weather forecasting with 3d neural networks. Nature, 619(7970): 533–538, 2023. 1, 3
- [2] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 4
- [3] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. Advances in neural information processing systems, 33:9912– 9924, 2020. 3
- [4] Zheng Chang, Xinfeng Zhang, Shanshe Wang, Siwei Ma, Yan Ye, Xiang Xinguang, and Wen Gao. Mau: A motionaware unit for video prediction and beyond. Advances in Neural Information Processing Systems, 34:26950–26962,

2021. 3, 5, 6, 7, 8

- [5] Kang Chen, Tao Han, Junchao Gong, Lei Bai, Fenghua Ling, Jing-Jia Luo, Xi Chen, Leiming Ma, Tianning Zhang, Rui Su, et al. Fengwu: Pushing the skillful global mediumrange weather forecast beyond 10 days lead. arXiv preprint arXiv:2304.02948, 2023. 1, 3
- [6] Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, and Dit-Yan Yeung. Mixed autoencoder for selfsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22742–22751, 2023. 2, 3
- [7] Lei Chen, Xiaohui Zhong, Feng Zhang, Yuan Cheng, Yinghui Xu, Yuan Qi, and Hao Li. Fuxi: A cascade machine learning forecasting system for 15-day global weather forecast. npj Climate and Atmospheric Science, 6(1):190,

2023. 1, 3

- [8] Min Chen, Hao Yang, Shaohan Li, and Xiaolin Qin. Staa: Spatiotemporal alignment attention for short-term precipitation forecasting. IEEE Geoscience and Remote Sensing Letters, 21:1–5, 2024. 1, 2, 3
- [9] Shengchao Chen, Guodong Long, Tao Shen, and Jing Jiang. Prompt federated learning for weather forecasting: Toward foundation models on meteorological data. arXiv preprint arXiv:2301.09152, 2023. 2
- [10] James Douris and Geunhye Kim. The atlas of mortality and economic losses from weather, climate and water extremes (1970-2019). 2021. 1

- [11] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 2
- [12] Ryan Eusebi, Gabriel A Vecchi, Ching-Yao Lai, and Mingjing Tong. Realistic tropical cyclone wind and pressure fields can be reconstructed from sparse data using deep learning. Communications Earth & Environment, 5(1):8, 2024. 1, 2, 3
- [13] Tryambak Gangopadhyay, Sin Yong Tan, Zhanhong Jiang, Rui Meng, and Soumik Sarkar. Spatiotemporal attention for multivariate time series prediction and interpretation. In ICASSP 2021-2021 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 3560–

3564. IEEE, 2021. 5

- [14] Zhangyang Gao, Cheng Tan, Lirong Wu, and Stan Z Li. Simvp: Simpler yet better video prediction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3170–3180, 2022. 3, 5, 6, 7, 8
- [15] Jean-Bastien Grill, Florian Strub, Florent Altch´e, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020. 3
- [16] Vincent Le Guen and Nicolas Thome. Disentangling physical dynamics from unknown factors for unsupervised video prediction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11474– 11484, 2020. 1, 3
- [17] Daehyeon Han, Minki Choo, Jungho Im, Yeji Shin, Juhyun Lee, and Sihun Jung. Precipitation nowcasting using ground radar data and simpler yet better video prediction deep learning. GIScience & Remote Sensing, 60(1):2203363, 2023. 1, 3
- [18] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020. 2, 3
- [19] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 4
- [20] Xiaotao Hu, Zhewei Huang, Ailin Huang, Jun Xu, and Shuchang Zhou. A dynamic multi-scale voxel flow network for video prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6121–6131, 2023. 3
- [21] Yaxuan Huang, Bin Guo, Haoxuan Sun, Huijie Liu, and Song Xi Chen. Relative importance of meteorological variables on air quality and role of boundary layer height. Atmospheric Environment, 267:118737, 2021. 2
- [22] Wenjun Jiang, Bo Liu, Yang Liang, Huanxiang Gao, Pengfei Lin, Dongqin Zhang, and Gang Hu. Applicability analysis of transformer to wind speed forecasting by a novel deep learn-

- ing framework with multiple atmospheric variables. Applied Energy, 353:122155, 2024. 3, 4
- [23] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015. 6
- [24] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jos´e Lezama, Jonathan Huang, Rachel Hornung, Hartwig Adam, Hassan Akbari, Yair Alon, Vighnesh Birodkar, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023. 2, 3, 4
- [25] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International conference on machine learning, pages 3519–3529. PMLR, 2019. 7
- [26] Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran Alet, Suman Ravuri, Timo Ewalds, Zach Eaton-Rosen, Weihua Hu, et al. Learning skillful medium-range global weather forecasting. Science, 382(6677):1416–1421, 2023. 4
- [27] Pengzhi Li, Yan Pei, and Jianqiang Li. A comprehensive survey on design and application of autoencoder in deep learning. Applied Soft Computing, 138:110176, 2023. 2, 3
- [28] Siyuan Li, Zedong Wang, Zicheng Liu, Cheng Tan, Haitao Lin, Di Wu, Zhiyuan Chen, Jiangbin Zheng, and Stan Z Li. Moganet: Multi-order gated aggregation network. In International Conference on Learning Representations, 2024. 8
- [29] Siyuan Li, Zedong Wang, Zicheng Liu, Cheng Tan, Haitao Lin, Di Wu, Zhiyuan Chen, Jiangbin Zheng, and Stan Z Li. Moganet: Multi-order gated aggregation network. In ICLR,

2024. 7

- [30] Siyuan Li, Luyuan Zhang, Zedong Wang, Juanxi Tian, Cheng Tan, Zicheng Liu, Chang Yu, Qingsong Xie, Haonan Lu, Haoqian Wang, and Zhen Lei. Mergevq: A unified framework for visual generation and representation with disentangled token merging and quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 2
- [31] Haitao Lin, Zhangyang Gao, Yongjie Xu, Lirong Wu, Ling Li, and Stan Z Li. Conditional local convolution for spatiotemporal meteorological forecasting. In Proceedings of the AAAI conference on artificial intelligence, pages 7470–7478,

2022. 1

- [32] XuDong Ling, ChaoRong Li, LiHong Zhu, FengQing Qin, Ping Zhu, and Yuanyuan Huang. Spacetime separable latent diffusion model with intensity structure information for precipitation nowcasting. IEEE Transactions on Geoscience and Remote Sensing, 2024. 1
- [33] Xingdou Liu, Li Zhang, Jiangong Wang, Yue Zhou, and Wei Gan. A unified multi-step wind speed forecasting framework based on numerical weather prediction grids and wind farm monitoring data. Renewable Energy, 211:948–963, 2023. 4
- [34] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986,

2022. 5, 6, 7

- [35] Minbo Ma, Peng Xie, Fei Teng, Bin Wang, Shenggong Ji, Junbo Zhang, and Tianrui Li. Histgnn: Hierarchical spatiotemporal graph neural network for weather forecasting. Information Sciences, 648:119580, 2023. 4
- [36] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 2, 3
- [37] Fadji Z Maina, Erica R Siirila-Woodburn, and Pouya Vahmani. Sensitivity of meteorological-forcing resolution on hydrologic variables. Hydrology and Earth System Sciences, 24

(7):3451–3474, 2020. 2

- [38] Haomiao Ni, Changhao Shi, Kai Li, Sharon X Huang, and Martin Renqiang Min. Conditional image-to-video generation with latent flow diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18444–18455, 2023. 2, 3
- [39] Xuesong Nie, Xi Chen, Haoyuan Jin, Zhihang Zhu, Yunfeng Yan, and Donglian Qi. Triplet attention transformer for spatiotemporal predictive learning. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 7036–7045, 2024. 3
- [40] Xuesong Nie, Yunfeng Yan, Siyuan Li, Cheng Tan, Xi Chen, Haoyuan Jin, Zhihang Zhu, Stan Z Li, and Donglian Qi. Wavelet-driven spatiotemporal predictive learning: Bridging frequency and time variations. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4334– 4342, 2024. 3, 5
- [41] Edward Appau Nketiah, Li Chenlong, Jing Yingchuan, and Simon Appah Aram. Recurrent neural network modeling of multivariate time series and its application in temperature forecasting. Plos one, 18(5):e0285713, 2023. 2
- [42] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 3
- [43] Qingyao Qiao, Hamidreza Eskandari, Hassan Saadatmand, and Mohammad Ali Sahraei. An interpretable multi-stage forecasting framework for energy consumption and co2 emissions for the transportation sector. Energy, 286:129499,

2024. 2

- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 4
- [45] Yongming Rao, Wenliang Zhao, Yansong Tang, Jie Zhou, Ser Nam Lim, and Jiwen Lu. Hornet: Efficient highorder spatial interactions with recursive gated convolutions. Advances in Neural Information Processing Systems, 35: 10353–10366, 2022. 5, 6, 7, 8
- [46] Stephan Rasp, Peter D Dueben, Sebastian Scher, Jonathan A Weyn, Soukayna Mouatadid, and Nils Thuerey. Weatherbench: a benchmark data set for data-driven weather forecasting. Journal of Advances in Modeling Earth Systems, 12

(11):e2020MS002203, 2020. 6

- [47] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3, 4
- [48] Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-Kin Wong, and Wang-chun Woo. Convolutional lstm network: A machine learning approach for precipitation nowcasting. Advances in neural information processing systems, 28, 2015. 2, 5, 6, 7, 8
- [49] Nitish Srivastava, Elman Mansimov, and Ruslan Salakhudinov. Unsupervised learning of video representations using lstms. In International conference on machine learning, pages 843–852. PMLR, 2015. 8
- [50] Cheng Tan, Zhangyang Gao, Lirong Wu, Yongjie Xu, Jun Xia, Siyuan Li, and Stan Z Li. Temporal attention unit: Towards efficient spatiotemporal predictive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18770–18782, 2023. 1, 3, 5, 6, 7, 8
- [51] Cheng Tan, Siyuan Li, Zhangyang Gao, Wenfei Guan, Zedong Wang, Zicheng Liu, Lirong Wu, and Stan Z Li. Openstl: A comprehensive benchmark of spatio-temporal predictive learning. Advances in Neural Information Processing Systems, 36:69819–69831, 2023. 1, 6
- [52] Meron Teferi Taye and Ellen Dyer. Hydrologic extremes in a changing climate: a review of extremes in east africa. Current Climate Change Reports, 10(1):1–11, 2024. 2
- [53] Yunbo Wang, Mingsheng Long, Jianmin Wang, Zhifeng Gao, and Philip S Yu. Predrnn: Recurrent neural networks for predictive learning using spatiotemporal lstms. Advances in neural information processing systems, 30, 2017. 2, 5, 6, 7, 8
- [54] Yunbo Wang, Zhifeng Gao, Mingsheng Long, Jianmin Wang, and S Yu Philip. Predrnn++: Towards a resolution of the deep-in-time dilemma in spatiotemporal predictive learning. In International conference on machine learning, pages 5123–5132. PMLR, 2018. 2, 5, 6, 8
- [55] Yunbo Wang, Lu Jiang, Ming-Hsuan Yang, Li-Jia Li, Mingsheng Long, and Li Fei-Fei. Eidetic 3d lstm: A model for video prediction and beyond. In International conference on learning representations, 2018. 3
- [56] Yunbo Wang, Jianjin Zhang, Hongyu Zhu, Mingsheng Long, Jianmin Wang, and Philip S Yu. Memory in memory: A predictive neural network for learning higher-order nonstationarity from spatiotemporal dynamics. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9154–9162, 2019. 3
- [57] Yun Wang, Runmin Zou, Fang Liu, Lingjun Zhang, and Qianyi Liu. A review of wind speed and wind power forecasting with deep neural networks. Applied Energy, 304: 117766, 2021. 4
- [58] Yunbo Wang, Haixu Wu, Jianjin Zhang, Zhifeng Gao, Jianmin Wang, S Yu Philip, and Mingsheng Long. Predrnn: A recurrent neural network for spatiotemporal predictive learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2):2208–2225, 2022. 3

- [59] Jialong Wu, Shaofeng Yin, Ningya Feng, Xu He, Dong Li, Jianye Hao, and Mingsheng Long. ivideogpt: Interactive videogpts are scalable world models. arXiv preprint arXiv:2405.15223, 2024. 2
- [60] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashionmnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017. 8
- [61] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021. 2
- [62] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. Magvit: Masked generative video transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10459–10469, 2023. 2, 3, 4
- [63] Sihyun Yu, Kihyuk Sohn, Subin Kim, and Jinwoo Shin. Video probabilistic diffusion models in projected latent space. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18456–18466,

2023. 2, 3

## Met2Net: A Decoupled Two-Stage Spatio-Temporal Forecasting Model for Complex Meteorological Systems

### Supplementary Material

Algorithm 1 Pseudocode of Implicit Two-Stage Process in a PyTorch-like Style Integrated Within a Inference Pipeline.

# E1, E2: encoders. # D1, D2: decoders. # H: translator.

for x in loader: # load a minibatch x with N samples

- x1, x2 = slice(x) # Slicing Operations in Python

- z1_x = E1(x1) # Independent encoding of Variables

- z2_x = E2(x2)

- z_x = torch.stack(z1_x,z2_x) # Spatio-Temporal learnning and variable fusion

- z_y = H(z_x)

z1_y, z2_y = slice(z_y)

- y1 = D1(z1_y) # Independent decoding of Variables y2 = D2(z2_y) y_pre = torch.stack(y1, y2)

Algorithm 2 Pseudocode of Implicit Two-Stage Process in a PyTorch-like Style Integrated Within a Training Pipeline.

# E1, E2, E1_m, E2_m: encoder that applies gradient updates and momentum updates to two different variables.

# D1, D2, D1_m, D2_m: decoder that applies gradient updates and momentum updates to two different variables.

# H, H_m: translator that gradient updates and

momentum updates. # a: momentum

- E1_m.params = E1.params # initialize

- E2_m.params = E2.params # initialize D1_m.params = D1.params # initialize D2_m.params = D2.params # initialize H_m.params = H.params # initialize

for x,y in loader: # load a minibatch x,y with N samples

- # stage 1

- x1, x2 = slice(x) # Slicing Operations in Python

- z1_x = E1(x1) # Independent encoding of Variables

- z2_x = E2(x2)

- z_x = torch.stack(z1_x,z2_x) # Spatio-Temporal learnning and variable fusion

- z_y = H_m(z_x)

z1_y, z2_y = slice(z_y)

- y1 = D1(z1_y) # Independent decoding of Variables

- y2 = D2(z2_y)

- y_rec = torch.stack(y1, y2) # momentum update

- E1_m.params = a*E1_m.params +(1-a)*E1.params

- E2_m.params = a*E2_m.params +(1-a)*E2.params D1_m.params = a*D1_m.params +(1-a)*D1.params D2_m.params = a*D2_m.params +(1-a)*D2.params loss_rec = MSE(y_rec,y)

# stage 2

- x1, x2 = slice(x) # Slicing Operations in Python

z1_x = E1_m(x1) # use momentum updates module z2_x = E2_m(x2)

- z_x = torch.stack(z1_x, z2_x) # Spatio-Temporal learnning and variable fusion z_y_pre = H(z_x)

z1_y, z2_y = slice(z_y) y1 = D1_m(z1_y) y2 = D2_m(z2_y)

- y_pre= torch.stack(y1, y2)

- y1, y2 = slice(y)

- z1_y = E1_m(y1) # use momentum updates module

- z2_y = E2_m(y2)

- z_y = torch.stack(z1_y, z2_y)

#### 6. Appendix

- 6.1. Experimental Setup for Variable Distributions

Data Distributions. The experiment utilized the 2018 T2M (2-meter air temperature) and TCC (Total Cloud Cover) data from the WeatherBench dataset to analyze spatial and temporal distribution patterns. The data were normalized to the range [0, 1] for comparability. Spatial analysis was based on grid data from a single time step to capture geographic variability, while temporal analysis focused on the time series data of a single grid point.

First-Order Differences. The experiment analyzed firstorder differences of 2018 T2M and TCC data from the WeatherBench dataset. Temporal differences were calculated between steps, and spatial differences from adjacent grid points along the h and w dimensions. Differences were standardized to zero mean and unit variance, with outliers exceeding three standard deviations removed.

- 6.2. Pseudocode of Inference Pipeline

- In Algorithm 1, we present the pseudocode of our method within a inference pipeline. For simplicity, we demonstrate the case with only two variables.

6.3. Pseudocode of Trainning Pipeline

- In Algorithm 2, we present the pseudocode of our method within a training pipeline. For simplicity, we demonstrate the case with only two variables.

# momentum update H_m.params = a*H_m.params +(1-a)*H.params loss_pre = MSE(z_y_pre,z_y) loss = loss_rec + loss_pre # Adam update: query network loss.backward()

##### 6.4. Impact of different blocks in translator

We tested different blocks within the Translator of our framework, as shown in Table 9. The results indicate that while the TAU block remains a competitive choice, our method consistently outperforms the baseline methods across all tested blocks. This demonstrates the robustness of our framework in handling various blocks. Regardless of the block selected, our method maintains superior performance, validating the effectiveness of the proposed framework across different configurations.

Table 9. Impact of using different Blocks in the translator on T2M and TCC prediction performance. The light gray background indicates results not applied in our framework. The white background indicates results obtained using different translators within our framework.

Method T2M TCC

MSE MAE RMSE MSE MAE RMSE

HorNet 1.2010 0.6906 1.0960 0.0469 0.1475 0.2166 TAU 1.1620 0.6707 1.0780 0.0472 0.1460 0.2173 Wast 1.0980 0.6338 1.0440 - 0.1452 0.2150

ConvNext 1.0238 0.6598 1.0105 0.0440 0.1426 0.2096 SimVPv2 0.9215 0.6148 0.9588 0.0425 0.1367 0.2061 PoolFormer 0.9493 0.6271 0.9730 0.0435 0.1426 0.2085 Hornet 0.8778 0.5987 0.9358 0.0423 0.1388 0.2055 Moga 1.0314 0.6643 1.0141 0.0445 0.1455 0.2109 TAU 0.8271 0.5770 0.9094 0.0422 0.1370 0.2054

##### 6.5. Performance evolves with time steps.

As shown in Figure 7, the performance of both models (TAU and our method) varies with the prediction time steps for different variables (T2M and TCC). Although the performance of both models declines as the time step increases, our method consistently outperforms TAU, exhibiting slower growth in MSE and more stable PCC.

(a) T2M on MSE. (b) T2M on PCC.

(c) TCC on MSE. (d) TCC on PCC.

- Figure 7. Performance comparison of T2M and TCC prediction using MSE and PCC across different prediction time steps.

##### 6.6. Single meteorological variables prediction

To validate the applicability and effectiveness of our method, we conducted single-variable prediction experiments. Table 10 presents quantitative comparison results for UV10 and T2M, showing that our method outperforms existing models across all key metrics. Although singlevariable accuracy is lower than multi-variable predictions (Table 1), this highlights the effectiveness of our multivariable fusion approach and the importance of considering multiple variables in meteorological forecasting.

Table 10. Quantitative comparison on the UV10 and T2M variables. The subscript S in Baselines indicates the single-variable model.

|Method<br><br>|UV10 MAE RMSE|T2M MAE RMSE<br><br>|
|---|---|---|
|ConvLSTM PredRNN++ SimVP ConvNeXt TAU Met2NetS|0.9215 1.3775 0.9019 1.3685 0.9510 1.4091 0.8698 1.3006 0.8426 1.2619 0.8197 1.2518<br><br>|0.7949 1.2330 0.7866 1.2070 0.7037 1.1130 0.7220 1.1300 0.6607 1.0780 0.6536 1.0753|

##### 6.7. Additional metrics and resource comparison

We report both the anomaly correlation coefficient (ACC) and the resource consumption of different models on the cropped ERA5 dataset, as presented in Table 11. All experiments are conducted under the same setting with fp32 precision and batch size 16 on a single NVIDIA RTX 4090 GPU.

Met2Net achieves the highest forecasting accuracy across all variables while maintaining moderate parameter count and competitive efficiency in terms of computation and memory usage.

Table 11. ACC and resource comparison on cropped ERA5.

Method Params FLOPs Mem Time MSL U10 V10 T2M

(M) (G) (MiB) (Min) ACC

ConvLSTM 7.44 135.0 4398 2:42 0.9671 0.9073 0.9427 0.9293 MogaNet 12.83 18.9 14408 1:37 0.9690 0.9181 0.9492 0.9533 TAU 12.29 18.3 11942 1:21 0.9652 0.9097 0.9432 0.9510 Met2Net 8.90 119.0 23078 2:24 0.9803 0.9340 0.9590 0.9711

Note: All experiments were conducted on a single NVIDIA RTX 4090 GPU. Mem indicates the peak GPU memory usage with fp32 and a batch size of 16; Time refers to the training time per epoch.

##### 6.8. Scalability under increased variable input

To evaluate the scalability of the proposed method, we expand the number of input meteorological variables in the WeatherL setting by introducing three additional physical fields: total precipitation (TP), geopotential height (Z), and top-of-atmosphere incoming shortwave radiation (TISR). Correspondingly, we increase the encoder–decoder pairs

from 4 to 8, while maintaining the same translator architecture. We compare the forecasting performance of TAU and Met2Net under varying numbers of encoder–decoder pairs. The results are summarized in Table 12.

The attention map reveals several meaningful patterns. For example, the model places strong attention between U10 and V10, and between T2M and TCC, which are physically correlated. This indicates that the translator can adaptively capture variable-specific influences, enhancing both forecasting performance and model interpretability.

Table 12. Forecasting performance with increased variables (UV10 and TCC) under different encoder–decoder configurations.

##### 6.10. Additional tracking tropical cyclones

|Method|# P (M)<br><br>|UV10 TCC MSE RMSE MSE RMSE|
|---|---|---|
| | | |
|TAU1 Met2Net4 TAU8 Met2Net8<br><br>|12.2 8.7 12.2 8.9<br><br>|1.5925 1.2619 0.0472 0.2173 1.5055 1.2270 0.0422 0.2054 1.7345 1.3161 0.0444 0.2107 1.4740 1.2129 0.0417 0.2043<br><br>|

115°E

120°E

125°E

130°E

135°E

140°E

115°E

120°E

125°E

130°E

135°E

140°E

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

Truth

Truth

42°N 42°N

42°N 42°N

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

TAU

TAU

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Ours

Ours

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

37°N 37°N

37°N 37°N

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

32°N 32°N

32°N 32°N

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

27°N 27°N

27°N 27°N

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

The results show that Met2Net maintains superior forecasting accuracy while scaling to more variables, with only a marginal increase in parameter count (from 8.65M to 8.87M). In contrast, TAU exhibits a performance drop despite using the same number of encoders. This demonstrates that Met2Net is well-suited for scalable spatiotemporal modeling in multi-variable settings.

22°N 22°N

22°N 22°N

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

17°N 17°N

17°N 17°N

115°E

120°E

125°E

130°E

135°E

140°E

115°E

120°E

125°E

130°E

135°E

140°E

(a) Predicted and ground truth tracks of Typhoon MAWAR (1-Hour lead time).

(b) Predicted and ground truth tracks of Typhoon DOKSURI (3-Hour lead time).

Figure 9. Tracking tropical cyclones.

##### 6.9. Cross-Variable Attention Analysis

##### 6.11. Additional visualization results

To better understand the inter-variable dependencies captured by the translator module, we analyze the crossvariable attention weights learned during the forecasting process. In our model, each meteorological variable is encoded independently as a token, and the translator performs self-attention over these variable tokens to enable dynamic information aggregation. Figure 8 presents the averaged attention map across all samples and heads. Each row represents the target variable being predicted, and each column indicates the source variable being attended to. The values are normalized attention weights, reflecting how much each variable contributes to the others.

1.0

[Figure 11]

0.37 0.50 0.12 0.01

MSL U10 V10 T2M MSLU10V10T2M

0.8

0.03 0.75 0.13 0.10

0.6

0.4

0.37 0.37 0.25 0.00

0.2

0.25 0.49 0.12 0.14

0.0

- Figure 8. Cross-variable attention map extracted from the translator module. Brighter colors indicate stronger attention. Variables include: MSL, U10, V10, T2M, TP, Z, TISR, and TCC.

[Figure 12]

[Figure 13]

(a) t=1. (b) t=12.

- Figure 10. Visualization of prediction results for different lead times. (a) Results at a forecast time of 1 hour. The background in white represents the absolute error (| GT-Prediction |) for each model. (b) Results at a forecast time of 12 hours.

[Figure 14]

(a) t=1. (b) t=12.

[Figure 15]

- Figure 11. Visualization of prediction results for different lead times. (a) Results at a forecast time of 1 hour. The background in white represents the absolute error (| GT-Prediction |) for each model. (b) Results at a forecast time of 12 hours.

[Figure 16]

(a) t=1. (b) t=12.

[Figure 17]

- Figure 12. Visualization of prediction results for different lead times. (a) Results at a forecast time of 1 hour. The background in white represents the absolute error (| GT-Prediction |) for each model. (b) Results at a forecast time of 12 hours.

[Figure 18]

(a) t=1. (b) t=12.

[Figure 19]

- Figure 13. Visualization of prediction results for different lead times. (a) Results at a forecast time of 1 hour. The background in white represents the absolute error (| GT-Prediction |) for each model. (b) Results at a forecast time of 12 hours.

[Figure 20]

[Figure 21]

(a) t=1. (b) t=10.

- Figure 14. Visualization of prediction results for different lead times on the Mv Mmfnist dataset. The last two columns represent the absolute error (| GT - Prediction |) for each model. (a) Results at a forecast time of 1 frame. (b) Results at a forecast time of 10 frame.

[Figure 22]

(a) t=1. (b) t=10.

[Figure 23]

- Figure 15. Visualization of prediction results for different lead times on the Mv Mmfnist dataset. The last two columns represent the absolute error (| GT - Prediction |) for each model. (a) Results at a forecast time of 1 frame. (b) Results at a forecast time of 10 frame.

[Figure 24]

(a) t=1. (b) t=10.

[Figure 25]

- Figure 16. Visualization of prediction results for different lead times on the Mv Mmfnist dataset. The last two columns represent the absolute error (| GT - Prediction |) for each model. (a) Results at a forecast time of 1 frame. (b) Results at a forecast time of 10 frame.

