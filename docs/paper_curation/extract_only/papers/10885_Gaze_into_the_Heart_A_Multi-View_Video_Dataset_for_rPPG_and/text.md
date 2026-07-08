# arXiv:2508.17924v1[cs.CV]25Aug2025

## Gaze into the Heart: A Multi-View Video Dataset for rPPG and Health Biomarkers Estimation

Konstantin Egorov

Stepan Botman

Pavel Blinov

Sber AI Lab Moscow, Russia egorov.k.ser@sber.ru

Sber AI Lab Moscow, Russia SABotman@sber.ru

Sber AI Lab Moscow, Russia Blinov.P.D@sber.ru

Galina Zubkova

Sber AI Lab Moscow, Russia GVZubkova@sber.ru

Anton Ivaschenko

Samara State Medical University Samara, Russia an.v.ivaschenko@samsmu.ru

Andrey Savchenko

Sber AI Lab ISP RAS Research Center for Trusted Artificial Intelligence Moscow, Russia avsavchenko@hse.ru

Alexander Kolsanov

Samara State Medical University Samara, Russia a.v.kolsanov@samsmu.ru

[Figure 1]

Figure 1: Our multimodal data capture setup (left) and images from three video sources with superimposed facial mesh.

### ABSTRACT

### KEYWORDS

Telemedicine, Video, rPPG, biosignals

Progress in remote PhotoPlethysmoGraphy (rPPG) is limited by the critical issues of existing publicly available datasets: small size, privacy concerns with facial videos, and lack of diversity in conditions. The paper introduces a novel comprehensive large-scale multi-view video dataset for rPPG and health biomarkers estimation. Our dataset comprises 3600 synchronized video recordings from 600 subjects, captured under varied conditions (resting and post-exercise) using multiple consumer-grade cameras at different angles. To enable multimodal analysis of physiological states, each recording is paired with a 100 Hz PPG signal and extended health metrics, such as electrocardiogram, arterial blood pressure, biomarkers, temperature, oxygen saturation, respiratory rate, and stress level. Using this data, we train an efficient rPPG model and compare its quality with existing approaches in cross-dataset scenarios. The public release of our dataset and model should significantly speed up the progress in the development of AI medical assistants.

### 1 INTRODUCTION

Pulse wave analysis is a non-invasive method widely used to evaluate cardiovascular health. Recent advancements in computer vision have enabled the estimation of pulse waves using standard video cameras and ambient light. Remote PhotoPlethysmoGraphy (rPPG) has garnered significant attention for its potential to allow background health monitoring without special medical procedures. This innovative approach is now transitioning from research to practical applications [2], with some technologies already integrated into everyday devices like health monitoring mirrors. These advancements facilitate the early detection of cardiovascular issues in seemingly healthy individuals, highlighting conditions such as elevated blood pressure or cardiac stiffness, which could indicate a higher risk of severe cardiovascular diseases [18].

However, factors such as shooting conditions, lighting, and video duration can significantly impact the accuracy of health parameter estimation via rPPG. Understanding how these variables affect pulse wave extraction is crucial for algorithm training and testing. Existing datasets [17, 21, 24] addressing this issue are often limited in size and lack comprehensive health data, such as temperature,

### CCS CONCEPTS

• Computing methodologies → Biometrics; • Social and professional topics → Remote medicine; Gender; Age.

pulse, photoplethysmogram, electrocardiogram (ECG), and other human biomarkers.

Existing datasets usually contain a medium number of subjects (10-140) recorded in a laboratory or office setting, see Table 1. Only some contain other health biomarkers, like blood pressure [10], or stress levels [21]. Moreover, most known datasets are subject to restricted access and require submitting a request without a guarantee of approval, often due to privacy concerns, proprietary rights, or institutional policies. This limitation impedes collaborative research and slows down the pace of innovation.

Our work aims to advance remote health monitoring by providing a novel publicly available dataset to streamline data handling and model training for further research. Dataset is available on Huggingface platform:

https://huggingface.co/datasets/kyegorov/mcd_rppg Additionally, we provide scripts for all experiments in GitHub

repository: https://github.com/ksyegorov/mcd_rppg In particular, our main contributions can be summarized as fol-

lows:

- (1) Comprehensive Dataset: We introduce the Multi-Camera Dataset for rPPG (MCD-rPPG), a large, diverse, open dataset designed to advance deep learning methods for pulse wave detection and human biomarker estimation. Featuring 600 subjects of varying genders and ages, the dataset includes three-minute multi-view videos recorded in resting states and after physical activity, synchronized with PPG signals, and enriched with 13 additional health biomarkers such as arterial blood pressure, ECG, and heart rate.
- (2) Fast Baseline Model: We introduce an efficient multi-task neural network that estimates pulse waves and other health biomarkers from facial video. The model operates in realtime on a CPU, achieving a speed improvement of up to 13% over leading models, while maintaining competitive accuracy compared to state-of-the-art approaches, even on mobile devices.
- (3) Benchmarking and Comparison: We thoroughly compare our rPPG model with state-of-the-art approaches, test its generalization capabilities in cross-dataset evaluation, and provide a baseline and benchmark for novel tasks, such as estimating various health biomarkers from facial videos.

#### Table 1: Existing open rPPG datasets

name year subjects open link PURE 2014 10 No [23] BP4D+ 2016 140 No [29] COHFACE 2017 40 No [9] LGI-PPGI 2018 25 Yes [20] UBFC-rPPG 2019 42 No [4] UBFC-Phys 2021 56 No [21] SCAMPS 2022 2800 (synthetic) Yes [17] MMPD 2023 33 No [24] VitalVideos 2023 900 No [25] iBVP 2024 33 No [10] MCD-rPPG 2024 600 Yes Ours

#### Table 2: Distribution of anthropometric parameters and biomarkers in the dataset

mean std min max

Weight, kg 65.92 15.79 43.00 168.00 Height, cm 169.75 8.87 147.00 201.00

BMI, kg/m2 22.73 4.34 15.39 47.03 Age, years 23.08 10.90 18.00 83.00

Systolic pressure, mm Hg 122.45 17.43 80.00 202.00 Diastolic pressure, mm Hg 73.79 9.25 50.00 108.00 Saturation, % 98.01 1.29 86.00 99.00 Temperature, ◦ 36.56 0.13 36.00 37.50 Hemoglobin, g/dL 13.59 1.66 8.10 17.30 Glycated hemoglobin, % 5.52 0.69 3.40 13.02 Cholesterol, mmol/L 4.16 0.83 0.90 8.00 Respiratory rate, rpm 18.05 1.71 15.00 24.00 Heart rate, bpm 91.93 18.37 49.00 153.00 Arterial stiffness 8.99 3.04 1.75 34.02 Stress (PSM-25) 3.04 1.46 1.00 7.52

### 2 MCD-RPPG DATASET

Our dataset was created with 600 subjects by simultaneously measuring PPG and ECG signals using medical-grade devices (Eldar and AXMA HemoCard-BT) directly from the subject and recording videos from three different angles using various recording devices (mobile phone, video camera, and webcam). For each subject, two recordings were made: one in the resting state and another after light physical activity (15 squats in 30 seconds). This approach enhances the resulting dataset, which can be used to train more stable models concerning the subject’s physical state. Differences between states can be illustrated by differences in pulse, blood pressure, and respiratory rate distributions, as shown in Fig. 2. A similar logic was applied when deciding whether to include different viewing angles.

Each recording session lasted approximately 3 minutes, during which the video was recorded at standard VGA resolution (640×480) with a frame rate of 24 or 30, depending on the device. The PPG signal was sampled at a rate of 100 Hz. Additional anthropometric and biometric data were gathered before the session, as shown in Table 2, using appropriate medical equipment when necessary (scales REKAM BS 630FT, thermometer Schwabe F01, tonometer AND UA-911BT-C, pulse oximeter Beurer PO 60, blood analyzers EasyTouch and EasyTouch 2, volumetric sphygmography system BPLab Angio). The stress assessment was conducted using the PSM25 psychological stress scale, measured through a questionnaire.

### 2.1 Data collection procedure

The data collection process was organized as follows. At first, the subjects filled out the consent form to participate in the study with medical intervention and the consent form to process data. Then, they were asked to fill out a Psychological Stress Scale PSM-25 (Lemyr-Tessier-Fillion) by indicating the assessment of statements on an 8-point scale.

Next, the subject proceeded to the medical parameter collection stand (Fig. 1), where the main physiological parameters were assessed (body weight, height, temperature, blood pressure, heart rate (HR), oxygen saturation, ECG, blood glucose, glycated hemoglobin, total cholesterol, respiratory rate, and arterial wall stiffness). Body weight was measured using the REKAM BS 630FT floor scale. Blood

300

before

after

| |
|---|

200

Count

100

0

50 100 150 200

(a) Diastolic BP, mm Hg

300

200

Count

100

0

50 100 150 200

(b) Systolic BP, mm Hg

200

150

Count

100

50

0

40 60 80 100 120 140 160

(c) Pulse, bpm

300

200

Count

100

0

14 16 18 20 22 24

(d) Respiratory rate, rpm

- Figure 2: Value distribution of diastolic blood pressure (a), systolic blood pressure (b), pulse (c) and respiratory rate (d), before and after physical activity.

pressure and HR were measured using an automatic tonometer AND UA-911BT-C. ECG was recorded by registering at least three channels of a 12-channel ECG using the ACSMA GemoCard-BT device. Oxygen saturation was determined by indirect oximetry of the index finger using the Beurer PO 60 pulse oximeter. All these devices can remotely transmit data to the database via Bluetooth. The blood glucose, total cholesterol, and glycated hemoglobin levels

were determined by puncturing the index finger pad with a disposable sterile scarifier, then using test strips and blood analyzers EasyTouch (glycated hemoglobin, total cholesterol) and EasyTouch

- 2 (glucose). Body temperature was measured using a non-contact thermometer. Arterial wall stiffness was assessed using the Eldar photoplethysmograph or the system for volumetric sphygmography - BPLab Angio. A physician measured the respiratory rate by counting the number of respiratory movements per minute.

Immediately after that, the subject proceeded to the video and photoplethysmography data collection stand, which was a laptop with three video cameras (mobile phone Samsung A3s, video camera Sony FDR-AX43A, and webcam Defender G-lens 2599) and a photoplethysmograph connected. The subject sat down in front of the cameras, put the photoplethysmograph probe on their finger, and waited motionlessly for three minutes. An electronic clock with a second value was placed behind the subject to synchronize the video. The operator’s task was to accompany the subject at this stand and monitor the lighting and validity of the collected data. After three minutes of recording, the subject performed a physical exercise of 15 squats in 30 seconds, and then all parameter recordings were repeated.

Specialized software was developed to record video and anthropometric and biometric parameters. The software operates as follows. In the user interface, the operator enters the subject’s data and the recording stage (initial or after physical exercises). After that, input data is automatically validated, and the initialization process for video and PPG capturing begins. This process is performed until the internal timer reaches a predefined number of seconds. Then, the operator displays the recorded PPG signal on the screen for visual control. Finally, the program reinitializes for the next recording.

- 3 SYNCHRONIZATION CHALLENGES

Precise time synchronization of data is essential, as it can significantly impact the achievable metrics for training AI models.

### 3.1 Video synchronization

Although time synchronization of video streams from different recording devices is assisted to some extent by software developed for this purpose, timestamps are assigned on the computer side, which does not consider the latency introduced by the recording device and the data transfer to the computer. Thus, we have enriched the data with an additional synchronization channel. As seen in Fig. 1, there is a tablet with a digital clock in the background, visible from all the cameras. There are two major things to take into account:

- • Firstly, the tablet clock is not perfectly synchronized with the computer, which does not prevent us from calculating time shifts between video sources pairwise;
- • Secondly, tablet clock lacks sub-second time resolution, which timestamps have.

The latter issue is solved by employing the following algorithm:

- • Tablet clock data is extracted from each frame using optical character recognition (EasyOCR) and cleansed;
- • For each adjacent pair of frames with different times displayed on the tablet clock, the time shift is calculated as the

- difference between the time displayed in the last frame and half of the sum of the frame timestamps;
- • The record time shift is obtained by taking the average of all the shifts calculated in the previous step;
- • The quality of video stream synchronization is estimated for each pair of video recording devices by calculating the direct difference between their corresponding time shifts.

The results for record time shifts (excluding 226 records, or approximately 6.3% of the total number of records for which OCR failed) are shown in Fig. 3. Here, the distributions for different cameras match quite closely, suggesting that the absolute value is determined by the drift of the tablet’s internal clock. Additionally, it can be shown that the record time shifts for different video recording devices are linearly dependent.

Probabilitydensity

- 0

- 0.5
- 1

- 1.5

cam1 cam2 cam3

−4 −2 0 2 Time shift, s

- Figure 3: Distribution of record time shifts (between frame timestamps and physical clock) estimated using KDE (cam1 is IriunWebcam, cam2 is FullHDwebcam and cam3 is USBVideo).

Fig. 4 shows that the time difference is generally within ±0.2 seconds, constituting only a fraction of the average heartbeat cycle length. Moreover, the proposed method can further refine the temporal data.

### 3.2 PPG synchronization

Considering possible rPPG applications, another important point to consider is the synchronization of video and PPG data. To estimate it, we compared ground truth PPG signal with PPG signals reconstructed from video using the POS algorithm as follows:

- • all signals are filtered using 4-order Chebyshev Type II band-pass filter with cutoff frequencies of 0.4 Hz and 8 Hz;
- • Discrete shift was determined in terms of the optimization problem for maximization of the Pearson correlation coefficient between reconstructed signals and ground truth.

The obtained results are shown in Fig. 5 (326 records, or approximately 9.1% of the total number of records for which the POS

300

cam2 − cam1 cam3 − cam2 cam3 − cam1

| |
|---|

| |
|---|

200

Count

100

0

−0.2 −0.1 0 0.1 0.2 0.3

Time difference, s

#### Figure 4: Distribution of time shift between different video sources (cam1 is IriunWebcam, cam2 is FullHDwebcam and cam3 is USBVideo).

300

cam1 cam2 cam3

| |
|---|

| |
|---|

200

Count

100

0

−20 −10 0 10 20

Optimal shift, frames

#### Figure 5: Distribution of time shift between ground truth PPG and reconstructed PPG (cam1 is IriunWebcam, cam2 is FullHDwebcam and cam3 is USBVideo).

algorithm failed, were excluded). The second set of peaks on the left side is most probably caused by a time shift approaching half a period of the PPG signal, in which case, the determination of the nearest maximum becomes ambiguous. Considering that the frontal facing camera (cam2) shows significantly better synchronization, the subpar results for other cameras are at least partially attributed to the shortcomings of the POS algorithms.

### 4 BASELINE MODEL

Modern rPPG models can be either unsupervised or supervised. Unsupervised methods can be divided into two subgroups: methods

[Figure 2]

#### Figure 6: Overview of our baseline model.

based on reading micro-movements (Ballistocardiographic) [1] and methods based on micro changes in the color of facial pixels [7, 26]. Supervised methods, such as [6, 27, 28], are based on deep learning and let the neural network determine where and what to look at. Such methods are more accurate and better predict the PPG shape, but require datasets for training and often are subject to overfitting. This can be seen in the results of [15], where unsupervised POS [26] and OMI [5] models showed results that were more robust across datasets compared to supervised approaches, as well as in [11], where all supervised models showed a significant drop in quality during cross-dataset evaluation.

In this regard, we decided to take a hybrid approach relying on domain-specific pre-processing, followed by processing with a specialized neural network adapter, similar to [14]. Because the network has fewer parameters, it is much less susceptible to overfitting while maintaining high quality and the ability for fine-tuning and domain adaptation, if necessary. Additionally, adding new targets like arterial pressure or respiratory rate to neural networks is possible without significantly altering their architecture.

For a domain-specific pre-processing tool, we used ideas from [7, 26]. It is essential to consider the anatomical properties of the blood supply to the face [22] and detect those areas of the face with the greatest influence of the pulse wave, significantly reducing the noise level in the signal. We select the face region-of-interest (ROI) using a technique from [12], allowing the neural network to choose the signal of different ROIs in the required proportions.

The final model architecture is shown in Fig. 6. First, we detect a face and highlight the ROI using the FaceMesh model from the mediapipe [16]. We thenselecta setof multipleregions and compute the mean pixel value in each ROI for each frame. The received signals are fed into a neural network, producing a PPG signal and medical parameter predictions. The model is a fully convolutional 1-dimensional feature pyramid network [13]. It allows the neural networks to operate on different lengths of the input signal without slicing it in a sliding window manner. The resulting pipeline is very fast and surpasses even unsupervised models in inference time on GPU and CPU. The proposed model is a good foundation for analyzing an extended set of biomarkers. While the ROI-based approach is indeed common in rPPG, our main contribution is developing a blazing-fast model capable of working on small devices

like phones and wearables while being on par with high-capacity models in terms of accuracy.

### 5 EXPERIMENTS

We trained our model by splitting videos and PPG signals into 20second windows and feeding them to the neural network. The main target of the training was to predict the PPG signal. Still, the dataset had additional targets: systolic and diastolic arterial pressure, glycated hemoglobin, cholesterol, respiratory rate, arterial stiffness, age, sex, BMI, stress level, and saturation. All targets were normalized with a standard scaler on the training subset, and the loss function was the sum of mean squared error (MSE) losses for each target. For optimization, we utilized the Adam optimizer. We used the mean average error (MAE) of HR for each 10-second segment as our primary metric. To determine HR, we use the algorithm [19]. Table 3 presents the PPG and HR estimation results. Two metrics are shown: the mean average error (MAE) of the predicted PPG signal and the MAE of HR estimation. HR is predicted by selecting the most powerful frequency of 0.5-3 Hz of the predicted PPG signal. The training procedure and all hyperparameters necessary for repeating the results can be obtained from the repository.

The result of the biomarkers estimation is presented in Table 5. Our model performs better than the straightforward baseline, the optimal constant value fitted on a training subset.

One of the goals of gathering this dataset was to research the impact of different circumstances and camera parameters on the quality of remote medical scanning. For this purpose, we provide multi-view videos from three sources and our experimental results. First, as seen in Fig. 1, three camera views cover the face of the patient differently, so we tested the drop in quality. The results are presented in Table 4. We can see the pattern where unsupervised methods, while strong in cross-dataset generalization, cannot adapt to different view angles, and neural networks generally perform better in these scenarios.

Table 4 also shows our model’s inference time and size advantage. Table speed advantage on CPU is 13% better than the previous best model. Time was measured by running 200 20-second video segments sequentially. However, the results in terms of model MAE are not straightforward and show multidirectional trends. This is consistent with independent benchmarks like rPPG-Toolbox [15] and Remote Biosensing Benchmark [11], where different models

#### Table 3: Comparison of performance of different models (MAE). With bold font, we highlight the best-performing model on each test dataset. With underline font, we highlight the best-performing model, which is not trained on the test dataset in question. The last line describes the performance of our model trained on the MCD-rPPG dataset.

Train MCD-rPPG MCD-rPPG MMPD MMPD SCAMPS SCAMPS UBFC-rPPG UBFC-rPPG Model dataset PPG (Ours) HR (Ours) PPG HR PPG HR PPG HR

|PBV[8]<br><br>|-<br><br>|0.85|15.37<br><br>|0.80|37.11<br><br>|0.88<br><br>|35.93|0.86|30.27|
|---|---|---|---|---|---|---|---|---|---|
|OMIT[5]<br><br>|-|0.80<br><br>|4.78|0.77<br><br>|15.33|0.86<br><br>|16.27|0.84|1.95<br><br>|
|POS[26]|-<br><br>|0.87|3.80<br><br>|1.08|15.36<br><br>|1.41|16.02<br><br>|1.52|1.17<br><br>|
|PhysFormer[28]<br><br>|MMPD SCAMPS UBFC-rPPG MCD-rPPG|0.89±0.02<br><br>1.10±0.00<br><br><br>0.81±0.01 0.46±0.01<br><br>|13.57±1.40 46.38±0.00 43.65±3.71 4.08±0.12<br><br>|0.82±0.02<br><br>0.97±0.00 0.76±0.01<br><br>0.98±0.03<br>|22.67±1.21 30.01±0.00 42.09±7.21 22.61±0.51<br><br>|0.90±0.02 0.13±0.00<br><br>0.84±0.02<br><br>1.08±0.03<br>|28.45±0.75 0.40±0.00 49.77±4.70 23.33±1.29<br><br>|0.79±0.04 0.67±0.00<br><br>0.77±0.01<br>1.30±0.06<br>|8.79±5.96 1.17±0.00 37.40±0.98 1.27±1.51<br><br>|
|iBVPNet[10]<br><br>|MMPD SCAMPS UBFC-rPPG MCD-rPPG|0.99±0.02<br><br>1.07±0.01<br><br><br>0.96±0.03 0.68±0.01<br><br>|36.28±6.10 59.48±1.93 35.31±7.24 4.83±0.44|0.87±0.01<br><br>1.01±0.00<br><br><br>0.86±0.01<br><br>1.01±0.01<br><br><br>|21.02±0.20 24.19±0.67 23.53±3.92 17.12±0.51|0.96±0.01 0.54±0.00<br><br>0.90±0.02<br><br>1.05±0.00<br><br><br>|34.60±2.04 1.63±0.08 31.36±2.17 25.74±0.23|0.93±0.01<br><br>0.84±0.03<br><br>0.85±0.01<br><br><br>1.21±0.01<br><br><br>|21.63±5.16 7.91±2.62 7.47±2.21 4.69±0.71|
|RhythmFormer[30]|MMPD<br><br>SCAMPS<br><br>UBFC-rPPG<br><br>MCD-rPPG<br><br>|0.94±0.02<br>1.05±0.01 0.87±0.01 0.43±0.00<br>|17.54±4.30 58.58±8.14 19.14±1.93 2.82±0.13<br><br>|0.77±0.00<br>1.00±0.01 0.82±0.01 0.98±0.01<br>|17.28±2.21 26.08±0.44 24.81±0.69 16.63±0.96<br><br>|0.91±0.01 0.08±0.00<br><br>0.99±0.01<br>1.06±0.02<br>|30.34±1.00 0.20±0.10 28.53±0.43 21.45±1.65<br><br>|0.85±0.01<br><br>0.75±0.03<br><br>0.86±0.02<br><br>1.46±0.02<br><br><br>|11.52±2.80 10.25±2.68 21.53±3.29 2.39±0.35<br><br>|
|Ours (Fig. 6)<br><br>|MMPD SCAMPS UBFC-rPPG MCD-rPPG<br><br>|1.32±0.02 1.20±0.01 1.17±0.05 0.68±0.03|7.58±1.3 41.17±4.30 23.09±6.67 4.86±0.36<br><br>|0.94±0.01<br>1.06±0.03 1.01±0.02 1.20±0.02<br>|15.26±0.49 34.44±3.81 27.93±3.41 17.53±0.68<br><br>|1.16±0.02 0.50±0.04<br><br>0.92±0.08<br>1.27±0.05<br>|27.77±1.75 10.60±0.77 37.46±0.62 20.20±1.39<br><br>|0.96±0.12 0.87±0.03<br><br>0.77±0.02<br>1.42±0.08<br>|2.39±0.46 13.62±3.57 6.84±5.27 4.13±0.92<br><br>|

#### Table 4: Performance for different camera views

Speed of Inference (ms) Metrics (MAE) CPU GPU Size Frontal Side Frontal Side

Model seconds seconds Mb PPG PPG HR HR

PBV 0.18 0.18 0 0.85 0.86 15.37 40.44 OMIT 0.17 0.17 0 0.80 1.11 4.78 22.35 POS 0.26 0.26 0 0.87 1.25 3.80 16.40 PhysFormer 0.93 0.31 28.4 0.46 0.97 4.08 10.68 RhythmFormer 0.97 0.33 12.9 0.43 0.91 2.82 7.33 iBVPNet 0.93 0.28 5.5 0.68 0.99 4.83 11.42 Ours 0.15 0.16 3.9 0.68 1.10 4.86 14.01

#### Table 5: Metric for predicting biomarkers compared to a naive baseline (all metrics are MAE, except for Sex)

Target Baseline Model Systolic pressure, mm Hg 13.78 12.82 Diastolic pressure, mm Hg 7.50 8.39 Glycated hemoglobin, % 0.43 0.41 Cholesterol, mmol/L 0.66 0.60 Respiratory rate, rpm 1.36 1.20 Arterial stiffness 2.19 2.04 Age, years 5.71 3.91 BMI 3.18 3.37 Stress (PSM-25) 1.20 1.07 Saturation, % 0.98 0.98 Sex (Accuracy) 0.61 0.64

excel in different training and testing setups. We believe it is due to the limited sizes of all available datasets. This comparison confirms our success in developing small and fast models that achieved performance metrics comparable to competitive models.

### 6 CONCLUSION

In this paper, we introduce a unique large MCD-rPPG dataset for rPPG and health biomarkers estimation. The dataset contains 3minute video recordings from 600 subjects of different genders and ages filmed in two states (quiet and after exercise), aligned with 13 biomarkers. A photoplethysmograph was synchronized with multiple webcams to ensure proper parameter alignment, enabling pulse wave detection from facial video recordings. Additionally, we developed a fast, lightweight multitask rPPG model (Fig. 6) trained on our dataset, which relies on domain-specific pre-processing.

We believe the public release of our dataset and model will accelerate progress in AI-driven medical assistants [3] and multimedia applications (e.g., emotion-aware telemedicine, stress monitoring via video calls, or fitness tracking via smartphone cameras), which rely on robust estimation of pulse waves and health biomarkers from everyday recording devices like mobile phones and webcams.

### ACKNOWLEDGMENTS

The work of A. Savchenko was supported by a grant, provided by the Ministry of Economic Development of the Russian Federation in accordance with the subsidy agreement (agreement identifier 000000C313925P4G0002) and the agreement with the Ivannikov Institute for System Programming of the Russian Academy of Sciences dated June 20, 2025 No. 139-15-2025-011.

### REFERENCES

- [1] Guha Balakrishnan, Fredo Durand, and John Guttag. 2013. Detecting Pulse from Head Motions in Video. In 2013 IEEE Conference on Computer Vision and Pattern Recognition. 3430–3437. doi:10.1109/CVPR.2013.440
- [2] Rohan Banerjee, Anirban Dutta Choudhury, Aniruddha Sinha, and Aishwarya Visvanathan. 2014. HeartSense: smart phones to estimate blood pressure from photoplethysmography. In Proceedings of the 12th ACM Conference on Embedded Network Sensor Systems. 322–323.
- [3] Pavel Blinov, Konstantin Egorov, Ivan Sviridov, Nikolay Ivanov, Stepan Botman, Evgeniy Tagin, Stepan Kudin, Galina Zubkova, and Andrey V Savchenko.

- 2024. GigaPevt: multimodal medical assistant. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence. 8614–8618.
- [4] Serge Bobbia and et al. 2019. Unsupervised skin tissue segmentation for remote photoplethysmography. Pattern Recognition Letters 124 (2019), 82–90. doi:10. 1016/j.patrec.2017.10.017
- [5] Constantino Alvarez Casado and Miguel Bordallo López. 2023. Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces. IEEE Journal of Biomedical and Health Informatics 27, 11 (2023), 5530–5541.
- [6] Weixuan Chen and Daniel McDuff. 2018. DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks. In Computer Vision – ECCV 2018, Vittorio Ferrari, Martial Hebert, Cristian Sminchisescu, and Yair Weiss (Eds.). Springer International Publishing, Cham, 356–373.
- [7] Gerard de Haan and Vincent Jeanne. 2013. Robust Pulse Rate From ChrominanceBased rPPG. IEEE Transactions on Biomedical Engineering 60 (2013), 2878–2886.
- [8] G de Haan and A van Leest. 2014. Improved motion robustness of remote-PPG by using the blood volume pulse signature. Physiological Measurement 35, 9 (aug 2014), 1913. doi:10.1088/0967-3334/35/9/1913
- [9] Guillaume Heusch, André Anjos, and Sébastien Marcel. 2017. A reproducible study on remote heart rate measurement. arXiv preprint arXiv:1709.00962 (2017).
- [10] Jitesh Joshi and Youngjun Cho. 2024. iBVP Dataset: RGB-Thermal rPPG Dataset with High Resolution Signal Quality Labels. Electronics 13, 7 (2024). doi:10.3390/ electronics13071334
- [11] Dae Yeol Kim and et al. 2023. Remote Bio-Sensing: Open Source Benchmark Framework for Fair Evaluation of rPPG. arXiv:2307.12644 [eess.IV]
- [12] Dae-Yeol Kim, Kwangkee Lee, and Chae-Bong Sohn. 2021. Assessment of ROI Selection for Facial Video-Based rPPG. Sensors 21, 23 (2021). doi:10.3390/s21237923
- [13] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. 2017. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition. 2117–2125.
- [14] Xin Liu and et al. 2021. Efficientphys: Enabling simple, fast and accurate camerabased vitals measurement. arXiv preprint arXiv:2110.04447 (2021).
- [15] Xin Liu and et al. 2022. rPPG-Toolbox: Deep Remote PPG Toolbox. arXiv preprint arXiv:2210.00716 (2022).
- [16] Camillo Lugaresi and et al. 2019. MediaPipe: A Framework for Perceiving and Processing Reality. In Third Workshop on Computer Vision for AR/VR at IEEE Computer Vision and Pattern Recognition (CVPR) 2019.
- [17] Daniel McDuff, Miah Wander, Xin Liu, Brian Hill, Javier Hernandez, Jonathan Lester, and Tadas Baltrusaitis. 2022. Scamps: Synthetics for camera measurement of physiological signals. Advances in Neural Information Processing Systems 35

(2022), 3744–3757.

- [18] Riccardo Miotto and et al. 2018. Reflecting health: smart mirrors for personalized medicine. NPJ digital medicine 1, 1 (2018), 62.
- [19] N. H. Mohd Sani and et al. 2015. Determination of heart rate from photoplethysmogram using Fast Fourier Transform. In 2015 International Conference on BioSignal Analysis, Processing and Systems (ICBAPS). 168–170. doi:10.1109/ ICBAPS.2015.7292239
- [20] Christian S. Pilz, Sebastian Zaunseder, Jarek Krajewski, and Vladimir Blazek.

2018. Local Group Invariance for Heart Rate Estimation from Face Videos in the Wild. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). 1335–13358. doi:10.1109/CVPRW.2018.00172

- [21] Rita Meziati Sabour, Yannick Benezeth, Pierre De Oliveira, Julien Chappé, and Fan Yang. 2023. UBFC-Phys: A Multimodal Database For Psychophysiological Studies of Social Stress. IEEE Transactions on Affective Computing 14, 1 (2023), 622–636. doi:10.1109/TAFFC.2021.3056960
- [22] V. A. Sinopalnikov and V. M. Zemskov. 2018. Telemonitoring of Capillary Blood Flow in the Human Skin: New Opportunities and Prospects.
- [23] Ronny Stricker, Steffen Müller, and Horst-Michael Gross. 2014. Non-contact video-based pulse rate measurement on a mobile service robot. In The 23rd IEEE International Symposium on Robot and Human Interactive Communication. 1056–1062. doi:10.1109/ROMAN.2014.6926392
- [24] Jiankai Tang, Kequan Chen, Yuntao Wang, Yuanchun Shi, Shwetak Patel, Daniel McDuff, and Xin Liu. 2023. Mmpd: multi-domain mobile video physiology dataset. In 2023 45th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). IEEE, 1–5.
- [25] Pieter-Jan Toye. 2023. Vital Videos: A dataset of videos with PPG and blood pressure ground truths. arXiv preprint arXiv:2306.11891 (2023).
- [26] Wenjin Wang, Albertus C. den Brinker, Sander Stuijk, and Gerard de Haan.

2017. Algorithmic Principles of Remote PPG. IEEE Transactions on Biomedical Engineering 64, 7 (2017), 1479–1491. doi:10.1109/TBME.2016.2609282

- [27] Zitong Yu, Xiaobai Li, and Guoying Zhao. 2019. Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks. In British Machine Vision Conference.
- [28] Z. Yu, Y. Shen, J. Shi, H. Zhao, P. Torr, and G. Zhao. 2022. PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE Computer Society, Los Alamitos, CA, USA, 4176–4186. doi:10.1109/CVPR52688.2022.00415

- [29] Zheng et al. Zhang. 2016. Multimodal Spontaneous Emotion Corpus for Human Behavior Analysis. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 3438–3446. doi:10.1109/CVPR.2016.374
- [30] Bochao Zou, Zizheng Guo, Jiansheng Chen, Junbao Zhuo, Weiran Huang, and Huimin Ma. 2024. RhythmFormer: Extracting Patterned rPPG Signals based on Periodic Sparse Attention. arXiv preprint arXiv:2402.12788 (2024).

