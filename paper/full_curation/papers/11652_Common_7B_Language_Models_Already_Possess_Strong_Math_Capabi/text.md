# arXiv:2403.04706v1[cs.CL]7Mar2024

## Common 7B Language Models Already Possess Strong Math Capabilities

### Chen Li1,4, Weiqi Wang2,4, Jingcheng Hu3,4, Yixuan Wei3,4,

Nanning Zheng1, Han Hu4, Zheng Zhang4*, Houwen Peng4* 1IAIR, Xi’an Jiaotong University 2University of Science and Technology of China 3Tsinghua University 4Microsoft Research Asia

edward82@stu.xjtu.edu.cn {v-weiqiwang, t-jingchu, t-yixuanwei, zhez, houwen.peng}@microsoft.com nnzheng@xjtu.edu.cn ancientmooner@gmail.com

### Abstract

Mathematical capabilities were previously believed to emerge in common language models only at a very large scale or require extensive math-related pre-training. This paper shows that the LLaMA-2 7B model with common pretraining already exhibits strong mathematical abilities, as evidenced by its impressive accuracy of 97.7% and 72.0% on the GSM8K and MATH benchmarks, respectively, when selecting the best response from 256 random generations. The primary issue with the current base model is the difficulty in consistently eliciting its inherent mathematical capabilities. Notably, the accuracy for the first answer drops to 49.5% and 7.9% on the GSM8K and MATH benchmarks, respectively. We find that simply scaling up the SFT data can significantly enhance the reliability of generating correct answers. However, the potential for extensive scaling is constrained by the scarcity of publicly available math questions. To overcome this limitation, we employ synthetic data, which proves to be nearly as effective as real data and shows no clear saturation when scaled up to approximately one million samples. This straightforward approach achieves an accuracy of 82.6% on GSM8K and 40.6% on MATH using LLaMA-2 7B models, surpassing previous models by 14.2% and 20.8%, respectively. We also provide insights into scaling behaviors across different reasoning complexities and error types.

### 1 Introduction

Mathematical capabilities have long been considered so challenging that they are thought to emerge in common language models only at a very large scale. For instance, studies by (Wei et al., 2022a,b) suggest that only models with size exceeding 50

*Project leader. Chen, Weiqi, Jingcheng and Yixuan are interns at MSRA. GitHub: Xwin-Math This repository will be continually updated.

[Figure 1]

Figure 1: The orange star markers represent the accuracy achieved by selecting the best response from 256 random generations of the LLaMA-2 7B model. The high accuracy on the MATH (left) and GSM8K (right) benchmarks (72.0% and 97.7%, respectively) suggest that the LLaMA-2 7B already possesses strong mathematical capabilities, although the stability in generating correct answers could be enhanced. This paper demonstrates that by scaling synthetic SFT data, the stability can be significantly improved as evidenced by the curves. Through this straightforward scaling of SFT data, the top-performing model has exceeded an early GPT-4 model by 10.3% on the MATH benchmark.

billion parameters can attain meaningful accuracy or benefit from chain-of-thought processing on math problems. A strategy to equip smaller language models with mathematical abilities involves creating math-specific base models trained on hundreds of billions of math-related pre-training data (Lewkowycz et al., 2022; Azerbayev et al., 2023). However, the accuracy of such models remains modest; for example, Llemma-7B (Azerbayev et al., 2023) only achieves 36.4% on the GSM8K dataset (Cobbe et al., 2021) and 18.0% on the MATH dataset (Hendrycks et al., 2021).

In this paper, we demonstrate that common language models of small size, such as the LLaMA-2 7B model (Touvron et al., 2023b), already possess strong mathematical capabilities without specific pre-training on math-related data. Surprisingly, we find that with supervised fine-tuning on just thousands of math questions (noting that the SFT stage does not enhance capabilities as stated in

- Table 1: Comparison of SFT data scaling with real versus synthetic math questions. It reveals that synthetic math questions are nearly as effective as real ones.

Data size GSM8K-real GSM8K-syn MATH-real MATH-syn

- 0.94K 26.7 25.9 4.2 3.9

- 1.88K 32.8 31.9 5.6 4.9 3.75K 43.3 42.2 6.6 6.0 7.50K 50.2 49.5 8.4 7.9

(Bai et al., 2022; Ouyang et al., 2022)), the model can correctly solve 97.7% of GSM8K questions and 72.0% of MATH questions, when selecting the best answer from 256 random generations, as indicated by the orange star marks in Figure 1. It is noteworthy that the accuracy has even outperformed those reported for the GPT-4 model, which achieved 92.0% on GSM8K and 42.5% on MATH

- 1. Therefore, we conclude that the LLaMA-2 7B model has indeed developed strong mathematical capabilities. The primary issue is the lack of guarantee that the correct answer will be digged out, as most generations are incorrect. In fact, the accuracy drops to 49.5% on GSM8K and 7.9% on MATH if we consider only one random generation per question. We refer to this as the instability issue.

To address the instability issue, we first observe that the accuracy improves almost in linear or even super-linear with exponentially increased supervised fine-tuning (SFT) data. Moreover, we note that the accuracy is far from reaching a plateau when utilizing all available GSM8K and MATH training data (as shown in Table 1). This observation encourages us to further scale up the SFT data. However, we face a challenge as there is a lack of publicly accessible real data to support this continuous scaling.

To overcome this limitation, we turn to synthetic data, employing a prestigious language model, namely GPT-4 Turbo, to produce synthetic math questions. We find that a straightforward “brandnew” generation strategy, which prompts the GPT-4 Turbo to create a completely new question based on preference ones and then applies a simple verifier (also GPT-4 Turbo based), has been highly effective. Specifically, as indicated in Table 1, the use of synthetically generated math questions can

1The accuracy numbers are reported in the GPT-4 technical report (OpenAI, 2023b). GPT-4 models are continuously being improved. The latest GPT-4 Turbo (1106) API has increased accuracy to 94.8% on GSM8K and 64.5% on MATH. However, the LLaMA-2 7B model using the best of 256 generations still outperforms the latest GPT-4 models.

achieve accuracy nearly on par with that of real questions, highlighting the potential of synthetic SFT math questions for the scaling purpose.

Leveraging synthetic data has allowed us to scale our SFT data significantly, for instance, from 7.5K to 960K on GSM8K and from 7.5K to 480K on MATH. This data scaling shows nearly perfect scaling behavior, as drawn in Figure 1. Specifically, by simply scaling the SFT data, our model has become the first to exceed 80% and 40% accuracy on GSM8K and MATH, respectively, using a standard LLaMA-2 7B base model (achieving 82.6% and 40.6% respectively)2.

The straightforward synthetic SFT data proves effective from stronger base models as well, such as LLaMA-2 70B, which achieves 90.6% on GSM8K and 52.8% on MATH. To the best of our knowledge, this is the first open-source model to exceed 90% accuracy on GSM8K. It is also the first open-source model to outperform GPT-4 (i.e., GPT-4-0314) on the MATH benchmark, demonstrating the efficacy of our simple synthetic scaling method.

In addition to the strong results, we have also gleaned insights into the effectiveness of our approach: 1) As the scale of SFT data increases, the model’s accuracy tends to plateau when utilizing 256 attempts; however, there is a marked increase using 1 response. This indicates that while the model’s upper capability limit remains fairly constant, the performance gains are primarily due to enhanced stability in generating correct answers. 2) The accuracy of solving math problems follows a power law with respect to the number of chain-ofthought (CoT) steps with different SFT data quantities. An expanded SFT dataset improves the reliability of each reasoning step. Further increasing the proportion of training samples with longer CoT steps through resampling can significantly improve the accuracy of the model for difficult questions. 3) An analysis of error types during the scaling process reveals that calculation errors are more readily mitigated compared to reasoning errors.

### 2 Examine Math Capability of Language Models

Metrics We employ two metrics to examine the math capabilities of language models.

2Concurrently, DeepSeek-MATH-7B (Shao et al., 2024) also surpasses 80% accuracy. However, their approach relies on a much stronger base model extensively pre-trained on math-related corpora and a sophisticated RL algorithm. Our results are complementary to theirs.

The first is a Pass@N metric Pass@N = E

###### [min(c,1)], (1)

Problems

where c represents the number of correct answers out of N responses. This metric considers a question to be solved if at least one correct answer is produced from N random generations. We employ this metric to reflect the potential or capability of a model in solving a math question. To enhance the diversity of the N generations, we set the temperature of the generation process to 0.73.

The second is a PassRatio@N metric PassRatio@N = E

c N

, (2)

Problems

which measures the percentage of correct answers within the N generated answers. This metric is somewhat equivalent to Pass@1, but with reduced variance.

Observations Based on these two metrics, we examine the performance of the LLaMA-2 models on the GSM8K and the MATH benchmarks4 as shown in Figure 1. To adapt models for these two benchmarks in instruction-following settings, we use their SFT versions, which are trained with a limited amount of SFT data (i.e., 7.5K). As demonstrated in (Bai et al., 2022; Ouyang et al., 2022), the SFT stage does not enhance capabilities (and may even lead to a reduction, as mentioned in the context of “alignment taxes”). Therefore, employing the SFT version provides a fair assessment of the models’ mathematical capabilities.

We first observe that the Pass@256 metrics for the LLaMA-2 7B model on both benchmarks are remarkably high: 97.7% on GSM8K and 72.0% on MATH. This suggests that the LLaMA-2 7B model possesses a strong capability for solving mathematical problems.

We then notice that the PassRatio@256 is significantly lower than that of Pass@256, being 48.2% on GSM8K and 7.9% on MATH. This suggests that while the correct answers to most math questions are present within 256 random generations, there is no assurance that the correct answers will consistently be extracted, a phenomenon we refer to as an "instability issue".

- 3It is worth noting that most math models utilize a greedy generation strategy with the temperature set to 0. However, the impact of this difference is minimal.
- 4Following (Lightman et al., 2023), we utilize a subset of 500 test samples from the MATH benchmark for experimental efficiency.

In the following, we will present a simple approach to significantly reduce the instability issue.

### 3 Scaling SFT Data using Synthetic Math Questions

In this section, we first demonstrate that scaling up the limited real SFT data can significantly alleviate the instability issue. We also observe that the accuracy has not yet plateaued when using the full available GSM8K and MATH training data. We consider further scaling up SFT data using synthetic math questions. To this aim, we introduce a straight-forward method for synthetic data generation utilizing the GPT-4 Turbo API. The synthetic data proves to be as effective as real math questions. Consequently, we boldly scale the synthetic SFT data to 960K on GSM8K and 480K on MATH, respectively, resulting in nearly perfect scaling behavior, and reach state-of-the-art accuracy.

Scaling using Real Math Questions We begin by examining the scaling behavior of real math questions across the entire GSM8K and MATH training sets. As indicated in Table 1, we observe a consistent accuracy improvement, increasing from 26.7% to 50.2% on GSM8K, and from 4.2% to 8.4% on MATH, with no signs of saturation.

Synthetic SFT Data Generation Since the real data has been exhausted, we contemplate further scaling up SFT data using synthetically generated math questions.

We introduce a straightforward three-step approach with the assistance of the GPT-4 Turbo API:

- • Step 1. Generate a new math question. We request the GPT-4 Turbo API to generate a brand-new question using a reference math question as a starting point. To improve the validity of the new questions, we incorporate three rules into the prompt: Firstly, the new question must obey common knowledge; secondly, it should be solvable independently of the original question; and thirdly, it must not include any answer responses. Besides, we have set specific formatting requirements for questions and answers tailored to various target datasets.

- • Step 2. Verify the question. We further enhance the quality of the generated questions by validating and refining them through attempted solutions. By integrating solving and

verification steps into a single prompt, we have found that this approach consistently elevates the validity of questions across different benchmarks.

• Step 3. Generate chain-of-thought (CoT) answers. We request GPT-4 Turbo to produce a chain-of-thought (CoT) answer response for each newly generated question.

The detailed prompt designs are shown in Appendix A.

Comparison of Synthetic SFT Data versus Real Data To assess the quality of the synthetically generated math questions, we evaluate their effectiveness against real questions from the GSM8K and MATH training sets, utilizing a LLaMA-2 7B model, as detailed in Table 1. The results indicate that the synthetic math questions are nearly as effective as the real ones.

We also explored various other synthetic methods as proposed in previous works (Xu et al., 2023; Yu et al., 2023; An et al., 2023). These methods also prove to be effective, though marginally less so than the our approach, as illustrated in Figure 6.

Scaling to about a Million SFT Math Data Considering the effectiveness of the synthetic approach, we substantially increase the scale of the SFT data for both GSM8K and MATH problems, to 960K and 480K, respectively. Figure 1 presents the main reasults utilizing various sizes of the LLaMA-

- 2 series. The straightforward scaling strategy yields state-of-the-art accuracy.

It is also worth noting that the accuracy has not yet reached its peak. Exploring the effects of additional scaling will be left as our future research.

### 4 Experiments

4.1 Datasets and Evaluations We conduct experiments on 5 benchmarks to evaluate the efficacy of the proposed method.

GSM8K (Cobbe et al., 2021). This is a highquality, linguistically diverse math dataset, whose math knowledge mainly covers grade school level. It includes 7,473 training examples and 1,319 test cases. In this work, we use its training set as the given questions to generate new synthetic data.

MATH (Hendrycks et al., 2021). This dataset focuses on competitive-level math problems that requires high levels of reasoning ability and mathematical knowledge. It consists of 7,500 training

examples and 5,000 test cases. We use the training examples to generate synthetic data.

SVAMP (Patel et al., 2021). This dataset comprises elementary-level math problems. We utilize all 1,000 of its test cases to assess the cross-dataset performance of our models.

ASDiv (Miao et al., 2021). This dataset contains a set of math problems with diverse language patterns and types of questions. We adopt the test set of 2,305 problems as evaluation benchmark.

Hungarian National High School Exam This evaluation benchmark is first introduced by Grok1 (xAI, 2023), which is designed for evaluating the out-of-domain capability of math models. It consists of 33 challenging problems.

It is worth noting that the final answers of Hungarian National High School Exam dataset is annotated by human, while other benchmarks are labelled using automatic scripts, similar to previous works (Luo et al., 2023; Gou et al., 2023).

##### 4.2 Implementation Details

In data synthesis, we utilize the GPT-4 Turbo API, setting the temperature to 1.0 for both question and answer generation.

For supervised fine-tuning, we employ the Adam optimizer with a cosine learning rate schedule spanning a total of 3 epochs of training. The maximum learning rate is set 2e-5 (except that 2e-6 for the Mistral-7b model) and there is a 4% linear warmup. The maximum token length is set 2048, and the Vicuna-v1.1 (Zheng et al., 2023) system prompt is used. All experiments are conducted on 8×Nvidia H100 GPUs. Our most resource-intensive experiment, involving a 70B model and 960K data points, takes 1900 H100 GPU hours.

For evaluation, we use the same prompt as used in SFT and set the maximum sequence length to 2048. The vLLM (Kwon et al., 2023) is used in answer generation.

##### 4.3 Main Results and Comparison with State-of-the-art Models

In this comparison, we examine both in-domain benchmarks, GSM8K/MATH, and out-of-domain benchmarks, such as the Hungarian National High School Exam. For in-domain evaluation of each benchmark, we utilize data synthesized from its respective training samples. For GSM8K, 960K synthetic data is employed, while for MATH, 480K synthetic data is used. For out-domain evaluation,

- Table 2: Math reasoning performances of various LLMs.

Model GSM8K MATH

Closed-source models GPT-4 Turbo (1106) 94.8 64.5 GPT-4-0314 94.7 52.6 GPT-4 (Achiam et al., 2023) 92.0 42.5 Claude-2 (Anthropic, 2023) 88.0 GPT-3.5-Turbo (OpenAI, 2023a) 80.8 34.1

Open-source models LLaMA-2-7B WizardMath-7B (Luo et al., 2023) 54.9 10.7 MuggleMath-7B (Li et al., 2023) 68.4 MetaMath-7B (Yu et al., 2023) 66.5 19.8 LEMA-LLaMA-2-7B (An et al., 2023) 54.1 9.4 Xwin-Math-7B (ours) 82.6 40.6

Open-source models Mistral-7B

WizardMath-7B-v1.1 (Luo et al., 2023) 83.2 33.0 MetaMath-Mistral-7B (Yu et al., 2023) 77.4 28.2 Xwin-Math-Mistral-7B (ours) 89.2 43.7

Open-source models Llemma-7B

MetaMath-Llemma-7B (Yu et al., 2023) 69.2 30.0 Xwin-Math-Llemma-7B (ours) 84.2 47.2

Open-source models LLaMA-2-13B WizardMath-13B (Luo et al., 2023) 63.9 14.0 MuggleMath-13B (Li et al., 2023) 74.0 MetaMath-13B (Yu et al., 2023) 72.3 22.4 LEMA-LLaMA-2-13B (An et al., 2023) 65.7 12.6 Xwin-Math-13B (ours) 88.1 44.9

Open-source models LLaMA-2-70B WizardMath-70B (Luo et al., 2023) 81.6 22.7 MuggleMath-70B (Li et al., 2023) 82.3 MetaMath-70B (Yu et al., 2023) 82.3 26.6 LEMA-LLaMA-2-70B (An et al., 2023) 83.5 25.0 Xwin-Math-70B (ours) 90.6 52.8

we test models trained using GSM8K, MATH, or a mixed of two synthetic sets.

For base models, we consider both common language models, i.e., LLaMA-2 7B/13B/70B/Mistral7B, and math-specific models, such as Llemma-7B, to assess the generality of the proposed approach.

In-Domain Results Table 2 presents a comparison of the proposed approach with the state-ofthe-art open and closed-source models. Across all base models, our method significantly outperforms the previous best approaches that use the same pretrained base model.

On LLaMA-2-7B, our approach exceeds the prior best by absolutely +14.2 on GSM8K (compared to MuggleMath-7B (Li et al., 2023)), and by +20.8 on MATH (compared to MetaMath-7B (Yu

- et al., 2023)), respectively. It even surpasses several latest 70B models dedicated for math capabilities, such as WizardMath-70B (Luo et al., 2023) (82.6 versus 81.6 on GSM8K). On LLaMA-2-13B, the

Table 3: Hungarian national high school exam test result of various LLMs.

Model Test Score (%)

GPT-4 (Achiam et al., 2023) 68 Grok-1 (xAI, 2023) 59 Claude-2 (Anthropic, 2023) 55 GPT-3.5 Turbo (OpenAI, 2023a) 41 DeepSeek-LLM-67B-Chat (Bi et al., 2024) 58

Xwin-Math-70B (480K GSM8K) 22 Xwin-Math-70B (120K MATH) 51 Xwin-Math-70B (480K MATH) 59 Xwin-Math-70B (480K Mix) 65

improvements are +14.1 on GSM8K (compared to MuggleMath-13B (Li et al., 2023)) and +22.5 on MATH (compared to MetaMath-13B (Yu et al., 2023)), respectively. On LLaMA-2-70B, the gains are +7.1 on GSM8K (compared to LEMA-LLaMA2-70B (An et al., 2023)) and +26.2 on MATH (compared to MetaMath-70B (Yu et al., 2023)), respectively.

On a stronger common language model, i.e., Mistral-7B, the improvements are +6.0 on GSM8K and +10.7 on MATH (compared to WizardMath7B-v1.1 (Luo et al., 2023)), respectively.

On a math-specific base model, such as Llemma7B, the gains are +15.0 on GSM8K and +17.2 on MATH (compared to MetaMath-Llemma-7B (Luo et al., 2023)), respectively.

It is also noteworthy that our LLaMA-2-70B model achieves competitive accuracy with early versions of GPT-4 on GSM8K and MATH. To our knowledge, this is the first LLaMA-based model to outperform GPT-4-0314 on MATH.

These results demonstrate the significant effectiveness and broad applicability of scaling synthetic math SFT data.

Out-of-Domain Results We test the models trained using GSM8K, MATH, or a mixed of two synthetic sets on an out-of-domain benchmark, Hungarian National High-School Exam Test, following the practice in (xAI, 2023).

Table 3 shows the results. Our model trained on the mixing data (240K MATH synthetic data + 240K GSM8K synthetic data) ranked as the second, just behind the GPT-4 and much better than other models. Additionally, we plot the correlation between GSM8K and Hungarian national high-school exam scores in Appendix B. The results show that there is no significant benchmark overfitting in our model.

Figure. 2 (Left) presents the results of the model

[Figure 2]

Figure 2: Comparing the increase in SFT data scale using either a single dataset or mixed datasets.

[Figure 3]

- Figure 3: The Pass@256 and PassRatio@256 curve with increasing data size on GSM8K and MATH benchmark.

trained on GSM8K synthetic data, while Figure. 2 (Middle) presents the results of the model trained on MATH. We find that the accuracy of other benchmarks also improves as the amount of data increases for models trained with either GSM8K or MATH synthetic data. We also note that the generalization behaviors differ for GSM8K and MATH models: 1) SVAMP and ASDiv benefit more from GSM8K models than from MATH models. 2) While MATH models perform relatively well on the GSM8K benchmark, GSM8K models perform considerably worse on MATH benchmarks.

Figure. 2 (Right) shows the results of models using a mixture of GSM8K and MATH in a 1:1 ratio. These models exhibit balanced scaling behaviors in both in-domain and out-of-domain benchmarks.

4.4 What Happens behind Performance Improvements?

Pass@256 v.s. PassRatio@256 To deepen the understanding behind the performance improvements, we tracked Pass@N metric and PassRatio@N metric under different data size. The results are shown in Figure 3. With very limited synthetic data (e.g. 7.5K samples), the Xwin-Math-70B model already has very high Pass@256, indicating the strong ability to generate correct answers through multiple

attempts. Meanwhile, the Pass@256 metric only changed slightly with increasing the amount of used data. In contrast, PassRatio@256, which reflects the stability to generate correct answer, increases significantly with the amount of synthetic data, and its growth trend is similar to that of Pass@1. This result confirms our hypothesize that the performance improvements is mainly caused by the better stability in answer generation rather than stronger ability to answer the question.

Estimated Single-step Reasoning Accuracy Because of the Chain-of-Thought (CoT) are adopted in inference, the process of answer mathematical problems is completed by a multi-step reasoning process. Therefore, we hypothesize that the increase in final answer accuracy can be interpreted by the improvement in single-step reasoning accuracy. Based on this assumption, if one question can be theoretically answered by s reasoning steps in CoT, the final answer accuracy can be approximate by the power function of the single-step reasoning accuracy:

Accfinal = Accsstep (3)

With this equation, step accuracy can be estimated from the final answer accuracy. We experimented on GSM8K. For each question in the test

[Figure 4]

- Figure 4: Left: The relationship between the mean accuracy on the GSM8K and the number of annotated CoT steps with data increasing. The solid line is fitted using all seven points, while the dashed line is fitted using the first four points. Right: Changes in mean accuracy when resampling is used to increase the CoT lengeh of training data.

set, we generated 256 responses and used the number of steps in the GSM8k test set’s CoT annotations as the theoretical CoT steps. We draw the curve to show the relationship between the number of CoT reasoning steps and mean final answer accuracy and show the fitted curve based on Equation. 3. We test Xwin-Math-7B models with different synthetic data, and the results are shown in Figure 4. The solid line is fitted using all seven points and Table 4 shows the estimated single-step accuracy when using different amounts of data using all data points, and it can be seen that the single-step accuracy improve significantly with more data.

However, when we fit based on Equation. 3 to the first four points, as shown in dashed lines, we found that the latter three points were significantly below the curve. We believe this phenomenon may be related to the smaller proportion of more complex problems in the training data. Therefore, we resampled the 960K synthetic data according to the number of sentences in CoT solution. As can be seen from Figure 4 (right), when the proportion of complex problems is increased, the accuracy for simpler problems remains virtually unchanged, but the accuracy for more complex problems can be significantly improved. Moreover, the utilization of data resampling can increase the model’s PassRatio@256 from 71.1 to 72.8. This experimental result provides new insights into data selection for mathematical reasoning tasks.

In addition, we further used the GPT-4 Turbo to find the position where the first step in our answer was wrong and normalized that position by the total number of steps in each answer. As the estimated single-step accuracy gets higher, the first

Table 4: The estimated single-step reasoning accuracy and the average normalized first error position by GPT-4 Turbo in Xwin-Math-7B on GSM8K benchmark.

Data size Estimated Accstep Normalized first error position

7.5K 78.9 67.1 120K 89.7 83.9 960K 94.2 90.9

[Figure 5]

Figure 5: Changes in the proportion of calculation and reasoning mistake during data increased.

error position of the normalization is postponed.

The Improvement in the Accuracy of Numerical Calculations is More Significant than Logical Reasoning The performance of the model gradually improves as the synthetic data increases. For a deeper understanding, we analyze the error proportion for different types of errors on GSM8K. We categorized errors into two types: reasoning errors and calculation errors. Reasoning errors primarily encompass issues such as loss of conditions and concept confusion, while calculation errors include incorrect analysis of quantitative relationships and numerical computation mistakes. Based on the experimental results illustrated in Figure 5, we observe a gradual decrease in the percentage of calculation errors, suggesting that GSM8K is correcting calculation errors at a faster rate than reasoning errors.

##### 4.5 Ablations on the Data Synthetic Schema

Comparison with Other Data Synthetic Methods We compared our approach with following common used data synthetic methods:

Add Constraint. Adding one more constrain to the original question while keeping others unchanged, which is used in WizardMath and MuggleMath.

Change Numbers. Changing the numbers that appear in the problem while keeping the context intact. which is used in MuggleMath.

Change Background. Changing the background in

[Figure 6]

- Figure 6: GSM8K and MATH performance of different synthetic methods.

- Table 5: Ablation of question verification on MATH.

Model Pass@1 (%) Xwin-Math-70B (7.5K data) 28.9 Xwin-Math-70B (7.5K data) w/o verification 28.1 (-0.8) Xwin-Math-70B (30K data) 37.6 Xwin-Math-70B (30K data) w/o verification 36.6 (-1.0)

the question while keeping others the same.

The Combination of Changing Numbers and Background. A hybrid approach that combines changing both numbers and background.

MetaMath Approach. The synthetic methods proposed in MetaMath, including answer augmentation, rephrasing question, self-verification question and FOBAR question. In experiments, we follow the implementation of MetaMath but use GPT-4 Turbo instead of GPT-3.5 Turbo to generate response data using their released questions.

The experimental results in the Figure 6 show that when the data size is relatively small, e.g., 7.5k and 30k samples, the performance gap between the different methods is negligible. However, as the data size increases, our method and the method with added constraints show stronger performance. This suggests that the choice of data synthetic strategy becomes more critical as the data size increases, and that some methods can scale the data more efficiently, thus improving the performance.

Effects of Question Verification. The question verification is used to further improve the generation quality. In our experiments, we found it can improve the performance on MATH benchmark, the results are shown in Table 5, while we do not see significantly impact on GSM8K dataset.

### 5 Related Works

Large Language Models Large language models (Brown et al., 2020; Achiam et al., 2023; Touvron et al., 2023a,b) have made significant achieve-

ments, with impressive performance on a wide range of tasks. Currently, closed-source large language models, represented by GPT (Brown et al., 2020; Achiam et al., 2023), Gemini (Team et al., 2023), Grok (xAI, 2023), and Claude-2 (Anthropic,

- 2023), are the most advanced models in terms of performance. However, open-source models, represented by LLaMA (Touvron et al., 2023a), LLaMA2 (Touvron et al., 2023b) and Mixtral (Jiang et al.,
- 2024), have also progressed rapidly, and have even shown competitive performance with the closedsource models on some tasks. Our work, which aims to improve the performance of open-source LLMs on mathematical tasks by fine-tuning them on synthetic data.

Reasoning Framework for Improving Mathematical Capability Chain-of-thoughts (Wei et al., 2022b) encourages the LLMs perform multistep reasoning by specific designed prompts and can improve reasoning performance. Based on this work, many subsequent works suggesting further improvements (Fu et al., 2022; Zhang et al.,

- 2022; Kojima et al., 2022). The above works focus primarily on how to improve performance through better prompt design or inference strategies without fine-tuning the model, whereas our work focuses on how to improve the model itself, and thus these approaches are complementary to ours.

Fine-tuned LLM for Math Another sort of works (Lightman et al., 2023; Luo et al., 2023; Azerbayev et al., 2023; Yue et al., 2023; Yu et al.,

- 2023; An et al., 2023; Li et al., 2023; Gou et al.,

2023) try to improve performance directly by training the model on mathematical data. A direct way is to use fine-tuning to improve models. One widely used method is to use synthetic data, which is very close to our approach: MetaMath (Yu et al., 2023) presents to bootstrap questions to augment data. LeMA (An et al., 2023) collects mistake-correction data pairs by using GPT-4 as a corrector. And MuggleMath (Li et al., 2023) augments the GSM8K dataset by incorporating GPT-4 with a series of predefined operations. Compared to these synthetic data based efforts, our data synthetic method is much simpler and more scalable due to introduce less prior and constraint.

SFT Data Scaling Recently, some research efforts have focused on the data scale for supervised fine-tuning. For instance, LIMA (Zhou et al., 2023) mentions that fine-tuning with 1,000 high-quality

instructions can yield impressive results in various general tasks. Other studies have indicated that performance scales with data size in mathematical and coding tasks (Dong et al., 2023). Recent work (Bi

- et al., 2024) even uses 1.5 million data for instruct fine-tuning to obtain top performance. However, the intrinsic reasons behind this scaling effect have not been thoroughly investigated.

- 6 Conclusion This study reveals that common 7B language models, such as LLaMA-2 7B, already exhibit strong mathematical capabilities, challenging the previous belief that advanced mathematical reasoning is exclusive to larger, more extensively pre-trained models. By significantly scaling up SFT data, we have markedly improved the stability of the model’s mathematical problem-solving skills. Our methodology has enabled the Xwin-Math models to reach performance levels comparable to, and in some instances surpassing, those of their larger counterparts. Our analysis also indicates that the enhancements are primarily attributable to heightened accuracy in single-step reasoning and a extra resampling of training data can improve the accuracy of harder questions. Additionally, we see more substantial reduction of calculation errors as opposed to logical reasoning errors. Our research contributes valuable insights into the mathematical capabilities of large language models. Acknowledgments

Chen Li and Nanning Zheng were supported in part by NSFC under grant No. 62088102. Thank Shengnan An at IAIR, Xi’an Jiaotong University for his valuable advice on this work.

### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou, and Weizhu Chen. 2023. Learning from mistakes makes llm better reasoner. arXiv preprint arXiv:2310.20689.

Anthropic. 2023. Model card and evaluations for claude models.

Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q Jiang, Jia Deng, Stella Biderman, and Sean Welleck. 2023. Llemma: An open language model for mathematics. arXiv preprint arXiv:2310.10631.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback.

Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. 2024. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Guanting Dong, Hongyi Yuan, Keming Lu, Chengpeng Li, Mingfeng Xue, Dayiheng Liu, Wei Wang, Zheng Yuan, Chang Zhou, and Jingren Zhou. 2023. How abilities in large language models are affected by supervised fine-tuning data composition. arXiv preprint arXiv:2310.05492.

Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. 2022. Complexitybased prompting for multi-step reasoning. arXiv preprint arXiv:2210.00720.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yujiu Yang, Minlie Huang, Nan Duan, Weizhu Chen, et al. 2023. Tora: A tool-integrated reasoning agent for mathematical problem solving. arXiv preprint arXiv:2309.17452.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. 2022. Solving quantitative reasoning problems with language models.

Chengpeng Li, Zheng Yuan, Guanting Dong, Keming Lu, Jiancan Wu, Chuanqi Tan, Xiang Wang, and Chang Zhou. 2023. Query and response augmentation cannot help out-of-domain math reasoning generalization. arXiv preprint arXiv:2310.05506.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. arXiv preprint arXiv:2305.20050.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. 2023. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583.

Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. 2021. A diverse corpus for evaluating and developing english math word problem solvers. arXiv preprint arXiv:2106.15772.

- OpenAI. 2023a. Gpt-3.5 turbo fine-tuning and api updates.
- OpenAI. 2023b. GPT-4 technical report. CoRR, abs/2303.08774.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback.

Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are nlp models really able to solve simple math word problems? arXiv preprint arXiv:2103.07191.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. 2022a. Emergent abilities of large language models.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022b. Chain-ofthought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837.

xAI. 2023. Grok-1.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. 2023. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2023. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2023. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv:2309.05653.

Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. 2022. Automatic chain of thought prompting in large language models. arXiv preprint arXiv:2210.03493.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mtbench and chatbot arena.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. 2023. Lima: Less is more for alignment. arXiv preprint arXiv:2305.11206.

### A Synthetic Prompt on GSM8K

##### Prompt 1: Question Generation

Please act as a professional math teacher. Your goal is to create high quality math word problems to help students learn math. You will be given a math question. Please create a new question based on the Given Question and following instructions. To achieve the goal, you have three jobs. # Please generate a similar but new question according to the Given Question. # Check the question by solving it step-by-step to find out if it adheres to all principles. # Modify the created question according to your checking comment to ensure it is of high quality. You have five principles to do this. # Ensure the new question only asks for one thing, be reasonable, be based on the Given Question, and can be answered with only a number (float or integer). For example, DO NOT ask, ‘what is the amount of A, B and C?’. # Ensure the new question is in line with common sense of life. For example, the amount someone has or pays must be a positive number, and the number of people must be an integer. # Ensure your student can answer the new question without the given question. If you want to use some numbers, conditions or background in the given question, please restate them to ensure no information is omitted in your new question. # Please DO NOT include solution in your question. # If the created question already follows these principles upon your verification. Just keep it without any modification. Given Question: given question Your output should be in the following format: CREATED QUESTION: <your created question> VERIFICATION AND MODIFICATION: <solve the question step-by-step and modify it to follow all principles>

FINAL CREATED QUESTION: <your final created question>

##### Prompt 2: Answer Generation

Please act as a professional math teacher. Your goal is to accurately solve a math word problem. To achieve the goal, you have two jobs. # Write detailed solution to a Given Question. # Write the final answer to this question. You have two principles to do this. # Ensure the solution is step-by-step. # Ensure the final answer is just a number (float or integer). Given Question: given question Your output should be in the following format: SOLUTION: <your detailed solution to the given question>

FINAL ANSWER: <your final answer to the question with only an integer or float number>

##### Prompt 3: Question Generation w/o verification

Please act as a professional math teacher. Your goal is to create high quality math word problems to help students learn math. You will be given a math question. Please create a new question based on the Given Question and following instructions. To achieve the goal, you have one job. # Please generate a similar but new question according to the Given Question. You have four principles to do this. # Ensure the new question only asks for one thing, be reasonable, be based on the Given Question, and can be answered with only a number(float or integer). For example, DO NOT ask, ‘what is the amount of A, B and C?’. # Ensure the new question is in line with common sense of life. For example, the amount someone has or pays must be a positive number, and the number of people must be an integer. # Ensure your student can answer the new question without the given question. If you want to use some numbers, conditions or background in the given question, please restate them to ensure no information is omitted in your new question. # You only need to create the new question. Please DO NOT solve it. Given Question: given question Your output should be in the following format:

CREATED QUESTION: <your created question>

### B Additional Results

- Figure 7: Xwin-Math’s aggregate performance on these two benchmarks is second only to GPT-4, demonstrating our model’s robust generalization capabilities.

Hungarian Exam Score vs GSM8K Score

| |Xwin-Ma|th-70B (480|K GSM8K)| |GPT-4|
|---|---|---|---|---|---|
| | | |Clau<br><br>Xwin|de-2<br><br>-Math-70B|(480K Mix)|
| | |D<br><br>Xwin-Ma|eepSeek-LLM-<br><br>th-70B (480|67B-Chat<br><br>K MATH)| |
| |Xwin-M|ath-70B (1|20K MATH)| | |
| | | | | | |
| | | | | | |
| | | | | |Grok-1|
| | | |GPT-3.5 Tur|bo| |
| | | | | | |

90

85

GSM8KScore(%)

80

75

70

65

60

20 30 40 50 60 70

Hungarian Exam Score (%)

- Table 6: To validate benchmark data leakage during data generation, we compare LM loss on: 1) a training subset, which is a small subset with 256 samples from all synthetic data; 2) a regenerated training subset, where we maintain the original questions from the training subset and use GPT-4 Turbo to rewrite answers; 3) a regenerated test set, where we keep the questions from the evaluation metrics unchanged and use GPT-4 Turbo to rewrite answers; 4) a reference test set, where we use the test set as seed to generate new questions and answers via GPT-4 Turbo. Referring to Skywork, we also report two key metrics: ∆1 = Ltest-regen − Ltest-ref,∆2 = Ltest-regen − Ltrain-regen, As ∆1 is close to 0 and ∆2 is significantly greater than 0 in two benchmarks, we believe that there is no leakage during the process of data synthesis.

Dataset Ltest-regen Ltest-ref Ltrain Ltrain-regen ∆1 ∆2

GSM8K 0.52 0.50 0.11 0.33 0.02 0.19 MATH 0.59 0.58 0.23 0.39 0.01 0.20

### C Case Study of Synthetic Data

##### C.1 GSM8K Original Question

Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?

##### Synthetic Question w/o Verification

- Maria has 80 apple trees in her orchard. She also has 4 less than a third as many cherry trees as apple trees. How many trees does Maria have in all in her orchard?

Verification Comment Let’s check the created question:

- - Maria has 80 apple trees.
- - She has 4 less than a third as many cherry trees as apple trees. A third of the apple trees is 80/3, which is approximately 26.67.
- - Four less than that would be around 22.67, but since we cannot have a fraction of a tree, we round it to the nearest whole number, which is 27.
- - The modified question should ask for the number of cherry trees without the decimal. So we adjust the question to have a whole number as a result.
- - Since 80 is not divisible by 3, to get a whole number of cherry trees, we can change the number so that it is divisible by 3. Let’s use 81 apple trees instead of 80.

Synthetic Question with Verification

- Maria has 81 apple trees in her orchard. She also has 4 less than a third as many cherry trees as apple trees. How many trees does Maria have in all in her orchard?

##### Synthetic Solution and Final Answer

First, let’s determine the number of cherry trees Maria has. We are told she has 4 less than a third as many cherry trees as apple trees. Since Maria has 81 apple trees, let’s find one third of that number: 1/3 of 81 apple trees = 81 apple trees / 3 = 27 apple trees. Now, since she has 4 less than this amount in cherry trees, we subtract 4 from 27 to find the number of cherry trees: 27 cherry trees - 4 = 23 cherry trees. To find the total number of trees Maria has in her orchard, we add the number of apple trees to the number of cherry trees: 81 apple trees + 23 cherry trees = 104 trees. The answer is

|104|
|---|

.

- C.2 MATH Original Question The value of log10 28471 is between the consecutive integers a and b. Find a + b.

Synthetic Question w/o Verification The value of log2 16384 is between the consecutive integers x and y. Find x + y. Verification Comment

To ensure the question can be solved, we need to check if the logarithm given can be evaluated and falls between two consecutive integers.

- 1. Identify the value of 2n that equals 16384.
- 2. If 16384 is a power of 2, then log2 16384 will be an integer.
- 3. Check to ensure the result is an integer and find the consecutive integers around it. Let’s perform the check: 20 = 1, 21 = 2, 22 = 4, 23 = 8, 24 = 16

... 214 = 16384 So, log2 16384 = 14. The consecutive integers around 14 are (13 and 14) or (14 and 15). Since the logarithm gives an exact integer, to make the question valid, we will modify it to use a number that is not an exact power of 2. MODIFIED CREATED QUESTION: The value of log2 10000 is between the consecutive integers m and n. Find m + n. Now, let’s check the modified question:

- 1. Since 10000 is not a power of 2, log2 10000 will not be an integer.
- 2. The goal is to find two consecutive integers between which this logarithm value lies.
- 3. Knowing that 213 = 8192 and 214 = 16384, log2 10000 is between 13 and 14. The consecutive integers around log2 10000 are 13 and 14.

##### Synthetic Question with Verification The value of log2 10000 is between the consecutive integers m and n. Find m + n. Synthetic Solution and Final Answer

To find the consecutive integers between which log2 10000 lies, we need to find the powers of 2 that are immediately less than and greater than 10000. First, let’s find the largest power of 2 that is less than 10000. Starting with 21,22,... and so on, we find: 210 = 1024, 211 = 2048, 212 = 4096, 213 = 8192, 214 = 16384 Here, we see that 213 < 10000 < 214, which means that log2 10000 is greater than 13 but less than 14.

Therefore, the consecutive integers m and n between which log2 10000 lies are m = 13 and n = 14. Now we will add these two integers to find the required sum: m + n = 13 + 14 = 27. The answer is

|27|
|---|

.

### D Case Study of Model Response D.1 GSM8K

##### Question and Ground Truth Solution

Question: A treasure hunter found a buried treasure chest filled with gems. There were 175 diamonds, 35 fewer rubies than diamonds, and twice the number of emeralds than the rubies. How many of the gems were there in the chest?

Solution: There were 175 - 35 = 140 rubies in the treasure chest. There were 140 x 2 = 280 emeralds in the treasure chest. So, the total number of gems in the treasure chest is 175 + 140 + 280 =

|595|
|---|

.

##### Response – Xwin-Math-7B (7.5k data) Greedy Decode

First, let’s find out the number of rubies. We are told there are 35 fewer rubies than diamonds. Since there are 175 diamonds, we can calculate the number of rubies as follows: Number of rubies = Number of diamonds - 35 = 175 - 35 = 140 Next, we need to find out the number of emeralds. We are told there are twice the number of emeralds than the rubies. So we can calculate the number of emeralds as follows: Number of emeralds = 2 × Number of rubies = 2 × 140 = 280 Now, we can find the total number of gems by adding the number of diamonds, rubies, and emeralds together: Total number of gems = Number of diamonds + Number of rubies + Number of emeralds =

175 + 140 + 280 = 605 The answer is:

|605|
|---|

. Incorrect Wrong Calculation

Response – Xwin-Math-7B (7.5k data) One Sample from Temperature=0.7

Let’s find the number of each type of gem in the chest step by step. First, we are given the number of diamonds: Number of diamonds = 175 Next, we are told there are 35 fewer rubies than diamonds. To find the number of rubies, we subtract 35 from the number of diamonds: Number of rubies = Number of diamonds - 35 = 175 - 35 = 140 Then, we are told there are twice the number of emeralds than the rubies. To find the number of emeralds, we multiply the number of rubies by 2: Number of emeralds = 2 * Number of rubies = 2 * 140 = 280 Now, we add the number of diamonds, rubies, and emeralds together to find the total number of gems in the chest: Total number of gems = Number of diamonds + Number of rubies + Number of emeralds = 175 + 140 + 280 = 595 The answer is:

|595|
|---|

. Correct

Response – Xwin-Math-7B (960k data) Greedy Decode

Let’s break down the information given in the problem step by step.

- 1. We know there are 175 diamonds.
- 2. There are 35 fewer rubies than diamonds. Since there are 175 diamonds, we can calculate the number of rubies by subtracting 35 from 175. Number of rubies = Number of diamonds - 35 = 175 - 35 = 140
- 3. There are twice the number of emeralds than the rubies. We can find the number of emeralds by multiplying the number of rubies by 2. Number of emeralds = 2 × Number of rubies = 2 × 140 = 280 Now, to find the total number of gems in the chest, we add the number of diamonds, rubies, and emeralds together. Total number of gems = Number of diamonds + Number of rubies + Number of emeralds = 175 + 140 + 280 = 595

|595|
|---|

#### . Correct

The answer is:

D.2 MATH Question and Ground Truth Solution Question: If f(x) = 3xx−−22, what is the value of f(−2) + f(−1) + f(0)? Express your answer as a common fraction. Solution: f(−2)+f(−1)+f(0) = 3(−−22)−−22+3(−−11)−−22+3(0)0−−22 = −−84+−−35+−−22 = 2+53+1 = Response – Xwin-Math-7B (7.5k data) Greedy Decode

|14 3<br><br>|
|---|

To find the value of f(−2) + f(−1) + f(0), we need to evaluate the function f(x) at these three values of x. First, let’s find f(−2): f(x) = 3xx−−22 Plug in x = −2: f(−2) = 3(−−22)−−22 = −−6−42 = −−84 Now, let’s find f(−1): Plug in x = −1: f(−1) = 3(−−11)−−22 = −−3−32 = −−53 Finally, let’s find f(0): Plug in x = 0: f(0) = 3(0)0−−22 = −−22 = 1 Now, we can find the sum: f(−2) + f(−1) + f(0) = −−84 + −−53 + 1 To add these fractions, we need to find a common denominator, which is the least common multiple (LCM) of the three denominators (-4, -2, and -3) . The LCM of these denominators is -12. Now, we can rewrite each fraction with the common denominator:

−8 −4 = 34 , −5

−3 = 35, 1 = 1212 Now, we can add the fractions: 43 + 35 + 1212 = 4+5+123 = 193 The answer is:

Unreasonable

|19 3<br><br>|
|---|

. Incorrect

Wrong calculation

Response – Xwin-Math-7B (7.5k data) One Sample from Temperature=0.7 To find the value of f(−2) + f(−1) + f(0), we first need to evaluate each function at these specific values of x. For f(−2): f(x) = 3xx−−22 Substitute x = −2 into the equation: f(−2) = 3(−−22)−−22 = −−6−42 = −−84 =

- 2 For f(−1): f(x) = 3xx−−22 Substitute x = −1 into the equation: f(−1) = 3(−−11)−−22 = −−3−32 = −−53 = 5

- 3 For f(0): f(x) = 3xx−−22 Substitute x = 0 into the equation: f(0) = 3(0)0−−22 = −−22 = 1 Now we can add the values of f(−2), f(−1), and f(0): f(−2) + f(−1) + f(0) = 2 + 53 + 1 = 143

|14 3<br><br>|
|---|

. Correct

The answer is:

Response – Xwin-Math-7B (480k data) Greedy Decode

To find the value of f(−2) + f(−1) + f(0), we need to evaluate the function f(x) at each of the given points and then sum the results.

First, let’s find f(−2): f(−2) = 3((−−2)2)−−22 = −−62−−22 = −−84 = 2 Next, let’s find f(−1): f(−1) = 3((−−1)1)−−22 = −−31−−22 = −−53 = 53 Finally, let’s find f(0): f(0) = 3(0)0−−22 = 0−−22 = −−22 = 1 Now, we can sum these values: f(−2) + f(−1) + f(0) = 2 + 53 + 1 To add these, we need a common denominator, which is 3: 2 + 35 + 1 = 23·3 + 53 + 13·3 = 63 + 53 + 33 Now, we add the numerators: 63 + 53 + 33 = 6+5+33 = 143 The answer is:

|14 3<br><br>|
|---|

##### . Correct

