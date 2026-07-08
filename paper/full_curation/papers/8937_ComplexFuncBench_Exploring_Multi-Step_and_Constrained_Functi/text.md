## ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario

Lucen Zhong1 Zhengxiao Du1,2 Xiaohan Zhang1 Haiyi Hu1 Jie Tang2 1Zhipu AI 2Tsinghua University lucen.zhong@aminer.cn zx-du20@mails.tsinghua.edu.cn

# arXiv:2501.10132v1[cs.CL]17Jan2025

### Abstract

Enhancing large language models (LLMs) with real-time APIs can help generate more accurate and up-to-date responses. However, evaluating the function calling abilities of LLMs in real-world scenarios remains underexplored due to the complexity of data collection and evaluation. In this work, we introduce ComplexFuncBench, a benchmark for complex function calling across five real-world scenarios. Compared to existing benchmarks, ComplexFuncBench encompasses multi-step and constrained function calling, which requires long-parameter filing, parameter value reasoning, and 128k long context. Additionally, we propose an automatic framework, ComplexEval, for quantitatively evaluating complex function calling tasks. Through comprehensive experiments, we demonstrate the deficiencies of state-of-the-art LLMs in function calling and suggest future directions for optimizing these capabilities. The data and code are available at https://github.com/THUDM/ ComplexFuncBench.

### 1 Introduction

Large Language Models (LLMs) achieve remarkable results on many general tasks (Hendrycks et al.; Xu et al., 2023; Chen et al., 2021; Cobbe et al., 2021). However, LLMs lack real-time and factual knowledge because they can only respond based on the information they were trained on (Zhao et al., 2023). To address this limitation, recent research focuses on enhancing LLMs with function calling capabilities (Qin et al., 2024, 2023; Chen et al., 2024a; MeetKai, 2024). By integrating external tools and APIs, LLMs can deliver more accurate and up-to-date outputs.

While many models (OpenAI, 2023; Anthropic, 2024; TeamGLM et al., 2024) are enhanced with function calling capabilities, evaluating function calling, especially complex calls, is still challenging. Researchers try various methods to evaluate

(a) Simple Function Calling

Query: Book a flight from Washington to New York on December 23, 2024.

###### (b) Complex Function Calling

Query: Help me find the cheapest flight from Washington to New York on the day before Christmas Eve 2024. I want to know the flight's baggage allowances and also arrange a taxi to Times Square 1 hour after landing.

- Step 1

- Step 2

Function Calls: {"name": "Search_Flights", "arguments": {"departDate": "2024-12-23", ...}} API Response: [{"number": "d7699", "price": "1287 USD", "arriveTime": "13:00", "arriveAirport": "John F. Kennedy International Airport", ...},

128k context length

{"number": "d7740", "price": "1025 USD", "arriveTime": "22:00", "arriveAirport": "LaGuardia Airport", "token": "d7699_H4sIAAAA..."}, ...]

Function Calls: {"name": "Get_Flights_Details", "arguments": {"token": "d7699_H4sIAAAA..."}} {"name": "Search_Taxi", "arguments": {"pick_up_time": "23:00", "pick_up_place": "LaGuardia Airport", ...}} API Response: [...]

Assistant Response: The cheapest flight is ...

Figure 1: (a) Simple Function Calling. (b) Complex Function Calling with multi-step, constraints, parameter value reasoning, long parameter values and long context. Different colors correspond to the corresponding features marked in the figure.

the function calling capabilities of LLMs. Some works propose the prompt-based methods to assess the tool planning ability (Chen et al., 2024b; Qiao et al., 2024) of LLMs, but these do not assess the correctness of calling parameters. Meanwhile, some works evaluate the final state of the execution environments (Qin et al., 2023; Lu et al., 2024), yet ignore the correctness of the intermediate process. Moreover, some work uses rule-based matching methods(Yan et al., 2024; Wang et al., 2024) to calculate the precision of function calls, but these methods are limited to simple scenarios and cannot effectively assess complex function calls.

As shown in Figure 1, existing benchmarks directly provide parameters in the query and only require one-step function calling. In contrast, in real-world scenarios, users often implicitly provide parameter values rather than explicitly. LLMs are expected to infer the correct parameter values based on user constraints and API responses. Therefore, we define complex function calling from five aspects: (1) Multi-step in a single turn; (2) User-

Real API Response Multi-Step Constraints Parameter Value Reasoning Long Parameter Values Long-Context

API-Bench ✗ ✗ ✗ ✗ ✗ ✗ ToolBench ✓ ✓ ✗ ✗ ✗ ✗

T-Eval ✓ ✓ ✗ ✗ ✗ ✗ BFCL ✗ ✓ ✗ ✗ ✓ ✓

Tool Sandbox ✗ ✓ ✗ ✗ ✗ ✗ ComplexFuncBench ✓ ✓ ✓ ✓ ✓ ✓

Table 1: ComplexFuncBench compare with other function calling benchmarks.

provided constraints; (3) Parameter value reasoning from implicit information; (4) Long parameter values; and (5) 128k long context length.

The evaluation of complex function calling is currently underexplored due to several challenges. First, data annotation for complex function calling is time-consuming and laborious. Since current LLMs cannot do this task well, specially-trained senior annotators are required for annotation. Second, annotating a single valid path without disambiguating API response is challenging. Real-time API responses may contain multiple choices that satisfy user constraints at the current step, resulting in ambiguities in the parameters for subsequent calls. Third, traditional rule-based matching methods are inadequate for evaluating complex function calling since function calls with slightly different parameters can be equivalent. For example, parameter values can be expressed differently, such as NY and New York.

To address the above problems, we introduce ComplexFuncBench, a complex function calling benchmark with multi-step and constrained function calling under long-context scenario. We collect functions from the Booking.com 1 website and manually annotate 1,000 complex function calling samples across five real-world domains. The annotation process is divided into three stages. At the coarse generation stage, we use GPT-4o to generate 1,000 queries along with the corresponding function calling paths to build a preliminary dataset to reduce the costs of human annotation. At the finegrained annotation stage, we train senior annotators to select and label 100 complex samples with different function call paths based on the preliminary dataset to build a template dataset. Specifically, we first ensure that the annotated function calling path is the shortest one that fully addresses the query requirements while correcting any errors in parameter values. Then, we remove ambiguous information from API responses to ensure a single valid function calling path for each sample, avoiding

1https://rapidapi.com/DataCrawler/api/booking-com15

parameter ambiguity during evaluation. At the generalization stage, we train junior annotators to generalize the template dataset from 100 to 1,000 samples by modifying specific information in the template queries to build ComplexFuncBench. Table 1 shows the difference between ComplexFuncBench and other function calling benchmarks.

We further propose an automatic evaluation framework ComplexEval for complex function calling evaluation. ComplexEval overcomes the limitations of traditional exact matching methods by using a multi-dimensional matching approach: (1) Rule-based matching treats two identical function calls as equivalent. (2) Response-based matching treats function calls with the same API responses as equivalent. (3) LLM-based matching prompts LLMs to decide the equivalence of function calls, allowing for different expressions of parameter values. Additionally, ComplexEval provides error information as the API response to assess the self-correction ability of LLMs. Furthermore, we utilize LLMs to evaluate the completeness and correctness of the final model-generated responses.

We conduct extensive experiments on various LLMs on ComplexFuncBench. We find that a significant portion of errors come from parameter value errors. Additionally, different models exhibit specific weaknesses in particular scenarios. Our contributions are as follows:

- • We propose ComplexFuncBench, a dataset consisting of 1,000 samples with multi-step and constrained function calling that requires 128k long context.
- • We propose ComplexEval, an automatic evaluation framework for complex function calling. ComplexEval addresses previous evaluation challenges by integrating a multidimensional matching method.
- • The detailed analysis of the experiment results will guide future optimization of function calling capabilities.

Function Set Building

###### Stage 1: Coarse Generation

Stage 2: Fine-grained Annotation Data Point

###### Stage 3: Generalization

[Figure 1]

Query

Sample Functions

[Figure 2]

[Figure 3]

[Figure 4]

senior

Function Collection

Query Generation

Correction

Query Generalization

[Figure 5]

[Figure 6]

senior

[Figure 7]

junior

Correction & Disambiguation

Add Description

Path Annotation

Disambiguation

quality check quality check

###### Function Set

###### Preliminary Dataset Template Dataset

ComplexFuncBench

(a)

Correction

Disambigutation

rewrite query API response

function call order annotation Step 1 Step 2

[Figure 8]

[Figure 9]

function list

[Figure 10]

[Figure 11]

Find the cheapest NY to LA flight on Dec 1, 2024.

{"id": "NYC.CITY", "type": "CITY", "name": "New York", ...}, {"id": 'JFK.AIRPORT", "type": "AIRPORT", "name": "John F. Kennedy International Airport", "cityName": "New York", ...}

Annotated

1 2 3 4 5

{"name": "Search Hotel Destination", "params": {...}}

- 3

5

- 4

- 1

- 2

Valid 1 1 3 2 4 5

{"name": "Location lat to long", "params": {...}}

fix param error

[Figure 12]

...

...

Flights(from=xx, ... , sort=price)

Valid n 2 1 3 5 4

(b) (c)

- Figure 2: Overview of the data collection process. (a) is the high-level process of data collection. (b) is the example of human correction process.(c) is the example of disambiguation process. The grey part is removed during annotation. A detailed annotation example is shown in Appendix A.1.

### 2 ComplexFuncBench

##### 2.1 Data Collection

We collect daily-life APIs from Booking.com provided by RapidAPI to build our candidate function set, which includes 43 real-time APIs across five domains: Hotel, Flight, Attraction, Car Rental, and Taxi. We find that the function descriptions from the website are incomplete and even incorrect. Therefore, we manually correct and annotate the detailed function description for each function to build a high-quality function set.

The ComplexFuncBench dataset collection has three stages: coarse generation, fine-grained annotation, and generalization. The overview of the data collection process is shown in Figure 2.

##### 2.1.1 Stage 1: Coarse Generation

Annotating complex function calling samples is a multifaceted process that includes query construction, parameter annotation, function call order arrangement, and API response disambiguation. To mitigate the costs of human annotation, we leverage the function calling ability of GPT-4o to generate 1,000 preliminary complex function calling samples as a reference to assist human annotators.

Specifically, we randomly sampled several functions from the function set and specially designed

a prompt for GPT-4o to generate 1,000 queries requiring complex function calling across different domains. The detailed prompt is shown in Appendix A.3. We leverage GPT-4o’s function calling interface to automatically annotate these queries with function calling results to construct a preliminary dataset. However, the automatically annotated data exhibits severe errors due to current models’ limited complex function calling capabilities, such as parameter value and function call order errors.

##### 2.1.2 Stage 2: Fine-grained Annotation

To ensure the precision and reliability of the benchmark dataset, we propose a fine-grained annotation process to refine the preliminary dataset. Many automatically generated samples in the preliminary dataset failed at the first step, making subsequent human annotation difficult. Therefore, we engage human annotators to select 100 samples with relatively complete function calling paths from the preliminary dataset, encompassing various function calling steps and different function calling paths. A senior annotator will annotate the selected samples through the following steps.

Correction Annotators must correct the GPTgenerated samples from three aspects. First, rewrite the query with detailed information, like the ARRIVE_TIME of the taxi and the CHECK_IN_DATE

of the hotel. Second, annotate the function call order. Specifically, they add missing function calls and remove unnecessary ones to get the whole function call path that can comprehensively address the query requirements. Then, they adjust the function call order to build the shortest function call path as the final annotated path. As shown in Figure 2(b), a query may have multiple valid calling paths to complete the task. We annotate the shortest path for quantitative evaluation later. Third, correct function calling parameter errors, such as parameter type mismatches, incorrect parameter values, parameter hallucinations, and missing parameters.

Disambiguation After correction, human annotators must ensure that the functions and the calling parameters are unambiguous. First, overlapping functionalities are removed from the function list for each sample to eliminate ambiguity in functions. For example, obtaining the latitude and longitude of a city can be achieved through both SEARCH_HOTEL_DESTINATION and LOCATION_LAT_TO_LONG function. Therefore, these two functions will not be available in the same sample. Second, to disambiguate the calling parameter values, we will delete API responses that may cause parameter ambiguity for subsequent function calls. As shown in Figure 2(c), when searching hotels in New York City, the initial step might return multiple location IDs of NY(e.g., for downtown, airports, attractions, etc.). Only one of these location IDs is retained during the annotation process to prevent ambiguity in subsequent calls. Note that there is no need to delete results unrelated to New York City, as those do not introduce ambiguity.

We perform quality checks on all 100 samples until there are no errors. Finally, we get 100 complex function calling paths, including singledomain and cross-domains.

##### 2.1.3 Stage 3: Generalization

To reduce the variance of the evaluation results, we generalize each sample in the template dataset to 10 samples. This expansion increases the ComplexFuncBench dataset size from 100 to 1,000. By building a larger dataset, we can get more reliable and stable results from the benchmark.

Specifically, we use the 100 queries in the template dataset as examples and design prompts for GPT-4o to modify elements such as locations and constraints, generating nine new queries for each original query. The detailed prompt is shown in

Appendix A.4. These 10 queries share the same function calling path but differ in parameter values for each call. We then trained junior annotators to fill in the respective function call parameter values based on the given path, delete API responses to avoid disambiguation, and obtain a new complex function calling sample. The annotation difficulty and human effort required for generalization are significantly lower than the cost of fine-grained annotation, enabling several junior annotators to complete large-scale annotations effectively.

##### 2.2 Data Composition

Finally, we obtained 1,000 complex function calling samples, which comprises 600 single-domain samples, with 150 samples each from the Hotels, Flights, Car Rental, and Attraction domains, and 400 cross-domain samples. The taxi domain only has two functions, so it is only used for crossdomain. The average number of function calling steps in ComplexFuncBench is 3.26, while the average number of function calls per sample is 5.07. Detailed statistics are shown in Appendix A.2.

3 ComplexEval: Automatic Evaluation

We propose ComplexEval to quantitatively evaluate models’ complex function calling ability and response generation ability. Figure 3 is the overview of ComplexEval.

##### 3.1 Function Calling Evaluation

Given the user query q and the available function list f, the model continually generates the next-step function calls until the final response is generated.

Specifically, at step t, the model generates a list consisting of n predicted function calls, denote as [p0t,··· ,pnt ]. And the golden function call list contains m function calls [gt0,··· ,gtm]. We aim to evaluate the correctness of each predicted function call pit against the golden function call list. We will append the corresponding API response to the dialogue history for each pit. Based on the updated dialogue history, the model then predicts the nextstep function calls. The evaluation process includes format checking, Hungarian mapping for function calls, and multi-dimensional matching.

Format Checking First, we check the format of each pit in the predicted function call list. The format check includes three parts: (1) verifying that the called function is in the available function list, (2) ensuring that all required parameters are

Model Response Evaluation

Output Response

query functions

FC Model Output FC Format Checking Hungarian Mapping

API Response

[Figure 13]

[Figure 14]

Multi-Dimensional Matching

###### API Response ✅

|Format Error|
|---|
|{"error":"parameter {param_name} of {func_name} should be a string, but got float."}<br><br>|

|Success Matched|
|---|
|{"dest_id":"13264", "latitude":37.8, "longitude":-122.4, "name":"Golden Gate<br><br>Bridge"}|

|Invalid FC|
|---|
|{"error": "function call is not correct, please double-check for possible problems."}<br><br>|

Rule-based Matching

[Figure 15]

Response-based Matching

LLM-based Matching

- Figure 3: Overview of ComplexEval. Different colors represent different API response types. Color blue represents format error with specific error message. Color green represents correct function call with corresponding golden API response. Color red represents invalid function call with general error message.

included in the call, and (3) confirming that the type of each parameter meets the requirement in the function description. If pit fails the format checking, we use a specific error message as the API response. This approach allows the model to undergo a trialand-error process, and we can evaluate the model’s self-correction ability.

Hungarian Mapping For remaining predicted function calls with correct formats, we use bgelarge-en-v1.5 model(Xiao et al., 2023) to obtain the text embedding for each function call. Specifically, we concatenate the function name and parameters into a single string for text encoding. The predicted and golden function calls list embedding are denoted as ep and eg, respectively. Then, we calculate the cosine similarity between ep and eg and employ the Hungarian mapping algorithm to get a mapping list, e.g., (pit,gtj) denotes the ith predict function call mapping with the jth golden function call in at step t.

(i,j) = HungarianMatch(−ep × eTg ) (1)

Multi-Dimensional Matching Based on the mapping list, we propose a multi-dimensional matching method to decide the equivalence between pit and gtj.

- • Rule-based Matching determines equivalence through exact matching. Two identical function calls are considered equivalent.
- • Response-based Matching determines equivalence by comparing the returned API re-

sponse of the two function calls. Two function calls are considered equivalent when the returned results are identical. For example, if parameter adults default value is 1, then Search_Hotel(New_York, adults=1) and Search_Hotel(New_York) are equivalent.

• LLM-based Matching leverages GPT-4o to determine the equivalence. Specifically, Parameter values can be expressed in different forms as long as the meaning is the same. For example, New York and NY, Narita International Airport and Tokyo Narita International Airport are equivalent. Detailed prompt is shown in Appendix B.3.

If pit is equivalent to gtj, the annotated API response of gtj is used as the API response of pit. If pit is not equivalent to gtj, we use a fixed system error message as the API response of pit. We will update the golden function call list by removing successfully matched calls and adding the next-step calls, as shown in Appendix B.2. We continually call the model to generate the next-step function calls until the final model response is generated.

Metrics We use Success Rate and Call Accuracy as metrics. Success Rate measures the overall task completion by calculating the proportion of samples that successfully complete the task. Call Accuracy calculates the proportion of correct function calls as in equation (2).

N i=1 ci N i=1 ni

Call Acc =

(2)

where N is the number of samples, ni is the total number of function calls in sample i, and ci is the number of correct function calls in sample i.

##### 3.2 Model Response Evaluation

After obtaining the model’s final response, we use GPT-4o to further evaluate the response quality from two aspects: completeness and correctness. The detailed prompt is shown in Appendix B.4

Completeness Whether the response comprehensively addresses all the requirements proposed by the user. Given the query q and generated response rgen, GPT-4o will give a score ∈ {0,1,2} as the completeness score. score = 0 indicates none of the user requirements were addressed, score = 1 indicates partial fulfillment and score = 2 indicates full fulfillment. We calculate the average score of all samples as the final completeness score.

Correctness Whether the provided answers are accurate and align with the API response. Given the complete dialogue history h and generated response rgen, GPT-4o will give a score ∈ {0,1,2} as the correctness score. score = 0 indicates totally incorrect, score = 1 indicates partially correct and score = 2 indicates totally correct. We calculate the average score of all samples as the final correctness score.

4 Experiments

##### 4.1 Settings

We select function calling models with 128k context length, encompassing 12 models from 6 institutions, to comprehensively investigate current model performance on ComplexFuncBench. Concretely, for closed-source models, we evaluate the latest version of GPT-4o and GPT-4-Turbo(OpenAI, 2024), Claude 3.5 Sonnet and Haiku(Anthropic, 2024) and GLM-4-Long(TeamGLM et al., 2024). For opensource models, we evaluate Llama-3.1 series from 8B to 405B(Dubey et al., 2024), Qwen2.5(Qwen Team, 2024) 7B and 72B, and Mistral Large 2(Mistral AI, 2024).

4.2 Main Results The main results of function call evaluation and model response evaluation are shown in Table 2.

Function Call Evaluation The complex function calling ability of the closed-source models is superior to that of the open-source models. Claude-3.5Sonnet, GPT-4o, and GLM-4-Long perform compa-

rably among the closed-source models, with overall success rates of 61.0%, 60.5%, and 57.1%, respectively. Among the open-source models, Qwen2.572B performs the best, achieving a task success rate of 40.1%. The Llama3.1 series models do not possess the ability for complex function calling; the experiment results indicate that Llama3.1 suffers from disordered function call sequences and severe parameter hallucinations during the function calling process. Moreover, models with less than 10B parameters cannot handle the complex function calling well; the best-performing model, GLM-4-9B, only achieved a success rate of 8.4%.

Different models’ performance varies in different domains due to the distinct challenges presented by each domain. Specifically, GPT-4o achieved notable success rates of 70% and 82% in the Hotel and Attraction domains, respectively, significantly outperforming other models. These domains require a model to comprehensively understand the function descriptions to infer the function calling order. Conversely, Claude-3.5-Sonnet handles more complex function calls in cross-domain scenarios better. Meanwhile, GLM-4-Long excelled in the Flights and Car Rental domains, which require handing long parameter values approaching 600 characters.

Model Response Evaluation For closed-source models, Claude-3.5-Sonnet outperforms other models in both the completeness and correctness of responses, achieving scores of 1.84 and 1.85, respectively. However, despite GPT-4o’s high success rate, it shows the least complete performance with a score of 1.66. For the open-source model, Qwen2.5-72B generates the most complete and correct response, with completeness and correctness scores of 1.80 and 1.75, which is superior to other open-source models.

##### 4.3 Results Analysis

To compare the performance between different models more precisely, we select the top three closed-source models and the leading open-source model for in-depth analysis.

##### 4.3.1 Error Analysis

We classify the errors encountered during the complex function calling process into five types: func_error, param_missing, hallucination, value_error, and stop_early. func_error indicates calling a wrong or invalid function. param_missing refers to the omission of cru-

Hotels Flights Car Rental Attraction Cross Overall

Completeness Correctness Success Call Acc Success Call Acc Success Call Acc Success Call Acc Success Call Acc Success Call Acc

Model

close-source models

Claude-3.5-Haiku 36.00 50.62 50.67 75.63 59.33 74.05 58.00 75.37 38.00 70.00 45.80 69.50 1.79 1.71 Claude-3.5-Sonnet 54.67 68.17 54.00 79.50 76.67 86.01 69.33 83.33 57.00 79.33 61.00 79.27 1.84 1.85

GLM-4-Long 56.00 63.98 66.67 84.38 77.33 85.71 72.67 83.33 40.50 72.75 57.10 76.35 1.72 1.74 GPT-4-Turbo 54.67 68.48 48.67 76.5 44.67 71.14 70.67 76.48 41.75 69.38 49.50 71.38 1.72 1.81

GPT-4o 70.00 81.99 65.33 85.50 72.00 86.88 82.00 87.59 42.75 75.13 60.50 80.55 1.66 1.75

open-source models Qwen2.5-7B 2.00 20.65 0.00 5.13 4.67 6.41 14.67 35.18 4.5 21.41 5.0 18.19 1.5 1.47

Llama-3.1-8B 0.00 0.00 0.00 1.00 0.00 1.89 0.67 2.78 0.00 1.00 0.10 1.34 0.18 0.09 GLM-4-9B 19.33 31.52 11.33 34.00 16.0 25.36 10.67 29.26 2.00 25.46 9.40 27.97 1.15 1.03

Llama-3.1-70B 2.00 10.71 0.67 2.63 6.67 10.06 4.67 11.11 1.50 8.13 2.70 8.17 0.67 0.36 Llama-3.1-405B 3.33 13.51 2.66 10.75 4.00 15.74 14.00 18.52 1.00 9.21 4.00 11.87 0.43 0.30

Qwen2.5-72B 40.00 60.24 31.33 49.25 48.67 57.58 63.33 67.41 31.50 59.00 40.10 58.32 1.80 1.75 Mistral Large 2 19.33 34.32 20.67 52.88 40.67 58.16 25.33 40.18 10.50 50.54 20.10 48.78 0.94 1.0

Table 2: Main Results. We categorize models as close-source, open-source under 10B, and open-source above 10B. Top two results for each category are highlighted in bold and underline. The specific endpoint of open-source models are: gpt-4o-2024-08-06, gpt-4-turbo-2024-04-09, claude-3-5-sonnet-20241022 and claude-3-5-haiku-20241022.

[Figure 16]

Figure 4: Error type analysis for different models.

cial parameters. hallucination refers to generating random parameters not provided by the user. value_error denotes parameter values errors. stop_early means generating the final response without calling all required functions.

Figure 4 presents the error analysis results for four models. We can see that value_error accounts for a significant portion of errors in all models, particularly for Qwen2.5-72B, which has a value_error rate of 78.8%. The high value_error rate indicates that ComplexFuncBench poses significant challenges for LLMs in constrained parameter value reasoning and long-context parameter extraction. In addition to value_error, all four models tend to stop function calling without gathering complete information. This issue is especially severe for Claude-

- 3.5-Sonnet and GPT-4o, with stop_early rates of 19.7% and 21.0%, respectively. Moreover, GLM4-Long suffers from missing parameters with a param_missing rate of 11.0%, while other models show fewer parameter missing problems. Although less common compared to value_error, func_error still occurs, showing the model lacks understanding of function documentation.

We analyze the error rate for each parameter type to further investigate the causes of the

value_error in different models. Specifically, we divide all parameters into different types, such as date-related, time-related, and location-related parameters. Appendix C gives examples of different parameter types. Then, we calculate the error rates of different models on different parameter types. The result is shown in Figure 5.

Although slight variations exist in the error distribution for different models, the incorrect parameters related to filter and legs are relatively higher than other parameters. The filter parameter challenges the model to infer function calling orders from the parameter description. GLM-4Long fails more than 40% on this parameter. The legs parameter needs the model to gather multiple information from the user query and API response. Qwen2.5-72B fails 38.1% on this parameter. Claude-3.5-Sonnet cannot understand the constrained user query and consistently retrieves the wrong token parameter. Claude-3.5-Sonnet and GPT-4o cannot understand the relationships between parameters specified in the function description and often retrieve the wrong slug parameter from the API response. The results indicate that it is challenging for models to complete complex function calling with constraints and long context, and there is still room for improvement.

##### 4.3.2 Function Calling Steps Analysis

Figure 6 compares the annotated shortest calling steps with the predicted calling steps among different models. We can see that the predicted calling steps often exceed the shortest calling steps due to the models’ distinct planning strategies. Specifically, Claude-3.5-Sonnet and Qwen2.5-72B require an average of 4.47 and 4.28 steps, respectively, be-

[Figure 17]

Figure 5: Error rates for each parameter type of different models

Claude-3.5-Sonnet

GPT-4o

Function Calling Evaluation A reliable and comprehensive function calling benchmark is crucial for providing optimization directions for function calling models. Several works propose to use llm-based methods to evaluate the tool planning capabilities of LLMs (Qiao et al., 2024; Nexusflow.ai, 2023). However, these methods can not capture the correctness of the function call parameters since they generate natural language thoughts instead of formatted function calls. Therefore, some studies propose rule-based matching methods, such as AST tree analysis, to evaluate the precision of function calling (Yan et al., 2024; Li et al., 2023; Wang et al., 2024; Chen et al., 2024b). However, these studies only evaluate single function calling without assessing complex function calling scenarios. Furthermore, some work proposes to use state-based (Qin et al., 2023; Lu et al., 2024; Yan et al., 2024) methods to evaluate the final state of the system without considering the intermediate function call procedure, resulting in the randomness of the evaluation results. While previous research lacks a thorough evaluation of complex function calling in real-world scenarios, this work introduces a benchmark for function calling in more complex settings.

1.2

1.2

| ||Mean: 3.05 Std: 0.75|
|---|
<br><br>|Mean: Std: 1.|
|---|
|4.47 65<br><br>Shortest Step Predicted Step<br><br>| |
|---|
<br><br>0.0<br><br>0.2<br><br>0.4<br><br>0.6<br><br>0.8<br><br>1.0<br><br>|Mean: 3.02 Std: 0.71|
|---|
<br><br>|Mean: 3 Std: 1.0|
|---|
|.25 5<br><br>Shortest Step Predicted Step<br><br>| |
|---|
|
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

1.0

0.8

Density

Density

0.6

0.4

0.2

0.0

0 2 4 6 8 10

0 2 4 6 8 10

Steps

Steps

GLM-4-Long

Qwen2.5-72B

1.2

1.2

| ||Mean: 3.06 Std: 0.69|
|---|
<br><br>|Mean Std: 0|
|---|
|: 3.16<br><br>.91<br><br>Shortest Step Predicted Step<br><br>| |
|---|
| |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| ||Mean: 3.02 Std: 0.75|
|---|
<br><br>|Mean: 4.28 Std: 1.62|
|---|
|Shortest Step Predicted Step<br><br>| |
|---|
|
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

1.0

1.0

0.8

0.8

Density

Density

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

0 2 4 6 8 10

0 2 4 6 8 10

Steps

Steps

Figure 6: Function calling steps distribution.

yond the shortest steps. These models generally need 1–3 extra steps more than the shortest steps. As the task complexity increases, these models need more than 10 steps for task completion. In contrast, models such as GLM-4-Long and GPT-

- 4o tend to complete tasks with fewer steps. The GLM-4-Long’s average calling step is only 0.13 steps longer than the shortest steps.
- 5 Related Work

Function Calling with LLMs Numerous models have been developed to enable function calling within Large Language Models (LLMs) to address challenges in real-world applications. Some foundation models (OpenAI, 2023; TeamGLM et al., 2024; Anthropic, 2024) are inherently equipped with native function calling capabilities, allowing seamless integration of API functionalities. In addition to these, some work proposes to fine-tune base models with specialized function calling datasets (Qin et al., 2023; Patil et al., 2023; Gao et al., 2024; MeetKai, 2024; Liu et al., 2024). By training models on curated datasets, these works enable the models to dynamically generate function calls and execute functions, bridging the gap between language model generation and real-world scenarios.

### 6 Conclusion

In this work, we introduce ComplexFuncBench, a benchmark dataset consisting of 1,000 samples from five realistic scenarios, and an automatic evaluation framework ComplexEval for complex function calling. Unlike existing benchmarks, ComplexFuncBench features multi-step and constrained function calling with parameter value reasoning, long-context parameter extraction, and long parameter value filling. We show the deficiency of current state-of-the-art function calling models through comprehensive experiments. We hope ComplexFuncBench can guide the research community in developing function calling capabilities of foundation models.

### References

Anthropic. 2024. The claude 3 model family: Opus, sonnet, haiku. https://www-cdn.anthropic.com/ de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/ Model_Card_Claude_3.pdf. Accessed: 2024-1021.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Mingyang Chen, Haoze Sun, Tianpeng Li, Fan Yang, Hao Liang, Keer Lu, Bin Cui, Wentao Zhang, Zenan Zhou, and Weipeng Chen. 2024a. Facilitating multiturn function calling for llms via compositional instruction tuning. arXiv preprint arXiv:2410.12952.

Zehui Chen, Weihua Du, Wenwei Zhang, Kuikun Liu, Jiangning Liu, Miao Zheng, Jingming Zhuo, Songyang Zhang, Dahua Lin, Kai Chen, et al. 2024b. T-eval: Evaluating the tool utilization capability of large language models step by step. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9510–9529.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Shen Gao, Zhengliang Shi, Minghang Zhu, Bowen Fang, Xin Xin, Pengjie Ren, Zhumin Chen, Jun Ma, and Zhaochun Ren. 2024. Confucius: Iterative tool learning from introspection feedback by easy-to-difficult curriculum. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 18030– 18038.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li API-bank. 2023. A comprehensive benchmark for tool-augmented llms. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3102–3116.

Weiwen Liu, Xu Huang, Xingshan Zeng, Xinlong Hao, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Zhengying Liu, Yuanqing Yu, et al. 2024. Toolace:

Winning the points of llm function calling. arXiv preprint arXiv:2409.00920.

Jiarui Lu, Thomas Holleis, Yizhe Zhang, Bernhard Aumayer, Feng Nan, Felix Bai, Shuang Ma, Shen Ma, Mengyu Li, Guoli Yin, et al. 2024. Toolsandbox: A stateful, conversational, interactive evaluation benchmark for llm tool use capabilities. arXiv preprint arXiv:2408.04682.

MeetKai. 2024. Meetkai functionary. https://

functionary.meetkai.com/. Mistral AI. 2024. Mistral large 2. Nexusflow.ai. 2023. Nexusraven-v2: Surpassing gpt-4

for zero-shot function calling.

- OpenAI. 2023. Openai api documentation: Function calling. https://platform.openai.com/docs/ guides/function-calling. Accessed: 2024-1021.
- OpenAI. 2024. Openai models.

Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. 2023. Gorilla: Large language model connected with massive apis. arXiv preprint arXiv:2305.15334.

Shuofei Qiao, Runnan Fang, Zhisong Qiu, Xiaobin Wang, Ningyu Zhang, Yong Jiang, Pengjun Xie, Fei Huang, and Huajun Chen. 2024. Benchmarking agentic workflow generation. arXiv preprint arXiv:2410.07869.

Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Yufei Huang, Chaojun Xiao, Chi Han, Yi Ren Fung, Yusheng Su, Huadong Wang, Cheng Qian, Runchu Tian, Kunlun Zhu, Shihao Liang, Xingyu Shen, Bokai Xu, Zhen Zhang, Yining Ye, Bowen Li, Ziwei Tang, Jing Yi, Yuzhang Zhu, Zhenning Dai, Lan Yan, Xin Cong, Yaxi Lu, Weilin Zhao, Yuxiang Huang, Junxi Yan, Xu Han, Xian Sun, Dahai Li, Jason Phang, Cheng Yang, Tongshuang Wu, Heng Ji, Zhiyuan Liu, and Maosong Sun. 2024. Tool learning with foundation models. Preprint, arXiv:2304.08354.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. 2023. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789.

Qwen Team. 2024. Qwen2.5: A party of foundation models.

TeamGLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong, Mingdao Liu, Minlie Huang, Peng Zhang, Qinkai Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin Cao, Shuxun Yang, Weng Lam Tam, Wenyi

Zhao, Xiao Liu, Xiao Xia, Xiaohan Zhang, Xiaotao Gu, Xin Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan Song, Xunkai Zhang, Yifan An, Yifan Xu, Yilin Niu, Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao Dong, Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, and Zihan Wang. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv e-prints, pages arXiv–2406.

Pei Wang, Yanan Wu, Zekun Wang, Jiaheng Liu, Xiaoshuai Song, Zhongyuan Peng, Ken Deng, Chenchen Zhang, Jiakai Wang, Junran Peng, et al. 2024. Mtu-bench: A multi-granularity tool-use benchmark for large language models. arXiv preprint arXiv:2410.11710.

Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023. C-pack: Packaged resources to advance general chinese embedding. Preprint, arXiv:2309.07597.

Liang Xu, Anqi Li, Lei Zhu, Hang Xue, Changtai Zhu, Kangkang Zhao, Haonan He, Xuanwei Zhang, Qiyue Kang, and Zhenzhong Lan. 2023. Superclue: A comprehensive chinese large language model benchmark. arXiv preprint arXiv:2307.15020.

Fanjia Yan, Huanzhi Mao, Charlie Cheng-Jie Ji, Tianjun Zhang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2024. Berkeley function calling leaderboard.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223.

### A Data Annotation

- A.1 An Example of Annotation Process

Table 5 gives a detailed example of the annotation process.

- A.2 Data Statistics

The data statistics for ComplexFuncBench are shown in Table 3.

Hotels Flights Car Rental Attraction Cross Total

# Samples 150 150 150 150 400 1000 Avg. Steps 3.33 3.33 2.87 2.86 3.5 3.26 Avg. Calls 4.29 5.33 4.57 3.6 6.0 5.07

Table 3: Data statistics for ComplexFuncBench.

- A.3 Prompt for Query Generation. Prompt for query generation is shown in Figure 8.
- A.4 Prompt for Query Generalization

Prompt for query generalization at the generalization stage is shown in Figure 9.

### B Automatic Evaluation

- B.1 Experiment Setting

For a fair comparison of the LLM’s complex function calling, we use the same generation parameter for all models. We use greedy sampling during generation and the max generation token is set to 2048. For close-source models, we set tool_choice = ”auto” to let the LLM decide which tool to choose.

- B.2 Golden Function Call Updating

As shown in Algorithm 1, the model predicts a function call list with n function calls at step t, denoted as pt = [p0t,··· ,pnt ], while the golden function call list contains m function calls gt = [gt0,··· ,gtm]. We aim to evaluate the correctness of pit based on gt. If Any pit matched with gtj under the multidimensional matching method, the annotated API response of gtj is used as the API response of pit, and we will update the golden function call list by adding the next-step golden function calls. Figure 7 is an example if golden function call updating where t = 0.

6

1 2 3 4 5

1 2 1

Update

=

2 3 4 5

Figure 7: An example for golden function call updating. Path on the left is the annotated shortest function call path with three steps.

- B.3 Prompt for LLM-based Match Prompt for llm-based match is shown in Figure 10.
- B.4 Prompt for Model Response Evaluation

Prompt for model response evaluation are shown in Figure 11 and Figure 12.

### C Parameter Type Examples

Specific examples of each parameter type are shown in Table 4.

Algorithm 1 Golden Function Call List Updating Strategy

- 1: Annotated shortest function call path: G = [[g01,··· ,g0m1],··· ,[gs1,··· ,gsms]]
- 2: Initialization: ind ← 0,t ← 0
- 3: Golden function calls at current step t: gt = G[ind] = G[0] = [g01,··· ,g0m1]
- 4: repeat
- 5: Model predict function calls at current step t: pt = [p0t,··· ,pnt ]
- 6: Evaluation pt based on gt based on Section 3.1
- 7: Obtain the success matched golden function calls list gtsuccess = [gtj,··· ,gtk]
- 8: Remain golden function calls gt′ = gt − gtsuccess
- 9: if ind < s then
- 10: ind ← ind + 1
- 11: Update Golden function calls gt+1 = gt′ + G[ind]
- 12: else
- 13: Update Golden function calls gt+1 = gt′
- 14: end if
- 15: t ← t + 1
- 16: until Model generate the final response

Parameter Example Explanation

filter filter = "facility::433, facility::107") The filter parameter can be retrieved from the API response of the GET_FILTER function. It often appears in queries with constraints, like: Find a few hotels with a pool and free wifi. "facility::433" and "facility::107" denote pool and free wifi, respectively.

legs legs=[{"fromId": "DFW.CITY", "toId": "MUC.AIRPORT", "date": "2024-11-

The legs parameter is the multi-stop flights list which contains location and date.

- 25"},{"fromId": "MUC.AIRPORT", "toId": "STO.CITY", "date": "2024-11-
- 26"}, {"fromId": "STO.CITY","toId": "DFW.CITY","date": "2024-12-09"}])

token token="d7699_H4sIAAAAAAAA_ ... AAA."

The token parameter can be retrieved from the API response of multiple functions. For example, the SEARCH_FLIGHTS function will return the token for different flights.

slug slug="pr7jepixwlvr-private-guidedtour-orsay-museum-rare-languages"

The slug parameter can be retrieved from the API response of SEARCH_ATTRACTION_LOCATION function as as ’productSlug’ inside ’products’ or ’destinations’.

date date="2024-11-22" Parameters related to dates, like: check-in-date, check-out-date,etc. location location="Amsterdam" Parameters related to locations, like: country, city ,etc.

key key= "eyJkcml2ZXJzQWdlIjozMCwiZ HJvcE...19GRUVTIl19"

The key parameter can be retrieved from the API response of multiple functions. For example, the SEARCH_CAR_RENTALS function will return the key for different cars.

id id="eyJ1ZmkiOi01NjQwNjR9" The id parameter can be retrieved from the API response of multiple functions. For example, the SEARCH_ATTRACTION_LOCATION function will return the id for different attractions.

time time="08:00" Parameters related to time, like: pick-up-time, drop-off-time,etc. sort_by sort_by="popularity" The sort_by parameter controls the order in which hotel or flight results are

presented. For example, the hotels can be sorted by price or popularity. type type="landmark" The type parameter is the entity type, like landmark, city, etc. age age="8" The age parameter is the age of people.

people people=2 The people parameter is the number of people.

Table 4: Examples of different parameter types.

|Prompt for Query Generation|
|---|
|### You are an intelligent assistant with various tools with several related APIs. You need to propose some real-world tasks and use the apis we provide to solve them. ### You can use the following APIs: ```JSON {function_list} ``` ### Here are some usage examples: ```JSON [{"query": "I'm thinking about heading to Paris. Can you tell me which museums are currently trending?", "function_list": ["Search_Attraction_Location", "Search_Attractions", "thought": "First, call `Search_Attraction_Location` to query the ID of Paris. Then, call `Search_Attractions` to retrieve the most popular museums, using the ID returned in the previous step for Paris."]}] ``` where `query` is the user query, `api_list` is the api used to solve the query, and `thought` is the LLM's thought to finished the query.<br><br>Please come up with extra ten queries and use the apis to solve them step-by-step. Specific requirements are as follows:<br><br>1. These ten queries should display a diverse range of sentence structures, encompass a variety of tones and contain a wide range of subjects: myself, my friends, family, and company. These ten queries should display a diverse conbination of all apis.<br>2. You must incorporate the information required for each API call in the query. To achieve this, generate random information for required parameters such as date, distance, etc. However, you must not fabricate information such as IDs in the query. For instance, instead of saying "today" or "a month ago," provide an exact date. You cannot say 'team ID is 1' in the query because this ID is likely not valid. Don’t merely say ‘an address’, provide the exact road and district names. Don’t just mention ‘high-protein’, specify the protein percentage. Do not explicitly specifying which API to employ in the query.<br>3. Keep in mind that **each query involves [args3] sequential api calls at least**. Each query can call the same api multiple times, each step can contain several parallel calls.<br>4. The `number` arguments for all APIs must be fewer than 5.<br>5. Please ensure that your query uses only the APIs available in the current list. Avoid using any APIs provided in the example, as they may not be available.<br>6. The output format should follow the usage examples in JSON format. `api_list` only have api name, do not generate api calls. output:<br>|

###### Figure 8: Prompt for Query Generation.

#### Prompt for Query Generalization

|# Instructions You are a master of keyword replacement. I will provide you with a series of queries, and you need to replace 6-10 key pieces of information in each query, but ensure that the sentence structure remains unchanged. Information that can be replaced includes: Quantities: Number of people, e.g., two people, five people, not exceeding six. Age, if children are involved, the age should preferably be under 10. Number of items, e.g., three, two, not exceeding three.<br><br>Duration in days, if hotel stay duration is involved, it should be between 1 day and two weeks. Dates: Should be between November 1, 2024, and December 30, 2024. Distances: Between 10 kilometers and 50 kilometers, in multiples of 10. Times: In various formats, e.g., 14:00, 2 PM, 9 AM, 10:00 (if car rental times are involved, they should be between 8 AM and 8 PM). Latitude and Longitude: Select the latitude and longitude of major cities, within ±1.67. Airport: If the query mentions a city, the airport name should correspond to that city. Flight cabin class: enum [Economy Class, Premium Economy Class, Business Class, First Class]. Flight sorting: enum [Best, Cheapest, Fastest]. Hotels: If the query mentions a city, the hotel name should correspond to that city. Hotel chains: e.g., Four Seasons, Shangri-La, Marriott, InterContinental, Hyatt, Holiday Inn. Attractions/buildings corresponding to cities: If a city is replaced, the attractions/buildings in the original query should be replaced with the corresponding attractions/buildings of the new city. Attraction search keywords: e.g., Harry Potter, Mario, Barbie, Shakespeare, Ballet, Musical, Christmas, Marvel Movies, Star Wars, Simpsons. Attraction sorting: enum [Most Popular, Highest Rated, Lowest Price]. Car rental companies: enum [Dollar, Budget, Sixt, Avis, Alamo, Fox, Hertz, Thrifty]. Information that absolutely must not be replaced includes: Hotel filter criteria Hotel sorting criteria Hotel user review sorting Additional car rental components Child policies User reviews Payment methods Car rental company rating sorting<br><br># Input query {query}<br><br># Output output:|
|---|

- Figure 9: Prompt for Query Generalization.

|Prompt for LLM-based Match|
|---|
|You are an assistant for function call comparison. Your task is to determine whether two function calls are equivalent based on the conversation history and the function descriptions, and provide specific reasons. # Instructions: You need to determine whether two function calls are equivalent based on the following criteria:<br><br>1. The same parameter can be expressed in different languages. For example: `America`, `美国` and `アメ リカ` are equivalent.<br>2. The same parameter can be expressed in different forms as long as the meaning is the same. For example, `New York` and `NY` are equivalent, `Shanghai` and `Shanghai City` are equivalent. `Narita International Airport` and `Tokyo Narita International Airport` are equivalent.<br>3. A location with or without a country suffix is considered equivalent. For example, Van Gogh Museum, Amsterdam and Van Gogh Museum are equivalent.<br>4. The order of parameters in a function call can differ, for example: `add(1, 2)` and `add(2, 1)` are equivalent.<br>5. If a parameter in the function description has a default value, and the current parameter value is equal to that default value, then the parameter can be omitted in the function call. For example, if the adults parameter in the Search_Hotels function has a default value of 1, then the following two function calls are equivalent: `Search_Hotels(New_York, adults=1)` and `Search_Hotels(New_York)`. # Output: You need to output the result in JSON format, containing the following fields:<br><br><br>- **is_equal**: A boolean indicating whether the two function calls are equivalent.<br>- **reason**: Please provide the reason for your judgment.<br><br><br>## Example 1 output: ```JSON {"is_equal": true, "reason": "xx and yy are equivalent."} ```<br>## Example 2 output: ```JSON {"is_equal": false, "reason": "The parameter in call xx is not present in call yy."} ``` Note: The output must be a valid JSON format that can be properly loaded.<br><br><br>Function list: ```JSON<br><br>[args1] ``` Conversation history: ```JSON<br>[args2] ```<br><br>Function call 1: ```JSON<br><br>[args3] ```<br><br>Function call 2: ```JSON<br><br><br>[args4] ``` Please determine whether Function call 1 and Function call 2 are equivalent and provide your reason. output:\n<br>|

###### Figure 10: Prompt for LLM-based Match.

|Prompt for Completeness Evaluation|
|---|
|You are a helpful response completeness detect assistant. Your task is to evaluate the response based on whether it fully addresses all aspects of the user's query. # Your Task For each user query and corresponding response, you should determine the completeness of the response using the following criteria:<br><br>- If the response covers all requested information and addresses all parts of the user's query, it should be considered complete and receive a score of 2.<br>- If the response addresses some but not all parts of the user's query, it should be considered partial and receive a score of 1.<br>- If the response does not address any of the requested information in the user's query, it should be considered incomplete and receive a score of 0. # Output Format You should output the score for each user query and corresponding response in JSON format with following keys:<br>- score: the completeness score for the response (0, 1, or 2)<br>- reason: a string describing the reason for the score (e.g. "all requested information is addressed", "some requested information is not addressed", "no requested information is addressed") # Example<br><br><br>## Example 1 input: query: I'm thinking about heading to Paris. Can you tell me which museums are currently trending? response: The Louvre is the most treding museums in Paris. output: ```JSON {"score": 2, "reason": "all requested information is addressed"} ```<br>## Example 2 input: query: I'm planning to stay in Milan for a week starting on November 2nd, 2024. Help me find detailed information about the most trending attractions in the center of Milan and book a hotel nearby this attraction. response: The Milan Cathedral (Duomo di Milano), located in the central square of Milan, Italy, is the largest Gothic cathedral in the world. It is a Roman Catholic cathedral and ranks among the top five churches in the world. \nConstruction of the Milan Cathedral began in 1386, the vault was completed in 1500, and the highest Gothic spire, topped with the gilded statue of the Virgin Mary (La Madonnina), was built by Giuseppe Perego in 1774. It is a symbol of Milan city. The entire cathedral was completed in 1965, taking five centuries to finish. Napoleon held his coronation ceremony at the Milan Cathedral in 1805. output: ```JSON {"score": 1, "reason": "Only give the detailed information about the most trending attractions in the center of Milan. Miss the hotel booking part."} ```<br>## Example 3 input: query: My partner and I are gonna fly from Chicago to Tokyo on September 5th, 2024. Then we'll fly from Tokyo to Shanghai on September 9th, 2024. Finally, we'll go back to Chicago on September 19th, 2024. Can you give us the info about economy class? response: Sorry, I can't find any info about economy class. output: ```JSON {"score": 0, "reason": "The response did not give any information about economy class."}<br><br><br>input: query: {} response: {} output:|

###### Figure 11: Prompt for Completeness Evaluation.

|Prompt for Correctness Evaluation|
|---|
|You are a helpful response correctness detect assistant. Your task is to evaluate the response based on its accuracy in matching the details provided by API response. # Your task Give a dialogue history containing user query, function calls and api responses, you should determine the correctness of the corresponding respons using the following criteria:<br><br>- If the response is consistent with the information provided in the API response, it should be considered entirely correct and receive a score of 2.<br>- If the response partially matches the information provided in the API response (with some correct and some incorrect details), it should be considered partially correct and receive a score of 1.<br>- If the response does not match any of the information provided in the API response, it should be considered incorrect and receive a score of 0. # Output Format You should output the score for each dialogue history and corresponding response in JSON format with following keys:<br>- score: the completeness score for the response (0, 1, or 2)<br>- reason: a string describing the reason for the score (e.g. "all mentioned information is correct.", "Some information is correct, and some are incorrect.", "All information in the response is incorrect.")<br><br><br>## example output 1 output: ```JSON {"score": 2, "reason": "All mentioned information is correct."} ```<br>## example output 2 output: ```JSON {"score": 1, "reason": "xx is correct, and yy is incorrect."} ```<br>## example output 3 output: ```JSON {"score": 0, "reason": "All information in the response is incorrect."} ```<br><br><br>dialogue history: {} response: {} output:|

###### Figure 12: Prompt for Correctness Evaluation.

##### Query

Initial My friend is planning a trip between December 15, 2024 and December 20, 2024 and he wants to fly from Sydney to Melbourne. Please help find the best flight options and book a 4star hotel near Fitzroy Gardens. They also need to rent a car and a taxi service from Melbourne Airport to the hotel.

Human Annotated Please help my friend finds the best flight from Sydney to Melbourne on 15 December 2024 and book a hotel within 10km of Fitzroy Gardens, Melbourne for one night. Remember to book a taxi to pick him up from the airport to the hotel an hour after the plane lands.

Explanation Correction. Rewrite query for clarity, such as the number of days for the hotel reservation, the departure location,

and the time for the taxi.

##### Function call at step 1

- 1. {"name": "Search_Flight_Location", "arguments": {"query": "Sydney"}}
- 2. {"name": "Search_Flight_Location", "arguments": {"query": "Melbourne"}}
- 3. {"name": "Location_to_Lat_Long", "arguments": {"query": "Fitzroy Gardens, Melbourne"}}
- 4. {"name": "Taxi_Search_Location", "arguments": {"query": "Melbourne Airport"}}

GPT Generated

- 1. {"name": "Search_Flight_Location", "arguments": {"query": "Sydney"}}
- 2. {"name": "Search_Flight_Location", "arguments": {"query": "Melbourne"}}
- 3. {"name": "Location_to_Lat_Long", "arguments": {"query": "Fitzroy Gardens, Melbourne"}}

Human Annotated

Explanation Correction. The arrive airport for the best flight may not be "Melbourne Airport". The model need to get the

arrive airports before use "Taxi_Search_Location".

##### API response at step 1

- 1. {"message": "Success", "data": [{"id": "SYD.AIRPORT", "name": "Sydney Kingsford Smith Airport"}, {"id": "YQY.AIRPORT", "name": "Sydney (Nova Scotia) Airport"}]}
- 2. {"message": "Success", "data": [{"id": "MEL.CITY", "name": "Melbourne"}, {"id": "MEL.AIRPORT", "name": "Melbourne Airport"}, {"id": "AVV.AIRPORT", "name": "Avalon Airport"}, {"id": "MEB.AIRPORT", "name": "Essendon Fields Airport"}, {"id": "MLB.AIRPORT", "name": "Melbourne International Airport"}]}
- 3. {"message": "Success", "data": [{"business_status": "OPERATIONAL", "formatted_address": "Wellington Parade, East Melbourne VIC 3002, Australia" "geometry": {"location": {"lat": -37.81272149999999, "lng": 144.9800679}},"name": "Fitzroy Gardens"}]}
- 4. {"message": "Success", "data": [{"longitude": 144.849, "city": "Melbourne", "name": "Melbourne Airport", "latitude": -37.6696}, {"city": "Melbourne", "longitude": -80.6456, "name": "Melbourne International Airport", "latitude": 28.1025}]}

Initial

- 1. {"message": "Success", "data": [{ "id": "SYD.AIRPORT", "name": "Sydney Kingsford Smith Airport"}]},
- 2. {"message": "Success", "data": [{"id": "MEL.CITY", "name": "Melbourne"}]},
- 3. {"message": "Success", "data": [{"business_status": "OPERATIONAL", "formatted_address": "Wellington Parade, East Melbourne VIC 3002, Australia" "geometry": {"location": {"lat": -37.81272149999999, "lng": 144.9800679}},"name": "Fitzroy Gardens"}]}

Human Annotated

Explanation Disambiguation. Delete API response that may cause ambiguation for subsequent steps.

##### Function call at step 2

- 1. {"name": "Search_Flights", "arguments": {"fromId": "SYD.AIRPORT", "toId": "MEL.CITY", "departDate": "2024-12-15", "sort": "BEST"}}
- 2. {"name": "Search_Flights", "arguments": {"fromId": "YQY.AIRPORT", "toId": "MEL.CITY", "departDate": "2024-12-15", "sort": "BEST"}}
- 3. {"name": "Search_Flights", "arguments": {"fromId": "SYD.AIRPORT", "toId": "AVV.AIRPORT", "departDate": "2024-12-15", "sort": "BEST"}}
- 4. {"name": "Search_Flights", "arguments": {"fromId": "YQY.AIRPORT", "toId": "AVV.AIRPORT", "departDate": "2024-12-15", "sort": "BEST"}}
- 5. {"name": "Search_Hotels_By_Coordinates", "arguments": {"latitude": "-37.81272149999999", "longitude": "144.9800679", "arrival_date": "2024-12-15", "departure_date": "2024-12-20", "radius": 10}}

GPT Generated

- 1. {"name": "Search_Flights", "arguments": {"fromId": "SYD.AIRPORT", "toId": "MEL.CITY", "departDate": "2024-12-15", "sort": "BEST"}}
- 2. {"name": "Search_Hotels_By_Coordinates", "arguments": {"latitude": "-37.81272149999999", "longitude": "144.9800679", "arrival_date": "2024-12-15", "departure_date": "2024-12-20", "radius": 10}}

Human Annotated

Explanation Correction. After delete ambiguous information in the API response, we need to remove redundant function

calls.

Table 5: Annotation Example. To make the content easy for reading, we have removed most of the content from the API response. The actual API response contains a large amount of information, reaching a context length of 128k.

