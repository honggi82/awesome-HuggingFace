# arXiv:2407.14402v1[cs.AI]19Jul2024

## The Vision of Autonomic Computing: Can LLMs Make It a Reality?

Zhiyang Zhang 1∗ Fangkai Yang 2 Xiaoting Qin 2 Jue Zhang 2† Qingwei Lin 2 Gong Cheng 1 Dongmei Zhang 2 Saravan Rajmohan 2 Qi Zhang 2 1State Key Laboratory for Novel Software Technology, Nanjing University 2Microsoft

### ABSTRACT

task automation capabilities [7, 45, 46] represent a significant leap forward in our ability to create truly autonomic systems. Successful demonstrations of LLMs in tasks such as anomaly detection and incident mitigation [60] illustrate their potential to provide the contextual understanding and adaptive decision-making necessary to achieve the goals of ACV. This prompts an intriguing question: Can LLMs make the Vision of Autonomic Computing a reality?

The Vision of Autonomic Computing (ACV), proposed over two decades ago, envisions computing systems that self-manage akin to biological organisms, adapting seamlessly to changing environments. Despite decades of research, achieving ACV remains challenging due to the dynamic and complex nature of modern computing systems. Recent advancements in Large Language Models (LLMs) offer promising solutions to these challenges by leveraging their extensive knowledge, language understanding, and task automation capabilities. This paper explores the feasibility of realizing ACV through an LLM-based multi-agent framework for microservice management. We introduce a five-level taxonomy for autonomous service maintenance and present an online evaluation benchmark based on the Sock Shop microservice demo project to assess our framework’s performance. Our findings demonstrate significant progress towards achieving Level 3 autonomy, highlighting the effectiveness of LLMs in detecting and resolving issues within microservice architectures. This study contributes to advancing autonomic computing by pioneering the integration of LLMs into microservice management frameworks, paving the way for more adaptive and self-managing computing systems. The code will be made available at https://aka.ms/ACV-LLM.

Addressing this question is complex due to the broad scope of ACV. To systematically assess the feasibility of achieving ACV using LLMs, evaluation studies in concrete settings are necessary. This work proposes to study this feasibility within the context of microservice self-management, a popular architecture for managing cloud services. Specifically, we propose an LLM-based multi-agent self-management framework for microservices and assess its performance using a live online evaluation benchmark built on the known microservice demo project Sock Shop [1].

Our proposed LLM-based microservice management system employs a hierarchical multi-agent architecture. High-level group manager handles declarative tasks that span multiple service components, such as optimizing end-to-end latency to under 200 ms. In contrast, low-level autonomic agents focus on specific tasks within their managed service components. To evaluate our system, we introduce a five-level taxonomy of autonomous service maintenance, emphasizing Self-Optimization and Self-Healing. We then design specific evaluation tasks within the Sock Shop microservice, employing chaos engineering techniques to deliberately introduce faults and observe how our management system resolves these issues. Our findings demonstrate that the LLM-based multi-agent framework achieves Level 3 autonomy in our five-level taxonomy. While it effectively detects issues and performs specific imperative tasks, there are opportunities for further enhancements in root cause analysis and issue mitigation capabilities. Our contributions can be summarized as follows:

### 1 INTRODUCTION

As we enter the modern era, computing infrastructure is becoming increasingly distributed and large-scale, presenting significant challenges for human management. This complexity underscores the necessity for developing systems capable of self-management. This vision, reminiscent of the Vision of Autonomic Computing (ACV) [27] proposed two decades ago, envisions computing systems that can manage themselves according to an administrator’s goals, integrating new components as seamlessly as biological cells.

Realizing an autonomic system with self-management poses significant challenges. Early approaches to autonomic systems often relied on rule-based mechanisms and predefined policies, which, while effective to some extent, struggled to adapt to the increasingly dynamic and complex environments seen in modern computing systems [23, 33]. Over the years, significant strides have been made towards achieving this vision through the development of selfadaptive and self-managing systems [15, 36]. Despite significant efforts and progress over the past two decades, the realization of ACV is still elusive due to numerous grand challenges outlined in the ACV paper, many of which hinge on breakthroughs in AI.

- • We advance the domain of autonomic computing for microservice management through an LLM-based multi-agent framework. To the best of our knowledge, we are the first research work to explore microservice self-management using LLM-based agents.
- • We establish a taxonomy consisting of five levels for autonomous service maintenance. We also present an online evaluation benchmark designed to assess tasks corresponding to each level of autonomy within the microservice demo project Sock Shop.
- • We conduct a rigorous evaluation of our LLM-based microservice management framework using the aforementioned benchmark.

Recent advancements in AI, particularly through Large Language Models (LLMs), offer promising new avenues to address these challenges. LLMs’ extensive knowledge, language understanding, and

### 2 BACKGROUND AND RELATED WORK

Autonomic Computing. The goal of autonomic computing is to develop self-managing systems which reduces the complexity and cost of IT management that increases system reliability and

∗Work is done during an internship at Microsoft. †Corresponding author.

performance. It is inspired by the biological autonomic nervous system, which autonomously regulates functions such as heart rate and body temperature, thereby reducing cognitive load. The ACV paper identified four key objectives of self-management in autonomic systems [12, 27]:

- • Self-Configuration: Systems can configure and re-adjust themselves to meet the agreed objectives.
- • Self-Optimization: Continuously monitor themselves and seek opportunities to improve performance and costs.
- • Self-Healing: The ability to autonomously recover from failures and even predict them.
- • Self-Protection: Against malicious attacks or cascading failures.

It also proposed the Monitor, Analyze, Plan, Execute, and Knowledge (MAPE-K) loop [27] as a structured approach to autonomic computing, providing a clear and systematic methodology for implementing self-managing capabilities. The MAPE-K model forms the foundation of self-adaptive and self-management systems, enabling continuous adaptation and optimization through feedback loops [5, 12, 40, 55] and creating autonomous systems to handle complex and dynamic environments.

Subsequent works have explored various aspects of autonomic systems, leading to significant advancements in self-adaptive systems, such as handling uncertainties and runtime variabilities [4, 14, 24, 28, 61], as well as self-protecting ability in terms of security risks [20]. Autonomic computing principles have also been applied beyond software systems, such as in robotics [8], autonomous driving systems [6], and the development of digital twins [18]. These systems require sophisticated self-adaptation mechanisms to cope with dynamic and unpredictable environments.

Initially,autonomicsystemsoftenrelied on rule-based approaches

and predefined policies to achieve self-management. While effective in certain contexts, these approaches were limited in their ability to handle complex, dynamic scenarios requiring adaptive decisionmaking and contextual understanding. Recent advancements in AI and machine learning (ML) have introduced new possibilities. Leveraging AI and ML to provide deeper insights (such as event correlation and predictive analytics) and automation capabilities has further enhanced the self-managing capabilities of IT operations, leading to the development of a field known as AIOps.

Autonomic Management of Cloud-Native Applications. Cloud computing has emerged as a critical component of modern distributed computing systems, offering scalable resources, cost efficiency, and flexibility through on-demand access to computing power, storage, and applications. This shift has led to the widespread adoption of cloud-native applications, leveraging architectures such as microservices. However, managing these applications poses significant challenges, including ensuring security, navigating complex microservices architectures, maintaining observability, managing resource allocation, and ensuring reliable performance and scalability [17, 40, 50, 58].

To address these challenges, numerous management tools have beendeveloped. TheCloudNativeComputing Foundation (CNCF) [10] hosts various projects aimed at fostering innovation and collaboration within the cloud-native community, such as Kubernetes [11]

and Prometheus [47]. Kubernetes, for example, automates deployment, scaling, and management of containerized applications, offering features like service discovery and automated rollouts. Despite these capabilities, Kubernetes lacks comprehensive high-level management features aligned closely with user intent, such as built-in middleware or advanced configuration management systems. On the research side, recent advancements, such as the AMoCNA [29– 31]framework,extendtraditionalautonomic management approaches like the MAPE-K loop [5] to enhance autonomy in cloud-native environments. However, these frameworks often rely on rule-based systems, limiting their adaptability in handling complex scenarios requiring contextual understanding and adaptive decision-making. To address these limitations, recent works have further extended the traditional MAPE-K loop to incorporate AI and ML advancements [43, 51], proposing a novel self-adaptation approach [32, 52]. LLM-based Kubernetes and Cloud Service Management. In the era of LLMs, integrating LLMs into Kubernetes management represents a promising avenue to overcome limitations in traditional approaches. For instance, GenKubeSec [39] applies LLM-based methods to detect and remediate Kubernetes configuration file misconfigurations with precision and detailed insights, surpassing the capabilities of rule-based tools. Similarly, the open-source project K8sGPT [26] provides natural language-based diagnostics and issue triaging for Kubernetes clusters. Despite these advancements, a cohesive integration of LLMs across all facets of self-management in Kubernetes remains an ongoing area of development.

Similarly, current cloud service management systems are starting to integrate LLMs for AIOps to ensure the high availability and reliability of large-scale cloud infrastructures [57, 59, 60]. The key AIOps tasks within cloud service management include data collection and pre-processing, root cause analysis (RCA), and incident mitigation. Cloud platforms consist of numerous services with diverse configurations, posing challenges for traditional AIOps models in terms of cross-service and cross-task flexibility. LLMs offer a solution by efficiently interpreting unstructured data such as service logs and troubleshooting guides, extracting essential information from large data volumes [25, 34]. Once an anomaly or incident is detected, RCA is crucial for identifying the underlying causes. Leveraging information extracted and summarized by LLM-based data pre-processing, LLMs can comprehend and localize incidents effectively [2, 9]. Incident mitigation follows RCA, where LLMs assist in automating mitigation steps previously managed by on-call engineers (OCEs) [3, 48]. These LLM-based solutions empower each AIOps task by reducing human effort and increasing automation. However, despite these advancements in each AIOps task, the entire AIOps process is not yet fully self-managed, requiring additional efforts for managing large-scale cloud service systems. The concrete realization of ACV in our paper offers a demonstration that unifies and coordinates LLM-based AIOps task solutions to achieve a self-managed cloud service system with high efficiency, availability and reliability.

### 3 SERVICE MANAGEMENT WITH LLM-BASED MULTI-AGENTS

In this section, we demonstrate the application of ACV in service management using LLM-based agents, realizing the general architectural considerations discussed in the ACV paper.

[Figure 1]

- Figure 1: Hierarchical service management framework with LLM-based multi-agent design.

Architecture Overview. The ACV paper emphasizes that autonomic management elements operate at various levels, from individual service components to service groups fulfilling specific features, and up to entire business functions. This concept aligns with the design of LLM-based multi-agent system [19, 54], where a hierarchical architecture is naturally suited to manage services at these different levels. Considering the varied complexity of management requests at different levels, we propose a two-level autonomic management architecture in this paper. As shown in Figure 1, this hierarchical service management framework provides flexibility for the maintainers to manage the system at different appropriate levels. For complex management requests, such as “ensure end-to-end latency below 20 ms”, which involve multiple services, the high-level LLM-based group manager processes the request by breaking it down into simpler low-level tasks that can be handled by the respective low-level autonomic agents. Conversely, simpler requests can be sent directly to the low-level autonomic agents. For example, a request to “scale replicas to 3” can be directed straight to the corresponding low-level autonomic agent, bypassing the high-level group manager. The detailed working mechanism of this two-level architecture will be discussed later in this section.

Low-Level Autonomic Agent. Traditionally, a system consists of elements with limited autonomic capabilities responsible for basic

service functions, such as a pod in Kubernetes. These elements operate reactively, executing tasks like responding to API calls upon request. By integrating an autonomic layer to manage each element, these managed elements can enhance their functionality significantly. They can monitor their own health, analyze potential issues, and develop mitigation plans to resolve problems autonomously.

Before the advent of LLMs, autonomic capabilities were achieved through explicit development of Monitor, Analyze, Plan, Execute, and Knowledge modules [27]. However, LLMs streamline these functions into two core modules: Planner and Executor. The Planner generates execution steps to achieve specific goals, including monitoring and analysis, which maybe passing down from the highlevel group manager (discussed in the below section) or requested directly by maintainers. These steps are often executable maintenance code (e.g., kubectl commands in Kubernetes), leveraging the coding capabilities and general knowledge of LLMs. The Executor carries out these steps, and the results are sent back to the Planner to verify goal achievement. If the goal is unmet, another self-correction cycle is initiated. This Plan-Execute feedback loop simplifies the traditional design of self-management agents, facilitating the development of autonomic elements at scale and across different hierarchical levels.

High-Level Group Manager. Low-level autonomic agents can interact through natural upstream and downstream dependencies or form a high-level service group managed by a high-level autonomic group manager. This high-level group manager is often necessary for complex tasks requiring coordinated group actions, such as resolving complex service incidents in cloud environments. It is also easier and flexible to express goal-oriented terms in natural language at the high-level service group, such as “reduce end-to-end service latency to 20 ms”.

The LLM-based agent in the high-level group manager also uses the Plan-Execute feedback mechanism. Unlike the lower-level autonomic elements, the Planner in the high-level group manager breaks down complex management requests into sub-tasks and generates detailed, step-by-step plans. Each step corresponds to a specific low-level autonomic agent and includes executable code, such as assigning sub-tasks and analyzing collected metrics. The Executor then carries out these steps by executing the code. Feedback from the low-level autonomic agents is sent back to the high-level Planner to determine if the goals have been met. The Planner can adjust the plan based on this feedback and track the progress of the execution. This proactive approach allows the system to maintain optimal performance and reduce maintenance time, minimizing the need for direct human intervention.

Three Working Mechanisms. We define three working mechanisms with our hierarchical service management framework, each designed for particular task management and execution scenarios: a) Low-Level Autonomic Agent Working Alone; b) Multiple Low-Level Autonomic Agents Collaborating Under a Manager; c) Intra-Communication among Low-Level Autonomic Agents. To be more specific:

Low-Level Autonomic Agent Working Alone. In this mode, the maintainer directly assign requests or tasks to a single low-level autonomic agent, requesting it to answer maintenance-related inquiries or take actions to fulfill maintenance requests. The agent operates independently to complete the assigned task. This mechanism is

[Figure 2]

- Figure 2: Illustration of the Sock Shop example: Each microservice is converted into an LLM-based autonomic agent, managed by an autonomic layer comprising a Planner and Executor. The high-level group manager employs a Plan-Execute feedback mechanism to generate plans and assign sub-tasks to low-level autonomic agents. A decoupled message queue system serves as the middleware to manage communication, collecting feedback and unresolved issues.

straightforward and effective for simple, isolated tasks that do not require coordination with other low-level autonomic agents.

Multiple Low-Level Autonomic Agents Collaborating Under a Manager. In this mode, the high-level group manager either receives tasks from the maintainer or issues are raised by the low-level autonomic agents themselves. The manager first decomposes the task based on the received message, generate a plan, and then assigns the sub-tasks to the relevant low-level autonomic agents for execution. The execution results from each agent are collected, and the manager may modify or proceed to the next steps based on these outcomes. If the original plan cannot be followed, it is adjusted accordingly. This iterative process continues until the task is either completed or deemed unachievable. This mechanism is suitable for complex tasks that require collaborative efforts among multiple low-level autonomic agents.

Intra-Communication among Low-Level Autonomic Agents. When a low-level autonomic agent encounters an issue it cannot resolve independently, it can seek assistance from other agents without involving of the high-level group manager. This mechanism facilitates internal communication among low-level autonomic agents, allowing them to collaborate and attempt to fix the issue by directly communicating with each other.

To facilitate these working mechanisms, we use a message queue system as middleware for message passing and storage. This approach unifies the basic framework of all three working mechanisms, enhancing overall service availability and reliability. Additionally, the message queue system decouples message passing through asynchronous message queues, improving system robustness and flexibility. Note that we implement the first two mechanisms in the example below for simplicity and better illustration.

Example Implementation on Sock Shop. There are several microservce demos such as Online Boutique [16] and Death Star [13]. We select Sock Shop [1] as the microservices demo application due to its comprehensive example of a real-world e-commerce application, modular microservices design, and Kubernetes-native setup. It offers a simple yet accessible start point to demonstrate sophisticated microservices and Kubernetes concepts.

Sock Shop is a microservices demo application that emulates an e-commerce website for selling socks, designed to help users understand Kubernetes and microservices architecture. It consists of various microservices managed by Kubernetes, each handling different aspects of the e-commerce site, such as the website frontend, product catalog, orders, and payment (shown in Figure 2 left). Each microservice is containerized using Docker, ensuring consistent and isolated deployment across different environments. These microservices are deployed and orchestrated by Kubernetes, which offers certain capabilities for managing them. However, Kubernetes relies on additional tools like Prometheus [47] to perform tasks like monitoring and metrics collection. It also requires other essential declarative actions include auto-scaling, resource management, alerting, fault troubleshooting, and performance optimization. Currently, these actions are mostly performed manually by maintainers, requiring significant effort and domain expertise.

As shown in Figure 2 right, we demonstrate the application of LLM-based autonomic management on the Sock Shop microservice demo project in Kubernetes. This autonomic management framework is based on the previously discussed LLM-based two-level hierarchical framework. Briefly, the high-level group manager oversees the entire group of microservices, interprets high-level tasks from maintainers, and breaks down these tasks into executable steps assigned to low-level autonomic agents. The high-level group manager also handles issues raised by low-level autonomic agents. Note that the high-level group manager does not directly operate any microservice; instead, each low-level autonomic agent manages its corresponding microservice, responding to tasks assigned by the high-level group manager, such as monitoring and maintenance.

To be more specific, each Sock Shop containerized microservice is managed by a low-level LLM-based autonomic agent, making it an LLM-enhanced service component. Each agent is responsible for managing its own component. For example, the Front-end component is converted into the Front-end agent, enhanced by the LLM-based autonomic agent. Messages containing sub-tasks or maintainer requests are passed to the Front-end agent. Through the iterative process with the Planner and Executor, maintenance tasks are completed within the agent. Responses and feedback

are then sent to the high-level group manager through the manager’s message queue. If there is an unresolved problem, an issue is sent to the manager’s message queue, seeking assistance from the high-level group manager (shown in the red line in Figure 2). The high-level group manager analyzes the message, makes necessary adjustments to the plan, and then transmits the revised steps to the message queues of the corresponding low-level autonomic agents, coordinating collaboration among various agents.

### 4 EVALUATION BENCHMARK

After introducing the hierarchical LLM-based autonomic management framework and presenting an example ACV implementation, we now evaluate its performance to address the central question of this paper: how close are we to achieving ACV with LLMs? To answer this, we introduce a taxonomy categorizing tasks by their levels of autonomy and define specific task cases using Sock Shop.

### 4.1 Autonomous Levels in Service Maintenance

Inspired by the six levels of autonomous driving and the categorization of Personal LLM Agents’ duties based on intelligence levels in [37], we propose a taxonomy of five autonomy levels for service maintenance, depicted in Figure 3. The basic L1 and L2 levels represent the foundational maintenance capabilities of autonomic management agents, such as understanding users’ intent and possessing necessary service maintenance knowledge (e.g., writing correct maintenance code using kubectl commands). At Level 1 (“Simple Step Following”), we assess whether the agent can determine correct operational commands to fulfill specific imperative instructions, such as scaling replicas to three. Level 2 (“Deterministic Task Automation”) additionally requires the agent to possess the planning capability by decomposing a complex task into smaller executable steps, such as checking the service health status by generating a series of metric collection and understanding steps.

L1 and L2 levels are characterized by imperative tasks, focusing on fulfilling specified tasks. Higher autonomy levels require the agent to fulfill declarative tasks, proactively performing actions to achieve predefined goals or states. These high-level goals align with the four self-management objectives outlined in the ACV paper. For simplicity, this work focus on tasks related to Self-Optimization and Self-Healing, leaving the other two areas for future research.

Achieving Self-Optimization and Self-Healing necessitates three key capabilities in the autonomic management agent: “Proactive Issue Detection”, “Automatic Root cause Analysis”, and “Full SelfMaintenance” with the generation and execution of mitigation solutions. These capabilities correspond to L3, L4, and L5 levels in our taxonomy shown in Figure 3. With this five-level taxonomy established, we proceed to design specific evaluation tasks tailored to each level within the Sock Shop context. These tasks will serve as benchmarks to quantitatively assess the capabilities of current LLMs in advancing towards the realization of ACV.

### 4.2 Online Live Evaluation Benchmark

In traditional benchmarking, evaluations are typically performed against a specific dataset in an offline manner. However, to assess whether an agent can achieve the complete Detection-RCAMitigation maintenance cycle, we need an Online Live Evaluation

Benchmark that operates within a functional service environment. To create this environment, we deploy the Sock Shop service using Kubernetes and simulate traffic to ensure it functions as a live system. This setup allows us to introduce various tasks, such as metric collection and health checks, to evaluate the basic L1 and L2 capabilities. For evaluating L3, L4, and L5 capabilities, we further employ Chaos Engineering techniques [44] to intentionally inject faults or induce performance issues.

Table 1 lists 16 evaluation tasks for L1 and L2 levels, distinguishing between 12 tasks for low-level and 4 tasks for high-level group manager. Low-level tasks are directly applied to agents that manage individual service component, while high-level tasks involve the manager agent, allowing us to examine individual and collaborative task handling. For this study, we focus on the Catalogue for low-level evaluation and include the high-level group manager and Front-end for high-level evaluation tasks. These tasks encompass basic Deployment and Creation Management (DCM) operations as well as Runtime Management (RM) activities, reflecting common microservice management operations. Traffic load levels are specified to ensure tasks, such as reducing latency, are meaningful.

In contrast, L3, L4, and L5 evaluations involve injecting specific faults or issues to trigger the evaluation. We introduce three types of faults and performance issues:

- • Pod Failure: Replace Catalogue pod image with a fake and non-functional one.
- • CPU Stress: Occupy 100% of Catalogue pod’s CPU resources.
- • Rising Traffic: Gradually increase traffic, leading to high resource usage and extended service latency. This pattern is applied directly to Catalogue for low-level tasks and to Front-end for high-level tasks.

While many other faults and issues exist in microservice management, these three are representative and pertinent for this study.

We then evaluatethe system’sability to perform self-management operations to meet predefined Service Level Objectives (SLOs) under these injected conditions. Within the Sock Shop context, we define the following SLOs:

- (1) All service components maintain a healthy READY state.
- (2) CPU and memory usage for each component remains under 50% of allocated resources.;
- (3) The P99 latency for each component is stable, with an average P99 latency below 200ms.

Evaluation of L3/L4/L5 tasks are also distinguished by both lowlevel autonomic agents and high-level group manager, with the above SLOs communicated to these agents based on task requirements. Given that L3, L4, and L5 tasks are typically executed in a unified Detection-RCA-Mitigation sequence, we implement a combined task to evaluate all three levels simultaneously. A detailed description of these tasks is provided in Section A.2 in the Appendix.

### 4.3 Evaluation Metric

Our evaluation focuses on two main types of metrics: efficiency and quality. For efficiency metrics, we monitor the number of steps taken in each evaluation, with each step defined as a single call to the base LLM. We also track the number of communication rounds between high-level group manager and low-level autonomic agents for tasks initiated by the high-level group manager. Steps that result

[Figure 3]

#### Figure 3: Taxonomy of autonomous levels in service maintenance, focusing on Self-Healing and Self-Optimization.

|Request Level|Operation Type<br><br>|Level|Task Name<br><br>|Task Description<br><br>|Traffic Load|
|---|---|---|---|---|---|
|Low<br><br>|DCM DCM DCM DCM RM RM RM RM|L1 L1 L1 L1 L1 L1 L1 L2<br><br>|Deployment Creation Deployment Update Roll Back Pod Restart Metric Collection-CPU Metric Collection-Latency Manual Scaling Health Check|Create a deployment using Catalogue’s deployment YAML file. Update Catalogue’s environment variable logger_flag to true. Rollback Catalogue’s image version to 0.3.4. Restart all the pods of Catalogue immediately. Report the CPU usage of Catalogue. Report the P99 latency of Catalogue. Scale Catalogue’s replicas to 3. Check ’s healthy status immediately.<br><br>|Moderate on Cat. Moderate on Cat. Moderate on Cat. Moderate on Cat. Moderate on Cat. Moderate on Cat. Heavy on Cat. Heavy on|
| |RM RM RM<br><br>RM<br><br>|L2 L2 L2<br><br>L2|Performance Check Auto Scaling Latency Reduction<br><br>CPU Reduction|Catalogue<br><br>Check whether Catalogue’s performance is normal immediately. Implement auto-scaling with sensible thresholds for Catalogue. Reduce the P99 latency of Catalogue to under 300 ms. Reduce the average CPU usage of Catalogue to under 30% of resource limit.<br><br>|Cat.<br><br>Moderate on Cat. Heavy on Cat. Heavy on Cat.<br><br>Moderate on Cat.|
|High|RM RM<br><br>RM<br><br>RM|L2 L2<br><br>L2<br><br>L2<br><br>|Latency Reduction-Cat. CPU Reduction-Cat.<br><br>Latency Reduction-Group<br><br>CPU Reduction-Group<br><br>|Reduce the P99 latency of Catalogue to under 300 ms. Reduce the average CPU usage of Catalogue to under 30% of resource limit. Reduce the total P99 latency of Catalogue and Front-end to under 400 ms. Reduce each Catalogue and Front-end’s average CPU usage to under 30%.|Heavy on FE. Heavy on FE.<br><br>Heavy on FE.<br><br>Heavy on FE.|

- Note1: No initial deployment of Catalogue for the Deployment Creation task, and for the Roll Back task, the initial pod image version is 0.3.6.
- Note2: Definition of each Traffic Load can be found in Table 8. Cat and FE are abbreviations for Catalogue and Front-end, respectively. Table 1: Definition of L1 and L2 tasks within the context of Sock Shop.

in execution errors are included as well, as these errors reflect the base LLM’s ability to generate correct service maintenance code.

On the quality side, we measure the task fulfillment rate at various levels. For L1 and L2 tasks, we assess whether the assigned tasks are completed successfully. For L3 tasks, we evaluate the agent’s ability to correctly judge if the system meets the SLOs. In L4 tasks, we determine if the management agents can identify the correct root causes. Finally, for L5 tasks, we check whether injected faults or issues are successfully mitigated. Additionally, for tasks assigned to high-level group manager, we evaluate the high-level agent’s ability to correctly delegate tasks to low-level autonomic agents.

### 5 EXPERIMENT 5.1 Setup

SockShopDeployment.SockShopisdeployedona Minikube [42] cluster with 6 cores and 16 GB of memory, utilizing an Intel Xeon Gold 6338 CPU @ 2 GHz server. The deployment includes an enabled metric server and a Prometheus server, configured according to the Sock Shop deployment files. Load simulation is applied using Locust [38], with traffic levels detailed in Table 8 in the Appendix. The faults of Pod Failure and CPU Stress are injected by utilizing the Chaos Mesh tool [41].

Agent-based Management System. The agent-based management system introduced in Section 3 is implemented by leveraging

| |Low-Level|High-Level|
|---|---|---|
|L1 L2|1. Deploy Sock Shop and ensure stable traffic;<br>2. Send task to Catalogue’s management agent;<br>3. Task evaluation;<br>4. Repeat the above 3 times.<br>|1. Deploy Sock Shop and ensure stable traffic;<br>2. Send task to high-level manager;<br>3. Task evaluation;<br>4. Repeat the above 3 times.<br><br><br>|
|L3 L4 L5<br><br>|1. Deploy Sock Shop and ensure stable traffic;<br>2. Inject faults/issues;<br>3. Send task to Catalogue’s management agent;<br>4. Perform 3 consecutive evals;<br>5. Repeat the above 3 times.<br>|1. Deploy Sock Shop and ensure stable traffic;<br>2. Inject faults/issues;<br>3. Send task to high-level manager;<br>4. Perform one evaluation;<br>5. Repeat the above 3 times.<br>|

#### Table 2: Experiment procedures for different configurations.

the multi-agent framework AutoGen [56]. It incorporates agent groups consisting of LLM-based agents and non-LLM-based code executors for each microservice component, serving as low-level autonomic agents. Similarly, the high-level group manager comprises a LLM-based agent and a non-LLM-based code executor. Communication among these agents is facilitated by RabbitMQ [49], and GPT-4 Turbo is adopted as the underlying LLM.

The prompts for low-level autonomic agents and high-level group manager are given in Sections C in the Appendix. Most of instructions in them are generally applicable to other microservices managed by Kubernetes. However, in the prompt for lowlevel autonomic agents, we include instructions that is tailored specifically to the Sock Shop service. Those instructions are effective at eliminating inefficient self-correcting steps. For example, agents might use the wrong label selector “-l app=Catalogue” instead of the correct “-l name=Catalogue” to identify Catalogue. While agents can correct such errors through several self-correction rounds, these mistakes significantly disrupt the execution flow since many kubectl commands depend on accurate label selectors. Similarly, querying Prometheus metrics correctly is often challenging because it requires precise input of metric names and filters, which are service-specific and not typically known to LLMs. Although these specific instructions reduce generality, they are necessary, especially for low-level autonomic agents managing elements with domain-specific information that LLMs lack.

Experiment Procedure. As detailed in Section 4.2, tasks are categorized for management agents at both low and high levels. Given the distinct characteristics of L1/L2 and L3/L4/L5 tasks, we establish four distinct experiment configurations, each outlined comprehensively in Table 2. Overall, the procedure of each experiment run involves the environment setup (i.e., deploying Sock Shop and ensuring stable traffic), sending task requirement to corresponding management agents and finally evaluating the task performance. All experiments are randomly repeated three times to reduce statistical fluctuation. For L3/L4/L5 tasks managed by low-level autonomic agents, three consecutive evaluations are allowed to account for potential missed detections in initial trials. Conversely, for L3/L4/L5 tasks managed by high-level agents, a single evaluation suffices as multiple agents collaborate to detect issues.

| |Task Name|Is Passed? Steps<br><br>Steps with Exec. Error|
|---|---|---|
|L1<br><br>|Deployment Creation Deployment Update Roll back Pod Restart Metric Collection-CPU Metric Collection-Latency Manual Scaling<br><br>|[1,1,1] [3,4,4] [0,0,0] [1,1,1] [5,5,5] [0,0,0] [1,1,1] [5,9,6] [0,0,0]<br>[1,1,1] [4,5,5] [0,0,0]<br>[1,1,1] [5,5,4] [0,0,0] [1,1,1] [3,3,3] [0,0,0] [1,1,1] [9,5,6] [0,0,0]<br>|
|L2<br><br>|Health Check Performance Check Auto Scaling Latency Reduction CPU Reduction|[1,1,1] [7,7,4] [0,0,0] [1,1,1] [4,4,4] [1,0,0] [1,1,1] [6,11,4] [0,2,0] [1,1,0] [11,13,15] [0,0,1] [1,0,1] [11,10,8] [0,0,0]<br><br>|
|L1 L2<br><br>|Average Average|1.0 4.9 0 0.87 7.9 0.3<br><br>|

Table 3: Results for L1 and L2 tasks applied to the low-level autonomic agent of Catalogue. Each element in the 3-tuple list corresponds to one of the 3 repeated experiment runs.

Evaluation. The evaluation metrics discussed in Section 4.3 were gathered through a manual review of the agents’ chat history. This review was performed independently by two authors with expertise in Kubernetes service maintenance. A sample log of the agents’ chat history corresponding to the Latency Reduction task applied to Catalogue can be accessed via this link.1

### 5.2 Results for Tasks Applied to the Low-Level Autonomic Agent

Table 3 presents the experimental results for the L1 and L2 tasks applied on the low-level autonomic agent of Catalogue. Details include the pass status, the number of steps taken, and any steps with execution errors. The key findings are summarized as follows:

- • The L1 and L2 tasks demonstrated high task completion rates, achieving 100% for L1 and 87% for L2. One of the failed L2 experiments was the Latency Reduction, which failed because an ineffective action was made by editing another service configuration. The other unsuccessful experiment, CPU Reduction, failed due to misguided actions, particularly the reduction of CPU requests and limits, resulting in non-operational pods.
- • On average, the L1 task required approximately 5 steps to complete, while the L2 task required 8 steps, indicating that L2 tasks often involve additional planning steps. While core operations often required only 1 or 2 steps, those extra steps were related to precautionary checks before and after taking actions.
- • Code execution errors were minimal, and the system generally self-corrected these errors, underscoring the robustness of the LLM-based management system. Theseresultsindicatethatlow-level LLM-based autonomic agents

are highly effective in performing basic service maintenance tasks in Kubernetes, establishing a solid foundation for advancing towards higher levels of autonomy.

The results for L3, L4, and L5 tasks applied to the low-level autonomic agent of Catalogue are summarized in Table 4. Detailed evaluations, including failure reasons, are provided in Table 10

1https://paste.ubuntu.com/p/HnnGJ9qPgM/

|Injected Faults|Trial Index<br><br>|Steps (3 Evals)<br><br>Is Task Passed? L3 L4 L5 First Overall|
|---|---|---|
|Pod Failure<br><br>|1<br>2<br>3<br>|[7,17,15] 3/3 2/3 0/3 0 0 [9,11,11] 2/3 1/1 1/1 1 1<br><br>[21,6,7] 3/3 1/1 1/1 1 1<br><br>|
|CPU Stress<br><br>|1<br><br>2<br><br>3<br><br><br>|[7,9,9] 2/3 1/2 1/2 0 1 [13,11,12] 3/3 0/3 0/3 0 0<br><br>[4,9,1] 1/3 0/1 0/1 0 0|
|Rising Traffic<br><br>|1<br>2<br>3<br>|[4,4,1] 2/3 0/0 0/0 0 0 [18,7,7] 3/3 0/0 0/0 - [13,7,4] 1/3 1/1 0/1 0 0<br><br>|
|Average<br><br>| |8.8 (per eval) 0.74 0.5 0.25 0.25 0.38|

Table 4: Result summary for L3/L4/L5 tasks applied to the low-level autonomic agent of Catalogue. “Trial Index” represents each experiment run, and the element in the 3-tuple list of “Steps” corresponds to one of the three consecutive evaluations. The “x/y” pattern in the “Is Task Passed?” columns denotes that “x” out of “y” cases are passed. More result details can be found in Table 10.

|Task Name|Trial Index<br><br>|Rounds Steps<br><br>Is Correct Task Assignment?<br><br>Is Task Passed?|
|---|---|---|
|Latency Reduction -Cat.|1<br><br>2<br><br>3<br><br><br>|3 19 3/3 1<br>4 35 4/4 1 2 8 2/2 1<br>|
|CPU Reduction -Cat.|1<br><br>2<br><br>3<br><br><br>|3 19 2/3 0 3 27 3/3 1 3 19 2/3 0|
|Latency Reduction -Group|1<br><br>2<br><br>3<br>|3 29 3/3 1 3 46 3/3 1 3 36 2/3 1<br><br>|
|CPU Reduction -Group<br><br>|1<br>2<br>3<br>|3 35 1/3 0<br><br>3 36 3/3 0<br><br><br>5 63 5/5 0|
|Average<br><br>| |3.2 31 0.87 0.58|

#### Table 5: Result summary for L2 tasks applied to the highlevel group manager. More details are given in Table 11.

in the Appendix. Table 4 presents the number of steps for three consecutive evaluations per experiment run and assesses task completion at different granularities: detection of failures/issues (L3), identification of root causes (L4), issue mitigation (L5), and task resolution in the first evaluation (First) or overall after three evaluations (Overall). Notably, not all evaluations involved valid L4/L5 tasks, as some issues were resolved prior to evaluation due to previous interventions or unsuccessful fault injection (e.g., Trial 2 in the Rising Traffic case). Key observations from Table 4 include:

- • On average,approximately9stepswere needed to perform L3/L4/L5 tasks, indicating higher complexity compared to L1/L2 tasks. The number of required steps varied widely across the three evaluations in one experiment run. This variation is attributed to the necessity of the full Detection-RCA-Mitigation cycle (e.g., some evaluation terminated early if no issue was detected).
- • The task pass rates decreased from 0.74 (L3) to 0.5 (L4) and 0.25 (L5), reflecting the increased complexity of higher-level selfmanagement tasks. Coupled with the high completion rates for L1 and L2 tasks, we conclude that the low-level autonomic agent can achieve Level 3 autonomy in Self-Optimization and Self-Healing, according to the taxonomy in Section 4.1. Although full L5 autonomy has not been achieved by current LLMs, these results demonstrate significant potential for LLMs in advancing ACV.
- • The task completion rate improved from 0.25 to 0.38 when evaluations were allowed to be performed up to three times. This indicates that repeated Detection-RCA-Mitigation cycles enhance the system’s ability to mitigate issues.

### 5.3 Results for Tasks Applied to the High-Level Group Manager

We next present the results of applying tasks to the high-level group manager, focusing on the task completion rate, task assignment to low-level autonomic agents, and how these agents fulfill sub-tasks.

- Table 5 summarizes the results for L2 tasks applied to the high-

level group manager, with detailed results in Table 11. Besides

the number of steps and the overall task pass status, Table 5 also indicates the number of communication rounds between high-level and low-level autonomicagents and the accuracy of task assignment by the high-level agent. Key observations include:

- • On average, it took about 3 rounds to complete each L2 task. These rounds typically involve: i) the first round of collecting metrics to determine task necessity; ii) the second round where low-level autonomic agents perform the actual task; iii) the final round of summarizing and reporting results, and terminating the task. For instance, Figure 4 illustrates the sequence for the Latency Reduction-Group task. Upon receiving the task to reduce P99 latency of Front-end and Catalogue to under 400 ms, the high-level group manager first gathered P99 latency metrics for both services. After finding that the latencies were about 400 ms and 700 ms respectively, the manager assigned the task of reducing latency to under 200 ms. Low-level autonomic agents carry out the required actions and reported back, enabling the high-level group manager to summarize and terminate the task.
- • Each task took approximately 31 steps to complete, significantly higher than the 8 steps required for L2 tasks handled by lowlevel autonomic agents (Table 3). The increase in steps is due to the higher complexity of tasks involving multiple low-level autonomic agents (e.g., Latency Reduction-Group task) and additional planning and communication steps required in the hierarchical management architecture. Notably, the steps taken by low-level autonomic agents during task execution (e.g., the round for the latency reduction task) were comparable to those in Table 3. Thus, since a hierarchical management structure introduces extra planning and communication costs, directly assigning maintenance requests to specific low-level autonomic agents might sometimes be more efficient.
- • The high-level management system demonstrates strong task assignment capabilities, achieving an accuracy of 0.87 for L2 tasks. Analysis of task completion reveals that while latency reduction tasks exhibit high success rates, tasks aimed at CPU reduction suffer substantial accuracy declines. Detailed examination of failure instances, detailed in Table 11, highlights common issues

Figure 4: Sequence diagram for the Latency Reduction-Group task applied to the high-level group manager.

|Injected Faults<br><br>|Trial Index<br><br>|Rounds Steps<br><br>Is Task Passed? L3 L4 L5 Overall|
|---|---|---|
|Pod Failure|1<br><br>2<br><br>3<br>|2 28 1/1 1/1 1/1 1<br><br>3 31 2/2 0/1 1/1 1<br><br><br>2 30 1/1 0/1 1/1 1|
|CPU Stress<br><br>|1<br>2<br>3<br>|5 61 3/4 1/3 1/3 1 4 69 0/0 0/0 2/4 1 4 50 0/0 0/0 2/3 1<br><br>|
|Rising Traffic|1<br><br>2<br><br>3<br><br><br>|4 66 0/0 0/0 0/4 0 2 21 0/0 0/0 0/0 0 4 49 2/2 2/2 0/2 0|
|Average| |3.3 45 0.9 0.5 0.42 0.67|

- Table 6: Result summary for L3/L4/L5 tasks applied to the high-level group manager. The “x/y” pattern in the “Is Task Passed?” columns denotes that “x” out of “y” cases are passed. More result details can be found in Table 12.

such as erroneous CPU request reductions (mirroring the L2 Latency Reduction task failures in Table 3) and horizontal scaling actions insufficient for task fulfillment.

The results for the L3, L4, and L5 tasks are summarized in Table 6, with detailed information provided in Table 12. These tables list the number of rounds and steps performed, as well as the task completion rates at each level and overall, including whether the injected faults were resolved. Key findings include:

- • The average number of rounds required was approximately 3, similar to the L2 task. The procedure for these rounds typically followed the similar pattern of metric collection, task execution, and result summary. However, additional rounds of task execution were sometimes necessary if the previous ones did not complete the task.
- • On average, 45 steps were required, significantly more than the 31 steps needed for L2 tasks. This increase may be due to the

greater complexity and deeper analysis required for each round of the L3/L4/L5 sub-tasks.

• The overall task completion rate decreased from 0.9 (L3) to 0.5 (L4) and 0.42 (L5), consistent with the pattern observed in low-level autonomic agent tasks. However, the actual completion rates were higher than those for the corresponding low-level tasks, suggesting that including a hierarchical management structure may improve the overall task completion rate. This observation further confirms our previous proposition that attaining Level 3 autonomy is feasible with the existing LLM-based agent management system.

### 5.4 Failure Analysis

Lastly, we present an in-depth analysis of failed cases in tasks related to L3 (incorrect detection), L4 (incorrect root cause analysis), L5 (mitigation failure), and erroneous high-level task assignment. The detailed results, categorized by failure type and potential improvement directions, are shown in Table 7. These failures stem from issues such as instruction-following errors (e.g., task omission), hallucinations, deficiencies in reasoning capabilities, and insufficient domain knowledge.

To address these issues, we propose two primary strategies. First, we can employ a superior base LLM, either a state-of-the-art general LLM or a domain-specific LLM fine-tuned with service maintenance knowledge. Second, we can enhance our existing agent framework by integrating additional modules. These modules might include mechanisms for incorporating domain knowledge using retrievalaugmented generation techniques [35] or the addition of critic agents to mitigate hallucinations and bolster reasoning capabilities.

### 6 DISCUSSION

This study systematically evaluates the feasibility of realizing autonomic computing using LLM-based agents. Our comprehensive

|Task with Failure<br><br>|Failure Reason|Failure Count|Potential Solution Direction|
|---|---|---|---|
|L3 (Wrong Detection)|hallucination during numerical comparison task omission<br><br>|4 3<br><br>|reduce hallucination better instruction following|
| | | | |
| |mis-guided by previous action execution output<br><br>|1|improve reasoning<br><br>|
|L4 (Wrong RCA)|unable to collect important information (e.g., CPU limit) mis-guided by previous action execution output<br><br>|5 3|better reasoning; more domain knowledge improve reasoning|
| | | | |
| |task omission|2<br><br>|better instruction following|
|L5 (Failed Mitigation)|ineffective action adopted action insufficient to resolve issue<br><br>|4 4<br><br>|improve reasoning improve reasoning|
| | | | |
| |unable to report correctly (e.g., hallucinated report, omit report)<br><br>|4<br><br>|better instruction following; reduce hallucination|
| |wrong choice of action|3<br><br>|improve reasoning|
| |failed to execute action (e.g., action generated but not applied)|3<br><br>|better instruction following|
|Wrong High-level Task Assignment<br><br>|task omission affected by incorrect low-level report|7 5<br><br>|better instruction following improve reporting at low-level|
| | | | |
| |wrong reasoning based on low-level report|2<br><br>|improve reasoning|

#### Table 7: Failure reason analysis and potential solution directions for tasks with failures.

experiments, conducted with the microservice Sock Shop in Kubernetes, highlight several key topics that merit further discussion.

Alternative LLM-based Management System with Agents. We utilize a hierarchical multi-agent system to decompose tasks at different levels, employing the Plan-Execution feedback mechanism in each agent for autonomic task management. Other agent-based solutions could also be considered, such as integrating critic agents to oversee the entire process, implementing a memory mechanism to store short/long-term maintenance history, and including a lifelong self-learning module to continuously enhance agent performance [48, 53, 56]. Additionally, different LLMs could be adopted as base models. While GPT-4 Turbo serves as a capable representative, other LLMs might exhibit different behaviors in service maintenance contexts. Fine-tuning LLMs on service maintenance-related data is also an existing area worth exploration.

Evaluation Benchmark Enrichment. The benchmark tasks in this study focus on Self-Optimization and Self-Healing. While this is a first attempt to systematically assess ACV using LLM-based agents, expanding the scope of the evaluation benchmark in future work would be valuable. Including Self-Configuration and SelfProtection would demonstrate the system’s flexibility, evolution capability, and resilience in hostile cyber environments. Additionally, while we aim to cover most failure or issue types, some, such as grey failures [22] and metastable failures [21], are not included. Further, although the Sock Shop microservice covers major aspects of microservice management using Kubernetes, it lacks the complexity, scalability, and diversity of real-world large-scale cloud services. Evaluating more diverse benchmarks, such as Online Boutique [16] and Death Star [13], and those beyond microservice management, would be a logical next step. Finally, although Section 3 outlines the intra-communication mechanisms among low-level autonomic agents, concrete evaluation tasks are needed to assess agents’ collaborative issue-solving capabilities in self-organizing scenarios without high-level supervision. Such patterns are crucial for managing extensive distributed computing systems that do not have centralized management.

Application in Practice. Two immediate real-world implications of our research are discernible. Firstly, our agent-based management system, initially developed for Sock Shop, could be generalized as a

standard component for existing Kubernetes management frameworks. This abstraction could manifest as middleware tools akin to Prometheus and other metric/log collection tools, seamlessly transforming each microservice component into autonomic units during application development. Such integration would augment Kubernetes’ capabilities in declarative, autonomic, and proactive management of service deployments. Second, building a live evaluation benchmark would be an effective way to facilitate replicating real-world issues and assessing management systems’ ability to resolve them dynamically. This live benchmark would serve as a versatile test bed for various experiments and the development of new solutions to improve existing management systems.

### 7 CONCLUSION

In conclusion, this study has presented a novel approach to advancing the Vision of Autonomic Computing by leveraging Large Language Models within a hierarchical multi-agent framework for microservice management. Through the introduction of a five-level taxonomy for autonomous service maintenance and the development of an online evaluation benchmark based on the Sock Shop microservice demo, we systematically assessed the performance of our proposed system. Our findings reveal that the LLM-based multiagent framework achieves Level 3 autonomy, effectively handling detection and resolution of specific issues, although there remains room for enhancement in areas such as root cause analysis and mitigation. This work not only demonstrates the potential of LLMs to significantly contribute towards realizing ACV but also sets the stage for future research to address the remaining challenges and push the boundaries of self-managing computing systems further.

### ACKNOWLEDGMENTS

We thank Lingxiang Hu, Shurun Yuan and Haiyang Ding for early participation of this work. We also appreciate David Liu for the insightful discussions and Bo Qiao for the assistance in setting up the experiment server.

### REFERENCES

- [1] [n.d.]. Sock Shop: A Microservices Demo Application. https://github.com/ microservices-demo/microservices-demo. Accessed: 2024-06-18.
- [2] Toufique Ahmed, Supriyo Ghosh, Chetan Bansal, Thomas Zimmermann, Xuchao Zhang, and Saravan Rajmohan. 2023. Recommending Root-Cause and Mitigation Steps for Cloud Incidents using Large Language Models. In 45th IEEE/ACM International Conference on Software Engineering, ICSE 2023, Melbourne, Australia, May 14-20, 2023. IEEE, 1737–1749. https://doi.org/10.1109/ICSE48619.2023.00149
- [3] Kaikai An, Fangkai Yang, Liqun Li, Zhixing Ren, Hao Huang, Lu Wang, Pu Zhao, Yu Kang, Hua Ding, Qingwei Lin, Saravan Rajmohan, and Qi Zhang.

2024. Nissist: An Incident Mitigation Copilot based on Troubleshooting Guides. CoRR abs/2402.17531 (2024). https://doi.org/10.48550/ARXIV.2402.17531 arXiv:2402.17531

- [4] Aradea, Iping Supriana, and Kridanto Surendro. 2023. ARAS: adaptation requirements for adaptive systems. Autom Softw Eng 30, Article 2 (2023). https: //doi.org/10.1007/s10515-022-00369-3
- [5] Paolo Arcaini, Elvinia Riccobene, and Patrizia Scandurra. 2015. Modeling and Analyzing MAPE-K Feedback Loops for Self-Adaptation. 2015 IEEE/ACM 10th International Symposium on Software Engineering for Adaptive and Self-Managing Systems (2015), 13–23. https://api.semanticscholar.org/CorpusID:11757466
- [6] Sagar Behere and Martin Törngren. 2016. A functional reference architecture for autonomous driving. Information and Software Technology 73 (2016), 136–150. https://doi.org/10.1016/j.infsof.2015.12.008
- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners. In Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 1877–1901. https://proceedings.neurips.cc/paper_files/paper/2020/file/ 1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf
- [8] Matteo Camilli, Raffaela Mirandola, and Patrizia Scandurra. 2023. XSA: eXplainable Self-Adaptation. In Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering (Rochester, MI, USA) (ASE ’22). Association for Computing Machinery, New York, NY, USA, Article 189, 5 pages. https://doi.org/10.1145/3551349.3559552
- [9] Yinfang Chen, Huaibing Xie, Minghua Ma, Yu Kang, Xin Gao, Liu Shi, Yunjie Cao, Xuedong Gao, Hao Fan, Ming Wen, Jun Zeng, Supriyo Ghosh, Xuchao Zhang, Chaoyun Zhang, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, and Tianyin Xu. 2024. Automatic Root Cause Analysis via Large Language Models for Cloud Incidents. In Proceedings of the Nineteenth European Conference on Computer Systems, EuroSys 2024, Athens, Greece, April 22-25, 2024. ACM, 674–688. https://doi.org/10.1145/3627703.3629553
- [10] Cloud Native Computing Foundation. 2024. Cloud Native Computing Foundation (CNCF). https://www.cncf.io/ Accessed: 2024-07-06.
- [11] Kubernetes Documentation. 2024. Why you need Kubernetes and what can it do. https://kubernetes.io/docs/concepts/overview/#why-you-need-kubernetesand-what-can-it-do Accessed: 2024-07-06.
- [12] João Figueira and Carlos Coutinho. 2024. Developing self-adaptive microservices. Procedia Computer Science 232 (2024), 264–273. https://doi.org/10.1016/ j.procs.2024.01.026 5th International Conference on Industry 4.0 and Smart Manufacturing (ISM 2023).
- [13] Yu Gan, Yanqi Zhang, Dailun Cheng, Ankitha Shetty, Priyal Rathi, Nayan Katarki, Ariana Bruno, Justin Hu, Brian Ritchken, Brendon Jackson, Kelvin Hu, Meghna Pancholi, Yuan He, Brett Clancy, Chris Colen, Fukang Wen, Catherine Leung, Siyuan Wang, Leon Zaruvinsky, Mateo Espinosa, Rick Lin, Zhongling Liu, Jake Padilla, and Christina Delimitrou. 2019. An Open-Source Benchmark Suite for Microservices and Their Hardware-Software Implications for Cloud & Edge Systems. In Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems (Providence, RI, USA) (ASPLOS ’19). Association for Computing Machinery, New York, NY, USA, 3–18. https://doi.org/10.1145/3297858.3304013
- [14] Simos Gerasimou, Radu Calinescu, and Giordano Tamburrelli. 2018. Synthesis of probabilistic models for quality-of-service software engineering. Autom Softw Eng 25 (2018), 785–831. https://doi.org/10.1007/s10515-018-0235-8

- [15] Sukhpal Singh Gill, Minxian Xu, Carlo Ottaviani, Panos Patros, Rami Bahsoon, Arash Shaghaghi, Muhammed Golec, Vlado Stankovski, Huaming Wu, Ajith Abraham, Manmeet Singh, Harshit Mehta, Soumya K. Ghosh, Thar Baker, Ajith Kumar Parlikad, Hanan Lutfiyya, Salil S. Kanhere, Rizos Sakellariou, Schahram Dustdar, Omer Rana, Ivona Brandic, and Steve Uhlig. 2022. AI for next generation computing: Emerging trends and future directions. Internet of Things 19 (2022),

100514. https://doi.org/10.1016/j.iot.2022.100514

- [16] GoogleCloudPlatform. [n. d.]. microservices-demo. Retrieved July 11, 2024 from https://github.com/GoogleCloudPlatform/microservices-demo
- [17] Hemanth Gopal, Guanqun Song, and Ting Zhu. 2022. Security, Privacy and Challenges in Microservices Architecture and Cloud Computing- Survey. arXiv:arXiv:2212.14422
- [18] Miguel A. Guinea-Cabrera and Juan A. Holgado-Terriza. 2024. Digital Twins in Software Engineering—A Systematic Literature Review and Vision. Applied Sciences 14, 3 (2024). https://doi.org/10.3390/app14030977
- [19] Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. 2024. Large language model based multi-agents: A survey of progress and challenges. arXiv preprint arXiv:2402.01680

(2024).

- [20] Mahmoud Hammad, Joshua Garcia, and Sam Malek. 2018. Self-Protection of Android Systems from Inter-component Communication Attacks. In 2018 33rd IEEE/ACM International Conference on Automated Software Engineering (ASE). 726–737. https://doi.org/10.1145/3238147.3238207
- [21] Lexiang Huang, Matthew Magnusson, Abishek Bangalore Muralikrishna, Salman Estyak, Rebecca Isaacs, Abutalib Aghayev, Timothy Zhu, and Aleksey Charapko.

2022. Metastable Failures in the Wild. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). USENIX Association, Carlsbad, CA, 73–90. https://www.usenix.org/conference/osdi22/presentation/huang-lexiang

- [22] Peng Huang, Chuanxiong Guo, Lidong Zhou, Jacob R. Lorch, Yingnong Dang, Murali Chintalapati, and Randolph Yao. 2017. Gray Failure: The Achilles’ Heel of Cloud-Scale Systems. In Proceedings of the 16th Workshop on Hot Topics in Operating Systems (Whistler, BC, Canada) (HotOS ’17). Association for Computing Machinery, New York, NY, USA, 150–155. https://doi.org/10.1145/3102980.3103005
- [23] Markus C. Huebscher and Julie A. McCann. 2008. A survey of autonomic computing—degrees, models, and applications. ACM Comput. Surv. 40, 3, Article 7 (aug 2008), 28 pages. https://doi.org/10.1145/1380584.1380585
- [24] Emilio Incerto, Mirco Tribastone, and Catia Trubiani. 2017. Software performance self-adaptation through efficient model predictive control. In 2017 32nd IEEE/ACM International Conference on Automated Software Engineering (ASE). 485–496. https: //doi.org/10.1109/ASE.2017.8115660
- [25] Pengxiang Jin, Shenglin Zhang, Minghua Ma, Haozhe Li, Yu Kang, Liqun Li, Yudong Liu, Bo Qiao, Chaoyun Zhang, Pu Zhao, Shilin He, Federica Sarro, Yingnong Dang, Saravan Rajmohan, Qingwei Lin, and Dongmei Zhang. 2023. Assess and Summarize: Improve Outage Understanding with Large Language Models. In Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ESEC/FSE 2023, San Francisco, CA, USA, December 3-9, 2023, Satish Chandra, Kelly Blincoe, and Paolo Tonella (Eds.). ACM, 1657–1668. https://doi.org/10.1145/3611643.3613891
- [26] K8sGPT. 2024. K8sGPT Documentation. https://docs.k8sgpt.ai/ Accessed: 2024-07-06.
- [27] J. Kephart and David M. Chess. 2003. The Vision of Autonomic Computing. Computer 36 (2003), 41–50. https://api.semanticscholar.org/CorpusID:44705503
- [28] Joerg Kienzle, Benoit Combemale, Gunter Mussbacher, Omar Alam, Francis Bordeleau, Lola Burgueno, Gregor Engels, Jessie Galasso, Jean-Marc Jézéquel, Bettina Kemme, Sébastien Mosser, Houari Sahraoui, Maximilian Schiedermeier, and Eugene Syriani. 2023. Global Decision Making Over Deep Variability in Feedback-Driven Software Development. In Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering (Rochester, MI, USA) (ASE ’22). Association for Computing Machinery, New York, NY, USA, Article 178, 6 pages. https://doi.org/10.1145/3551349.3559551
- [29] Joanna Kosińska and Krzysztof Zielinski. 2020. Autonomic Management Framework for Cloud-Native Applications. Journal of Grid Computing 18 (2020), 779 –

796. https://api.semanticscholar.org/CorpusID:225001553

- [30] Joanna Kosińska and Krzysztof Zielinski. 2023. Enhancement of Cloud-native applications with Autonomic Features. Journal of Grid Computing 21 (2023). https://api.semanticscholar.org/CorpusID:259859370
- [31] Joanna Kosińska and Krzysztof Zielinski. 2023. Experimental Evaluation of Rule-Based Autonomic Computing Management Framework for Cloud-Native Applications. IEEE Transactions on Services Computing 16 (2023), 1172–1183. https://api.semanticscholar.org/CorpusID:247489235
- [32] Shubham Kulkarni, Arya Marda, and Karthik Vaidhyanathan. 2023. Towards Self-Adaptive Machine Learning-Enabled Systems Through QoS-Aware Model Switching. In 2023 38th IEEE/ACM International Conference on Automated Software Engineering (ASE). 1721–1725. https://doi.org/10.1109/ASE56229.2023.00172
- [33] Philippe Lalanda, Julie A. McCann, and Ada Diaconescu. 2013. Future of Autonomic Computing and Conclusions. Springer London, London, 263–278. https://doi.org/10.1007/978-1-4471-5007-7_10

- [34] Van-Hoang Le and Hongyu Zhang. 2023. Log Parsing: How Far Can ChatGPT Go?. In 38th IEEE/ACM International Conference on Automated Software Engineering, ASE 2023, Luxembourg, September 11-15, 2023. IEEE, 1699–1704. https://doi.org/ 10.1109/ASE56229.2023.00206
- [35] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 9459–9474. https://proceedings.neurips.cc/ paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
- [36] Jialong Li, Mingyue Zhang, Nianyu Li, Danny Weyns, Zhi Jin, and Kenji Tei.

2024. Exploring the Potential of Large Language Models in Self-adaptive Systems. In Proceedings of the 19th International Symposium on Software Engineering for Adaptive and Self-Managing Systems (Lisbon, AA, Portugal) (SEAMS ’24). Association for Computing Machinery, New York, NY, USA, 77–83. https: //doi.org/10.1145/3643915.3644088

- [37] Yuanchun Li, Hao Wen, Weijun Wang, Xiangyu Li, Yizhen Yuan, Guohong Liu, Jiacheng Liu, Wenxing Xu, Xiang Wang, Yi Sun, Rui Kong, Yile Wang, Hanfei Geng, Jian Luan, Xuefeng Jin, Zilong Ye, Guanjing Xiong, Fan Zhang, Xiang Li, Mengwei Xu, Zhijun Li, Peng Li, Yang Liu, Ya-Qin Zhang, and Yunxin Liu. 2024. Personal LLM Agents: Insights and Survey about the Capability, Efficiency and Security. ArXiv abs/2401.05459 (2024). https://api.semanticscholar.org/CorpusID: 266933252
- [38] Locust. 2024. Locust. https://locust.io/ Accessed: 2024-07-02.
- [39] Ehud Malul, Yair Meidan, Dudu Mimran, Yuval Elovici, and Asaf Shabtai. 2024. GenKubeSec: LLM-Based Kubernetes Misconfiguration Detection, Localization, Reasoning, and Remediation. arXiv:2405.19954 [cs.CR] https://arxiv.org/abs/ 2405.19954
- [40] Nabor C. Mendonca, Pooyan Jamshidi, David Garlan, and Claus Pahl. 2021. Developing Self-Adaptive Microservice Systems: Challenges and Directions. IEEE Software 38, 2 (2021), 70–79. https://doi.org/10.1109/MS.2019.2955937
- [41] Chaos Mesh. 2024. Chaos Mesh. https://chaos-mesh.org/ Accessed: 2024-07-02.
- [42] minikube. 2024. minikube. https://github.com/kubernetes/minikube Accessed: 2024-07-16.
- [43] João Paulo Karol Santos Nunes, Shiva Nejati, Mehrdad Sabetzadeh, and Elisa Yumi Nakagawa. 2024. Self-Adaptive, Requirements-Driven Autoscaling of Microservices. In 2024 IEEE/ACM 19th Symposium on Software Engineering for Adaptive and Self-Managing Systems (SEAMS). 168–174. https://doi.org/10.1145/3643915. 3644094
- [44] Principles of Chaos Engineering. 2024. Principles of Chaos Engineering. https: //principlesofchaos.org/ Accessed: 2024-07-02.
- [45] OpenAI. 2023. GPT-4 Technical Report. arXiv:2303.08774 [cs.CL]
- [46] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), Vol. 35. Curran Associates, Inc., 27730–27744. https://proceedings.neurips.cc/paper_files/paper/2022/file/ b1efde53be364a73914f58805a001731-Paper-Conference.pdf
- [47] Prometheus. 2024. Prometheus - From metrics to insight. https://prometheus.io/ Accessed: 2024-07-06.
- [48] Bo Qiao, Liqun Li, Xu Zhang, Shilin He, Yu Kang, Chaoyun Zhang, Fangkai Yang, Hang Dong, Jue Zhang, Lu Wang, et al. 2023. Taskweaver: A code-first agent framework. arXiv preprint arXiv:2311.17541 (2023).
- [49] RabbitMQ. 2024. RabbitMQ. https://www.rabbitmq.com/ Accessed: 2024-07-02.
- [50] Sandro Speth. 2022. Semi-automated cross-component issue management and impact analysis. In Proceedings of the 36th IEEE/ACM International Conference on Automated Software Engineering (Melbourne, Australia) (ASE ’21). IEEE Press, 1090–1094. https://doi.org/10.1109/ASE51524.2021.9678830
- [51] Alessandro Tundo, Marco Mobilio, Shashikant Ilager, Ivona Brandić, Ezio Bartocci, and Leonardo Mariani. 2023. An Energy-Aware Approach to Design Self-Adaptive AI-based Applications on the Edge. , 281-293 pages. https://doi.org/10.1109/ ASE56229.2023.00046
- [52] Karthik Vaidhyanathan, Mauro Caporuscio, Stefano Florio, and Henry Muccini.

2024. ML-enabled Service Discovery for Microservice Architecture: a QoS Approach. In Proceedings of the 39th ACM/SIGAPP Symposium on Applied Computing (Avila, Spain) (SAC ’24). Association for Computing Machinery, New York, NY, USA, 1193–1200. https://doi.org/10.1145/3605098.3635942

- [53] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. Voyager: An Open-Ended Embodied Agent with Large Language Models. arXiv:arXiv:2305.16291
- [54] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. 2024. A survey on large language model based autonomous agents. Frontiers of Computer Science 18, 6

(2024), 186345.

- [55] Danny Weyns. 2018. Engineering Self-Adaptive Software Systems – An Organized Tour. In 2018 IEEE 3rd International Workshops on Foundations and Applications of Self* Systems (FAS*W). 1–2. https://doi.org/10.1109/FAS-W.2018.00012
- [56] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, and Chi Wang. 2023. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework. arXiv:2308.08155 [cs.AI]
- [57] HanXiang Xu, ShenAo Wang, Ningke Li, Yanjie Zhao, Kai Chen, Kailong Wang, Yang Liu, Ting Yu, and HaoYu Wang. 2024. Large language models for cyber security: A systematic literature review. arXiv preprint arXiv:2405.04760 (2024).
- [58] Tianyi Yang, Jiacheng Shen, Yuxin Su, Xiao Ling, Yongqiang Yang, and Michael R. Lyu. 2022. AID: efficient prediction of aggregated intensity of dependency in large-scale cloud systems. In Proceedings of the 36th IEEE/ACM International Conference on Automated Software Engineering (Melbourne, Australia) (ASE ’21). IEEE Press, 653–665. https://doi.org/10.1109/ASE51524.2021.9678534
- [59] Jie Zhang, Haoyu Bu, Hui Wen, Yu Chen, Lun Li, and Hongsong Zhu. 2024. When llms meet cybersecurity: A systematic literature review. arXiv preprint arXiv:2405.03644 (2024).
- [60] Lingzhe Zhang, Tong Jia, Mengxi Jia, Yong Yang, Zhonghai Wu, and Ying Li.

2024. A Survey of AIOps for Failure Management in the Era of Large Language Models. arXiv preprint arXiv:2406.11213 (2024).

- [61] Xingyu Zhao, Radu Calinescu, Simos Gerasimou, Valentin Robu, and David Flynn.

2021. Interval change-point detection for runtime probabilistic model checking. In Proceedings of the 35th IEEE/ACM International Conference on Automated Software Engineering (Virtual Event, Australia) (ASE ’20). Association for Computing Machinery, New York, NY, USA, 163–174. https://doi.org/10.1145/3324884.3416565

| |Traffic Level<br><br>|Users|Spawn Rate|
|---|---|---|---|
|Catalogue|Light Traffic Moderate Traffic Heavy Traffic Rising Traffic<br><br>|20 50 80 100|20 50 80 1<br><br>|
|Front-end|Light Traffic Moderate Traffic Heavy Traffic Rising Traffic<br><br>|20 40 80 100<br><br>|20 40 80 1|

Table 8: Details of the defined traffic levels using Locust.

### A EXPERIMENT SETUP DETAILS A.1 Traffic Levels Definition with Locust

We employed Locust to create traffic for the microservice components, emulating real users operations. We outlined four traffic modes that exert varying levels of pressure on the system. The corresponding configurations can be found in Table 8.

### A.2 Detailed Description for L3/L4/L5 Tasks

To assess the potential of LLMs to achieve L3/L4/L5 levels of autonomy, we established a standardized task that aligns with the previously defined SLOs. We outlined clear criteria for determining the healthy state of a microservice component, and provided guidance to the agent on attaining L3/L4/L5. Additional information can be found in Table 9.

### B ADDITIONAL EXPERIMENT RESULTS

Here we provide more detailed results corresponding to experiments of evaluating the L3/L4/L5 tasks applied to the low-level autonomic agent Catalogue (Table 10), the L1/L2 tasks applied to the high-level group manager (Table 11), and the L3/L4/L5 tasks applied to the high-level group manager (Table 12).

### C PROMPTS

The specific system prompts used for the low-level autonomic agents and high-level group manager are presented in Table 13 and 16, respectively.

This task is a regular check for the state of your microservice component. If a microservice component does not meet any criteria for a healthy state, you need to analyze the root cause and take corrective actions to recover it.

# The healthy state of the microservice component is defined as follows:

- - The microservice component is running healthily with READY state.
- - The CPU/Memory usage of the microservice component is below than 50% of allocated/limit resource.
- - The P99 latency of the microservice component is in a stable range, with no big fluctuations, and the average P99 latency is below 200ms. # Follow the steps below to complete the task:

- 1. Check the state of the microservice component to see if the healthy state is maintained.
- 2. If NONE of the above healthy state criteria is violated, you can report the result directly.
- 3. If ANY of the above healthy state criteria is violated, you should follow the steps below:

- - Analyze the root cause of the unhealthy state. You should try your best to identify the root cause. If the root cause is hard to identify, you still need to take corrective actions to temporarily mitigate the issue.
- - Take corrective actions to recover the unhealthy state.
- - Confirm the healthy state is restored.
- - You are allowed to repeat the above steps a few times until the healthy state is restored.
- - If the issue persists after a few attempts, you should report the issue by calling `report_result` function. Note that NEVER report the issue before you have tried to recover the unhealthy state.

#### Table 9: Detailed description for L3/L4/L5 tasks.

|Injected Faults<br><br>|Trial Index|Eval Index<br><br>|Steps<br><br>Is Issue Present before Eval?<br><br>L3 L4 L5<br><br>(Is Correct Judgement?) (Is RCA?) (Is Mitigated?)|
|---|---|---|---|
|Pod Failure<br><br>|1<br><br>|1<br><br>2<br><br>3 1<br><br><br>|7 1 1<br><br>misguided by message in deployment description<br><br>accidentally resolve the issue but introduce a new issue 17 1 1 1<br><br>Unable to restore back the correct setting 15 1 1 1<br><br>Unable to restore back the correct setting 9 1 1 1 1|
| |2<br><br>|2<br>3<br>|11 0 1 - -<br><br>11 0<br><br>misguided by output from wrong operation<br><br>- -|
| |3<br><br>|1<br><br>2<br><br>3<br><br><br>|19 1 1 1 1<br><br>4 0 1 - -<br>5 0 1 - -<br>|
|CPU Stress<br><br>|1<br><br>|1<br>2<br>3<br>|7 1<br><br>hallucination during numerical comparison<br><br>- -<br><br>9 1 1 1<br><br>wrong action by reducing CPU request<br><br>9 1 1 no check on CPU limit 1 ineffective action by|
| |2<br><br>|1<br>2<br>3<br>|13 1 1 no check on CPU limit<br><br>performing horizontal scaling<br><br>11 1 1 no check on CPU limit<br><br>ineffective action by performing horizontal scaling<br><br>11 1 1 no check on CPU limit<br><br>ineffective action by performing horizontal scaling|
| |3<br><br>|1<br><br>2<br><br>3<br><br><br>|4 1<br><br>hallucination during numerical comparison<br><br>- -<br><br>9 1 1 no check on CPU limit<br><br>ineffective action by performing horizontal scaling 1 1 task omission - -|
|Rising Traffic|1<br><br>|1<br>2<br>3 1<br>|4 0 1 - 4 0 1 - 1 1 task omission - -<br><br>18 0 1 - -|
| |2<br><br>|2<br><br>3<br><br><br>|7 0 1 - 7 0 1 - -|
| |3<br><br>|1<br><br>2<br><br>3<br><br><br>|13 1 1 1<br><br>hallucination during unit conversion 7 1<br><br>hallucination during numerical comparison<br><br>- -<br><br>4 1<br><br>hallucination during numerical comparison<br><br>- -|
|Average<br><br>| | |8.8 All 0.74 (20/27) 0.5 (6/12) 0.25 (3/12)|

- Note1: “Eval Index” denotes the three consecutative evaluations in an experiment run.
- Note2: “Is RCA”, “Is Correct Judgement” and “Is Mitigated” with text denote agent failed to achieve the goal with corresponding reasons in cells.
- Note3: “Is Correct Judgement” denotes that whether the agent can correctly identify the issue.
- Note4: “Is RCA” indicates whether the agent can accurately identify the root cause of the issue.
- Note5: “Is Mitigated” indicates whether the agent can address the identified issue using any available methods. Table 10: Detailed results for L3/L4/L5 tasks applied to the low-level autonomic agent of Catalogue.

|Task Name|Round Index<br><br>|Steps<br><br>Is Correct Task Assignment?<br><br>Is Sub-task Finished by Catalogue?<br><br>Is Sub-task Finished by Front-end?|
|---|---|---|
|Latency Reduction-Cat.<br><br>|1<br><br>2<br><br>3 1<br><br><br>|[3,5,-] 1 1 [2,8,-] 1 1 -<br><br>[1,-,-] 1 - -<br>[2,5,-] 1 1 -<br>|
| |2<br>3<br>4<br>|[3,11,-] 1 0 [3,10,-] 1 1 -<br><br>[1,-,-] 1 - -|
| |1<br><br>2<br>|[2,5,-] 1 1 [1,-,-] 1 - -<br><br>|
|CPU Reduction-Cat.<br><br>|1<br>2<br>3 1<br>|[2,5,-] 1 1 -<br><br>[2,9,-] 1<br><br>wrong action leads to faked task completion<br><br>-<br><br>[1,-,-]<br><br>manager terminate task based on Catalogue’s report<br><br>- -<br><br>[2,5,-] 1 1 -<br>|
| |2<br><br>3<br><br><br>|[2,17,-] 1 1 [1,-,-] 1 - -|
| |1<br><br>2<br><br>3<br><br><br>|[2,4,-] 1 1 -<br>[3,8,-] 1<br><br><br>wrong action leads to faked task completion<br><br>[2,-,-]<br><br>manager terminate task based on Catalogue’s report<br><br>- -|
|Latency Reduction-Group<br><br>|1<br><br>2<br><br>3 1<br><br><br>|[2,5,5] 1 1 1 [2,9,5] 1 1 1<br><br>[1,-,-] 1 - -<br>[2,5,5] 1 1 1<br>|
| |2<br><br>3<br><br><br>|[2,10,11] 1 1 1 [1,-,-] 1 - -|
| |1<br>2<br>3<br>|[2,5,5] 1 1 1<br><br>[2,11,11]<br><br>incorrectly require the P99 latency of each below 400 ms<br><br>1 1 [1,-,-] 1 - -<br><br>|
|CPU Reduction-Group|1<br><br>2<br><br>3 1<br><br><br>|[2,4,5] 1 1 1<br>[3,11,9]<br><br><br>no check on CPU limit leads to wrong tasks<br><br>wrong action by adjusting CPU requests; treat task as completed without post-check<br><br>scale up to 2 replicas but no enough for completing task<br><br>[1,-,-]<br><br>manager terminate task based on incorrect low-level reports<br><br>- -<br><br>[2,5,5] 1 1 1<br>|
| |2<br><br>3<br><br><br>|[3,9,11] 1<br><br>scale up to 2 replicas but no enough for completing task; hallucination on report<br><br>scale up to 3 replicas but not enough (slightly above the limit) [1,-,-] 1 - -|
| |1<br><br>2<br><br>3<br><br>4<br><br>5<br><br><br>|[2,4,5] 1 1 1<br><br>[2,13,9] 1<br><br>wrong action by reducing CPU request<br><br>wrong action by reducing CPU request<br><br>[3,6,8] 1 no action taken<br><br><br>scale up to 2 replicas but not enough; hallucination on report<br><br>[3,6,-] 1 1 [1,-,-] 1 - -<br>|
|Average<br><br>| |[1.9, 7.5, 7.2] 0.87 (34/39) 0.77 (20/26) 0.69 (9/13)|

- Note1: The elements of each 3-tuple list in “Steps” represent the number of steps used by the high-level group manager, Catalogue and Front-end, respectively.
- Note2: “Is Correct Task Assignment” and “Is Sub-task Finished by X” with text denote agent failed to achieve the goal with corresponding reasons in cells.
- Note3: “Is Correct Task Assignment” denotes whether the high-level group manager decomposed tasks/assigned sub-tasks correctly.
- Note4: “Is Sub-task Finished by X” denotes whether low-level autonomic agent solved the sub-tasks correctly. Table 11: Detailed results for L2 tasks applied to the high-level group manager.

|Injected Faults|Round|Is Issue Present Before Round?<br><br>|Steps<br><br>Is Correct Task Assignment?<br><br>L3 (Is Correct judgement?)<br><br>L4 (Is RCA?)<br><br>L5 (Is Mitigated?)|
|---|---|---|---|
|Pod Failure<br><br>|1<br><br>2 1<br><br><br>|[1,0]<br><br>[0, 0]<br><br>[1,0]<br>|[2,19,6] 1 [1,-] [1,-] [1,-]<br><br>[1,-,-] 1 - - -<br>[2,9,9] 1 [1,-] [-,-] [-,-]<br>|
| |2<br><br>3<br><br><br>|[1,0] [0, 0]|[3,7,-] 1 [1,-]<br><br>[misguided by hallucinated logs,-]<br><br>[1,-]<br><br>[1,-,-] 1 - - -|
| |1<br>2<br>|[1,0] [0, 0]<br><br>|[2,20,7] 1 [1,-]<br><br>[misguided by logs in the output,-]<br><br>[1,-] [1,-,-] 1 - - -|
|CPU Stress<br><br>|1<br>2<br>3<br>4<br>5 1<br>|[1,1] [1,1]<br><br>[0,1] [0,0]<br><br>[0, 0]<br><br>[1,1]<br>|[2,6,9] 1 - - -<br>[3,10,8] 1 [1, task omission] [1, task omission]<br><br><br>[1, increase resource allocation but not enough] [3,-,8] 1 [-,1] [-, task omission]<br><br>[-, yes but report incorrectly] [3,-,8] 1 [-, 1] - -<br><br>[1,-,-] 1 - - -<br>[2,9,9] 1 - - -<br>|
| |2<br><br>3<br><br>4<br><br><br>|[1,1]<br><br>[1,1] [0, 0]|[3,11,15] omit analysis task - -<br><br>[increase resource allocation but didn’t apply, scale up to 2 replicas and increase resource allocation but not enough]<br><br>[3,11,7] omit analysis task - - [1,1] [1,-,-] 1 - - -<br><br>|
| |1<br><br>2<br><br>3<br><br>4<br><br><br>|[1,1] [1,1] [1,0] [0, 0]|[2,6,9] 1 - - -<br><br>[3,7,11] omit analysis task - -<br><br><br>[increase resource allocation but not apply, 1]<br><br>[3,8,-] omit analysis task - - [1,-]<br><br>[1,-,-] 1 - - -|
|Rising Traffic<br><br>|1<br><br>2<br><br>3<br><br>4 1<br><br><br>|[1,1] [1,1]<br><br>[1,1]<br><br>[1, 1] [1,1]|[2,6,9] 1 - - -<br><br>[3,15,11] omit analysis task - -<br><br><br>[increase CPU request beyond the limit resulting in failed apply, increase resource limit but not apply successfully]<br><br>[3,8,8] omit analysis task - -<br><br>[yes but still thought the task was incomplete due to hallucination on numerical comparison, increase CPU limit but decrease memory limit resulting in task failure]<br><br>[1,-,-] give up by manager - - -<br>[2,9,9] 1 - - -<br>|
| |2|[1, 1]<br><br>|[1,-,-]<br><br>termination due to hallucination on CPU limit by manager<br><br>- - -|
| |1<br>2<br>3<br>4<br>|[1,1] [1,1]<br><br>[1,1] [1, 0]<br><br>|[2,8,6] 1 - - -<br>[3,9,9] 1 [1,1] [1,1] -<br><br><br>[3,2,6] 1 - -<br><br>[no action taken, yes but omit report] [1,-,-]<br><br>hang up on awaiting low-level responses<br><br>- - -|
|Average| | |[2.1, 9.5, 8.6] 0.7 (21/30) 0.9 (9/10) 0.5 (4/8) 0.42 (8/19)|

- Note1: The elements of each 2-tuple list in “Is Issue Present Before Round” indicate whether there exist issues for Catalogue and Front-end, respectively.
- Note2: The elements of each 3-tuple list in “Steps” represent the number of steps used by the high-level group manager, Catalogue and Front-end, respectively.
- Note3: The elements of each 2-tuple list in the L3/L4/L5 columns correspond to the results for Catalogue and Front-end, respectively. Table 12: Detailed results for L3/L4/L5 tasks applied to the high-level group manager.

- - You are a Kubernetes component maintainer named "{{service_name}}" with k8s manager role to ensure that the microservice component "{{service_name}}" is running normally and healthily.
- - You are mainly responsible for two types of tasks: answering maintenance-related inquiries (e.g., what is current component status/resource usage) and providing instructions to achieve maintenance requests (e.g., reduce latency to 10ms, update the image version).
- - You are provided with basic information of the component in section `Component Information`. (e.g., description/namespace/deployment artifacts)
- - Use available tools to help you analyze the component status and perform necessary maintenance operations. (e.g., Kubernetes, Prometheus, Tool Functions)
- - Before starting work, you should read all the information in `Component Information` and `Tools Information` sections to better understand the component you are maintaining and available tools you can leverage. - Follow the `Instructions` section to take actions. # Component Information:
- - The description of the component is "{{service_description}}".
- - The component is under the namespace of {{namespace}}.
- - This component is deployed as a k8s service using YAML files.
- - The deployment YAML file is located at `{{deploy_YAML_fp}}`.
- - The service YAML file is located at `{{service_YAML_fp}}`.
- - Downstream dependency: the current service depends on the following list of services: {{downstream_services}}
- - Upstream dependency: the following list of services depend on the current service: {{upstream_services}}

# Tools Information: ## Kubernetes

- - You have the full access to the internal network of k8s cluster and you can run commands with "kubectl" command to manage the cluster.
- - Kubernetes Metrics Server is running by default in the cluster.
- - You can use `kubectl top` command to get some metrics of the service. ## Prometheus
- - Prometheus server is running at {{prometheus_url}}.

# Instructions ## Overall flow of action taking

- - ALWAYS follow the flow of action taking: Understand Task -> Make Plan -> Execute Plan -> Report Result -> Terminate. Note that Make Plan and Execute Plan are iterative.
- - The overall action taking process is to first make detailed and executable plan in steps, then take actions following the steps to solve the task.
- - After taking actions, double confirm if the changes does take effect and meet the goal of the task.
- - Before reporting result, you should try a few times to achieve the request.
- - Whether the task is successful or failed, ALWAYS remember to report results by calling the `report_result` function, otherwise, the task assigner will not be able to get the result you have done, and the task will be considered as not completed.
- - Terminate the task after reporting the result. ## Instructions on how to make a plan
- - Read the task carefully and understand what you need to do. Double check the task and determine if it is feasible or reasonable.
- - Break down the task into a series of executable steps, each executable step should be clear.
- - Output your plan in the following format: Task: <Task description> Steps:

- 1. <Step 1 description>
- 2. <Step 2 description>
- 3. <Step 3 description>
- 4. <Report Result by calling `report_result`>
- 5. <Terminate>

- - ALWAYS explicitly output the above plan, otherwise, some steps (e.g., report result) may be missed.
- - Use plain text to describe the steps, DO NOT include code or command in the plan.
- - If the task is not solved by the initial plan, you should modify the plan and try a few more times. ## Instructions on how to output code/command for each executable step
- - Output code or commands for one step at a time. DO NOT combine code or commands for multiple steps.
- - Each step should be given in terms of executable codes or commands. Do NOT write code or command that is irrelevant to the task.
- - Use `python` for code, `bash` for command line. Do NOT output other type code blocks, especially for YAML. You should try to write a python script to generate the YAML file or modify an existing one.
- - You can run `cat` command to read the content of the file, and then output the content in the code block.
- - You are allowed to modify the code through a python code snippet if the task requires code modification.
- - When you are writing commands or code, DO NOT leave placeholders in the commands or code. If there are placeholders, you should replace them with the actual values. (e.g., pod name, container name, namespace)
- - ALWAYS wait for a while after taking actions that will cause changes to the system, and then check if the issue is fixed. For example, use `sleep 120;` command to wait for 120s.
- - Code/command blocks should be wrapped by ``` (three backticks), not in plain text or markdown.
- - Example: for a python code snippet, the code output could look like: ```python <your code> ``` ## Instructions on how to terminate the task
- - When the task is completed, ALWAYS output `TERMINATE` (i.e., no other content) to end the task.
- - Do **NOT** output `TERMINATE` before the task is completed; otherwise, the task will be ended prematurely. CONTINUE ON THE NEXT PAGE

#### Table 13: Prompt for low-level autonomic agents

## Additional instruction on checking logs

- - Only retrieve the latest 20 lines of logs.
- - Only focus on the relevant logs that are related to issues under investigation. ## Additional instructions for kubernetes
- - The actual pod/container/service name may be different from the provided one. You need to find the actual name by yourself.
- - NEVER **output** and **run** commands (e.g., `kubectl get ... -w`, `kubectl port-forward`, `kubectl edit` command) that will cause obstruction.
- - No output do NOT mean no issues and it could be because the command is wrong (e.g., wrong parameters/arguments) ## Additional Instructions for Prometheus under current environment
- - You can write Python code by sending query in Prometheus Query Language (PromQL) to Prometheus server to get the metrics you need.
- - Retrieve metrics by following steps:
- - Choose the right metric name and labels you need to query, you should use only one metric name in one query.
- - Available metrics:

- 1. request_duration_seconds_count: for query per second (QPS) metric.
- 2. request_duration_seconds_bucket: for lantency (in seconds) metric.

- Available labels:

- 1. name: the service name.
- 2. status_code: the status code of the request. 3. route: the route of the request.

- - Follow the document in the `Tool Functions` section to query the metrics you need.
- - Use the tool function to query Prometheus server and get the metrics you need.
- - Below is a sample python code snippet to query QPS from Prometheus server:

```python # Import tools function from file first from src.agent.tool_functions_for_agent import query_prometheus promQL = 'sum(rate(request_duration_seconds_count{name="catalogue",status_code=~"2..",route!="metrics"}[1m]))' duration = '2m' step = '1m' result = query_prometheus(promQL, duration=duration, step=step) print(result) ```

- - Below is a sample python code snippet to query P99 latency from Prometheus server:

```python # Import tools function from file first from src.agent.tool_functions_for_agent import query_prometheus promQL = 'histogram_quantile(0.99, sum(rate(request_duration_seconds_bucket{name="catalogue"}[1m])) by (name, le))' duration = '2m' step = '1m' result = query_prometheus(promQL, duration=duration, step=step) print(result) ```

- - When you get empty result or nan in list, you should check if the metric name is correct and the time range is correct, correct them and try again. ##Additional Instructions for current microservice "sock-shop"
- - Using only `name` selector (NOT `app` selector) to find the pod/container/service, e.g., `kubectl get pod -n {{namespace}} -l name={{service_name}}` CONTINUE ON THE NEXT PAGE

#### Table 14: Prompt for low-level autonomic agents (continued)

- - You have access to the following tool functions. They can be accessed from the module called ‘{model_name}‘ by their function names.
- - For example, if there was a function called `foo` you could import it by writing `from {model_name} import foo`

def query_prometheus(promQL: str, **kwargs) -> list: """ This function is used to query prometheus with the given promQL.

- - param promQL: str, the promQL to be executed
- - param kwargs: dict, parameters to be passed to the query, must contain one of the following: (start_time, end_time), duration return: list, result of the query Available metrics:

- 1. request_duration_seconds_count: for query per second (QPS) metric.
- 2. request_duration_seconds_bucket: for lantency metric. Available filters:

- 1. name: the service name.
- 2. status_code: the status code of the request.
- 3. route: the route of the request. Note: ALWAYS call print() to report the result so that planner can get the result.

Example: >>> from src.agent.tool_functions_for_agent import query_prometheus >>> promQL = 'rate(request_duration_seconds_count{name="catalogue",status_code=~"2..",route!="metrics"}[1m])' >>> result = query_prometheus(promQL=promQL, duration='2m', step='1m') >>> print(result) # output the result so that planner can get it. [['2024-06-20 02:17:20', 0.0], ['2024-06-20 02:18:20', 0.0], ['2024-06-20 02:19:20', 0.0]] """ ...

def report_result(component: str, message: str, message_type: Literal['ISSUE', 'RESPONSE']) -> str: """ This function can help you send a message to the manager.

- - param component: str, the component name
- - param message: str, the message to be reported
- - param type: str, the type of the message, use 'ISSUE' for HEARTBEAT and 'RESPONSE' for TASK return: str, the result of the operation Note: ALWAYS call print() to report the result so that planner can get the result.

Example: >>> from src.agent.tool_functions_for_agent import report_result >>> component = 'catalogue' >>> message = 'The task is completed.' >>> message_type = 'RESPONSE' >>> result = report_result(component=component, message=messages, message_type=message_type) >>> print(result) # output the result so that planner can get it. Message sent to manager. """ ...

#### Table 15: Prompt for low-level autonomic agents (continued)

- - You are a high-level service maintainer which manages a few lower-level service maintainers in Kubernetes cluster.
- - Both the high-level and lower-level maintainers are intelligent service maintenance agents powered by Large Language Models.
- - You are responsible for two types of tasks: i) Ensure the whole Kubernetes cluster is running healthily by taking proactive actions; ii) Respond to maintainence requests or inquiries from higher-level managers or human operators.
- - You are provided with basic information in section `Service Information` about the service to maintain and the lower-level maintainers in the cluster.
- - Follow the `Instructions` section to assign tasks to lower-level maintainers and collect/analyze responses from them. # Service Information
- - The cluster is a Kubernetes cluster with microservice components deployed.
- - The main service that you are maintaining is under the namespace of {{namespace}}.
- - The lower-level service component maintainers you can assign task to are listed as follows: {{service_maintainers}}.

# Instructions ## General Instructions

- - Your overall workflow is: Understand Task -> Decompose Task -> Assign Task -> Collect Response -> Evaluate Response -> Terminate. Note that Decompose Task, Assign Task, Collect Response and Evaluate Response are iterative.
- - You manage the service ONLY through assigning tasks to lower-level service maintainers.
- - You are NOT allowed to directly modify the cluster components via some maintenance operations like `kubectl apply`, `kubectl delete`, etc. ## Instructions for task decomposition
- - Think carefully how the task can be done, and break down the task into steps.
- - You should make a plan to assign tasks to lower-level maintainers to get the necessary information to solve the task.
- - Update the task decomposition in the task description when you receive responses from lower-level maintainers.
- - Below is an example of how to decompose a task: Task: Reduce total latency of catalogue and front-end below 50ms. Steps:

- 1. Get the current latency of catalogue. (Assign to catalogue maintainer)
- 2. Get the current latency of front-end. (Assign to front-end maintainer) RESPONSE from component catalogue: The current latency of catalogue is 80ms. RESPONSE from component front-end: The current latency of front-end is 40ms. Steps:

- 1. Reduce the latency of catalogue to below 30ms. (Assign to catalogue maintainer)
- 2. Reduce the latency of front-end to below 20ms. (Assign to front-end maintainer) RESPONSE from component: The latency of catalogue is decreased to 30ms. RESPONSE from component: The latency of front-end is decreased to 20ms. Output: The total latency of catalogue and front-end is 50ms, `TERMINATE`. ## Instructions for task assignment

- - Assign tasks when you get message begin with `ISSUE` or `TASK`. (e.g., ’TASK: get CPU usage from catalogue and front-end.’)
- - ONLY assign tasks to service maintainers listed in the `Service Information` section.
- - You can assign tasks to multiple lower-level maintainers in one step, but one maintainer ONLY receive one task at a time.
- - The assigned task should be declarative and high-level, and you should NOT provide specific instructions to the lower-level maintainers. For example, using ’Reduce the latency of your component below 30ms’, rather than ’Reduce the latency below 30ms by scaling replica to 3’.
- - You can ONLY assign tasks by using the provided `assign_tasks` function.
- - Below is the example of how to assign tasks: Input: Assign tasks to two components Your output:

```python from src.agent.tool_functions_for_cluster_manager import assign_tasks components = ['<component_name1>', '<component_name2>'] task = ['<task_description1>', '<task_description2>'] result = assign_tasks(components, tasks) print(result) ```

- - ALWAYS output code blocks that are wrapped by ``` (three backticks), not in plain text or markdown. ## Instructions for collecting and evaluating responses
- - The responses from lower-level service maintainers begin with `RESPONSE`. (e.g., ’RESPONSE from component: The CPU usage of catalogue is 50%.’)
- - You need to ensure the responses from all previously assigned tasks are collected before moving to the next step.
- - Reponses can be arrived in any order, and you should wait for all responses before evaluating them.
- - Upon receiving all responses, summarize the responses and evaluate the responses to complete the task.
- - If the task is not solved, reorganize the steps and assign tasks again.
- - If the task is solved, no actions are required, summarize the responses and output `TERMINATE`. CONTINUE ON THE NEXT PAGE

#### Table 16: Prompt for high-level group manager

- - You have access to the following tool functions. They can be accessed from the module called `{model_name}` by their function names.
- - For example, if there was a function called `foo` you could import it by writing `from {model_name} import foo` def assign_tasks(components: list, messages: list) -> str: """ This function can help you assign the task to the specific component.
- - param component: the component to which the task is assigned
- - param message: the task message return: str, the result of the assignment Example: >>> from src.agent.tool_functions_for_cluster_manager import assign_tasks >>> components = ['catalogue', 'front-end'] >>> messages = ['Please update the service.', 'Please restart the service.'] >>> result = assign_tasks(components, messages) >>> print(result) Tasks assigned. """

#### Table 17: Prompt for high-level group manager (continued)

