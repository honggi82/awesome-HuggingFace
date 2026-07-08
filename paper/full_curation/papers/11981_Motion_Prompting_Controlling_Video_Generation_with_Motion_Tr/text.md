## Motion Prompting: Controlling Video Generation with Motion Trajectories

# arXiv:2412.02700v2[cs.CV]27Mar2025

Daniel Geng1,2,∗ Charles Herrmann1,† Junhwa Hur1 Forrester Cole1 Serena Zhang1 Tobias Pfaff1 Tatiana Lopez-Guevara1 Carl Doersch1 Yusuf Aytar1 Michael Rubinstein1 Chen Sun1,3 Oliver Wang1 Andrew Owens2 Deqing Sun1

1Google DeepMind 2University of Michigan 3Brown University https://motion-prompting.github.io/

### Abstract

Motion control is crucial for generating expressive and compelling video content; however, most existing video generation models rely mainly on text prompts for control, which struggle to capture the nuances of dynamic actions and temporal compositions. To this end, we train a video generation model conditioned on spatio-temporally sparse or dense motion trajectories. In contrast to prior motion conditioning work, this flexible representation can encode any number of trajectories, object-specific or global scene motion, and temporally sparse motion; due to its flexibility we refer to this conditioning as motion prompts. While users may directly specify sparse trajectories, we also show how to translate high-level user requests into detailed, semi-dense motion prompts, a process we term motion prompt expansion. We demonstrate the versatility of our approach through various applications, including camera and object motion control, “interacting” with an image, motion transfer, and image editing. Our results showcase emergent behaviors, such as realistic physics, suggesting the potential of motion prompts for probing video models and interacting with future generative world models. Finally, we evaluate quantitatively, conduct a human study, and demonstrate strong performance.

### 1. Introduction

In video generation, motion is paramount. It can elevate a video from the uncanny valley to realistic or from amateur to professional. Motion guides attention, enhances storytelling, and defines visual style. Skilled filmmakers, like Kubrick and Kurosawa, masterfully use motion to create captivating, immersive experiences. Achieving realistic and expressive motion, coupled with granular control, is essential for generating compelling video. While text remains the

∗Work done as intern, †Project lead

main control signal for generation, its limitations become apparent when focusing on motion. Although effective for describing static scenes in images or high level actions, text struggles to convey the subtleties of motion: e.g., a prompt like “a bear quickly turns its head” could be interpreted in countless ways. How quick is “quickly”? What is the exact trajectory? Should it accelerate? Even detailed descriptions fail to capture nuances like ease-in-ease-out timing or synchronized movements. These nuances are often better conveyed through the motion itself.

Motivated by this, we explore motion as a powerful, complementary control scheme to text. Our first observation is that in order to fully harness the expressiveness of motion, we require a representation that can encode any type of motion. To this end, we identify spatio-temporally sparse or dense motion trajectories [22, 58] as an ideal candidate. Motion trajectories, a.k.a. particle video or point tracks, track the movement and visibility of a set of points throughout a video, offering a highly expressive encoding of motion. This representation can capture the trajectories of any number of points, represent object-specific or global scene motion, and even handle temporally sparse motion constraints. Furthermore, recent advances in point track estimation have yielded robust and efficient algorithms [12, 13, 36, 37] that are capable of processing diverse real-world videos to generate constraints for training. Given the comprehensive and flexible nature of this motion representation, akin to text, we designate our motion conditioning as motion prompts. We then train a motion trajectory ControlNet [87] on top of a pre-trained video diffusion model [3] to accept the motion prompt conditioning.

While these motion prompts can define any type of video motion, what is less clear is how a user would generate them in practice. Sparse trajectories, which give the rough direction of a few pixels or patches, may be easy to specify with mouse drags, but do not sufficiently constrain the generation process and fall short with respect to fine-grained control. Conversely, dense trajectories, though offering precise con-

| | | |
|---|---|---|
|[Figure 5]|[Figure 6]|[Figure 7]|
|[Figure 8]|[Figure 9]|[Figure 10]|

|[Figure 11]|
|---|

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

| | | |
|---|---|---|
|[Figure 17]|[Figure 18]|[Figure 19]|
|[Figure 20]|[Figure 21]|[Figure 22]|

| | | |
|---|---|---|
|[Figure 23]|[Figure 24]|[Figure 25]|
|[Figure 26]|[Figure 27]|[Figure 28]|

| | | |
|---|---|---|
|[Figure 29]|[Figure 30]|[Figure 31]|
|[Figure 32]|[Figure 33]|[Figure 34]|

| | | |
|---|---|---|
|[Figure 35]|[Figure 36]|[Figure 37]|
|[Figure 38]|[Figure 39]|[Figure 40]|

| | | |
|---|---|---|
|[Figure 41]|[Figure 42]|[Figure 43]|
|[Figure 44]|[Figure 45]|[Figure 46]|

| | | |
|---|---|---|
|[Figure 47]|[Figure 48]|[Figure 49]|
|[Figure 50]|[Figure 51]|[Figure 52]|

- Figure 1. Motion Prompting. 1) We train a general-purpose track-conditioned ControlNet adapter on top of a video diffusion model. 2) To use this model, we design motion prompts from user inputs, and show a variety of capabilities from this single trained model, such as object control, camera control, simultaneous object and camera control, motion transfer, and model probing. We visualize the motion prompt tracks and corresponding frames from the generated videos underneath. The tracks are colored only for the purpose of visualization, with trails denoting the direction and magnitude of motion. Additionally, some of our motion prompts are derived from user mouse motions, for which we visualize the mouse locations. We highly encourage the reader to view video results on our webpage .

trol, are impractical to design manually. To address this, our second observation is that we can often translate high level user requests (e.g., “move the camera around the xz plane”, “rotate the head of the cat”) into detailed motion trajectories through computer vision signals. We denote this process as motion prompt expansion due to its similarities to prompt expansion [11] or rewriting [4] for text in image generation. This method is intended to bridge the gap between user goals and our motion representation.

We identify several instances where motion prompt expansion can be an effective tool including (Fig. 1): converting user mouse drags into semi-dense motion trajectories

allowing users to “interact” with an image by manipulating hair or sand (Sec. 4.1); simultaneously specifying camera and object motion (Sec. 4.4); performing motion transfer where motion from a given video is applied to a different first frame (Sec. 4.5); and performing drag-based image editing (Sec. 4.1). While these results are not yet real time or causal, they strongly hint at how users may interact with generative world models, and allows us to probe the video prior of the generator to understand the aspects of physics and general world knowledge it has learned.

Finally we present quantitative results and human studies against baselines, indicating that our model performs well.

We also present ablations to validate our design choices and give insight. In summary, our contributions are:

- • We focus on motion as a conditioning signal and identify spatio-temporally sparse or dense motion tracks as a flexible motion representation that can accomplish many aspects of motion control. We train a ControlNet to accept these motion prompts as conditioning.
- • We propose motion prompt expansion, a process which takes simple user input and produces more complex motion tracks, which allow for more fine-grained control.
- • We then apply our approach to a wide range of tasks, such as object control, camera control, motion transfer, or drag-based image editing.
- • We also show emergent behavior, such as physics, which suggests that these motion prompts may be used to probe video models or interact with future world models.
- • We evaluate our method against baselines with quantitative metrics and a human study, showing that our model performs well compared to baselines.

### 2. Related Work

Video Diffusion Models. Diffusion models [25, 65, 66] have demonstrated amazing capabilities for video generation, conditioned on natural language [3, 19, 20, 26, 27] or by “animating” static images into videos [6, 64, 81]. Beyond content creation [52], they can be seen as a path to the ambitious goal of creating world simulators [7], showing preliminary success in visual planning for embodied agents [15, 16, 83]. Meanwhile, whether the video prior captures sufficient understanding of the physical world is still under debate [35], and explicit integration of physics rules appears to be necessary [41, 85, 86]. Our motion prompting technique, applicable to any video diffusion model, not only offers a more flexible and accurate interface to specify motion patterns for video generation, but also serves as a framework to probe a trained generative model for their 3D or physics understanding.

Motion-conditioned Video Generation. A pre-trained text-to-video model can be adapted to follow new motion patterns or additional motion conditioning signals. Lowrank adaptation (LoRA) [29], a generic technique for parameter fine-tuning, can be utilized for few-shot motion customization [55, 90]. DreamBooth [57], originally for personalized image generation, can also be applied to video generation [78] with motion control.

Early work proposes video control through sparse motions [2, 21]. More recent work explores similar ideas with more powerful models. The approaches vary in their design choices but often require certain complicated engineering techniques for stable training and better convergence. Tora [89], MotionCtrl [75], DragNUWA [84], Image Conductor [39], and MCDiff [9] adopt two-stage (e.g., finetun-

ing with first dense and then sparse trajectories, or training adapters sequentially), specialized losses [39, 45], architectures [17, 80], or multi-stage fine-tuning for multiple modules [9, 61, 73]. MOFA-Video [46] requires separate adapters for different motion types, TrackGo [92] uses custom losses and layers, while other works [39, 46, 75, 84, 89] engineer data filtering pipelines. In contrast, we find that a simpler training recipe yields high quality results. Our model is trained in a single stage, with uniformly sampled dense trajectories, and without any specialized engineering efforts. Yet it handles a wide range of tasks and motions, generalizing to both sparse and dense trajectories during inference.

Other approaches use entity-centric control signals such as bounding boxes [72, 78], segmentation masks [10, 79], human pose [30, 82], or camera pose [23, 76]. Zero-shot motion adaptation approaches (e.g., SG-I2V [45], Trailblazer [43], FreeTraj [54], and Peekaboo [32]) adopt a similar strategy, guiding the video generation based on the motion of entity-centric masks and thus avoiding training or fine-tuning video models. Our motion prompts offer a more flexible interface to control motion generation at various granularity. Unlike the test-time approaches which explicitly control the diffusion feature maps, our framework naturally balances the strength of controlling signals and that of the encoded video priors.

Motion Representations. As our goal is to condition a video generation model on motion of any type, it is crucial to choose a suitable motion representation. The most common representation is optical flow [8, 14, 28, 42, 67, 68]. While flow can be chained over time, errors can accumulate. The lack of occlusion handling also makes it unsuitable for our needs, which we find necessary for good camera control (Sec. 4.3). In contrast, long-range feature matching [5, 31, 33, 63] or point trajectories [12, 13, 22, 36, 37, 91] is a well-suited representation for our application. It can handle occlusions and allows for both sparse and dense tracking over any arbitrary temporal durations.

### 3. Method

Our video generation method takes as input a single frame, a text prompt, and a motion prompt in the form of point tracks—which we explain how to create in Sec. 4. Full implementation details can be found in Appendix A.

#### 3.1. Motion Prompts

To fully harness the expressiveness of motion, we need to be able to represent any type of motion. To this end, we use point trajectories for our motion prompts, which can encode both spatially (and temporally) sparse and dense motions, motion on a single object or of an entire scene, and even occlusions via a visibility flag. Using this representation

|[Figure 53]|
|---|

|[Figure 54]|
|---|

t

t

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

T

|[Figure 58]|
|---|

###### T

H

H

W

W

- Figure 2. Conditioning Tracks. During training, we take estimated tracks from a video (left) and encode them into a T ×H× W×C dimensional space-time volume (middle). Each track has a unique embedding (right), written to every location the track visits and is visible at. All other locations are set to zeros. This strategy can encode any number and configuration of tracks.

enables a broad range of capabilities such as object control (Sec. 4.1), camera control (Sec. 4.3), both simultaneously (Sec. 4.4), motion transfer (Sec. 4.5), and drag-based image editing (Sec. 4.1) under a unified model.

Formally, we denote a set of N point trajectories of length T by p ∈ RN×T×2, where the 2D coordinate of the nth track at the tth timestep is p[n,t] = (xnt ,ytn). In addition, we denote the visibility of the tracks as v ∈ RN×T, an array of 1’s and 0’s where 0 indicates an off-screen or occluded track, and 1 indicates a visible track.

- 3.2. Architecture

We build our model on top of Lumiere, a pre-trained video diffusion model [3] which has been trained to generate 5 seconds of video at 16 fps given text and first frame conditioning. In order to train in track conditioning, we use a ControlNet [87] which requires encoding tracks in a spatialtemporal volume, c ∈ RT×H×W×C, where T is the number of frames, H and W are the height and width of the generated video, and C is the channel dimension. To do this, we associate with each track, p[n,:], a unique and random embedding vector ϕn ∈ RC. Then, for each space-time location a track visits, and is visible at, we simply place the embedding ϕn ∈ RC in that location. All other values in the conditioning signal are set to 0. Fig. 2 illustrates this process. In other words, we zero-initialize c and set

##### c[t,xnt ,ytn] = v[n,t]ϕn (1)

for all tracks at each timestep t, where multiplying by the visibility v[n,t] zeros out the embedding if the track is not visible at that location and time. We quantize xnt and ytn to the nearest integer for simplicity. When multiple tracks pass through the same space-time location, we add the embeddings together. The track embeddings ϕn are randomly drawn from a fixed pool, and act simply as a unique identifier for each track. For completely dense tracks, this representation is equivalent to starting with a dense grid of embeddings and forward warping, similar to [59].

#### 3.3. Data

To train our model, we prepare a video dataset paired with tracks. We run BootsTAP [13], an off-the-shelf point tracking method, on an internal dataset consisting of 2.2M videos resized to 128×128, the output size of our base model. We extract tracks densely, resulting in 16,384 tracks per video as well as predicted occlusions, which we can sample from during training. We do not filter the videos in any way, with the hypothesis being that training on diverse motions will result in a more powerful and flexible model.

#### 3.4. Training

Training follows ControlNet [87], where the conditioning signal is given to a trainable copy of the base model’s encoder and the standard diffusion loss is optimized. For every video, we sample a random number of tracks from a uniform distribution and construct the conditioning signal as explained above. More details can be found in Appendix A.

We observe various phenomena during training. For one, we find that the loss is not correlated with the performance of the model at following tracks. Also, similar to [87], we observe a “sudden convergence phenomena” in which the model goes from completely ignoring the conditioning signal to fully trained in a short number of training steps. More details can be found in Appendix B.

Finally, we observe that our model exhibits fairly strong generalization in multiple directions. For example, while our model is trained on randomly sampled tracks, resulting in spatially uniformly distributed tracks during training, the model can generalize to spatially localized track conditioning (Figs. 3 and 6). In addition, while our model is trained for specific numbers of tracks, it generalizes surprisingly well to more (Fig. 5) or fewer number of tracks (Figs. 3, 4 and 6). Finally, we find that our model generalizes to tracks that don’t necessarily start from the first frame, despite only being trained on these tracks (Fig. 3b). We hypothesize this generalization is due to a combination of inductive biases from the convolutions in the network and the fact that we train the model on a large variety of trajectories.

### 4. Motion Prompts

In this section, we discuss different types of effects achievable through motion prompts and prompt expansion. In particular, we identify and demonstrate several different types of expansion, as shown in Fig. 1. Text prompts and other parameters for each video may be found in Tab. A1. We strongly encourage readers to view generated videos on our webpage.

#### 4.1. “Interacting” with an Image

Our model enables the ability to “interact” with images. To do this, we build a GUI that displays a still image and

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

|[Figure 65]|
|---|

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|

|[Figure 69]|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

|[Figure 75]|
|---|

|[Figure 76]|
|---|

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

- Figure 3. “Interacting” with an Image. We translate a simple user input, mouse motions and drags, and expand it into a more complex motion prompt which helps to achieve the user’s intention. The mouse trajectories are visualized as a hand when dragging, and as a black cursor otherwise. A grid of tracks centered on the cursor are created when the mouse is dragged, as shown in the top row. Frames from the generated video are shown in the bottom row. Prompting our model in this way, we can (a) move the head of a parrot or (c) a cow (b) play with hair or (d) “interact” with an image of sand. We can also keep the background still by specifying static tracks, as in (b) or (d). Note these samples are not generated in real-time and are not temporally causal. More examples can be found on our webpage.

|[Figure 99]|
|---|

|[Figure 100]|
|---|

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

- Figure 4. Drag-Based Image Editing. We show the input images in the first row, and resulting drag-based edits in the bottom row, with the drag visualized in both rows. In addition, in the final example we show how we can keep areas of the images static.

and size of this grid can be chosen by the user, similar to the Gaussian blurring of tracks in [39, 75, 79, 84] to specify the spatial extent of the motion. However, note that in our approach this step is done only at inference time, and not at train time. Additionally, a user may choose to place a grid of static tracks down to keep the background still, as in Fig. 3b and Fig. 3d, or have tracks persist after a mouse drag as in Fig. 3d.

Emergent Phenomena. We find that these “interaction” motion prompts can result in straightforward motions, such as turning the head of a parrot in Fig. 3a. But interestingly, we also observe more complex dynamics: e.g., in Fig. 3b, where the tracks toss the hair of the subject, or in Fig. 3d where the sand is swept around. In these examples, we are essentially probing the video prior learned by the model, and by doing so are able to visualize the physics and general world understanding that the model has learned. Furthermore, because our model supports temporally sparse track conditioning, we can effectively do prediction. That is, we can query our model with a motion for a short duration, and then let the model predict the future, allowing us to answer questions such as ”how will the hair behave if I pull on it this way or that?” as in Fig. 3b.

records mouse drags from a user. This recording is then converted to tracks, as described below, and is fed to the model along with the initial frame and text. More information about the GUI can be found in Appendix A. For simple mouse motions, where the mouse is constantly dragged, this approach is similar to prior work on sparse trajectory conditioned video generation [39, 45, 46, 61, 73, 75, 79, 84, 89, 92]. However, because our model generalizes to partial tracks, we can also handle multiple mouse drags in different locations at different times, resulting in natural user control as in Fig. 3b and Fig. 3d. Please note that while we record mouse inputs in real-time, our method requires sampling from the video diffusion model, which is not real-time – it takes about 12 minutes to generate an output video.

Drag-Based Image Editing. A natural application of this “interaction” ability is drag-based image editing [1, 18, 44, 48, 56, 62]. This task involves taking user supplied “drags” and editing an image such that objects follow these drags. We shows qualitative results in Fig. 4.

To create the motion prompts, we translate mouse drags into a grid of point tracks as shown in Fig. 3. The density

|[Figure 109]|
|---|

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

[Figure 115]

|[Figure 116]|
|---|

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

|[Figure 124]|
|---|

|[Figure 125]|
|---|

|[Figure 126]|
|---|

|[Figure 127]|
|---|

|[Figure 128]|
|---|

|[Figure 129]|
|---|

|[Figure 130]|
|---|

|[Figure 131]|
|---|

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]|
|---|

|[Figure 136]|
|---|

|[Figure 137]|
|---|

|[Figure 138]|
|---|

|[Figure 139]|
|---|

|[Figure 140]|
|---|

- Figure 5. Camera Control with Depth. We can construct motion prompts for camera control by specifying a camera trajectory and computing a point cloud with an off-the-shelf monocular depth estimator. We then project the points onto the sequence of cameras, which results in the shown point trajectories. We can also convert user mouse input into camera trajectories, as in example (c).

|[Figure 141]|
|---|

|[Figure 142]|
|---|

|[Figure 143]|
|---|

|[Figure 144]|
|---|

|[Figure 145]|
|---|

|[Figure 146]|
|---|

|[Figure 147]|
|---|

|[Figure 148]|
|---|

|[Figure 149]|
|---|

|[Figure 150]|
|---|

|[Figure 151]|
|---|

|[Figure 152]|
|---|

|[Figure 153]|
|---|

|[Figure 154]|
|---|

|[Figure 155]|
|---|

|[Figure 156]|
|---|

|[Figure 157]|
|---|

|[Figure 158]|
|---|

|[Figure 159]|
|---|

|[Figure 160]|
|---|

- Figure 6. Object Control with Primitives. By defining geometric primitives (e.g., a sphere) manipulated by a user with a mouse, we can obtain tracks exerting more fine-grain control over objects (e.g., rotations), which cannot be specified with a single track.

mouse-drag created trajectory. For implementation details, please see Appendix A.

#### 4.3. Camera Control with Depth

We can also design motion prompts to achieve camera control with our model. We do this by first running an off-theshelf monocular depth estimator [51] on the input frame to obtain a point cloud of the scene. Then, given a trajectory of camera poses we can re-project the point cloud onto each camera, resulting in 2D tracks for input. We can further improve quality by running z-buffering to get occlusion flags.

Prompting our model with these motion prompts results in camera control, as shown in Fig. 5. We can orbit a camera in circles as in Fig. 5a or have it arc upwards as in Fig. 5b. In addition, we can combine this camera control with mouse recordings for even greater ease of use. To do so, we record mouse inputs as is done in Sec. 4.1. We then construct a camera trajectory such that a single point in the point cloud follows the mouse trajectory, and that the camera is constrained to a vertical plane, which we show in Fig. 5c. For implementation details, please see Appendix A.

#### 4.2. Object Control with Primitives

Note that our model is neither trained on nor conditioned on camera poses, as with prior work [39, 75, 76]. Furthermore, none of our training data includes pose annotations. Despite this, we find that our model can achieve compelling camera control. This shows that instead of training a video model on specific types of motion, we can train a model on general motions and tease out specific capabilities by using motion prompts.

We can also reinterpret mouse motions as manipulating a proxy geometric primitive, such as a sphere. By placing these tracks over an object that can be roughly approximated by the primitive, we can effect more fine-grained control over the object than with sparse mouse tracks alone. For example, in Fig. 6, we place a sphere over the head of a cat and the eye of a frog to precisely rotate these objects to different positions, and in Fig. 1 we animate a bear. In this setting, the user must supply both the mouse motion, and also the location and radius of the sphere to use. This allows for the user to specify more complex motions than translations, which would be hard to express with a single

#### 4.4. Composing Motions

By composing motion prompts together we can combine capabilities. For example, in Fig. 7, we add together tracks

|[Figure 161]|
|---|

|[Figure 162]|
|---|

|[Figure 163]|
|---|

|[Figure 164]|
|---|

|[Figure 165]|
|---|

|[Figure 166]|
|---|

|[Figure 167]|
|---|

|[Figure 168]|
|---|

|[Figure 169]|
|---|

|[Figure 170]|
|---|

|[Figure 171]|
|---|

|[Figure 172]|
|---|

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

|[Figure 176]|
|---|

|[Figure 177]|
|---|

|[Figure 178]|
|---|

|[Figure 179]|
|---|

|[Figure 180]|
|---|

- Figure 7. Compositions of Motion Prompts. By composing motion prompts together, we can attain simultaneous object and camera control. For example, here we move the dog and horse’s head while orbiting the camera from left to right.

for object control and camera control, resulting in simultaneous control of both. This is done by converting the object tracks to displacements, and adding these deltas to the camera control tracks. In two dimensions, this composition is an approximation and will fail for extreme camera motion, but we find it works well for small to moderate camera motion. Again note, that we do not specifically train for this capability in contrast to prior work [39, 75, 76].

#### 4.5. Motion Transfer

Many types of motions may be hard to design a motion prompt for. Given a video with a desired motion, we can perform motion transfer [18, 73], where we extract motion tracks from a source video and apply it to an image. For example, we can extract the motion of a person turning their head and use it to puppeteer a macaque, as in Fig. 8. Moreover, we find that our model is surprisingly robust, in that we can apply motions to fairly out-of-domain images. For example, in Fig. 8 we apply the motion of a monkey chewing to a bird’s eye view image of trees. The resulting videos exhibit an interesting effect in which pausing the video on any frame removes the percept of the source video [24]. The monkey can only be perceived when the video is playing, where a Gestalt common-fate effect occurs [34].

#### 4.6. Failures, Limitations, and Probing Models

We differentiate failures into two broad categories. The first are failures of our motion conditioning or our motion prompting. For example, in Fig. 9a we show an example in which the cow’s head is unnaturally stretched due to the horns being mistakingly “locked” to the background. The second category are failures due to the underlying video model. For example, in Fig. 9b, we drag the chess piece but a new one spontaneously forms, which is a less plausible

|[Figure 181]|
|---|

|[Figure 182]|
|---|

|[Figure 183]|
|---|

|[Figure 184]|
|---|

|[Figure 185]|
|---|

|[Figure 186]|
|---|

|[Figure 187]|
|---|

|[Figure 188]|
|---|

|[Figure 189]|
|---|

|[Figure 190]|
|---|

|[Figure 191]|
|---|

|[Figure 192]|
|---|

|[Figure 193]|
|---|

|[Figure 194]|
|---|

|[Figure 195]|
|---|

|[Figure 196]|
|---|

|[Figure 197]|
|---|

|[Figure 198]|
|---|

|[Figure 199]|
|---|

|[Figure 200]|
|---|

|[Figure 201]|
|---|

|[Figure 202]|
|---|

|[Figure 203]|
|---|

|[Figure 204]|
|---|

|[Figure 205]|
|---|

|[Figure 206]|
|---|

|[Figure 207]|
|---|

|[Figure 208]|
|---|

|[Figure 209]|
|---|

|[Figure 210]|
|---|

- Figure 8. Motion Transfer. By conditioning our model on extracted motion from a source video we can puppeteer a macaque, or even transfer the motion of a monkey chewing to a photo of trees. Best viewed as videos on our webpage.

|[Figure 211]|
|---|

|[Figure 212]|
|---|

|[Figure 213]|
|---|

|[Figure 214]|
|---|

|[Figure 215]|
|---|

|[Figure 216]|
|---|

|[Figure 217]|
|---|

|[Figure 218]|
|---|

|[Figure 219]|
|---|

|[Figure 220]|
|---|

|[Figure 221]|
|---|

|[Figure 222]|
|---|

|[Figure 223]|
|---|

|[Figure 224]|
|---|

|[Figure 225]|
|---|

|[Figure 226]|
|---|

|[Figure 227]|
|---|

|[Figure 228]|
|---|

|[Figure 229]|
|---|

|[Figure 230]|
|---|

- Figure 9. Probing by Failures. We can use motion prompts to probe limitations of the underlying model. For example, dragging the chess piece results in the creation of a new piece.

video given the constraint. These types of failures suggest that we might be able to use motion prompts as a way to probe video models and discover limitations in their learned representations.

### 5. Quantitative Results

In addition to the qualitative examples above, we describe a quantitative benchmark, and evaluate our method against

Table 1. Quantitative Evaluations. We evaluate the appearance (PSNR, SSIM, LPIPS, FVD) and motion (EPE) of generated videos on the validation set of the DAVIS dataset. Please note that each method is trained from a different base model.

# Tracks Method PSNR ↑ SSIM ↑ LPIPS ↓ FVD ↓ EPE ↓

Image Conductor 11.468 0.145 0.529 1919.8 19.224 DragAnything 14.589 0.241 0.420 1544.9 9.135

N = 1

###### Ours 15.431 0.266 0.368 1445.2 14.619

Image Conductor 12.184 0.175 0.502 1838.9 24.263 DragAnything 15.119 0.305 0.378 1282.8 9.800

N = 16

###### Ours 16.618 0.405 0.319 1322.0 8.319

Image Conductor 11.902 0.132 0.524 1966.3 30.734 DragAnything 15.055 0.289 0.381 1379.8 10.948

N = 512

- Ours 18.968 0.583 0.229 688.7 4.055

N = 2048

Image Conductor 11.609 0.120 0.538 1890.7 33.561 DragAnything 14.845 0.286 0.397 1468.4 12.485

- Ours 19.327 0.608 0.227 655.9 3.887

recent baselines. Furthermore, we conduct a human study, and describe ablations in this section.

#### 5.1. Track-Conditioned Generation Evaluation

To evaluate our track-text-and-first frame conditioned video generation, we use the validation split of the DAVIS video dataset [53]. We extract first frames and tracks from the dataset and feed this to the models along with an automatically generated text prompt. For exact implementation details, please see Appendix A. To evaluate a range of track densities, we vary the number of conditioning tracks from just a single track to 2048 tracks.

We compare our method with two recent works: ImageConductor [39], which finetunes AnimateDiff [20] for camera and object motion, and DragAnything [79], which is designed to move “entities” along tracks by finetuning Stable Video Diffusion [6]. To evaluate the appearance of the generated videos we compute PSNR, SSIM [74], LPIPS [88], and FVD [69] between the generated videos and ground truth videos. To evaluate how well the generated video matches the motion conditioning, we use end-point error (EPE) which is computed as the L2 distance between the conditioning tracks and tracks estimated from the generated videos.

As shown in Tab. 1, our model outperforms the baselines in almost all cases. On some examples, DragAnything performs better in terms of EPE with fewer tracks. This is because DragAnything includes a module that effectively warps latents. While this warping effect may result in accurate motion, it also creates visual artifacts as evidenced by the underperforming PSNR, SSIM, LPIPS, and FVD results. We also provide numbers for 4 and 64 tracks in Appendix C, which we omit here for brevity.

#### 5.2. Human Study

We run a human study where we manually create a set of 30 inputs consisting of a single trajectory. We run a two alternative forced choice test where we ask (1) which video

- Table 2. Human Study. We present % win rates of our method against baselines in 2AFC human study results. Sample sizes are N = 103, N = 103, and N = 115 for each column respectively.

Method Motion Adherence Motion Quality Visual Quality

Image Conductor 74.3 (±1.1) 80.5 (±1.0) 77.3 (±1.0) Drag Anything 74.5 (±1.1) 75.7 (±1.1) 73.7 (±1.0)

- Table 3. Ablation. We ablate the density of tracks during training and find that training on dense tracks works best for our model.

# Tracks Method PSNR ↑ SSIM ↑ LPIPS ↓ FVD ↓ EPE ↓

Sparse 15.075 0.241 0.384 1209.2 30.712 Dense + Sparse 15.162 0.252 0.379 1230.6 29.466 Dense 15.638 0.296 0.349 1254.9 24.553

N = 4

Sparse 15.697 0.284 0.355 1322.0 26.724 Dense + Sparse 15.294 0.246 0.375 1267.8 27.931 Dense 19.197 0.582 0.230 729.0 4.806

N = 2048

follows the motion conditioning better (2) which video has more realistic motion and (3) which video has higher visual quality. There are 180 questions total, and win rates for our method as well as 95% confidence intervals are presented in Tab. 2. When considering both motion and appearance together, our approach is preferred over baselines in all categories. Implementation details can be found in Appendix A.

#### 5.3. Ablations

In Tab. 3 we present an ablation where we train our model on only Sparse point trajectories (1-8 tracks) and Dense + Sparse, where the number of tracks is sampled logarithmically from 20 to 213. We find that dense training is most effective, especially for large number of tracks. Surprisingly, dense training is also better for sparse tracks. We hypothesize that this is because using sparse tracks gives so little training signal that it is more efficient to train on dense tracks, which then generalizes to sparser tracks, though this may be influenced by our usage of ControlNet and zero convolutions. We use a subset of the DAVIS evaluation from Sec. 5.1, but we note that the numbers do not match as we use less data and fewer training steps for the ablations.

### 6. Conclusion

We have introduced a framework for motion-conditioned video generation that leverages flexible motion prompts – spatio-temporal trajectories that can encode arbitrary motion complexity. Unlike prior work, this representation allows specifying sparse or dense motion for cameras, objects, or full scenes. We also introduce motion prompt expansion to translate high-level motion requests into detailed prompts. Our versatile approach enables applications like motion control, motion transfer, image editing, and showcasing emergent behaviors like realistic physics with a single unified model. Quantitative and human evaluation demonstrate the effectiveness of our framework.

Acknowledgements We would like to thank Sarah Rumbley, Roni Paiss, Jeong Joon Park, Liyue Shen, Stella Yu, Alyosha Efros, Boqing Gong, Daniel Watson, David Fleet, and Bill Freeman for their invaluable feedback and discussions.

### References

- [1] Hadi Alzayer, Zhihao Xia, Xuaner Zhang, Eli Shechtman, Jia-Bin Huang, and Michael Gharbi. Magic Fixup: Streamlining photo editing by watching dynamic videos. arXiv preprint arXiv:2403.13044, 2024. 5
- [2] Pierfrancesco Ardino, Marco De Nadai, Bruno Lepri, Elisa Ricci, and St´ephane Lathuili`ere. Click to move: Controlling video generation with sparse motion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14749–14758, 2021. 3
- [3] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Yuanzhen Li, Tomer Michaeli, et al. Lumiere: A spacetime diffusion model for video generation. arXiv preprint arXiv:2401.12945, 2024. 1, 3, 4
- [4] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Technical report, 2023. 2
- [5] Zhangxing Bian, Allan Jabri, Alexei A. Efros, and Andrew Owens. Learning pixel trajectories with multiscale contrastive random walks. In CVPR, 2022. 3
- [6] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3, 8
- [7] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. Technical report, 2024. 3
- [8] Thomas Brox and Jitendra Malik. Large displacement optical flow: descriptor matching in variational motion estimation. IEEE TPAMI, 33(3):500–513, 2010. 3
- [9] Tsai-Shien Chen, Chieh Hubert Lin, Hung-Yu Tseng, TsungYi Lin, and Ming-Hsuan Yang. Motion-conditioned diffusion model for controllable video synthesis. arXiv preprint arXiv:2304.14404, 2023. 3
- [10] Zuozhuo Dai, Zhenghao Zhang, Yao Yao, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Fine-grained open domain image animation with motion guidance, 2023. 3
- [11] Siddhartha Datta, Alexander Ku, Deepak Ramachandran, and Peter Anderson. Prompt expansion for adaptive text-toimage generation. arXiv preprint arXiv:2312.16720, 2023.

- 2

[12] Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman. TAPIR: Tracking any point with per-frame initialization and temporal refinement. In ICCV, pages 10061–10072, 2023. 1,

- 3, 4

- [13] Carl Doersch, Yi Yang, Dilara Gokay, Pauline Luc, Skanda Koppula, Ankush Gupta, Joseph Heyward, Ross Goroshin, Jo˜ao Carreira, and Andrew Zisserman. BootsTAP: Bootstrapped training for tracking-any-point. arXiv preprint arXiv:2402.00847, 2024. 1, 3, 4
- [14] Alexey Dosovitskiy, Philipp Fischer, Eddy Ilg, Philip Hausser, Caner Hazirbas, Vladimir Golkov, Patrick Van Der Smagt, Daniel Cremers, and Thomas Brox. FlowNet: Learning optical flow with convolutional networks. In ICCV, pages 2758–2766, 2015. 3
- [15] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. NeurIPS, 36, 2024. 3
- [16] Alejandro Escontrela, Ademi Adeniji, Wilson Yan, Ajay Jain, Xue Bin Peng, Ken Goldberg, Youngwoon Lee, Danijar Hafner, and Pieter Abbeel. Video prediction models as rewards for reinforcement learning. NeurIPS, 36, 2024. 3
- [17] Wanquan Feng, Tianhao Qi, Jiawei Liu, Mingzhen Sun, Pengqi Tu, Tianxiang Ma, Fei Dai, Songtao Zhao, Siyu Zhou, and Qian He. I2vcontrol: Disentangled and unified video motion synthesis control. arXiv preprint arXiv:2411.17765, 2024. 3
- [18] Daniel Geng and Andrew Owens. Motion guidance: Diffusion-based image editing with differentiable motion estimators. In ICLR, 2024. 5, 7
- [19] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709, 2023. 3
- [20] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. AnimateDiff: Animate your personalized text-toimage diffusion models without specific tuning. In ICLR,

2024. 3, 8

- [21] Zekun Hao, Xun Huang, and Serge Belongie. Controllable video generation with sparse trajectories. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 7854–7863, 2018. 3
- [22] Adam W. Harley, Zhaoyuan Fang, and Katerina Fragkiadaki. Particle video revisited: Tracking through occlusions using point trajectories. In ECCV, 2022. 1, 3
- [23] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. CameraCtrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 3
- [24] Herolias. My first try with video. https://www. reddit.com/r/StableDiffusion/comments/ 17b4dfc/my_first_try_with_video/, 2023. Accessed: 2024-11-13. 7
- [25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020. 3
- [26] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, et al. Imagen

- video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 3
- [27] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. NeurIPS, 35:8633–8646, 2022. 3
- [28] Berthold K.P. Horn and Brian G. Schunck. Determining optical flow. Artificial intelligence, 17(1-3):185–203, 1981. 3
- [29] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 3
- [30] Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. Animate Anyone: Consistent and controllable image-to-video synthesis for character animation. arXiv preprint arXiv:2311.17117, 2023. 3
- [31] Allan Jabri, Andrew Owens, and Alexei Efros. Space-time correspondence as a contrastive random walk. NeurIPS, 33: 19545–19560, 2020. 3
- [32] Yash Jain, Anshul Nasery, Vibhav Vineet, and Harkirat Behl. Peekaboo: Interactive video generation via maskeddiffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8079– 8088, 2024. 3
- [33] Wei Jiang, Eduard Trulls, Jan Hosang, Andrea Tagliasacchi, and Kwang Moo Yi. COTR: Correspondence transformer for matching across images. In ICCV, pages 6207–6217, 2021. 3
- [34] Gunnar Johansson. Visual perception of biological motion and a model for its analysis. Perception & psychophysics, 14:201–211, 1973. 7
- [35] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. arXiv preprint arXiv:2411.02385, 2024. 3
- [36] Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. CoTracker3: Simpler and better point tracking by pseudolabelling real videos. arXiv preprint arXiv:2410.11831,

2024. 1, 3

- [37] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. CoTracker: It is better to track together. In ECCV, 2024. 1, 3
- [38] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023. 3
- [39] Yaowei Li, Xintao Wang, Zhaoyang Zhang, Zhouxia Wang, Ziyang Yuan, Liangbin Xie, Yuexian Zou, and Ying Shan. Image conductor: Precision control for interactive video synthesis. arXiv preprint arXiv:2406.15339, 2024. 3, 5, 6, 7, 8
- [40] Ce Liu, Antonio Torralba, William T Freeman, Fr´edo Durand, and Edward H Adelson. Motion magnification. ACM transactions on graphics (TOG), 24(3):519–526, 2005. 4
- [41] Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. PhysGen: Rigid-body physics-grounded image-

to-video generation. In ECCV, pages 360–378. Springer,

2024. 3

- [42] Bruce D. Lucas and Takeo Kanade. An iterative image registration technique with an application to stereo vision. In IJCAI, pages 674–679, 1981. 3
- [43] Wan-Duo Kurt Ma, John P. Lewis, and W Bastiaan Kleijn. Trailblazer: Trajectory control for diffusion-based video generation. arXiv preprint arXiv:2401.00896, 2023. 3
- [44] Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. DragonDiffusion: Enabling dragstyle manipulation on diffusion models. arXiv preprint arXiv:2307.02421, 2023. 5
- [45] Koichi Namekata, Sherwin Bahmani, Ziyi Wu, Yash Kant, Igor Gilitschenski, and David B. Lindell. SG-I2V: Selfguided trajectory control in image-to-video generation. arXiv preprint arXiv:2411.04989, 2024. 3, 5
- [46] Muyao Niu, Xiaodong Cun, Xintao Wang, Yong Zhang, Ying Shan, and Yinqiang Zheng. MOFA-Video: Controllable image animation via generative motion field adaptions in frozen image-to-video diffusion model. In ECCV, 2024. 3, 5
- [47] Tae-Hyun Oh, Ronnachai Jaroensri, Changil Kim, Mohamed Elgharib, Fr’edo Durand, William T Freeman, and Wojciech Matusik. Learning-based video motion magnification. In Proceedings of the European Conference on Computer Vision (ECCV), pages 633–648, 2018. 4
- [48] Xingang Pan, Ayush Tewari, Thomas Leimk¨uhler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. Drag Your GAN: Interactive point-based manipulation on the generative image manifold. In SIGGRAPH, pages 1–11, 2023. 5
- [49] Zhaoying Pan, Daniel Geng, and Andrew Owens. Selfsupervised motion magnification by backpropagating through optical flow. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 4
- [50] Bohao Peng, Jian Wang, Yuechen Zhang, Wenbo Li, MingChang Yang, and Jiaya Jia. Controlnext: Powerful and efficient control for image and video generation. arXiv preprint arXiv:2408.06070, 2024. 4
- [51] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. UniDepth: Universal monocular metric depth estimation. In CVPR,

2024. 6

- [52] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

2024. 3

- [53] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 DAVIS challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 8
- [54] Haonan Qiu, Zhaoxi Chen, Zhouxia Wang, Yingqing He, Menghan Xia, and Ziwei Liu. FreeTraj: Tuning-free trajectory control in video diffusion models. arXiv preprint arXiv:2406.16863, 2024. 3
- [55] Yixuan Ren, Yang Zhou, Jimei Yang, Jing Shi, Difan Liu, Feng Liu, Mingi Kwon, and Abhinav Shrivastava.

- Customize-a-video: One-shot motion customization of textto-video diffusion models. In ECCV, 2024. 3
- [56] Noam Rotstein, Gal Yona, Daniel Silver, Roy Velich, David Bensa¨ıd, and Ron Kimmel. Pathways on the image manifold: Image editing via video generation. arXiv preprint arXiv:2411.16819, 2024. 5
- [57] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, pages 22500–22510, 2023. 3
- [58] Peter Sand and Seth Teller. Particle video: Long-range motion estimation using point trajectories. IJCV, 80:72–91,

2008. 1

- [59] Junyoung Seo, Kazumi Fukuda, Takashi Shibuya, Takuya Narihira, Naoki Murata, Shoukang Hu, Chieh-Hsin Lai, Seungryong Kim, and Yuki Mitsufuji. GenWarp: Single image to novel views with semantic-preserving generative warping. NeurIPS, 2024. 4
- [60] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. In International Conference on Machine Learning, pages 4596–4604. PMLR,

2018. 1

- [61] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. Motion-I2V: Consistent and controllable image-to-video generation with explicit motion modeling. In SIGGRAPH, pages 1–11, 2024. 3, 5
- [62] Yujun Shi, Chuhui Xue, Jiachun Pan, Wenqing Zhang, Vincent YF Tan, and Song Bai. Dragdiffusion: Harnessing diffusion models for interactive point-based image editing. arXiv preprint arXiv:2306.14435, 2023. 5
- [63] Ayush Shrivastava and Andrew Owens. Self-supervised anypoint tracking by contrastive random walks. ECCV, 2024. 3
- [64] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-A-Video: Text-to-video generation without text-video data. In ICLR, 2023. 3
- [65] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, pages 2256– 2265, 2015. 3
- [66] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 3
- [67] Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz. PWC-Net: CNNs for optical flow using pyramid, warping, and cost volume. In CVPR, pages 8934–8943, 2018. 3
- [68] Zachary Teed and Jia Deng. RAFT: Recurrent all-pairs field transforms for optical flow. In ECCV, pages 402–419, 2020. 3
- [69] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. FVD: A new metric for video generation, 2019. 8
- [70] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, pages 5998–6008, 2017. 1

- [71] Neal Wadhwa, Michael Rubinstein, Fr´edo Durand, and William T Freeman. Phase-based video motion processing. ACM Transactions on Graphics (ToG), 32(4):1–10, 2013. 4
- [72] Jiawei Wang, Yuchen Zhang, Jiaxin Zou, Yan Zeng, Guoqiang Wei, Liping Yuan, and Hang Li. Boximator: Generating rich and controllable motions for video synthesis, 2024. 3
- [73] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. VideoComposer: Compositional video synthesis with motion controllability. NeurIPS, 36, 2023. 3, 5, 7
- [74] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE TIP, 13(4):600–612, 2004. 8
- [75] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 3, 5, 6, 7
- [76] Daniel Watson, Saurabh Saxena, Lala Li, Andrea Tagliasacchi, and David J. Fleet. Controlling space and time with diffusion models. arXiv preprint arXiv:2407.07860, 2024. 3, 6, 7
- [77] Hao-Yu Wu, Michael Rubinstein, Eugene Shih, John Guttag, Fr´edo Durand, and William Freeman. Eulerian video magnification for revealing subtle changes in the world. ACM transactions on graphics (TOG), 31(4):1–8, 2012. 4
- [78] Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, and Kai Chen. Motionbooth: Motion-aware customized text-to-video generation. arXiv preprint arXiv:2406.17758, 2024. 3
- [79] Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang. DragAnything: Motion control for anything using entity representation. In ECCV, pages 331–348,

2024. 3, 5, 8

- [80] Zeqi Xiao, Wenqi Ouyang, Yifan Zhou, Shuai Yang, Lei Yang, Jianlou Si, and Xingang Pan. Trajectory attention for fine-grained video motion control. arXiv preprint arXiv:2411.19324, 2024. 3
- [81] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. DynamiCrafter: Animating open-domain images with video diffusion priors. In ECCV, pages 399–417. Springer, 2024. 3
- [82] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. MagicAnimate: Temporally consistent human image animation using diffusion model. In CVPR, 2024. 3
- [83] Sherry Yang, Jacob Walker, Jack Parker-Holder, Yilun Du, Jake Bruce, Andre Barreto, Pieter Abbeel, and Dale Schuurmans. Video as the new language for real-world decision making. arXiv preprint arXiv:2402.17139, 2024. 3
- [84] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. DragNUWA: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023. 3, 5

- [85] Alan Yu, Ge Yang, Ran Choi, Yajvan Ravan, John Leonard, and Phillip Isola. Lucidsim: Learning agile visual locomotion from generated images. In 8th Annual Conference on Robot Learning, 2024. 3
- [86] Ye Yuan, Jiaming Song, Umar Iqbal, Arash Vahdat, and Jan Kautz. PhysDiff: Physics-guided human motion diffusion model. In ICCV, pages 16010–16021, 2023. 3
- [87] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In IEEE International Conference on Computer Vision (ICCV),

2023. 1, 4

- [88] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 8
- [89] Zhenghao Zhang, Junchao Liao, Menghao Li, Long Qin, and Weizhi Wang. Tora: Trajectory-oriented diffusion transformer for video generation. arXiv preprint arXiv:2407.21705, 2024. 3, 5
- [90] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jia-Wei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. MotionDirector: Motion customization of text-to-video diffusion models. In ECCV, pages 273–290. Springer, 2025. 3
- [91] Yang Zheng, Adam W. Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J. Guibas. PointOdyssey: A largescale synthetic dataset for long-term point tracking. In ICCV,

2023. 3

- [92] Haitao Zhou, Chuang Wang, Rui Nie, Jinxiao Lin, Dongdong Yu, Qian Yu, and Changhu Wang. TrackGo: A flexible and efficient method for controllable video generation. arXiv preprint arXiv:2408.11475, 2024. 3, 5

## Motion Prompting: Controlling Video Generation with Motion Trajectories Supplementary Material

### A. Implementation Details

#### A.1. Architecture and Training

We train our model for 70,000 steps using Adafactor [60] with a learning rate of 1 × 10−4. We do not use any learning rate decay. For the ControlNet, we copy Lumiere’s encoder stack, add in zero convolutions as in [87], and replace the first convolutional layer with a new layer that accepts a T ×H×W ×C conditioning signal. From the constraints of the base architecture, we have T = 80 and H = W = 128. We set C = 64. During training, we sample the number of input tracks uniformly from 1000 to 2000 inclusive. For each track we randomly assign a sinusoidal positional encoding [70], of 64 dimensions, by sampling integers without replacement from 0 to 16384 – the maximum number of tracks for a 128 × 128 image, and using the corresponding positional embedding for that integer. Note that the encoding is completely randomly assigned. In particular, its spatial location has no bearing on its embedding.

All sampled videos are passed through Lumiere’s spatial super resolution (SSR) model, resulting in a 1024 × 1024, 80 frame video at 16 frame per second, for a total of 5 seconds. We use Lumiere’s SSR model as is, without finetuning it for motion conditioning, as we find that the 128×128 output of the base model already contains all of the motion conditioned dynamics.

Data. We train on an internal dataset of 2.2 million videos. We precompute trajectories on this dataset by center cropping each video to a square, resizing it to 256 × 256, and then running BootsTAP [13] with a dense gird of query points, resulting in 16,384 tracks per video. During training, a video is sampled, and then tracks are randomly sampled from this dataset. During the Lumiere fine-tuning, videos are resized to match Lumiere’s 128 × 128 input and output size.

#### A.2. Qualitative Results

We provide additional details for qualitative results in Tab. A1, including text prompt and licensing information. All images and videos are used with permission and under open and free licenses. In addition, as can be seen we construct text prompts to describe the scene but not the motion, in order to limit the influence of text conditioning on the motion as much as possible.

#### A.3. Mouse GUI

We record mouse motions through a simple HTML GUI, which is shown in Fig. A1. It consists of a canvas el-

|[Figure 231]|
|---|

Figure A1. Mouse Motion GUI. We show a screenshot of the GUI that we use to record mouse motions. For more information please see Appendix A.3.

ement which displays the first frame conditioning, labels that indicate the position of the mouse in the canvas, and whether or not it is currently being clicked, a button to start the recording, a countdown timer which gives three seconds before recording starts, and a second countdown timer which shows when the recording will end. We record 80 frames of mouse input to match the five seconds of video that our model outputs at 16 frames per second. For each frame we record the mouse (x,y) position, and a flag indicating whether the mouse is being clicked.

#### A.4. Interacting with and Drag Editing Images

In order to feed mouse motions to our model, we create a grid of tracks that is centered on the mouse whenever it is being dragged. The user may choose the stride of these tracks, and the size of the grid. We use a square grid of tracks for simplicity. In addition, a user may choose to have the tracks “persist,” in that before and after the mouse drag the tracks remain. This is useful in cases where objects should stay in place after a drag. A user may also place down a grid of tracks to “pin” the background in place. Note

- Table A1. Figure Details. We provide details about qualitative samples shown in our figures, including text prompts fed to the model and licensing information. In general, these are sorted by the order that they appear in the paper, moving from left to right, top to bottom.

Description Figure Text Prompt Source URL License License URL two elephants Fig. 1, Fig. 5 elephants Unsplash link Unsplash license owl Fig. 1 a close up of a great horned owl Unsplash link Unsplash license brown bear Fig. 1 a brown bear Unsplash link Unsplash license squirrel Fig. 1 a squirrel sitting on the ground in the woods Unsplash link Unsplash license golden retriever Fig. 1, Fig. 7 a golden retriever laying in the grass Unsplash link Unsplash license man (motion source) Fig. 1, Fig. 8 – private correspondence – permission granted – macaque Fig. 1, Fig. 8 a macaque monkey Unsplash link Unsplash license sand Fig. 1, Fig. 3 sand Unsplash link Unsplash license woman Fig. 1, Fig. 3 a woman private correspondence – permission granted – parrot Fig. 3 a close up of a green parrot Unsplash link Unsplash license

a highland cow standing in a grassy scottish wilderness

cow Fig. 3, Fig. 4, Fig. 9

Unsplash link Unsplash license

skull Fig. 4 a white skull on a black background Unsplash link Unsplash license stool Fig. 4 a living room Unsplash link Unsplash license

a serene scene of multiple hot air balloons floating over Cappadocia, Turkey, during sunset

hot air balloons Fig. 4

Unsplash link Unsplash license

arches in arches national park, with shrubbery in the foreground and a bright blue sky

arches Fig. 5

Unsplash link Unsplash license

roses Fig. 5 a red rose Unsplash link Unsplash license cat Fig. 6 a cat Unsplash link Unsplash license frog Fig. 6 a close up of a frog Unsplash link Unsplash license horse Fig. 7 a horse Unsplash link Unsplash license Earth (motion source) Fig. 8 – Pexels link Pexels license panda Fig. 8 a panda Unsplash link Unsplash license monkey (motion source) Fig. 8 – Pexels link Pexels license trees Fig. 8 birds eye view of trees Unsplash link Unsplash license

close-up of a chessboard with strong depth of field. The white king piece is in focus, surrounded by black pawns

chess Fig. 9

Unsplash link Unsplash license

that this setup is identical to how we obtain the “drag-based image editing” results.

#### A.5. Geometric Primitives

To make spherical tracks we take points on a sphere and follow them as the sphere is spun. This gives us a trajectory of 3D points, which when orthographically projected gives us 2D tracks. The density of the points, the radius of the sphere, and the location of the sphere are determined by the user. Mouse motions are converted to sphere spins by rotating the sphere through a single axis such that the initial mouse location matches with the current mouse location at each frame. This uniquely defines a rotation and ensures that the sphere tracks the mouse.

#### A.6. Camera Control

In order to obtain camera control, we run a monocular depth estimator on the first frame input to the model. This gives us camera intrinsics as well as depths, allowing us to unproject into a point cloud. We then project this point cloud onto a sequence of camera poses forming the desired camera trajectory, resulting in 2D point tracks. In addition, we run z-buffering to determine occlusions, where only the

closest point that has been projected to some neighborhood is visible while all other points in that neighborhood are occluded—unless that point is sufficiently close to the visible point. This requires choosing a radius for the neighborhood size, and a threshold for a point to remain visible if it is close enough to the visible point. Both are set manually to constant values that we find to work well for all examples.

We also discuss translating mouse motion to camera motion. This is done by having the camera move in such a way that the mouse is always above the same point. Because this is underdetermined, we also add the constraint that the camera should stay fixed in the vertical plane. Note that this is not the only constraint possible. Other constraints may restrict the camera to the surface of a sphere around the scene for example.

#### A.7. Track Sparsity

For camera control and motion transfer motion prompts, we obtain a dense set of tracks. Empirically, we find that it is helpful to randomly subsample these tracks, as using too many tracks suppresses the video model’s learned priors from working, while using too few affords too little control. Somewhere in the middle is a sweet spot. For exam-

ple, for the majority of the depth-based motion prompts, we use 1024 tracks, which we find offers a good balance between control and emergent video prior effects. In other cases, such as transferring the motion of the person’s face in Fig. 8, we find that fewer tracks is helpful in dealing with misalignments between the source video and the input first frame. Finally, for out-of-domain motion transfer as in the monkey chewing example in Fig. 8, we find that very dense tracks help. We use 1500 tracks, as we need a lot of control to apply such an unnatural motion to the first frames.

#### A.8. Davis Eval

We conduct a quantitative evaluation of first frame, text, and track conditioned video generation using the DAVIS validation dataset, with a subset of results in Tab. 1 and full results in Tab. A2. The validation dataset contains 30 videos from a wide range of scenes, involving subjects from sports to humans to animals to cars. In order to create inputs for the models, we extract tracks using BootsTAP [13]. First frame inputs are square crops of the first frames of the videos, and text prompts are the titles of the videos given by the dataset and typically consist of a word or two. For each evaluation for a given number of tracks, we randomly sample that number of tracks for conditioning.

To ensure a fair comparison, we make the following accommodations for baselines. In addition to a first frame, tracks, and a text prompt, DragAnything requires segmentation masks for objects that the tracks move. To get this, we use the provided ground truth segmentations in the DAVIS dataset. Image Conductor is a finetuned version of AnimateDiff and is trained on videos of resolution 384 × 256. We initially gave the model 256 × 256 images, and found that we got reasonable results. However, we experimented with reflection padding the input frame to 384×256 and cropping the output, which gave slightly better results which we report.

#### A.9. Human Studies

To perform the human studies, we handcraft 30 inputs with diverse image subjects and input motions. Motions consist of a single uninterrupted trajectory. Text prompts are designed to describe the image, but not the desired motion, to factor out the influence of text as much as possible. DragAnything requires masks, which we obtain by running SAM [38] on the first frame with the initial location of the tracks as query points. For our method, we turn the trajectory into a grid of tracks as described above. We then feed these inputs to the models and take a single random sample. We follow the same protocol as above for Image Conductor. This results in 30 samples for each model and 90 samples in total.

We run a two alternative forced choice (2AFC) test between our model and the baselines. We display a sample

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

Figure A2. Test and Train Metrics. Here we plot out training loss, along with PSNR, SSIM, LPIPS, and EPE on our DAVIS test set. Note that there is no correlation between the training loss and the test metrics, and that the test metrics show no signs of improvement until step 20,000 at which point the network learns quite rapidly.

from our method and a sample from the baseline in a random order with a video of the corresponding motion conditioning in the middle, visualized as a moving red dot. Participants are then asked three questions. Verbatim, we ask (1) Which video better follows the motion of the red dot? (2) Which video has the more realistic motion? (3) Which video is of higher visual quality? These questions are designed to measure the adherence of the motion to the conditioning, the quality of the motion, and the overall visual quality of video, respectively. This results in a total of 180 questions.

We recruit participants for our study through Amazon Mechanical Turk (MTurk). To ensure responses are of high quality, we add three “vigilance” questions with clearly correct answers. We discard all responses that fail any of these three questions. Each question is conducted as a separate study, and the resulting number of participants are N = 103, N = 103, and N = 115 for each question respectively. This results in a total of 19,260 answers.

### B. Training Observations

In training, we observe similar behavior as noted in ControlNet [87] and ControlNext [50]: 1) training loss does not directly correlate with model performance, and 2) “sudden convergence” where in a few epochs the model goes from not adhering to control signal to full adherence. ControlNext identifies both of these behaviors as coming from the zero initialization and offers cross normalization as a potential solution. We believe this and other future control scheme is a promising avenue for future work in track conditioned video generation. We show training loss and test metrics in Fig. A2. As can be seen, the training loss is fairly inscrutable, while the test losses do not begin to decrease until step 20,000.

### C. Full Quantitative Results

In Sec. 5 we present DAVIS evaluation results for N = {1,16,512,2048}. In Tab. A2 we present results for N = 4 and N = 64 as well, which we omit from the main paper for brevity.

- Table A2. Quantitative Evaluations. We evaluate the appearance (PSNR, SSIM, LPIPS, FVD) and motion (EPE) of generated videos using the validation set of the DAVIS dataset. Please note that each method is trained from a different base model.

# Tracks Method PSNR ↑ SSIM ↑ LPIPS ↓ FVD ↓ EPE ↓

Image Conductor 11.468 0.145 0.529 1919.8 19.224 DragAnything 14.589 0.241 0.420 1544.9 9.135 Ours 15.431 0.266 0.368 1445.2 14.619

N = 1

Image Conductor 12.017 0.176 0.499 1735.0 18.921 DragAnything 15.040 0.272 0.397 1497.2 8.946

N = 4

- Ours 15.820 0.319 0.353 1207.7 12.985

N = 16

Image Conductor 12.184 0.175 0.502 1838.9 24.263 DragAnything 15.119 0.305 0.378 1282.8 9.800

- Ours 16.618 0.405 0.319 1322.0 8.319

Image Conductor 12.513 0.180 0.503 1947.7 26.316 DragAnything 14.499 0.274 0.393 1342.0 10.642 Ours 18.000 0.513 0.265 951.4 4.127

N = 64

Image Conductor 11.902 0.132 0.524 1966.3 30.734 DragAnything 15.055 0.289 0.381 1379.8 10.948

N = 512

- Ours 18.968 0.583 0.229 688.7 4.055

N = 2048

Image Conductor 11.609 0.120 0.538 1890.7 33.561 DragAnything 14.845 0.286 0.397 1468.4 12.485

- Ours 19.327 0.608 0.227 655.9 3.887

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Figure A3. Pose Conditioning. We estimate human pose, animate it, translate it to tracks, and then feed it to our model. In each row, we show frames from generated videos with input tracks overlaid on top.

### E. Motion Magnification

One additional application of our model is motion magnification [40, 47, 49, 71, 77]. This task involves taking a video with subtle motions and generating a new video in which these motions have been magnified, so that they are easier to see. We do this by running a tracking algorithm [12] on an input video, smoothing the tracks by applying a Gaussian blur over space and time, and then magnifying the resulting tracks. We then feed the first frame of the input video and the magnified tracks to our model. We show results, along with space-time slices, in Fig. A4. We found that smoothing was necessary to reduce noise in the estimated tracks. As a result the magnified tracks are not exactly at the specified magnification factor, but nonetheless are qualitatively useful in revealing subtle motions. We expect more accurate point tracking algorithms will remove the need for this smoothing step.

### D. Human Pose Control

We show results on using our method to control humans through human pose estimated keypoints in Fig. A3. To do this, we first estimate the pose with an off-the-shelf model, and then apply motions to desired keypoints and feed it to our model.

|[Figure 252]|
|---|

|[Figure 253]|
|---|

|[Figure 254]|
|---|

|[Figure 255]|
|---|

|[Figure 256]|
|---|

|[Figure 257]|
|---|

|[Figure 258]|
|---|

|[Figure 259]|
|---|

|[Figure 260]|
|---|

|[Figure 261]|
|---|

Figure A4. Motion Magnification. We show the result of using our model to perform motion magnification. We show the first frame of two videos, and space-time slices through the blue line at different magnification factors.

