# Phase I — Generative AI Interaction Log

**Project:** Drone-Based Object Detection  
**Model:** YOLO11n (pretrained on COCO)  
**Dataset:** VisDrone2019-DET validation set  
**Research question:** How does object size influence detection performance in aerial images?

## Our approach

We wanted to investigate how YOLO11n detects objects of different sizes in drone images, rather than simply report an overall accuracy score. In particular, we asked whether smaller objects are missed more often, whether detected objects of different sizes receive different confidence scores, and whether similarly sized objects can have different outcomes.

The model was pretrained on COCO, but our images and annotations came from VisDrone. We therefore needed consistent class labels and bounding-box coordinates before comparing YOLO predictions with ground truth. For example, VisDrone has separate *pedestrian* and *people* labels, whereas COCO uses *person*. We used the existing model predictions and developed an analysis in Jupyter with Generative AI as a collaborator. The prompts below are **representative paraphrases of our discussions**, not a verbatim transcript.

## Key interactions with Generative AI

### 1. Preparing the data

**Prompt:** *How should we map the VisDrone categories to COCO classes, and how do we convert the bounding boxes so that the annotations and YOLO predictions can be compared?*

We discussed the class mapping, the difference between VisDrone's `(x, y, width, height)` annotations and YOLO's bounding-box formats, and which classes could be compared. Generative AI helped us refine the conversion and loading code. We then matched predictions against annotated objects using intersection over union (IoU) to identify correct detections, missed objects and possible wrong-class detections.

### 2. Deciding how to compare object sizes

**Prompt:** *Should we measure size across the whole dataset, or compare objects of the same category within each image?*

We considered different definitions of size and decided to prioritize **within-image, same-category comparisons**. The code calculates bounding-box area and divides eligible image/category groups into smaller and larger halves. We also added a dataset-level interactive plot showing the relative size rank and detection outcome of objects from different images. This gave us both an overview of possible patterns and a way to inspect them in individual images.

### 3. Designing the interactive image explorer

**Prompt:** *Can we select an image and category from dropdown menus, see which objects were detected or missed, and compare their sizes in that image?*

We worked through several versions of this visualization. The final image explorer displays the selected photograph with color-coded ground-truth boxes and a plot of object size against detection outcome. Generative AI helped with the implementation, but we had to review and correct the code. At one point the matching cell failed because the IoU function had not been defined. Later, a class filter in an earlier cell still restricted the ground-truth data to three categories even though we wanted to include more. Revising these parts was an important part of the collaboration.

### 4. Exploring confidence scores

**Prompt:** *Can we show YOLO's confidence beside each predicted bounding box and, separately, visualize confidence versus object size for the selected category?*

This required repeated clarification. Some initial suggestions combined several analyses into one chart or showed confidence as a bar chart, which was not what we wanted. We refined the request until we had a **separate interactive confidence view**: the image with prediction boxes and confidence labels on one side, and a size-versus-confidence scatter plot on the other. We also explored confidence across correctly matched detections at dataset level.

The confidence values are taken from the model predictions; we calculate bounding-box area to compare size with confidence. Objects with no matching detection do not have a prediction confidence. We therefore kept the questions of **whether an object is detected** and **how confident the model is when it predicts an object** separate.

### 5. Reviewing and validating the analysis

**Prompt:** *Here is our notebook and the results CSV. Can you review it critically, check whether the numbers are correct, and point out anything that could mislead us before we push it?*

The second member of our pair used Generative AI to review the notebook before we shared it. It confirmed the headline figures from the results CSV (for example 63% vs 20% detection for the larger and smaller halves of cars), but it also pointed out several problems that we then fixed:

- **A plotting artifact.** In the relative-rank scatter, images with only one object of a category always received rank 1.0, which created a misleading vertical stripe. We now exclude single-object groups, centre the ranks with `(rank − 0.5) / n`, and add a binned detection-rate line calculated from all objects, not only the 4,000 plotted points.
- **Plots not saved.** The `browser` renderer opened plots in a separate tab, so they were never stored in the notebook. We removed it.
- **An ambiguous confidence view.** All predictions were drawn in green, including false positives. Predictions are now coloured as correct, wrong class or false positive, and missed objects are shown as dashed boxes.
- **Outdated descriptions.** The introduction still said we analyse three categories (the code uses six), and parts of it were written as if addressed to us by the AI.

It also suggested three additional analyses, which we accepted and implemented: an **absolute-size detection curve** (detection rises from about 1% below 256 px² to 83% above 9,216 px²), a **size × occlusion comparison** (heavily occluded objects are rarely detected at any size), and an **IoU-threshold sensitivity check**, since IoU ≥ 0.5 is particularly strict for tiny boxes. It also pointed out that inference at `imgsz=640` shrinks the 1920×1080 images about three times, so most VisDrone objects fall far below COCO's "small" threshold of 32 × 32 px at the model's input. We will follow this up in Phase II.

Not every suggestion fit our design. We kept the within-image comparison as our main analysis and treat the absolute-size curve as a complement, not a replacement. As before, the AI could not run the full notebook because it did not have our images and annotations, so we re-ran everything locally and checked the new figures ourselves.

## Strengths and limitations observed

GenAI helped us explore alternatives, implement interactive visualizations and revise the analysis without starting from scratch each time. It was particularly useful when we could describe a specific problem, test a suggestion and then give feedback on the result.

However, it sometimes misunderstood the intended design, suggested more analysis than we needed or produced code that required debugging. The missing function and outdated class filter were concrete examples. We learned that **specific prompts, incremental testing and checking results against the original data** were more effective than accepting an entire generated solution. Our detection matches and emerging patterns still require manual checks before we draw strong conclusions.
