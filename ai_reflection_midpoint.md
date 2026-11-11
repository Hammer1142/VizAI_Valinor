# AI Reflection — Phase I Midpoint

**Project:** Drone-Based Object Detection

**AI-assisted pair**

| First name | Last name | Student ID |
|Ashraf Rumman|Rumman|K12455624|

| Yasaman | Afsharnaderi | k01229654 |


## Reflection on our Phase I experience

During Phase I, we worked with GenAI to investigate whether object size influences YOLO11n's detection performance on aerial images from VisDrone. What we found most valuable was not simply getting help with Python code, but having the flexibility to develop our ideas through discussion. We could consider an approach, see what the visualization revealed, and then adjust it if it did not answer our question clearly.

At first, we were mainly thinking about comparing the sizes of detected and missed objects. As the work progressed, we realized that we needed both a broader view across the dataset and a closer look at individual images. The interactive visualizations made this possible. We could examine relative object sizes across images, then select a particular image and category from the dropdown menus to inspect its bounding boxes and detection outcomes. This was much more informative than relying on a single performance score, as it allowed us to see the objects behind the numbers.

The visualizations also helped us notice a possible pattern. In our current results, larger cars were detected more often than smaller cars when we compared objects of the same category within each image. For example, the notebook reports a detection rate of approximately **63% for the larger half of cars versus 20% for the smaller half**. These values still need further verification, but seeing the pattern in both charts and individual photographs gave us a clearer direction for the investigation. We also looked at similarly sized objects with different detection outcomes, which suggests that object size may not explain every failure.

Another useful step was adding a separate confidence visualization. Being able to view confidence scores directly beside predicted bounding boxes and compare confidence with object size made the results easier to explore. It also helped us distinguish two questions that we initially treated together: whether YOLO detects an object at all, and how confident it is in the predictions it makes.

A further step was asking GenAI to review the finished notebook critically before we pushed it. This review turned out to be as valuable as the development itself. It confirmed our headline numbers, but it also revealed that one of our scatter plots contained a misleading stripe caused by how we ranked images containing a single object, and that our confidence view did not distinguish correct predictions from false positives. It also prompted us to look beyond size: comparing occlusion levels within the same size range showed that heavily occluded objects are rarely detected even when they are large. We had not considered that our input resolution (`imgsz=640`) shrinks most VisDrone objects to only a few pixels. This gives us a concrete starting point for Phase II.

The collaboration was not always straightforward. Some suggestions were different from what we had in mind, and parts of the code needed correction. We had to explain our intentions more precisely, test the implementations and decide which ideas were worth keeping. GenAI supported the process, but the research direction and final design choices remained our responsibility.

The experience gave us more freedom to experiment with different views of the same data and helped us identify patterns worth investigating further, particularly using an interactive visualization.
