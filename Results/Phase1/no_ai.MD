# Qualitative

## Images with no detections at all at Yolo size 640 (17 total):

- 0000103_04948_d_0000035.jpg : could be considered a truck, depending on what a truck is in COCO dataset, however, weirdly shaped so could also be the shape

- 0000244_06388_d_0000014.jpg : shot from far away clearly containing multiple cars

- 0000253_00001_d_0000001.jpg : shot from huge distance from overhead, contains multiple cars

- 0000276_05601_d_0000535.jpg : shot from rather close distance, from the front, a few cars spottable but hardly seeable even by eye due to bright sunlight

- 0000276_05801_d_0000536.jpg : same scene, so similar setting to "0000276_05601_d_0000535.jpg" containing just a little fewer cars

- 0000287_05001_d_0000782.jpg : shot from overhead showing a lot of cars

- 0000333_01765_d_0000010.jpg : similar to "0000287_05001_d_0000782.jpg" showing lots of cars from an overhead view

- 0000333_01961_d_0000011.jpg : exactly the same as "0000333_01765_d_0000010.jpg" and "0000287_05001_d_0000782.jpg"

- 0000333_02157_d_0000012.jpg : same as "0000333_01961_d_0000011.jpg"

- 0000333_02353_d_0000013.jpg : same

- 0000333_02549_d_0000014.jpg : same

- 0000333_02745_d_0000015.jpg : same

- 0000333_02941_d_0000016.jpg : same

- 0000333_03137_d_0000017.jpg : same

- 0000333_03333_d_0000018.jpg : same

- 0000333_03725_d_0000020.jpg : similar to 0000011 to 0000018 but shows a new scene but also overhead with multiple cars

## Images with no detections at all at Yolo size 1280 (11 total):

All images with no detections at imgsz=1280 already had no detection in imgsz=640

The 11 left are 10 from the same overhead scene similar to imgsz=640 and 1 from another overhead scene

## Combining both sizes:

Number of Detections for imgsz=640: 16068
Number of Detections for imgsz=1280: 29828

Images that were fixed (at least a little, as they just don't have 0 detections anymore, 6 total):

- 0000103_04948_d_0000035.jpg : bounding box fixed, class is wrong though

- 0000244_06388_d_0000014.jpg : not properly fixed, only detected one ground truth object and even that is split in two detections

- 0000276_05601_d_0000535.jpg : not really fixed, one car detected, half of the objects are partly occluded (gt 1) and light conditions are hindering the detection

- 0000276_05801_d_0000536.jpg : similar to "0000276_05601_d_0000535.jpg", even though the one proper visible car is detected, other farther away or at least party occluded cars are not detected though

- 0000287_05001_d_0000782.jpg : overhead shot, around 1/3 ob the objects are detected, while similar visible objects are not

- 0000333_00001_d_0000001.jpg : same as "0000287_05001_d_0000782.jpg" for another scene

## Findings:

- For imgsz=640: Shots from straight overhead are the most common type of image to fail to detect cars at all.
- For imgsz=1280 a similar behaviour remains, even though most of the failures are from the same overhead scene
- cars a little further away can be skipped in detection due to light
- increasing the image size to 1280 (scaling the amount of pixels by factor 4) nearly doubled the amount of detections

# Quantitative

We use IoU 0.7 and conf >=0.1

- So far only Recall, No Precision (so false alarms are not considered)
- Recall per Image Size:
  found_640 0.175407
  found_1280 0.322226

- Recall per Class shows a clear increase of recall for imgsz=1280 vs 640. Smaller classes like People or Bycicles appear to have a stronger increase

- Improvement/Worsening after size increase:
  Only 6.2% of Persons are detected in both imgsz while 15.6% are only detected in the imgsz=1280 case.

- Recall per Image and Image Size:
  images with a lower median object size show a lower recall. In general there appears to be a logarithmic trend with some extreme values for both imgsz=1280 and imgsz=640

- Objects per Image vs Recall:
  Images containing more objects, so possibly images with a load of people since they are smaller and more of them can fit, show a significantly smaller recall. So recall is decreasing with amount of objects in image for both imgsz.

- Sensitivity to IoU Threshold:
  A larger IoU threshold results in a lower recall. Recall is dropping faster and stronger for imgsz=1280 than for imgsz=640

## Recall vs Object sizes per Image Class

### Orignal size

- Person: imgsz=1280 detects smaller persons faster and shows a quicker increase in recall, for large person sizes both imgsz end up at a similar recall

- bycicles are barely detected at all, even though imgsz=1280 shows an improvement, still very low recall though

- for cars, motorcycles, buses and trucks, the behaviour of the curves is similar, both increase closely together with increasing object sizes. The difference between imgsz=1280 recall and 640 recall is mostly consistent, even though for cars imgsz=1280 gets off faster, similar to people, meaning it's recall increases faster for small object sizes than for imgsz=640, but not that much different as for people

### Effective Size

- Disclaimer: the effective size depends on the imgsz, meaning that imgsz=1280 contains fewer very small objects while imgsz=640 on the other hand, contains fewer very large objects. So even though we use the same x-Axis, the number of objects per bin, especially for the small and large bins, differs a lot.

- we can see that mostly both lines are very closely together, hinting that the recall may increase with the object size independent of the imgsz. For Person and Car classes, the recall for 32-64px bin is visbly larger for imgsz=640 than for imgsz=1280, but we have to consider the smaller sample size for that.

- for person the perfect (or near perfect) recall for size>64 is not represantative, as it only contains very few persons that are of this size.

- the higher resolution (imgsz=1280) is mostly helpful because objects that shrink below the threshold of detectability (approx. 8 to 16 px, as recall <8 is 0 for all classes, below 8px nothing is detected, between 8 to 16px recall starts to rise) at a resolution of 640 remain above that threshold at 1280. The threshold of detectability lies at roughly the same effective object size (approx. 8–16 px at network input) in both runs

- The concluding finding. A higher resolution is mostly helpful because it increases the object size over the threshold of detectability.
