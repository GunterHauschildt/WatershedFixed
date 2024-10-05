# WatershedFixed
A post-processing algorithm to clean up the markers from OpenCV's 'watershed' segmentation.
![borders](https://github.com/user-attachments/assets/830463d8-1613-487b-8623-1ea89125373e)

OpenCV's watershed example, https://docs.opencv.org/3.4/d3/db4/tutorial_py_watershed.html, closes with
"For some coins, the region where they touch are segmented properly and for some, they are not."

The post processing algorithm presented here segments all coins properly. My algorithm
1) calls CV's watershed in the same manner as their example above.
2) finds the 'curvy' borders between objects
3) matches those borders to the nearest points on the overall mask as returned from CV's watershed
4) segments based on those minimum-distance pairs - the expected result for touching objects.
![objects](https://github.com/user-attachments/assets/f695ddb0-b293-4865-aeed-96afc39a1b38)


