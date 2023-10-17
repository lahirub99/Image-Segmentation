# Inter-means Algorithm

The inter-means algorithm is an iterative technique that starts with an approximate threshold and then successively refines it to estimate the correct threshold.

## Algorithm Steps

1. Select an initial estimate of the threshold, 𝑇. A good initial value is the average intensity of the image
2. Partition the image into two groups, 𝑅1 and 𝑅2, using this threshold 𝑇.
3. Calculate the mean grey values 𝜇1 and 𝜇2 for the respective partitions of 𝑅1 and 𝑅2.
4. Select a new threshold as 

    $T = \frac{\mu_1 + \mu_2}{2}$

5. Repeat steps 2 -> 4 until 𝜇1 and 𝜇2 does not significantly change in successive iterations

##
## Usage
#### 1. Read the input images
`img1 = cv2.imread('image1.jpg')`

`img2 = cv2.imread('image2.jpg')`

##
#### 2 .Apply the inter-means threshold algorithm to the images
`segmented_image1 = inter_means_threshold(img1)`

`segmented_image2 = inter_means_threshold(img2)`

##
#### 3. Save the original and segmented images
`cv2.imwrite('image1_segmented.jpg', segmented_image1)`

`cv2.imwrite('image2_segmented.jpg', segmented_image2)`

import cv2

# Read the input image
img = cv2.imread('image2.jpg')

# Read the segmented image
segmented_img = cv2.imread('image2_segmented.jpg')

# Display the images
cv2.imshow('Input Image', img)
cv2.imshow('Segmented Image', segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
### Sample Input and Output

#### Input Image
![Input Image](image2.jpg)

#### Segmented Image
![Segmented Image](image2_segmented.jpg)

##
## Dependencies

For calculations: `import numpy as np`

For read the images and save them: 
`import cv2`

##
## Author

lahirub99
