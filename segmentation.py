'''
**** Inter-means algorithm ****
The inter-means algorithm is an iterative technique that starts with an 
approximate threshold and then successively refines it ot estimate eth correct 
threshold.

1. Select an initial estimate of the threshold, 𝑇. A good initial value is the average 
intensity of the image
2. Partition the image into two groups, 𝑅1 and 𝑅2, using this threshold 𝑇.
3. Calculate the mean grey values 𝜇1 and 𝜇2 for the respective partitions of 𝑅1 and 𝑅2.
4. Select a new threshold as
        𝑇 = (𝜇1 + 𝜇2)/2
5. Repeat steps 2 -> 4 until 𝜇1 and 𝜇2 does not significantly change in successive iterations
'''

import numpy as np
import cv2

# Convert the image to gray-scale (8bpp format) 
def grayscale(image_rgb):
    height, width, _ = image_rgb.shape

    image_gray = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            # Extracting the RGB values
            r, g, b = image_rgb[i][j]
            # Converting to grayscale considering the Luminance level as it was widely use than others in as in YUV and YCrCb formats
            # Formula: Y = 0.299 R + 0.587 G + 0.114 B
            image_gray[i][j] = int(0.299*r + 0.587*g + 0.114*b)

    print('Image converted to grayscale successfully!')
    return image_gray



def inter_means_threshold(img):
    # Convert the image to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = grayscale(img)

    # Initialize the threshold value to the average intensity of the image
    # with the assumption of a perfectly bimodal histogram will have more probability of occuring
    threshold = np.mean(gray)

    # Repeat until the threshold value does not change significantly
    while True:
        # Partition the image into two groups based on the threshold value
        group1 = gray[gray <= threshold]
        group2 = gray[gray > threshold]

        # Calculate the mean intensity of each group
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)

        # Calculate the new threshold value as the average of the mean intensities of the two groups
        new_threshold = (mean1 + mean2) / 2

        # Check if the threshold value has changed significantly
        # 0.5 declared as the level of significance for the change in threshold value
        if abs(threshold - new_threshold) < 0.5:
            break

        threshold = new_threshold

    # Threshold the image using the final threshold value
    thresholded = gray.copy()
    thresholded[gray <= threshold] = 0
    thresholded[gray > threshold] = 255

    return thresholded


# Read the input images
# Used opencv library to read the images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Apply the inter-means threshold algorithm to the images
segmented_image1 = inter_means_threshold(img1)
segmented_image2 = inter_means_threshold(img2)

# Save the original and segmented images
# Used opencv library to save the images 
cv2.imwrite('image1_segmented.jpg', segmented_image1)
cv2.imwrite('image2_segmented.jpg', segmented_image2)

