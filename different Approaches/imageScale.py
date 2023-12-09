import cv2

def resize_image_aspect_ratio(reference_image_path, target_image_path, target_size):
    # Load the reference image to get its aspect ratio
    reference_image = cv2.imread(reference_image_path)
    print(reference_image.shape[::1])
    
    # Get the aspect ratio of the reference image
    reference_aspect_ratio = reference_image.shape[1] / reference_image.shape[0]

    # Load the target image
    target_image = cv2.imread(target_image_path)
    print(target_image.shape[::1])
    # Calculate the new dimensions while maintaining the aspect ratio
    if reference_aspect_ratio > 1:  # landscape orientation
        new_width = target_size
        new_height = int(target_size / reference_aspect_ratio)
    else:  # portrait or square orientation
        new_height = target_size
        new_width = int(target_size * reference_aspect_ratio)
    
    # Resize the target image
    resized_target_image = cv2.resize(target_image, (new_width, new_height))
    
    return resized_target_image

# Provide the paths to the images
one_image_path = "roi.jpg"
three_image_path = "images/8_montego.jpg"

# Resize "three" while maintaining aspect ratio based on "one" image
resized_three_image = resize_image_aspect_ratio(one_image_path, three_image_path, target_size=500)
print(resized_three_image.shape[::1])
# Save or display the resized image
cv2.imwrite("resized_three.jpg", resized_three_image)
cv2.imshow("Resized Three", resized_three_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
