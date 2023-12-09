import cv2
from cig_database import Cigarette  # Assuming cig_database is the name of your module

# Load the Cigarette class and create an instance
cig = Cigarette()
# List of images to compare
image_paths_to_compare = []
image_paths_to_compare.append(f'images/box/cropped25.jpg')
image_paths_to_compare.append(f'images/box/cropped32.jpg')
image_paths_to_compare.append(f'images/box/cropped23.jpg')

roi =cv2.imread(image_paths_to_compare[2], 0)
cv2.imshow("1", roi)
cv2.waitKey(0)

# Call the function
most_similar_product, best_similarity, similarity_scores, hist_template_scores = cig.find_most_similar_productV3(image_paths_to_compare)

# Print the results
print(f"The most similar product is '{most_similar_product}' with a similarity index of {best_similarity}")
print("Similarity scores:")
for product_name, similarity_index in similarity_scores.items():
    print(f"{product_name}: {similarity_index}")

print("Best template scores:")
for product_name, best_template_score in hist_template_scores.items():
    print(f"{product_name}: {best_template_score}")    