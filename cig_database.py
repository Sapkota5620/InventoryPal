import cv2
import numpy as np

class Cigarette:

    def __init__(self):
        self.products = {}
        for product_name, product_info in self.product_data.items():
            product = self.create_product(product_name, product_info["variations"])
            self.products[product_name] = product
    
    def create_product (self, name , variations):
        product = {
            "name": name,
            "variations": [],
            "color_variations": set(),
            "images": set(),
            "loc": set()
        }

        for variation in variations:
            size = variation["size"]
            color = variation["color"]
            image = variation["image"]
            loc = f"images/src/{image}"

            product["variations"].append({"size": size, "color": color, "image": image})
            product["color_variations"].add(color)
            product["images"].add(image)
            product["loc"].add(loc)
        return product
    
    def __str__(self):
        product_name = list(self.products.keys())[0]  # You may need to adjust this depending on your use case
        product = self.products[product_name]
        return f"{product['name']} - Variations: {len(product['variations'])}, Color Variations: {len(product['color_variations'])}, Images: {len(product['images'])}, loc: {len(product['loc'])}"
        # Cigarette data
    product_data = {
        "247": {
            "variations": [
                {"size": "Kings", "color": "Red", "image": "247_Kings_red.jpg"},
                {"size": "Kings", "color": "Blue", "image": "247_Kings_blue.jpg"},
                {"size": "Kings", "color": "Green", "image": "247_Kings_green.jpg"},
                {"size": "100s", "color": "Red", "image": "247_100s_red.jpg"},
                {"size": "100s", "color": "Blue", "image": "247_100s_blue.jpg"},
                {"size": "100s", "color": "Green", "image": "247_100s_green.jpg"},
                {"size": "100s", "color": "Silver", "image": "247_100s_silver.jpg"},
                # Add more variations as needed
            ],
        },
        "Edgefield": {
            "variations": [
                {"size": "Kings", "color": "Red", "image": "Edgefield_Kings_red.jpg"},
                {"size": "Kings", "color": "Blue", "image": "Edgefield_Kings_blue.jpg"},
                {"size": "Kings", "color": "Green", "image": "Edgefield_Kings_green.jpg"},
                {"size": "Kings", "color": "Lgreen", "image": "Edgefield_Kings_lgreen.jpg"},   
                {"size": "Kings", "color": "Silver", "image": "Edgefield_Kings_silver.jpg"},
                {"size": "Kings", "color": "nonfilter", "image": "Edgefield_Kings_nonfilter.jpg"},

                {"size": "100s", "color": "Red", "image": "Edgefield_100s_red.jpg"},
                {"size": "100s", "color": "Blue", "image": "Edgefield_100s_blue.jpg"},
                {"size": "100s", "color": "Green", "image": "Edgefield_100s_green.jpg"},
                {"size": "100s", "color": "LGreen", "image": "Edgefield_100s_lgreen.jpg"},
                {"size": "100s", "color": "Silver", "image": "Edgefield_100s_silver.jpg"},
            ],
        },
        "LD": {
            "variations": [
                {"size": "Kings", "color": "Red", "image": "LD_Kings_red.jpg"},
                {"size": "Kings", "color": "Blue", "image": "LD_Kings_blue.jpg"},
                {"size": "Kings", "color": "Green", "image": "LD_Kings_green.jpg"},
                {"size": "Kings", "color": "LGreen", "image": "LD_Kings_lgreen.jpg"},   
                {"size": "Kings", "color": "Silver", "image": "LD_Kings_silver.jpg"},
                
                {"size": "100s", "color": "Red", "image": "LD_100s_red.jpg"},
                {"size": "100s", "color": "Blue", "image": "LD_100s_blue.jpg"},
                {"size": "100s", "color": "Green", "image": "LD_100s_green.jpg"},
                {"size": "100s", "color": "LGreen", "image": "LD_100s_lgreen.jpg"},
                {"size": "100s", "color": "Silver", "image": "LD_100s_silver.jpg"},
                # Add more variations as needed
            ],
        },
        "Montego": {
            "variations": [
                {"size": "Kings", "color": "Red", "image": "Montego_Kings_red.jpg"},
                {"size": "Kings", "color": "Blue", "image": "Montego_Kings_blue.jpg"},
                {"size": "Kings", "color": "Green", "image": "Montego_Kings_green.jpg"},
                {"size": "Kings", "color": "LGreen", "image": "Montego_Kings_lgreen.jpg"},   

                
                {"size": "100s", "color": "Red", "image": "Montego_100s_red.jpg"},
                {"size": "100s", "color": "Blue", "image": "Montego_100s_blue.jpg"},
                {"size": "100s", "color": "Green", "image": "Montego_100s_green.jpg"},
                {"size": "100s", "color": "LGreen", "image": "Montego_100s_lgreen.jpg"},
                {"size": "100s", "color": "Silver", "image": "Montego_100s_silver.jpg"},
                # Add more variations as needed
            ],
        },
        # Add more products as needed
    }

    def find_most_similar_productV2(self, image_paths):
        best_similarity = float('-inf')
        most_similar_product = None
        similarity_scores = {}
        hist_template_scores = {}

        for product_name, product in self.products.items():
            for variation in product["variations"]:
                image = variation["image"]
                image_path_template = f"images/src/{image}"

                total_similarity = 0
                for image_path in image_paths:
                    product_image_path = image_path_template.replace("{variation}", image_path)
                    product_image = cv2.imread(product_image_path)

                    if product_image is not None:
                        # Convert images to grayscale
                        gray_cropped_image = cv2.cvtColor(product_image, cv2.COLOR_BGR2GRAY)
                        gray_comparison_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                        w, h = gray_comparison_image.shape[::-1]
                        gray_cropped_image = cv2.resize(gray_cropped_image, (w, h), cv2.INTER_AREA)

                        # Compute histogram - commutative_image_diff
                        hist_cropped_image = cv2.calcHist([gray_cropped_image], [0], None, [256], [0, 256])
                        hist_comparison_image = cv2.calcHist([gray_comparison_image], [0], None, [256], [0, 256])

                        # Normalize histograms
                        cv2.normalize(hist_cropped_image, hist_cropped_image, 0, 1, cv2.NORM_MINMAX)
                        cv2.normalize(hist_comparison_image, hist_comparison_image, 0, 1, cv2.NORM_MINMAX)

                        # Calculate histogram comparison using Bhattacharyya distance
                        similarity_index = cv2.compareHist(hist_cropped_image, hist_comparison_image,
                                                            cv2.HISTCMP_BHATTACHARYYA)
                        total_similarity += similarity_index

                        # Store similarity score in the dictionary
                        similarity_scores[(product_name)] = similarity_index

                average_similarity = total_similarity / len(image_paths)
                

                if average_similarity > best_similarity:
                    best_similarity = average_similarity
                    most_similar_product = product_name

        return most_similar_product, best_similarity, similarity_scores
    
    def find_most_similar_productV3(self, image_paths):
        best_similarity = float('-inf')
        most_similar_product = None
        similarity_scores = {}
        hist_template_scores = {}

        for product_name, product in self.products.items():
            best_template_score = float('-inf')  # Initialize the best template score for each product

            for variation in product["variations"]:
                image = variation["image"]
                image_path_template = f"images/src/{image}"

                total_similarity = 0
                for image_path in image_paths:
                    product_image_path = image_path_template.replace("{variation}", image_path)
                    product_image = cv2.imread(product_image_path)

                    if product_image is not None:
                        # Convert images to grayscale
                        gray_cropped_image = cv2.cvtColor(product_image, cv2.COLOR_BGR2GRAY)
                        gray_comparison_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                        w, h = gray_comparison_image.shape[::-1]
                        gray_cropped_image = cv2.resize(gray_cropped_image, (w, h), cv2.INTER_AREA)

                        # Compute histogram - commutative_image_diff
                        hist_cropped_image = cv2.calcHist([gray_cropped_image], [0], None, [256], [0, 256])
                        hist_comparison_image = cv2.calcHist([gray_comparison_image], [0], None, [256], [0, 256])

                        # Normalize histograms
                        cv2.normalize(hist_cropped_image, hist_cropped_image, 0, 1, cv2.NORM_MINMAX)
                        cv2.normalize(hist_comparison_image, hist_comparison_image, 0, 1, cv2.NORM_MINMAX)

                        # Calculate histogram comparison using Bhattacharyya distance
                        similarity_index = cv2.compareHist(hist_cropped_image, hist_comparison_image,
                                                            cv2.HISTCMP_BHATTACHARYYA)
                        img_template_similarity = cv2.matchTemplate(gray_cropped_image, gray_comparison_image, cv2.TM_CCOEFF_NORMED)[0][0]
                        img_hist_diff = 1 - similarity_index

                        # taking only 10% of histogram diff, since it's less accurate than the template method
                        commutative_image_diff = (img_hist_diff / 10) + img_template_similarity

                        # Adjust the threshold as needed
                        if commutative_image_diff < 1:
                            hist_template_scores.setdefault(product_name, []).append(commutative_image_diff)

                        total_similarity += similarity_index

                        # Store similarity score in the dictionary
                        similarity_scores[product_name] = similarity_index

                average_similarity = total_similarity / len(image_paths)

                if average_similarity > best_similarity:
                    best_similarity = average_similarity
                    most_similar_product = product_name

                # Update the best_template_score for the current product
                if hist_template_scores.get(product_name):
                    best_template_score = max(best_template_score, max(hist_template_scores[product_name]))

            # Store the best template score for each product
            hist_template_scores[product_name] = best_template_score

        return most_similar_product, best_similarity, similarity_scores, hist_template_scores

