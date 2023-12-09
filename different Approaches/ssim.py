import cv2

class CompareImage(object):
    
    def __init__(self, image_1_path, image_2_path):
        self.minimum_commutative_image_diff = 1
        self.image_1_path = image_1_path
        self.image_2_path = image_2_path
    
    def compare_image(self):
        image_1 = cv2.imread(self.image_1_path, 0)
        image_2 = cv2.imread(self.image_2_path, 0)

        w, h = image_2.shape[::-1]
        image_1  = cv2.resize(image_1, (w , h), cv2.INTER_AREA )
        cv2.imshow("1", image_1)
        cv2.imshow("2", image_2)
        cv2.waitKey(0)

        commutative_image_diff = self.get_image_difference(image_1, image_2)
    
        if commutative_image_diff < self.minimum_commutative_image_diff:
            print("Matched")
            return commutative_image_diff
        print(commutative_image_diff, ": It is not matched.")
        return 10000  # random failure value
    
    @staticmethod
    def get_image_difference(image_1, image_2):
        first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])
    
        img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
        img_template_probability_match = cv2.matchTemplate(image_1, image_2, cv2.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match
    
        # taking only 10% of histogram diff, since it's less accurate than the template method
        commutative_image_diff = (img_hist_diff / 10) + img_template_diff
        return commutative_image_diff

if __name__ == '__main__':
    compare_image = CompareImage('images/box/cropped2.jpg', 'images/box/cropped2.jpg')
    image_difference = compare_image.compare_image()
    print(image_difference)
