from inspect import unwrap
from multiprocessing.resource_sharer import stop

import cv2
import numpy as np
from numpy import linalg

class Evaluator:

    def __init__(self, initial_images):
        if(initial_images is None):
            print("ERROR: The initial image is None")
            return
        self.frame_accumulator = initial_images
        self.original_image = initial_images[0].copy()  # original image
        self.original_image_gray = cv2.cvtColor(initial_images[0].copy(), cv2.COLOR_BGR2GRAY)
        self.original_image_unscrewed = None
        self.stage_images = []                              # [img name] [image]

        self.prev_lines = []

        self.homography_matrix = None
        self.unscrewed_image_width = 600
        self.unscrewed_image_height = 600

        self.add_stage("0. Initialisation: Original Image ", self.original_image)

        self.unscrew()


    ## Homography matrix things
    def unscrew(self):

        # Params ------------------------------------------------------
        canny_lower = 50
        canny_upper = 150

        # Edge Detection ----------------------------------------------
        img = self.original_image_gray.copy()
        img = cv2.GaussianBlur(img, (5, 5), 0)

        edges = np.zeros_like(img)
        for frame in self.frame_accumulator:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
            edge_n = cv2.Canny(frame_gray, canny_lower, canny_upper)
            edges = cv2.bitwise_or(edges, edge_n, mask=edge_n)

        edges_previously = cv2.Canny(img, canny_lower, canny_upper)

        cv2.imshow("EDGES PREVIOUSLY", edges_previously)
        cv2.waitKey(0)
        cv2.imshow("EDGES ACCUMULATED", edges)
        cv2.waitKey(0)








        # Contours ----------------------------------------------------
        # First parameter cv2.RETR_TREE how to return the relationship between the contours
        # Second parameter cv2.CHAIN_APPROX_NONE should it be simplified or not the contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) <= 0:
            print("ERROR in unscrew: No contours were found found!")
            return True
        print("Found " + str(len(contours)) + " contours")

        #Find biggest contour based on Area and shape
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            #print("The contour area is: " + str(cv2.contourArea(contour)))
            epsilon = 0.02 * cv2.arcLength(contour, True)       # Calculates the perimeter of the contour and Checks how much the simplified (aproximated point) can be from real thing
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):  # must be a square/rectangle
                biggest_contour = contour

                break

        print("The biggest contour is: " + str(cv2.contourArea(biggest_contour)))

        # This can all be deleted after debuging is finished------
        #debug_img = self.original_image.copy()
        #cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)  # all contours in green
        #cv2.drawContours(debug_img, [biggest_contour], -1, (0, 0, 255), 10)
        #cv2.imshow("Debug Image", debug_img)
        #cv2.waitKey(0)

        if(biggest_contour is None):
            print("ERROR in unscrew: Could not find biggest contour");

        # Homography ----------------------------------------------------
        # ASSUMES BIGGEST CONTOUR IS THE SQUARED

        corners = approx.reshape(4, 2).astype(np.float32)
        square = np.zeros((4, 2), dtype=np.float32)
        sum = corners.sum(axis=1)               # x+y for each point
        diff = np.diff(corners, axis=1)         # x-y for each point

        square[0] = corners[np.argmin(sum)]     # (x+y).min = Top Left
        square[2] = corners[np.argmax(sum)]     # (x+y).max = Bottom Right
        square[1] = corners[np.argmin(diff)]    # (x-y).min = Top Right
        square[3] = corners[np.argmax(diff)]    # (x-y).max = Bottom Left

        target_format = np.array([
            [0, 0],                                                                 # top-left corner
            [self.unscrewed_image_width-1, 0],                                      # top-right corner
            [self.unscrewed_image_width - 1, self.unscrewed_image_height - 1],      # bottom-right corner
            [0, self.unscrewed_image_height - 1]                                    # bottom-left corner
        ], dtype=np.float32)

        self.homography_matrix = cv2.getPerspectiveTransform(square, target_format) #homography matrix

        unwarped = cv2.warpPerspective(self.original_image.copy(), self.homography_matrix, (self.unscrewed_image_width, self.unscrewed_image_height))
        self.original_image_unscrewed = unwarped.copy()
        self.add_stage("1. Unscrew: Unwrapped original image", unwarped)

        unwarped_gray = cv2.cvtColor(unwarped, cv2.COLOR_BGR2GRAY)
        #unwarped_gray = cv2.GaussianBlur(unwarped_gray, (5, 5), 0)
        unwarped_edges = cv2.Canny(unwarped_gray, canny_lower, canny_upper)
        self.prev_lines = self.detectLines(unwarped_edges, treshold=20)
        self.add_stage("2. Unscrew: Detected lines From Unscrewed image", self.drawLines(unwarped, self.prev_lines))

        return None






    def evaluate(self, eval_img):
        """
            1. Warp (homography)
            2. Canny
            3. Line detection / XOR
            4. XOR (Isolate arrow)
        """
        if(eval_img is None):
            print("ERROR in evaluate: The evaluate image is None")
            exit(0)
        if(self.homography_matrix is None):
            print("ERROR in evaluate: The homography matrix is None")
            exit(0)

        #Params -------------------------------------------
        score = 0
        img = cv2.cvtColor(eval_img.copy(), cv2.COLOR_BGR2GRAY)

        canny_lower = 50
        canny_upper = 150

        target_size_in_cm = 50  # the rectangle
        target_specifications = np.array([      #[<cm_radius, score]
            [2, 10],
            [4, 9],
            [6, 8],
            [8, 7],
            [10, 6],
        ], dtype=np.int32)

        center_pixel = [self.unscrewed_image_height // 2, self.unscrewed_image_width // 2]  # [y,x] cos cv2 does it like that
        pixel_per_cm = self.unscrewed_image_width / target_size_in_cm  # Pixels per cms

        # 1 WARP -------------------------------------------------------------------
        unwrapped_img = cv2.warpPerspective(img.copy(), self.homography_matrix, (self.unscrewed_image_width, self.unscrewed_image_height))
        empty_img = np.zeros_like(unwrapped_img)
        empty_img = cv2.cvtColor(empty_img, cv2.COLOR_GRAY2BGR)

        # 2 Canny edge detection ---------------------------------------------------
        edges = cv2.Canny(unwrapped_img, canny_lower, canny_upper)

        # 3 Hough Line detction ---------------------------------------------------
        lines = self.detectLines(edges, treshold=20)
        if(lines is None):
            print("ERROR in evaluate3: The evaluate lines is None")



        # 4 Isolate individual arrow ---------------------------------------------------
        current_lines_img = self.drawLines(empty_img, lines)
        arrow_isolated_img = self.drawLines(current_lines_img.copy(), self.prev_lines, (0, 0, 0), 10) #thick so it paints black over it
        arrow_isolated_img_gray = cv2.cvtColor(arrow_isolated_img.copy(), cv2.COLOR_BGR2GRAY)
        arrow_isolated_line = self.detectLines(arrow_isolated_img_gray,treshold=20)                                                 #this might be useful if it has big jumping gap so it can connect cross over arrows

        if arrow_isolated_line is None or len(arrow_isolated_line) == 0:
            print("ERROR in Evaluation4: No isolated arrow detected")
            return 0

        # 5 Update previous lines ---------------------------------------------------
        self.prev_lines = np.concatenate((self.prev_lines, lines), axis=0)

        # 6 Evaluate  -------------------------------------------------------------------
        print("Number of isolated lines of just one arrow is: ", len(arrow_isolated_line))

        min_point = None

        for line in arrow_isolated_line:
            for x1, y1, x2, y2 in line:
                if min_point is None or x1 < min_point[1]:
                    min_point = np.array([y1, x1])

        distance_from_center = np.linalg.norm(min_point - center_pixel) #euclidian distance from center
        #min_distance = np.min(distance_from_center)     #argmin - index vs min - number

        for i in range(len(target_specifications)):
            radius_pixels = target_specifications[i][0] * pixel_per_cm
            if distance_from_center <= radius_pixels:
                score = target_specifications[i][1]
                break

        print("SCOREEE IS : " + str(score))

        # 7 Debug -------------------------------------------------------------------

        visual = empty_img.copy()
        for radius, _ in target_specifications:
            cv2.circle(visual, (center_pixel[1], center_pixel[0]), int (radius*pixel_per_cm), (0, 255, 0), 1)
        for line in arrow_isolated_line:
            for x1, y1, x2, y2 in line:
                cv2.line(visual, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.circle(visual, (min_point[1], min_point[0]), 10, (0, 0, 255), 2)
        cv2.circle(visual, (center_pixel[1], center_pixel[0]), 5, (0, 0, 255), -1)
        cv2.line(visual, (center_pixel[1], center_pixel[0]), (min_point[1], min_point[0]), (255, 0, 0), 1)
        cv2.putText(visual,f"Score: {score}",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255, 255, 255),2,cv2.LINE_AA)

        self.add_stage("3. Evaluate: Unwrapped image", unwrapped_img)
        self.add_stage("4. Evaluate: Edge detection image", edges)
        self.add_stage("5. Evaluate: Line detection image", current_lines_img)
        self.add_stage("6. Evaluate: Arrow isolated image", arrow_isolated_img)
        self.add_stage("7. Evaluate: Visualisation image", visual)

        return score


    def debug(self):
        for i, (name, img) in enumerate(self.stage_images):
            cv2.imshow(f"{i} - {name}", img)
            cv2.waitKey(0)

    def add_stage(self, label, img):
        self.stage_images.append((label, img.copy()))


    #A method so line detection parameters are equal everywhere
    def detectLines(self, img, treshold=50, minLineLength=20,maxLineGap=15):
        if img is None:
            print("ERROR in drawLines: No image to draw onto")
        img=img.copy()
        return  cv2.HoughLinesP(img,                                    # Returns a wird [ [[x1,x2,y1,y2]], [[x1,x2,y1,y2]] ] array
                                1,                                  # Precision of line length ~ 1px
                                np.pi / 180,                            # Precision of line angle i degrees pi/180 is 1
                                threshold=treshold,                     # How many white pixels it has to pass trough before calling it a line
                                minLineLength=minLineLength,            # Whats the minimal length on the line
                                maxLineGap=maxLineGap)

    def drawLines(self, img, lines, color=(0, 255, 0), line_thickness=1):

        if (img is None):
            print("ERROR in drawLines: No image to draw onto")
            return img
        draw_img = img.copy()
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(draw_img, (x1, y1), (x2, y2), color, line_thickness)
        return draw_img






"""
if __name__ == '__main__':


   original_img = cv2.imread("/Users/al/Desktop/Graduation Thesis/doortest4.jpeg")
   img_with_arrow = cv2.imread("/Users/al/Desktop/Graduation Thesis/doortest4_arrow.jpeg")


   if img_with_arrow is None:
       print("ERROR: Could not find image with value")




   evaluator = Evaluator(original_img)
   score = evaluator.evaluate(img_with_arrow)
   evaluator.debug()
"""

