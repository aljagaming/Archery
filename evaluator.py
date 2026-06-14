from multiprocessing.resource_sharer import stop

import cv2
import numpy as np


class Evaluator:

    def __init__(self, initial_image):
        if(initial_image is None):
            print("ERROR: The initial image is None")
            return
        self.original_image = initial_image.copy()  # original image
        self.zoom = 3
        self.original_image_gray = cv2.cvtColor(initial_image.copy(), cv2.COLOR_BGR2GRAY)
        self.original_image_unscrewed = None
        self.stage_images = []                              # [img name] [image]
        self.Arrows_OR = None                               # all previous images
        self.homography_matrix = None
        self.unscrewed_image_width = 600
        self.unscrewed_image_height = 600

        self.stage_images.append(("Original Image for Homography Calculation", self.original_image))

        self.unscrew()

        print("0. Evaluator successfully initialized!")


    ## Homography matrix things
    def unscrew(self):

        # Params ------------------------------------------------------
        canny_lower = 50
        canny_upper = 150

        # Edge Detection ----------------------------------------------

        img = self.original_image_gray.copy()

        #img_zoomed_in = self.zoom_in(img)

        #cv2.imshow("Zoomed in", img_zoomed_in)
        #cv2.waitKey(0)

        # Preproccesing ----------------------------


        # -----------------------------------------
        img = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(img, canny_lower, canny_upper)

        #cv2.imshow("Edges detected", edges)
        #cv2.waitKey(0)
        # Contours ----------------------------------------------------

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # First parameter cv2.RETR_TREE how to return the relationship between the contours
        # Second parameter cv2.CHAIN_APPROX_NONE should it be simplified or not the contours

        if len(contours) <= 0:
            print("ERROR: No contours were found found!")
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


        #biggest_contour = max(contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3])

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
            [0, 0],                                                                 # top-left
            [self.unscrewed_image_width-1, 0],                                      # top-right
            [self.unscrewed_image_width - 1, self.unscrewed_image_height - 1],      # bottom-right
            [0, self.unscrewed_image_height - 1]                                    # bottom-left
        ], dtype=np.float32)

        self.homography_matrix = cv2.getPerspectiveTransform(square, target_format) #homography matrix

        unwarped = cv2.warpPerspective(self.original_image.copy(), self.homography_matrix, (self.unscrewed_image_width, self.unscrewed_image_height))
        self.original_image_unscrewed = unwarped.copy()
        self.stage_images.append(("Unwrapped original image", unwarped))
        return None






    def evaluate(self, eval_img):
        if(eval_img is None):
            print("ERROR: The evaluate image is None")
            exit(0)
        if(self.homography_matrix is None):
            print("ERROR: The homography matrix is None")
            exit(0)

        img = cv2.cvtColor(eval_img.copy(), cv2.COLOR_BGR2GRAY)

        score = 0

        # Target parameters -----------------------------------------------------
        canny_lower = 50
        canny_upper = 150

        target_size_in_cm = 50  # the rectangle
        center_pixel = [self.unscrewed_image_height // 2, self.unscrewed_image_width // 2] # [y,x] cos cv2 does it like that
        pixel_per_cm = self.unscrewed_image_width / target_size_in_cm    #Pixels per cms

        target_specifications = np.array([      #[ <cm_radius, score]
            [2, 10],
            [4, 9],  # top-right
            [6, 8],
            [8, 7],
            [10, 6],
        ], dtype=np.int32)

        # XOR -------------------------------------------------------------------
        # Isolate only the last arrow ROUGHLY
        current_arrow_edges = cv2.Canny(img, canny_lower, canny_upper)
        self.stage_images.append(("Canny edges of current ARROW (not isolated)", current_arrow_edges))


        if self.Arrows_OR is None:
            edges_prev = cv2.Canny(self.original_image_gray, canny_lower, canny_upper)
        else:
            edges_prev = self.Arrows_OR

        # add the current arrow to all the edges that are present
        self.stage_images.append(("ALL PREVIOUS EDGES - OR IMAGE", edges_prev))

        #1. Update OR picture (that contains all previous things)
        self.Arrows_OR = cv2.bitwise_or(edges_prev, current_arrow_edges)

        #2. Get the XOR image to isolate arrow edges
        XOR_img = cv2.bitwise_xor(current_arrow_edges, edges_prev)

        self.stage_images.append(("CURRENT ARROW ALONE - XOR IMAGE", XOR_img))


        # Line Detection/Filtering -------------------------------------------------------------------
        # XOR first the unwrap than line detection
        # Right now line detection is performed befor warping the image

        XOR_img_unwrapped = cv2.warpPerspective(XOR_img, self.homography_matrix, (self.unscrewed_image_width, self.unscrewed_image_height))

        lines = cv2.HoughLinesP(XOR_img_unwrapped,                # Returns a wird [ [[x1,x2,y1,y2]], [[x1,x2,y1,y2]] ] array
                                1,                  # Precision of line length ~ 1px
                                np.pi / 180,            # Precision of line angle i degrees pi/180 is 1
                                threshold=50,           # How many white pixels it has to pass trough before calling it a line
                                minLineLength=2,        # Whats the minimal length on the line
                                maxLineGap=10)          # whats the max gap between the line

        lines_img = np.zeros_like(XOR_img_unwrapped)
        if (lines is None):
            print("ERROR: Arrow Line not found")
            return score
        else:
            for line in lines:
                x1, y1, x2, y2 = line[0]                #[ [[x1,x2,y1,y2]], [[x1,x2,y1,y2]] ] because of this line[0]
                cv2.line(lines_img, (x1, y1), (x2, y2), 255, 2)
            print(f"Found {len(lines)} lines")

        self.stage_images.append(("Hough lines", lines_img))

        white_pixels = np.argwhere(lines_img == 255)

        # Evaluation--------------------------
        if len(white_pixels) == 0:
            print("ERROR: Could not find any white pixels!")
            return score

        distance_from_center = np.linalg.norm(white_pixels - center_pixel, axis=1) #euclidian distance from center
        min_distance = np.min(distance_from_center)     #argmin - index vs min - number

        # --- Visualize closest pixel ---
        closest_idx = np.argmin(distance_from_center)
        closest_pixel = white_pixels[closest_idx]  # [y, x]

        visualise = cv2.cvtColor(lines_img, cv2.COLOR_GRAY2BGR)
        cv2.circle(visualise, (closest_pixel[1], closest_pixel[0]), 10, (0, 0, 255), 2)  # green circle
        cv2.circle(visualise, (center_pixel[1], center_pixel[0]), 5, (0, 0, 255), -1)  # red dot = center
        cv2.line(visualise, (center_pixel[1], center_pixel[0]), (closest_pixel[1], closest_pixel[0]), (255, 0, 0),1)  # blue line
        for radius, score in target_specifications:
            cv2.circle(visualise, (center_pixel[1], center_pixel[0]), int (radius*pixel_per_cm), (0, 255, 0), 1)

        self.stage_images.append(("Clossest pixel visualisation", visualise))
        #--------------------------------

        # Pixel closest to the center -------------------------------------------
        for i in range(len(target_specifications)):
            radius_pixels = target_specifications[i][0] * pixel_per_cm
            if min_distance <= radius_pixels:
                score = target_specifications[i][1]
                break

        print("Score: " + str(score))

        return score



    def debug(self):
        for i, (name, img) in enumerate(self.stage_images):
            cv2.imshow(f"{i} - {name}", img)
            cv2.waitKey(0)
