
import cv2
import numpy as np
import os


class TLE:
    def __init__(self, image_name):
        #self.__image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        path = os.path.join(os.path.dirname(__file__), 'test images')
        self.__image = cv2.imread(os.path.join(path, image_name), cv2.IMREAD_GRAYSCALE)
        if self.__image is None:
            raise NameError("Can't find image: \"" + image_name + '\"')

        self.__img_name = image_name
        self.__threshed = None

        row_num, col_num = self.__image.shape

        self.__flat = []
        self.__is_text_row = {x: x * 0 for x in range(0, row_num)}
        self.__first_text_row = -1
        self.__last_text_row = -1
        self.__contours = []
        self.__res = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)


    # input: point array represents polygon to be tighten * tight level of 2^level -> point array represents more thight polygon
    def tighten_polygon (self, polygon, level):

        self.__prepare_tighten(polygon)

        #check if there is a line to be tighten
        if self.__first_text_row == -1:
            return []

        self.__devide_line_contour(level)

        # now we have "dirty" and more accurate draw in Improved.png. This code make it cleaner
        _, new_threshed = cv2.threshold(self.__res, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # find contours and reverse the array for chronological row order
        _, improved_contours, _ = cv2.findContours(self.__threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        improved_contours = improved_contours[::-1]

        return self.__get_tighten_contour(improved_contours)


    def __draw_num(self, cont, i):
        cv2.drawContours(self.__res, cont, i, (100, 100, 100), 5)
        self.show_image("draw num " + str(i), self.__res)
        return

    def __find_next_line_row(self, points_sortByRow, last_line):
        for point in points_sortByRow:
            if point[1] > last_line:
                return point[1]

        return -1


    def __find_next_blank_row(self, img, curr_row):
        row_num, col_num = img.shape

        if curr_row >= row_num:  # Illegal row number
            return -1

        for x in range(curr_row, row_num):
            if self.__is_text_row[x] == 0:
                return x

        return -1

    def __find_last_blank_row(self, img, curr_row):
        row_num, col_num = img.shape

        i = row_num - 1
        while i > curr_row:
            if self.__is_text_row[i] == 1:
                return i+1
            i -= 1

        return -1


    def __prepare_tighten(self, polygon):
        self.__res = cv2.imread(self.__img_name, cv2.IMREAD_GRAYSCALE)

        # apply Gaussian filter
        blur = self.__image
        blur = cv2.GaussianBlur(blur, (5, 5), 0)

        # threshold
        th, self.__threshed = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        self.__threshed = self.__mask_image(polygon)
        #self.show_image("threshed after mask", self.__threshed)

        pts = cv2.findNonZero(self.__threshed)
        #return empty array if there are no points
        if pts is None:
            return []

        # after the loop, flat is sorted by row, when row is the right number at each array
        self.__flat = []
        for sublist in pts:
            for item in sublist:
                self.__flat.append(item)

        # initial dictionary
        row_num, col_num = self.__image.shape
        self.__is_text_row = {x: x * 0 for x in range(0, row_num)}
        for point in self.__flat:
            self.__is_text_row[point[1]] = 1

        self.__first_text_row = self.__find_next_line_row(self.__flat, 0)  # first text row
        self.__last_text_row = self.__find_last_blank_row(self.__threshed, self.__first_text_row)

        # find contours and reverse the array for chronological row order
        _, self.__contours, _ = cv2.findContours(self.__threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.__contours = self.__contours[::-1]

        return


    #filter only the relevant image's polygon
    def __mask_image(self, polygon):
        mask = np.zeros(self.__image.shape, dtype=np.uint8)
        hull = [cv2.convexHull(np.array(polygon, np.int32))]
        cv2.drawContours(mask, hull, -1, (255, 255, 255), -1)
        #self.show_image("mask1", mask)

        bit_and = cv2.bitwise_and(self.__threshed, mask)

        return bit_and


    # devide line contour to 2^precision parts, and add convex hull for each part
    def __devide_line_contour(self, precision):
        col_range = self.__find_col_text_range(self.__first_text_row, self.__last_text_row)

        row_length = col_range[1] - col_range[0] + 1
        part_width = round(row_length / pow(2, precision))

        low_col = col_range[0]
        high_col = low_col + part_width
        counter = 0
        prev_right = None
        while counter < pow(2, precision):
            partial = self.__get_partial_line_contour(self.__first_text_row, self.__last_text_row, [low_col, high_col])

            if partial is not None:
                self.__add_convexHull(self.__threshed, partial)

                if prev_right is None:
                    prev_right = self.__find_linking_points("right", -1, partial)
                else:
                    linking_points = self.__find_linking_points("left", 1000000, partial)
                    # don't draw line if the parts are overlap
                    if prev_right[0][0] < linking_points[0][0] and prev_right[1][0] < linking_points[1][0]:
                        cv2.line(self.__threshed, prev_right[0], linking_points[0], (255, 255, 255), 1)
                        cv2.line(self.__threshed, prev_right[1], linking_points[1], (255, 255, 255), 1)
                    prev_right = self.__find_linking_points("right", -1, partial)

            low_col = high_col + 1
            high_col += part_width
            if high_col > col_range[1]:
                high_col = col_range[1]
            counter += 1

        #self.show_image("Dirty improved result", self.__threshed)

        return

    def __get_row_edges(self, row):
        left = -1
        right = -1

        for point in self.__flat:
            if point[1] == row and left == -1:
                left = point[0]
            elif point[1] == row:
                right = point[0]
            elif point[1] > row:
                return [left, right]

        return [left, right]

    def __find_col_text_range(self, first_row, last_row):
        most_left = 1000000
        most_right = -1

        if first_row == -1 or last_row == -1:
            return [-1, -1]

        for row in range(first_row, last_row):
            edges = self.__get_row_edges(row)
            if most_left > edges[0] > -1:
                most_left = edges[0]
            if most_right < edges[1]:
                most_right = edges[1]

        return [most_left, most_right]


    def __get_partial_line_contour(self, first_row, last_row, col_range):
        partial_line_contour = None

        for x in self.__contours:
            found = 0
            if first_row <= x[0][0][1] <= last_row:
                for point in x:
                    if col_range[0] <= point[0][0] <= col_range[1]:
                        found = 1
                        break
                if found == 1:
                    if partial_line_contour is None:
                        partial_line_contour = x
                    else:
                        partial_line_contour = np.concatenate((partial_line_contour, x))
            elif x[0][0][1] > last_row:
                break

        return partial_line_contour

    def __get_contour_middle(self, contour):
        low_row = 1000000
        high_row = -1
        for point in contour:
            if point[0][1] > high_row:
                high_row = point[0][1]
            elif point[0][1] < low_row:
                low_row = point[0][1]

        return round((high_row - low_row) / 2) + low_row


    # side is "right" if the contour is from the left contour, and "left" for right contour
    def __find_linking_points(self, side, initial_value, contour):
        col = initial_value
        upper_limit = 1000000
        bottom_limit = 0
        high_row = -1
        low_row = 1000000
        middle_row = self.__get_contour_middle(contour)

        if contour is None:
            return

        # print("side:", side, "init:",initial_value, "cont:", contour)
        while True:
            if side == "right":
                for point in contour:
                    if col < point[0][0] <= upper_limit:
                        col = point[0][0]
            elif side == "left":
                for point in contour:
                    if bottom_limit <= point[0][0] < col:
                        col = point[0][0]

            if low_row > middle_row:
                for point in contour:
                    if point[0][0] == col:
                        if point[0][1] < low_row and point[0][1] < middle_row:
                            low_row = point[0][1]
            if high_row < middle_row:
                for point in contour:
                    if point[0][0] == col:
                        if point[0][1] > high_row and point[0][1] > middle_row:
                            high_row = point[0][1]

            if low_row == high_row or low_row > middle_row or high_row < middle_row:
                if side == "right":
                    upper_limit = col - 1
                    col = -1
                else:
                    bottom_limit = col + 1
                    col = 1000000
            else:
                break

        return [(col, low_row), (col, high_row)]


    # return contour length (number of points)
    def __get_length(self, arr):
        return len(arr.tolist())


    # return the distance btween contour's column
    def __get_contour_width(self, contour):
        low_col = 1000000
        high_col = -1
        for point in contour:
            if point[0][0] > high_col:
                high_col = point[0][0]
            elif point[0][0] < low_col:
                low_col = point[0][0]

        return high_col - low_col


    # return the tighten contour (the contour with the maximal width)
    def __get_tighten_contour(self, contours):
        widest_cont = []
        max_width = -1

        for cont in contours:
            cont_width = self.__get_contour_width(cont)
            if cont_width > max_width:
                widest_cont = cont
                max_width = cont_width

        return widest_cont


    def show_image(self, name, image):
        if image is None:
            return

        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def __add_convexHull(self, image, cont):
        hull = [cv2.convexHull(cont)]
        cv2.drawContours(image, hull, -1, (100, 100, 100), 1)
        #self.show_image("after add", image)

        return




test = TLE("test.png")

polygn1 = [[70, 50], [400, 50], [60, 190], [250, 190]]
polygn2 = [[70, 130], [400, 130], [60, 250], [250, 250]]
polygn3 = [[30, 125], [650, 125], [30, 155], [650, 155]]
polygn4 = [[70, 50], [330, 50], [60, 190], [320, 190], [300, 350]]
polygn5 = [[30, 125], [650, 125], [30, 193], [650, 193]]

img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
res1 = test.tighten_polygon(polygn1, 2)
res2 = test.tighten_polygon(polygn2, 3)
res3 = test.tighten_polygon(polygn3, 4)
res4 = test.tighten_polygon(polygn4, 4)
res5 = test.tighten_polygon(polygn5, 1)
cv2.drawContours(img, [res3], -1, (100, 100, 100), 2)
test.show_image("testing!!", img)

#print("res1 ", res1, "res1 list ", res1.tolist())




