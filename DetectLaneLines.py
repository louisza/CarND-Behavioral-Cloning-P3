#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from scipy import stats


image = mpimg.imread('./data/IMG/center_2016_12_01_13_32_57_507.jpg')


#reading in an image
#printing out some stats and plotting
#print('This image is:', type(image), 'with dimesions:', image.shape)
#plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

# plt.imshow(image)
# newimg = image
#
# newimg = grayscale(newimg)
# plt.imshow(newimg,cmap='gray')
#
# newimg = gaussian_blur(newimg, 5)
# plt.imshow(newimg,cmap='gray')
#
# newimg = canny(newimg,120,150)
# plt.imshow(newimg,cmap='gray')
#
# imshape = newimg.shape
# top_line_perc = 0.25
# vertices = np.array(
#     [[(0, imshape[0]/3), ((imshape[1] / 2 ), imshape[0] / 4 ),
#       (imshape[1], imshape[0]/3),(imshape[1],imshape[0]),
#       ((imshape[1] / 2 ), imshape[0] / 2 ),
#       (0,imshape[0])
#       ]], dtype=np.int32)
# newimg = region_of_interest(newimg, vertices)
#
# plt.imshow(newimg,cmap='gray')
#
#
# rho = 2  # distance resolution in pixels of the Hough grid
# theta = np.pi / 180  # angular resolution in radians of the Hough grid
# threshold = 18  # minimum number of votes (intersections in Hough grid cell)
# min_line_length = 30 #minimum number of pixels making up a line
# max_line_gap = 15  # maximum gap in pixels between connectable line segments
#
# # Run Hough on edge detected image
# # Output "lines" is an array containing endpoints of detected line segments
# lin_img = hough_lines(newimg, rho, theta, threshold,
#                       min_line_length, max_line_gap)
# plt.imshow(lin_img,cmap='gray')

#ToDO consider using stats.linear regression

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    #for line in lines:
     #   for x1, y1, x2, y2 in line:
      #      cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    neg_slope_dist = 0
    negslope_avg = 0
    negslope_count = 0

    pos_slope_dist = 0
    posslope_avg = 0
    posslope_count = 0

    leftx = []
    lefty = []
    rightx = []
    righty = []

    imshape = img.shape
    top_line_perc = -0.1

    #topline =  imshape[0] / 3

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = round(((y2 - y1) / (x2 - x1)), 2)
            distance = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
            #print("Line: ", line, " Slope=", slope, " Distance = ", distance)
            if slope < -0.4 and slope > -0.8:
                neg_slope_dist += distance
                negslope_avg += slope * distance
                negslope_count += 1
                rightx.append(x1)
                rightx.append(x2)
                righty.append(y1)
                righty.append(y2)
            if slope > 0.4 and slope < 0.8:
                pos_slope_dist += distance
                posslope_avg += slope * distance
                posslope_count += 1
                leftx.append(x1)
                leftx.append(x2)
                lefty.append(y1)
                lefty.append(y2)

    if pos_slope_dist == 0:
        pos_slope_dist = 1
    if neg_slope_dist == 0:
        neg_slope_dist = 1

    neg_slope = negslope_avg / (neg_slope_dist)
    pos_slope = posslope_avg / (pos_slope_dist)
    if len(rightx) > 0 and len(leftx) > 0:
        slope_r, intercept_r, r_value_r, p_value_r, std_err_r = stats.linregress(rightx, righty)

        rx1 = int((imshape[0]-intercept_r)/slope_r)
        rx2 = int((imshape[0] / 2 + (imshape[0] * top_line_perc)-intercept_r)/slope_r)
        ry1 = int(slope_r*rx1+intercept_r)
        ry2 = int(slope_r*rx2+intercept_r)

        slope_l, intercept_l, r_value_l, p_value_l, std_err_l = stats.linregress(leftx, lefty)

        lx1 = int((imshape[0] - intercept_l) / slope_l)
        lx2 = int((imshape[0] / 2 + (imshape[0] * top_line_perc) - intercept_l) / slope_l)
        ly1 = int(slope_l * lx1 + intercept_l)
        ly2 = int(slope_l * lx2 + intercept_l)

        cv2.line(img, (rx1, ry1), (rx2, ry2), color, thickness)
        cv2.line(img, (lx1, ly1), (lx2, ly2), color, thickness)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    #for line in lines:
     #  for x1, y1, x2, y2 in line:
      #    cv2.line(line_img, (x1, y1), (x2, y2), [255, 0, 0], 2)

    draw_lines(line_img, lines, thickness= 1)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)




# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
def process_image(image):
    grayimg = grayscale(image)
    blur_gray = gaussian_blur(grayimg, 5)
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    imshape = image.shape
    # top_line_perc = 0.08
    vertices = np.array(
        [[(0, imshape[0] / 3), ((imshape[1] / 2), imshape[0] / 4),
          (imshape[1], imshape[0] / 3), (imshape[1], imshape[0]),
          ((imshape[1] / 2), imshape[0] / 2),
          (0, imshape[0])
          ]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 18  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 25  # minimum number of pixels making up a line
    max_line_gap = 30  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lin_img = hough_lines(masked_edges, rho, theta, threshold,
                            min_line_length, max_line_gap)

   # plt.imshow(lin_img)
   # plt.show()

    #print(len(lines), " lines detected")

    # Iterate over the output "lines" and draw lines on a blank image
    #for line in lines:
    #    for x1, y1, x2, y2 in line:
    #        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

            # print(lines)



#    print("Avg Slopes : ", neg_slope,pos_slope)


                # Create a "color" binary image to combine with line image
    #color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
   # lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

    w_img = weighted_img(lin_img, image, α=0.8, β=1., λ=0.)
    #wimg_BGR = cv2.cvtColor(wimg, cv2.COLOR_RGB2BGR)

    return w_img




