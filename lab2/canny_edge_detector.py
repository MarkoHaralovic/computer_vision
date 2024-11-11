from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage

def non_maximum_suppresion(gradient_image, angles):
   assert gradient_image.shape == angles.shape

   suppressed_image = np.zeros_like(gradient_image)

   for y in range(gradient_image.shape[0]):
      for x in range(gradient_image.shape[1]):
         angle = angles[y,x]
         angle = abs(angle % 180)
         
         if 0 <= angle < 22.5:
            neighbours = [[0,-1],[0,1]] 
         elif 22.5 <= angle < 67.5:
            neighbours = [[-1,+1],[1,-1]] 
         elif 67.5 <= angle < 112.5:
            neighbours = [[-1,0],[1,0]] 
         elif 112.5 <= angle < 157.5:
            neighbours = [[-1,-1],[1,1]]  
         else:
            neighbours = [[0,-1], [0,1]] 

         g = gradient_image[y, x]
         y1, x1 = max(min(y + neighbours[0][0], gradient_image.shape[0] - 1), 0), max(min(x + neighbours[0][1], gradient_image.shape[1] - 1), 0)
         y2, x2 = max(min(y + neighbours[1][0], gradient_image.shape[0] - 1), 0), max(min(x + neighbours[1][1], gradient_image.shape[1] - 1), 0)

         g1 = gradient_image[y1, x1]
         g2 = gradient_image[y2, x2]

         if g >= g1 and g >= g2:
                suppressed_image[y, x] = g
 
   return suppressed_image

def detect_edges(gradient_image, upper_threshold, lower_threshold, binarize=False):
   image_w_strong_edges = np.zeros_like(gradient_image)
   image_with_weak_edges = np.zeros_like(gradient_image)

   if binarize:
      image_w_strong_edges = np.where(gradient_image >= upper_threshold, 255, 0)
   else:
      image_w_strong_edges = np.where(gradient_image >= upper_threshold, gradient_image, 0)

   kernel = np.ones((3,3))
   
   strong_neighbours = scipy.ndimage.convolve((image_w_strong_edges > 0).astype(int), kernel, mode='constant')

   weak_edges_mask = np.logical_and(gradient_image > lower_threshold, gradient_image < upper_threshold)

   has_strong_neigbour = strong_neighbours > 0

   if binarize:
      image_with_weak_edges = np.where(np.logical_and(weak_edges_mask, has_strong_neigbour), 255, 0)
   else:
      image_with_weak_edges = np.where(np.logical_and(weak_edges_mask, has_strong_neigbour), gradient_image, 0)

   return image_w_strong_edges, image_with_weak_edges

PARAMS = {
   'fer_logo' : {
      'path' : "./images/fer_logo.jpg",
      'sigma' : 1.5,
      'min_val' : 10,
      'max_val' : 90
   },
   'house'  : {
      'path' : "./images/house.jpg",
      'sigma' : 1.5,
      'min_val' : 10,
      'max_val' : 90
   }
}

sobel_operator_x = np.array([
   [-1, 0, 1],
   [-2, 0, 2],
   [-1, 0, 1]
])

sobel_operator_y = np.array([
   [1, 2, 1],
   [0, 0, 0],
   [-1, -2, -1]
])

image_name = "house"
image = np.array(Image.open(PARAMS[image_name]['path']),dtype=np.float32)

image_smoothed = scipy.ndimage.gaussian_filter(input = image,
                                                  sigma = PARAMS[image_name]['sigma'])

gradient_x = scipy.ndimage.convolve(input = image_smoothed,
                                    weights=sobel_operator_x
                                    )

gradient_y = scipy.ndimage.convolve(input = image_smoothed,
                                    weights=sobel_operator_y
                                    )

G = np.sqrt(gradient_x**2 + gradient_y**2)
G = (G / G.max()) * 255

phi = np.degrees(np.arctan2(gradient_y, gradient_x))

fig, ax = plt.subplots(1,1)
ax.imshow(G,cmap="gray")
ax.set_title("Normalized values of G")
plt.savefig(f"./images/interim/canny/{image_name}/gradient.jpg")

suppressed_image = non_maximum_suppresion(G,phi)

plt.imshow(suppressed_image,cmap="gray")
ax.set_title("G after nms")
plt.savefig(f"./images/interim/canny/{image_name}/suppressed.jpg")


img_strong_edges, img_weak_edges = detect_edges(gradient_image=suppressed_image, 
                                                lower_threshold= PARAMS[image_name]['min_val'],
                                                upper_threshold= PARAMS[image_name]['max_val'],
                                                binarize= True
                                                )

plt.imshow(img_strong_edges,cmap="gray")
ax.set_title("Strong edges")
plt.savefig(f"./images/interim/canny/{image_name}/strong_edges.jpg")

plt.imshow(img_weak_edges,cmap="gray")
ax.set_title("Weak edges")
plt.savefig(f"./images/interim/canny/{image_name}/weak_edges.jpg")

plt.imshow(img_strong_edges + img_weak_edges,cmap="gray")
ax.set_title("Strong + weak edges")
plt.savefig(f"./images/interim/canny/{image_name}/all_edges.jpg")

plt.show()
