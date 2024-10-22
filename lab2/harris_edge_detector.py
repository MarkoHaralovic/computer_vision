from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage

def convert_to_grayscale(rgb_image):
   assert len(rgb_image.shape) == 3, "Image must be RGB of format (H,W,C)"
   grayscale_image = np.zeros((rgb_image.shape[0],rgb_image.shape[1]))
   grayscale_image = rgb_image.mean(axis=2)
   return grayscale_image

def non_maximum_suppression(image, neighbourhood_size=(3,3),threshold=0):
   assert threshold >=0

   rows, cols = image.shape
   suppressed = np.zeros_like(image)

   hs_x = neighbourhood_size[0] / 2
   hs_y = neighbourhood_size[1] / 2

   for y in range(image.shape[0]):
      for x in range(image.shape[1]):
         if image[y,x] < threshold:
            continue
         y_start = max(0, y - int(np.floor(hs_y)))
         y_end = min(image.shape[0], y + int(np.ceil(hs_y)))
         x_start = max(0, x - int(np.floor(hs_x)))
         x_end = min(image.shape[1], x + int(np.ceil(hs_y)))

         neighbourhood = image[y_start:y_end, x_start:x_end]
         
         if neighbourhood.max() == image[y,x]:
            suppressed[y,x] = image[y,x]
      
   return suppressed

def k_highest_responses(image, k=1):
   assert k > 0, "k must be positive number"

   coords = np.nonzero(image)

   k = min(k,len(coords[0]))

   values = image[coords]

   top_k_ind = np.argsort(values)[-k:]

   top_k_coords = tuple(c[top_k_ind] for c in coords)
   return top_k_coords

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

PARAMS = {
   'fer_logo' : {
      'path' : "./images/fer_logo.jpg",
      'sigma' : 1,
      'threshold' : 1e-10,
      'k' : 0.04,
      'topk'  : 22,
      'sliding_window_sum' : (5,5),
      'sliding_window_nms' : (14,14)
   },
   'house'  : {
      'path' : "./images/house.jpg",
      'sigma' : 1.5,
      'threshold' : 1e-9,
      'k' : 0.04,
      'topk'  : 38,
      'sliding_window_sum' : (5,5),
      'sliding_window_nms' : (32,32)
   }
}
image_name = "house" # "house"
image = np.array(Image.open(PARAMS[image_name]['path']),dtype=np.float32)

if len(image.shape) == 3:
   image = convert_to_grayscale(image)

image_smoothed = scipy.ndimage.gaussian_filter(input = image,
                                                  sigma = PARAMS[image_name]['sigma'])

gradient_x = scipy.ndimage.convolve(input = image_smoothed,
                                    weights=sobel_operator_x
                                    )

gradient_y = scipy.ndimage.convolve(input = image_smoothed,
                                    weights=sobel_operator_y
                                    )

kernel = np.ones((PARAMS[image_name]['sliding_window_sum']))

G_xx = gradient_x**2 
G_xy = gradient_x*gradient_y 
G_yy = gradient_y**2

fig, axes = plt.subplots(2,3)
axes = axes.flatten()

axes[0].imshow(image,cmap='gray')
axes[0].set_title("Original image")
axes[1].imshow(gradient_x,cmap='gray')
axes[1].set_title("G_x")
axes[2].imshow(gradient_y,cmap='gray')
axes[2].set_title("G_y")
axes[3].imshow(G_xx,cmap='gray')
axes[3].set_title("G_xx")
axes[4].imshow(G_yy**2,cmap='gray')
axes[4].set_title("G_yy")
axes[5].imshow(gradient_x*gradient_y,cmap='gray')
axes[5].set_title("G_x * G_y")

plt.savefig(f"./images/interim/harris/{image_name}_gradients.jpg")

Sxx = scipy.ndimage.convolve(G_xx, kernel)
Sxy = scipy.ndimage.convolve(G_xy, kernel)
Syy = scipy.ndimage.convolve(G_yy, kernel)

det_G = (Sxx * Syy) - (Sxy ** 2)
trace_G = Sxx + Syy
harris_response = det_G - PARAMS[image_name]['k'] * (trace_G ** 2)

harris_response_suppressed = non_maximum_suppression(
                        image=harris_response,
                        neighbourhood_size=PARAMS[image_name]['sliding_window_nms'],
                        threshold=PARAMS[image_name]['threshold'])

fig, ax = plt.subplots(1, 1) 
ax.imshow(harris_response_suppressed, cmap='gray')  
ax.set_title("Harris response") 
plt.savefig(f"./images/interim/harris/{image_name}_harris_response.jpg")  

k_highest_res = k_highest_responses(
                        image=harris_response_suppressed,
                        k = PARAMS[image_name]['topk']
                        )

print(len(k_highest_res[0]))
plt.imshow(image_smoothed,cmap='gray')
plt.scatter(k_highest_res[1], k_highest_res[0], edgecolor='red', facecolors='none', s=10)
plt.show()
