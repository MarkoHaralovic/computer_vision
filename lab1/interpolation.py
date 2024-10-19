import numpy as np

def recover_affine_center(source_h, source_w, destination_h, destination_w):
   A = 0.25*np.eye(2) + np.random.normal(size=(2, 2))
   source_center = np.array([source_w // 2, source_h // 2])
   destination_center = np.array([destination_w // 2, destination_h // 2])
   b = source_center - A @ destination_center 
   return A, b

def recover_affine_diamond(source_h, source_w, destination_h, destination_w):
    x1, y1 = 0, 0  
    x2, y2 = destination_w - 1 , 0             
    x3, y3 = destination_w - 1 , destination_h - 1

    x1_d, y1_d = 0, np.floor((source_h -1) // 2)
    x2_d, y2_d = np.floor((source_w -1) // 2) , 0       
    x3_d, y3_d = (source_w - 1), np.floor((source_h -1 ) // 2 ) 

    S = np.array(
        [[x1, y1, 0, 0, 1, 0],    
        [0, 0, x1, y1, 0, 1],
        [x2, y2, 0, 0, 1, 0],
        [0, 0, x2, y2, 0, 1],
        [x3, y3, 0, 0, 1, 0],
        [0, 0, x3, y3, 0, 1]]
    )

    x = np.array([x1_d, y1_d, x2_d, y2_d, x3_d, y3_d])

    t = np.linalg.solve(S,x)

    A = np.array([
        [t[0], t[1]],
        [t[2], t[3]]
    ])
    b = np.array([t[4], t[5]])

    return A, b

def recover_projective(Qs, Qd):
    x1, y1 = Qs[0]
    x2, y2 = Qs[1]
    x3, y3 = Qs[2]
    x4, y4 = Qs[3]

    x1_d, y1_d = Qd[0]
    x2_d, y2_d = Qd[1]
    x3_d, y3_d = Qd[2]
    x4_d, y4_d = Qd[3]

    S = np.array([
        [-x1, -y1, -1, 0, 0, 0, x1*x1_d, y1*x1_d, -x1_d],
        [0, 0, 0, -x1, -y1, -1, x1*y1_d, y1*y1_d, -y1_d],
        [-x2, -y2, -1, 0, 0, 0, x2*x2_d, y2*x2_d, -x2_d],
        [0, 0, 0, -x2, -y2, -1, x2*y2_d, y2*y2_d, -y2_d],
        [-x3, -y3, -1, 0, 0, 0, x3*x3_d, y3*x3_d, -x3_d],
        [0, 0, 0, -x3, -y3, -1, x3*y3_d, y3*y3_d, -y3_d],
        [-x4, -y4, -1, 0, 0, 0, x4*x4_d, y4*x4_d, -x4_d],
        [0, 0, 0, -x4, -y4, -1, x4*y4_d, y4*y4_d, -y4_d],
    ])

    _, _ , V = np.linalg.svd(S)

    h = (V[-1] / V[-1][-1]).reshape(3, 3)
    
    return h

def affine_nn(source_image, A, b, destination_width, destination_height):
    if len(source_image.shape) == 3:
        destination_image = np.zeros((destination_height, destination_width, source_image.shape[2]), dtype=source_image.dtype)
    else:
        destination_image = np.zeros((destination_height, destination_width), dtype=source_image.dtype)

    for y in range(destination_height):
        for x in range(destination_width):
            destination_coords = np.array([x,y])
            source_coords = A @ destination_coords + b

            src_x, src_y = int(np.round(source_coords[0])), int(np.round(source_coords[1]))

            if 0 <= src_x < source_image.shape[1] and 0 <= src_y < source_image.shape[0]:
                if len(source_image.shape) == 2:
                    destination_image[y, x] = source_image[src_y, src_x]
                else:
                    destination_image[y, x, :] = source_image[src_y, src_x, :]

    return destination_image

def affine_bilin(source_image, A, b, destination_width, destination_height):
    if len(source_image.shape) == 3:
        destination_image = np.zeros((destination_height, destination_width, source_image.shape[2]), dtype=source_image.dtype)
    else:
        destination_image = np.zeros((destination_height, destination_width), dtype=source_image.dtype)

    for y in range(destination_height):
        for x in range(destination_width):
            destination_coords = np.array([x, y])
            source_coords = A @ destination_coords + b

            src_x, src_y = source_coords[0], source_coords[1]
            
            if 0 <= src_x < source_image.shape[1] and 0 <= src_y < source_image.shape[0]:
                if len(source_image.shape) == 2:
                    destination_image[y, x] = source_image[int(src_y), int(src_x)]
                else:
                    destination_image[y, x, :] = source_image[int(src_y), int(src_x), :]
            else:
                x1, x2 = min(max(0, int(np.floor(src_x))),source_image.shape[1]-1), max(min(source_image.shape[1] - 1, int(np.ceil(src_x))),-source_image.shape[1])
                y1, y2 = min(max(0, int(np.floor(src_y))),source_image.shape[0]-1), max(min(source_image.shape[0] - 1, int(np.ceil(src_y))),-source_image.shape[0])
                
                if x1 != x2 and y1 != y2 and 0 <= x1 < source_image.shape[1] and 0 <= y1 < source_image.shape[0] and 0 <= x2 < source_image.shape[1] and 0 <= y2 < source_image.shape[0]:
                    fx1 = (x2 - src_x) / (x2 - x1)
                    fx2 = (src_x - x1) / (x2 - x1)
                    fy1 = (y2 - src_y) / (y2 - y1)
                    fy2 = (src_y - y1) / (y2 - y1)

                    if len(source_image.shape) == 2:
                        destination_image[y, x] = (fx1 * fy1 * source_image[y1, x1] +
                                                   fx1 * fy2 * source_image[y2, x1] +
                                                   fx2 * fy1 * source_image[y1, x2] +
                                                   fx2 * fy2 * source_image[y2, x2])
                    else:
                        print(x1,x2,y1,y2)
                        destination_image[y, x, :] = (fx1 * fy1 * source_image[y1, x1, :] +
                                                      fx1 * fy2 * source_image[y2, x1, :] +
                                                      fx2 * fy1 * source_image[y1, x2, :] +
                                                      fx2 * fy2 * source_image[y2, x2, :])
    return destination_image

def projective_nn(source_image, H, destination_width, destination_height):
    if len(source_image.shape) == 3:
        destination_image = np.zeros((destination_height, destination_width, source_image.shape[2]), dtype=source_image.dtype)
    else:
        destination_image = np.zeros((destination_height, destination_width), dtype=source_image.dtype)
    
    for y in range(destination_height):
        for x in range(destination_width):
            destination_coords = np.array([x,y,1])

            source_coords_homogeneous = H @ destination_coords

            if source_coords_homogeneous[2] != 0 :
                src_x = source_coords_homogeneous[0] / source_coords_homogeneous[2]
                src_y = source_coords_homogeneous[1] / source_coords_homogeneous[2]
            else:
                continue

            src_x = int(np.round(src_x))
            src_y = int(np.round(src_y))

            if 0 <= src_x < source_image.shape[1] and 0 <= src_y < source_image.shape[0]:
                if len(source_image.shape) == 2: 
                    destination_image[y, x] = source_image[src_y, src_x]
                else:  
                    destination_image[y, x, :] = source_image[src_y, src_x, :]

    return destination_image

def rmse(src_img, dst_img):
    assert src_img.shape == dst_img.shape
    
    se = (src_img - dst_img) ** 2
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    return rmse