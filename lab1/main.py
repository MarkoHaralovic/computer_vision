import matplotlib.pyplot as plt
import scipy.misc as misc
import numpy as np

from interpolation import *

Hd,Wd = 200, 200

Is = misc.face()
Is = np.asarray(Is)

projective = False

if projective:
    Q_s = np.array([
        [[np.random.randint(0, Wd), np.random.randint(0, Hd)] for _ in range(4)],
        [[np.random.randint(0, Wd), np.random.randint(0, Hd)] for _ in range(4)],
        [[np.random.randint(0, Wd), np.random.randint(0, Hd)] for _ in range(4)],
        [[np.random.randint(0, Wd), np.random.randint(0, Hd)] for _ in range(4)]
    ])

    Q_d = np.array([[0,0],[Wd,0],[0,Hd],[Wd,Hd]])

    H_1= recover_projective(Qs=Q_s[0], Qd=Q_d)
    H_2= recover_projective(Qs=Q_s[1], Qd=Q_d)
    H_3= recover_projective(Qs=Q_s[2], Qd=Q_d)
    H_4= recover_projective(Qs=Q_s[3], Qd=Q_d)

    Id1 = projective_nn(source_image=Is,H=H_1,destination_height=Hd,destination_width=Wd)
    Id2 = projective_nn(source_image=Is,H=H_2,destination_height=Hd,destination_width=Wd)
    Id3 = projective_nn(source_image=Is,H=H_3,destination_height=Hd,destination_width=Wd)
    Id4 = projective_nn(source_image=Is,H=H_4,destination_height=Hd,destination_width=Wd)

    images = [Is, Id1,Id2,Id3,Id4]
else:
    A,b = recover_affine_center(Is.shape[0],Is.shape[1], Hd,Wd)
    # A,b= recover_affine_diamond(Is.shape[0],Is.shape[1], Hd,Wd)

    Id1 = affine_nn(Is, A, b, Hd,Wd)
    Id2 = affine_bilin(Is, A, b, Hd, Wd)

    RMSE = rmse(Id1,Id2)
    print(f"RMSE between image reconstructed using nearest neighbour interpolation and bilinear interpolation : {RMSE}")
    
    images = [Is, Id1,Id2]

fig = plt.figure()
if len(Is.shape)==2: plt.gray()
for i,im in enumerate(images):
  fig.add_subplot(1,len(images), i+1)
  plt.imshow(im.astype(int))
plt.show()