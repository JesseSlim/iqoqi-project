import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

spotA_image = Image.open("spotA.png")
spotB_image = Image.open("spotB.png")

spotA_imgarray = np.array(spotA_image)
spotB_imgarray = np.array(spotB_image)

spotA_green = spotA_imgarray[:,:,1]
spotB_green = spotB_imgarray[:,:,1]



spotA_max_val = np.max(spotA_green)
spotA_max_idx = np.flatnonzero(spotA_green == spotA_max_val)

spotB_max_val = np.max(spotB_green)
spotB_max_idx = np.flatnonzero(spotB_green == spotB_max_val)

spotA_max_pos = np.unravel_index(spotA_max_idx, spotA_green.shape)
spotB_max_pos = np.unravel_index(spotB_max_idx, spotB_green.shape)

spotA_center = np.mean(spotA_max_pos, axis=1)
spotB_center = np.mean(spotB_max_pos, axis=1)

print(spotA_center)
print(spotB_center)


plt.imshow(spotA_green)
plt.scatter(spotA_center[1], spotA_center[0])
plt.show()

plt.imshow(spotB_green)
plt.scatter(spotB_center[1], spotB_center[0])
plt.show()