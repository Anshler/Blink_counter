import csv
import numpy as np
from PIL import Image
#convert csv fole to image
f=open('dataset.csv','r')
f=csv.reader(f)
countO=0
countC=0
for a in f:
    if a[0] != 'state':
        a[1]=a[1].replace("[","")
        a[1] = a[1].replace("]", "")
        a[1] = a[1].split(', ')
        print(a[1])
        img = np.array(a[1], dtype=np.uint8)
        img = img.reshape((26, 34))
        img = np.expand_dims(img, axis=2)
        # Use PIL to create an image from the new array of pixels
        new_image = Image.fromarray(img)
        if a[0] == 'open':
            new_image.save('dataset\\open{}.png'.format(countO))
            countO+=1
        else:
            new_image.save('dataset\\close{}.png'.format(countC))
            countC += 1

