from sklearn.cluster import MiniBatchKMeans
#from skimage import color
import scipy
from scipy import ndimage
import warnings; warnings.simplefilter('ignore') 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy import fliplr
import cv2


#class ImageEffects:
def sharpen(f):
    #f = scipy.misc.face(gray=True).astype(float)
    blurred_f = ndimage.gaussian_filter(f, 3)
    filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)

    alpha = 30
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)

    plt.figure(figsize=(12, 4))
  
    plt.subplot(131)
    plt.imshow(f, cmap=plt.cm.gray)
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(blurred_f, cmap=plt.cm.gray)
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(sharpened, cmap=plt.cm.gray)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def invert(img):
    ## imvert colors
    img = 255 - img
    ax = plt.axes(xticks=[], yticks=[])
    ax.imshow(img)
    plt.show()
# end invert colors

def shift(val, img):
    ## imvert colors
    img = abs(img - val)
    ax = plt.axes(xticks=[], yticks=[])
    ax.imshow(img)
    plt.show()
# end invert colors

def cool_scale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ax = plt.axes(xticks=[], yticks=[])
    ax.imshow(gray_img)
    plt.show()

def grayscale(img):
    img = np.dot(img[...,:3],[.2989, .5870, .1140])
    ax = plt.axes(xticks=[], yticks=[])
    ax.imshow(img)
    plt.show()

def render_image(img):
    ax = plt.axes(xticks=[], yticks=[])
    ax.imshow(img)
    plt.show()  

def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    
    # choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20)
    plt.show()
def blackWhite(c, threshold):
    img = c.copy()
    rd = 255
    gd = 255
    bd = 255
    rl = 0
    gl = 0
    bl = 0
    for i in img:
        for a in i:
            c = (a[0] + a[1] + a[2]) / 3
            #print(c)
            if c > threshold:
                a[0] = rd
                a[1] = gd
                a[2] = bd
            else:
                a[0]=rl
                a[1]=gl
                a[2]=bl
    render_image(img)
 
##--------------------------------------------------------------------------------------##
#def main():
#
#if __name__ == "__main__":
#main()


a='IMG_8868.JPG'
b='IMG_8870.JPG'
c='IMG_8869.JPG'
d='IMG_8855.JPG'
e='IMG_8871.JPG'
A = e
val = 60

img = mpimg.imread('/root/Downloads/' + A)
img2 = mpimg.imread('/root/Downloads/' + A)

print(img)
print("-----")
print(img2)

blackWhite(img, 50) # black and white but changes it for all following
	
img2 = np.fliplr(img)
render_image(img[::-1])# flip upside down
render_image(np.fliplr(img)) ## reverse image
shift(40, img)
shift(80, img)
shift(117, img)
shift(231, img)
shift(180, img)
shift(100, img)
sharpen(img)
grayscale(img)
cool_scale(img)
invert(img)
x = img.shape[0]
y  = img.shape[1]
#img2 = img2.resize(x, y)
render_image(img2)
render_image(img)
render_image(img+img2)
render_image(img-img2)

# here below is 0 to 1 scale
data = img / 255.0 # use 0...1 scalepip ls

print(img.shape)

data = data.reshape(img.shape[0] * img.shape[1], 3)
data.shape

## below here render out in NUM_COLORS colors
 
for NUM_COLORS in range(13, 50):
	#plot_pixels(data, title='Input color space: 16 million possible colors')

	kmeans = MiniBatchKMeans(NUM_COLORS)
	kmeans.fit(data)
	new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

	s = str(NUM_COLORS) + '-color Image'
	china_recolored = new_colors.reshape(img.shape)

	fig, ax = plt.subplots(1, 2, figsize=(16, 6),
		               subplot_kw=dict(xticks=[], yticks=[]))
	fig.subplots_adjust(wspace=0.01)
	ax[0].imshow(img)
	ax[0].set_title('Original Image', size=16)
	ax[1].imshow(china_recolored)
	ax[1].set_title(s, size=NUM_COLORS)
	plt.show()
