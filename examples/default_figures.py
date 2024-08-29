import numpy as np
import matplotlib.pyplot as plt
import trevorarp as tb

fi = tb.display.figure_inches("Default Title","2","2")
ax1 = fi.make_axes()
ax2, ax2cb = fi.make_img_axes()
ax3 = fi.make_axes()
ax4 = fi.make_axes()

plt.suptitle("Default Figures Format")
fi.stamp("unique identifier(s)")

# Random test data
np.random.seed(42)
data = np.zeros((100,100))
rows, cols = data.shape
for i in range(rows):
    data[i,:] = 0.25*np.random.rand(100) + np.linspace(0,1,100)

for i in range(3):
    ax1.plot(data[i,:]+i)
ax1.set_xlabel("X Axis Label")
ax1.set_ylabel("Y Axis Label")
ax1.set_title("Title")

cmap, cnorm, smap = tb.display.colorscale_map(data, mapname='inferno')
tb.display.make_colorbar(ax2cb, cmap, cnorm)
ax2.imshow(data, cmap=cmap, norm=cnorm)
ax2.set_xlabel("X Axis Label")
ax2.set_ylabel("Y Axis Label")
ax2.set_title("Title", loc='left')

for i in range(3):
    ax3.plot(data[i+3,::-1]+i)
ax3.set_xlabel("X Axis Label")
ax3.set_ylabel("Y Axis Label")
ax3.set_title("Title")

for i in range(4):
    if i % 2 == 0:
        ax4.plot(data[i+3,:]+0.25*i)
    else:
        ax4.plot(data[i+3,::-1]+0.25*i)
ax4.set_xlabel("X Axis Label")
ax4.set_ylabel("Y Axis Label")
ax4.set_title("Title")

plt.show()
