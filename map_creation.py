import numpy as np
import matplotlib.pyplot as plt

def interpolate(a0, a1, w):
    #w in [0,1]
    # Smootherstep
    return (a1 - a0) * ((w * (w * 6.0 - 15.0) + 10.0) * w * w * w) + a0

def dotGridGrad(ix, iy, x, y, grad):
    dx = x-ix
    dy = y-iy
    return dx*grad[0] + dy*grad[1]

def perlin(x, y, res, seed):
    unit = 1 / np.sqrt(2)  # diagonale
    grad = np.array([[unit, unit], [-unit, unit], [unit, -unit], [-unit, -unit], [1, 0], [-1, 0], [0, 1], [0, -1]])
    np.random.seed(seed)
    perm = np.arange(256, dtype=int)
    np.random.shuffle(perm)
    x /= res
    y /= res
    #determine grid cell coord
    x0, x1 = int(x), int(x) + 1
    y0, y1 = int(y), int(y) + 1
    #interpolation weights
    sx = x - x0
    sy = y - y0
    #compute random grad
    ii, jj = x0 % 255, y0 % 255
    g0 = perm[(ii + perm[jj]) % 255] % 8
    g1 = perm[(ii + 1 + perm[jj]) % 255] % 8
    g2 = perm[(ii + perm[jj + 1]) % 255] % 8
    g3 = perm[(ii + 1 + perm[jj + 1]) % 255] % 8

    n0 = dotGridGrad(x0, y0, x, y, grad[g0])
    n1 = dotGridGrad(x1, y0, x, y, grad[g1])
    ix0 = interpolate(n0, n1, sx)

    n0 = dotGridGrad(x0, y1, x, y, grad[g2])
    n1 = dotGridGrad(x1, y1, x, y, grad[g3])
    ix1 = interpolate(n0, n1, sx)

    value = interpolate(ix0, ix1, sy)
    return value

def compute():

    min, max = -1,1

    grid = np.zeros([size, size])

    for x in range(size):
        for y in range(size):
            grid[x, y] = perlin(x, y, scale, seed)
            coef = 1
            k = scale / 2
            while k >= 1:
                k /= 2
                grid[x, y] += perlin(x, y, k, seed) * coef
                coef /= 2  # l'importance diminue quand k diminue
            if min > grid[x, y]:
                grid[x, y] = min
            if max < grid[x, y]:
                grid[x, y] = max
    map = np.zeros([size, size, 3])
    for x in range(size):
        for y in range(size):
            value = grid[x, y]
            if value < -0.2:
                bleu = 219 * value + 299  # 80 = f(-1) = -a+b ; 255 = f(-0.2) = -0.2a+b
                map[x, y] = (0, 70, bleu)
            elif value < -0.15:
                map[x, y] = (194, 178, 128)

            elif value < 0.2:
                red = 80 * value + 76
                green = -280 * value + 186
                blue = 50 * value + 18
                if red > 255:
                    red = 255
                if green > 255:
                    green = 255
                if blue > 255:
                    blue = 255
                map[x, y] = (red, green, blue)
            else:
                red = 204 * value + 51
                green = 237 * value + 17.5
                blue = 238 * value - 28
                if red > 255:
                    red = 255
                if green > 255:
                    green = 255
                if blue > 255:
                    blue = 255
                map[x, y] = (red, green, blue)
    plt.title(seed)
    plt.imshow(grid, cmap="gray")
    plt.show()
    plt.title(seed)
    plt.imshow(map.astype('uint8'))
    plt.show()



#SETTINGS
size = 600 #size of the image
scale = 1000 #like "zoom lever"
seed = 54545 #random number
#start
compute()
