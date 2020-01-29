import dask
import dask.array as da
import numpy as np
from distributed import Client


def read_image():
    return np.random.randint(0, 4000, (120, 120), dtype=np.uint16)

n_images = 2000000
delayed_read = dask.delayed(read_image, pure=True)
lazy_images = [da.from_delayed(delayed_read(), dtype=np.uint16, shape=(120, 120))
               for x in range(n_images)]

stack = da.stack(lazy_images, axis=0)
stack

stack = stack.rechunk(1000, 120, 120)
stack

client = Client('35.197.27.240:8786')
imgs = client.persist(stack)