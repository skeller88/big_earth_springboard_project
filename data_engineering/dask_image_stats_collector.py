import time
import dask
import dask.array as da
import gcsfs
import imageio
import numpy as np
from distributed import Client

client = Client()


def read_filenames_from_gcs(filenames):
    bands = ["B02", "B03", "B04"]
    fs = gcsfs.GCSFileSystem(project='big_earth')
#     gcs_client = storage.Client()
#     bucket = gcs_client.bucket("big_earth")

    def read(filename):
        imgs = []
        for band in bands:
            image_path = f"{filename}{filename.split('/')[-2]}_{band}.tif"
#             image_path = image_path.replace("big_earth/", "")
#             blob = bucket.blob(image_path)
#             r = blob.download_as_string()
            r = fs.cat(image_path)
            imgs.append(imageio.core.asarray(imageio.imread(r, 'TIFF')))
        return np.stack(imgs, axis=-1).flatten()

    delayed_read = dask.delayed(read)
    return [da.from_delayed(delayed_read(filename), shape=(14400 * 3, ), dtype=np.uint16) for filename in filenames]

fs = gcsfs.GCSFileSystem(project='big_earth')
filenames = fs.ls("big_earth/raw_rgb/tiff")

imgs = read_filenames_from_gcs(filenames)
imgs = da.stack(imgs, axis=0)
imgs_rechunked = imgs.rechunk((50, 43200))

st = time.time()

chunk_size = 40000
chunk_num = 0
chunk_futures = []
start = 0
end = start + chunk_size
image_paths_to_submit = filenames
is_last_chunk = False

for dataset in client.list_datasets():
    client.unpublish_dataset(dataset)

while True:
    cst = time.time()
    chunk = image_paths_to_submit[start:end]
    cst1 = time.time()

    if start == 0:
        print('loaded chunk in', cst1 - cst)

    if len(chunk) == 0:
        break

    chunk_future = client.submit(read_filenames_from_gcs, chunk)
    chunk_futures.append(chunk_future)
    dataset_name = "chunk_{}".format(chunk_num)
    client.publish_dataset(**{dataset_name: chunk_future})

    if start == 0:
        print('submitted chunk in', time.time() - cst1)
    start = end

    if is_last_chunk:
        break

    chunk_num += 1
    end = start + chunk_size
    if end > len(image_paths_to_submit):
        is_last_chunk = True
        end = len(image_paths_to_submit)

    if start == end:
        break

    if chunk_num % 3 == 0:
        print('chunk_num', chunk_num, 'start', start)

print('completed in', time.time() - st)

persisted_chunks = []
start = time.time()
for idx, chunk in enumerate(chunk_futures):
    startc = time.time()
    persisted_chunk = client.persist(chunk.result())
    persisted_chunks.append(persisted_chunk)
    dataset_name = 'persisted_chunk_{}'.format(idx)
    client.publish_dataset(**{dataset_name: persisted_chunk})
    if idx == 0:
        print('submitted chunk in', time.time() - startc)

print('submitted all chunk_futures in', time.time() - start)

da.concatenate()


############## old code - not used right now
persisted_chunks = []
start = time.time()
for idx, chunk in enumerate(chunk_futures):
    if idx == 0:
        startc = time.time()
        persisted_chunks.append(client.persist(chunk.result()))
        print('submitted chunk in', time.time() - startc)
    else:
        persisted_chunks.append(client.persist(chunk.result()))

print('submitted all chunk_futures in', time.time() - start)

