fs = gcsfs.GCSFileSystem(project='big_earth', token='cloud')

metadata_paths = bucket.ls("big_earth/metadata")

def load_metadata_from_gcs(filename):
    r = bucket.cat(filename)
    return pd.read_csv(io.BytesIO(r))

def load_image_bands_from_gcs(base_filename):
    bands = []
    for band in ["02", "03", "04"]:
        r = bucket.cat(base_filename.format(band))
        bands.append(imageio.core.asarray(imageio.imread(r, 'TIFF')))
    return np.stack(bands, axis=-1)

df = pd.concat(map(load_metadata_from_gcs, metadata_paths))
df = df.set_index('image_prefix', drop=False)

base_filename = os.path.join(train_path, df.index[0], df.index[0] + "_B{}.tif")
img = load_image_bands_from_gcs(base_filename)
print(img.shape)
print("stacked pixel for 3 bands\n", img[0][0])

print(len(df))
df.head()