import tarfile
from pathlib import Path

import pandas

filepath = Path.home() / "Downloads/BigEarthNet-v1.0.tar.gz"

with tarfile.open(filepath, 'r') as fileobj:
    names = fileobj.getnames()
    df = pandas.DataFrame(names)
    df.to_csv("big_earth_filenames.csv", index=False)
