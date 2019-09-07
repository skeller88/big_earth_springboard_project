# Environment
```bash
export PROJECT_DIR=<project-dir>
cd $PROJECT_DIR
python3 -m venv venv
pip install -r requirements.txt

# Jupyter should already be installed
python -m ipykernel install --user --name=big_earth_springboard_project
```

# Data preparation
Download BigEarth data

```bash
export DIR_WITHOUT_CLOUDS_AND_SNOW=<put BigEarth data without clouds and snow here>

```

```bash
python $PROJECT_DIR/eliminate_snowy_cloudy_patches.py -r ~/Documents/BigEarthNet-v1.0/ -e \
patches_with_cloud_and_shadow.csv patches_with_seasonal_snow.csv -d DIR_WITHOUT_CLOUDS_AND_SNOW
```

