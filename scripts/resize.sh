# Path to a directory `base/` with images in `base/images/`.
DATASET_PATH=$1

# Resize images.

cp -r "$DATASET_PATH"/images "$DATASET_PATH"/images_4

pushd "$DATASET_PATH"/images_4
ls | xargs -P 8 -I {} mogrify -resize 25% {}
popd
