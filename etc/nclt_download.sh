#!/bin/bash

NCLT_URL=http://robots.engin.umich.edu/nclt

FilePython=("read_vel_hits.py"
            "read_vel_sync.py"
            "project_vel_to_cam.py"
            "undistort.py"
)

FileData=("2012-04-29"
          "2013-01-10"
)

FileCalib="cam_params.zip"
FileUndist=("U2D_ALL_1616X1232.tar.gz"
            "D2U_ALL_1616X1232.tar.gz"
)

# Download Python Scripts
echo "Downloading python scripts from NCLT"

for FILE in "${FilePython[@]}"
do
  if [ -f $FILE ]; then
    echo "File already exists: ${FILE}"
  else
    wget "${NCLT_URL}/python/${FILE}"
  fi
done

# Download camera and LIDAR data
echo "Downloading camera and LIDAR data"

for FILE in "${FileData[@]}"
do
  FILE_CAM="${FILE}_lb3.tar.gz"
  FILE_LIDAR="${FILE}_vel.tar.gz"
  FILE_GT="groundtruth_${FILE}.csv"
  if [ -f $FILE_CAM ]; then
    echo "File already exists: ${FILE_CAM}"
  else
    wget "${NCLT_URL}/images/${FILE_CAM}"
  fi
  if [ -f $FILE_LIDAR ]; then
    echo "File already exists: ${FILE_LIDAR}"
  else
    wget "${NCLT_URL}/velodyne_data/${FILE_LIDAR}"
  fi
  if [ -f $FILE_GT ]; then
    echo "File already exists: ${FILE_GT}"
  else
    wget "${NCLT_URL}/ground_truth/${FILE_GT}"
  fi
done

if [ -f $FileCalib ]; then
  echo "File already exists: ${FileCalib}"
else
  wget "${NCLT_URL}/ladybug3_calib/${FileCalib}"
fi

for FILE in "${FileUndist[@]}"
do
  if [ -f $FILE ]; then
    echo "File already exists: ${FILE}"
  else
    wget "${NCLT_URL}/ladybug3_calib/${FILE}"
  fi
done

# Extracting Files
echo "Extracting Files"
for FILE in "${FileData[@]}"
do
  FILE_CAM="${FILE}_lb3.tar.gz"
  FILE_LIDAR="${FILE}_vel.tar.gz"
  
  tar -xzf $FILE_CAM
  echo "    Extraction is done: ${FILE_CAM}"
  tar -xzf $FILE_LIDAR
  echo "    Extraction is done: ${FILE_LIDAR}"
done

unzip $FileCalib -d cam_params
for FILE in "${FileUndist[@]}"
do
  tar -xzf $FILE -C cam_params
done
