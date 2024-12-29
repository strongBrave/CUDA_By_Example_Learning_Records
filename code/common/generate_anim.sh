#!/bin/bash

# This script is used to streamline the process of making videos of anim.cu file in chapter_5.
# It consists of 3 steps, including complie anim.cu file, run anim exe to generate frames(imgs), run 
# imgs_to_video.py to make videos.

# Navigate to the parent directory
cd ../
echo "Current directory: $(pwd)"
# Define variables
CHAPTER_DIR="/nas/home/yujunhao/cuda_by_example/code/chapter_7"
ANIM_FILE_NAME="heater_w_texture_memory"
ANIM_CU_FILE_PATH="$CHAPTER_DIR/$ANIM_FILE_NAME.cu"          # Path to the CUDA source file
ANIM_EXE_SAVE_PATH="$CHAPTER_DIR/res/$ANIM_FILE_NAME"        # Path where the compiled executable will be saved
CONDA_ENV_NAME="gen6d"               # Conda environment name
I2V_PY_PATH="/nas/home/yujunhao/cuda_by_example/code/common/imgs_to_video.py"       # Path to the Python script for converting images to video
IMG_FOLDER="$CHAPTER_DIR/imgs"                  # Directory where the generated frames (images) will be stored
OUTPUT_VIDEO_DIR=""
OUTPUT_VIDEO_PATH="$CHAPTER_DIR/$ANIM_FILE_NAME.mp4"

# Check if the required CUDA source file exists
if [ ! -f "$ANIM_CU_FILE_PATH" ]; then
  echo "Error: CUDA source file '$ANIM_CU_FILE_PATH' not found!"
  exit 1
fi

# Check if the Python script exists
if [ ! -f "$I2V_PY_PATH" ]; then
  echo "Error: Python script '$I2V_PY_PATH' not found!"
  exit 1
fi

# Ensure the directory for generated frames exists
if [ ! -d "$IMG_FOLDER" ]; then
  mkdir -p "$IMG_FOLDER"
  echo "Created directory for frames: $IMG_FOLDER"
fi

# Compile the CUDA file into an executable
echo "Compiling $ANIM_CU_FILE_PATH into $ANIM_EXE_SAVE_PATH..."
nvcc -o "$ANIM_EXE_SAVE_PATH" "$ANIM_CU_FILE_PATH" -lglut -lGL -lGLU
if [ $? -ne 0 ]; then
  echo "Error: Failed to compile $ANIM_CU_FILE_PATH!"
  exit 1
fi
echo "Compilation successful!"

# Run the compiled executable to generate frames
echo "Running $ANIM_EXE_SAVE_PATH to generate frames..."
"$ANIM_EXE_SAVE_PATH"
if [ $? -ne 0 ]; then
  echo "Error: Failed to execute $ANIM_EXE_SAVE_PATH!"
  exit 1
fi
echo "Frames generated successfully in '$IMG_FOLDER'."

# Activate the Conda environment
echo "Activating Conda environment: $CONDA_ENV_NAME..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
if [ $? -ne 0 ]; then
  echo "Error: Failed to activate Conda environment '$CONDA_ENV_NAME'!"
  exit 1
fi
echo "Conda environment '$CONDA_ENV_NAME' activated."

# Run the Python script to create the video
echo "Running Python script $I2V_PY_PATH to create video..."
python "$I2V_PY_PATH" --image_folder "$IMG_FOLDER" --output_video_path "$OUTPUT_VIDEO_PATH"
if [ $? -ne 0 ]; then
  echo "Error: Failed to create video using $I2V_PY_PATH!"
  exit 1
fi
echo "Video successfully created as 'output_video.mp4'."

# Final success message
echo "Process completed successfully!"
