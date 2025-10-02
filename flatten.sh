#!/usr/bin/env bash
# flatten.sh
# Create flattened sibling dir with symlinks
# 
# This script takes a nested directory structure and creates a "flat" version
# where all files are accessible in a single directory using symlinks.
# The original directory structure is preserved, but filenames encode the path.

# Enable strict error handling:
# -e: exit immediately if any command fails
# -u: treat undefined variables as errors  
# -o pipefail: pipes fail if any command in the pipe fails
set -euo pipefail

# Check if user provided at least one argument (the directory to flatten)
# $# is the number of command line arguments
if [ $# -lt 1 ]; then
  echo "Usage: $0 <directory-to-flatten>"
  exit 1
fi

# Get the directory path from the first command line argument
TOP_RAW="$1"

# Remove any trailing slash from the path using parameter expansion
# ${variable%pattern} removes the shortest match of pattern from the end
TOP="${TOP_RAW%/}"                           # strip trailing slash

# Check if TOP is actually a directory, exit with error message if not
# [ -d "$TOP" ] tests if TOP is a directory
# || means "or" - if the test fails, execute the right side
# { } groups commands together
[ -d "$TOP" ] || { echo "Not a directory: $TOP"; exit 1; }

# Extract just the directory name (last part of the path)
# basename removes the directory path, leaving just the final component
# e.g., basename "/path/to/dataset" returns "dataset"
BASE="$(basename "$TOP")"                    # e.g., dataset

# Get the absolute path of the parent directory
# dirname gets the parent directory path
# cd changes directory, pwd prints current directory
# $( ) captures command output as a variable
# && means "and" - only run pwd if cd succeeds
PARENT="$(cd "$(dirname "$TOP")" && pwd)"    # absolute parent dir

# Create the name for our flattened directory
# It will be a sibling of the original directory with "_flat" suffix
FLAT="${PARENT}/${BASE}_flat"                # sibling: dataset_flat

# Create the flattened directory (and any parent directories if needed)
# -p flag means "create parent directories as needed, don't error if already exists"
mkdir -p "$FLAT"

# Find all files in the directory tree and process each one
# This is the main loop that creates symlinks for every file
#
# find "$TOP" -type f -print0: 
#   - find: search for files and directories
#   - "$TOP": start searching from our target directory
#   - -type f: only find regular files (not directories)
#   - -print0: separate results with null characters instead of newlines
#     (this handles filenames with spaces or special characters safely)
#
# | while IFS= read -r -d '' f; do:
#   - | pipes the find output to a while loop
#   - IFS= sets Internal Field Separator to empty (preserves whitespace)
#   - read -r: read without interpreting backslashes as escape characters
#   - -d '': use null character as delimiter (matches find's -print0)
#   - f: variable name to store each filename
find "$TOP" -type f -print0 | while IFS= read -r -d '' f; do
  
  # Extract the relative path from the full file path
  # ${f#${TOP}/} removes the TOP directory path from the beginning
  # e.g., if f="/path/to/EN/folder/file.wav" and TOP="/path/to/EN"
  # then rel="folder/file.wav"
  rel="${f#${TOP}/}"                         # path relative to TOP
  
  # Create a "safe" filename by replacing forward slashes with hyphens
  # ${rel//\//-} replaces ALL occurrences of / with -
  # e.g., "folder/subfolder/file.wav" becomes "folder-subfolder-file.wav"
  safe="${rel//\//-}"                        # replace '/' with '-'
  
  # Create the destination path for our symlink
  dest="$FLAT/$safe"

  # Handle filename collisions by adding a number suffix
  # This loop runs if a file with this name already exists
  i=1
  while [ -e "$dest" ]; do
    # -e tests if file exists (any type: file, directory, symlink, etc.)
    # If it exists, append a number to make it unique
    dest="$FLAT/${safe}-${i}"
    i=$((i+1))  # $((arithmetic)) performs integer arithmetic
  done

  # Create the symbolic link
  # ln -s creates a symbolic link
  # "../$BASE/$rel": the target path (what the link points to)
  #   - ../ goes up one directory from FLAT to PARENT
  #   - $BASE is the original directory name
  #   - $rel is the relative path within that directory
  # "$dest": the link name/location
  #
  # Example: if we're in /data/EN_flat/ and want to link to /data/EN/audio/file.wav
  # we create: ln -s "../EN/audio/file.wav" "audio-file.wav"
  ln -s "../$BASE/$rel" "$dest"
done

# Print completion message
echo "Done. Symlinks created in: $FLAT"
