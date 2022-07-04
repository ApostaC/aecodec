#!/bin/bash
set -e

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <modelfile name> <out folder>"
fi

ROOT=`pwd`/..
PROFILER=${ROOT}/src/sim_profiler.py
VIDEO_FOLDER=${ROOT}/data/test_videos/

MODEL_FILE=$1
OUTFOLDER=$2
mkdir -p $OUTFOLDER

if [[ -f $MODEL_FILE ]]; then
    # run autoencoder
    echo "Found the model file! will profile the model"
    for video in $(ls $VIDEO_FOLDER); do
        vname=$(basename $VIDEO_FOLDER/$video .mp4)
        echo "video name is $vname"
        outfile=$OUTFOLDER/$vname-ae.csv
        python3 $PROFILER $VIDEO_FOLDER/$video $outfile ae $MODEL_FILE
    done
else
    # run mpeg
    echo "Not found the model, run mpeg!"
    for video in $(ls $VIDEO_FOLDER); do
        vname=$(basename $VIDEO_FOLDER/$video .mp4)
        echo "video name is $vname"
        outfile=$OUTFOLDER/$vname-mpeg.csv
        python3 $PROFILER $VIDEO_FOLDER/$video $outfile mpeg 
    done
fi
