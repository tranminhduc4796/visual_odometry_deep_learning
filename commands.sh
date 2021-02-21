#!/bin/bash
python -B main.py -datadir ./KITTI/dataset/ -gradClip 45. -imageWidth 448 -imageHeight 192 -outputParameterization default -expID tmp
# -loadFlowNet ./models/flownets_EPE1.951.pth.tar