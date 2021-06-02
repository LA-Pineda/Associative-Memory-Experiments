#!/bin/bash

memdir="runs/images/memories-004"
backdir="../../.."

cd $memdir
for i in 0 1 2 3 4 5 6 7 8 9; do
    cd stage_$i/msize_7
    g=`ls -l ${i}_* | fgrep -w 77 | head -1`
    echo $g
    cd ../..
done | cut -d\  -f9
