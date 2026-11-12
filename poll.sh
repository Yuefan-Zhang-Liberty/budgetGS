#!/bin/bash
echo "room_images=$(ls /home/stud/zyuef/storage/user/zyuef/datasets/room/images/ | wc -l)"
echo "bonsai_images=$(ls /home/stud/zyuef/storage/user/zyuef/datasets/bonsai/images/ | wc -l)"
echo "unzip_procs=$(pgrep -c unzip)"
tail -2 /home/stud/zyuef/storage/user/zyuef/datasets/unzip360.log
echo "render_jobs=$(sacct -n -X -j 1980814,1980815 --format=JobID,State,NodeList | tr '\n' ' ')"
echo "train_job=$(sacct -n -X -j 1980769 --format=JobID,State,Elapsed,NodeList | tr '\n' ' ')"
