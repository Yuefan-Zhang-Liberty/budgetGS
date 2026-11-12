#!/bin/bash
cd /home/stud/zyuef/storage/user/zyuef/datasets
nohup unzip -o -q /home/stud/zyuef/360_v2.zip '*/images/*' </dev/null > unzip360.log 2>&1 &
sleep 3
echo "unzip_procs=$(pgrep -c unzip)"
echo "room_images_after=$(ls /home/stud/zyuef/storage/user/zyuef/datasets/room/images/ 2>/dev/null | wc -l)"
