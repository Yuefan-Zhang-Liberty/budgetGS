#!/bin/bash
D=/home/stud/zyuef/storage/user/zyuef/datasets/360_v2
echo "A1 room_images=$(ls "$D/room/images/" 2>&1 | wc -l)"
echo "A2 room_images2=$(ls "$D/room/images_2/" 2>&1 | wc -l)"
echo "A3 bonsai_images=$(ls "$D/bonsai/images/" 2>&1 | wc -l)"
echo "A4 bonsai_images2=$(ls "$D/bonsai/images_2/" 2>&1 | wc -l)"
echo "A5 room_top=$(ls "$D/room/" | tr '\n' ' ')"
echo "A6 wget=$(which wget 2>&1)"
echo "A7 unzip=$(which unzip 2>&1)"
echo "A8 jobs=$(sacct -n -X -j 1980769,1980770,1980771 --format=JobID,State,NodeList 2>&1 | tr '\n' ' ')"
echo "A9 ziplog=$(ls -la /home/stud/zyuef/360_v2.zip* 2>/dev/null | tail -2 | tr '\n' ' ')"
echo "A10 wgetrunning=$(pgrep -f 'wget.*360_v2' >/dev/null && echo yes || echo no)"
echo "A11 zipsize=$(stat -c%s /home/stud/zyuef/360_v2.zip 2>/dev/null || echo 0)"
echo "A12 disk=$(df -h / | tail -1)"
