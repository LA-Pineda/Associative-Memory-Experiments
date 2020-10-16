#! /usr/bin/bash

runs_dir='runs'
imag_dir="${runs_dir}/images"
test_dir="${imag_dir}/test/partial"
mems_dir="${imag_dir}/memories/partial"
random_dir=`openssl rand -hex 4`
random_dir=${imag_dir}/${random_dir}

if [ "$#" -ne 0 ]; then
    echo "This script does not use any parameters"
    exit 1
fi

mkdir ${random_dir}
digs_fn="${random_dir}/digits.txt"
stgs_fn="${random_dir}/stages.txt"
shuf -i 0-9 > $digs_fn
shuf -i 0-9 > $stgs_fn

pair_fn="${random_dir}/pairs.txt"
paste -d, $digs_fn $stgs_fn > $pair_fn

for i in `cat $pair_fn`; do 
    IFS=',' read digit stage <<< "${i}"

    ts_dir="${test_dir}/stage_${stage}"
    ms_dir="${mems_dir}/stage_${stage}"
    dig_fn=`ls ${ts_dir}/${digit}_?????.png | shuf -n 1`
    dig_fn=`basename $dig_fn .png`
    join_imgs="${random_dir}/${dig_fn}-join.png"

    original_img=${ts_dir}/${dig_fn}-original.png
    decoded_img=${ts_dir}/${dig_fn}.png

    memories_imgs=`find ${ms_dir} -name ${dig_fn}.png -print | sort`
    echo $memories_imgs
    convert ${original_img} ${decoded_img} $(echo $memories_imgs) -border 2 -bordercolor white -append $join_imgs     
done

join_imgs=`find ${random_dir} -type f -name '*.png' -print| sort`
final_img=${random_dir}/'all.png'
convert $(echo $join_imgs) +append ${final_img}
mogrify -scale '1000%' ${final_img}

