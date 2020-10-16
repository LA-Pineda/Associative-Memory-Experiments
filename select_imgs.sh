#! /usr/bin/bash

runs_dir='runs'
imag_dir="${runs_dir}/images"
test_dir="${imag_dir}/test/partial"
mems_dir="${imag_dir}/memories/partial"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 stage-id.txt"
    echo "Where stage-id.txt is a text file (must have .txt extension) with pairs of stage and id"
    exit 1
fi

random_dir=`basename $1 .txt`
random_dir=${imag_dir}/${random_dir}

if [ ! -d ${random_dir}  ]; then
    mkdir ${random_dir}
fi

pair_fn=$1

for i in `cat $pair_fn`; do 
    IFS=',' read stage id <<< "${i}"

    echo $stage $id
    ts_dir="${test_dir}/stage_${stage}"
    ms_dir="${mems_dir}/stage_${stage}"
    dig_fn=$id
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

