#! /bin/bash

# Copyright [2020] Luis Alberto Pineda Cortés, Gibrán Fuentes Pineda,
# Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

runs_dir='runs'
imag_dir="${runs_dir}/images"
test_dir="${imag_dir}/test-004"
mems_dir="${imag_dir}/memories-004"
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

