#!/bin/bash

# 设置最大并行进程数
max_processes=10

# 创建一个函数，用于运行每个任务
run_task() {
    python katacr/build_train_dataset/cut_episodes.py --path-video "$1"
}

export -f run_task

# 定义要运行的视频文件列表
video_files=(
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/2.6HogDaily_20231014.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20210528.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230203.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230211.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230212.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230224.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230305.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230314.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230319.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230327.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230329.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230430.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230604.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230621.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230630.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230704.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230729.mp4"
    "/home/wty/Coding/datasets/CR/fast_pig_2.6/OYASSU_20230910.mp4"
)

# 计算任务总数
total_tasks=${#video_files[@]}

# 使用 parallel 命令并发运行任务，并通过 pv 显示进度
parallel -j $max_processes run_task ::: "${video_files[@]}" | pv -l -s $total_tasks > /dev/null

# 等待所有任务完成
wait

