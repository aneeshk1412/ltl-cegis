#!/bin/bash

rm -f loop_only.txt
touch loop_only.txt

for (( i = 1; i < 100; i+=1 )); do
	#statements
	echo Seed $i
	python bfs_minigrid.py --env-name MiniGrid-MultiKeyDoorKey-16x16-1 --verifier-seed "${i}" >>loop_only.txt
done
