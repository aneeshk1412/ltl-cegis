#!/bin/bash

for envname in MiniGrid-MultiKeyDoorKey-16x16-2; do
	for algo in bfs priority; do
		for seed in 100 200 300 400 500; do
			python bfs_minigrid.py --env-name "${envname}" --verifier-seed "${seed}" --algorithm "${algo}" >"logs/${envname}-${algo}-${seed}.log" &
		done
		wait
	done
done
