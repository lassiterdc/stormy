#!/bin/bash

## tips on how to use:
	# source __utils.sh
	# #confirm working directory exists
	# mkdir -p ${assar_dirs[dir_repo]}${assar_dirs[subdir]}
	# # move to a directory
	# cd ${assar_dirs[dir_repo]}${assar_dirs[subdir]}

format_time() {
  ((h=${1}/3600))
  ((m=(${1}%3600)/60))
  ((s=${1}%60))
  printf "%02d:%02d:%02d\n" $h $m $s
 }