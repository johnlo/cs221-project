#!/bin/bash

i=0
for dir in `ls data`; do
    for mood in `cat data/$dir/mood`; do
	mkdir -p tmp/$mood
	cp -f data/$dir/text tmp/$mood/$i
	let i=i+1
    done
done
