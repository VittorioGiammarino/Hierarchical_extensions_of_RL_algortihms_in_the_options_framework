#!/bin/bash

for job in $(seq 1634997 1635055);
do
qdel $job
done
