#!/bin/bash

find /home/z52lu/projects/def-vganesh/z52lu/smtlib24 -type f -name '*=*.smt2' | while IFS= read -r fname; do
    # Remove '=' from filenames
    newname=$(echo "$fname" | sed 's/=//g')
    if [[ "$fname" != "$newname" ]]; then
        mv "$fname" "$newname"
    fi
done
