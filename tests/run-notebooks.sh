#!/bin/bash
# Run jupyter notebooks as given on command line, show output only on error.
# If a directory is provided, run all contained notebooks non-recursively.
set -x
set -e

SCRIPT_PATH=$(dirname $BASH_SOURCE)
SCRIPT_PATH=$(cd $SCRIPT_PATH && pwd)

runNotebook () {
    set +e
    tempfile=$(mktemp)
    jupyter nbconvert --debug --stdout --execute --ExecutePreprocessor.timeout=300 --to markdown $@ &> $tempfile
    ret=$?
    if [[ $ret != 0 ]]; then
      cat $tempfile
      exit $ret
    fi
    rm $tempfile
    set -e
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 [notebook.ipynb] [dirContainingNotebooks/]"
    exit 1
fi

pip3 install jupyter ipykernel nbconvert

for arg in "$@"; do
    if [ -d $arg ]; then
        for notebook in $(ls -1 $arg | grep -E ipynb\$); do
            runNotebook $arg/$notebook
        done
    elif [ -f $arg ]; then
        runNotebook $arg
    fi
done
