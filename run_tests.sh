for testfile in tests/test*.txt
do
    base=${testfile##*'/'}
    echo Running ${base%%.*}
    python interiorpt.py < $testfile
    echo
    echo
done
