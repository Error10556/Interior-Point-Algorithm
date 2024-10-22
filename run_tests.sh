for testfile in tests/test*.txt
do
    base=${testfile##*'/'}
    echo Running ${base%%.*}
    python main.py < $testfile
    echo
    echo
done
