for testfile in tests/test*.txt
do
    echo Running ${${testfile##*'/'}#.}
    python main.py < testfile
    echo "\n\n"
done
