#!/bin/sh

cd "$(dirname "$0")"

wget http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip
unzip cornell_movie_dialogs_corpus.zip "cornell movie-dialogs corpus/movie_lines.txt"
mv "cornell movie-dialogs corpus/movie_lines.txt" movie_lines_unsorted.txt
sed -i 's#^L##' movie_lines_unsorted.txt
sort -n movie_lines_unsorted.txt > movie_lines.txt
rm movie_lines_unsorted.txt
