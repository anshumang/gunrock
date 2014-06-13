#!/bin/sh

OPTION1="" #directed and do not mark-pred"
OPTION2="--mark-pred" #directed and mark-pred"
OPTION3="--undirected" #undirected and do not mark-pred"
OPTION4="--undirected --mark-pred" #undirected and mark-pred"

#put OS and Device type here
SUFFIX="linuxmint15.k40cx4_6.0_metis"
EXCUTION="./bin/test_bfs_6.0_x86_64"
DATADIR="/data/gunrock_dataset/large"
OPTION="--src=randomize --device=0,1,2,3 --partition_method=metis --queue-sizing=4.0"

mkdir -p eval/$SUFFIX

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 kron_g500-logn21 webbase-1M soc-LiveJournal1
do
    echo $EXCUTION market $DATADIR/$i/$i.mtx $OPTION $OPTION1
         $EXCUTION market $DATADIR/$i/$i.mtx $OPTION $OPTION1 > eval/$SUFFIX/$i.$SUFFIX.dir_no_mark_pred.txt
    sleep 10

    echo $EXCUTION market $DATADIR/$i/$i.mtx $OPTION $OPTION2
         $EXCUTION market $DATADIR/$i/$i.mtx $OPTION $OPTION2 > eval/$SUFFIX/$i.$SUFFIX.dir_mark_pred.txt
    sleep 10

    echo $EXCUTION market $DATADIR/$i/$i.mtx $OPTION $OPTION3
         $EXCUTION market $DATADIR/$i/$i.mtx $OPTION $OPTION3 > eval/$SUFFIX/$i.$SUFFIX.undir_no_mark_pred.txt
    sleep 10

    echo $EXCUTION market $DATADIR/$i/$i.mtx $OPTION $OPTION4
         $EXCUTION market $DATADIR/$i/$i.mtx $OPTION $OPTION4 > eval/$SUFFIX/$i.$SUFFIX.undir_mark_pred.txt
    sleep 10
done
