
#for((i = 40; i<85; i=i+5))
#do
#    echo 15 $i 100 3
#    for((j = 0; j<5; j++))
#    do
#        python rnncnn.py 10 1000 20 15 $i 100 3 0.0005 0.5 
#    done
#done

for((i = 85; i<120; i=i+5))
do
    echo 15 85 $i 3
    for((j = 0; j<5; j++))
    do
        python rnncnn.py 10 1000 20 15 85 $i 3 0.0005 0.5
    done
done



