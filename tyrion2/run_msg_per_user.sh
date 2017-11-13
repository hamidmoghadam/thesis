
for((i = 5; i=<25; i=i+5))
do
    echo 70 $i 85 3
    for((j = 0; j<5; j++))
    do
        python hybrid_classifier.py 10 1000 20 70 $i 85 3 
        python hybrid_classifier.py 10 1000 20 70 $i 85 3 
    done
done




