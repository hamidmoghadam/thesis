echo 70 15 85 3
for((i = 5; i=<20; i=i+5))
do
    python hybrid_classifier.py 10 1000 20 70 $i 85 3 
    python hybrid_classifier.py 10 1000 20 70 $i 85 3 
done




