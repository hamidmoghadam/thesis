
for((i = 40; i<85; i=i+5))
do
    echo 15 $i 100 3
    for((j = 0; j<5; j++))
    do
        python rnn_cnn.py 20 1000 15 $i 100 3  
    done
done

for((i = 50; i<120; i=i+5))
do
    echo 15 85 $i 3
    for((j = 0; j<5; j++))
    do
        python rnn_cnn.py 20 1000 15 85 $i 3  
    done
done



