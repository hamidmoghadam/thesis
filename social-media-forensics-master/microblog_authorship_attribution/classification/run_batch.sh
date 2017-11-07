max=4000

for((i=50;i<500; i= i+50))
do
    echo "============================================= number of user 25  number posts is $i ============================================================="
    for((j=0; j<10; j++))
    do
        python pmsvm_classifier2.py --source-dir-data /Users/hamidmoghaddam/Documents/thesis/social-media-forensics-master/microblog_authorship_attribution/dataset/out_train --output-dir hamid_output --minimal-number-tweets $i  --number-authors 25 --number-tweets $i
    done
done


for((i=1000;i<$max; i= i +500))
do
    echo "============================================= number of user 25 number posts is $i ============================================================="
    for((j=0; j<10; j++))
    do
        python pmsvm_classifier2.py --source-dir-data /Users/hamidmoghaddam/Documents/thesis/social-media-forensics-master/microblog_authorship_attribution/dataset/out_train --output-dir hamid_output --minimal-number-tweets $i  --number-authors 25 --number-tweets $i
    done
done
