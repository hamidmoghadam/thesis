
for((i=1000;i<3500; i= i +500))
do
    echo "============== number of user 10 number posts is $i ==============="
    for((j=0; j<10; j++))
    do
        python pmsvm_classifier2.py --source-dir-data /home/hamidm/Documents/thesis/thesis/social-media-forensics-master/microblog_authorship_attribution/dataset/out_train --output-dir hamid_output --minimal-number-tweets $i  --number-authors 10 --number-tweets $i
    done
done
