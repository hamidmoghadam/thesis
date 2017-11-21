


for((i=5;i<50; i= i+5))
do
    echo "=================== number of user 5 number posts is $i ==================================="
    python pmsvm_classifier2.py --source-dir-data /Users/hamidmoghaddam/Documents/thesis/social-media-forensics-master/microblog_authorship_attribution/dataset/out_train --output-dir hamid_output --minimal-number-tweets 2000  --number-authors $i --number-tweets 2000 -f word-1-gram


   
done
