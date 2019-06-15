from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def load_predictions(predict_path):
    predicts=np.loadtxt(predict_path).astype(int).tolist()
    return predicts

def load_groundTruth(path):
    file=open(path,'r')
    flag=0
    grounds=[]
    for line in file:
        if(flag==0):
            flag=1
            continue
        line=line.strip().split('\t')
        ground=int(line[1])
        grounds.append(ground)
    return grounds
def accuracyEachClass(grounds_B,predicts_B):
    total=len(grounds_B)
    correct0=0
    total0=0
    correct1=0
    total1=0
    correct2=0
    total2=0
    correct3=0
    total3=0
    for i in range(total):
        if grounds_B[i]==0:
            total0+=1
            if predicts_B[i]==0:
                correct0+=1
        if grounds_B[i]==1:
            total1+=1
            if predicts_B[i]==1:
                correct1+=1
        if grounds_B[i]==2:
            total2+=1
            if predicts_B[i]==2:
                correct2+=1
        if grounds_B[i]==3:
            total3+=1
            if predicts_B[i]==3:
                correct3+=1
    # print(total0, total1, total2,total3)
    return [correct0/total0,correct1/total1,correct2/total2,correct3/total3]

if __name__ == '__main__':
    # predictA_path="res/predict_A_MLP.txt"
    # predictB_path="res/predict_B_MLP.txt"
    predictA_path = "res/pred_A_1.txt"
    predictB_path = "res/pred_B_1.txt"
    groundTruthA_path='data/SemEval2018-T3_gold_test_taskA_emoji.txt'
    groundTruthB_path='data/SemEval2018-T3_gold_test_taskB_emoji.txt'

    predicts_A=load_predictions(predictA_path)
    predicts_B=load_predictions(predictB_path)
    grounds_A=load_groundTruth(groundTruthA_path)
    grounds_B=load_groundTruth(groundTruthB_path)
    # print(predicts_A[0:10], grounds_A[0:10])
    # print(np.unique(predicts_B), np.unique(grounds_B))

    accuracy_A=accuracy_score(grounds_A,predicts_A)
    accuracy_B=accuracy_score(grounds_B,predicts_B)
    print(" Accuracy of A and B are: {}, {}".format(accuracy_A,accuracy_B))

    precision_A=precision_score(grounds_A,predicts_A,average='binary')
    precision_B=precision_score(grounds_B,predicts_B,average='macro')
    print(" Precision of A and B are: {}, {}".format(precision_A,precision_B))

    recall_A=recall_score(grounds_A,predicts_A, average='binary')
    recall_B=recall_score(grounds_B,predicts_B, average='macro')
    print(" Recall of A and B are: {}, {}".format(recall_A,recall_B))

    F1_A=f1_score(grounds_A,predicts_A,average='binary')
    F1_B=f1_score(grounds_B,predicts_B,average='macro')
    print(" F1-macro of A and B are: {}, {}".format(F1_A, F1_B))

    print("task B accuracy on each class:")
    accuracy_list=accuracyEachClass(grounds_B,predicts_B)
    print(accuracy_list)