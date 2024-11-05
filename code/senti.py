import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics.sequence_labeling import get_entities
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import logging
import numpy as np
logging.basicConfig(filename='./log/senti20.log',level=logging.INFO,format='%(asctime)s - %(message)s')
logger = logging.getLogger('test')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time

def to_sentiment(inp,outp,modelname,task):
    df = pd.read_csv(inp)
    model = AutoModelForSequenceClassification.from_pretrained(modelname,local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    text_classification = pipeline(task, model=model,tokenizer=tokenizer)
    # sentiment = df[1].apply(lambda x: classifier(x)["score"])
    pred = []
    df_trimmed = df["微博正文"].apply(lambda x: x[:510] if isinstance(x, str) else x)
    df_trimmed=pd.DataFrame(df_trimmed,columns=['微博正文'])
    errorlist=[]
    for index, row in df_trimmed.iterrows():
        try:
            row.astype(str)
        except:
            errorlist.append(index)
    df.drop(errorlist)
    df_trimmed.drop(errorlist)

    preds=[]
    logger.info(f"{errorlist.__len__()} message has been deleted:")
    textlist=np.array(df_trimmed).astype(str)
    textlist_split = np.array_split(textlist, 1000, axis=0)
    for i,arr in enumerate(textlist_split):
        start = time.time()
        pred = text_classification(arr.flatten().tolist())
        logger.info(f"{str(0.1 * (i + 1))}" + f"% has finished in {(time.time() - start) / 60} minutes")
        print(f"{str(0.1 * (i + 1))}" + f"% has finished in {(time.time() - start) / 60} minutes")
        preds+=pred

    result = pd.concat([df, pd.DataFrame(preds)], axis=1)
    # 将DataFrame保存回txt文件
    result.to_csv(outp, sep=',', header=False, index=False)
    logger.info(f"{inp} file has completed")

if __name__ == '__main__':
    to_sentiment('./sentiment/senti2020.csv', "/exstorage/sjf/ner_result/2020_label.csv","./roberta-base-finetuned-chinanews-chinese","sentiment-analysis")
