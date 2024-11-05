import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics.sequence_labeling import get_entities
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import logging
logging.basicConfig(filename='nerpy.log',level=logging.INFO,format='%(asctime)s - %(message)s')
logger = logging.getLogger('test')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def get_entity(sentence):
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained("./bert4ner-base-chinese")
    model = AutoModelForTokenClassification.from_pretrained("./bert4ner-base-chinese")
    label_list = ['I-ORG', 'B-LOC', 'O', 'B-ORG', 'I-LOC', 'I-PER', 'B-TIME', 'I-TIME', 'B-PER']

    tokens = tokenizer.tokenize(sentence)
    inputs = tokenizer.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs).logits
    predictions = torch.argmax(outputs, dim=2)
    char_tags = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())][1:-1]

    pred_labels = [i[1] for i in char_tags]
    entities = []
    line_entities = get_entities(pred_labels)
    for i in line_entities:
        word = sentence[i[1]: i[2] + 1]
        entity_type = i[0]
        entities.append((word, entity_type))

    return entities

def to_output(inp,oup):
    df = pd.read_csv(inp)
    list=[]
    errorlist=[]
    dftext=df['微博正文']
    dftext=pd.DataFrame(dftext,columns=['微博正文'])
    for index,value in dftext.iterrows():
        try:
            value.astype(str)
        except:
            errorlist.append(index)
    dftext.drop(errorlist)
    logger.info(f"{len(errorlist)} message has been deleted:")

    for index,value in dftext.iterrows():
        list.append(get_entity(value.iloc[0][:510]))

    dfresult=pd.DataFrame(list)
    combine=pd.concat([df,dfresult], axis=1)
    combine.to_csv(oup,sep=',')

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
    logger.info(f"{errorlist.__len__()} message has been deleted:")
    pred = text_classification(df_trimmed['微博正文'].values.tolist())
    preds = pd.DataFrame(pred)

    # senti_id_time = pd.concat([df[0],df[1],df[2],sentiment],axis=1)
    # senti_id_time.to_csv('senti_id_time2022.csv',sep=',', header=False, index=False)
    result = pd.concat([df, preds], axis=1)
    # 将DataFrame保存回txt文件
    result.to_csv(outp, sep=',', header=False, index=False)
    logger.info(f"{inp} file has completed")

if __name__ == '__main__':
    to_output('./sentiment/senti2021.csv', "/exstorage/sjf/ner_result/2021_ner.csv")
