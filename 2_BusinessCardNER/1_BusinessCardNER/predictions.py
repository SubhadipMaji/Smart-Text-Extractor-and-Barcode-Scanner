import numpy as np
import pandas as pd
import cv2
import pytesseract
import glob as glob
import spacy
# from spacy import displacy
import re 
import string 
import warnings
warnings.filterwarnings('ignore')

# load NER model
model_NER = spacy.load('output/model-best/')

# group the label
class groupgen():
    def __init__(self):
        self.id = 0
        self.text = ''
        
    def getgroup(self,text):
        if self.text == text:
            return self.id
        else:
            self.id +=1
            self.text = text
            return self.id
grp_gen = groupgen()

def cleanText(txt):
    whitespace = string.whitespace
    punctuation = "!#$%&\'()*+:;<=>?[\\]^`{|}~"
    tableWhitespace = str.maketrans('','',whitespace)
    tablePunctuation = str.maketrans('','',punctuation)
    text = str(txt)
    text = text.lower()
    removewhitespace = text.translate(tableWhitespace)
    removepunctuation = removewhitespace.translate(tablePunctuation)
    
    return str(removepunctuation)

# PARSER
def parser(text, label):
    if label == 'PHONE':
        text = text.lower()
        text = re.sub(r'\D','',text)
    
    elif label == 'EMAIL':
        text = text.lower()
        allow_special_char = '@_.\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char),'',text)

    elif label == 'WEB':
        text = text.lower()
        allow_special_char = ':/.%#\-'
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(allow_special_char),'',text)

    elif label in ('NAME', 'DES'):
        text = text.lower()
        text = re.sub(r'[^a-z ]','',text)
        text = text.title()

    elif label == 'ORG':
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]','',text)
        text = text.title()

    return text


def get_predictions(image):
    #extract data using pytesseract
    tess_data = pytesseract.image_to_data(image)

    #convert list into dataframe
    tess_list = list(map(lambda x:x.split('\t'), tess_data.split('\n')))
    df = pd.DataFrame(tess_list[1:], columns=tess_list[0])
    df.dropna(inplace=True)
    df['text'] = df['text'].apply(lambda x: cleanText(x))

    #convert data into content
    df_clean = df.query('text != ""')
    content = " ".join([text for text in df_clean['text']])
    print(content)

    #get predictions from NER model
    doc = model_NER(content)

    # convert doc to json
    doc_json = doc.to_json()
    doc_text = doc_json['text']

    #creating tokens
    dataframe_tokens = pd.DataFrame(doc_json['tokens'])
    dataframe_tokens['token'] = dataframe_tokens[['start', 'end']].apply(
        lambda x: doc_text[x[0]:x[1]], axis=1)

    right_table = pd.DataFrame(doc_json['ents'])[['start','label']]
    dataframe_tokens = pd.merge(dataframe_tokens,right_table,how='left',on='start')
    dataframe_tokens.fillna('O',inplace=True)

    # join tokens to df_clean dataframe
    df_clean['end'] = df_clean['text'].apply(lambda x: len(x)+1).cumsum() - 1 
    df_clean['start'] = df_clean[['text','end']].apply(lambda x: x[1] - len(x[0]),axis=1)

    # inner join with start 
    dataframe_info = pd.merge(df_clean,dataframe_tokens[['start','token','label']],how='inner',on='start')


    #BOUNDUING BOX
    bb_df = dataframe_info.query("label != 'O' ")

    bb_df['label'] = bb_df['label'].apply(lambda x: x[2:])
    bb_df['group'] = bb_df['label'].apply(grp_gen.getgroup)

    # right and bottom of bounding box
    bb_df[['left','top','width','height']] = bb_df[['left','top','width','height']].astype(int)
    bb_df['right'] = bb_df['left'] + bb_df['width']
    bb_df['bottom'] = bb_df['top'] + bb_df['height']

    # tagging: groupby group
    col_group = ['left','top','right','bottom','label','token','group']
    group_tag_img = bb_df[col_group].groupby(by='group')
    img_tagging = group_tag_img.agg({
        
        'left':min,
        'right':max,
        'top':min,
        'bottom':max,
        'label':np.unique,
        'token':lambda x: " ".join(x)
        
    })

    # draw bounding box
    img = image.copy()

    for x,y,w,h,label in img_tagging[['left','top','right','bottom','label']].values:
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        
        cv2.rectangle(img,(x,y),(w,h),(0,255,0),2)
        cv2.putText(img,str(label),(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2)


    ### Entities
    info_array = dataframe_info[['token','label']].values
    entities = dict(NAME = [], ORG = [], DES = [], PHONE = [], EMAIL = [], WEB = [])
    previous = 'O'

    for token, label in info_array:
        bio_tag = label[0]
        label_tag = label[2:]

        #step-1 : parse the token
        text = parser(token, label_tag)
        if bio_tag in ('B','I'):
            if label_tag != previous:
                entities[label_tag].append(text)
            else:
                if bio_tag == 'B':
                    entities[label_tag].append(text)
                else:
                    if label_tag in ('NAME','ORG','DES'):
                        entities[label_tag][-1] += ' ' + text
                    else:
                        entities[label_tag][-1] += text

        previous = label_tag

    return img, entities