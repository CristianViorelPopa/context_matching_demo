import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np


st.title('Context Matching Demo')


# important to initialize the model once and then predict using it while loaded into memory
# model = SentenceTransformer('paraphrase-distilroberta-base-v1', device='cpu')
# model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v1', device='cpu')


@st.cache(allow_output_mutation=True)
def load_sentence_transformers_model(model_name):
    return SentenceTransformer(model_name, device='cpu')


# the function for computing the string similarity using the model
def string_similarity(ref, sent):
    ref_embeddings = model.encode(ref, convert_to_tensor=True)
    sent_embeddings = model.encode(sent, convert_to_tensor=True)
    return util.pytorch_cos_sim(ref_embeddings, sent_embeddings).item()


model_name = st.selectbox(
     'Model used',
     (
         'paraphrase-distilroberta-base-v1',
         'paraphrase-xlm-r-multilingual-v1',
         'paraphrase-TinyBERT-L6-v2',
         'stsb-distilbert-base',
         'stsb-roberta-large'
      ))
# initial setup
with st.spinner(text='In progress'):
    model = load_sentence_transformers_model(model_name)

num_replies = st.slider('Select the number of replies:', 2, 20, 5)
num_context_replies = st.slider('Select the number of previous replies considered for the computing the context score:',
                                1, num_replies - 1, 5)

replies = []
score_containers = []
for idx in range(num_replies):
    replies.append(st.text_input('Enter reply  # {}:'.format(len(replies) + 1)))
    score_containers.append(st.empty())
    score_containers[-1].text('-')

recompute_button = st.button('Compute scores')

if recompute_button:
    for idx in range(len(replies)):
        if idx == 0:
            continue
        scores = []
        for reply in replies[:idx][-num_context_replies:]:
            scores.append(string_similarity(reply, replies[idx]))
        avg_score = np.mean(scores)
        score_containers[idx].empty()
        score_containers[idx].text('Average context score: ' + str(avg_score))
