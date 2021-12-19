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

num_context_replies = st.slider('Select the number of previous replies considered for computing the context score:',
                                1, 20, 1)

user_turn = st.radio('The user is ...', ('First', 'Second'))

replies_text_area = st.text_area('Enter replies (one per line):', height=275)

recompute_button = st.button('Compute scores')

if recompute_button:
    replies = replies_text_area.splitlines()

    average_scores = []
    for idx in range(1, len(replies)):
        # Remember: indexing is offset by 1
        if user_turn == 'First' and idx % 2 != 0:
            continue
        if user_turn == 'Second' and idx % 2 == 0:
            continue

        scores = []
        for reply in replies[:idx][-num_context_replies:]:
            scores.append(string_similarity(reply, replies[idx]))
        average_scores.append(np.mean(scores))

    st.write('The total number of replies in the dialog: ' + str(len(average_scores)))
    st.write('The average score for the entire dialog: ' + str(np.mean(average_scores)))
    st.write('The score standard deviation for the entire dialog: ' + str(np.std(average_scores)))
    st.write('The minimum score for the entire dialog: ' + str(np.min(average_scores)))
    st.write('The maximum score for the entire dialog: ' + str(np.max(average_scores)))
