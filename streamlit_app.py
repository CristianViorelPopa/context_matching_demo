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

num_samples = st.slider('Select the number of top/bottom conversation turns by score to show after computation:',
                        0, 100, 3)

user_turn = st.radio('The user is ...', ('First', 'Second'))

computation_type = st.radio('The computation will consider ...', ('Spok & user', 'User-only'))

replies_text_area = st.text_area('Enter replies (one per line):', height=275)

recompute_button = st.button('Compute scores')

if recompute_button:
    replies = replies_text_area.splitlines()

    average_scores = []
    reply_batches = []
    if computation_type == 'Spok & user':
        for idx in range(1, len(replies)):
            # Remember: indexing is offset by 1
            if user_turn == 'First' and idx % 2 != 0:
                continue
            if user_turn == 'Second' and idx % 2 == 0:
                continue

            current_scores = []
            current_replies = []
            for reply in replies[:idx][-num_context_replies:]:
                current_replies.append(reply)
                current_scores.append(string_similarity(reply, replies[idx]))
            current_replies.append(replies[idx])

            average_scores.append(np.mean(current_scores))
            reply_batches.append(current_replies)

    elif computation_type == 'User-only':
        for idx in range(1, len(replies)):
            # Remember: indexing is offset by 1
            if user_turn == 'First' and (idx % 2 != 0 or idx < 2):
                continue
            if user_turn == 'Second' and (idx % 2 == 0 or idx < 3):
                continue

            current_scores = []
            current_replies = []
            if user_turn == 'First':
                starting_index = -2 * num_context_replies
            elif user_turn == 'Second':
                starting_index = -2 * num_context_replies
                if starting_index + idx < 0:
                    starting_index = 1
            else:
                raise RuntimeError('User turn is not first or second')

            for reply in replies[:idx][starting_index::2]:
                current_replies.append(reply)
                current_scores.append(string_similarity(reply, replies[idx]))
            current_replies.append(replies[idx])

            average_scores.append(np.mean(current_scores))
            reply_batches.append(current_replies)

    st.write('## General statistics')
    st.write(f'The total number of turns in the dialog: **{len(average_scores)}**')
    st.write(f'The average score for the entire dialog: **{np.mean(average_scores)}**')
    st.write(f'The score standard deviation for the entire dialog: **{np.std(average_scores)}**')
    st.write(f'The minimum score for the entire dialog: **{np.min(average_scores)}**')
    st.write(f'The maximum score for the entire dialog: **{np.max(average_scores)}**')

    if num_samples > 0:
        st.write('## Top/bottom conversation turns by score')
        average_scores = np.array(average_scores)
        reply_batches = np.array(reply_batches)

        sorted_indices = np.argsort(average_scores)
        st.write(f'#### The top {num_samples} conversation turns by score:')
        for idx in range(min(len(average_scores), num_samples)):
            output = ''
            output += f'{idx + 1}.'
            for reply in reply_batches[sorted_indices][-idx - 1]:
                output += f'\t{reply}\n\n'
            output += f'\tAverage score: {average_scores[sorted_indices][-idx - 1]}'
            st.write(output)

        st.write(f'#### The bottom {num_samples} conversation turns by score:')
        for idx in range(min(len(average_scores), num_samples)):
            output = ''
            output += f'{idx + 1}.'
            for reply in reply_batches[sorted_indices][idx]:
                output += f'\t{reply}\n\n'
            output += f'\tAverage score: {average_scores[sorted_indices][idx]}'
            st.write(output)
