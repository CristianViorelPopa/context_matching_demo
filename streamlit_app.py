import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np


st.title('Context Matching Demo')


# important to initialize the model once and then predict using it while loaded into memory
# model = SentenceTransformer('paraphrase-distilroberta-base-v1', device='cpu')
# model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v1', device='cpu')


@st.cache(allow_output_mutation=True)
def load_sentence_transformers_model():
    return SentenceTransformer('paraphrase-distilroberta-base-v1', device='cpu')


# initial setup
with st.spinner(text='In progress'):
    model = load_sentence_transformers_model()


# the function for computing the string similarity using the model
def string_similarity(ref, sent):
    ref_embeddings = model.encode(ref, convert_to_tensor=True)
    sent_embeddings = model.encode(sent, convert_to_tensor=True)
    return util.pytorch_cos_sim(ref_embeddings, sent_embeddings).item()


add_reply_button = st.button('Add more replies')
recompute_button = st.button('Recompute scores')

if 'replies' not in st.session_state:
    st.session_state['replies'] = [st.text_input('Enter reply #1:')]

if 'score_containers' not in st.session_state:
    st.session_state['score_containers'] = [st.empty()]
    st.session_state['score_containers'][0].text('-')

# replies = [st.text_input('Enter reply #1:')]
# score_containers = [st.empty()]
# score_containers[0].text('-')

if add_reply_button:
    for idx in range(len(st.session_state['replies'])):
        if idx == 0:
            continue
        scores = []
        for reply in st.session_state['replies'][:idx]:
            scores.append(string_similarity(reply, st.session_state['replies'][idx]))
        avg_score = np.mean(scores)
        st.session_state['score_containers'][idx].empty()
        st.session_state['score_containers'][idx].text('Average context score: ' + str(avg_score))

    st.session_state['replies'].append(st.text_input('Enter reply #{}:'.format(len(st.session_state['replies']) + 1)))
    st.session_state['score_containers'].append(st.empty())
    st.session_state['score_containers'][-1].text('-')
    # del add_reply_button
    # add_reply_button = st.form_submit_button('+')
    # del recompute_button
    # recompute_button = st.form_submit_button('Recompute scores')

    if recompute_button:
        for idx in range(len(st.session_state['replies'])):
            if idx == 0:
                continue
            scores = []
            for reply in st.session_state['replies'][:idx]:
                scores.append(string_similarity(reply, st.session_state['replies'][idx]))
            avg_score = np.mean(scores)
            st.session_state['score_containers'][idx].empty()
            st.session_state['score_containers'][idx].text('Average context score: ' + str(avg_score))


# # user form
# with st.form(key='sentence_transformers_form'):
#     add_reply_button = st.form_submit_button('Add more replies')
#     recompute_button = st.form_submit_button('Recompute scores')
#     replies = [st.text_input('Enter reply #1:')]
#     score_containers = [st.empty()]
#     score_containers[0].text('-')
#
#     # on form submission
#     if add_reply_button:
#         for idx in range(len(replies)):
#             if idx == 0:
#                 continue
#             scores = []
#             for reply in replies[:idx]:
#                 scores.append(string_similarity(reply, replies[idx]))
#             avg_score = np.mean(scores)
#             score_containers[idx].empty()
#             score_containers[idx].text('Average context score: ' + str(avg_score))
#
#         replies.append(st.text_input('Enter reply #{}:'.format(len(replies) + 1)))
#         score_containers.append(st.empty())
#         score_containers[-1].text('-')
#         # del add_reply_button
#         # add_reply_button = st.form_submit_button('+')
#         # del recompute_button
#         # recompute_button = st.form_submit_button('Recompute scores')
#
#     if recompute_button:
#         for idx in range(len(replies)):
#             if idx == 0:
#                 continue
#             scores = []
#             for reply in replies[:idx]:
#                 scores.append(string_similarity(reply, replies[idx]))
#             avg_score = np.mean(scores)
#             score_containers[idx].empty()
#             score_containers[idx].text('Average context score: ' + str(avg_score))

#
#
#
#
#
# st.write('# Gramformer')
#
#
# # initial setup
# with st.spinner(text='In progress'):
#     gf = setup_gramformer()
#
# num_candidates = st.number_input('Number of candidate corrections', min_value=1, max_value=20, value=1,
#                                  format='%d', help='The Gramformer is a generative model that may produce '
#                                                    'more than one correction for the same sentence')
#
# # user form
# with st.form(key='gramformer'):
#     gf_text = st.text_input('Enter your text here:')
#     gf_submit = st.form_submit_button('Correct the text')
#
#     # on form submission
#     if gf_submit:
#         # with st.spinner(text='In progress'):
#         corrections = gf.correct(gf_text, max_candidates=num_candidates)
#
#         st.success('Done! These are the candidate corrections by the Gramformer model:')
#         for idx, correction in enumerate(corrections):
#             st.write(str(idx + 1) + '. ' + correction[0])
