import streamlit as st
from sentence_transformers import SentenceTransformer, util


st.title('Context Matching Demo')


# important to initialize the model once and then predict using it while loaded into memory
model = SentenceTransformer('paraphrase-distilroberta-base-v1', device='cpu')
# model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v1', device='cpu')


# the function for computing the string similarity using the model
def string_similarity(ref, sent):
    ref_embeddings = model.encode(ref, convert_to_tensor=True)
    sent_embeddings = model.encode(sent, convert_to_tensor=True)
    return util.pytorch_cos_sim(ref_embeddings, sent_embeddings).item()


@st.cache(allow_output_mutation=True)
def load_sentence_transformers_model():
    return SentenceTransformer('paraphrase-distilroberta-base-v1', device='cpu')


# initial setup
with st.spinner(text='In progress'):
    model = load_sentence_transformers_model()

# user form
with st.form(key='sentence_transformers_form'):
    lt_text = st.text_input('Enter your text here:')
    lt_submit = st.form_submit_button('Find mistakes')

#     # on form submission
#     if lt_submit:
#         # with st.spinner(text='In progress'):
#         lt_matches = tool.check(lt_text)
#         lt_corrected_text = tool.correct(lt_text)
#
#         st.success('Done! There were ' + str(len(lt_matches)) + ' mistakes found in the text:')
#         for idx, match in enumerate(lt_matches):
#             st.write(str(idx + 1) + '. __' + match.ruleIssueType.upper() + '__: "' + match.message + '"')
#
#         st.write('The corrected text is: __"' + lt_corrected_text + '"__')
#
#         st.write('The raw output from LanguageTool:')
#         st.write(lt_matches)
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
