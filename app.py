""" Streamlit app tying everything together """

import streamlit as st
from ingest import ingest_query
from vstore import search
from lc_agent import rag_summarize, agent_plan_and_execute
from utils import clean_title

st.set_page_config(page_title='Smart Research Agent (LangChain + LangGraph)', layout='wide')
st.title('Smart Research & Summarization Agent — LangChain + LangGraph')

query = st.text_input('Enter research query/topic')

col1, col2 = st.columns([1, 2])

with col1:
    n = st.number_input('Num papers to ingest', min_value=5, max_value=200, value=50)
    if st.button('Ingest from arXiv'):
        if not query:
            st.error('Enter a query first')
        else:
            with st.spinner('Fetching and ingesting papers...'):
                count = ingest_query(query, n)
            st.success(f'Ingested {count} papers for "{query}"')

with col2:
    k = st.number_input('Top-k retrieve', min_value=1, max_value=10, value=5)
    if st.button('Retrieve & Summarize (RAG)'):
        if not query:
            st.error('Enter a query')
        else:
            with st.spinner('Searching vector DB...'):
                docs = search(query, k=k)
            st.subheader('Top retrieved documents')
            for d in docs:
                st.write(f"**{clean_title(d['title'])}** — {d['url']}")
                st.write(d['abstract'][:600] + ('...' if len(d['abstract']) > 600 else ''))
                st.markdown('---')

            st.info('Generating structured summary...')
            summary = rag_summarize(query, top_k=k)
            st.subheader('Structured summary')
            st.write(summary)

st.markdown('---')
st.header('Agentic tasks (LangGraph-style planner)')

if st.button('Run multi-step planner (summarize + compare + slides)'):
    if not query:
        st.error('Enter a query')
    else:
        with st.spinner('Running planner...'):
            out = agent_plan_and_execute(query, tasks=['summarize', 'compare_methods', 'create_slide_outline'])
        st.subheader('Planner outputs')
        for k, v in out.items():
            st.write(f'**{k}**')
            st.write(v)
            st.markdown('---')
