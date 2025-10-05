""" LangChain agent + LangGraph planner usage with Groq """

from typing import List
import os
from langchain import LLMChain, PromptTemplate
from groq_llm import GroqLLM
from vstore import search
from utils import build_system_prompt, build_user_prompt

# Always use Groq
llm = GroqLLM(model="qwen/qwen3-32b")


def rag_summarize(query: str, top_k: int = 5):
    docs = search(query, k=top_k)
    user_prompt = build_user_prompt(
        query,
        [{'title': d['title'], 'url': d['url'], 'snippet': d['abstract']} for d in docs]
    )
    system_prompt = build_system_prompt()

    prompt = PromptTemplate(
        input_variables=['context'],
        template=system_prompt + "\n\n{context}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain.run(user_prompt)


def agent_plan_and_execute(query: str, tasks: List[str]):
    """ tasks: e.g. ['summarize', 'compare_methods', 'create_slide_outline'] """
    plan_results = {}

    if 'summarize' in tasks:
        plan_results['summary'] = rag_summarize(query, top_k=5)

    if 'compare_methods' in tasks:
        docs = search(query, k=6)
        compare_prompt = "Compare methods used across these papers and list pros/cons: \n"
        for i, d in enumerate(docs):
            compare_prompt += f"{i + 1}. {d['title']} | {d['url']}\n{d['abstract'][:600]}\n\n"
        plan_results['methods_comparison'] = llm._call(compare_prompt)

    if 'create_slide_outline' in tasks:
        outline_prompt = f"Create an 8-slide presentation outline for: {query} based on the summary. Provide bullets per slide."
        plan_results['slides'] = llm._call(outline_prompt)

    return plan_results
