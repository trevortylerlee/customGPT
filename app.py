import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = ''

## Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
description_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

## App
st.title('Streamlit + Langchain + OpenAI')
prompt = st.text_input("Prompt?")

## Prompt template
course_template = PromptTemplate(
    input_variables=['topic', 'wikipedia_research'],
    template="Create a name and course number for a university class about {topic} while also leveraging this wikipedia research: {wikipedia_research}."
)

description_template = PromptTemplate(
    input_variables=['title'],
    template="Create a description and course outline for this course. COURSE: {title}."
)

## Create an instance of the LLM
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=course_template, verbose=True, output_key='title', memory=title_memory)
description_chain = LLMChain(llm=llm, prompt=description_template, verbose=True, output_key='description', memory=description_memory)

wiki = WikipediaAPIWrapper()

if prompt:
    title_input = {'topic': prompt, 'wikipedia_research': ''}
    title = title_chain.run(title_input)
    wiki_research = wiki.run(query=prompt)
    description = description_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(description)

    with st.expander('Title History'):
        st.info(title_memory.buffer)
    
    with st.expander('Description History'):
        st.info(description_memory.buffer)
    
    with st.expander('Wikipedia Research'):
        st.info(wiki_research)