import streamlit as st
import json
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import finalpreprocess
from langchain.chains.llm import LLMChain

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


paragraphs_df = pd.read_csv('test1.csv') 
paragraphs = paragraphs_df['2'].tolist()
df = pd.read_csv('for_bot.csv')
para = df['2'].tolist()
vectorizer = TfidfVectorizer()
paragraph_vectors = vectorizer.fit_transform(paragraphs)



def get_top_n_relevant_paragraphs(query, n=5):
    query = finalpreprocess(query)
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, paragraph_vectors).flatten()
    top_n_indices = similarity_scores.argsort()[-n:][::-1]
    return top_n_indices



def get_conversational_chain():

    prompt_template = """
    You are a chatbot having a conversation with a human.

    Given the following extracted parts of a long document and a question, create a short final answer.
    If the answer is not in the provided context, just say "False". Don't provide a wrong answer. 
    If the answer is in the context, say "True", return the whole paragraph that contains the answer, 
    and then provide the answer to the question.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Respond with a JSON object like this:
    
    {{
        "Can be answered": "True" or "False", 
        "Answering Paragraph": "paragraph number that can answer the question",
        "Answer": "answer to the question"
    }}

    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)
    

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain



def user_input(user_question):

    ind = get_top_n_relevant_paragraphs(user_question)
    docs = f"""paragraph 1: {para[ind[0]]},
            paragraph 2: {para[ind[1]]},
            paragraph 3: {para[ind[2]]},
            paragraph 3: {para[ind[3]]},
            paragraph 3: {para[ind[4]]}"""

    chain = get_conversational_chain()

    response = chain(
        {"context":docs, "question": user_question}
        , return_only_outputs=True)
    nt=1
    for i in ind:
        rd = f"paragraph {nt} \n: {para[i]}"
        st.write(rd)
        nt=nt+1

    parsed_json = json.loads(response["text"])

        
    st.write("*Here is the response* :sunglasses: :\n",parsed_json)



def main():
    st.set_page_config("Chat")
    st.header("Ask any query from Gemini about the paragraphs in the dataset 💁")

    user_question = st.text_input("Ask a Question from the paragraphs given")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()