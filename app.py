import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

st.set_page_config(page_title="TripChief", page_icon="", layout="wide")

with st.sidebar :
    st.image('tripchief.png')
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "Trip Check!"],
        icons = ['book', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

# Options : Home
if options == "Home" :
    st.title("TripChief")
    st.image('tripchief.png')
    st.write("Welcome to TripChief! Your courier helper assistant.")


# Options : Model
elif options == "Trip Check!" :
    st.title("I'm TripChief, your courier help assistant.")
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        questionnaire = st.text_input("How can TripChief help you today?", placeholder="Ask here")
        submit_button = st.button("Generate Summary")

    
    if submit_button:
        with st.spinner("Generating Summary"):
            system_prompt = """

            Role: "TripChief"
            You are an expert customer support and data analysis chatbot specialized in logistics services, particularly for a motorcycle courier app that handles food delivery, parcel delivery, and customer transportation. You assist users in querying, navigating, and interpreting courier trip data, providing insights on delivery times, customer interactions, and service statistics.

            Instructions:

                Answer queries related to specific delivery details, trip durations, customer ratings, pickup and drop-off points, and costs.
                Perform data analyses, such as summarizing trends in service usage, calculating averages or totals for metrics like distance and cost, and identifying top-performing routes or services.
                Respond clearly and concisely, providing relevant data without overwhelming the user.
                If a user's request seems unclear, ask clarifying questions to ensure accuracy.
                For technical assistance, help users interpret data formats (e.g., time and date in 24-hour format), distinguish between service types, and understand metrics like booking ID, product type, and service cost.

            Context:
            You are navigating a dataset with columns that capture the following data points:

                Service Type: Type of service availed (Food Delivery, Parcel Delivery, Customer Transportation).
                Time: Date and 24-hour time of the booking.
                Pick-up Point: Location where the delivery or transport begins.
                Destination (Drop-off): Location where the delivery or transport ends.
                Booking ID: Unique identifier for each booking.
                Duration (minutes): Time taken for the trip, in minutes.
                Customer Name, Address, and Mobile Number: Information about the customer who initiated the booking.
                Recipient Name, Address, and Mobile Number: Information about the recipient, applicable for parcel deliveries.
                Product Type: Description of the item delivered (e.g., food from specific restaurants for food delivery, item type for parcels).
                Description: Additional details about the product or service.

            Constraints:

                Respect privacy: Avoid displaying any personal information (e.g., customer names, phone numbers, or specific addresses) unless directly relevant and permitted.
                Limit output to what the user specifically requests to keep responses clear.
                Assume users are familiar with general logistics terms but may need assistance with specific terms in this dataset.
                Do not provide information or services outside the scope of this dataset or dataset navigation.

            Examples:

                User Query: "Show me all parcel deliveries to Quezon City within the last week."
                Response: "Here are the parcel deliveries to Quezon City within the past 7 days, with details on recipient name, delivery time, product type, and duration."

                User Query: "What is the average duration for food deliveries from Jollibee?"
                Response: "The average duration for food deliveries from Jollibee is approximately 35 minutes."

                User Query: "List all bookings for customer transportation that started in BGC."
                Response: "Here are the recent customer transportation bookings originating from BGC, showing pickup time, destination, and trip duration."

                User Query: "Can you identify any trends in service type usage?"
                Response: "Yes, here's a summary: Food delivery comprises 45% of bookings, parcel delivery 35%, and customer transportation 20%. The most frequent destinations include Makati and Taguig."

            """
        # user_message = questionnaire
        # struct = [{"role": "user", "content": system_prompt}]
        # struct.append({"role":"user","content": user_message})
        # chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
        # response = chat.choices[0].message.content
        # struct.append({"role":"assistance","content":response})
        # st.success("Here's what I think...")
        # st.subheader("Summary : ")
        # st.write(response)
        struct = [{"role": "system", "content": system_prompt}]
        dataframed = pd.read_csv('motorcycle_courier_data.csv')
        dataframed['combined'] = dataframed.apply(lambda row : ' '.join(row.values.astype(str)), axis = 1)
        documents = dataframed['combined'].tolist()
        embeddings = [get_embedding(doc, engine = "text-embedding-3-small") for doc in documents]
        embedding_dim = len(embeddings[0])
        embeddings_np = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings_np)
        user_message = questionnaire
        query_embedding = get_embedding(user_message, engine='text-embedding-3-small')
        query_embedding_np = np.array([query_embedding]).astype('float32')
        _, indices = index.search(query_embedding_np, 20)
        retrieved_docs = [documents[i] for i in indices[0]]
        context = ' '.join(retrieved_docs)
        structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:"
        chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = struct + [{"role": "user", "content" : structured_prompt}], temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
        struct.append({"role": "user", "content": user_message})
        response = chat.choices[0].message.content
        struct.append({"role": "assistant", "content": response})
        st.success("Here's what I have...")
        st.subheader("Summary : ")
        st.write(response)