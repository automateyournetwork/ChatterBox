import os
import json
import aiohttp
import asyncio
import aiofiles
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader,JSONLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Instantiate openAI client
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

class NetboxGiftwrap():
    def __init__(self,
                url,
                token
            ):
        self.url = url
        self.token = token

    def netbox_giftwrap(self):
        asyncio.run(self.main())

    def netbox_api_list(self):
        self.api_list = [
            "/api/ipam/aggregates/",
            "/api/ipam/asns/",
            "/api/dcim/cables/",
            "/api/circuits/circuit-terminations/",
            "/api/circuits/circuit-types/",
            "/api/circuits/circuits/",
            "/api/virtualization/cluster-groups/",
            "/api/virtualization/cluster-types/",
            "/api/virtualization/clusters/",
            "/api/dcim/console-port-templates/",
            "/api/dcim/console-ports/",
            "/api/tenancy/contact-assignments/",
            "/api/tenancy/contact-groups/",
            "/api/tenancy/contact-roles/",
            "/api/tenancy/contacts/",
            "/api/dcim/device-bay-templates/",
            "/api/dcim/device-bays/",
            "/api/dcim/device-roles/",
            "/api/dcim/device-types/",
            "/api/dcim/devices/",
            "/api/dcim/front-port-templates/",
            "/api/dcim/front-ports/",
            "/api/users/groups/",
            "/api/dcim/interface-templates/",
            "/api/dcim/interfaces/",
            "/api/dcim/inventory-items/",
            "/api/ipam/ip-addresses/",
            "/api/ipam/ip-ranges/",
            "/api/dcim/locations/",
            "/api/dcim/manufacturers/",
            "/api/dcim/module-bay-templates/",
            "/api/dcim/module-bays/",
            "/api/dcim/module-types/",
            "/api/dcim/modules/",
            "/api/dcim/platforms/",
            "/api/dcim/power-feeds/",
            "/api/dcim/power-outlet-templates/",
            "/api/dcim/power-outlets/",
            "/api/dcim/power-panels/",
            "/api/dcim/power-port-templates/",
            "/api/dcim/power-ports/",
            "/api/ipam/prefixes/",
            "/api/circuits/provider-networks/",
            "/api/circuits/providers/",
            "/api/dcim/rack-reservations/",
            "/api/dcim/rack-roles/",
            "/api/dcim/racks/",
            "/api/dcim/rear-port-templates/",
            "/api/dcim/rear-ports/",
            "/api/dcim/regions/",
            "/api/ipam/rirs/",
            "/api/ipam/roles/",
            "/api/ipam/route-targets/",
            "/api/ipam/service-templates/",
            "/api/ipam/services/",
            "/api/dcim/site-groups/",
            "/api/dcim/sites/",
            "/api/status/",
            "/api/tenancy/tenant-groups/",
            "/api/tenancy/tenants/",
            "/api/users/tokens/",
            "/api/users/users/",
            "/api/dcim/virtual-chassis/",
            "/api/virtualization/interfaces/",
            "/api/virtualization/virtual-machines/",
            "/api/ipam/vlan-groups/",
            "/api/ipam/vlans/",
            "/api/ipam/vrfs/"
        ]
        return self.api_list        

    async def get_api(self,api_url):
        payload={}
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Token { self.token }',
            }
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.url}{api_url}", headers=headers, data=payload, verify_ssl=False) as resp:
                self.api_count += 1
                responseJSON = await resp.json()
                if api_url == "/api/status/":
                    responseList = responseJSON            
                else:
                    responseList = responseJSON['results']
                    offset = 50
                    total_pages = responseJSON['count'] / 50
                    while total_pages > 1:
                        async with session.get(f"{self.url}{api_url}?limit=50&offset={offset}", headers=headers, data=payload, verify_ssl=False) as resp:
                            self.api_count += 1
                            responseDict = await resp.json()
                            responseList.extend(responseDict['results'])
                            offset = offset +50
                            total_pages = total_pages - 1
                            print(f"{total_pages} pages remaining")
        return(api_url, responseList)

    async def main(self):
        self.api_count = 0
        self.file_count = 0
        api_list = self.netbox_api_list()
        results = await asyncio.gather(*(self.get_api(api_url) for api_url in api_list))
        await self.all_files(json.dumps(results, indent=4, sort_keys=True))

    async def json_file(self, parsed_json):
        for api, payload in json.loads(parsed_json):
            data = {api: payload}  # Create a dictionary with API name as key and payload as value
            wrapped_data = {"info": data }
            async with aiofiles.open(f'{api}.json'.replace("api","").replace("/"," "), 'w') as f:
                await f.write(json.dumps(wrapped_data, indent=4, sort_keys=True))

    async def all_files(self, parsed_json):
        await asyncio.gather(self.json_file(parsed_json))

class Chatterbox:
    def __init__(self):
        self.conversation_history = []
        self.load_text()
        self.split_into_chunks()
        self.store_in_chroma()
        self.setup_conversation_memory()
        self.setup_conversation_retrieval_chain()

    def load_text(self):
        self.loader = DirectoryLoader("", glob='**/*.json', show_progress=True, loader_cls=JSONLoader, loader_kwargs={'text_content': False, 'jq_schema': '.info[]'})
        self.pages = self.loader.load_and_split()

    def split_into_chunks(self):
        # Create a text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        self.docs = self.text_splitter.split_documents(self.pages)

    def store_in_chroma(self):
        embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.docs, embedding=embeddings)
        self.vectordb.persist()

    def setup_conversation_memory(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def setup_conversation_retrieval_chain(self):
        self.qa = ConversationalRetrievalChain.from_llm(llm, self.vectordb.as_retriever(search_kwargs={"k": 25}), memory=self.memory)

    def chat(self, question):
        # Format the user's prompt and add it to the conversation history
        user_prompt = f"User: {question}"
        self.conversation_history.append({"text": user_prompt, "sender": "user"})

        # Format the entire conversation history for context, excluding the current prompt
        conversation_context = self.format_conversation_history(include_current=False)

        # Concatenate the current question with conversation context
        combined_input = f"Context: {conversation_context}\nQuestion: {question}"

        # Generate a response using the ConversationalRetrievalChain
        response = self.qa.invoke(combined_input)

        # Extract the answer from the response
        answer = response.get('answer', 'No answer found.')

        # Format the AI's response
        ai_response = f"NetBox: {answer}"
        self.conversation_history.append({"text": ai_response, "sender": "bot"})

        # Update the Streamlit session state by appending new history with both user prompt and AI response
        st.session_state['conversation_history'] += f"\n{user_prompt}\n{ai_response}"

        # Return the formatted AI response for immediate display
        return ai_response


    def format_conversation_history(self, include_current=True):
        formatted_history = ""
        history_to_format = self.conversation_history[:-1] if not include_current else self.conversation_history
        for msg in history_to_format:
            speaker = "You: " if msg["sender"] == "user" else "Bot: "
            formatted_history += f"{speaker}{msg['text']}\n"
        return formatted_history
    
netbox_url = os.getenv('NETBOX_URL')
netbox_token = os.getenv('NETBOX_TOKEN')

# Page functions
def netbox_data_gathering_page():
    st.title("Gathering JSON data from NetBox")
    if 'data_gathered' not in st.session_state:
        st.session_state['data_gathered'] = False

    if not st.session_state['data_gathered']:
        netbox = NetboxGiftwrap(url=netbox_url, token=netbox_token)
        with st.spinner('Gathering JSON data from NetBox...'):
            st.text("Please wait, this may take a moment, we are gathering up your source of truth...")
            netbox.netbox_giftwrap()  # Assuming this is synchronous
            st.session_state['data_gathered'] = True
        
        st.success("Data successfully gathered! Proceed to the Q&A interface.")

    # Here, ensure that the page state matches what you check in the main app flow.
    if st.button('Proceed to Chat with NetBox'):
        st.session_state['page'] = 'Q&A'  # This must match the check in your main app flow
        st.rerun()

def qa_page():
    st.title("ChatterBox - Talk to Your Netbox")
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = ""

    user_input = st.text_input("Ask a question about NetBox:", key="user_input")
    if st.button("Ask"):
        with st.spinner('Processing...'):
            chat_instance = Chatterbox()
            ai_response = chat_instance.chat(user_input)
            st.session_state['conversation_history'] += f"\nUser: {user_input}\nAI: {ai_response}"
            st.text_area("Conversation History:", value=st.session_state['conversation_history'], height=300, key="conversation_history_display")

# Main app flow
if 'page' not in st.session_state:
    st.session_state['page'] = 'NetBox Data Gathering'

# Ensure this matches the state set in the netbox_data_gathering_page function
if st.session_state['page'] == 'NetBox Data Gathering':
    netbox_data_gathering_page()
elif st.session_state['page'] == 'Q&A':  # This matches the new state set when button is clicked
    qa_page()
