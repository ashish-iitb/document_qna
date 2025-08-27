import os
from typing import Dict, List
from pypdf import PdfReader
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
import json
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),base_url = os.getenv("OPENAI_BASE_URL") )
def read_pdfs_from_folder(folder_path: str) -> Dict[str, str]:
    """
    Reads all PDF files from the given folder (non-recursive) and extracts text.

    Args:
        folder_path (str): Path to the folder containing PDF files.

    Returns:
        Dict[str, str]: Dictionary with filenames as keys and extracted text as values.
    """
    pdf_texts = {}

    # Ensure folder exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"Provided path {folder_path} is not a directory.")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                pdf_texts[filename] = text.strip()
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return pdf_texts


def recursive_character_split(
    pdf_texts: Dict[str, str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: List[str] = ["\n\n", "\n", " ", ""]
) -> Dict[str, List[str]]:
    """
    Splits text content from PDFs into chunks using a recursive character splitting strategy,
    similar to LangChain's RecursiveCharacterTextSplitter.

    Args:
        pdf_texts (Dict[str, str]): Dictionary with filenames as keys and full text as values.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
        separators (List[str]): Ordered list of separators to try splitting on.

    Returns:
        Dict[str, List[str]]: Dictionary with filenames as keys and list of text chunks as values.
    """

    def split_text(text: str, chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[str]:
        # Base case: if text fits in chunk_size, return it
        if len(text) <= chunk_size:
            return [text]

        # Try to split by separators in order
        for sep in separators:
            if sep == "":  # fallback to character-level split
                splits = list(text)
            else:
                splits = text.split(sep)

            if len(splits) == 1:
                continue  # couldn't split with this separator

            chunks = []
            current_chunk = ""

            for part in splits:
                if current_chunk and len(current_chunk) + len(part) + len(sep) > chunk_size:
                    chunks.append(current_chunk)
                    # Keep overlap
                    overlap = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
                    current_chunk = overlap + part + sep
                else:
                    current_chunk += part + sep

            if current_chunk:
                chunks.append(current_chunk)

            # Recursively ensure each chunk fits the size
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > chunk_size:
                    final_chunks.extend(split_text(chunk, chunk_size, chunk_overlap, separators[1:]))
                else:
                    final_chunks.append(chunk)

            return final_chunks  # ✅ return after successful split

        return [text]  # fallback

    # Process all PDFs
    chunked_texts = {}
    for filename, text in pdf_texts.items():
        chunked_texts[filename] = split_text(text, chunk_size, chunk_overlap, separators)

    return chunked_texts




def get_embedding(text: str, model: str = "text-embedding-3-small"):
    """
    Generate embedding for a given text using OpenAI API.
    """
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def load_or_create_collection(
    chunked_texts: Dict[str, List[str]] = None,
    persist_dir: str = "./chroma_db",
    collection_name: str = "pdf_chunks"
):
    """
    Load persisted ChromaDB collection if available.
    If not, create collection and store embeddings.

    Args:
        chunked_texts (Dict[str, List[str]], optional):
            Required only when creating collection for the first time.
        persist_dir (str): Directory where Chroma DB persists data.
        collection_name (str): Name of the ChromaDB collection.
    """
    client_chroma = chromadb.PersistentClient(path=persist_dir)

    try:
        # Try loading existing collection
        collection = client_chroma.get_collection(name=collection_name)
        print(f"✅ Loaded existing collection: {collection_name}")
        return collection

    except Exception:
        # Collection not found, must create
        if not chunked_texts:
            raise ValueError("No collection found. Please provide chunked_texts to create one.")

        collection = client_chroma.get_or_create_collection(name=collection_name)

        ids, documents, metadatas, embeddings = [], [], [], []
        i = 0

        for filename, chunks in chunked_texts.items():
            for chunk in chunks:
                ids.append(f"{filename}_{i}")
                documents.append(chunk)
                metadatas.append({"source": filename})
                embeddings.append(get_embedding(chunk))  # ✅ using your embedding function
                i += 1

        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"✅ Created and stored {len(documents)} chunks into ChromaDB (collection: {collection_name})")
        return collection


def get_embedding(text: str, model: str = "text-embedding-3-small"):
    """
    Generate embedding for a given text using OpenAI API.
    """
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def query_collection(
    query: str,
    chat_history: list=  [],
    collection_name: str = "pdf_chunks", 
    persist_dir: str = "./chroma_db", # if after first run  , you want to add more pdfs or less pdfs then the first run , then rename this db name to avoid any mix up
    top_k: int = 5
):
    """
    Query the ChromaDB collection, combine with conversation history,
    and return GPT-5 answer.
    """
    # Load ChromaDB client + collection
    client_chroma = chromadb.PersistentClient(path=persist_dir)
    collection = client_chroma.get_collection(name=collection_name)

    # Get embedding of query
    query_embedding = get_embedding(query)

    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_docs = results["documents"][0]
    context = "\n\n".join(retrieved_docs)

    return context
    
def llm_query (user_prompt,system_prompt): 

  # Ask GPT-5
  response = client.chat.completions.create(
        model="gpt-5",
        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    )
  return response.choices[0].message.content

def ask_question(query,top_k , chat_history= [],):
  context = query_collection(query, chat_history,top_k = top_k)
  system_prompt = """ You are an assistant that must answer ONLY from context from PDFs knowledgebase and previous conversation history if available. 
                  - If the answer is clearly and explicitly stated in the context, provide it directly.  
                  - If the answer is missing, ambiguous, or cannot be determined with certainty, respond with exactly: "not found".  
                  - Do NOT guess, assume, or generate extra information beyond what is given in the context.  
                  - If multiple possible answers exist, respond with "not found" 
                  -  if  format is mention for the naswe then stricly follow that format
                  """

  # Build messages with history
  system_prompt = system_prompt

   
  user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer accurately using the context above and conversation history if available"
  user_prompt = f"""
                    You are a helpful, knowledgeable, and conversational AI assistant. 
                    Use the provided context ,conversation history(if available) and the current user query to give the most accurate, 
                    context-aware, and human-like response.

                    conversation history:
                    {chat_history}

                    Context:
                    {context}

                    User Query:
                    {query}

                    Instructions:
                    - Answer in a natural, conversational tone.  
                    - Always consider the context from the history when responding.  
                    - If the answer is not directly available in context, strictly follow system prompt  
                    - Keep responses concise and to the point do not provide any suffix or prefix in the answer.  
                    """
  
  answer =  llm_query (user_prompt,system_prompt)
  return answer


def perform_collection_loading(source_folder,vector_db_perists_dir,collection_name,chunk_size,chunk_overlap):
  pdf_texts = read_pdfs_from_folder(source_folder)
  chunks = recursive_character_split(pdf_texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  # Load existing collection or create new one
  collection = load_or_create_collection(chunks, persist_dir=vector_db_perists_dir, collection_name= collection_name)
  return True


if __name__ == "__main__":
    source_folder = "./pdfs"
    vector_db_perists_dir = "./chroma_db"
    collection_name = "pdf_chunks"
    chunk_size=800  # character length of each chuncks
    chunk_overlap=100 # overlap between two consecutive chucks
    top_K_context =20  # top N relevant doc fecthed from vector db

    perform_collection_loading(source_folder,vector_db_perists_dir,collection_name,chunk_size,chunk_overlap)

    chat_history = []  # stores conversation memory
    

    # get responses : 
    # Define prompts for each field
    trustName = ""
    trustee_name = ""
    settlor_name = ""
    appointor_name  = ""

    trustName_prompt = "What is the name of the trust, provide the exact name only?"

    basicInformation_dateOfDeed = "On what date was the trust deed executed? (Format: YYYY-MM-DD)"
    basicInformation_settledSum = "What is the settled sum (initial amount) for the trust?"
    basicInformation_governingLaw = "Which jurisdiction’s law governs this trust?"

    trustTerm_commencementDate = "What is the commencement date of the trust? (Format: YYYY-MM-DD)"
    trustTerm_terminationDate = "What is the termination date or vesting day of the trust?"

    parties_settlor_name = f"who is the settlor of the trust: {trustName} ?"
    parties_settlor_address = f"What is the residential or business address of the settlor of trust {parties_settlor_name}?"
    parties_settlor_restrictions = f"Are there any restrictions on the settlor {parties_settlor_name} (e.g., cannot benefit from the trust) of trust {trustName}?"

    parties_trustee_name = f"who is the trustee of the trust : {trustName} ?"
    parties_trustee_acn = f"What is the ACN (Australian Company Number) of the trustee {parties_trustee_name} ?"
    parties_trustee_address = f"What is the address of the trustee {parties_trustee_name}?"
    parties_trustee_director = f"Who is the director (or key contact person) of the trustee {parties_trustee_name}?"

    parties_appointor_name = f"Who is the the appointor of trust {trustName} , if more than one appointers are there then provide the answer in string having all the appointers seperated by commas, do not provide any suffix or prefix ?"
    parties_appointor_powers = f"What powers does the appointor {parties_appointor_name} have? (e.g., remove trustees, appoint new ones, consent for actions, request auditor), provide the andwer in paragraph format without suffix, prefix or  any bullet points or sepcial character"


    ## Now call ask_question on each prompt (your required style)

    print("data fetching begins")

    trustName = ask_question(trustName_prompt, top_K_context)

    basicInformation_dateOfDeed = ask_question(basicInformation_dateOfDeed ,top_K_context)
    basicInformation_settledSum = ask_question(basicInformation_settledSum ,top_K_context)
    basicInformation_governingLaw = ask_question(basicInformation_governingLaw ,top_K_context)

    trustTerm_commencementDate = ask_question(trustTerm_commencementDate ,top_K_context)
    trustTerm_terminationDate = ask_question(trustTerm_terminationDate ,top_K_context)

    parties_settlor_name = ask_question(trustName_prompt.format(trustName=trustName) ,top_K_context)
    parties_settlor_address = ask_question(parties_settlor_address.format(parties_settlor_name=parties_settlor_name) ,top_K_context)
    parties_settlor_restrictions = ask_question(parties_settlor_restrictions.format(trustName=trustName,parties_settlor_name=parties_settlor_name),top_K_context)

    parties_trustee_name = ask_question(parties_trustee_name.format(trustName=trustName) ,top_K_context)
    parties_trustee_acn = ask_question(parties_trustee_acn.format(parties_trustee_name=parties_trustee_name) ,top_K_context)
    parties_trustee_address = ask_question(parties_trustee_address.format(parties_trustee_name=parties_trustee_name) ,top_K_context)
    parties_trustee_director = ask_question(parties_trustee_director.format(parties_trustee_name=parties_trustee_name) ,top_K_context)

    parties_appointor_name  = ask_question(parties_appointor_name.format(trustName=trustName,parties_appointor_name=parties_appointor_name) ,top_K_context)
    parties_appointor_powers = ask_question(parties_appointor_powers.format(parties_appointor_name=parties_appointor_name) ,top_K_context)

    print("data fetching concludes")

    # print(trustName)
    # print(basicInformation_dateOfDeed)
    # print(basicInformation_settledSum)
    # print(basicInformation_governingLaw)
    # print(trustTerm_commencementDate)
    # print(trustTerm_terminationDate)
    # print(parties_settlor_name)
    # print(parties_settlor_address)
    # print(parties_settlor_restrictions)
    # print(parties_trustee_name)
    # print(parties_trustee_acn)
    # print(parties_trustee_address)
    # print(parties_trustee_director)
    # print(parties_appointor_name)
    # print(parties_appointor_powers)

    basicInformation_info = {
        "dateOfDeed": basicInformation_dateOfDeed,
        "settledSum": basicInformation_settledSum,
        "governingLaw": basicInformation_governingLaw
    }

    trustTerm_info = {
        "commencementDate": trustTerm_commencementDate,
        "terminationDate": trustTerm_terminationDate
    }

    parties_info = {
        "settlor": {
            "name": parties_settlor_name,
            "address": parties_settlor_address,
            "restrictions": parties_settlor_restrictions
        },
        "trustee": {
            "name": parties_trustee_name,
            "acn": parties_trustee_acn,
            "address": parties_trustee_address,
            "director": parties_trustee_director
        },
        "appointor": {
            "name": parties_appointor_name,
            "powers": [parties_appointor_powers]
        }
    }

    # Output JSON
    trust_json = {
        "trustName": trustName,
        "basicInformation": basicInformation_info,
        "trustTerm": trustTerm_info,
        "parties": parties_info
    }
    
    print("printing output json")
    print(json.dumps(trust_json, indent=4))

    # Save to JSON file
    with open("trust_info.json", "w", encoding="utf-8") as f:
        json.dump(trust_json, f, indent=4, ensure_ascii=False)



    #######################  uncomment below block to have conversational chat   ############
    # #chat experience :
    # chat_history = []
    # perform_collection_loading(source_folder,vector_db_perists_dir,collection_name,chunk_size,chunk_overlap)
    # while True:
    #     query = input("\n🔍 Enter your query (or 'exit' to quit): ")
    #     if query.lower() == "exit":
    #         break

    #     answer = ask_question(query,top_k= top_K_context,chat_history=  chat_history)

    #     # Save conversation to memory
    #     chat_history.append({"role": "user", "content": query})
    #     chat_history.append({"role": "assistant", "content": answer})

    #     print(f"\n🤖 Answer: {answer}")
