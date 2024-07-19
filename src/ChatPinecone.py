import os
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.evaluation import load_evaluator


class LineList(BaseModel):
    """Model to represent a list of lines parsed from LLM output."""
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    """Parser to convert LLM output text into a LineList object."""

    def __init__(self) -> None:
        """Initialize the LineListOutputParser with the LineList model."""
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        """Parse the input text into a LineList object.

        Args:
            text (str): The text output from the LLM.

        Returns:
            LineList: An instance of LineList containing the parsed lines.
        """
        lines = text.strip().split("\n")
        return LineList(lines=lines)


class ChatPinecone:
    """Class to handle interactions with the ChatOpenAI and Pinecone services."""

    def __init__(self, pinecone_api_key, openai_api_key):
        """Initialize the ChatPinecone instance.

        Args:
            pinecone_api_key (str): API key for Pinecone.
            openai_api_key (str): API key for OpenAI.
        """
        self.pinecone_api_key = pinecone_api_key
        self.openai_api_key = openai_api_key
        self.prompt = PromptTemplate.from_template(
            """
            You are a bot tasked to answer company questions based on user reviews on our company music app called Spotify.

            You will be given a company question and several user reviews that will likely contain the insight on how to answer the question.

            User Reviews: {context}
            Company Question: {question}

            Behaviour Guideline
            1. Do not ever answer a question if the question is not about our company and cannot be answered using User Reviews.
            2. Always refer your answer based on the user reviews.
            3. If the User Reviews do not contain any insight to the company question, do not answer and either:
                - Ask a follow-up question 
                - Answer the user with apologies that you don't have any answer.
            4. Important! Just output your answer without the prompt/user input.
            5. Your answer tone should be formal, concise, and as detailed as possible.
            Answer:
            """
        )

        os.environ["PINECONE_API_KEY"] = self.pinecone_api_key

        self.llm = ChatOpenAI(
            api_key=self.openai_api_key, model="gpt-4o-mini", temperature=0.0
        )

        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name="spotify-review-small-embed",
            embedding=embedding,
        )

        self.retriever = self.build_retriever()
        self.chain = (
            self.build_standalone_question_chain() | self.build_rag_chain_with_source()
        )
        self.scoring_chain = self.build_scoring_chain()

    def build_retriever(self):
        """Build a retriever that generates user reviews based on company questions.

        Returns:
            MultiQueryRetriever: A retriever that uses the vector store and LLM to generate reviews.
        """
        q_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are a user that asked to help Spotify to review their app.
            You will be given a question from the company, 
            then write 5 examples of review sentences that could answer the company question, make it short and general.
            The example reviews should be as diverse as possible.

            Write the user reviews as best you can. Answer as though you were writing several reviews that could answer the user question. 
            Provide these reviews separated by newlines.
            Company question: {question}""",
        )

        retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(), llm=self.llm, prompt=q_prompt
        )

        return retriever

    def format_docs(self, docs):
        """Format the retrieved documents into a single string.

        Args:
            docs (list): A list of documents to format.

        Returns:
            str: A formatted string containing the content of the documents.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def build_rag_chain_with_source(self):
        """Build a RAG (Retrieval-Augmented Generation) chain that combines context and question.

        Returns:
            RunnableParallel: A runnable that processes the context and question in parallel.
        """
        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: self.format_docs(x["context"]))
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {
                "context": self.retriever,
                "question": RunnablePassthrough(),
            }
        ).assign(output=rag_chain_from_docs)

        return rag_chain_with_source

    def build_standalone_question_chain(self):
        """Build a chain to reformulate questions based on chat history.

        Returns:
            Runnable: A runnable that reformulates the question based on the chat history.
        """
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )

        standalone_question_chain = prompt | self.llm | StrOutputParser()
        return standalone_question_chain

    def build_scoring_chain(self):
        """Build a scoring chain to evaluate the quality of the assistant's answers.

        Returns:
            Evaluator: An evaluator that scores the assistant's answers based on defined criteria.
        """
        hh_criteria = {
            "helpful": "The assistant's answer should be helpful to the user.",
            "harmless": "The assistant's answer should not be illegal, harmful, offensive, or unethical.",
            "hallucinations": "The assistant's answer should come from User Reviews that are given as reference.",
        }

        llm_scoring = ChatOpenAI(
            api_key=self.openai_api_key, model="gpt-4o", temperature=0.0
        )

        evaluator = load_evaluator(
            "score_string", criteria=hh_criteria, llm=llm_scoring
        )
        return evaluator