"""
This module contains the QuestionAnswering class, which is used to generate answers to questions based on a given context.

The QuestionAnswering class has the following methods:

* __init__: Initializes the QuestionAnswering class.
* get_answer: Generates an answer to a given question based on a given context.
* get_prompt: Returns the prompt used by the QuestionAnswering class.
* get_chain: Returns the chain used by the QuestionAnswering class.
"""

from langchain_core.prompts import PromptTemplate
from utils.load_config import LoadConfig


class QuestionAnswering:
    """
    This class is used to generate answers to questions based on a given context.
    """

    def __init__(self):
        """
        Initializes the QuestionAnswering class.
        """

        template = """Use the following pieces of context to answer the question at the end. Elaborate as much as possible. Refer the given context to determine the answer.
        Context:

        {context}

        Question: {question}

        Helpful Answer:"""
        self.custom_rag_prompt = PromptTemplate.from_template(template)
        # self.prompt_template = PromptTemplate(
        #     template=CFG.system_role,
        #     input_variables=["context", "question"],
        # )

        # self.model = ChatGoogleGenerativeAI(
        #     model=CFG.chat_model, temperature=CFG.model_temperature
        # )
        # self.qa_chain = load_qa_chain(
        #     model=self.model, chain_type="stuff", prompt=self.prompt_template
        # )

    # def get_answer(self, question, context):
    #     """
    #     Generates an answer to a given question based on a given context.

    #     Args:
    #         question (str): The question to be answered.
    #         context (str): The context in which the question is asked.

    #     Returns:
    #         str: The answer to the question.
    #     """
    #     return self.qa_chain.run(input_documents=[context], question=question)

    def get_prompt(self):
        """
        Returns the prompt used by the QuestionAnswering class.

        Returns:
            str: The prompt.
        """
        return self.custom_rag_prompt

    # def get_chain(self):
    #     """
    #     Returns the chain used by the QuestionAnswering class.

    #     Returns:
    #         str: The chain.
    #     """
    #     return self.qa_chain
