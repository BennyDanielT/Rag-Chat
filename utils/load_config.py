import os
from dotenv import load_dotenv
import yaml
import shutil
from pyprojroot import here
import google.generativeai as genai


class LoadConfig:
    """
    This class loads the configuration file and stores the parameters in the class.
    """

    def __init__(self) -> None:

        with open(here("config/config.yaml")) as confg:
            config = yaml.load(confg, Loader=yaml.FullLoader)
        self.chunk_size = config["splitter_parameters"]["chunk_size"]
        self.chunk_overlap = config["splitter_parameters"]["chunk_overlap"]
        self.embedding_model = config["embedding_config"]["model"]
        self.vectordb_directory = config["vectordb"]["faiss"]["directory"]
        self.system_role = config["prompt"]["system"]
        self.chat_model = config["llm"]["model"]
        self.model_temperature = config["llm"]["temperature"]

    def load_model(self):
        load_dotenv()
        self.GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
        genai.configure(api_key=self.GOOGLE_API_KEY)
