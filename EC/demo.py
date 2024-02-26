from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import yaml


def load_config():
    """
    Load configuration from the 'config.yaml' file.
 
    Returns:
        dict: Configuration settings.
    """
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()


def get_embeddings(model_name=config["embeddings"]["name"],
                    model_kwargs={'device': config["embeddings"]["device"]}):
    """
    Load HuggingFace embeddings.
 
    Args:
        model_name (str): The name of the HuggingFace model.
        model_kwargs (dict): Keyword arguments for the model.
 
    Returns:
        HuggingFaceEmbeddings: Embeddings model.
    """
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

embeddings = get_embeddings()
print(embeddings)