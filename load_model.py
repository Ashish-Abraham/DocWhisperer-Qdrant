from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP

def load_models():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
    model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
    llm = LlamaCPP(
        model_url=model_url,
        temperature=0.1,
        max_new_tokens=256,
        context_window=3900,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 1},
        verbose=True,
    )
    return embed_model, llm
