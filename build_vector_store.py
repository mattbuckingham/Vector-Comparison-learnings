from PIL import ImageFile
import warnings
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext
import qdrant_client
from llama_index.core import SimpleDirectoryReader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from llama_index.core.schema import ImageDocument
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from random import randrange

def main():
    with open("api_key", "r") as f:
        OPENAI_API_TOKEN = f.read().strip()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN

    IMAGE_SEARCH = "./mixed_wiki/" + randrage(108) + ".jpg"

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Create a local Qdrant vector store
    print("Creating local quadrant V/S")
    client = qdrant_client.QdrantClient(path="qdrant_img_db")

    print("text store")
    text_store = QdrantVectorStore(
        client=client, collection_name="text_collection"
    )

    print("image store")
    image_store = QdrantVectorStore(
        client=client, collection_name="image_collection"
    )
    print("storage context")
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )

    # Create the MultiModal index
    print("document")
    documents = SimpleDirectoryReader("./mixed_wiki/").load_data()
    print("multimodal index \n\n")
    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    # generate Text retrieval results
    retriever_engine = index.as_retriever(image_similarity_top_k=10)

    print("Searching for: " + IMAGE_SEARCH )

    # Retrieve more information from the GPT4V response using an image
    retrieval_results = retriever_engine.image_to_image_retrieve("./search_images/8.jpg" )

    print(f"Retrieved {len(retrieval_results)} results.")
    for i, res in enumerate(retrieval_results):
        print(f"Result {i+1}: {res.node.metadata['file_path']} - Score: {res.score}")


    # Extract the file paths of the retrieved images
    retrieved_images = []
    for res in retrieval_results:
        retrieved_images.append(res.node.metadata["file_path"])

    # Remove the first retrieved image as it is the input image
    # since the input image will get the highest similarity score
    retrieved_images = retrieved_images[1:]

    # Plot the retrieved images
    def plot_images(image_paths):
        plt.figure(figsize=(100, 100))  # Adjust the size as needed
        for i, img_path in enumerate(image_paths):
            img = mpimg.imread(img_path)
            plt.subplot(1, len(image_paths), i + 1)  # Create subplots
            plt.imshow(img)
            plt.axis('off')
        plt.show()

    plot_images(retrieved_images)

if __name__ == "__main__":
    main()
