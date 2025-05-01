import download_images
import build_vector_store
import plot_images

def main():
    print("Starting pipeline...")

    download_images.main()
    build_vector_store.main()
    plot_images.main()

    print("Pipeline complete.")

if __name__ == "__main__":
    main()
