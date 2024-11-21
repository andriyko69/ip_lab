import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    """Load an image from the specified path."""
    return cv2.imread(image_path)


def calculate_histogram(image):
    """Calculate histograms for each channel of the image."""
    channels = cv2.split(image)
    return [np.histogram(channel.ravel(), bins=256, range=(0, 256))[0] for channel in channels]


def equalize_histogram(image):
    """Equalize the histogram of each channel and merge them."""
    channels = cv2.split(image)
    equalized_channels = [cv2.equalizeHist(channel) for channel in channels]
    return cv2.merge(equalized_channels), calculate_histogram(cv2.merge(equalized_channels))


def apply_roberts_operator(image):
    """Apply Roberts operator on the grayscale image."""
    kernel_roberts_x = np.array([[1, 0], [0, -1]], dtype=int)
    kernel_roberts_y = np.array([[0, 1], [-1, 0]], dtype=int)
    roberts_x = cv2.filter2D(image, -1, kernel_roberts_x)
    roberts_y = cv2.filter2D(image, -1, kernel_roberts_y)
    return cv2.addWeighted(roberts_x, 0.5, roberts_y, 0.5, 0)


def apply_prewitt_operator(image):
    """Apply Prewitt operator on the grayscale image."""
    kernel_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
    kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    prewitt_x = cv2.filter2D(image, -1, kernel_prewitt_x)
    prewitt_y = cv2.filter2D(image, -1, kernel_prewitt_y)
    return cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)


def apply_sobel_operator(image):
    """Apply Sobel operator on the grayscale image."""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.addWeighted(np.abs(sobel_x), 0.5, np.abs(sobel_y), 0.5, 0)


def save_image(image, filename):
    """Save an image to a file."""
    cv2.imwrite(filename, image)


def plot_results(image, hist_original, equalized_image, hist_equalized, roberts, prewitt, sobel):
    """Plot the results including images and histograms."""
    plt.figure(figsize=(16, 18))

    # Original image and its histogram
    plt.subplot(4, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(4, 2, 2)
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        plt.bar(range(256), hist_original[i], color=color, alpha=0.7, label=f'{color} channel')
    plt.xlim(10, 100)
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Count")
    plt.legend()
    plt.grid(True)

    # Equalized image and its histogram
    plt.subplot(4, 2, 3)
    plt.title("Equalized Image")
    plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(4, 2, 4)
    for i, color in enumerate(colors):
        plt.bar(range(256), hist_equalized[i], color=color, alpha=0.7, label=f'{color} channel')
    plt.xlim(10, 100)
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Count")
    plt.legend()
    plt.grid(True)

    # Roberts operator
    plt.subplot(4, 2, 5)
    plt.title("Roberts Operator")
    plt.imshow(roberts, cmap='gray')
    plt.axis('off')

    # Prewitt operator
    plt.subplot(4, 2, 6)
    plt.title("Prewitt Operator")
    plt.imshow(prewitt, cmap='gray')
    plt.axis('off')

    # Sobel operator
    plt.subplot(4, 2, 7)
    plt.title("Sobel Operator")
    plt.imshow(sobel, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    image_path = "lab2_image.jpg"
    image = load_image(image_path)

    # Original histogram
    hist_original = calculate_histogram(image)

    # Histogram equalization
    equalized_image, hist_equalized = equalize_histogram(image)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply operators
    roberts = apply_roberts_operator(gray_image)
    prewitt = apply_prewitt_operator(gray_image)
    sobel = apply_sobel_operator(gray_image)

    # Save processed images
    save_image(equalized_image, "equalized_image.jpg")
    save_image(roberts, "roberts_operator.jpg")
    save_image(prewitt, "prewitt_operator.jpg")
    save_image(sobel, "sobel_operator.jpg")

    # Plot results
    plot_results(image, hist_original, equalized_image, hist_equalized, roberts, prewitt, sobel)


if __name__ == "__main__":
    main()
