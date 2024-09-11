import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from PIL import Image
from io import BytesIO
import requests

def download_image(image_url, save_path):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(save_path)
            print(f'Image saved to {save_path}')
        else:
            print('Failed to retrieve image')
    except Exception as e:
        print(f'Error downloading image: {e}')

def search_and_download_images(query, num_images, save_folder):
    # Create folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setup Selenium WebDriver (Assuming Chrome)
    driver = webdriver.Chrome()
    search_url = f"https://www.google.com/search?hl=en&tbm=isch&q={query}"
    driver.get(search_url)

    # Scroll and load more images
    image_urls = set()
    while len(image_urls) < num_images:
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(2)  # Wait for images to load
        thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")

        for img in thumbnails[len(image_urls):]:
            try:
                img.click()
                time.sleep(1)
                large_image = driver.find_element(By.CSS_SELECTOR, "img.n3VNCb")
                src = large_image.get_attribute("src")
                if src and 'http' in src:
                    image_urls.add(src)
                if len(image_urls) >= num_images:
                    break
            except Exception as e:
                print(f"Error clicking image thumbnail: {e}")

    # Download images
    for idx, img_url in enumerate(image_urls):
        save_path = os.path.join(save_folder, f"{query}_{idx+1}.jpg")
        download_image(img_url, save_path)

    driver.quit()

# Example usage
search_and_download_images('car', 25, 'foto')
