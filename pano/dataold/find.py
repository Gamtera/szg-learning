import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

def download_image(image_url, save_path):
    response = requests.get(image_url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
        print(f'Image saved to {save_path}')
    else:
        print('Failed to retrieve image')

def search_and_download_images(query, num_images, save_folder):
    # Create folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    search_url = f'https://www.google.com/search?hl=en&tbm=isch&q={query}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find image URLs
    img_tags = soup.find_all('img')
    img_urls = [img['src'] for img in img_tags[1:num_images+1]]  # Skip the first img tag which is usually not a result image

    for idx, img_url in enumerate(img_urls):
        save_path = os.path.join(save_folder, f'{query}_{idx+1}.jpg')
        download_image(img_url, save_path)

# Example usage
search_and_download_images('geçmeli Yeşil Klemensler', 20, 'görsel bul\Klemensler')
