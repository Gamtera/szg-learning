from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import os
import urllib.request

# ChromeDriver yolunu buraya girin
chrome_driver_path = "/Users/burakbuz/Downloads/chromedriver-mac-arm64"

# İndirmek istediğiniz görsel sayısı ve arama terimi
query = "kedi"
num_images = 20

# WebDriver başlat
driver = webdriver.Chrome(executable_path=chrome_driver_path)
driver.get(f"https://www.google.com/search?q={query}&source=lnms&tbm=isch")

# Ekranda aşağı kaydırarak daha fazla görsel yükle
last_height = driver.execute_script("return document.body.scrollHeight")

while len(driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")) < num_images:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        try:
            driver.find_element(By.CSS_SELECTOR, ".mye4qd").click()
        except:
            break
    last_height = new_height

# İndirilecek klasörü oluştur
save_folder = "downloads"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Görselleri indir
images = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")
for index, image in enumerate(images):
    if index >= num_images:
        break
    try:
        image_url = image.get_attribute("src")
        if image_url:
            urllib.request.urlretrieve(image_url, os.path.join(save_folder, f"{query}_{index + 1}.jpg"))
            print(f"{index + 1}. görsel indirildi.")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

driver.quit()
print("İndirme işlemi tamamlandı!")
