import os
import requests
from bs4 import BeautifulSoup

def scrape_tshirt_images(url, destination_folder):
    # Ensure destination folder exists, create it if it doesn't
    os.makedirs(destination_folder, exist_ok=True)

    # Send a GET request to the URL
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return
    
    # Parse the HTML content of the page with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all image tags on the page
    img_tags = soup.find_all('img')
    
    # Filter image tags that seem to be T-shirt images based on their alt text or other attributes
    tshirt_imgs = [img for img in img_tags if 'tshirt' in img.get('alt', '').lower()]
    
    print(f"Found {len(tshirt_imgs)} T-shirt images.")

    # Download each image
    for img in tshirt_imgs:
        img_url = img.get('src')
        if not img_url:
            continue
        
        # Ensure the image URL is absolute
        if not img_url.startswith(('http://', 'https://')):
            img_url = os.path.join(url, img_url)
        
        # Download the image
        img_response = requests.get(img_url)
        if img_response.status_code == 200:
            # Get the image name from the URL
            img_name = os.path.basename(img_url)
            img_path = os.path.join(destination_folder, img_name)
            
            # Save the image to the destination folder
            with open(img_path, 'wb') as img_file:
                img_file.write(img_response.content)
            
            print(f"Downloaded {img_url} to {img_path}")
        else:
            print(f"Failed to download image {img_url}. Status code: {img_response.status_code}")

# Example usage:
url = 'https://www.myntra.com/men-tshirt?rawQuery=men%20tshirt'  # Replace with the actual URL
destination_folder = './tshirt_images'

scrape_tshirt_images(url, destination_folder)
