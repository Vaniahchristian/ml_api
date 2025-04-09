import requests
from PIL import Image
import io

# Load and optimize the image
image_path = r'C:\Users\Karim\Desktop\dataset\test\Eczema Photos\03DermatitisArm1.jpg'
img = Image.open(image_path)
img = img.resize((180, 180))  # Resize to match the model's input size
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG', quality=85)
img_byte_arr = img_byte_arr.getvalue()

url = 'https://ml-api-213769180782.us-central1.run.app/predict'

files = {'image': ('03DermatitisArm1.jpg', img_byte_arr, 'image/jpeg')}
response = requests.post(url, files=files)

print('Status Code:', response.status_code)
print('Response:', response.text)
