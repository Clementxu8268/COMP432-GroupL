import os
from PIL import Image  # Import necessary libraries


dataset_paths = {
    'Dataset1': {
        'input': '/COMP 432/Project/Dataset 1/Colorectal Cancer',  
        'output': '/COMP 432/Project/Dataset 1/Modified'             
    },
    'Dataset2': {
        'input': '/COMP 432/Project/Dataset 2/Prostate Cancer',   
        'output': '/COMP 432/Project/Dataset 2/Modified'              
    },
    'Dataset3': {
        'input': '//COMP 432/Project/Dataset 3/Animal Faces',        
        'output': '/COMP 432/Project/Dataset 3/Modified'             
    },
}


folder_structure = {
    'Dataset1': ['MUS', 'NORM', 'STR'],         
    'Dataset2': ['gland', 'nongland', 'tumor'], 
    'Dataset3': ['cat', 'dog', 'wild'],         
}


for dataset_name, folders in folder_structure.items():
    for folder in folders:
        folder_path = os.path.join(dataset_paths[dataset_name]['input'], folder)
        output_folder = os.path.join(dataset_paths[dataset_name]['output'], f'{folder}_modified')

        # Create the output folder if it does not exist
        os.makedirs(output_folder, exist_ok=True)

        for file_name in os.listdir(folder_path):
            file = os.path.join(folder_path, file_name)
            if file_name.lower().endswith(('jpeg', 'jpg', 'png', 'bmp', 'tif')):
                try:
                    image = Image.open(file)
                    # Resize the image to 224 x 224
                    modified_image = image.resize((224, 224))

                    # Determine the output file path and format based on dataset
                    if dataset_name == 'Dataset1':
                        output_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.tif')
                        modified_image.save(output_file, format='TIFF')
                    elif dataset_name == 'Dataset2':
                        output_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.tif')
                        modified_image.save(output_file, format='TIFF')
                    elif dataset_name == 'Dataset3':
                        output_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.jpg')
                        modified_image.save(output_file, format='JPEG')

                    print(f"Image '{file_name}' has been successfully modified and saved to: {output_file}")
                except OSError as e:
                    print(f"OSError: Could not process image '{file_name}': {e}")
                except Exception as e:
                    print(f"Could not process image '{file_name}': {e}")
