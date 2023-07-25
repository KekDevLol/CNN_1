import os

def rename_images(folder_path):
    count = 5000
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            new_filename = str(count) + os.path.splitext(filename)[1]
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
            count += 1


folder_path = 'train'


rename_images(folder_path)
print('Готово')