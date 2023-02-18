import shutil
import os

def delete_directory(dir_path):
    """
    Deletes a directory and all its contents.
    
    Args:
        dir_path (str): The path to the directory to be deleted.
        
    Returns:
        bool: True if the directory was deleted successfully, False otherwise.
    """
    try:
        shutil.rmtree(dir_path)
        if os.path.exists(dir_path):
            return False
        else:
            return True
    except Exception as e:
        print("Error deleting directory:", e)
        return False
