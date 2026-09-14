import os
import json
import gdown
import requests
import mimetypes
from typing import Tuple, Optional
from Utils.ConfigUtils import ConfigUtils

class GoogleUtils:

    def __init__(self, folder_id: str = None):

        config = ConfigUtils()

        REFRESH_TOKEN = config.get("google.REFRESH_TOKEN")
        CLIENT_SECRET = config.get("google.CLIENT_SECRET")
        PROJECT_ID = config.get("google.PROJECT_ID")

        self.client_json = {"web":{
                "client_id":"138516851437-86cqss3urm46bn5i08loq240i3g03o69.apps.googleusercontent.com",
                "project_id":PROJECT_ID,
                "auth_uri":"https://accounts.google.com/o/oauth2/auth",
                "token_uri":"https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs",
                "client_secret":CLIENT_SECRET,
                "redirect_uris":["https://developers.google.com/oauthplayground"],
                "javascript_origins":["https://developers.google.com"],
                "refresh_token": REFRESH_TOKEN}}
        
        self.headers = self.set_headers()
        self.folder_id = config.get("google.FOLDER_ID") if folder_id is None else folder_id

    def set_headers(self):
        payload = {
            'client_id': self.client_json['web']['client_id'],
            'client_secret': self.client_json['web']['client_secret'],
            'refresh_token': self.client_json["web"]['refresh_token'],
            'grant_type': 'refresh_token',
        }

        response = requests.post(self.client_json["web"]['token_uri'], data=payload)
        token_data = response.json()

        if 'access_token' in token_data:
            access_token = token_data['access_token']
        else:
            raise Exception(f"Failed to refresh access token: {token_data}")
        return {
            "authorization": f"Bearer {access_token}",
        }

    def download_file(self, file_id: str, file_name: str) -> Tuple[requests.Response, Optional[str]]:
        """
        Downloads a file from Google Drive.
        """
        try:
            response = requests.get(
                f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media",
                headers=self.headers
            )
            if response.status_code == 200:
                file_path = os.path.join(os.getcwd(), file_name)
                with open(file_path, "wb") as f:
                    f.write(response.content)
                return response, file_path
            else:
                return response, None
        except Exception as e:
            return e, None

    def download_file_by_ids(self, ids: list) -> None:
        """
        Downloads a list of files from Google Drive.
        """
        for id in ids:
            gdown.download(id=id, quiet=False)

    import os
    import requests

    def download_folder(self, folder_id: str, output_dir: str) -> None:
        """
        Download all the files in a google drive folder and subfolder up to 1 lvl
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            url = f"https://www.googleapis.com/drive/v3/files?q='{folder_id}'+in+parents&fields=files(id,name,mimeType)"
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                files = response.json().get('files', [])
                for file in files:
                    file_id = file['id']
                    file_name = file['name']
                    mime_type = file['mimeType']
                    if mime_type == 'application/vnd.google-apps.folder':
                        subfolder_path = os.path.join(output_dir, file_name)
                        self.download_folder(file_id, subfolder_path)
                    else:
                        file_response, file_path = self.download_file(file_id, os.path.join(output_dir, file_name))
                        print(f"{file_path} Downloaded")
            else:
                return None, response.status_code
        except Exception as e:
            return e, None


    def upload_file(self, file_name: str, file_path: str, drive_folder_id: str) -> requests.Response:
        """
        Uploads a file to Google Drive.
        """
        try:
            params = {
                "name": file_name,
                "parents": [drive_folder_id]
            }
            if not os.path.exists(file_path):
                raise Exception("File not found")
            file_type = mimetypes.guess_type(file_path)[0]
            files = {
                'data': ('metadata', json.dumps(params), 'application/json;charset=UTF-8'),
                'file': (file_name, open(file_path, 'rb'), file_type)
            }
            res = requests.post(
                'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart',
                headers=self.headers,
                files=files
            )
            return res.json()['id']
        except Exception as e:
            return e

    def delete_file(self, file_id: str) -> requests.Response:
        """
        Deletes a file from Google Drive.
        """
        response = requests.delete(
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            headers=self.headers
        )
        return response

    def create_folder(self, folder_name: str, parent_folder_id: str) -> requests.Response:
        """
        Creates a folder in Google Drive.
        """
        folder_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_folder_id]
        }

        response = requests.post(
            "https://www.googleapis.com/drive/v3/files",
            headers=self.headers,
            data=json.dumps(folder_metadata)
        )
        return response

    def delete_folder(self, folder_id: str) -> requests.Response:
        """
        Deletes a folder from Google Drive.
        """
        headers_copy = self.headers
        headers_copy["Content-Type"] = "application/json"
        response = requests.delete(
            f"https://www.googleapis.com/drive/v3/files/{folder_id}",
            headers=headers_copy
        )
        return response

    def search_file(self, file_name: str) -> requests.Response:
        """
        Searches for a file in Google Drive.
        """
        response = requests.get(
            "https://www.googleapis.com/drive/v3/files",
            headers= self.headers,
            params={"q": f"name='{file_name}'"}
        )
        return response

    def upload_folder(self, local_folder_path: str, parent_folder_id: str = '1IlzJon8PASm_x9Pl8BS3outOBY0w2N9y') -> None:
        """
        Uploads an entire folder from local machine to Google Drive.
        """
        base_folder_name = os.path.basename(local_folder_path)
        folder_id = self.create_folder(base_folder_name, parent_folder_id).json().get('id')
        for root, dirs, files in os.walk(local_folder_path):
            print("Uploading folder", root)
            # Create the folder structure in Google Drive
            relative_path = os.path.relpath(root, local_folder_path)
            if relative_path == '.':
                relative_path = ''

            # Create subfolders in Google Drive
            for dir_name in dirs:
                if 'Neural-Spell-Checker' in dir_name or dir_name.startswith('.'):
                    continue
                self.upload_folder(os.path.join(root, dir_name), folder_id)

            # Upload files to the current folder
            for file_name in files:
                file_path = os.path.join(root, file_name)
                #if file type is log skip it
                if file_name.endswith('.log'):
                    continue
                self.upload_file(file_name, file_path, folder_id)
                print(file_name, "uploaded")
            # Break after the first iteration to avoid uploading the same folder multiple times
            break