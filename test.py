import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name("coords-83f772b5da97.json", scope)

client = gspread.authorize(creds)

sheet = client.open("coords").sheet1

data = sheet.get_all_records()

print(data)

coords = [10, 10, 10]

sheet.insert_row(coords ,2)

data2 = sheet.get_all_records()

print(data2)