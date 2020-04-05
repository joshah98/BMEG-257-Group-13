import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name("coords-83f772b5da97.json", scope)

client = gspread.authorize(creds)

sheet = client.open("coords").sheet1

data = sheet.get_all_records()

print(data)

coords = [15, 10, 5]

for i in range(0,3):
    sheet.update_cell(1,i+1,'joe')
    
    
for i in range(0,3):
    sheet.update_cell(2,i+1,'mama')
    
    
for i in range(0,3):
    sheet.update_cell(3,i+1,'hehe')
    
#cell_list = sheet.range('A1:C1')

#sheet.update_cells(cell_list, coords)