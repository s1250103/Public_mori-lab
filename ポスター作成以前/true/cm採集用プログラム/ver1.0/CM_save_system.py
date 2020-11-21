## ver1.0
## 1. input one cm.
## 2. calculate the hash value.
## 3. listen information for the database.
## 4. write to the database

import hashlib
import sqlite3

PATH = './CMdata_ts/'
FORMAT = '.ts'

def read_data():
    ## get the name, you want to record to database.
    ## and unite from a path of the directory, which in CMs.
    ## so calculates the hash
    ##
    ##
    ## no arguments
    ## return: name, hash value.

    print("What file, you want write to database?")
    print("The file name: ", end='')
    name = input()
    path = PATH + name + FORMAT

    with open(path, 'rb') as f: # read, binary.
        data = f.read()

    h = hashlib.sha256()
    h.update(data)
    h = h.hexdigest()

    return name, h
def listen_meta():
    print("What is the title?: ", end='')
    title = input()
    while(title == ''):
        title = input()

    print("When record? input like (09/19): ", end='')
    date = input()
    while(date == ''):
        date = input()

    print("What is the channel? input like (4): ", end='')
    channel = input()

    print("What is the name of program?: ",end='')
    program_name = input()

    print("What any description? if no, just enter.:", end='')
    description = input()

    return title, date, channel, program_name, description
def write_DB(title, file_name, date, channel, program_name, h_sha256, description):
    # initialize for connection to sqlite3.
    with sqlite3.connect('./cm.db') as connection:
        cursor = connection.cursor()

        # aim certain table.
        cursor.execute('CREATE TABLE IF NOT EXISTS cm_tbl(id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL UNIQUE, file_name TEXT NOT NULL, date TEXT NOT NULL, channel INTEGER, program_name TEXT, hash TEXT NOT NULL UNIQUE, description TEXT NULL)')

        # insert infomation.
        try:
            cursor.execute('INSERT INTO cm_tbl (title, file_name, date, channel, program_name, hash, description) VALUES (?, ?, ?, ?, ?, ?, ?)', [title, file_name, date, channel, program_name, h_sha256, description])
        except sqlite3.IntegrityError:
            print("The data seems to have be insearted already. The data does not be inserted.")


        # fetch information from the database.
        cursor.execute('SELECT * FROM cm_tbl')
        dbData = cursor.fetchall()

        print("Up to date in the DataBase:")
        print(dbData)


if __name__ == "__main__":
    file_name, h_sha256 = read_data() ## data is generateor format. (1), (2)
    title, date, channel, program_name, description = listen_meta()

    write_DB(title, file_name, date, channel, program_name, h_sha256, description)

   
