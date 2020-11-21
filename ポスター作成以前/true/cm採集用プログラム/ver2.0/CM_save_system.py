from glob import glob
import os

import hashlib
import sqlite3
import multiprocessing

PATH = './CMdata_ts'
FORMAT = '.mp4'

class metadata:
    title = None
    fineName = None
    date = None
    channel = None
    programName = None
    hashValue = None

    prevData = None

    desctiption = None
    @classmethod
    def Hash(self, dataPath):
        with open(dataPath, 'rb') as f: # read, binary.
            data = f.read()

        h = hashlib.sha256()
        h.update(data)
        h = h.hexdigest()
        return h

def read_data():
    ## A function, take in data from a path.
    ## Interface, only one connectiable to out of the system.
    print("Recurcible read data: ", PATH)
    wildForm = os.path.join(PATH, '**')
    Directories_Date = glob(wildForm) ##
    return Directories_Date

def setup_meta(Directories_Date):
    def listen_NOTNULL():
        one = input()
        while(one == ''):
            one = input()
        return one

    def setup(oneVideo, oneDate, oneProgram, prevData):
        print(oneVideo)
        metadata_oneVideo = metadata()

        ## type title
        print("What is the title?: ", end='')
        title = listen_NOTNULL()
        ## fileName
        fileName = os.path.basename(oneVideo)
        ## date
        date = os.path.basename(oneDate)
        ## type channel
        print("What is the channel? input like (4): ", end='')
        channel = input()
        ## programName
        programName = os.path.basename(oneProgram)
        ## hashValue
        hashValue = metadata.Hash(oneVideo)
        ## description
        print("What any description? if no, just enter: ", end='')
        description = input()

        ## save to the class
        metadata_oneVideo.title = title
        metadata_oneVideo.fileName = fileName
        metadata_oneVideo.date = date
        metadata_oneVideo.channel = channel
        metadata_oneVideo.programName = programName
        metadata_oneVideo.hashValue = hashValue
        metadata_oneVideo.prevData = prevData
        metadata_oneVideo.description = description

        return metadata_oneVideo

    # take 'one cm' in some 'cms' in some 'programs' in some 'dates'.

    metadata_All = []
    prevData = None # initialize
    for oneDate in Directories_Date:
        Directories_Program = glob(os.path.join(oneDate, '**'))
        for oneProgram in Directories_Program:
            Videos_CM = glob(os.path.join(oneProgram, '**'))
            for oneVideo in Videos_CM:
                metadata_oneVideo = setup(oneVideo, oneDate, oneProgram, prevData)
                prevData = metadata_oneVideo.hashValue
                print("_____________________________________________________________")
                print("Check Cashed :=")
                print("title: ", metadata_oneVideo.title)
                print("fileName: ", metadata_oneVideo.fileName)
                print("date: ", metadata_oneVideo.date)
                print("channel: ", metadata_oneVideo.channel)
                print("programName: ", metadata_oneVideo.programName)
                print("hashValue: ", metadata_oneVideo.hashValue)
                print("prevData: ", metadata_oneVideo.prevData)
                print("*************************************************************")

                metadata_All.append(metadata_oneVideo)

    return metadata_All

def write_toDB(metadata_All):
    # initialize for connection to sqlite3.
    with sqlite3.connect('./cm.db') as connection:
        cursor = connection.cursor()
        # aim certain table.
        cursor.execute('CREATE TABLE IF NOT EXISTS cm_tbl (\
        id INTEGER PRIMARY KEY AUTOINCREMENT, \
        title TEXT NOT NULL UNIQUE, \
        fileName TEXT NOT NULL, \
        date TEXT NOT NULL, \
        channel INTEGER, \
        programName TEXT, \
        hashValue TEXT NOT NULL UNIQUE, \
        previousData TEXT NOT NULL UNIQUE, \
        description TEXT NULL)')

    for metadata_oneVideo in metadata_All:
        # insert infomation.
        try:
            cursor.execute('INSERT INTO \
            cm_tbl (title, fileName, date, channel, programName, hashValue, previousData, description) \
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            [metadata_oneVideo.title, metadata_oneVideo.fileName, metadata_oneVideo.date, metadata_oneVideo.channel, metadata_oneVideo.programName, metadata_oneVideo.hashValue, metadata_oneVideo.prevData, metadata_oneVideo.description])
        except sqlite3.IntegrityError:
            print("The data seems to have be insearted already. The data does not be inserted.")


    # fetch information from the database.
    cursor.execute('SELECT * FROM cm_tbl')
    dbData = cursor.fetchall()

    print("Up to date in the DataBase:")
    print(dbData)

if __name__ == "__main__":
    data = read_data() ##
    metadata_All = setup_meta(data)
    #write_toDB(True)
    write_toDB(metadata_All)
