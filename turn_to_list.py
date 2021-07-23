import os


def generate(dir,label):
    files = os.listdir(dir)
    files.sort()
    print
    '****************'
    print
    'input :', dir
    print
    'start...'
    listText = open('D:\\attfdbtrain.txt', 'a')
    for file in files:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = dir +  '/' + file + ' ' + str(int(label)) + '\n'
        listText.write(name)
    listText.close()
    print
    'down!'
    print
    '****************'


outer_path = 'D:\\attfdbtrain'

if __name__ == '__main__':
    i = 0
    folderlist = os.listdir(outer_path)
    for folder in folderlist:
        generate(os.path.join(outer_path, folder),folder)
        i += 1
