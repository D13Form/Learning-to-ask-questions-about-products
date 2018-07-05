# Maps the usefull from the string into a list.
# Removes a lot of cases from the Amazon data definition
# such as \n \t ( ) ! , . \, for some of the cases a
# space gets replaced as it makes it easier for splitting
# into bag of words.
def mapper(string):
#    print(string)
    string = string.replace('.View larger.',' ')
    string = string.replace('.',' ')
    string = string.replace('\\n',' ')
    string = string.replace('\\t',' ')
#    string = string.strip('\\t')
    string = string.replace('\\','')
    string = string.replace('(',' ')
    string = string.replace(')',' ')
    string = string.replace('!',' ')
    string = string.replace(',',' ')
    string = string.replace('\'','')
    string = string.replace(':',' ')
    string = string.split(' ')
#    print(string)
    return(list(filter(None,string)))
