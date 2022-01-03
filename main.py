from inverted_index_colab import *
from os import path

if __name__ =="__main__":
    path = os.path.dirname(os.path.realpath(__file__))
    title = read_index(path.join(path , 'content/title/') ,'title' )