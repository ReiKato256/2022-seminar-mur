import random
def Picture_subject():
    #お題のリストを作成
    themelist = []
    with open("ThemeBox.txt","r",encoding='utf-8') as file:
        themelist = file.read().split("\n")

    #お題をランダムに設定
    quantity = len(themelist)
    theme = random.randint(0,(quantity-1))

    return themelist[theme]

'''確認用コード
a = Picture_subject()
print(a)
'''
