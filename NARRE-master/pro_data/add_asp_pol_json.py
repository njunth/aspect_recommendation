import codecs
import json

aspect_indicator = {}
aspect_indicator["service"] = 0
aspect_indicator["food"] = 1
aspect_indicator["miscellaneous"] = 2
aspect_indicator["price"] = 3
aspect_indicator["ambience"] = 4

def add_asp(file):
    sentence_asp = {}
    sentence_pol = {}
    f = codecs.open("../data/yelp13res/bert_0.9034.pt_bert_0.9168_polarity.txt", encoding='utf-8')
    temp_index = 0
    temp_id = ""
    n = 0
    for line in f:
        print n
        n += 1
        temp = line.split( "	" )
        id = temp[0]
        aspect = [0 for j in range( 5 )]
        polarity = [0 for j in range( 5 )]
        i = 2
        while i < len( temp ):
            aspect_tem = temp[i].split( "#" )
            aspect[aspect_indicator[aspect_tem[0]]] += 1
            if aspect_tem[1] == "positive":
                polarity[aspect_indicator[aspect_tem[0]]] += 1
            elif aspect_tem[1] == "negative":
                polarity[aspect_indicator[aspect_tem[0]]] += (-1)
            else:
                polarity[aspect_indicator[aspect_tem[0]]] += 0
            i += 1
        if id != temp_id:
            sentence_asp[id] = aspect
            sentence_pol[id] = polarity
            temp_id = id
        else:
            temp_index += 1
            sentence_asp[id] = [sentence_asp[id][i]+aspect[i] for i in range(len(aspect))]
            sentence_pol[id] = [sentence_pol[id][i]+polarity[i] for i in range(len(polarity))]
        # print id, content, aspect, polarity

    f = codecs.open("../data/yelp13res_split/"+file+".json", encoding='utf-8')
    final_dict = []
    n = 0
    for line in f:
        print n
        # print(line)
        js=json.loads(line)
        id = str( js['review_id'] )
        temp_dic = {}
        temp_dic['user_id'] = js['user_id']
        temp_dic['review_id'] = id
        temp_dic['ratings'] = js['ratings']
        temp_dic['reviews'] = js['reviews']
        temp_dic['item_id'] = js['item_id']
        if id in sentence_asp:
            temp_dic['aspect'] = sentence_asp[id]
            temp_dic['polarity'] = sentence_pol[id]
        else:
            temp_dic['aspect'] = [0 for j in range( 5 )]
            temp_dic['polarity'] = [0 for j in range( 5 )]
        final_dict.append( json.dumps( temp_dic ) + "\n" )
        n += 1

    with open("../data/yelp13res_split/"+file+"_asp_pol.json", 'w') as f:
        f.writelines(final_dict)

if __name__ == '__main__':
    add_asp("raw_valid_filtered5")
