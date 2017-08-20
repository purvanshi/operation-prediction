import falcon 
import json 
import sys 
import codecs 
import urllib 
import html 
import time 
import requests 
 
 
#from BeautifulSoup import BeautifulSoup 
 
sys.path.insert(0, '/home/puru/Documents/operator_prediction/operation-prediction') 
import model1
 
TOKEN = "441560082:AAEIdBz5_y0gfY-DgAbJ2TuJbWy3EVZeN9c" 
URL = "https://api.telegram.org/bot{}/".format(TOKEN) 
 
 
class ThingsResource(object): 
    def on_get(self, req, resp,filename): 
        """Handles GET requests""" 
        resp.status = falcon.HTTP_200 
        resp.content_type = 'text/html' 
        with open(filename, 'r') as f: 
            resp.body = f.read() 
 
class Starting(object): 
  #def on_get(self, req, resp): 
  #  p = ('\nTwo things awe me most, the starry sky ' 
    #                 'above me and the moral law within me.\n' 
    #                 '\n' 
    #                 '    ~ Immanuel Kant\n\n') 
  #  print("p") 
 
    def on_post(self,req,resp): 
        t0 = time.time() 
        print("called\n\n") 
        data=req.stream.read() 
        print(data) 
        print("\n\n\nnow printing decoded \n\n") 
        data=data.decode("utf-8") 
        print(data) 
        print("\n\n\nnow printing resolved \n\n") 
        

        data=json.loads(data) 
        
        print(data)
        question = urllib.request.unquote(data['question'])
        # question = urllib.parse.unquote(data['question'])
        print("\n\n\nnow printing parsed \n\n") 
        print(question)
        # print(data['question'])
        # answer= data
        answer=model1.main_func(question) 
        answer=str(answer) 
        # print(answer) 
        d={} 
        d["type"]=0 
        d["siaplayText"]="The answer is "+str(answer) 
        d["speech"]="The answer is "+str(answer) 
        d["answer"]=answer 
 
        resp.body = json.dumps(d, ensure_ascii=False) 
        resp.status = falcon.HTTP_200 
        t1 = time.time() 
        total = t1-t0 
        print("\n\n\n\n time taken") 
        print(total) 
app = falcon.API() 
things = ThingsResource() 
thi = Starting() 
 
app.add_route('/things/{filename}', things) 
app.add_route('/things',thi) 
